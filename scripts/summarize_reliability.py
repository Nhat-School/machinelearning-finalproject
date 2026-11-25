"""Audit every test example, review coverage and per-class automatic mistakes; draft cases are not gold."""
import argparse,csv,json,re,shutil,subprocess,sys,os
from pathlib import Path
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/reliability'
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',choices=['deberta_real_only','deberta_reviewed_custom'],default='deberta_real_only')
    args=parser.parse_args();OUT=R/args.run_name
    if not (OUT/'results.json').exists():raise SystemExit('Full DeBERTa training has not completed; no classifier result is available.')
    audits={};fixed_diagnostics=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for ax,source in zip(axes,['upenn','stanford']):
        rows=read(OUT/f'predictions_{source}_mixed.csv');y=np.array([int(r['label']) for r in rows]);p=np.array([float(r['score']) for r in rows])
        pred=np.array([int(r['predicted']) for r in rows]);accept=np.array([r['action']!='REVIEW' for r in rows])
        fixed=p>=.5;precision,recall,f1,_=precision_recall_fscore_support(y,fixed,average='binary',zero_division=0)
        fixed_diagnostics.append({'source':source,'threshold':.5,'precision':float(precision),'recall':float(recall),
            'f1':float(f1),'accuracy':float(accuracy_score(y,fixed)),'confusion_matrix':confusion_matrix(y,fixed).tolist(),
            'interpretation':'Fixed 0.5 diagnostic only; not selected by test and not the deployed review policy.'})
        audits[source]={'total':len(rows),'automatic_count':int(accept.sum()),'review_count':int((~accept).sum()),
            'automatic_coverage':float(accept.mean()),'automatic_errors':int(((pred!=y)&accept).sum()),
            'automatic_accuracy':float((pred[accept]==y[accept]).mean()) if accept.any() else None,
            'urgent_automatically_found':int(((y==1)&(pred==1)&accept).sum()),
            'urgent_automatically_missed':int(((y==1)&(pred==0)&accept).sum()),
            'urgent_requiring_review':int(((y==1)&(~accept)).sum()),
            'false_urgent_automatic':int(((y==0)&(pred==1)&accept).sum())}
        found=audits[source]['urgent_automatically_found'];false=audits[source]['false_urgent_automatic']
        audits[source]['automatic_urgent_precision']=found/(found+false) if found+false else None
        audits[source]['automatic_urgent_recall_of_all_urgent_posts']=found/int(y.sum())
        observed,estimated=calibration_curve(y,p,n_bins=8,strategy='quantile')
        ax.plot([0,1],[0,1],'--',color='#999');ax.plot(estimated,observed,'o-',color='#315c49')
        ax.set(title=source.upper(),xlabel='Estimated urgency probability',ylabel='Observed urgent fraction',xlim=(0,1),ylim=(0,1))
    figure_name='deberta_calibration.png' if args.run_name=='deberta_real_only' else 'deberta_reviewed_custom_calibration.png'
    fig.tight_layout();fig.savefig(ROOT/'reports/figures'/figure_name,dpi=190);plt.close(fig)
    figures=[figure_name];prefix='deberta' if args.run_name=='deberta_real_only' else 'deberta_reviewed_custom'
    def save(fig,suffix):
        name=prefix+'_'+suffix+'.png';fig.tight_layout();fig.savefig(ROOT/'reports/figures'/name,dpi=190);plt.close(fig);figures.append(name)
    history=json.loads((OUT/'history.json').read_text());epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    axes[0].plot(epochs,[r['train_loss'] for r in history],'o-');axes[0].set(title='Training loss',xlabel='Epoch',ylabel='Source-weighted cross entropy',xticks=epochs)
    for source in ['upenn','stanford']:axes[1].plot(epochs,[r['validation_ap_by_source'][source] for r in history],'o-',label=source)
    axes[1].set(title='Validation checkpoint diagnostic',xlabel='Epoch',ylabel='Average precision',xticks=epochs,ylim=(0,1));axes[1].legend();save(fig,'training')
    results=json.loads((OUT/'results.json').read_text());mixed={r['test_source']:r for r in results if r['policy']=='mixed'}
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.5))
    for ax,source in zip(axes,['upenn','stanford']):
        matrix=np.array(mixed[source]['confusion_matrix']);ax.imshow(matrix,cmap='Blues')
        for a in range(2):
            for b in range(2):ax.text(b,a,str(matrix[a,b]),ha='center',va='center',color='white' if matrix[a,b]>matrix.max()*.5 else 'black')
        ax.set(title=source.upper()+' shared policy',xlabel='Predicted',ylabel='True',xticks=[0,1],yticks=[0,1])
    save(fig,'confusion')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for ax,source in zip(axes,['upenn','stanford']):
        audit=audits[source];counts=[audit['automatic_count']-audit['automatic_errors'],audit['automatic_errors'],audit['review_count']]
        ax.bar(['Automatic correct','Automatic wrong','Human review'],counts,color=['#315c49','#b55243','#b5a26c'])
        for i,n in enumerate(counts):ax.text(i,n,str(n),ha='center',va='bottom',fontsize=9)
        ax.set(title=source.upper()+' all test posts',ylabel='Post count');ax.tick_params(axis='x',labelsize=8)
    save(fig,'review')
    baselines=[]
    for relative,label in [('fine_tuning/results.json','MiniLM separate'),('ensembles/results.json','Blend separate')]:
        path=ROOT/'reports'/relative
        if path.exists():baselines.extend((label,r) for r in json.loads(path.read_text()) if r['train_source']==r['test_source'])
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
    for ax,source in zip(axes,['upenn','stanford']):
        candidates=[(label,r) for label,r in baselines if r['test_source']==source]+[('DeBERTa joint',mixed[source])]
        x=np.arange(len(candidates));width=.24
        for j,(metric,title) in enumerate([('precision','Precision'),('recall','Recall'),('f1','F1')]):ax.bar(x+(j-1)*width,[r[metric] for _,r in candidates],width,label=title)
        ax.set(title=source.upper()+' historical test',xticks=x,xticklabels=[label for label,_ in candidates],ylim=(0,1));ax.tick_params(axis='x',labelsize=8)
    axes[0].legend(fontsize=8);save(fig,'comparison')
    (OUT/'figure_files.json').write_text(json.dumps(figures,indent=2)+'\n')
    (OUT/'reliability_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    (OUT/'fixed_threshold_diagnostic.json').write_text(json.dumps(fixed_diagnostics,indent=2)+'\n')
    folder=ROOT/'models'/('urgency_'+args.run_name)
    shutil.copyfile(ROOT/'scripts/predict_reliable_deberta.py',folder/'predict.py')
    import transformers
    (folder/'requirements.txt').write_text(f'torch>=2.8\ntransformers=={transformers.__version__}\nsentencepiece==0.2.1\nprotobuf==6.33.6\n')
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name in ['protocol.json','history.json']:
        shutil.copyfile(OUT/name,folder/name)
    shutil.copyfile(R/'data_plan.json',folder/'data_plan.json')
    source_card=R/'model_source.json'
    if args.run_name=='deberta_real_only':
        source_card.write_text(json.dumps({'model_id':protocol['model_id'],'revision':protocol['revision'],
            'status':protocol['weights_status'],'publisher':'Microsoft','run':'deberta_real_only'},indent=2)+'\n')
    # Choose an actual saved REVIEW post for a reproducible browser check.
    candidates=[r for source in ['upenn','stanford'] for r in read(OUT/f'predictions_{source}_mixed.csv')
                if r['action']=='REVIEW' and re.search('[a-zA-Z]',r['text'])]
    central=[r for r in candidates if .25<=float(r['score'])<=.75 and len(r['text'])>=15]
    candidates=central or candidates
    if candidates:
        chosen=min(candidates,key=lambda row:(len(row['text']),row['post_id']))
        (OUT/'browser_review_input.json').write_text(json.dumps({'text':chosen['text'],'post_id':chosen['post_id'],
            'source':'historical test prediction; browser check only'},indent=2)+'\n')
    (folder/'README.md').write_text('# Joint DeBERTa urgency model\n\n'
        f"Trained on UPenn and Stanford fit partitions, with {protocol['custom_rows_used']} approved custom training rows. Unreviewed drafts excluded.\n"
        'Use: python predict.py --text "I cannot submit my assignment and the deadline is tonight."\n'
        'Default mixed policy is shared across both sources. Source-specific policies are diagnostic alternatives.\n'
        'REVIEW means a lecturer should judge the post. Calibration does not guarantee correctness.\n'
        'Supported scope: English student forum posts. No validated semantic out-of-domain detector is included.\n')
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(ROOT/'.cache/deberta-empty-cache'),HF_HUB_CACHE=str(ROOT/'.cache/deberta-empty-cache/hub'))
    examples=[]
    for text in ['I cannot submit my assignment and the deadline is tonight. Please help!','Thank you for the helpful explanation. I understand now.']:
        output=subprocess.run([sys.executable,str(folder/'predict.py'),'--text',text],env=env,check=True,capture_output=True,text=True)
        examples.append(json.loads(output.stdout))
    assert all(np.isfinite(r['score']) for r in examples)
    (OUT/'standalone_check.json').write_text(json.dumps({'passed':True,'offline':True,'examples':examples},indent=2)+'\n')
    print(json.dumps(audits,indent=2));print('Offline DeBERTa export verified; examples retain their actual predictions.')

if __name__=='__main__':main()
