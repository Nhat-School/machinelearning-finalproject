"""Select a blend from saved source OOF predictions, then evaluate the fixed choice."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from train import metrics,tune_threshold

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';OUT=R/'ensembles'

def read(path):
    with path.open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    result={r['post_id']:r for r in rows};assert len(result)==len(rows),'Duplicate post IDs'
    return result

def aligned(fine_path,lr_path):
    fine,lr=read(fine_path),read(lr_path)
    assert set(fine)==set(lr),'Mismatched split IDs'
    ids=sorted(fine);assert all(fine[i]['label']==lr[i]['label'] for i in ids),'Mismatched labels'
    y=np.array([int(fine[i]['label']) for i in ids])
    a=np.array([float(fine[i]['score']) for i in ids]);b=np.array([float(lr[i]['score']) for i in ids])
    return ids,y,a,b,fine

def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',nargs='+',choices=['upenn','stanford'],default=['upenn','stanford'])
    args=parser.parse_args();results=[]
    for source in args.sources:
        directory=OUT/source;directory.mkdir(parents=True,exist_ok=True)
        files=[R/f'fine_tuning/{source}/oof.csv',R/f'experiments/oof_{source}_tfidf_logreg.csv']
        ids,y,a,b,_=aligned(*files);candidates=[]
        for weight in np.linspace(0,1,21):
            scores=weight*a+(1-weight)*b;threshold,f1=tune_threshold(y,scores)
            candidates.append({'minilm_weight':float(weight),'tfidf_weight':float(1-weight),
                               'threshold':threshold,'oof_f1':f1,'oof_ap':float(average_precision_score(y,scores))})
        chosen=max(candidates,key=lambda c:(c['oof_f1'],c['oof_ap'],c['minilm_weight']))
        config={'train_source':source,**chosen,'selection':'source OOF F1; tie AP, then higher MiniLM weight',
                'validation_status':'Development diagnostics: base OOF predictions also participated in model/checkpoint selection; not nested CV.',
                'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
        # Persist the decision before opening either test prediction file.
        dump(directory/'selection.json',{'selected':config,'candidates':candidates})
        dump(directory/'decision_config.json',config)
        source_results=[]
        for target in ['upenn','stanford']:
            ids,y,a,b,fine=aligned(R/f'fine_tuning/{source}/predictions_{target}.csv',
                                  R/f'experiments/predictions_{source}_{target}_tfidf_logreg.csv')
            scores=config['minilm_weight']*a+config['tfidf_weight']*b;threshold=config['threshold']
            row={'train_source':source,'test_source':target,'model':'minilm_tfidf_blend',
                 'threshold':threshold,'minilm_weight':config['minilm_weight'],**metrics(y,scores,threshold,ids)}
            source_results.append(row);results.append(row)
            with (directory/f'predictions_{target}.csv').open('w',encoding='utf-8',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=['post_id','label','score','predicted','minilm_score','tfidf_score','text'])
                writer.writeheader()
                for i,key in enumerate(ids):writer.writerow({'post_id':key,'label':int(y[i]),'score':float(scores[i]),
                    'predicted':int(scores[i]>=threshold),'minilm_score':float(a[i]),'tfidf_score':float(b[i]),'text':fine[key]['text']})
            print('BLEND TEST',json.dumps(row),flush=True)
        dump(directory/'test_results.json',source_results)
    dump(OUT/'results.json',results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,len(args.sources),figsize=(4.5*len(args.sources),3.5),squeeze=False)
    for ax,source in zip(axes.flat,args.sources):
        data=json.loads((OUT/source/'selection.json').read_text());chosen=data['selected']
        ax.plot([c['minilm_weight'] for c in data['candidates']],[c['oof_f1'] for c in data['candidates']],'o-',markersize=3,color='#315c49')
        ax.axvline(chosen['minilm_weight'],linestyle='--',color='#b57646',label=f"Selected {chosen['minilm_weight']:.2f}")
        ax.set(xlabel='MiniLM weight (remaining weight: TF-IDF LR)',ylabel='Source OOF F1 after threshold tuning',title=source.upper());ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(R/'figures/blend_oof_search.png',dpi=190,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
