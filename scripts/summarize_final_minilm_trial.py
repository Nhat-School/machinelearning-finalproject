"""Verify saved trial predictions and recreate report figures without training."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score
from final_minilm_trial import ROOT,BASE,read,dump,load_real,load_custom
from evaluate_final_minilm_trial import context_metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--controls-only',action='store_true');args=parser.parse_args()
    full=(BASE/'results.json').exists() and not args.controls_only;path=BASE/('results.json' if full else 'control_results.json')
    results=json.loads(path.read_text());real=load_real();custom=load_custom(True);checks=[];errors={};contexts={};protocols={}
    for name in dict.fromkeys(r['run'] for r in results):
        directory=BASE/name;protocols[name]=json.loads((directory/'protocol.json').read_text())
        training=read(directory/'training_ids.csv');trained={r['post_id'] for r in training}
        for role in ['validation','calibration','test']:
            assert not trained.intersection(r['post_id'] for rows in real[role].values() for r in rows)
        assert not trained.intersection(r['post_id'] for split in ['validation','challenge'] for r in custom[split])
        errors[name]={}
        for source in ['upenn','stanford']:
            rows=read(directory/f'predictions_{source}.csv');result=next(r for r in results if r['run']==name and r['test_source']==source)
            expected={r['post_id']:int(r['priority']) for r in real['test'][source]}
            assert len(rows)==len(expected) and {r['post_id']:int(r['label']) for r in rows}==expected
            y=[int(r['label']) for r in rows];pred=[int(float(r['score'])>=.5) for r in rows]
            assert confusion_matrix(y,pred,labels=[0,1]).tolist()==result['confusion_matrix']
            assert abs(f1_score(y,pred)-result['f1'])<1e-12
            errors[name][source]={
                'FP':sorted([r for r in rows if r['label']=='0' and r['predicted']=='1'],key=lambda r:-float(r['score']))[:2],
                'FN':sorted([r for r in rows if r['label']=='1' and r['predicted']=='0'],key=lambda r:float(r['score']))[:2]}
        rows=read(directory/'custom_all_predictions.csv')
        contexts[name]={split:context_metrics([r for r in rows if r['split']==split],custom[split]) for split in custom}
        contexts[name]['submission_examples']=[r for r in rows if r['scenario_group']=='submission_button']
        checks.append({'run':name,'real_test_ids_labels_and_confusion_counts_verified':True,
            'fit_disjoint_from_real_calibration_validation_test':True,'custom_train_disjoint_from_validation_and_challenge':True})
        script=ROOT/'scripts/final_minilm_trial.py'
        if hashlib.sha256(script.read_bytes()).hexdigest()==protocols[name]['training_script_sha256']:
            (directory/'training_source.py').write_bytes(script.read_bytes())
    fine=json.loads((ROOT/'reports/fine_tuning/results.json').read_text())
    large=json.loads((ROOT/'reports/reliability/deberta_real_only/fixed_threshold_diagnostic.json').read_text())
    comparisons=[]
    for source in ['upenn','stanford']:
        old=next(r for r in fine if r['train_source']==r['test_source']==source)
        comparisons.append({'test_source':source,'label':'Previous source MiniLM','policy':'raw head score >= 0.5',**old['default_threshold_metrics']})
        comparisons.append({'test_source':source,'label':'Joint DeBERTa','policy':'calibrated score >= 0.5',**next(r for r in large if r['source']==source)})
        comparisons.extend({**r,'label':r['run'],'policy':'calibrated score >= 0.5'} for r in results if r['test_source']==source)
    data={'status':'three-arm trial evaluated' if full else 'two-source joint MiniLM controls complete; no custom training in these runs',
        'results':results,'comparisons':comparisons,'contexts':contexts,'protocols':protocols,'checks':checks,
        'historical_tests_are_exploratory':True,'custom_is_synthetic_not_independent_gold':True,
        'custom_reviewed_count':sum(r['review_status']=='approved' for r in read(ROOT/'Datasets/CustomForum/reviewed_labels.csv'))}
    dump(BASE/('control_summary.json' if args.controls_only else 'summary.json'),data)
    dump(BASE/('control_error_examples.json' if args.controls_only else 'error_examples.json'),errors)
    prefix='best_minilm' if args.controls_only else 'final_minilm'
    figures=ROOT/'reports/figures'
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for ax,source in zip(axes,['upenn','stanford']):
        rows=[r for r in comparisons if r['test_source']==source];x=np.arange(len(rows));width=.25
        for i,(key,title) in enumerate([('precision','Precision'),('recall','Recall'),('f1','F1')]):
            ax.bar(x+(i-1)*width,[r[key] for r in rows],width,label=title)
        ax.set(xticks=x,xticklabels=[r['label'].replace('real_only_','Joint MiniLM\n').replace('custom_pairwise_','Custom pair\n').replace('custom_ce_','Custom CE\n') for r in rows],
            ylim=(0,1),title=source.upper()+' | all historical test posts')
        ax.tick_params(axis='x',labelsize=7,rotation=30);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Fixed 0.5 decision rule; old pipelines differ in calibration and training composition',fontsize=10)
    fig.tight_layout();fig.savefig(figures/(prefix+'_metrics.png'),dpi=180);plt.close(fig)
    primary=[r for r in results if r['seed']==42 and r['test_source'] in ['upenn','stanford']]
    arms=list(dict.fromkeys(r['arm'] for r in primary));fig,axes=plt.subplots(len(arms),2,figsize=(8,3.1*len(arms)),squeeze=False)
    for i,arm in enumerate(arms):
        for j,source in enumerate(['upenn','stanford']):
            result=next(r for r in primary if r['arm']==arm and r['test_source']==source);cm=np.array(result['confusion_matrix']);ax=axes[i,j]
            ax.imshow(cm,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Urgent'],yticklabels=['Normal','Urgent'],xlabel='Predicted',ylabel='Original label',title=arm+' / '+source)
            for y in range(2):
                for x in range(2):ax.text(x,y,str(cm[y,x]),ha='center',va='center',color='white' if cm[y,x]>cm.max()/2 else 'black',fontsize=13)
    fig.tight_layout();fig.savefig(figures/(prefix+'_confusion.png'),dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for name,protocol in protocols.items():
        history=json.loads((BASE/name/'history.json').read_text());epochs=[r['epoch'] for r in history]
        axes[0].plot(epochs,[r['train_loss'] for r in history],marker='o',label=name)
        axes[1].plot(epochs,[r['mean_real_validation_ap'] for r in history],marker='o',label=name)
    axes[0].set(title='Training loss',xlabel='Epoch');axes[1].set(title='Mean real validation AP',xlabel='Epoch',ylim=(0,1))
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(figures/(prefix+'_learning.png'),dpi=180);plt.close(fig)
    print(data['status']);print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
