"""Scientific plots generated directly from saved held-out predictions and metrics."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/experiments';FIG=ROOT/'reports/figures'
NAMES={'tfidf_logreg':'TF-IDF + LR','tfidf_svm':'TF-IDF + SVM','tfidf_nb':'TF-IDF + NB','minilm_logreg':'MiniLM + LR','tfidf_dummy':'Dummy'}
ORDER=['tfidf_dummy','tfidf_nb','tfidf_logreg','tfidf_svm','minilm_logreg']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})

def read(name):
    with (OUT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def save(name):
    plt.savefig(FIG/name,dpi=190,bbox_inches='tight',facecolor='white');plt.close()

def main():
    results=json.loads((OUT/'results.json').read_text());cv=json.loads((OUT/'cv_results.json').read_text())
    for source in ['upenn','stanford']:
        rows={r['model']:r for r in results if r['train_source']==source and r['test_source']==source}
        fig,ax=plt.subplots(figsize=(8.7,3.7));x=np.arange(5)
        for offset,key,color in [(-.24,'precision','#809aa7'),(0,'recall','#b57646'),(.24,'f1','#315c49')]:
            bars=ax.bar(x+offset,[rows[m][key] for m in ORDER],.23,label=key.capitalize(),color=color)
            ax.bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
        ax.set_xticks(x,[NAMES[m] for m in ORDER]);ax.set_ylim(0,1.12);ax.set_ylabel('Positive-class score');ax.legend(ncol=3,loc='upper left');ax.set_title(f'{source.upper()} held-out test · same split for all methods');save(f'{source}_metrics.png')
        fig,ax=plt.subplots(figsize=(8.7,3.7))
        for m in ORDER:
            predictions=read(f'predictions_{source}_{source}_{m}.csv')
            y=np.array([int(r['label']) for r in predictions]);s=np.array([float(r['score']) for r in predictions])
            precision,recall,_=precision_recall_curve(y,s)
            ax.step(recall,precision,where='post',label=f'{NAMES[m]} · AP {rows[m]["average_precision"]:.3f}',linewidth=1.7)
        ax.axhline(y.mean(),color='#999999',ls='--',label=f'Prevalence {y.mean():.3f}')
        ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.03),title=f'{source.upper()} precision–recall curves');ax.legend(fontsize=8);save(f'{source}_pr.png')
    fig,axes=plt.subplots(2,2,figsize=(9,6.6))
    selected={source:max([r for r in cv if r['source']==source and r['name']!='dummy'],key=lambda r:r['oof_tuned_f1'])['model'] for source in ['upenn','stanford']}
    for ax,(source,target) in zip(axes.flat,[('upenn','upenn'),('upenn','stanford'),('stanford','stanford'),('stanford','upenn')]):
        row=next(r for r in results if r['train_source']==source and r['test_source']==target and r['model']==selected[source])
        matrix=np.asarray(row['confusion_matrix']);ax.imshow(matrix,cmap='Blues')
        for a in range(2):
            for b in range(2):ax.text(b,a,str(matrix[a,b]),ha='center',va='center',fontsize=17,color='white' if matrix[a,b]>matrix.max()/2 else '#142e36')
        ax.set_xticks([0,1],['Not priority','Priority']);ax.set_yticks([0,1],['Not priority','Priority']);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(f'{source} → {target}\n{NAMES[row["model"]]}')
    fig.tight_layout();save('confusion_four.png')
    fig,ax=plt.subplots(figsize=(8.7,3.8));x=np.arange(4);methods=ORDER[1:]
    for offset,(source,target),color in [(-.15,('upenn','stanford'),'#315c49'),(.15,('stanford','upenn'),'#b57646')]:
        values=[next(r['f1'] for r in results if r['train_source']==source and r['test_source']==target and r['model']==m) for m in methods]
        bars=ax.bar(x+offset,values,.29,label=f'{source} → {target}',color=color);ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    ax.set_xticks(x,[NAMES[m] for m in methods]);ax.set_ylim(0,1);ax.set_ylabel('Positive-class F1');ax.legend();ax.set_title('Cross-source transfer · no target tuning');save('transfer.png')
    deployment=json.loads((ROOT/'models/deployment.json').read_text());source=deployment['source'];model=deployment['model']
    rows=read(f'oof_{source}_{model}.csv');y=np.array([int(r['label']) for r in rows]);s=np.array([float(r['score']) for r in rows]);p,r,t=precision_recall_curve(y,s);f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    fig,ax=plt.subplots(figsize=(8.7,3.4));ax.plot(t,p[:-1],label='Precision');ax.plot(t,r[:-1],label='Recall');ax.plot(t,f,label='F1');ax.axvline(deployment['threshold'],color='#333',ls='--',label=f"Chosen {deployment['threshold']:.3f}");ax.set(xlabel='Decision threshold',ylabel='Source OOF score',ylim=(0,1.04),title='Threshold chosen from training out-of-fold predictions');ax.legend(ncol=4,fontsize=8);save('threshold.png')
    # Exact source rows for manual discussion in the DOCX.
    examples={}
    for target in ['upenn','stanford']:
        rows=read(f'predictions_{source}_{target}_{model}.csv')
        examples[target]={kind:sorted([r for r in rows if r['error']==kind],key=lambda r:float(r['score']),reverse=(kind=='FP'))[:3] for kind in ['FN','FP']}
    (OUT/'error_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Generated metrics, PR, confusion, transfer, threshold figures and exact error examples.')

if __name__=='__main__':main()
