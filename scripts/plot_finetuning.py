"""Plot real fine-tuning histories, validation choices and held-out comparisons."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/fine_tuning';FIG=ROOT/'reports/figures'

def read(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    baseline=json.loads((ROOT/'reports/experiments/results.json').read_text())
    results=[];selections={};cv={}
    for source in ['upenn','stanford']:
        results+=json.loads((OUT/source/'test_results.json').read_text())
        selections[source]=json.loads((OUT/source/'selection.json').read_text())
        cv[source]=json.loads((OUT/source/'cv.json').read_text())
    (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(10,6.3))
    for col,source in enumerate(['upenn','stanford']):
        for card in selections[source]['scout_runs']:
            history=card['history'];epochs=[r['epoch'] for r in history]
            axes[0,col].plot(epochs,[r['train_loss'] for r in history],marker='o',label=card['name'].replace('scout_',''))
            axes[1,col].plot(epochs,[r['val_ap'] for r in history],marker='o',label=card['name'].replace('scout_',''))
        axes[0,col].set(title=f'{source.upper()} · training loss',xlabel='Epoch',ylabel='Cross-entropy')
        axes[1,col].set(title=f'{source.upper()} · scout validation',xlabel='Epoch',ylabel='Average precision',ylim=(0,1))
        for ax in axes[:,col]:ax.legend(fontsize=8);ax.set_xticks([1,2,3])
    fig.tight_layout();fig.savefig(FIG/'ft_learning_curves.png',dpi=190,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8));x=np.arange(5)
    for source,color in [('upenn','#b57646'),('stanford','#315c49')]:
        values=[r['validation_f1_tuned'] for r in cv[source]['folds']]
        ax.plot(x+1,values,'o-',label=f'{source} · mean {np.mean(values):.3f} ± {np.std(values):.3f}',color=color)
    ax.set(xlabel='Fold',ylabel='Validation F1 after threshold tuning',ylim=(0,1),title='5-fold development diagnostics · early stopping uses fold validation');ax.set_xticks(range(1,6));ax.legend();fig.tight_layout();fig.savefig(FIG/'ft_cv.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,source in zip(axes,['upenn','stanford']):
        records=[next(r for r in baseline if r['train_source']==source and r['test_source']==source and r['model']==name) for name in ['tfidf_logreg','minilm_logreg']]
        records.append(next(r for r in results if r['train_source']==source and r['test_source']==source))
        for offset,key,color in [(-.25,'precision','#7e9ba8'),(0,'recall','#bd8051'),(.25,'f1','#315c49')]:
            bars=ax.bar(np.arange(3)+offset,[r[key] for r in records],.24,label=key,color=color);ax.bar_label(bars,fmt='%.3f',fontsize=7)
        ax.set_xticks(range(3),['TF-IDF + LR','Frozen + LR','Fine-tuned']);ax.set_ylim(0,1.12);ax.set_title(source.upper());ax.set_ylabel('Positive-class score');ax.legend(fontsize=7,ncol=3)
    fig.tight_layout();fig.savefig(FIG/'ft_comparison.png',dpi=190,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8.5,6.5))
    for ax,r in zip(axes.flat,results):
        matrix=np.array(r['confusion_matrix']);ax.imshow(matrix,cmap='Greens')
        for a in range(2):
            for b in range(2):ax.text(b,a,str(matrix[a,b]),ha='center',va='center',fontsize=17,color='white' if matrix[a,b]>matrix.max()/2 else '#142e36')
        ax.set_xticks([0,1],['Not urgent','Urgent']);ax.set_yticks([0,1],['Not urgent','Urgent'])
        ax.set(xlabel='Predicted',ylabel='Actual',title=f"Fine-tuned {r['train_source']} → {r['test_source']}")
    fig.tight_layout();fig.savefig(FIG/'ft_confusion.png',dpi=190,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for ax,source in zip(axes,['upenn','stanford']):
        for label,path in [('TF-IDF + LR',ROOT/f'reports/experiments/predictions_{source}_{source}_tfidf_logreg.csv'),
                           ('Frozen MiniLM + LR',ROOT/f'reports/experiments/predictions_{source}_{source}_minilm_logreg.csv'),
                           ('Fine-tuned MiniLM',OUT/source/f'predictions_{source}.csv')]:
            rows=read(path);y=[int(r['label']) for r in rows];s=[float(r['score']) for r in rows]
            p,recall,_=precision_recall_curve(y,s);ax.step(recall,p,where='post',label=label)
        ax.set(title=source.upper(),xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIG/'ft_pr.png',dpi=190,bbox_inches='tight');plt.close(fig)
    print('Fine-tuning results and 5 figures generated from saved runs.')

if __name__=='__main__':main()
