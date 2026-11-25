"""Validate and plot results parsed from the user's supplied Kaggle log excerpt."""
import argparse,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/kaggle_run'

def load(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',type=Path,default=OUT/'provided_log_excerpt.txt')
    args=parser.parse_args()
    records={}
    for line in args.log.read_text(encoding='utf-8').splitlines():
        match=re.search(r'FINAL TEST (\{.*\})',line)
        if match:
            row=json.loads(match.group(1));key=(row['train_source'],row['test_source'])
            if key in records:assert records[key]==row,'Conflicting duplicated results in this log.'
            records[key]=row
    assert set(records)=={(a,b) for a in ['upenn','stanford'] for b in ['upenn','stanford']}
    rows=list(records.values())
    for row in rows:
        for result in [row,row['default_threshold_metrics']]:
            (tn,fp),(fn,tp)=result['confusion_matrix']
            assert tn+fp+fn+tp==result['test_size']
            assert fn+tp==result['positive_count']
            expected={'precision':tp/(tp+fp),'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),
                      'accuracy':(tn+tp)/(tn+fp+fn+tp)}
            for key,value in expected.items():assert abs(result[key]-value)<1e-12,(key,result[key],value)
    (OUT/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    local=load(ROOT/'reports/fine_tuning/results.json')
    order=[('upenn','upenn'),('stanford','stanford'),('upenn','stanford'),('stanford','upenn')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,3.7));x=np.arange(4)
    for offset,data,label,color in [(-.17,local,'Mac MPS / original run','#809aa7'),(.17,rows,'Kaggle T4 / supplied log','#315c49')]:
        values=[next(r['f1'] for r in data if (r['train_source'],r['test_source'])==direction) for direction in order]
        bars=ax.bar(x+offset,values,.32,label=label,color=color);ax.bar_label(bars,fmt='%.4f',fontsize=8,padding=2)
    ax.set_xticks(x,[a.title()+' → '+b.title() for a,b in order]);ax.set_ylim(0,1)
    ax.set_ylabel('Urgent-class F1');ax.set_title('Two recorded runs: hardware, libraries and selected configurations differ')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'reports/figures/kaggle_comparison.png',dpi=190,bbox_inches='tight');plt.close(fig)
    row=records[('stanford','stanford')]
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.5))
    for ax,result,title in [(axes[0],row,f"OOF threshold {row['threshold']:.4f}"),(axes[1],row['default_threshold_metrics'],'Default threshold 0.5')]:
        matrix=np.array(result['confusion_matrix']);ax.imshow(matrix,cmap='Greens')
        for a in range(2):
            for b in range(2):ax.text(b,a,str(matrix[a,b]),ha='center',va='center',fontsize=16,color='white' if matrix[a,b]>matrix.max()/2 else '#142e36')
        ax.set_xticks([0,1],['Not urgent','Urgent']);ax.set_yticks([0,1],['Not urgent','Urgent'])
        ax.set(xlabel='Predicted',ylabel='Actual',title=title+f" · F1 {result['f1']:.4f}")
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/kaggle_stanford_confusion.png',dpi=190,bbox_inches='tight');plt.close(fig)
    print('PASS: all four Kaggle results agree with their confusion matrices. Two evidence figures regenerated.')

if __name__=='__main__':main()
