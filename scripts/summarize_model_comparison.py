"""Recreate the MiniLM/DeBERTa discussion evidence from completed runs; no training."""
import csv,json,math,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPORTS=ROOT/'reports';OUT=REPORTS/'reliability'
def load(path):return json.loads(path.read_text())
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def tensor_count(path):
    with path.open('rb') as f:header=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
    return sum(math.prod(v['shape']) for k,v in header.items() if k!='__metadata__')
def main():
    folder=OUT/'deberta_real_only'
    fine=load(REPORTS/'fine_tuning/results.json');deberta=load(folder/'results.json')
    fixed=load(folder/'fixed_threshold_diagnostic.json');comparisons=[];alignment={}
    for source in ['upenn','stanford']:
        mini=next(r for r in fine if r['train_source']==r['test_source']==source)
        large=next(r for r in deberta if r['test_source']==source and r['policy']=='mixed')
        diagnostic=next(r for r in fixed if r['source']==source)
        mini_rows=rows(REPORTS/f'fine_tuning/{source}/predictions_{source}.csv')
        large_rows=rows(folder/f'predictions_{source}_mixed.csv')
        assert {r['post_id']:r['label'] for r in mini_rows}=={r['post_id']:r['label'] for r in large_rows}
        assert len(mini_rows)==len({r['post_id'] for r in mini_rows})
        alignment[source]={'same_test_ids_and_labels':True,'test_posts':len(mini_rows)}
        for model,metrics,threshold,policy in [('MiniLM',mini,mini['threshold'],'published_selected'),
                ('DeBERTa',large,large['threshold'],'published_selected'),
                ('MiniLM',mini['default_threshold_metrics'],.5,'fixed_0_5'),
                ('DeBERTa',diagnostic,.5,'fixed_0_5')]:
            matrix=metrics['confusion_matrix']
            records=mini_rows if model=='MiniLM' else large_rows
            counted=[[0,0],[0,0]]
            for record in records:counted[int(record['label'])][int(float(record['score'])>=threshold)]+=1
            assert counted==matrix,(source,model,policy,'Confusion counts disagree with saved scores.')
            tn,fp=matrix[0];fn,tp=matrix[1]
            assert abs(metrics['precision']-tp/(tp+fp))<1e-12
            assert abs(metrics['recall']-tp/(tp+fn))<1e-12
            assert abs(metrics['f1']-2*tp/(2*tp+fp+fn))<1e-12
            comparisons.append({'source':source,'model':model,'policy':policy,'threshold':threshold,
                **{key:metrics[key] for key in ['precision','recall','f1','accuracy']},
                'confusion_matrix':matrix,'false_positives':matrix[0][1],'false_negatives':matrix[1][0]})
    folders={'MiniLM':ROOT/'models/urgency_minilm_stanford','DeBERTa':ROOT/'models/urgency_deberta_real_only'}
    models={key:{'stored_tensor_values':tensor_count(value/'model.safetensors'),
                 'max_length':load(value/'decision_config.json')['max_length']} for key,value in folders.items()}
    drafted=rows(folder/'custom_draft_predictions.csv')
    examples=[r for r in drafted if r['scenario_group']=='submission_button' and r['case_type'] in ['blocked','resolved']]
    from transformers import AutoTokenizer
    for key,path in folders.items():
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=key!='DeBERTa')
        for example in examples:example[key.lower()+'_tokens']=len(tokenizer(example['text'])['input_ids'])
    metadata=load(ROOT/'Datasets/CustomForum/metadata.json');plan=load(OUT/'data_plan.json')
    protocol=load(folder/'protocol.json');history=load(folder/'history.json')
    summary={'model_size_basis':'Tensor elements counted from saved safetensors headers, including any saved buffers; approximate model size.',
        'models':models,'comparisons':comparisons,'test_alignment_checks':alignment,'context_examples':examples,
        'actual_deberta_custom_training_rows':protocol['custom_rows_used'],
        'custom_dataset':metadata,'data_plan':plan,'deberta_history':history,
        'interpretation':'These are different pipelines, training compositions, calibration and selection budgets. Same numerical threshold is not a controlled architecture-only comparison.',
        'three_dataset_minilm_status':('See final_minilm_trial/summary.json for the measured matched MiniLM extension. Synthetic weak supervision is distinct from human-reviewed labels.'
            if (REPORTS/'final_minilm_trial/summary.json').exists() else 'Proposed; no joint MiniLM run completed.')}
    (OUT/'model_comparison_discussion.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
    for ax,source in zip(axes,['upenn','stanford']):
        selected=[r for r in comparisons if r['source']==source and r['policy']=='fixed_0_5'];x=np.arange(2);width=.24
        for i,(metric,title) in enumerate([('precision','Precision'),('recall','Recall'),('f1','F1')]):
            ax.bar(x+(i-1)*width,[r[metric] for r in selected],width,label=title)
        ax.set(title=source.upper()+' fixed threshold 0.5',xticks=x,xticklabels=[r['model'] for r in selected],ylim=(0,1))
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(REPORTS/'figures/minilm_deberta_fixed_threshold.png',dpi=190);plt.close(fig)
    print('Recreated model-size discussion: 8 measured comparisons, 2 context examples, no new model training.')
if __name__=='__main__':main()
