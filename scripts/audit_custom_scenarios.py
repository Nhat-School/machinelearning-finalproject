"""Record model behaviour on synthetic drafts without treating the draft labels as ground truth."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path(__file__).resolve().parents[1]
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',choices=['deberta_real_only','deberta_reviewed_custom'],default='deberta_real_only')
    args=parser.parse_args();OUT=ROOT/'reports/reliability'/args.run_name
    rows=read(ROOT/'Datasets/CustomForum/draft_examples.csv');texts=[r['text'] for r in rows]
    scores={};actions={}
    for key,folder in [('deberta',ROOT/'models'/('urgency_'+args.run_name)),('minilm',ROOT/'models/urgency_minilm_stanford')]:
        if not (folder/'decision_config.json').exists():
            if key=='minilm':continue
            raise SystemExit('Train/export DeBERTa before the context audit.')
        card=json.loads((folder/'decision_config.json').read_text())
        tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,use_fast=key!='deberta')
        model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True).eval();raw=[]
        with torch.inference_mode():
            for i in range(0,len(texts),4):
                batch=tokenizer(texts[i:i+4],padding=True,truncation=True,max_length=card['max_length'],return_tensors='pt')
                raw.extend(torch.softmax(model(**batch).logits.float(),dim=-1)[:,1].numpy().tolist())
        del model,tokenizer
        if key=='deberta':
            policy=card['policies']['mixed'];r=np.clip(raw,1e-6,1-1e-6)
            z=policy['coefficient']*np.log(r/(1-r))+policy['intercept'];p=1/(1+np.exp(-np.clip(z,-40,40)))
            threshold=policy['threshold'];scores[key]=p
            actions[key]=['NORMAL' if s<=min(.15,threshold) else ('PRIORITY' if s>=max(.85,threshold) else 'REVIEW') for s in p]
        else:
            scores[key]=np.array(raw);actions[key]=['PRIORITY' if s>=card['threshold'] else 'NORMAL' for s in raw]
    saved=[]
    for i,row in enumerate(rows):
        saved.append({**row,'evaluation_status':'synthetic diagnostic only; human labels pending',
                      **{key+'_score':float(scores[key][i]) for key in scores},
                      **{key+'_action':actions[key][i] for key in actions}})
    with (OUT/'custom_draft_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(saved[0]));writer.writeheader();writer.writerows(saved)
    pairs={}
    for i,row in enumerate(rows):
        if row['case_type'] in ['blocked','resolved']:pairs.setdefault(row['scenario_group'],{})[row['case_type']]=i
    summaries={key:{'paired_scenarios':len(pairs),'score_decreases_after_resolution':int(sum(scores[key][p['resolved']]<scores[key][p['blocked']] for p in pairs.values())),
                    'interpretation':'Directional behaviour on drafted scenarios, not accuracy or a fresh benchmark.'} for key in scores}
    (OUT/'custom_context_diagnostics.json').write_text(json.dumps(summaries,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(scores),figsize=(5*len(scores),3.7),sharey=True,squeeze=False)
    for ax,key in zip(axes[0],scores):
        for pair in pairs.values():ax.plot([0,1],[scores[key][pair['blocked']],scores[key][pair['resolved']]],'o-',alpha=.35,lw=.8,ms=3)
        ax.set(xticks=[0,1],xticklabels=['Unresolved issue','Resolved update'],ylabel='Model score',title=key.upper(),ylim=(0,1))
    fig.suptitle('Synthetic draft context pairs; labels await human review',fontsize=11);fig.tight_layout()
    fig.savefig(ROOT/'reports/figures'/('deberta_context_drafts.png' if args.run_name=='deberta_real_only' else 'deberta_reviewed_custom_context_drafts.png'),dpi=190);plt.close(fig)
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
