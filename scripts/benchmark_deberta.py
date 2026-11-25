"""A disposable three-step training smoke check, not an accuracy evaluation."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from train import read_csv
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--bf16',action='store_true');parser.add_argument('--max-length',type=int,default=512);args=parser.parse_args()
card=json.loads((ROOT/'reports/reliability/model_source.json').read_text())
snapshot=ROOT/'.cache/huggingface/hub'/('models--'+card['model_id'].replace('/','--'))/'snapshots'/card['revision']
device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
start=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,use_fast=False)
model=AutoModelForSequenceClassification.from_pretrained(snapshot,num_labels=2,local_files_only=True,attn_implementation='eager').to(device)
fit_ids={r['post_id'] for r in read_csv(ROOT/'reports/reliability/split_plan.csv') if r['role']=='fit'}
rows=sorted([r for r in read_csv(ROOT/'Datasets/Stanford/processed.csv') if r['post_id'] in fit_ids],key=lambda r:len(r['text_clean']),reverse=True)[:4]
inputs=tokenizer([r['text_clean'] for r in rows],truncation=True,max_length=args.max_length,padding=True,return_tensors='pt')
inputs={k:v.to(device) for k,v in inputs.items()};labels=torch.tensor([int(r['priority']) for r in rows],device=device)
optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5);times=[];losses=[]
for step in range(3):
    begin=time.perf_counter();optimizer.zero_grad(set_to_none=True)
    from contextlib import nullcontext
    with torch.autocast(device_type=device,dtype=torch.bfloat16) if args.bf16 else nullcontext():loss=model(**inputs,labels=labels).loss
    loss.backward();optimizer.step()
    if device=='mps':torch.mps.synchronize()
    if device=='cuda':torch.cuda.synchronize()
    times.append(time.perf_counter()-begin);losses.append(float(loss.detach().cpu()))
    print('step',step,'seconds',round(times[-1],3),'loss',losses[-1],flush=True)
result={'status':'training smoke test only; discarded weights; no classifier evaluation','device':device,
        'batch_size':len(rows),'sequence_length':inputs['input_ids'].shape[1],'step_seconds':times,'loss':losses,
        'parameters':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.perf_counter()-start,'bf16':args.bf16}
if device=='mps':result['driver_memory_gb']=torch.mps.driver_allocated_memory()/1024**3
target='benchmark_bf16.json' if args.bf16 else 'benchmark.json'
(ROOT/'reports/reliability'/target).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
