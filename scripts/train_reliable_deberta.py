"""Joint DeBERTa experiment with separate fit, calibration, validation and historical test roles."""
import argparse,csv,json,math,random,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,balanced_accuracy_score,brier_score_loss,roc_curve
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
from finetune_minilm import encode,batches,seed_all
from train import read_csv,write_csv,metrics
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/reliability'
MODEL='microsoft/deberta-v3-base';REVISION='8ccc9b6f36199bec6961081d44eb72fb3f7353f3'

def dump(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def logit(p):p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))
def calibrate(p,card):return 1/(1+np.exp(-np.clip(card['coefficient']*logit(p)+card['intercept'],-40,40)))
def tensor_batch(tokenizer,encoded,idx,device):
    # Fewer shape variants reduce MPS compilation overhead; CUDA also benefits from aligned lengths.
    return {k:v.to(device) for k,v in tokenizer.pad([encoded[i] for i in idx],padding=True,
            pad_to_multiple_of=32,return_tensors='pt').items()}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=5);parser.add_argument('--patience',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=8);parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--max-length',type=int,default=512);parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--include-approved-custom',action='store_true')
    parser.add_argument('--smoke-steps',type=int,default=0)
    parser.add_argument('--mps-bf16',action='store_true')
    args=parser.parse_args();seed_all(args.seed)
    name='deberta_reviewed_custom' if args.include_approved_custom else 'deberta_real_only'
    if args.smoke_steps:name='deberta_smoke'
    out=R/name;out.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    def amp():
        if device=='cuda':return torch.autocast('cuda',dtype=dtype)
        if device=='mps' and args.mps_bf16:return torch.autocast('mps',dtype=torch.bfloat16)
        return nullcontext()
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and dtype==torch.float16)
    cache=ROOT/'.cache/huggingface/hub'/('models--'+MODEL.replace('/','--'))/'snapshots'/REVISION
    if not (cache/'config.json').exists() or not any((cache/f).exists() for f in ['model.safetensors','pytorch_model.bin']):
        from huggingface_hub import snapshot_download
        cache=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','spm.model','pytorch_model.bin','model.safetensors']))
    tokenizer=AutoTokenizer.from_pretrained(cache,local_files_only=True,use_fast=False)
    model=AutoModelForSequenceClassification.from_pretrained(cache,local_files_only=True,num_labels=2,
        id2label={0:'NORMAL',1:'PRIORITY'},label2id={'NORMAL':0,'PRIORITY':1},attn_implementation='eager').to(device)
    names={'upenn':'UPenn','stanford':'Stanford'}
    plan={r['post_id']:r for r in read_csv(R/'split_plan.csv')};data={s:read_csv(ROOT/f'Datasets/{names[s]}/processed.csv') for s in names}
    fit=[r for source in data.values() for r in source if plan[r['post_id']]['role']=='fit']
    custom=read_csv(R/'approved_custom_train.csv') if args.include_approved_custom else []
    if args.include_approved_custom and not custom:raise ValueError('No approved custom training rows. Human review is required before this ablation.')
    fit+=custom;source_counts={s:sum(r['source']==s for r in fit) for s in {r['source'] for r in fit}}
    weight=np.array([len(fit)/(len(source_counts)*source_counts[r['source']]) for r in fit],dtype=np.float32)
    encoded=encode(tokenizer,[r['text_clean'] for r in fit],args.max_length);labels=np.array([int(r['priority']) for r in fit])
    validation={s:[r for r in rows if plan[r['post_id']]['role']=='validation'] for s,rows in data.items()}
    val_encoded={s:encode(tokenizer,[r['text_clean'] for r in rows],args.max_length) for s,rows in validation.items()}
    def predict(records):
        model.eval();values=np.empty(len(records))
        with torch.inference_mode():
            for idx in batches(records,range(len(records)),args.batch_size):
                with amp():p=torch.softmax(model(**tensor_batch(tokenizer,records,idx,device)).logits.float(),dim=-1)[:,1]
                values[idx]=p.cpu().numpy()
        return values
    protocol={'model_id':MODEL,'revision':REVISION,'configuration':vars(args),'device':device,
              'precision':str(dtype) if device=='cuda' else ('bfloat16' if device=='mps' and args.mps_bf16 else 'float32'),'source_fit_counts':source_counts,
              'source_balancing':'loss weights inverse source sample count; natural class ratios retained',
              'checkpoint_selection':'mean per-source validation AP; no F1/test-driven selection',
              'calibration':'per-source and source-balanced pooled Platt scaling on calibration role only',
              'decision_threshold':'validation balanced accuracy; pooled policy averages source balanced accuracy',
              'review_policy':'calibrated p <= 0.15 NORMAL or >= 0.85 PRIORITY, consistent with binary threshold; otherwise REVIEW',
              'custom_rows_used':len(custom),'historical_test_status':'previously inspected; exploratory comparison',
              'weights_status':'untrained initialization' if args.smoke_steps else 'training in progress'}
    dump(out/'protocol.json',protocol)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    steps=math.ceil(math.ceil(len(fit)/args.batch_size)/args.accumulation)*args.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,max(1,int(steps*.1)),steps)
    history=[];best=-1.;stale=0;update_count=0;start=time.perf_counter();checkpoint=out/'best_state.pt'
    for epoch in range(1,args.epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.;seen=0;epoch_start=time.perf_counter()
        chunks=list(batches(encoded,range(len(fit)),args.batch_size,True,args.seed+epoch))
        for i,idx in enumerate(chunks):
            block_size=min(args.accumulation,len(chunks)-(i//args.accumulation)*args.accumulation)
            with amp():
                logits=model(**tensor_batch(tokenizer,encoded,idx,device)).logits
                losses=torch.nn.functional.cross_entropy(logits,torch.tensor(labels[idx],device=device),reduction='none')
                loss=(losses*torch.tensor(weight[idx],device=device)).mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss; do not export this run.')
            scaler.scale(loss/block_size).backward();loss_sum+=float(loss.detach().cpu())*len(idx);seen+=len(idx)
            if (i+1)%args.accumulation==0 or i+1==len(chunks):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True);update_count+=1
                if update_count%50==0:print('TRAIN PROGRESS',json.dumps({'epoch':epoch,'fit_rows_seen':seen,'optimizer_updates':update_count,'epoch_seconds':time.perf_counter()-epoch_start}),flush=True)
                if args.smoke_steps and update_count>=args.smoke_steps:
                    dump(out/'smoke_check.json',{'passed':True,'optimizer_updates':update_count,'seconds':time.perf_counter()-start,
                        'status':'discarded smoke weights; no classifier quality evaluation','device':device})
                    print('SMOKE PASSED',flush=True);return
        ap={s:float(average_precision_score([int(r['priority']) for r in validation[s]],predict(val_encoded[s]))) for s in data}
        score=float(np.mean(list(ap.values())));row={'epoch':epoch,'train_loss':loss_sum/seen,'validation_ap_by_source':ap,
            'mean_validation_ap':score,'epoch_seconds':time.perf_counter()-epoch_start};history.append(row);dump(out/'history.json',history)
        print('DEBERTA EPOCH',json.dumps(row),flush=True)
        if score>best+1e-5:
            best=score;stale=0;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
        else:stale+=1
        if stale>=args.patience:break
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));checkpoint.unlink()
    policies={};cal_rows={};val_scores={}
    for source,rows in data.items():
        cal_rows[source]=[r for r in rows if plan[r['post_id']]['role']=='calibration']
    pooled_cal=[r for rows in cal_rows.values() for r in rows]
    pooled_raw=predict(encode(tokenizer,[r['text_clean'] for r in pooled_cal],args.max_length))
    for domain in ['upenn','stanford','mixed']:
        use=np.array([domain=='mixed' or r['source']==domain for r in pooled_cal]);y=np.array([int(r['priority']) for r in pooled_cal])[use]
        balance=np.array([len(pooled_cal)/(2*len(cal_rows[r['source']])) for r in pooled_cal])[use] if domain=='mixed' else None
        calibrator=LogisticRegression(C=1.,max_iter=1000).fit(logit(pooled_raw[use]).reshape(-1,1),y,sample_weight=balance)
        policies[domain]={'coefficient':float(calibrator.coef_[0,0]),'intercept':float(calibrator.intercept_[0]),'review_low':.15,'review_high':.85}
    for source in data:val_scores[source]=predict(val_encoded[source])
    for domain,card in policies.items():
        sources=list(data) if domain=='mixed' else [domain]
        calibrated={s:calibrate(val_scores[s],card) for s in sources}
        candidates=np.unique(np.concatenate([np.linspace(0,1,101),*[np.quantile(p,np.linspace(0,1,101)) for p in calibrated.values()]]))
        scored=[(float(np.mean([balanced_accuracy_score([int(r['priority']) for r in validation[s]],calibrated[s]>=t) for s in sources])),float(t)) for t in candidates]
        card['validation_balanced_accuracy'],card['threshold']=max(scored,key=lambda pair:(pair[0],-abs(pair[1]-.5)))
    folder=ROOT/'models'/('urgency_'+name);folder.mkdir(exist_ok=True)
    model.save_pretrained(folder,safe_serialization=True);tokenizer.save_pretrained(folder)
    decision={'model_id':MODEL,'revision':REVISION,'max_length':args.max_length,'policies':policies,'default_domain':'mixed',
              'scope':'English student forum posts; no validated semantic out-of-domain detector',
              'default_review_note':'REVIEW is a model decision-support policy, not a correctness guarantee.'}
    dump(folder/'decision_config.json',decision);results=[]
    for source,rows in data.items():
        test=[r for r in rows if plan[r['post_id']]['role']=='test'];raw=predict(encode(tokenizer,[r['text_clean'] for r in test],args.max_length))
        y=np.array([int(r['priority']) for r in test]);ids=[r['post_id'] for r in test]
        for domain in [source,'mixed']:
            card=policies[domain];p=calibrate(raw,card);threshold=card['threshold'];prediction=p>=threshold
            accepted=(p<=min(card['review_low'],threshold))|(p>=max(card['review_high'],threshold))
            result={'model':'joint_deberta','train_source':'upenn+stanford','test_source':source,'policy':domain,
                    'threshold':threshold,**metrics(y,p,threshold,ids),'brier_score':float(brier_score_loss(y,p)),
                    'automatic_coverage':float(accepted.mean()),'review_count':int((~accepted).sum()),
                    'automatic_accuracy':float((prediction[accepted]==y[accepted]).mean()) if accepted.any() else None,
                    'urgent_sent_to_review':int(((~accepted)&(y==1)).sum())}
            results.append(result)
            write_csv(out/f'predictions_{source}_{domain}.csv',[{'post_id':r['post_id'],'label':int(label),'score':float(score),
                'predicted':int(pred),'action':'REVIEW' if not accept else ('PRIORITY' if pred else 'NORMAL'),'text':r['text_clean']}
                for r,label,score,pred,accept in zip(test,y,p,prediction,accepted)])
            print('DEBERTA TEST',json.dumps(result),flush=True)
    dump(out/'results.json',results);protocol['weights_status']='trained, exported and evaluated';protocol['training_seconds']=time.perf_counter()-start
    dump(out/'protocol.json',protocol)

if __name__=='__main__':main()
