"""Matched MiniLM controls and custom-data ablations, with test evaluation deferred."""
import argparse,csv,gc,hashlib,json,math,os,platform,random,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,balanced_accuracy_score,brier_score_loss
from transformers import AutoModelForSequenceClassification,AutoTokenizer,get_linear_schedule_with_warmup
from finetune_minilm import batches,encode,seed_all,snapshot
from train import metrics,read_csv,write_csv,ENCODER,ENCODER_REV
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reports/final_minilm_trial'

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def fingerprint(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def logit(values):
    values=np.clip(values,1e-6,1-1e-6);return np.log(values/(1-values))
def calibrated(values,policy):
    z=policy['coefficient']*logit(values)+policy['intercept'];return 1/(1+np.exp(-np.clip(z,-40,40)))
def load_real():
    plan={r['post_id']:r['role'] for r in read(ROOT/'reports/reliability/split_plan.csv')}
    data={key:read(ROOT/f'Datasets/{folder}/processed.csv') for key,folder in [('upenn','UPenn'),('stanford','Stanford')]}
    return {role:{source:[r for r in records if plan[r['post_id']]==role] for source,records in data.items()}
            for role in ['fit','calibration','validation','test']}
def load_custom(allow_drafts=False):
    drafts=read(ROOT/'Datasets/CustomForum/draft_examples.csv')
    review={r['post_id']:r for r in read(ROOT/'Datasets/CustomForum/reviewed_labels.csv')};result=[]
    for row in drafts:
        judgement=review.get(row['post_id'],{})
        if judgement.get('review_status')=='approved':
            if not judgement.get('reviewer','').strip():raise ValueError('Human-approved labels need a named reviewer.')
            label=judgement.get('adjudicated_priority');origin='human_reviewed'
        elif allow_drafts:label=row['draft_priority'];origin='assistant_draft_weak_label'
        else:continue
        if label not in ['0','1']:continue
        result.append({**row,'text_clean':row['text'],'priority':label,'source':'custom','training_label_origin':origin})
    return {split:[r for r in result if r['split']==split] for split in ['train','validation','challenge']}
def tensor_batch(tokenizer,encoded,indices,device):
    return {key:value.to(device) for key,value in tokenizer.pad([encoded[i] for i in indices],padding=True,
            pad_to_multiple_of=8,return_tensors='pt').items()}
def scores(model,tokenizer,encoded,device,batch_size,amp=lambda:nullcontext()):
    model.eval();result=np.empty(len(encoded),dtype=np.float64)
    with torch.inference_mode():
        for indices in batches(encoded,range(len(encoded)),batch_size):
            with amp():values=torch.softmax(model(**tensor_batch(tokenizer,encoded,indices,device)).logits.float(),dim=-1)[:,1]
            result[indices]=values.cpu().numpy()
    return result
def run_name(arm,seed):return arm+'_seed'+str(seed)

def train(args):
    seed_all(args.seed);name=run_name(args.arm,args.seed);out=BASE/name
    folder=ROOT/'models'/('urgency_final_minilm_'+name)
    if (out/'protocol.json').exists() and not args.force:
        previous=json.loads((out/'protocol.json').read_text())
        if previous.get('status')=='trained and exported; historical test deferred':
            print('Reusing completed training:',name,flush=True);return
        raise ValueError('Incomplete run already exists. Inspect it and use --force to restart this same arm.')
    real=load_real();custom=load_custom(args.allow_synthetic_drafts)
    using_custom=args.arm!='real_only'
    if using_custom and not custom['train']:
        raise ValueError('No eligible custom labels. Use reviewed labels or an explicitly authorized synthetic weak-label trial.')
    if using_custom and args.allow_synthetic_drafts:
        authorization=BASE/'synthetic_authorization.json'
        if not authorization.exists() or not json.loads(authorization.read_text()).get('user_authorized_synthetic_trial'):
            raise ValueError('The experimental synthetic-label exception needs recorded user authorization; never forge human review.')
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    cuda_bf16=device=='cuda' and torch.cuda.is_bf16_supported(including_emulation=False)
    def amp():
        if device=='cuda':return torch.autocast('cuda',dtype=torch.bfloat16 if cuda_bf16 else torch.float16)
        if device=='mps' and args.bf16:return torch.autocast('mps',dtype=torch.bfloat16)
        return nullcontext()
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and not cuda_bf16)
    tokenizer=AutoTokenizer.from_pretrained(snapshot(),local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(snapshot(),num_labels=2,local_files_only=True,
        id2label={0:'NORMAL',1:'PRIORITY'},label2id={'NORMAL':0,'PRIORITY':1},attn_implementation='sdpa').to(device)
    fit=[row for records in real['fit'].values() for row in records]
    encoded=encode(tokenizer,[r['text_clean'] for r in fit],args.max_length);y=np.array([int(r['priority']) for r in fit])
    counts={source:len(records) for source,records in real['fit'].items()}
    weights=np.array([len(fit)/(2*counts[r['source']]) for r in fit],dtype=np.float32)
    val_encoded={source:encode(tokenizer,[r['text_clean'] for r in records],args.max_length) for source,records in real['validation'].items()}
    custom_encoded=encode(tokenizer,[r['text_clean'] for r in custom['train']],args.max_length) if using_custom else []
    custom_y=np.array([int(r['priority']) for r in custom['train']]) if using_custom else np.array([])
    paired={}
    for i,row in enumerate(custom['train']):
        if row['case_type'] in ['blocked','resolved','still_blocked_update']:paired.setdefault(row['scenario_group'],{})[row['case_type']]=i
    paired=[group for group in paired.values() if {'blocked','resolved','still_blocked_update'}<=set(group)]
    if using_custom:
        positive=int(custom_y.sum());negative=len(custom_y)-positive
        if not positive or not negative:raise ValueError('Custom training must contain both binary classes.')
        custom_class_weight=torch.tensor([len(custom_y)/(2*negative),len(custom_y)/(2*positive)],device=device)
    protocol={'arm':args.arm,'seed':args.seed,'base_model':ENCODER,'base_revision':ENCODER_REV,'configuration':vars(args),
        'real_fit_counts':counts,'real_weighting':'inverse source count, equal aggregate contributions from UPenn and Stanford',
        'custom_training_rows':len(custom['train']) if using_custom else 0,
        'custom_label_origin':sorted({r['training_label_origin'] for r in custom['train']}) if using_custom else [],
        'custom_loss_share':args.custom_share if using_custom else 0.,
        'pairwise_coefficient_inside_custom_loss':args.pairwise_coefficient if args.arm=='custom_pairwise' else 0.,
        'checkpoint_selection':'highest mean real-source validation AP; identical rule for every arm',
        'calibration':'source-balanced pooled Platt scaling, separate real calibration role, CPU float32',
        'binary_threshold':.5,'review_bounds':[.2,.8],
        'test_status':'historical tests previously inspected; new test evaluation deferred until candidate selection is frozen',
        'training_device':device,'training_precision':'bfloat16' if (device=='mps' and args.bf16) or cuda_bf16 else ('float16' if device=='cuda' else 'float32'),
        'evaluation_precision':'CPU float32','versions':{'python':platform.python_version(),'torch':torch.__version__},
        'split_sha256':fingerprint(ROOT/'reports/reliability/split_plan.csv'),
        'custom_draft_sha256':fingerprint(ROOT/'Datasets/CustomForum/draft_examples.csv'),
        'training_script_sha256':fingerprint(Path(__file__)),'status':'training in progress'}
    out.mkdir(parents=True,exist_ok=True);dump(out/'protocol.json',protocol)
    write_csv(out/'training_ids.csv',[{'post_id':r['post_id'],'source':r['source'],'label':r['priority'],'label_origin':'original_research_label'} for r in fit]+
        ([{'post_id':r['post_id'],'source':'custom','label':r['priority'],'label_origin':r['training_label_origin']} for r in custom['train']] if using_custom else []))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    updates=math.ceil(len(fit)/args.batch_size)*args.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,max(1,int(updates*.1)),updates)
    history=[];best=-1.;best_epoch=None;stale=0;checkpoint=out/'best_state.pt';start=time.perf_counter();steps=0
    for epoch in range(1,args.epochs+1):
        model.train();loss_total=0.;real_total=0.;custom_total=0.;ranking_total=0.;seen=0;begin=time.perf_counter()
        for indices in batches(encoded,range(len(fit)),args.batch_size,True,args.seed+epoch):
            optimizer.zero_grad(set_to_none=True)
            with amp():
                logits=model(**tensor_batch(tokenizer,encoded,indices,device)).logits
                real_losses=torch.nn.functional.cross_entropy(logits,torch.tensor(y[indices],device=device),reduction='none')
                real_loss=(real_losses*torch.tensor(weights[indices],device=device)).mean();custom_loss=real_loss.new_zeros(());ranking=real_loss.new_zeros(())
                if using_custom:
                    rng=random.Random(args.seed*100000+epoch*10000+steps);selected=[];pairs=[]
                    for group in rng.sample(paired,min(2,len(paired))):
                        offset=len(selected);selected.extend([group['blocked'],group['resolved'],group['still_blocked_update']]);pairs.extend([(offset,offset+1),(offset+2,offset+1)])
                    while len(selected)<12:selected.append(rng.randrange(len(custom_encoded)))
                    custom_logits=model(**tensor_batch(tokenizer,custom_encoded,selected,device)).logits
                    custom_loss=torch.nn.functional.cross_entropy(custom_logits,torch.tensor(custom_y[selected],device=device),weight=custom_class_weight)
                    if pairs:
                        margin=custom_logits[:,1]-custom_logits[:,0]
                        ranking=torch.stack([torch.nn.functional.softplus(-(margin[u]-margin[n])) for u,n in pairs]).mean()
                    extra=custom_loss+(args.pairwise_coefficient*ranking if args.arm=='custom_pairwise' else 0.)
                    loss=(1-args.custom_share)*real_loss+args.custom_share*extra
                else:loss=real_loss
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss; no model should be exported.')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();scheduler.step();steps+=1
            amount=len(indices);seen+=amount;loss_total+=float(loss.detach().cpu())*amount;real_total+=float(real_loss.detach().cpu())*amount
            custom_total+=float(custom_loss.detach().cpu())*amount;ranking_total+=float(ranking.detach().cpu())*amount
            if steps%100==0:print('FINAL TRIAL PROGRESS',name,'epoch',epoch,'steps',steps,'seconds',round(time.perf_counter()-begin,1),flush=True)
        ap={source:float(average_precision_score([int(r['priority']) for r in real['validation'][source]],
            scores(model,tokenizer,records,device,args.batch_size,amp))) for source,records in val_encoded.items()}
        quality=float(np.mean(list(ap.values())));row={'epoch':epoch,'train_loss':loss_total/seen,'real_loss':real_total/seen,
            'custom_loss':custom_total/seen,'pairwise_loss':ranking_total/seen,'validation_ap_by_source':ap,
            'mean_real_validation_ap':quality,'seconds':time.perf_counter()-begin};history.append(row);dump(out/'history.json',history)
        print('FINAL TRIAL EPOCH',name,json.dumps(row),flush=True)
        if quality>best+1e-5:
            best=quality;best_epoch=epoch;stale=0;torch.save({key:value.detach().cpu() for key,value in model.state_dict().items()},checkpoint)
        else:stale+=1
        if stale>=args.patience:break
    model=model.to('cpu');model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));checkpoint.unlink()
    if device=='mps':torch.mps.empty_cache()
    if device=='cuda':torch.cuda.empty_cache()
    cal=[r for records in real['calibration'].values() for r in records]
    raw=scores(model,tokenizer,encode(tokenizer,[r['text_clean'] for r in cal],args.max_length),'cpu',64)
    balance=np.array([len(cal)/(2*len(real['calibration'][r['source']])) for r in cal])
    calibrator=LogisticRegression(C=1.,max_iter=1000).fit(logit(raw).reshape(-1,1),[int(r['priority']) for r in cal],sample_weight=balance)
    policy={'coefficient':float(calibrator.coef_[0,0]),'intercept':float(calibrator.intercept_[0]),'threshold':.5,'review_low':.2,'review_high':.8}
    folder.mkdir(parents=True,exist_ok=True);model.save_pretrained(folder,safe_serialization=True);tokenizer.save_pretrained(folder)
    dump(folder/'decision_config.json',{'model_id':ENCODER,'revision':ENCODER_REV,'max_length':args.max_length,'policies':{'mixed':policy},
        'default_domain':'mixed','arm':args.arm,'seed':args.seed,'custom_label_origin':protocol['custom_label_origin'],
        'scope':'English student forum posts; synthetic custom labels are not independent gold labels'})
    validation=[]
    for source,records in real['validation'].items():
        raw=scores(model,tokenizer,encode(tokenizer,[r['text_clean'] for r in records],args.max_length),'cpu',64);p=calibrated(raw,policy)
        y_val=[int(r['priority']) for r in records]
        validation.append({'source':source,**metrics(y_val,p,.5,[r['post_id'] for r in records]),'balanced_accuracy':float(balanced_accuracy_score(y_val,p>=.5))})
        write_csv(out/f'validation_{source}.csv',[{'post_id':r['post_id'],'label':r['priority'],'raw_score':float(a),'score':float(b),'text':r['text_clean']} for r,a,b in zip(records,raw,p)])
    # Save predictions on draft validation texts for later diagnostic interpretation, never as human gold.
    diagnostic=[r for r in read(ROOT/'Datasets/CustomForum/draft_examples.csv') if r['split']=='validation']
    raw=scores(model,tokenizer,encode(tokenizer,[r['text'] for r in diagnostic],args.max_length),'cpu',64);p=calibrated(raw,policy)
    write_csv(out/'custom_validation_diagnostic.csv',[{**r,'score':float(s),'predicted':int(s>=.5),'label_status':'assistant draft; not human reviewed'} for r,s in zip(diagnostic,p)])
    dump(out/'validation_results.json',validation)
    protocol.update(status='trained and exported; historical test deferred',best_epoch=best_epoch,best_mean_validation_ap=best,
        training_seconds=time.perf_counter()-start,optimizer_updates=steps)
    dump(out/'protocol.json',protocol);dump(folder/'training_protocol.json',protocol)
    del model,optimizer;gc.collect();print('TRAINING COMPLETED; TEST DEFERRED',name,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['real_only','custom_ce','custom_pairwise'],default='real_only')
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--patience',type=int,default=2);parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--max-length',type=int,default=256);parser.add_argument('--lr',type=float,default=2e-5)
    parser.add_argument('--custom-share',type=float,default=.2);parser.add_argument('--pairwise-coefficient',type=float,default=.25)
    parser.add_argument('--allow-synthetic-drafts',action='store_true');parser.add_argument('--bf16',action='store_true');parser.add_argument('--force',action='store_true')
    args=parser.parse_args();train(args)
if __name__=='__main__':main()
