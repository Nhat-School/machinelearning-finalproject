"""Portable supervised MiniLM fine-tuning, source-only selection, CV and offline export."""
from __future__ import annotations
import argparse,csv,gc,hashlib,json,math,os,random,time,sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit,StratifiedGroupKFold,StratifiedKFold,train_test_split
from sklearn.metrics import average_precision_score,f1_score
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
from train import metrics,tune_threshold,read_csv,write_csv,ENCODER,ENCODER_REV
from prepare_data import dump_json,sha256,clean_text

ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1])).resolve()
OUT=ROOT/'reports/fine_tuning'
CONFIGS={
    'plain':{'weighted':False,'augmentation':False,'lr':2e-5},
    'weighted':{'weighted':True,'augmentation':False,'lr':2e-5},
    'weighted_aug':{'weighted':True,'augmentation':True,'lr':2e-5},
}
MIXED_PRECISION=False

def autocast(device):
    return torch.autocast(device_type=device,dtype=torch.bfloat16) if MIXED_PRECISION else nullcontext()
CONTRACTIONS={"can't":"cannot","won't":"will not","don't":"do not","doesn't":"does not",
    "didn't":"did not","isn't":"is not","aren't":"are not","wasn't":"was not","weren't":"were not",
    "couldn't":"could not","wouldn't":"would not","shouldn't":"should not","haven't":"have not",
    "hasn't":"has not","hadn't":"had not","i'm":"I am","you're":"you are","we're":"we are",
    "they're":"they are","i've":"I have","we've":"we have","i'll":"I will","we'll":"we will"}

def augment(text):
    import re
    text=text.replace('’',"'")
    return re.sub(r"\b(?:"+'|'.join(re.escape(k) for k in CONTRACTIONS)+r")\b",
                  lambda m:CONTRACTIONS[m.group(0).lower()],text,flags=re.I)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def snapshot():
    local=ROOT/'.cache/huggingface/hub'/('models--'+ENCODER.replace('/','--'))/'snapshots'/ENCODER_REV
    if local.exists():return str(local)
    from huggingface_hub import snapshot_download
    return snapshot_download(ENCODER,revision=ENCODER_REV,allow_patterns=['*.json','*.txt','*.safetensors','1_Pooling/*'])

def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(snapshot(),local_files_only=True)
    return device,tokenizer

def batches(encodings,indices,batch_size,shuffle=False,seed=42):
    # Sort by length to reduce padding; shuffle batch order for each training epoch.
    order=sorted(map(int,indices),key=lambda i:len(encodings[i]['input_ids']))
    chunks=[order[i:i+batch_size] for i in range(0,len(order),batch_size)]
    if shuffle:random.Random(seed).shuffle(chunks)
    yield from chunks

def encode(tokenizer,texts,max_length):
    result=[]
    for start in range(0,len(texts),256):
        encoded=tokenizer(texts[start:start+256],truncation=True,max_length=max_length,padding=False)
        result.extend({k:encoded[k][i] for k in encoded} for i in range(len(encoded['input_ids'])))
    return result

def tensor_batch(tokenizer,encoded,idx,device):
    return {k:v.to(device) for k,v in tokenizer.pad([encoded[i] for i in idx],padding=True,return_tensors='pt').items()}

def predict(model,tokenizer,encoded,device,batch_size):
    model.eval();scores=np.empty(len(encoded),dtype=np.float64)
    with torch.inference_mode():
        for idx in batches(encoded,range(len(encoded)),batch_size):
            with autocast(device):
                values=torch.softmax(model(**tensor_batch(tokenizer,encoded,idx,device)).logits.float(),dim=-1)[:,1]
            scores[idx]=values.float().cpu().numpy()
    return scores

def new_model(seed,device):
    seed_all(seed)
    model=AutoModelForSequenceClassification.from_pretrained(snapshot(),num_labels=2,
        id2label={0:'NOT_URGENT',1:'URGENT'},label2id={'NOT_URGENT':0,'URGENT':1},
        attn_implementation='sdpa',local_files_only=True)
    return model.to(device)

def fit_run(source,name,config,train_rows,val_rows,device,tokenizer,args,fixed_epochs=None,export=None):
    directory=OUT/source/name;directory.mkdir(parents=True,exist_ok=True)
    texts=[r['text_clean'] for r in train_rows];labels=[int(r['priority']) for r in train_rows]
    augmented=[]
    if config['augmentation']:
        for r in train_rows:
            changed=augment(r['text_clean'])
            if changed!=r['text_clean']:
                texts.append(changed);labels.append(int(r['priority']))
                augmented.append({'parent_post_id':r['post_id'],'label':r['priority'],'text':changed})
    if augmented:write_csv(directory/'augmented_rows.csv',augmented)
    encoded=encode(tokenizer,texts,args.max_length);y=np.array(labels)
    val_enc=encode(tokenizer,[r['text_clean'] for r in val_rows],args.max_length) if val_rows else []
    val_y=np.array([int(r['priority']) for r in val_rows])
    model=new_model(args.seed,device)
    positive=int(y.sum());negative=len(y)-positive
    weights=torch.tensor([len(y)/(2*negative),len(y)/(2*positive)],dtype=torch.float32,device=device) if config['weighted'] else None
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.01)
    epochs=fixed_epochs or args.epochs
    steps_per_epoch=math.ceil(len(y)/args.batch_size)
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=max(1,int(steps_per_epoch*epochs*.1)),num_training_steps=steps_per_epoch*epochs)
    best_ap=-1.;best_epoch=0;best_scores=None;stale=0;history=[]
    start=time.perf_counter();checkpoint=directory/'best_state.pt'
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0.;seen=0;epoch_start=time.perf_counter()
        for step,idx in enumerate(batches(encoded,range(len(y)),args.batch_size,True,args.seed+epoch)):
            optimizer.zero_grad(set_to_none=True)
            inputs=tensor_batch(tokenizer,encoded,idx,device)
            target=torch.tensor(y[idx],device=device,dtype=torch.long)
            with autocast(device):
                loss=criterion(model(**inputs).logits,target)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step()
            loss_sum+=float(loss.detach().cpu())*len(idx);seen+=len(idx)
        record={'epoch':epoch,'train_loss':loss_sum/seen,'seconds':time.perf_counter()-epoch_start}
        if val_rows:
            scores=predict(model,tokenizer,val_enc,device,args.batch_size*2)
            ap=float(average_precision_score(val_y,scores));threshold,tuned_f1=tune_threshold(val_y,scores)
            record.update(val_ap=ap,val_f1_default=float(f1_score(val_y,scores>=.5,zero_division=0)),
                          val_f1_tuned=tuned_f1,threshold=threshold)
            if ap>best_ap+1e-5:
                best_ap=ap;best_epoch=epoch;best_scores=scores.copy();stale=0
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
            else:stale+=1
        else:best_epoch=epoch
        history.append(record);dump_json(directory/'history.json',history)
        print(source,name,json.dumps(record),flush=True)
        if val_rows and stale>=args.patience:break
    if val_rows:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state);del state
        threshold,tuned_f1=tune_threshold(val_y,best_scores)
        write_csv(directory/'validation_predictions.csv',[{'post_id':r['post_id'],'label':int(r['priority']),'score':float(s)} for r,s in zip(val_rows,best_scores)])
    else:threshold=.5;tuned_f1=None
    card={'source':source,'name':name,'config':config,'best_epoch':best_epoch,'epochs_run':len(history),
          'validation_ap':best_ap if val_rows else None,'validation_f1_tuned':tuned_f1,'validation_threshold':threshold,
          'original_training_posts':len(train_rows),'augmented_posts':len(augmented),'training_rows':len(y),
          'validation_rows':len(val_rows),'seconds':time.perf_counter()-start,'device':device,
          'max_length':args.max_length,'batch_size':args.batch_size,'seed':args.seed,'history':history}
    dump_json(directory/'run.json',card)
    if export:
        model.save_pretrained(export,safe_serialization=True);tokenizer.save_pretrained(export)
    checkpoint.unlink(missing_ok=True)
    del optimizer,scheduler,encoded,val_enc
    gc.collect()
    return model,card,best_scores

def release(model,device):
    model.to('cpu')
    del model;gc.collect()
    if device=='mps':torch.mps.empty_cache()
    if device=='cuda':torch.cuda.empty_cache()

def load_data():
    data={s:read_csv(ROOT/f'data/processed/{s}.csv') for s in ['upenn','stanford']}
    assignment={r['post_id']:r['split'] for r in read_csv(ROOT/'reports/experiments/split_ids.csv')}
    training={s:[r for r in rows if assignment[r['post_id']]=='train'] for s,rows in data.items()}
    testing={s:[r for r in rows if assignment[r['post_id']]=='test'] for s,rows in data.items()}
    return data,training,testing

def main():
    global MIXED_PRECISION
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['benchmark','scout','cv','final','all'],default='all')
    parser.add_argument('--sources',nargs='+',choices=['upenn','stanford'],default=['upenn','stanford'])
    parser.add_argument('--epochs',type=int,default=4);parser.add_argument('--patience',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--max-length',type=int,default=256)
    parser.add_argument('--bf16',action='store_true',help='Use bfloat16 autocast on supported MPS/CUDA hardware')
    parser.add_argument('--folds',type=int,default=5);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();device,tokenizer=setup();MIXED_PRECISION=args.bf16 and device!='cpu';data,training,testing=load_data()
    if args.stage=='benchmark':
        rows=training['stanford'][:512]
        model,card,_=fit_run('stanford','benchmark',CONFIGS['plain'],rows,[],device,tokenizer,args,fixed_epochs=1)
        card['estimated_full_stanford_epoch_seconds']=card['seconds']*len(training['stanford'])/len(rows)
        dump_json(OUT/'benchmark.json',card);release(model,device);return
    protocol={'seed':args.seed,'base_model':ENCODER,'revision':ENCODER_REV,'max_epochs':args.epochs,
        'patience':args.patience,'early_stopping_metric':'validation AP','folds':args.folds,'max_length':args.max_length,
        'batch_size':args.batch_size,'bf16':MIXED_PRECISION,'attention':'sdpa','configs':CONFIGS,'selection':'source scout validation tuned F1; tie AP',
        'final_epochs':'rounded median of selected-config CV best epochs','threshold':'source OOF F1 maximum',
        'tests':'same previously inspected hold-outs; exploratory comparisons, no test-based tuning',
        'data_hashes':{s:sha256(ROOT/f'data/processed/{s}.csv') for s in data},
        'augmentation':'one contraction-expanded variant when text changes; training rows only'}
    dump_json(OUT/'protocol.json',protocol)
    for source in args.sources:
        rows=training[source];y=np.array([int(r['priority']) for r in rows]);groups=np.array([r['source_id'] for r in rows])
        selection_file=OUT/source/'selection.json'
        if args.stage in ['scout','all']:
            if source=='upenn':
                tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed+1).split(y,y,groups))
                assert not set(groups[tr])&set(groups[va])
            else:tr,va=train_test_split(np.arange(len(rows)),test_size=.15,stratify=y,random_state=args.seed+1)
            scout_train_set=set(tr)
            write_csv(OUT/source/'scout_split.csv',[{'post_id':rows[i]['post_id'],'role':'train' if i in scout_train_set else 'validation'} for i in range(len(rows))])
            cards=[]
            for name,config in CONFIGS.items():
                model,card,_=fit_run(source,'scout_'+name,config,[rows[i] for i in tr],[rows[i] for i in va],device,tokenizer,args)
                cards.append(card);release(model,device)
            chosen=max(cards,key=lambda c:(c['validation_f1_tuned'],c['validation_ap']))
            dump_json(selection_file,{'selected':chosen['name'].replace('scout_',''),'scout_runs':cards})
            print('SELECTED',source,chosen['name'],flush=True)
        selection=json.loads(selection_file.read_text());name=selection['selected'];config=CONFIGS[name]
        cv_file=OUT/source/'cv.json'
        if args.stage in ['cv','all']:
            splitter=StratifiedGroupKFold(n_splits=args.folds,shuffle=True,random_state=args.seed) if source=='upenn' else StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=args.seed)
            folds=splitter.split(np.arange(len(rows)),y,groups) if source=='upenn' else splitter.split(np.arange(len(rows)),y)
            oof=np.zeros(len(rows));cards=[];fold_assign=[]
            for fold,(tr,va) in enumerate(folds,1):
                if source=='upenn':assert not set(groups[tr])&set(groups[va])
                model,card,scores=fit_run(source,f'fold_{fold}',config,[rows[i] for i in tr],[rows[i] for i in va],device,tokenizer,args)
                oof[va]=scores;cards.append(card);release(model,device)
                fold_assign.extend({'post_id':rows[i]['post_id'],'fold':fold} for i in va)
                write_csv(OUT/source/'cv_fold_ids.csv',fold_assign)
            threshold,_=tune_threshold(y,oof)
            result={'source':source,'selected':name,'folds':cards,'threshold':threshold,
                'oof_metrics':metrics(y,oof,threshold,[r['post_id'] for r in rows]),
                'epochs_for_final':max(1,int(np.floor(np.median([c['best_epoch'] for c in cards])+.5))),
                'fold_f1_mean':float(np.mean([c['validation_f1_tuned'] for c in cards])),
                'fold_f1_std':float(np.std([c['validation_f1_tuned'] for c in cards]))}
            dump_json(cv_file,result)
            write_csv(OUT/source/'oof.csv',[{'post_id':r['post_id'],'label':int(r['priority']),'score':float(s)} for r,s in zip(rows,oof)])
        if args.stage in ['final','all']:
            cv=json.loads(cv_file.read_text());export=ROOT/'models'/f'urgency_minilm_{source}';export.mkdir(parents=True,exist_ok=True)
            model,card,_=fit_run(source,'final_fit',config,rows,[],device,tokenizer,args,fixed_epochs=cv['epochs_for_final'],export=export)
            threshold=cv['threshold'];dump_json(export/'decision_config.json',{'threshold':threshold,'max_length':args.max_length,
                'source':source,'labels':{'0':'NOT_URGENT','1':'URGENT'},'preprocessing':'prepare_data.clean_text NFC HTML whitespace',
                'seed':args.seed,'base_revision':ENCODER_REV,'cv_config':name,'final_epochs':cv['epochs_for_final']})
            comparisons=[]
            for target,test_rows in testing.items():
                enc=encode(tokenizer,[r['text_clean'] for r in test_rows],args.max_length)
                scores=predict(model,tokenizer,enc,device,args.batch_size*2);labels=np.array([int(r['priority']) for r in test_rows])
                result={'train_source':source,'test_source':target,'model':'finetuned_minilm','threshold':threshold,
                    **metrics(labels,scores,threshold,[r['post_id'] for r in test_rows])}
                result['default_threshold_metrics']=metrics(labels,scores,.5,[r['post_id'] for r in test_rows]);comparisons.append(result)
                write_csv(OUT/source/f'predictions_{target}.csv',[{'post_id':r['post_id'],'label':int(r['priority']),
                    'predicted':int(s>=threshold),'score':float(s),'text':r['text_clean']} for r,s in zip(test_rows,scores)])
                print('FINAL TEST',json.dumps(result),flush=True)
            dump_json(OUT/source/'test_results.json',comparisons)
            # Verify the exported directory can reload without its original cache or network.
            reloaded=AutoModelForSequenceClassification.from_pretrained(export,local_files_only=True).to(device)
            local_tokenizer=AutoTokenizer.from_pretrained(export,local_files_only=True)
            texts=['I cannot submit my assignment and the deadline is tonight. Please help!','Thank you for the helpful explanation.']
            enc=encode(local_tokenizer,texts,args.max_length);a=predict(model,tokenizer,enc,device,2);b=predict(reloaded,local_tokenizer,enc,device,2)
            assert np.allclose(a,b,atol=1e-6)
            dump_json(OUT/source/'export_check.json',{'passed':True,'max_score_difference':float(np.max(np.abs(a-b))),
                'examples':[{'text':t,'score':float(s),'predicted':int(s>=threshold)} for t,s in zip(texts,b)]})
            (export/'model_card.md').write_text(f'# Fine-tuned MiniLM urgency classifier\n\nSource: {source}. English only.\n'
                f'Base revision: {ENCODER_REV}. Configuration: {name}.\n'
                f'Full encoder and classification head trained for {cv["epochs_for_final"]} epochs.\n'
                'Load this directory locally with AutoTokenizer and AutoModelForSequenceClassification.\n'
                'Apply decision_config.json threshold to softmax(logits)[:,1].\n'
                'Previously inspected test sets are exploratory comparisons.\n',encoding='utf-8')
            release(reloaded,device);release(model,device)
    dump_json(OUT/'environment.json',{'python':sys.version,'torch':torch.__version__,'device':device})

if __name__=='__main__':main()
