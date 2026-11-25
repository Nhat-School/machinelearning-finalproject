"""Reproducible source-only model selection and two-source held-out evaluation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedKFold, StratifiedGroupKFold,
    train_test_split, GridSearchCV, cross_val_predict)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from prepare_data import dump_json, sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/experiments'
SEED=42
ENCODER='sentence-transformers/all-MiniLM-L6-v2'
ENCODER_REV='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL_KEYS=['tfidf_logreg','tfidf_svm','tfidf_nb','tfidf_dummy','minilm_logreg']

def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def scores(model,x):
    if hasattr(model,'predict_proba'):
        return model.predict_proba(x)[:,1]
    return model.decision_function(x)

def metrics(y,s,t,ids):
    p=np.asarray(s)>=t
    matrix=confusion_matrix(y,p,labels=[0,1])
    order=np.lexsort((np.asarray(ids),-np.asarray(s)))
    k=min(20,len(y));found=int(np.asarray(y)[order[:k]].sum())
    return dict(precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p,zero_division=0)),
        f1=float(f1_score(y,p,zero_division=0)),macro_f1=float(f1_score(y,p,average='macro')),
        accuracy=float(accuracy_score(y,p)),average_precision=float(average_precision_score(y,s)),
        roc_auc=float(roc_auc_score(y,s)),precision_at_20=found/k,recall_at_20=found/int(np.sum(y)),
        top20_found=found,confusion_matrix=matrix.tolist(),test_size=len(y),positive_count=int(np.sum(y)))

def tune_threshold(y,s):
    precision,recall,thresholds=precision_recall_curve(y,s)
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    # Highest threshold among exact F1 ties; no target test labels are used.
    index=np.flatnonzero(np.isclose(f,np.max(f),atol=1e-12,rtol=0))[-1]
    return float(thresholds[index]),float(f[index])

def embeddings(data):
    import torch
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    snapshot=snapshot_download(ENCODER,revision=ENCODER_REV,
        allow_patterns=['*.json','*.txt','*.safetensors','1_Pooling/*'])
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(SEED)
    encoder=SentenceTransformer(snapshot,device=device)
    result={};info={'model':ENCODER,'revision':ENCODER_REV,'device':device,
        'snapshot':str(Path(snapshot).relative_to(ROOT)),'max_seq_length':encoder.max_seq_length,
        'dimension':(encoder.get_embedding_dimension() if hasattr(encoder,'get_embedding_dimension')
                     else encoder.get_sentence_embedding_dimension()),'sources':{}}
    for source,rows in data.items():
        path=ROOT/'.cache/embeddings'/f'{source}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        fingerprint=sha256(ROOT/f'data/processed/{source}.csv')
        if path.exists():
            cached=np.load(path)
            if str(cached['fingerprint'])==fingerprint and str(cached['revision'])==ENCODER_REV:
                result[source]=cached['vectors'];info['sources'][source]={'cache_used':True};continue
        texts=[row['text_clean'] for row in rows]
        start=time.perf_counter()
        lengths=[]
        for i in range(0,len(texts),128):
            batch=encoder.tokenizer(texts[i:i+128],truncation=False,add_special_tokens=True)['input_ids']
            lengths.extend(map(len,batch))
        vectors=encoder.encode(texts,batch_size=64,normalize_embeddings=True,show_progress_bar=True).astype(np.float32)
        elapsed=time.perf_counter()-start
        np.savez_compressed(path,vectors=vectors,fingerprint=fingerprint,revision=ENCODER_REV)
        result[source]=vectors
        info['sources'][source]={'cache_used':False,'encoding_seconds':elapsed,
            'truncated_posts':sum(n>encoder.max_seq_length for n in lengths),'total':len(texts),
            'tokens_p95':float(np.percentile(lengths,95)),'tokens_max':max(lengths)}
        dump_json(OUT/'encoder_info.json',info)
    # Preserve original measured token/encoding stats when embeddings are reused.
    if (OUT/'encoder_info.json').exists():
        previous=json.loads((OUT/'encoder_info.json').read_text())
        for source,entry in info['sources'].items():
            if entry.get('cache_used') and source in previous.get('sources',{}):
                info['sources'][source]=previous['sources'][source]
    dump_json(OUT/'encoder_info.json',info)
    return result,info

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=MODEL_KEYS,default=MODEL_KEYS,
                        help='Train only these existing methods; default preserves the original full experiment.')
    parser.add_argument('--plan-only',action='store_true',help='Print the selected methods without preparing data or fitting anything.')
    args=parser.parse_args();requested=set(args.models)
    if args.plan_only:
        print(json.dumps({'models':[key for key in MODEL_KEYS if key in requested],
                          'representations':[rep for rep in ['tfidf','minilm'] if any(key.startswith(rep+'_') for key in requested)],
                          'training_performed':False},indent=2));return
    OUT.mkdir(parents=True,exist_ok=True)
    (ROOT/'models').mkdir(exist_ok=True)
    data={source:read_csv(ROOT/f'data/processed/{source}.csv') for source in ['upenn','stanford']}
    splits={};folds={};split_rows=[];split_info={}
    for source,rows in data.items():
        y=np.array([int(row['priority']) for row in rows]);indices=np.arange(len(rows))
        if source=='upenn':
            groups=np.array([row['source_id'] for row in rows])
            assert all(groups)
            train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(indices,y,groups))
            folds[source]=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED).split(train,y[train],groups[train]))
            assert not (set(groups[train]) & set(groups[test]))
            policy='GroupShuffleSplit by UPenn student id; 3-fold StratifiedGroupKFold for tuning'
        else:
            train,test=train_test_split(indices,test_size=.2,random_state=SEED,stratify=y)
            folds[source]=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED).split(train,y[train]))
            policy='Stratified hold-out and 3-fold StratifiedKFold; author/thread IDs unavailable'
        for a,b in folds[source]:
            assert len(set(y[train[a]]))==2 and len(set(y[train[b]]))==2
        splits[source]=(train,test)
        split_info[source]={'policy':policy,'train':len(train),'test':len(test),
            'train_positives':int(y[train].sum()),'test_positives':int(y[test].sum()),
            'processed_sha256':sha256(ROOT/f'data/processed/{source}.csv')}
        train_set=set(train)
        assignments={int(i):('train' if i in train_set else 'test') for i in indices}
        split_rows.extend({'source':source,'post_id':row['post_id'],'split':assignments[i]} for i,row in enumerate(rows))
    write_csv(OUT/'split_ids.csv',split_rows);dump_json(OUT/'split_info.json',split_info)
    dump_json(OUT/'config.json',{'seed':SEED,'folds':3,'test_fraction':.2,'tfidf':{
        'ngram_range':[1,2],'min_df':2,'max_df':.98,'max_features':40000,'sublinear_tf':True},
        'threshold_rule':'Maximize positive F1 on source-only out-of-fold predictions; ties use largest threshold',
        'deployment_rule':'Best tuned OOF F1 among Stanford-trained models; independent test scores not used',
        'encoder_revision':ENCODER_REV,'requested_models':[key for key in MODEL_KEYS if key in requested],
        'data_hashes':{s:i['processed_sha256'] for s,i in split_info.items()}})
    results=[];cv_records=[]
    # Train TF-IDF first, so useful artifacts exist before downloading the optional encoder.
    embedded=None;encoder_info=None
    for representation in ['tfidf','minilm']:
        if not any(key.startswith(representation+'_') for key in requested):continue
        if representation=='minilm':
            embedded,encoder_info=embeddings(data)
        for source,rows in data.items():
            train,_=splits[source]
            y=np.array([int(row['priority']) for row in rows]);texts=np.array([row['text_clean'] for row in rows],dtype=object)
            x=texts if representation=='tfidf' else embedded[source]
            if representation=='tfidf':
                candidates=[('logreg',LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=2000,random_state=SEED),{'classifier__C':[.3,1.,3.]}),
                    ('svm',LinearSVC(class_weight='balanced',max_iter=10000,random_state=SEED),{'classifier__C':[.3,1.,3.]}),
                    ('nb',MultinomialNB(),{'classifier__alpha':[.1,.5,1.]}),
                    ('dummy',DummyClassifier(strategy='prior'),{})]
            else:
                candidates=[('logreg',LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=2000,random_state=SEED),{'C':[.3,1.,3.,10.]})]
            for name,classifier,grid in candidates:
                slug=f'{representation}_{name}'
                if slug not in requested:continue
                print(f'TRAIN {source} / {slug}',flush=True)
                if representation=='tfidf':
                    model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,max_features=40000,
                        sublinear_tf=True,strip_accents='unicode',dtype=np.float32)),('classifier',classifier)])
                else:model=classifier
                start=time.perf_counter()
                search=GridSearchCV(model,grid,cv=folds[source],scoring='f1',n_jobs=1,refit=True,return_train_score=False)
                search.fit(x[train],y[train]);model=search.best_estimator_
                method='decision_function' if name=='svm' else 'predict_proba'
                oof=cross_val_predict(clone(model),x[train],y[train],cv=folds[source],method=method,n_jobs=1)
                oof=oof if name=='svm' else oof[:,1]
                default=0. if name=='svm' else .5
                threshold,oof_f1=tune_threshold(y[train],oof)
                if name=='dummy': threshold=default;oof_f1=float(f1_score(y[train],oof>=threshold,zero_division=0))
                elapsed=time.perf_counter()-start
                card={'source':source,'model':slug,'representation':representation,'name':name,
                    'threshold':threshold,'default_threshold':default,'cv_mean_f1':float(search.best_score_),
                    'cv_std_f1':float(search.cv_results_['std_test_score'][search.best_index_]),
                    'oof_tuned_f1':oof_f1,'best_params':search.best_params_,'training_seconds':elapsed}
                if representation=='minilm':card['encoder']=encoder_info
                cv_records.append(card)
                bundle={'model':model,'card':card}
                joblib.dump(bundle,ROOT/'models'/f'{source}__{slug}.joblib',compress=3)
                write_csv(OUT/f'oof_{source}_{slug}.csv',[{'post_id':rows[i]['post_id'],'label':int(y[i]),'score':float(s)} for i,s in zip(train,oof)])
                for target,target_rows in data.items():
                    _,test=splits[target]
                    test_x=np.array([row['text_clean'] for row in target_rows],dtype=object)[test] if representation=='tfidf' else embedded[target][test]
                    test_y=np.array([int(row['priority']) for row in target_rows])[test]
                    ids=[target_rows[i]['post_id'] for i in test]
                    start=time.perf_counter();s=scores(model,test_x);infer_seconds=time.perf_counter()-start
                    record={'train_source':source,'test_source':target,'model':slug,'threshold':threshold,
                        'training_seconds':elapsed,'inference_seconds':infer_seconds,**metrics(test_y,s,threshold,ids)}
                    record['default_threshold_metrics']=metrics(test_y,s,default,ids)
                    results.append(record)
                    prediction_rows=[]
                    for i,label,score in zip(test,test_y,s):
                        row=target_rows[i];pred=int(score>=threshold)
                        prediction_rows.append({'post_id':row['post_id'],'source':target,'source_id':row['source_id'],
                            'course_type':row['source_course_type'],'text':row['text_clean'],'urgency':row['urgency'],
                            'label':int(label),'predicted':pred,'score':float(score),
                            'error':'FN' if label==1 and pred==0 else 'FP' if label==0 and pred==1 else ''})
                    write_csv(OUT/f'predictions_{source}_{target}_{slug}.csv',prediction_rows)
                    print(f'  TEST {target}: P={record["precision"]:.3f} R={record["recall"]:.3f} F1={record["f1"]:.3f}',flush=True)
                dump_json(OUT/'results.json',results);dump_json(OUT/'cv_results.json',cv_records)
    eligible=[r for r in cv_records if r['source']=='stanford' and r['name']!='dummy']
    selected=max(eligible or [r for r in cv_records if r['source']=='stanford'],key=lambda r:r['oof_tuned_f1'])
    dump_json(ROOT/'models/deployment.json',{'source':selected['source'],'model':selected['model'],
        'file':f"models/{selected['source']}__{selected['model']}.joblib",'selection_rule':'Stanford source training OOF F1 only',
        'oof_tuned_f1':selected['oof_tuned_f1'],'threshold':selected['threshold']})
    dump_json(OUT/'training_environment.json',{'python':sys.version,'sklearn':sklearn.__version__,
        'numpy':np.__version__,'platform':platform.platform(),'seed':SEED})
    print('DEPLOYMENT',selected['source'],selected['model'],flush=True)

if __name__=='__main__':main()
