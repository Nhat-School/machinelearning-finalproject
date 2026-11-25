"""Local inference server. Loads fitted artifacts; never trains during prediction."""
from __future__ import annotations
import csv
import io
import json
import re
import sys
import threading
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_data import clean_text

app=FastAPI(title='Forum Priority',version='1.0')
cache={};encoder_cache={};lock=threading.Lock()

class PredictRequest(BaseModel):
    texts:list[str]=Field(min_length=1,max_length=500)
    model_key:str|None=None

class BatchRequest(BaseModel):
    csv_text:str=Field(max_length=2_000_000)
    model_key:str|None=None

def catalog():
    cards=json.loads((ROOT/'reports/experiments/cv_results.json').read_text())
    result={f"{c['source']}__{c['model']}":c for c in cards if c['name']!='dummy'}
    trial=ROOT/'reports/final_minilm_trial'
    trial_arms=[('real_only','minilm_real_control')]
    if (trial/'selection.json').exists():
        chosen=json.loads((trial/'selection.json').read_text())['selected_custom_configuration']
        trial_arms.append((chosen,'minilm_custom_trial'))
    for arm,key in trial_arms:
        folder=ROOT/'models'/f'urgency_final_minilm_{arm}_seed42'
        if (folder/'decision_config.json').exists() and (trial/f'{arm}_seed42/standalone_check.json').exists():
            settings=json.loads((folder/'decision_config.json').read_text())
            result['joint__'+key]={'source':'joint','model':key,'name':'final_minilm','representation':'reliable',
                'folder':str(folder),'max_length':settings['max_length'],**settings['policies']['mixed']}
    reliable=ROOT/'models/urgency_deberta_real_only'
    if (reliable/'decision_config.json').exists() and (ROOT/'reports/reliability/deberta_real_only/standalone_check.json').exists():
        settings=json.loads((reliable/'decision_config.json').read_text())
        result['joint__reliable_deberta']={'source':'joint','model':'reliable_deberta','name':'deberta',
            'representation':'reliable','folder':str(reliable),'max_length':settings['max_length'],**settings['policies']['mixed']}
    for source in ['upenn','stanford']:
        hybrid=ROOT/'models'/f'urgency_hybrid_{source}'
        if (hybrid/'decision_config.json').exists():
            decision=json.loads((hybrid/'decision_config.json').read_text())
            result[f'{source}__minilm_tfidf_blend']={'source':source,'model':'minilm_tfidf_blend','name':'hybrid',
                'representation':'hybrid','folder':str(hybrid),**decision}
        folder=ROOT/'models'/f'urgency_minilm_{source}'
        if (folder/'decision_config.json').exists():
            decision=json.loads((folder/'decision_config.json').read_text())
            result[f'{source}__finetuned_minilm']={'source':source,'model':'finetuned_minilm','name':'transformer',
                'representation':'finetuned','threshold':decision['threshold'],'folder':str(folder),
                'max_length':decision['max_length']}
    return result

def predict(texts,key):
    deployment=json.loads((ROOT/'models/deployment.json').read_text())
    selection=ROOT/'models/product_selection.json'
    default=json.loads(selection.read_text())['model_key'] if selection.exists() else f"{deployment['source']}__{deployment['model']}"
    key=key or default
    if key not in catalog():raise HTTPException(400,'Mô hình không hợp lệ.')
    cleaned=[clean_text(text) for text in texts]
    if any(len(text)>12000 for text in texts):raise HTTPException(422,'Mỗi bài tối đa 12.000 ký tự.')
    if any(not re.search('[a-zA-Z]',text) for text in cleaned):
        raise HTTPException(422,'Hãy nhập nội dung bài đăng bằng tiếng Anh, có ít nhất một chữ cái.')
    with lock:
        descriptor=catalog()[key]
        if descriptor['representation']=='reliable':
            import torch
            from transformers import AutoTokenizer,AutoModelForSequenceClassification
            if key not in cache:
                cache[key]={'model':AutoModelForSequenceClassification.from_pretrained(descriptor['folder'],local_files_only=True).eval(),
                            'tokenizer':AutoTokenizer.from_pretrained(descriptor['folder'],local_files_only=True,use_fast=False)}
            bundle=cache[key]
            chunks=[]
            with torch.inference_mode():
                for start in range(0,len(cleaned),4):
                    inputs=bundle['tokenizer'](cleaned[start:start+4],padding=True,truncation=True,max_length=descriptor['max_length'],return_tensors='pt')
                    chunks.append(torch.softmax(bundle['model'](**inputs).logits.float(),dim=-1)[:,1].numpy())
            raw=np.concatenate(chunks)
            raw=np.clip(raw,1e-6,1-1e-6);z=descriptor['coefficient']*np.log(raw/(1-raw))+descriptor['intercept']
            values=1/(1+np.exp(-np.clip(z,-40,40)));threshold=descriptor['threshold'];items=[]
            for i,(text,value) in enumerate(zip(texts,values)):
                priority=int(value>=threshold);automatic=value<=min(descriptor['review_low'],threshold) or value>=max(descriptor['review_high'],threshold)
                items.append({'index':i,'text':text,'priority':priority,'score':float(value),'threshold':threshold,'terms':[],
                              'action':('PRIORITY' if priority else 'NORMAL') if automatic else 'REVIEW','score_type':'calibrated_urgency_estimate'})
            return {'model_key':key,'items':items,'language_scope':'English student forum posts'}
        if descriptor['representation'] in ['finetuned','hybrid']:
            import torch
            from transformers import AutoTokenizer,AutoModelForSequenceClassification
            if key not in cache:
                folder=descriptor['folder']
                cache[key]={'card':descriptor,'model':AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True).eval(),
                            'tokenizer':AutoTokenizer.from_pretrained(folder,local_files_only=True)}
                if descriptor['representation']=='hybrid':cache[key]['linear']=joblib.load(Path(folder)/'tfidf_logreg.joblib')['model']
            bundle=cache[key]
            inputs=bundle['tokenizer'](cleaned,padding=True,truncation=True,max_length=descriptor['max_length'],return_tensors='pt')
            with torch.inference_mode():values=torch.softmax(bundle['model'](**inputs).logits.float(),dim=-1)[:,1].numpy()
            if descriptor['representation']=='hybrid':
                values=descriptor['minilm_weight']*values+descriptor['tfidf_weight']*bundle['linear'].predict_proba(cleaned)[:,1]
            return {'model_key':key,'items':[{'index':i,'text':text,'priority':int(value>=descriptor['threshold']),
                'score':float(value),'threshold':descriptor['threshold'],'terms':[],'score_type':'estimated_probability'}
                for i,(text,value) in enumerate(zip(texts,values))],'language_scope':'English'}
        if key not in cache:cache[key]=joblib.load(ROOT/'models'/f'{key}.joblib')
        bundle=cache[key];model=bundle['model'];card=bundle['card']
        if card['representation']=='minilm':
            if key not in encoder_cache:
                from sentence_transformers import SentenceTransformer
                encoder_cache[key]=SentenceTransformer(str(ROOT/card['encoder']['snapshot']),device='cpu',local_files_only=True)
            x=encoder_cache[key].encode(cleaned,normalize_embeddings=True,show_progress_bar=False)
        else:x=cleaned
        values=model.predict_proba(x)[:,1] if hasattr(model,'predict_proba') else model.decision_function(x)
        output=[]
        for i,(text,value) in enumerate(zip(texts,values)):
            terms=[]
            if card['representation']=='tfidf':
                vector=model.named_steps['tfidf'].transform([cleaned[i]])
                classifier=model.named_steps['classifier']
                coefficients=(classifier.coef_[0] if hasattr(classifier,'coef_') else
                              classifier.feature_log_prob_[1]-classifier.feature_log_prob_[0])
                names=model.named_steps['tfidf'].get_feature_names_out()
                contribution=vector.data*coefficients[vector.indices]
                order=np.argsort(-np.abs(contribution))[:6]
                terms=[{'term':str(names[vector.indices[j]]),'contribution':float(contribution[j])} for j in order]
            output.append({'index':i,'text':text,'priority':int(value>=card['threshold']),
                'score':float(value),'threshold':card['threshold'],'terms':terms,
                'score_type':'decision_margin' if card['name']=='svm' else 'estimated_probability'})
    return {'model_key':key,'items':output,'language_scope':'English'}

@app.get('/api/info')
def info():
    selected=ROOT/'models/product_selection.json'
    return {'deployment':json.loads((ROOT/'models/deployment.json').read_text()),
            'default_model_key':json.loads(selected.read_text())['model_key'] if selected.exists() else None,'models':catalog(),
            'audit':json.loads((ROOT/'reports/data_audit.json').read_text()),
            'splits':json.loads((ROOT/'reports/experiments/split_info.json').read_text())}

@app.get('/api/results')
def results():
    rows=json.loads((ROOT/'reports/experiments/results.json').read_text())
    extra=ROOT/'reports/fine_tuning/results.json'
    hybrid=ROOT/'reports/ensembles/results.json'
    reliable=ROOT/'reports/reliability/deberta_real_only/results.json'
    joint=[{**r,'train_source':'joint','model':'reliable_deberta'} for r in json.loads(reliable.read_text()) if r['policy']=='mixed'] if reliable.exists() else []
    trial=ROOT/'reports/final_minilm_trial'
    trial_results=trial/('results.json' if (trial/'results.json').exists() else 'control_results.json')
    if trial_results.exists():
        for record in json.loads(trial_results.read_text()):
            if record['test_source'] not in ['upenn','stanford']:continue
            key={'real_only':'minilm_real_control','custom_ce':'minilm_custom_ce','custom_pairwise':'minilm_custom_pairwise'}[record['arm']]
            joint.append({**record,'train_source':'joint','model':key+('_seed43' if record['seed']==43 else '')})
    return rows+(json.loads(extra.read_text()) if extra.exists() else [])+(json.loads(hybrid.read_text()) if hybrid.exists() else [])+joint

@app.post('/api/predict')
def predict_endpoint(body:PredictRequest):return predict(body.texts,body.model_key)

@app.post('/api/batch')
def batch(body:BatchRequest):
    reader=csv.DictReader(io.StringIO(body.csv_text.lstrip('\ufeff')))
    column=next((c for c in ['text','post_text','text_clean'] if c in (reader.fieldnames or [])),None)
    if not column:raise HTTPException(422,'CSV cần cột text, post_text hoặc text_clean.')
    texts=[]
    for row in reader:
        if len(texts)>=500:raise HTTPException(422,'Mỗi lần tối đa 500 bài.')
        texts.append(row.get(column) or '')
    if not texts:raise HTTPException(422,'CSV không có bài đăng.')
    return predict(texts,body.model_key)

app.mount('/static',StaticFiles(directory=ROOT/'app/static'),name='static')

@app.get('/')
def index():return FileResponse(ROOT/'app/static/index.html')
