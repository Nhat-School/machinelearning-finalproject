"""Offline binary urgency estimate plus an explicit manual-review policy."""
import argparse,html,json,math,re,unicodedata
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

def clean(text):
    pattern=r'</?(?:p|br|div|span|a|b|i|strong|em|ul|ol|li|blockquote|pre|code|hr)(?:\s[^<>]*?)?\s*/?>'
    return ' '.join(re.sub(pattern,' ',unicodedata.normalize('NFC',html.unescape(text)),flags=re.I).split())

def classify(folder,text,domain='mixed'):
    folder=Path(folder);settings=json.loads((folder/'decision_config.json').read_text());policy=settings['policies'][domain]
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,use_fast=False)
    model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True).eval()
    encoded=tokenizer(clean(text),truncation=True,max_length=settings['max_length'],return_tensors='pt')
    with torch.inference_mode():raw=float(torch.softmax(model(**encoded).logits.float(),dim=-1)[0,1])
    raw=max(1e-6,min(1-1e-6,raw));z=policy['coefficient']*math.log(raw/(1-raw))+policy['intercept'];p=1/(1+math.exp(-max(-40,min(40,z))))
    label=int(p>=policy['threshold']);automatic=(p<=min(policy['review_low'],policy['threshold']) or p>=max(policy['review_high'],policy['threshold']))
    return {'text':text,'score':p,'raw_score':raw,'threshold':policy['threshold'],'binary_priority':label,
            'action':('PRIORITY' if label else 'NORMAL') if automatic else 'REVIEW',
            'domain':domain,'scope':'English student forum posts','note':'An estimate, not a guarantee of correctness.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--domain',choices=['mixed'],default='mixed');parser.add_argument('--text',required=True)
    args=parser.parse_args();print(json.dumps(classify(args.model,args.text,args.domain),ensure_ascii=False,indent=2))
