"""Predict offline from a folder containing MiniLM and TF-IDF Logistic Regression."""
import argparse,html,json,re,unicodedata
from pathlib import Path
import joblib,torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

def classify(folder,text):
    folder=Path(folder);settings=json.loads((folder/'decision_config.json').read_text())
    pattern=r'</?(?:p|br|div|span|a|b|i|strong|em|ul|ol|li|blockquote|pre|code|hr)(?:\s[^<>]*?)?\s*/?>'
    cleaned=' '.join(re.sub(pattern,' ',unicodedata.normalize('NFC',html.unescape(text)),flags=re.I).split())
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True).eval()
    encoded=tokenizer(cleaned,truncation=True,max_length=settings['max_length'],return_tensors='pt')
    with torch.inference_mode():fine=float(torch.softmax(model(**encoded).logits.float(),dim=-1)[0,1])
    linear=float(joblib.load(folder/'tfidf_logreg.joblib')['model'].predict_proba([cleaned])[0,1])
    score=settings['minilm_weight']*fine+settings['tfidf_weight']*linear;threshold=settings['threshold']
    return {'text':text,'score':score,'threshold':threshold,'urgent':score>=threshold,
            'label':'URGENT' if score>=threshold else 'NOT_URGENT','minilm_score':fine,'tfidf_score':linear}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--text',required=True);args=parser.parse_args()
    print(json.dumps(classify(args.model,args.text),ensure_ascii=False,indent=2))
