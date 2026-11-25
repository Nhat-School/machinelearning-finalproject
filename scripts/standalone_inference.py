"""Self-contained inference; copy this file beside config.json and model.safetensors."""
import argparse,html,json,re,unicodedata
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

def clean_text(text):
    pattern=r"</?(?:p|br|div|span|a|b|i|strong|em|ul|ol|li|blockquote|pre|code|hr)(?:\s[^<>]*?)?\s*/?>"
    return ' '.join(re.sub(pattern,' ',unicodedata.normalize('NFC',html.unescape(text)),flags=re.I).split())

def classify(folder,text):
    folder=Path(folder);settings=json.loads((folder/'decision_config.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True).eval()
    encoded=tokenizer(clean_text(text),truncation=True,max_length=settings['max_length'],return_tensors='pt')
    with torch.inference_mode():score=float(torch.softmax(model(**encoded).logits.float(),dim=-1)[0,1])
    return {'text':text,'score':score,'threshold':settings['threshold'],'urgent':score>=settings['threshold'],
            'label':'URGENT' if score>=settings['threshold'] else 'NOT_URGENT'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=str(Path(__file__).resolve().parent));parser.add_argument('--text',required=True)
    args=parser.parse_args();print(json.dumps(classify(args.model,args.text),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
