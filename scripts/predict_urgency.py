"""Predict from a self-contained exported MiniLM directory, without network access."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from prepare_data import clean_text

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--text',required=True)
    args=parser.parse_args();path=Path(args.model);decision=json.loads((path/'decision_config.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).eval()
    inputs=tokenizer(clean_text(args.text),truncation=True,max_length=decision['max_length'],return_tensors='pt')
    with torch.inference_mode():score=float(torch.softmax(model(**inputs).logits,dim=-1)[0,1])
    print(json.dumps({'text':args.text,'score':score,'threshold':decision['threshold'],
        'urgent':score>=decision['threshold'],'label':'URGENT' if score>=decision['threshold'] else 'NOT_URGENT'},indent=2))

if __name__=='__main__':main()
