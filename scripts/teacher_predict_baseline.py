"""Predict with one project TF-IDF or frozen-MiniLM classifier, preserving its saved threshold."""
import argparse,json,html,re,unicodedata
from pathlib import Path
import joblib


def clean_text(text):
    tag=re.compile(r'</?(?:p|br|div|span|a|b|i|strong|em|ul|ol|li|blockquote|pre|code|hr)(?:\s[^<>]*?)?\s*/?>',re.I)
    return ' '.join(tag.sub(' ',unicodedata.normalize('NFC',html.unescape(text))).split())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path);parser.add_argument('--text',required=True)
    args=parser.parse_args();bundle=joblib.load(args.model);card=bundle['card'];model=bundle['model']
    text=clean_text(args.text)
    if not text:raise ValueError('Enter a nonempty English learning-forum post.')
    if card['representation']=='minilm':
        from sentence_transformers import SentenceTransformer
        encoder_card=card['encoder']
        encoder=SentenceTransformer(encoder_card['model'],revision=encoder_card['revision'],device='cpu')
        x=encoder.encode([text],normalize_embeddings=True,show_progress_bar=False)
    else:x=[text]
    value=float(model.predict_proba(x)[0,1]) if hasattr(model,'predict_proba') else float(model.decision_function(x)[0])
    threshold=float(card['threshold'])
    print(json.dumps({'text':args.text,'model':card['model'],'source':card['source'],'score':value,
        'score_type':'decision_margin' if card['name']=='svm' else 'model_probability_estimate',
        'threshold':threshold,'binary_priority':int(value>=threshold),
        'note':'English student forum posts; a model estimate can be wrong.'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
