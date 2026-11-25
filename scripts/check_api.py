"""Compare served pipelines with saved predictions and exercise the review-aware batch route."""
import csv,json,os,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URL=os.environ.get('PROJECT_APP_URL','http://127.0.0.1:8765')
info=json.load(urllib.request.urlopen(URL+'/api/info',timeout=20))
def post(path,data):
    request=urllib.request.Request(URL+path,json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(request,timeout=90))
checks=[]
for key in info['models']:
    source,model=key.split('__')
    if model in ['minilm_real_control','minilm_custom_trial']:
        arm='real_only' if model=='minilm_real_control' else json.loads((ROOT/'reports/final_minilm_trial/selection.json').read_text())['selected_custom_configuration']
        example=json.loads((ROOT/f'reports/final_minilm_trial/{arm}_seed42/standalone_check.json').read_text())['examples'][0]
        expected={'text':example['text'],'score':example['score'],'predicted':example['binary_priority']}
    elif model=='reliable_deberta':
        example=json.loads((ROOT/'reports/reliability/deberta_real_only/standalone_check.json').read_text())['examples'][0]
        expected={'text':example['text'],'score':example['score'],'predicted':example['binary_priority']}
    elif model in ['finetuned_minilm','minilm_tfidf_blend']:
        category='fine_tuning' if model=='finetuned_minilm' else 'ensembles'
        example=json.loads((ROOT/f'reports/{category}/{source}/standalone_check.json').read_text())['examples'][0]
        expected={'text':example['text'],'score':example['score'],'predicted':int(example['urgent'])}
    else:
        with (ROOT/f'reports/experiments/predictions_{source}_stanford_{model}.csv').open() as handle:
            expected=next(csv.DictReader(handle))
    actual=post('/api/predict',{'texts':[expected['text']],'model_key':key})['items'][0]
    delta=abs(actual['score']-float(expected['score']))
    assert actual['priority']==int(expected['predicted']),(key,'label mismatch')
    assert delta<1e-5,(key,delta)
    checks.append({'model':key,'same_label_as_saved_prediction':True,'score_difference':delta})
if 'joint__reliable_deberta' in info['models']:
    example=json.loads((ROOT/'reports/reliability/deberta_real_only/standalone_check.json').read_text())['examples'][0]
    items=post('/api/predict',{'texts':[example['text']]*9,'model_key':'joint__reliable_deberta'})['items']
    assert len(items)==9 and all(abs(item['score']-example['score'])<1e-5 for item in items)
    assert all(item['action']==example['action'] for item in items)
    checks.append({'deberta_multiple_inference_batches':True,'post_count':9,'action_preserved':True})
batch=post('/api/batch',{'csv_text':'text\n"Thank you, this was helpful."\n"I cannot submit my assignment. Please help!"\n'})
assert len(batch['items'])==2
checks.append({'csv_quoted_comma':True})
(ROOT/'reports/experiments/api_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(f"PASS: all {len(info['models'])} served models match saved test/standalone predictions, quoted CSV passed.")
