"""Run fixed context probes through the actual app; labels are assistant draft diagnostics."""
import argparse,csv,json,os,urllib.request
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/final_minilm_trial'
PROBES=[
 ('unresolved_upload',1,'My report is finished, but every upload attempt fails. It is due in an hour and I need assistance.'),
 ('resolved_upload',0,'The upload failure is fixed. I have checked the receipt and my report was accepted. No action is needed.'),
 ('longer_resolution',0,'I was unable to enter the quiz and asked for help this morning. Since then the administrator reset my access. I completed it successfully, so please close this request.'),
 ('problem_returned',1,'The quiz opened briefly after a reset, but now the same error is back. I still cannot start and the timer is running. Please investigate.'),
 ('denied_access',1,'I paid for the course, yet access is denied. Support has not replied and the assessed activity closes today.'),
 ('past_problem',0,'Yesterday I was locked out of the course. Today access has been restored and everything works. Thanks for resolving it.'),
 ('negative_statement',1,'It is not true that my login has been repaired. I remain locked out and need someone to restore access before the exam.'),
 ('no_remaining_issue',0,'There is no remaining login issue. I tested it again, entered the exam and finished without errors.'),
 ('quoted_problem',0,'The tutorial uses the sentence "I cannot log in" as an example of a support request. I am quoting the tutorial, not reporting a problem with my account.'),
 ('reference_example',0,'Here is an example error message for our documentation: "Submission failed. Contact your instructor." My own submission completed correctly.'),
 ('still_waiting',1,'I followed the suggested workaround. It changed nothing: my answers are still missing and I cannot finish the graded exercise.'),
 ('different_failure',1,'The original password issue is fixed, but the payment system has charged me twice. I need help recovering the extra charge.'),
 ('thank_you',0,'Thank you for answering my question. The explanation makes sense and I do not need any additional help.'),
 ('sharing_solution',0,'For classmates who saw the installation error: restarting the kernel fixed it for me. I am sharing a working solution.'),
 ('lost_work',1,'My saved project disappeared after the lab crashed. The submission window closes shortly. Can staff help recover it?'),
 ('received_feedback',0,'I have now received the missing feedback and the grade is correct. Please disregard my earlier complaint.'),
 ('vague',None,'Can someone help me with this?'),
 ('missing_reference',None,'The second one is wrong again.'),
 ('outside_weather',None,'What will the weather be like in Paris next weekend?'),
 ('outside_recipe',None,'Give me a recipe for chocolate cake.')]
def write_definition():
    path=OUT/'behavior_probe_definition.json';definition={'label_status':'assistant-authored draft expectations, not human-adjudicated gold',
        'use':'diagnostic only; never training, checkpoint selection, calibration, threshold selection or configuration selection',
        'probes':[{'id':key,'expected_binary_priority':label,'text':text,'expected_handling':'context needed or outside supported scope' if label is None else 'binary context diagnostic'} for key,label,text in PROBES]}
    if path.exists():assert json.loads(path.read_text())==definition,'Probe definition changed after being frozen.'
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(definition,indent=2)+'\n')
    return definition
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--exported-only',action='store_true');parser.add_argument('--control-only',action='store_true');args=parser.parse_args()
    definition=write_definition()
    if os.environ.get('FORUM_DEFINE_PROBES_ONLY')=='1':print('Frozen 20 probes; no model evaluated.');return
    url=os.environ.get('PROJECT_APP_URL','http://127.0.0.1:8765')
    if args.exported_only:
        keys=['joint__minilm_real_control']
        if (OUT/'selection.json').exists() and not args.control_only:keys.append('joint__minilm_custom_trial')
    else:
        info=json.load(urllib.request.urlopen(url+'/api/info',timeout=30))
        keys=[key for key in ['stanford__finetuned_minilm','joint__reliable_deberta','joint__minilm_real_control','joint__minilm_custom_trial'] if key in info['models']]
    results=[]
    with (ROOT/'Datasets/CustomForum/draft_examples.csv').open() as handle:custom=list(csv.DictReader(handle))
    assert not {r['text'] for r in custom}.intersection(text for _,_,text in PROBES)
    for key in keys:
        if args.exported_only:
            from final_minilm_trial import scores,calibrated
            from transformers import AutoTokenizer,AutoModelForSequenceClassification
            from finetune_minilm import encode
            arm='real_only' if key.endswith('real_control') else json.loads((OUT/'selection.json').read_text())['selected_custom_configuration']
            folder=ROOT/'models'/f'urgency_final_minilm_{arm}_seed42';settings=json.loads((folder/'decision_config.json').read_text());policy=settings['policies']['mixed']
            tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True);model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True).eval()
            raw=scores(model,tokenizer,encode(tokenizer,[r['text'] for r in definition['probes']],settings['max_length']),'cpu',4);values=calibrated(raw,policy)
            items=[{'priority':int(v>=.5),'score':float(v),'action':'NORMAL' if v<=.2 else ('PRIORITY' if v>=.8 else 'REVIEW')} for v in values];del model,tokenizer
        else:
            request=urllib.request.Request(url+'/api/predict',json.dumps({'model_key':key,'texts':[r['text'] for r in definition['probes']]}).encode(),headers={'Content-Type':'application/json'})
            items=json.load(urllib.request.urlopen(request,timeout=180))['items']
        rows=[{**probe,**item} for probe,item in zip(definition['probes'],items)]
        labelled=[r for r in rows if r['expected_binary_priority'] is not None]
        y=[r['expected_binary_priority'] for r in labelled];pred=[r['priority'] for r in labelled]
        result={'model_key':key,'label_status':definition['label_status'],'diagnostic_binary_count':len(labelled),
            'diagnostic_accuracy':float(accuracy_score(y,pred)),'diagnostic_f1':float(f1_score(y,pred)),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),'correct_binary_probes':sum(a==b for a,b in zip(y,pred)),
            'binary_errors':[r['id'] for r in labelled if r['priority']!=r['expected_binary_priority']],
            'unlabelled_probes_automatically_classified':[r['id'] for r in rows if r['expected_binary_priority'] is None and r.get('action')!='REVIEW'],
            'predictions':rows};results.append(result)
    filename='control_behavior_probe_results.json' if args.control_only else ('exported_behavior_probe_results.json' if args.exported_only else 'behavior_probe_results.json')
    (OUT/filename).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['model_key','correct_binary_probes','diagnostic_binary_count','binary_errors','unlabelled_probes_automatically_classified']} for r in results],indent=2))
if __name__=='__main__':main()
