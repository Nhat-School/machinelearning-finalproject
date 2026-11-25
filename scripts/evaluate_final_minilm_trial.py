"""Freeze custom configuration using validation only, then evaluate the completed trial."""
import argparse,csv,json,shutil,subprocess,sys,os,math
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score,brier_score_loss
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from final_minilm_trial import BASE,ROOT,load_real,load_custom,read,dump,run_name,scores,calibrated
from finetune_minilm import encode
from train import metrics,write_csv

def load(path):return json.loads(path.read_text())
def weak_authorized():
    path=BASE/'synthetic_authorization.json'
    return path.exists() and load(path).get('user_authorized_synthetic_trial',False)
def context_metrics(records,labelled):
    lookup={r['post_id']:r for r in labelled};use=[r for r in records if r['post_id'] in lookup]
    if not use or len({lookup[r['post_id']]['priority'] for r in use})!=2:raise ValueError('Context validation requires both labelled classes.')
    y=np.array([int(lookup[r['post_id']]['priority']) for r in use]);p=np.array([float(r['score']) for r in use])
    result=metrics(y,p,.5,[r['post_id'] for r in use]);result['balanced_accuracy']=float(balanced_accuracy_score(y,p>=.5))
    grouped={}
    for r in use:
        if r['case_type'] in ['blocked','resolved','still_blocked_update']:grouped.setdefault(r['scenario_group'],{})[r['case_type']]=int(float(r['score'])>=.5)
    groups=[r for r in grouped.values() if {'blocked','resolved','still_blocked_update'}<=set(r)]
    result['complete_scenario_groups']=len(groups)
    result['all_three_states_correct']=sum(r['blocked']==1 and r['resolved']==0 and r['still_blocked_update']==1 for r in groups)
    result['label_origin']=sorted({r['training_label_origin'] for r in labelled})
    result['interpretation']='Synthetic scenario diagnostic, not independently verified real-world accuracy.' if any('draft' in origin for origin in result['label_origin']) else 'Human-reviewed custom scenario evaluation.'
    return result
def validation_card(arm,seed):
    folder=BASE/run_name(arm,seed);real=load(folder/'validation_results.json')
    return {'arm':arm,'seed':seed,'mean_real_validation_ap':float(np.mean([r['average_precision'] for r in real])),
            'real_validation':real,'context_validation':context_metrics(read(folder/'custom_validation_diagnostic.csv'),load_custom(weak_authorized())['validation'])}
def select():
    reference=validation_card('real_only',42);candidates=[]
    for arm in ['custom_ce','custom_pairwise']:
        card=validation_card(arm,42);card['passes_real_ap_guard']=card['mean_real_validation_ap']>=reference['mean_real_validation_ap']-.03;candidates.append(card)
    eligible=[r for r in candidates if r['passes_real_ap_guard']]
    chosen=max(eligible or candidates,key=lambda r:(r['context_validation']['balanced_accuracy'],r['mean_real_validation_ap']))
    selection={'status':'frozen before new historical test evaluation','primary_seed':42,'confirmation_seed':43,
        'reference':reference,'candidates':candidates,'selected_custom_configuration':chosen['arm'],
        'recommendation_eligible_on_primary_validation':bool(eligible),
        'selection_rule':'Require mean real validation AP >= control minus 0.03; maximize custom validation balanced accuracy, tie real AP. If guard fails, keep real-only recommendation.',
        'custom_label_use':'explicitly authorized assistant-draft weak supervision' if weak_authorized() else 'human reviewed labels only',
        'test_metrics_used_for_selection':False}
    dump(BASE/'selection.json',selection);print(json.dumps(selection,indent=2))
def evaluate():
    selection=load(BASE/'selection.json');chosen=selection['selected_custom_configuration']
    runs=[('real_only',42),('custom_ce',42),('custom_pairwise',42),('real_only',43),(chosen,43)]
    # Confirm the frozen selection and record second-seed validation before reading any test prediction.
    confirmation={arm:validation_card(arm,43) for arm in ['real_only',chosen]}
    reference=confirmation['real_only'];custom=confirmation[chosen]
    confirmation_guard=custom['mean_real_validation_ap']>=reference['mean_real_validation_ap']-.03
    context_improves=custom['context_validation']['balanced_accuracy']>reference['context_validation']['balanced_accuracy']
    primary_context_improves=next(r for r in selection['candidates'] if r['arm']==chosen)['context_validation']['balanced_accuracy']>selection['reference']['context_validation']['balanced_accuracy']
    preferred=chosen if selection['recommendation_eligible_on_primary_validation'] and confirmation_guard and context_improves and primary_context_improves else 'real_only'
    frozen={'selected_custom_configuration':chosen,'preferred_trial_arm':preferred,'deployment_seed':42,
        'basis':'validation only; both seeds must pass the real AP guard and improve context validation for a custom recommendation',
        'second_seed':confirmation,'confirmation_guard_passed':confirmation_guard,'context_improved_on_both_seeds':context_improves and primary_context_improves,
        'test_metrics_used':False,'custom_label_use':selection['custom_label_use']}
    dump(BASE/'frozen_recommendation.json',frozen)
    all_results=score_runs(runs)
    dump(BASE/'results.json',all_results)
    summarize_seeds(all_results,['real_only',chosen],BASE/'two_seed_summary.json')
    print('Final trial evaluated; previous model folders preserved.',flush=True)

def score_runs(runs):
    # Draft labels may be inspected as synthetic diagnostics without authorizing training on them.
    all_results=[];real=load_real();custom_labels=load_custom(True);drafts=read(ROOT/'Datasets/CustomForum/draft_examples.csv')
    for arm,seed in runs:
        name=run_name(arm,seed);directory=BASE/name;folder=ROOT/'models'/('urgency_final_minilm_'+name)
        protocol=load(directory/'protocol.json');decision=load(folder/'decision_config.json');policy=decision['policies']['mixed']
        if (directory/'test_results.json').exists() and (directory/'standalone_check.json').exists():
            all_results.extend(load(directory/'test_results.json'));print('Reusing completed test evaluation:',name,flush=True);continue
        tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True).eval();results=[]
        for source,records in real['test'].items():
            raw=scores(model,tokenizer,encode(tokenizer,[r['text_clean'] for r in records],decision['max_length']),'cpu',64)
            p=calibrated(raw,policy);y=np.array([int(r['priority']) for r in records]);pred=p>=.5;accepted=(p<=.2)|(p>=.8)
            result={'run':name,'arm':arm,'seed':seed,'test_source':source,'threshold':.5,
                **metrics(y,p,.5,[r['post_id'] for r in records]),'brier_score':float(brier_score_loss(y,p)),
                'automatic_coverage':float(accepted.mean()),'review_count':int((~accepted).sum()),
                'automatic_accuracy':float((pred[accepted]==y[accepted]).mean()) if accepted.any() else None,
                'automatic_errors':int(((pred!=y)&accepted).sum()),'urgent_automatically_found':int(((pred==1)&(y==1)&accepted).sum()),
                'urgent_automatically_missed':int(((pred==0)&(y==1)&accepted).sum()),'urgent_requiring_review':int(((y==1)&(~accepted)).sum())}
            results.append(result);print('FINAL MINILM TEST',json.dumps(result),flush=True)
            write_csv(directory/f'predictions_{source}.csv',[{'post_id':r['post_id'],'label':int(label),'raw_score':float(raw_s),'score':float(score),
                'predicted':int(pr),'action':('PRIORITY' if pr else 'NORMAL') if accept else 'REVIEW','text':r['text_clean']}
                for r,label,raw_s,score,pr,accept in zip(records,y,raw,p,pred,accepted)])
        raw=scores(model,tokenizer,encode(tokenizer,[r['text'] for r in drafts],decision['max_length']),'cpu',64);p=calibrated(raw,policy)
        records=[{**r,'raw_score':float(a),'score':float(b),'predicted':int(b>=.5),'action':'NORMAL' if b<=.2 else ('PRIORITY' if b>=.8 else 'REVIEW'),
            'label_status':'assistant draft; review worksheet remains separate'} for r,a,b in zip(drafts,raw,p)]
        write_csv(directory/'custom_all_predictions.csv',records)
        challenge=context_metrics([r for r in records if r['split']=='challenge'],custom_labels['challenge'])
        results.append({'run':name,'arm':arm,'seed':seed,'test_source':'custom_challenge',**challenge})
        dump(directory/'test_results.json',results);all_results.extend(results)
        # This inference file supports any sequence-classification model with the same calibration policy schema.
        source=(ROOT/'scripts/predict_reliable_deberta.py').read_text().replace("choices=['mixed','upenn','stanford']","choices=['mixed']")
        (folder/'predict.py').write_text(source)
        import transformers
        (folder/'requirements.txt').write_text(f'torch>=2.8\ntransformers=={transformers.__version__}\n')
        (folder/'README.md').write_text('# Final MiniLM trial\n\n'
            f'Arm: {arm}; seed: {seed}; custom training labels: {protocol["custom_label_origin"]}.\n'
            'Use: python predict.py --text "I cannot submit my assignment. Please help!"\n'
            'English student forum posts. Synthetic labels are not independently verified gold labels.\n'
            'Middle scores require human review. High scores can still be wrong.\n')
        env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(ROOT/'.cache/final-minilm-empty-cache'))
        standalone=[]
        for text in ['I cannot submit my assignment and the deadline is tonight. Please help!',
                     'The submit button works now and my assignment has a receipt. No further help is needed.']:
            output=subprocess.run([sys.executable,str(folder/'predict.py'),'--text',text],env=env,capture_output=True,text=True,check=True)
            standalone.append(json.loads(output.stdout))
        assert all(math.isfinite(r['score']) for r in standalone)
        dump(directory/'standalone_check.json',{'passed':True,'offline':True,'examples':standalone})
        del model,tokenizer
    return all_results

def summarize_seeds(all_results,arms,target):
    summary={}
    for arm in arms:
        summary[arm]={}
        for source in ['upenn','stanford','custom_challenge']:
            records=[r for r in all_results if r['arm']==arm and r['test_source']==source]
            summary[arm][source]={metric:{'mean':float(np.mean([r[metric] for r in records])),
                'std_across_two_seeds':float(np.std([r[metric] for r in records])),
                'values_by_seed':{str(r['seed']):r[metric] for r in records}} for metric in ['precision','recall','f1','accuracy']}
    dump(target,summary)

def controls():
    dump(BASE/'control_evaluation_plan.json',{'runs':['real_only_seed42','real_only_seed43'],
        'binary_threshold':.5,'review_bounds':[.2,.8],'selection':'seed 42 fixed in advance; seed 43 reports initialization variation',
        'custom_training_rows':0,'custom_metrics':'synthetic draft diagnostics only; no custom gradient updates authorized',
        'historical_test_status':'previously inspected; exploratory'})
    results=score_runs([('real_only',42),('real_only',43)])
    dump(BASE/'control_results.json',results)
    summarize_seeds(results,['real_only'],BASE/'control_two_seed_summary.json')
    print('Both real-only controls evaluated; custom training remains gated.',flush=True)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['select','evaluate','controls'],required=True);args=parser.parse_args()
    {'select':select,'evaluate':evaluate,'controls':controls}[args.stage]()
if __name__=='__main__':main()
