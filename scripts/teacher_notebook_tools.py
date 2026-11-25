"""Shared data checks, saved-run validation, statistics and figures for teacher notebooks.

Every training notebook embeds this file. Saved-run QA is explicitly separate
from fresh training and never writes through links to the original model files.
"""
import argparse,csv,hashlib,json,os,shutil,textwrap
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/teacher'


def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def copy_tree_for_qa(source,target):
    source,target=Path(source),Path(target)
    if not source.exists():raise FileNotFoundError(source)
    for path in source.rglob('*'):
        if not path.is_file():continue
        destination=target/path.relative_to(source);destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.is_symlink():destination.unlink()
        if path.suffix in {'.safetensors','.bin','.pt','.joblib'}:
            if destination.exists():destination.unlink()
            destination.symlink_to(path.resolve())
        else:shutil.copy2(path,destination)


def hydrate_qa(profile,model_keys):
    """Populate only the saved evidence required by this notebook, without training."""
    if os.environ.get('FORUM_TEACHER_VALIDATE')!='1':return
    saved=Path(os.environ['FORUM_TEACHER_SAVED_PROJECT']).resolve()
    assert saved!=ROOT and saved.is_dir()
    for folder in ['scripts','models','reports/figures','reports/experiments','data/raw','.cache/tmp']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    cache=ROOT/'.cache/huggingface'
    if not cache.exists():cache.symlink_to(saved/'.cache/huggingface',target_is_directory=True)
    for name,filename in [('UPenn','All_Courses_REDACTED_CODED.csv'),('Stanford','Stanford.csv')]:
        target=ROOT/'data/raw'/filename
        if not target.exists():shutil.copy2(saved/'Datasets'/name/'raw.csv',target)
    split=saved/'reports/experiments/split_ids.csv'
    shutil.copy2(split,ROOT/'reports/experiments/split_ids.csv')
    families=[];models=[]
    if profile=='baseline':
        families=['experiments']
        models=[f'{source}__{key}.joblib' for source in ['upenn','stanford'] for key in model_keys]
    elif profile in ['source_minilm','blend']:
        families=['fine_tuning']
        models=[f'urgency_minilm_{source}' for source in ['upenn','stanford']]
        if profile=='blend':
            families+=['experiments','ensembles']
            models += [f'{source}__tfidf_logreg.joblib' for source in ['upenn','stanford']]
            models += [f'urgency_hybrid_{source}' for source in ['upenn','stanford']]
    elif profile=='deberta':
        families=['reliability/deberta_real_only'];models=['urgency_deberta_real_only']
    elif profile in ['joint_real','joint_custom']:
        families=['final_minilm_trial']
        models=[p.name for p in (saved/'models').glob('urgency_final_minilm_*')
                if profile=='joint_custom' or p.name.startswith('urgency_final_minilm_real_only_')]
    for family in families:copy_tree_for_qa(saved/'reports'/family,ROOT/'reports'/family)
    for name in models:
        source=saved/'models'/name;target=ROOT/'models'/name
        if source.is_dir():copy_tree_for_qa(source,target)
        else:
            if target.exists() or target.is_symlink():target.unlink()
            target.symlink_to(source.resolve())
    if profile=='baseline':
        for filename in ['results.json','cv_results.json']:
            path=ROOT/'reports/experiments'/filename
            write_json(path,[r for r in read_json(path) if r['model'] in model_keys])
    write_json(OUT/'validation_origin.json',{
        'mode':'LOCAL VALIDATION USING COMPLETED TRAINING; no fitting performed',
        'saved_project':str(saved),'profile':profile,'model_keys':model_keys})


def check_data():
    audit=read_json(ROOT/'reports/data_audit.json')
    assert all(audit['checks'].values())
    plan=read_csv(ROOT/'reports/reliability/split_plan.csv')
    historical={r['post_id']:r for r in read_csv(ROOT/'reports/experiments/split_ids.csv')}
    assert len(plan)==len({r['post_id'] for r in plan})==32563
    assert all((r['role']=='test')==(historical[r['post_id']]['split']=='test') for r in plan)
    students=defaultdict(set)
    for row in plan:
        if row['source']=='upenn':students[row['group_id']].add(row['role'])
    assert all(len(roles)==1 for roles in students.values())
    groups=defaultdict(set)
    for row in read_csv(ROOT/'Datasets/CustomForum/draft_examples.csv'):groups[row['scenario_group']].add(row['split'])
    assert all(len(roles)==1 for roles in groups.values())
    result={'passed':True,'hashes_and_cleaning_checks':audit['checks'],
        'historical_test_membership_preserved':True,'upenn_student_roles_disjoint':True,
        'custom_scenario_groups_disjoint':True,'rows':len(plan),
        'split_plan_sha256':hashlib.sha256((ROOT/'reports/reliability/split_plan.csv').read_bytes()).hexdigest()}
    write_json(OUT/'data_checks.json',result)
    return result


def count_metrics(matrix):
    (tn,fp),(fn,tp)=matrix
    ratio=lambda a,b:a/b if b else 0.
    f1=ratio(2*tp,2*tp+fp+fn)
    return {'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),'f1':f1,
        'accuracy':ratio(tn+tp,tn+fp+fn+tp),
        'macro_f1':(f1+ratio(2*tn,2*tn+fp+fn))/2}


def result_rows(profile,model_keys):
    if profile=='baseline':return [r for r in read_json(ROOT/'reports/experiments/results.json') if r['model'] in model_keys]
    if profile=='source_minilm':
        selected=[r for source in ['upenn','stanford'] for r in read_json(ROOT/f'reports/fine_tuning/{source}/test_results.json')]
        write_json(ROOT/'reports/fine_tuning/results.json',selected)
        return [record for r in selected for record in [dict(r,policy='selected threshold'),
            dict(r['default_threshold_metrics'],train_source=r['train_source'],test_source=r['test_source'],model=r['model'],threshold=.5,policy='fixed 0.5 diagnostic')]]
    if profile=='blend':return read_json(ROOT/'reports/ensembles/results.json')
    if profile=='deberta':
        rows=read_json(ROOT/'reports/reliability/deberta_real_only/results.json')
        rows += [dict(r,model='joint_deberta',train_source='upenn+stanford',test_source=r['source'],policy='fixed 0.5 diagnostic')
                 for r in read_json(ROOT/'reports/reliability/deberta_real_only/fixed_threshold_diagnostic.json')]
        return rows
    if profile in ['joint_real','joint_custom']:
        filename='control_results.json' if profile=='joint_real' else 'results.json'
        return [dict(r,model='joint_minilm_'+r['arm'],train_source='joint') for r in read_json(ROOT/'reports/final_minilm_trial'/filename)]
    if profile=='archive':return read_json(ROOT/'reports/teacher/archived_comparison.json')
    return []


def prediction_rows(record,profile):
    source=record.get('test_source');train=record.get('train_source')
    if profile=='baseline':path=ROOT/f'reports/experiments/predictions_{train}_{source}_{record["model"]}.csv'
    elif profile=='source_minilm':path=ROOT/f'reports/fine_tuning/{train}/predictions_{source}.csv'
    elif profile=='blend':path=ROOT/f'reports/ensembles/{train}/predictions_{source}.csv'
    elif profile=='deberta':
        policy=record['policy'] if record['policy'] in ['upenn','stanford','mixed'] else 'mixed'
        path=ROOT/f'reports/reliability/deberta_real_only/predictions_{source}_{policy}.csv'
    elif profile in ['joint_real','joint_custom']:
        directory=ROOT/'reports/final_minilm_trial'/record['run']
        path=directory/('custom_all_predictions.csv' if source=='custom_challenge' else f'predictions_{source}.csv')
    else:return []
    rows=read_csv(path)
    if source=='custom_challenge':
        rows=[dict(r,label=r['draft_priority']) for r in rows if r['split']=='challenge' and r['draft_priority'] in ['0','1']]
    return rows


def label(row):
    method=row.get('trial_label',row.get('run',row.get('model','method')))
    policy=row.get('policy','')
    return f"{method} {row.get('train_source','joint')} → {row['test_source']}"+(f' / {policy}' if policy else '')


def verify_results(rows,profile):
    errors=[];checked=[];curves=[]
    for row in rows:
        calculated=count_metrics(row['confusion_matrix'])
        for key,value in calculated.items():
            if row.get(key) is not None:assert abs(float(row[key])-value)<1e-10,(label(row),key,row[key],value)
        predicted_rows=prediction_rows(row,profile)
        if predicted_rows:
            y=np.array([int(r['label']) for r in predicted_rows]);p=np.array([float(r['score']) for r in predicted_rows])
            threshold=float(row.get('threshold',.5));pred=p>=threshold
            matrix=[[int(((y==0)&(~pred)).sum()),int(((y==0)&pred).sum())],
                    [int(((y==1)&(~pred)).sum()),int(((y==1)&pred).sum())]]
            assert matrix==row['confusion_matrix'],label(row)
            for key,value in [('average_precision',average_precision_score(y,p)),('roc_auc',roc_auc_score(y,p))]:
                if row.get(key) is not None:assert abs(float(row[key])-float(value))<1e-8,(label(row),key)
            wrong=[]
            for item,true,score,guess in zip(predicted_rows,y,p,pred):
                if int(true)!=int(guess):wrong.append({'post_id':item['post_id'],'true':int(true),'predicted':int(guess),
                    'score':float(score),'threshold':threshold,'error':'FN' if true else 'FP','text':item.get('text','')})
            errors.append({'experiment':label(row),'all_error_count':len(wrong),
                'examples':sorted(wrong,key=lambda r:abs(r['score']-threshold),reverse=True)[:4]})
            curves.append((label(row),y,p,threshold))
        checked.append({'experiment':label(row),'count_metrics_match':True,
                        'per_post_predictions_checked':bool(predicted_rows)})
    write_json(OUT/'checks.json',checked);write_json(OUT/'error_examples.json',errors)
    return curves


def make_figures(rows,curves,profile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=[]
    def save(fig,name):
        path=OUT/(name+'.png');fig.tight_layout();fig.savefig(path,dpi=145,bbox_inches='tight');plt.close(fig);figures.append(str(path.relative_to(ROOT)))
    if not rows:
        audit=read_json(ROOT/'reports/data_audit.json');fig,ax=plt.subplots(figsize=(7,3.7))
        sources=list(audit['datasets']);x=np.arange(len(sources));width=.35
        for i,(key,title) in enumerate([('0','Not urgent'),('1','Urgent')]):
            ax.bar(x+(i-.5)*width,[audit['datasets'][s]['label_counts'][key] for s in sources],width,label=title)
        ax.set(xticks=x,xticklabels=sources,ylabel='Clean posts',title='Actual prepared dataset counts');ax.legend();save(fig,'dataset_labels')
    for start in range(0,len(rows),8):
        chunk=rows[start:start+8];x=np.arange(len(chunk));width=.25
        fig,ax=plt.subplots(figsize=(max(8,len(chunk)*1.35),4.8))
        for i,key in enumerate(['precision','recall','f1']):ax.bar(x+(i-1)*width,[row[key] for row in chunk],width,label=key)
        ax.set(ylim=(0,1),xticks=x,xticklabels=[textwrap.fill(label(row),24) for row in chunk],title='All-post binary metrics; review rows remain included')
        ax.tick_params(axis='x',labelsize=7);ax.legend();save(fig,f'metrics_{start//8+1:02d}')
        fig,axes=plt.subplots((len(chunk)+1)//2,2,figsize=(8,3.2*((len(chunk)+1)//2)),squeeze=False)
        for ax,row in zip(axes.flat,chunk):
            matrix=np.array(row['confusion_matrix']);ax.imshow(matrix,cmap='Blues')
            for a in range(2):
                for b in range(2):ax.text(b,a,str(matrix[a,b]),ha='center',va='center',color='white' if matrix[a,b]>matrix.max()*.5 else 'black')
            ax.set(title=textwrap.fill(label(row),42),xlabel='Predicted',ylabel='True',xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Urgent'],yticklabels=['Normal','Urgent'])
            ax.title.set_fontsize(8)
        for ax in list(axes.flat)[len(chunk):]:ax.axis('off')
        save(fig,f'confusions_{start//8+1:02d}')
    for start in range(0,len(curves),8):
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for title,y,p,t in curves[start:start+8]:
            precision,recall,thresholds=precision_recall_curve(y,p)
            axes[0].plot(recall,precision,label=title)
            f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
            axes[1].plot(thresholds,f1,label=title)
        axes[0].set(xlabel='Recall',ylabel='Precision',title='Precision–recall curves',xlim=(0,1),ylim=(0,1))
        axes[1].set(xlabel='Decision threshold',ylabel='F1',title='Test threshold diagnostic only; no selection',ylim=(0,1))
        axes[0].legend(fontsize=6);save(fig,f'precision_recall_threshold_{start//8+1:02d}')
    histories=[]
    if profile=='source_minilm':
        for source in ['upenn','stanford']:
            for run in read_json(ROOT/f'reports/fine_tuning/{source}/selection.json')['scout_runs']:
                histories.append((source+'/'+run['name'],run['history'],'val_ap'))
    elif profile=='deberta':histories=[('deberta',read_json(ROOT/'reports/reliability/deberta_real_only/history.json'),'mean_validation_ap')]
    elif profile in ['joint_real','joint_custom']:
        for run in dict.fromkeys(row['run'] for row in rows):
            histories.append((run,read_json(ROOT/'reports/final_minilm_trial'/run/'history.json'),'mean_real_validation_ap'))
    if histories:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for title,history,metric in histories:
            axes[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history],'o-',label=title)
            if all(metric in r for r in history):axes[1].plot([r['epoch'] for r in history],[r[metric] for r in history],'o-',label=title)
        axes[0].set(xlabel='Epoch',ylabel='Training loss');axes[1].set(xlabel='Epoch',ylabel='Validation AP',ylim=(0,1));axes[0].legend(fontsize=6);save(fig,'learning_curves')
    write_json(OUT/'figure_files.json',figures)
    return figures


def emit(profile,model_keys):
    OUT.mkdir(parents=True,exist_ok=True);rows=result_rows(profile,model_keys)
    curves=verify_results(rows,profile);figures=make_figures(rows,curves,profile)
    write_json(OUT/'results.json',rows)
    fields=['environment','run','train_source','test_source','model','policy','seed','threshold','precision','recall','f1','macro_f1','accuracy','average_precision','roc_auc','brier_score','precision_at_20','recall_at_20','automatic_coverage','review_count','confusion_matrix']
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in rows:writer.writerow(row)
    print(json.dumps({'profile':profile,'rows':len(rows),'figures':figures,
        'mode':'SAVED-TRAINING VALIDATION' if os.environ.get('FORUM_TEACHER_VALIDATE')=='1' else ('HISTORICAL LOG REPLAY; NO TRAINING' if profile=='archive' else 'FRESH EXPERIMENT'),
        'checks':'All reported count metrics recomputed; per-post scores checked when available.'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['hydrate','check-data','emit'])
    parser.add_argument('--profile',default='data');parser.add_argument('--models',nargs='*',default=[])
    args=parser.parse_args()
    if args.action=='hydrate':hydrate_qa(args.profile,args.models)
    elif args.action=='check-data':print(json.dumps(check_data(),indent=2))
    else:emit(args.profile,args.models)


if __name__=='__main__':main()
