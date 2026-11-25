"""Preserve the historical test sets and split only their training pools for the new experiment."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit,train_test_split
from train import read_csv,write_csv
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/reliability';SEED=42

def main():
    OUT.mkdir(exist_ok=True);assignment=ROOT/'reports/experiments/split_ids.csv'
    names={'upenn':'UPenn','stanford':'Stanford'}
    data={s:read_csv(ROOT/f'Datasets/{names[s]}/processed.csv') for s in names}
    if not assignment.exists():
        split_rows=[]
        for source,rows in data.items():
            y=np.array([int(r['priority']) for r in rows]);idx=np.arange(len(rows))
            if source=='upenn':tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(idx,y,[r['source_id'] for r in rows]))
            else:tr,te=train_test_split(idx,test_size=.2,stratify=y,random_state=SEED)
            train_ids=set(tr)
            split_rows.extend({'post_id':r['post_id'],'source':source,'split':'train' if i in train_ids else 'test'} for i,r in enumerate(rows))
        write_csv(assignment,split_rows)
    historical={r['post_id']:r['split'] for r in read_csv(assignment)};plan=[];summary={}
    for source,rows in data.items():
        training=[r for r in rows if historical[r['post_id']]=='train'];idx=np.arange(len(training));y=np.array([int(r['priority']) for r in training])
        if source=='upenn':
            groups=np.array([r['source_id'] for r in training])
            fit,dev=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED+101).split(idx,y,groups))
            c,v=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=SEED+102).split(dev,y[dev],groups[dev]))
            calibration,validation=dev[c],dev[v]
            partitions=[set(groups[x]) for x in [fit,calibration,validation]]
            assert all(not partitions[a]&partitions[b] for a,b in [(0,1),(0,2),(1,2)])
        else:
            fit,dev=train_test_split(idx,test_size=.2,stratify=y,random_state=SEED+101)
            calibration,validation=train_test_split(dev,test_size=.5,stratify=y[dev],random_state=SEED+102)
        roles={}
        for role,indices in [('fit',fit),('calibration',calibration),('validation',validation)]:
            assert set(y[indices])=={0,1}
            for i in indices:roles[training[i]['post_id']]=role
        for r in rows:
            role='test' if historical[r['post_id']]=='test' else roles[r['post_id']]
            plan.append({'post_id':r['post_id'],'source':source,'role':role,'priority':r['priority'],'group_id':r['source_id']})
        summary[source]={role:{'posts':sum(r['source']==source and r['role']==role for r in plan),
            'urgent':sum(r['source']==source and r['role']==role and int(r['priority'])==1 for r in plan)} for role in ['fit','calibration','validation','test']}
    write_csv(OUT/'split_plan.csv',plan)
    def reviewed_csv(path):
        with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    drafts={r['post_id']:r for r in reviewed_csv(ROOT/'Datasets/CustomForum/draft_examples.csv')}
    approved=[]
    for review in reviewed_csv(ROOT/'Datasets/CustomForum/reviewed_labels.csv'):
        if review['review_status']!='approved':continue
        assert review['reviewer'].strip(),'Approved rows require a named reviewer.'
        example=drafts[review['post_id']];label=review['adjudicated_priority'];action=review['adjudicated_action']
        assert action in ['PRIORITY','NORMAL','REVIEW','OUT_OF_SCOPE']
        if action in ['REVIEW','OUT_OF_SCOPE']:assert not label;continue
        assert label in ['0','1'] and (label=='1')==(action=='PRIORITY')
        if example['split']=='train':approved.append({'post_id':example['post_id'],'source':'custom','text_clean':example['text'],
                                                    'priority':label,'scenario_group':example['scenario_group'],'reviewer':review['reviewer']})
    # A fixed header also makes the empty approved set explicit and readable.
    with (OUT/'approved_custom_train.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['post_id','source','text_clean','priority','scenario_group','reviewer']);writer.writeheader();writer.writerows(approved)
    audit={'seed':SEED,'source_splits':summary,'approved_custom_train_rows':len(approved),
           'historical_test_sets_preserved':True,'upenn_students_disjoint_across_roles':True,
           'input_split_sha256':hashlib.sha256(assignment.read_bytes()).hexdigest(),
           'custom_default':'unreviewed drafts excluded','test_status':'previously inspected historical tests; exploratory, not a fresh external test'}
    (OUT/'data_plan.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
