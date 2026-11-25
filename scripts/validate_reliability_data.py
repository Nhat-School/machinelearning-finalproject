"""Check source hashes, test membership and group separation before new training."""
import csv,hashlib,json,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    plan=read(ROOT/'reports/reliability/split_plan.csv');historical={r['post_id']:r for r in read(ROOT/'reports/experiments/split_ids.csv')}
    assert len(plan)==len({r['post_id'] for r in plan})==32563
    for source in ['UPenn','Stanford']:
        folder=ROOT/'Datasets'/source;card=json.loads((folder/'metadata.json').read_text())
        for name in ['raw','processed']:assert hashlib.sha256((folder/(name+'.csv')).read_bytes()).hexdigest()==card[name+'_sha256']
    assert all((r['role']=='test')==(historical[r['post_id']]['split']=='test') for r in plan)
    student_roles={}
    for r in plan:
        if r['source']=='upenn':student_roles.setdefault(r['group_id'],set()).add(r['role'])
    assert all(len(roles)==1 for roles in student_roles.values())
    drafts=read(ROOT/'Datasets/CustomForum/draft_examples.csv');groups={}
    for r in drafts:groups.setdefault(r['scenario_group'],set()).add(r['split'])
    assert all(len(splits)==1 for splits in groups.values())
    approved=read(ROOT/'reports/reliability/approved_custom_train.csv');by_id={r['post_id']:r for r in drafts}
    assert all(by_id[r['post_id']]['split']=='train' and r['reviewer'].strip() for r in approved)
    import torch,transformers,sklearn,sentencepiece
    result={'passed':True,'original_source_hashes_match':True,'historical_test_membership_preserved':True,
            'upenn_students_disjoint_across_all_roles':True,'custom_scenario_groups_disjoint':True,
            'custom_training_rows_require_human_review':True,'approved_custom_training_rows':len(approved),
            'versions':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
                        'sklearn':sklearn.__version__,'sentencepiece':sentencepiece.__version__}}
    (ROOT/'reports/reliability/data_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
