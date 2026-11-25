"""Refit all models from fixed splits/cache and compare metrics, excluding timings."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read():return json.loads((ROOT/'reports/experiments/results.json').read_text())
before=read();split_hash=hashlib.sha256((ROOT/'reports/experiments/split_ids.csv').read_bytes()).hexdigest()
subprocess.run(['bash','-c','source scripts/launchers/env.sh; .venv/bin/python scripts/train.py'],cwd=ROOT,check=True)
after=read();assert len(before)==len(after)==20
assert split_hash==hashlib.sha256((ROOT/'reports/experiments/split_ids.csv').read_bytes()).hexdigest()
keys=['precision','recall','f1','macro_f1','accuracy','average_precision','roc_auc','precision_at_20','recall_at_20']
maximum=0.
for a,b in zip(before,after):
    assert (a['train_source'],a['test_source'],a['model'])==(b['train_source'],b['test_source'],b['model'])
    assert a['confusion_matrix']==b['confusion_matrix']
    for key in keys:
        delta=abs(a[key]-b[key]);maximum=max(maximum,delta);assert delta<1e-8,(a['model'],key,delta)
result={'passed':True,'experiments':len(after),'split_ids_sha256':split_hash,'maximum_metric_difference':maximum,
        'tolerance':1e-8,'embedding_cache_reused':True,'timings_excluded':True}
(ROOT/'reports/experiments/reproducibility_ml.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: 20 experiments, same splits and confusion matrices; maximum metric difference',maximum)
