"""Run the notebook setup/data cells in an empty project folder, without training."""
import csv,json,os,time
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
def main():
    notebook=nbformat.read(ROOT/'kaggle/reliable_deberta_training.ipynb',as_version=4);nbformat.validate(notebook)
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type=='code':compile(cell.source,f'cell-{i}','exec')
    fresh=ROOT/'.cache/reliability-fresh-project'
    os.environ['FORUM_PROJECT_DIR']=str(fresh);os.environ['FORUM_REUSE_RELIABILITY']='1'
    namespace={};executed=0;start=time.perf_counter()
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        if cell.source.startswith('if RUN_TRAINING:'):break
        exec(compile(cell.source,'fresh-notebook-bootstrap','exec'),namespace);executed+=1
    validation=json.loads((fresh/'reports/reliability/data_validation.json').read_text());assert validation['passed']
    original=json.loads((ROOT/'reports/reliability/data_plan.json').read_text());new=json.loads((fresh/'reports/reliability/data_plan.json').read_text())
    assert original['source_splits']==new['source_splits']
    def membership(path):
        with path.open(newline='') as f:return {r['post_id']:(r['source'],r['split']) for r in csv.DictReader(f)}
    assert membership(ROOT/'reports/experiments/split_ids.csv')==membership(fresh/'reports/experiments/split_ids.csv')
    result={'passed':True,'mode':'Fresh folder notebook bootstrap only; no model training','executed_code_cells':executed,
            'original_test_membership_recreated':True,'seconds':time.perf_counter()-start}
    (ROOT/'reports/reliability/fresh_notebook_bootstrap.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
