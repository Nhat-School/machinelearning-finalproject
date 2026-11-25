"""Execute every notebook cell locally, explicitly reusing the separately completed training runs."""
import argparse,json,os,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--notebook',default='urgency_minilm_training.ipynb',choices=['urgency_minilm_training.ipynb','optimize_existing_models.ipynb','reliable_deberta_training.ipynb','final_minilm_trial.ipynb','final_best_model.ipynb'])
args=parser.parse_args()
directory=ROOT/'.cache/jupyter/kernels/forum-project';directory.mkdir(parents=True,exist_ok=True)
(directory/'kernel.json').write_text(json.dumps({'argv':[str(ROOT/'.venv/bin/python'),'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Forum Project local validation','language':'python'}))
os.environ['FORUM_REUSE_EXISTING']='1';os.environ['FORUM_REUSE_RELIABILITY']='1';os.environ['FORUM_PROJECT_DIR']=str(ROOT)
os.environ['FORUM_REUSE_FINAL_TRIAL']='1'
manager=KernelManager(kernel_name='forum-project',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(directory.parent)],ensure_native_kernel=False))
path=ROOT/'kaggle'/args.notebook;notebook=nbformat.read(path,as_version=4)
nbformat.validate(notebook)
start=time.perf_counter()
client=NotebookClient(notebook,km=manager,timeout=180,kernel_name='forum-project',resources={'metadata':{'path':str(ROOT.parent)}})
try:
    client.execute()
finally:
    if manager.has_kernel:manager.shutdown_kernel(now=True)
errors=[output for cell in notebook.cells if cell.cell_type=='code' for output in cell.outputs if output.output_type=='error']
assert not errors
summary={'passed':True,'executed_code_cells':sum(c.cell_type=='code' for c in notebook.cells),
    'mode':('Local execution reused completed actual training artifacts. Kaggle default trains models.'
            if args.notebook!='optimize_existing_models.ipynb' else
            'Optimization only: reused trained models, selected OOF blend and evaluated it. No encoder training.'),
    'seconds':time.perf_counter()-start,'kaggle_gpu_execution_verified':False}
notebook.metadata['local_validation']=summary
nbformat.write(notebook,path)
validation_name={'urgency_minilm_training.ipynb':'notebook_validation.json',
                 'optimize_existing_models.ipynb':'optimization_notebook_validation.json',
                 'reliable_deberta_training.ipynb':'reliability_notebook_validation.json',
                 'final_minilm_trial.ipynb':'notebook_validation.json',
                 'final_best_model.ipynb':'best_notebook_validation.json'}[args.notebook]
validation_dir=ROOT/('reports/final_minilm_trial' if args.notebook in ['final_minilm_trial.ipynb','final_best_model.ipynb'] else 'reports/reliability' if args.notebook=='reliable_deberta_training.ipynb' else 'reports/fine_tuning')
(validation_dir/validation_name).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
