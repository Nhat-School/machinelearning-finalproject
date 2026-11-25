"""Execute teacher notebooks in isolated workspaces using completed saved training.

This validates bootstrap, data preparation, statistics, figures and inference;
it deliberately does not repeat training or claim a fresh Kaggle GPU execution.
"""
import argparse,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'kaggle/teacher_experiments'
QA=ROOT/'.cache/teacher-notebooks-validation'
EVIDENCE=ROOT/'reports/teacher_notebooks'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',nargs='*',help='Notebook numeric prefixes to validate')
    parser.add_argument('--keep-workspaces',action='store_true')
    args=parser.parse_args()
    EVIDENCE.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
    directory=ROOT/'.cache/jupyter/kernels/forum-teacher'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'kernel.json').write_text(json.dumps({'argv':[str(ROOT/'.venv/bin/python'),'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Project teacher notebook QA','language':'python'}))
    records=[]
    for path in sorted(DEST.glob('*.ipynb')):
        if args.only and path.name[:2] not in args.only:continue
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        for n,cell in enumerate(notebook.cells):
            if cell.cell_type=='code':compile(cell.source,str(path)+f':cell{n}','exec')
        work=QA/path.stem
        if work.exists():shutil.rmtree(work)
        work.mkdir(parents=True)
        env=os.environ.copy()
        env.update(FORUM_TEACHER_VALIDATE='1',FORUM_TEACHER_SAVED_PROJECT=str(ROOT),FORUM_TEACHER_RUN_DIR=str(work))
        manager=KernelManager(kernel_name='forum-teacher',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(directory.parent)],ensure_native_kernel=False))
        start=time.perf_counter();record={'file':path.name,'training_performed':False,'kaggle_gpu_execution_verified':False}
        print('VALIDATING',path.name,flush=True)
        client=NotebookClient(notebook,km=manager,timeout=360,kernel_name='forum-teacher',resources={'metadata':{'path':str(ROOT.parent)}})
        try:
            client.execute(env=env)
            errors=[out for cell in notebook.cells if cell.cell_type=='code' for out in cell.outputs if out.output_type=='error']
            assert not errors
            results=json.loads((work/'reports/teacher/results.json').read_text())
            checks=json.loads((work/'reports/teacher/checks.json').read_text())
            figures=json.loads((work/'reports/teacher/figure_files.json').read_text())
            assert all((work/f).is_file() for f in figures)
            record.update(passed=True,executed_code_cells=sum(c.cell_type=='code' for c in notebook.cells),
                result_rows=len(results),prediction_rows_checked=sum(r['per_post_predictions_checked'] for r in checks),
                figure_count=len(figures),data_checks=(json.loads((work/'reports/teacher/data_checks.json').read_text()) if (work/'reports/teacher/data_checks.json').exists() else None),
                environment=json.loads((work/'reports/teacher/environment.json').read_text()),
                mode='Local isolated execution; pinned raw inputs and completed trained artifacts reused; no fitting')
            notebook.metadata['local_validation']=record
            nbformat.write(notebook,path)
        except Exception:
            record.update(passed=False,error=traceback.format_exc())
            nbformat.write(notebook,EVIDENCE/(path.stem+'_FAILED.ipynb'))
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        record['seconds']=round(time.perf_counter()-start,2)
        (EVIDENCE/(path.stem+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        output=[]
        for i,cell in enumerate(notebook.cells):
            if cell.cell_type!='code':continue
            for out in cell.outputs:
                if out.output_type=='stream':output.append(f'Cell {i}: '+out.text)
                if out.output_type=='error':output.append('\n'.join(out.traceback))
        (EVIDENCE/(path.stem+'.log')).write_text('\n'.join(output))
        records.append(record)
        print(json.dumps({k:record[k] for k in ['file','passed','seconds']}),flush=True)
        if not record['passed']:
            print(record['error'],flush=True)
            raise SystemExit(1)
        if not args.keep_workspaces:shutil.rmtree(work)
    all_records=[json.loads((EVIDENCE/(p.stem+'.json')).read_text()) for p in sorted(DEST.glob('*.ipynb')) if (EVIDENCE/(p.stem+'.json')).exists()]
    summary={'passed':len(all_records)==11 and all(r['passed'] for r in all_records),'notebook_count':len(all_records),
        'fresh_training_performed':False,'fresh_kaggle_gpu_execution_verified':False,
        'scope':'All cells executed locally in isolated workspaces. Data preparation, saved prediction metrics, plots and demo inference checked. Fresh training and package installation on Kaggle not performed.',
        'notebooks':all_records}
    (DEST/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('COMPLETE',len(records),'notebooks in this invocation.',flush=True)


if __name__=='__main__':main()
