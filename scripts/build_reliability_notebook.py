"""Build the fully reproducible, real-data DeBERTa experiment and three-dataset workspace."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];cells=[]
def cell(kind,s):
    item={'cell_type':kind,'id':hashlib.sha256(s.encode()).hexdigest()[:12],'metadata':{},'source':s.splitlines(keepends=True)}
    if kind=='code':item.update(execution_count=None,outputs=[])
    cells.append(item)
def md(s):cell('markdown',s)
def code(s):cell('code',s)
md('''# Reliable classification of student posts with DeBERTa

This experiment prioritizes classification reliability, errors and manual review alongside Precision, Recall and F1.
It trains one DeBERTa-v3-base encoder on the real UPenn and Stanford fit partitions. The source contributions are balanced
in the loss. Calibration and validation rows never participate in gradient training. Historical test sets remain separate.

Three datasets are organized under `Project/Datasets`: UPenn, Stanford and CustomForum.
The 160 custom examples are **assistant-authored synthetic drafts**, not real student records or verified gold labels.
They are excluded from training by default. An approved-custom training ablation requires reviewed labels and keeps related scenarios together.

On Kaggle, enable Internet and GPU T4 ×2. The trainer uses one T4 with FP16 plus gradient scaling.
No API key or separately uploaded dataset is needed for the default real-only run.
The historical test sets were inspected previously, so results are exploratory. A better model does not guarantee correct classification of arbitrary text.
''')
code('''import os,sys,json,subprocess,importlib.util
from pathlib import Path
from IPython.display import display,Image,Markdown
IS_KAGGLE=Path('/kaggle/working').exists()
PROJECT=Path(os.environ.get('FORUM_PROJECT_DIR','/kaggle/working/Project' if IS_KAGGLE else str(Path.cwd()/'Project'))).resolve()
for folder in ['scripts','models','reports/reliability','reports/figures','.cache/tmp']:(PROJECT/folder).mkdir(parents=True,exist_ok=True)
os.environ.update(PROJECT_ROOT=str(PROJECT),HF_HOME=str(PROJECT/'.cache/huggingface'),
    HF_HUB_CACHE=str(PROJECT/'.cache/huggingface/hub'),HF_XET_CACHE=str(PROJECT/'.cache/huggingface/xet'),
    HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',MPLCONFIGDIR=str(PROJECT/'.cache/matplotlib'),
    TMPDIR=str(PROJECT/'.cache/tmp'),PIP_CACHE_DIR=str(PROJECT/'.cache/pip'),
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTORCH_ENABLE_MPS_FALLBACK='1')
RUN_TRAINING=os.environ.get('FORUM_REUSE_RELIABILITY','0')!='1'
MAX_EPOCHS=3
PATIENCE=2
MAX_LENGTH=512
USE_APPROVED_CUSTOM=False
RUN_NAME='deberta_reviewed_custom' if USE_APPROVED_CUSTOM else 'deberta_real_only'
RUN_DIR=PROJECT/'reports/reliability'/RUN_NAME
FIGURE_NAME='deberta_reviewed_custom_calibration.png' if USE_APPROVED_CUSTOM else 'deberta_calibration.png' 
print('Project:',PROJECT,'Train:',RUN_TRAINING,'Approved custom ablation:',USE_APPROVED_CUSTOM)
''')
md('## Dependencies and device\n\nExisting PyTorch is retained. Tokenizer dependencies are pinned to a compatible pair.\n')
code('''from importlib.metadata import version,PackageNotFoundError
required={'torch':'torch>=2.8','transformers':'transformers>=4.55,<6','sklearn':'scikit-learn>=1.6,<2',
          'numpy':'numpy','matplotlib':'matplotlib','joblib':'joblib','sentencepiece':'sentencepiece==0.2.1','google.protobuf':'protobuf==6.33.6'}
packages=[]
for module,spec in required.items():
    try:available=importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:available=False
    if not available:packages.append(spec)
for package,expected in [('sentencepiece','0.2.1'),('protobuf','6.33.6')]:
    try:installed=version(package)
    except PackageNotFoundError:installed=None
    if installed!=expected and f'{package}=={expected}' not in packages:packages.append(f'{package}=={expected}')
if packages:subprocess.run([sys.executable,'-m','pip','install',*packages],check=True)
import torch,transformers,sklearn,numpy as np
DEVICE='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
if IS_KAGGLE and RUN_TRAINING and DEVICE!='cuda':raise RuntimeError('Enable a Kaggle GPU before full training.')
BATCH_SIZE=8 if DEVICE=='cuda' else 4
ACCUMULATION=4 if DEVICE=='cuda' else 8
print({'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,
       'sklearn':sklearn.__version__,'device':DEVICE,'batch_size':BATCH_SIZE,'accumulation':ACCUMULATION})
''')
sources=['prepare_data.py','train.py','finetune_minilm.py','organize_datasets.py','build_custom_dataset.py',
         'prepare_reliability_data.py','train_reliable_deberta.py','predict_reliable_deberta.py','summarize_reliability.py','audit_custom_scenarios.py','validate_reliability_data.py']
for name in sources:
    md('## Source '+name+'\n')
    source=(ROOT/'scripts'/name).read_text();literal="r'''"+source+"'''"
    code('(PROJECT/"scripts"/'+repr(name)+').write_text('+literal+',encoding="utf-8")\n')
md('''## Prepare the three datasets and preserve test membership

The two original CSVs are downloaded at a pinned revision and checked by SHA-256. Compatibility paths use relative links.
Custom examples and the human-review worksheet are separate. Draft labels are never automatically approved.
''')
code('''def run(name,*args):
    subprocess.run([sys.executable,str(PROJECT/'scripts'/name),*map(str,args)],cwd=PROJECT,check=True,env=os.environ.copy())
for name,args in [('prepare_data.py',['--download']),('organize_datasets.py',[]),('build_custom_dataset.py',[]),('prepare_reliability_data.py',[]),('validate_reliability_data.py',[])]:run(name,*args)
print((PROJECT/'reports/reliability/data_plan.json').read_text())
print((PROJECT/'Datasets/CustomForum/metadata.json').read_text())
''')
md('''## Train the joint encoder and fit independent calibration

The checkpoint is selected using mean validation AP across the two sources. The decision thresholds use validation balanced accuracy,
not the largest test F1. Platt calibration is fitted on the separate calibration role. The shared mixed policy is the default for a new post.
Scores in the middle range are routed to REVIEW. Review coverage and mistakes among automatic classifications are reported explicitly.
This is not a validated semantic detector for all unrelated text; the supported scope is English student forum posts.
''')
code('''if RUN_TRAINING:
    args=['--epochs',MAX_EPOCHS,'--patience',PATIENCE,'--max-length',MAX_LENGTH,'--batch-size',BATCH_SIZE,'--accumulation',ACCUMULATION]
    if DEVICE=='mps':args.append('--mps-bf16')
    if USE_APPROVED_CUSTOM:args.append('--include-approved-custom')
    run('train_reliable_deberta.py',*args)
run('summarize_reliability.py','--run-name',RUN_NAME)
run('audit_custom_scenarios.py','--run-name',RUN_NAME)
''')
md('## Test metrics and review coverage\n\nAll test examples remain in binary metrics. REVIEW examples are counted separately, never silently dropped from F1.\n')
code('''results=json.loads((RUN_DIR/'results.json').read_text())
for row in results:
    print('Policy:',row['policy'],'Test source:',row['test_source'])
    display({k:row[k] for k in ['precision','recall','f1','macro_f1','accuracy','average_precision','roc_auc','brier_score','confusion_matrix','automatic_coverage','automatic_accuracy','review_count']})
print((RUN_DIR/'reliability_audit.json').read_text())
print('Fixed 0.5 diagnostic; deployed policy unchanged:',(RUN_DIR/'fixed_threshold_diagnostic.json').read_text())
import csv
for source in ['upenn','stanford']:
    with (RUN_DIR/f'predictions_{source}_mixed.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    errors=[r for r in rows if r['label']!=r['predicted']]
    errors.sort(key=lambda r:abs(float(r['score'])-.5),reverse=True)
    print(source,'Most confident mistakes; full text retained in the prediction CSV')
    for error in errors[:3]:display(error)
for figure_name in json.loads((RUN_DIR/'figure_files.json').read_text()):display(Image(filename=str(PROJECT/'reports/figures'/figure_name)))
print((RUN_DIR/'custom_context_diagnostics.json').read_text())
context_figure='deberta_context_drafts.png' if RUN_NAME=='deberta_real_only' else 'deberta_reviewed_custom_context_drafts.png'
display(Image(filename=str(PROJECT/'reports/figures'/context_figure)))
''')
md('## Predict a new post\n\nChange TEXT_TO_PREDICT. The folder includes model weights, tokenizer and calibration policies. REVIEW requires human judgement.\n')
code('''TEXT_TO_PREDICT='I cannot submit my assignment and the deadline is tonight. Please help!'
folder=PROJECT/'models'/('urgency_'+RUN_NAME)
run('predict_reliable_deberta.py','--model',folder,'--text',TEXT_TO_PREDICT)
print((RUN_DIR/'standalone_check.json').read_text())
''')
md('## Download the trained model, three datasets and statistics\n')
code('''from zipfile import ZipFile,ZIP_DEFLATED
archive=PROJECT/'reliable_deberta_and_datasets.zip'
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for folder in [PROJECT/'models'/('urgency_'+RUN_NAME),PROJECT/'Datasets',PROJECT/'reports/reliability']:
        for file in folder.rglob('*'):
            if file.is_file() and file.name!='best_state.pt':z.write(file,str(file.relative_to(PROJECT)))
    for figure_name in [*json.loads((RUN_DIR/'figure_files.json').read_text()),context_figure]:z.write(PROJECT/'reports/figures'/figure_name,'reports/figures/'+figure_name)
print('Download:',archive)
''')
md('''## Sources and interpretation

- Real datasets and original annotation protocol: https://github.com/pcla-code/forum-posts-urgency
- DeBERTa-v3-base: https://huggingface.co/microsoft/deberta-v3-base
- Calibration: https://scikit-learn.org/stable/modules/calibration.html

The custom dataset is explicitly synthetic and awaits human review. It does not replace fresh independently labelled real-world evaluation.
The new pipeline changes architecture, training composition and calibration together, so differences do not isolate the effect of architecture alone.
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'},
    'language_info':{'name':'python'},'kaggle':{'accelerator':'gpu','isInternetEnabled':True}},'nbformat':4,'nbformat_minor':5}
target=ROOT/'kaggle/reliable_deberta_training.ipynb';target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
print('Created:',target,'code cells:',sum(c['cell_type']=='code' for c in cells))
