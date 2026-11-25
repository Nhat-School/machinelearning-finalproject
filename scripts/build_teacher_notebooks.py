"""Create one self-contained Kaggle notebook per completed experiment family."""
import hashlib,json,textwrap
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'kaggle/teacher_experiments'
COMMON=['prepare_data.py','train.py','organize_datasets.py','build_custom_dataset.py',
        'prepare_reliability_data.py','audit_custom_labels_ai.py','teacher_notebook_tools.py']
FINE=['finetune_minilm.py','standalone_inference.py','package_finetuned_models.py']
JOINT=['finetune_minilm.py','predict_reliable_deberta.py','final_minilm_trial.py',
       'evaluate_final_minilm_trial.py','summarize_final_minilm_trial.py','run_final_minilm_trial.py','check_final_behavior.py']
SPECS=[
    dict(slug='00_data_preparation_and_label_audit',profile='data',title='Data preparation and the explicit AI label audit',pages='5–6, 16',
         purpose='Download the pinned UPenn/Stanford sources, clean and organize all three datasets, recreate the historical partitions, and regenerate the recorded AI audit. No classifier is trained.',gpu='CPU',models=[],extra=[],
         finding='The real sources contain 3,493 UPenn and 29,070 Stanford posts after cleaning. CustomForum has 160 synthetic drafts, 0 human-approved rows, and 82 binary training examples. AI inspection does not create human approval.'),
    dict(slug='01_tfidf_logistic_regression_and_dummy',profile='baseline',title='TF-IDF Logistic Regression and majority baseline',pages='7, 10',
         purpose='Train TF-IDF Logistic Regression and the majority baseline on each source. Evaluate same-source and cross-source test sets, including OOF threshold selection.',gpu='CPU',models=['tfidf_logreg','tfidf_dummy'],extra=['teacher_predict_baseline.py'],
         finding='Logistic Regression was a strong lexical baseline. The majority classifier has urgent-class F1 of zero despite high overall accuracy. Earlier Kaggle same-source LR F1 was 0.5781 / 0.7215 (UPenn / Stanford).'),
    dict(slug='02_tfidf_svm',profile='baseline',title='TF-IDF linear SVM',pages='7, 10',
         purpose='Train only the existing linear SVM experiment, with 3-fold development tuning, source OOF threshold selection, and four train-to-test directions.',gpu='CPU',models=['tfidf_svm'],extra=['teacher_predict_baseline.py'],
         finding='The earlier Kaggle same-source SVM F1 was 0.5289 / 0.7185. The method remained useful for comparison, but did not beat fine-tuned MiniLM on those same-source benchmarks. SVM scores are decision margins, not probabilities.'),
    dict(slug='03_tfidf_naive_bayes',profile='baseline',title='TF-IDF Multinomial Naive Bayes',pages='7, 10',
         purpose='Train only Multinomial Naive Bayes, tune smoothing inside training data, choose source OOF thresholds, and evaluate four transfer directions.',gpu='CPU',models=['tfidf_nb'],extra=['teacher_predict_baseline.py'],
         finding='Earlier Kaggle same-source F1 was 0.3919 / 0.6453. The recorded false positives and false negatives explain the lower score; conditional-independence assumptions are a possible mechanism, not an isolated proven cause.'),
    dict(slug='04_frozen_minilm_logistic_regression',profile='baseline',title='Frozen MiniLM embeddings with Logistic Regression',pages='7, 10',
         purpose='Extract normalized embeddings with the pinned pretrained MiniLM, keep the encoder frozen, and fit Logistic Regression using source-only development data.',gpu='GPU recommended',models=['minilm_logreg'],extra=['teacher_predict_baseline.py'],
         finding='Earlier Kaggle same-source F1 was 0.5395 / 0.6210. General pretrained embeddings did not match supervised fine-tuning here. This notebook downloads the pinned encoder but does not update its weights.'),
    dict(slug='05_source_minilm_finetuning_augmentation_cv',profile='source_minilm',title='Source-specific MiniLM fine-tuning augmentation and five-fold CV',pages='8–9, 11',
         purpose='Run all three original scouts (plain, weighted, weighted plus contraction augmentation), select by validation, run five folds, fit source models, and evaluate same-source and transfer tests.',gpu='GPU required',models=[],extra=FINE,
         finding='Augmentation helped the UPenn scout; Stanford selected plain training. Historical Mac selected-model F1 was 0.6222 / 0.8036. CV supplied development OOF scores and thresholds; it was not an independent nested-CV evaluation.'),
    dict(slug='06_minilm_tfidf_blending',profile='blend',title='MiniLM and TF-IDF Logistic Regression blending',pages='8, 17',
         purpose='From a fresh session, train TF-IDF LR and source MiniLM prerequisites, then select one of 21 blend weights and a threshold from aligned OOF predictions before reading test predictions.',gpu='GPU required',models=['tfidf_logreg'],extra=FINE+['blend_trained_models.py','standalone_blend.py','package_blended_models.py'],
         finding='The historical Mac blend reached F1 0.6364 / 0.8092. It improved same-source benchmarks but did not improve every transfer direction. No previous notebook output is required: this file trains its prerequisites when run fresh.'),
    dict(slug='07_joint_deberta_calibration_review',profile='deberta',title='Joint DeBERTa calibration thresholds and review coverage',pages='8, 12, 18',
         purpose='Train the completed real-only DeBERTa protocol, calibrate on separate rows, choose validation policies, and display mixed/source policies plus the fixed-0.5 diagnostic and Review coverage.',gpu='GPU required',models=[],extra=['finetune_minilm.py','train_reliable_deberta.py','predict_reliable_deberta.py','summarize_reliability.py','audit_custom_scenarios.py'],
         finding='Historical Mac fixed-0.5 F1 was 0.6119 / 0.7964. The selected mixed policy emphasized Recall and had more false alarms. The interrupted Mac FP32 attempt had no classifier-quality result and is archived in notebook 10, not presented as a successful training run.'),
    dict(slug='08_joint_minilm_two_datasets_selected',profile='joint_real',title='Selected joint MiniLM on the two real datasets',pages='11, 13–18',
         purpose='Train only the selected two-source MiniLM method with seeds 42 and 43, separate fit/calibration/validation roles, fixed 0.5 binary threshold, and 0.2/0.8 Review bounds.',gpu='GPU required',models=[],extra=JOINT,
         finding='This is the selected shared prototype, equivalent to final_best_model.ipynb. Historical Mac seed-42 F1 was 0.6667 / 0.7742; the supplied Kaggle run was 0.6613 / 0.7701 and 12/16 draft probes. It is not universally highest-F1; the Stanford blend remains higher on that benchmark.'),
    dict(slug='09_joint_minilm_three_dataset_comparison',profile='joint_custom',title='Three-dataset MiniLM weak-supervision comparison',pages='13–18',
         purpose='Run the completed five-run comparison: two-source control (42,43), custom cross-entropy (42), and the validation-selected custom variant confirmed with seed 43. Synthetic draft labels remain explicitly weak supervision.',gpu='GPU required',models=[],extra=JOINT,
         finding='Custom pairwise supervision lowered real-test F1 versus the matched control on both sources and both seeds. It scored 24/24 templated synthetic challenge examples, but 13/16 new-phrasing probes versus control 14/16 in the Mac run. Those are synthetic diagnostics, not independent human gold.'),
    dict(slug='10_archived_logs_and_interrupted_attempt',profile='archive',title='Archived measured results and the interrupted FP32 attempt',pages='10–12; full comparison',
         purpose='Recompute all count-derived metrics from the 83 archived comparison records, plot their confusion matrices, and display the supplied Kaggle environments/probes and the incomplete Mac FP32 attempt. No model is trained.',gpu='CPU',models=[],extra=[],
         finding='This is an evidence replay. AP/AUC/Brier values without imported Kaggle per-post predictions remain transcribed log values. A memory-pressure failure on a Mac cannot be promised to recur on a different Kaggle GPU. Fresh training is provided by the other notebooks.'),
]


def cells_for(spec):
    cells=[]
    def add(kind,source):
        cell={'cell_type':kind,'id':hashlib.sha256((spec['slug']+kind+source).encode()).hexdigest()[:16],
              'metadata':{},'source':source.splitlines(keepends=True)}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    md=lambda value:add('markdown',value)
    code=lambda value:add('code',value)
    profile=spec['profile'];neural=profile in ['source_minilm','blend','deberta','joint_real','joint_custom'] or 'minilm_logreg' in spec['models']
    md(f'''# {spec['title']}

**DOCX pages: {spec['pages']}. Hardware: {spec['gpu']}.**

{spec['purpose']}

Upload this single `.ipynb` into a fresh Kaggle notebook, enable Internet, and Run All. GPU notebooks use one visible GPU; T4 ×2 is compatible. No dataset upload, previous notebook output, personal repository, Hugging Face token or AI API key is required. Training code and fixed input revisions are embedded below. Each file writes into its own folder under `/kaggle/working/Project/teacher_runs`.

**Historical finding:** {spec['finding']}

Fresh results are generated by this execution. Historical values quoted in explanations are references, not substitutes for the new outputs. Hardware, Python and library versions can change numerical results. The real historical test sets were inspected during development, so this reproduces the reported exploratory protocol, not fresh external validation. Ten epochs has not been tested in the joint pipeline and is not silently added here.
''')
    if profile in ['data','deberta','joint_real','joint_custom']:
        md('''## Meaning of the third dataset

CustomForum contains 160 assistant-authored synthetic drafts. The recorded AI audit is clearly marked as AI review; zero rows are human-approved. Related scenario states stay in the same split. Only notebook 09 opts into the previously requested synthetic weak-supervision training; data preparation or AI inspection alone never approves human labels. The 12 ambiguous and 12 unrelated examples do not become binary training labels.
''')
    code(f'''import os,sys,json,subprocess,importlib.util
from pathlib import Path
from IPython.display import display,Image,Markdown
SLUG={spec['slug']!r}
PROFILE={profile!r}
MODEL_KEYS={spec['models']!r}
NEURAL={neural!r}
GPU_REQUIRED={bool(spec['gpu']=='GPU required')!r}
IS_KAGGLE=Path('/kaggle/working').exists()
VALIDATION_REUSE=os.environ.get('FORUM_TEACHER_VALIDATE','0')=='1'
default_root=(Path('/kaggle/working/Project/teacher_runs') if IS_KAGGLE else Path.cwd()/'Project/teacher_runs')/SLUG
PROJECT=Path(os.environ.get('FORUM_TEACHER_RUN_DIR',str(default_root))).resolve()
if VALIDATION_REUSE:
    saved_root=Path(os.environ['FORUM_TEACHER_SAVED_PROJECT']).resolve()
    assert PROJECT.is_relative_to(saved_root) and PROJECT!=saved_root
for name in ['scripts','models','reports/teacher','reports/figures','reports/experiments','.cache/tmp']:(PROJECT/name).mkdir(parents=True,exist_ok=True)
os.environ.update(PROJECT_ROOT=str(PROJECT),HF_HOME=str(PROJECT/'.cache/huggingface'),HF_HUB_CACHE=str(PROJECT/'.cache/huggingface/hub'),
    HF_XET_CACHE=str(PROJECT/'.cache/huggingface/xet'),HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',
    MPLCONFIGDIR=str(PROJECT/'.cache/matplotlib'),TMPDIR=str(PROJECT/'.cache/tmp'),PIP_CACHE_DIR=str(PROJECT/'.cache/pip'),
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTORCH_ENABLE_MPS_FALLBACK='1')
RUN_TRAINING=not VALIDATION_REUSE and PROFILE not in ['data','archive']
print('Project:',PROJECT)
print('Mode:','LOCAL QA: saved completed training; no fitting' if VALIDATION_REUSE else ('FRESH TRAINING' if RUN_TRAINING else 'DATA AUDIT / HISTORICAL EVIDENCE; NO TRAINING'))
''')
    md('''## Dependencies and recorded environment

Unlike the older presence-only dependency check, this cell checks installed versions against the requested requirements. Existing PyTorch is kept when compatible. Installation occurs only in the Kaggle notebook environment; local QA never installs packages. Start in a fresh kernel. If a changed package was already imported and this cell requests a restart, restart once and Run All.
''')
    requirements=['numpy>=2,<3','scikit-learn==1.9.1','matplotlib>=3.10,<4','joblib>=1.4,<2','packaging>=24']
    if neural:requirements+=['torch>=2.8','transformers==4.57.6','sentencepiece==0.2.1']
    if 'minilm_logreg' in spec['models']:requirements+=['sentence-transformers==5.7.0']
    if profile=='deberta':requirements+=['protobuf==6.33.6']
    code(f'''from importlib.metadata import version,PackageNotFoundError
try:
    from packaging.requirements import Requirement
except ImportError:
    if not IS_KAGGLE:raise RuntimeError('Use the existing Project/.venv; local package installation is not performed.')
    subprocess.run([sys.executable,'-m','pip','install','packaging>=24'],check=True)
    from packaging.requirements import Requirement
REQUIREMENTS={requirements!r}
changes=[];loaded_changes=[]
module_names={{'scikit-learn':'sklearn','sentence-transformers':'sentence_transformers','protobuf':'google.protobuf'}}
for text in REQUIREMENTS:
    requirement=Requirement(text)
    try:installed=version(requirement.name)
    except PackageNotFoundError:installed=None
    if installed is None or installed not in requirement.specifier:
        changes.append(text)
        if module_names.get(requirement.name,requirement.name.replace('-','_')) in sys.modules:loaded_changes.append(requirement.name)
if changes:
    if VALIDATION_REUSE or not IS_KAGGLE:raise RuntimeError('Use the existing Project/.venv with compatible packages; no local installation: '+str(changes))
    subprocess.run([sys.executable,'-m','pip','install',*changes],check=True)
    if loaded_changes:raise RuntimeError('Packages changed after import. Restart the kernel once, then Run All: '+str(loaded_changes))
for text in REQUIREMENTS:
    requirement=Requirement(text)
    assert version(requirement.name) in requirement.specifier,(requirement.name,version(requirement.name),str(requirement.specifier))
DEVICE='cpu';USE_BF16=False
if NEURAL:
    import torch
    DEVICE='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    USE_BF16=DEVICE=='mps' or (DEVICE=='cuda' and torch.cuda.is_bf16_supported(including_emulation=False))
if RUN_TRAINING and IS_KAGGLE and GPU_REQUIRED and DEVICE!='cuda':raise RuntimeError('Enable a Kaggle GPU before training this notebook.')
environment={{'python':sys.version,'requirements':REQUIREMENTS,'versions':{{Requirement(text).name:version(Requirement(text).name) for text in REQUIREMENTS}},
    'device':DEVICE,'visible_gpus':torch.cuda.device_count() if NEURAL else 0,'training_gpu_count':1 if RUN_TRAINING and DEVICE=='cuda' else 0,
    'validation_reuse':VALIDATION_REUSE,'fresh_training_requested':RUN_TRAINING}}
(PROJECT/'reports/teacher/environment.json').write_text(json.dumps(environment,indent=2)+'\\n')
print(json.dumps(environment,indent=2))
''')
    files=list(dict.fromkeys((['teacher_notebook_tools.py'] if profile=='archive' else COMMON)+spec['extra']))
    md('## Embedded source files\n\nEach module below is written into this experiment folder. Defaults and algorithm parameters match the completed project protocol; the baseline CLI merely selects the requested existing method.\n')
    code('SOURCES={}\n')
    for name in files:
        md('### '+name+'\n')
        source=(ROOT/'scripts'/name).read_text()
        literal="r'''"+source+"'''" if "'''" not in source else repr(source)
        code('SOURCES['+repr(name)+']='+literal+'\n(PROJECT/"scripts"/'+repr(name)+').write_text(SOURCES['+repr(name)+'],encoding="utf-8")\n')
    code('''import hashlib
def run(name,*args):subprocess.run([sys.executable,str(PROJECT/'scripts'/name),*map(str,args)],cwd=PROJECT,env=os.environ.copy(),check=True)
def read_json(relative):return json.loads((PROJECT/relative).read_text())
print(json.dumps({name:hashlib.sha256(source.encode()).hexdigest() for name,source in SOURCES.items()},indent=2))
if VALIDATION_REUSE:run('teacher_notebook_tools.py','hydrate','--profile',PROFILE,'--models',*MODEL_KEYS)
''')
    if profile=='archive':
        paths=['reports/all_trials_comparison.json','reports/kaggle_run/environment.json',
               'reports/kaggle_run/joint_minilm_20261117_run.json','reports/kaggle_run/joint_minilm_20261117_probes.json',
               'reports/reliability/fp32_runtime_attempt.json','reports/reliability/benchmark.json','reports/reliability/benchmark_bf16.json']
        references={path:json.loads((ROOT/path).read_text()) for path in paths}
        md('''## Archived evidence and its limits

These are explicitly embedded measured records from the DOCX, not fresh training. P/R/F1/Accuracy are recomputed from every confusion matrix. Kaggle AP/AUC/Brier values are transcribed log statistics when per-post Kaggle files are unavailable. The FP32 record was stopped before validation and has no test score or exported classifier; the three-step timing benchmarks discarded their weights. Their timings and memory failure depend on the original hardware.
''')
        code('ARCHIVED='+repr(references)+'''
for relative,value in ARCHIVED.items():
    target=PROJECT/'reports/teacher/archive'/Path(relative).name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(value,indent=2)+'\\n')
(PROJECT/'reports/teacher/archived_comparison.json').write_text(json.dumps(ARCHIVED['reports/all_trials_comparison.json'],indent=2)+'\\n')
for relative,value in ARCHIVED.items():
    if relative!='reports/all_trials_comparison.json':print(relative,json.dumps(value,indent=2))
''')
    else:
        md('''## Prepare pinned inputs and check separation

The two real CSVs are downloaded at a fixed revision and verified by SHA-256. Cleaning and exact-duplicate decisions are recorded. UPenn is grouped by student; Stanford is stratified because equivalent author/thread IDs are unavailable. No learned preprocessing is fitted on test. Related synthetic scenarios remain together.
''')
        code('''for name,args in [('prepare_data.py',['--download']),('organize_datasets.py',[]),('build_custom_dataset.py',[]),
                  ('prepare_reliability_data.py',[])]:run(name,*args)
run('teacher_notebook_tools.py','check-data')
print(json.dumps(read_json('reports/data_audit.json')['checks'],indent=2))
print(json.dumps(read_json('reports/reliability/data_plan.json'),indent=2))
''')
    if profile=='data':
        md('## Recorded AI semantic audit\n\nThis regenerates the saved judgments on the fixed 160 draft texts and checks their identity. It is not a fresh independent model review or human adjudication. The human worksheet remains unchanged.\n')
        code("run('audit_custom_labels_ai.py')\nprint(json.dumps(read_json('Datasets/CustomForum/ai_label_audit_20261117.json'),indent=2))\n")
    if profile=='baseline':
        md('''## Train the selected method and evaluate four directions

Hyperparameters are tuned with three source-training folds. A source OOF threshold is fixed before either source's test is scored. SVM uses its decision margin; the other pipelines use probability estimates. Both same-source and cross-source results are retained. Frozen MiniLM keeps its pretrained encoder unchanged; the other notebooks here use TF-IDF fitted within training folds.
''')
        code("if RUN_TRAINING:run('train.py','--models',*MODEL_KEYS)\n")
        code("print(json.dumps(read_json('reports/experiments/cv_results.json'),indent=2))\n")
    if profile in ['source_minilm','blend']:
        md('''## Original source MiniLM protocol

Maximum three epochs, patience one, batch 64, length 256 and LR 2e-5. The scouts test plain cross-entropy, class weighting, and weighting plus training-only contraction expansion. Scout selection uses tuned validation F1 with AP as tie-breaker; checkpoints use validation AP. The chosen configuration runs five folds. Median best epoch and OOF threshold determine the final source fit. CV is development evidence, not unbiased nested CV. T4 follows the original source trainer's FP32 path; supported devices may use BF16.
''')
        code('''MAX_EPOCHS=3
PATIENCE=1
FOLDS=5
if RUN_TRAINING:
    args=['--stage','all','--epochs',MAX_EPOCHS,'--patience',PATIENCE,'--folds',FOLDS,'--batch-size',64,'--max-length',256]
    if USE_BF16:args.append('--bf16')
    if PROFILE=='blend':run('train.py','--models','tfidf_logreg')
    run('finetune_minilm.py',*args)
    run('package_finetuned_models.py')
selected=[r for source in ['upenn','stanford'] for r in read_json(f'reports/fine_tuning/{source}/test_results.json')]
(PROJECT/'reports/fine_tuning/results.json').write_text(json.dumps(selected,indent=2)+'\\n')
for source in ['upenn','stanford']:
    print(source,'scouts and selection',json.dumps(read_json(f'reports/fine_tuning/{source}/selection.json'),indent=2))
    print(source,'five-fold development',json.dumps(read_json(f'reports/fine_tuning/{source}/cv.json'),indent=2))
''')
    if profile=='blend':
        md('''## Select blend on development OOF predictions

All required base training is included above for a fresh Kaggle session. Twenty-one weights from 0 to 1 are compared using source OOF F1, with AP as tie-breaker. The chosen weight/threshold are saved before reading either test prediction file. Base OOF predictions also participated in earlier model/checkpoint selection, so the blend development results are not an independent nested-CV assessment.
''')
        code("run('blend_trained_models.py')\nif RUN_TRAINING:run('package_blended_models.py')\nfor source in ['upenn','stanford']:print(source,json.dumps(read_json(f'reports/ensembles/{source}/selection.json'),indent=2))\n")
    if profile=='deberta':
        md('''## Completed DeBERTa protocol and calibration policies

Use the completed Mac experiment's batch size 4 and accumulation 8 (effective 32), maximum three epochs, patience two, length 512, LR 1e-5 and seed 42. CUDA chooses mixed precision appropriate to its hardware; the completed Mac run used MPS BF16. Both real sources have equal aggregate loss contribution. Separate calibration fits Platt scaling; validation selects balanced-accuracy policies. Mixed and source policies, fixed 0.5, all-post metrics, Review coverage and confident errors are reported. Custom drafts do not update this model.
''')
        code('''if RUN_TRAINING:
    args=['--epochs',3,'--patience',2,'--batch-size',4,'--accumulation',8,'--max-length',512,'--lr',1e-5,'--seed',42]
    if DEVICE=='mps':args.append('--mps-bf16')
    run('train_reliable_deberta.py',*args)
    run('summarize_reliability.py')
    run('audit_custom_scenarios.py')
print(json.dumps(read_json('reports/reliability/deberta_real_only/reliability_audit.json'),indent=2))
print(json.dumps(read_json('reports/reliability/deberta_real_only/custom_context_diagnostics.json'),indent=2))
''')
    if profile in ['joint_real','joint_custom']:
        reference_paths=['reports/fine_tuning/results.json','reports/reliability/deberta_real_only/fixed_threshold_diagnostic.json']
        refs={path:json.loads((ROOT/path).read_text()) for path in reference_paths}
        plan=json.loads((ROOT/'reports/final_minilm_trial/experiment_plan.json').read_text())
        md('''## Historical references for comparison only

These two small files contain previously measured Mac source-MiniLM and DeBERTa reference metrics for the existing comparison plotter. Those old models are not trained here and their metrics are not measurements from the current Kaggle run. The current joint experiment writes separate results below.
''')
        code('REFERENCES='+repr(refs)+'''
for relative,value in REFERENCES.items():
    target=PROJECT/relative
    if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value,indent=2)+'\\n')
target=PROJECT/'reports/final_minilm_trial/experiment_plan.json';target.parent.mkdir(parents=True,exist_ok=True)
if not target.exists():target.write_text(json.dumps('''+repr(plan)+''',indent=2)+'\\n')
''')
        md('''## Train the existing joint protocol

Maximum four epochs, patience two, batch 64, length 256, LR 2e-5, equal aggregate real-source loss, and checkpoint selection by mean real validation AP. Platt scaling uses separate real calibration rows. Threshold 0.5 and Review bounds 0.2/0.8 are fixed before test. Seed 42 is the inference export; seed 43 reports variation and is not chosen by test score.

Notebook 09 additionally tests custom CE (20% loss share) and custom CE plus ranking (0.25 inside the custom term), explicitly as weak supervision. It freezes the custom configuration on validation, then confirms it with seed 43. Better synthetic challenge accuracy does not establish better classification on real data. This is the completed comparison, not a new ten-epoch experiment.
''')
        code('''if RUN_TRAINING:
    args=['--synthetic-drafts'] if PROFILE=='joint_custom' else []
    if DEVICE=='mps':args.append('--bf16')
    run('run_final_minilm_trial.py',*args)
run('summarize_final_minilm_trial.py',*(['--controls-only'] if PROFILE=='joint_real' else []))
base=PROJECT/'reports/final_minilm_trial'
for name in (['control_two_seed_summary.json'] if PROFILE=='joint_real' else ['selection.json','frozen_recommendation.json','two_seed_summary.json']):
    print(name,(base/name).read_text())
run('check_final_behavior.py','--exported-only',*(['--control-only'] if PROFILE=='joint_real' else []))
probe_name='control_behavior_probe_results.json' if PROFILE=='joint_real' else 'exported_behavior_probe_results.json'
print('Extra draft diagnostics:',(base/probe_name).read_text())
''')
    md('''## All statistics figures and actual errors

Precision/Recall/F1 describe the urgent class, while Accuracy includes every test post. Review rows remain in binary metrics. Each available prediction CSV is checked against its confusion matrix and AP/AUC; full wrong-prediction CSVs remain in the archive. Test threshold curves below are labelled diagnostics and never used to choose a threshold. Historical log replay has count checks only where per-post files were not supplied.
''')
    code('''run('teacher_notebook_tools.py','emit','--profile',PROFILE,'--models',*MODEL_KEYS)
rows=read_json('reports/teacher/results.json')
if rows:
    headers=['Method/run','Train → test','Policy','P','R','F1','Accuracy','FP','FN']
    table='| '+' | '.join(headers)+' |\\n|'+ '|'.join(['---']*len(headers))+'|\\n'
    for r in rows:
        table+='| '+' | '.join([str(r.get('run',r.get('trial_label',r.get('model','method')))),str(r.get('train_source','joint'))+' → '+r['test_source'],str(r.get('policy','')),
            *[f"{r[k]:.4f}" for k in ['precision','recall','f1','accuracy']],str(r['confusion_matrix'][0][1]),str(r['confusion_matrix'][1][0])])+' |\\n'
    display(Markdown(table))
    for r in rows:print(json.dumps(r,indent=2))
for relative in read_json('reports/teacher/figure_files.json'):display(Image(filename=str(PROJECT/relative)))
print('Checks:',json.dumps(read_json('reports/teacher/checks.json'),indent=2))
print('Actual error examples:',json.dumps(read_json('reports/teacher/error_examples.json'),indent=2))
''')
    if profile not in ['data','archive']:
        md('''## Predict a new English post

Change TEXT_TO_PREDICT and rerun this cell after training is complete. Scores are model estimates; a high score can still be wrong. Review and outside-scope handling are not independently validated semantic classes. Source models use their source-specific saved thresholds.
''')
        if profile=='baseline':
            code("TEXT_TO_PREDICT='I cannot submit my assignment and the deadline is tonight. Please help!'\nMODEL_SOURCE='stanford'\nMODEL_KEY=MODEL_KEYS[0]\nrun('teacher_predict_baseline.py','--model',PROJECT/'models'/f'{MODEL_SOURCE}__{MODEL_KEY}.joblib','--text',TEXT_TO_PREDICT)\n")
        elif profile=='source_minilm':
            code("TEXT_TO_PREDICT='I cannot submit my assignment and the deadline is tonight. Please help!'\nMODEL_SOURCE='stanford'\nsubprocess.run([sys.executable,str(PROJECT/'models'/f'urgency_minilm_{MODEL_SOURCE}'/'predict.py'),'--text',TEXT_TO_PREDICT],check=True)\n")
        elif profile=='blend':
            code("TEXT_TO_PREDICT='I cannot submit my assignment and the deadline is tonight. Please help!'\nsubprocess.run([sys.executable,str(PROJECT/'models/urgency_hybrid_stanford/predict.py'),'--text',TEXT_TO_PREDICT],check=True)\n")
        elif profile=='deberta':
            code("TEXT_TO_PREDICT='I cannot submit my assignment and the deadline is tonight. Please help!'\nsubprocess.run([sys.executable,str(PROJECT/'models/urgency_deberta_real_only/predict.py'),'--text',TEXT_TO_PREDICT],check=True)\n")
        else:
            code("TEXT_TO_PREDICT='I cannot submit my assignment and the deadline is tonight. Please help!'\nARM='real_only'\n# Notebook 09 also exports custom_ce and the selected custom configuration; choose deliberately.\nsubprocess.run([sys.executable,str(PROJECT/'models'/f'urgency_final_minilm_{ARM}_seed42'/'predict.py'),'--text',TEXT_TO_PREDICT],check=True)\n")
    md('''## Download the complete experiment output

The ZIP includes source code, dataset provenance/splits, statistics, figures, predictions, and trained model folders when training occurred. Caches and temporary checkpoints are excluded. Explicit local validation omits linked model weights from its QA ZIP; a fresh Kaggle training run includes its actual trained weights. Download the displayed ZIP from Kaggle Output. This notebook does not launch the local web application.
''')
    code('''from zipfile import ZipFile,ZIP_DEFLATED
archive=PROJECT/(SLUG+'_output.zip')
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for directory in ['scripts','Datasets','reports','models']:
        for path in (PROJECT/directory).rglob('*'):
            if not path.is_file() or path.name=='best_state.pt':continue
            if VALIDATION_REUSE and path.suffix in ['.safetensors','.bin','.pt']:continue
            z.write(path,str(path.relative_to(PROJECT)))
print('Download from Kaggle Output:',archive,'MB:',round(archive.stat().st_size/1e6,1))
''')
    return cells,requirements,files


def main():
    DEST.mkdir(parents=True,exist_ok=True);manifest=[]
    for spec in SPECS:
        cells,requirements,sources=cells_for(spec)
        metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},
            'kaggle':{'isInternetEnabled':True,'accelerator':'gpu' if spec['gpu']!='CPU' else 'none'},
            'teacher_experiment':{**spec,'training_default':spec['profile'] not in ['data','archive'],
                                  'requirements':requirements,'source_files':sources,'gpu_execution_verified':False}}
        notebook={'cells':cells,'metadata':metadata,'nbformat':4,'nbformat_minor':5}
        filename=spec['slug']+'.ipynb';path=DEST/filename
        path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
        manifest.append({**spec,'file':filename,'code_cells':sum(c['cell_type']=='code' for c in cells),
                         'requirements':requirements,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (DEST/'experiment_index.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    lines=['# Chọn notebook theo thí nghiệm trong DOCX','',
        'Đây là bộ notebook dành cho giảng viên. Mỗi file tự chứa mã và tải đúng dữ liệu; chỉ upload một file `.ipynb` tương ứng vào Kaggle.',
        'Không cần chạy tất cả. Chọn đúng thí nghiệm được hỏi. Các notebook training tự huấn luyện khi Run All; bản 00 và 10 không train model.','',
        '| File | Nội dung | Trang DOCX | Thiết bị |','|---|---|---|---|']
    for r in manifest:lines.append(f"| [{r['file']}]({r['file']}) | {r['title']} | {r['pages']} | {r['gpu']} |")
    lines += ['', '## Cách chạy','',
        '1. Chọn một file theo bảng; notebook 08 là model mặc định đã chọn.',
        '2. Upload file vào Kaggle, bật Internet; dùng GPU T4 ×2 cho các bản neural. Mã dùng một GPU.',
        '3. Run All. Không cần Add Input, API key hoặc notebook output cũ. Bản blend 06 tự train các model cần trước khi trộn.',
        '4. Xem bảng chỉ số, ma trận, đường học/PR khi có, các lỗi thật và kiểm tra tái tạo. Đổi TEXT_TO_PREDICT để demo câu mới.',
        '5. Tải ZIP output được in ở cell cuối. ZIP có model đã train, cấu hình, datasets/splits, code và mọi số đo của lần chạy.',
        '', '## Phạm vi và các trường hợp dễ nhầm','',
        '- Bảng tham khảo trong phần giải thích là kết quả lịch sử. Kết quả mới nằm trong các cell chạy; không giả vờ số đo cũ là lần train mới.',
        '- Các notebook mới kiểm tra phiên bản thư viện, khắc phục việc chỉ kiểm tra module đã tồn tại ở các notebook cũ. Python/GPU/PyTorch khác vẫn có thể làm số đo khác.',
        '- Bản 09 dùng nhãn tổng hợp theo ngoại lệ synthetic weak supervision đã được yêu cầu. AI audit không phải human review; không tự duyệt worksheet.',
        '- Bản 10 phát lại 83 dòng kết quả và hai log Kaggle đã cung cấp; nó không tái tạo weights từ các số đếm. Notebook khác dùng để train lại phương pháp.',
        '- Lượt DeBERTa FP32 bị dừng trên Mac chưa có F1. Bản 10 giữ bằng chứng lỗi này; bản 07 chạy protocol DeBERTa hoàn tất. Không ép máy khác phải lỗi bộ nhớ giống Mac.',
        '- Lượt 10 epoch chưa được thử nên không có notebook tuyên bố đây là thí nghiệm đã đo. Các cấu hình trong bộ này khớp những lượt đã báo cáo.',
        '- Sản phẩm web ở trang 19–20 chạy local bằng bash Project/scripts/launchers/start_app.sh. Các notebook cung cấp cell demo dự đoán, không host web local.',
        '- Bộ này không chứa model weights và không cài gì lên Mac khi chỉ tải/giải nén. Mọi mã/temp của quá trình tạo và kiểm tra nằm trong Project.',
        '', '## Kiểm tra đã thực hiện','',
        'Xem VALIDATION.md và validation.json: phân biệt compile/bootstrap, đối soát dữ liệu, thực thi với kết quả training đã lưu và việc thực sự train trên Kaggle. Chưa tự nhận đã chạy GPU cho các notebook đóng gói mới.',
        '', 'Tạo lại bộ notebook: Project/.venv/bin/python Project/scripts/build_teacher_notebooks.py từ thư mục machine.',
    ]
    (DEST/'README.md').write_text('\n'.join(lines)+'\n')
    print('Created',len(manifest),'self-contained teacher notebooks in',DEST)


if __name__=='__main__':main()
