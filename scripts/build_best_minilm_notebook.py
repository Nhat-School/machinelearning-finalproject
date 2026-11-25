"""Derive the teacher-facing notebook for the selected two-source joint MiniLM prototype."""
import json,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
comparison=ROOT/'kaggle/final_minilm_trial.ipynb'
if not comparison.exists():runpy.run_path(str(ROOT/'scripts/build_final_minilm_notebook.py'),run_name='__main__')
n=json.loads(comparison.read_text());n['metadata'].pop('local_validation',None)
n['cells'][0]['source']=['# Final project model: joint MiniLM trained on UPenn + Stanford\n\n',
    'This is the **final selected prototype** documented in the official report. It trains on the two real datasets only, with equal aggregate source loss weight, separate calibration and validation, and two seeds. Seed 42 is the fixed inference export.\n\n',
    'The three-dataset variants remain in the separate comparison notebook. In the historical Mac runs they improved templated synthetic scenarios but lost real-test F1 on both datasets and scored 13/16 extra binary draft probes versus 14/16 for this control. The subsequently supplied Kaggle run scored 12/16. These are separate measured runs. This does not prove universal accuracy: the older Stanford blend has higher Stanford benchmark F1.\n\n',
    'Enable **Internet ON and GPU T4 ×2**, then Run All. One T4 is used. No dataset upload or API key is required. The three dataset folders are generated, but CustomForum is **diagnostic only** and its draft labels never update this model. Historical test results and assistant-authored context probes are exploratory.\n']
for cell in n['cells']:
    source=''.join(cell['source'])
    if cell['cell_type']=='code':
        cell.update(execution_count=None,outputs=[])
        if source.startswith('(PROJECT/"scripts"/'):
            continue
        source=source.replace('RUN_SYNTHETIC_TRIAL=True','RUN_SYNTHETIC_TRIAL=False')
        if "run('run_final_minilm_trial.py'" in source:
            source="""if RUN_TRAINING:
    args=['--bf16'] if DEVICE=='mps' else []
    run('run_final_minilm_trial.py',*args)
run('summarize_final_minilm_trial.py','--controls-only')
BASE=PROJECT/'reports/final_minilm_trial'
print('Final export: real_only seed 42; no synthetic gradient training.')
"""
        source=source.replace("BASE/'summary.json'","BASE/'control_summary.json'")
        source=source.replace('final_minilm_metrics.png','best_minilm_metrics.png').replace('final_minilm_confusion.png','best_minilm_confusion.png').replace('final_minilm_learning.png','best_minilm_learning.png')
        source=source.replace("BASE/('two_seed_summary.json' if (BASE/'results.json').exists() else 'control_two_seed_summary.json')","BASE/'control_two_seed_summary.json'")
        source=source.replace("BASE/'error_examples.json'","BASE/'control_error_examples.json'")
        source=source.replace("run('check_final_behavior.py','--exported-only')","run('check_final_behavior.py','--exported-only','--control-only')").replace("BASE/'exported_behavior_probe_results.json'","BASE/'control_behavior_probe_results.json'")
        if "recommendation=json.loads" in source:
            source="""TEXT_TO_PREDICT='I cannot submit my assignment and the deadline is tonight. Please help!'
ARM_TO_USE='real_only'
MODEL_FOLDER=PROJECT/'models/urgency_final_minilm_real_only_seed42'
subprocess.run([sys.executable,str(MODEL_FOLDER/'predict.py'),'--text',TEXT_TO_PREDICT],check=True)
print('Final model folder:',MODEL_FOLDER)
"""
        source=source.replace("'final_minilm_models_and_statistics.zip'","'best_minilm_model_and_statistics.zip'").replace("glob('urgency_final_minilm_*')","glob('urgency_final_minilm_real_only_*')").replace("glob('final_minilm_*.png')","glob('best_minilm_*.png')")
        cell['source']=source.splitlines(keepends=True)
    else:
        if '## Train the predeclared arms' in source:
            cell['source']=['## Train and independently calibrate the selected model\n\nFour maximum epochs, patience 2, LR 2e-5, batch 64, length 256. Equal aggregate real-source contributions; natural class ratios. Checkpoint maximizes mean real validation AP. Separate real calibration fits Platt scaling. Binary threshold 0.5 and review bounds 0.2/0.8 are fixed before test. Both seeds are reported; custom examples are diagnostic only.\n']
        if 'The matched arms' in source:cell['source']=[source.replace('The matched arms','The historical comparison arms')]

def load(relative):return json.loads((ROOT/'reports'/relative).read_text())
def same_source_f1(records,model,source):
    return next(row['f1'] for row in records if row['model']==model and row['train_source']==row['test_source']==source)
baseline=load('kaggle_run/baseline_results_from_log.json')
fine=load('fine_tuning/results.json');blend=load('ensembles/results.json')
fixed=load('reliability/deberta_real_only/fixed_threshold_diagnostic.json')
control=load('final_minilm_trial/control_results.json');custom=load('final_minilm_trial/results.json')
def arm_f1(records,arm,source,seed=42):
    return next(row['f1'] for row in records if row['arm']==arm and row['seed']==seed and row['test_source']==source)
history=[]
for model,label,reason in [
    ('tfidf_logreg','TF-IDF + Logistic Regression','Useful baseline; lower same-source F1 than source-specific fine-tuned MiniLM in that earlier Kaggle run. This does not establish that it loses on every future post.'),
    ('tfidf_svm','TF-IDF + linear SVM','Already trained and evaluated. Lower same-source F1 in that run; SVM is an alternative classifier, not an automatic extra layer of contextual understanding.'),
    ('tfidf_nb','TF-IDF + Naive Bayes','Already tested; lower same-source F1 in that run. It remains part of the comparison evidence.'),
    ('minilm_logreg','Frozen MiniLM embeddings + LR','Already tested; keeping the encoder frozen did not match the source-specific fine-tuned classifier in that run.'),
]:
    history.append((label,'Earlier user-supplied Kaggle log',f"{same_source_f1(baseline,model,'upenn'):.4f} / {same_source_f1(baseline,model,'stanford'):.4f}",reason))
history.extend([
    ('Source-specific MiniLM fine-tuning','Historical Mac runs',f"{same_source_f1(fine,'finetuned_minilm','upenn'):.4f} / {same_source_f1(fine,'finetuned_minilm','stanford'):.4f}",'Successful reference method. Separate models per source; different fit pools, checkpoint selection and score policies from this joint model.'),
    ('MiniLM plus TF-IDF blend','Historical Mac runs',f"{same_source_f1(blend,'minilm_tfidf_blend','upenn'):.4f} / {same_source_f1(blend,'minilm_tfidf_blend','stanford'):.4f}",'Improved same-source benchmark F1 and remains the highest measured Stanford F1 here. It did not improve every transfer direction; the selected joint prototype is not universally best.'),
    ('Larger DeBERTa, fixed calibrated threshold 0.5','Historical Mac diagnostic', ' / '.join(f"{next(row['f1'] for row in fixed if row['source']==source):.4f}" for source in ['upenn','stanford']),'Mixed policy had lower F1 with higher Recall. Threshold, calibration, fit pools and tuning budget also differ; these runs do not prove a larger architecture is inherently inferior.'),
    ('Joint MiniLM on two real sources','Historical Mac, seed 42', ' / '.join(f"{arm_f1(control,'real_only',source):.4f}" for source in ['upenn','stanford']),'Chosen shared prototype; improved UPenn relative to the older source model, but lost Stanford Recall/F1. The improvement on 16 draft probes is small and not independent deployment evidence.'),
    ('Joint MiniLM plus synthetic CE and pairwise loss','Historical Mac, seed 42', ' / '.join(f"{arm_f1(custom,'custom_pairwise',source):.4f}" for source in ['upenn','stanford']),'Lower real-test F1 than the matched two-source control on both sources and both seeds. Synthetic challenge 24/24 did not generalize consistently to new phrasing; 13/16 extra draft probes versus control 14/16.'),
])
history_text='''## What was already tested and why it was not selected

Read this section before proposing another experiment. These are **previously measured results**, not models retrained by Run All in this notebook. F1 refers to the urgent class. A method not being selected does not mean it never works. All causal explanations remain hypotheses unless isolated by an experiment.

| Previous method | Where measured | UPenn F1 / Stanford F1 | Finding and decision |
|---|---|---|---|
'''+''.join('| '+' | '.join(row)+' |\n' for row in history)
up=load('fine_tuning/upenn/selection.json');st=load('fine_tuning/stanford/selection.json')
def scout(selection,name):return next(row for row in selection['scout_runs'] if row['name']==name)['validation_f1_tuned']
history_text+=f'''
**Augmentation was tested, with mixed outcomes.** The older source-only experiment expanded contractions and tested plain, weighted and weighted-plus-augmentation scouts. UPenn weighted-plus-augmentation validation F1 was {scout(up,'scout_weighted_aug'):.4f}, versus {scout(up,'scout_weighted'):.4f} for weighting alone; it was selected. Stanford plain was {scout(st,'scout_plain'):.4f}, versus {scout(st,'scout_weighted_aug'):.4f} for weighting plus augmentation; plain was selected. The three-source trial was a separate synthetic weak-supervision test. These results do not establish that all possible paraphrase augmentation fails.

**Five-fold CV was already used** by the older source-specific MiniLM pipeline for out-of-fold threshold/epoch selection. This joint notebook instead keeps fit, calibration, validation and test separate and reports two training seeds. Repeating CV does not create additional independent labels.

**AI audit of CustomForum:** all 160 fixed texts were inspected under the current-support rubric; 136 retain binary draft labels, 12 need context and 12 are outside the supported scope. Only 82 binary examples belong to training. All 40 still-blocked updates share one suffix. AI reinspection is not independent human approval and does not turn synthetic scenarios into real student evidence. The earlier authorized synthetic experiment already trained on draft labels; the two-source notebook deliberately excludes them after comparison.
'''
kaggle_path=ROOT/'reports/kaggle_run/joint_minilm_20261117_results.json'
if kaggle_path.exists():
    measured=json.loads(kaggle_path.read_text())
    history_text+='\n**Subsequent actual Kaggle run, supplied by the user:**\n\n| Seed | UPenn F1 | Stanford F1 |\n|---|---:|---:|\n'
    for seed in [42,43]:
        history_text+=f"| {seed} | {arm_f1(measured,'real_only','upenn',seed):.4f} | {arm_f1(measured,'real_only','stanford',seed):.4f} |\n"
    history_text+='\nThat run completed in about 16.8 minutes, exported 182.7 MB and scored 12/16 extra binary draft probes. Its synthetic challenge F1 was 0.80, accuracy 0.7083, with 0/7 complete three-state groups correct. This demonstrates continuing errors on resolution and quotation. These numbers are historical Kaggle references; fresh outputs below may differ. Its Transformers version was 5.20.2, while the local run used 4.57.6. The setup cell checks module presence, so its requested version pin is not enforced for preinstalled packages; keep the printed versions with every run. No single cause of the environment-related differences has been isolated.\n'
history_text+='''
**Ten epochs has not been tested in this joint pipeline.** Both supplied Kaggle seeds peaked in mean validation AP at epoch 3 and declined at epoch 4. That does not rule out later improvement. A future ten-epoch cap with early stopping would be a new experiment, retaining the best validation checkpoint rather than automatically exporting epoch 10. This code also bases its learning-rate schedule on the epoch cap, so changing the cap changes optimization before epoch 4 as well. No ten-epoch result or improvement is claimed.

Run All below trains only the selected two-source method (seeds 42 and 43). It does not repeat the older SVM, augmentation, blend, DeBERTa or synthetic-training experiments. The full historical comparison notebook remains a separate optional artifact.
'''
n['cells'].insert(1,{'cell_type':'markdown','metadata':{},'source':history_text.splitlines(keepends=True)})
for i,cell in enumerate(n['cells']):cell['id']='best-'+str(i)
path=ROOT/'kaggle/final_best_model.ipynb';path.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n');print('Created final selected notebook:',path)
