"""Export self-contained hybrid folders from existing fitted models, then check offline CPU inference."""
import json,os,shutil,subprocess,sys
from pathlib import Path
import sklearn,transformers
ROOT=Path(__file__).resolve().parents[1]
for source in ['upenn','stanford']:
    selection=ROOT/f'reports/ensembles/{source}/decision_config.json'
    if not selection.exists():continue
    fine=ROOT/f'models/urgency_minilm_{source}';folder=ROOT/f'models/urgency_hybrid_{source}';folder.mkdir(exist_ok=True)
    for path in fine.iterdir():
        if path.is_file() and path.name not in {'decision_config.json','predict.py','README.md','requirements.txt','model_card.md'}:
            shutil.copyfile(path,folder/path.name)
    settings=json.loads(selection.read_text());settings['max_length']=json.loads((fine/'decision_config.json').read_text())['max_length']
    settings['model']='minilm_tfidf_blend';settings['labels']={'0':'NOT_URGENT','1':'URGENT'}
    (folder/'decision_config.json').write_text(json.dumps(settings,indent=2)+'\n')
    shutil.copyfile(ROOT/f'models/{source}__tfidf_logreg.joblib',folder/'tfidf_logreg.joblib')
    shutil.copyfile(ROOT/'scripts/standalone_blend.py',folder/'predict.py')
    (folder/'requirements.txt').write_text(f'torch>=2.8\ntransformers=={transformers.__version__}\nscikit-learn=={sklearn.__version__}\njoblib\nnumpy\n')
    (folder/'README.md').write_text('# Offline MiniLM and TF-IDF blend\n\nCopy this entire folder. Run: python predict.py --text "I cannot submit my assignment. Please help!"\n'
        'Uses existing fine-tuned MiniLM and TF-IDF Logistic Regression. No encoder retraining.\n'
        'Blend weight and threshold were selected on source OOF predictions, not test labels.\n'
        'CPU FP32 scores may differ slightly from MPS BF16 evaluation.\n'
        'Install requirements.txt in an isolated environment. scikit-learn is pinned for joblib compatibility.\n')
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(ROOT/'.cache/blend-empty-cache'),HF_HUB_CACHE=str(ROOT/'.cache/blend-empty-cache/hub'))
    checks=[]
    for text in ['I cannot submit my assignment and the deadline is tonight. Please help!','Thank you for the helpful explanation. I understand now.']:
        output=subprocess.run([sys.executable,str(folder/'predict.py'),'--text',text],env=env,cwd=ROOT/'.cache/tmp',check=True,capture_output=True,text=True)
        checks.append(json.loads(output.stdout))
    assert checks[0]['urgent'] and not checks[1]['urgent']
    (ROOT/f'reports/ensembles/{source}/standalone_check.json').write_text(json.dumps({'passed':True,'offline':True,'examples':checks},indent=2)+'\n')
    print('Offline blend export verified:',source)
