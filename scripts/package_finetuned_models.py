"""Complete each offline export and verify standalone CPU inference with an empty HF cache."""
import json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for source in ['upenn','stanford']:
    folder=ROOT/f'models/urgency_minilm_{source}'
    if not (folder/'model.safetensors').exists():continue
    shutil.copyfile(ROOT/'scripts/standalone_inference.py',folder/'predict.py')
    (folder/'requirements.txt').write_text('torch>=2.8\ntransformers>=4.55,<5\nsafetensors\n')
    (folder/'README.md').write_text('# Offline urgency model\n\nCopy this entire folder to another project.\n'
        'Run: python predict.py --text "I cannot submit my assignment. Please help!"\n'
        'Python needs torch and transformers; no Hugging Face cache, token or Internet is required for inference.\n'
        'The exact text cleaning function is included in predict.py. Labels and threshold are in decision_config.json.\n'
        'Training/validation used the project protocol. FP32 CPU inference can differ slightly from bfloat16 evaluation.\n')
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(ROOT/'.cache/standalone-empty-cache'),HF_HUB_CACHE=str(ROOT/'.cache/standalone-empty-cache/hub'))
    checks=[]
    for text in ['I cannot submit my assignment and the deadline is tonight. Please help!',
                 'Thank you for the helpful explanation. I understand now.']:
        result=subprocess.run([sys.executable,str(folder/'predict.py'),'--text',text],env=env,cwd=ROOT/'.cache/tmp',
            check=True,capture_output=True,text=True)
        checks.append(json.loads(result.stdout))
    target=ROOT/f'reports/fine_tuning/{source}/standalone_check.json'
    target.write_text(json.dumps({'passed':True,'offline':True,'empty_hf_cache':True,'examples':checks},indent=2)+'\n')
    print('Offline standalone export verified:',source)
