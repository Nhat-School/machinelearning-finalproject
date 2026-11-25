"""Run the predeclared MiniLM comparison, preserving completed runs and old models."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reports/final_minilm_trial'
def run(name,*args):subprocess.run([sys.executable,str(ROOT/'scripts'/name),*map(str,args)],check=True)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-drafts',action='store_true',help='Explicitly opt into unreviewed synthetic weak supervision; does not approve human labels.')
    parser.add_argument('--bf16',action='store_true',help='Use BF16 for Apple MPS; CUDA chooses supported precision automatically.')
    args=parser.parse_args();BASE.mkdir(parents=True,exist_ok=True)
    common=['--epochs',4,'--patience',2]+(['--bf16'] if args.bf16 else [])
    for seed in [42,43]:run('final_minilm_trial.py','--arm','real_only','--seed',seed,*common)
    if not args.synthetic_drafts:
        if not (BASE/'control_results.json').exists():run('evaluate_final_minilm_trial.py','--stage','controls')
        run('summarize_final_minilm_trial.py');return
    authorization=BASE/'synthetic_authorization.json'
    if not authorization.exists():
        authorization.write_text(json.dumps({'user_authorized_synthetic_trial':True,
            'authorization_basis':'Explicit --synthetic-drafts option supplied by the notebook/CLI operator.',
            'authorized_at_utc':datetime.now(timezone.utc).isoformat(),'human_label_adjudication':False,
            'review_worksheet_modified':False},indent=2)+'\n')
    for arm in ['custom_ce','custom_pairwise']:
        run('final_minilm_trial.py','--arm',arm,'--seed',42,'--allow-synthetic-drafts',*common)
    if not (BASE/'selection.json').exists():run('evaluate_final_minilm_trial.py','--stage','select')
    chosen=json.loads((BASE/'selection.json').read_text())['selected_custom_configuration']
    run('final_minilm_trial.py','--arm',chosen,'--seed',43,'--allow-synthetic-drafts',*common)
    if not (BASE/'results.json').exists():run('evaluate_final_minilm_trial.py','--stage','evaluate')
    run('summarize_final_minilm_trial.py')
if __name__=='__main__':main()
