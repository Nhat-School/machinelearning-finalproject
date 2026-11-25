#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
PROJECT_DOC_PYTHON="${HOME}/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
PROJECT_NODE="${HOME}/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' 'Chạy bash Project/setup.sh từ machine trước.' >&2; exit 1
fi
"$PROJECT_DOC_PYTHON" scripts/prepare_data.py --download
.venv/bin/python scripts/organize_datasets.py
.venv/bin/python scripts/build_custom_dataset.py
.venv/bin/python scripts/prepare_reliability_data.py
.venv/bin/python scripts/validate_reliability_data.py
if [[ "${1:-}" == '--train' ]]; then
  .venv/bin/python scripts/train.py
elif [[ "${1:-}" == '--finetune' ]]; then
  .venv/bin/python scripts/finetune_minilm.py --stage all --epochs 3 --patience 1 --bf16
elif [[ "${1:-}" == '--reliable' ]]; then
  .venv/bin/python scripts/train_reliable_deberta.py --epochs 3 --patience 2 --batch-size 4 --accumulation 8 --max-length 512 --mps-bf16
elif [[ "${1:-}" == '--final-minilm' ]]; then
  .venv/bin/python scripts/run_final_minilm_trial.py --synthetic-drafts --bf16
elif [[ "${1:-}" == '--best-minilm' ]]; then
  .venv/bin/python scripts/run_final_minilm_trial.py --bf16
  .venv/bin/python scripts/summarize_final_minilm_trial.py --controls-only
elif [[ $# -gt 0 ]]; then
  printf '%s\n' 'Cách dùng: bash Project/reproduce.sh [--train|--finetune|--reliable|--final-minilm|--best-minilm]' >&2; exit 1
fi
.venv/bin/python scripts/plot_results.py
if [[ -f reports/fine_tuning/stanford/test_results.json && -f reports/fine_tuning/upenn/test_results.json ]]; then
  .venv/bin/python scripts/package_finetuned_models.py
  .venv/bin/python scripts/plot_finetuning.py
  .venv/bin/python scripts/blend_trained_models.py
  .venv/bin/python scripts/package_blended_models.py
fi
if [[ -f reports/kaggle_run/provided_log_excerpt.txt ]]; then
  .venv/bin/python scripts/summarize_kaggle_run.py
  .venv/bin/python scripts/summarize_kaggle_baselines.py
fi
if [[ -f reports/reliability/deberta_real_only/results.json ]]; then
  .venv/bin/python scripts/summarize_reliability.py
  .venv/bin/python scripts/audit_custom_scenarios.py
  .venv/bin/python scripts/summarize_model_comparison.py
fi
if [[ -f reports/final_minilm_trial/control_results.json || -f reports/final_minilm_trial/results.json ]]; then
  .venv/bin/python scripts/summarize_final_minilm_trial.py
fi
"$PROJECT_DOC_PYTHON" scripts/build_report.py
PROJECT_SERVER_PID=''
cleanup() { if [[ -n "$PROJECT_SERVER_PID" ]]; then kill "$PROJECT_SERVER_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
if curl -q --silent --fail http://127.0.0.1:8766/api/info >/dev/null; then
  printf '%s\n' 'Cổng minh chứng 8766 đang được dùng; dừng tiến trình đó rồi chạy lại.' >&2; exit 1
fi
.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8766 > .cache/app-evidence.log 2>&1 &
PROJECT_SERVER_PID=$!
for attempt in {1..30}; do
  if curl -q --silent --fail http://127.0.0.1:8766/api/info >/dev/null; then break; fi
  sleep 1
done
PROJECT_APP_URL=http://127.0.0.1:8766 .venv/bin/python scripts/check_api.py
if [[ -f reports/final_minilm_trial/summary.json ]]; then
  PROJECT_APP_URL=http://127.0.0.1:8766 .venv/bin/python scripts/check_final_behavior.py
fi
PROJECT_APP_URL=http://127.0.0.1:8766 "$PROJECT_NODE" scripts/capture_app.cjs
"$PROJECT_DOC_PYTHON" scripts/build_complete_report.py
.venv/bin/python scripts/make_manifest.py
.venv/bin/python scripts/verify_evidence.py
du -sh "$PROJECT_ROOT"
