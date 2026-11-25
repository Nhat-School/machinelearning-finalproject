#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi
.venv/bin/python -m pip install --only-binary=:all: -r requirements-ml.txt
.venv/bin/python -m pip freeze > requirements-ml.lock.txt
.venv/bin/python -m pip check
