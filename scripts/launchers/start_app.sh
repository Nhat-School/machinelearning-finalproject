#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
if [[ ! -x .venv/bin/python || ! -f models/deployment.json ]]; then
  printf '%s\n' 'Cần chạy setup.sh và huấn luyện trước. Xem README.md.' >&2
  exit 1
fi
exec .venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8765
