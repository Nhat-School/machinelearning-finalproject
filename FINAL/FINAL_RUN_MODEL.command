#!/usr/bin/env bash
set -euo pipefail
FINAL_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$FINAL_DIR/../scripts/launchers/env.sh"
if [[ $# -gt 0 ]]; then
  FINAL_TEXT="$*"
else
  read -r -p "Enter an English forum post: " FINAL_TEXT
fi
"$PROJECT_ROOT/.venv/bin/python" "$FINAL_DIR/FINAL_MODEL/predict.py" --text "$FINAL_TEXT"
if [[ $# -eq 0 ]]; then
  read -r -p "Press Enter to close." FINAL_DONE
fi
