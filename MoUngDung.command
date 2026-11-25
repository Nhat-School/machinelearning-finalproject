#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if curl -q --silent --fail http://127.0.0.1:8765/api/info >/dev/null; then
  open http://127.0.0.1:8765
  exit 0
fi
bash scripts/launchers/start_app.sh &
PROJECT_WEB_PID=$!
trap 'kill "$PROJECT_WEB_PID" 2>/dev/null || true' EXIT
for attempt in {1..30}; do
  if curl -q --silent --fail http://127.0.0.1:8765/api/info >/dev/null; then
    open http://127.0.0.1:8765
    wait "$PROJECT_WEB_PID"
    exit 0
  fi
  sleep 1
done
printf '%s\n' 'Không khởi động được web. Xem thông báo lỗi trong Terminal.' >&2
exit 1
