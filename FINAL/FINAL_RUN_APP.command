#!/usr/bin/env bash
set -euo pipefail
FINAL_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$FINAL_DIR/../MoUngDung.command"
