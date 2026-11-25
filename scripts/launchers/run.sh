#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == '--download' ]]; then shift; fi
exec bash "$(dirname "$0")/reproduce.sh" "$@"
