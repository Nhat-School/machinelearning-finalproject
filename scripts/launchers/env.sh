#!/usr/bin/env bash
# Source only from a project subprocess; no shell profile or global config edits.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_REQUIRE_VIRTUALENV=true
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export SENTENCE_TRANSFORMERS_HOME="$PROJECT_ROOT/.cache/sentence-transformers"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export PYTHONPYCACHEPREFIX="$PROJECT_ROOT/.cache/pycache"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export IPYTHONDIR="$PROJECT_ROOT/.cache/ipython"
export JUPYTER_CONFIG_DIR="$PROJECT_ROOT/.cache/jupyter/config"
export JUPYTER_DATA_DIR="$PROJECT_ROOT/.cache/jupyter/data"
export JUPYTER_RUNTIME_DIR="$PROJECT_ROOT/.cache/jupyter/runtime"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
unset PYTHONPATH PYTHONHOME
mkdir -p "$TMPDIR" "$PROJECT_ROOT/data/raw" "$PROJECT_ROOT/data/processed" \
  "$PROJECT_ROOT/reports/figures" "$PROJECT_ROOT/models"
