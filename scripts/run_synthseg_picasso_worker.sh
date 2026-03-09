#!/usr/bin/env bash
#SBATCH -J synthseg_worker
#SBATCH --time=0-08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# =============================================================================
# SynthSeg Evaluation Worker for Picasso
#
# Runs export, segment, or analyze stage depending on the first argument.
# Called by run_synthseg_picasso.sh via sbatch dependencies.
#
# Usage (internal — called by sbatch):
#   run_synthseg_picasso_worker.sh export
#   run_synthseg_picasso_worker.sh segment
#   run_synthseg_picasso_worker.sh analyze
#
# Expected env vars (exported by run_synthseg_picasso.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_ABS, OUTPUT_DIR
# =============================================================================

set -euo pipefail

MODE="${1:-export}"
START_TIME=$(date +%s)

echo "=========================================="
echo " SynthSeg Worker (${MODE})"
echo "=========================================="
echo "Time:     $(date)"
echo "Hostname: $(hostname)"
echo "SLURM ID: ${SLURM_JOB_ID:-local}"
echo "Config:   ${CONFIG_ABS:-not set}"
echo ""

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done

if [ "$module_loaded" -eq 0 ]; then
  echo "[env] No conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"

cd "${REPO_SRC}"

# ========================================================================
# GPU INFO (for segment stage)
# ========================================================================
if [ "${MODE}" = "segment" ]; then
  echo ""
  echo "[gpu] Checking GPU availability..."
  python -c "
import os
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f'TensorFlow GPUs: {len(gpus)}')
except ImportError:
    print('TensorFlow not in this env (expected — SynthSeg runs via subprocess)')
" 2>/dev/null || true
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[gpu] nvidia-smi not available"
fi

# ========================================================================
# RUN STAGE
# ========================================================================
echo ""
echo "=== Running ${MODE} stage ==="
echo ""

python -m pacs_sr.cli.synthseg_eval --config "${CONFIG_ABS}" "${MODE}"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "=== ${MODE} complete in ${ELAPSED}s ==="
