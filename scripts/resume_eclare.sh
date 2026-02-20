#!/usr/bin/env bash
# =============================================================================
# SERAM: Resume ECLARE Generation — Sharded SLURM Launcher
#
# Submits a SLURM job array to parallelize ECLARE across multiple GPUs.
# Each array task processes every Nth volume (round-robin sharding),
# writing to its own HDF5 shard file. A final merge job combines shards
# into the main eclare.h5.
#
# Automatically skips volumes already present in eclare.h5 from previous
# runs, so it's safe to resubmit after partial completion.
#
# Usage (from Picasso login node):
#   bash scripts/resume_eclare.sh              # 4 shards (default)
#   bash scripts/resume_eclare.sh 8            # 8 shards for faster throughput
#
# Time estimate (A100, ~3 min/volume):
#   657 remaining tasks / 4 GPUs ≈ 8h
#   657 remaining tasks / 8 GPUs ≈ 4h
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_SHARDS="${1:-4}"

echo "=========================================="
echo " SERAM: Resume ECLARE (${NUM_SHARDS} shards)"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="pacs"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/PaCS-SR"
export CONFIG="${REPO_SRC}/configs/seram_glioma_picasso.yaml"

export SOURCE_H5="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/gliomas/source_data.h5"
export EXPERTS_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/gliomas/experts"

export SPACINGS="3mm 5mm 7mm"
export PULSES="t1c t2w t2f"

export LOG_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/pacs_sr/logs"
mkdir -p "${LOG_DIR}"

# Activate conda on the login node for pre-flight checks
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

echo "Configuration:"
echo "  Shards:      ${NUM_SHARDS}"
echo "  Repo:        ${REPO_SRC}"
echo "  Source H5:   ${SOURCE_H5}"
echo "  Experts dir: ${EXPERTS_DIR}"
echo "  Spacings:    ${SPACINGS}"
echo "  Pulses:      ${PULSES}"
echo ""

# ========================================================================
# PRE-FLIGHT: Check existing progress
# ========================================================================
echo "[Pre-flight] Checking existing eclare.h5..."
ECLARE_H5="${EXPERTS_DIR}/eclare.h5"
if [ -f "${ECLARE_H5}" ]; then
    python -c "
import h5py
def count_ds(grp):
    n = 0
    for k in grp:
        if isinstance(grp[k], h5py.Dataset): n += 1
        else: n += count_ds(grp[k])
    return n
with h5py.File('${ECLARE_H5}', 'r') as f:
    total = count_ds(f)
    print(f'  Existing datasets: {total}/900')
    print(f'  Remaining:         {900 - total}')
"
else
    echo "  No existing eclare.h5 — starting fresh"
fi
echo ""

if [ ! -f "${SOURCE_H5}" ]; then
    echo "ERROR: Source HDF5 missing: ${SOURCE_H5}"
    exit 1
fi

# ========================================================================
# HELPER
# ========================================================================
_jobid() { echo "$1" | grep -oE '[0-9]+' | tail -1; }

# ========================================================================
# SUBMIT ECLARE JOB ARRAY (N shards, each gets 1 GPU)
# ========================================================================
# Time per shard ≈ (remaining / num_shards) × 3 min
# Conservative: 12h covers up to ~240 tasks/shard
LAST_SHARD=$((NUM_SHARDS - 1))

ECLARE_ARRAY_ID=$(_jobid "$(sbatch --parsable \
    --array=0-${LAST_SHARD} \
    --job-name="seram_eclare" \
    --time=0-12:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --constraint=dgx \
    --gres=gpu:1 \
    --output="${LOG_DIR}/eclare_shard_%a_%j.out" \
    --error="${LOG_DIR}/eclare_shard_%a_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/generate_eclare_worker.sh" "${NUM_SHARDS}" 2>&1)")

echo "ECLARE array submitted: ${ECLARE_ARRAY_ID} (${NUM_SHARDS} tasks, 0-${LAST_SHARD})"

# ========================================================================
# SUBMIT MERGE JOB (depends on all array tasks completing)
# ========================================================================
MERGE_JOB_ID=$(_jobid "$(sbatch --parsable \
    --dependency=afterok:${ECLARE_ARRAY_ID} \
    --job-name="seram_eclare_merge" \
    --time=0-01:00:00 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=16G \
    --constraint=cpu \
    --output="${LOG_DIR}/eclare_merge_%j.out" \
    --error="${LOG_DIR}/eclare_merge_%j.err" \
    --export=ALL \
    --wrap="cd ${REPO_SRC} && \
module load miniconda3 2>/dev/null || true && \
source \$(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || true && \
conda activate ${CONDA_ENV_NAME} 2>/dev/null || source activate ${CONDA_ENV_NAME} && \
python scripts/merge_eclare_shards.py --experts-dir ${EXPERTS_DIR} --num-shards ${NUM_SHARDS}" \
    2>&1)")

echo "MERGE job submitted:   ${MERGE_JOB_ID} (after all shards)"

# ========================================================================
# SUMMARY
# ========================================================================
echo ""
echo "=========================================="
echo " JOBS SUBMITTED"
echo "=========================================="
echo "ECLARE array: ${ECLARE_ARRAY_ID}  (${NUM_SHARDS} × 1 GPU, ~8-12h each)"
echo "MERGE:        ${MERGE_JOB_ID}  (after all shards, ~30 min)"
echo ""
echo "Dependency chain:"
echo "  ECLARE_SHARD_0..${LAST_SHARD} → MERGE"
echo ""
echo "After merge completes, run the PaCS-SR pipeline:"
echo "  bash scripts/run_seram_picasso.sh"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/eclare_shard_0_*.out"
echo "  tail -f ${LOG_DIR}/eclare_shard_1_*.out"
echo ""
