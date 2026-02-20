#!/usr/bin/env bash
# =============================================================================
# SERAM: Post-Expert Pipeline — SLURM Launcher
#
# Submits manifest → fold training → metrics/figures jobs on Picasso,
# assuming expert HDF5 files (bspline.h5, eclare.h5) already exist.
#
# This is the second half of the pipeline — run after experts are generated
# via generate_experts.sh (or manually).
#
# Usage (from Picasso login node):
#   bash scripts/run_seram_picasso.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo " SERAM: Post-Expert Pipeline Launcher"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="pacs"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/PaCS-SR"
export CONFIG="${REPO_SRC}/configs/seram_glioma_picasso.yaml"

# HDF5 data paths on Picasso
export SOURCE_H5="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/gliomas/source_data.h5"
export EXPERTS_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/gliomas/experts"

# Experiment parameters
export SPACINGS="3mm 5mm 7mm"
export PULSES="t1c t2w t2f"

# Log directory
export LOG_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/pacs_sr/logs"
mkdir -p "${LOG_DIR}"

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Source H5:   ${SOURCE_H5}"
echo "  Experts dir: ${EXPERTS_DIR}"
echo "  Spacings:    ${SPACINGS}"
echo "  Pulses:      ${PULSES}"
echo "  Conda env:   ${CONDA_ENV_NAME}"
echo ""

# ========================================================================
# PRE-FLIGHT: Verify expert HDF5 files exist before submitting
# ========================================================================
echo "[Pre-flight] Checking expert HDF5 files..."
MISSING=0

if [ ! -f "${SOURCE_H5}" ]; then
    echo "  [MISS] source_data.h5: ${SOURCE_H5}"
    MISSING=1
else
    echo "  [OK]   source_data.h5: ${SOURCE_H5}"
fi

for expert in bspline eclare; do
    h5_file="${EXPERTS_DIR}/${expert}.h5"
    if [ ! -f "${h5_file}" ]; then
        echo "  [MISS] ${expert}.h5: ${h5_file}"
        MISSING=1
    else
        echo "  [OK]   ${expert}.h5: ${h5_file}"
    fi
done

if [ "${MISSING}" -eq 1 ]; then
    echo ""
    echo "ERROR: Required HDF5 files missing. Generate experts first:"
    echo "  bash scripts/generate_experts.sh"
    exit 1
fi
echo ""

# Helper: extract numeric job ID from sbatch output (Picasso lua wrapper
# may print warnings to stdout that corrupt --parsable output).
_jobid() { echo "$1" | grep -oE '[0-9]+' | tail -1; }

# ========================================================================
# SUBMIT MANIFEST JOB (no dependency — experts already exist)
# ========================================================================
MANIFEST_JOB_ID=$(_jobid "$(sbatch --parsable \
    --job-name="seram_manifest" \
    --time=0-00:30:00 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --constraint=cpu \
    --output="${LOG_DIR}/manifest_%j.out" \
    --error="${LOG_DIR}/manifest_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/run_seram_picasso_worker.sh" manifest 2>&1)")

echo "MANIFEST job submitted: ${MANIFEST_JOB_ID}"

# ========================================================================
# SUBMIT PaCS-SR FOLD JOBS (each depends on manifest)
# ========================================================================
FOLD_JOBS=""
for fold in 1 2 3 4 5; do
    FOLD_JOB_ID=$(_jobid "$(sbatch --parsable \
        --dependency=afterok:${MANIFEST_JOB_ID} \
        --job-name="seram_fold${fold}" \
        --time=0-02:00:00 \
        --ntasks=1 \
        --cpus-per-task=12 \
        --mem=32G \
        --constraint=cpu \
        --output="${LOG_DIR}/fold${fold}_%j.out" \
        --error="${LOG_DIR}/fold${fold}_%j.err" \
        --export=ALL \
        "${SCRIPT_DIR}/run_seram_picasso_worker.sh" train "${fold}" 2>&1)")

    echo "FOLD ${fold} job submitted: ${FOLD_JOB_ID} (after manifest)"
    FOLD_JOBS="${FOLD_JOBS}:${FOLD_JOB_ID}"
done

# ========================================================================
# SUBMIT METRICS + FIGURES JOB (depends on all folds completing)
# ========================================================================
METRICS_JOB_ID=$(_jobid "$(sbatch --parsable \
    --dependency=afterok${FOLD_JOBS} \
    --job-name="seram_metrics" \
    --time=0-01:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=16G \
    --constraint=cpu \
    --output="${LOG_DIR}/metrics_%j.out" \
    --error="${LOG_DIR}/metrics_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/run_seram_picasso_worker.sh" metrics 2>&1)")

echo "METRICS job submitted: ${METRICS_JOB_ID} (after all folds)"

# ========================================================================
# SUMMARY
# ========================================================================
echo ""
echo "=========================================="
echo " JOBS SUBMITTED"
echo "=========================================="
echo "MANIFEST: ${MANIFEST_JOB_ID}  (immediate, ~10 min)"
echo "FOLDS:    ${FOLD_JOBS#:}  (after manifest, ~2 hrs each)"
echo "METRICS:  ${METRICS_JOB_ID}  (after all folds, ~1 hr)"
echo ""
echo "Dependency chain:"
echo "  MANIFEST → FOLD_1..5 → METRICS"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/manifest_${MANIFEST_JOB_ID}.out"
echo ""
