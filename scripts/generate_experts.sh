#!/usr/bin/env bash
# =============================================================================
# SERAM: Expert Generation — SLURM Launcher
#
# Login-node script that submits BSPLINE (CPU) and ECLARE (GPU) generation
# jobs on Picasso for all BraTS-GLI patients × spacings × pulses.
#
# Output:
#   {EXPERTS_DIR}/bspline.h5   — BSPLINE expert outputs
#   {EXPERTS_DIR}/eclare.h5    — ECLARE expert outputs
#
# Usage (from Picasso login node):
#   bash scripts/generate_experts.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo " SERAM: Expert Generation Launcher"
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

# Create experts directory
mkdir -p "${EXPERTS_DIR}"

# ========================================================================
# SUBMIT BSPLINE JOB (CPU-only, high parallelism)
# ========================================================================
BSPLINE_JOB_ID=$(sbatch --parsable \
    --job-name="seram_bspline" \
    --time=0-02:00:00 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=32G \
    --output="${LOG_DIR}/bspline_%j.out" \
    --error="${LOG_DIR}/bspline_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/generate_experts_worker.sh" bspline)

echo "BSPLINE job submitted: ${BSPLINE_JOB_ID}"

# ========================================================================
# SUBMIT ECLARE JOB (GPU, sequential per volume)
# ========================================================================
ECLARE_JOB_ID=$(sbatch --parsable \
    --job-name="seram_eclare" \
    --time=0-12:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --constraint=dgx \
    --gres=gpu:1 \
    --output="${LOG_DIR}/eclare_%j.out" \
    --error="${LOG_DIR}/eclare_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/generate_experts_worker.sh" eclare)

echo "ECLARE job submitted:  ${ECLARE_JOB_ID}"

# ========================================================================
# SUBMIT MANIFEST JOB (depends on both experts completing)
# ========================================================================
MANIFEST_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${BSPLINE_JOB_ID},${ECLARE_JOB_ID} \
    --job-name="seram_manifest" \
    --time=0-00:30:00 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --output="${LOG_DIR}/manifest_%j.out" \
    --error="${LOG_DIR}/manifest_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/run_seram_picasso_worker.sh" manifest)

echo "MANIFEST job submitted: ${MANIFEST_JOB_ID} (after experts)"

# ========================================================================
# SUBMIT PaCS-SR FOLD JOBS (each depends on manifest)
# ========================================================================
FOLD_JOBS=""
for fold in 1 2 3 4 5; do
    FOLD_JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${MANIFEST_JOB_ID} \
        --job-name="seram_fold${fold}" \
        --time=0-02:00:00 \
        --ntasks=1 \
        --cpus-per-task=12 \
        --mem=32G \
        --output="${LOG_DIR}/fold${fold}_%j.out" \
        --error="${LOG_DIR}/fold${fold}_%j.err" \
        --export=ALL \
        "${SCRIPT_DIR}/run_seram_picasso_worker.sh" train "${fold}")

    echo "FOLD ${fold} job submitted: ${FOLD_JOB_ID} (after manifest)"
    FOLD_JOBS="${FOLD_JOBS}:${FOLD_JOB_ID}"
done

# ========================================================================
# SUBMIT METRICS + FIGURES JOB (depends on all folds completing)
# ========================================================================
METRICS_JOB_ID=$(sbatch --parsable \
    --dependency=afterok${FOLD_JOBS} \
    --job-name="seram_metrics" \
    --time=0-01:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=16G \
    --output="${LOG_DIR}/metrics_%j.out" \
    --error="${LOG_DIR}/metrics_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/run_seram_picasso_worker.sh" metrics)

echo "METRICS job submitted: ${METRICS_JOB_ID} (after all folds)"

# ========================================================================
# SUMMARY
# ========================================================================
echo ""
echo "=========================================="
echo " JOBS SUBMITTED"
echo "=========================================="
echo "BSPLINE:  ${BSPLINE_JOB_ID}  (CPU, ~30 min)"
echo "ECLARE:   ${ECLARE_JOB_ID}  (GPU, ~6-10 hrs)"
echo "MANIFEST: ${MANIFEST_JOB_ID}  (after experts, ~10 min)"
echo "FOLDS:    ${FOLD_JOBS#:}  (after manifest, ~2 hrs each)"
echo "METRICS:  ${METRICS_JOB_ID}  (after all folds, ~1 hr)"
echo ""
echo "Dependency chain:"
echo "  BSPLINE + ECLARE → MANIFEST → FOLD_1..5 → METRICS"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/bspline_${BSPLINE_JOB_ID}.out"
echo "  tail -f ${LOG_DIR}/eclare_${ECLARE_JOB_ID}.out"
echo ""
echo "After completion, verify:"
echo "  python -c \"import h5py; f=h5py.File('${EXPERTS_DIR}/bspline.h5','r'); print(len(list(f.keys())))\""
echo "  python -c \"import h5py; f=h5py.File('${EXPERTS_DIR}/eclare.h5','r'); print(len(list(f.keys())))\""
