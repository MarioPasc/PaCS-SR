#!/usr/bin/env bash
# =============================================================================
# SynthSeg Evaluation Pipeline for PaCS-SR on Picasso
#
# Submits a 3-stage SLURM pipeline:
#   1. Export:  HDF5 → NIfTI (CPU, ~1h)
#   2. Segment: Run SynthSeg on GPU (DGX constraint, ~2h)
#   3. Analyze: Compute metrics + stats (CPU, ~30min)
#
# SynthSeg command path is read from the YAML config (synthseg.command).
#
# Usage (from Picasso login node):
#   bash scripts/run_synthseg_picasso.sh --config configs/seram_glioma_picasso.yaml
#
# Prerequisites:
#   - SynthSeg installed (see neuromf setup_synthseg.sh)
#   - synthseg.command configured in the YAML config file
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
CONFIG=""

for arg in "$@"; do
    case "${arg}" in
        --config)  shift; CONFIG="$1" ;;
        -h|--help)
            echo "Usage: bash scripts/run_synthseg_picasso.sh --config <yaml>"
            exit 0
            ;;
    esac
    shift 2>/dev/null || true
done

if [ -z "${CONFIG}" ]; then
    echo "Error: --config is required"
    echo "Usage: bash scripts/run_synthseg_picasso.sh --config <yaml>"
    exit 1
fi

if [ ! -f "${CONFIG}" ]; then
    echo "Error: config file not found: ${CONFIG}"
    exit 1
fi

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="pacs"
export REPO_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG_ABS="$(readlink -f "${CONFIG}")"

# Parse output_dir from YAML using grep/sed (no Python dependency on login node)
OUTPUT_DIR=$(grep -A0 '^\s*output_dir:' "${CONFIG}" | head -1 | sed 's/.*output_dir:\s*//' | sed 's/["\x27]//g' | xargs)
if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="results/synthseg_evaluation"
fi
export OUTPUT_DIR

# Log directory
export LOG_DIR="${REPO_SRC}/${OUTPUT_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo " PaCS-SR SynthSeg Evaluation Pipeline"
echo "=========================================="
echo "Time:       $(date)"
echo "Config:     ${CONFIG_ABS}"
echo "Repo:       ${REPO_SRC}"
echo "Conda env:  ${CONDA_ENV_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log dir:    ${LOG_DIR}"
echo ""

# Helper: extract numeric job ID from sbatch output (Picasso lua wrapper
# may print warnings to stdout that corrupt --parsable output).
_jobid() { echo "$1" | grep -oE '[0-9]+' | tail -1; }

# ========================================================================
# STAGE 1: EXPORT (HDF5 → NIfTI) — CPU
# ========================================================================
EXPORT_JOB_ID=$(_jobid "$(sbatch --parsable \
    --job-name="synthseg_export" \
    --time=0-01:00:00 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=8G \
    --constraint=cpu \
    --output="${LOG_DIR}/export_%j.out" \
    --error="${LOG_DIR}/export_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/run_synthseg_picasso_worker.sh" export 2>&1)")

echo "EXPORT job submitted: ${EXPORT_JOB_ID}"

# ========================================================================
# STAGE 2: SEGMENT (Run SynthSeg) — GPU (DGX)
# ========================================================================
SEGMENT_JOB_ID=$(_jobid "$(sbatch --parsable \
    --dependency=afterok:${EXPORT_JOB_ID} \
    --job-name="synthseg_segment" \
    --time=0-08:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=16G \
    --constraint=dgx \
    --gres=gpu:1 \
    --output="${LOG_DIR}/segment_%j.out" \
    --error="${LOG_DIR}/segment_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/run_synthseg_picasso_worker.sh" segment 2>&1)")

echo "SEGMENT job submitted: ${SEGMENT_JOB_ID} (after export)"

# ========================================================================
# STAGE 3: ANALYZE (Metrics + Statistics) — CPU
# ========================================================================
ANALYZE_JOB_ID=$(_jobid "$(sbatch --parsable \
    --dependency=afterok:${SEGMENT_JOB_ID} \
    --job-name="synthseg_analyze" \
    --time=0-00:30:00 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=4G \
    --constraint=cpu \
    --output="${LOG_DIR}/analyze_%j.out" \
    --error="${LOG_DIR}/analyze_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/run_synthseg_picasso_worker.sh" analyze 2>&1)")

echo "ANALYZE job submitted: ${ANALYZE_JOB_ID} (after segment)"

echo ""
echo "=========================================="
echo " JOBS SUBMITTED"
echo "=========================================="
echo "EXPORT:  ${EXPORT_JOB_ID}  (immediate, ~30 min)"
echo "SEGMENT: ${SEGMENT_JOB_ID}  (after export, ~4 hrs on GPU)"
echo "ANALYZE: ${ANALYZE_JOB_ID}  (after segment, ~15 min)"
echo ""
echo "Dependency chain:"
echo "  EXPORT → SEGMENT (GPU) → ANALYZE"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/export_${EXPORT_JOB_ID}.out"
echo ""
