#!/usr/bin/env bash
# =============================================================================
# SynthSeg Evaluation Pipeline for PaCS-SR on Picasso
#
# Submits a 3-stage SLURM pipeline:
#   1. Export:  HDF5 → NIfTI (CPU, ~1h)
#   2. Segment: Run SynthSeg per method/spacing (CPU/GPU, ~2h each)
#   3. Analyze: Compute metrics + stats (CPU, ~30min)
#
# The SynthSeg command path is read from the YAML config (synthseg.command).
#
# Usage:
#   bash scripts/run_synthseg_picasso.sh \
#       --config configs/seram_glioma_picasso.yaml \
#       [--partition gpu] [--account myaccount] [--pacs-sr-env /path/to/env]
#
# Prerequisites:
#   - SynthSeg installed (see neuromf setup_synthseg.sh)
#   - PaCS-SR conda environment activated (or pass --pacs-sr-env)
#   - synthseg.command configured in the YAML config file
# =============================================================================

set -euo pipefail

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
CONFIG=""
PARTITION="batch"
ACCOUNT=""
PACS_SR_ENV="${CONDA_PREFIX:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";       shift 2 ;;
        --partition)    PARTITION="$2";     shift 2 ;;
        --account)      ACCOUNT="$2";      shift 2 ;;
        --pacs-sr-env)  PACS_SR_ENV="$2";  shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/run_synthseg_picasso.sh --config <yaml>"
            exit 1
            ;;
    esac
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

if [ -z "${PACS_SR_ENV}" ]; then
    echo "Error: activate PaCS-SR conda env or pass --pacs-sr-env"
    exit 1
fi

ACCOUNT_FLAG=""
if [ -n "${ACCOUNT}" ]; then
    ACCOUNT_FLAG="#SBATCH --account=${ACCOUNT}"
fi

# Parse output_dir from YAML using grep/sed (no Python dependency)
OUTPUT_DIR=$(grep -A0 '^\s*output_dir:' "${CONFIG}" | head -1 | sed 's/.*output_dir:\s*//' | sed 's/["\x27]//g' | xargs)
if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="results/synthseg_evaluation"
fi

# Get absolute path to config for use inside SLURM jobs
CONFIG_ABS=$(readlink -f "${CONFIG}")

echo "=========================================="
echo "PaCS-SR SynthSeg Evaluation Pipeline"
echo "=========================================="
echo "Config:        ${CONFIG_ABS}"
echo "PaCS-SR env:   ${PACS_SR_ENV}"
echo "Partition:     ${PARTITION}"
echo "Output dir:    ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}/slurm_logs"

# ========================================================================
# STAGE 1: EXPORT (HDF5 → NIfTI)
# ========================================================================
EXPORT_SCRIPT="${OUTPUT_DIR}/slurm_logs/export.sh"
cat > "${EXPORT_SCRIPT}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=synthseg-export
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/export_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/export_%j.err
${ACCOUNT_FLAG}

source activate ${PACS_SR_ENV} 2>/dev/null || conda activate ${PACS_SR_ENV}

echo "=== Stage 1: Export ==="
python -m pacs_sr.cli.synthseg_eval --config ${CONFIG_ABS} export
echo "=== Export complete ==="
HEREDOC

EXPORT_JOB=$(sbatch --parsable "${EXPORT_SCRIPT}")
echo "Submitted export job: ${EXPORT_JOB}"

# ========================================================================
# STAGE 2: SEGMENT (Run SynthSeg)
# ========================================================================
SEGMENT_SCRIPT="${OUTPUT_DIR}/slurm_logs/segment.sh"
cat > "${SEGMENT_SCRIPT}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=synthseg-segment
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/segment_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/segment_%j.err
${ACCOUNT_FLAG}

source activate ${PACS_SR_ENV} 2>/dev/null || conda activate ${PACS_SR_ENV}

echo "=== Stage 2: Segment (SynthSeg) ==="
python -m pacs_sr.cli.synthseg_eval --config ${CONFIG_ABS} segment
echo "=== Segment complete ==="
HEREDOC

SEGMENT_JOB=$(sbatch --parsable --dependency=afterok:${EXPORT_JOB} "${SEGMENT_SCRIPT}")
echo "Submitted segment job: ${SEGMENT_JOB} (depends on ${EXPORT_JOB})"

# ========================================================================
# STAGE 3: ANALYZE (Metrics + Statistics)
# ========================================================================
ANALYZE_SCRIPT="${OUTPUT_DIR}/slurm_logs/analyze.sh"
cat > "${ANALYZE_SCRIPT}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=synthseg-analyze
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/analyze_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/analyze_%j.err
${ACCOUNT_FLAG}

source activate ${PACS_SR_ENV} 2>/dev/null || conda activate ${PACS_SR_ENV}

echo "=== Stage 3: Analyze ==="
python -m pacs_sr.cli.synthseg_eval --config ${CONFIG_ABS} analyze
echo "=== Analysis complete ==="
HEREDOC

ANALYZE_JOB=$(sbatch --parsable --dependency=afterok:${SEGMENT_JOB} "${ANALYZE_SCRIPT}")
echo "Submitted analyze job: ${ANALYZE_JOB} (depends on ${SEGMENT_JOB})"

echo ""
echo "=========================================="
echo "Pipeline submitted:"
echo "  Export:  ${EXPORT_JOB}"
echo "  Segment: ${SEGMENT_JOB}"
echo "  Analyze: ${ANALYZE_JOB}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT_DIR}/"
echo "=========================================="
