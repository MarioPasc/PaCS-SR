#!/usr/bin/env bash
#SBATCH -J seram_pacs
#SBATCH --time=0-08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

# =============================================================================
# SERAM: PaCS-SR Pipeline Worker
#
# Runs manifest building, per-fold training, or metrics/figures depending on
# the first argument. Called by generate_experts.sh via sbatch dependencies.
#
# Usage (internal — called by sbatch):
#   run_seram_picasso_worker.sh manifest
#   run_seram_picasso_worker.sh train <fold_number>
#   run_seram_picasso_worker.sh metrics
#   run_seram_picasso_worker.sh figures
#
# Expected env vars (exported by generate_experts.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG, SOURCE_H5, EXPERTS_DIR
# =============================================================================

set -euo pipefail

MODE="${1:-manifest}"
FOLD="${2:-}"
START_TIME=$(date +%s)

echo "=========================================="
echo " SERAM PaCS-SR Worker (${MODE})"
echo "=========================================="
echo "Time:     $(date)"
echo "Hostname: $(hostname)"
echo "SLURM ID: ${SLURM_JOB_ID:-local}"
[ -n "${FOLD}" ] && echo "Fold:     ${FOLD}"
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
# PRE-FLIGHT: Verify HDF5 files
# ========================================================================
echo ""
echo "[pre-flight] Verifying HDF5 files..."

if [ -f "${SOURCE_H5}" ]; then
    echo "  [OK] source_data.h5: ${SOURCE_H5}"
else
    echo "  [MISS] source_data.h5: ${SOURCE_H5}"
    echo "ERROR: Source HDF5 missing. Aborting."
    exit 1
fi

if [ -d "${EXPERTS_DIR}" ]; then
    echo "  [OK] experts dir: ${EXPERTS_DIR}"
else
    echo "  [MISS] experts dir: ${EXPERTS_DIR}"
    echo "ERROR: Experts directory missing. Aborting."
    exit 1
fi

# ========================================================================
# DISPATCH
# ========================================================================
case "${MODE}" in
    manifest)
        echo ""
        echo "[manifest] Building K-fold manifest..."
        python -m pacs_sr.cli.build_manifest --config "${CONFIG}"
        echo "[manifest] Done."
        ;;

    train)
        if [ -z "${FOLD}" ]; then
            echo "ERROR: train mode requires fold number as second argument."
            exit 1
        fi
        echo ""
        echo "[train] Training fold ${FOLD}..."
        python -m pacs_sr.cli.train --config "${CONFIG}" --fold "${FOLD}"
        echo "[train] Fold ${FOLD} done."
        ;;

    metrics)
        echo ""
        echo "[metrics] Computing SERAM metrics..."
        python scripts/seram_compute_metrics.py --config "${CONFIG}"

        echo ""
        echo "[figures] Generating figures and table..."
        python -m pacs_sr.seram.figure_comparison_grid --config "${CONFIG}"
        python -m pacs_sr.seram.figure_performance_plot --config "${CONFIG}"
        python -m pacs_sr.seram.table_msssim --config "${CONFIG}"
        echo "[figures] Done."
        ;;

    figures)
        echo ""
        echo "[figures] Generating figures and table..."
        python -m pacs_sr.seram.figure_comparison_grid --config "${CONFIG}"
        python -m pacs_sr.seram.figure_performance_plot --config "${CONFIG}"
        python -m pacs_sr.seram.table_msssim --config "${CONFIG}"
        echo "[figures] Done."
        ;;

    *)
        echo "Unknown mode: ${MODE}. Use 'manifest', 'train', 'metrics', or 'figures'."
        exit 1
        ;;
esac

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo " ${MODE} COMPLETED"
echo "=========================================="
echo "Finished: $(date)"
echo "Duration: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
