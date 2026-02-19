#!/usr/bin/env bash
#SBATCH -J seram_expert
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# =============================================================================
# SERAM: Expert Generation Worker
#
# Runs BSPLINE or ECLARE generation depending on the first argument.
# Called by generate_experts.sh launcher.
#
# Usage (internal — called by sbatch):
#   generate_experts_worker.sh bspline
#   generate_experts_worker.sh eclare
#
# Expected env vars (exported by generate_experts.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG, SOURCE_H5, EXPERTS_DIR,
#   SPACINGS, PULSES
# =============================================================================

set -euo pipefail

MODE="${1:-bspline}"
START_TIME=$(date +%s)
echo "SERAM expert generation (${MODE}) started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"

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

echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"

if [ "${MODE}" = "eclare" ]; then
    python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Devices:', torch.cuda.device_count())"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true
fi

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

# Verify HDF5 source file
if [ -f "${SOURCE_H5}" ]; then
    echo "[OK]   ${SOURCE_H5}"
else
    echo "[MISS] ${SOURCE_H5}"
    echo "ERROR: Source HDF5 file missing. Aborting."
    exit 1
fi

# Verify experts directory exists (create if needed)
mkdir -p "${EXPERTS_DIR}"
echo "[OK]   ${EXPERTS_DIR}"

# Verify HDF5 has expected structure
python -c "
import h5py
with h5py.File('${SOURCE_H5}', 'r') as f:
    n_hr = len([k for k in f['high_resolution']])
    n_lr_spacings = len([k for k in f['low_resolution']])
    print(f'HR patients: {n_hr}')
    print(f'LR spacings: {n_lr_spacings}')
"

if [ "${MODE}" = "eclare" ]; then
    python -c "import eclare; print('ECLARE available')" || {
        echo "ERROR: eclare package not found. Run: pip install eclare"
        exit 1
    }
fi

# ========================================================================
# BSPLINE GENERATION (CPU, parallelized via scipy.ndimage.zoom)
# ========================================================================
generate_bspline() {
    echo ""
    echo "=========================================="
    echo "GENERATING BSPLINE EXPERT OUTPUTS"
    echo "=========================================="

    python scripts/generate_bspline.py \
        --config "${CONFIG}" \
        --n-jobs "${SLURM_CPUS_PER_TASK:-8}"

    echo "BSPLINE generation complete."
}

# ========================================================================
# ECLARE GENERATION (GPU, sequential per volume via eclare_h5_adapter.py)
# ========================================================================
generate_eclare() {
    echo ""
    echo "=========================================="
    echo "GENERATING ECLARE EXPERT OUTPUTS"
    echo "=========================================="

    ECLARE_H5="${EXPERTS_DIR}/eclare.h5"

    TOTAL=0
    DONE=0
    SKIP=0
    FAIL=0

    # Get patient list from source_data.h5
    PATIENTS=$(python -c "
from pacs_sr.data.hdf5_io import list_groups
pts = list_groups('${SOURCE_H5}', 'high_resolution')
print(' '.join(pts))
")

    # Count total tasks
    for patient_id in ${PATIENTS}; do
        for spacing in ${SPACINGS}; do
            for pulse in ${PULSES}; do
                TOTAL=$((TOTAL + 1))
            done
        done
    done
    echo "Total tasks: ${TOTAL}"

    TASK_IDX=0
    for spacing in ${SPACINGS}; do
        # Extract numeric thickness from spacing (e.g., "3mm" -> 3)
        THICKNESS=$(echo "${spacing}" | sed 's/mm$//')

        for patient_id in ${PATIENTS}; do
            for pulse in ${PULSES}; do
                TASK_IDX=$((TASK_IDX + 1))

                echo "[${TASK_IDX}/${TOTAL}] Processing: ${patient_id} | ${spacing} | ${pulse}"

                if python scripts/eclare_h5_adapter.py \
                    --source-h5 "${SOURCE_H5}" \
                    --expert-h5 "${ECLARE_H5}" \
                    --patient-id "${patient_id}" \
                    --spacing "${spacing}" \
                    --pulse "${pulse}" \
                    --thickness "${THICKNESS}" \
                    --gpu-id 0 2>&1; then
                    DONE=$((DONE + 1))
                else
                    FAIL=$((FAIL + 1))
                fi

                # Progress report every 50 tasks
                if [ $((TASK_IDX % 50)) -eq 0 ]; then
                    ELAPSED=$(($(date +%s) - START_TIME))
                    AVG=$((ELAPSED / TASK_IDX))
                    ETA=$(( AVG * (TOTAL - TASK_IDX) / 60 ))
                    echo "  Progress: ${TASK_IDX}/${TOTAL} | Done: ${DONE} Skip: ${SKIP} Fail: ${FAIL} | ETA: ~${ETA}min"
                fi
            done
        done
    done

    echo ""
    echo "ECLARE generation complete."
    echo "  Done: ${DONE} | Skipped: ${SKIP} | Failed: ${FAIL} | Total: ${TOTAL}"
}

# ========================================================================
# DISPATCH
# ========================================================================
case "${MODE}" in
    bspline)
        generate_bspline
        ;;
    eclare)
        generate_eclare
        ;;
    *)
        echo "Unknown mode: ${MODE}. Use 'bspline' or 'eclare'."
        exit 1
        ;;
esac

# ========================================================================
# OUTPUT VERIFICATION
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

if [ "${MODE}" = "bspline" ]; then
    H5_FILE="${EXPERTS_DIR}/bspline.h5"
else
    H5_FILE="${EXPERTS_DIR}/eclare.h5"
fi

if [ -f "${H5_FILE}" ]; then
    python -c "
import h5py
with h5py.File('${H5_FILE}', 'r') as f:
    def count_datasets(grp, prefix=''):
        n = 0
        for k in grp:
            path = f'{prefix}/{k}' if prefix else k
            if isinstance(grp[k], h5py.Dataset):
                n += 1
            else:
                n += count_datasets(grp[k], path)
        return n
    total = count_datasets(f)
    print(f'Total datasets in ${H5_FILE}: {total}')
    for spacing in f:
        n = count_datasets(f[spacing])
        print(f'  {spacing}: {n} datasets')
"
else
    echo "WARNING: ${H5_FILE} not found"
fi

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "EXPERT GENERATION (${MODE}) COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
