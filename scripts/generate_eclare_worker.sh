#!/usr/bin/env bash
#SBATCH -J seram_eclare
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# =============================================================================
# SERAM: Sharded ECLARE Worker
#
# Processes a subset of (spacing, patient, pulse) tasks based on
# SLURM_ARRAY_TASK_ID. Each shard writes to its own HDF5 file to avoid
# concurrent write conflicts.
#
# Usage (internal — called via sbatch --array):
#   generate_eclare_worker.sh <num_shards>
#
# Expected env vars (exported by launcher):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG, SOURCE_H5, EXPERTS_DIR,
#   SPACINGS, PULSES, SLURM_ARRAY_TASK_ID
# =============================================================================

set -euo pipefail

NUM_SHARDS="${1:-4}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_TIME=$(date +%s)

echo "=========================================="
echo " ECLARE Sharded Worker"
echo "=========================================="
echo "Shard:    ${SHARD_ID} / ${NUM_SHARDS}"
echo "Time:     $(date)"
echo "Hostname: $(hostname)"
echo "SLURM ID: ${SLURM_JOB_ID:-local}"
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
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Devices:', torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true

# ========================================================================
# PRE-FLIGHT
# ========================================================================
cd "${REPO_SRC}"

if [ ! -f "${SOURCE_H5}" ]; then
    echo "ERROR: Source HDF5 missing: ${SOURCE_H5}"
    exit 1
fi

mkdir -p "${EXPERTS_DIR}"

python -c "import eclare; print('ECLARE available')" || {
    echo "ERROR: eclare package not found."
    exit 1
}

# ========================================================================
# BUILD TASK LIST AND SELECT THIS SHARD'S SUBSET
# ========================================================================
echo ""
echo "[shard ${SHARD_ID}] Building task list..."

# Existing eclare.h5 for skip-checking (read-only)
ECLARE_MAIN="${EXPERTS_DIR}/eclare.h5"

# This shard writes to its own file
ECLARE_SHARD="${EXPERTS_DIR}/eclare_shard_${SHARD_ID}.h5"

# Get patient list
PATIENTS=$(python -c "
from pacs_sr.data.hdf5_io import list_groups
pts = list_groups('${SOURCE_H5}', 'high_resolution')
print(' '.join(pts))
")

# Build the full task list (spacing × patient × pulse) and select this shard
TASK_IDX=0
SHARD_TOTAL=0
SHARD_DONE=0
SHARD_FAIL=0

# Count total and shard tasks first
TOTAL=0
MY_TOTAL=0
for spacing in ${SPACINGS}; do
    for patient_id in ${PATIENTS}; do
        for pulse in ${PULSES}; do
            if [ $((TOTAL % NUM_SHARDS)) -eq "${SHARD_ID}" ]; then
                MY_TOTAL=$((MY_TOTAL + 1))
            fi
            TOTAL=$((TOTAL + 1))
        done
    done
done

echo "[shard ${SHARD_ID}] Total tasks: ${TOTAL}, this shard: ${MY_TOTAL}"
echo ""

# ========================================================================
# PROCESS THIS SHARD'S TASKS
# ========================================================================
TASK_IDX=0
MY_IDX=0

for spacing in ${SPACINGS}; do
    THICKNESS=$(echo "${spacing}" | sed 's/mm$//')

    for patient_id in ${PATIENTS}; do
        for pulse in ${PULSES}; do
            # Only process tasks assigned to this shard
            if [ $((TASK_IDX % NUM_SHARDS)) -eq "${SHARD_ID}" ]; then
                MY_IDX=$((MY_IDX + 1))

                echo "[${MY_IDX}/${MY_TOTAL}] Processing: ${patient_id} | ${spacing} | ${pulse}"

                # Skip if already in the main eclare.h5 (from previous runs)
                ALREADY_DONE=$(python -c "
from pacs_sr.data.hdf5_io import has_key, expert_key
k = expert_key('${spacing}', '${patient_id}', '${pulse}')
print('yes' if has_key('${ECLARE_MAIN}', k) else 'no')
" 2>/dev/null || echo "no")

                if [ "${ALREADY_DONE}" = "yes" ]; then
                    echo "  SKIP (already in eclare.h5)"
                    SHARD_DONE=$((SHARD_DONE + 1))
                else
                    if python scripts/eclare_h5_adapter.py \
                        --source-h5 "${SOURCE_H5}" \
                        --expert-h5 "${ECLARE_SHARD}" \
                        --patient-id "${patient_id}" \
                        --spacing "${spacing}" \
                        --pulse "${pulse}" \
                        --thickness "${THICKNESS}" \
                        --gpu-id 0 2>&1; then
                        SHARD_DONE=$((SHARD_DONE + 1))
                    else
                        SHARD_FAIL=$((SHARD_FAIL + 1))
                    fi
                fi

                # Progress report every 25 tasks
                if [ $((MY_IDX % 25)) -eq 0 ]; then
                    ELAPSED=$(($(date +%s) - START_TIME))
                    AVG=$((ELAPSED / MY_IDX))
                    ETA=$(( AVG * (MY_TOTAL - MY_IDX) / 60 ))
                    echo "  Progress: ${MY_IDX}/${MY_TOTAL} | Done: ${SHARD_DONE} Fail: ${SHARD_FAIL} | ETA: ~${ETA}min"
                fi
            fi

            TASK_IDX=$((TASK_IDX + 1))
        done
    done
done

# ========================================================================
# OUTPUT VERIFICATION
# ========================================================================
echo ""
echo "=========================================="
echo " SHARD ${SHARD_ID} OUTPUT"
echo "=========================================="

if [ -f "${ECLARE_SHARD}" ]; then
    python -c "
import h5py
with h5py.File('${ECLARE_SHARD}', 'r') as f:
    def count_datasets(grp):
        n = 0
        for k in grp:
            if isinstance(grp[k], h5py.Dataset): n += 1
            else: n += count_datasets(grp[k])
        return n
    total = count_datasets(f)
    print(f'Datasets in shard ${SHARD_ID}: {total}')
"
else
    echo "WARNING: Shard file not created (all tasks pre-existing?)"
fi

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo " ECLARE SHARD ${SHARD_ID} COMPLETED"
echo "=========================================="
echo "Done:     ${SHARD_DONE} | Failed: ${SHARD_FAIL} | Total: ${MY_TOTAL}"
echo "Finished: $(date)"
echo "Duration: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
