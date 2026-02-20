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

# Build task list in Python: get patients, existing keys, and shard assignment
# in a single process launch instead of one per task.
TASK_FILE=$(mktemp /tmp/eclare_tasks_XXXXXX.txt)
python -c "
from pacs_sr.data.hdf5_io import list_groups, has_key, expert_key
import h5py

source_h5 = '${SOURCE_H5}'
eclare_main = '${ECLARE_MAIN}'
eclare_shard = '${ECLARE_SHARD}'
spacings = '${SPACINGS}'.split()
pulses = '${PULSES}'.split()
num_shards = ${NUM_SHARDS}
shard_id = ${SHARD_ID}

patients = list_groups(source_h5, 'high_resolution')

# Build set of existing keys in main eclare.h5 + this shard (single open each)
existing = set()
for h5_path in [eclare_main, eclare_shard]:
    try:
        with h5py.File(h5_path, 'r') as f:
            def collect(grp, prefix=''):
                for k in grp:
                    path = f'{prefix}/{k}' if prefix else k
                    if isinstance(grp[k], h5py.Dataset):
                        existing.add(path)
                    else:
                        collect(grp[k], path)
            collect(f)
    except (OSError, FileNotFoundError):
        pass

task_idx = 0
my_total = 0
my_skip = 0
for spacing in spacings:
    thickness = spacing.replace('mm', '')
    for patient_id in patients:
        for pulse in pulses:
            if task_idx % num_shards == shard_id:
                key = expert_key(spacing, patient_id, pulse)
                if key in existing:
                    my_skip += 1
                else:
                    print(f'{spacing} {patient_id} {pulse} {thickness}')
                my_total += 1
            task_idx += 1

import sys
print(f'Shard {shard_id}: {my_total} assigned, {my_skip} already done, {my_total - my_skip} to process', file=sys.stderr)
" > "${TASK_FILE}"

MY_TOTAL=$(wc -l < "${TASK_FILE}")
echo "[shard ${SHARD_ID}] Tasks to process: ${MY_TOTAL}"
echo ""

# ========================================================================
# PROCESS THIS SHARD'S TASKS
# ========================================================================
MY_IDX=0
SHARD_DONE=0
SHARD_FAIL=0

while IFS=' ' read -r spacing patient_id pulse thickness; do
    MY_IDX=$((MY_IDX + 1))

    echo "[${MY_IDX}/${MY_TOTAL}] Processing: ${patient_id} | ${spacing} | ${pulse}"

    if python scripts/eclare_h5_adapter.py \
        --source-h5 "${SOURCE_H5}" \
        --expert-h5 "${ECLARE_SHARD}" \
        --patient-id "${patient_id}" \
        --spacing "${spacing}" \
        --pulse "${pulse}" \
        --thickness "${thickness}" \
        --gpu-id 0 2>&1; then
        SHARD_DONE=$((SHARD_DONE + 1))
    else
        SHARD_FAIL=$((SHARD_FAIL + 1))
    fi

    # Progress report every 25 tasks
    if [ $((MY_IDX % 25)) -eq 0 ]; then
        ELAPSED=$(($(date +%s) - START_TIME))
        AVG=$((ELAPSED / MY_IDX))
        ETA=$(( AVG * (MY_TOTAL - MY_IDX) / 60 ))
        echo "  Progress: ${MY_IDX}/${MY_TOTAL} | Done: ${SHARD_DONE} Fail: ${SHARD_FAIL} | ETA: ~${ETA}min"
    fi
done < "${TASK_FILE}"

rm -f "${TASK_FILE}"

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
