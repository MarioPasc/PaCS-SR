#!/usr/bin/env bash
#SBATCH -J pacs-sr-array
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=1-60
#SBATCH --constraint=cal
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err

# cd ~/fscratch/repos/PaCS-SR/; git pull; cd ~/execs/PACS_SR/; cp ~/fscratch/repos/PaCS-SR/scripts/picasso_job.sh .
# sbatch --export=ALL,CONFIG_YAML=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/PaCS-SR/configs/picasso.yaml,CONDA_ENV=pacs,RESULTS_HOME=$HOME/execs/pacs-sr/results picasso_job.sh


set -euo pipefail

# -------- user inputs ----------
: "${CONFIG_YAML:?set CONFIG_YAML=/abs/path/to/config.yaml before sbatch}"
CONDA_ENV="${CONDA_ENV:-pacs-sr}"
RESULTS_HOME="${RESULTS_HOME:-$HOME/pacs-sr/results}"   # shared, persistent
# -------------------------------

# Picasso env
module purge
module load anaconda
source activate "$CONDA_ENV"

# prevent BLAS over-subscription; joblib owns parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# working dirs in FSCRATCH (no localscratch)
WORKDIR="${FSCRATCH}/${USER}/pacs-sr/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORKDIR"
export JOBLIB_TEMP_FOLDER="${WORKDIR}/joblib_tmp"
mkdir -p "$JOBLIB_TEMP_FOLDER"

# runtime config (YAML edits done via Python for correctness)
export RUNTIME_CFG="${WORKDIR}/config.runtime.yaml"
cp "$CONFIG_YAML" "$RUNTIME_CFG"

python - <<'PY'
import sys, yaml, os, pathlib
cfg_path = os.environ["RUNTIME_CFG"]
with open(cfg_path) as f:
    y = yaml.safe_load(f)

# manifest path shared across array tasks
manifest_path = os.path.join(os.environ["RESULTS_HOME"], "kfolds_manifest.json")
# shared output root for all tasks (final, persistent)
out_root = os.environ["RESULTS_HOME"]

# update YAML
y.setdefault("data", {})["out"] = manifest_path
y.setdefault("pacs_sr", {})["out_root"] = out_root
y["pacs_sr"]["num_workers"] = int(os.environ["SLURM_CPUS_PER_TASK"])

# Enable SLURM-friendly logging (disable tqdm progress bars)
y["pacs_sr"]["disable_tqdm"] = True

with open(cfg_path, "w") as f:
    yaml.safe_dump(y, f, sort_keys=False)
print(cfg_path)
PY

# build manifest once with file lock
LOCK="${RESULTS_HOME}/.manifest.lock"
MANIFEST="${RESULTS_HOME}/kfolds_manifest.json"
mkdir -p "$RESULTS_HOME"
(
  flock -n 200 || true
  if [ ! -f "$MANIFEST" ]; then
    echo "Building manifest at $MANIFEST"
    srun pacs-sr-build-manifest --config "$RUNTIME_CFG"
  fi
) 200>"$LOCK"

# array mapping: 5 folds × 3 spacings × 4 pulses = 60 tasks
FOLDS=(1 2 3 4 5)
SPACINGS=("3mm" "5mm" "7mm")
PULSES=("t1c" "t1n" "t2w" "t2f")

IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
NF=${#FOLDS[@]}    # 5
NS=${#SPACINGS[@]} # 3
NP=${#PULSES[@]}   # 4

fi=$(( IDX / (NS*NP) ))
rem=$(( IDX % (NS*NP) ))
si=$(( rem / NP ))
pi=$(( rem % NP ))

FOLD="${FOLDS[$fi]}"
SPACING="${SPACINGS[$si]}"
PULSE="${PULSES[$pi]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: fold=${FOLD} spacing=${SPACING} pulse=${PULSE}"

# train one triple
time srun pacs-sr-train --config "$RUNTIME_CFG" --fold "$FOLD" --spacing "$SPACING" --pulse "$PULSE"

# efficiency report
seff "$SLURM_JOB_ID" || true
