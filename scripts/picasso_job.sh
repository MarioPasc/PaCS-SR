#!/usr/bin/env bash
#SBATCH -J pacs-sr-train
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --constraint=cal
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

# --- user inputs --------------------------------------------------------------
: "${CONFIG_YAML:?set CONFIG_YAML=/path/to/config.yaml before sbatch}"
CONDA_ENV="${CONDA_ENV:-pacs-sr}"               # name of your conda env (change if needed)
RESULTS_HOME="${RESULTS_HOME:-$HOME/pacs-sr}"   # where to persist final results
FOLD_OPT="${FOLD:+--fold $FOLD}"                # optional: export FOLD=1
SPACING_OPT="${SPACING:+--spacing $SPACING}"    # optional: export SPACING=3mm
PULSE_OPT="${PULSE:+--pulse $PULSE}"            # optional: export PULSE=t1c
# ------------------------------------------------------------------------------

# Load software via modules, as required on Picasso
module purge
module load anaconda                             # load the site-provided Anaconda
# (list modules with `module avail`; keep `module load` inside the SBATCH script)  # :contentReference[oaicite:0]{index=0}
source activate "$CONDA_ENV"

# Avoid BLAS oversubscription; let joblib control threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Use FSCRATCH for working directory and joblib temps (not LOCALSCRATCH)
WORKDIR="${FSCRATCH}/${USER}/pacs-sr/${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
export JOBLIB_TEMP_FOLDER="${WORKDIR}/joblib_tmp"
mkdir -p "$JOBLIB_TEMP_FOLDER"

# Prepare a runtime config that:
#   - writes the manifest to FSCRATCH
#   - writes results under FSCRATCH
#   - sets pacs_sr.num_workers = $SLURM_CPUS_PER_TASK
RUNTIME_CFG="${WORKDIR}/config.runtime.yaml"
cp "$CONFIG_YAML" "$RUNTIME_CFG"

MANIFEST_PATH="${WORKDIR}/kfolds_manifest.json"
RESULTS_DIR="${WORKDIR}/results"
# crude but effective YAML line edits (one occurrence per key expected)
sed -i -E "s|(^[[:space:]]*out:[[:space:]]*).*$|\1 ${MANIFEST_PATH}|" "$RUNTIME_CFG"
sed -i -E "s|(^[[:space:]]*out_root:[[:space:]]*).*$|\1 ${RESULTS_DIR}|" "$RUNTIME_CFG"
sed -i -E "s|(^[[:space:]]*num_workers:[[:space:]]*).*$|\1 ${SLURM_CPUS_PER_TASK}|" "$RUNTIME_CFG"

# Build manifest and train; run under srun to bind resources from Slurm
time srun pacs-sr-build-manifest --config "$RUNTIME_CFG"
time srun pacs-sr-train --config "$RUNTIME_CFG" ${FOLD_OPT:-} ${SPACING_OPT:-} ${PULSE_OPT:-}

# Persist outputs to HOME (FSCRATCH is purged periodically)
mkdir -p "$RESULTS_HOME/${SLURM_JOB_ID}"
rsync -a "${RESULTS_DIR}/" "$RESULTS_HOME/${SLURM_JOB_ID}/"
cp -f "$RUNTIME_CFG" "$RESULTS_HOME/${SLURM_JOB_ID}/"
cp -f "$MANIFEST_PATH" "$RESULTS_HOME/${SLURM_JOB_ID}/"

# Post-run efficiency report
seff "$SLURM_JOB_ID" || true
