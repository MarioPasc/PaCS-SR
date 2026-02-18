#!/bin/bash
# =============================================================================
# SERAM Experiment Pipeline — End-to-End Orchestration
#
# Two-stage workflow:
#   Stage A (Picasso HPC): Pre-generate expert outputs via SLURM
#     bash scripts/generate_experts.sh
#
#   Stage B (Local): Train PaCS-SR, compute metrics, generate figures
#     bash scripts/run_seram.sh
#
# Prerequisites:
#   - source_data.h5 created via scripts/convert_nifti_to_hdf5.py
#   - Expert HDF5 files (bspline.h5, eclare.h5) in experts-dir
#   - Config paths in seram_glioma.yaml point to local HDF5 files
#
# Usage:
#   bash scripts/run_seram.sh
# =============================================================================
set -euo pipefail

CONDA=~/.conda/envs/pacs/bin
CONFIG=configs/seram_glioma.yaml

echo "=========================================="
echo " SERAM Experiment Pipeline (Local)"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Time:   $(date)"

# ── Pre-flight: verify HDF5 files exist ──────────────────────────────────
echo ""
echo "[Pre-flight] Checking HDF5 files..."

SOURCE_H5=$(python -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${CONFIG}').read_text())
print(cfg['data']['source-h5'])
")
EXPERTS_DIR=$(python -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${CONFIG}').read_text())
print(cfg['data']['experts-dir'])
")
MODELS=$(python -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${CONFIG}').read_text())
print(' '.join(cfg['data']['models']))
")

MISSING=0

# Check source HDF5
if [ -f "${SOURCE_H5}" ]; then
    echo "  [OK] source_data.h5: ${SOURCE_H5}"
    python -c "
import h5py
with h5py.File('${SOURCE_H5}', 'r') as f:
    n_hr = len(list(f['high_resolution']))
    print(f'       HR patients: {n_hr}')
"
else
    echo "  [MISS] source_data.h5: ${SOURCE_H5}"
    MISSING=1
fi

# Check expert HDF5 files
for model in ${MODELS}; do
    model_lower=$(echo "${model}" | tr '[:upper:]' '[:lower:]')
    h5_file="${EXPERTS_DIR}/${model_lower}.h5"
    if [ -f "${h5_file}" ]; then
        python -c "
import h5py
with h5py.File('${h5_file}', 'r') as f:
    def count_ds(g):
        n = 0
        for k in g:
            if isinstance(g[k], h5py.Dataset): n += 1
            else: n += count_ds(g[k])
        return n
    print(f'  [OK] ${model}: {count_ds(f)} datasets in ${h5_file}')
"
    else
        echo "  [MISS] ${model}: ${h5_file}"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "ERROR: Required HDF5 files missing. Generate them first:"
    echo "  1. python scripts/convert_nifti_to_hdf5.py --hr-root ... --lr-root ... --output ${SOURCE_H5}"
    echo "  2. python scripts/generate_bspline.py --config ${CONFIG}"
    echo "  3. (Picasso) bash scripts/generate_experts.sh"
    exit 1
fi

# ── Step 1: Build K-fold manifest ────────────────────────────────────────────
echo ""
echo "[Step 1/4] Building K-fold manifest..."
$CONDA/python -m pacs_sr.cli.build_manifest --config $CONFIG

# ── Step 2: Train PaCS-SR (all folds × spacings × pulses) ───────────────────
echo ""
echo "[Step 2/4] Training PaCS-SR..."
$CONDA/python -m pacs_sr.cli.train --config $CONFIG

# ── Step 3: Compute SERAM metrics (3D-MS-SSIM + MMD-MF) ─────────────────────
echo ""
echo "[Step 3/4] Computing SERAM metrics..."
$CONDA/python scripts/seram_compute_metrics.py --config $CONFIG

# ── Step 4: Generate figures and table ───────────────────────────────────────
echo ""
echo "[Step 4/4] Generating figures and table..."
$CONDA/python -m pacs_sr.seram.figure_comparison_grid --config $CONFIG
$CONDA/python -m pacs_sr.seram.figure_performance_plot --config $CONFIG
$CONDA/python -m pacs_sr.seram.table_msssim --config $CONFIG

echo ""
echo "=========================================="
echo " SERAM Pipeline Complete"
echo "=========================================="
echo " Results: $(grep 'out_root' $CONFIG | head -1 | awk '{print $2}')"
echo "=========================================="
