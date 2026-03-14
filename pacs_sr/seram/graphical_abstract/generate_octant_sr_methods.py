#!/usr/bin/env python3
"""Generate 3D octant figures for SR methods (BSPLINE, ECLARE, PaCS-SR).

Same camera/octant as the anisotropy figures, but showing the super-resolved
outputs instead of the nearest-neighbour upsampled LR inputs.  Also includes
HR ground truth for reference.

Produces PNG + PDF for each (method, spacing) combination:
    methods:  HR, BSPLINE, ECLARE, PaCS_SR
    spacings: 3mm, 5mm, 7mm

Output directory:
    .../graphical_abstract/octant_sr_methods/

Usage:
    python pacs_sr/seram/graphical_abstract/generate_octant_sr_methods.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_glioma_octant import (
    RenderConfig,
    compute_brain_mask,
    find_optimal_octant,
    render_glioma_octant,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

PATIENT_ID = "BraTS-GLI-02355-100"
PULSE = "t1c"

SOURCE_H5 = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/data/gliomas/source_data.h5"
)
EXPERTS_DIR = Path("/media/mpascual/Sandisk2TB/research/pacs_sr/data/gliomas/experts")
RESULTS_H5 = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/res_folder/results_fold_1.h5"
)
SEG_NIFTI = Path(
    "/media/mpascual/PortableSSD/BraTS_GLI/source/"
    "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2/"
    f"{PATIENT_ID}/{PATIENT_ID}-seg.nii.gz"
)

OUTPUT_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/article/graphical_abstract/"
    "octant_sr_methods"
)

SPACINGS = ["3mm", "5mm", "7mm"]

# Rendering — same zoom as the anisotropy figures
ZOOM = 3.8
WINDOW_SIZE = (1400, 1200)
MARGIN = 0.05

SLICE_AXIAL: int | None = 90
SLICE_CORONAL: int | None = None
SLICE_SAGITTAL: int | None = None


# =============================================================================
# Volume loaders
# =============================================================================


def load_hr(h5_path: Path, patient_id: str, pulse: str) -> np.ndarray:
    """Load high-resolution volume from source HDF5."""
    with h5py.File(h5_path, "r") as f:
        return np.array(f[f"high_resolution/{patient_id}/{pulse}"])


def load_expert(
    experts_dir: Path,
    method: str,
    spacing: str,
    patient_id: str,
    pulse: str,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Load an expert SR volume, padding to target_shape if needed."""
    with h5py.File(experts_dir / f"{method}.h5", "r") as f:
        vol = np.array(f[f"{spacing}/{patient_id}/{pulse}"])

    # ECLARE can produce volumes 1-2 voxels shorter in z — zero-pad to match
    if vol.shape != target_shape:
        logger.warning(
            f"  {method} {spacing} shape {vol.shape} != target {target_shape}, padding"
        )
        padded = np.zeros(target_shape, dtype=vol.dtype)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(vol.shape, target_shape))
        padded[slices] = vol[slices]
        vol = padded

    return vol


def load_pacs_sr(
    results_h5: Path,
    spacing: str,
    patient_id: str,
    pulse: str,
) -> np.ndarray:
    """Load PaCS-SR blend volume from results HDF5."""
    with h5py.File(results_h5, "r") as f:
        return np.array(f[f"{spacing}/blends/{patient_id}/{pulse}"])


def save_figure(plotter, out_path: Path) -> None:
    """Save as PNG and PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_path.with_suffix(".png")))
    logger.info(f"  Saved: {out_path.with_suffix('.png').name}")
    plotter.save_graphic(str(out_path.with_suffix(".pdf")))
    logger.info(f"  Saved: {out_path.with_suffix('.pdf').name}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Generate octant figures for HR + 3 SR methods x 3 spacings."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Reference geometry (identical to anisotropy figures) ---
    logger.info("Loading HR t1c for reference geometry...")
    hr_vol = load_hr(SOURCE_H5, PATIENT_ID, PULSE)
    target_shape = hr_vol.shape
    logger.info(f"  HR shape: {target_shape}")

    brain_mask = compute_brain_mask(hr_vol)

    # Compute octant from real segmentation (geometry only — not rendered)
    real_seg = (
        nib.as_closest_canonical(nib.load(str(SEG_NIFTI))).get_fdata().astype(np.int32)
    )
    slice_indices, octant = find_optimal_octant(hr_vol, real_seg, margin_frac=MARGIN)

    k, i, j = slice_indices
    if SLICE_AXIAL is not None:
        k = SLICE_AXIAL
    if SLICE_CORONAL is not None:
        i = SLICE_CORONAL
    if SLICE_SAGITTAL is not None:
        j = SLICE_SAGITTAL
    slice_indices = (k, i, j)
    logger.info(f"  Octant: {octant}, slices: {slice_indices}")

    empty_seg = np.zeros(target_shape, dtype=np.int32)
    base_cfg = RenderConfig(octant=octant, zoom=ZOOM, window_size=WINDOW_SIZE)

    # --- Build render list: (label, volume) ---
    jobs: list[tuple[str, np.ndarray]] = []

    # HR ground truth (spacing-independent)
    jobs.append(("HR", hr_vol))

    # Methods x spacings
    for spacing in SPACINGS:
        logger.info(f"Loading volumes for {spacing}...")

        bspline = load_expert(
            EXPERTS_DIR, "bspline", spacing, PATIENT_ID, PULSE, target_shape
        )
        jobs.append((f"BSPLINE_{spacing}", bspline))

        eclare = load_expert(
            EXPERTS_DIR, "eclare", spacing, PATIENT_ID, PULSE, target_shape
        )
        jobs.append((f"ECLARE_{spacing}", eclare))

        pacs = load_pacs_sr(RESULTS_H5, spacing, PATIENT_ID, PULSE)
        jobs.append((f"PaCS_SR_{spacing}", pacs))

    total = len(jobs)
    for idx, (label, vol) in enumerate(jobs, 1):
        logger.info(f"[{idx}/{total}] Rendering {label} ...")
        plotter = render_glioma_octant(
            vol,
            empty_seg,
            slice_indices,
            cfg=base_cfg,
            brain_mask=brain_mask,
            off_screen=True,
        )
        save_figure(plotter, OUTPUT_DIR / f"{PATIENT_ID}_{PULSE}_{label}")
        plotter.close()

    logger.info(f"\nDone — {total} figure pairs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
