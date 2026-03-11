#!/usr/bin/env python3
"""Generate 3D octant figures for one glioma patient across pulse/spacing combos.

Produces PNG + PDF figures for every combination of:
    pulses:   {t1c, t2w, t2f}
    spacings: {3mm, 5mm, 7mm}

Each low-resolution volume is upscaled to 1mm³ isotropic with nearest-neighbour
interpolation before rendering. The segmentation is loaded from the original
BraTS-GLI NIfTI directory and used at its native 1mm³ resolution to overlay
the tumor with alpha=0.3 for all labels.

All figures share identical camera orientation (computed once from the HR t1c).
Figures are saved without titles, axes, or scalar bars.

Output directory:
    /media/mpascual/Sandisk2TB/research/pacs_sr/article/graphical_abstract/octant_spacing/

Usage:
    python pacs_sr/seram/graphical_abstract/generate_octant_spacing_grid.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as ndizoom

# Allow importing the local visualize_glioma_octant module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_glioma_octant import (
    RenderConfig,
    compute_brain_mask,
    find_optimal_octant,
    render_glioma_octant,
    replace_octant,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

PATIENT_ID = "BraTS-GLI-02355-100"

SOURCE_H5 = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/data/gliomas/source_data.h5"
)
SEG_NIFTI = Path(
    "/media/mpascual/PortableSSD/BraTS_GLI/source/"
    "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2/"
    f"{PATIENT_ID}/{PATIENT_ID}-seg.nii.gz"
)

OUTPUT_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/article/graphical_abstract/"
    "octant_spacing"
)

PULSES = ["t1c", "t2w", "t2f"]
SPACINGS = ["3mm", "5mm", "7mm"]

TUMOR_ALPHA = 0.3
ZOOM = 2.5
WINDOW_SIZE = (1400, 1200)
MARGIN = 0.05

# Manual slice overrides — set to an int to override the auto-computed value,
# or leave as None to use the automatic tumor-centred slice.
SLICE_AXIAL: int | None = 90      # k index (dim 0)
SLICE_CORONAL: int | None = None    # i index (dim 1)
SLICE_SAGITTAL: int | None = None   # j index (dim 2)


# =============================================================================
# Helpers
# =============================================================================


def load_seg_nifti(seg_path: Path) -> np.ndarray:
    """Load segmentation NIfTI and convert to RAS."""
    img = nib.load(str(seg_path))
    img = nib.as_closest_canonical(img)
    return np.asanyarray(img.get_fdata()).astype(np.int32)


def load_hr_volume(h5_path: Path, patient_id: str, pulse: str) -> np.ndarray:
    """Load high-resolution volume from HDF5."""
    with h5py.File(h5_path, "r") as f:
        return np.array(f[f"high_resolution/{patient_id}/{pulse}"])


def load_lr_volume(
    h5_path: Path, spacing: str, patient_id: str, pulse: str
) -> np.ndarray:
    """Load low-resolution volume from HDF5."""
    with h5py.File(h5_path, "r") as f:
        return np.array(f[f"low_resolution/{spacing}/{patient_id}/{pulse}"])


def upsample_nearest(lr_vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Upsample volume to target_shape using nearest-neighbour interpolation.

    Args:
        lr_vol: Low-resolution 3D volume.
        target_shape: Desired (nx, ny, nz) after upsampling.

    Returns:
        Upsampled volume at target resolution.
    """
    factors = tuple(t / s for t, s in zip(target_shape, lr_vol.shape))
    return ndizoom(lr_vol, factors, order=0)


def save_figure(plotter, out_path: Path) -> None:
    """Save the plotter as PNG and PDF (same stem)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_path.with_suffix(".png")
    pdf_path = out_path.with_suffix(".pdf")

    plotter.screenshot(str(png_path))
    logger.info(f"  Saved: {png_path.name}")

    plotter.save_graphic(str(pdf_path))
    logger.info(f"  Saved: {pdf_path.name}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Generate all octant figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load segmentation (1mm³ native)
    logger.info(f"Loading segmentation: {SEG_NIFTI}")
    seg = load_seg_nifti(SEG_NIFTI)
    logger.info(f"  Seg shape: {seg.shape}, labels: {np.unique(seg).tolist()}")

    # Load HR t1c as reference for brain mask and camera
    logger.info("Loading HR t1c for reference brain mask and camera...")
    hr_ref = load_hr_volume(SOURCE_H5, PATIENT_ID, "t1c")
    target_shape = hr_ref.shape
    logger.info(f"  HR shape: {target_shape}")

    brain_mask = compute_brain_mask(hr_ref)
    slice_indices, octant = find_optimal_octant(
        hr_ref, seg, margin_frac=MARGIN
    )
    # Apply manual overrides per plane
    k, i, j = slice_indices
    if SLICE_AXIAL is not None:
        k = SLICE_AXIAL
    if SLICE_CORONAL is not None:
        i = SLICE_CORONAL
    if SLICE_SAGITTAL is not None:
        j = SLICE_SAGITTAL
    slice_indices = (k, i, j)
    logger.info(f"  Octant: {octant}, slices: {slice_indices}")

    # Base config — shared across all figures
    base_cfg = RenderConfig(
        octant=octant,
        zoom=ZOOM,
        window_size=WINDOW_SIZE,
    )

    total = len(PULSES) * len(SPACINGS)
    done = 0

    for pulse in PULSES:
        for spacing in SPACINGS:
            done += 1
            tag = f"{pulse}_{spacing}"
            logger.info(f"[{done}/{total}] Rendering {tag} ...")

            lr_vol = load_lr_volume(SOURCE_H5, spacing, PATIENT_ID, pulse)
            vol_up = upsample_nearest(lr_vol, target_shape)
            logger.info(
                f"  LR {lr_vol.shape} -> upsampled {vol_up.shape}"
            )

            out_stem = OUTPUT_DIR / f"{PATIENT_ID}_{tag}"
            plotter = render_glioma_octant(
                vol_up,
                seg,
                slice_indices,
                cfg=base_cfg,
                brain_mask=brain_mask,
                tumor_alpha_override=TUMOR_ALPHA,
                off_screen=True,
            )
            save_figure(plotter, out_stem)
            plotter.close()

    logger.info(f"\nDone — {total} figure pairs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
