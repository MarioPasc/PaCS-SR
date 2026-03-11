#!/usr/bin/env python3
"""Generate 3D octant expert comparison figures for one glioma patient.

Produces PNG + PDF for each of: HR, BSPLINE (5mm), ECLARE (5mm), PaCS-SR (5mm).
All figures share the same camera orientation and octant (computed from the HR).
Tumor overlay uses alpha=0.3 for all labels, no titles/axes/scalar bars.

Output directory:
    /media/mpascual/Sandisk2TB/research/pacs_sr/article/graphical_abstract/expert_comparison/

Usage:
    python pacs_sr/seram/graphical_abstract/generate_expert_comparison.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as ndizoom

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
SPACING = "5mm"
PULSE = "t1c"
FOLD = 1  # fold where this patient is in the test set

SOURCE_H5 = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/data/gliomas/source_data.h5"
)
BSPLINE_H5 = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/data/gliomas/experts/bspline.h5"
)
ECLARE_H5 = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/data/gliomas/experts/eclare.h5"
)
RESULTS_H5 = Path(
    f"/media/mpascual/Sandisk2TB/research/pacs_sr/SERAM_GLI/results_fold_{FOLD}.h5"
)
SEG_NIFTI = Path(
    "/media/mpascual/PortableSSD/BraTS_GLI/source/"
    "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2/"
    f"{PATIENT_ID}/{PATIENT_ID}-seg.nii.gz"
)

OUTPUT_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/article/graphical_abstract/"
    "expert_comparison"
)

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


def pad_to_shape(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center-pad volume to target shape (needed when expert output is smaller).

    Args:
        vol: Input 3D volume.
        target_shape: Desired (nx, ny, nz).

    Returns:
        Padded volume.
    """
    if vol.shape == target_shape:
        return vol

    pad_widths = []
    for s, t in zip(vol.shape, target_shape):
        diff = t - s
        if diff < 0:
            # crop instead
            start = (-diff) // 2
            vol = vol[
                tuple(
                    slice(start, start + t) if i == len(pad_widths) else slice(None)
                    for i in range(3)
                )
            ]
            pad_widths.append((0, 0))
        else:
            before = diff // 2
            after = diff - before
            pad_widths.append((before, after))

    return np.pad(vol, pad_widths, mode="constant", constant_values=0.0)


def resize_to_shape(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize volume to target shape using cubic interpolation (consistent with PaCS-SR).

    Args:
        vol: Input 3D volume.
        target_shape: Desired (nx, ny, nz).

    Returns:
        Resized volume.
    """
    if vol.shape == target_shape:
        return vol
    factors = tuple(t / s for t, s in zip(target_shape, vol.shape))
    return ndizoom(vol, factors, order=3)


def save_figure(plotter, out_path: Path) -> None:
    """Save the plotter as PNG and PDF."""
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
    """Generate expert comparison figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load segmentation
    logger.info(f"Loading segmentation: {SEG_NIFTI}")
    seg = load_seg_nifti(SEG_NIFTI)

    # Load HR as reference
    logger.info("Loading HR volume...")
    with h5py.File(SOURCE_H5, "r") as f:
        hr_vol = np.array(f[f"high_resolution/{PATIENT_ID}/{PULSE}"])
    target_shape = hr_vol.shape
    logger.info(f"  HR shape: {target_shape}")

    # Compute shared brain mask, octant, and camera from HR
    brain_mask = compute_brain_mask(hr_vol)
    slice_indices, octant = find_optimal_octant(hr_vol, seg, margin_frac=MARGIN)
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

    base_cfg = RenderConfig(
        octant=octant,
        zoom=ZOOM,
        window_size=WINDOW_SIZE,
    )

    # Load expert volumes
    logger.info(f"Loading BSPLINE {SPACING}...")
    with h5py.File(BSPLINE_H5, "r") as f:
        bspline_vol = np.array(f[f"{SPACING}/{PATIENT_ID}/{PULSE}"])

    logger.info(f"Loading ECLARE {SPACING}...")
    with h5py.File(ECLARE_H5, "r") as f:
        eclare_vol = np.array(f[f"{SPACING}/{PATIENT_ID}/{PULSE}"])

    # Resize ECLARE if needed (ECLARE often has slightly different z-dim)
    if eclare_vol.shape != target_shape:
        logger.info(
            f"  Resizing ECLARE {eclare_vol.shape} -> {target_shape} (cubic)"
        )
        eclare_vol = resize_to_shape(eclare_vol, target_shape)

    logger.info(f"Loading PaCS-SR blend (fold {FOLD}, {SPACING})...")
    with h5py.File(RESULTS_H5, "r") as f:
        pacs_vol = np.array(f[f"{SPACING}/blends/{PATIENT_ID}/{PULSE}"])

    # Define methods to render
    methods = {
        "HR": hr_vol,
        "BSPLINE": bspline_vol,
        "ECLARE": eclare_vol,
        "PACS_SR": pacs_vol,
    }

    for name, vol in methods.items():
        tag = f"{PATIENT_ID}_{SPACING}_{PULSE}_{name}"
        logger.info(f"Rendering {name} (shape={vol.shape})...")

        out_stem = OUTPUT_DIR / tag
        plotter = render_glioma_octant(
            vol,
            seg,
            slice_indices,
            cfg=base_cfg,
            brain_mask=brain_mask,
            tumor_alpha_override=TUMOR_ALPHA,
            off_screen=True,
        )
        save_figure(plotter, out_stem)
        plotter.close()

    logger.info(f"\nDone — {len(methods)} figure pairs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
