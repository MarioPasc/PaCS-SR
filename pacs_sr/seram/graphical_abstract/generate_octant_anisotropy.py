#!/usr/bin/env python3
"""Generate 3D octant figures focused on anisotropy — no tumor, zoomed into slices.

Produces PNG + PDF for HR and each spacing (3mm, 5mm, 7mm) using a single
pulse (t1c). The camera is positioned closer to the octant opening so the
staircase artefact from anisotropic acquisitions is clearly visible on the
orthogonal MRI slice planes.

Output directory:
    /media/mpascual/Sandisk2TB/research/pacs_sr/article/graphical_abstract/octant_anisotropy/

Usage:
    python pacs_sr/seram/graphical_abstract/generate_octant_anisotropy.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom as ndizoom

# Allow importing the local visualize_glioma_octant module
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
PULSES = ["t1c", "t2w", "t2f"]

SOURCE_H5 = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/data/gliomas/source_data.h5"
)

OUTPUT_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/article/graphical_abstract/"
    "octant_anisotropy"
)

SPACINGS = ["3mm", "5mm", "7mm"]

# Rendering — tighter zoom to focus on the slice planes inside the cutaway
ZOOM = 3.8
WINDOW_SIZE = (1400, 1200)
MARGIN = 0.05

# Manual slice overrides (same as the spacing grid for consistency)
SLICE_AXIAL: int | None = 90
SLICE_CORONAL: int | None = None
SLICE_SAGITTAL: int | None = None


# =============================================================================
# Helpers
# =============================================================================


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
    """Generate octant figures without tumor, zoomed into the anisotropy."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use t1c as geometry reference (brain mask, octant, slices)
    logger.info("Loading HR t1c for reference geometry...")
    hr_ref = load_hr_volume(SOURCE_H5, PATIENT_ID, "t1c")
    target_shape = hr_ref.shape
    logger.info(f"  HR shape: {target_shape}")

    brain_mask = compute_brain_mask(hr_ref)
    empty_seg = np.zeros(target_shape, dtype=np.int32)

    # Load the real segmentation for octant orientation only
    seg_nifti = Path(
        "/media/mpascual/PortableSSD/BraTS_GLI/source/"
        "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2/"
        f"{PATIENT_ID}/{PATIENT_ID}-seg.nii.gz"
    )
    import nibabel as nib

    real_seg = (
        nib.as_closest_canonical(nib.load(str(seg_nifti))).get_fdata().astype(np.int32)
    )

    slice_indices, octant = find_optimal_octant(hr_ref, real_seg, margin_frac=MARGIN)
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

    total = len(PULSES) * (1 + len(SPACINGS))  # HR + spacings per pulse
    done = 0

    for pulse in PULSES:
        logger.info(f"=== Pulse: {pulse} ===")

        # --- HR figure ---
        done += 1
        logger.info(f"[{done}/{total}] Rendering {pulse} HR (1mm isotropic) ...")
        hr_vol = load_hr_volume(SOURCE_H5, PATIENT_ID, pulse)
        plotter = render_glioma_octant(
            hr_vol,
            empty_seg,
            slice_indices,
            cfg=base_cfg,
            brain_mask=brain_mask,
            off_screen=True,
        )
        save_figure(plotter, OUTPUT_DIR / f"{PATIENT_ID}_{pulse}_HR")
        plotter.close()

        # --- LR figures at each spacing ---
        for spacing in SPACINGS:
            done += 1
            logger.info(f"[{done}/{total}] Rendering {pulse} {spacing} ...")

            lr_vol = load_lr_volume(SOURCE_H5, spacing, PATIENT_ID, pulse)
            vol_up = upsample_nearest(lr_vol, target_shape)
            logger.info(f"  LR {lr_vol.shape} -> upsampled {vol_up.shape}")

            plotter = render_glioma_octant(
                vol_up,
                empty_seg,
                slice_indices,
                cfg=base_cfg,
                brain_mask=brain_mask,
                off_screen=True,
            )
            save_figure(plotter, OUTPUT_DIR / f"{PATIENT_ID}_{pulse}_{spacing}")
            plotter.close()

    logger.info(f"\nDone — {total} figure pairs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
