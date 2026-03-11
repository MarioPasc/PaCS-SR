#!/usr/bin/env python3
"""Generate PaCS-SR weight visualizations for one glioma patient.

Produces two visualizations:

1. **Brain-projected weight octant**: The 4D weight maps from a .npz file are
   masked to the brain volume and rendered as coloured octant slices (same
   camera / octant logic as the MRI figures, but without tumor overlay).
   The slice textures use a diverging colormap centered at 0.5 — one end
   represents full reliance on Expert 1 (BSPLINE), the other on Expert 2
   (ECLARE).

2. **Patch cube comparison**: A single patch-sized cube is extracted from both
   the MRI volume and the weight map.  The MRI cube is rendered in grayscale
   and the weight cube with the same diverging colormap + a colorbar.

Output directory:
    /media/mpascual/Sandisk2TB/research/pacs_sr/article/graphical_abstract/weight_viz/

Usage:
    python pacs_sr/seram/graphical_abstract/generate_weight_visualization.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

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
FOLD = 1

PATCH_SIZE = 32
STRIDE = 16

SOURCE_H5 = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/data/gliomas/source_data.h5"
)
WEIGHTS_NPZ = Path(
    f"/media/mpascual/Sandisk2TB/research/pacs_sr/SERAM_GLI/{SPACING}/"
    f"model_data/fold_{FOLD}/{PULSE}/{PATIENT_ID}_weights_test.npz"
)

OUTPUT_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/pacs_sr/article/graphical_abstract/"
    "weight_viz"
)

# Octant / camera settings (same as expert comparison for consistency)
ZOOM = 2.5
WINDOW_SIZE = (1400, 1200)
MARGIN = 0.05

# Manual slice overrides (None = auto from tumor centroid)
SLICE_AXIAL: int | None = None
SLICE_CORONAL: int | None = None
SLICE_SAGITTAL: int | None = None

# Patch cube location — set to (x0, y0, z0) to override, None = auto-centre on
# the voxel with the most extreme weight (furthest from 0.5).
PATCH_ORIGIN: Tuple[int, int, int] | None = None

# Diverging colormap: Expert 1 colour ← 0.5 → Expert 2 colour
WEIGHT_CMAP = "RdBu_r"  # Red = Expert 1 (BSPLINE), Blue = Expert 2 (ECLARE)


# =============================================================================
# Data loading helpers
# =============================================================================


def load_hr_volume(h5_path: Path, patient_id: str, pulse: str) -> np.ndarray:
    """Load high-resolution volume from HDF5."""
    with h5py.File(h5_path, "r") as f:
        return np.array(f[f"high_resolution/{patient_id}/{pulse}"])


def load_weight_maps(npz_path: Path) -> Tuple[np.ndarray, list[str]]:
    """Load weight maps and model names from NPZ.

    Returns:
        weight_maps: 4D array (Z, Y, X, n_models).
        model_names: List of expert model name strings.
    """
    data = np.load(npz_path, allow_pickle=True)
    wm = data.get("weight_maps")
    if wm is None:
        wm = data.get("weights")
    if wm is None:
        raise ValueError(f"No weight_maps found in {npz_path}")

    names = data.get("model_names", None)
    if names is not None and hasattr(names, "tolist"):
        names = names.tolist()
    if names is None:
        names = [f"Expert{i}" for i in range(wm.shape[-1])]
    return wm, names


def compute_weight_difference(weight_maps: np.ndarray) -> np.ndarray:
    """Compute per-voxel weight difference for a 2-expert system.

    Returns a scalar volume in [0, 1] where 0 = full Expert 1, 1 = full
    Expert 2, 0.5 = equal weighting.

    For >2 experts, returns the weight of the second expert (index 1).

    Args:
        weight_maps: 4D array (Z, Y, X, n_models).

    Returns:
        3D array (Z, Y, X) of weight values for Expert 2.
    """
    return weight_maps[..., 1].copy()


# =============================================================================
# Visualisation 1 — Brain-projected weight octant
# =============================================================================


def save_pyvista_figure(plotter, out_path: Path) -> None:
    """Save plotter as PNG and PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_path.with_suffix(".png")))
    logger.info(f"  Saved: {out_path.with_suffix('.png').name}")
    plotter.save_graphic(str(out_path.with_suffix(".pdf")))
    logger.info(f"  Saved: {out_path.with_suffix('.pdf').name}")


def generate_weight_octant(
    hr_vol: np.ndarray,
    weight_maps: np.ndarray,
    model_names: list[str],
    brain_mask: np.ndarray,
    output_dir: Path,
) -> None:
    """Render the octant figure with weight-coloured slices instead of MRI.

    We pass the weight-difference volume (Expert 2 weight) as if it were an
    MRI intensity volume to ``render_glioma_octant``, using a diverging cmap.
    The segmentation is set to all-zeros so no tumor overlay is drawn.
    """
    w_diff = compute_weight_difference(weight_maps)

    # Mask to brain
    w_diff[~brain_mask] = np.nan

    # Use a dummy all-zero segmentation (no tumor)
    dummy_seg = np.zeros(hr_vol.shape, dtype=np.int32)

    # Auto-compute octant from HR (we need a seg with nonzero for
    # find_optimal_octant — fall back to volume centre)
    nx, ny, nz = hr_vol.shape
    slice_indices = (nz // 2, nx // 2, ny // 2)
    octant = (False, False, False)

    # Try to get better placement from the actual expert comparison settings
    import nibabel as nib

    seg_path = Path(
        "/media/mpascual/PortableSSD/BraTS_GLI/source/"
        "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2/"
        f"{PATIENT_ID}/{PATIENT_ID}-seg.nii.gz"
    )
    if seg_path.exists():
        seg_img = nib.load(str(seg_path))
        seg_img = nib.as_closest_canonical(seg_img)
        real_seg = np.asanyarray(seg_img.get_fdata()).astype(np.int32)
        slice_indices, octant = find_optimal_octant(
            hr_vol, real_seg, margin_frac=MARGIN
        )

    # Apply manual overrides
    k, i, j = slice_indices
    if SLICE_AXIAL is not None:
        k = SLICE_AXIAL
    if SLICE_CORONAL is not None:
        i = SLICE_CORONAL
    if SLICE_SAGITTAL is not None:
        j = SLICE_SAGITTAL
    slice_indices = (k, i, j)

    logger.info(f"  Weight octant — octant: {octant}, slices: {slice_indices}")

    cfg = RenderConfig(
        octant=octant,
        zoom=ZOOM,
        window_size=WINDOW_SIZE,
        cmap=WEIGHT_CMAP,
    )

    plotter = render_glioma_octant(
        w_diff,
        dummy_seg,
        slice_indices,
        cfg=cfg,
        brain_mask=brain_mask,
        off_screen=True,
    )

    out_stem = output_dir / f"{PATIENT_ID}_{SPACING}_{PULSE}_weight_octant"
    save_pyvista_figure(plotter, out_stem)
    plotter.close()

    # Also render per-expert weight maps
    for idx, name in enumerate(model_names):
        w_expert = weight_maps[..., idx].copy()
        w_expert[~brain_mask] = np.nan

        plotter = render_glioma_octant(
            w_expert,
            dummy_seg,
            slice_indices,
            cfg=cfg,
            brain_mask=brain_mask,
            off_screen=True,
        )
        out_stem = output_dir / f"{PATIENT_ID}_{SPACING}_{PULSE}_weight_{name}"
        save_pyvista_figure(plotter, out_stem)
        plotter.close()


# =============================================================================
# Visualisation 2 — Patch cube comparison (MRI vs weight cube)
# =============================================================================


def find_interesting_patch(
    weight_maps: np.ndarray,
    brain_mask: np.ndarray,
    patch_size: int,
) -> Tuple[int, int, int]:
    """Find the patch origin whose weight is most extreme (furthest from 0.5).

    Args:
        weight_maps: 4D array (Z, Y, X, n_models).
        brain_mask: 3D boolean brain mask.
        patch_size: Cube side length.

    Returns:
        (z0, y0, x0) origin of the best patch.
    """
    w_diff = compute_weight_difference(weight_maps)
    w_diff[~brain_mask] = 0.5  # Neutral outside brain

    deviation = np.abs(w_diff - 0.5)

    Z, Y, X = w_diff.shape
    best_score = -1.0
    best_origin = (0, 0, 0)

    # Scan on stride grid for efficiency
    step = max(1, patch_size // 2)
    for z0 in range(0, Z - patch_size + 1, step):
        for y0 in range(0, Y - patch_size + 1, step):
            for x0 in range(0, X - patch_size + 1, step):
                p = patch_size
                patch_mask = brain_mask[z0:z0+p, y0:y0+p, x0:x0+p]
                if patch_mask.mean() < 0.5:
                    continue  # Skip mostly-outside-brain patches
                score = deviation[z0:z0+p, y0:y0+p, x0:x0+p].mean()
                if score > best_score:
                    best_score = score
                    best_origin = (z0, y0, x0)

    logger.info(
        f"  Best patch origin: {best_origin}, "
        f"mean |w-0.5|={best_score:.4f}"
    )
    return best_origin


def render_cube_slices(
    cube: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    vcenter: float | None,
    title: str,
    out_path: Path,
) -> None:
    """Render three orthogonal mid-slices of a cube and save as PNG + PDF.

    Args:
        cube: 3D array (patch_size, patch_size, patch_size).
        cmap: Matplotlib colormap name.
        vmin: Colorbar minimum.
        vmax: Colorbar maximum.
        vcenter: If not None, centre the diverging norm here.
        title: Figure suptitle (empty string for no title).
        out_path: Stem path (without extension).
    """
    p = cube.shape[0]
    mid = p // 2

    slices = {
        "Axial": cube[mid, :, :],
        "Coronal": cube[:, mid, :],
        "Sagittal": cube[:, :, mid],
    }

    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, sl) in zip(axes, slices.items()):
        im = ax.imshow(
            sl.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path.with_suffix(".png")), dpi=200, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path.with_suffix('.png').name}")


def generate_patch_cubes(
    hr_vol: np.ndarray,
    weight_maps: np.ndarray,
    model_names: list[str],
    brain_mask: np.ndarray,
    output_dir: Path,
) -> None:
    """Extract and render patch cubes for MRI and weight maps."""
    p = PATCH_SIZE

    if PATCH_ORIGIN is not None:
        z0, y0, x0 = PATCH_ORIGIN
    else:
        z0, y0, x0 = find_interesting_patch(weight_maps, brain_mask, p)

    logger.info(f"Patch cube origin: ({z0}, {y0}, {x0}), size: {p}")

    mri_cube = hr_vol[z0:z0+p, y0:y0+p, x0:x0+p]
    w_diff = compute_weight_difference(weight_maps)
    weight_cube = w_diff[z0:z0+p, y0:y0+p, x0:x0+p]

    # MRI cube — grayscale
    render_cube_slices(
        mri_cube,
        cmap="gray",
        vmin=float(np.nanpercentile(mri_cube, 1)),
        vmax=float(np.nanpercentile(mri_cube, 99)),
        vcenter=None,
        title="",
        out_path=output_dir / f"{PATIENT_ID}_{SPACING}_patch_mri",
    )

    # Weight cube — diverging cmap centred at 0.5
    e1 = model_names[0] if len(model_names) > 0 else "Expert1"
    e2 = model_names[1] if len(model_names) > 1 else "Expert2"

    render_cube_slices(
        weight_cube,
        cmap=WEIGHT_CMAP,
        vmin=0.0,
        vmax=1.0,
        vcenter=0.5,
        title="",
        out_path=output_dir / f"{PATIENT_ID}_{SPACING}_patch_weight",
    )

    # Also save a combined figure with cbar labels
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    mid = p // 2
    plane_names = ["Axial", "Coronal", "Sagittal"]
    mri_slices = [mri_cube[mid, :, :], mri_cube[:, mid, :], mri_cube[:, :, mid]]
    w_slices = [weight_cube[mid, :, :], weight_cube[:, mid, :], weight_cube[:, :, mid]]

    mri_vmin = float(np.nanpercentile(mri_cube, 1))
    mri_vmax = float(np.nanpercentile(mri_cube, 99))
    w_norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    for col in range(3):
        # MRI row
        axes[0, col].imshow(
            mri_slices[col].T, origin="lower", cmap="gray",
            vmin=mri_vmin, vmax=mri_vmax, interpolation="nearest",
        )
        axes[0, col].set_title(plane_names[col], fontsize=10)
        axes[0, col].axis("off")

        # Weight row
        im = axes[1, col].imshow(
            w_slices[col].T, origin="lower", cmap=WEIGHT_CMAP,
            norm=w_norm, interpolation="nearest",
        )
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("MRI", fontsize=11)
    axes[1, 0].set_ylabel("Weight", fontsize=11)

    cbar = fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8, pad=0.04)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([f"1.0 {e1}", "0.75", "0.5 (equal)", "0.75", f"1.0 {e2}"])
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = output_dir / f"{PATIENT_ID}_{SPACING}_patch_combined"
    fig.savefig(str(out.with_suffix(".png")), dpi=200, bbox_inches="tight")
    fig.savefig(str(out.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out.with_suffix('.png').name}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Generate weight visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load HR volume
    logger.info(f"Loading HR volume: {SOURCE_H5}")
    hr_vol = load_hr_volume(SOURCE_H5, PATIENT_ID, PULSE)
    logger.info(f"  HR shape: {hr_vol.shape}")

    # Load weight maps
    logger.info(f"Loading weight maps: {WEIGHTS_NPZ}")
    weight_maps, model_names = load_weight_maps(WEIGHTS_NPZ)
    logger.info(
        f"  Weight shape: {weight_maps.shape}, "
        f"experts: {model_names}"
    )

    # Ensure weight maps match HR volume shape
    if weight_maps.shape[:3] != hr_vol.shape:
        logger.warning(
            f"  Weight shape {weight_maps.shape[:3]} != HR shape {hr_vol.shape}. "
            "Attempting to continue anyway."
        )

    # Compute brain mask from HR
    brain_mask = compute_brain_mask(hr_vol)
    logger.info(f"  Brain mask: {brain_mask.sum()} voxels")

    # --- Visualisation 1: Brain-projected weight octant ---
    logger.info("Generating weight octant figures...")
    generate_weight_octant(hr_vol, weight_maps, model_names, brain_mask, OUTPUT_DIR)

    # --- Visualisation 2: Patch cube comparison ---
    logger.info("Generating patch cube figures...")
    generate_patch_cubes(hr_vol, weight_maps, model_names, brain_mask, OUTPUT_DIR)

    logger.info(f"\nDone — all figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
