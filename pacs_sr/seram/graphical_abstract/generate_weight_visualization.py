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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm

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
    "/media/mpascual/Sandisk2TB/research/pacs_sr/article/graphical_abstract/weight_viz"
)

# Octant / camera settings (same zoom as anisotropy/sr_methods figures)
ZOOM = 3.8
WINDOW_SIZE = (1400, 1200)
MARGIN = 0.05

# Manual slice overrides (None = auto from tumor centroid)
SLICE_AXIAL: int | None = 90
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
                patch_mask = brain_mask[z0 : z0 + p, y0 : y0 + p, x0 : x0 + p]
                if patch_mask.mean() < 0.5:
                    continue  # Skip mostly-outside-brain patches
                score = deviation[z0 : z0 + p, y0 : y0 + p, x0 : x0 + p].mean()
                if score > best_score:
                    best_score = score
                    best_origin = (z0, y0, x0)

    logger.info(f"  Best patch origin: {best_origin}, mean |w-0.5|={best_score:.4f}")
    return best_origin


def _draw_cube_edges(
    ax,
    color: str = "black",
    linewidth: float = 0.8,
) -> None:
    """Draw the 12 edges of a unit cube on a 3D axes."""
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],  # bottom
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],  # top
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # vertical
    ]
    for i, j in edges:
        ax.plot3D(
            [verts[i][0], verts[j][0]],
            [verts[i][1], verts[j][1]],
            [verts[i][2], verts[j][2]],
            color=color,
            linewidth=linewidth,
        )


def render_3d_cube(
    cube: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    vcenter: float | None,
    out_path: Path,
) -> None:
    """Render a 3D cube with outer slices textured on three visible faces.

    The three visible faces (top, front, right) are textured with the
    corresponding outer slices of the input volume. Camera is positioned
    to show all three faces in an isometric-like projection.

    Args:
        cube: 3D array (Z, Y, X).
        cmap: Matplotlib colormap name.
        vmin: Colormap minimum.
        vmax: Colormap maximum.
        vcenter: If not None, centre the diverging norm here.
        out_path: Stem path (without extension).
    """
    p = cube.shape[0]

    # Outer slices for the three visible faces
    top_slice = cube[-1, :, :]  # Z=max -> top face
    front_slice = cube[:, 0, :]  # Y=0   -> front face
    right_slice = cube[:, :, -1]  # X=max -> right face

    # Build normalizer and colormap
    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Vertex grid: (p+1) edge coordinates to produce p x p face cells
    edges = np.linspace(0, 1, p + 1)

    def _to_rgba(slice_2d: np.ndarray) -> np.ndarray:
        """Map a 2D slice to RGBA, padded to (p+1, p+1, 4) for plot_surface."""
        rgba = np.ones((p + 1, p + 1, 4))
        rgba[:p, :p] = colormap(norm(slice_2d))
        return rgba

    # Top face: Z=1, varying X (col) and Y (row)
    X_t, Y_t = np.meshgrid(edges, edges)
    Z_t = np.ones_like(X_t)
    ax.plot_surface(
        X_t,
        Y_t,
        Z_t,
        facecolors=_to_rgba(top_slice),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        linewidth=0,
    )

    # Front face: Y=0, varying X (col) and Z (row)
    X_f, Z_f = np.meshgrid(edges, edges)
    Y_f = np.zeros_like(X_f)
    ax.plot_surface(
        X_f,
        Y_f,
        Z_f,
        facecolors=_to_rgba(front_slice),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        linewidth=0,
    )

    # Right face: X=1, varying Y (col) and Z (row)
    Y_r, Z_r = np.meshgrid(edges, edges)
    X_r = np.ones_like(Y_r)
    ax.plot_surface(
        X_r,
        Y_r,
        Z_r,
        facecolors=_to_rgba(right_slice),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        linewidth=0,
    )

    # Draw cube boundary edges
    _draw_cube_edges(ax)

    # Camera and style
    ax.view_init(elev=25, azim=-60)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(out_path.with_suffix(".png")),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )
    fig.savefig(
        str(out_path.with_suffix(".pdf")),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    logger.info(f"  Saved: {out_path.with_suffix('.png').name}")


def generate_standalone_colorbar(
    cmap: str,
    vmin: float,
    vmax: float,
    vcenter: float | None,
    model_names: list[str],
    out_path: Path,
    orientation: str = "horizontal",
) -> None:
    """Generate a standalone colorbar image (no data plot).

    Args:
        cmap: Matplotlib colormap name.
        vmin: Colormap minimum.
        vmax: Colormap maximum.
        vcenter: If not None, centre the diverging norm here.
        model_names: Expert model names for tick labels.
        out_path: Stem path (without extension).
        orientation: 'horizontal' or 'vertical'.
    """
    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    e1 = model_names[0] if len(model_names) > 0 else "Expert1"
    e2 = model_names[1] if len(model_names) > 1 else "Expert2"

    if orientation == "horizontal":
        fig, ax = plt.subplots(figsize=(6, 0.5))
    else:
        fig, ax = plt.subplots(figsize=(0.5, 6))

    cbar = fig.colorbar(sm, cax=ax, orientation=orientation)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([f"1.0 {e1}", "0.75", "0.5 (equal)", "0.75", f"1.0 {e2}"])
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(out_path.with_suffix(".png")),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    fig.savefig(
        str(out_path.with_suffix(".pdf")),
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"  Saved: {out_path.with_suffix('.png').name}")


def find_diverse_patches(
    weight_maps: np.ndarray,
    brain_mask: np.ndarray,
    patch_size: int,
    n_patches: int = 3,
    min_distance: int = 40,
    seed: int = 42,
) -> list[Tuple[int, int, int]]:
    """Find diverse, brain-interior patches with varied weight distributions.

    Selects patches that are (a) mostly inside the brain, (b) spatially
    separated by at least ``min_distance`` voxels, and (c) span a range of
    weight deviation levels.

    Args:
        weight_maps: 4D array (Z, Y, X, n_models).
        brain_mask: 3D boolean brain mask.
        patch_size: Cube side length.
        n_patches: Number of patches to return.
        min_distance: Minimum L2 distance between patch centres.
        seed: Random seed for reproducibility.

    Returns:
        List of (z0, y0, x0) origins.
    """
    rng = np.random.default_rng(seed)
    w_diff = compute_weight_difference(weight_maps)
    w_diff[~brain_mask] = 0.5

    deviation = np.abs(w_diff - 0.5)
    Z, Y, X = w_diff.shape
    p = patch_size
    step = max(1, p // 2)

    # Score all candidate patches
    candidates: list[Tuple[float, Tuple[int, int, int]]] = []
    for z0 in range(0, Z - p + 1, step):
        for y0 in range(0, Y - p + 1, step):
            for x0 in range(0, X - p + 1, step):
                patch_mask = brain_mask[z0 : z0 + p, y0 : y0 + p, x0 : x0 + p]
                if patch_mask.mean() < 0.7:
                    continue
                score = deviation[z0 : z0 + p, y0 : y0 + p, x0 : x0 + p].mean()
                candidates.append((score, (z0, y0, x0)))

    # Sort by score and pick diverse set
    candidates.sort(key=lambda c: c[0], reverse=True)

    selected: list[Tuple[int, int, int]] = []
    for _, origin in candidates:
        centre = np.array(origin) + p // 2
        if all(
            np.linalg.norm(centre - (np.array(s) + p // 2)) >= min_distance
            for s in selected
        ):
            selected.append(origin)
        if len(selected) >= n_patches:
            break

    # If we couldn't fill via diversity, add random brain-interior patches
    while len(selected) < n_patches and candidates:
        _, origin = candidates[rng.integers(len(candidates))]
        centre = np.array(origin) + p // 2
        if all(
            np.linalg.norm(centre - (np.array(s) + p // 2)) >= min_distance // 2
            for s in selected
        ):
            selected.append(origin)

    logger.info(f"  Selected {len(selected)} diverse patches")
    for idx, s in enumerate(selected):
        logger.info(f"    Patch {idx}: origin={s}")
    return selected


def generate_patch_cubes(
    hr_vol: np.ndarray,
    weight_maps: np.ndarray,
    model_names: list[str],
    brain_mask: np.ndarray,
    output_dir: Path,
) -> None:
    """Extract and render 3D patch cubes for MRI and weight maps."""
    p = PATCH_SIZE
    w_diff = compute_weight_difference(weight_maps)

    # --- Primary patch (most extreme weights) ---
    if PATCH_ORIGIN is not None:
        primary_origin = PATCH_ORIGIN
    else:
        primary_origin = find_interesting_patch(weight_maps, brain_mask, p)

    all_origins = [("patch", primary_origin)]

    # --- Additional diverse patches ---
    extra_origins = find_diverse_patches(
        weight_maps, brain_mask, p, n_patches=3, min_distance=40
    )
    for idx, origin in enumerate(extra_origins):
        all_origins.append((f"patch_extra{idx + 1}", origin))

    # Render each patch pair (MRI + weight cube)
    for label, (z0, y0, x0) in all_origins:
        logger.info(f"  Rendering {label} at ({z0}, {y0}, {x0})...")

        mri_cube = hr_vol[z0 : z0 + p, y0 : y0 + p, x0 : x0 + p]
        weight_cube = w_diff[z0 : z0 + p, y0 : y0 + p, x0 : x0 + p]

        render_3d_cube(
            mri_cube,
            cmap="gray",
            vmin=float(np.nanpercentile(mri_cube, 1)),
            vmax=float(np.nanpercentile(mri_cube, 99)),
            vcenter=None,
            out_path=output_dir / f"{PATIENT_ID}_{SPACING}_{label}_mri",
        )

        render_3d_cube(
            weight_cube,
            cmap=WEIGHT_CMAP,
            vmin=0.0,
            vmax=1.0,
            vcenter=0.5,
            out_path=output_dir / f"{PATIENT_ID}_{SPACING}_{label}_weight",
        )

    # Standalone colorbars (once)
    generate_standalone_colorbar(
        cmap=WEIGHT_CMAP,
        vmin=0.0,
        vmax=1.0,
        vcenter=0.5,
        model_names=model_names,
        out_path=output_dir / f"{PATIENT_ID}_{SPACING}_weight_colorbar_horizontal",
        orientation="horizontal",
    )
    generate_standalone_colorbar(
        cmap=WEIGHT_CMAP,
        vmin=0.0,
        vmax=1.0,
        vcenter=0.5,
        model_names=model_names,
        out_path=output_dir / f"{PATIENT_ID}_{SPACING}_weight_colorbar_vertical",
        orientation="vertical",
    )


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
    logger.info(f"  Weight shape: {weight_maps.shape}, experts: {model_names}")

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
