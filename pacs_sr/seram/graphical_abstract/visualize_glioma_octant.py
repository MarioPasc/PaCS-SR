#!/usr/bin/env python3
"""
visualize_glioma_octant.py - 3D octant visualization for BraTS glioma data.

Creates a 3D PyVista visualization showing a brain surface with a cutaway octant
revealing three orthogonal MRI slices inside. Glioma tumor compartments are
rendered as colored surface meshes:

    Label 1: Non-Enhancing Tumor / Necrotic Core (NCR)  — yellow  (#DDCC77)
    Label 2: Peritumoral Edema (ED)                      — teal    (#44AA99)
    Label 3: Enhancing Tumor (ET)                        — red     (#CC3311)
    Label 4: Resection Cavity / Other (RC)               — purple  (#AA3377)

Input: BraTS-GLI subject directory containing {id}-{modality}.nii.gz + {id}-seg.nii.gz
       OR volumes/segmentation passed directly as numpy arrays (for HDF5 pipelines).

Usage:
    python pacs_sr/seram/graphical_abstract/visualize_glioma_octant.py \\
        --subject-dir /path/to/BraTS-GLI-00512-100 \\
        --output /path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from dataclasses import dataclass, fields
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None

try:
    from scipy import ndimage

    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    from skimage import measure

    HAS_PYVISTA = True
except ModuleNotFoundError:
    HAS_PYVISTA = False
    pv = None

logger = logging.getLogger(__name__)

# =============================================================================
# BraTS Glioma Label Configuration
# =============================================================================

GLIOMA_LABELS: Final[Dict[int, str]] = {
    1: "NCR (Necrotic Core)",
    2: "ED (Peritumoral Edema)",
    3: "ET (Enhancing Tumor)",
    4: "RC (Resection Cavity)",
}

# Colorblind-friendly palette (Paul Tol's vibrant)
GLIOMA_COLORS: Final[Dict[int, Tuple[float, float, float]]] = {
    1: (0.867, 0.800, 0.467),  # Yellow  (#DDCC77) — NCR
    2: (0.267, 0.667, 0.600),  # Teal    (#44AA99) — ED
    3: (0.800, 0.200, 0.067),  # Red     (#CC3311) — ET
    4: (0.667, 0.200, 0.467),  # Purple  (#AA3377) — RC
}

# Per-label opacity: tumor cores opaque, edema semi-transparent.
GLIOMA_ALPHA: Dict[int, float] = {
    1: 0.85,
    2: 0.35,
    3: 1.0,
    4: 0.6,
}

VALID_MODALITIES: Final[List[str]] = ["t1c", "t1n", "t2w", "t2f"]


@dataclass(frozen=True)
class RenderConfig:
    """Configuration for PyVista surface rendering."""

    iso_level: float = 0.5
    mesh_alpha: float = 1.0
    mesh_color: Tuple[float, float, float] = (0.82, 0.82, 0.82)
    slice_alpha: float = 0.95
    cmap: str = "gray"
    specular: float = 0.3
    specular_power: float = 20.0
    plane_bias: float = 0.01
    smooth_iterations: int = 5
    smooth_lambda: float = 0.5
    tumor_alpha: float = 0.85
    tumor_smooth_iterations: int = 3
    octant: Tuple[bool, bool, bool] = (False, False, False)
    camera_azimuth: Optional[float] = None
    camera_elevation: Optional[float] = None
    zoom: float = 2.2
    window_size: Tuple[int, int] = (1400, 1200)


# =============================================================================
# Data I/O
# =============================================================================


def resolve_brats_paths(
    subject_dir: pathlib.Path, modality: str = "t1c"
) -> Tuple[pathlib.Path, pathlib.Path]:
    """Resolve image and segmentation paths from a BraTS-GLI subject directory.

    Args:
        subject_dir: Path to BraTS subject directory.
        modality: MRI modality to load.

    Returns:
        Tuple of (image_path, segmentation_path).
    """
    if modality not in VALID_MODALITIES:
        raise ValueError(
            f"Invalid modality '{modality}'. Must be one of {VALID_MODALITIES}"
        )

    image_matches = sorted(subject_dir.glob(f"*{modality}.nii.gz"))
    seg_matches = sorted(subject_dir.glob("*seg.nii.gz"))

    if not image_matches:
        raise FileNotFoundError(
            f"No *{modality}.nii.gz found in {subject_dir}"
        )
    if len(image_matches) > 1:
        raise ValueError(
            f"Multiple *{modality}.nii.gz matches in {subject_dir}: {image_matches}"
        )
    if not seg_matches:
        raise FileNotFoundError(f"No *seg.nii.gz found in {subject_dir}")
    if len(seg_matches) > 1:
        raise ValueError(
            f"Multiple *seg.nii.gz matches in {subject_dir}: {seg_matches}"
        )

    return image_matches[0], seg_matches[0]


def load_volume(path: pathlib.Path) -> np.ndarray:
    """Load a 3D NIfTI volume, converting to RAS orientation.

    Args:
        path: Path to NIfTI file.

    Returns:
        3D float64 array in RAS orientation.
    """
    if nib is None:
        raise ImportError("Reading NIfTI requires nibabel: pip install nibabel")
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return np.asanyarray(img.get_fdata())


# =============================================================================
# Brain Mask & Mesh Helpers
# =============================================================================


def compute_brain_mask(
    volume: np.ndarray, threshold_percentile: float = 5.0
) -> np.ndarray:
    """Compute a brain mask via thresholding + morphological cleanup.

    Args:
        volume: 3D MRI volume.
        threshold_percentile: Intensity percentile for foreground detection.

    Returns:
        Binary mask array.
    """
    if not HAS_SCIPY:
        raise ImportError("Brain mask computation requires scipy")

    finite_vals = volume[np.isfinite(volume)]
    if finite_vals.size == 0:
        return np.ones(volume.shape, dtype=bool)

    threshold = np.percentile(finite_vals, threshold_percentile)
    mask = volume > threshold

    struct = ndimage.generate_binary_structure(3, 1)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=struct, iterations=2)

    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        mask = labeled == largest

    mask = ndimage.binary_closing(mask, structure=struct, iterations=2)
    return mask.astype(bool)


def laplacian_smooth(
    vertices: np.ndarray, faces: np.ndarray, iterations: int = 5, lam: float = 0.5
) -> np.ndarray:
    """Apply Laplacian smoothing to a triangle mesh.

    Args:
        vertices: (N, 3) vertex positions.
        faces: (F, 3) triangle indices.
        iterations: Smoothing iterations.
        lam: Smoothing factor (0=none, 1=full averaging).

    Returns:
        Smoothed vertex positions.
    """
    n_verts = len(vertices)
    verts = vertices.copy()

    adjacency: List[set] = [set() for _ in range(n_verts)]
    for f in faces:
        for idx in range(3):
            adjacency[f[idx]].add(f[(idx + 1) % 3])
            adjacency[f[idx]].add(f[(idx + 2) % 3])

    for _ in range(iterations):
        new_verts = verts.copy()
        for idx in range(n_verts):
            neighbors = list(adjacency[idx])
            if neighbors:
                centroid = verts[neighbors].mean(axis=0)
                new_verts[idx] = (1 - lam) * verts[idx] + lam * centroid
        verts = new_verts

    return verts


def faces_to_pyvista(faces_tri: np.ndarray) -> np.ndarray:
    """Convert (F, 3) triangle array to PyVista face format [3, i, j, k, ...]."""
    f = np.hstack(
        [
            np.full((faces_tri.shape[0], 1), 3, dtype=np.int64),
            faces_tri.astype(np.int64),
        ]
    )
    return f.ravel()


def find_tumor_center(segmentation: np.ndarray) -> Tuple[int, int, int]:
    """Find the centroid of the tumor (all non-zero labels).

    Args:
        segmentation: 3D label array.

    Returns:
        (x, y, z) centroid voxel indices.
    """
    coords = np.array(np.where(segmentation > 0))
    if coords.size == 0:
        raise ValueError("No tumor found in segmentation")
    center = coords.mean(axis=1).astype(int)
    return tuple(center)


def find_optimal_octant(
    volume: np.ndarray,
    segmentation: np.ndarray,
    margin_frac: float = 0.05,
) -> Tuple[Tuple[int, int, int], Tuple[bool, bool, bool]]:
    """Find optimal slice indices and octant orientation for tumor visualization.

    Positions the octant cut to reveal the maximum tumor cross-section.

    Args:
        volume: 3D MRI volume.
        segmentation: 3D label array.
        margin_frac: Extra margin beyond tumor inner edge.

    Returns:
        Tuple of (slice_indices_kij, octant_bools).
    """
    nx, ny, nz = volume.shape

    if not np.any(segmentation > 0):
        return (nz // 2, nx // 2, ny // 2), (False, False, False)

    cx, cy, cz = find_tumor_center(segmentation)

    inv_x = cx < nx // 2
    inv_y = cy < ny // 2
    inv_z = cz < nz // 2

    tumor_coords = np.argwhere(segmentation > 0)
    t_min = tumor_coords.min(axis=0)
    t_max = tumor_coords.max(axis=0)

    margin = np.array([nx, ny, nz]) * margin_frac

    cap_min = np.array([nx, ny, nz]) * 0.35
    cap_max = np.array([nx, ny, nz]) * 0.65

    if inv_x:
        sx = int(np.clip(t_max[0] + margin[0], cap_min[0], nx - 2))
    else:
        sx = int(np.clip(t_min[0] - margin[0], 1, cap_max[0]))

    if inv_y:
        sy = int(np.clip(t_max[1] + margin[1], cap_min[1], ny - 2))
    else:
        sy = int(np.clip(t_min[1] - margin[1], 1, cap_max[1]))

    if inv_z:
        sz = int(np.clip(t_max[2] + margin[2], cap_min[2], nz - 2))
    else:
        sz = int(np.clip(t_min[2] - margin[2], 1, cap_max[2]))

    logger.info(
        f"  Tumor bbox: x=[{t_min[0]},{t_max[0]}] y=[{t_min[1]},{t_max[1]}] z=[{t_min[2]},{t_max[2]}]"
    )
    return (sz, sx, sy), (inv_x, inv_y, inv_z)


# =============================================================================
# PyVista Rendering
# =============================================================================


def render_glioma_octant(
    volume: np.ndarray,
    segmentation: np.ndarray,
    slice_indices: Tuple[int, int, int],
    *,
    cfg: RenderConfig = RenderConfig(),
    brain_mask: Optional[np.ndarray] = None,
    tumor_alpha_override: Optional[float] = None,
    save: Optional[pathlib.Path] = None,
    off_screen: bool = True,
) -> Any:
    """Render 3D octant visualization with brain surface and multi-label glioma tumor.

    Args:
        volume: 3D MRI volume (nx, ny, nz), RAS orientation.
        segmentation: 3D integer label array (0=bg, 1=NCR, 2=ED, 3=ET, 4=RC).
        slice_indices: (k_axial, i_coronal, j_sagittal).
        cfg: Rendering configuration.
        brain_mask: Pre-computed brain mask (if None, computed from volume).
        tumor_alpha_override: If set, use this alpha for ALL tumor labels.
        save: Output file path (PNG or PDF).
        off_screen: Render without display window.

    Returns:
        pv.Plotter object.
    """
    if not HAS_PYVISTA:
        raise ImportError("Requires: pip install pyvista scikit-image scipy")

    nx, ny, nz = volume.shape
    k_a, i_c, j_s = slice_indices

    margin = 1
    k = int(np.clip(k_a, margin, nz - 1 - margin))
    i = int(np.clip(i_c, margin, nx - 1 - margin))
    j = int(np.clip(j_s, margin, ny - 1 - margin))

    logger.info(f"Volume shape: {volume.shape}, octant origin: (i={i}, j={j}, k={k})")

    if brain_mask is not None:
        mask = brain_mask
    else:
        mask = compute_brain_mask(volume)
    logger.info(
        f"Brain mask: {mask.sum()} voxels ({100 * mask.sum() / mask.size:.1f}%)"
    )

    vol_masked = volume.copy().astype(np.float32)
    vol_masked[~mask] = np.nan

    inside = vol_masked[np.isfinite(vol_masked)]
    if inside.size > 0:
        vmin = float(np.percentile(inside, 1.0))
        vmax = float(np.percentile(inside, 99.0))
    else:
        vmin, vmax = float(np.nanmin(vol_masked)), float(np.nanmax(vol_masked))

    # -- Plotter setup --
    pv.global_theme.background = "white"
    plotter = pv.Plotter(
        off_screen=off_screen or (save is not None),
        window_size=list(cfg.window_size),
    )

    try:
        plotter.enable_anti_aliasing("msaa")
    except Exception:
        pass
    try:
        plotter.enable_depth_peeling()
    except Exception:
        pass

    # -- Brain surface mesh --
    verts, face_arr, _, _ = measure.marching_cubes(
        mask.astype(np.float32), level=cfg.iso_level
    )
    verts = laplacian_smooth(
        verts, face_arr, iterations=cfg.smooth_iterations, lam=cfg.smooth_lambda
    )
    mesh_full = pv.PolyData(verts, faces_to_pyvista(face_arr))

    inv_x, inv_y, inv_z = cfg.octant
    bx = (-0.5, i + 0.5) if inv_x else (i - 0.5, nx - 0.5)
    by = (-0.5, j + 0.5) if inv_y else (j - 0.5, ny - 0.5)
    bz = (-0.5, k + 0.5) if inv_z else (k - 0.5, nz - 0.5)
    bounds = (bx[0], bx[1], by[0], by[1], bz[0], bz[1])

    mesh_clip = mesh_full.clip_box(bounds=bounds, invert=True, merge_points=True)

    plotter.add_mesh(
        mesh_clip,
        color=cfg.mesh_color,
        opacity=cfg.mesh_alpha,
        smooth_shading=True,
        specular=float(np.clip(cfg.specular, 0.0, 1.0)),
        specular_power=float(cfg.specular_power),
        show_edges=False,
    )

    # -- Tumor surface meshes (one per label) --
    # Build a voxel-space mask of the visible interior (the removed corner
    # box that creates the cutaway opening) so we can crop the segmentation
    # *before* marching cubes.  This avoids clip_box on the mesh, which
    # breaks transparency under software renderers.
    visible_interior = np.zeros((nx, ny, nz), dtype=bool)
    x_sl = slice(0, i + 1) if inv_x else slice(i, nx)
    y_sl = slice(0, j + 1) if inv_y else slice(j, ny)
    z_sl = slice(0, k + 1) if inv_z else slice(k, nz)
    visible_interior[x_sl, y_sl, z_sl] = True

    present_labels = [l for l in GLIOMA_LABELS if np.any(segmentation == l)]
    for label_id in present_labels:
        color = GLIOMA_COLORS[label_id]
        label_mask = ((segmentation == label_id) & visible_interior).astype(np.float32)
        if label_mask.sum() < 10:
            continue

        try:
            if HAS_SCIPY:
                binary = label_mask > 0.5
                labeled_cc, n_cc = ndimage.label(binary)
                if n_cc > 1:
                    cc_sizes = ndimage.sum(binary, labeled_cc, range(1, n_cc + 1))
                    largest_cc = np.argmax(cc_sizes) + 1
                    min_size = max(100, cc_sizes[largest_cc - 1] * 0.05)
                    for cc_id in range(1, n_cc + 1):
                        if cc_sizes[cc_id - 1] < min_size:
                            label_mask[labeled_cc == cc_id] = 0.0

            if HAS_SCIPY:
                label_mask = ndimage.gaussian_filter(label_mask, sigma=0.5)

            t_verts, t_faces, _, _ = measure.marching_cubes(label_mask, level=0.5)

            if cfg.tumor_smooth_iterations > 0:
                t_verts = laplacian_smooth(
                    t_verts,
                    t_faces,
                    iterations=cfg.tumor_smooth_iterations,
                    lam=cfg.smooth_lambda,
                )

            tumor_mesh = pv.PolyData(t_verts, faces_to_pyvista(t_faces))

            if tumor_mesh.n_points == 0:
                logger.info(
                    f"Label {label_id} ({GLIOMA_LABELS[label_id]}): "
                    "fully outside visible octant — skipped"
                )
                continue

            if tumor_alpha_override is not None:
                label_alpha = tumor_alpha_override
            else:
                label_alpha = GLIOMA_ALPHA.get(label_id, cfg.tumor_alpha)

            plotter.add_mesh(
                tumor_mesh,
                color=color,
                opacity=label_alpha,
                smooth_shading=True,
                show_edges=False,
            )

            logger.info(
                f"Label {label_id} ({GLIOMA_LABELS[label_id]}): "
                f"{tumor_mesh.n_points} verts, alpha={label_alpha}"
            )
        except Exception as e:
            logger.warning(f"Could not create surface for label {label_id}: {e}")

    # -- MRI slice surfaces --
    grid = pv.ImageData()
    grid.dimensions = np.array(vol_masked.shape, dtype=int) + 1
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (-0.5, -0.5, -0.5)
    grid.cell_data["I"] = vol_masked.ravel(order="F")
    grid = grid.cell_data_to_point_data(pass_cell_data=True)

    clip_x_normal = (-1, 0, 0) if inv_x else (1, 0, 0)
    clip_y_normal = (0, -1, 0) if inv_y else (0, 1, 0)
    clip_z_normal = (0, 0, -1) if inv_z else (0, 0, 1)

    slice_kwargs = dict(
        scalars="I",
        cmap=cfg.cmap,
        clim=(vmin, vmax),
        opacity=cfg.slice_alpha,
        nan_opacity=0.0,
        show_scalar_bar=False,
    )

    # Axial (XY @ z=k)
    z0 = float(k) + (cfg.plane_bias if not inv_z else -cfg.plane_bias)
    slc = grid.slice(normal=(0, 0, 1), origin=(0.0, 0.0, z0))
    slc = slc.clip(normal=clip_x_normal, origin=(float(i), 0.0, 0.0), invert=False)
    slc = slc.clip(normal=clip_y_normal, origin=(0.0, float(j), 0.0), invert=False)
    plotter.add_mesh(slc, **slice_kwargs)

    # Coronal (YZ @ x=i)
    x0 = float(i) + (cfg.plane_bias if not inv_x else -cfg.plane_bias)
    slc = grid.slice(normal=(1, 0, 0), origin=(x0, 0.0, 0.0))
    slc = slc.clip(normal=clip_y_normal, origin=(0.0, float(j), 0.0), invert=False)
    slc = slc.clip(normal=clip_z_normal, origin=(0.0, 0.0, float(k)), invert=False)
    plotter.add_mesh(slc, **slice_kwargs)

    # Sagittal (XZ @ y=j)
    y0 = float(j) + (cfg.plane_bias if not inv_y else -cfg.plane_bias)
    slc = grid.slice(normal=(0, 1, 0), origin=(0.0, y0, 0.0))
    slc = slc.clip(normal=clip_x_normal, origin=(float(i), 0.0, 0.0), invert=False)
    slc = slc.clip(normal=clip_z_normal, origin=(0.0, 0.0, float(k)), invert=False)
    plotter.add_mesh(slc, **slice_kwargs)

    # -- Camera --
    if mask.any():
        idx_arr = np.argwhere(mask)
        (xmin, ymin, zmin), (xmax, ymax, zmax) = idx_arr.min(0), idx_arr.max(0)
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
        brain_size = max(xmax - xmin, ymax - ymin, zmax - zmin)
    else:
        center = (nx / 2, ny / 2, nz / 2)
        brain_size = max(nx, ny, nz)

    plotter.set_focus(center)
    dist = 1.8 * brain_size

    if cfg.camera_azimuth is not None and cfg.camera_elevation is not None:
        azim_rad = np.radians(cfg.camera_azimuth)
        elev_rad = np.radians(cfg.camera_elevation)
        cam_pos = (
            center[0] + dist * np.cos(elev_rad) * np.cos(azim_rad),
            center[1] + dist * np.cos(elev_rad) * np.sin(azim_rad),
            center[2] + dist * np.sin(elev_rad),
        )
    else:
        x_sign = -1 if inv_x else 1
        y_sign = -1 if inv_y else 1
        z_sign = 0.7 if inv_z else 0.9
        cam_pos = (
            center[0] + x_sign * dist * 0.7,
            center[1] + y_sign * dist * 0.7,
            center[2] + z_sign * dist * 0.5,
        )

    plotter.set_position(cam_pos)
    plotter.set_viewup((0, 0, 1))
    plotter.camera.SetViewAngle(35)
    plotter.reset_camera_clipping_range()

    if cfg.zoom != 1.0:
        plotter.camera.zoom(cfg.zoom)

    # -- Save --
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        suffix = save.suffix.lower()
        if suffix == ".pdf":
            plotter.save_graphic(str(save))
        else:
            plotter.screenshot(str(save))
        logger.info(f"Saved: {save}")

    return plotter


def replace_octant(cfg: RenderConfig, **kwargs: Any) -> RenderConfig:
    """Create a new RenderConfig with specified fields replaced.

    Args:
        cfg: Original config.
        **kwargs: Fields to replace.

    Returns:
        New RenderConfig instance.
    """
    d = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    d.update(kwargs)
    return RenderConfig(**d)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="3D octant visualization for BraTS glioma data.",
    )

    parser.add_argument(
        "--subject-dir", type=pathlib.Path, required=True,
        help="Path to single BraTS-GLI subject directory",
    )
    parser.add_argument(
        "--modality", type=str, default="t1c", choices=VALID_MODALITIES,
    )
    parser.add_argument("--output", "-o", type=pathlib.Path, default=None)
    parser.add_argument("--octant", type=str, default=None)
    parser.add_argument("--camera-azimuth", type=float, default=None)
    parser.add_argument("--camera-elevation", type=float, default=None)
    parser.add_argument("--zoom", type=float, default=2.2)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument(
        "--tumor-alpha", type=float, default=None,
        help="Override alpha for ALL tumor labels.",
    )
    parser.add_argument(
        "--window-size", type=int, nargs=2, default=[1400, 1200],
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    octant = (False, False, False)
    if args.octant is not None:
        o = args.octant.lower().strip()
        if len(o) != 3 or not all(c in "np" for c in o):
            raise ValueError(f"--octant must be 3 chars of 'n'/'p', got '{args.octant}'")
        octant = (o[0] == "n", o[1] == "n", o[2] == "n")

    cfg = RenderConfig(
        octant=octant,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        zoom=args.zoom,
        window_size=tuple(args.window_size),
    )

    image_path, seg_path = resolve_brats_paths(args.subject_dir, args.modality)
    volume = load_volume(image_path)
    segmentation = load_volume(seg_path).astype(np.int32)

    slice_indices, auto_octant = find_optimal_octant(
        volume, segmentation, margin_frac=args.margin
    )
    if args.octant is None and cfg.camera_azimuth is None:
        cfg = replace_octant(cfg, octant=auto_octant)

    output = args.output or pathlib.Path(
        f"{args.subject_dir.name}_{args.modality}_octant.png"
    )

    render_glioma_octant(
        volume,
        segmentation,
        slice_indices,
        cfg=cfg,
        tumor_alpha_override=args.tumor_alpha,
        save=output,
        off_screen=True,
    )
    print(f"[OK] -> {output}")


if __name__ == "__main__":
    main()
