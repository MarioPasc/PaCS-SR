#!/usr/bin/env python3
"""SERAM Figure 1: Per-sequence comparison grid.

Generates a grid (spacings x models) showing axial slices + MAE maps
with zoomed tumor insets for each MRI sequence.

Usage:
    python -m pacs_sr.seram.figure_comparison_grid --config configs/seram_glioma.yaml
    python -m pacs_sr.seram.figure_comparison_grid \\
        --config configs/seram_glioma_local.yaml \\
        --patient-id BraTS-GLI-02152-100 \\
        --seg-dir /media/mpascual/PortableSSD/BraTS_GLI/LowRes_HighRes/high_resolution \\
        --showcase-weight-maps --tumor-slices --show-tumor-contours \\
        --slice-range 90 130
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pacs_sr.data.hdf5_io import (
    blend_key,
    expert_h5_path,
    expert_key,
    has_key,
    hr_key,
    list_groups,
    read_volume,
    results_h5_path,
)
from pacs_sr.utils.visualize_pacs_sr import normalize01
from pacs_sr.utils.settings import apply_ieee_style, PLOT_SETTINGS

LOG = logging.getLogger(__name__)

# BraTS segmentation label colours (for contour drawing)
_TUMOR_LABEL_COLORS: Dict[int, Tuple[str, str]] = {
    1: ("#FF4444", "NCR"),  # necrotic core – red
    2: ("#44FF44", "ED"),  # peritumoral edema – green
    3: ("#4488FF", "ET"),  # enhancing tumour – blue
    4: ("#FFAA00", "WT"),  # whole tumour (if present) – orange
}


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------


def _reorient_to_ras(
    data: np.ndarray, affine: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorient a 3D volume to RAS+ orientation."""
    img = nib.Nifti1Image(data, affine)
    orig_ornt = io_orientation(affine)
    ras_ornt = axcodes2ornt(("R", "A", "S"))
    transform = ornt_transform(orig_ornt, ras_ornt)
    reoriented = img.as_reoriented(transform)
    return np.asarray(reoriented.dataobj), reoriented.affine


def _axial_slice(vol: np.ndarray, idx: int) -> np.ndarray:
    """Extract an axial slice (axis-2) from an RAS-oriented volume."""
    return vol[:, :, idx]


# ---------------------------------------------------------------------------
# Slice / bbox selection
# ---------------------------------------------------------------------------


def _find_tumor_bbox(seg_slice: np.ndarray, margin: int = 10) -> np.ndarray:
    """Bounding box around nonzero voxels in a 2D segmentation slice."""
    rs, cs = np.where(seg_slice > 0)
    if len(rs) == 0:
        cy, cx = seg_slice.shape[0] // 2, seg_slice.shape[1] // 2
        return np.array([cy - 30, cy + 30, cx - 30, cx + 30])
    return np.array(
        [
            max(0, rs.min() - margin),
            min(seg_slice.shape[0], rs.max() + margin),
            max(0, cs.min() - margin),
            min(seg_slice.shape[1], cs.max() + margin),
        ]
    )


def _find_high_variance_slice_bbox(
    vol: np.ndarray,
    slice_range: Optional[Tuple[int, int]] = None,
) -> Tuple[int, np.ndarray]:
    """Find axial slice with highest intensity variance and a tight bbox."""
    from scipy.ndimage import uniform_filter

    n_slices = vol.shape[2]
    scores = np.zeros(n_slices)
    lo = 0 if slice_range is None else max(0, slice_range[0])
    hi = n_slices if slice_range is None else min(n_slices, slice_range[1])
    for z in range(lo, hi):
        sl = vol[:, :, z]
        mask = sl > sl.max() * 0.05
        if mask.sum() > 100:
            scores[z] = sl[mask].var()

    best_z = int(np.argmax(scores))
    sl = vol[:, :, best_z]
    brain_mask = sl > sl.max() * 0.05
    if brain_mask.sum() < 50:
        cr, ca = vol.shape[0] // 2, vol.shape[1] // 2
        return best_z, np.array([cr - 30, cr + 30, ca - 30, ca + 30])

    k = 15
    mean_local = uniform_filter(sl.astype(np.float64), size=k)
    mean_sq = uniform_filter(sl.astype(np.float64) ** 2, size=k)
    var_map = mean_sq - mean_local**2
    var_map[~brain_mask] = 0
    peak = np.unravel_index(np.argmax(var_map), var_map.shape)
    m = 25
    return best_z, np.array(
        [
            max(0, peak[0] - m),
            min(vol.shape[0], peak[0] + m),
            max(0, peak[1] - m),
            min(vol.shape[1], peak[1] + m),
        ]
    )


def _find_best_tumor_slice(
    seg_ras: np.ndarray,
    gt_vol: np.ndarray,
    expert_vols: Dict[str, np.ndarray],
    blend_vol: Optional[np.ndarray],
    slice_range: Optional[Tuple[int, int]] = None,
    min_tumor_pixels: int = 200,
) -> Tuple[int, np.ndarray]:
    """Select the axial slice that best showcases PaCS-SR advantage.

    Filters to slices in *slice_range* that contain at least *min_tumor_pixels*
    tumor voxels, then ranks by ``mean_expert_MAE - pacs_sr_MAE`` (higher =
    better showcase).

    Args:
        seg_ras: Segmentation volume in RAS (nonzero = tumor).
        gt_vol: Ground-truth HR volume in RAS.
        expert_vols: ``{model_name: vol_ras}`` for each expert.
        blend_vol: PaCS-SR blend volume in RAS (can be None).
        slice_range: Optional (lo, hi) for axial indices.
        min_tumor_pixels: Minimum nonzero seg voxels to consider a slice.

    Returns:
        (best_z, tumor_bbox) — index and bounding box in full-volume coords.
    """
    n_slices = seg_ras.shape[2]
    lo = 0 if slice_range is None else max(0, slice_range[0])
    hi = n_slices if slice_range is None else min(n_slices, slice_range[1])

    best_score = -np.inf
    best_z = (lo + hi) // 2
    best_bbox = np.array([0, seg_ras.shape[0], 0, seg_ras.shape[1]])

    for z in range(lo, hi):
        seg_sl = seg_ras[:, :, z]
        tumor_mask = seg_sl > 0
        if tumor_mask.sum() < min_tumor_pixels:
            continue

        gt_sl = gt_vol[:, :, z]

        # Mean MAE over tumour voxels for each expert
        expert_maes = []
        for name, evol in expert_vols.items():
            sz = min(z, evol.shape[2] - 1)
            e_sl = evol[:, :, sz]
            if e_sl.shape != gt_sl.shape:
                fac = (gt_sl.shape[0] / e_sl.shape[0], gt_sl.shape[1] / e_sl.shape[1])
                e_sl = zoom(e_sl, fac, order=3)
            expert_maes.append(np.abs(e_sl - gt_sl)[tumor_mask].mean())

        pacs_mae = 0.0
        if blend_vol is not None:
            bz = min(z, blend_vol.shape[2] - 1)
            b_sl = blend_vol[:, :, bz]
            if b_sl.shape != gt_sl.shape:
                fac = (gt_sl.shape[0] / b_sl.shape[0], gt_sl.shape[1] / b_sl.shape[1])
                b_sl = zoom(b_sl, fac, order=3)
            pacs_mae = np.abs(b_sl - gt_sl)[tumor_mask].mean()

        score = float(np.mean(expert_maes)) - pacs_mae
        if score > best_score:
            best_score = score
            best_z = z
            best_bbox = _find_tumor_bbox(seg_sl)

    LOG.info(
        "Tumor-slice selection: z=%d, score=%.4f (expert_mae - pacs_mae)",
        best_z,
        best_score,
    )
    return best_z, best_bbox


# ---------------------------------------------------------------------------
# Brain crop
# ---------------------------------------------------------------------------


def _brain_crop_bbox(
    vol_slice: np.ndarray, pad: int = 5, extra_top_right: int = 0
) -> np.ndarray:
    """Tight bounding box around brain tissue with optional asymmetric padding.

    Args:
        vol_slice: 2D array (axial slice).
        pad: Symmetric padding in voxels.
        extra_top_right: Extra padding added to the top and right edges
            to create space for the zoom inset.

    Returns:
        Array [r0, r1, a0, a1].
    """
    mask = vol_slice > vol_slice.max() * 0.02
    if mask.sum() < 10:
        return np.array([0, vol_slice.shape[0], 0, vol_slice.shape[1]])
    rs, cs = np.where(mask)
    return np.array(
        [
            max(0, rs.min() - pad),
            min(vol_slice.shape[0], rs.max() + pad + 1 + extra_top_right),
            max(0, cs.min() - pad),
            min(vol_slice.shape[1], cs.max() + pad + 1 + extra_top_right),
        ]
    )


# ---------------------------------------------------------------------------
# Zoom inset
# ---------------------------------------------------------------------------


def _add_zoom_inset(
    ax: plt.Axes,
    img: np.ndarray,
    bbox: np.ndarray,
    cmap: str = "gray",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Add a yellow-bordered zoom inset to the top-right of an axes."""
    y0, y1, x0, x1 = bbox
    crop = img[y0:y1, x0:x1]

    inset = inset_axes(ax, width="28%", height="28%", loc="upper right", borderpad=0)
    inset.imshow(crop, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("#FFD700")
        spine.set_linewidth(0.8)

    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=0.8,
        edgecolor="#FFD700",
        facecolor="none",
    )
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Tumor contour drawing
# ---------------------------------------------------------------------------


def _draw_tumor_contours(
    ax: plt.Axes,
    seg_slice: np.ndarray,
    linewidth: float = 0.6,
) -> None:
    """Draw contour lines per segmentation label on *ax*.

    Each BraTS label gets a distinct colour. Only boundary pixels are drawn
    (no filled overlay), keeping the underlying image fully visible.
    """
    for label_val, (color, _name) in _TUMOR_LABEL_COLORS.items():
        binary = (seg_slice == label_val).astype(np.float32)
        if binary.sum() < 5:
            continue
        ax.contour(
            binary,
            levels=[0.5],
            colors=[color],
            linewidths=linewidth,
            origin="lower",
        )


# ---------------------------------------------------------------------------
# HDF5 / weight-map finders
# ---------------------------------------------------------------------------


def _find_blend_h5(
    patient_id: str,
    spacing: str,
    pulse: str,
    out_root: Path,
    experiment_name: str,
    n_folds: int,
) -> Tuple[Optional[Path], Optional[int]]:
    """Find which fold's results HDF5 contains a blend for a given patient."""
    key = blend_key(spacing, patient_id, pulse)
    for fold_num in range(1, n_folds + 1):
        h5_path = results_h5_path(out_root, experiment_name, fold_num)
        if h5_path.exists() and has_key(h5_path, key):
            return h5_path, fold_num
    return None, None


def _find_weight_npz(
    patient_id: str,
    spacing: str,
    pulse: str,
    out_root: Path,
    experiment_name: str,
    n_folds: int,
) -> Optional[Path]:
    """Find the weight NPZ file for a patient across folds."""
    for fold_num in range(1, n_folds + 1):
        for base in [out_root / experiment_name, out_root]:
            npz_path = (
                base
                / spacing
                / "model_data"
                / f"fold_{fold_num}"
                / pulse
                / f"{patient_id}_weights_test.npz"
            )
            if npz_path.exists():
                return npz_path
    return None


# ---------------------------------------------------------------------------
# Main grid generation
# ---------------------------------------------------------------------------


def generate_comparison_grid(
    source_h5: Path,
    experts_dir: Path,
    out_root: Path,
    experiment_name: str,
    n_folds: int,
    spacings: List[str],
    models: List[str],
    pulse: str,
    patient_id: str,
    out_path: Path,
    seg_path: Optional[Path] = None,
    showcase_weight_maps: bool = False,
    slice_range: Optional[Tuple[int, int]] = None,
    tumor_slices: bool = False,
    show_tumor_contours: bool = False,
    fixed_slice_z: Optional[int] = None,
) -> int:
    """Generate the comparison grid for one pulse sequence.

    Args:
        source_h5: Path to source_data.h5.
        experts_dir: Directory containing {model}.h5 files.
        out_root: PaCS-SR output root.
        experiment_name: Experiment name.
        n_folds: Number of folds.
        spacings: List of spacing strings.
        models: Expert model names (e.g., ["BSPLINE", "ECLARE"]).
        pulse: MRI pulse sequence (e.g., "t1c").
        patient_id: Patient identifier.
        out_path: Output file path.
        seg_path: Optional segmentation NIfTI for tumor bounding box.
        showcase_weight_maps: If True, add a Weights column for PaCS-SR.
        slice_range: Optional (lo, hi) to restrict axial slice search.
        tumor_slices: If True, pick the slice that maximises expert MAE and
            minimises PaCS-SR MAE over tumour pixels (requires seg_path).
        show_tumor_contours: If True, draw label-specific tumour contour
            lines on the SR images.
        fixed_slice_z: If given, skip slice selection and use this index.

    Returns:
        The axial slice index used (so callers can reuse it across pulses).
    """
    apply_ieee_style()
    display_names = list(models) + ["PaCS-SR"]
    n_rows = len(spacings)

    # ---- column layout (GridSpec) ----------------------------------------
    cols_per_model = []
    for name in display_names:
        if name == "PaCS-SR" and showcase_weight_maps:
            cols_per_model.append(3)
        else:
            cols_per_model.append(2)

    width_ratios: List[float] = []
    col_map: List[List[int]] = []
    gc = 0
    for i, ncols in enumerate(cols_per_model):
        if i > 0:
            width_ratios.append(0.15)
            gc += 1
        model_cols = []
        for _ in range(ncols):
            width_ratios.append(1.0)
            model_cols.append(gc)
            gc += 1
        col_map.append(model_cols)

    total_gcols = len(width_ratios)
    fig = plt.figure(figsize=(PLOT_SETTINGS["figure_width_double"], n_rows * 1.6))
    gs = GridSpec(
        n_rows,
        total_gcols,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.0,
        wspace=0.04,
    )

    # ---- load GT & reorient -----------------------------------------------
    gt_vol_raw, gt_affine = read_volume(source_h5, hr_key(patient_id, pulse))
    gt_vol, gt_aff_ras = _reorient_to_ras(gt_vol_raw, gt_affine)

    # ---- optional segmentation --------------------------------------------
    seg_ras: Optional[np.ndarray] = None
    if seg_path is not None and seg_path.exists():
        seg_nii = nib.load(str(seg_path))
        seg_ras = np.asarray(nib.as_closest_canonical(seg_nii).dataobj)

    # ---- slice & bbox selection -------------------------------------------
    if fixed_slice_z is not None:
        slice_z = fixed_slice_z
        if seg_ras is not None:
            bbox = _find_tumor_bbox(seg_ras[:, :, slice_z])
        else:
            _, bbox = _find_high_variance_slice_bbox(
                gt_vol, slice_range=(slice_z, slice_z + 1)
            )
    elif tumor_slices and seg_ras is not None:
        # Pre-load one spacing's expert + blend vols for scoring
        ref_spacing = spacings[len(spacings) // 2]  # middle spacing
        exp_vols_ras: Dict[str, np.ndarray] = {}
        for m in models:
            mh5 = expert_h5_path(experts_dir, m)
            ek = expert_key(ref_spacing, patient_id, pulse)
            if has_key(mh5, ek):
                v, a = read_volume(mh5, ek)
                exp_vols_ras[m], _ = _reorient_to_ras(v, a)
        blend_vol_ras: Optional[np.ndarray] = None
        bh5, _ = _find_blend_h5(
            patient_id,
            ref_spacing,
            pulse,
            out_root,
            experiment_name,
            n_folds,
        )
        if bh5 is not None:
            bv, ba = read_volume(bh5, blend_key(ref_spacing, patient_id, pulse))
            blend_vol_ras, _ = _reorient_to_ras(bv, ba)

        slice_z, bbox = _find_best_tumor_slice(
            seg_ras,
            gt_vol,
            exp_vols_ras,
            blend_vol_ras,
            slice_range=slice_range,
        )
    elif seg_ras is not None:
        # Use segmentation for bbox but standard variance for slice
        slice_z, bbox = _find_high_variance_slice_bbox(gt_vol, slice_range=slice_range)
        bbox = _find_tumor_bbox(seg_ras[:, :, slice_z])
    else:
        slice_z, bbox = _find_high_variance_slice_bbox(gt_vol, slice_range=slice_range)

    LOG.info("Selected slice z=%d for pulse=%s patient=%s", slice_z, pulse, patient_id)

    # ---- brain crop (with extra top-right padding for inset) --------------
    gt_slice_full = _axial_slice(gt_vol, slice_z)
    brain_bbox = _brain_crop_bbox(gt_slice_full, pad=8, extra_top_right=40)
    br0, br1, bc0, bc1 = brain_bbox

    def _crop2d(img: np.ndarray) -> np.ndarray:
        return img[br0:br1, bc0:bc1]

    gt_slice = _crop2d(gt_slice_full)

    # crop seg slice (if available)
    seg_slice_cropped: Optional[np.ndarray] = None
    if seg_ras is not None:
        seg_slice_cropped = _crop2d(seg_ras[:, :, slice_z])

    # zoom bbox → cropped coords
    bbox_c = np.array(
        [
            bbox[0] - br0,
            bbox[1] - br0,
            bbox[2] - bc0,
            bbox[3] - bc0,
        ]
    )
    bbox_c = np.clip(bbox_c, 0, [br1 - br0, br1 - br0, bc1 - bc0, bc1 - bc0])

    # ---- per-spacing row --------------------------------------------------
    for row_idx, spacing in enumerate(spacings):
        mae_maps: List[np.ndarray] = []
        sr_slices: List[np.ndarray] = []
        weight_slice: Optional[np.ndarray] = None

        for model_name in display_names:
            if model_name == "PaCS-SR":
                blend_h5, fold_num = _find_blend_h5(
                    patient_id,
                    spacing,
                    pulse,
                    out_root,
                    experiment_name,
                    n_folds,
                )
                if blend_h5 is not None:
                    sr_vol_raw, sr_aff = read_volume(
                        blend_h5,
                        blend_key(spacing, patient_id, pulse),
                    )
                    sr_vol, _ = _reorient_to_ras(sr_vol_raw, sr_aff)
                    sz = min(slice_z, sr_vol.shape[2] - 1)
                    sr_slice = _crop2d(_axial_slice(sr_vol, sz))
                else:
                    LOG.warning(
                        "Missing blend for %s %s %s", patient_id, spacing, pulse
                    )
                    sr_slice = np.zeros_like(gt_slice)

                if showcase_weight_maps:
                    npz_path = _find_weight_npz(
                        patient_id,
                        spacing,
                        pulse,
                        out_root,
                        experiment_name,
                        n_folds,
                    )
                    if npz_path is not None:
                        npz = np.load(npz_path)
                        wm = npz["weight_maps"]
                        model_names_npz = list(npz["model_names"])
                        wm_ras_chs = []
                        for ch in range(wm.shape[-1]):
                            ch_ras, _ = _reorient_to_ras(
                                wm[..., ch],
                                sr_aff if blend_h5 else gt_aff_ras,
                            )
                            wm_ras_chs.append(ch_ras)
                        wm_ras = np.stack(wm_ras_chs, axis=-1)
                        wz = min(slice_z, wm_ras.shape[2] - 1)
                        weight_slice = _crop2d(wm_ras[:, :, wz, :])
                        LOG.info(
                            "Weight map from %s (models: %s)", npz_path, model_names_npz
                        )
                    else:
                        LOG.warning(
                            "No weight NPZ for %s %s %s", patient_id, spacing, pulse
                        )
            else:
                model_h5 = expert_h5_path(experts_dir, model_name)
                exp_k = expert_key(spacing, patient_id, pulse)
                if has_key(model_h5, exp_k):
                    sr_vol_raw, sr_aff = read_volume(model_h5, exp_k)
                    sr_vol, _ = _reorient_to_ras(sr_vol_raw, sr_aff)
                    sz = min(slice_z, sr_vol.shape[2] - 1)
                    sr_slice = _crop2d(_axial_slice(sr_vol, sz))
                else:
                    LOG.warning("Missing: %s in %s", exp_k, model_h5)
                    sr_slice = np.zeros_like(gt_slice)

            if sr_slice.shape != gt_slice.shape:
                fac = (
                    gt_slice.shape[0] / sr_slice.shape[0],
                    gt_slice.shape[1] / sr_slice.shape[1],
                )
                sr_slice = zoom(sr_slice, fac, order=3)

            sr_slices.append(normalize01(sr_slice))
            mae_maps.append(np.abs(sr_slice - gt_slice))

        row_vmax = max(m.max() for m in mae_maps) if mae_maps else 1.0
        if row_vmax < 1e-8:
            row_vmax = 1.0

        # ---- plot each model group ----------------------------------------
        for col_idx, (model_name, sr_norm, mae_map) in enumerate(
            zip(display_names, sr_slices, mae_maps)
        ):
            gcols = col_map[col_idx]
            ax_sr = fig.add_subplot(gs[row_idx, gcols[0]])
            ax_mae = fig.add_subplot(gs[row_idx, gcols[1]])

            ax_sr.imshow(
                sr_norm, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal"
            )
            _add_zoom_inset(ax_sr, sr_norm, bbox_c, cmap="gray", vmin=0, vmax=1)
            if show_tumor_contours and seg_slice_cropped is not None:
                _draw_tumor_contours(ax_sr, seg_slice_cropped)
            ax_sr.set_xticks([])
            ax_sr.set_yticks([])

            ax_mae.imshow(
                mae_map,
                cmap="inferno",
                vmin=0,
                vmax=row_vmax,
                origin="lower",
                aspect="equal",
            )
            _add_zoom_inset(
                ax_mae, mae_map, bbox_c, cmap="inferno", vmin=0, vmax=row_vmax
            )
            ax_mae.set_xticks([])
            ax_mae.set_yticks([])

            # Weight map column
            if model_name == "PaCS-SR" and showcase_weight_maps and len(gcols) == 3:
                ax_wm = fig.add_subplot(gs[row_idx, gcols[2]])
                if weight_slice is not None:
                    eclare_idx = min(1, weight_slice.shape[-1] - 1)
                    wm_disp = weight_slice[:, :, eclare_idx]
                    ax_wm.imshow(
                        wm_disp,
                        cmap="RdYlBu_r",
                        vmin=0,
                        vmax=1,
                        origin="lower",
                        aspect="equal",
                    )
                    _add_zoom_inset(
                        ax_wm, wm_disp, bbox_c, cmap="RdYlBu_r", vmin=0, vmax=1
                    )
                    if show_tumor_contours and seg_slice_cropped is not None:
                        _draw_tumor_contours(ax_wm, seg_slice_cropped)
                else:
                    ax_wm.imshow(
                        np.zeros_like(gt_slice),
                        cmap="RdYlBu_r",
                        vmin=0,
                        vmax=1,
                        origin="lower",
                        aspect="equal",
                    )
                ax_wm.set_xticks([])
                ax_wm.set_yticks([])
                if row_idx == 0:
                    ax_wm.set_title(
                        "Weights", fontsize=PLOT_SETTINGS["annotation_fontsize"]
                    )

            if row_idx == 0:
                ax_sr.set_title(
                    f"{model_name}", fontsize=PLOT_SETTINGS["tick_labelsize"]
                )
                ax_mae.set_title("MAE", fontsize=PLOT_SETTINGS["annotation_fontsize"])
            if col_idx == 0:
                ax_sr.set_ylabel(
                    spacing,
                    fontsize=PLOT_SETTINGS["font_size"],
                    rotation=0,
                    labelpad=30,
                    va="center",
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        save_path = out_path.with_suffix(f".{ext}")
        fig.savefig(save_path, dpi=PLOT_SETTINGS["dpi_print"], bbox_inches="tight")
        LOG.info("Saved %s", save_path)
    plt.close(fig)
    return slice_z


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SERAM Figure 1: Comparison grid")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Patient ID (auto-selects if omitted)",
    )
    parser.add_argument(
        "--seg-dir",
        type=Path,
        default=None,
        help="Directory with {patient}/{patient}-seg.nii.gz",
    )
    parser.add_argument(
        "--showcase-weight-maps",
        action="store_true",
        default=False,
        help="Add a Weights column next to PaCS-SR MAE",
    )
    parser.add_argument(
        "--slice-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Restrict axial slice search to [LO, HI)",
    )
    parser.add_argument(
        "--tumor-slices",
        action="store_true",
        default=False,
        help="Pick slice maximising expert MAE / minimising PaCS-SR MAE over tumour pixels (needs --seg-dir)",
    )
    parser.add_argument(
        "--show-tumor-contours",
        action="store_true",
        default=False,
        help="Draw tumour label contour lines on the SR images",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(name)s] %(levelname)s | %(message)s"
    )

    from pacs_sr.config.config import load_full_config

    full = load_full_config(args.config)
    data = full.data
    pacs_sr = full.pacs_sr

    out_dir = Path(pacs_sr.out_root) / pacs_sr.experiment_name / "figures"
    n_folds = data.kfolds

    patient_id = args.patient_id
    if patient_id is None:
        patients = list_groups(data.source_h5, "high_resolution")
        if patients:
            patient_id = patients[0]
    if patient_id is None:
        LOG.error("No patients found")
        return

    seg_path = None
    if args.seg_dir is not None:
        seg_path = args.seg_dir / patient_id / f"{patient_id}-seg.nii.gz"

    shared_slice_z: Optional[int] = None
    for pulse in pacs_sr.pulses:
        used_z = generate_comparison_grid(
            source_h5=data.source_h5,
            experts_dir=data.experts_dir,
            out_root=Path(pacs_sr.out_root),
            experiment_name=pacs_sr.experiment_name,
            n_folds=n_folds,
            spacings=list(pacs_sr.spacings),
            models=list(pacs_sr.models),
            pulse=pulse,
            patient_id=patient_id,
            out_path=out_dir / f"figure1_{pulse}.pdf",
            seg_path=seg_path,
            showcase_weight_maps=args.showcase_weight_maps,
            slice_range=tuple(args.slice_range) if args.slice_range else None,
            tumor_slices=args.tumor_slices,
            show_tumor_contours=args.show_tumor_contours,
            fixed_slice_z=shared_slice_z,
        )
        if shared_slice_z is None:
            shared_slice_z = used_z
            LOG.info("Shared slice z=%d (from first pulse %s)", used_z, pulse)


if __name__ == "__main__":
    main()
