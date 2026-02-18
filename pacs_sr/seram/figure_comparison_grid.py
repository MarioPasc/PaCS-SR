#!/usr/bin/env python3
"""SERAM Figure 1: Per-sequence comparison grid.

Generates a 3x3 grid (spacings x models) showing axial slices + MAE maps
with zoomed tumor insets for each MRI sequence.

Layout per sequence:
          BSPLINE          ECLARE           PaCS-SR
         SR | MAE         SR | MAE         SR | MAE
3mm   [img][err]      [img][err]      [img][err]
5mm   [img][err]      [img][err]      [img][err]
7mm   [img][err]      [img][err]      [img][err]

Reads all volumes from HDF5 files:
  - GT from source_data.h5
  - Expert SR from {model}.h5
  - PaCS-SR blends from results_fold_{N}.h5

Usage:
    python -m pacs_sr.seram.figure_comparison_grid --config configs/seram_glioma.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
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

LOG = logging.getLogger(__name__)


def _find_tumor_slice(seg: np.ndarray) -> Tuple[int, np.ndarray]:
    """Find the axial slice with the largest tumor area.

    Args:
        seg: 3D segmentation volume (Z, Y, X) with nonzero = tumor.

    Returns:
        (slice_index, 2D bounding box as (y0, y1, x0, x1)).
    """
    tumor_area = np.array([np.count_nonzero(seg[z]) for z in range(seg.shape[0])])
    best_z = int(np.argmax(tumor_area))

    seg_slice = seg[best_z] > 0
    ys, xs = np.where(seg_slice)
    if len(ys) == 0:
        cy, cx = seg.shape[1] // 2, seg.shape[2] // 2
        margin = 30
        bbox = np.array([cy - margin, cy + margin, cx - margin, cx + margin])
    else:
        margin = 10
        bbox = np.array(
            [
                max(0, ys.min() - margin),
                min(seg.shape[1], ys.max() + margin),
                max(0, xs.min() - margin),
                min(seg.shape[2], xs.max() + margin),
            ]
        )
    return best_z, bbox


def _find_mid_slice_bbox(vol: np.ndarray) -> Tuple[int, np.ndarray]:
    """Fallback when no segmentation: use middle slice and center crop."""
    z = vol.shape[0] // 2
    cy, cx = vol.shape[1] // 2, vol.shape[2] // 2
    margin = 30
    bbox = np.array(
        [
            max(0, cy - margin),
            min(vol.shape[1], cy + margin),
            max(0, cx - margin),
            min(vol.shape[2], cx + margin),
        ]
    )
    return z, bbox


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

    inset = inset_axes(ax, width="35%", height="35%", loc="upper right")
    inset.imshow(crop, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("#FFD700")
        spine.set_linewidth(2)

    # Draw rectangle on main image
    rect = Rectangle(
        (x0, y0), x1 - x0, y1 - y0, linewidth=1.5, edgecolor="#FFD700", facecolor="none"
    )
    ax.add_patch(rect)


def _find_blend_h5(
    patient_id: str,
    spacing: str,
    pulse: str,
    out_root: Path,
    experiment_name: str,
    n_folds: int,
) -> Optional[Path]:
    """Find which fold's results HDF5 contains a blend for a given patient.

    Args:
        patient_id: Patient ID.
        spacing: Spacing string.
        pulse: Pulse sequence.
        out_root: Output root directory.
        experiment_name: Experiment name.
        n_folds: Number of folds to search.

    Returns:
        Path to the results HDF5 file, or None.
    """
    key = blend_key(spacing, patient_id, pulse)
    for fold_num in range(1, n_folds + 1):
        h5_path = results_h5_path(out_root, experiment_name, fold_num)
        if h5_path.exists() and has_key(h5_path, key):
            return h5_path
    return None


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
) -> None:
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
    """
    apply_ieee_style()
    display_names = list(models) + ["PaCS-SR"]
    n_rows = len(spacings)
    n_cols = len(display_names) * 2  # SR + MAE per model

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(PLOT_SETTINGS["figure_width_double"], n_rows * 1.8),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Load GT from HDF5
    gt_vol, _ = read_volume(source_h5, hr_key(patient_id, pulse))

    if seg_path is not None and seg_path.exists():
        seg_vol = nib.load(str(seg_path)).get_fdata()
        slice_z, bbox = _find_tumor_slice(seg_vol)
    else:
        slice_z, bbox = _find_mid_slice_bbox(gt_vol)

    gt_slice = gt_vol[slice_z]
    gt_norm = normalize01(gt_slice)

    for row_idx, spacing in enumerate(spacings):
        mae_maps = []
        sr_slices = []

        for model_name in display_names:
            if model_name == "PaCS-SR":
                # Load blend from results HDF5
                blend_h5 = _find_blend_h5(
                    patient_id, spacing, pulse, out_root, experiment_name, n_folds
                )
                if blend_h5 is not None:
                    sr_vol, _ = read_volume(
                        blend_h5, blend_key(spacing, patient_id, pulse)
                    )
                    sr_slice = sr_vol[slice_z]
                else:
                    LOG.warning(
                        "Missing blend for %s %s %s", patient_id, spacing, pulse
                    )
                    sr_slice = np.zeros_like(gt_slice)
            else:
                # Load expert from {model}.h5
                model_h5 = expert_h5_path(experts_dir, model_name)
                exp_k = expert_key(spacing, patient_id, pulse)
                if has_key(model_h5, exp_k):
                    sr_vol, _ = read_volume(model_h5, exp_k)
                    sr_slice = sr_vol[slice_z]
                else:
                    LOG.warning("Missing: %s in %s", exp_k, model_h5)
                    sr_slice = np.zeros_like(gt_slice)

            sr_slices.append(normalize01(sr_slice))
            mae_maps.append(np.abs(sr_slice - gt_slice))

        # Consistent MAE vmax across the row
        row_vmax = max(m.max() for m in mae_maps) if mae_maps else 1.0
        if row_vmax < 1e-8:
            row_vmax = 1.0

        for col_idx, (model_name, sr_norm, mae_map) in enumerate(
            zip(display_names, sr_slices, mae_maps)
        ):
            ax_sr = axes[row_idx, col_idx * 2]
            ax_mae = axes[row_idx, col_idx * 2 + 1]

            # SR image
            ax_sr.imshow(
                sr_norm, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal"
            )
            _add_zoom_inset(ax_sr, sr_norm, bbox, cmap="gray", vmin=0, vmax=1)
            ax_sr.set_xticks([])
            ax_sr.set_yticks([])

            # MAE map
            ax_mae.imshow(
                mae_map,
                cmap="inferno",
                vmin=0,
                vmax=row_vmax,
                origin="lower",
                aspect="equal",
            )
            _add_zoom_inset(
                ax_mae, mae_map, bbox, cmap="inferno", vmin=0, vmax=row_vmax
            )
            ax_mae.set_xticks([])
            ax_mae.set_yticks([])

            # Column titles (top row only)
            if row_idx == 0:
                ax_sr.set_title(
                    f"{model_name}", fontsize=PLOT_SETTINGS["tick_labelsize"]
                )
                ax_mae.set_title("MAE", fontsize=PLOT_SETTINGS["annotation_fontsize"])

            # Row labels (left column only)
            if col_idx == 0:
                ax_sr.set_ylabel(
                    spacing,
                    fontsize=PLOT_SETTINGS["font_size"],
                    rotation=0,
                    labelpad=30,
                    va="center",
                )

    fig.suptitle(
        f"{pulse.upper()} — Patient {patient_id}",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        y=1.02,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        save_path = out_path.with_suffix(f".{ext}")
        fig.savefig(save_path, dpi=PLOT_SETTINGS["dpi_print"])
        LOG.info("Saved %s", save_path)
    plt.close(fig)


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
        "--seg-dir", type=Path, default=None, help="Segmentation directory"
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

    for pulse in pacs_sr.pulses:
        # Auto-select patient: pick first one from HR group in source_data.h5
        patient_id = args.patient_id
        if patient_id is None:
            patients = list_groups(data.source_h5, "high_resolution")
            if patients:
                patient_id = patients[0]
            if patient_id is None:
                LOG.warning("No patients found; skipping %s", pulse)
                continue

        seg_path = None
        if args.seg_dir is not None:
            seg_path = args.seg_dir / patient_id / f"{patient_id}-seg.nii.gz"

        generate_comparison_grid(
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
        )


if __name__ == "__main__":
    main()
