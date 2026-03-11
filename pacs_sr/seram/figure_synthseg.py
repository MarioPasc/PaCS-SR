#!/usr/bin/env python3
"""SynthSeg evaluation figures for PaCS-SR paper.

Figure 1 (QC Scatter): Per-tissue QC scores plotted against HR reference,
    with per-method regression lines showing agreement with ground truth.
    Includes HR label map insets per subplot for anatomical context.

Figure 2 (Segmentation Grid): Qualitative comparison of SynthSeg label maps
    across methods and spacings, with difference maps against HR.

Usage:
    python -m pacs_sr.seram.figure_synthseg --config configs/seram_glioma_local.yaml --figure all
    python -m pacs_sr.seram.figure_synthseg --config configs/seram_glioma_local.yaml --figure qc-scatter
    python -m pacs_sr.seram.figure_synthseg --config configs/seram_glioma_local.yaml \\
        --figure seg-grid --plane axial --patient-id BraTS-GLI-00547-100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats as sp_stats

from pacs_sr.experiments.synthseg_evaluation import (
    _find_label_file,
    load_synthseg_eval_config,
)
from pacs_sr.seram.figure_comparison_grid import (
    _brain_crop_bbox,
    _reorient_to_ras,
)
from pacs_sr.utils.settings import (
    PAUL_TOL_MUTED,
    PLOT_SETTINGS,
    SERAM_MODEL_COLORS,
    apply_ieee_style,
)

LOG = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

MODEL_ORDER: List[str] = ["BSPLINE", "ECLARE", "PaCS-SR"]

# Distinct marker shapes per method for scatter plots
MODEL_MARKERS: Dict[str, str] = {
    "BSPLINE": "s",  # square
    "ECLARE": "^",  # triangle up
    "PaCS-SR": "o",  # circle
}

# Internal method keys (used in file/directory names)
_METHOD_KEYS: Dict[str, str] = {
    "BSPLINE": "BSPLINE",
    "ECLARE": "ECLARE",
    "PaCS-SR": "PACS_SR",
}

TISSUE_COLUMNS: List[str] = [
    "general white matter",
    "general grey matter",
    "general csf",
    "cerebellum",
    "brainstem",
    "thalamus",
    "putamen+pallidum",
    "hippocampus+amygdala",
]

# Display names with capitalised first letters
TISSUE_DISPLAY_NAMES: Dict[str, str] = {
    "general white matter": "General White Matter",
    "general grey matter": "General Grey Matter",
    "general csf": "General CSF",
    "cerebellum": "Cerebellum",
    "brainstem": "Brainstem",
    "thalamus": "Thalamus",
    "putamen+pallidum": "Putamen+Pallidum",
    "hippocampus+amygdala": "Hippocampus+Amygdala",
}

TISSUE_COLORS: Dict[str, str] = {
    "general white matter": PAUL_TOL_MUTED[0],  # #CC6677 rose
    "general grey matter": PAUL_TOL_MUTED[1],  # #332288 indigo
    "general csf": PAUL_TOL_MUTED[2],  # #DDCC77 sand
    "cerebellum": PAUL_TOL_MUTED[3],  # #117733 green
    "brainstem": PAUL_TOL_MUTED[4],  # #88CCEE cyan
    "thalamus": PAUL_TOL_MUTED[5],  # #882255 wine
    "putamen+pallidum": PAUL_TOL_MUTED[6],  # #44AA99 teal
    "hippocampus+amygdala": PAUL_TOL_MUTED[7],  # #999933 olive
}

# FreeSurfer LUT: label_id -> (R, G, B) [0-255]
# Standard FreeSurfer colors for the 34 SynthSeg labels.
FREESURFER_LUT: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),  # background
    2: (245, 245, 245),  # Left Cerebral WM
    3: (205, 62, 78),  # Left Cerebral Cortex
    4: (120, 18, 134),  # Left Lateral Ventricle
    5: (196, 58, 250),  # Left Inf Lateral Ventricle
    7: (220, 248, 164),  # Left Cerebellum WM
    8: (230, 148, 34),  # Left Cerebellum Cortex
    10: (0, 118, 14),  # Left Thalamus
    11: (122, 186, 220),  # Left Caudate
    12: (236, 13, 176),  # Left Putamen
    13: (12, 48, 255),  # Left Pallidum
    14: (204, 182, 142),  # 3rd Ventricle
    15: (42, 204, 164),  # 4th Ventricle
    16: (119, 159, 176),  # Brain Stem
    17: (220, 216, 20),  # Left Hippocampus
    18: (103, 255, 255),  # Left Amygdala
    24: (60, 60, 60),  # CSF
    26: (255, 165, 0),  # Left Accumbens
    28: (165, 42, 42),  # Left VentralDC
    41: (245, 245, 245),  # Right Cerebral WM
    42: (205, 62, 78),  # Right Cerebral Cortex
    43: (120, 18, 134),  # Right Lateral Ventricle
    44: (196, 58, 250),  # Right Inf Lateral Ventricle
    46: (220, 248, 164),  # Right Cerebellum WM
    47: (230, 148, 34),  # Right Cerebellum Cortex
    49: (0, 118, 14),  # Right Thalamus
    50: (122, 186, 220),  # Right Caudate
    51: (236, 13, 176),  # Right Putamen
    52: (12, 48, 255),  # Right Pallidum
    53: (220, 216, 20),  # Right Hippocampus
    54: (103, 255, 255),  # Right Amygdala
    58: (255, 165, 0),  # Right Accumbens
    60: (165, 42, 42),  # Right VentralDC
}

# Mapping from spacing to which plane the QC scatter inset should show
# (axial for 3mm, sagittal for 5mm, coronal for 7mm)
_QC_INSET_PLANES: Dict[str, str] = {
    "3mm": "axial",
    "5mm": "sagittal",
    "7mm": "coronal",
}

# Spacing display labels
_SPACING_DISPLAY: Dict[str, str] = {
    "3mm": "3 mm",
    "5mm": "5 mm",
    "7mm": "7 mm",
}


# ============================================================================
# Data loading
# ============================================================================


def _load_qc_tissue_data(
    qc_dir: Path,
    methods: List[str],
    spacings: List[str],
) -> pd.DataFrame:
    """Load per-tissue QC scores into a tidy DataFrame.

    Reads individual QC CSVs from ``qc_dir`` and the HR reference CSV.
    Returns a DataFrame with columns:
    ``patient_id, method, spacing, tissue, qc_score, hr_qc_score``.

    Args:
        qc_dir: Directory containing ``{METHOD}_{SPACING}_qc.csv`` files
            and ``HR_qc.csv``.
        methods: SR method display names to load.
        spacings: Spacing strings (e.g. ``["3mm", "5mm", "7mm"]``).

    Returns:
        Tidy DataFrame with one row per (patient, method, spacing, tissue).
    """
    hr_csv = qc_dir / "HR_qc.csv"
    hr_df = pd.read_csv(hr_csv)
    hr_long = hr_df.melt(
        id_vars="subject",
        value_vars=TISSUE_COLUMNS,
        var_name="tissue",
        value_name="hr_qc_score",
    )
    hr_long = hr_long.rename(columns={"subject": "patient_id"})

    rows = []
    for method in methods:
        method_key = _METHOD_KEYS[method]
        for spacing in spacings:
            csv_path = qc_dir / f"{method_key}_{spacing}_qc.csv"
            if not csv_path.exists():
                LOG.warning("Missing QC CSV: %s", csv_path)
                continue
            df = pd.read_csv(csv_path)
            long = df.melt(
                id_vars="subject",
                value_vars=TISSUE_COLUMNS,
                var_name="tissue",
                value_name="qc_score",
            )
            long = long.rename(columns={"subject": "patient_id"})
            long["method"] = method
            long["spacing"] = spacing
            rows.append(long)

    method_df = pd.concat(rows, ignore_index=True)
    merged = method_df.merge(hr_long, on=["patient_id", "tissue"], how="left")
    return merged


def _find_highest_qc_patient(qc_dir: Path) -> str:
    """Find the HR patient with highest mean QC score across tissues.

    Args:
        qc_dir: Directory containing ``HR_qc.csv``.

    Returns:
        Patient ID (subject column value) with best QC.
    """
    hr_csv = qc_dir / "HR_qc.csv"
    hr_df = pd.read_csv(hr_csv)
    hr_df["mean_qc"] = hr_df[TISSUE_COLUMNS].mean(axis=1)
    best_row = hr_df.loc[hr_df["mean_qc"].idxmax()]
    return str(best_row["subject"])


# ============================================================================
# Label map rendering helpers (shared by both figures)
# ============================================================================


def _label_slice_to_rgb(
    label_slice: np.ndarray,
    lut: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    """Convert a 2D integer label map to an RGB image.

    Args:
        label_slice: 2D array of integer labels.
        lut: Label -> (R, G, B) mapping.

    Returns:
        RGB array of shape (H, W, 3) with values in [0, 1].
    """
    h, w = label_slice.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for lab_id, color in lut.items():
        mask = label_slice == lab_id
        if mask.any():
            rgb[mask] = [c / 255.0 for c in color]
    return rgb


def _extract_slice(
    vol: np.ndarray,
    plane: str,
    idx: int,
) -> np.ndarray:
    """Extract a 2D slice from a 3D RAS volume, oriented for display.

    In RAS: dim0=R (left-right), dim1=A (post-ant), dim2=S (inf-sup).
    For display with ``origin="lower"``, the superior-inferior axis must
    be vertical (rows). Coronal and sagittal slices are transposed so
    that S runs along rows.

    Args:
        vol: 3D volume in RAS orientation.
        plane: One of "axial", "coronal", "sagittal".
        idx: Slice index.

    Returns:
        2D array oriented for display.
    """
    if plane == "axial":
        # vol[:, :, z] → rows=R, cols=A — standard orientation
        return vol[:, :, idx]
    elif plane == "coronal":
        # vol[:, y, :] → rows=R, cols=S — transpose so S is vertical
        return vol[:, idx, :].T
    elif plane == "sagittal":
        # vol[x, :, :] → rows=A, cols=S — transpose so S is vertical
        return vol[idx, :, :].T
    else:
        raise ValueError(f"Unknown plane: {plane}")


def _mid_brain_index(vol: np.ndarray, plane: str) -> int:
    """Find the mid-brain slice index for a given plane.

    Args:
        vol: 3D volume (label map or intensity).
        plane: One of "axial", "coronal", "sagittal".

    Returns:
        Slice index at the center of the brain.
    """
    axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    axis = axis_map[plane]
    nonzero = vol > 0
    proj = nonzero.any(axis=tuple(i for i in range(3) if i != axis))
    indices = np.where(proj)[0]
    if len(indices) == 0:
        return vol.shape[axis] // 2
    return int(indices[len(indices) // 2])


def _crop_to_common_shape(
    volumes: List[np.ndarray],
) -> List[np.ndarray]:
    """Crop all volumes to the common minimum shape.

    Args:
        volumes: List of 3D arrays.

    Returns:
        List of cropped arrays with identical shapes.
    """
    min_shape = tuple(min(v.shape[ax] for v in volumes) for ax in range(3))
    return [v[: min_shape[0], : min_shape[1], : min_shape[2]] for v in volumes]


def _render_hr_label_inset(
    labels_dir: Path,
    patient_subject: str,
    plane: str,
    pulse: str = "t1c",
) -> Optional[np.ndarray]:
    """Render an HR label map slice as an RGB image for inset embedding.

    Args:
        labels_dir: Root labels directory containing ``HR/`` subdir.
        patient_subject: Subject string from QC CSV (e.g. ``"BraTS-...-t1c"``).
        plane: Anatomical plane.
        pulse: Pulse sequence.

    Returns:
        Cropped RGB image (H, W, 3) in [0, 1], or None on failure.
    """
    # Extract patient_id from subject string (remove -pulse suffix)
    patient_id = patient_subject.rsplit(f"-{pulse}", 1)[0]
    hr_dir = labels_dir / "HR"
    lbl_path = _find_label_file(hr_dir, patient_id, pulse)
    if lbl_path is None:
        LOG.warning("HR label file not found for inset: %s", patient_id)
        return None

    nii = nib.load(str(lbl_path))
    vol_raw = np.asarray(nii.dataobj, dtype=np.int32)
    vol_ras, _ = _reorient_to_ras(vol_raw.astype(np.float32), nii.affine)
    vol_ras = np.round(vol_ras).astype(np.int32)

    si = _mid_brain_index(vol_ras, plane)
    sl = _extract_slice(vol_ras, plane, si)

    # Tight crop around brain
    bbox = _brain_crop_bbox((sl > 0).astype(np.float32), pad=2)
    r0, r1, c0, c1 = bbox
    sl_crop = sl[r0:r1, c0:c1]
    return _label_slice_to_rgb(sl_crop, FREESURFER_LUT)


# ============================================================================
# Figure 1: QC Scatter
# ============================================================================


def generate_qc_scatter(
    qc_dir: Path,
    spacings: List[str],
    out_path: Path,
    labels_dir: Optional[Path] = None,
) -> None:
    """Generate QC scatter figure: method QC vs HR QC per tissue.

    Layout: 1x3 subplots (one per spacing). Each subplot shows per-tissue
    scatter points (muted) and per-patient mean points (strong, distinct
    markers per method) with linear regression lines and R^2 annotations.
    An HR label map inset is embedded in the top-left of each subplot.

    Args:
        qc_dir: Directory containing QC CSVs.
        spacings: List of spacing strings.
        out_path: Output file path (saved as .pdf and .png).
        labels_dir: Root labels directory for HR inset label maps.
            If None, insets are skipped.
    """
    apply_ieee_style()
    data = _load_qc_tissue_data(qc_dir, MODEL_ORDER, spacings)

    fig, axes = plt.subplots(
        1,
        len(spacings),
        figsize=(PLOT_SETTINGS["figure_width_double"], 3.5),
        sharey=True,
        sharex=True,
    )
    if len(spacings) == 1:
        axes = [axes]

    # Find best HR patient for inset label maps
    inset_patient = None
    if labels_dir is not None:
        inset_patient = _find_highest_qc_patient(qc_dir)
        LOG.info("HR inset patient: %s", inset_patient)

    for ax_idx, (ax, spacing) in enumerate(zip(axes, spacings)):
        sp_data = data[data["spacing"] == spacing]

        # x=y identity line
        ax.plot([0.5, 1.0], [0.5, 1.0], ls="--", color="0.4", lw=0.8, zorder=0)

        for method in MODEL_ORDER:
            m_data = sp_data[sp_data["method"] == method]
            if m_data.empty:
                continue

            method_color = SERAM_MODEL_COLORS.get(method, "#333333")
            method_marker = MODEL_MARKERS.get(method, "o")

            # Per-tissue scatter (muted, same marker as method for consistency)
            for tissue in TISSUE_COLUMNS:
                t_data = m_data[m_data["tissue"] == tissue]
                ax.scatter(
                    t_data["hr_qc_score"],
                    t_data["qc_score"],
                    color=TISSUE_COLORS[tissue],
                    marker=method_marker,
                    alpha=0.35,
                    s=3,
                    zorder=1,
                    rasterized=True,
                )

            # Per-patient mean (strong color, distinct marker)
            patient_means = (
                m_data.groupby("patient_id")
                .agg(
                    hr_mean=("hr_qc_score", "mean"),
                    method_mean=("qc_score", "mean"),
                )
                .reset_index()
            )

            ax.scatter(
                patient_means["hr_mean"],
                patient_means["method_mean"],
                color=method_color,
                marker=method_marker,
                alpha=1.0,
                s=12,
                zorder=3,
                edgecolors="white",
                linewidths=0.3,
            )

            # Linear regression on means -> annotate R^2
            if len(patient_means) >= 3:
                slope, intercept, r_val, _, _ = sp_stats.linregress(
                    patient_means["hr_mean"], patient_means["method_mean"]
                )
                x_fit = np.linspace(
                    patient_means["hr_mean"].min(),
                    patient_means["hr_mean"].max(),
                    50,
                )
                ax.plot(
                    x_fit,
                    slope * x_fit + intercept,
                    color=method_color,
                    lw=1.0,
                    zorder=2,
                )
                r_squared = r_val**2
                ax.annotate(
                    f"$R^2$={r_squared:.3f}",
                    xy=(0.98, 0.02 + MODEL_ORDER.index(method) * 0.08),
                    xycoords="axes fraction",
                    ha="right",
                    va="bottom",
                    fontsize=PLOT_SETTINGS["annotation_fontsize"],
                    color=method_color,
                )

        # HR label map inset (top-left)
        if inset_patient is not None and labels_dir is not None:
            plane = _QC_INSET_PLANES.get(spacing, "axial")
            inset_rgb = _render_hr_label_inset(labels_dir, inset_patient, plane)
            if inset_rgb is not None:
                ax_inset = inset_axes(
                    ax,
                    width="30%",
                    height="30%",
                    loc="upper left",
                    borderpad=0.3,
                )
                ax_inset.imshow(
                    inset_rgb,
                    origin="lower",
                    aspect="equal",
                    interpolation="nearest",
                )
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                for spine in ax_inset.spines.values():
                    spine.set_edgecolor("0.5")
                    spine.set_linewidth(0.5)

        ax.set_title(
            _SPACING_DISPLAY.get(spacing, spacing),
            fontsize=PLOT_SETTINGS["axes_titlesize"],
        )
        ax.set_xlabel("HR QC Score", fontsize=PLOT_SETTINGS["axes_labelsize"])
        if ax_idx == 0:
            ax.set_ylabel("Method QC Score", fontsize=PLOT_SETTINGS["axes_labelsize"])

    # Method legend (top-center, with distinct markers)
    method_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[m],
            color="w",
            markerfacecolor=SERAM_MODEL_COLORS[m],
            markeredgecolor="white",
            markeredgewidth=0.3,
            markersize=5,
            label=m,
        )
        for m in MODEL_ORDER
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        ncol=len(MODEL_ORDER),
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.5,
        handletextpad=0.4,
    )

    # Tissue legend (bottom-center, capitalised names)
    tissue_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=TISSUE_COLORS[t],
            markersize=3,
            alpha=0.7,
            label=TISSUE_DISPLAY_NAMES[t],
        )
        for t in TISSUE_COLUMNS
    ]
    fig.legend(
        handles=tissue_handles,
        loc="lower center",
        ncol=4,
        fontsize=PLOT_SETTINGS["legend_fontsize"] - 1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        columnspacing=0.8,
        handletextpad=0.3,
    )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88, wspace=0.08)
    _save_figure(fig, out_path)


# ============================================================================
# Figure 2: Segmentation Grid helpers
# ============================================================================


def _diff_map_rgb(
    method_labels: np.ndarray,
    hr_labels: np.ndarray,
    lut: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    """Build an RGB difference map between method and HR labels.

    Agreement voxels -> light grey. Disagreement -> HR label color.
    Background stays black.

    Args:
        method_labels: 2D integer label array from the SR method.
        hr_labels: 2D integer label array from HR ground truth.
        lut: Label -> (R, G, B) mapping.

    Returns:
        RGB array of shape (H, W, 3) with values in [0, 1].
    """
    h, w = method_labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    bg_mask = (method_labels == 0) & (hr_labels == 0)
    agree_mask = (method_labels == hr_labels) & ~bg_mask
    rgb[agree_mask] = 0.85

    disagree_mask = (method_labels != hr_labels) & ~bg_mask
    for lab_id, color in lut.items():
        lab_mask = disagree_mask & (hr_labels == lab_id)
        if lab_mask.any():
            rgb[lab_mask] = [c / 255.0 for c in color]

    return rgb


# ============================================================================
# Patient selection
# ============================================================================


def _available_label_patients(labels_dir: Path) -> set:
    """Return patient IDs that have HR label maps on disk."""
    hr_labels_dir = labels_dir / "HR"
    pids: set = set()
    if hr_labels_dir.exists():
        for f in hr_labels_dir.iterdir():
            if f.suffix == ".gz":
                pid = f.name.split("-t1c")[0]
                if pid:
                    pids.add(pid)
    return pids


def select_showcase_patient(
    raw_csv: Path,
    labels_dir: Path,
    methods: List[str],
    spacings: List[str],
    mode: str = "representative",
) -> str:
    """Select a showcase patient for segmentation grid figures.

    Modes:
        ``"representative"``: Highest consistent PaCS-SR Dice
            (``mean - 0.3 * std``). Shows a clean, typical case.
        ``"best-case"``: Smallest PaCS-SR disadvantage vs best expert.
            Shows where PaCS-SR comes closest to individual experts.
        ``"worst-case"``: Largest PaCS-SR disadvantage vs best expert.
            Shows an honest failure mode.

    Only considers patients with complete data for all methods and spacings.

    Args:
        raw_csv: Path to ``raw_results.csv``.
        labels_dir: Root labels directory (with method subdirs).
        methods: Method names.
        spacings: Spacing strings.
        mode: Selection mode.

    Returns:
        Patient ID.
    """
    df = pd.read_csv(raw_csv)
    available = _available_label_patients(labels_dir)
    LOG.info("Found %d patients with HR label maps", len(available))

    scored: List[Tuple[str, float]] = []
    for pid in available:
        pacs = df[(df["patient_id"] == pid) & (df["method"] == "PACS_SR")]
        if pacs.empty or pacs["dice_mean"].isna().any():
            continue

        pacs_dices = pacs["dice_mean"].values
        pacs_mean = float(np.mean(pacs_dices))

        if mode == "representative":
            score = pacs_mean - 0.3 * float(np.std(pacs_dices))
        elif mode in ("best-case", "worst-case"):
            # Require complete data for all experts at all spacings
            expert_means = []
            skip = False
            for m in ["ECLARE", "BSPLINE"]:
                m_data = df[(df["patient_id"] == pid) & (df["method"] == m)]
                if (
                    m_data.empty
                    or m_data["dice_mean"].isna().any()
                    or len(m_data) < len(spacings)
                ):
                    skip = True
                    break
                expert_means.append(float(m_data["dice_mean"].mean()))
            if skip or not expert_means:
                continue
            advantage = pacs_mean - max(expert_means)
            # best-case: maximize advantage (least negative)
            # worst-case: minimize advantage (most negative)
            score = advantage if mode == "best-case" else -advantage
        else:
            raise ValueError(f"Unknown mode: {mode}")

        scored.append((pid, score))

    if not scored:
        LOG.error("No valid patients found")
        return ""

    scored.sort(key=lambda x: x[1], reverse=True)
    best_pid, best_score = scored[0]
    LOG.info("Selected %s patient: %s (score=%.4f)", mode, best_pid, best_score)
    return best_pid


# ============================================================================
# Figure 2: Segmentation Grid
# ============================================================================


def generate_segmentation_grid(
    labels_dir: Path,
    raw_csv: Path,
    spacings: List[str],
    methods: List[str],
    out_path: Path,
    plane: str = "axial",
    patient_id: Optional[str] = None,
    pulse: str = "t1c",
    selection_mode: str = "representative",
) -> None:
    """Generate a segmentation comparison grid for one anatomical plane.

    Layout: rows = spacings, columns = [HR | M1 | M1_diff | M2 | M2_diff | ...]
    with column gaps between method groups. Black background.

    Args:
        labels_dir: Root labels directory with ``{METHOD}/`` subdirs and
            ``{METHOD}/{spacing}/`` for SR methods.
        raw_csv: Path to ``raw_results.csv`` for patient selection.
        spacings: List of spacing strings.
        methods: SR method display names.
        out_path: Output file path.
        plane: Anatomical plane.
        patient_id: Patient ID to visualize. If None, auto-selected.
        pulse: MRI pulse sequence.
        selection_mode: Patient selection mode for auto-selection.
    """
    apply_ieee_style()

    if patient_id is None:
        patient_id = select_showcase_patient(
            raw_csv,
            labels_dir,
            methods,
            spacings,
            mode=selection_mode,
        )
    if not patient_id:
        LOG.error("No valid patient found for segmentation grid")
        return
    LOG.info("Generating seg grid for patient=%s, plane=%s", patient_id, plane)

    # Load HR label volume
    hr_label_path = _find_label_file(labels_dir / "HR", patient_id, pulse)
    if hr_label_path is None:
        LOG.error("HR label file not found for %s", patient_id)
        return

    hr_nii = nib.load(str(hr_label_path))
    hr_data_raw = np.asarray(hr_nii.dataobj, dtype=np.int32)
    hr_data, _ = _reorient_to_ras(hr_data_raw.astype(np.float32), hr_nii.affine)
    hr_data = np.round(hr_data).astype(np.int32)

    slice_idx = _mid_brain_index(hr_data, plane)
    LOG.info("Mid-brain %s slice index: %d", plane, slice_idx)

    # Load method label volumes
    method_labels: Dict[str, Dict[str, np.ndarray]] = {}
    for spacing in spacings:
        method_labels[spacing] = {}
        for method in methods:
            method_key = _METHOD_KEYS[method]
            method_lbl_dir = labels_dir / method_key / spacing
            lbl_path = _find_label_file(method_lbl_dir, patient_id, pulse)
            if lbl_path is None:
                LOG.warning(
                    "Label file not found: %s/%s/%s",
                    method_key,
                    spacing,
                    patient_id,
                )
                continue
            nii = nib.load(str(lbl_path))
            vol_raw = np.asarray(nii.dataobj, dtype=np.int32)
            vol_ras, _ = _reorient_to_ras(vol_raw.astype(np.float32), nii.affine)
            method_labels[spacing][method_key] = np.round(vol_ras).astype(np.int32)

    # --- GridSpec layout ---
    n_rows = len(spacings)
    width_ratios: List[float] = [1.0]  # HR column
    col_assignments: List[Tuple[str, str]] = [("HR", "label")]

    for method in methods:
        width_ratios.append(0.12)
        col_assignments.append(("gap", "gap"))
        width_ratios.append(1.0)
        col_assignments.append((method, "label"))
        width_ratios.append(1.0)
        col_assignments.append((method, "diff"))

    total_cols = len(width_ratios)
    fig_height = n_rows * 1.45
    fig = plt.figure(
        figsize=(PLOT_SETTINGS["figure_width_double"], fig_height),
        facecolor="black",
    )
    gs = GridSpec(
        n_rows,
        total_cols,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.04,
        wspace=0.02,
    )

    # Brain crop from HR slice
    hr_slice = _extract_slice(hr_data, plane, slice_idx)
    crop_bbox = _brain_crop_bbox((hr_slice > 0).astype(np.float32), pad=3)
    r0, r1, c0, c1 = crop_bbox

    def _crop(s: np.ndarray) -> np.ndarray:
        return s[r0:r1, c0:c1]

    for row_idx, spacing in enumerate(spacings):
        sp_vols = method_labels.get(spacing, {})

        all_vols = [hr_data]
        for mk in [_METHOD_KEYS[m] for m in methods]:
            if mk in sp_vols:
                all_vols.append(sp_vols[mk])
        cropped_vols = _crop_to_common_shape(all_vols)
        hr_cropped = cropped_vols[0]

        sp_vols_cropped: Dict[str, np.ndarray] = {}
        vol_i = 1
        for m in methods:
            mk = _METHOD_KEYS[m]
            if mk in sp_vols:
                sp_vols_cropped[mk] = cropped_vols[vol_i]
                vol_i += 1

        axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
        si = min(slice_idx, hr_cropped.shape[axis_map[plane]] - 1)
        hr_sl = _extract_slice(hr_cropped, plane, si)
        hr_sl_crop = _crop(hr_sl)
        hr_rgb = _label_slice_to_rgb(hr_sl_crop, FREESURFER_LUT)

        for col_idx, (col_id, col_type) in enumerate(col_assignments):
            if col_type == "gap":
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.set_visible(False)
                continue

            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_facecolor("black")

            if col_id == "HR":
                ax.imshow(
                    hr_rgb,
                    origin="lower",
                    aspect="equal",
                    interpolation="nearest",
                )
            elif col_type == "label":
                mk = _METHOD_KEYS[col_id]
                if mk in sp_vols_cropped:
                    m_sl = _extract_slice(sp_vols_cropped[mk], plane, si)
                    m_sl_crop = _crop(m_sl)
                    m_rgb = _label_slice_to_rgb(m_sl_crop, FREESURFER_LUT)
                    ax.imshow(
                        m_rgb,
                        origin="lower",
                        aspect="equal",
                        interpolation="nearest",
                    )
                else:
                    ax.imshow(
                        np.zeros_like(hr_rgb),
                        origin="lower",
                        aspect="equal",
                    )
            elif col_type == "diff":
                mk = _METHOD_KEYS[col_id]
                if mk in sp_vols_cropped:
                    m_sl = _extract_slice(sp_vols_cropped[mk], plane, si)
                    m_sl_crop = _crop(m_sl)
                    diff_rgb = _diff_map_rgb(m_sl_crop, hr_sl_crop, FREESURFER_LUT)
                    ax.imshow(
                        diff_rgb,
                        origin="lower",
                        aspect="equal",
                        interpolation="nearest",
                    )
                else:
                    ax.imshow(
                        np.zeros_like(hr_rgb),
                        origin="lower",
                        aspect="equal",
                    )

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Column titles (top row)
            if row_idx == 0:
                if col_id == "HR" and col_type == "label":
                    ax.set_title(
                        "HR",
                        fontsize=PLOT_SETTINGS["tick_labelsize"],
                        color="white",
                        pad=3,
                    )
                elif col_type == "label" and col_id != "HR":
                    ax.set_title(
                        col_id,
                        fontsize=PLOT_SETTINGS["tick_labelsize"],
                        color="white",
                        pad=3,
                    )
                elif col_type == "diff":
                    ax.set_title(
                        r"$\Delta$",
                        fontsize=PLOT_SETTINGS["tick_labelsize"],
                        color="white",
                        pad=3,
                    )

            # Row labels (first column)
            if col_idx == 0:
                ax.set_ylabel(
                    _SPACING_DISPLAY.get(spacing, spacing),
                    fontsize=PLOT_SETTINGS["font_size"],
                    rotation=0,
                    labelpad=25,
                    va="center",
                    color="white",
                )

    _save_figure(fig, out_path, facecolor="black")
    LOG.info("Patient used: %s", patient_id)


# ============================================================================
# I/O helpers
# ============================================================================


def _save_figure(
    fig: plt.Figure,
    out_path: Path,
    facecolor: str = "white",
) -> None:
    """Save figure as PDF and PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        save_path = out_path.with_suffix(f".{ext}")
        fig.savefig(
            save_path,
            dpi=PLOT_SETTINGS["dpi_print"],
            bbox_inches="tight",
            facecolor=facecolor,
        )
        LOG.info("Saved %s", save_path)
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """CLI entry point for SynthSeg figures."""
    parser = argparse.ArgumentParser(
        description="SynthSeg evaluation figures for PaCS-SR"
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument(
        "--figure",
        choices=["qc-scatter", "seg-grid", "all"],
        default="all",
        help="Which figure to generate",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Patient ID for seg-grid (auto-selects if omitted)",
    )
    parser.add_argument(
        "--plane",
        choices=["axial", "coronal", "sagittal", "all"],
        default="all",
        help="Anatomical plane for seg-grid",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["representative", "best-case", "worst-case", "all"],
        default="representative",
        help="Patient selection mode for seg-grid: "
        "'representative' (highest consistent PaCS-SR Dice), "
        "'best-case' (smallest PaCS-SR disadvantage vs experts), "
        "'worst-case' (largest PaCS-SR disadvantage), "
        "'all' (generate all three)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s | %(message)s",
    )

    cfg = load_synthseg_eval_config(args.config)
    output_dir = cfg.output_dir
    qc_dir = output_dir / "qc"
    labels_dir = output_dir / "labels"
    raw_csv = output_dir / "raw_results.csv"
    spacings = list(cfg.spacings)
    fig_dir = output_dir / "figures"

    if args.figure in ("qc-scatter", "all"):
        generate_qc_scatter(
            qc_dir=qc_dir,
            spacings=spacings,
            out_path=fig_dir / "synthseg_qc_scatter",
            labels_dir=labels_dir,
        )

    if args.figure in ("seg-grid", "all"):
        sr_methods = [
            m
            for m in MODEL_ORDER
            if _METHOD_KEYS[m] in [mk for mk in cfg.methods if mk != "HR"]
        ]
        planes = (
            ["axial", "coronal", "sagittal"] if args.plane == "all" else [args.plane]
        )
        modes = (
            ["representative", "best-case", "worst-case"]
            if args.selection_mode == "all"
            else [args.selection_mode]
        )
        for mode in modes:
            for plane in planes:
                suffix = f"_{mode}" if mode != "representative" else ""
                generate_segmentation_grid(
                    labels_dir=labels_dir,
                    raw_csv=raw_csv,
                    spacings=spacings,
                    methods=sr_methods,
                    out_path=fig_dir / f"synthseg_seg_grid_{plane}{suffix}",
                    plane=plane,
                    patient_id=args.patient_id,
                    pulse=cfg.pulse,
                    selection_mode=mode,
                )


if __name__ == "__main__":
    main()
