"""SynthSeg-based brain segmentation evaluation for PaCS-SR.

Evaluates anatomical plausibility of super-resolved volumes by running
SynthSeg segmentation and comparing quality scores, Dice overlap, and
regional volumes against ground-truth HR segmentations.

Hypothesis: PaCS-SR yields anatomically more plausible volumes than
individual SR experts, as measured by higher SynthSeg QC scores and
closer segmentation agreement with HR ground truth.

The pipeline has 3 stages (designed for independent SLURM jobs):

1. **Export**: HDF5 volumes → NIfTI files with real affines.
2. **Segment**: Run SynthSeg (subprocess-isolated TF env) on each directory.
3. **Analyze**: Compute Dice, volume errors, QC comparison, statistical tests.

Reference:
    Billot et al., "SynthSeg: Segmentation of brain MRI scans of any
    contrast and resolution without retraining", Medical Image Analysis, 2023.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

from pacs_sr.analysis.analyze_metrics import (
    _fdr_adjust,
    cliffs_delta,
    cohen_dz,
)
from pacs_sr.data.hdf5_io import (
    blend_key,
    expert_h5_path,
    expert_key,
    hr_key,
    read_volume,
    results_h5_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FreeSurfer label IDs for SynthSeg regions of interest
# ---------------------------------------------------------------------------

SYNTHSEG_LABEL_MAP: Dict[str, int] = {
    "Left-Cerebral-White-Matter": 2,
    "Left-Cerebral-Cortex": 3,
    "Left-Lateral-Ventricle": 4,
    "Left-Cerebellum-White-Matter": 7,
    "Left-Cerebellum-Cortex": 8,
    "Left-Thalamus": 10,
    "Left-Caudate": 11,
    "Left-Putamen": 12,
    "Left-Pallidum": 13,
    "3rd-Ventricle": 14,
    "4th-Ventricle": 15,
    "Brain-Stem": 16,
    "Left-Hippocampus": 17,
    "Left-Amygdala": 18,
    "CSF": 24,
    "Left-Accumbens-area": 26,
    "Left-VentralDC": 28,
    "Right-Cerebral-White-Matter": 41,
    "Right-Cerebral-Cortex": 42,
    "Right-Lateral-Ventricle": 43,
    "Right-Cerebellum-White-Matter": 46,
    "Right-Cerebellum-Cortex": 47,
    "Right-Thalamus": 49,
    "Right-Caudate": 50,
    "Right-Putamen": 51,
    "Right-Pallidum": 52,
    "Right-Hippocampus": 53,
    "Right-Amygdala": 54,
    "Right-Accumbens-area": 58,
    "Right-VentralDC": 60,
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthSegEvalConfig:
    """Configuration for SynthSeg evaluation pipeline.

    Args:
        command: Command prefix list for invoking SynthSeg subprocess.
        robust: Use SynthSeg robust mode (``--robust`` flag).
        threads: CPU threads for SynthSeg (0 = system default).
        crop: Crop size for SynthSeg (must be divisible by 32, 0 = default 192).
        methods: SR methods to evaluate.
        spacings: Anisotropy spacings to sweep.
        pulse: MRI pulse sequence (e.g. ``"t1c"``).
        alpha: Significance level for statistical tests.
        correction: Multiple-comparison correction method.
        reference_method: Method treated as the reference for pairwise tests.
        output_dir: Root output directory for all SynthSeg artifacts.
        cleanup_nifti: If True, delete NIfTI files after SynthSeg run.
        source_h5: Path to source_data.h5 with HR volumes.
        experts_dir: Directory containing ``{model}.h5`` expert files.
        pacs_sr_out_root: PaCS-SR results output root.
        experiment_name: PaCS-SR experiment name (for results HDF5 lookup).
        manifest_path: Path to K-fold manifest JSON.
    """

    command: Tuple[str, ...] = ("mri_synthseg",)
    robust: bool = True
    threads: int = 1
    crop: int = 0
    methods: Tuple[str, ...] = ("HR", "BSPLINE", "ECLARE", "PACS_SR")
    spacings: Tuple[str, ...] = ("3mm", "5mm", "7mm")
    pulse: str = "t1c"
    alpha: float = 0.05
    correction: str = "fdr_bh"
    reference_method: str = "PACS_SR"
    output_dir: Path = Path("results/synthseg_evaluation")
    cleanup_nifti: bool = False
    source_h5: Path = Path()
    experts_dir: Path = Path()
    pacs_sr_out_root: Path = Path()
    experiment_name: str = "PaCS_SR"
    manifest_path: Path = Path()


def load_synthseg_eval_config(yaml_path: Path) -> SynthSegEvalConfig:
    """Parse YAML config into SynthSegEvalConfig, merging data paths.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Fully populated SynthSegEvalConfig.
    """
    import yaml

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    ss = cfg.get("synthseg", {})
    data = cfg.get("data", {})
    pacs = cfg.get("pacs_sr", {})

    return SynthSegEvalConfig(
        command=tuple(ss.get("command", ["mri_synthseg"])),
        robust=bool(ss.get("robust", True)),
        threads=int(ss.get("threads", 1)),
        crop=int(ss.get("crop", 0)),
        methods=tuple(ss.get("methods", ["HR", "BSPLINE", "ECLARE", "PACS_SR"])),
        spacings=tuple(ss.get("spacings", data.get("spacings", ["3mm", "5mm", "7mm"]))),
        pulse=str(ss.get("pulse", "t1c")),
        alpha=float(ss.get("alpha", 0.05)),
        correction=str(ss.get("correction", "fdr_bh")),
        reference_method=str(ss.get("reference_method", "PACS_SR")),
        output_dir=Path(ss.get("output_dir", "results/synthseg_evaluation")),
        cleanup_nifti=bool(ss.get("cleanup_nifti", False)),
        source_h5=Path(data.get("source-h5", "")),
        experts_dir=Path(data.get("experts-dir", "")),
        pacs_sr_out_root=Path(pacs.get("out_root", "")),
        experiment_name=str(pacs.get("experiment_name", "PaCS_SR")),
        manifest_path=Path(data.get("out", "")),
    )


# ---------------------------------------------------------------------------
# K-fold utilities
# ---------------------------------------------------------------------------


def build_patient_fold_map(manifest: dict) -> Dict[str, int]:
    """Map each patient to the 1-indexed fold where they appear in the test set.

    Args:
        manifest: K-fold manifest with structure
            ``{"folds": [{"train": [...], "test": [...]}, ...]}``.

    Returns:
        Dict mapping ``patient_id → fold_number`` (1-indexed).
    """
    patient_fold: Dict[str, int] = {}
    for fold_idx, fold in enumerate(manifest["folds"]):
        for pid in fold["test"]:
            patient_fold[pid] = fold_idx + 1
    return patient_fold


def collect_all_test_patients(manifest: dict) -> List[str]:
    """Collect the union of all test-set patients across folds.

    Args:
        manifest: K-fold manifest.

    Returns:
        Sorted list of unique patient IDs.
    """
    patients = set()
    for fold in manifest["folds"]:
        patients.update(fold["test"])
    return sorted(patients)


# ---------------------------------------------------------------------------
# Stage 1: Export HDF5 → NIfTI
# ---------------------------------------------------------------------------


def _export_single_volume(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
) -> Path:
    """Write a volume to NIfTI, skipping if the file already exists.

    Args:
        data: 3D volume array.
        affine: 4×4 affine matrix.
        output_path: Target NIfTI path.

    Returns:
        Path to the written NIfTI file.
    """
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, str(output_path))
    return output_path


def export_volumes_to_nifti(
    config: SynthSegEvalConfig,
    manifest: dict,
) -> Dict[str, List[Path]]:
    """Export all method volumes from HDF5 to NIfTI files.

    HR volumes are exported once (no spacing dimension). Expert and PaCS-SR
    volumes are exported per spacing.

    Args:
        config: SynthSeg evaluation configuration.
        manifest: K-fold manifest for patient lists and PaCS-SR fold lookup.

    Returns:
        Dict mapping ``"{method}" or "{method}/{spacing}"`` to list of NIfTI paths.
    """
    patients = collect_all_test_patients(manifest)
    patient_fold_map = build_patient_fold_map(manifest)
    pulse = config.pulse
    base_dir = config.output_dir / "nifti"
    exported: Dict[str, List[Path]] = {}

    logger.info(
        "Exporting %d patients × %d methods × %d spacings to NIfTI",
        len(patients),
        len(config.methods),
        len(config.spacings),
    )

    for method in config.methods:
        if method == "HR":
            # HR has no spacing: export once
            key_label = "HR"
            paths: List[Path] = []
            for pid in patients:
                try:
                    data, affine = read_volume(config.source_h5, hr_key(pid, pulse))
                    out_path = base_dir / "HR" / f"{pid}-{pulse}.nii.gz"
                    _export_single_volume(data, affine, out_path)
                    paths.append(out_path)
                except (KeyError, FileNotFoundError) as exc:
                    logger.warning("Skipping HR export for %s: %s", pid, exc)
            exported[key_label] = paths
            logger.info("Exported %d HR volumes", len(paths))

        elif method == "PACS_SR":
            for spacing in config.spacings:
                key_label = f"PACS_SR/{spacing}"
                paths = []
                for pid in patients:
                    fold_num = patient_fold_map.get(pid)
                    if fold_num is None:
                        logger.warning(
                            "Patient %s not in any test fold, skipping PACS_SR", pid
                        )
                        continue
                    try:
                        h5_path = results_h5_path(
                            config.pacs_sr_out_root,
                            config.experiment_name,
                            fold_num,
                        )
                        data, affine = read_volume(
                            h5_path, blend_key(spacing, pid, pulse)
                        )
                        out_path = (
                            base_dir / "PACS_SR" / spacing / f"{pid}-{pulse}.nii.gz"
                        )
                        _export_single_volume(data, affine, out_path)
                        paths.append(out_path)
                    except (KeyError, FileNotFoundError) as exc:
                        logger.warning(
                            "Skipping PACS_SR export for %s/%s: %s",
                            pid,
                            spacing,
                            exc,
                        )
                exported[key_label] = paths
                logger.info("Exported %d PACS_SR/%s volumes", len(paths), spacing)

        else:
            # Expert methods (BSPLINE, ECLARE, etc.)
            for spacing in config.spacings:
                key_label = f"{method}/{spacing}"
                paths = []
                h5_path = expert_h5_path(config.experts_dir, method)
                for pid in patients:
                    try:
                        data, affine = read_volume(
                            h5_path, expert_key(spacing, pid, pulse)
                        )
                        out_path = base_dir / method / spacing / f"{pid}-{pulse}.nii.gz"
                        _export_single_volume(data, affine, out_path)
                        paths.append(out_path)
                    except (KeyError, FileNotFoundError) as exc:
                        logger.warning(
                            "Skipping %s export for %s/%s: %s",
                            method,
                            pid,
                            spacing,
                            exc,
                        )
                exported[key_label] = paths
                logger.info("Exported %d %s/%s volumes", len(paths), method, spacing)

    return exported


# ---------------------------------------------------------------------------
# Stage 2: Run SynthSeg
# ---------------------------------------------------------------------------


def check_synthseg_available(command: List[str] | Tuple[str, ...]) -> bool:
    """Check if the SynthSeg command is available.

    Args:
        command: Command prefix for SynthSeg invocation.

    Returns:
        True if the command can be executed.
    """
    if not command:
        return False
    exe = command[0]
    if not Path(exe).is_absolute():
        if shutil.which(exe) is None:
            return False
    elif not Path(exe).exists():
        return False
    if len(command) > 1 and not Path(command[1]).exists():
        return False
    return True


def run_synthseg_on_directory(
    input_dir: Path,
    labels_dir: Path,
    volumes_csv: Path,
    qc_csv: Path,
    command: List[str] | Tuple[str, ...],
    robust: bool = True,
    threads: int = 1,
    crop: int = 0,
) -> bool:
    """Run SynthSeg on a directory of NIfTI volumes.

    Args:
        input_dir: Directory containing input ``.nii.gz`` files.
        labels_dir: Directory for output segmentation label maps.
        volumes_csv: Path to output CSV with regional volumes.
        qc_csv: Path to output CSV with QC scores.
        command: SynthSeg command prefix.
        robust: Enable robust mode.
        threads: CPU threads (0 = system default).
        crop: Crop size (0 = default 192).

    Returns:
        True if SynthSeg completed successfully.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    volumes_csv.parent.mkdir(parents=True, exist_ok=True)
    qc_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        *command,
        "--i",
        str(input_dir),
        "--o",
        str(labels_dir),
        "--vol",
        str(volumes_csv),
        "--qc",
        str(qc_csv),
    ]
    if robust:
        cmd.append("--robust")
    if threads > 0:
        cmd.extend(["--threads", str(threads)])
    if crop > 0:
        cmd.extend(["--crop", str(crop)])

    logger.info("Running SynthSeg: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if result.returncode != 0:
            logger.error(
                "SynthSeg failed (rc=%d):\nstdout: %s\nstderr: %s",
                result.returncode,
                result.stdout[-500:] if result.stdout else "",
                result.stderr[-500:] if result.stderr else "",
            )
            return False
        logger.info("SynthSeg completed successfully for %s", input_dir)
        return True
    except FileNotFoundError:
        logger.error("SynthSeg command not found: %s", command)
        return False
    except subprocess.TimeoutExpired:
        logger.error("SynthSeg timed out after 2 hours for %s", input_dir)
        return False


def run_all_synthseg(config: SynthSegEvalConfig) -> Dict[str, bool]:
    """Run SynthSeg on all exported NIfTI directories.

    Skips runs where the volumes CSV already exists (caching/resumability).

    Args:
        config: SynthSeg evaluation configuration.

    Returns:
        Dict mapping run label to success status.
    """
    if not check_synthseg_available(list(config.command)):
        logger.error(
            "SynthSeg not available (command=%s). Cannot proceed.", config.command
        )
        return {}

    base_nifti = config.output_dir / "nifti"
    base_labels = config.output_dir / "labels"
    base_volumes = config.output_dir / "volumes"
    base_qc = config.output_dir / "qc"

    results: Dict[str, bool] = {}

    # Build list of (label, input_dir, labels_dir, vol_csv, qc_csv)
    runs: List[Tuple[str, Path, Path, Path, Path]] = []

    for method in config.methods:
        if method == "HR":
            input_dir = base_nifti / "HR"
            runs.append(
                (
                    "HR",
                    input_dir,
                    base_labels / "HR",
                    base_volumes / "HR_volumes.csv",
                    base_qc / "HR_qc.csv",
                )
            )
        else:
            for spacing in config.spacings:
                label = f"{method}_{spacing}"
                input_dir = base_nifti / method / spacing
                runs.append(
                    (
                        label,
                        input_dir,
                        base_labels / method / spacing,
                        base_volumes / f"{method}_{spacing}_volumes.csv",
                        base_qc / f"{method}_{spacing}_qc.csv",
                    )
                )

    for label, input_dir, labels_dir, vol_csv, qc_csv in runs:
        if vol_csv.exists():
            logger.info("SynthSeg results for %s already cached, skipping.", label)
            results[label] = True
            continue

        if not input_dir.exists() or not any(input_dir.glob("*.nii.gz")):
            logger.warning("No NIfTI files for %s in %s, skipping.", label, input_dir)
            results[label] = False
            continue

        ok = run_synthseg_on_directory(
            input_dir=input_dir,
            labels_dir=labels_dir,
            volumes_csv=vol_csv,
            qc_csv=qc_csv,
            command=list(config.command),
            robust=config.robust,
            threads=config.threads,
            crop=config.crop,
        )
        results[label] = ok

    return results


# ---------------------------------------------------------------------------
# Stage 3: Analyze — parsing
# ---------------------------------------------------------------------------


def parse_qc_csv(qc_csv: Path) -> Dict[str, float]:
    """Parse SynthSeg QC CSV into per-subject scores.

    The QC CSV has columns: ``subject, qc`` (or similar).

    Args:
        qc_csv: Path to the QC CSV output by SynthSeg ``--qc``.

    Returns:
        Dict mapping ``patient_id → qc_score``.
    """
    scores: Dict[str, float] = {}
    with open(qc_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row.get("subject", row.get("", ""))
            # Strip path and extension to get patient ID
            subject = Path(subject).stem.replace(".nii", "")
            qc_val = row.get("qc", row.get("qc_score", ""))
            try:
                scores[subject] = float(qc_val)
            except (ValueError, TypeError):
                logger.warning("Invalid QC value for %s: %s", subject, qc_val)
    return scores


def parse_volumes_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Parse SynthSeg volumes CSV into nested dict.

    Args:
        csv_path: Path to the volumes CSV output by SynthSeg ``--vol``.

    Returns:
        Dict mapping ``{patient_id: {region_name: volume_mm3}}``.
    """
    volumes: Dict[str, Dict[str, float]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row.get("subject", row.get("", ""))
            subject = Path(subject).stem.replace(".nii", "")
            region_vols: Dict[str, float] = {}
            for key, val in row.items():
                if key in ("subject", ""):
                    continue
                try:
                    region_vols[key] = float(val)
                except (ValueError, TypeError):
                    pass
            volumes[subject] = region_vols
    return volumes


# ---------------------------------------------------------------------------
# Stage 3: Analyze — metrics computation
# ---------------------------------------------------------------------------


def compute_dice_per_region(
    method_label_path: Path,
    hr_label_path: Path,
    label_map: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Compute Dice overlap per region between two SynthSeg label maps.

    Args:
        method_label_path: Path to method's SynthSeg label map NIfTI.
        hr_label_path: Path to HR reference label map NIfTI.
        label_map: Region name → FreeSurfer label ID. Defaults to full map.

    Returns:
        Dict with per-region Dice scores and ``"mean"`` Dice.
    """
    if label_map is None:
        label_map = SYNTHSEG_LABEL_MAP

    method_data = np.asarray(nib.load(str(method_label_path)).dataobj, dtype=np.int32)
    hr_data = np.asarray(nib.load(str(hr_label_path)).dataobj, dtype=np.int32)

    dices: Dict[str, float] = {}
    for name, lid in label_map.items():
        method_mask = method_data == lid
        hr_mask = hr_data == lid
        intersection = np.sum(method_mask & hr_mask)
        denom = np.sum(method_mask) + np.sum(hr_mask)
        if denom == 0:
            dices[name] = 1.0  # Both empty → perfect match
        else:
            dices[name] = float(2.0 * intersection / denom)

    valid_dices = [v for v in dices.values()]
    if valid_dices:
        dices["mean"] = float(np.mean(valid_dices))

    return dices


def compute_volume_errors(
    method_volumes: Dict[str, float],
    hr_volumes: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Compute absolute and relative volume errors per region.

    Args:
        method_volumes: Method's regional volumes (from SynthSeg CSV).
        hr_volumes: HR reference regional volumes.

    Returns:
        Dict with ``"absolute"`` and ``"relative"`` sub-dicts per region.
    """
    errors: Dict[str, Dict[str, float]] = {"absolute": {}, "relative": {}}
    for region in hr_volumes:
        if region not in method_volumes:
            continue
        hr_vol = hr_volumes[region]
        method_vol = method_volumes[region]
        abs_err = abs(method_vol - hr_vol)
        errors["absolute"][region] = abs_err
        if hr_vol > 1e-10:
            errors["relative"][region] = abs_err / hr_vol
        else:
            errors["relative"][region] = float("nan")
    return errors


def _find_label_file(
    labels_dir: Path,
    patient_id: str,
    pulse: str,
) -> Optional[Path]:
    """Locate the SynthSeg label map for a given patient.

    SynthSeg preserves the input filename, so we look for
    ``{patient_id}-{pulse}.nii.gz`` in the labels directory.

    Args:
        labels_dir: Directory containing SynthSeg label NIfTIs.
        patient_id: Patient identifier.
        pulse: Pulse sequence name.

    Returns:
        Path to the label file, or None if not found.
    """
    candidates = [
        labels_dir / f"{patient_id}-{pulse}.nii.gz",
        labels_dir / f"{patient_id}-{pulse}_synthseg.nii.gz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    # Fallback: glob
    matches = list(labels_dir.glob(f"*{patient_id}*{pulse}*.nii.gz"))
    return matches[0] if matches else None


def assemble_results(
    config: SynthSegEvalConfig,
    patients: List[str],
) -> Dict[str, Any]:
    """Assemble all per-patient metrics from SynthSeg outputs.

    Reads QC CSVs, volumes CSVs, and label maps to compute Dice and
    volume errors for each (method, spacing, patient) vs HR.

    Args:
        config: SynthSeg evaluation configuration.
        patients: List of patient IDs to analyze.

    Returns:
        Dict with ``"metadata"``, ``"per_patient"``, and ``"hr_reference"`` keys.
    """
    pulse = config.pulse
    base_labels = config.output_dir / "labels"
    base_volumes = config.output_dir / "volumes"
    base_qc = config.output_dir / "qc"

    # Parse HR reference data
    hr_qc = _safe_parse_qc(base_qc / "HR_qc.csv")
    hr_vols = _safe_parse_volumes(base_volumes / "HR_volumes.csv")

    hr_reference: Dict[str, Any] = {}
    for pid in patients:
        pid_key = f"{pid}-{pulse}"
        hr_reference[pid] = {
            "qc_score": hr_qc.get(pid_key, float("nan")),
            "volumes": hr_vols.get(pid_key, {}),
        }

    # Parse method data and compute metrics
    sr_methods = [m for m in config.methods if m != "HR"]
    per_patient: Dict[str, Dict[str, Any]] = {}

    for pid in patients:
        per_patient[pid] = {}
        pid_key = f"{pid}-{pulse}"
        hr_label_path = _find_label_file(base_labels / "HR", pid, pulse)

        for spacing in config.spacings:
            per_patient[pid][spacing] = {}

            for method in sr_methods:
                label = f"{method}_{spacing}"
                method_qc = _safe_parse_qc(base_qc / f"{label}_qc.csv")
                method_vols = _safe_parse_volumes(base_volumes / f"{label}_volumes.csv")

                result: Dict[str, Any] = {
                    "qc_score": method_qc.get(pid_key, float("nan")),
                }

                # Dice vs HR labels
                if hr_label_path is not None:
                    method_label_path = _find_label_file(
                        base_labels / method / spacing, pid, pulse
                    )
                    if method_label_path is not None:
                        result["dice"] = compute_dice_per_region(
                            method_label_path, hr_label_path
                        )
                    else:
                        result["dice"] = {}
                else:
                    result["dice"] = {}

                # Volume errors vs HR
                hr_v = hr_vols.get(pid_key, {})
                method_v = method_vols.get(pid_key, {})
                if hr_v and method_v:
                    vol_err = compute_volume_errors(method_v, hr_v)
                    result["volume_error"] = vol_err["absolute"]
                    result["rel_volume_error"] = vol_err["relative"]
                else:
                    result["volume_error"] = {}
                    result["rel_volume_error"] = {}

                per_patient[pid][spacing][method] = result

    return {
        "metadata": {
            "pulse": pulse,
            "spacings": list(config.spacings),
            "methods": list(sr_methods),
            "n_patients": len(patients),
            "label_map": SYNTHSEG_LABEL_MAP,
        },
        "per_patient": per_patient,
        "hr_reference": hr_reference,
    }


def _safe_parse_qc(path: Path) -> Dict[str, float]:
    """Parse QC CSV, returning empty dict if file doesn't exist."""
    if not path.exists():
        logger.warning("QC CSV not found: %s", path)
        return {}
    return parse_qc_csv(path)


def _safe_parse_volumes(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse volumes CSV, returning empty dict if file doesn't exist."""
    if not path.exists():
        logger.warning("Volumes CSV not found: %s", path)
        return {}
    return parse_volumes_csv(path)


def results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert raw results dict to a tidy long-format DataFrame.

    Columns: patient_id, spacing, method, qc_score, dice_mean,
    dice_{region}, vol_error_{region}, rel_vol_error_{region}.

    Args:
        results: Output from ``assemble_results()``.

    Returns:
        Tidy DataFrame with one row per (patient, spacing, method).
    """
    rows: List[Dict[str, Any]] = []
    per_patient = results["per_patient"]
    hr_ref = results["hr_reference"]

    # Add HR reference rows
    for pid, hr_data in hr_ref.items():
        for spacing in results["metadata"]["spacings"]:
            row: Dict[str, Any] = {
                "patient_id": pid,
                "spacing": spacing,
                "method": "HR",
                "qc_score": hr_data.get("qc_score", float("nan")),
                "dice_mean": 1.0,  # HR vs itself
            }
            for region in SYNTHSEG_LABEL_MAP:
                row[f"dice_{region}"] = 1.0
                row[f"vol_error_{region}"] = 0.0
                row[f"rel_vol_error_{region}"] = 0.0
            rows.append(row)

    # Add SR method rows
    for pid, spacing_data in per_patient.items():
        for spacing, method_data in spacing_data.items():
            for method, metrics in method_data.items():
                row = {
                    "patient_id": pid,
                    "spacing": spacing,
                    "method": method,
                    "qc_score": metrics.get("qc_score", float("nan")),
                    "dice_mean": metrics.get("dice", {}).get("mean", float("nan")),
                }
                dice = metrics.get("dice", {})
                vol_err = metrics.get("volume_error", {})
                rel_vol_err = metrics.get("rel_volume_error", {})
                for region in SYNTHSEG_LABEL_MAP:
                    row[f"dice_{region}"] = dice.get(region, float("nan"))
                    row[f"vol_error_{region}"] = vol_err.get(region, float("nan"))
                    row[f"rel_vol_error_{region}"] = rel_vol_err.get(
                        region, float("nan")
                    )
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 3: Analyze — statistical tests
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    x: np.ndarray,
    n_resamples: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Args:
        x: 1D array of observations.
        n_resamples: Number of bootstrap resamples.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (lower, upper) bounds.
    """
    rng = np.random.default_rng(seed)
    means = np.array(
        [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n_resamples)]
    )
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def run_statistical_analysis(
    df: pd.DataFrame,
    config: SynthSegEvalConfig,
) -> Dict[str, Any]:
    """Run comprehensive statistical analysis on assembled results.

    For each spacing and metric, computes:
    - Friedman test across all SR methods
    - Pairwise Wilcoxon signed-rank vs reference method
    - Effect sizes (Cohen's dz, Cliff's delta)
    - Bootstrap CIs for mean differences
    - BH-FDR correction across all tests

    Args:
        df: Tidy DataFrame from ``results_to_dataframe()``.
        config: SynthSeg evaluation configuration.

    Returns:
        Dict with statistical test results organized by metric and spacing.
    """
    sr_methods = [m for m in config.methods if m != "HR"]
    ref = config.reference_method
    comparisons = [(m, ref) for m in sr_methods if m != ref]

    all_results: Dict[str, Any] = {}

    # Metrics to test
    metric_cols = ["qc_score", "dice_mean"]
    # Add per-region dice and volume error
    for region in SYNTHSEG_LABEL_MAP:
        metric_cols.append(f"dice_{region}")
        metric_cols.append(f"vol_error_{region}")
        metric_cols.append(f"rel_vol_error_{region}")

    # Collect all p-values for FDR correction
    all_pvals: List[float] = []
    pval_indices: List[Tuple[str, str, str, str]] = []  # metric, spacing, test, comp

    for metric in metric_cols:
        if metric not in df.columns:
            continue
        metric_results: Dict[str, Any] = {}

        for spacing in config.spacings:
            spacing_df = df[df["spacing"] == spacing]
            spacing_results: Dict[str, Any] = {}

            # Get paired data (only patients present in all methods)
            patients = spacing_df[spacing_df["method"] == ref]["patient_id"].values
            method_arrays: Dict[str, np.ndarray] = {}

            for method in sr_methods:
                method_df = spacing_df[spacing_df["method"] == method]
                # Align by patient_id
                merged = pd.merge(
                    pd.DataFrame({"patient_id": patients}),
                    method_df[["patient_id", metric]],
                    on="patient_id",
                    how="inner",
                )
                vals = merged[metric].values.astype(float)
                valid = ~np.isnan(vals)
                method_arrays[method] = vals[valid]

            # Friedman test (requires ≥3 methods with same patients)
            friedman_arrays = []
            min_len = min((len(v) for v in method_arrays.values()), default=0)
            if len(method_arrays) >= 3 and min_len >= 5:
                for method in sr_methods:
                    friedman_arrays.append(method_arrays[method][:min_len])
                try:
                    stat, pval = stats.friedmanchisquare(*friedman_arrays)
                    spacing_results["friedman"] = {
                        "statistic": float(stat),
                        "p_value": float(pval),
                    }
                    all_pvals.append(pval)
                    pval_indices.append((metric, spacing, "friedman", "global"))
                except Exception as exc:
                    logger.warning(
                        "Friedman test failed for %s/%s: %s", metric, spacing, exc
                    )
                    spacing_results["friedman"] = {
                        "statistic": float("nan"),
                        "p_value": float("nan"),
                    }

            # Pairwise Wilcoxon tests
            pairwise: Dict[str, Any] = {}
            for method_a, method_b in comparisons:
                comp_key = f"{method_b}_vs_{method_a}"

                a_vals = method_arrays.get(method_a, np.array([]))
                b_vals = method_arrays.get(method_b, np.array([]))

                # Align lengths
                n = min(len(a_vals), len(b_vals))
                if n < 5:
                    pairwise[comp_key] = {
                        "wilcoxon_statistic": float("nan"),
                        "p_value": float("nan"),
                        "p_adjusted": float("nan"),
                        "cohens_dz": float("nan"),
                        "cliffs_delta": float("nan"),
                        "mean_diff": float("nan"),
                        "ci_95": [float("nan"), float("nan")],
                        "n": n,
                    }
                    continue

                a = a_vals[:n]
                b = b_vals[:n]
                diff = b - a

                # Wilcoxon signed-rank
                try:
                    w_stat, w_pval = stats.wilcoxon(a, b)
                except Exception:
                    w_stat, w_pval = float("nan"), float("nan")

                # Effect sizes
                dz = cohen_dz(b, a)
                cd = cliffs_delta(b, a)

                # Bootstrap CI for mean difference
                ci_lo, ci_hi = _bootstrap_ci(diff)

                pairwise[comp_key] = {
                    "wilcoxon_statistic": float(w_stat),
                    "p_value": float(w_pval),
                    "p_adjusted": float("nan"),  # filled after FDR
                    "cohens_dz": dz,
                    "cliffs_delta": cd,
                    "mean_diff": float(np.mean(diff)),
                    "ci_95": [ci_lo, ci_hi],
                    "n": n,
                }
                if not np.isnan(w_pval):
                    all_pvals.append(w_pval)
                    pval_indices.append((metric, spacing, "wilcoxon", comp_key))

            spacing_results["pairwise"] = pairwise
            metric_results[spacing] = spacing_results

        all_results[metric] = metric_results

    # Apply FDR correction across all tests
    if all_pvals:
        adjusted = _fdr_adjust(np.array(all_pvals), config.correction)
        for i, (metric, spacing, test_type, comp_key) in enumerate(pval_indices):
            if test_type == "friedman":
                all_results[metric][spacing]["friedman"]["p_adjusted"] = float(
                    adjusted[i]
                )
            elif test_type == "wilcoxon":
                all_results[metric][spacing]["pairwise"][comp_key]["p_adjusted"] = (
                    float(adjusted[i])
                )

    return all_results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_export(config: SynthSegEvalConfig) -> Dict[str, List[Path]]:
    """Stage 1: Export HDF5 volumes to NIfTI.

    Args:
        config: SynthSeg evaluation configuration.

    Returns:
        Dict of exported paths per method/spacing.
    """
    with open(config.manifest_path) as f:
        manifest = json.load(f)
    return export_volumes_to_nifti(config, manifest)


def run_segment(config: SynthSegEvalConfig) -> Dict[str, bool]:
    """Stage 2: Run SynthSeg on all exported directories.

    Args:
        config: SynthSeg evaluation configuration.

    Returns:
        Dict of success status per run.
    """
    return run_all_synthseg(config)


def run_analyze(config: SynthSegEvalConfig) -> Dict[str, Any]:
    """Stage 3: Compute metrics and statistical tests, save all outputs.

    Args:
        config: SynthSeg evaluation configuration.

    Returns:
        Dict with raw results, DataFrame, and statistical test results.
    """
    with open(config.manifest_path) as f:
        manifest = json.load(f)
    patients = collect_all_test_patients(manifest)

    # Assemble per-patient results
    logger.info("Assembling results for %d patients...", len(patients))
    raw_results = assemble_results(config, patients)

    # Convert to tidy DataFrame
    df = results_to_dataframe(raw_results)

    # Statistical analysis
    logger.info("Running statistical analysis...")
    stat_results = run_statistical_analysis(df, config)

    # Save outputs
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw results JSON
    raw_path = output_dir / "raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(raw_results, f, indent=2, default=_json_default)
    logger.info("Saved raw results to %s", raw_path)

    # Tidy CSV
    csv_path = output_dir / "raw_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved tidy DataFrame to %s", csv_path)

    # Statistical tests JSON
    stats_path = output_dir / "statistical_tests.json"
    with open(stats_path, "w") as f:
        json.dump(stat_results, f, indent=2, default=_json_default)
    logger.info("Saved statistical tests to %s", stats_path)

    # Summary CSV (aggregated mean ± std per method/spacing)
    summary_path = output_dir / "summary.csv"
    summary_df = _build_summary(df)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)

    return {
        "raw_results": raw_results,
        "dataframe": df,
        "statistical_tests": stat_results,
        "summary": summary_df,
    }


def run_full_pipeline(config: SynthSegEvalConfig) -> Dict[str, Any]:
    """Run the complete 3-stage SynthSeg evaluation pipeline.

    Args:
        config: SynthSeg evaluation configuration.

    Returns:
        Dict with results from all stages.
    """
    logger.info("=== Stage 1: Export HDF5 → NIfTI ===")
    exported = run_export(config)

    logger.info("=== Stage 2: Run SynthSeg ===")
    synthseg_status = run_segment(config)
    n_ok = sum(1 for v in synthseg_status.values() if v)
    n_total = len(synthseg_status)
    logger.info("SynthSeg completed: %d/%d successful", n_ok, n_total)

    if n_ok == 0:
        logger.error("All SynthSeg runs failed. Cannot proceed to analysis.")
        return {"exported": exported, "synthseg_status": synthseg_status}

    logger.info("=== Stage 3: Analyze ===")
    analysis = run_analyze(config)

    return {
        "exported": exported,
        "synthseg_status": synthseg_status,
        **analysis,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics as mean ± std per method/spacing.

    Args:
        df: Tidy DataFrame from ``results_to_dataframe()``.

    Returns:
        Summary DataFrame with columns: method, spacing, metric, mean, std, n.
    """
    summary_rows: List[Dict[str, Any]] = []
    metrics = ["qc_score", "dice_mean"]
    # Include per-region dice means
    for region in SYNTHSEG_LABEL_MAP:
        metrics.append(f"dice_{region}")

    for (method, spacing), group in df.groupby(["method", "spacing"]):
        for metric in metrics:
            if metric not in group.columns:
                continue
            vals = group[metric].dropna().values
            if len(vals) == 0:
                continue
            summary_rows.append(
                {
                    "method": method,
                    "spacing": spacing,
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "median": float(np.median(vals)),
                    "n": len(vals),
                }
            )

    return pd.DataFrame(summary_rows)


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types and Paths."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
