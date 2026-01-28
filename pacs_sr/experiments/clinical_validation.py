#!/usr/bin/env python
"""
Clinical Validation with ROI-specific Performance
==================================================

Computes super-resolution metrics within clinically relevant ROIs:
- Tumor core
- Edema
- Surrounding tissue
- Whole brain

This enables answering: "Which SR method performs best in which anatomical region?"

Usage:
    python -m pacs_sr.experiments.clinical_validation \\
        --results-dir ./results \\
        --gt-dir ./high_resolution \\
        --methods BSPLINE ECLARE SMORE UNIRES PACS_SR \\
        --output results/clinical/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# Default ROI labels for brain tumor segmentations
DEFAULT_ROI_LABELS = {
    "all": None,      # Special: entire brain mask
    "core": 1,        # Tumor core (necrosis + enhancing)
    "edema": 2,       # Peritumoral edema
    "surround": 3,    # Surrounding tissue
}


def load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI volume as numpy array."""
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for NIfTI loading")
    return nib.load(path).get_fdata().astype(np.float32)


def compute_psnr_masked(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Compute PSNR within masked region."""
    p = pred[mask]
    t = target[mask]

    if len(p) == 0:
        return float('nan')

    mse = np.mean((p - t) ** 2)
    if mse <= 0:
        return float('inf')

    data_range = float(np.max(t) - np.min(t))
    if data_range == 0:
        return float('nan')

    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def compute_ssim_slicewise(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    axis: int = 0,
) -> float:
    """Compute SSIM averaged over slices within masked region."""
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image required for SSIM")

    scores = []
    n_slices = pred.shape[axis]

    for i in range(n_slices):
        if axis == 0:
            p_slice = pred[i]
            t_slice = target[i]
            m_slice = mask[i]
        elif axis == 1:
            p_slice = pred[:, i, :]
            t_slice = target[:, i, :]
            m_slice = mask[:, i, :]
        else:
            p_slice = pred[:, :, i]
            t_slice = target[:, :, i]
            m_slice = mask[:, :, i]

        if not np.any(m_slice):
            continue

        # Apply mask by zeroing outside
        p_masked = p_slice * m_slice
        t_masked = t_slice * m_slice

        data_range = float(np.max(t_masked) - np.min(t_masked))
        if data_range < 1e-7:
            continue

        try:
            score = structural_similarity(t_masked, p_masked, data_range=data_range)
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue

    return float(np.mean(scores)) if scores else 0.0


def compute_roi_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    segmentation: Optional[np.ndarray] = None,
    roi_labels: Optional[Dict[str, Optional[int]]] = None,
    brain_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute PSNR and SSIM within each ROI.

    Args:
        pred: Predicted SR volume (Z, Y, X)
        target: Ground truth HR volume (Z, Y, X)
        segmentation: Segmentation volume with ROI labels
        roi_labels: Mapping of ROI names to label values
        brain_mask: Optional brain mask for "all" ROI

    Returns:
        Dict mapping ROI name -> {psnr, ssim, n_voxels}
    """
    if roi_labels is None:
        roi_labels = DEFAULT_ROI_LABELS

    results = {}

    for roi_name, roi_label in roi_labels.items():
        if roi_label is None:
            # "all" region - use brain mask or nonzero target
            if brain_mask is not None:
                mask = brain_mask.astype(bool)
            else:
                mask = target > 0
        elif segmentation is not None:
            mask = segmentation == roi_label
        else:
            continue

        if not np.any(mask):
            continue

        psnr = compute_psnr_masked(pred, target, mask)
        ssim = compute_ssim_slicewise(pred, target, mask.astype(np.float32))

        results[roi_name] = {
            "psnr": psnr,
            "ssim": ssim,
            "n_voxels": int(np.sum(mask)),
        }

    return results


def find_matching_files(
    patient_id: str,
    pulse: str,
    results_dir: Path,
    gt_dir: Path,
    seg_dir: Optional[Path] = None,
) -> Dict[str, Optional[Path]]:
    """
    Find matching prediction, ground truth, and segmentation files.
    """
    patterns = [
        f"{patient_id}-{pulse}.nii.gz",
        f"{patient_id}_{pulse}.nii.gz",
        f"{patient_id}-{pulse}.nii",
        f"{patient_id}_{pulse}.nii",
    ]

    files = {"pred": None, "gt": None, "seg": None}

    # Find prediction
    for pattern in patterns:
        pred_path = results_dir / pattern
        if pred_path.exists():
            files["pred"] = pred_path
            break
        # Check in output_volumes subdirectory
        pred_path = results_dir / "output_volumes" / pattern
        if pred_path.exists():
            files["pred"] = pred_path
            break

    # Find ground truth
    for pattern in patterns:
        gt_path = gt_dir / patient_id / pattern
        if gt_path.exists():
            files["gt"] = gt_path
            break
        gt_path = gt_dir / pattern
        if gt_path.exists():
            files["gt"] = gt_path
            break

    # Find segmentation
    if seg_dir:
        seg_patterns = [
            f"{patient_id}-seg.nii.gz",
            f"{patient_id}_seg.nii.gz",
            f"{patient_id}-seg.nii",
        ]
        for pattern in seg_patterns:
            seg_path = seg_dir / patient_id / pattern
            if seg_path.exists():
                files["seg"] = seg_path
                break
            seg_path = seg_dir / pattern
            if seg_path.exists():
                files["seg"] = seg_path
                break

    return files


def run_clinical_validation(
    results_dirs: Dict[str, Path],
    gt_dir: Path,
    seg_dir: Optional[Path] = None,
    spacings: List[str] = None,
    pulses: List[str] = None,
    patient_ids: Optional[List[str]] = None,
) -> Dict:
    """
    Run clinical validation across multiple methods and patients.

    Args:
        results_dirs: Mapping of method name -> results directory
        gt_dir: Ground truth directory
        seg_dir: Segmentation directory
        spacings: List of spacing values to evaluate
        pulses: List of pulse sequences to evaluate
        patient_ids: Optional list of patient IDs (auto-discovered if None)

    Returns:
        Dict with per-method, per-ROI metrics
    """
    if spacings is None:
        spacings = ["3mm", "5mm", "7mm"]
    if pulses is None:
        pulses = ["t1c", "t2w", "t2f"]

    # Discover patient IDs if not provided
    if patient_ids is None:
        patient_ids = set()
        for method_dir in results_dirs.values():
            for nii_path in method_dir.rglob("*.nii.gz"):
                # Extract patient ID from filename
                name = nii_path.stem.replace(".nii", "")
                parts = name.rsplit("-", 1)
                if len(parts) == 2:
                    patient_ids.add(parts[0])
        patient_ids = sorted(patient_ids)

    print(f"Found {len(patient_ids)} patients")

    all_results = {method: {} for method in results_dirs}

    for method, method_dir in results_dirs.items():
        print(f"\nEvaluating {method}...")

        for spacing in spacings:
            spacing_dir = method_dir / spacing

            if not spacing_dir.exists():
                spacing_dir = method_dir  # Flat structure

            for pulse in pulses:
                key = f"{spacing}_{pulse}"
                all_results[method][key] = {"rois": {}, "patients": []}

                for patient_id in patient_ids:
                    files = find_matching_files(
                        patient_id, pulse, spacing_dir, gt_dir, seg_dir
                    )

                    if files["pred"] is None or files["gt"] is None:
                        continue

                    try:
                        pred = load_nifti(files["pred"])
                        gt = load_nifti(files["gt"])

                        seg = None
                        if files["seg"]:
                            seg = load_nifti(files["seg"]).astype(np.int32)

                        roi_metrics = compute_roi_metrics(pred, gt, seg)

                        for roi, metrics in roi_metrics.items():
                            if roi not in all_results[method][key]["rois"]:
                                all_results[method][key]["rois"][roi] = {
                                    "psnr": [], "ssim": []
                                }
                            all_results[method][key]["rois"][roi]["psnr"].append(metrics["psnr"])
                            all_results[method][key]["rois"][roi]["ssim"].append(metrics["ssim"])

                        all_results[method][key]["patients"].append(patient_id)

                    except Exception as e:
                        print(f"  Error with {patient_id}/{pulse}: {e}")
                        continue

    # Aggregate statistics
    summary = {}
    for method in results_dirs:
        summary[method] = {}
        for key in all_results[method]:
            summary[method][key] = {}
            for roi, metrics in all_results[method][key]["rois"].items():
                psnr_vals = [v for v in metrics["psnr"] if not np.isnan(v)]
                ssim_vals = [v for v in metrics["ssim"] if not np.isnan(v)]

                summary[method][key][roi] = {
                    "psnr_mean": float(np.mean(psnr_vals)) if psnr_vals else None,
                    "psnr_std": float(np.std(psnr_vals)) if psnr_vals else None,
                    "ssim_mean": float(np.mean(ssim_vals)) if ssim_vals else None,
                    "ssim_std": float(np.std(ssim_vals)) if ssim_vals else None,
                    "n_patients": len(psnr_vals),
                }

    return {"raw": all_results, "summary": summary}


def run_clinical_validation_demo(
    results_dir: Path,
    output_dir: Path,
) -> Dict:
    """
    Simplified clinical validation for demos.
    Analyzes a single results directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to find predictions in the results directory
    nii_files = list(results_dir.rglob("*.nii.gz"))

    if not nii_files:
        print(f"No NIfTI files found in {results_dir}")
        return {}

    print(f"Found {len(nii_files)} NIfTI files")

    # For demo, just compute basic statistics on the predictions
    results = {
        "n_files": len(nii_files),
        "file_list": [str(p) for p in nii_files[:10]],  # First 10
        "note": "Full clinical validation requires ground truth and segmentation data",
    }

    with open(output_dir / "clinical_validation_demo.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved demo results to {output_dir / 'clinical_validation_demo.json'}")
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical validation of SR methods with ROI-specific metrics"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing SR predictions"
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        help="Directory containing ground truth HR volumes"
    )
    parser.add_argument(
        "--seg-dir",
        type=Path,
        help="Directory containing segmentation files"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["PACS_SR"],
        help="Methods to evaluate (directories under results-dir)"
    )
    parser.add_argument(
        "--spacings",
        nargs="+",
        default=["3mm", "5mm", "7mm"],
        help="Spacing values to evaluate"
    )
    parser.add_argument(
        "--pulses",
        nargs="+",
        default=["t1c", "t2w", "t2f"],
        help="Pulse sequences to evaluate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("clinical_validation"),
        help="Output directory"
    )
    args = parser.parse_args()

    if args.gt_dir:
        # Full validation
        results_dirs = {}
        for method in args.methods:
            method_dir = args.results_dir / method
            if method_dir.exists():
                results_dirs[method] = method_dir
            elif args.results_dir.exists():
                results_dirs[method] = args.results_dir

        results = run_clinical_validation(
            results_dirs,
            args.gt_dir,
            args.seg_dir,
            args.spacings,
            args.pulses,
        )

        args.output.mkdir(parents=True, exist_ok=True)
        with open(args.output / "clinical_validation.json", "w") as f:
            json.dump(results["summary"], f, indent=2)

        print(f"\nResults saved to {args.output / 'clinical_validation.json'}")

        # Print summary
        print("\n" + "=" * 60)
        print("CLINICAL VALIDATION SUMMARY")
        print("=" * 60)
        for method in results["summary"]:
            print(f"\n{method}:")
            for key, rois in results["summary"][method].items():
                print(f"  {key}:")
                for roi, metrics in rois.items():
                    if metrics["psnr_mean"]:
                        print(f"    {roi}: PSNR={metrics['psnr_mean']:.2f}, SSIM={metrics['ssim_mean']:.4f}")

    else:
        # Demo mode
        run_clinical_validation_demo(args.results_dir, args.output)


if __name__ == "__main__":
    main()
