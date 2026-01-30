#!/usr/bin/env python
"""
Regional Specialization Analysis
=================================

Analyzes which SR experts excel in which brain regions, providing evidence
for PaCS-SR's explainability claims.

Key outputs:
- Per-ROI weight statistics (mean, std per expert per region)
- Specialization index (how much the ensemble relies on specific experts)
- Heatmaps showing expert-region correlations

Usage:
    python -m pacs_sr.experiments.regional_specialization \\
        --npz path/to/weights.npz \\
        --seg path/to/segmentation.nii.gz \\
        --output results/regional/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_weight_maps(npz_path: Path) -> Tuple[np.ndarray, List[str], dict]:
    """
    Load weight maps from NPZ file.

    Returns:
        weight_maps: 4D array (Z, Y, X, n_models)
        model_names: List of expert model names
        metadata: Dictionary with additional metadata
    """
    data = np.load(npz_path, allow_pickle=True)

    weight_maps = data.get("weight_maps")
    if weight_maps is None:
        weight_maps = data.get("weights")
    if weight_maps is None:
        raise ValueError(f"No weight_maps found in {npz_path}")

    model_names = data.get("model_names", ["Expert1", "Expert2", "Expert3", "Expert4"])
    if hasattr(model_names, 'tolist'):
        model_names = model_names.tolist()

    def _convert_value(v):
        """Convert numpy array to Python scalar or list."""
        if hasattr(v, 'item') and v.ndim == 0:
            return v.item()  # 0-d array (scalar)
        elif hasattr(v, 'tolist'):
            return v.tolist()  # Multi-element array
        return v

    metadata = {
        k: _convert_value(v)
        for k, v in data.items()
        if k not in ("weight_maps", "weights", "model_names")
    }

    return weight_maps, model_names, metadata


def load_segmentation(seg_path: Path) -> np.ndarray:
    """Load segmentation volume."""
    try:
        import nibabel as nib
        seg = nib.load(seg_path).get_fdata().astype(np.int32)
        return seg
    except ImportError:
        raise ImportError("nibabel required for loading segmentations")


def analyze_regional_weights(
    weight_maps: np.ndarray,
    model_names: List[str],
    segmentation: Optional[np.ndarray] = None,
    roi_labels: Optional[Dict[str, int]] = None,
) -> Dict:
    """
    Analyze weight distribution across brain regions.

    Args:
        weight_maps: 4D array (Z, Y, X, n_models)
        model_names: List of expert names
        segmentation: Optional 3D array with region labels
        roi_labels: Mapping of region names to label values

    Returns:
        Dictionary containing:
        - 'global_stats': Per-model statistics across entire volume
        - 'roi_stats': Per-model statistics per ROI (if segmentation provided)
        - 'dominant_counts': Count of voxels where each model dominates
        - 'specialization_index': Overall specialization measure
    """
    n_models = weight_maps.shape[-1]

    # Global statistics
    global_stats = {}
    for i, name in enumerate(model_names):
        w = weight_maps[..., i].flatten()
        w = w[~np.isnan(w)]  # Remove NaN values
        global_stats[name] = {
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "median": float(np.median(w)),
            "q25": float(np.percentile(w, 25)),
            "q75": float(np.percentile(w, 75)),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
        }

    # Dominant model counts
    dominant = np.argmax(weight_maps, axis=-1)
    dominant_counts = {}
    for i, name in enumerate(model_names):
        dominant_counts[name] = int(np.sum(dominant == i))

    # ROI-specific statistics (if segmentation provided)
    roi_stats = {}
    if segmentation is not None:
        if roi_labels is None:
            # Default ROI labels for brain tumors
            roi_labels = {
                "background": 0,
                "core": 1,
                "edema": 2,
                "surround": 3,
            }

        # Ensure shape compatibility
        if segmentation.shape != weight_maps.shape[:3]:
            # Attempt to resize segmentation
            from scipy.ndimage import zoom
            scale = np.array(weight_maps.shape[:3]) / np.array(segmentation.shape)
            segmentation = zoom(segmentation, scale, order=0)

        for roi_name, roi_label in roi_labels.items():
            if roi_label == 0:  # Skip background
                continue

            mask = segmentation == roi_label
            if not np.any(mask):
                continue

            roi_stats[roi_name] = {}
            for i, model_name in enumerate(model_names):
                w = weight_maps[..., i][mask]
                w = w[~np.isnan(w)]
                if len(w) == 0:
                    continue

                roi_stats[roi_name][model_name] = {
                    "mean": float(np.mean(w)),
                    "std": float(np.std(w)),
                    "n_voxels": int(len(w)),
                }

    # Specialization index
    specialization_index = compute_specialization_index(weight_maps)

    return {
        "global_stats": global_stats,
        "roi_stats": roi_stats,
        "dominant_counts": dominant_counts,
        "specialization_index": specialization_index,
    }


def compute_specialization_index(weight_maps: np.ndarray) -> float:
    """
    Compute a measure of how specialized the ensemble is.

    Higher values indicate more reliance on specific experts per region.
    A value of 1.0 means each voxel uses only one expert.
    A value of 0.0 means uniform weighting everywhere.

    Uses the average "concentration" of weights:
        SI = 1 - mean_entropy / max_entropy

    Returns:
        Specialization index in [0, 1]
    """
    n_models = weight_maps.shape[-1]
    max_entropy = np.log(n_models)

    if max_entropy == 0:
        return 1.0

    # Compute per-voxel entropy
    eps = 1e-8
    entropy = -np.sum(weight_maps * np.log(weight_maps + eps), axis=-1)

    # Normalize by max entropy
    normalized_entropy = entropy / max_entropy

    # Specialization = 1 - normalized_entropy
    mean_normalized_entropy = float(np.nanmean(normalized_entropy))

    return 1.0 - mean_normalized_entropy


def plot_regional_heatmap(
    results: Dict,
    model_names: List[str],
    output_path: Path,
) -> None:
    """
    Create heatmap showing average weights per ROI per expert.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping heatmap")
        return

    roi_stats = results.get("roi_stats", {})
    if not roi_stats:
        print("No ROI stats available for heatmap")
        return

    rois = list(roi_stats.keys())
    n_rois = len(rois)
    n_models = len(model_names)

    # Build weight matrix
    weight_matrix = np.zeros((n_rois, n_models))
    for i, roi in enumerate(rois):
        for j, model in enumerate(model_names):
            if model in roi_stats[roi]:
                weight_matrix[i, j] = roi_stats[roi][model]["mean"]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weight_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticks(range(n_rois))
    ax.set_yticklabels(rois)

    ax.set_xlabel("Expert Model")
    ax.set_ylabel("Brain Region")
    ax.set_title("Average Expert Weights by Region")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Weight")

    # Add text annotations
    for i in range(n_rois):
        for j in range(n_models):
            text = ax.text(j, i, f"{weight_matrix[i, j]:.2f}",
                           ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_path}")


def analyze_all_weights(
    weight_files: List[Path],
    output_dir: Path,
    seg_pattern: Optional[str] = None,
) -> Dict:
    """
    Analyze weights from multiple patients and aggregate results.

    Args:
        weight_files: List of paths to weight NPZ files
        output_dir: Directory for output files
        seg_pattern: Optional pattern to find corresponding segmentation files

    Returns:
        Aggregated analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_specialization = []

    for npz_path in weight_files:
        try:
            weight_maps, model_names, metadata = load_weight_maps(npz_path)

            # Try to find corresponding segmentation
            seg = None
            if seg_pattern:
                seg_path = Path(str(npz_path).replace("_weights_", "_seg_").replace(".npz", ".nii.gz"))
                if seg_path.exists():
                    seg = load_segmentation(seg_path)

            results = analyze_regional_weights(weight_maps, model_names, seg)
            results["patient_id"] = metadata.get("patient_id", npz_path.stem)
            all_results.append(results)
            all_specialization.append(results["specialization_index"])

        except Exception as e:
            print(f"Error processing {npz_path}: {e}")
            continue

    if not all_results:
        print("No results to aggregate")
        return {}

    # Aggregate global statistics
    aggregated = {
        "n_patients": len(all_results),
        "mean_specialization_index": float(np.mean(all_specialization)),
        "std_specialization_index": float(np.std(all_specialization)),
        "model_names": model_names,
        "global_stats_aggregated": {},
    }

    # Aggregate per-model stats
    for model in model_names:
        means = [r["global_stats"][model]["mean"] for r in all_results if model in r["global_stats"]]
        aggregated["global_stats_aggregated"][model] = {
            "mean_of_means": float(np.mean(means)),
            "std_of_means": float(np.std(means)),
        }

    # Save results
    with open(output_dir / "regional_analysis.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Plot aggregated heatmap if possible
    if all_results and "roi_stats" in all_results[0]:
        # Average ROI stats across patients
        avg_roi_stats = {}
        for roi in all_results[0]["roi_stats"]:
            avg_roi_stats[roi] = {}
            for model in model_names:
                means = [r["roi_stats"].get(roi, {}).get(model, {}).get("mean", 0) for r in all_results]
                avg_roi_stats[roi][model] = {"mean": float(np.mean(means))}

        aggregated_results = {"roi_stats": avg_roi_stats}
        plot_regional_heatmap(aggregated_results, model_names, output_dir / "regional_heatmap.png")

    print(f"Saved analysis to {output_dir / 'regional_analysis.json'}")
    return aggregated


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze regional specialization of PaCS-SR weights"
    )
    parser.add_argument(
        "--npz",
        type=Path,
        help="Path to single weight NPZ file"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory containing weight NPZ files"
    )
    parser.add_argument(
        "--seg",
        type=Path,
        help="Path to segmentation file (for single NPZ)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("regional_analysis"),
        help="Output directory"
    )
    args = parser.parse_args()

    if args.npz:
        # Single file analysis
        weight_maps, model_names, metadata = load_weight_maps(args.npz)

        seg = None
        if args.seg and args.seg.exists():
            seg = load_segmentation(args.seg)

        results = analyze_regional_weights(weight_maps, model_names, seg)
        results["patient_id"] = metadata.get("patient_id", args.npz.stem)

        args.output.mkdir(parents=True, exist_ok=True)
        with open(args.output / "analysis.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"Specialization Index: {results['specialization_index']:.3f}")
        print("\nDominant Model Counts:")
        for model, count in results["dominant_counts"].items():
            print(f"  {model}: {count}")

    elif args.dir:
        # Directory analysis
        weight_files = list(args.dir.rglob("*_weights_*.npz"))
        if not weight_files:
            print(f"No weight files found in {args.dir}")
            return

        print(f"Found {len(weight_files)} weight files")
        results = analyze_all_weights(weight_files, args.output)
        print(f"\nMean Specialization Index: {results['mean_specialization_index']:.3f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
