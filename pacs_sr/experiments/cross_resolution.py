#!/usr/bin/env python
"""
Cross-Resolution Generalization Experiment
==========================================

Tests whether PaCS-SR weights learned at one resolution generalize to others.

Protocol:
1. Train weights on source spacing (e.g., 3mm)
2. Apply same weights to target spacings (e.g., 5mm, 7mm)
3. Compare: transferred weights vs. directly trained weights

This tests the hypothesis that expert preferences are consistent across
different levels of clinical anisotropy.

Usage:
    python -m pacs_sr.experiments.cross_resolution \\
        --config configs/config.yaml \\
        --train-spacing 3mm \\
        --test-spacings 5mm 7mm \\
        --output results/generalization/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def compute_generalization_gap(
    performance_direct: float,
    performance_transfer: float,
) -> float:
    """
    Compute the generalization gap between direct and transferred weights.

    Gap = (direct - transfer) / direct * 100

    Positive gap means direct training is better (expected).
    Negative gap means transferred weights are better (surprising but possible).

    Returns:
        Gap as percentage
    """
    if performance_direct == 0:
        return float('nan')
    return (performance_direct - performance_transfer) / performance_direct * 100


def load_weights_from_json(weights_path: Path) -> Dict[int, np.ndarray]:
    """Load weight dictionary from JSON file."""
    with open(weights_path, "r") as f:
        weights_raw = json.load(f)

    # Convert to proper format
    weights = {}
    for key, value in weights_raw.items():
        if isinstance(key, str):
            key = int(key)
        weights[key] = np.array(value, dtype=np.float32)

    return weights


def apply_weights_to_spacing(
    weights: Dict[int, np.ndarray],
    fold_data: Dict,
    spacing: str,
    pulse: str,
    config,
) -> Tuple[float, float]:
    """
    Apply pre-trained weights to a different spacing and compute metrics.

    Returns:
        (psnr, ssim) tuple
    """
    from pacs_sr.model.model import PatchwiseConvexStacker

    # Create model with transferred weights
    model = PatchwiseConvexStacker(config)
    model.weights = weights

    # Evaluate using the model's evaluation method
    try:
        results = model.evaluate_split(fold_data, spacing, pulse)
        test_metrics = results.get("test", {})
        return test_metrics.get("psnr", float('nan')), test_metrics.get("ssim", float('nan'))
    except Exception as e:
        print(f"Error evaluating: {e}")
        return float('nan'), float('nan')


def train_and_transfer(
    config_path: Path,
    train_spacing: str,
    test_spacings: List[str],
    pulses: List[str] = None,
    folds: List[int] = None,
) -> Dict:
    """
    Train on one spacing and test transfer to others.

    Args:
        config_path: Path to configuration file
        train_spacing: Spacing to train on (source)
        test_spacings: Spacings to transfer to (targets)
        pulses: Pulse sequences to evaluate
        folds: Folds to use (default: all)

    Returns:
        Dictionary with transfer performance results
    """
    from pacs_sr.config.config import load_full_config
    from pacs_sr.model.model import PatchwiseConvexStacker

    # Load configuration
    full_config = load_full_config(config_path)
    pacs_sr_config = full_config.pacs_sr
    data_config = full_config.data

    if pulses is None:
        pulses = list(pacs_sr_config.pulses)

    # Load manifest
    manifest_path = data_config.out
    with open(manifest_path, "r") as f:
        full_manifest = json.load(f)

    if folds is None:
        folds = list(range(len(full_manifest["folds"])))

    results = {
        "train_spacing": train_spacing,
        "test_spacings": test_spacings,
        "experiments": [],
    }

    for fold_idx in folds:
        fold_data = full_manifest["folds"][fold_idx]
        fold_num = fold_idx + 1

        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_num}")
        print(f"{'=' * 60}")

        for pulse in pulses:
            print(f"\nPulse: {pulse}")
            print("-" * 40)

            # Step 1: Train on source spacing
            print(f"  Training on {train_spacing}...")
            model = PatchwiseConvexStacker(pacs_sr_config, fold_num=fold_num)

            try:
                source_weights = model.fit_one(fold_data, train_spacing, pulse)
                source_results = model.evaluate_split(fold_data, train_spacing, pulse)
                source_psnr = source_results.get("test", {}).get("psnr", float('nan'))
                source_ssim = source_results.get("test", {}).get("ssim", float('nan'))
                print(f"    Source ({train_spacing}): PSNR={source_psnr:.4f}, SSIM={source_ssim:.4f}")
            except Exception as e:
                print(f"    Error training source: {e}")
                continue

            experiment = {
                "fold": fold_num,
                "pulse": pulse,
                "source_spacing": train_spacing,
                "source_psnr": source_psnr,
                "source_ssim": source_ssim,
                "transfers": [],
            }

            # Step 2: Apply to target spacings
            for target_spacing in test_spacings:
                if target_spacing == train_spacing:
                    continue

                print(f"  Transferring to {target_spacing}...")

                # Direct training on target
                try:
                    direct_model = PatchwiseConvexStacker(pacs_sr_config, fold_num=fold_num)
                    direct_model.fit_one(fold_data, target_spacing, pulse)
                    direct_results = direct_model.evaluate_split(fold_data, target_spacing, pulse)
                    direct_psnr = direct_results.get("test", {}).get("psnr", float('nan'))
                    direct_ssim = direct_results.get("test", {}).get("ssim", float('nan'))
                except Exception as e:
                    print(f"    Error direct training: {e}")
                    direct_psnr, direct_ssim = float('nan'), float('nan')

                # Transfer weights from source
                try:
                    transfer_model = PatchwiseConvexStacker(pacs_sr_config, fold_num=fold_num)
                    transfer_model.weights = source_weights
                    transfer_results = transfer_model.evaluate_split(fold_data, target_spacing, pulse)
                    transfer_psnr = transfer_results.get("test", {}).get("psnr", float('nan'))
                    transfer_ssim = transfer_results.get("test", {}).get("ssim", float('nan'))
                except Exception as e:
                    print(f"    Error transfer eval: {e}")
                    transfer_psnr, transfer_ssim = float('nan'), float('nan')

                # Compute gaps
                psnr_gap = compute_generalization_gap(direct_psnr, transfer_psnr)
                ssim_gap = compute_generalization_gap(direct_ssim, transfer_ssim)

                print(f"    Direct:   PSNR={direct_psnr:.4f}, SSIM={direct_ssim:.4f}")
                print(f"    Transfer: PSNR={transfer_psnr:.4f}, SSIM={transfer_ssim:.4f}")
                print(f"    Gap:      PSNR={psnr_gap:.2f}%, SSIM={ssim_gap:.2f}%")

                experiment["transfers"].append({
                    "target_spacing": target_spacing,
                    "direct_psnr": direct_psnr,
                    "direct_ssim": direct_ssim,
                    "transfer_psnr": transfer_psnr,
                    "transfer_ssim": transfer_ssim,
                    "psnr_gap_percent": psnr_gap,
                    "ssim_gap_percent": ssim_gap,
                })

            results["experiments"].append(experiment)

    return results


def plot_generalization_matrix(
    results: Dict,
    output_path: Path,
    metric: str = "psnr",
) -> None:
    """
    Create matrix plot showing generalization performance.

    Rows: Source spacing (where weights were learned)
    Cols: Target spacing (where weights were applied)
    Values: Performance (color-coded)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    # Extract unique spacings
    source_spacing = results["train_spacing"]
    target_spacings = results["test_spacings"]
    all_spacings = [source_spacing] + [s for s in target_spacings if s != source_spacing]

    n = len(all_spacings)
    matrix = np.full((n, n), np.nan)

    # Fill diagonal with direct performance
    # Fill off-diagonal with transfer performance
    for exp in results["experiments"]:
        source_idx = all_spacings.index(exp["source_spacing"])

        # Source performance on diagonal
        source_val = exp.get(f"source_{metric}", np.nan)
        if not np.isnan(source_val):
            matrix[source_idx, source_idx] = source_val

        # Transfer performance
        for transfer in exp["transfers"]:
            target_idx = all_spacings.index(transfer["target_spacing"])
            transfer_val = transfer.get(f"transfer_{metric}", np.nan)
            if not np.isnan(transfer_val):
                matrix[source_idx, target_idx] = transfer_val

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(range(n))
    ax.set_xticklabels(all_spacings)
    ax.set_yticks(range(n))
    ax.set_yticklabels(all_spacings)

    ax.set_xlabel("Target Spacing (Applied)")
    ax.set_ylabel("Source Spacing (Trained)")
    ax.set_title(f"Cross-Resolution Generalization ({metric.upper()})")

    plt.colorbar(im, ax=ax, label=metric.upper())

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                               ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved generalization matrix to {output_path}")


def summarize_results(results: Dict) -> Dict:
    """Aggregate results across folds and pulses."""
    summary = {
        "train_spacing": results["train_spacing"],
        "test_spacings": results["test_spacings"],
        "aggregated": {},
    }

    # Collect all gaps
    for target in results["test_spacings"]:
        if target == results["train_spacing"]:
            continue

        psnr_gaps = []
        ssim_gaps = []

        for exp in results["experiments"]:
            for transfer in exp["transfers"]:
                if transfer["target_spacing"] == target:
                    if not np.isnan(transfer["psnr_gap_percent"]):
                        psnr_gaps.append(transfer["psnr_gap_percent"])
                    if not np.isnan(transfer["ssim_gap_percent"]):
                        ssim_gaps.append(transfer["ssim_gap_percent"])

        summary["aggregated"][target] = {
            "mean_psnr_gap_percent": float(np.mean(psnr_gaps)) if psnr_gaps else None,
            "std_psnr_gap_percent": float(np.std(psnr_gaps)) if psnr_gaps else None,
            "mean_ssim_gap_percent": float(np.mean(ssim_gaps)) if ssim_gaps else None,
            "std_ssim_gap_percent": float(np.std(ssim_gaps)) if ssim_gaps else None,
            "n_experiments": len(psnr_gaps),
        }

    return summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Test cross-resolution generalization of PaCS-SR weights"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--train-spacing",
        type=str,
        default="3mm",
        help="Spacing to train on (source)"
    )
    parser.add_argument(
        "--test-spacings",
        nargs="+",
        default=["5mm", "7mm"],
        help="Spacings to transfer to (targets)"
    )
    parser.add_argument(
        "--pulses",
        nargs="+",
        default=None,
        help="Pulse sequences to evaluate (default: all from config)"
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Folds to use (default: all)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generalization_results"),
        help="Output directory"
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Run experiment
    results = train_and_transfer(
        args.config,
        args.train_spacing,
        args.test_spacings,
        args.pulses,
        args.folds,
    )

    # Save raw results
    with open(args.output / "generalization_raw.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json.dump(results, f, indent=2, default=convert)

    # Summarize and save
    summary = summarize_results(results)
    with open(args.output / "generalization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    plot_generalization_matrix(results, args.output / "generalization_psnr.png", "psnr")
    plot_generalization_matrix(results, args.output / "generalization_ssim.png", "ssim")

    # Print summary
    print("\n" + "=" * 60)
    print("CROSS-RESOLUTION GENERALIZATION SUMMARY")
    print("=" * 60)
    print(f"Source: {args.train_spacing}")
    print("-" * 60)
    for target, stats in summary["aggregated"].items():
        print(f"\n{args.train_spacing} -> {target}:")
        if stats["mean_psnr_gap_percent"] is not None:
            print(f"  PSNR gap: {stats['mean_psnr_gap_percent']:.2f}% +/- {stats['std_psnr_gap_percent']:.2f}%")
        if stats["mean_ssim_gap_percent"] is not None:
            print(f"  SSIM gap: {stats['mean_ssim_gap_percent']:.2f}% +/- {stats['std_ssim_gap_percent']:.2f}%")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
