#!/usr/bin/env python
"""
PaCS-SR Demo Script
===================
Simplified entry points for medical congress reproducibility.

Usage:
    python run_demo.py --mode train [--fold 1] [--spacing 3mm] [--pulse t1c]
    python run_demo.py --mode analyze [--results-dir path/to/results]
    python run_demo.py --mode visualize [--results-dir path/to/results]
    python run_demo.py --mode quick-start  # Full demo pipeline

Modes:
    train      - Train PaCS-SR model (wraps pacs-sr-train)
    analyze    - Run regional specialization and clinical validation analysis
    visualize  - Generate weight maps, entropy maps, and figures
    quick-start - Complete demo: train -> analyze -> visualize

For detailed documentation, see CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PaCS-SR Demo - Simplified entry points for reproducibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on one fold with specific settings
  python run_demo.py --mode train --fold 1 --spacing 3mm --pulse t1c

  # Analyze trained results
  python run_demo.py --mode analyze --results-dir ./results/PaCS_SR

  # Generate visualizations
  python run_demo.py --mode visualize --results-dir ./results/PaCS_SR

  # Run complete demo pipeline
  python run_demo.py --mode quick-start --config configs/demo_config.yaml
"""
    )
    parser.add_argument(
        "--mode",
        choices=["train", "analyze", "visualize", "quick-start"],
        required=True,
        help="Operation mode"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/demo_config.yaml"),
        help="Path to configuration file (default: configs/demo_config.yaml)"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold number to train (default: 1)"
    )
    parser.add_argument(
        "--spacing",
        type=str,
        default="3mm",
        help="Resolution spacing (default: 3mm)"
    )
    parser.add_argument(
        "--pulse",
        type=str,
        default="t1c",
        help="MRI pulse sequence (default: t1c)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing PaCS-SR results for analysis/visualization"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_output"),
        help="Output directory for demo results (default: demo_output)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit number of patients to visualize (default: 5)"
    )
    return parser.parse_args()


def run_train_demo(args):
    """
    Train PaCS-SR model with demo settings.
    Wraps the pacs-sr-train CLI with sensible defaults.
    """
    print("\n" + "=" * 60)
    print("PaCS-SR TRAINING DEMO")
    print("=" * 60)

    # Check config exists
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        print("Please create a config file or specify --config path", file=sys.stderr)
        sys.exit(1)

    # Import and run training
    from pacs_sr.config.config import load_full_config
    from pacs_sr.model.model import PatchwiseConvexStacker

    print(f"Config:  {args.config}")
    print(f"Fold:    {args.fold}")
    print(f"Spacing: {args.spacing}")
    print(f"Pulse:   {args.pulse}")
    print("-" * 60)

    # Load configuration
    full_config = load_full_config(args.config)
    pacs_sr_config = full_config.pacs_sr
    data_config = full_config.data

    # Load manifest
    manifest_path = data_config.out
    if not Path(manifest_path).exists():
        print(f"Error: Manifest not found: {manifest_path}", file=sys.stderr)
        print("Run 'pacs-sr-build-manifest --config <config>' first", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r") as f:
        full_manifest = json.load(f)

    fold_idx = args.fold - 1
    if fold_idx >= len(full_manifest["folds"]):
        print(f"Error: Fold {args.fold} not found (only {len(full_manifest['folds'])} folds)", file=sys.stderr)
        sys.exit(1)

    fold_data = full_manifest["folds"][fold_idx]

    print(f"Train patients: {len(fold_data['train'])}")
    print(f"Test patients:  {len(fold_data['test'])}")
    print("-" * 60)

    # Create and train model
    model = PatchwiseConvexStacker(pacs_sr_config, fold_num=args.fold)

    import time
    start_time = time.time()

    print(f"\nTraining {args.spacing} / {args.pulse}...")
    weights = model.fit_one(fold_data, args.spacing, args.pulse)
    print(f"Trained {len(weights)} regions")

    print("\nEvaluating...")
    results = model.evaluate_split(fold_data, args.spacing, args.pulse)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if 'train' in results and results['train']:
        print(f"Train - PSNR: {results['train']['psnr']:.4f}, SSIM: {results['train']['ssim']:.4f}")
    if 'test' in results and results['test']:
        print(f"Test  - PSNR: {results['test']['psnr']:.4f}, SSIM: {results['test']['ssim']:.4f}")
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Results saved to: {pacs_sr_config.out_root}")

    model.logger.log_session_end()
    return pacs_sr_config.out_root


def run_analyze_demo(args):
    """
    Run regional specialization and clinical validation analysis.
    """
    print("\n" + "=" * 60)
    print("PaCS-SR ANALYSIS DEMO")
    print("=" * 60)

    results_dir = args.results_dir
    if results_dir is None:
        # Try to infer from config
        if args.config.exists():
            from pacs_sr.config.config import load_full_config
            full_config = load_full_config(args.config)
            results_dir = Path(full_config.pacs_sr.out_root)
        else:
            print("Error: Please specify --results-dir or --config", file=sys.stderr)
            sys.exit(1)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {args.output_dir}")
    print("-" * 60)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find weight NPZ files
    weight_files = list(results_dir.rglob("*_weights_*.npz"))
    print(f"Found {len(weight_files)} weight map files")

    if len(weight_files) == 0:
        print("No weight files found. Run training first.", file=sys.stderr)
        sys.exit(1)

    # Run regional analysis
    print("\n[1/2] Regional Specialization Analysis...")
    try:
        from pacs_sr.experiments.regional_specialization import analyze_all_weights
        regional_results = analyze_all_weights(weight_files, args.output_dir / "regional")
        print(f"  Saved regional analysis to {args.output_dir / 'regional'}")
    except ImportError:
        print("  (regional_specialization module not available - skipping)")
        regional_results = None
    except Exception as e:
        print(f"  Warning: Regional analysis failed: {e}")
        regional_results = None

    # Run clinical validation (if segmentations available)
    print("\n[2/2] Clinical Validation Analysis...")
    try:
        from pacs_sr.experiments.clinical_validation import run_clinical_validation_demo
        clinical_results = run_clinical_validation_demo(results_dir, args.output_dir / "clinical")
        print(f"  Saved clinical validation to {args.output_dir / 'clinical'}")
    except ImportError:
        print("  (clinical_validation module not available - skipping)")
        clinical_results = None
    except Exception as e:
        print(f"  Warning: Clinical validation failed: {e}")
        clinical_results = None

    print("\n" + "=" * 60)
    print(f"Analysis complete! Results in: {args.output_dir}")
    print("=" * 60)

    return args.output_dir


def run_visualize_demo(args):
    """
    Generate weight maps, entropy maps, and figures.
    """
    print("\n" + "=" * 60)
    print("PaCS-SR VISUALIZATION DEMO")
    print("=" * 60)

    results_dir = args.results_dir
    if results_dir is None:
        if args.config.exists():
            from pacs_sr.config.config import load_full_config
            full_config = load_full_config(args.config)
            results_dir = Path(full_config.pacs_sr.out_root)
        else:
            print("Error: Please specify --results-dir or --config", file=sys.stderr)
            sys.exit(1)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Max patients: {args.limit}")
    print("-" * 60)

    # Create output directory
    viz_dir = args.output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Find weight NPZ files
    weight_files = list(results_dir.rglob("*_weights_*.npz"))[:args.limit]
    print(f"Visualizing {len(weight_files)} patients...")

    if len(weight_files) == 0:
        print("No weight files found. Run training first.", file=sys.stderr)
        sys.exit(1)

    import numpy as np

    for i, npz_path in enumerate(weight_files, 1):
        print(f"\n[{i}/{len(weight_files)}] {npz_path.stem}")

        try:
            data = np.load(npz_path, allow_pickle=True)
            weight_maps = data.get("weight_maps") or data.get("weights")

            if weight_maps is None:
                print(f"  Warning: No weight_maps found in {npz_path}")
                continue

            model_names = data.get("model_names", ["Expert1", "Expert2", "Expert3", "Expert4"])
            if hasattr(model_names, 'tolist'):
                model_names = model_names.tolist()

            # Create simple visualizations
            patient_dir = viz_dir / npz_path.stem
            patient_dir.mkdir(exist_ok=True)

            # Plot middle slices for each expert
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                mid_z = weight_maps.shape[0] // 2

                # Weight maps per expert
                fig, axes = plt.subplots(1, len(model_names), figsize=(4*len(model_names), 4))
                if len(model_names) == 1:
                    axes = [axes]

                for j, name in enumerate(model_names):
                    im = axes[j].imshow(weight_maps[mid_z, :, :, j], cmap='viridis', vmin=0, vmax=1)
                    axes[j].set_title(f"{name} weights")
                    axes[j].axis('off')
                    plt.colorbar(im, ax=axes[j], fraction=0.046)

                plt.tight_layout()
                plt.savefig(patient_dir / "weight_maps.png", dpi=150, bbox_inches='tight')
                plt.close()

                # Entropy map
                eps = 1e-8
                entropy = -np.sum(weight_maps * np.log(weight_maps + eps), axis=-1)

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(entropy[mid_z], cmap='hot')
                ax.set_title("Entropy Map (lower = more confident)")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
                plt.savefig(patient_dir / "entropy_map.png", dpi=150, bbox_inches='tight')
                plt.close()

                # Dominant model map
                dominant = np.argmax(weight_maps, axis=-1)

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(dominant[mid_z], cmap='tab10', vmin=0, vmax=len(model_names)-1)
                ax.set_title("Dominant Expert")
                ax.axis('off')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046)
                cbar.set_ticks(range(len(model_names)))
                cbar.set_ticklabels(model_names)
                plt.savefig(patient_dir / "dominant_model.png", dpi=150, bbox_inches='tight')
                plt.close()

                print(f"  Saved to {patient_dir}")

            except ImportError:
                print("  Warning: matplotlib not available, skipping plots")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print("\n" + "=" * 60)
    print(f"Visualization complete! Results in: {viz_dir}")
    print("=" * 60)

    return viz_dir


def run_quick_start_demo(args):
    """
    Run complete demo pipeline: train -> analyze -> visualize.
    """
    print("\n" + "=" * 60)
    print("PaCS-SR QUICK START DEMO")
    print("=" * 60)
    print("This will run the complete pipeline:")
    print("  1. Train model (fold 1, 3mm, t1c)")
    print("  2. Analyze results (regional specialization)")
    print("  3. Generate visualizations")
    print("=" * 60)

    # Step 1: Train
    print("\n>>> STEP 1/3: Training...")
    results_dir = run_train_demo(args)

    # Update results_dir for subsequent steps
    args.results_dir = Path(results_dir)

    # Step 2: Analyze
    print("\n>>> STEP 2/3: Analyzing...")
    run_analyze_demo(args)

    # Step 3: Visualize
    print("\n>>> STEP 3/3: Visualizing...")
    run_visualize_demo(args)

    print("\n" + "=" * 60)
    print("QUICK START COMPLETE!")
    print("=" * 60)
    print(f"Training results: {results_dir}")
    print(f"Analysis output:  {args.output_dir}")
    print("\nNext steps:")
    print("  - Review weight maps in demo_output/visualizations/")
    print("  - Check metrics in the model_data/ directories")
    print("  - See CLAUDE.md for detailed documentation")
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    if args.mode == "train":
        run_train_demo(args)
    elif args.mode == "analyze":
        run_analyze_demo(args)
    elif args.mode == "visualize":
        run_visualize_demo(args)
    elif args.mode == "quick-start":
        run_quick_start_demo(args)


if __name__ == "__main__":
    main()
