#!/usr/bin/env python
"""
CLI entry point for training PaCS-SR model.

This script reads the configuration file, loads the K-fold manifest,
and trains the PaCS-SR model for all spacings and pulses across all folds.
"""

import argparse
import json
import sys
from pathlib import Path

from pacs_sr.config.config import load_full_config
from pacs_sr.model.model import PatchwiseConvexStacker


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PaCS-SR model from configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  pacs-sr-train --config configs/config.yaml

  # Train specific fold only
  pacs-sr-train --config configs/config.yaml --fold 1

  # Train specific spacing and pulse
  pacs-sr-train --config configs/config.yaml --spacing 3mm --pulse t1c

This will:
  1. Load configuration and K-fold manifest
  2. Train PaCS-SR for each fold, spacing, and pulse
  3. Evaluate on train and test sets
  4. Save weights, metrics, and predictions
"""
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Train specific fold only (1-indexed). If not specified, trains all folds."
    )
    parser.add_argument(
        "--spacing",
        type=str,
        default=None,
        help="Train specific spacing only (e.g., '3mm'). If not specified, trains all spacings."
    )
    parser.add_argument(
        "--pulse",
        type=str,
        default=None,
        help="Train specific pulse only (e.g., 't1c'). If not specified, trains all pulses."
    )
    return parser.parse_args()


def main():
    """Main entry point for train CLI."""
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    try:
        full_config = load_full_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    data_config = full_config.data
    pacs_sr_config = full_config.pacs_sr

    # Load manifest
    manifest_path = data_config.out
    print(f"Loading K-fold manifest from: {manifest_path}")
    try:
        with open(manifest_path, "r") as f:
            full_manifest = json.load(f)
    except Exception as e:
        print(f"Error loading manifest: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine which folds/spacings/pulses to train
    all_folds = list(range(len(full_manifest["folds"])))
    folds_to_train = [args.fold - 1] if args.fold is not None else all_folds

    spacings_to_train = [args.spacing] if args.spacing is not None else list(pacs_sr_config.spacings)
    pulses_to_train = [args.pulse] if args.pulse is not None else list(pacs_sr_config.pulses)

    print("\nTraining Configuration:")
    print("=" * 80)
    print(f"Experiment:  {pacs_sr_config.experiment_name}")
    print(f"Folds:       {[f+1 for f in folds_to_train]}")
    print(f"Spacings:    {spacings_to_train}")
    print(f"Pulses:      {pulses_to_train}")
    print(f"Patch size:  {pacs_sr_config.patch_size}")
    print(f"Stride:      {pacs_sr_config.stride}")
    print(f"Logging:     {'SLURM-friendly' if pacs_sr_config.disable_tqdm else 'tqdm progress bars'}")
    print(f"Output root: {pacs_sr_config.out_root}")
    print("=" * 80)

    # Train each fold
    total_tasks = len(folds_to_train) * len(spacings_to_train) * len(pulses_to_train)
    task_idx = 0

    for fold_idx in folds_to_train:
        fold_data = full_manifest["folds"][fold_idx]
        fold_num = fold_idx + 1

        print(f"\n{'=' * 80}")
        print(f"FOLD {fold_num}/{len(full_manifest['folds'])}")
        print(f"{'=' * 80}")
        print(f"Train patients: {len(fold_data['train'])}")
        print(f"Test patients:  {len(fold_data['test'])}")

        # Create model for this fold
        # Pass fold_num to model for proper directory structure
        model = PatchwiseConvexStacker(pacs_sr_config, fold_num=fold_num)

        # Train for each spacing and pulse
        import time
        for spacing in spacings_to_train:
            for pulse in pulses_to_train:
                task_idx += 1
                task_start_time = time.time()
                print(f"\n[Task {task_idx}/{total_tasks}] Training: Fold {fold_num} | {spacing} | {pulse}")

                try:
                    # Train
                    weights = model.fit_one(fold_data, spacing, pulse)
                    task_elapsed = time.time() - task_start_time
                    print(f"  ✓ Trained {len(weights)} regions in {task_elapsed:.1f}s")

                    # Evaluate
                    eval_start = time.time()
                    results = model.evaluate_split(fold_data, spacing, pulse)
                    eval_elapsed = time.time() - eval_start
                    print(f"  ✓ Evaluation complete in {eval_elapsed:.1f}s")
                    if 'train' in results and results['train']:
                        print(f"    Train - PSNR: {results['train']['psnr']:.4f}, SSIM: {results['train']['ssim']:.4f}")
                    if 'test' in results and results['test']:
                        print(f"    Test  - PSNR: {results['test']['psnr']:.4f}, SSIM: {results['test']['ssim']:.4f}")

                    # Report total task time
                    total_task_time = time.time() - task_start_time
                    print(f"  ✓ Total task time: {total_task_time/60:.1f}min")

                except Exception as e:
                    print(f"  ✗ Error: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                    continue

        # Log session end for this fold
        model.logger.log_session_end()

    print(f"\n{'=' * 80}")
    print("✓ Training completed successfully")
    print(f"{'=' * 80}")
    print(f"Results saved to: {pacs_sr_config.out_root}")


if __name__ == "__main__":
    main()
