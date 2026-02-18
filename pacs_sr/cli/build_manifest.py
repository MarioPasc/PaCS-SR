#!/usr/bin/env python
"""CLI entry point for building K-fold CV manifest.

Reads the configuration file and generates a K-fold cross-validation manifest
JSON file containing train/test splits with patient IDs. HDF5 keys are
constructed at runtime from these IDs.
"""

import argparse
import json
import sys
from pathlib import Path

from pacs_sr.config.config import load_full_config
from pacs_sr.data.folds_builder import BuilderConfig, build_manifest, setup_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build K-fold CV manifest from configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  pacs-sr-build-manifest --config configs/seram_glioma.yaml

This will:
  1. Read the data section from config.yaml
  2. Scan HDF5 files for patients with complete coverage
  3. Generate K-fold splits
  4. Save manifest JSON to path specified in config.yaml (data.out)
""",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan HDF5 files and report statistics without writing manifest",
    )
    return parser.parse_args()


def main():
    """Main entry point for build-manifest CLI."""
    args = parse_args()

    setup_logging()

    print(f"Loading configuration from: {args.config}")
    try:
        full_config = load_full_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    data_config = full_config.data

    builder_config = BuilderConfig(
        source_h5=data_config.source_h5,
        experts_dir=data_config.experts_dir,
        spacings=data_config.spacings,
        pulses=data_config.pulses,
        models=data_config.models,
        k_folds=data_config.kfolds,
        seed=data_config.seed,
    )

    print("\nBuilding K-fold manifest...")
    print(f"  Source HDF5: {builder_config.source_h5}")
    print(f"  Experts dir: {builder_config.experts_dir}")
    print(f"  Spacings:    {', '.join(builder_config.spacings)}")
    print(f"  Pulses:      {', '.join(builder_config.pulses)}")
    print(f"  Models:      {', '.join(builder_config.models)}")
    print(f"  K-folds:     {builder_config.k_folds}")
    print(f"  Seed:        {builder_config.seed}")
    print()

    try:
        manifest = build_manifest(builder_config)
    except Exception as e:
        print(f"Error building manifest: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nManifest Statistics:")
    print("=" * 60)
    for fold_idx, fold_data in enumerate(manifest["folds"], start=1):
        n_train = len(fold_data["train"])
        n_test = len(fold_data["test"])
        print(f"Fold {fold_idx}: {n_train} train, {n_test} test patients")
    print("=" * 60)

    if args.dry_run:
        print("\nDry-run mode: manifest not saved")
    else:
        out_path = data_config.out
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        print(f"\nManifest saved to: {out_path}")
        print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")

    print("\nBuild manifest completed successfully")


if __name__ == "__main__":
    main()
