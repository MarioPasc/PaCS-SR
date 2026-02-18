#!/usr/bin/env python
"""CLI entry point for making predictions with trained PaCS-SR weights.

Loads saved weights and blends expert predictions from HDF5 files to generate
super-resolved images.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from pacs_sr.config.config import load_full_config
from pacs_sr.data.hdf5_io import (
    expert_h5_path,
    expert_key,
    read_volume,
    write_volume,
)
from pacs_sr.utils.patches import region_labels
from pacs_sr.utils.registration import apply_brain_mask


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Make predictions using trained PaCS-SR weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  pacs-sr-predict --config configs/seram_glioma.yaml --weights path/to/weights.json \\
      --patient-id BraTS-GLI-00033-100 --spacing 3mm --pulse t1c
""",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to weights file (JSON)"
    )
    parser.add_argument("--patient-id", type=str, required=True, help="Patient ID")
    parser.add_argument(
        "--spacing", type=str, required=True, help="Spacing (e.g., '3mm')"
    )
    parser.add_argument(
        "--pulse", type=str, required=True, help="Pulse sequence (e.g., 't1c')"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 path (default: write to results dir)",
    )
    parser.add_argument(
        "--patch-size", type=int, default=32, help="Patch size (default: 32)"
    )
    parser.add_argument("--stride", type=int, default=16, help="Stride (default: 16)")
    return parser.parse_args()


def load_weights(weights_path: Path) -> Dict[int, np.ndarray]:
    """Load weights from JSON file.

    Args:
        weights_path: Path to weights JSON file.

    Returns:
        Dictionary mapping region_id -> weight_vector.
    """
    with open(weights_path, "r") as f:
        weights_dict = json.load(f)
    return {int(k): np.array(v, dtype=np.float32) for k, v in weights_dict.items()}


def blend_predictions(
    expert_volumes: list[np.ndarray],
    weights_dict: Dict[int, np.ndarray],
    patch_size: int,
    stride: int,
    pulse: str | None = None,
    use_registration: bool = False,
    atlas_dir: Path | None = None,
) -> np.ndarray:
    """Blend expert predictions using learned regional weights.

    Args:
        expert_volumes: List of expert prediction arrays (in model order).
        weights_dict: Dict mapping region_id -> weight_vector.
        patch_size: Patch size used during training.
        stride: Stride used during training.
        pulse: Pulse sequence name (needed for brain masking).
        use_registration: Whether to apply brain masking.
        atlas_dir: Atlas directory containing brain mask.

    Returns:
        Blended prediction array.
    """
    volume_shape = expert_volumes[0].shape[:3]
    X = np.stack(expert_volumes, axis=0)  # (M, Z, Y, X)

    labels = region_labels(volume_shape, patch_size, stride)
    unique_regions = sorted(set(int(r) for r in np.unique(labels)))

    blended = np.zeros(volume_shape, dtype=np.float32)

    for region_id in unique_regions:
        sel = labels == region_id
        w = weights_dict.get(region_id)
        if w is None:
            w = np.ones(X.shape[0], dtype=np.float32) / X.shape[0]
        tile = np.tensordot(w.astype(np.float32), X[:, sel], axes=(0, 0))
        blended[sel] = tile

    if use_registration and atlas_dir is not None and pulse is not None:
        mask_cache: dict = {}
        blended = apply_brain_mask(blended, pulse, atlas_dir, mask_cache)

    return blended


def main():
    """Main entry point for predict CLI."""
    args = parse_args()

    print(f"Loading configuration from: {args.config}")
    try:
        full_config = load_full_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    data_config = full_config.data

    # Load weights
    print(f"Loading weights from: {args.weights}")
    weights_dict = load_weights(args.weights)
    print(f"  Loaded {len(weights_dict)} regional weight vectors")

    # Load expert predictions from HDF5
    print("\nLoading expert predictions from HDF5:")
    expert_volumes = []
    for model_name in data_config.models:
        h5_path = expert_h5_path(data_config.experts_dir, model_name)
        key = expert_key(args.spacing, args.patient_id, args.pulse)
        vol, affine = read_volume(h5_path, key)
        expert_volumes.append(vol)
        print(f"  {model_name}: shape={vol.shape}")

    # Blend
    print("\nBlending predictions...")
    blended = blend_predictions(
        expert_volumes=expert_volumes,
        weights_dict=weights_dict,
        patch_size=args.patch_size,
        stride=args.stride,
        pulse=args.pulse,
        use_registration=full_config.pacs_sr.use_registration,
        atlas_dir=full_config.pacs_sr.atlas_dir,
    )
    print(f"  Blended shape: {blended.shape}")

    # Save
    if args.output is not None:
        output_path = args.output
    else:
        output_path = Path(full_config.pacs_sr.out_root) / "predictions" / "blended.h5"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    key = f"{args.spacing}/{args.patient_id}/{args.pulse}"
    write_volume(output_path, key, blended, affine)
    print(f"\nSaved blended prediction to: {output_path} (key: {key})")


if __name__ == "__main__":
    main()
