#!/usr/bin/env python
"""
CLI entry point for making predictions with trained PaCS-SR weights.

This script loads saved weights and blends expert predictions to generate
super-resolved images.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import nibabel as nib

from pacs_sr.config.config import load_full_config
from pacs_sr.utils.patches import region_labels


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Make predictions using trained PaCS-SR weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  pacs-sr-predict --config configs/config.yaml

  # Predict with specific weights file
  pacs-sr-predict --config configs/config.yaml --weights path/to/weights.json

This will:
  1. Load trained weights from specified path
  2. Load expert predictions from paths in config
  3. Blend predictions using learned regional weights
  4. Save blended output to config.predict.out_root
"""
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to weights file (JSON or NPZ). Overrides config.predict.pacs_sr_weights_path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for blended prediction. Overrides config.predict.out_root"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size used during training (default: 16)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Stride used during training (default: 16)"
    )
    return parser.parse_args()


def load_weights(weights_path: Path) -> Dict[int, np.ndarray]:
    """
    Load weights from JSON or NPZ file.

    Args:
        weights_path: Path to weights file

    Returns:
        Dictionary mapping region_id -> weight_vector
    """
    if weights_path.suffix == ".json":
        with open(weights_path, "r") as f:
            weights_dict = json.load(f)
        # Convert string keys to int
        return {int(k): np.array(v, dtype=np.float32) for k, v in weights_dict.items()}
    elif weights_path.suffix == ".npz":
        # Load from NPZ (assuming it's a saved weights dict)
        data = np.load(weights_path, allow_pickle=True)
        if "weights_dict" in data:
            weights_dict = data["weights_dict"].item()
            return {int(k): np.array(v, dtype=np.float32) for k, v in weights_dict.items()}
        else:
            raise ValueError(f"NPZ file {weights_path} does not contain 'weights_dict' key")
    else:
        raise ValueError(f"Unsupported weights file format: {weights_path.suffix}")


def blend_predictions(
    expert_predictions: Dict[str, np.ndarray],
    weights_dict: Dict[int, np.ndarray],
    patch_size: int,
    stride: int,
    model_order: list
) -> np.ndarray:
    """
    Blend expert predictions using learned regional weights.

    Args:
        expert_predictions: Dict mapping model_name -> prediction_array
        weights_dict: Dict mapping region_id -> weight_vector
        patch_size: Patch size used during training
        stride: Stride used during training
        model_order: Order of models (must match weight vector order)

    Returns:
        Blended prediction array
    """
    # Get volume shape from first expert
    first_expert = next(iter(expert_predictions.values()))
    volume_shape = first_expert.shape[:3]

    # Stack expert predictions in correct order
    X = []
    for model_name in model_order:
        if model_name not in expert_predictions:
            raise ValueError(f"Expert prediction for '{model_name}' not found")
        X.append(expert_predictions[model_name])
    X = np.stack(X, axis=0)  # (M, Z, Y, X)

    # Create region labels
    labels = region_labels(volume_shape, patch_size, stride)
    unique_regions = sorted(set(int(r) for r in np.unique(labels)))

    # Blend per region
    blended = np.zeros(volume_shape, dtype=np.float32)

    for region_id in unique_regions:
        sel = labels == region_id

        # Get weights for this region
        if region_id in weights_dict:
            w = weights_dict[region_id]
        else:
            # Uniform weights if not found
            w = np.ones(X.shape[0], dtype=np.float32) / X.shape[0]

        # Linear blend: blend[sel] = sum_i w[i] * X[i, sel]
        tile = np.tensordot(w.astype(np.float32), X[:, sel], axes=(0, 0))
        blended[sel] = tile

    return blended


def main():
    """Main entry point for predict CLI."""
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    try:
        full_config = load_full_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    if full_config.predict is None:
        print("Error: No 'predict' section found in configuration", file=sys.stderr)
        sys.exit(1)

    predict_config = full_config.predict

    # Determine weights path
    weights_path = args.weights if args.weights is not None else predict_config.pacs_sr_weights_path
    weights_path = Path(weights_path)

    if not weights_path.exists():
        print(f"Error: Weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    # Load weights
    print(f"Loading weights from: {weights_path}")
    try:
        weights_dict = load_weights(weights_path)
        print(f"  ✓ Loaded {len(weights_dict)} regional weight vectors")
    except Exception as e:
        print(f"Error loading weights: {e}", file=sys.stderr)
        sys.exit(1)

    # Load expert predictions
    print("\nLoading expert predictions:")
    expert_predictions = {}
    model_order = []

    for expert_id, expert_config in predict_config.experts.items():
        model_name = expert_config.name
        opinion_path = expert_config.opinion_path

        if not opinion_path.exists():
            print(f"  ✗ Warning: Expert prediction not found: {opinion_path}", file=sys.stderr)
            continue

        try:
            img = nib.load(str(opinion_path))
            data = img.get_fdata(dtype=np.float32)
            expert_predictions[model_name] = data
            model_order.append(model_name)
            print(f"  ✓ {model_name}: {opinion_path.name} (shape: {data.shape})")
        except Exception as e:
            print(f"  ✗ Error loading {model_name}: {e}", file=sys.stderr)
            continue

    if not expert_predictions:
        print("Error: No expert predictions loaded", file=sys.stderr)
        sys.exit(1)

    # Get reference image for affine and header
    first_expert_config = next(iter(predict_config.experts.values()))
    ref_img = nib.load(str(first_expert_config.opinion_path))

    # Blend predictions
    print("\nBlending predictions...")
    try:
        blended = blend_predictions(
            expert_predictions=expert_predictions,
            weights_dict=weights_dict,
            patch_size=args.patch_size,
            stride=args.stride,
            model_order=model_order
        )
        print(f"  ✓ Blended shape: {blended.shape}")
    except Exception as e:
        print(f"Error blending predictions: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Determine output path
    if args.output is not None:
        output_path = args.output
    else:
        output_path = predict_config.out_root / "blended_prediction.nii.gz"

    # Save blended prediction
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        output_img = nib.Nifti1Image(blended, ref_img.affine, ref_img.header)
        nib.save(output_img, str(output_path))
        print(f"\n✓ Saved blended prediction to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"Error saving prediction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
