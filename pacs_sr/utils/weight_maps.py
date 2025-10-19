"""
Weight map utilities for PaCS-SR.

This module provides functions to convert per-region weight vectors into
full-volume weight maps and save them in NPZ format for efficient storage
and analysis.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def expand_weights_to_volume(
    region_labels: np.ndarray,
    weights_dict: Dict[int, np.ndarray],
    n_models: int
) -> np.ndarray:
    """
    Expand per-region weight vectors to full-volume weight maps.

    Takes a 3D array of region labels and a dictionary mapping region IDs
    to weight vectors, and creates a 4D array where each voxel contains
    the weight vector for its region.

    Args:
        region_labels: 3D array of shape (Z, Y, X) with region IDs
        weights_dict: Dictionary mapping region_id -> weight_vector
        n_models: Number of expert models (length of weight vectors)

    Returns:
        weight_maps: 4D array of shape (Z, Y, X, n_models) where
                     weight_maps[z, y, x, :] contains the weight vector
                     for the region at voxel (z, y, x)
    """
    Z, Y, X = region_labels.shape
    weight_maps = np.zeros((Z, Y, X, n_models), dtype=np.float32)

    # Get unique region IDs
    unique_regions = np.unique(region_labels)
    unique_regions = unique_regions[unique_regions >= 0]  # Exclude -1 if present

    for region_id in unique_regions:
        # Get mask for this region
        mask = region_labels == region_id

        # Get weights for this region
        if region_id in weights_dict:
            weights = weights_dict[region_id]
        else:
            # If weights not found, use uniform weights as fallback
            weights = np.ones(n_models, dtype=np.float32) / n_models

        # Assign weights to all voxels in this region
        weight_maps[mask] = weights

    return weight_maps


def save_weight_maps_npz(
    weight_maps: np.ndarray,
    save_path: Path,
    patient_id: str,
    split: str,
    spacing: str,
    pulse: str,
    model_names: List[str],
    patch_size: int,
    stride: int,
    metadata: Optional[Dict] = None
):
    """
    Save weight maps to NPZ file with comprehensive metadata.

    Args:
        weight_maps: 4D array of shape (Z, Y, X, n_models)
        save_path: Path to save the NPZ file
        patient_id: Patient identifier
        split: "train", "val", or "test"
        spacing: Spacing identifier (e.g., "3mm")
        pulse: Pulse sequence (e.g., "t1c")
        model_names: List of expert model names
        patch_size: Size of patches used for tiling
        stride: Stride used for tiling
        metadata: Optional additional metadata dictionary
    """
    # Prepare metadata
    save_dict = {
        "weight_maps": weight_maps.astype(np.float32),
        "patient_id": patient_id,
        "split": split,
        "spacing": spacing,
        "pulse": pulse,
        "model_names": model_names,
        "patch_size": patch_size,
        "stride": stride,
        "shape": weight_maps.shape[:3],
        "n_models": weight_maps.shape[3]
    }

    # Add optional metadata
    if metadata is not None:
        for key, value in metadata.items():
            if key not in save_dict:
                save_dict[key] = value

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as compressed NPZ
    np.savez_compressed(save_path, **save_dict)


def load_weight_maps_npz(npz_path: Path) -> Dict:
    """
    Load weight maps from NPZ file.

    Args:
        npz_path: Path to NPZ file

    Returns:
        Dictionary containing weight_maps and metadata
    """
    data = np.load(npz_path, allow_pickle=True)

    # Convert to regular dict for easier access
    result = {}
    for key in data.files:
        result[key] = data[key]

    return result


def analyze_weight_statistics(weight_maps: np.ndarray, model_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for each model's weights across the volume.

    Args:
        weight_maps: 4D array of shape (Z, Y, X, n_models)
        model_names: List of expert model names

    Returns:
        Dictionary mapping model_name -> statistics dict
        Statistics include: mean, std, min, max, median, q25, q75
    """
    n_models = weight_maps.shape[3]
    stats = {}

    for i, model_name in enumerate(model_names):
        weights = weight_maps[..., i]
        stats[model_name] = {
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "median": float(np.median(weights)),
            "q25": float(np.percentile(weights, 25)),
            "q75": float(np.percentile(weights, 75))
        }

    return stats


def compute_weight_entropy(weight_maps: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute per-voxel entropy of weight distributions.

    High entropy indicates uniform blending across models.
    Low entropy indicates preference for specific models.

    Args:
        weight_maps: 4D array of shape (Z, Y, X, n_models)
        eps: Small constant for numerical stability

    Returns:
        entropy_map: 3D array of shape (Z, Y, X) with entropy values
    """
    # Normalize weights to ensure they sum to 1 (in case of numerical errors)
    weights_sum = np.sum(weight_maps, axis=-1, keepdims=True)
    weights_normalized = weight_maps / (weights_sum + eps)

    # Compute Shannon entropy: -sum(p * log(p))
    log_weights = np.log(weights_normalized + eps)
    entropy_map = -np.sum(weights_normalized * log_weights, axis=-1)

    return entropy_map


def get_dominant_model_map(weight_maps: np.ndarray) -> np.ndarray:
    """
    Get the index of the dominant (highest weight) model for each voxel.

    Args:
        weight_maps: 4D array of shape (Z, Y, X, n_models)

    Returns:
        dominant_map: 3D array of shape (Z, Y, X) with model indices
    """
    return np.argmax(weight_maps, axis=-1).astype(np.int32)


def save_weight_analysis(
    weight_maps: np.ndarray,
    model_names: List[str],
    save_path: Path,
    patient_id: str
):
    """
    Save comprehensive weight map analysis including statistics,
    entropy, and dominant models.

    Args:
        weight_maps: 4D array of shape (Z, Y, X, n_models)
        model_names: List of expert model names
        save_path: Path to save analysis (will create .npz file)
        patient_id: Patient identifier
    """
    # Compute statistics
    stats = analyze_weight_statistics(weight_maps, model_names)

    # Compute entropy
    entropy_map = compute_weight_entropy(weight_maps)

    # Get dominant models
    dominant_map = get_dominant_model_map(weight_maps)

    # Save everything
    analysis_dict = {
        "weight_maps": weight_maps.astype(np.float32),
        "entropy_map": entropy_map.astype(np.float32),
        "dominant_map": dominant_map,
        "model_names": model_names,
        "patient_id": patient_id,
        "statistics": stats
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **analysis_dict)


def weights_dict_to_npz(
    weights_dict: Dict[int, np.ndarray],
    volume_shape: tuple,
    patch_size: int,
    stride: int,
    model_names: List[str],
    save_path: Path,
    patient_id: str,
    split: str,
    spacing: str,
    pulse: str,
    include_analysis: bool = False
) -> np.ndarray:
    """
    Convert a weights dictionary to NPZ format in a single call.

    This is a convenience function that combines region label generation,
    weight expansion, and saving.

    Args:
        weights_dict: Dictionary mapping region_id -> weight_vector
        volume_shape: Shape of the volume (Z, Y, X)
        patch_size: Patch size used for tiling
        stride: Stride used for tiling
        model_names: List of expert model names
        save_path: Path to save the NPZ file
        patient_id: Patient identifier
        split: "train", "val", or "test"
        spacing: Spacing identifier
        pulse: Pulse sequence
        include_analysis: If True, also save analysis file

    Returns:
        weight_maps: The generated 4D weight maps array
    """
    from .patches import region_labels

    # Generate region labels
    labels = region_labels(volume_shape, patch_size, stride)

    # Expand weights to volume
    n_models = len(model_names)
    weight_maps = expand_weights_to_volume(labels, weights_dict, n_models)

    # Save weight maps
    save_weight_maps_npz(
        weight_maps=weight_maps,
        save_path=save_path,
        patient_id=patient_id,
        split=split,
        spacing=spacing,
        pulse=pulse,
        model_names=model_names,
        patch_size=patch_size,
        stride=stride
    )

    # Optionally save analysis
    if include_analysis:
        analysis_path = save_path.parent / f"{patient_id}_weight_analysis_{split}.npz"
        save_weight_analysis(weight_maps, model_names, analysis_path, patient_id)

    return weight_maps
