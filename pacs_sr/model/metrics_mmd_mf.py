"""MMD-MF: Maximum Mean Discrepancy on Morphological Features.

Pure numpy/scipy implementation. Extracts local morphological features
(gradient magnitude, multi-scale LoG, local variance) and computes
polynomial kernel MMD between predicted and target feature distributions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage as ndi

LOG = logging.getLogger(__name__)


def _sobel_magnitude(vol: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude for a 3D volume."""
    v = vol.astype(np.float32, copy=False)
    gx = ndi.sobel(v, axis=2, mode="nearest")
    gy = ndi.sobel(v, axis=1, mode="nearest")
    gz = ndi.sobel(v, axis=0, mode="nearest")
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def _log_response(vol: np.ndarray, sigma: float) -> np.ndarray:
    """Laplacian of Gaussian at a single scale."""
    return ndi.gaussian_laplace(vol.astype(np.float32, copy=False), sigma=sigma)


def _local_variance(vol: np.ndarray, radius: int = 2) -> np.ndarray:
    """Local variance in a (2*radius+1)^3 neighborhood via uniform filter."""
    v = vol.astype(np.float32, copy=False)
    size = 2 * radius + 1
    mean = ndi.uniform_filter(v, size=size, mode="nearest")
    mean_sq = ndi.uniform_filter(v * v, size=size, mode="nearest")
    var = mean_sq - mean * mean
    var[var < 0] = 0.0
    return var


def _extract_features(vol: np.ndarray) -> np.ndarray:
    """Extract 5 morphological features per voxel.

    Features:
        0: Sobel gradient magnitude
        1: LoG at sigma=1
        2: LoG at sigma=2
        3: LoG at sigma=4
        4: Local variance (5x5x5)

    Args:
        vol: 3D volume (Z, Y, X).

    Returns:
        Feature array of shape (Z*Y*X, 5).
    """
    feats = [
        _sobel_magnitude(vol).ravel(),
        _log_response(vol, sigma=1.0).ravel(),
        _log_response(vol, sigma=2.0).ravel(),
        _log_response(vol, sigma=4.0).ravel(),
        _local_variance(vol, radius=2).ravel(),
    ]
    return np.column_stack(feats)


def _polynomial_mmd(
    features_x: np.ndarray,
    features_y: np.ndarray,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
) -> float:
    """Compute polynomial kernel Maximum Mean Discrepancy.

    Replicates the pattern from pacs_sr/model/metrics.py:233-256.

    Args:
        features_x: Feature matrix (N, D).
        features_y: Feature matrix (M, D).
        degree: Polynomial kernel degree.
        gamma: Kernel coefficient. Defaults to 1/n_features.
        coef0: Kernel intercept.

    Returns:
        MMD^2 estimate (non-negative float).
    """
    n, m = len(features_x), len(features_y)
    if n == 0 or m == 0:
        return float("nan")

    if gamma is None:
        gamma = 1.0 / features_x.shape[1]

    def kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (gamma * x @ y.T + coef0) ** degree

    k_xx = kernel(features_x, features_x)
    k_yy = kernel(features_y, features_y)
    k_xy = kernel(features_x, features_y)

    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1)) if n > 1 else 0.0
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1)) if m > 1 else 0.0
    sum_xy = k_xy.mean()

    mmd_sq = sum_xx + sum_yy - 2 * sum_xy
    return float(max(0.0, mmd_sq))


def mmd_mf(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_samples: int = 10_000,
    seed: int = 42,
) -> float:
    """Compute MMD on Morphological Features between two 3D volumes.

    Args:
        pred: Predicted volume (Z, Y, X).
        target: Ground-truth volume (Z, Y, X).
        mask: Optional boolean brain mask. If None, uses target > 0.
        n_samples: Number of voxels to subsample for MMD computation.
        seed: Random seed for reproducible subsampling.

    Returns:
        MMD-MF value (lower = better morphological fidelity).
    """
    if mask is None:
        mask = target > 0

    mask = mask.astype(bool)
    n_fg = int(np.count_nonzero(mask))
    if n_fg == 0:
        LOG.warning("Empty mask; returning NaN")
        return float("nan")

    # Extract features under mask
    feat_pred = _extract_features(pred)
    feat_target = _extract_features(target)

    flat_mask = mask.ravel()
    feat_pred = feat_pred[flat_mask]
    feat_target = feat_target[flat_mask]

    # Subsample for tractable MMD computation
    rng = np.random.RandomState(seed)
    if n_fg > n_samples:
        idx = rng.choice(n_fg, size=n_samples, replace=False)
        feat_pred = feat_pred[idx]
        feat_target = feat_target[idx]

    # Z-score normalize features jointly for numerical stability
    combined = np.concatenate([feat_pred, feat_target], axis=0)
    mu = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std < 1e-8] = 1.0
    feat_pred = (feat_pred - mu) / std
    feat_target = (feat_target - mu) / std

    return _polynomial_mmd(feat_pred, feat_target)
