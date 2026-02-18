"""3D Multi-Scale SSIM metric using pytorch-msssim."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

LOG = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    from pytorch_msssim import ms_ssim as _ms_ssim_fn

    _TORCH_AVAILABLE = True
except ImportError:
    pass


def ms_ssim_3d(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute 3D Multi-Scale SSIM between two volumes.

    Args:
        pred: Predicted volume (Z, Y, X).
        target: Ground-truth volume (Z, Y, X).
        data_range: Dynamic range of the data. If None, computed from target.
        mask: Optional boolean mask. Voxels outside mask are zeroed before computation.

    Returns:
        Scalar MS-SSIM value in [0, 1]. Returns NaN if torch/pytorch-msssim unavailable.
    """
    if not _TORCH_AVAILABLE:
        LOG.warning("pytorch-msssim not available; returning NaN")
        return float("nan")

    p = pred.astype(np.float32)
    t = target.astype(np.float32)

    if mask is not None:
        m = mask.astype(bool)
        p = p * m
        t = t * m

    if data_range is None:
        data_range = float(t.max() - t.min())
    if data_range < 1e-8:
        return 1.0

    # pytorch-msssim expects (N, C, D, H, W) for 3D
    p_tensor = torch.from_numpy(p).unsqueeze(0).unsqueeze(0)
    t_tensor = torch.from_numpy(t).unsqueeze(0).unsqueeze(0)

    min_dim = min(p_tensor.shape[2:])
    win_size = 11
    n_scales = 5

    # pytorch-msssim v1.0.0 has a hardcoded check:
    #   smaller_side > (win_size - 1) * (2 ** 4)
    # It always checks against 4 downsamples regardless of actual weights.
    # We reduce win_size first (must stay odd), then reduce scales if needed.
    while win_size > 3 and min_dim <= (win_size - 1) * (2**4):
        win_size -= 2

    # Also ensure each scale level can support the pooling operations:
    # after i downsamplings by factor 2, spatial dim = min_dim / 2^i
    # each level needs dim > win_size
    while n_scales > 1 and (min_dim // (2 ** (n_scales - 1))) < win_size:
        n_scales -= 1

    if n_scales < 2 or win_size < 3:
        LOG.warning("Volume too small for MS-SSIM (min_dim=%d); returning NaN", min_dim)
        return float("nan")

    # Truncate default weights to match reduced scale count
    default_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = default_weights[:n_scales]
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    try:
        val = _ms_ssim_fn(
            p_tensor,
            t_tensor,
            data_range=data_range,
            size_average=True,
            win_size=win_size,
            weights=weights,
        )
        return float(val.item())
    except Exception as exc:
        LOG.warning("MS-SSIM computation failed: %s", exc)
        return float("nan")
