from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Tuple
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error

def psnr(pred: np.ndarray, target: np.ndarray, data_range: Optional[float]=None, mask: Optional[np.ndarray]=None) -> float:
    """
    PSNR using scikit-image. If mask is provided, compute on masked voxels only.
    """
    if mask is not None:
        p = pred[mask]
        t = target[mask]
        # scikit-image expects images; fall back to formula
        mse = np.mean((p.astype(np.float32) - t.astype(np.float32))**2)
        if data_range is None:
            data_range = float(np.max(t) - np.min(t))
        if mse <= 0:
            return float("inf")
        return float(20.0 * np.log10((data_range + 1e-12) / np.sqrt(mse + 1e-12)))
    else:
        return float(peak_signal_noise_ratio(target, pred, data_range=data_range))

def ssim3d_slicewise(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]=None, axis: str="axial") -> float:
    """
    Slice-wise SSIM averaged across a chosen axis. This avoids heavy 3D SSIM implementations.
    axis in {'axial', 'coronal', 'sagittal'}.
    """
    axes = {"axial": 0, "coronal": 1, "sagittal": 2}
    a = axes.get(axis, 0)
    p = pred.astype(np.float32)
    t = target.astype(np.float32)
    if mask is not None:
        m = mask.astype(bool)
    else:
        m = None
    scores = []
    for i in range(p.shape[a]):
        if a == 0:
            ps, ts = p[i], t[i]
            ms = m[i] if m is not None else None
        elif a == 1:
            ps, ts = p[:, i, :], t[:, i, :]
            ms = m[:, i, :] if m is not None else None
        else:
            ps, ts = p[:, :, i], t[:, :, i]
            ms = m[:, :, i] if m is not None else None
        if ms is not None:
            # compute SSIM only on masked region by zeroing outside
            # note: structural_similarity requires same shape arrays
            ps = ps * ms
            ts = ts * ms
        scores.append(structural_similarity(ts, ps, data_range=float(ts.max() - ts.min()) if ts.size else 1.0))
    return float(np.mean(scores)) if scores else float("nan")

def mae(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]=None) -> float:
    if mask is not None:
        p = pred[mask]
        t = target[mask]
    else:
        p = pred.reshape(-1)
        t = target.reshape(-1)
    return float(np.mean(np.abs(p.astype(np.float32) - t.astype(np.float32))))

def mse(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]=None) -> float:
    if mask is not None:
        p = pred[mask]
        t = target[mask]
    else:
        p = pred.reshape(-1)
        t = target.reshape(-1)
    return float(mean_squared_error(t, p))
