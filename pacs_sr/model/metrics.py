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

        # Skip slices with no foreground voxels
        if ms is not None and not np.any(ms):
            continue

        if ms is not None:
            # compute SSIM only on masked region by zeroing outside
            # note: structural_similarity requires same shape arrays
            ps = ps * ms
            ts = ts * ms

        # Calculate data range with safeguards
        data_range = float(ts.max() - ts.min())
        if data_range < 1e-7:  # Avoid division by zero or near-zero values
            continue

        try:
            score = structural_similarity(ts, ps, data_range=data_range)
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            # Skip slices that cause errors in SSIM computation
            continue

    return float(np.mean(scores)) if scores else 0.0

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


# --------------------------------------------------------------------------
# Optional deep perceptual metrics (require torch)
# --------------------------------------------------------------------------

_TORCH_AVAILABLE = False
_KID_MODEL = None
_DEVICE = None

try:
    import torch
    from torchvision.models import inception_v3, Inception_V3_Weights
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def _load_kid_model():
    """Lazily load KID model on first use."""
    global _KID_MODEL
    if _KID_MODEL is None and _TORCH_AVAILABLE:
        _inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        _inception.fc = torch.nn.Identity()
        _inception = _inception.to(_DEVICE).eval()
        _inception.requires_grad_(False)
        _KID_MODEL = _inception
    return _KID_MODEL


def kid_slicewise(pred: np.ndarray, target: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  axis: str = "axial") -> float:
    """
    Slice-wise KID (Kernel Inception Distance) averaged across a chosen axis.

    Uses Inception V3 features with polynomial kernel MMD.
    Lower values indicate more similar distributions (better SR quality).

    Args:
        pred: Predicted SR volume (Z, Y, X)
        target: Ground truth HR volume (Z, Y, X)
        mask: Optional mask for ROI
        axis: Slice axis {'axial', 'coronal', 'sagittal'}

    Returns:
        Average KID across slices (lower = better)
    """
    if not _TORCH_AVAILABLE:
        return float('nan')

    model = _load_kid_model()
    if model is None:
        return float('nan')

    axes = {"axial": 0, "coronal": 1, "sagittal": 2}
    a = axes.get(axis, 0)

    p = pred.astype(np.float32)
    t = target.astype(np.float32)

    if mask is not None:
        m = mask.astype(bool)
    else:
        m = None

    kid_scores = []

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

        # Skip slices with no foreground
        if ms is not None and not np.any(ms):
            continue

        # Apply mask
        if ms is not None:
            ps = ps * ms.astype(np.float32)
            ts = ts * ms.astype(np.float32)

        # Normalize to [0, 1]
        vmin = min(ps.min(), ts.min())
        vmax = max(ps.max(), ts.max())
        if vmax - vmin < 1e-7:
            continue

        ps_norm = (ps - vmin) / (vmax - vmin)
        ts_norm = (ts - vmin) / (vmax - vmin)

        try:
            kid_val = _compute_kid_single_slice(ps_norm, ts_norm, model)
            if not np.isnan(kid_val):
                kid_scores.append(kid_val)
        except Exception:
            continue

    return float(np.mean(kid_scores)) if kid_scores else float('nan')


def _compute_kid_single_slice(pred_slice: np.ndarray, target_slice: np.ndarray,
                               model) -> float:
    """Compute KID for a single 2D slice."""
    from scipy.ndimage import zoom

    # Resize to 299x299 for Inception
    target_size = 299
    h, w = pred_slice.shape
    zoom_factors = (target_size / h, target_size / w)

    pred_resized = zoom(pred_slice, zoom_factors, order=1)
    target_resized = zoom(target_slice, zoom_factors, order=1)

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def _to_tensor(img: np.ndarray):
        rgb = np.stack([img, img, img], axis=0)
        for c in range(3):
            rgb[c] = (rgb[c] - mean[c]) / std[c]
        t = torch.from_numpy(rgb).float().unsqueeze(0)
        return t.to(_DEVICE)

    pred_t = _to_tensor(pred_resized)
    target_t = _to_tensor(target_resized)

    with torch.no_grad():
        pred_feat = model(pred_t).cpu().numpy().flatten().reshape(1, -1)
        target_feat = model(target_t).cpu().numpy().flatten().reshape(1, -1)

    # Polynomial kernel MMD
    return _polynomial_mmd(pred_feat, target_feat)


def _polynomial_mmd(features_x: np.ndarray, features_y: np.ndarray,
                    degree: int = 3, gamma: float = None,
                    coef0: float = 1.0) -> float:
    """Compute polynomial kernel Maximum Mean Discrepancy."""
    n, m = len(features_x), len(features_y)
    if n == 0 or m == 0:
        return float('nan')

    if gamma is None:
        gamma = 1.0 / features_x.shape[1]

    def kernel(x, y):
        return (gamma * x @ y.T + coef0) ** degree

    k_xx = kernel(features_x, features_x)
    k_yy = kernel(features_y, features_y)
    k_xy = kernel(features_x, features_y)

    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1)) if n > 1 else 0
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1)) if m > 1 else 0
    sum_xy = k_xy.mean()

    mmd_sq = sum_xx + sum_yy - 2 * sum_xy
    return float(max(0, mmd_sq))
