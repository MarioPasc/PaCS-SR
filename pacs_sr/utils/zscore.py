from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ZScoreParams:
    mean: float
    std: float

def zscore(x: np.ndarray, mask: Optional[np.ndarray]=None, eps: float=1e-8) -> tuple[np.ndarray, ZScoreParams]:
    """
    Per-patient z-score normalization with optional foreground mask.
    Returns normalized array and parameters for inverse transformation.
    """
    if mask is not None:
        vals = x[mask]
    else:
        vals = x.reshape(-1)
    m = float(vals.mean())
    s = float(vals.std())
    s = s if s > eps else eps
    xn = (x - m) / s
    return xn.astype(np.float32), ZScoreParams(mean=m, std=s)

def inverse_zscore(xn: np.ndarray, params: ZScoreParams) -> np.ndarray:
    """
    Invert z-score using stored parameters.
    """
    return (xn * params.std + params.mean).astype(np.float32)
