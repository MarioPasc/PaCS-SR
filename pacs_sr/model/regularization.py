from __future__ import annotations
import numpy as np
from typing import Dict, Iterable

def laplacian_smooth_weights(
    W: np.ndarray,
    adjacency: Dict[int, Iterable[int]],
    tau: float,
    n_iter: int = 1
) -> np.ndarray:
    """
    Simple Laplacian smoothing over the region-adjacency graph.
    W: (R, M) weights per region, sum-to-one along M assumed.
    adjacency: dict node -> neighbors (6-neighborhood typically)
    tau: smoothing coefficient in [0,1]. tau=0 returns W.
    n_iter: number of Jacobi smoothing iterations.
    """
    if tau <= 0:
        return W
    R, M = W.shape
    Wout = W.copy()
    for _ in range(n_iter):
        newW = Wout.copy()
        for r in range(R):
            nbrs = list(adjacency.get(r, []))
            if not nbrs:
                continue
            avg = np.mean(Wout[nbrs, :], axis=0)
            newW[r, :] = (1.0 - tau) * Wout[r, :] + tau * avg
        Wout = newW
    # Renormalize to simplex
    Wout[Wout < 0] = 0
    row_sums = np.sum(Wout, axis=1, keepdims=True) + 1e-8
    Wout = Wout / row_sums
    return Wout
