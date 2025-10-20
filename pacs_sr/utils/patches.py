from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple

def tile_boxes(shape: Tuple[int,int,int], patch_size: int, stride: int
               ) -> Iterator[Tuple[int, Tuple[int,int,int,int,int,int]]]:
    """
    Yield (rid, (z0,z1,y0,y1,x0,x1)) in the same order used by region_labels().
    """
    Z, Y, X = shape
    p, s = patch_size, stride
    rid = 0
    for z0 in range(0, Z, s):
        for y0 in range(0, Y, s):
            for x0 in range(0, X, s):
                z1, y1, x1 = min(z0+p, Z), min(y0+p, Y), min(x0+p, X)
                yield rid, (z0, z1, y0, y1, x0, x1)
                rid += 1

def region_labels(shape: Tuple[int, int, int], patch_size: int, stride: int) -> np.ndarray:
    """
    Compute an integer region label per voxel given a regular 3D tiling.
    Regions are enumerated in z-major order. No padding; trailing border is a smaller tile if not divisible.

    Returns
    -------
    labels : np.ndarray[int32], shape=shape
        Region id for each voxel in the volume.
    """
    Z, Y, X = shape
    p = patch_size
    s = stride

    # Compute grid anchors
    zs = list(range(0, Z, s))
    ys = list(range(0, Y, s))
    xs = list(range(0, X, s))

    labels = np.full(shape, -1, dtype=np.int32)
    rid = 0
    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                z1 = min(z0 + p, Z)
                y1 = min(y0 + p, Y)
                x1 = min(x0 + p, X)
                labels[z0:z1, y0:y1, x0:x1] = rid
                rid += 1
    return labels

def region_adjacency_from_labels(labels: np.ndarray) -> dict[int, set[int]]:
    """
    Build a 6-neighborhood adjacency graph over tile ids to support Laplacian smoothing.
    """
    Z, Y, X = labels.shape
    adj: dict[int, set[int]] = {}
    def add(u, v):
        if u == v: return
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    # Check faces between tiles by scanning lines where labels change
    # Along x
    for z in range(Z):
        for y in range(Y):
            row = labels[z, y, :]
            diffs = row[:-1] != row[1:]
            idxs = np.where(diffs)[0]
            for i in idxs:
                add(int(row[i]), int(row[i+1]))
    # Along y
    for z in range(Z):
        for x in range(X):
            col = labels[z, :, x]
            diffs = col[:-1] != col[1:]
            idxs = np.where(diffs)[0]
            for i in idxs:
                add(int(col[i]), int(col[i+1]))
    # Along z
    for y in range(Y):
        for x in range(X):
            dep = labels[:, y, x]
            diffs = dep[:-1] != dep[1:]
            idxs = np.where(diffs)[0]
            for i in idxs:
                add(int(dep[i]), int(dep[i+1]))
    return adj
