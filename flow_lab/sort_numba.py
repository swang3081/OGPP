import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# ============================================================
# Hilbert index + sorting implementation
# ============================================================
@njit
def hilbert_index(x, y, p):
    index = 0
    n = 1 << p
    rx, ry = 0, 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        index += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s >>= 1
    return index

@njit(parallel=True)
def hilbert_indices(coords, p):
    N = coords.shape[0]
    out = np.empty(N, dtype=np.int64)
    for i in prange(N):
        out[i] = hilbert_index(coords[i,0], coords[i,1], p)
    return out

def hilbert_sort_xy_fast(xy, p=10):
    if xy.ndim == 2:
        xy = xy[None, ...]
    elif xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError(f"expected (N,2) or (K,N,2), got {xy.shape}")

    K, N, _ = xy.shape
    grid_max = (2 ** p) - 1
    results = np.empty_like(xy)

    for k in range(K):
        pts = xy[k]
        xy_min = pts.min(axis=0)
        xy_max = pts.max(axis=0)
        span = xy_max - xy_min
        span[span == 0] = 1.0
        xy_norm = (pts - xy_min) / span
        coords = np.floor(xy_norm * grid_max + 1e-12).astype(np.int64)
        coords = np.clip(coords, 0, grid_max)

        dists = hilbert_indices(coords, p)
        order = np.argsort(dists)
        results[k] = pts[order]

    return results[0] if results.shape[0] == 1 else results