import os
import numpy as np
import torch


def write_ply_points(path, xyz: np.ndarray, vec: np.ndarray | None = None):
    """
    Write an ASCII PLY point cloud:
      - xyz: (N,3)
      - vec: (N,3) optional; writes nx,ny,nz and also writes vertex color (rgb) = (vec*0.5+0.5)
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    N = xyz.shape[0]

    has_vec = vec is not None
    if has_vec:
        vec = np.asarray(vec, dtype=np.float32)
        assert vec.shape == (N, 3)
        rgb = np.clip((vec * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_vec:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        if not has_vec:
            for i in range(N):
                x, y, z = xyz[i]
                f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
        else:
            for i in range(N):
                x, y, z = xyz[i]
                nx, ny, nz = vec[i]
                r, g, b = rgb[i]
                f.write(f"{x:.8f} {y:.8f} {z:.8f} {nx:.8f} {ny:.8f} {nz:.8f} {int(r)} {int(g)} {int(b)}\n")
