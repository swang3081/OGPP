import torch, numpy as np
import os
from torchvision.utils import save_image
from .dynamics import EulerSimulator, VectorFieldODE
# Optionally use scipy's cKDTree for acceleration; otherwise fall back to torch's vectorized version
try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

@torch.no_grad()
def reconstruct_voronoi_images(
    sites: torch.Tensor,
    img_size: int = 32,
    coords_are_normalized: bool = True,   # True: x,y are already in [0,1]
    prefer_scipy: bool = True             # prefer KDTree; on failure, fall back to torch automatically
) -> torch.Tensor:
    """
    Accepts arbitrary prefix dims + (N,5), returns the corresponding prefix + (3,H,W).
    sites[..., :2] = (x,y), sites[..., 2:5] = (r,g,b)
    If colors are in [-1,1], they are automatically mapped to [0,1].
    If coordinates are not in [0,1] (e.g. pixel coordinates or [-1,1]), they are automatically normalized.
    """
    device = sites.device
    dtype = sites.dtype

    # ---- Unify shape: flatten the prefix ----
    if sites.dim() < 2 or sites.size(-1) != 5:
        raise ValueError(f"Expect last dim=5 [x,y,r,g,b], got shape {tuple(sites.shape)}")
    *prefix, N, D = sites.shape
    Bflat = int(np.prod(prefix)) if len(prefix) > 0 else 1
    sites_flat = sites.view(Bflat, N, D).contiguous()  # (Bflat,N,5)

    # ---- Split & normalize to [0,1] ----
    sites_flat = sites_flat.clamp_(-1.0, 1.0)
    sites_flat = (sites_flat + 1.0) / 2.0
    xy = sites_flat[..., :2].to(dtype)   # (Bflat,N,2)
    rgb = sites_flat[..., 2:5].to(dtype) # (Bflat,N,3)
    xy_u = xy
    xy_u[:, :, 1] = 1.0 - xy_u[:, :, 1]


    # ---- Pixel grid (u,v) in [0,1] ----
    u = (torch.arange(img_size, device=device, dtype=dtype) + 0.5) / img_size
    v = (torch.arange(img_size, device=device, dtype=dtype) + 0.5) / img_size
    uu, vv = torch.meshgrid(u, v, indexing="xy")               # (H,W)
    grid = torch.stack([uu, vv], dim=-1).view(-1, 2)           # (HW,2)

    # ---- Compute nearest neighbors: prefer KDTree, fall back to torch on failure ----
    out_imgs = []
    use_scipy = prefer_scipy and _HAS_SCIPY
    # print(prefer_scipy)
    if use_scipy:
        grid_np = grid.detach().cpu().numpy()
        for b in range(Bflat):
            pts = xy_u[b].detach().cpu().numpy()
            # print(pts.shape)
            # exit(0)
            if pts.ndim != 2:
                pts = pts.reshape(-1, 2)
            elif pts.shape[1] != 2 and pts.shape[0] == 2:
                pts = pts.T
            if pts.ndim != 2 or pts.shape[1] != 2:
                use_scipy = False
                break
            try:
                tree = cKDTree(pts)
                _, idx = tree.query(grid_np, k=1)             # (HW,)
                cols = rgb[b].detach().cpu().numpy()[idx]     # (HW,3)
                img = torch.from_numpy(cols).to(device=device, dtype=dtype).view(img_size, img_size, 3).permute(2,0,1)
                out_imgs.append(img)
            except Exception:
                use_scipy = False
                break

    if not use_scipy:
        # Pure torch: broadcast to compute distances, (Bflat,HW,N)
        G = grid.unsqueeze(0)                                  # (1,HW,2)
        P = xy_u.unsqueeze(1)                                  # (Bflat,1,N,2)
        d2 = (G.unsqueeze(2) - P).pow(2).sum(dim=-1)           # (Bflat,HW,N)
        nn_idx = d2.argmin(dim=-1)                             # (Bflat,HW)
        cols = rgb.gather(1, nn_idx.unsqueeze(-1).expand(-1, -1, 3))  # (Bflat,HW,3)
        imgs = cols.view(Bflat, img_size, img_size, 3).permute(0,3,1,2)
        out_imgs = [imgs[i] for i in range(Bflat)]

    out = torch.stack(out_imgs, dim=0)                         # (Bflat,3,H,W)
    if len(prefix) > 0:
        out = out.view(*prefix, 3, img_size, img_size)
    return out

@torch.no_grad()
def export_voronoi_samples_for_fid(
    model, path, out_dir: str, total: int = 10000, batch: int = 256,
    steps: int = 250, device: torch.device = torch.device("cpu"),
    img_size: int = 32, coords_are_normalized: bool = True
):
    """
    Generate Voronoi point sets -> ODE integration -> KDTree reconstruction -> save PNG
    """

    os.makedirs(out_dir, exist_ok=True)
    ode = VectorFieldODE(model)
    sim = EulerSimulator(ode)

    done = 0
    while done < total:
        b = min(batch, total - done)
        x0, _ = path.p_simple.sample(b)
        ts = torch.linspace(0, 1, steps, device=device).view(1,-1,1,1).expand(b,-1,1,1)
        x1 = sim.simulate(x0, ts)                                                # (B,N,5)
        imgs = reconstruct_voronoi_images(x1, img_size=img_size,
                                          coords_are_normalized=coords_are_normalized)  # (B,3,H,W) [0,1]
        for i in range(b):
            save_image(imgs[i].cpu(), os.path.join(out_dir, f"{done+i:06d}.png"))
        done += b
        if done % 1024 == 0 or done == total:
            print(f"[gen] {done}/{total}")
