import os, torch
import torch.nn as nn
from .paths import GaussianConditionalProbabilityPath
import time
from .dynamics import VectorFieldODE, EulerSimulator, EulerMaruyamaSimulator
from torchvision.utils import save_image
import torch.nn.functional as F
import re
import numpy as np
from typing import Optional, Tuple
from torchvision.utils import make_grid
import glob
from pathlib import Path
import igl
from numba import njit, prange
from typing import Union, List, Tuple
import math
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False
def get_device() -> torch.device:
    d = os.getenv("DEVICE", "auto").lower()
    if d == "cuda": return torch.device("cuda")
    if d == "mps":  return torch.device("mps")
    if d == "cpu":  return torch.device("cpu")
    if torch.cuda.is_available(): return torch.device("cuda")
    try:
        if torch.backends.mps.is_available(): return torch.device("mps")
    except Exception: pass
    return torch.device("cpu")
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
import open3d as o3d

import open3d.core as o3c
from joblib import Parallel, delayed

from typing import Dict

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    _HAS_SPARSE_BIPARTITE = True
except Exception:
    _HAS_SPARSE_BIPARTITE = False

_BVH_CACHE: Dict[int, o3d.t.geometry.RaycastingScene] = {}
_IGL_AABB_CACHE: dict[int, igl.AABB] = {}

def get_or_build_igl_aabb(mesh_id: int, V: np.ndarray, F: np.ndarray) -> igl.AABB:
    """
    Cache the AABB within the current process.
    Do not pass the tree across processes; only pass V, F, mesh_id.
    """
    tree = _IGL_AABB_CACHE.get(mesh_id)
    if tree is None:
        tree = igl.AABB()
        tree.init(
            V.astype(np.float64, copy=False),
            F.astype(np.int32,  copy=False),
        )
        _IGL_AABB_CACHE[mesh_id] = tree
    return tree


def save_checkpoint(
    model: nn.Module,
    args,
    save_dir: str,
    arch_name: str = "UncondRGBUNet",
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    epoch: Optional[int] = None      ,    # 0-based current epoch
    global_step: Optional[int] = None ,  # number of optimization steps already executed
    max_retries: int = 3,
):
    """
    Save a checkpoint with retries and exception handling to avoid training
    interruptions caused by intermittent Unity Catalog Volumes issues.
    """
    # Note: this function should only be called by rank 0, and directory creation also runs only once
    os.makedirs(save_dir, exist_ok=True)

    base_model = model.module if hasattr(model, "module") else model

    ckpt = {
        "model": base_model.state_dict(),
        # Training progress (used for resume)
        "epoch": int(epoch if epoch is not None else 0),
        "global_step": int(global_step if global_step is not None else 0),
        "scheduler_last_epoch": getattr(scheduler, "last_epoch", -1),
        # Optional components
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        # Metadata / configuration
        "arch": arch_name,
        "args": vars(args) if args is not None else None,
        # Fields originally kept (in case they are still needed)
        "p_simple_shape": [3, 32, 32],
        "normalize": "[-1,1]",
        "sample_steps": getattr(args, "sample_steps", None),
        "seed": torch.initial_seed(),
    }

    # Use the "current progress" in the filename instead of args.epochs, so you can tell at a glance which epoch was saved
    fname = f"{arch_name}_e{ckpt['epoch']+1:03d}_gs{ckpt['global_step']}_lr{getattr(args,'lr',None)}_bs{getattr(args,'batch_size',None)}_{int(time.time())}.pt"
    path = os.path.join(save_dir, fname)

    # Save logic with retries
    for attempt in range(max_retries):
        try:
            torch.save(ckpt, path)
            print(f"[ckpt] saved to {path}")
            return path
        except (PermissionError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"[WARNING] save_checkpoint failed (attempt {attempt+1}/{max_retries}): {e}, retrying in 2s...")
                time.sleep(2)
            else:
                print(f"[ERROR] save_checkpoint failed after {max_retries} attempts: {e}")
                print(f"[ERROR] Training will continue, but checkpoint was NOT saved!")
                return None

def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    add_min_lr_back: bool = True,   # You adopted the "base_lr = lr - min_lr + add min_lr back each step" scheme
    min_lr: float = 0.0,
    only_load_model_weight: bool = False,
) -> Tuple[int, int]:
    """
    Returns: (epoch, global_step)
    - Only model passed: load weights only; if epoch/global_step are absent, return (0, 0)
    - optimizer passed: also restore momentum, etc.
    - scheduler passed: also align last_epoch; if add_min_lr_back=True and min_lr>0, add min_lr back after step
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # 1) Model weights
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    # Compatible with DDP wrapping
    base_model = model.module if hasattr(model, "module") else model
    missing = base_model.load_state_dict(state_dict, strict=strict)

    if hasattr(missing, "missing_keys") and (missing.missing_keys or missing.unexpected_keys):
        print(f"[ckpt] load_state_dict missing={missing.missing_keys}, unexpected={missing.unexpected_keys}")

    # If only the model weights are needed, return directly
    if only_load_model_weight:
        print(f"[ckpt] (weights-only) loaded from {ckpt_path}")
        return 0, 0

    # 2) Optimizer
    if optimizer is not None and ckpt.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[ckpt] optimizer load failed: {e} (continue with fresh optimizer)")

    # 3) Training progress
    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))

    # 4) Scheduler progress
    if scheduler is not None:
        # last_epoch semantics: the number of steps already completed; the next scheduler.step() uses last_epoch+1
        last_ep = int(ckpt.get("scheduler_last_epoch", global_step - 1))
        try:
            scheduler.last_epoch = last_ep
            scheduler.step()   # advance to the position corresponding to last_epoch+1
            if add_min_lr_back and min_lr > 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["lr"] + min_lr
        except Exception as e:
            print(f"[ckpt] scheduler restore failed: {e} (continue with fresh schedule)")

    print(f"[ckpt] loaded from {ckpt_path}, epoch={epoch}, global_step={global_step}")

    return epoch, global_step


@torch.no_grad()
def export_samples_for_fid(model: nn.Module, path_obj: GaussianConditionalProbabilityPath,
                           out_dir: str, total: int = 50000, batch: int = 256, steps: int = 250, device="cuda"):
    """
    Optional: export PNGs to a folder so FID can later be computed with pytorch-fid/torchmetrics
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    ode = VectorFieldODE(model)
    sim = EulerSimulator(ode)
    done = 0
    while done < total:
        b = min(batch, total - done)
        x0, _ = path_obj.p_simple.sample(b)  # (B,3,32,32) in [-1,1]
        ts = torch.linspace(0, 1, steps, device=device).view(1, -1, 1, 1, 1).expand(b, -1, 1, 1, 1)
        x1 = sim.simulate(x0, ts)            # Unconditional: no y needed
        imgs = ((x1.clamp(-1, 1) + 1) / 2.0).cpu()
        for i in range(b):
            save_image(imgs[i], os.path.join(out_dir, f"{done+i:06d}.png"))
        done += b
        if done % 1024 == 0 or done == total:
            print(f"[export] {done}/{total}")
    

def make_red_dot_images(xy, img_size=64, coords_are_normalized=True):
    """
    xy: (B, N, 2)
    return: (B, 3, H, W) — image of red dots on a black background
    """
    B, N, _ = xy.shape
    imgs = torch.zeros(B, 3, img_size, img_size, device=xy.device)
    for b in range(B):
        coords = xy[b]
        if coords_are_normalized:
            coords = (coords + 1) / 2 * (img_size - 1)
            coords[:, 1] = img_size - coords[:, 1]
        coords = coords.long().clamp(0, img_size - 1)
        imgs[b, 0, coords[:, 1], coords[:, 0]] = 1.0  # R channel
    return imgs

def render_point_images_newvis(
    xt: torch.Tensor,                 # (B, N, 2)
    img_size: int = 128,
    coords_are_normalized: bool = True,
    point_radius: int = 2,
    background: float = 1.0,
    point_value: float = 0.0,
    channels: int = 3,
    antialias: bool = False,
    # === Hilbert connecting lines (new) ===
    connect_points: bool = False,
    line_color=(1.0, 0., 0.0),       # orange line
    line_width: int = 1,
    line_alpha: float = 0.7,
    # === 32 equal divisions (solid gray lines) ===
    draw_grid_x_32: bool = False,
    n_splits_x_32: int = 32,
    line_thickness_32: int = 4,
    line_value_32: float = 0.7,
    # === 64 equal divisions (red dashed lines) ===
    draw_grid_x_64_dashed: bool = False,
    n_splits_x_64: int = 64,
    dash_thickness_64: int = 2,
    dash_len_64: int = 6,
    gap_len_64: int = 6,
    # === 64-bin counts ===
    show_counts_64: bool = False,
    min_text_gap_px: int = 14,
    font_size: Optional[int] = None,
    text_color=(0, 0, 0),
    text_stroke_width: int = 1,
    text_stroke_fill=(255, 255, 255),
):
    """
    Render a point set, with support for:
      - small dots
      - 32 equal-division gray lines
      - 64 equal-division red dashed lines
      - per-bin point counts
      - Hilbert connecting lines (connect_points=True)
    """
    assert xt.ndim == 3 and xt.shape[-1] == 2
    device = xt.device
    B, N, _ = xt.shape
    H = W = int(img_size)

    # Map to pixels
    x, y = xt[..., 0], xt[..., 1]
    if coords_are_normalized:
        xmin, xmax = float(x.min()), float(x.max())
        is_minus1_1 = (xmin < -1e-2) or (xmax > 1 + 1e-2) or (x.min() < 0) or (y.min() < 0)
        if is_minus1_1 and (x.min() >= -1.1 and x.max() <= 1.1 and y.min() >= -1.1 and y.max() <= 1.1):
            x_pix = ((x + 1.0) * 0.5) * (W - 1)
            y_pix = ((y + 1.0) * 0.5) * (H - 1)
        else:
            x_pix = x * (W - 1)
            y_pix = y * (H - 1)
    else:
        x_pix, y_pix = x, y

    finite_mask = torch.isfinite(x_pix) & torch.isfinite(y_pix)
    x_idx = x_pix.round().clamp(0, W - 1).long()
    y_idx = y_pix.round().clamp(0, H - 1).long()

    # impulses: point impulses
    impulses = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    b_ids = torch.arange(B, device=device).view(B, 1).expand(B, N)
    lin = (b_ids * (H * W) + y_idx * W + x_idx)
    lin = lin[finite_mask]
    impulses.view(-1).index_put_((lin.reshape(-1),), torch.ones(lin.numel(), device=device), accumulate=True)

    # Dilate the dots
    r = int(max(0, point_radius))
    if r == 0:
        dots = impulses
    else:
        yy, xx = torch.meshgrid(
            torch.arange(-r, r + 1, device=device),
            torch.arange(-r, r + 1, device=device),
            indexing='ij'
        )
        if antialias:
            sigma = max(0.5, r / 2)
            ker = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        else:
            ker = ((xx**2 + yy**2) <= (r * r)).float()
        ker = ker / ker.max()
        ker = ker.view(1, 1, *ker.shape)
        dots = F.conv2d(impulses, ker, padding=r)
        dots = (dots > 0).float() if not antialias else dots.clamp(0, 1)

    # Single-channel grayscale background
    img = background * torch.ones_like(dots)
    img = img * (1 - dots) + point_value * dots

    # 32 equal divisions
    if draw_grid_x_32 and n_splits_x_32 > 1 and line_thickness_32 > 0:
        xs32 = torch.linspace(0, W - 1, steps=n_splits_x_32 + 1, device=device)[1:-1]
        cols = torch.arange(W, device=device)
        half32 = max(0.5, (line_thickness_32 - 1.0) / 2.0)
        dist32 = (cols.unsqueeze(0) - xs32.unsqueeze(1)).abs()
        mask_1d_32 = (dist32 <= half32).any(dim=0).float()
        line_mask_32 = mask_1d_32.view(1, 1, 1, W).expand(B, 1, H, W)
        img = img * (1.0 - line_mask_32) + line_value_32 * line_mask_32

    # === Convert to RGB, to make drawing lines and text easier ===
    if channels == 3:
        img = img.repeat(1, 3, 1, 1)
    else:
        img = img.repeat(1, 3, 1, 1)
        _to_gray_later = True

    # === Hilbert connecting lines ===
    if connect_points:
        img_cpu = (img.clamp(0,1).cpu() * 255).byte().permute(0,2,3,1).numpy()
        x_i = x_pix.round().clamp(0,W-1).long().cpu().numpy()
        y_i = y_pix.round().clamp(0,H-1).long().cpu().numpy()

        for b in range(B):
            pil_img = Image.fromarray(img_cpu[b], mode="RGB")
            draw = ImageDraw.Draw(pil_img)
            pts = list(zip(x_i[b], y_i[b]))

            # Main line
            draw.line(
                pts,
                fill=tuple(int(c*255) for c in line_color),
                width=int(line_width)
            )

            # Alpha blending
            if line_alpha < 1.0:
                pil_arr = np.array(pil_img).astype(np.float32)
                orig = img_cpu[b].astype(np.float32)
                pil_arr = pil_arr * line_alpha + orig * (1 - line_alpha)
                img_cpu[b] = pil_arr.clip(0,255).astype(np.uint8)
            else:
                img_cpu[b] = np.array(pil_img, dtype=np.uint8)

        img = torch.from_numpy(img_cpu).permute(0,3,1,2).to(device, dtype=torch.uint8).float()/255.0

    # === 64 equal-division red dashed lines ===
    if draw_grid_x_64_dashed and n_splits_x_64 > 1 and dash_thickness_64 > 0:
        n_bins = int(n_splits_x_64)
        xs64 = (torch.arange(1, n_bins, device=device).float() * (W / n_bins))
        cols = torch.arange(W, device=device).float()
        half64 = max(0.5, (dash_thickness_64 - 1.0) / 2.0)
        dist64 = (cols.unsqueeze(0) - xs64.unsqueeze(1)).abs()
        mask_cols_64 = (dist64 <= half64).any(dim=0)

        period = int(dash_len_64 + gap_len_64)
        rows = torch.arange(H, device=device)
        mask_rows_dash = (rows % period) < dash_len_64

        red_mask = (mask_rows_dash.view(1,1,H,1) & mask_cols_64.view(1,1,1,W)).expand(B,1,H,W)
        red_color = torch.tensor([1.0,0.0,0.0], device=device).view(1,3,1,1)
        img = torch.where(red_mask, red_color, img)

    # === 64-bin counts ===
    counts_64 = None
    if show_counts_64:
        n_bins = int(n_splits_x_64)
        x_pix_clamped = x_pix.clamp(0, W - 1)
        bin_idx = (x_pix_clamped * n_bins / float(W)).floor().long().clamp(0, n_bins - 1)
        valid = finite_mask
        counts_64 = torch.zeros((B, n_bins), device=device, dtype=torch.int32)
        b_flat = b_ids[valid]
        bin_flat = bin_idx[valid]
        counts_64.index_put_((b_flat, bin_flat), torch.ones(bin_flat.numel(), device=device, dtype=torch.int32), accumulate=True)

    # Draw text
    if show_counts_64 and _HAS_PIL:
        img_cpu = (img.clamp(0,1).cpu() * 255).byte().permute(0,2,3,1).numpy()

        if font_size is None:
            fs = max(8, int(round(W / 64)))
        else:
            fs = int(font_size)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", fs)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", fs)
            except:
                font = ImageFont.load_default()

        n_bins = int(n_splits_x_64)
        boundary_cols = [int(round(j * W / n_bins)) for j in range(1, n_bins)]
        bin_width = W / n_bins
        step = max(1, int(np.ceil(min_text_gap_px / bin_width)))
        n_rows = 2

        for b in range(B):
            pil_img = Image.fromarray(img_cpu[b], mode="RGB")
            draw = ImageDraw.Draw(pil_img)
            for idx, j in enumerate(range(1, n_bins, step)):
                col = boundary_cols[j - 1]
                left_c = int(counts_64[b,j-1].item())
                right_c = int(counts_64[b,j].item())

                row_id = (idx % n_rows)
                y_top = 1 + row_id * (fs + 2)

                left_text = str(left_c)
                bbox_l = draw.textbbox((0,0), left_text, font=font, stroke_width=text_stroke_width)
                tw_l = bbox_l[2] - bbox_l[0]
                x_left = max(0, col - 2 - tw_l)

                right_text = str(right_c)
                bbox_r = draw.textbbox((0,0), right_text, font=font, stroke_width=text_stroke_width)
                tw_r = bbox_r[2] - bbox_r[0]
                x_right = min(W - tw_r, col + 2)

                draw.text((x_left, y_top), left_text, font=font,
                          fill=text_color, stroke_width=text_stroke_width, stroke_fill=text_stroke_fill)
                draw.text((x_right, y_top), right_text, font=font,
                          fill=text_color, stroke_width=text_stroke_width, stroke_fill=text_stroke_fill)

            img_cpu[b] = np.array(pil_img, dtype=np.uint8)

        img = torch.from_numpy(img_cpu).permute(0,3,1,2).to(device, dtype=torch.uint8).float()/255.0

    # Restore single channel
    if channels == 1:
        img = img[:, :1, :, :]

    return img

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def _render_point_images_3d(
    xt: torch.Tensor,                 # (B, N, 3) or (B, N, 6=xyz+normal)
    img_size: int = 512,
    coords_are_normalized: bool = True,
    point_radius: int = 2,
    background: float = 1.0,          # kept for interface compatibility; canvas defaults to white background
    channels: int = 3,
    color_normal: bool = False,       # <<< new flag
):
    """
    3D point cloud rendering:
      - xt[..., :3] is used as point coordinates (x, y, z)
      - if xt.shape[-1] == 6, then xt[..., 3:6] is used as the normal:
          * color_normal=False: draw normals as arrows (at most 256 per batch)
          * color_normal=True : color points by normal direction (no arrows)
    """
    assert xt.ndim == 3 and xt.shape[-1] in (3, 6), "Expect (B,N,3) or (B,N,6=xyz+normal)"
    device = xt.device
    B, N, C = xt.shape
    has_normals = (C == 6)

    out = torch.empty((B, 3, img_size, img_size), dtype=torch.float32)

    special_colors = [(1.0, 0.0, 0.0),   # first - red
                      (1.0, 1.0, 0.0),   # middle - yellow
                      (0.0, 0.0, 1.0)]   # last - blue
    special_size = max(8, int(point_radius * 8))
    base_size    = 0.1  # max(1, int(point_radius * 1.5))

    for b in range(B):
        pts_all = xt[b].detach().cpu().numpy()  # (N,C)
        pts = pts_all[:, :3].copy()             # (N,3)

        normals = None
        if has_normals:
            normals = pts_all[:, 3:6].copy()   # (N,3)

        # === Apply one rotation to the 3D points here, to "turn the airplane around" ===
        theta = np.deg2rad(90.0)
        Rx = np.array([
            [1.0,           0.0,           0.0],
            [0.0,  np.cos(theta), -np.sin(theta)],
            [0.0,  np.sin(theta),  np.cos(theta)],
        ], dtype=np.float32)
        pts = pts @ Rx.T   # (N,3)

        if normals is not None:
            normals = normals @ Rx.T

        pts[:, 0] *= -1.0  # flip x axis
        pts[:, 1] *= -1.0  # flip y axis
        if normals is not None:
            normals[:, 0] *= -1.0
            normals[:, 1] *= -1.0

        fig = Figure(
            figsize=(img_size/100.0, img_size/100.0),
            dpi=100,
            facecolor="white"
        )
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d')

        # === Draw the points first ===
        if has_normals and color_normal:
            # Color points by normal direction
            finite_mask = np.isfinite(pts).all(axis=1) & np.isfinite(normals).all(axis=1)
            p_valid = pts[finite_mask]
            n_valid = normals[finite_mask]

            if p_valid.shape[0] > 0:
                # Normalize the normal -> map to [0,1]
                n_norm = np.linalg.norm(n_valid, axis=1, keepdims=True)
                n_norm[n_norm < 1e-6] = 1.0
                n_dir = n_valid / n_norm              # (-1,1)
                colors = (n_dir + 1.0) / 2.0          # (M,3), mapped to [0,1]
                colors = np.clip(colors, 0.0, 1.0)

                ax.scatter(
                    p_valid[:, 0], p_valid[:, 1], p_valid[:, 2],
                    s=base_size,
                    c=colors.tolist(),
                    depthshade=True
                )
            # Non-finite points are not drawn
        else:
            # No normals, or color_normal disabled: draw everything in gray
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=base_size,
                c=[(0.35, 0.35, 0.35)],
                depthshade=True
            )

        # Color the first / middle / last 3 points (drawn on top)
        idxs = [0, N // 2, N - 1]
        for idx, color in zip(idxs, special_colors):
            if 0 <= idx < N and np.all(np.isfinite(pts[idx])):
                ax.scatter(
                    pts[idx, 0], pts[idx, 1], pts[idx, 2],
                    s=special_size,
                    c=[color],
                    depthshade=True
                )

        # Compute an equal-scale coordinate range to keep the three axes proportional
        mins = np.nanmin(pts, axis=0); maxs = np.nanmax(pts, axis=0)
        center = (mins + maxs) / 2.0
        half_range = (maxs - mins).max() / 2.0  # take half of the largest span
        half_range = max(half_range, 1e-6)  # avoid division by zero

        # Set equal-scale axis ranges
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)

        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

        # === Normal visualization (arrows); only drawn when has_normals and not color_normal ===
        if has_normals and not color_normal:
            finite_mask = np.isfinite(pts).all(axis=1) & np.isfinite(normals).all(axis=1)
            valid_idx = np.where(finite_mask)[0]

            if valid_idx.size > 0:
                max_arrows = 2048
                if valid_idx.size <= max_arrows:
                    idx_sel = valid_idx
                else:
                    step = max(1, valid_idx.size // max_arrows)
                    idx_sel = valid_idx[::step][:max_arrows]

                p_sel = pts[idx_sel]        # (M,3)
                n_sel = normals[idx_sel]    # (M,3)

                # Normalize directions
                n_norm = np.linalg.norm(n_sel, axis=1, keepdims=True)
                n_norm[n_norm < 1e-6] = 1.0
                n_dir = n_sel / n_norm      # (M,3)

                # Arrow length is determined by the point cloud bbox size
                bbox_size = float((maxs - mins).max()) if np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs)) else 1.0
                arrow_len = bbox_size * 0.1 if bbox_size > 0 else 0.1

                ax.quiver(
                    p_sel[:, 0], p_sel[:, 1], p_sel[:, 2],
                    n_dir[:, 0], n_dir[:, 1], n_dir[:, 2],
                    length=arrow_len,
                    normalize=True,
                    color=(0.0, 0.8, 0.0),
                    linewidth=0.1
                )

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.grid(False)

        # Viewpoint
        ax.view_init(elev=20, azim=-60)

        fig.tight_layout(pad=0)

        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        rgb = rgba[..., :3].astype(np.float32)
        a   = (rgba[..., 3:4].astype(np.float32) / 255.0)
        rgb = rgb * a + 255.0 * (1.0 - a)

        img = torch.from_numpy(rgb).permute(2, 0, 1) / 255.0
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(img_size, img_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        out[b] = img

    return out.to(device)

def render_point_images(
    xt: torch.Tensor,                 # (B, N, 2)
    img_size: int = 128,
    coords_are_normalized: bool = True,
    point_radius: int = 2,            # dot radius (pixels)
    in_out_dim: int = 2,
    background: float = 1.0,          # 1=white background
    point_value: float = 0.0,         # 0=black dots
    channels: int = 3,                # default to 3 channels for compatibility with imshow(grid.permute(1,2,0))
    antialias: bool = False,          # set True for smooth edges
    color_point: bool = True,
    # === 32 equal divisions (solid lines) ===
    draw_grid_x_32: bool = False,
    n_splits_x_32: int = 32,
    line_thickness_32: int = 4,
    line_value_32: float = 0.7,       # grayscale line (0=black, 1=white)
    # === 64 equal divisions (red dashed lines) ===
    draw_grid_x_64_dashed: bool = False,
    n_splits_x_64: int = 64,
    dash_thickness_64: int = 2,
    dash_len_64: int = 6,             # solid dash segment length
    gap_len_64: int = 6,              # gap length
    # === 64-bin count text ===
    show_counts_64: bool = False,
    min_text_gap_px: int = 14,        # minimum horizontal spacing between adjacent labels, to avoid overlap
    font_size: Optional[int] = None,     # font pixel height; None=automatic (~W/16, minimum 8)
    text_color=(0, 0, 0),             # black text
    text_stroke_width: int = 1,
    text_stroke_fill=(255, 255, 255), # white outline to improve readability
    x_scale = 1.0,
    y_scale = 1.0,
    color_normal: bool = False,
):
    """
    Render a point set:
      - 32 equal-division solid gray lines
      - 64 equal-division red dashed lines
      - point count per 1/64 bin, labeled on the "top left and right sides" of the corresponding dashed line
    """
    assert xt.ndim == 3
    device = xt.device
    B, N, _ = xt.shape
    H = W = int(img_size)

    if in_out_dim == 3 or  in_out_dim==6:
        return _render_point_images_3d(
            xt,
            img_size,
            coords_are_normalized,
            point_radius,
            background,
            channels,
            color_normal
        )

    # 1) Map to pixel coordinates
    x, y = xt[..., 0], xt[..., 1]
    if coords_are_normalized:
        xmin, xmax = float(x.min()), float(x.max())
        is_minus1_1 = (xmin < -1e-2) or (xmax > 1 + 1e-2) or (x.min() < 0) or (y.min() < 0)
        if is_minus1_1 and (x.min() >= -1.1 and x.max() <= 1.1 and y.min() >= -1.1 and y.max() <= 1.1):
            x_pix = ((x + 1.0) * 0.5 * x_scale) * (W - 1)
            y_pix = ((y + 1.0) * 0.5 * y_scale) * (H - 1)
        else:
            x_pix = x * (W - 1)
            y_pix = y * (H - 1)
    else:
        x_pix, y_pix = x, y
    
    y_pix = H - y_pix

    # Clean up invalid values and discretize to pixels
    finite_mask = torch.isfinite(x_pix) & torch.isfinite(y_pix)
    x_idx = x_pix.round().clamp(0, W - 1).long()
    y_idx = y_pix.round().clamp(0, H - 1).long()

    # 2) Drop the points as "impulses" onto the pixel plane (one-shot scatter)
    impulses = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    b_ids = torch.arange(B, device=device).view(B, 1).expand(B, N)
    lin = (b_ids * (H * W) + y_idx * W + x_idx)
    lin = lin[finite_mask]
    impulses.view(-1).index_put_((lin.reshape(-1),), torch.ones(lin.numel(), device=device), accumulate=True)

    # 3) Dots (dilation / antialiasing)
    r = int(max(0, point_radius))
    if r == 0:
        dots = impulses
    else:
        yy, xx = torch.meshgrid(
            torch.arange(-r, r + 1, device=device),
            torch.arange(-r, r + 1, device=device),
            indexing='ij'
        )
        if antialias:
            sigma = max(0.5, r / 2)
            ker = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        else:
            ker = ((xx**2 + yy**2) <= (r * r)).float()
        ker = ker / ker.max()
        ker = ker.view(1, 1, *ker.shape)                # (1,1,Kh,Kw)
        dots = F.conv2d(impulses, ker, padding=r)       # (B,1,H,W)
        dots = (dots > 0).float() if not antialias else dots.clamp(0, 1)

    # 4) Composite foreground/background (single-channel grayscale)
    img = background * torch.ones_like(dots)
    img = img * (1 - dots) + point_value * dots  # (B,1,H,W)

    # 5) 32 equal divisions: solid gray lines
    if draw_grid_x_32 and n_splits_x_32 and n_splits_x_32 > 1 and line_thickness_32 > 0:
        xs32 = torch.linspace(0, W - 1, steps=n_splits_x_32 + 1, device=device, dtype=torch.float32)[1:-1]  # (L,)
        cols = torch.arange(W, device=device, dtype=torch.float32)
        half32 = max(0.5, (float(line_thickness_32) - 1.0) / 2.0)
        dist32 = (cols.unsqueeze(0) - xs32.unsqueeze(1)).abs()                                              # (L, W)
        mask_1d_32 = (dist32 <= half32).any(dim=0).float()                                                  # (W,)
        line_mask_32 = mask_1d_32.view(1, 1, 1, W).expand(B, 1, H, W)
        img = img * (1.0 - line_mask_32) + line_value_32 * line_mask_32

    # 6) Three-channel output (produce color first, then overlay red lines/text)
    if channels == 3:
        img = img.repeat(1, 3, 1, 1)  # (B,3,H,W)
    else:
        # Even for single channel, first convert to three channels for visualization enhancement, then restore at the end
        img = img.repeat(1, 3, 1, 1)
        _to_gray_later = True
    if color_point:
        B, N = xt.shape[0], xt.shape[1]
        H = W = int(img_size)
        r = int(max(0, point_radius))
        r_sel = int(math.ceil(2.0 * r)) if r > 0 else 1  # 1.5x; if the original is 0, set to 1 pixel to ensure visibility

        # Reuse the same-style kernel, but with radius r_sel
        if r_sel > 0:
            yy_sel, xx_sel = torch.meshgrid(
                torch.arange(-r_sel, r_sel + 1, device=xt.device),
                torch.arange(-r_sel, r_sel + 1, device=xt.device),
                indexing='ij'
            )
            if antialias:
                sigma_sel = max(0.5, r_sel / 2)
                ker_sel = torch.exp(-(xx_sel**2 + yy_sel**2) / (2 * sigma_sel**2))
            else:
                ker_sel = ((xx_sel**2 + yy_sel**2) <= (r_sel * r_sel)).float()
            ker_sel = ker_sel / ker_sel.max()
            ker_sel = ker_sel.view(1, 1, *ker_sel.shape)  # (1,1,Kh,Kw)

        # Color table (RGB, 0..1)
        color_map = {
            0:   torch.tensor([1.0, 0.0, 0.0], device=xt.device),  # red: first point
            -1:  torch.tensor([0.0, 0.0, 1.0], device=xt.device),  # blue: last point
            511: torch.tensor([1.0, 1.0, 0.0], device=xt.device),  # yellow: 512th point (0-based=511)
        }

        sel_indices = [0]
        if N >= 1:   sel_indices.append(N - 1)
        if N >= 512: sel_indices.append(511)

        for idx in sel_indices:
            key = (idx if idx != (N - 1) else -1)
            color = color_map[key].view(1, 3, 1, 1)

            valid = finite_mask[:, idx]  # (B,)
            if not valid.any():
                continue

            # Single-point impulse image
            imp = torch.zeros((B, 1, H, W), device=xt.device, dtype=torch.float32)
            b_lin = torch.arange(B, device=xt.device) * (H * W)
            lin = b_lin + (y_idx[:, idx] * W + x_idx[:, idx])
            lin = lin[valid]
            imp.view(-1).index_put_((lin,), torch.ones_like(lin, dtype=torch.float32, device=xt.device))

            # Expand into a "dot": radius r_sel
            if r_sel == 0:
                dots_sel = imp
            else:
                dots_sel = F.conv2d(imp, ker_sel, padding=r_sel)
                dots_sel = dots_sel.clamp(0, 1) if antialias else (dots_sel > 0).float()

            # Overlay onto the color image: alpha blending
            alpha = dots_sel.expand(B, 3, H, W)
            img = img * (1.0 - alpha) + color.expand_as(img) * alpha

    if channels == 1:
        img = img[:, :1, :, :]

    return img
def extract_epoch(ckpt_name: str) -> int:
    """
    Extract the epoch number from a checkpoint filename.
    For example:
    'UncondVoronoiTransformer_epoch201_e200000_1761086335' -> 201
    """
    match = re.search(r'epoch(\d+)', ckpt_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot find epoch number in '{ckpt_name}'")

def build_k_radial(max_norm: int = 5, exclude_zero: bool = True) -> torch.Tensor:
    """Generate the list of integer wave vectors with |k|<=max_norm, excluding (0,0). Returns a [K,2] float32 tensor."""
    ks = []
    for kx in range(-max_norm, max_norm + 1):
        for ky in range(-max_norm, max_norm + 1):
            if exclude_zero and kx == 0 and ky == 0:
                continue
            if kx * kx + ky * ky <= max_norm * max_norm:
                ks.append([kx, ky])
    K = torch.tensor(ks, dtype=torch.float32)  # [K,2]
    return K

@torch.no_grad()
def save_trajectory_frames(
    xs: torch.Tensor,
    frames_dir: str,
    img_size: int,
    point_radius: int,
    nrow: int,
):
    """
    xs: (B, T, N, 2), coordinate range consistent with render_point_images at training time (here it is [-1,1])
    """
    os.makedirs(frames_dir, exist_ok=True)
    B, T, _, _ = xs.shape

    for i in range(T):
        # Take the B sample point sets of the i-th frame -> (B, N, 2)
        x_t = xs[:, i]  # (B,N,2)

        # Use the existing rendering tool to render the point sets into B images
        imgs = render_point_images(
            x_t,
            img_size=img_size,
            point_radius=point_radius,
            channels=3,
            background=1.0,
            point_value=0.0,
            antialias=False
        )  # imgs: (B,3,H,W)

        grid = make_grid(imgs, nrow=nrow)  # (3, H*, W*)
        fpath = os.path.join(frames_dir, f"fr_{i}.png")
        save_image(grid, fpath)  # Save directly as PNG
    return T


def write_video_from_frames(frames_dir: str, out_video_path: str, fps: int = 12) -> bool:
    import imageio.v2 as iio
    pattern = os.path.join(frames_dir, "fr_*.png")
    frame_paths = glob.glob(pattern)

    # Natural sort: order by the <num> in fr_<num>.png
    def frame_idx(p):
        m = re.search(r'fr_(\d+)\.png$', os.path.basename(p))
        return int(m.group(1)) if m else float('inf')

    frame_paths.sort(key=frame_idx)

    if not frame_paths:
        print(f"[trajectory] no frames found under: {frames_dir}")
        return False

    try:
        with iio.get_writer(out_video_path, fps=fps, codec="libx264", macro_block_size=None) as w:
            for fp in frame_paths:
                w.append_data(iio.imread(fp))
        print(f"[trajectory] video saved: {os.path.abspath(out_video_path)}")
        return True
    except Exception as e:
        # If you kept the fallback hint to ffmpeg, it is recommended to also switch to fr_%d or %06d here, depending on your save naming
        print(f"[trajectory] mp4 compose failed ({e}). Frames are saved; "
              f'you can run:\n  ffmpeg -y -r {fps} -i "{frames_dir}/fr_%d.png" '
              f'-pix_fmt yuv420p "{out_video_path}"')
        return False

def _dilate_1bit_32_to_64(v: torch.Tensor) -> torch.Tensor:
    # v: int64 tensor, but values must be in [0, 2^32-1]
    v = v & 0x00000000FFFFFFFF
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8))  & 0x00FF00FF00FF00FF
    v = (v | (v << 4))  & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2))  & 0x3333333333333333
    v = (v | (v << 1))  & 0x5555555555555555
    return v

def morton2d_codes_from_int_xy(xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
    """
    xi, yi: (B, N) int64, with each value < 2^p; p<=31 (ensures 2p<=62 so the sign bit is not triggered)
    return: (B, N) int64 Morton code
    """
    x = _dilate_1bit_32_to_64(xi.long())
    y = _dilate_1bit_32_to_64(yi.long())
    return x | (y << 1)

# ---- Main function: convert (B,N,2) float coordinates to Morton codes and sort by them ----
def z_order_sort(points: torch.Tensor,
                 p: int = 16,
                 in_unit_square: bool = True,
                 return_codes: bool = False):
    """
    points: (B, N, 2) float (half/single/double precision all work)
    p: quantize to p bits (per coordinate), recommended 8~21; maximum 31
    in_unit_square: True means coordinates are already in [0,1]; False applies per-batch min/max linear normalization
    return_codes: whether to return the Morton code

    Returns:
      sorted_points: (B, N, 2)
      sort_idx:      (B, N) sort indices along dim=1
      [optional] codes:  (B, N) Morton code (int64)
    """
    assert points.ndim == 3 and points.shape[-1] == 2, "points must be (B,N,2)"
    assert 1 <= p <= 31, "p must be in [1,31] so that 2p <= 62 bits"
    B, N, _ = points.shape
    device = points.device
    scale = (1 << p) - 1

    xy = points
    if in_unit_square:
        xy_norm = xy.clamp(0.0, 1.0)
    else:
        # Adaptively normalize each batch to [0,1]
        xy_min = xy.amin(dim=1, keepdim=True)
        xy_max = xy.amax(dim=1, keepdim=True)
        xy_range = (xy_max - xy_min).clamp_min(1e-12)
        xy_norm = (xy - xy_min) / xy_range
        xy_norm = xy_norm.clamp(0.0, 1.0)

    # Quantize to integer grid coordinates
    xi = (xy_norm[..., 0] * scale).floor().clamp(0, scale).to(torch.int64)
    yi = (xy_norm[..., 1] * scale).floor().clamp(0, scale).to(torch.int64)

    # Morton/Z-order code
    codes = morton2d_codes_from_int_xy(xi, yi)  # (B, N) int64

    # Sort by Morton code (dim=1); for stable tie-breaking within the same cell, optionally do a secondary sort by x/y
    sort_idx = torch.argsort(codes, dim=1)
    sorted_points = torch.gather(xy, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))

    if return_codes:
        return sorted_points, sort_idx, codes
    return sorted_points, sort_idx

# ---- Convenience function returning only the indices ----
def z_order_argsort(points: torch.Tensor, **kwargs) -> torch.Tensor:
    _, idx = z_order_sort(points, return_codes=False, **kwargs)
    return idx




import os
import numpy as np

def save_pointcloud_ply(points_np: np.ndarray,
                        out_path: str,
                        colors_np: np.ndarray = None,
                        binary: bool = True,
                        sanitize: bool = True):
    """
    Write a PLY correctly (per-vertex interleaved: x y z r g b)
    points_np: (N,3) float
    colors_np: (N,3) uint8, optional; if not provided, everything is gray
    """
    assert points_np.ndim == 2 and points_np.shape[1] == 3, "points must be (N,3)"
    pts = np.asarray(points_np, dtype=np.float32)

    # Clean up invalid values (can be disabled)
    if sanitize:
        mask = np.isfinite(pts).all(axis=1)
        if colors_np is not None:
            colors_np = np.asarray(colors_np, dtype=np.uint8)[mask]
        pts = pts[mask]

    N = pts.shape[0]
    if colors_np is None:
        colors = np.full((N, 3), 200, dtype=np.uint8)
    else:
        colors = np.asarray(colors_np, dtype=np.uint8)
        assert colors.shape == (N, 3)

    header = (
        "ply\n"
        f"{'format binary_little_endian 1.0' if binary else 'format ascii 1.0'}\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if binary:
        # Use a structured dtype to guarantee per-vertex interleaving (no padding)
        dtype = np.dtype([
            ('x','<f4'), ('y','<f4'), ('z','<f4'),
            ('red','u1'), ('green','u1'), ('blue','u1'),
        ], align=False)
        rec = np.empty(N, dtype=dtype)
        rec['x'] = pts[:, 0]
        rec['y'] = pts[:, 1]
        rec['z'] = pts[:, 2]
        rec['red']   = colors[:, 0]
        rec['green'] = colors[:, 1]
        rec['blue']  = colors[:, 2]

        with open(out_path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(rec.tobytes())
    else:
        with open(out_path, "w") as f:
            f.write(header)
            for (x, y, z), (r, g, b) in zip(pts, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")



def export_ply_batch_with_highlights(X: torch.Tensor, out_dir: str, prefix: str = "pts", normalize: str = "keep"):
    """
    X: (B,N,3) torch.Tensor
    Save one ply per batch, with annotations:
      - first: red (255,0,0)
      - middle: yellow (255,255,0)
      - last: blue (0,0,255)
    normalize: 'keep' | '01' | 'm11' (whether to remap coordinates)
    """
    os.makedirs(out_dir, exist_ok=True)
    B, N, D = X.shape
    assert D == 3, "Only for 3D"

    for i in range(B):
        pts = X[i].detach().cpu().numpy().astype(np.float32)  # (N,3)

        if normalize in ("01", "m11"):
            mins = np.nanmin(pts, axis=0)
            maxs = np.nanmax(pts, axis=0)
            span = np.maximum(maxs - mins, 1e-8)
            pts = (pts - mins) / span
            if normalize == "m11":
                pts = pts * 2.0 - 1.0

        colors = np.full((N, 3), 200, dtype=np.uint8)
        special = [0, N // 2, N - 1]
        special_colors = [(255, 0, 0), (255, 255, 0), (0, 0, 255)]
        for idx, col in zip(special, special_colors):
            if 0 <= idx < N:
                colors[idx] = np.array(col, dtype=np.uint8)

        out_path = os.path.join(out_dir, f"{prefix}_{i}.ply")
        save_pointcloud_ply(pts, out_path, colors_np=colors, binary=True)

def normalize_3d_unit_box(t: torch.Tensor, dims: int) -> torch.Tensor:
    """
    t: (N, dims) float32
    If dims==3 and enable=True, scale the point cloud to the [-1,1]^3 bounding box.
    """
    if dims != 3:
        return t
    # Maximum absolute value over all coordinates of the entire point cloud
    max_abs = t.abs().max()
    if max_abs > 0:
        t = t / max_abs
    t = t * 0.5 + 0.5
    return t


def normalize_and_orient_vertices(V: np.ndarray) -> np.ndarray:
    """
    Apply a basic normalization to the mesh vertices + place them into the [0,1]^3 unit box.

    1. Translate the centroid to the origin
    2. Normalize to [-1,1] based on max_abs
    3. Map to [0,1]^3:  t = t * 0.5 + 0.5
    """
    V = np.asarray(V, dtype=np.float32)
    if V.size == 0:
        return V

    centroid = V.mean(axis=0, keepdims=True)
    Vc = V - centroid

    max_abs = np.abs(Vc).max()
    if max_abs > 0:
        Vc = Vc / max_abs

    Vc = Vc * 0.5 + 0.5

    # To [-1, 1]^3
    Vc = Vc * 2.0 - 1.0
    return Vc


def find_all_ply(root: Path) -> List[Path]:
    """Recursively search for all .ply files under root."""
    return sorted(root.rglob("*.ply"))


def nearest_one_job_proc(bi: int, V: np.ndarray, F: np.ndarray, P: np.ndarray):
    """
    A single task for multiprocessing:
    - Build the AABB for (V,F) inside the subprocess
    - Run a squared_distance query for the point set P (N,3)
    Returns:
        (bi, C) where C is the (N,3) nearest-point coordinates
    """
    tree = igl.AABB()
    tree.init(V, F)
    _, _, C = tree.squared_distance(V, F, P)
    return bi, C


@njit
def hilbert_index_3d(ix: np.int64, iy: np.int64, iz: np.int64, p_bits: int) -> np.int64:
    """
    Compute the 3D Hilbert index of a single point.
    Assumes (ix, iy, iz) have already been quantized to [0, 2^p_bits - 1].
    """
    x0 = ix
    x1 = iy
    x2 = iz

    # Inverse Gray code (the standard approach in Skilling's paper)
    t = x2 >> 1
    x2 ^= x1
    x1 ^= x0
    x0 ^= t

    # Rotation / swap
    Q = np.int64(2)
    limit = np.int64(1 << p_bits)
    while Q < limit:
        P = Q - 1

        if (x2 & Q) != 0:
            x0 ^= P
        else:
            tt = (x0 ^ x2) & P
            x0 ^= tt
            x2 ^= tt

        if (x1 & Q) != 0:
            x0 ^= P
        else:
            tt = (x0 ^ x1) & P
            x0 ^= tt
            x1 ^= tt

        if (x0 & Q) != 0:
            x0 ^= P
        Q <<= 1

    # Assemble into a linear index (3 dimensions, 3 bits per level)
    idx = np.int64(0)
    for bit in range(p_bits - 1, -1, -1):
        idx = (idx << 1) | ((x0 >> bit) & 1)
        idx = (idx << 1) | ((x1 >> bit) & 1)
        idx = (idx << 1) | ((x2 >> bit) & 1)
    return idx

@njit  # No longer parallel=True, to avoid competing for threads with the outer multiprocessing
def hilbert_indices_int_3d(coords_q: np.ndarray, p_bits: int) -> np.ndarray:
    """
    coords_q: (N,3) int64, each in [0, 2^p_bits-1]
    """
    N = coords_q.shape[0]
    out = np.empty(N, dtype=np.int64)
    for i in range(N):  # prange would also work, but there is no need to parallelize
        out[i] = hilbert_index_3d(coords_q[i, 0], coords_q[i, 1], coords_q[i, 2], p_bits)
    return out


# ===================== Numba-accelerated N-dimensional Hilbert sort =====================

@njit(cache=True)
def _axes_to_transposed_index_numba(X: np.ndarray, bits: int) -> None:
    """
    In-place conversion of coordinate axes into the Hilbert transposed index (Skilling's algorithm)
    X: (D,) uint64 array, modified in place
    bits: number of bits per dimension
    """
    n = X.shape[0]
    M = np.uint64(1) << np.uint64(bits - 1)
    Q = M

    # --- Inverse undo (Skilling: first half of AxestoTranspose) ---
    while Q > np.uint64(1):
        P = Q - np.uint64(1)
        for i in range(n):
            if (X[i] & Q) != np.uint64(0):
                X[0] ^= P  # invert
            else:
                t = (X[0] ^ X[i]) & P
                X[0] ^= t
                X[i] ^= t
        Q >>= np.uint64(1)

    # --- Gray encode (second half) ---
    for i in range(1, n):
        X[i] ^= X[i - 1]

    t = np.uint64(0)
    Q = M
    while Q > np.uint64(1):
        if (X[n - 1] & Q) != np.uint64(0):
            t ^= (Q - np.uint64(1))
        Q >>= np.uint64(1)

    for i in range(n):
        X[i] ^= t


@njit(cache=True)
def _transposed_to_hilbert_integer_numba(X: np.ndarray, bits: int) -> np.uint64:
    """
    Convert the transposed Hilbert index into a single integer
    X: (D,) uint64 array
    bits: number of bits per dimension
    Returns: Hilbert index (uint64)
    """
    n = X.shape[0]
    h = np.uint64(0)

    # From high bit to low bit, interleave each dimension's bits into a single integer
    for bit in range(bits - 1, -1, -1):
        mask = np.uint64(1) << np.uint64(bit)
        for i in range(n):
            h = (h << np.uint64(1)) | ((X[i] & mask) >> np.uint64(bit))

    return h


@njit(cache=True)
def hilbert_indices_int_nd(coords: np.ndarray, bits: int) -> np.ndarray:
    """
    Compute the Hilbert indices of an (N, D) point set (Numba-accelerated version)

    coords: (N, D) uint64 integer coordinates, each dimension in [0, 2^bits - 1]
    bits: number of bits per dimension

    Returns:
        dists: (N,) Hilbert index (uint64), can be used for sorting
    """
    N, D = coords.shape
    dists = np.empty(N, dtype=np.uint64)
    X = np.empty(D, dtype=np.uint64)  # Preallocate the work array to avoid allocating on every loop

    for i in range(N):
        # Copy coordinates into the work array (modified in place)
        for j in range(D):
            X[j] = coords[i, j]
        # Convert and compute the Hilbert index
        _axes_to_transposed_index_numba(X, bits)
        dists[i] = _transposed_to_hilbert_integer_numba(X, bits)

    return dists


def hilbert_sort_nd(points: np.ndarray, p_bits: int = 10, return_order: bool = False):
    """
    Sort an (N,D) point set along a D-dimensional Hilbert curve (Skilling's algorithm, Numba-accelerated, D>=2)

    points: (N,D) float or int both work
    p_bits: quantization bits per dimension (e.g. 10 -> 0~1023 per dimension)

    Returns:
      - return_order=False: (N,D) sorted points
      - return_order=True : (pts_sorted, order)
    """
    if points.ndim != 2:
        raise ValueError(f"Expected (N,D), got {points.shape}")

    N, D = points.shape
    if D < 2:
        raise ValueError("The Hilbert curve requires dimension D >= 2")

    points = np.asarray(points, dtype=np.float32)

    # 1. Normalize to [0,1]^D
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = maxs - mins
    span[span == 0.0] = 1.0
    norm = (points - mins) / span

    # 2. Quantize to the integer grid [0, 2^p_bits - 1]
    grid_max = (1 << p_bits) - 1
    coords = np.floor(norm * grid_max + 1e-12).astype(np.uint64)
    coords = np.clip(coords, 0, grid_max).astype(np.uint64)

    # 3. Compute the N-dimensional Hilbert index (Numba-accelerated)
    dists = hilbert_indices_int_nd(coords, p_bits)

    # 4. Stable sort
    order = np.argsort(dists, kind="mergesort")
    pts_sorted = points[order]

    if return_order:
        return pts_sorted, order
    return pts_sorted


# ===================== Pure Python version (kept for reference) =====================

def transposed_to_hilbert_integer_python(X: np.ndarray, bits: int) -> np.uint64:
    """
    X: (D,) transposed Hilbert index, integer
    bits: number of bits per dimension
    Returns:
        a single Hilbert index (uint64)
    """
    X = np.asarray(X, dtype=np.uint64)  # extra safety
    n = X.size
    h = np.uint64(0)

    # From high bit to low bit, interleave each dimension's bits into a single integer
    for bit in range(bits - 1, -1, -1):
        mask = np.uint64(1) << np.uint64(bit)
        for i in range(n):
            h = (h << np.uint64(1)) | ((X[i] & mask) >> np.uint64(bit))

    return h

def axes_to_transposed_index_python(axes, bits: int) -> np.ndarray:
    """
    axes: 1D coordinate vector, length = dimension D, elements are integers in [0, 2^bits - 1]
    bits: number of bits used per dimension
    Returns:
        X: (D,) transposed Hilbert index, dtype=uint64
    """
    # Force conversion to uint64 to avoid issues with float / int64, etc.
    X = np.asarray(axes, dtype=np.uint64).copy()
    n = X.size

    M = np.uint64(1) << np.uint64(bits - 1)  # highest-bit mask, e.g. bits=5 -> 16
    Q = M

    # --- Inverse undo (Skilling: first half of AxestoTranspose) ---
    while Q > 1:
        P = Q - np.uint64(1)
        for i in range(n):
            if (X[i] & Q) != 0:
                X[0] ^= P  # invert
            else:
                t = (X[0] ^ X[i]) & P
                X[0] ^= t
                X[i] ^= t
        Q >>= np.uint64(1)

    # --- Gray encode (second half) ---
    for i in range(1, n):
        X[i] ^= X[i - 1]

    t = np.uint64(0)
    Q = M
    while Q > 1:
        if (X[n - 1] & Q) != 0:
            t ^= (Q - np.uint64(1))
        Q >>= np.uint64(1)

    for i in range(n):
        X[i] ^= t

    return X

def hilbert_indices_int_nd_python(coords: np.ndarray, bits: int) -> np.ndarray:
    """
    coords: (N, D) integer coordinates, each dimension in [0, 2^bits - 1]
    bits: number of bits per dimension

    Returns:
        dists: (N,) Hilbert index (uint64), can be used for sorting
    """
    if coords.ndim != 2:
        raise ValueError(f"Expected (N,D), got {coords.shape}")

    N, D = coords.shape
    dists = np.empty(N, dtype=np.uint64)

    for i in range(N):
        X = axes_to_transposed_index_python(coords[i], bits)
        dists[i] = transposed_to_hilbert_integer_python(X, bits)

    return dists

def hilbert_sort_nd_python(points: np.ndarray, p_bits: int = 10, return_order: bool = False):
    """
    Sort an (N,D) point set along a D-dimensional Hilbert curve (Skilling's algorithm, D>=2)
    Pure Python implementation, slower; prefer hilbert_sort_nd (Numba-accelerated version)

    points: (N,D) float or int both work
    p_bits: quantization bits per dimension (e.g. 10 -> 0~1023 per dimension)

    Returns:
      - return_order=False: (N,D) sorted points
      - return_order=True : (pts_sorted, order)
    """
    if points.ndim != 2:
        raise ValueError(f"Expected (N,D), got {points.shape}")

    N, D = points.shape
    if D < 2:
        raise ValueError("The Hilbert curve requires dimension D >= 2")

    points = np.asarray(points, dtype=np.float32)

    # 1. Normalize to [0,1]^D
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = maxs - mins
    span[span == 0.0] = 1.0
    norm = (points - mins) / span

    # 2. Quantize to the integer grid [0, 2^p_bits - 1]
    grid_max = (1 << p_bits) - 1
    coords = np.floor(norm * grid_max + 1e-12).astype(np.uint64)
    coords = np.clip(coords, 0, grid_max).astype(np.uint64)

    # 3. Compute the N-dimensional Hilbert index
    dists = hilbert_indices_int_nd_python(coords, p_bits)

    # 4. Stable sort
    order = np.argsort(dists, kind="mergesort")
    pts_sorted = points[order]

    if return_order:
        return pts_sorted, order
    return pts_sorted

def hilbert_sort_xyz_numba(points: np.ndarray, p_bits: int = 10, return_order: bool = False) -> np.ndarray:
    """
    Sort an (N,3) point set along a 3D Hilbert curve (strict Hilbert, Numba-accelerated).

    Steps:
      1) min-max to [0,1]^3
      2) quantize to [0, 2^p_bits-1]^3
      3) compute the Hilbert index and use a stable sort
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N,3), got {points.shape}")

    # 1. Normalize to [0,1]^3
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = maxs - mins
    span[span == 0.0] = 1.0
    norm = (points - mins) / span

    # 2. Quantize to the integer grid
    grid_max = (1 << p_bits) - 1
    coords = np.floor(norm * grid_max + 1e-12).astype(np.int64)
    coords = np.clip(coords, 0, grid_max)

    # 3. Compute the Hilbert index and stable-sort
    dists = hilbert_indices_int_3d(coords, p_bits)
    order = np.argsort(dists, kind="mergesort")
    pts_sorted = points[order]

    if return_order:
        return pts_sorted, order
    return pts_sorted



def sample_points_on_mesh(V: np.ndarray, F: np.ndarray, num_points: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Uniformly sample num_points points from the surface of the (V, F) triangle mesh.

    V: (M,3) float
    F: (K,3) int
    Returns: (num_points,3) float64
    """
    V = np.asarray(V, dtype=np.float32)
    F = np.asarray(F, dtype=np.int64)

    tris = V[F]               # (K, 3, 3)
    v0 = tris[:, 0, :]
    v1 = tris[:, 1, :]
    v2 = tris[:, 2, :]

    # Area of each triangle
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()
    if total_area <= 0:
        # Degenerate mesh: just jitter slightly around all vertices
        idx = rng.integers(0, V.shape[0], size=num_points)
        return V[idx] + 1e-6 * rng.standard_normal((num_points, 3))

    probs = areas / total_area

    # First draw triangle indices according to the area distribution
    tri_idx = rng.choice(len(F), size=num_points, replace=True, p=probs)
    tri = tris[tri_idx]  # (num_points, 3, 3)

    # Then sample within each triangle using uniform barycentric coordinates
    r1 = rng.random(num_points)
    r2 = rng.random(num_points)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = r2 * sqrt_r1
    w = 1.0 - u - v

    pts = (
        tri[:, 0, :] * u[:, None] +
        tri[:, 1, :] * v[:, None] +
        tri[:, 2, :] * w[:, None]
    )
    return pts


def sample_and_hilbert_one_mesh(
    bi: int,
    V: np.ndarray,
    F: np.ndarray,
    num_points: int,
    hilbert_p: int,
    seed: int,
) -> Tuple[int, np.ndarray]:
    """
    Sample num_points surface points on a single mesh and apply a Hilbert sort.
    Returns (batch_index, points_sorted) for convenient aggregation with Parallel.
    """
    rng = np.random.default_rng(seed)
    pts = sample_points_on_mesh(V, F, num_points, rng)           # (N,3) float64
    pts_sorted = hilbert_sort_xyz_numba(pts, p_bits=hilbert_p)   # (N,3) float64
    return bi, pts_sorted




def normalize_vertices_minus1_1(V: np.ndarray) -> np.ndarray:
    """
    Normalize the vertices V to [-1,1]^3:
      1) translate so the centroid is 0
      2) scale by max_abs to [-1,1]^3
    """
    V = np.asarray(V, dtype=np.float32)
    if V.size == 0:
        return V
    centroid = V.mean(axis=0, keepdims=True)
    Vc = V - centroid
    max_abs = np.abs(Vc).max()
    if max_abs > 0:
        Vc = Vc / max_abs
    return Vc


def project_points_to_sphere(points: np.ndarray, radius: float = math.sqrt(2.0)) -> np.ndarray:
    """
    Radially project each point onto the sphere centered at the origin with radius `radius`.
    points: (N,3) -> (N,3)
    """
    pts = np.asarray(points, dtype=np.float32)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)  # (N,1)
    out = np.zeros_like(pts)

    eps = 1e-12
    mask = norms[:, 0] > eps
    out[mask] = pts[mask] / norms[mask] * radius
    out[~mask] = np.array([radius, 0.0, 0.0], dtype=np.float32)
    return out

def poisson_points_one_mesh_noclean_and_hermite_ot_one_batch(
    bi: int,
    x0: np.ndarray,   # (N,3)
    V: np.ndarray,
    F: np.ndarray,
    num_points: int,
    init_factor: float,
    lambda_orient: float,
    ot_solver: str,
    hermite_degree: str,
    mesh_idx: int,
    zero_t0: bool,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    For a single batch index = bi:
      1) Run Poisson-disk sampling on the mesh (without cleaning)
      2) Construct n1 = -normal
      3) Call hermite_ot_one_batch for Hermite-style cost + OT / greedy matching

    Returns:
      (bi, x1_matched, n0, n1_matched)
    """
    # Poisson sampling first
    _, pts, normals = poisson_points_one_mesh_noclean(
        batch_idx=bi,
        V=V,
        F=F,
        num_points=num_points,
        init_factor=init_factor,
    )

    if pts.shape != (num_points, 3) or normals.shape != (num_points, 3):
        raise RuntimeError(
            f"Unexpected Poisson result shape: "
            f"pts={pts.shape}, normals={normals.shape}, expected {(num_points,3)}"
        )

    # Consistent with the original dataset logic: negate the normals
    n1 = -normals

    # Then run Hermite OT
    _, x1_matched, n0, n1_matched = hermite_ot_one_batch(
        bi=bi,
        x0=pts if x0 is None else x0,   # under normal circumstances x0 is not None
        x1=pts,
        n1=n1,
        lambda_orient=lambda_orient,
        ot_solver=ot_solver,
        hermite_degree=hermite_degree,
        mesh_idx=mesh_idx,
        V=V,
        F=F,
        zero_t0=zero_t0,
    )
    return bi, x1_matched, n0, n1_matched


def random_points_one_mesh_noclean_and_hermite_ot_one_batch(
    bi: int,
    x0: np.ndarray,   # (N,3)
    V: np.ndarray,
    F: np.ndarray,
    num_points: int,
    lambda_orient: float,
    ot_solver: str,
    hermite_degree: str,
    mesh_idx: int,
    zero_t0: bool,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Uniform surface sampling + Hermite OT for a single mesh:

      1) Build an Open3D TriangleMesh from V, F
      2) sample_points_uniformly to sample num_points points
      3) n1 = -normal
      4) Call hermite_ot_one_batch for Hermite-style cost + OT / greedy matching

    Returns:
      (bi, x1_matched, n0, n1_matched)
    """
    # Uniform surface sampling (consistent with the original code in compute_batch)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)

    if hermite_degree == "linear":
        pcd = mesh.sample_points_uniformly(
            number_of_points=int(num_points),
        )
        pts = np.asarray(pcd.points, dtype=np.float32)

        if pts.shape != (num_points, 3):
            raise RuntimeError(
                f"Unexpected uniform sample result shape: "
                f"pts={pts.shape}, expected {(num_points,3)}"
            )

        _, x1_matched, n0, n1_matched = hermite_ot_one_batch(
            bi=bi,
            x0=x0,
            x1=pts,
            n1=None,
            lambda_orient=lambda_orient,
            ot_solver=ot_solver,
            hermite_degree=hermite_degree,
            mesh_idx=mesh_idx,
            V=V,
            F=F,
            zero_t0=zero_t0,
        )
        return bi, x1_matched, None, None


    # Vertex normals are needed so that sample_points_uniformly interpolates point-cloud normals
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_uniformly(
        number_of_points=int(num_points),
        use_triangle_normal=False
    )

    pts = np.asarray(pcd.points, dtype=np.float32)
    normals = -np.asarray(pcd.normals, dtype=np.float32)

    if pts.shape != (num_points, 3) or normals.shape != (num_points, 3):
        raise RuntimeError(
            f"Unexpected uniform sample result shape: "
            f"pts={pts.shape}, normals={normals.shape}, expected {(num_points,3)}"
        )

    n1 = -normals

    _, x1_matched, n0, n1_matched = hermite_ot_one_batch(
        bi=bi,
        x0=x0,
        x1=pts,
        n1=n1,
        lambda_orient=lambda_orient,
        ot_solver=ot_solver,
        hermite_degree=hermite_degree,
        mesh_idx=mesh_idx,
        V=V,
        F=F,
        zero_t0=zero_t0,
    )
    return bi, x1_matched, n0, n1_matched


def poisson_points_one_mesh_noclean(
    batch_idx: int,
    V: np.ndarray,
    F: np.ndarray,
    num_points: int,
    init_factor: float = 4.0,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Poisson sampling for a single mesh (no cleaning, since cleaning was already done in dataset._load_meshes):

      - Build an Open3D TriangleMesh from V, F
      - Compute vertex normals and unify their orientation
      - sample_points_poisson_disk to sample N points
      - Return the Nx3 point coordinates and corresponding normals

    Returns:
      (batch_idx, pts, normals)
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    mesh.compute_vertex_normals()


    pcd = mesh.sample_points_poisson_disk(
        number_of_points=int(num_points),
        init_factor=float(init_factor),
    )
    pts = np.asarray(pcd.points, dtype=np.float32)

    if pcd.has_normals():
        normals = -np.asarray(pcd.normals, dtype=np.float32)
    else:
        raise RuntimeError("No normal!")

    if pts.shape[0] != num_points:
        RuntimeError("Poisson sampling produced less than N points.")

    return batch_idx, pts, normals

def poisson_sphere_one_mesh(
    bi: int,
    V: np.ndarray,
    F: np.ndarray,
    num_points: int,
    radius: float,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Poisson + sphere projection for a single mesh:
      - Build an Open3D TriangleMesh from V, F
      - sample_points_poisson_disk to sample N points as x1
      - Project x1 onto the sphere of radius `radius` to obtain x0

    Returns:
      (bi, x0, x1), where x0/x1 shape: (N,3), float64
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    # mesh.compute_vertex_normals()

    pcd = mesh.sample_points_poisson_disk(
        number_of_points=int(num_points),
        init_factor=4.0,
        use_triangle_normal=False,
    )
    pts = np.asarray(pcd.points, dtype=np.float32)  # (M,3)
    if pts.shape[0] != num_points:
        # Usually they are equal; if not, simply pad/truncate
        if pts.shape[0] > 0 and pts.shape[0] < num_points:
            repeat = num_points - pts.shape[0]
            extra_idx = np.random.choice(pts.shape[0], size=repeat, replace=True)
            pts = np.concatenate([pts, pts[extra_idx]], axis=0)
        elif pts.shape[0] == 0:
            pts = np.zeros((num_points, 3), dtype=np.float32)
        else:
            pts = pts[:num_points]

    x1 = pts
    x0 = project_points_to_sphere(x1, radius=radius)
    return bi, x0, x1


import csv
from pathlib import Path
from typing import List

def find_ply_files_from_all_csv(
    mesh_root: Path,
    split: str = "train",
    csv_name: str = "all.csv",
) -> List[Path]:
    """
    Read all.csv under mesh_root, filter modelIds by split,
    then collect only the .ply paths in models/ under the subdirectories of those modelIds.
    """
    csv_path = mesh_root / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Parse all.csv: modelId -> split
    model_split = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("modelId")
            sp = row.get("split")
            if mid is None or sp is None:
                continue
            model_split[mid.strip()] = sp.strip().lower()

    target_split = split.strip().lower()
    ply_files: List[Path] = []

    # Iterate over the immediate subdirectories under mesh_root (their names are the modelIds)
    for subdir in mesh_root.iterdir():
        if not subdir.is_dir():
            continue

        model_id = subdir.name
        sp = model_split.get(model_id)
        # Skip if not found in all.csv
        if sp is None:
            continue
        # Keep only the specified split
        if sp != target_split:
            continue

        models_dir = subdir / "models"
        if not models_dir.is_dir():
            print(f"[WARN] models/ folder not found for {model_id} under {subdir}, skip.")
            continue

        cur_plys = sorted(models_dir.glob("*.ply"))
        if not cur_plys:
            print(f"[WARN] No .ply found in {models_dir}, skip.")
            continue

        ply_files.extend(cur_plys)

    ply_files = sorted(ply_files)
    return ply_files

def poisson_hilbert_one_mesh_noclean(
    batch_idx: int,
    V: np.ndarray,
    F: np.ndarray,
    num_points: int,
    hilbert_p: int,
    init_factor: float = 4.0,
    use_normal: bool = False,
    use_normal_in_sort: bool = False,
) -> Tuple[int, np.ndarray]:
    """
    On an "already preprocessed" mesh, run Poisson disk sampling of num_points points and apply a 3D Hilbert sort.

    Note: here V, F are assumed to have already been processed upstream:
      - remove_* cleaning
      - orient_triangles to unify triangle orientation
      - normalize to [-1,1]^3
    """
    # Construct the Open3D mesh (lightweight operation)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(V.astype(np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))

    # If normals are needed, use triangle normals to assign normals to the Poisson-sampled points
    if use_normal:
        # mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

    # Poisson disk sampling
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=int(num_points),
        init_factor=float(init_factor),
        use_triangle_normal=False,
    )

    pts = np.asarray(pcd.points, dtype=np.float32)  # (num_points, 3)
    if pts.shape[0] != num_points:
        raise RuntimeError(
            f"Poisson sampling returned {pts.shape[0]} points, expected {num_points}"
        )

    if not use_normal:
        # No normals needed, same as before
        pts_sorted = hilbert_sort_xyz_numba(pts, p_bits=hilbert_p)
        return batch_idx, pts_sorted

    # ----- Branch with normals -----
    if len(pcd.normals) == 0:
        raise RuntimeError("Normal is not initialized (expected vertex normals)!")

    normals = -np.asarray(pcd.normals, dtype=np.float32)  # (num_points, 3)
    if normals.shape[0] != num_points:
        raise RuntimeError(
            f"pcd.normals has shape {normals.shape}, expected ({num_points}, 3)"
        )

    if not use_normal_in_sort:
        pts_sorted, order = hilbert_sort_xyz_numba(pts, p_bits=hilbert_p, return_order=True)
        normals_sorted = normals[order]

        pts_with_normals = np.concatenate([pts_sorted, normals_sorted], axis=-1)  # (N,6)
        return batch_idx, pts_with_normals
    else:
        pts_with_normals_unsorted = np.concatenate([pts, normals], axis=-1)
        pts_with_normals_sorted = hilbert_sort_nd(pts_with_normals_unsorted, p_bits=hilbert_p)
        return batch_idx, pts_with_normals_sorted


def random_hilbert_one_mesh_noclean(
    batch_idx: int,
    V: np.ndarray,
    F: np.ndarray,
    num_points: int,
    hilbert_p: int,
    init_factor: float = 4.0,
    use_normal: bool = False,
    use_normal_in_sort: bool = False,
) -> Tuple[int, np.ndarray]:
    """
    On an "already preprocessed" mesh, run Poisson disk sampling of num_points points and apply a 3D Hilbert sort.

    Note: here V, F are assumed to have already been processed upstream:
      - remove_* cleaning
      - orient_triangles to unify triangle orientation
      - normalize to [-1,1]^3
    """
    # Construct the Open3D mesh (lightweight operation)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(V.astype(np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))

    # If normals are needed, use triangle normals to assign normals to the sampled points
    if use_normal:
        # mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

    # Poisson disk sampling
    pcd = mesh.sample_points_uniformly(
        number_of_points=int(num_points),
        use_triangle_normal=False,
    )

    pts = np.asarray(pcd.points, dtype=np.float32)  # (num_points, 3)
    if pts.shape[0] != num_points:
        raise RuntimeError(
            f"Random sampling returned {pts.shape[0]} points, expected {num_points}"
        )

    if not use_normal:
        # No normals needed, same as before
        pts_sorted = hilbert_sort_xyz_numba(pts, p_bits=hilbert_p)
        return batch_idx, pts_sorted

    # ----- Branch with normals -----
    if len(pcd.normals) == 0:
        raise RuntimeError("Normal is not initialized (expected vertex normals)!")

    normals = -np.asarray(pcd.normals, dtype=np.float32)  # (num_points, 3)
    if normals.shape[0] != num_points:
        raise RuntimeError(
            f"pcd.normals has shape {normals.shape}, expected ({num_points}, 3)"
        )

    if not use_normal_in_sort:
        pts_sorted, order = hilbert_sort_xyz_numba(pts, p_bits=hilbert_p, return_order=True)
        normals_sorted = normals[order]

        pts_with_normals = np.concatenate([pts_sorted, normals_sorted], axis=-1)  # (N,6)
        return batch_idx, pts_with_normals
    else:
        pts_with_normals_unsorted = np.concatenate([pts, normals], axis=-1)
        pts_with_normals_sorted = hilbert_sort_nd(pts_with_normals_unsorted, p_bits=hilbert_p)
        return batch_idx, pts_with_normals_sorted


def poisson_nosort_one_mesh_noclean(
    batch_idx: int,
    V: np.ndarray,
    F: np.ndarray,
    num_points: int,
    init_factor: float = 4.0,
    use_normal: bool = False,
) -> Tuple[int, np.ndarray]:
    """
    On an "already preprocessed" mesh, run Poisson disk sampling of num_points points, without a Hilbert sort.

    Note: here V, F are assumed to have already been processed upstream:
      - remove_* cleaning
      - orient_triangles to unify triangle orientation
      - normalize to [-1,1]^3
    """
    # Construct the Open3D mesh (lightweight operation)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(V.astype(np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))

    # If normals are needed, use triangle normals to assign normals to the Poisson-sampled points
    if use_normal:
        mesh.compute_vertex_normals()

    # Poisson disk sampling
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=int(num_points),
        init_factor=float(init_factor),
        use_triangle_normal=False,
    )

    pts = np.asarray(pcd.points, dtype=np.float32)  # (num_points, 3)
    if pts.shape[0] != num_points:
        raise RuntimeError(
            f"Poisson sampling returned {pts.shape[0]} points, expected {num_points}"
        )

    if not use_normal:
        # No normals needed, return the sampled points directly (no sorting)
        return batch_idx, pts

    # ----- Branch with normals -----
    if len(pcd.normals) == 0:
        raise RuntimeError("Normal is not initialized (expected vertex normals)!")

    normals = -np.asarray(pcd.normals, dtype=np.float32)  # (num_points, 3)
    if normals.shape[0] != num_points:
        raise RuntimeError(
            f"pcd.normals has shape {normals.shape}, expected ({num_points}, 3)"
        )

    pts_with_normals = np.concatenate([pts, normals], axis=-1)  # (N,6)
    return batch_idx, pts_with_normals


def random_nosort_one_mesh_noclean(
    batch_idx: int,
    V: np.ndarray,
    F: np.ndarray,
    num_points: int,
    use_normal: bool = False,
) -> Tuple[int, np.ndarray]:
    """
    On an "already preprocessed" mesh, run uniform random sampling of num_points points, without a Hilbert sort.

    Note: here V, F are assumed to have already been processed upstream:
      - remove_* cleaning
      - orient_triangles to unify triangle orientation
      - normalize to [-1,1]^3
    """
    # Construct the Open3D mesh (lightweight operation)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(V.astype(np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))

    # If normals are needed, use triangle normals to assign normals to the sampled points
    if use_normal:
        mesh.compute_vertex_normals()

    # Uniform random sampling
    pcd = mesh.sample_points_uniformly(
        number_of_points=int(num_points),
        use_triangle_normal=False,
    )

    pts = np.asarray(pcd.points, dtype=np.float32)  # (num_points, 3)
    if pts.shape[0] != num_points:
        raise RuntimeError(
            f"Random sampling returned {pts.shape[0]} points, expected {num_points}"
        )

    if not use_normal:
        # No normals needed, return the sampled points directly (no sorting)
        return batch_idx, pts

    # ----- Branch with normals -----
    if len(pcd.normals) == 0:
        raise RuntimeError("Normal is not initialized (expected vertex normals)!")

    normals = -np.asarray(pcd.normals, dtype=np.float32)  # (num_points, 3)
    if normals.shape[0] != num_points:
        raise RuntimeError(
            f"pcd.normals has shape {normals.shape}, expected ({num_points}, 3)"
        )

    pts_with_normals = np.concatenate([pts, normals], axis=-1)  # (N,6)
    return batch_idx, pts_with_normals


def random_hilbert_from_pointcloud(
    batch_idx: int,
    points: np.ndarray,      # (M, 3) pre-sampled points
    normals: np.ndarray,     # (M, 3) pre-sampled normals, may be None
    num_points: int,
    hilbert_p: int,
    use_normal: bool = False,
    use_normal_in_sort: bool = False,
) -> Tuple[int, np.ndarray]:
    """
    Randomly select num_points points from a pre-sampled point cloud and apply a Hilbert sort.
    No need to sample from a mesh; select directly from the existing point cloud.

    Args:
        batch_idx: the index of this result within the batch (for convenient write-back)
        points: (M, 3) pre-sampled point coordinates
        normals: (M, 3) pre-sampled normals; if use_normal=False, None can be passed
        num_points: number of points to select
        hilbert_p: number of bits for the Hilbert curve
        use_normal: whether to use normals
        use_normal_in_sort: whether to use normals during sorting (6D Hilbert sort)

    Returns:
        (batch_idx, pts_sorted) where pts_sorted shape = (num_points, 3) or (num_points, 6)
    """
    M = points.shape[0]

    # Randomly select num_points indices
    indices = np.random.choice(M, size=num_points, replace=False)
    pts = points[indices].astype(np.float32)

    if not use_normal:
        # No normals needed, only do a 3D Hilbert sort
        pts_sorted = hilbert_sort_xyz_numba(pts, p_bits=hilbert_p)
        return batch_idx, pts_sorted

    # ----- Branch with normals -----
    norms = normals[indices].astype(np.float32)

    if not use_normal_in_sort:
        # Sort by coordinates only, then reorder the normals according to the sort order
        pts_sorted, order = hilbert_sort_xyz_numba(pts, p_bits=hilbert_p, return_order=True)
        normals_sorted = norms[order]
        pts_with_normals = np.concatenate([pts_sorted, normals_sorted], axis=-1)  # (N,6)
        return batch_idx, pts_with_normals
    else:
        # Use a 6D Hilbert sort (coordinates + normals participate in sorting together)
        pts_with_normals_unsorted = np.concatenate([pts, norms], axis=-1)
        pts_with_normals_sorted = hilbert_sort_nd(pts_with_normals_unsorted, p_bits=hilbert_p)
        return batch_idx, pts_with_normals_sorted


def random_nosort_from_pointcloud(
    batch_idx: int,
    points: np.ndarray,      # (M, 3) pre-sampled points
    normals: np.ndarray,     # (M, 3) pre-sampled normals, may be None
    num_points: int,
    use_normal: bool = False,
) -> Tuple[int, np.ndarray]:
    """
    Randomly select num_points points from a pre-sampled point cloud, without a Hilbert sort.

    Args:
        batch_idx: the index of this result within the batch (for convenient write-back)
        points: (M, 3) pre-sampled point coordinates
        normals: (M, 3) pre-sampled normals; if use_normal=False, None can be passed
        num_points: number of points to select
        use_normal: whether to use normals

    Returns:
        (batch_idx, pts) where pts shape = (num_points, 3) or (num_points, 6)
    """
    M = points.shape[0]

    # Randomly select num_points indices
    indices = np.random.choice(M, size=num_points, replace=False)
    pts = points[indices].astype(np.float32)

    if not use_normal:
        return batch_idx, pts

    # ----- Branch with normals -----
    norms = normals[indices].astype(np.float32)
    pts_with_normals = np.concatenate([pts, norms], axis=-1)  # (N,6)
    return batch_idx, pts_with_normals


def poisson_hilbert_one_mesh(
    batch_idx: int,
    V: np.ndarray,
    F: np.ndarray,
    num_points: int,
    hilbert_p: int,
    init_factor: float = 4.0,
    use_normal: bool = False,
    use_normal_in_sort: bool = False,
) -> Tuple[int, np.ndarray]:
    """
    On a single mesh, run Poisson disk sampling of num_points points and apply a 3D Hilbert sort.

    Args:
        batch_idx: the index of this result within the batch (for convenient write-back)
        V: (Nv, 3) vertices, already normalized to [-1,1]^3
        F: (Nf, 3) triangle face indices
        num_points: target number of sampled points
        hilbert_p: number of bits for the Hilbert curve
        init_factor: the init_factor parameter for Open3D Poisson disk sampling

    Returns:
        (batch_idx, pts_sorted) where pts_sorted shape = (num_points, 3)
    """
    # Construct the Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))

    # (Optional) do a bit of cleaning to guard against bad meshes
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()

    if use_normal:
        mesh.compute_vertex_normals()

    # Poisson disk sampling
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=int(num_points),
        init_factor=float(init_factor),
    )
    pts = np.asarray(pcd.points, dtype=np.float32)  # (num_points, 3)
    if pts.shape[0] != num_points:
        # In theory open3d returns number_of_points points; add a sanity check here
        raise RuntimeError(
            f"Poisson sampling returned {pts.shape[0]} points, expected {num_points}"
        )

    if use_normal:
        if len(pcd.normals) == 0:
            raise RuntimeError(
                f"Normal is not initialized!"
            )
        normals = -np.asarray(pcd.normals, dtype=np.float32)  # (num_points, 3)
        if normals.shape[0] != num_points:
            raise RuntimeError(
                f"pcd.normals has shape {normals.shape}, expected ({num_points}, 3)"
            )

        if not use_normal_in_sort:
            pts_sorted, order = hilbert_sort_xyz_numba(pts, p_bits=hilbert_p, return_order=True)
            normals_sorted = normals[order]

            # Assemble into (N, 6)
            pts_with_normals = np.concatenate([pts_sorted, normals_sorted], axis=-1)
            return batch_idx, pts_with_normals
        else:
            pts_with_normals_unsorted = np.concatenate([pts, normals], axis=-1)
            pts_with_normals_sorted = hilbert_sort_nd(pts_with_normals_unsorted, p_bits=10)
            return batch_idx, pts_with_normals_sorted

    else:
        # The no-normals case is the same as before
        pts_sorted = hilbert_sort_xyz_numba(pts, p_bits=hilbert_p)
        return batch_idx, pts_sorted

def get_or_build_raycast_scene(
    mesh_idx: int,
    V: np.ndarray,
    F: np.ndarray,
) -> o3d.t.geometry.RaycastingScene:
    """
    Return the RaycastingScene (BVH) corresponding to mesh_idx within the current process.

    - If it already exists in the cache, reuse it directly;
    - If not, build one from the given V, F and cache it.
    """
    scene = _BVH_CACHE.get(mesh_idx, None)
    if scene is not None:
        return scene
    assert V.ndim == 2 and V.shape[1] == 3, f"V shape invalid: {V.shape}, idx={mesh_idx}"
    assert F.ndim == 2 and F.shape[1] == 3, f"F shape invalid: {F.shape}, idx={mesh_idx}"
    assert F.min() >= 0, f"F has negative index, idx={mesh_idx}"
    vmax = V.shape[0]
    assert F.max() < vmax, f"F has index >= num_verts ({vmax}), idx={mesh_idx}"

    verts = o3c.Tensor(V.astype(np.float32))   # (Nv,3)
    tris  = o3c.Tensor(F.astype(np.int32))     # (Nt,3)

    tmesh = o3d.t.geometry.TriangleMesh(verts, tris)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    _BVH_CACHE[mesh_idx] = scene
    return scene





def hermite_ot_one_batch(
    bi: int,
    x0: np.ndarray,   # (N,3)
    x1: np.ndarray,   # (N,3)
    n1: Optional[np.ndarray],   # (N,3)
    lambda_orient: float,
    ot_solver: str,
    hermite_degree: str,
    mesh_idx: int,
    V: np.ndarray,
    F: np.ndarray,
    quadratic_use_analytic: bool = False,   # new flag, defaults to the analytic quadratic
    zero_t0: bool = False,
)  -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Hermite-style cost + OT / greedy matching for a single batch index = bi.

    - If hermite_degree == "cubic":
        use BVH + approximate cubic Hermite cost (original logic)
    - If hermite_degree == "quadratic":
        no raycast; options are:
          * quadratic_use_analytic=True: analytic quadratic arc length
          * quadratic_use_analytic=False: a cubic-like approximate cost (much faster)
    """
    # tree = get_or_build_igl_aabb(mesh_idx, V, F)
    hermite_degree = (hermite_degree or "cubic").lower()

    if hermite_degree == "linear":
        # === linear mode: pure straight-line distance, no normals needed ===
        C = build_linear_cost_matrix(x0=x0, x1=x1)
        n0 = None  # do not return normals
        n1_for_match = None

    elif hermite_degree == "quadratic":
        if quadratic_use_analytic:
            # Exact quadratic Hermite arc length (analytic)
            C = build_quadratic_hermite_cost_matrix(
                x0=x0,
                x1=x1,
                n1=n1,
                tangent_scale=1.0,   # adjust this coefficient to tune the amount of curvature
            )
        else:
            # Approximate version: no raycast, just use x0,x1,n1 to build a
            # D * (1 + λ (1 - S)) style cost
            C = build_quadratic_approx_cost_matrix_opt(
                x0=x0,
                x1=x1,
                n1=n1,
                lambda_orient=lambda_orient,
            )

        # In quadratic mode n0 is currently unused, so fill it with 0 for now (keeps the interface compatible)
        n0 = np.zeros_like(x0, dtype=np.float32)
        n1_for_match = n1

    else:
        # cubic mode: keep the original logic, using BVH + approximate Hermite cost
        # scene = get_or_build_raycast_scene(mesh_idx, V, F)
        return_t0 = True if not zero_t0 else False

        ret =  build_hermite_cost_matrix_opt(
            x0=x0,
            x1=x1,
            n1=n1,
            lambda_orient=lambda_orient,
            # mesh_scene=scene,
            # tree = tree,
            V=V,
            F=F,
            return_t0=return_t0,
            zero_t0=zero_t0
        )
        C, n0 = ret if return_t0 else (ret, None)
        n1_for_match = n1

    # Regardless of quadratic / cubic, perform a single 1-to-1 OT assignment afterward
    perm = solve_ot_assignment(C, mode=ot_solver)

    x1_matched = x1[perm]   # (N,3)
    if n1_for_match is None:
        n1_matched = None
    else:
        n1_matched = n1_for_match[perm]
    return bi, x1_matched, n0, n1_matched


def build_linear_cost_matrix(
    x0: np.ndarray,  # (N,3)
    x1: np.ndarray,  # (N,3)
) -> np.ndarray:
    """
    Linear (straight-line) cost: C_ij = ||x0_i - x1_j||_2

    Use the (a+b-2ab) form to avoid creating the large (N,N,3) intermediate tensor.
    Returns float32 (N,N).
    """
    assert x0.ndim == 2 and x1.ndim == 2 and x0.shape[1] == 3 and x1.shape[1] == 3
    x0f = x0.astype(np.float32, copy=False)
    x1f = x1.astype(np.float32, copy=False)

    a = (x0f * x0f).sum(axis=1, keepdims=True)      # (N,1)
    b = (x1f * x1f).sum(axis=1, keepdims=True).T    # (1,N)
    # D^2 = ||x0||^2 + ||x1||^2 - 2 x0·x1
    d2 = a + b - 2.0 * (x0f @ x1f.T)                # (N,N)
    np.maximum(d2, 0.0, out=d2)                     # guard against numerical error
    C = np.sqrt(d2, dtype=np.float32)               # (N,N)
    return C



def build_quadratic_hermite_cost_matrix(
    x0: np.ndarray,  # (N,3)
    x1: np.ndarray,  # (N,3)
    n1: np.ndarray,  # (N,3)
    tangent_scale: float = 1.0,
) -> np.ndarray:
    """
    For each pair (x0_i, x1_j, n1_j), construct a quadratic Hermite curve:
        x(t) = x0_i + α(t) * d_ij + β(t) * v1_j
      where:
        d_ij = x1_j - x0_i
        v1_j ~ n1_j
        α(t) = 2t - t^2
        β(t) = t^2 - t

    Returns the cost matrix C (N,N), where C[i,j] = the arc length of this curve over t∈[0,1] (computed analytically).
    """
    N = x0.shape[0]
    assert x1.shape == (N, 3) and n1.shape == (N, 3)

    # 1) Normalize n1 -> v1
    v1 = n1.astype(np.float32).copy()
    n1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
    n1_norm[n1_norm < 1e-12] = 1.0
    v1 = tangent_scale * (v1 / n1_norm)   # (N,3)

    # 2) pairwise d_ij = x1_j - x0_i
    x0_i = x0[:, None, :]      # (N,1,3)
    x1_j = x1[None, :, :]      # (1,N,3)
    v1_j = v1[None, :, :]      # (1,N,3)

    d = x1_j - x0_i            # (N,N,3)

    # A_ij, B_ij
    A = 2.0 * d - v1_j         # (N,N,3)
    B = 2.0 * (v1_j - d)       # (N,N,3)

    # a_ij, b_ij, c_ij
    a = np.sum(B * B, axis=-1)         # (N,N)
    c = np.sum(A * A, axis=-1)         # (N,N)
    b = 2.0 * np.sum(A * B, axis=-1)   # (N,N)

    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    c = c.astype(np.float32, copy=False)

    # 3) Discriminant 4ac - b^2 > 0, always positive under the current geometry; apply a numerical lower bound
    disc = 4.0 * a * c - b * b
    disc = np.maximum(disc, 1e-24)

    a_eps = 1e-12
    mask = a > a_eps

    L = np.zeros_like(a)

    # 4) When a ≈ 0, the speed is approximately constant and degenerates to a straight line: length ~ ||A|| = sqrt(c)
    idx_lin = ~mask
    if np.any(idx_lin):
        c_lin = np.maximum(c[idx_lin], 0.0)
        L[idx_lin] = np.sqrt(c_lin)

    # 5) When a > 0, use the analytic formula:
    #    ∫ sqrt(a t^2 + b t + c) dt from 0 to 1
    #    = F(1) - F(0)
    #    F(t) = (2 a t + b) sqrt(...) / (4 a)
    #           + (4ac - b^2) / (8 a^(3/2)) * asinh((2 a t + b) / sqrt(4ac - b^2))
    if np.any(mask):
        am = a[mask]
        bm = b[mask]
        cm = c[mask]
        dm = disc[mask]

        sqrt_dm = np.sqrt(dm)
        a_sqrt = np.sqrt(am)
        denom2 = 8.0 * am * a_sqrt      # 8 * a^(3/2)

        # t = 1
        q1 = am + bm + cm
        q1 = np.maximum(q1, 0.0)
        sqrt_q1 = np.sqrt(q1)
        term1_1 = (2.0 * am + bm) * sqrt_q1 / (4.0 * am)
        u1 = (2.0 * am + bm) / sqrt_dm
        term2_1 = dm * np.arcsinh(u1) / denom2
        F1 = term1_1 + term2_1

        # t = 0
        q0 = cm
        q0 = np.maximum(q0, 0.0)
        sqrt_q0 = np.sqrt(q0)
        term1_0 = bm * sqrt_q0 / (4.0 * am)
        u0 = bm / sqrt_dm
        term2_0 = dm * np.arcsinh(u0) / denom2
        F0 = term1_0 + term2_0

        L[mask] = F1 - F0

    return L

def build_quadratic_approx_cost_matrix(
    x0: np.ndarray,  # (N,3)
    x1: np.ndarray,  # (N,3)
    n1: np.ndarray,  # (N,3)
    lambda_orient: float = 0.2,
) -> np.ndarray:
    """
    Approximate cost in quadratic mode:
      - no raycast
      - use the chord direction d_ij = x1_j - x0_i and the alignment S_ij with the endpoint normal
      - the form is similar to the cubic approximation:
            C_ij = D_ij * (1 + lambda_orient * (1 - S_ij))
    """
    N = x0.shape[0]
    assert x1.shape == (N, 3) and n1.shape == (N, 3)

    # pairwise distance
    diff = x0[:, None, :] - x1[None, :, :]   # (N,N,3)
    D2 = np.sum(diff ** 2, axis=-1)          # (N,N)
    D = np.sqrt(D2 + 1e-12)

    if lambda_orient <= 0:
        return D

    # chord direction d_ij = x1_j - x0_i
    d = -diff                                # (N,N,3) = x1_j - x0_i
    d_norm = np.linalg.norm(d, axis=-1, keepdims=True)
    d_norm[d_norm < 1e-12] = 1.0
    d_hat = d / d_norm                       # (N,N,3)

    # endpoint normal direction t1_j
    t1 = n1.astype(np.float32).copy()        # (N,3)
    t1_norm = np.linalg.norm(t1, axis=-1, keepdims=True)
    t1_norm[t1_norm < 1e-12] = 1.0
    t1_hat = t1 / t1_norm                    # (N,3)

    # broadcast to (N,N,3)
    t1_pair = t1_hat[None, :, :]             # (1,N,3) broadcast to (N,N,3)

    # S_ij = <d_hat_ij, t1_j>
    S = np.sum(d_hat * t1_pair, axis=-1)
    S = np.clip(S, -1.0, 1.0)

    C = D * (1.0 + lambda_orient * (1.0 - S))
    return C

def build_quadratic_approx_cost_matrix_opt(
    x0: np.ndarray,  # (N,3)
    x1: np.ndarray,  # (N,3)
    n1: np.ndarray,  # (N,3)
    lambda_orient: float = 0.2,
) -> np.ndarray:
    """
    Approximate cost in quadratic mode:
      C_ij = D_ij * (1 + λ (1 - S_ij)),
    where S_ij = <d_hat_ij, t1_j>,
    d_hat_ij = (x1_j - x0_i) / ||x1_j - x0_i||,
    and t1_j is the normalized normal.

    This implementation reuses intermediate quantities as much as possible to avoid an extra large (N,N,3) matrix.
    """
    N = x0.shape[0]
    assert x1.shape == (N, 3) and n1.shape == (N, 3)

    # Use float32 throughout
    x0 = x0.astype(np.float32, copy=False)
    x1 = x1.astype(np.float32, copy=False)
    n1 = n1.astype(np.float32, copy=False)

    # diff = x0_i - x1_j
    diff = x0[:, None, :] - x1[None, :, :]   # (N,N,3), float32

    # D_ij = ||x0_i - x1_j||
    D2 = np.sum(diff ** 2, axis=-1)          # (N,N)
    D = np.sqrt(D2 + 1e-12)                  # (N,N)

    if lambda_orient <= 0:
        # Pure Euclidean distance
        return D.astype(np.float32, copy=False)

    # Clamp slightly to avoid division by 0
    D[D < 1e-6] = 1e-6

    # Turn diff into d_hat in place:
    # originally diff = x0 - x1, but we want d_hat = (x1 - x0) / ||x1 - x0||
    # so diff *= -1 / ||.||
    diff *= (-1.0 / D[..., None]).astype(np.float32, copy=False)  # (N,N,3) is now d_hat

    # Normalize the normal to get t1_hat
    t1 = n1  # view
    t1_norm = np.linalg.norm(t1, axis=-1, keepdims=True)
    t1_norm[t1_norm < 1e-12] = 1.0
    t1_hat = t1 / t1_norm                   # (N,3)

    # S_ij = <d_hat_ij, t1_j>, using broadcasting (1,N,3) -> (N,N,3) without building an extra t1_pair
    S = np.sum(diff * t1_hat[None, :, :], axis=-1)  # (N,N)
    S = np.clip(S, -1.0, 1.0)

    C = D * (1.0 + lambda_orient * (1.0 - S))      # (N,N)
    return C.astype(np.float32, copy=False)

def build_hermite_cost_matrix(
    x0: np.ndarray,          # (N,3)
    x1: np.ndarray,          # (N,3)
    n1: np.ndarray,          # (N,3) normals
    lambda_orient: float = 0.2,
    # mesh_scene: o3d.t.geometry.RaycastingScene | None = None,
    # tree: igl.AABB | None = None,
    V: np.ndarray | None = None,
    F: np.ndarray | None = None,
    return_t0: bool = False,
) -> np.ndarray:
    """
    Construct the cost matrix C (N,N) approximating Hermite curve lengths:

      - D_ij = ||x0_i - x1_j||_2
      - for each x0_i:
          * if mesh_scene (BVH) is provided:
                find the nearest point p_i^* on the mesh surface (closest triangle point)
            otherwise (fallback):
                find the nearest Poisson point x1_{j*}, p_i^* = x1_{j*}
          * t0_i = (p_i^* - x0_i) / ||p_i^* - x0_i||
      - for each x1_j:
          * t1_j = normalized(n1_j)
      - S_ij = <t0_i, t1_j>
      - C_ij = D_ij * (1 + lambda_orient * (1 - S_ij))

    If lambda_orient <= 0, this degenerates to Euclidean distance.
    """
    N = x0.shape[0]
    assert x1.shape[0] == N and n1.shape[0] == N

    # pairwise Euclidean distance matrix D
    diff = x0[:, None, :] - x1[None, :, :]   # (N,N,3)
    D2 = np.sum(diff ** 2, axis=2)           # (N,N)
    D = np.sqrt(D2 + 1e-12)                  # prevent 0

    if lambda_orient <= 0:
        return (D, np.zeros_like(x0)) if return_t0 else D

    # Normalize n1 -> t1
    t1 = n1.copy()
    t1_norm = np.linalg.norm(t1, axis=1, keepdims=True)
    t1_norm[t1_norm < 1e-12] = 1.0
    t1 = t1 / t1_norm                         # (N,3)

    # ----- t0_i: points to the nearest point on the mesh (prefer BVH) -----
    # if mesh_scene is not None:
    #     # Use BVH to compute the nearest surface point directly
    #     queries = o3c.Tensor(x0.astype(np.float32))  # (N,3)
    #     ans = mesh_scene.compute_closest_points(queries)
    #     nearest_pts = ans["points"].numpy()          # (N,3)
    # else:
    #     # fallback: use the nearest Poisson point (guards against some odd cases)
    #     nn_indices = np.argmin(D2, axis=1)           # (N,)
    #     nearest_pts = x1[nn_indices]                 # (N,3)

    if V is not None and F is not None:
        # Use the libigl.AABB nearest point (consistent with nearest_one_job_proc above)
        tree = igl.AABB()
        tree.init(V, F)
        # P = x0
        _, _, nearest_pts = tree.squared_distance(
            V.astype(np.float64, copy=False),
            F.astype(np.int32, copy=False),
            x0.astype(np.float64, copy=False),
        )                                      # (N,3)
    else:
        raise RuntimeError("NNP failed")
        # fallback: use the nearest Poisson point (guards against some odd cases)
        # nn_indices = np.argmin(D2, axis=1)     # (N,)
        # nearest_pts = x1[nn_indices]           # (N,3)


    t0 = nearest_pts - x0                            # (N,3)
    t0_norm = np.linalg.norm(t0, axis=1, keepdims=True)
    t0_norm[t0_norm < 1e-12] = 1.0
    t0 = t0 / t0_norm                                # (N,3)

    # S_ij = <t0_i, t1_j> = (N,3) @ (3,N)
    S = t0 @ t1.T
    S = np.clip(S, -1.0, 1.0)

    C = D * (1.0 + lambda_orient * (1.0 - S))
    if return_t0:
        return C, t0
    else:
        return C
def pairwise_dist(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """
    Returns D (N,N), where D[i,j] = ||x0_i - x1_j||_2, mathematically
    fully equivalent to
    diff = x0[:,None,:] - x1[None,:,:]; np.sum(diff**2, axis=2).
    """
    # Ensure contiguous arrays
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)

    # (N,1), (1,N)
    x0_sq = np.sum(x0 * x0, axis=1, keepdims=True)
    x1_sq = np.sum(x1 * x1, axis=1, keepdims=True).T

    # D2 = ||x0||^2 + ||x1||^2 - 2 x0·x1
    D2 = -2.0 * (x0 @ x1.T)      # (N,N)
    D2 += x0_sq                  # broadcast-add columns
    D2 += x1_sq                  # broadcast-add rows

    # In theory D2 >= 0; as before, add 1e-12 inside the sqrt
    D = np.sqrt(D2 + 1e-12)
    return D

def build_hermite_cost_matrix_opt(
    x0: np.ndarray,          # (N,3)
    x1: np.ndarray,          # (N,3)
    n1: np.ndarray,          # (N,3) normals
    lambda_orient: float = 0.2,
    # mesh_scene: o3d.t.geometry.RaycastingScene | None = None,
    # tree: igl.AABB | None = None,
    V: np.ndarray | None = None,
    F: np.ndarray | None = None,
    return_t0: bool = False,
    zero_t0: bool = False,   # * new: if True, set the start-point tangent ("normal") to 0
) -> np.ndarray:
    """
    Construct the cost matrix C (N,N) approximating Hermite curve lengths:

      - D_ij = ||x0_i - x1_j||_2
      - for each x0_i:
          * if mesh_scene (BVH) is provided:
                find the nearest point p_i^* on the mesh surface (closest triangle point)
            otherwise (fallback):
                find the nearest Poisson point x1_{j*}, p_i^* = x1_{j*}
          * t0_i = (p_i^* - x0_i) / ||p_i^* - x0_i||
      - for each x1_j:
          * t1_j = normalized(n1_j)
      - S_ij = <t0_i, t1_j>
      - C_ij = D_ij * (1 + lambda_orient * (1 - S_ij))

    * new zero_t0=True:
      - when the start-point tangent is 0, t0 no longer provides a directional constraint.
      - a more suitable approximation is to use the alignment of the chord direction d_hat_ij with the endpoint normal t1_j:
            d_hat_ij = (x1_j - x0_i) / ||x1_j - x0_i||
            S_ij = <d_hat_ij, t1_j>
        then still apply the same penalty form:
            C_ij = D_ij * (1 + lambda_orient * (1 - S_ij))

    If lambda_orient <= 0, this degenerates to Euclidean distance.
    """
    N = x0.shape[0]
    assert x1.shape[0] == N and n1.shape[0] == N

    # pairwise Euclidean distance matrix D
    # diff = x0[:, None, :] - x1[None, :, :]   # (N,N,3)
    # D2 = np.sum(diff ** 2, axis=2)           # (N,N)
    # D = np.sqrt(D2 + 1e-12)                  # prevent 0
    D = pairwise_dist(x0, x1)
    if lambda_orient <= 0:
        # * when zero_t0=True, t0 should be 0; otherwise the original code returned np.zeros_like(x0)
        return (D, np.zeros_like(x0)) if return_t0 else D

    # Normalize n1 -> t1
    t1 = n1.astype(n1.dtype, copy=True)
    t1_norm = np.linalg.norm(t1, axis=1, keepdims=True)
    t1_norm[t1_norm < 1e-12] = 1.0
    t1 /= t1_norm                     # in-place, one fewer new array

    # ====== new: zero_t0 branch (no BVH / nearest_pts needed) ======
    if zero_t0:
        # Approximate S_ij with the dot product of the chord direction d_hat_ij and t1_j
        # S_ij = < (x1_j - x0_i)/D_ij , t1_j >
        #      = (x1_j·t1_j - x0_i·t1_j) / D_ij
        # To avoid constructing the (N,N,3) diff, expand using matrix multiplication:
        #   X0T1[i,j] = x0_i · t1_j
        #   X1T1[j]   = x1_j · t1_j
        X0T1 = x0 @ t1.T                           # (N,N)
        X1T1 = np.sum(x1 * t1, axis=1)             # (N,)
        denom = np.maximum(D, 1e-12)               # prevent division by 0
        S = (X1T1[None, :] - X0T1) / denom         # (N,N)
        S = np.clip(S, -1.0, 1.0)

        C = D * (1.0 + lambda_orient * (1.0 - S))
        return C

    # ----- t0_i: points to the nearest point on the mesh (prefer BVH) -----
    # if mesh_scene is not None:
    #     # Use BVH to compute the nearest surface point directly
    #     queries = o3c.Tensor(x0.astype(np.float32))  # (N,3)
    #     ans = mesh_scene.compute_closest_points(queries)
    #     nearest_pts = ans["points"].numpy()          # (N,3)
    # else:
    #     # fallback: use the nearest Poisson point (guards against some odd cases)
    #     nn_indices = np.argmin(D2, axis=1)           # (N,)
    #     nearest_pts = x1[nn_indices]                 # (N,3)

    if V is not None and F is not None:
        # Use the libigl.AABB nearest point (consistent with nearest_one_job_proc above)
        tree = igl.AABB()
        tree.init(V, F)
        # P = x0
        _, _, nearest_pts = tree.squared_distance(
            V.astype(np.float64, copy=False),
            F.astype(np.int32, copy=False),
            x0.astype(np.float64, copy=False),
        )                                      # (N,3)
    # if tree is not None:
    #     _, _, nearest_pts = tree.squared_distance(
    #         V.astype(np.float64, copy=False),
    #         F.astype(np.int32,  copy=False),
    #         x0.astype(np.float64, copy=False),
    #     )
    # elif V is not None and F is not None:
    #     tree = igl.AABB()
    #     tree.init(
    #         V.astype(np.float64, copy=False),
    #         F.astype(np.int32,  copy=False),
    #     )
    #     _, _, nearest_pts = tree.squared_distance(
    #         V.astype(np.float64, copy=False),
    #         F.astype(np.int32,  copy=False),
    #         x0.astype(np.float64, copy=False),
    #     )
    else:
        raise RuntimeError("NNP failed")
        # fallback: use the nearest Poisson point (guards against some odd cases)
        # nn_indices = np.argmin(D2, axis=1)     # (N,)
        # nearest_pts = x1[nn_indices]           # (N,3)

    t0 = nearest_pts - x0
    t0_norm = np.linalg.norm(t0, axis=1, keepdims=True)
    t0_norm[t0_norm < 1e-12] = 1.0
    t0 /= t0_norm

    # S_ij = <t0_i, t1_j> = (N,3) @ (3,N)
    S = t0 @ t1.T
    S = np.clip(S, -1.0, 1.0)

    C = D * (1.0 + lambda_orient * (1.0 - S))
    if return_t0:
        return C, t0
    else:
        return C


@njit
def _greedy_ot_core(C: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated core logic for greedy OT matching.
    """
    N = C.shape[0]

    # Compute the minimum of each row
    row_min = np.empty(N, dtype=np.float64)
    for i in range(N):
        row_min[i] = np.min(C[i])

    # Sort by the per-row minimum
    row_order = np.argsort(row_min)

    perm = np.empty(N, dtype=np.int64)
    unmatched = np.ones(N, dtype=np.bool_)

    # Greedy matching
    for idx in range(N):
        i = row_order[idx]
        best_j = 0
        best_val = np.inf
        for j in range(N):
            if unmatched[j] and C[i, j] < best_val:
                best_val = C[i, j]
                best_j = j
        perm[i] = best_j
        unmatched[best_j] = False

    # 2-swap optimization
    eps = 1e-12
    for idx in range(N - 1):
        i = row_order[idx]
        k = row_order[idx + 1]
        j = perm[i]
        l = perm[k]
        if j != l:
            current_cost = C[i, j] + C[k, l]
            swapped_cost = C[i, l] + C[k, j]
            if swapped_cost + eps < current_cost:
                perm[i] = l
                perm[k] = j

    return perm


def solve_ot_assignment(
    C: np.ndarray,
    mode: str = "greedy",
    knn_k: int = 256,
) -> np.ndarray:
    """
    Given the cost matrix C (N,N), return a perm array p (N,):
        x0_i corresponds to x1_{p[i]}.

    mode:
      - "greedy": O(N^2) approximate KNN matching
      - "hungarian": call SciPy's Hungarian (linear_sum_assignment), O(N^3)
    """
    N = C.shape[0]
    assert C.shape[1] == N

    mode = mode.lower()
    if mode == "hungarian":
        if linear_sum_assignment is None:
            raise ImportError(
                "scipy.optimize.linear_sum_assignment not available. "
                "Install SciPy or use mode='greedy'."
            )
        row_ind, col_ind = linear_sum_assignment(C)
        # row_ind is not necessarily [0..N), so construct perm explicitly
        perm = np.empty(N, dtype=np.int64)
        perm[row_ind] = col_ind
        return perm

    if mode in ("hungarian_knn", "knn_hungarian"):
        # Environment does not support it; fall back to dense Hungarian (acceptable for small N)
        if not _HAS_SPARSE_BIPARTITE or linear_sum_assignment is None:
            row_ind, col_ind = linear_sum_assignment(C)
            perm = np.empty(N, dtype=np.int64)
            perm[row_ind] = col_ind
            return perm

        # Try several times: k, 2k, 4k ...; if no full matching is ever found, fall back to greedy
        base_k = min(knn_k, N)
        max_attempts = 3

        for attempt in range(max_attempts):
            k = min(base_k * (2 ** attempt), N)

            rows = []
            cols = []
            data = []
            seen = set()  # deduplicate to avoid (i,j) repeats doubling the cost

            # --- row-KNN: keep the k smallest per row ---
            for i in range(N):
                row = C[i]
                if k < N:
                    idx = np.argpartition(row, k)[:k]
                else:
                    idx = np.arange(N, dtype=np.int64)
                for j in idx:
                    key = (i, int(j))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(i)
                    cols.append(int(j))
                    data.append(row[j])

            # --- col-KNN: keep the k smallest per column as well ---
            for j in range(N):
                col = C[:, j]
                if k < N:
                    idx = np.argpartition(col, k)[:k]
                else:
                    idx = np.arange(N, dtype=np.int64)
                for i in idx:
                    key = (int(i), j)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(int(i))
                    cols.append(j)
                    data.append(col[i])

            cost_sparse = csr_matrix((data, (rows, cols)), shape=(N, N))

            try:
                row_ind, col_ind = min_weight_full_bipartite_matching(cost_sparse)
                perm = np.empty(N, dtype=np.int64)
                perm[row_ind] = col_ind
                return perm
            except Exception as e:
                msg = str(e)
                print(f"[WARN] sparse Hungarian (k={k}) failed ({msg}).")
                # If it is "no full matching exists", try increasing k; otherwise break directly
                if "no full matching exists" not in msg:
                    break

        # Reaching here means sparse never produced a full matching;
        # for speed, fall back to the improved greedy (rather than dense Hungarian).
        print("[WARN] sparse Hungarian exhausted attempts, fallback to greedy.")

    # Default: greedy, Numba-accelerated
    return _greedy_ot_core(C.astype(np.float64))
