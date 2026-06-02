#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_dla.py

Evaluation script for DLA (Diffusion Limited Aggregation)

Supported modes:
1. 3D mode (default): outputs xyz coordinates directly
2. 2D mode (--mode 2d): outputs xy coordinates

Visualization: first two dims as xy, point index as color
"""

import os
import sys
import argparse
import torch
import matplotlib
matplotlib.use('Agg')  # Force a non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from flow_lab.models import UncondUniGBNTransformer, UncondUniGBNTransformer_PE
from flow_lab.dynamics import EulerSimulator, RK4Simulator, VectorFieldODE
from flow_lab.utils import load_checkpoint, save_pointcloud_ply
from flow_lab.distributions import Uniform


# ----------------- Argparser -----------------

def build_argparser():
    p = argparse.ArgumentParser(description="DLA evaluation")

    # Mode selection
    p.add_argument("--mode", type=str, default="3d",
                   choices=["3d", "2d"],
                   help="Evaluation mode: 3d or 2d")

    # Required parameters
    p.add_argument("--ckpt", type=str, required=True,
                   help="Checkpoint path (.pt/.pth)")

    # Sampling parameters
    p.add_argument("--n_points", type=int, default=1024,
                   help="Number of points per sample")
    p.add_argument("--n_samples", type=int, default=4,
                   help="Number of samples to generate")
    p.add_argument("--sample_steps", type=int, default=50,
                   help="Number of ODE integration steps")
    p.add_argument("--use_rk4", action="store_true",
                   help="Use RK4 integrator instead of Euler")
    p.add_argument("--use_PE", action="store_true",)

    # Model parameters (must match training)
    p.add_argument("--in_out_dim", type=int, default=3,
                   help="Spatial dimension (default: 3 for 3D, 2 for 2D)")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)

    # Output parameters
    p.add_argument("--out_dir", type=str, default="outputs/dla_eval",
                   help="Output directory")
    p.add_argument("--exp_name", type=str, default=None,
                   help="Experiment name (default: ckpt filename)")
    p.add_argument("--render_image", action="store_true")

    # Miscellaneous
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=123,
                   help="Random seed for x0 initialization")

    return p


# ----------------- Visualization functions (from hilbert_sort_npz.py) -----------------

def visualize_samples(points, save_path, sample_indices=None):
    """
    Visualize samples, using the first two dims as xy, the point index as color,
    and marking the 0th, middle, and last points

    Args:
        points: (B, N, 2) or (B, N, 3) numpy array or torch.Tensor
        save_path: save path
        sample_indices: sample indices to visualize; if None, use 0,1,2,...
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    B = points.shape[0]
    num_cols = min(4, B)
    num_rows = (B + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 7 * num_rows))
    if B == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Select samples
    if sample_indices is None:
        sample_indices = list(range(B))

    N = points.shape[1]
    special_indices = [0, N // 2, N - 1]  # the 0th, middle, and last points
    special_colors = ['red', 'green', 'blue']
    special_labels = ['Point 0 (start)', f'Point {N // 2} (mid)', f'Point {N - 1} (end)']

    # Use the point index as color
    c = np.arange(N)

    for ax_idx, (ax, idx) in enumerate(zip(axes, sample_indices)):
        if ax_idx >= B:
            ax.axis('off')
            continue

        sample = points[ax_idx]  # (N, 2) or (N, 3)
        x = sample[:, 0]
        y = sample[:, 1]

        # Draw all points, colored by index
        scatter = ax.scatter(x, y, c=c, cmap='viridis', s=5, alpha=0.6)

        # Mark the special points
        for si, sc, sl in zip(special_indices, special_colors, special_labels):
            ax.scatter(x[si], y[si], c=sc, s=150, marker='*', edgecolors='black', linewidths=1, label=sl, zorder=10)

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'Sample {idx}')
        plt.colorbar(scatter, ax=ax, label='point index')
        ax.legend(loc='upper right', fontsize=8)

    # Hide the extra subplots
    for ax_idx in range(B, len(axes)):
        axes[ax_idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIS] Visualization saved to: {save_path}")


def r_to_color(r: np.ndarray) -> np.ndarray:
    """
    Convert r values to RGB colors (using the viridis colormap)

    Args:
        r: (N,) array

    Returns:
        (N, 3) uint8 color array
    """
    cmap = plt.cm.viridis
    r_min, r_max = r.min(), r.max()
    if r_max > r_min:
        r_norm = (r - r_min) / (r_max - r_min)
    else:
        r_norm = np.zeros_like(r)
    colors = cmap(r_norm)[:, :3]  # take RGB, drop alpha
    return (colors * 255).astype(np.uint8)


# ----------------- Main logic -----------------

def main():
    args = build_argparser().parse_args()

    # Set in_out_dim based on mode
    if args.mode == "2d":
        args.in_out_dim = 2
    else:
        args.in_out_dim = 3

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Config] mode = {args.mode}, in_out_dim = {args.in_out_dim}")

    # Set the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Experiment name
    exp_name = args.exp_name or Path(args.ckpt).stem
    exp_dir = os.path.join(args.out_dir, exp_name)
    txt_dir = os.path.join(exp_dir, "txt")
    ply_dir = os.path.join(exp_dir, "ply")
    vis_dir = os.path.join(exp_dir, "vis")

    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"[Output] {exp_dir}")

    # Build p_simple (uniform distribution in [-1,1]^3)
    p_simple = Uniform(shape=[args.n_points, args.in_out_dim], a=1.0).to(device)

    # Build the model
    ModelCls = UncondUniGBNTransformer_PE if args.use_PE else UncondUniGBNTransformer

    model = ModelCls(
        n_points=args.n_points,
        in_dim=args.in_out_dim,
        out_dim=args.in_out_dim,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        t_embed_dim=40,
    ).to(device)

    # Load checkpoint
    load_checkpoint(model, args.ckpt, map_location=device)
    model.eval()
    print(f"[ckpt] Loaded from {args.ckpt}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] UncondUniGBNTransformer with {total_params/1e6:.2f}M parameters")

    # ODE sampling
    with torch.no_grad():
        ode = VectorFieldODE(model)
        if args.use_rk4:
            print("[ODE] Using RK4 integrator")
            simulator = RK4Simulator(ode)
        else:
            print("[ODE] Using Euler integrator")
            simulator = EulerSimulator(ode)

        # Sample x0
        b = args.n_samples
        x0, _ = p_simple.sample(b)  # (B, N, D) where D=2 or 3
        print(f"[Sample] x0 shape: {x0.shape}, range: [{x0.min():.4f}, {x0.max():.4f}]")

        # Build the time steps
        ts = torch.linspace(0, 1, args.sample_steps, device=device)
        ts = ts.view(1, -1, 1, 1).expand(b, -1, 1, 1)

        # ODE integration
        x_final = simulator.simulate(x0, ts)  # (B, N, D)
        print(f"[Sample] x_final shape: {x_final.shape}")

        # Convert to numpy
        x_final_np = x_final.detach().cpu().numpy()  # (B, N, D)

        # Coordinate data
        pts_all = x_final_np.astype(np.float32)  # (B, N, D)
        # Compute r = ||xy(z)|| (distance to the origin), used for PLY color
        r_final_np = np.linalg.norm(x_final_np, axis=-1)  # (B, N)

        # Print ranges
        print(f"[x_final] x range: [{x_final_np[..., 0].min():.4f}, {x_final_np[..., 0].max():.4f}]")
        print(f"[x_final] y range: [{x_final_np[..., 1].min():.4f}, {x_final_np[..., 1].max():.4f}]")
        if args.in_out_dim == 3:
            print(f"[x_final] z range: [{x_final_np[..., 2].min():.4f}, {x_final_np[..., 2].max():.4f}]")
        print(f"[r_final] range: [{r_final_np.min():.4f}, {r_final_np.max():.4f}]")

    print(f"[Output] pts shape: {pts_all.shape}")

    # Save txt and ply
    B, N, D = pts_all.shape
    for i in range(B):
        pts = pts_all[i].astype(np.float32)  # (N, D)

        # Write txt: first line N, then coordinates on each subsequent line
        txt_path = os.path.join(txt_dir, f"pts_{i}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{N}\n")
            for j in range(N):
                if D == 2:
                    x, y = pts[j].tolist()
                    f.write(f"{x:.6f} {y:.6f}\n")
                else:
                    x, y, z = pts[j].tolist()
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        # Write ply: color represents r (distance to the origin)
        # For 2D, append z=0 so it can be saved as a 3D PLY
        if D == 2:
            pts_3d = np.concatenate([pts, np.zeros((N, 1), dtype=np.float32)], axis=-1)
        else:
            pts_3d = pts
        r_i = r_final_np[i]  # (N,)
        colors = r_to_color(r_i)  # (N, 3) uint8
        ply_path = os.path.join(ply_dir, f"pts_{i}.ply")
        save_pointcloud_ply(pts_3d, ply_path, colors_np=colors, binary=True)

    print(f"[txt] Wrote {B} files to: {os.path.abspath(txt_dir)}")
    print(f"[ply] Wrote {B} files to: {os.path.abspath(ply_dir)}")

    # Visualization: use the visualize_samples style (first two dims as xy, colored by index)
    if args.render_image:
        vis_path = os.path.join(vis_dir, "dla_vis.png")
        visualize_samples(pts_all, vis_path, sample_indices=list(range(B)))

    print("[Done] Evaluation complete!")


if __name__ == "__main__":
    main()
