#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_thomson_multi.py

Evaluation script for the Thomson problem (3D):
   - Outputs xyz coordinates directly
   - 3D point cloud visualization
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
import math
from pathlib import Path
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from flow_lab.models import UncondUniGBNTransformer, UncondUniGBNTransformer_PE
from flow_lab.dynamics import EulerSimulator, RK4Simulator, VectorFieldODE
from flow_lab.utils import load_checkpoint, save_pointcloud_ply
from flow_lab.distributions import Uniform


# ----------------- Argparser -----------------

def build_argparser():
    p = argparse.ArgumentParser(description="Thomson problem evaluation")

    # Required parameters
    p.add_argument("--ckpt", type=str, required=True,
                   help="Checkpoint path (.pt/.pth)")

    # Sampling parameters
    p.add_argument("--n_points", type=int, default=384,
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
                   help="Spatial dimension (3 for 3D xyz)")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)

    # Output parameters
    p.add_argument("--out_dir", type=str, default="outputs/thomson_eval",
                   help="Output directory")
    p.add_argument("--exp_name", type=str, default=None,
                   help="Experiment name (default: ckpt filename)")

    # Miscellaneous
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=123,
                   help="Random seed for x0 initialization")

    # Trajectory output
    p.add_argument("--output_trajectory", action="store_true",
                   help="Output trajectory PLY sequence for animation")
    p.add_argument("--is_single_mesh", action="store_true",
                   help="Merge all batches into a single PLY per frame")
    p.add_argument("--is_single_pcd", action="store_true",
                   help="Single point cloud mode (xyz+rgb)")
    p.add_argument("--output_ply_rgb", action="store_true",
                   help="Output velocity as RGB color instead of normals")
    p.add_argument("--num_workers", type=int, default=None,
                   help="Number of workers for parallel PLY writing")

    return p


# ----------------- Trajectory output helper functions -----------------

def _write_ply_with_batch_idx(ply_path: str, pts: np.ndarray, normals: np.ndarray, batch_idx: np.ndarray):
    """
    Write a PLY file (binary format) with a batch_idx attribute.
    Each point contains: x, y, z, nx, ny, nz, batch_idx
    """
    n_points = pts.shape[0]

    header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property int batch_idx
end_header
"""

    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
        ('batch_idx', '<i4')
    ])
    data = np.empty(n_points, dtype=dtype)
    data['x'] = pts[:, 0].astype(np.float32)
    data['y'] = pts[:, 1].astype(np.float32)
    data['z'] = pts[:, 2].astype(np.float32)
    data['nx'] = normals[:, 0].astype(np.float32)
    data['ny'] = normals[:, 1].astype(np.float32)
    data['nz'] = normals[:, 2].astype(np.float32)
    data['batch_idx'] = batch_idx.astype(np.int32)

    with open(ply_path, 'wb') as f:
        f.write(header.encode('ascii'))
        data.tofile(f)


def _write_ply_with_rgb(ply_path: str, pts: np.ndarray, rgb: np.ndarray, batch_idx: np.ndarray):
    """
    Write a PLY file (binary format) with RGB color and a batch_idx attribute.
    Each point contains: x, y, z, red, green, blue, batch_idx
    """
    n_points = pts.shape[0]

    header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int batch_idx
end_header
"""

    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('batch_idx', '<i4')
    ])
    data = np.empty(n_points, dtype=dtype)
    data['x'] = pts[:, 0].astype(np.float32)
    data['y'] = pts[:, 1].astype(np.float32)
    data['z'] = pts[:, 2].astype(np.float32)
    data['red'] = rgb[:, 0].astype(np.uint8)
    data['green'] = rgb[:, 1].astype(np.uint8)
    data['blue'] = rgb[:, 2].astype(np.uint8)
    data['batch_idx'] = batch_idx.astype(np.int32)

    with open(ply_path, 'wb') as f:
        f.write(header.encode('ascii'))
        data.tofile(f)


def _velocity_to_rgb(vel: np.ndarray) -> np.ndarray:
    """
    Convert the velocity field to RGB color.
    Map velocity direction to color: map the normalized velocity vector from [-1,1] to [0,255].
    """
    vel_norm = np.linalg.norm(vel, axis=-1, keepdims=True)
    vel_normalized = vel / np.maximum(vel_norm, 1e-8)  # [-1, 1]
    rgb = ((vel_normalized + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)  # [0, 255]
    return rgb


def _write_single_frame_ply(args_tuple):
    """
    Multiprocessing worker function: write a merged PLY for a single frame.
    args_tuple: (t_idx, Xs_t, Vs_t, B_, N_pts, traj_ply_dir, output_ply_rgb)
    """
    t_idx, Xs_t, Vs_t, B_, N_pts, traj_ply_dir, output_ply_rgb = args_tuple

    # Merge points and velocities of all batches
    all_pts = Xs_t.reshape(-1, 3)  # (B*N, 3)
    all_vel = Vs_t.reshape(-1, 3)  # (B*N, 3)

    # Generate the batch_idx attribute
    batch_idx = np.repeat(np.arange(B_, dtype=np.int32), N_pts)  # (B*N,)

    # Write PLY
    ply_name = f"frame_{t_idx:04d}.ply"
    ply_path = os.path.join(traj_ply_dir, ply_name)

    if output_ply_rgb:
        # Convert the velocity field to RGB color
        rgb = _velocity_to_rgb(all_vel)
        _write_ply_with_rgb(ply_path, all_pts, rgb, batch_idx)
    else:
        # Use normalized velocity as the normal vector
        vel_norm = np.linalg.norm(all_vel, axis=-1, keepdims=True)
        all_normals = all_vel / np.maximum(vel_norm, 1e-8)
        _write_ply_with_batch_idx(ply_path, all_pts, all_normals, batch_idx)

    return t_idx


# ----------------- Color mapping -----------------

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


# ----------------- 3D XYZ visualization (for 3D mode) -----------------

def visualize_xyz_pointcloud(xyz_list: list, r_list: list, save_path: str, sample_indices: list = None):
    """
    3D XYZ point cloud visualization, with r shown by color

    Args:
        xyz_list: list of (N, 3) arrays, each one is the xyz coordinates of a sample
        r_list: list of (N,) arrays, the r value of each point (distance to the origin)
        save_path: save path
        sample_indices: list of sample indices (used for titles)
    """
    n_samples = len(xyz_list)
    num_cols = min(4, n_samples)
    num_rows = (n_samples + num_cols - 1) // num_cols

    fig = plt.figure(figsize=(5 * num_cols, 4 * num_rows))

    for i, (xyz, r_i) in enumerate(zip(xyz_list, r_list)):
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection='3d')

        # Draw points, colored by r
        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                           c=r_i, cmap='viridis', s=5, alpha=0.8)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('r (distance to origin)', fontsize=8)

        # Mark the first and last points
        ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2],
                   c='red', s=30, marker='o', label='first')
        ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2],
                   c='green', s=30, marker='^', label='last')

        ax.set_box_aspect([1, 1, 1])
        r_max = r_i.max()
        max_range = max(1.2, r_max * 1.1) if r_max > 0 else 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        r_min, r_max_val = r_i.min(), r_i.max()
        idx_str = sample_indices[i] if sample_indices else i
        ax.set_title(f"Sample {idx_str}\nr: [{r_min:.3f}, {r_max_val:.3f}]")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[VIS] Saved 3D xyz visualization to: {save_path}")


# ----------------- Main logic -----------------

def main():
    args = build_argparser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Config] in_out_dim = {args.in_out_dim}")

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

    # Build p_simple (uniform distribution)
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

        if args.output_trajectory:
            # ===== Trajectory mode =====
            print("[trajectory] Running simulate_with_trajectory...")
            xs, vs = simulator.simulate_with_trajectory(x0, ts, return_velocity=True)
            # xs: (B, T, N, D), vs: (B, T, N, D)
            x_final = xs[:, -1]  # (B, N, D)
            print(f"[trajectory] xs shape: {xs.shape}")
            print(f"[trajectory] x_final shape: {x_final.shape}")

            # === Export the trajectory PLY sequence ===
            traj_ply_dir = os.path.join(exp_dir, "trajectory_ply")
            os.makedirs(traj_ply_dir, exist_ok=True)

            Xs = xs.detach().cpu().float().numpy()  # (B, T, N, 3)
            Vs = vs.detach().cpu().float().numpy()  # (B, T, N, 3)
            B_, T_steps, N_pts, D = Xs.shape
            assert D == 3

            if args.is_single_mesh or args.is_single_pcd:
                # ===== Single-file mode: merge all batches of each frame into one PLY, parallelized with multiprocessing =====
                n_workers = args.num_workers if args.num_workers else cpu_count()
                print(
                    f"[trajectory_ply] exporting {T_steps} frames (single mesh mode), "
                    f"each frame has {B_} batches × {N_pts} points = {B_ * N_pts} points, "
                    f"using {n_workers} workers"
                )

                # Prepare multiprocessing arguments
                task_args = [
                    (t_idx, Xs[:, t_idx], Vs[:, t_idx], B_, N_pts, traj_ply_dir, args.output_ply_rgb)
                    for t_idx in range(T_steps)
                ]

                with Pool(n_workers) as pool:
                    results = pool.map(_write_single_frame_ply, task_args)

                print(f"[trajectory_ply] wrote {len(results)} merged PLY files to: {os.path.abspath(traj_ply_dir)}")

            else:
                # ===== Original mode: one PLY per batch =====
                print(
                    f"[trajectory_ply] exporting {T_steps} time steps "
                    f"for {B_} batches to {traj_ply_dir} (with velocity normals)"
                )

                for t_idx in range(T_steps):
                    for b_idx in range(B_):
                        pts = Xs[b_idx, t_idx]  # (N, 3)
                        vel = Vs[b_idx, t_idx]  # (N, 3)

                        # Normalize the velocity to a unit vector and use it as the normal vector
                        vel_norm = np.linalg.norm(vel, axis=-1, keepdims=True)
                        normals = vel / np.maximum(vel_norm, 1e-8)

                        # Generate batch_idx
                        batch_idx = np.full(N_pts, b_idx, dtype=np.int32)

                        ply_name = f"frame_{t_idx:04d}_b{b_idx}.ply"
                        ply_path = os.path.join(traj_ply_dir, ply_name)

                        if args.output_ply_rgb:
                            rgb = _velocity_to_rgb(vel)
                            _write_ply_with_rgb(ply_path, pts, rgb, batch_idx)
                        else:
                            _write_ply_with_batch_idx(ply_path, pts, normals, batch_idx)

                print(f"[trajectory_ply] wrote PLY sequence to: {os.path.abspath(traj_ply_dir)}")
        else:
            # ===== Only the final step =====
            x_final = simulator.simulate(x0, ts)  # (B, N, D)

        print(f"[Sample] x_final shape: {x_final.shape}")

        # Convert to numpy
        x_final_np = x_final.detach().cpu().numpy()  # (B, N, 3)

        # These are directly the xyz coordinates
        xyz_all = x_final_np.astype(np.float32)  # (B, N, 3)
        # Compute r = ||xyz|| (distance to the origin)
        r_final_np = np.linalg.norm(x_final_np, axis=-1)  # (B, N)

        # Print ranges
        print(f"[x_final] x range: [{x_final_np[..., 0].min():.4f}, {x_final_np[..., 0].max():.4f}]")
        print(f"[x_final] y range: [{x_final_np[..., 1].min():.4f}, {x_final_np[..., 1].max():.4f}]")
        print(f"[x_final] z range: [{x_final_np[..., 2].min():.4f}, {x_final_np[..., 2].max():.4f}]")
        print(f"[r_final] range: [{r_final_np.min():.4f}, {r_final_np.max():.4f}]")

    print(f"[Output] xyz shape: {xyz_all.shape}")

    # Save txt and ply
    B, N, _ = xyz_all.shape
    for i in range(B):
        pts = xyz_all[i].astype(np.float32)  # (N, 3)

        # Write txt: first line N, then x y z on each subsequent line
        txt_path = os.path.join(txt_dir, f"pts_{i}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{N}\n")
            for j in range(N):
                x, y, z = pts[j].tolist()
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        # Write ply: color represents r
        r_i = r_final_np[i]  # (N,)
        colors = r_to_color(r_i)  # (N, 3) uint8
        ply_path = os.path.join(ply_dir, f"pts_{i}.ply")
        save_pointcloud_ply(pts, ply_path, colors_np=colors, binary=True)

    print(f"[txt] Wrote {B} files to: {os.path.abspath(txt_dir)}")
    print(f"[ply] Wrote {B} files to: {os.path.abspath(ply_dir)}")

    # Visualization
    xyz_list = [xyz_all[i] for i in range(B)]

    r_list = [r_final_np[i] for i in range(B)]

    # Directly visualize the xyz point cloud, with r shown by color
    vis_path = os.path.join(vis_dir, "xyz_vis.png")
    visualize_xyz_pointcloud(xyz_list, r_list, vis_path, sample_indices=list(range(B)))

    print("[Done] Evaluation complete!")


if __name__ == "__main__":
    main()
