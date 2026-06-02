#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_minimal_surface.py

Evaluation script for Minimal Surface Conditional Generation

- Uses anchors as conditions to generate edge_points
- Supports CFG (Classifier-Free Guidance) inference
- Supports loading anchors from a dataset or randomly generating boundary anchors
- Visualizes the generated edge points + anchors
"""

from __future__ import annotations
import os
import sys
import argparse
import torch
import matplotlib
matplotlib.use('Agg')  # Force a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Add data path for minimal_surface_init module
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "minimal-surface")
sys.path.append(DATA_PATH)

from flow_lab.models_conditional import CondPointTransformerSDPA, CondPointTransformerSDPA_PE

from flow_lab.dynamics import EulerSimulator, RK4Simulator, VectorFieldODE
from flow_lab.utils import load_checkpoint
from flow_lab.datasets import MinimalSurfaceDataset
from flow_lab.distributions import Uniform


# ----------------- Argparser -----------------

def build_argparser():
    p = argparse.ArgumentParser(description="Minimal Surface Conditional Evaluation")

    # Required parameters
    p.add_argument("--ckpt", type=str, required=True,
                   help="Checkpoint path (.pt/.pth)")

    # Data parameters
    p.add_argument("--data_path", type=str, default=None,
                   help="Path to NPZ file for loading anchor conditions (optional)")
    p.add_argument("--sample_indices", type=str, default=None,
                   help="Comma-separated indices to sample from dataset (e.g., '0,1,2,3')")

    # Sampling parameters
    p.add_argument("--n_points", type=int, default=256,
                   help="Number of edge points per sample")
    p.add_argument("--n_samples", type=int, default=4,
                   help="Number of samples to generate (used when no data_path)")
    p.add_argument("--sample_steps", type=int, default=50,
                   help="Number of ODE integration steps")
    p.add_argument("--use_rk4", action="store_true",
                   help="Use RK4 integrator instead of Euler")

    # CFG parameters
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="CFG guidance scale (1.0 = no guidance, >1.0 = stronger conditioning)")

    # Model parameters (must match training)
    p.add_argument("--in_out_dim", type=int, default=2,
                   help="Spatial dimension (default: 2 for 2D)")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--cross_every", type=int, default=2)
    p.add_argument("--use_PE", action="store_true",
                   help="Use model without learnable positional encoding")

    # Output parameters
    p.add_argument("--out_dir", type=str, default="outputs/minimal_surface_eval",
                   help="Output directory")
    p.add_argument("--exp_name", type=str, default=None,
                   help="Experiment name (default: ckpt filename)")

    # Random anchor generation
    p.add_argument("--random_anchors", action="store_true",
                   help="Generate random boundary anchors (like training data)")
    p.add_argument("--grid_size", type=int, default=256,
                   help="Grid size for random anchor generation (default: 256)")
    p.add_argument("--n_anchors", type=int, default=3,
                   help="Number of anchors to generate (default: 3)")

    # Miscellaneous
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=123,
                   help="Random seed for x0 initialization")

    return p


# ----------------- Random Anchor Generation -----------------

def generate_random_boundary_anchors(n_samples: int, grid_size: int = 256,
                                     n_anchors: int = 3, seed: int = 42) -> np.ndarray:
    """
    Generate random anchor points on the domain boundary, similar to
    random_boundary_anchors in data/minimal-surface/minimal_surface_init.py

    Args:
        n_samples: Number of samples to generate
        grid_size: Size of the grid (W = H = grid_size)
        n_anchors: Number of anchor points per sample
        seed: Random seed

    Returns:
        anchors: (n_samples, n_anchors, 2) in normalized [-1, 1] range
    """
    rng = np.random.default_rng(seed)

    W = H = grid_size
    perimeter = 2 * (W - 1) + 2 * (H - 1)

    all_anchors = []

    for i in range(n_samples):
        # Generate evenly-spaced positions with some jitter
        base_positions = np.linspace(0, perimeter, n_anchors, endpoint=False)
        jitter = rng.uniform(-perimeter / (3 * n_anchors), perimeter / (3 * n_anchors), n_anchors)
        positions = (base_positions + jitter) % perimeter

        anchor_coords = []
        for pos in positions:
            if pos < W - 1:
                # Bottom edge
                x = pos
                y = 0
            elif pos < W - 1 + H - 1:
                # Right edge
                x = W - 1
                y = pos - (W - 1)
            elif pos < 2 * (W - 1) + H - 1:
                # Top edge (reverse direction)
                x = (W - 1) - (pos - (W - 1 + H - 1))
                y = H - 1
            else:
                # Left edge (reverse direction)
                x = 0
                y = (H - 1) - (pos - (2 * (W - 1) + H - 1))

            anchor_coords.append([x, y])

        points = np.array(anchor_coords, dtype=np.float32)

        # Order points by angle from centroid (counter-clockwise)
        centroid = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        order = np.argsort(angles)
        ordered_points = points[order]

        all_anchors.append(ordered_points)

    anchors = np.stack(all_anchors, axis=0)  # (n_samples, n_anchors, 2)

    # Normalize from [0, grid_size-1] to [-1, 1]
    anchors = (anchors / (grid_size - 1)) * 2 - 1  # [0, 255] -> [0, 1] -> [-1, 1]

    return anchors.astype(np.float32)


# ----------------- CFG ODE Wrapper -----------------

class ConditionalVectorFieldODE:
    """
    Conditional ODE wrapper with CFG support.
    Compatible with EulerSimulator/RK4Simulator interface.
    """
    def __init__(self, model, anchors: torch.Tensor, cfg_scale: float = 1.0):
        self.model = model
        self.anchors = anchors  # (B, 6)
        self.cfg_scale = cfg_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        ODE drift coefficient (velocity field).

        Args:
            x: (B, N, 2) - current state
            t: (B, 1, 1) - time
            y: ignored (for compatibility)

        Returns:
            v: (B, N, 2) - velocity
        """
        if self.cfg_scale == 1.0:
            # No CFG, just use conditional prediction
            return self.model(x, t, img_cond=self.anchors, cond_drop=False)
        else:
            # CFG: v = v_uncond + cfg_scale * (v_cond - v_uncond)
            v_cond = self.model(x, t, img_cond=self.anchors, cond_drop=False)
            v_uncond = self.model(x, t, img_cond=self.anchors, cond_drop=True)
            return v_uncond + self.cfg_scale * (v_cond - v_uncond)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return zero diffusion (deterministic ODE)."""
        return torch.zeros_like(x)


# ----------------- Visualization functions -----------------

def visualize_minimal_surface(edge_points, anchors, save_path, title_prefix="Sample"):
    """
    Visualize a minimal surface sample: edge points + anchors

    Args:
        edge_points: (B, N, 2) numpy array - edge points
        anchors: (B, 3, 2) numpy array - anchors
        save_path: save path
        title_prefix: title prefix
    """
    if isinstance(edge_points, torch.Tensor):
        edge_points = edge_points.cpu().numpy()
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.cpu().numpy()

    B = edge_points.shape[0]
    num_cols = min(4, B)
    num_rows = (B + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
    if B == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(B):
        ax = axes[idx]

        # Edge points
        pts = edge_points[idx]  # (N, 2)
        ax.scatter(pts[:, 0], pts[:, 1], c='blue', s=3, alpha=0.6, label='Edge points')

        # Anchors
        anch = anchors[idx]  # (3, 2)
        ax.scatter(anch[:, 0], anch[:, 1], c='red', s=100, marker='^',
                   edgecolors='black', linewidths=1, label='Anchors', zorder=10)

        # Label the anchor indices
        for i, (ax_x, ay_y) in enumerate(anch):
            ax.annotate(f'A{i}', (ax_x, ay_y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color='red')

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'{title_prefix} {idx}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide the extra subplots
    for idx in range(B, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIS] Visualization saved to: {save_path}")


def visualize_comparison(generated, ground_truth, anchors, save_path):
    """
    Side-by-side visualization: generated result vs. ground-truth data

    Args:
        generated: (B, N, 2) numpy array - generated edge points
        ground_truth: (B, N, 2) numpy array - ground-truth edge points
        anchors: (B, 3, 2) numpy array - anchors
        save_path: save path
    """
    if isinstance(generated, torch.Tensor):
        generated = generated.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.cpu().numpy()

    B = generated.shape[0]
    fig, axes = plt.subplots(B, 2, figsize=(12, 6 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for idx in range(B):
        # Generated result
        ax_gen = axes[idx, 0]
        pts_gen = generated[idx]
        ax_gen.scatter(pts_gen[:, 0], pts_gen[:, 1], c='blue', s=3, alpha=0.6, label='Generated')
        anch = anchors[idx]
        ax_gen.scatter(anch[:, 0], anch[:, 1], c='red', s=100, marker='^',
                       edgecolors='black', linewidths=1, label='Anchors', zorder=10)
        ax_gen.set_xlim(-1.1, 1.1)
        ax_gen.set_ylim(-1.1, 1.1)
        ax_gen.set_aspect('equal')
        ax_gen.set_title(f'Generated (Sample {idx})')
        ax_gen.legend(loc='upper right', fontsize=8)
        ax_gen.grid(True, alpha=0.3)

        # Ground-truth data
        ax_gt = axes[idx, 1]
        pts_gt = ground_truth[idx]
        ax_gt.scatter(pts_gt[:, 0], pts_gt[:, 1], c='green', s=3, alpha=0.6, label='Ground Truth')
        ax_gt.scatter(anch[:, 0], anch[:, 1], c='red', s=100, marker='^',
                      edgecolors='black', linewidths=1, label='Anchors', zorder=10)
        ax_gt.set_xlim(-1.1, 1.1)
        ax_gt.set_ylim(-1.1, 1.1)
        ax_gt.set_aspect('equal')
        ax_gt.set_title(f'Ground Truth (Sample {idx})')
        ax_gt.legend(loc='upper right', fontsize=8)
        ax_gt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIS] Comparison saved to: {save_path}")


# ----------------- Main logic -----------------

def main():
    args = build_argparser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Config] in_out_dim = {args.in_out_dim}, n_points = {args.n_points}")
    print(f"[Config] CFG scale = {args.cfg_scale}")

    # Set the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Experiment name
    exp_name = args.exp_name or Path(args.ckpt).stem
    exp_dir = os.path.join(args.out_dir, exp_name)
    npz_dir = os.path.join(exp_dir, "npz")
    vis_dir = os.path.join(exp_dir, "vis")

    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"[Output] {exp_dir}")

    # Build p_simple (uniform distribution in [-1,1]^2)
    p_simple = Uniform(shape=[args.n_points, args.in_out_dim], a=1.0).to(device)

    # Build the model
    ModelCls = CondPointTransformerSDPA_PE if args.use_PE else CondPointTransformerSDPA
    model = ModelCls(
        n_points=args.n_points,
        in_dim=args.in_out_dim,
        out_dim=args.in_out_dim,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        t_embed_dim=40,
        img_dim=6,                    # flattened anchor dimension (3*2=6)
        cross_every=args.cross_every,
        max_img_tokens=1,
    ).to(device)
    
    # Load checkpoint
    load_checkpoint(model, args.ckpt, map_location=device)
    model.eval()
    print(f"[ckpt] Loaded from {args.ckpt}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] CondPointTransformerSDPA with {total_params/1e6:.2f}M parameters")

    # Obtain the anchor conditions
    ground_truth = None
    if args.data_path:
        # Load anchors from the dataset
        dataset = MinimalSurfaceDataset(args.data_path)

        if args.sample_indices:
            indices = [int(i) for i in args.sample_indices.split(',')]
        else:
            # Randomly choose samples
            rng = np.random.default_rng(seed=args.seed)
            indices = rng.choice(dataset.num_meshes, size=args.n_samples, replace=False).tolist()

        print(f"[Data] Using samples: {indices}")

        # Get the data
        _, x1_np, anchors_np = dataset.compute_batch(indices, epoch=0, step=0)
        anchors = torch.from_numpy(anchors_np).float().to(device)  # (B, 6)
        ground_truth = x1_np  # (B, N, 2) numpy
        anchors_vis = anchors_np.reshape(-1, 3, 2)  # (B, 3, 2) for visualization

        b = len(indices)
    elif args.random_anchors:
        # Generate random boundary anchors (similar to the training data)
        b = args.n_samples
        print(f"[Data] Generating {b} random BOUNDARY anchors (grid_size={args.grid_size}, n_anchors={args.n_anchors})")

        anchors_vis = generate_random_boundary_anchors(
            n_samples=b,
            grid_size=args.grid_size,
            n_anchors=args.n_anchors,
            seed=args.seed
        )  # (B, n_anchors, 2) in [-1, 1]
        anchors = torch.from_numpy(anchors_vis.reshape(b, -1)).float().to(device)  # (B, 6)
    else:
        # Generate fully random anchors (not on the boundary)
        b = args.n_samples
        print(f"[Data] Generating {b} random anchor conditions (uniform in [-0.8, 0.8])")

        # Random anchors in [-0.8, 0.8]^2
        anchors_vis = np.random.uniform(-0.8, 0.8, size=(b, 3, 2)).astype(np.float32)
        anchors = torch.from_numpy(anchors_vis.reshape(b, -1)).float().to(device)  # (B, 6)

    print(f"[Anchors] shape: {anchors.shape}")

    # ODE sampling
    with torch.no_grad():
        # Create the conditional ODE
        cond_ode = ConditionalVectorFieldODE(model, anchors, cfg_scale=args.cfg_scale)

        if args.use_rk4:
            print("[ODE] Using RK4 integrator")
            simulator = RK4Simulator(cond_ode)
        else:
            print("[ODE] Using Euler integrator")
            simulator = EulerSimulator(cond_ode)

        # Sample x0
        x0, _ = p_simple.sample(b)  # (B, N, 2)
        print(f"[Sample] x0 shape: {x0.shape}, range: [{x0.min():.4f}, {x0.max():.4f}]")

        # Build the time steps
        ts = torch.linspace(0, 1, args.sample_steps, device=device)
        ts = ts.view(1, -1, 1, 1).expand(b, -1, 1, 1)

        # ODE integration
        x_final = simulator.simulate(x0, ts)  # (B, N, 2)
        print(f"[Sample] x_final shape: {x_final.shape}")

        # Convert to numpy
        x_final_np = x_final.detach().cpu().numpy().astype(np.float32)  # (B, N, 2)

        # Print ranges
        print(f"[x_final] x range: [{x_final_np[..., 0].min():.4f}, {x_final_np[..., 0].max():.4f}]")
        print(f"[x_final] y range: [{x_final_np[..., 1].min():.4f}, {x_final_np[..., 1].max():.4f}]")

    # Save NPZ
    npz_path = os.path.join(npz_dir, "generated.npz")
    np.savez_compressed(
        npz_path,
        edge_points=x_final_np,
        anchors=anchors_vis,
        ground_truth=ground_truth if ground_truth is not None else np.array([]),
    )
    print(f"[NPZ] Saved to: {npz_path}")

    # Visualize the generated result
    vis_path = os.path.join(vis_dir, "generated.png")
    visualize_minimal_surface(x_final_np, anchors_vis, vis_path, title_prefix="Generated")

    # If ground-truth data is available, produce a comparison visualization
    if ground_truth is not None:
        compare_path = os.path.join(vis_dir, "comparison.png")
        visualize_comparison(x_final_np, ground_truth, anchors_vis, compare_path)

        # Compute a simple MSE metric
        mse = np.mean((x_final_np - ground_truth) ** 2)
        print(f"[Metric] MSE (generated vs ground truth): {mse:.6f}")

    print("[Done] Evaluation complete!")


if __name__ == "__main__":
    main()
