#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_minimal_surface_variable.py

Minimal Surface Conditional Generation with VARIABLE anchors (3-8).

Key features:
- Pad-to-8: Anchors padded to max 8 with NaN
- Mask: Attention mask ignores padded anchor positions
- Missing embedding: Learnable embedding for padded anchor slots
- CFG: Classifier-Free Guidance support
"""

from __future__ import annotations
import os
import sys
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "minimal-surface")
sys.path.append(DATA_PATH)

from flow_lab.models_conditional import CondPointTransformerSDPAVariable
from flow_lab.dynamics import EulerSimulator, RK4Simulator
from flow_lab.utils import load_checkpoint
from flow_lab.datasets import MinimalSurfaceVariableDataset
from flow_lab.distributions import Uniform


# ----------------- Argparser -----------------

def build_argparser():
    p = argparse.ArgumentParser(description="Minimal Surface Variable Anchor Evaluation")

    # Required
    p.add_argument("--ckpt", type=str, required=True,
                   help="Checkpoint path (.pt/.pth)")

    # Data
    p.add_argument("--data_path", type=str, default=None,
                   help="Path to variable anchor NPZ file")
    p.add_argument("--sample_indices", type=str, default=None,
                   help="Comma-separated indices (e.g., '0,1,2,3')")

    # Sampling
    p.add_argument("--n_points", type=int, default=256,
                   help="Number of edge points per sample")
    p.add_argument("--n_samples", type=int, default=4,
                   help="Number of samples to generate")
    p.add_argument("--sample_steps", type=int, default=50,
                   help="ODE integration steps")
    p.add_argument("--use_rk4", action="store_true",
                   help="Use RK4 integrator")

    # CFG
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="CFG guidance scale (1.0 = no guidance)")

    # Model architecture (must match training)
    p.add_argument("--in_out_dim", type=int, default=2)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--cross_every", type=int, default=2)
    p.add_argument("--max_anchors", type=int, default=8,
                   help="Maximum number of anchors (default: 8)")

    # Output
    p.add_argument("--out_dir", type=str, default="outputs/minimal_surface_variable_eval")
    p.add_argument("--exp_name", type=str, default=None)

    # Random anchor generation
    p.add_argument("--random_anchors", action="store_true",
                   help="Generate random boundary anchors")
    p.add_argument("--grid_size", type=int, default=256)
    p.add_argument("--min_n_anchors", type=int, default=3,
                   help="Min number of random anchors")
    p.add_argument("--max_n_anchors", type=int, default=8,
                   help="Max number of random anchors")

    # Other
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=123)

    return p


# ----------------- Random Anchor Generation -----------------

def generate_random_variable_anchors(n_samples: int, grid_size: int = 256,
                                     min_anchors: int = 3, max_anchors: int = 8,
                                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random boundary anchors with variable count.

    Returns:
        anchors: (n_samples, max_anchors, 2) - NaN padded, normalized to [-1, 1]
        anchor_mask: (n_samples, max_anchors) - True for valid
    """
    rng = np.random.default_rng(seed)

    W = H = grid_size
    perimeter = 2 * (W - 1) + 2 * (H - 1)

    all_anchors = np.full((n_samples, max_anchors, 2), np.nan, dtype=np.float32)
    all_masks = np.zeros((n_samples, max_anchors), dtype=bool)

    for i in range(n_samples):
        n_anch = rng.integers(min_anchors, max_anchors + 1)

        # Generate evenly-spaced positions with jitter
        base_positions = np.linspace(0, perimeter, n_anch, endpoint=False)
        jitter = rng.uniform(-perimeter / (3 * n_anch), perimeter / (3 * n_anch), n_anch)
        positions = (base_positions + jitter) % perimeter

        anchor_coords = []
        for pos in positions:
            if pos < W - 1:
                x, y = pos, 0
            elif pos < W - 1 + H - 1:
                x, y = W - 1, pos - (W - 1)
            elif pos < 2 * (W - 1) + H - 1:
                x, y = (W - 1) - (pos - (W - 1 + H - 1)), H - 1
            else:
                x, y = 0, (H - 1) - (pos - (2 * (W - 1) + H - 1))
            anchor_coords.append([x, y])

        points = np.array(anchor_coords, dtype=np.float32)

        # Order by angle from centroid
        centroid = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        order = np.argsort(angles)
        ordered_points = points[order]

        # Normalize [0, grid_size-1] -> [-1, 1]
        ordered_points = (ordered_points / (grid_size - 1)) * 2 - 1

        all_anchors[i, :n_anch] = ordered_points
        all_masks[i, :n_anch] = True

    return all_anchors, all_masks


# ----------------- CFG ODE Wrapper -----------------

class ConditionalVectorFieldODEVariable:
    """
    Conditional ODE wrapper for variable anchors with CFG support.
    """
    def __init__(self, model, anchors: torch.Tensor, anchor_mask: torch.Tensor,
                 cfg_scale: float = 1.0):
        self.model = model
        self.anchors = anchors      # (B, max_anchors, 2)
        self.anchor_mask = anchor_mask  # (B, max_anchors)
        self.cfg_scale = cfg_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        if self.cfg_scale == 1.0:
            return self.model(x, t, anchors=self.anchors, anchor_mask=self.anchor_mask, cond_drop=False)
        else:
            v_cond = self.model(x, t, anchors=self.anchors, anchor_mask=self.anchor_mask, cond_drop=False)
            v_uncond = self.model(x, t, anchors=self.anchors, anchor_mask=self.anchor_mask, cond_drop=True)
            return v_uncond + self.cfg_scale * (v_cond - v_uncond)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


# ----------------- Visualization -----------------

def visualize_variable_anchors(edge_points, anchors, anchor_mask, save_path, title_prefix="Sample"):
    """
    Visualize minimal surface with variable anchors.

    Args:
        edge_points: (B, N, 2) numpy array
        anchors: (B, max_anchors, 2) numpy array - NaN padded
        anchor_mask: (B, max_anchors) numpy array - True for valid
        save_path: output path
    """
    if isinstance(edge_points, torch.Tensor):
        edge_points = edge_points.cpu().numpy()
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.cpu().numpy()
    if isinstance(anchor_mask, torch.Tensor):
        anchor_mask = anchor_mask.cpu().numpy()

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
        pts = edge_points[idx]
        ax.scatter(pts[:, 0], pts[:, 1], c='blue', s=3, alpha=0.6, label='Edge points')

        # Valid anchors only
        mask = anchor_mask[idx]
        anch = anchors[idx][mask]
        n_valid = mask.sum()

        ax.scatter(anch[:, 0], anch[:, 1], c='red', s=100, marker='^',
                   edgecolors='black', linewidths=1, label=f'Anchors ({n_valid})', zorder=10)

        for i, (ax_x, ay_y) in enumerate(anch):
            ax.annotate(f'A{i}', (ax_x, ay_y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color='red')

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'{title_prefix} {idx} ({n_valid} anchors)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(B, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIS] Saved to: {save_path}")


def visualize_comparison_variable(generated, ground_truth, anchors, anchor_mask, save_path):
    """Compare generated vs ground truth with variable anchors."""
    if isinstance(generated, torch.Tensor):
        generated = generated.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.cpu().numpy()
    if isinstance(anchor_mask, torch.Tensor):
        anchor_mask = anchor_mask.cpu().numpy()

    B = generated.shape[0]
    fig, axes = plt.subplots(B, 2, figsize=(12, 6 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for idx in range(B):
        mask = anchor_mask[idx]
        anch = anchors[idx][mask]
        n_valid = mask.sum()

        # Generated
        ax_gen = axes[idx, 0]
        pts_gen = generated[idx]
        ax_gen.scatter(pts_gen[:, 0], pts_gen[:, 1], c='blue', s=3, alpha=0.6, label='Generated')
        ax_gen.scatter(anch[:, 0], anch[:, 1], c='red', s=100, marker='^',
                       edgecolors='black', linewidths=1, zorder=10)
        ax_gen.set_xlim(-1.1, 1.1)
        ax_gen.set_ylim(-1.1, 1.1)
        ax_gen.set_aspect('equal')
        ax_gen.set_title(f'Generated (Sample {idx}, {n_valid} anchors)')
        ax_gen.grid(True, alpha=0.3)

        # Ground truth
        ax_gt = axes[idx, 1]
        pts_gt = ground_truth[idx]
        ax_gt.scatter(pts_gt[:, 0], pts_gt[:, 1], c='green', s=3, alpha=0.6, label='Ground Truth')
        ax_gt.scatter(anch[:, 0], anch[:, 1], c='red', s=100, marker='^',
                      edgecolors='black', linewidths=1, zorder=10)
        ax_gt.set_xlim(-1.1, 1.1)
        ax_gt.set_ylim(-1.1, 1.1)
        ax_gt.set_aspect('equal')
        ax_gt.set_title(f'Ground Truth (Sample {idx})')
        ax_gt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIS] Comparison saved to: {save_path}")


# ----------------- Main -----------------

def main():
    args = build_argparser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Config] n_points={args.n_points}, max_anchors={args.max_anchors}")
    print(f"[Config] CFG scale={args.cfg_scale}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_name = args.exp_name or Path(args.ckpt).stem
    exp_dir = os.path.join(args.out_dir, exp_name)
    npz_dir = os.path.join(exp_dir, "npz")
    vis_dir = os.path.join(exp_dir, "vis")

    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"[Output] {exp_dir}")

    # Uniform distribution for x0
    p_simple = Uniform(shape=[args.n_points, args.in_out_dim], a=1.0).to(device)

    # Build model
    model = CondPointTransformerSDPAVariable(
        n_points=args.n_points,
        in_dim=args.in_out_dim,
        out_dim=args.in_out_dim,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        t_embed_dim=40,
        max_anchors=args.max_anchors,
        anchor_dim=2,
        cross_every=args.cross_every,
    ).to(device)

    load_checkpoint(model, args.ckpt, map_location=device)
    model.eval()
    print(f"[ckpt] Loaded from {args.ckpt}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] CondPointTransformerSDPAVariable with {total_params/1e6:.2f}M params")

    # Get anchor conditions
    ground_truth = None
    if args.data_path:
        dataset = MinimalSurfaceVariableDataset(args.data_path, max_anchors=args.max_anchors)

        if args.sample_indices:
            indices = [int(i) for i in args.sample_indices.split(',')]
        else:
            rng = np.random.default_rng(seed=args.seed)
            indices = rng.choice(dataset.num_meshes, size=args.n_samples, replace=False).tolist()

        print(f"[Data] Using samples: {indices}")

        _, x1_np, anchors_np, anchor_mask_np = dataset.compute_batch(indices)
        anchors = torch.from_numpy(anchors_np).float().to(device)
        anchor_mask = torch.from_numpy(anchor_mask_np).to(device)
        ground_truth = x1_np

        b = len(indices)
    elif args.random_anchors:
        b = args.n_samples
        print(f"[Data] Generating {b} random boundary anchors "
              f"(n_anchors: {args.min_n_anchors}-{args.max_n_anchors})")

        anchors_np, anchor_mask_np = generate_random_variable_anchors(
            n_samples=b,
            grid_size=args.grid_size,
            min_anchors=args.min_n_anchors,
            max_anchors=args.max_anchors,
            seed=args.seed
        )
        anchors = torch.from_numpy(anchors_np).float().to(device)
        anchor_mask = torch.from_numpy(anchor_mask_np).to(device)
    else:
        # Random anchors in [-0.8, 0.8]
        b = args.n_samples
        print(f"[Data] Generating {b} random uniform anchors")

        anchors_np = np.full((b, args.max_anchors, 2), np.nan, dtype=np.float32)
        anchor_mask_np = np.zeros((b, args.max_anchors), dtype=bool)

        rng = np.random.default_rng(args.seed)
        for i in range(b):
            n_anch = rng.integers(args.min_n_anchors, args.max_n_anchors + 1)
            pts = rng.uniform(-0.8, 0.8, size=(n_anch, 2)).astype(np.float32)
            anchors_np[i, :n_anch] = pts
            anchor_mask_np[i, :n_anch] = True

        anchors = torch.from_numpy(anchors_np).float().to(device)
        anchor_mask = torch.from_numpy(anchor_mask_np).to(device)

    print(f"[Anchors] shape: {anchors.shape}, mask shape: {anchor_mask.shape}")
    print(f"[Anchors] valid counts: {anchor_mask.sum(dim=1).tolist()}")

    # ODE sampling
    with torch.no_grad():
        cond_ode = ConditionalVectorFieldODEVariable(model, anchors, anchor_mask, args.cfg_scale)

        if args.use_rk4:
            print("[ODE] Using RK4 integrator")
            simulator = RK4Simulator(cond_ode)
        else:
            print("[ODE] Using Euler integrator")
            simulator = EulerSimulator(cond_ode)

        x0, _ = p_simple.sample(b)
        print(f"[Sample] x0 shape: {x0.shape}")

        ts = torch.linspace(0, 1, args.sample_steps, device=device)
        ts = ts.view(1, -1, 1, 1).expand(b, -1, 1, 1)

        x_final = simulator.simulate(x0, ts)
        print(f"[Sample] x_final shape: {x_final.shape}")

        x_final_np = x_final.detach().cpu().numpy().astype(np.float32)
        anchors_np = anchors.cpu().numpy()
        anchor_mask_np = anchor_mask.cpu().numpy()

    # Save NPZ
    npz_path = os.path.join(npz_dir, "generated.npz")
    np.savez_compressed(
        npz_path,
        edge_points=x_final_np,
        anchors=anchors_np,
        anchor_mask=anchor_mask_np,
        ground_truth=ground_truth if ground_truth is not None else np.array([]),
    )
    print(f"[NPZ] Saved to: {npz_path}")

    # Visualize
    vis_path = os.path.join(vis_dir, "generated.png")
    visualize_variable_anchors(x_final_np, anchors_np, anchor_mask_np, vis_path, title_prefix="Generated")

    if ground_truth is not None:
        compare_path = os.path.join(vis_dir, "comparison.png")
        visualize_comparison_variable(x_final_np, ground_truth, anchors_np, anchor_mask_np, compare_path)

        mse = np.mean((x_final_np - ground_truth) ** 2)
        print(f"[Metric] MSE: {mse:.6f}")

    print("[Done] Evaluation complete!")


if __name__ == "__main__":
    main()
