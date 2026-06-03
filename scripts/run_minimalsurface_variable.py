#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_minimalsurface_variable.py

Minimal Surface Conditional Generation with VARIABLE anchors (3-8).

Key features:
- Variable anchor count per sample (3-8 anchors)
- Pad-to-8 with NaN, masked attention
- Missing embedding for padded positions
- CFG (Classifier-Free Guidance) training
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from flow_lab.paths import LinearConditionalProbabilityPath
from flow_lab.models_conditional import CondPointTransformerSDPAVariable
from flow_lab.utils import save_checkpoint
from flow_lab.datasets import MinimalSurfaceVariableDataset
from flow_lab.trainers import MinimalSurfaceVariableCondTrainer
from flow_lab.distributions import Uniform


# ----------------- Argparser -----------------

def build_argparser():
    p = argparse.ArgumentParser(description="Minimal Surface Variable Anchor Training")

    # Training
    p.add_argument("--epochs", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=1000)

    # CFG
    p.add_argument("--p_drop", type=float, default=0.15,
                   help="Condition dropout probability for CFG training")

    # Data
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to variable anchor NPZ file")
    p.add_argument("--n_points", type=int, default=256,
                   help="Number of edge points per sample")
    p.add_argument("--max_anchors", type=int, default=8,
                   help="Maximum number of anchors (default: 8)")

    # Model
    p.add_argument("--in_out_dim", type=int, default=2,
                   help="Spatial dimension (2D)")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--cross_every", type=int, default=2,
                   help="Cross-attention every N self-attention layers")

    # Visualization
    p.add_argument("--viz_steps", type=int, default=5)
    p.add_argument("--vis_begin", action="store_true",
                   help="Visualize trajectories at training start")
    p.add_argument("--output_begin_trajectory", action="store_true")

    # Logging
    p.add_argument("--exp_name", type=str, default="minimal_surface_variable")
    p.add_argument("--log_path", type=str, default="log")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--ckpt_path", type=str, default="",
                   help="Resume checkpoint path")
    p.add_argument("--if_load_ckpt", action="store_true")
    p.add_argument("--only_load_model_weight", action="store_true")
    p.add_argument("--use_tb", action="store_true", default=True)

    # Other
    p.add_argument("--prefetch_batches", type=int, default=3)
    p.add_argument("--use_warmup", action="store_true")
    p.add_argument("--use_cos_decay", action="store_true")
    p.add_argument("--overwrite_lr", action="store_true")

    # DDP
    p.add_argument("--no_ddp", action="store_true")
    p.add_argument("--force_ddp", action="store_true")

    return p


# ----------------- DDP Helpers -----------------

def _dist_available():
    return dist.is_available() and dist.is_initialized()


def _is_rank0():
    return (not _dist_available()) or dist.get_rank() == 0


def _maybe_init_dist(args):
    if getattr(args, "no_ddp", False):
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return False, 0

    world_from_env = int(os.getenv("WORLD_SIZE", "1"))
    must_init = bool(getattr(args, "force_ddp", False))
    use_ddp = must_init or (world_from_env > 1)

    local_rank = int(os.getenv("LOCAL_RANK", "0")) if use_ddp else 0

    if use_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    return use_ddp, local_rank


# ----------------- Visualization -----------------

def visualize_variable_anchors(edge_points, anchors, anchor_mask, save_dir, name_prefix, sample_indices=None):
    """
    Visualize minimal surface with variable anchors.

    Args:
        edge_points: (B, N, 2) numpy array
        anchors: (B, max_anchors, 2) numpy array - NaN padded
        anchor_mask: (B, max_anchors) numpy array - True for valid
        save_dir: output directory
        name_prefix: filename prefix
    """
    if isinstance(edge_points, torch.Tensor):
        edge_points = edge_points.cpu().numpy()
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.cpu().numpy()
    if isinstance(anchor_mask, torch.Tensor):
        anchor_mask = anchor_mask.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    if sample_indices is None:
        np.random.seed(42)
        sample_indices = np.random.choice(edge_points.shape[0], min(4, edge_points.shape[0]), replace=False)

    for ax, idx in zip(axes, sample_indices):
        pts = edge_points[idx]
        ax.scatter(pts[:, 0], pts[:, 1], c='blue', s=3, alpha=0.6, label='Edge points')

        # Valid anchors only
        mask = anchor_mask[idx]
        anch = anchors[idx][mask]
        n_valid = mask.sum()

        ax.scatter(anch[:, 0], anch[:, 1], c='red', s=100, marker='^',
                   edgecolors='black', linewidths=1, label=f'Anchors ({n_valid})', zorder=10)

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'Sample {idx} ({n_valid} anchors)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_fig = os.path.join(save_dir, f"{name_prefix}.png")
    plt.savefig(output_fig, dpi=150)
    plt.close()
    print(f"[VIS] Saved to: {output_fig}")


def vis_begin_variable(args, dataset, path, device):
    """Visualize variable anchor trajectories at training start."""
    num_samples = 4

    if dataset.num_meshes < num_samples:
        num_samples = dataset.num_meshes
    rng = np.random.default_rng(seed=123)
    idx_batch = rng.choice(dataset.num_meshes, size=num_samples, replace=False)
    print(f"[VIS] Visualizing samples: {idx_batch}")

    x0_np, x1_np, anchors_np, anchor_mask_np = dataset.compute_batch(idx_batch=idx_batch, epoch=0, step=0)
    x0 = torch.from_numpy(x0_np).to(device)
    x1 = torch.from_numpy(x1_np).to(device)

    ts = torch.linspace(0, 1, args.viz_steps, device=device)

    begin_dir = os.path.join(args.log_path, args.exp_name, "viz_begin")
    os.makedirs(begin_dir, exist_ok=True)

    begin_trajectory_dir = os.path.join(begin_dir, "trajectory")
    if args.output_begin_trajectory:
        os.makedirs(begin_trajectory_dir, exist_ok=True)
        traj_list = []

    for tidx, t in enumerate(ts):
        tt = t.view(1, 1, 1).expand(num_samples, 1, 1)
        xt = path.sample_conditional_path_inputx0(x0, x1, tt)

        visualize_variable_anchors(
            xt.cpu().numpy(),
            anchors_np,
            anchor_mask_np,
            begin_dir,
            f"vis_t{tidx:02d}_t{float(t):.2f}",
            sample_indices=list(range(num_samples))
        )

        if args.output_begin_trajectory:
            out = xt.detach().cpu().numpy().astype(np.float32)
            traj_list.append(out)

    if args.output_begin_trajectory:
        begin_npz = os.path.join(begin_dir, "begin_traj.npz")
        traj = np.stack(traj_list, axis=0)
        np.savez_compressed(begin_npz, traj=traj,
                           ts=ts.detach().cpu().numpy(),
                           idx_batch=idx_batch,
                           anchors=anchors_np,
                           anchor_mask=anchor_mask_np)
        print(f"[VIS] Saved trajectory NPZ: {begin_npz}")

    print(f"[VIS] Saved visualizations to: {begin_dir}")


# ----------------- Main -----------------

def main(args=None):
    if args is None:
        args = build_argparser().parse_args()

    use_ddp, local_rank = _maybe_init_dist(args)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = dist.get_world_size() if _dist_available() else 1
    rank = dist.get_rank() if _dist_available() else 0

    if _is_rank0():
        print(f"[Rank {rank}/{world_size}] device = {device}")
        print(f"[Config] n_points={args.n_points}, max_anchors={args.max_anchors}, p_drop={args.p_drop}")

    exp_dir = os.path.join(args.log_path, args.exp_name)
    if _is_rank0():
        os.makedirs(exp_dir, exist_ok=True)
        cmd_txt_path = os.path.join(exp_dir, "cmd.txt")
        try:
            with open(cmd_txt_path, "a", encoding="utf-8") as f:
                f.write(" ".join(sys.argv) + "\n")
        except Exception as e:
            print(f"[warn] Failed to write cmd.txt: {e}")

    if _dist_available():
        dist.barrier()

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if _is_rank0():
        os.makedirs(ckpt_dir, exist_ok=True)

    if _dist_available():
        dist.barrier()

    # Uniform distribution for x0
    p_simple = Uniform(shape=[args.n_points, args.in_out_dim], a=1.0).to(device)

    path = LinearConditionalProbabilityPath(
        p_simple=p_simple,
        p_data=p_simple,
    ).to(device)

    if _is_rank0():
        print("[Path] Using LinearConditionalProbabilityPath (2D)")

    # Load variable anchor dataset
    dataset = MinimalSurfaceVariableDataset(data_path=args.data_path, max_anchors=args.max_anchors)
    if _is_rank0():
        print(f"[Dataset] Loaded {dataset.num_meshes} samples")
        print(f"[Dataset] n_points={dataset.num_points}, max_anchors={dataset.max_anchors}")

    # Visualization
    if args.vis_begin and _is_rank0():
        vis_begin_variable(args, dataset, path, device)

    if _dist_available():
        dist.barrier()

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

    if _is_rank0():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Model] CondPointTransformerSDPAVariable with {total_params/1e6:.2f}M parameters")

    # DDP wrapper
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # Resume checkpoint
    start_epoch = 0
    if args.if_load_ckpt and args.ckpt_path and os.path.exists(args.ckpt_path):
        args.resume_path = args.ckpt_path
    else:
        args.resume_path = None

    # Trainer
    trainer = MinimalSurfaceVariableCondTrainer(
        path=path,
        mesh_dataset=dataset,
        model=model,
        p_drop=args.p_drop,
        prefetch_batches=args.prefetch_batches,
        overwrite_lr=args.overwrite_lr,
        start_epoch=start_epoch,
    )

    if _is_rank0():
        print(f"[Trainer] MinimalSurfaceVariableCondTrainer (p_drop={args.p_drop})")

    # Train
    try:
        trainer.train(
            num_epochs=args.epochs,
            device=device,
            ckpt_dir=ckpt_dir,
            args=args,
        )
    except KeyboardInterrupt:
        if _is_rank0():
            print("\n[Interrupt] Saving checkpoint...")
            ckpt_path = save_checkpoint(
                trainer.model,
                args,
                ckpt_dir,
                arch_name=f"MinimalSurfaceVariable_epoch{trainer.start_epoch+1:03d}",
            )
            print(f"[ckpt] Saved to: {ckpt_path}")
        if _dist_available():
            dist.barrier()
        sys.exit(0)
    finally:
        if _is_rank0() and getattr(trainer, "writer", None) is not None:
            try:
                trainer.writer.flush()
            finally:
                trainer.writer.close()

    # Cleanup
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()