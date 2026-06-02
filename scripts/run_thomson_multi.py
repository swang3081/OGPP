#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_thomson_multi.py

Training script for the Thomson problem, supporting multiple 3D modes:

1. 3D mode (--mode 3d):
   - Data format: xyz coordinates (already presorted, within [-1,1]^3)
   - x0: uniform random in [-1,1]^3
   - Uses LinearConditionalProbabilityPath

2. 3D MiniBatch OT mode (--mode 3d_minibatch_ot):
   - Same data format as 3D mode
   - x0-x1 pairing uses batch-level OT matching

3. 3D EqOTFM mode (--mode 3d_eqotfm):
   - Same data format as 3D mode
   - x0-x1 pairing uses two-level OT matching (point-level + batch-level)
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')  # Force a non-interactive backend to avoid Tkinter multi-threading issues
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from flow_lab.paths import LinearConditionalProbabilityPath
from flow_lab.models import UncondUniGBNTransformer, UncondUniGBNTransformer_PE
from flow_lab.utils import save_checkpoint
from flow_lab.datasets import XYZDataset, XYZMiniBatchOTDataset, XYZEqOTFMDataset
from flow_lab.trainers import MeshUncondTrainer
from flow_lab.distributions import Uniform


# ----------------- Argparser -----------------

def build_argparser():
    p = argparse.ArgumentParser(description="Thomson problem training with flow matching")

    # Mode selection
    p.add_argument("--mode", type=str, default="3d",
                   choices=["3d", "3d_minibatch_ot", "3d_eqotfm"],
                   help="Training mode: 3d, 3d_minibatch_ot, or 3d_eqotfm")

    # OT-related parameters (used by 3d_minibatch_ot and 3d_eqotfm modes)
    p.add_argument("--ot_solver", type=str, default="hungarian",
                   help="OT solver for batch-level matching (greedy/hungarian)")
    p.add_argument("--point_ot_solver", type=str, default="greedy",
                   help="OT solver for point-level matching in EqOTFM (greedy/hungarian)")

    # Data preprocessing parameters
    p.add_argument("--no_sort", action="store_true",
                   help="Randomly shuffle points within each sample (reproducible with fixed seed)")
    p.add_argument("--use_PE", action="store_true",)

    # Training parameters
    p.add_argument("--epochs", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=1000)

    # Data parameters
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to NPZ file containing Thomson data")
    p.add_argument("--n_points", type=int, default=384,
                   help="Number of points per point cloud")

    # Model parameters
    p.add_argument("--in_out_dim", type=int, default=3,
                   help="Spatial dimension (3 for 3D xyz)")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)

    # Visualization parameters
    p.add_argument("--viz_steps", type=int, default=5,
                   help="Number of time steps for visualization")
    p.add_argument("--vis_begin", action="store_true",
                   help="Visualize trajectories at the beginning of training")
    p.add_argument("--output_begin_trajectory", action="store_true",
                   help="Output trajectory as NPZ and PLY files")

    # Logging parameters
    p.add_argument("--exp_name", type=str, default="thomson_3d")
    p.add_argument("--log_path", type=str, default="log")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--ckpt_path", type=str, default="",
                   help="Resume checkpoint path")
    p.add_argument("--if_load_ckpt", action="store_true")
    p.add_argument("--only_load_model_weight", action="store_true")
    p.add_argument("--use_tb", action="store_true", default=True,
                   help="Use tensorboard logging")

    # Miscellaneous
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_batches", type=int, default=3)
    p.add_argument("--use_warmup", action="store_true")
    p.add_argument("--use_cos_decay", action="store_true")
    p.add_argument("--overwrite_lr", action="store_true")

    # DDP-related parameters
    p.add_argument("--no_ddp", action="store_true",
                   help="Force single-process mode")
    p.add_argument("--force_ddp", action="store_true",
                   help="Force DDP even with single GPU")

    return p


# ----------------- DDP helper functions -----------------

def _dist_available():
    return dist.is_available() and dist.is_initialized()


def _is_rank0():
    return (not _dist_available()) or dist.get_rank() == 0


def _maybe_init_dist(args):
    """Initialize distributed training"""
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


# ----------------- 3D XYZ visualization (for 3D mode) -----------------

def vis_begin_3d_xyz(args, dataset, path, device):
    """
    3D XYZ point cloud visualization

    Plot the trajectory from t=0 to t=1
    """
    num_rows, num_cols = 2, 2
    k = num_rows * num_cols

    # Select samples
    if dataset.num_meshes < k:
        k = dataset.num_meshes
    rng = np.random.default_rng(seed=123)
    idx_batch = rng.choice(dataset.num_meshes, size=k, replace=False)
    print(f"[VIS] Visualizing samples: {idx_batch}")

    # Compute (x0, x1)
    x0_np, x1_np = dataset.compute_batch(idx_batch=idx_batch, epoch=0, step=0)
    x0 = torch.from_numpy(x0_np).to(device)  # (k, N, 3)
    x1 = torch.from_numpy(x1_np).to(device)  # (k, N, 3)

    ts = torch.linspace(0, 1, args.viz_steps, device=device)

    begin_dir = os.path.join(args.log_path, args.exp_name, "viz_begin")
    os.makedirs(begin_dir, exist_ok=True)

    begin_trajectory_dir = os.path.join(begin_dir, "trajectory")
    if args.output_begin_trajectory:
        os.makedirs(begin_trajectory_dir, exist_ok=True)
        traj_list = []

    for tidx, t in enumerate(ts):
        tt = t.view(1, 1, 1).expand(k, 1, 1)
        xt = path.sample_conditional_path_inputx0(x0, x1, tt)  # (k, N, 3)

        # Create figure
        fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))

        for i in range(k):
            ax = fig.add_subplot(num_rows, num_cols, i + 1, projection='3d')

            # Convert to numpy
            xyz = xt[i].cpu().numpy()  # (N, 3)

            # Draw points
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                       c='blue', s=5, alpha=0.8)

            # Mark the first and last points
            ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], c='red', s=30, marker='o', label='first')
            ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], c='green', s=30, marker='^', label='last')

            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            ax.set_zlim([-1.2, 1.2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Sample {idx_batch[i]}, t={float(t):.2f}")

        plt.tight_layout()
        save_path = os.path.join(begin_dir, f"vis_t{tidx:02d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        # Save trajectory data
        if args.output_begin_trajectory:
            out = xt.detach().cpu().numpy().astype(np.float32)  # (k, N, 3)
            traj_list.append(out)

    # Save trajectory NPZ
    if args.output_begin_trajectory:
        begin_npz = os.path.join(begin_dir, "begin_traj.npz")
        traj = np.stack(traj_list, axis=0)  # (T, k, N, 3)
        np.savez_compressed(begin_npz, traj=traj,
                           ts=ts.detach().cpu().numpy(),
                           idx_batch=idx_batch)
        print(f"[VIS] Saved trajectory NPZ: {begin_npz}")

    print(f"[VIS] Saved 3D xyz visualizations to: {begin_dir}")


# ----------------- Main logic -----------------

def main(args=None):
    if args is None:
        args = build_argparser().parse_args()

    # DDP initialization
    use_ddp, local_rank = _maybe_init_dist(args)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = dist.get_world_size() if _dist_available() else 1
    rank = dist.get_rank() if _dist_available() else 0

    if _is_rank0():
        print(f"[Rank {rank}/{world_size}] device = {device}")
        print(f"[Config] mode = {args.mode}, in_out_dim = {args.in_out_dim}, n_points = {args.n_points}")

    # Create directories
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

    # Build p_simple (uniform distribution)
    p_simple = Uniform(shape=[args.n_points, args.in_out_dim], a=1.0).to(device)

    # Select path and dataset based on mode (all 3D)
    # All 3D modes use LinearConditionalProbabilityPath
    path = LinearConditionalProbabilityPath(
        p_simple=p_simple,
        p_data=p_simple,
    ).to(device)

    if _is_rank0():
        print("[Path] Using LinearConditionalProbabilityPath (3D)")

    # Select the dataset based on the specific 3D mode
    if args.mode == "3d_minibatch_ot":
        # MiniBatch OT mode
        dataset = XYZMiniBatchOTDataset(
            data_path=args.data_path,
            num_points=args.n_points,
            ot_solver=args.ot_solver,
            num_workers=args.num_workers,
        )
        if _is_rank0():
            print(f"[Dataset] Total samples: {dataset.num_meshes}")
            print(f"[Dataset] Mode: 3D MiniBatch OT (ot_solver={args.ot_solver})")

    elif args.mode == "3d_eqotfm":
        # Equivariant OT Flow Matching mode
        dataset = XYZEqOTFMDataset(
            data_path=args.data_path,
            num_points=args.n_points,
            point_ot_solver=args.point_ot_solver,
            batch_ot_solver=args.ot_solver,
            num_workers=args.num_workers,
        )
        if _is_rank0():
            print(f"[Dataset] Total samples: {dataset.num_meshes}")
            print(f"[Dataset] Mode: 3D EqOTFM (point_ot={args.point_ot_solver}, batch_ot={args.ot_solver})")

    else:
        # Plain 3D mode (no OT matching)
        dataset = XYZDataset(
            data_path=args.data_path,
            num_points=args.n_points,
        )
        if _is_rank0():
            print(f"[Dataset] Total samples: {dataset.num_meshes}")
            print(f"[Dataset] Mode: 3D (xyz, no OT)")

    # No 3D mode needs cond_vec_use_x0
    cond_vec_use_x0 = False

    # If --no_sort, apply a reproducible random shuffle to the points of each sample
    if args.no_sort and hasattr(dataset, 'points'):
        if _is_rank0():
            print("[Dataset] Shuffling points within each sample (--no_sort)...")
        shuffle_seed = 0x12345678  # fixed seed to ensure reproducibility
        rng = np.random.default_rng(seed=shuffle_seed)
        num_samples = dataset.points.shape[0]
        num_points = dataset.points.shape[1]
        for i in range(num_samples):
            perm = rng.permutation(num_points)
            dataset.points[i] = dataset.points[i][perm]
        if _is_rank0():
            print(f"[Dataset] Shuffled {num_samples} samples with seed={shuffle_seed}")

    # Visualization
    if args.vis_begin and _is_rank0():
        vis_begin_3d_xyz(args, dataset, path, device)

    if _dist_available():
        dist.barrier()

    # Build the model
    # Input: xt (B, N, 2) - (theta, phi) positions
    # Output: velocity field (B, N, 2)
    ModelCls = UncondUniGBNTransformer_PE if args.use_PE else UncondUniGBNTransformer
    model = ModelCls(
        n_points=args.n_points,
        in_dim=args.in_out_dim,      # 2
        out_dim=args.in_out_dim,     # 2
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        t_embed_dim=40,
    ).to(device)

    if _is_rank0():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Model] UncondUniGBNTransformer with {total_params/1e6:.2f}M parameters")

    # Wrap the model with DDP
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # checkpoint / resume
    start_epoch = 0
    if args.if_load_ckpt and args.ckpt_path and os.path.exists(args.ckpt_path):
        args.resume_path = args.ckpt_path
    else:
        args.resume_path = None

    # Build the Trainer
    # Use MeshUncondTrainer, which automatically uses MeshPairAsyncLoader
    trainer = MeshUncondTrainer(
        path=path,
        mesh_dataset=dataset,
        model=model,
        rotate4=False,
        zorder=False,
        output_x0=False,
        prefetch_batches=args.prefetch_batches,
        cond_vec_use_x0=cond_vec_use_x0,   # not needed in 3D mode
        cond_vec_use_x0_with_n0=False,     # not cubic
        sample_cond_use_n0=False,          # not cubic
        overwrite_lr=args.overwrite_lr,
        start_epoch=start_epoch,
        zero_t0=False,
    )

    if _is_rank0():
        print("[Trainer] MeshUncondTrainer initialized")
        print(f"[Trainer] cond_vec_use_x0={cond_vec_use_x0}, sample_cond_use_n0=False")

    # Training
    try:
        trainer.train(
            num_epochs=args.epochs,
            device=device,
            ckpt_dir=ckpt_dir,
            args=args,
        )
    except KeyboardInterrupt:
        if _is_rank0():
            print("\n[Interrupt] Training interrupted by user (Ctrl+C). Saving checkpoint...")
            ckpt_path = save_checkpoint(
                trainer.model,
                args,
                ckpt_dir,
                arch_name=f"Thomson3D_epoch{trainer.start_epoch+1:03d}",
            )
            print(f"[ckpt] Saved interrupted checkpoint to: {ckpt_path}")
        if _dist_available():
            dist.barrier()
        sys.exit(0)
    finally:
        if _is_rank0() and getattr(trainer, "writer", None) is not None:
            try:
                trainer.writer.flush()
            finally:
                trainer.writer.close()

    # DDP cleanup
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
