#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_mesh_uniGBN.py

- Uses a mesh dataset (.ply); for each batch:
    * x0 ~ Uniform([0,1]^3) (sampled inside MeshNearPointPairDataset)
    * x1 = the nearest point on the corresponding mesh surface
- Probability path: x_t = (1 - t) * x0 + t * x1
- Trainer: MeshUncondTrainer, which uses an async loader to prefetch x0/x1 on the CPU,
  while the GPU only runs the model and backward pass.

Note:
- The initial visualization also switches to synchronously computing a small batch of x0/x1
  from MeshNearPointPairDataset and then interpolating.
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

is_headless = (matplotlib.get_backend().lower() == "agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from flow_lab.paths import LinearConditionalProbabilityPath, QuadraticHermiteNormalConditionalProbabilityPath, CubicHermiteNormalConditionalProbabilityPath, GenVPConditionalProbabilityPath
from flow_lab.models import UncondUniGBNTransformer, UncondUniGBNTransformer_PE
from flow_lab.utils import save_checkpoint, render_point_images
from flow_lab.io import *
from flow_lab.datasets import MeshNearPointPairDataset, MeshPoissonSphereDataset, MeshSortDataset, MeshOTDataset, MeshEqOTFMDataset, MeshMiniBatchOTDataset
from flow_lab.trainers import MeshUncondTrainer
from flow_lab.distributions import Uniform

# ----------------- Argparser -----------------

def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--epochs", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=1000)

    p.add_argument("--in_out_dim", type=int, default=3)
    p.add_argument("--n_points", type=int, default=2048)
    p.add_argument("--mode", type=str, default="nearest")

    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)

    p.add_argument("--viz_steps", type=int, default=5)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--exp_name", type=str, default="mesh_uniGBN_uncond_linear")
    p.add_argument("--vis_begin", action="store_true")
    p.add_argument("--output_begin_trajectory", action="store_true")
    p.add_argument("--ot_solver", type=str, default="greedy", help="choose from hungarian (O(N^3)), or greedy (O(N^2))")

    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--log_hist_every", type=int, default=100)
    p.add_argument("--fid_total", type=int, default=10000)
    p.add_argument("--log_path", type=str, default="log", help="log root dir")
    p.add_argument("--ckpt_path", type=str, default="", help="resume checkpoint path")
    p.add_argument("--if_load_ckpt", action="store_true")
    p.add_argument("--only_load_model_weight", action="store_true")
    p.add_argument("--use_warmup", action="store_true")
    p.add_argument("--use_cos_decay", action="store_true")
    p.add_argument("--use_magnitude", action="store_true")
    p.add_argument("--overwrite_lr", action="store_true")

    p.add_argument("--use_multi", action="store_true", default=True)

    p.add_argument("--mesh_root", type=str, required=True,
                   help="Root folder containing .ply meshes (recursively searched).")

    p.add_argument("--hermite_degree", type=str, default=None)
    p.add_argument("--output_x0", action="store_true",
                   help="whether output x0 as npz every epoch.")
    p.add_argument("--use_normal", action="store_true",)
    p.add_argument("--use_poisson", action="store_true",)
    p.add_argument("--use_PE", action="store_true",)
    p.add_argument("--use_pvcnn", action="store_true",)

    p.add_argument("--cond_vec_use_x0", action="store_true",)
    p.add_argument("--cond_vec_use_x0_with_n0", action="store_true",)
    p.add_argument("--use_normal_in_sort", action="store_true",)
    p.add_argument("--sample_cond_use_n0", action="store_true",)
    p.add_argument("--use_gen_vp", action="store_true",)
    p.add_argument("--use_sphere", action="store_true",)
    p.add_argument("--use_shell", action="store_true",)
    p.add_argument("--zero_t0", action="store_true",)
    p.add_argument("--tangent_scale_mode", type=str, default="unit",
                   choices=["unit", "original", "chord", "arc_length"],
                   help="Endpoint velocity scaling mode (only for is_single_mesh + quadratic): "
                        "unit=unit vector, original=raw normal, chord=chord-length scaling, arc_length=arc-length-estimate scaling")
    p.add_argument("--lambda_orient", type=float, default=0.2,
                   help="Misalignment penalty coefficient for arc-length estimation (only in arc_length mode)")
    p.add_argument("--dataset_is_pcd", action="store_true",)
    p.add_argument("--no_sort", action="store_true",
                   help="Skip Hilbert sorting for x1 (directly use subsampled points)")
    p.add_argument("--dataset_is_mesh_npz", action="store_true",)
    p.add_argument("--is_single_mesh", action="store_true",
                   help="Single mesh mode: mesh_root points to a single mesh file")
    p.add_argument("--is_single_pcd", action="store_true",
                   help="Single point cloud mode: mesh_root points to a single .ply file with xyz+rgb")
    p.add_argument("--linear_6d", action="store_true",
                   help="6D linear flow matching: x0~Uniform([-1,1]^6), x1=xyz+normals (Hilbert-6D sorted). "
                        "Forces in_out_dim=6, use_normal=True, use_normal_in_sort=True.")

    p.add_argument("--num_workers", type=int, default=8,)
    p.add_argument("--prefetch_batches", type=int, default=3,)

    # Normalization method: choose one of three (mutually exclusive)
    norm_group = p.add_mutually_exclusive_group()
    norm_group.add_argument(
        "--normalize_globally", action="store_true",
        help="Use global normalization (LION style: global mean + global std)"
    )
    norm_group.add_argument(
        "--recenter_per_shape", action="store_true",
        help="Use per-shape recenter (LION default: bbox center + bbox half-extent)"
    )
    norm_group.add_argument(
        "--normalize_per_shape_maxabs", action="store_true",
        help="Use per-shape max-abs normalization (default)"
    )

    # DDP-related parameters
    p.add_argument("--no_ddp", action="store_true", help="Force single-process mode (useful in notebooks)")
    p.add_argument("--force_ddp", action="store_true", help="Even with 1 GPU, still initialize process group to test DDP plumbing.")

    return p


# ----------------- DDP helper functions -----------------

def _dist_available():
    return dist.is_available() and dist.is_initialized()

def _is_rank0():
    return (not _dist_available()) or dist.get_rank() == 0

def _maybe_init_dist(args):
    """
    Rules:
    - If --no_ddp: force single-process mode
    - If --force_ddp: always init_process_group, regardless of whether WORLD_SIZE is 1
    - Otherwise (not --force_ddp): normal logic, only use DDP when WORLD_SIZE>1, else single-node single-process
    """
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


# ----------------- Small utility: 3D -> 2D projection -----------------

def project_points_for_vis(x: torch.Tensor, plane: str = "xy") -> torch.Tensor:
    """
    x: (B, N, 3)
    Returns (B, N, 2), projected onto the specified plane.
    """
    if x.shape[-1] != 3:
        raise ValueError(f"Expect last dim=3 for 3D, got {x.shape}")

    if plane == "xy":
        return x[..., :2]
    elif plane == "xz":
        return x[..., (0, 2)]
    elif plane == "yz":
        return x[..., 1:]
    else:
        raise ValueError(f"Unknown plane: {plane}")


# ----------------- Main logic -----------------

def main(args=None):
    if args is None:  # CLI entry point
        args = build_argparser().parse_args()

    # ---- DDP initialization ----
    use_ddp, local_rank = _maybe_init_dist(args)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = dist.get_world_size() if _dist_available() else 1
    rank = dist.get_rank() if _dist_available() else 0

    if _is_rank0():
        print(f"[Rank {rank}/{world_size}] device = {device}")

    exp_dir = os.path.join(args.log_path, args.exp_name)
    if _is_rank0():
        os.makedirs(exp_dir, exist_ok=True)
        cmd_txt_path = os.path.join(exp_dir, "cmd.txt")
        try:
            with open(cmd_txt_path, "a", encoding="utf-8") as f:
                f.write(" ".join(sys.argv) + "\n")
        except Exception as e:
            print(f"[warn] Failed to write cmd.txt: {e}")

    # Wait for rank0 to create the directories
    if _dist_available():
        dist.barrier()

    # The original ckpt_dir definition now uses exp_dir
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if _is_rank0():
        os.makedirs(ckpt_dir, exist_ok=True)

    if _dist_available():
        dist.barrier()

    # ---- Automatic settings for linear_6d mode ----
    if args.linear_6d:
        args.in_out_dim = 6
        args.use_normal = True
        args.use_normal_in_sort = True
        if _is_rank0():
            print("[linear_6d] Forcing in_out_dim=6, use_normal=True, use_normal_in_sort=True")

    # ---- Build the base distribution p_simple (training no longer uses it to sample x0, but path needs an object) ----
    # This is only here to be compatible with the construction of LinearConditionalProbabilityPath.
    p_simple = Uniform(shape=[args.n_points, args.in_out_dim], a=math.sqrt(3)).to(device)

    args.cond_vec_use_x0 = True
    if _is_rank0():
        print("Automatically setting cond_vec_use_x0 due to mesh pair")

    if "hermite" not in args.mode and (args.hermite_degree == "quadratic" or args.hermite_degree == "cubic"):
        raise RuntimeError("Impossible combination: hermite not in mode but hermite curve is not linear")


    # p_data is not actually used here; just pass the same placeholder.
    if "hermite" not in args.mode or args.hermite_degree == "linear" or args.hermite_degree is None:
        if _is_rank0():
            print("Using Path: Linear")
        if not args.linear_6d:
            args.use_normal = False
            if _is_rank0():
                print("Automatically setting use_normal false due to linear path")

        if not args.use_gen_vp:
            path = LinearConditionalProbabilityPath(
                p_simple=p_simple,
                p_data=p_simple,
            ).to(device)
        else:
            path = GenVPConditionalProbabilityPath(
                p_simple=p_simple,
                p_data=p_simple,
            ).to(device)

    elif args.hermite_degree == "cubic":
        if _is_rank0():
            print("Using Path: Cubic Hermite Normal")
        args.sample_cond_use_n0 = True
        if _is_rank0():
            print("Automatically setting sample_cond_use_n0 true due to cubic Hermite")
        args.cond_vec_use_x0_with_n0 = True
        if _is_rank0():
            print("Automatically setting cond_vec_use_x0_with_n0 true due to cubic Hermite")
        args.use_normal = True
        if _is_rank0():
            print("Automatically setting use_normal true due to Hermite curve")

        args.cond_vec_use_x0 = False
        if _is_rank0():
            print("Automatically setting cond_vec_use_x0 false due to cubic Hermite")

        if args.zero_t0:
            if _is_rank0():
                print("Setting n0 equals zero")
        else:
            if _is_rank0():
                print("n0 is now determined by finding nearest mesh point")

        path = CubicHermiteNormalConditionalProbabilityPath(
            p_simple=p_simple,
            p_data=p_simple,
        ).to(device)
    elif args.hermite_degree == "quadratic":
        if _is_rank0():
            print("Using Path: Quadratic Hermite Normal")
        args.use_normal = True
        if _is_rank0():
            print("Automatically setting use_normal true due to Hermite curve")

        if args.zero_t0:
            raise RuntimeError("Quadratic path should not have zero_t0 = True")

        # is_single_pcd: normalize_tangent=False (preserve r information)
        # is_single_mesh: tangent_scale_mode controls arc-length scaling
        # otherwise: normalize_tangent=True, tangent_scale_mode="unit" (original behavior)
        if args.is_single_pcd:
            path = QuadraticHermiteNormalConditionalProbabilityPath(
                p_simple=p_simple,
                p_data=p_simple,
                normalize_tangent=False  # original logic: preserve magnitude information
            ).to(device)
        else:
            scale_mode = args.tangent_scale_mode if args.is_single_mesh else "unit"
            path = QuadraticHermiteNormalConditionalProbabilityPath(
                p_simple=p_simple,
                p_data=p_simple,
                normalize_tangent=True,
                tangent_scale_mode=scale_mode,
                lambda_orient=args.lambda_orient
            ).to(device)
            if args.is_single_mesh and scale_mode != "unit":
                if _is_rank0():
                    print(f"[is_single_mesh] Using tangent_scale_mode={scale_mode}, lambda_orient={args.lambda_orient}")
    else:
        raise RuntimeError("No path can be found that can be used.")
        


    # ---- Build the Mesh dataset (responsible for x0/x1) ----
    if args.mode == "nearest":
        mesh_dataset = MeshNearPointPairDataset(
            mesh_root=args.mesh_root,
            num_points=args.n_points,
            use_multi=args.use_multi,
            num_workers=args.num_workers,
        )
    elif args.mode == "poisson_sphere":

        mesh_dataset = MeshPoissonSphereDataset(
            mesh_root=args.mesh_root,
            num_points=args.n_points,
            use_multi=args.use_multi,
            num_workers=args.num_workers,
        )
    elif args.mode == "poisson_sort" or args.mode == "poisson_sort_hermite" or args.mode == "random_sort_hermite" or args.mode == "random_sort":

        mesh_dataset = MeshSortDataset(
            mesh_root=args.mesh_root,
            num_points=2048,
            use_multi=args.use_multi,
            num_workers=args.num_workers,
            use_normal=args.use_normal,
            use_normal_in_sort=args.use_normal_in_sort,
            use_poisson=args.use_poisson,
            use_sphere=args.use_sphere,
            use_shell=args.use_shell,
            hermite_degree=args.hermite_degree,
            zero_t0=args.zero_t0,
            dataset_is_pcd=args.dataset_is_pcd,
            normalize_globally=args.normalize_globally,
            recenter_per_shape=args.recenter_per_shape,
            normalize_per_shape_maxabs=args.normalize_per_shape_maxabs,
            no_sort=args.no_sort,
            dataset_is_mesh_npz=args.dataset_is_mesh_npz,
            is_single_mesh=args.is_single_mesh,
            is_single_pcd=args.is_single_pcd,
            linear_6d=args.linear_6d,
        )
    elif args.mode == "poisson_OT_hermite" or args.mode == "random_OT_hermite" or args.mode == "random_OT" or args.mode == "poisson_OT":
        mesh_dataset = MeshOTDataset(
            mesh_root=args.mesh_root,
            num_points=2048,
            use_multi=args.use_multi,
            num_workers=args.num_workers,
            use_normal=args.use_normal,
            use_poisson = args.use_poisson,
            use_sphere=args.use_sphere,
            use_shell=args.use_shell,
            ot_solver=args.ot_solver,
            hermite_degree=args.hermite_degree,
            lambda_orient=0.2,
            zero_t0=args.zero_t0,
            dataset_is_pcd=args.dataset_is_pcd,
            normalize_globally=args.normalize_globally,
            recenter_per_shape=args.recenter_per_shape,
            normalize_per_shape_maxabs=args.normalize_per_shape_maxabs,
            dataset_is_mesh_npz=args.dataset_is_mesh_npz
        )
    elif args.mode == "EqFM":
        mesh_dataset = MeshEqOTFMDataset(
            mesh_root=args.mesh_root,
            num_points=2048,
            use_multi=args.use_multi,
            num_workers=args.num_workers,
            use_sphere=args.use_sphere,
            use_shell=args.use_shell,
            normalize_globally=args.normalize_globally,
            recenter_per_shape=args.recenter_per_shape,
            normalize_per_shape_maxabs=args.normalize_per_shape_maxabs,
        )
    elif args.mode == "MBOT":
        mesh_dataset = MeshMiniBatchOTDataset(
            mesh_root=args.mesh_root,
            num_points=2048,
            use_multi=args.use_multi,
            num_workers=args.num_workers,
            use_sphere=args.use_sphere,
            use_shell=args.use_shell,
            normalize_globally=args.normalize_globally,
            recenter_per_shape=args.recenter_per_shape,
            normalize_per_shape_maxabs=args.normalize_per_shape_maxabs,
            ot_solver=args.ot_solver,
        )
    else:
        raise RuntimeError("Mode not supported!")

    if _is_rank0():
        print("[Mesh] Total number of meshes:", mesh_dataset.num_meshes)

    # ---- Visualization: now synchronously sample a small batch of x0/x1 from MeshNearPointPairDataset ----
    # Only run the visualization on rank0
    if args.vis_begin and _is_rank0():
        num_rows, num_cols = 4, 4
        k = num_rows * num_cols

        # Single mesh / single pcd mode: generate all-zero indices
        if getattr(args, 'is_single_mesh', False) or getattr(args, 'is_single_pcd', False):
            idx_batch = np.zeros(k, dtype=np.int64)
        else:
            if mesh_dataset.num_meshes < k:
                k = mesh_dataset.num_meshes
            rng = np.random.default_rng(seed=123)
            all_idx = np.arange(mesh_dataset.num_meshes)
            idx_batch = rng.choice(all_idx, size=k, replace=False)
        print(idx_batch)

        # Synchronously compute a batch of (x0, x1); epoch=0, step=0 are only used for the seed
        x0_np, x1_np = mesh_dataset.compute_batch(
            idx_batch=idx_batch,
            epoch=0,
            step=0,
        )
        x0 = torch.from_numpy(x0_np).to(device=device)
        x1 = torch.from_numpy(x1_np).to(device=device)

        fig, axes = plt.subplots(1, args.viz_steps, figsize=(4 * args.viz_steps, 4))
        ts = torch.linspace(0, 1, args.viz_steps, device=device)

        begin_dir = os.path.join(args.log_path, args.exp_name, "viz_begin")
        os.makedirs(begin_dir, exist_ok=True)

        begin_trajectory_dir = os.path.join(args.log_path, args.exp_name, "viz_begin", "trajectory")
        if args.output_begin_trajectory:
            os.makedirs(begin_trajectory_dir, exist_ok=True)

            with open(os.path.join(begin_trajectory_dir, "meta.txt"), "w") as f:
                f.write(f"idx_batch = {idx_batch.tolist()}\n")
                f.write(f"viz_steps = {args.viz_steps}\n")
                f.write(f"use_normal = {bool(args.use_normal)}\n")
                f.write(f"mode = {args.mode}\n")
                f.write(f"hermite_degree = {getattr(args, 'hermite_degree', None)}\n")

            traj_list = []


        for tidx, t in enumerate(ts):
            tt = t.view(1, 1, 1).expand(k, 1, 1)
            if not args.sample_cond_use_n0:
                xt = path.sample_conditional_path_inputx0(x0, x1, tt)  # (k, N, 3)
            else:
                if args.zero_t0:
                    xt = path.sample_conditional_path_inputx0_zeron0(x0, x1, tt)
                else:
                    xt = path.sample_conditional_path_inputx0_withn0(x0, x1, tt)  # (k, N, 3)
            
            if args.linear_6d:
                # linear_6d: xt is already 6D (xyz + normals interpolated)
                vis_6d_res = xt
                print(vis_6d_res.shape)
            elif "hermite" in args.mode:
                if args.sample_cond_use_n0: # cubic
                    if not args.zero_t0:
                        nt = path.conditional_vector_field_inputx0_withn0(x0, x1, tt)
                    else:
                        nt = path.conditional_vector_field_inputx0_zeron0(x0, x1, tt)
                    vis_6d_res = torch.cat([xt, nt], dim=-1)
                else:
                    nt = path.conditional_vector_field_inputx0(x0, x1, tt)
                    vis_6d_res = torch.cat([xt, nt], dim=-1)


            if args.output_begin_trajectory:
                if args.use_normal and (vis_6d_res is not None) and (vis_6d_res.shape[-1] == 6):
                    out = vis_6d_res.detach().cpu().numpy().astype(np.float32)  # (k,N,6)
                else:
                    out = xt.detach().cpu().numpy().astype(np.float32)          # (k,N,3)

                traj_list.append(out)

                # Write PLY: one subdirectory per sample, one ply per step
                for bi in range(k):
                    sample_dir = os.path.join(begin_trajectory_dir, f"sample_{bi:02d}_mesh_{int(idx_batch[bi])}")
                    ply_path = os.path.join(sample_dir, f"step_{tidx:03d}_t{float(t):.4f}.ply")

                    if out.shape[-1] == 6:
                        xyz = out[bi, :, 0:3]
                        vec = out[bi, :, 3:6]
                        write_ply_points(ply_path, xyz, vec=vec)
                    else:
                        xyz = out[bi, :, 0:3]
                        write_ply_points(ply_path, xyz, vec=None)
                continue

            imgs = render_point_images(
                xt if not args.use_normal else vis_6d_res,
                img_size=args.img_size,
                point_radius=getattr(args, "point_radius", 2),
                in_out_dim=args.in_out_dim,
                channels=3,
                background=1.0,
                point_value=0.0,
                antialias=False,
                # color_normal=True
            )

            grid = make_grid(imgs, nrow=num_cols)
            axes[tidx].imshow(grid.permute(1, 2, 0).cpu())
            axes[tidx].axis("off")

        if args.output_begin_trajectory:
            begin_npz = os.path.join(begin_dir, "begin_traj.npz")
            traj = np.stack(traj_list, axis=0)  # (T,k,N,dim)
            np.savez_compressed(begin_npz, traj=traj, ts=ts.detach().cpu().numpy(), idx_batch=idx_batch)
            print(f"[VIS] Saved begin trajectory PLYs + NPZ to: {begin_dir}")
            print(f"[VIS] NPZ: {begin_npz}")

        else:
            plt.tight_layout()
            save_path = os.path.join(begin_dir, "vis_begin.png")
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except OSError:
                    pass
            plt.savefig(save_path, dpi=200)
            print(f"[VIS] Saved visualization to: {save_path}")
            if not is_headless:
                plt.show()
            plt.close(fig)

    out_dim = 1 if args.use_magnitude else args.in_out_dim
    if _is_rank0():
        print("Out dim:", out_dim)
    # ---- Build the model ----
    if args.use_pvcnn:
        # Lazy import PVCNN models only when needed (requires CUDA compilation)
        from flow_lab.models import UncondPVCNN, UncondPVCNN_PosEmbed
        # PVCNN models: use_PE=True -> UncondPVCNN (no positional embedding, permutation equivariant)
        #              use_PE=False -> UncondPVCNN_PosEmbed (with positional embedding)
        ModelCls = UncondPVCNN if args.use_PE else UncondPVCNN_PosEmbed
        model = ModelCls(
            n_points=args.n_points,
            # embed_dim=args.embed_dim,
        ).to(device)
    else:
        ModelCls = UncondUniGBNTransformer_PE if args.use_PE else UncondUniGBNTransformer
        model = ModelCls(
            n_points=args.n_points,
            in_dim=args.in_out_dim,
            out_dim=out_dim,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            t_embed_dim=40,
        ).to(device)

    # ---- Wrap the model with DDP ----
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # ---- checkpoint / resume ----
    start_epoch = 0
    if args.if_load_ckpt and args.ckpt_path and os.path.exists(args.ckpt_path):
        args.resume_path = args.ckpt_path
    else:
        args.resume_path = None

    trainer = MeshUncondTrainer(
        path=path,
        mesh_dataset=mesh_dataset,
        model=model,
        rotate4=False,
        zorder=False,
        output_x0 = args.output_x0,
        prefetch_batches=args.prefetch_batches,
        cond_vec_use_x0 = args.cond_vec_use_x0,
        cond_vec_use_x0_with_n0 = args.cond_vec_use_x0_with_n0,
        sample_cond_use_n0 = args.sample_cond_use_n0,
        overwrite_lr = args.overwrite_lr,
        start_epoch=start_epoch,
        zero_t0=args.zero_t0
    )

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
                arch_name=f"UncondUniGBNTransformer_epoch{trainer.start_epoch+1:03d}",
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

    # ---- DDP cleanup ----
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
