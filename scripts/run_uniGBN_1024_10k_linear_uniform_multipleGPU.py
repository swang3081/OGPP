# scripts/train_main.py
import os, sys, argparse, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ----- imports from your project -----
from flow_lab.paths import LinearConditionalProbabilityPath
from flow_lab.datasets import UniGBNSampler
from flow_lab.dynamics import EulerSimulator, VectorFieldODE
from flow_lab.trainers import UniGBNUnconditionalTrainer
from flow_lab.models import *
from flow_lab.utils import save_checkpoint, render_point_images, extract_epoch
from flow_lab.voronoi import reconstruct_voronoi_images, export_voronoi_samples_for_fid
from flow_lab.distributions import JitterHilbertGridSample, Uniform

import torch.distributed as dist

# =============== Utilities ===============

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=1000)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)

    p.add_argument("--viz_steps", type=int, default=5)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--exp_name", type=str, default="uniGBN_uncond_linear_torus")

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--log_hist_every", type=int, default=100)
    p.add_argument("--fid_total", type=int, default=10000)
    p.add_argument("--n_points", type=int, default=1024)
    p.add_argument("--img_size", type=int, default=256)

    p.add_argument("--data_aug_rotate", action="store_true")
    p.add_argument("--random_shuffle_z", action="store_true")
    p.add_argument("--data_zorder", action="store_true")
    p.add_argument("--jitter_x0", action="store_true")
    p.add_argument("--if_load_ckpt", action="store_true")
    p.add_argument("--only_load_model_weight", action="store_true", help="whether only load model weight")


    p.add_argument("--data_path", type=str, default="data/1024_10k_original_sorted")
    p.add_argument("--log_path", type=str, default="log")
    p.add_argument("--ckpt_path", type=str, default="checkpoints/voronoi_uncond/UncondVoronoiTransformer_epoch651_e20000_1759864859.pt")

    p.add_argument("--use_warmup", action="store_true")
    p.add_argument("--use_cos_decay", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--load_schedule", action="store_true", help="load schedule")

    # Added: notebook switch to force single-process mode
    p.add_argument("--no_ddp", action="store_true", help="Force single-process mode (useful in notebooks)")
    p.add_argument("--force_ddp", action="store_true",
                help="Even with 1 GPU, still initialize process group to test DDP plumbing.")

    return p

def _dist_available():
    return dist.is_available() and dist.is_initialized()

def _is_rank0():
    return (not _dist_available()) or dist.get_rank() == 0

def _maybe_init_dist(args):
    """
    Rules:
    - If --force_ddp: always init_process_group regardless of whether WORLD_SIZE is 1 (handy for testing the flow on a single GPU)
    - Otherwise (not --force_ddp): normal logic, only use DDP when WORLD_SIZE>1, else single-node single-process
    - set_device is only called when CUDA is available
    """
    world_from_env = int(os.getenv("WORLD_SIZE", "1"))
    must_init = bool(getattr(args, "force_ddp", False))
    use_ddp = must_init or (world_from_env > 1)

    local_rank = int(os.getenv("LOCAL_RANK", "0")) if use_ddp else 0

    if use_ddp:
        # torchrun injects RANK/WORLD_SIZE/LOCAL_RANK/MASTER_*; we consistently use env://
        dist.init_process_group(backend="nccl", init_method="env://", timeout=torch.distributed.constants.default_pg_timeout)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    return use_ddp, local_rank

# =============== Main ===============

def main(args=None):
    # 1) Parse arguments: do not parse the CLI when a Notebook passes an object in
    if args is None:
        args = build_argparser().parse_args()

    use_ddp, local_rank = _maybe_init_dist(args)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = dist.get_world_size() if _dist_available() else 1
    rank = dist.get_rank() if _dist_available() else 0

    ckpt_dir = os.path.join(args.log_path, args.exp_name, "checkpoints")
    if _is_rank0():
        print(f"[Rank {rank}/{world_size}] device = {device}")
        print(f"[Exp] {args.exp_name}")

    # 2) Data distribution
    if args.jitter_x0:
        p_simple = JitterHilbertGridSample(grid_size=32, jitter="uniform", periodic=False, seed=1234).to(device)
    else:
        p_simple = Uniform(shape=[args.n_points, 2]).to(device)

    sampler = UniGBNSampler(
        data_dir=args.data_path,
        random_shuffle=args.random_shuffle_z,
        rotate4 = args.data_aug_rotate,
        preload=True
    ).to(device)

    if _is_rank0():
        print("Total number of samples:", len(sampler))

    path = LinearConditionalProbabilityPath(
        p_simple=p_simple,
        p_data=sampler
    ).to(device)

    # 3) Pre-training visualization (rank0 only; displayed inline in single-process Notebook runs)
    if _is_rank0():
        num_rows, num_cols = 2, 2
        k = num_rows * num_cols
        z, _ = path.p_data.get_batch(range(k))
        fig, axes = plt.subplots(1, args.viz_steps, figsize=(4 * args.viz_steps, 4))
        if args.viz_steps == 1:
            axes = [axes]
        ts = torch.linspace(0, 1, args.viz_steps, device=device)
        for tidx, t in enumerate(ts):
            tt = t.view(1, 1, 1).expand(k, 1, 1)
            xt = path.sample_conditional_path(z, tt)
            imgs = render_point_images(
                xt, img_size=args.img_size,
                point_radius=getattr(args, "point_radius", 2),
                channels=3, background=1.0, point_value=0.0, antialias=False
            )
            grid = make_grid(imgs, nrow=num_cols)
            axes[tidx].imshow(grid.permute(1, 2, 0).cpu())
            axes[tidx].axis("off")
        viz_dir = os.path.join(args.log_path, args.exp_name, "init_viz")
        os.makedirs(viz_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "viz_before_train.png"))
        plt.close(fig)

    # 4) Model
    model = UncondUniGBNTransformer(
        n_points=args.n_points, in_dim=2, out_dim=2,
        embed_dim=args.embed_dim, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, t_embed_dim=40
    ).to(device)

    start_epoch = 0
    if args.if_load_ckpt and args.ckpt_path and os.path.exists(args.ckpt_path):
        args.resume_path = args.ckpt_path
    else:
        args.resume_path = None

    # 5) Optional DDP wrapping
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # 6) Training
    trainer = UniGBNUnconditionalTrainer(
        path=path, model=model,
        rotate4=args.data_aug_rotate,
        zorder=args.data_zorder,
        start_epoch=start_epoch
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
            print("\n[Interrupt] KeyboardInterrupt: saving checkpoint...")
            ckpt_path = save_checkpoint(
                model=trainer.model,               # the function already handles .module internally
                args=args,
                save_dir=ckpt_dir,
                arch_name="UncondUniGBNTransformer",
                optimizer=getattr(trainer, "opt", None),
                scheduler=getattr(trainer, "scheduler", None),
                epoch=trainer.start_epoch,         # 0-based
                global_step=trainer.global_step,
            )
            print(f"[ckpt] Saved to: {ckpt_path}")
        # Optional: synchronize all ranks
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    finally:
        # Ensure flush + close regardless of whether we were interrupted
        if _is_rank0() and getattr(trainer, "writer", None) is not None:
            try:
                # TB: has flush(); MLflowWriter can implement an empty flush()
                trainer.writer.flush()
            finally:
                trainer.writer.close()

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
