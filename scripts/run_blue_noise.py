import os, sys, argparse, torch
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
is_headless = (matplotlib.get_backend().lower() == "agg")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from flow_lab.paths import LinearConditionalProbabilityPath
from flow_lab.datasets import UniGBNSampler, PointSetMiniBatchOTDataset, PointSetEqOTFMDataset
# dynamics / trainers
from flow_lab.dynamics import EulerSimulator, VectorFieldODE
from flow_lab.trainers import UniGBNUnconditionalTrainer
# models
from flow_lab.models import *
from flow_lab.models import UncondUniGBNTransformer, UncondUniGBNTransformer_PE
# utils
from flow_lab.utils import save_checkpoint, render_point_images, extract_epoch
from flow_lab.voronoi import reconstruct_voronoi_images, export_voronoi_samples_for_fid
from flow_lab.distributions import JitterHilbertGridSample, Uniform
# 简单无条件 ODE 包装（与 EulerSimulator 的接口一致）

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=1000)
    p.add_argument("--in_out_dim", type=int, default=2)

    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)

    p.add_argument("--viz_steps", type=int, default=5)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--exp_name", type=str, default="uniGBN_uncond_linear_torus")
    # p.add_argument("--ckpt_dir", type=str, default="./checkpoints/uniGBN_uncond_linear_uniform")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--log_hist_every", type=int, default=100)
    p.add_argument("--fid_total", type=int, default=10000)
    p.add_argument("--n_points", type=int, default=1024)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--data_aug_rotate", action="store_true", help="whether load checkpoint")
    p.add_argument("--random_shuffle_z", action="store_true", help="whether load checkpoint")
    p.add_argument("--data_zorder", action="store_true", help="whether use z-order")
    p.add_argument("--vis_begin", action="store_true", help="whether vis at begin")

    p.add_argument("--jitter_x0", action="store_true", help="whether use jitter x0")
    p.add_argument("--if_load_ckpt", action="store_true", help="whether use rotation as data augmentation")
    p.add_argument("--only_load_model_weight", action="store_true", help="whether use rotation as data augmentation")
    p.add_argument("--data_path", type=str, default="data/1024_10k_original_sorted", help="data path")
    p.add_argument("--log_path", type=str, default="log", help="data path")
    p.add_argument("--ckpt_path", type=str, default="checkpoints/voronoi_uncond/UncondVoronoiTransformer_epoch651_e20000_1759864859.pt", help="指定 checkpoint 文件路径")
    p.add_argument("--use_warmup", action="store_true")
    p.add_argument("--use_cos_decay", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=None)

    # Async loader 相关参数
    p.add_argument("--use_async_loader", action="store_true")
    p.add_argument("--overwrite_lr", action="store_true")
    p.add_argument("--use_ot_match", action="store_true", help="使用 OT 匹配 (PointSetMiniBatchOTDataset)")

    p.add_argument("--prefetch_batches", type=int, default=2, help="异步加载预取 batch 数")
    p.add_argument("--ot_num_workers", type=int, default=8, help="OT cost matrix 计算的并行 worker 数")
    p.add_argument("--use_PE", action="store_true", help="使用带位置编码的 Transformer (UncondUniGBNTransformer_PE)")
    p.add_argument("--use_eqfm", action="store_true", help="使用 Equivariant OT Flow Matching (PointSetEqOTFMDataset)")
    p.add_argument("--point_ot_solver", type=str, default="hungarian", choices=["greedy", "hungarian"],
                   help="点级别 OT 求解器类型 (用于 use_eqfm)")
    p.add_argument("--batch_ot_solver", type=str, default="greedy", choices=["greedy", "hungarian"],
                   help="Batch 级别 OT 求解器类型 (用于 use_eqfm)")

    return p

def main(args=None):
    if args is None:  # CLI 路径
        args = build_argparser().parse_args()

    ckpt_dir = os.path.join(args.log_path, args.exp_name, "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    # p_simple = JitterHilbertGridSample(grid_size=32, jitter="uniform", periodic=True, seed=42).to("cuda").to(device)
    p_simple = Uniform(shape = [args.n_points, args.in_out_dim],).to(device)
    if args.jitter_x0:
        p_simple = JitterHilbertGridSample(grid_size=32, jitter="uniform", periodic=False, seed=1234).to(device)        

    # Dataset 初始化：根据是否使用 async loader 和 OT 匹配选择不同的 dataset
    mesh_dataset = None
    if args.use_async_loader:
        if args.use_eqfm:
            # 使用 Equivariant OT Flow Matching dataset
            mesh_dataset = PointSetEqOTFMDataset(
                data_dir=args.data_path,
                rotate4=args.data_aug_rotate,
                preload=True,
                random_shuffle=args.random_shuffle_z,
                point_ot_solver=args.point_ot_solver,
                batch_ot_solver=args.batch_ot_solver,
                use_multi=True,
                num_workers=args.ot_num_workers,
            )
            print(f"[Async Mode] Using PointSetEqOTFMDataset with point_ot={args.point_ot_solver}, batch_ot={args.batch_ot_solver}, {args.ot_num_workers} workers")
        elif args.use_ot_match:
            # 使用带 OT 匹配的 dataset
            mesh_dataset = PointSetMiniBatchOTDataset(
                data_dir=args.data_path,
                rotate4=args.data_aug_rotate,
                preload=True,
                random_shuffle=args.random_shuffle_z,
                ot_solver=args.batch_ot_solver,
                use_multi=False,
                num_workers=args.ot_num_workers,
            )
            print(f"[Async Mode] Using PointSetMiniBatchOTDataset with {args.batch_ot_solver} OT solver, {args.ot_num_workers} workers")
        else:
            # 使用 UniGBNSampler (已添加 compute_batch 方法)
            mesh_dataset = UniGBNSampler(
                data_dir=args.data_path,
                random_shuffle=args.random_shuffle_z,
                rotate4=args.data_aug_rotate,
                preload=True,
            )
            print("[Async Mode] Using UniGBNSampler with compute_batch interface")
        print(f"Total number of data points: {mesh_dataset.num_meshes}")

        # 创建一个 dummy sampler 用于 path（async 模式下不实际使用）
        sampler = UniGBNSampler(
            data_dir=args.data_path,
            random_shuffle=args.random_shuffle_z,
            rotate4=args.data_aug_rotate,
            preload=True,
        ).to(device)
    else:
        # 原有同步模式
        sampler = UniGBNSampler(
            data_dir=args.data_path,
            random_shuffle=args.random_shuffle_z,
            rotate4=args.data_aug_rotate,
            preload=True,
        ).to(device)
        print("Total number of data points: ", sampler.__len__())

    path = LinearConditionalProbabilityPath(
        p_simple = p_simple,
        p_data=sampler,
    ).to(device)

    if args.vis_begin:
        import numpy as np
        num_rows, num_cols = 2, 2
        k = num_rows * num_cols

        # 根据是否使用 async loader 和特殊 dataset 来获取 x0, x1 pair
        if mesh_dataset is not None and hasattr(mesh_dataset, 'compute_batch'):
            # 使用 compute_batch 获取 (x0, x1) pair
            idx_batch = np.arange(k)
            x0_np, x1_np = mesh_dataset.compute_batch(idx_batch, epoch=0, step=0)
            x0 = torch.from_numpy(x0_np).to(device)
            x1 = torch.from_numpy(x1_np).to(device)
            print(f"[VIS] Using compute_batch: x0 shape={x0.shape}, x1 shape={x1.shape}")
        else:
            # 原来的方式：从 path.p_data 获取 x1，从 p_simple 采样 x0
            x1, _ = path.p_data.get_batch(range(k))  # (k, N, D)
            x0, _ = path.p_simple.sample(k)          # (k, N, D)
            print(f"[VIS] Using p_data/p_simple: x0 shape={x0.shape}, x1 shape={x1.shape}")

        fig, axes = plt.subplots(1, args.viz_steps, figsize=(4 * args.viz_steps, 4))
        ts = torch.linspace(0, 1, args.viz_steps, device=device)
        for tidx, t in enumerate(ts):
            tt = t.view(1, 1, 1).expand(k, 1, 1)
            # 使用线性插值: xt = (1-t)*x0 + t*x1
            xt = path.sample_conditional_path_inputx0(x0, x1, tt)  # (k, N, D)
            imgs = render_point_images(
                xt,
                img_size=args.img_size,
                point_radius=getattr(args, "point_radius", 2),
                in_out_dim=args.in_out_dim,
                channels=3,
                background=1.0,
                point_value=0.0,
                antialias=False
            )

            grid = make_grid(imgs, nrow=num_cols)
            axes[tidx].imshow(grid.permute(1, 2, 0).cpu())
            axes[tidx].axis("off")
        plt.tight_layout()
        if is_headless:
            save_path = os.path.join(args.log_path, args.exp_name, "vis_begin.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            print(f"[VIS] Saved visualization to: {save_path}")
        if not is_headless:
            plt.show()

    # 根据 --use_PE 选择模型
    ModelCls = UncondUniGBNTransformer_PE if args.use_PE else UncondUniGBNTransformer
    model = ModelCls(
        n_points=args.n_points, in_dim=args.in_out_dim, out_dim=args.in_out_dim, embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, t_embed_dim=40
    ).to(device)
    if args.use_PE:
        print("[Model] Using UncondUniGBNTransformer_PE (with positional encoding)")
    else:
        print("[Model] Using UncondUniGBNTransformer (permutation equivariant)")

    # model = flow_kl_d256_m256_l32(N=args.n_points).to(device)

    start_epoch = 0
    if args.if_load_ckpt and args.ckpt_path and os.path.exists(args.ckpt_path):
        args.resume_path = args.ckpt_path
    else:
        args.resume_path = None

    trainer = UniGBNUnconditionalTrainer(
        path=path,
        model=model,
        rotate4=args.data_aug_rotate,
        zorder=args.data_zorder,
        start_epoch=start_epoch,
        use_async_loader=args.use_async_loader,
        mesh_dataset=mesh_dataset,
        prefetch_batches=args.prefetch_batches,
    )
    try:
        trainer.train(
            num_epochs=args.epochs,
            device=device,
            ckpt_dir=ckpt_dir,
            args = args,
        )
    except KeyboardInterrupt:
        print("\n[Interrupt] Training interrupted by user (Ctrl+C). Saving checkpoint...")

        ckpt_path = save_checkpoint(
            trainer.model, args, ckpt_dir, arch_name=f"UncondUniGBNTransformer_epoch{trainer.start_epoch+1:03d}"
        )

        print(f"[ckpt] Saved interrupted checkpoint to: {ckpt_path}")
        sys.exit(0)    
    finally:
        # 无论是否被打断，都确保 flush + close
        if getattr(trainer, "writer", None) is not None:
            try:
                # TB: 有 flush(); MLflowWriter 可实现空的 flush()
                trainer.writer.flush()
            finally:
                trainer.writer.close()

    # ckpt_path = save_checkpoint(model, args, args.ckpt_dir, arch_name="UncondUniGBNTransformer")
    # print(f"[ckpt] saved to {ckpt_path}")

    with torch.no_grad():
        ode = VectorFieldODE(model)
        simulator = EulerSimulator(ode)
        b = 16
        x0, _ = path.p_simple.sample(b)                                                  # (B,N,2)
        ts = torch.linspace(0, 1, args.sample_steps, device=device).view(1,-1,1,1).expand(b,-1,1,1)
        x1 = simulator.simulate(x0, ts)                                                 # (B,N,2)
        print(x1.shape)
        imgs = render_point_images(
            x1,
            img_size=args.img_size,
            point_radius=getattr(args, "point_radius", 2),
            channels=3,
            background=1.0,
            point_value=0.0,
            antialias=False
        )
        grid = make_grid(imgs, nrow=4)
        plt.figure(figsize=(8, 8)); plt.imshow(grid.permute(1,2,0).cpu()); plt.axis("off"); plt.title("Voronoi unconditional samples"); plt.show()


if __name__ == "__main__":
    main()
