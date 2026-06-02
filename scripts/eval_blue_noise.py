import os, sys, argparse, torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# paths
from flow_lab.paths import LinearConditionalProbabilityPath
# datasets
from flow_lab.datasets import UniGBNSampler
# dynamics
from flow_lab.dynamics import EulerSimulator, RK4Simulator
# models
from flow_lab.models import UncondUniGBNTransformer, UncondUniGBNTransformer_PE
# voronoi utils
from flow_lab.voronoi import reconstruct_voronoi_images
from flow_lab.utils import *
from flow_lab.distributions import JitterHilbertGridSample, Uniform, IsotropicGaussian
import math

import numpy as np
from pathlib import Path
from flow_lab.sort_numba import hilbert_sort_xy_fast
# 简单无条件 ODE 包装
class VectorFieldODE:
    def __init__(self, net):
        self.net = net
    def drift_coefficient(self, x, t, y=None):
        return self.net(x, t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint path (.pt/.pth)")
    p.add_argument("--n_points", type=int, default=1024)
    p.add_argument("--n_point_set", type=int, default=4)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--in_out_dim", type=int, default=2)

    p.add_argument("--point_radius", type=float, default=4.0)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--coords_are_normalized", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data_path", type=str, default="data/1024_10k_original_sorted", help="data path")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--use_rk4", action="store_true", help="whether use RK4")
    p.add_argument("--save_fig", action="store_true", help="save fig")

    p.add_argument("--x_scale", type=float, default=1.0)
    p.add_argument("--y_scale", type=float, default=1.0)


    # 新增：文本输出与输出目录
    p.add_argument("--use_PE", action="store_true")
    p.add_argument("--jitter_x0", action="store_true")
    p.add_argument("--sort_x0", action="store_true")
    p.add_argument("--gaussian_init", action="store_true")

    p.add_argument("--periodic", action="store_true")
    p.add_argument("--output_txt", action="store_true", help="同时把采样的点写为 txt 文件")
    p.add_argument("--render_image", action="store_true", help="render image")
    p.add_argument("--out_dir", type=str, default="outputs", help="图与txt输出目录")
    p.add_argument("--output_trajectory", action="store_true",
                   help="若开启，则保存轨迹帧到 out_dir/trajectory 并导出视频")

    p.add_argument("--exp_name", type=str, default=None,
                help="folder name under out_dir/3d_pts for 3D point clouds. "
                        "Default = ckpt filename without ext")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    p.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated indices to select from x0 (e.g., '0,4,5,10'). If not provided, use all.",
    )


    args = p.parse_args()

    # 设置随机种子以确保可复现性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # dataset sampler (只为 p_data / p_simple shape 提供接口)
    if args.jitter_x0:
        print("jitter!")
        p_simple = JitterHilbertGridSample(grid_size=32, jitter="uniform", periodic=args.periodic, seed=1234).to(device)
    elif args.gaussian_init:
        p_simple = IsotropicGaussian(shape = [args.n_points, 2],).to("cuda").to(device)
    else:
        p_simple = Uniform(shape = [args.n_points, args.in_out_dim],).to("cuda").to(device)


    # sampler = UniGBNSampler(data_dir = args.data_path).to(device)
    # print("Total number of data points: ", sampler.__len__())
    # path = LinearConditionalProbabilityPath(
    #     p_simple = p_simple,
    #     p_data=sampler,
    # ).to(device)

    # init model (结构要和训练时一致)
    # model = UncondUniGBNTransformer(
    #     n_points=args.n_points, in_dim=args.in_out_dim, out_dim= args.in_out_dim, embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, t_embed_dim=40
    # ).to(device)
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


    # load ckpt
    load_checkpoint(model, args.ckpt, map_location=device)
    model.eval()
    print(f"[ckpt] loaded from {args.ckpt}")

    # 确保输出目录存在（用于 txt 或可能的图片保存）
    os.makedirs(args.out_dir, exist_ok=True)

    # sampling
    with torch.no_grad():
        ode = VectorFieldODE(model)
        if not args.use_rk4:
            simulator = EulerSimulator(ode)
        else:
            print("Using RK4!!!")
            simulator = RK4Simulator(ode)

        b = args.n_point_set
        x0, _ = p_simple.sample(b)  # (B,N,2)
        if args.sort_x0:
            x0 = torch.from_numpy(
                hilbert_sort_xy_fast(
                    x0.detach().cpu().numpy())
                ).to(x0.device)

        # 如果指定了 indices，则只取这些索引
        if args.indices is not None:
            indices = [int(i.strip()) for i in args.indices.split(",")]
            x0 = x0[indices]
            b = x0.shape[0]
            print(f"Selected indices: {indices}, new x0 shape: {x0.shape}")

        ts = torch.linspace(0, 1, args.sample_steps, device=device).view(1, -1, 1, 1).expand(b, -1, 1, 1)
        if args.output_trajectory:
            # ===== 轨迹模式 =====
            xs = simulator.simulate_with_trajectory(x0, ts, periodic=args.periodic)  # (B,T,N,2)
            print("[trajectory] xs shape:", xs.shape)

            # 应用 x_scale/y_scale 到整个轨迹
            xs[..., 0] = xs[..., 0] * args.x_scale
            xs[..., 1] = xs[..., 1] * args.y_scale

            x1 = xs[:, -1]  # 在 scale 之后取最终步
            print("[trajectory] x1 shape:", x1.shape)

            # 保存帧
            exp_name = args.exp_name or Path(args.ckpt).stem
            frames_dir = os.path.join(args.out_dir, exp_name, "trajectory")
            # nrow 尽量取 sqrt(b) 的上取整，更均衡的拼图排布
            nrow = int(math.ceil(math.sqrt(b)))
            T = save_trajectory_frames(
                xs=xs,
                frames_dir=frames_dir,
                img_size=args.img_size,
                point_radius=getattr(args, "point_radius", 4),
                nrow=nrow
            )
            print(f"[trajectory] saved {T} frames to: {os.path.abspath(frames_dir)}")

            # 合成视频
            # video_path = os.path.join(args.out_dir, "trajectory.mp4")
            # write_video_from_frames(frames_dir, video_path, fps=12)

            # print(f"[trajectory] wrote video to: {os.path.abspath(video_path)}")

            # === 新增：如果是 3D 且 output_trajectory=True，则导出每个时间步的 PLY 序列（用于 Blender 动画） ===
            if args.in_out_dim == 3:
                traj_ply_dir = os.path.join(args.out_dir, exp_name, "trajectory_ply")
                os.makedirs(traj_ply_dir, exist_ok=True)

                Xs = xs.detach().cpu()  # (B,T,N,3)
                B, T_steps, N_pts, D = Xs.shape
                assert D == 3

                print(f"[trajectory_ply] exporting {T_steps} time steps for {B} batches to {traj_ply_dir}")

                for t_idx in range(T_steps):
                    for b_idx in range(B):
                        pts = Xs[b_idx, t_idx].numpy().astype(np.float32)  # (N,3)

                        # 简单上个颜色（和静态 3D 输出保持一致：首/中/尾高亮）
                        colors = np.full((N_pts, 3), 200, dtype=np.uint8)
                        special_idx   = [0, N_pts // 2, N_pts - 1]
                        special_color = [(255, 0, 0), (255, 255, 0), (0, 0, 255)]  # 红/黄/蓝
                        for idx, col in zip(special_idx, special_color):
                            if 0 <= idx < N_pts:
                                colors[idx] = np.array(col, dtype=np.uint8)

                        # 文件名：frame_{t}_b{b}.ply
                        ply_name = f"frame_{t_idx:04d}_b{b_idx}.ply"
                        ply_path = os.path.join(traj_ply_dir, ply_name)
                        save_pointcloud_ply(pts, ply_path, colors_np=colors, binary=True)

                print(f"[trajectory_ply] wrote PLY sequence to: {os.path.abspath(traj_ply_dir)}")

        else:
            x1 = simulator.simulate(x0, ts, periodic=args.periodic)  # (B,N,2)
            # 非轨迹模式，在这里应用 scale
            x1[..., 0] = x1[..., 0] * args.x_scale
            x1[..., 1] = x1[..., 1] * args.y_scale

        print(x1.shape)
        x1_min_dim = x1.amin(dim=(0, 1)).detach().cpu().numpy()  # (D,)
        x1_max_dim = x1.amax(dim=(0, 1)).detach().cpu().numpy()  # (D,)
        x1_min_all = float(x1.min().item())
        x1_max_all = float(x1.max().item())
        print(f"[x1 range] per-dim min={x1_min_dim}, max={x1_max_dim} | global=[{x1_min_all:.6f}, {x1_max_all:.6f}]")
        if args.render_image:
            imgs = render_point_images(
                x1,
                img_size=args.img_size,
                point_radius=getattr(args, "point_radius", 4),
                in_out_dim = args.in_out_dim,
                channels=3,
                background=1.0,
                point_value=0.0,
                antialias=False,
                # x_scale=args.x_scale,
                # y_scale=args.y_scale,
                color_point=False
            )
            grid = make_grid(imgs, nrow=int(math.sqrt(b)))
            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0).cpu())
            plt.axis("off")
            # plt.title("UniGBN unconditional samples")
            if args.save_fig:
                plt.savefig("points_hd.png", dpi=600, bbox_inches="tight", pad_inches=0)

            plt.show()

        # 如果需要，同时把 x1 写成 txt
        if args.output_txt:
            if args.in_out_dim == 3:
                # —— 3D：同时导出 txt 和 PLY 到 outputs/3d_pts/<exp_name>/{txt,ply} —— #
                exp_name = args.exp_name or Path(args.ckpt).stem
                exp_root = os.path.join(args.out_dir, "3d_pts", exp_name)
                ply_dir  = os.path.join(exp_root, "ply")
                txt_dir  = os.path.join(exp_root, "txt")
                os.makedirs(ply_dir, exist_ok=True)
                os.makedirs(txt_dir, exist_ok=True)

                X = x1.detach().cpu()  # (B,N,3)
                B, N, _ = X.shape
                for i in range(B):
                    pts = X[i].numpy().astype(np.float32)  # (N,3) 原坐标不做归一化

                    # 1) 写 txt：第一行 N，后面每行 x y z
                    txt_path = os.path.join(txt_dir, f"pts_{i}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(f"{N}\n")
                        for j in range(N):
                            x, y, z = pts[j].tolist()
                            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

                    # 2) 写 ply：并高亮 first/middle/last
                    colors = np.full((N, 3), 200, dtype=np.uint8)
                    special_idx   = [0, N // 2, N - 1]
                    special_color = [(255, 0, 0), (255, 255, 0), (0, 0, 255)]  # 红/黄/蓝
                    for idx, col in zip(special_idx, special_color):
                        if 0 <= idx < N:
                            colors[idx] = np.array(col, dtype=np.uint8)

                    ply_path = os.path.join(ply_dir, f"pts_{i}.ply")
                    save_pointcloud_ply(pts, ply_path, colors_np=colors, binary=True)

                print(f"[3D txt] wrote {B} files to: {os.path.abspath(txt_dir)}")
                print(f"[3D ply] wrote {B} files to: {os.path.abspath(ply_dir)}")
                print("Blender: File > Import > Stanford (.ply) -> 选择上述 ply 目录")
            else:
                os.makedirs(os.path.join(args.out_dir, args.exp_name, "txt"), exist_ok=True)
                X = x1.detach().cpu()  # (B,N,2) on CPU
                X = (X + 1.0) / 2.0
                B, N, _ = X.shape
                for i in range(B):
                    # fpath = os.path.join(args.out_dir, f"pts_{i}.txt")
                    fpath = os.path.join(args.out_dir, args.exp_name, "txt", f"pts_{i}.txt")
                    with open(fpath, "w", encoding="utf-8") as f:
                        # 第一行写 N
                        f.write(f"{N}\n")
                        # 从第二行开始，每行写一个点: x y
                        for j in range(N):
                            x, y = X[i, j].tolist()
                            f.write(f"{x:.6f} {y:.6f}\n")
                print(f"[txt] wrote {B} files to: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
