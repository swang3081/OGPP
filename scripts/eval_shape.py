import os, sys, argparse, torch
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import math
import numpy as np
from pathlib import Path
import open3d as o3d

is_headless = (matplotlib.get_backend().lower() == "agg")


def _write_ply_with_batch_idx(ply_path: str, pts: np.ndarray, normals: np.ndarray, batch_idx: np.ndarray):
    """
    Write a PLY file (binary format) with a batch_idx attribute.
    Each point contains: x, y, z, nx, ny, nz, batch_idx
    Uses vectorized writing for better performance.
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

    # Build a structured array and write it in one pass
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
    RGB values are uint8 in the range 0-255.
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

    # Build a structured array and write it in one pass
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# models & dynamics
from flow_lab.models import UncondUniGBNTransformer, UncondUniGBNTransformer_PE
from flow_lab.dynamics import EulerSimulator, RK4Simulator, MidpointSimulator, VectorFieldODE

# utils
from flow_lab.utils import (
    load_checkpoint,
    render_point_images,
    save_pointcloud_ply,
    save_trajectory_frames,
    write_video_from_frames,
)


# ---- x0 initialization functions ----
def generate_sphere_init(B: int, N: int, radius: float = np.sqrt(2.0), seed: int = 0) -> np.ndarray:
    """Generate initial points uniformly distributed on a sphere surface"""
    rng = np.random.default_rng(seed=seed)
    x0 = rng.normal(size=(B, N, 3)).astype(np.float32)
    norm = np.linalg.norm(x0, axis=-1, keepdims=True)
    x0 = x0 / np.maximum(norm, 1e-12)
    x0 = x0 * radius
    return x0


def generate_shell_init(B: int, N: int, r_min: float = np.sqrt(2.0), r_max: float = 1.7, seed: int = 0) -> np.ndarray:
    """Generate initial points inside a spherical shell (uniform in volume)"""
    rng = np.random.default_rng(seed=seed)
    eps = np.float32(1e-12)

    # Directions
    dir_np = rng.standard_normal((B, N, 3), dtype=np.float32)
    norm2 = (dir_np * dir_np).sum(axis=-1, keepdims=True)
    np.maximum(norm2, eps, out=norm2)
    np.sqrt(norm2, out=norm2)
    dir_np /= norm2

    # Radius (uniform in volume)
    rmin3 = r_min ** 3
    rmax3 = r_max ** 3
    span3 = rmax3 - rmin3

    r = rng.random((B, N, 1), dtype=np.float32)
    r = np.cbrt(r * span3 + rmin3)

    return dir_np * r


def generate_box_init(B: int, N: int, dim: int = 3, seed: int = 0) -> np.ndarray:
    """Generate initial points uniformly distributed inside the [-1,1]^dim cube"""
    rng = np.random.default_rng(seed=seed)
    x0 = rng.random((B, N, dim), dtype=np.float32)
    x0 = x0 * 2.0 - 1.0
    return x0


class BF16InferODEWrapper:
    """Wrap the ODE so that drift_coefficient runs under bf16 autocast but returns float32"""
    def __init__(self, ode, use_bf16: bool = True):
        self.ode = ode
        self.use_bf16 = use_bf16

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v = self.ode.drift_coefficient(x, t, **kwargs)
            return v.float()  # return float32 to the ODE integrator
        else:
            return self.ode.drift_coefficient(x, t, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint path (.pt/.pth)")
    p.add_argument("--n_points", type=int, default=2048)
    p.add_argument("--n_point_set", type=int, default=16)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--in_out_dim", type=int, default=3)

    p.add_argument("--point_radius", type=float, default=4.0)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--coords_are_normalized", action="store_true")
    p.add_argument("--device", type=str, default="cuda")

    # Model architecture (must match training)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--use_rk4", action="store_true", help="whether use RK4")
    p.add_argument("--use_midpoint", action="store_true", help="whether use Midpoint method")
    p.add_argument("--save_fig", action="store_true", help="save fig")

    p.add_argument("--x_scale", type=float, default=1.0)
    p.add_argument("--y_scale", type=float, default=1.0)

    # Output-related
    p.add_argument("--output_txt", action="store_true", help="also write the sampled points as txt/ply files")
    p.add_argument("--render_image", action="store_true", help="render image")
    p.add_argument("--use_magnitude", action="store_true", help="whether use_magnitude")

    p.add_argument("--out_dir", type=str, default="outputs", help="output directory for figures and txt files")
    p.add_argument(
        "--output_trajectory",
        action="store_true",
    )
    p.add_argument(
        "--use_bf16_infer",
        action="store_true",
        help="Use bf16 for model forward pass",
    )
    p.add_argument(
        "--use_bf16_ode",
        action="store_true",
        help="Use bf16 for ODE integration (only effective when use_bf16_infer is True)",
    )

    p.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="folder name under out_dir/3d_pts for 3D point clouds. "
             "Default = ckpt filename without ext",
    )

    p.add_argument("--use_PE", action="store_true",)

    # x0 initialization method
    p.add_argument(
        "--use_sphere",
        action="store_true",
        help="Use sphere surface as initial distribution",
    )
    p.add_argument(
        "--use_shell",
        action="store_true",
        help="Use spherical shell as initial distribution",
    )

    p.add_argument(
        "--use_normal",
        action="store_true",
        help="Use predicted velocity as normal for Poisson reconstruction",
    )

    p.add_argument(
        "--is_single_mesh",
        action="store_true",
        help="When outputting trajectory, merge all batches into a single PLY per frame "
             "(each point has xyz + normal + batch_idx attribute). Uses multiprocessing.",
    )
    p.add_argument(
        "--is_single_pcd",
        action="store_true",
        help="Single point cloud mode (xyz+rgb). Same output behavior as is_single_mesh.",
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for parallel PLY writing (default: cpu_count())",
    )

    p.add_argument(
        "--output_ply_rgb",
        action="store_true",
        help="Output velocity as RGB color instead of normals in PLY files",
    )

    # Your previous eval had `periodic`; keep it here too
    p.add_argument("--periodic", action="store_true", help="periodic BC in simulator")

    # 6D linear mode
    p.add_argument("--linear_6d", action="store_true",
                   help="6D linear flow matching: x0~Uniform([-1,1]^6), model in_out_dim=6. "
                        "Output x_final[:,:,:3]=xyz, x_final[:,:,3:6]=normals.")

    # Random seed
    p.add_argument("--seed", type=int, default=123, help="random seed for x0 initialization")

    # Specify indices
    p.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated indices to select from n_point_set (e.g., '0,4,5,10'). If not provided, use all.",
    )

    # Batched inference
    p.add_argument("--eval_batch_size", type=int, default=64, help="batch size for inference (to avoid OOM)")

    # surface reconstruction
    p.add_argument(
        "--recon_surface",
        action="store_true",
        help="Run Poisson surface reconstruction",
    )

    args = p.parse_args()

    # ---- Automatic settings for linear_6d mode ----
    if args.linear_6d:
        args.in_out_dim = 6
        print("[linear_6d] Forcing in_out_dim=6")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Generate x0 (choose the initialization method based on use_sphere / use_shell) ----
    B = args.n_point_set
    N = args.n_points

    if args.linear_6d:
        print("Using 6D unit box init (linear_6d)")
        x0_np = generate_box_init(B, N, dim=6, seed=args.seed)
    elif args.use_sphere:
        print("Using sphere init")
        x0_np = generate_sphere_init(B, N, radius=np.sqrt(2.0), seed=args.seed)
    elif args.use_shell:
        print("Using shell init")
        x0_np = generate_shell_init(B, N, r_min=np.sqrt(2.0), r_max=1.7, seed=args.seed)
    else:
        print("Using unit box init")
        x0_np = generate_box_init(B, N, seed=args.seed)

    print(f"x0 shape: {x0_np.shape}, range: [{x0_np.min():.4f}, {x0_np.max():.4f}]")
    x0 = torch.from_numpy(x0_np).float().to(device=device)

    # If indices are specified, keep only those indices
    if args.indices is not None:
        indices = [int(i.strip()) for i in args.indices.split(",")]
        x0 = x0[indices]
        print(f"Selected indices: {indices}, new x0 shape: {x0.shape}")

    # ---- init model (architecture must match training) ----
    out_dim = 1 if args.use_magnitude else args.in_out_dim
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

    # load ckpt
    load_checkpoint(model, args.ckpt, map_location=device)
    model.eval()
    print(f"[ckpt] loaded from {args.ckpt}")

    # Make sure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- sampling from ODE: start from x0 ----
    # Decide the autocast strategy
    # use_bf16_infer=True, use_bf16_ode=True  -> the whole simulate runs in bf16
    # use_bf16_infer=True, use_bf16_ode=False -> bf16 only for the model forward pass; ODE integration in float32
    # use_bf16_infer=False                    -> float32 throughout
    use_full_bf16 = args.use_bf16_infer and args.use_bf16_ode
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_full_bf16 else torch.autocast(device_type="cuda", enabled=False)

    with torch.no_grad(), autocast_ctx:
        ode = VectorFieldODE(model, use_magnitude=args.use_magnitude)

        # If bf16 is used only for the model forward pass, use the wrapper
        if args.use_bf16_infer and not args.use_bf16_ode:
            ode = BF16InferODEWrapper(ode, use_bf16=True)
            print("[bf16] Using bf16 for model forward only, ODE integration in float32")
        elif use_full_bf16:
            print("[bf16] Using bf16 for both model forward and ODE integration")

        if args.use_rk4:
            print("Using RK4!!!")
            simulator = RK4Simulator(ode)
        elif args.use_midpoint:
            print("Using Midpoint!!!")
            simulator = MidpointSimulator(ode)
        else:
            simulator = EulerSimulator(ode)

        b = x0.shape[0]
        eval_batch_size = args.eval_batch_size
        ts_template = torch.linspace(0, 1.0, args.sample_steps, device=device)

        if args.output_trajectory:
            # ===== Trajectory mode (batched inference) =====
            xs_list = []
            num_batches = math.ceil(b / eval_batch_size)
            print(f"[trajectory] Running batched inference: {b} samples, batch_size={eval_batch_size}, {num_batches} batches")

            vs_list = []
            for batch_idx in range(num_batches):
                start = batch_idx * eval_batch_size
                end = min(start + eval_batch_size, b)
                x0_batch = x0[start:end]
                batch_b = x0_batch.shape[0]
                ts_batch = ts_template.view(1, -1, 1, 1).expand(batch_b, -1, 1, 1)

                xs_batch, vs_batch = simulator.simulate_with_trajectory(x0_batch, ts_batch, periodic=args.periodic, return_velocity=True)
                xs_list.append(xs_batch)
                vs_list.append(vs_batch)
                print(f"  [batch {batch_idx+1}/{num_batches}] samples {start}-{end-1} done")

            xs = torch.cat(xs_list, dim=0)  # (B, T, N, 3)
            vs = torch.cat(vs_list, dim=0)  # (B, T, N, 3) velocity field
            x_final = xs[:, -1]  # final step, used to align render / txt output
            print("[trajectory] xs shape:", xs.shape)
            print("[trajectory] x_final shape:", x_final.shape)

            # Save frames
            frames_dir = os.path.join(args.out_dir, "trajectory")
            # nrow = int(math.ceil(math.sqrt(b)))
            # T = save_trajectory_frames(
            #     xs=xs,
            #     frames_dir=frames_dir,
            #     img_size=args.img_size,
            #     point_radius=getattr(args, "point_radius", 4),
            #     nrow=nrow,
            # )
            # print(f"[trajectory] saved {T} frames to: {os.path.abspath(frames_dir)}")

            # Compose video
            # video_path = os.path.join(args.out_dir, "trajectory.mp4")
            # write_video_from_frames(frames_dir, video_path, fps=12)
            # print(f"[trajectory] wrote video to: {os.path.abspath(video_path)}")

            # === If 3D/6D and output_trajectory=True, export a PLY sequence for each time step (for Blender animation) ===
            if args.in_out_dim == 3 or args.linear_6d:
                traj_ply_dir = os.path.join(args.out_dir, "3d_pts", args.exp_name, "trajectory_ply")
                os.makedirs(traj_ply_dir, exist_ok=True)

                Xs_raw = xs.detach().cpu().float().numpy()  # (B,T,N,D) D=3 or 6
                Vs = vs.detach().cpu().float().numpy()  # (B,T,N,D) velocity field
                B_, T_steps, N_pts, D = Xs_raw.shape

                if args.linear_6d:
                    # 6D: first 3 dims = xyz, last 3 dims = normals
                    Xs = Xs_raw[:, :, :, :3]
                    Vs = Xs_raw[:, :, :, 3:6]  # use the last 3 dims of the flow output as normals
                else:
                    Xs = Xs_raw

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
                            pts = Xs[b_idx, t_idx]  # (N,3)
                            vel = Vs[b_idx, t_idx]  # (N,3) velocity

                            # Normalize the velocity to a unit vector and use it as the normal vector
                            vel_norm = np.linalg.norm(vel, axis=-1, keepdims=True)
                            normals = vel / np.maximum(vel_norm, 1e-8)

                            # Use open3d to save a PLY with normals
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pts)
                            pcd.normals = o3d.utility.Vector3dVector(normals)

                            # Set colors
                            colors = np.full((N_pts, 3), 200 / 255.0, dtype=np.float64)
                            special_idx = [0, N_pts // 2, N_pts - 1]
                            special_color = [
                                (1.0, 0.0, 0.0),    # red
                                (1.0, 1.0, 0.0),    # yellow
                                (0.0, 0.0, 1.0),    # blue
                            ]
                            for idx, col in zip(special_idx, special_color):
                                if 0 <= idx < N_pts:
                                    colors[idx] = np.array(col, dtype=np.float64)
                            pcd.colors = o3d.utility.Vector3dVector(colors)

                            ply_name = f"frame_{t_idx:04d}_b{b_idx}.ply"
                            ply_path = os.path.join(traj_ply_dir, ply_name)
                            o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)

                    print(f"[trajectory_ply] wrote PLY sequence to: {os.path.abspath(traj_ply_dir)}")
        else:
            # ===== Only the final step (batched inference) =====
            x_final_list = []
            num_batches = math.ceil(b / eval_batch_size)
            print(f"[inference] Running batched inference: {b} samples, batch_size={eval_batch_size}, {num_batches} batches")

            for batch_idx in range(num_batches):
                start = batch_idx * eval_batch_size
                end = min(start + eval_batch_size, b)
                x0_batch = x0[start:end]
                batch_b = x0_batch.shape[0]
                ts_batch = ts_template.view(1, -1, 1, 1).expand(batch_b, -1, 1, 1)

                x_final_batch = simulator.simulate(x0_batch, ts_batch, periodic=args.periodic)
                x_final_list.append(x_final_batch)
                print(f"  [batch {batch_idx+1}/{num_batches}] samples {start}-{end-1} done")

            x_final = torch.cat(x_final_list, dim=0)  # (B, N, 3)

        print("x_final shape:", x_final.shape)
        x1_min_dim = x_final.amin(dim=(0, 1)).detach().cpu().float().numpy()  # (D,)
        x1_max_dim = x_final.amax(dim=(0, 1)).detach().cpu().float().numpy()  # (D,)
        x1_min_all = float(x_final.min().item())
        x1_max_all = float(x_final.max().item())
        print(
            f"[x_final range] per-dim min={x1_min_dim}, max={x1_max_dim} "
            f"| global=[{x1_min_all:.6f}, {x1_max_all:.6f}]"
        )

        # ---- Render image ----
        if args.render_image:
            imgs = render_point_images(
                x_final,
                img_size=args.img_size,
                point_radius=getattr(args, "point_radius", 4),
                in_out_dim=args.in_out_dim,
                channels=3,
                background=1.0,
                point_value=0.0,
                antialias=False,
                x_scale=args.x_scale,
                y_scale=args.y_scale,
            )
            grid = make_grid(imgs, nrow=int(math.sqrt(b)))
            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0).cpu())
            plt.axis("off")
            plt.title("samples (from ODE)")
            if args.save_fig:
                plt.savefig(
                    os.path.join(args.out_dir, "points_hd.png"),
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0,
                )

            if is_headless:
                exp_name = args.exp_name or Path(args.ckpt).stem
                save_dir = os.path.join(args.out_dir, "3d_pts", exp_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "eval_render.png")
                plt.savefig(save_path, dpi=200)
                print(f"[VIS] Headless mode, saved to: {save_path}")
            else:
                plt.show()
            plt.close()

        # ---- txt / ply export ----
        if args.output_txt:
            if args.in_out_dim == 3 or args.linear_6d:
                # —— 3D/6D: export both txt and PLY to outputs/3d_pts/<exp_name>/{txt,ply} —— #
                exp_name = args.exp_name or Path(args.ckpt).stem
                exp_root = os.path.join(args.out_dir, "3d_pts", exp_name)
                ply_dir = os.path.join(exp_root, "ply")
                txt_dir = os.path.join(exp_root, "txt")
                os.makedirs(ply_dir, exist_ok=True)
                os.makedirs(txt_dir, exist_ok=True)

                X = x_final.detach().cpu().float()  # (B,N,D) D=3 or 6
                B_out, N, _ = X.shape

                for i in range(B_out):
                    pts = X[i, :, :3].numpy()  # (N,3) xyz

                    # 1) Write txt: first line N, then x y z on each subsequent line
                    txt_path = os.path.join(txt_dir, f"pts_{i}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(f"{N}\n")
                        for j in range(N):
                            x, y, z = pts[j].tolist()
                            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

                    # 2) Write ply
                    if args.linear_6d:
                        # 6D: PLY with normals
                        normals_i = X[i, :, 3:6].numpy()  # (N,3)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pts)
                        pcd.normals = o3d.utility.Vector3dVector(normals_i)
                        ply_path = os.path.join(ply_dir, f"pts_{i}.ply")
                        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
                    else:
                        # 3D: highlight first/middle/last
                        colors = np.full((N, 3), 200, dtype=np.uint8)
                        special_idx = [0, N // 2, N - 1]
                        special_color = [
                            (255, 0, 0),
                            (255, 255, 0),
                            (0, 0, 255),
                        ]
                        for idx, col in zip(special_idx, special_color):
                            if 0 <= idx < N:
                                colors[idx] = np.array(col, dtype=np.uint8)

                        ply_path = os.path.join(ply_dir, f"pts_{i}.ply")
                        save_pointcloud_ply(pts, ply_path, colors_np=colors, binary=True)

                print(f"[3D txt] wrote {B_out} files to: {os.path.abspath(txt_dir)}")
                print(f"[3D ply] wrote {B_out} files to: {os.path.abspath(ply_dir)}")
                print("Blender: File > Import > Stanford (.ply) -> select the ply directory above")
            else:
                os.makedirs(os.path.join(args.out_dir, "txt"), exist_ok=True)
                X = x_final.detach().cpu().float()  # (B,N,2)
                X = (X + 1.0) / 2.0
                B_out, N, _ = X.shape
                for i in range(B_out):
                    fpath = os.path.join(args.out_dir, "txt", f"pts_{i}.txt")
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(f"{N}\n")
                        for j in range(N):
                            x, y = X[i, j].tolist()
                            f.write(f"{x:.6f} {y:.6f}\n")
                print(f"[txt] wrote {B_out} files to: {os.path.abspath(os.path.join(args.out_dir, 'txt'))}")

        # ---- Poisson reconstruction ----
        if args.in_out_dim == 3 or args.linear_6d:
            exp_name = args.exp_name or Path(args.ckpt).stem
            mesh_recon_dir = os.path.join(args.out_dir, "3d_pts", exp_name, "mesh_recon")
            mesh_no_normal_dir = os.path.join(args.out_dir, "3d_pts", exp_name, "mesh_no_normal")
            os.makedirs(mesh_recon_dir, exist_ok=True)
            os.makedirs(mesh_no_normal_dir, exist_ok=True)

            X_raw = x_final.detach().cpu().float().numpy().astype(np.float32)
            B_out, N_pts, _ = X_raw.shape

            if args.linear_6d:
                # 6D: xyz from first 3 dims, normals from last 3 dims
                X = X_raw[:, :, :3]
                normals_np = X_raw[:, :, 3:6]
                # Normalize the normals
                n_norm = np.linalg.norm(normals_np, axis=-1, keepdims=True)
                normals_np = normals_np / np.maximum(n_norm, 1e-8)
                print("[linear_6d] Using normals from x_final[:,:,3:6]")
            else:
                X = X_raw
                # Get the predicted velocity as the normal (if use_normal)
                normals_np = None
                if args.use_normal:
                    t_final = torch.ones(b, 1, 1, device=device) * (1.0 - 1e-5)
                    # ode may be the wrapper (handles bf16 internally) or the raw ODE (requires an outer autocast)
                    v_final = ode.drift_coefficient(x_final.float(), t_final)  # (B, N, 3)
                    v_final_np = v_final.detach().cpu().float().numpy().astype(np.float32)
                    v_norm = np.linalg.norm(v_final_np, axis=-1, keepdims=True)
                    normals_np = v_final_np / np.maximum(v_norm, 1e-8)

            print(f"[mesh_recon] Running Poisson reconstruction for {B_out} point clouds...")

            def repair_mesh(mesh):
                """Repair the mesh: remove non-manifold edges, duplicate vertices, isolated vertices, fill holes, etc."""
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                mesh.remove_unreferenced_vertices()

                # Fill holes
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
                mesh = mesh.fill_holes().to_legacy()

                return mesh

            for i in range(B_out):
                pts = X[i]  # (N, 3)

                # ========== 1. Reconstruct using the predicted normals (if use_normal or linear_6d) ==========
                if (args.use_normal or args.linear_6d) and normals_np is not None:
                    pcd_with_normal = o3d.geometry.PointCloud()
                    pcd_with_normal.points = o3d.utility.Vector3dVector(pts)

                    if args.output_ply_rgb:
                        # Use the velocity field as RGB color
                        rgb = _velocity_to_rgb(normals_np[i])  # normals_np is already the normalized velocity
                        pcd_with_normal.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
                        pcd_path = os.path.join(mesh_recon_dir, f"pcd_with_rgb_{i}.ply")
                    else:
                        pcd_with_normal.normals = o3d.utility.Vector3dVector(normals_np[i])
                        pcd_path = os.path.join(mesh_recon_dir, f"pcd_with_normals_{i}.ply")

                    o3d.io.write_point_cloud(pcd_path, pcd_with_normal)

                    pcd_with_normal.normals = o3d.utility.Vector3dVector(-normals_np[i])

                    if args.recon_surface:
                        try:
                            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                                pcd_with_normal, depth=9, width=0, scale=1.1, linear_fit=False
                            )
                            mesh = repair_mesh(mesh)

                            mesh_path = os.path.join(mesh_recon_dir, f"mesh_{i}.ply")
                            o3d.io.write_triangle_mesh(mesh_path, mesh)

                        except Exception as e:
                            print(f"[mesh_recon] Failed for sample {i} (with normal): {e}")

                if args.recon_surface:
                    pcd_no_normal = o3d.geometry.PointCloud()
                    pcd_no_normal.points = o3d.utility.Vector3dVector(pts)
                    pcd_no_normal.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
                    )
                    pcd_no_normal.orient_normals_consistent_tangent_plane(k=30)

                    try:
                        mesh_no_normal, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                            pcd_no_normal, depth=9, width=0, scale=1.1, linear_fit=False
                        )
                        mesh_no_normal = repair_mesh(mesh_no_normal)

                        mesh_path = os.path.join(mesh_no_normal_dir, f"mesh_{i}.ply")
                        o3d.io.write_triangle_mesh(mesh_path, mesh_no_normal)

                        pcd_path = os.path.join(mesh_no_normal_dir, f"pcd_estimated_normals_{i}.ply")
                        o3d.io.write_point_cloud(pcd_path, pcd_no_normal)

                    except Exception as e:
                        print(f"[mesh_recon] Failed for sample {i} (no normal): {e}")

            print(f"[mesh_recon] wrote {B_out} meshes to: {os.path.abspath(mesh_recon_dir)}")
            print(f"[mesh_no_normal] wrote {B_out} meshes to: {os.path.abspath(mesh_no_normal_dir)}")

            # ---- Extra output: when use_normal + output_trajectory + (is_single_mesh or is_single_pcd), merge all pcd_with_normals into trajectory_ply ----
            if (args.use_normal or args.linear_6d) and args.output_trajectory and (args.is_single_mesh or args.is_single_pcd) and normals_np is not None:
                traj_ply_dir = os.path.join(args.out_dir, "3d_pts", exp_name, "trajectory_ply")
                merged_dir = os.path.join(traj_ply_dir, f"frames_{args.sample_steps}")
                os.makedirs(merged_dir, exist_ok=True)

                # Merge points and normals of all batches
                all_pts = X.reshape(-1, 3)  # (B*N, 3)
                all_normals = normals_np.reshape(-1, 3)  # (B*N, 3)

                # Generate the batch_idx attribute
                batch_idx = np.repeat(np.arange(B_out, dtype=np.int32), N_pts)  # (B*N,)

                if args.output_ply_rgb:
                    # Write a PLY with RGB color
                    rgb = _velocity_to_rgb(all_normals)
                    ply_path = os.path.join(merged_dir, "merged_pcd_with_rgb.ply")
                    _write_ply_with_rgb(ply_path, all_pts, rgb, batch_idx)
                else:
                    # Write the merged PLY (mimicking the format of _write_ply_with_batch_idx)
                    ply_path = os.path.join(merged_dir, "merged_pcd_with_normals.ply")
                    _write_ply_with_batch_idx(ply_path, all_pts, all_normals, batch_idx)

                print(f"[merged_pcd] wrote merged PLY ({B_out} batches × {N_pts} points = {B_out * N_pts} points) to: {os.path.abspath(ply_path)}")


if __name__ == "__main__":
    main()
