from typing import Optional
import numpy as np
import torch
from .distributions import Sampleable
import torch.nn as nn
from torchvision import datasets, transforms
from typing import Optional, List, Type, Tuple, Dict, Callable
from PIL import Image
import pickle
import os
import re
import glob
import math
from .sort_numba import hilbert_sort_xy_fast
from typing import Union, List
from .utils import *
import igl
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_config
import threading
import queue
from pathlib import Path



class MeshNearPointPairDataset:
    """
    Loads a mesh dataset, normalizes it to [0,1]^3, and precomputes
    (x0, x1) pairs each epoch according to a given permutation.

    - x0: (B, N, 3), each point is a uniform sample in [0,1]^3
    - x1: the corresponding nearest point on the mesh surface (B, N, 3)

    Usage:
        dataset = MeshNearPointPairDataset(
            mesh_root="path/to/meshes",
            num_points=2048,
            use_multi=True,
            num_workers=8,
            device="cpu",        # which device the precomputed results live on (usually cpu)
        )

        # In the trainer:
        dataset.prepare_epoch(local_perm, batch_size=batch_size, epoch=epoch)
        x0, x1 = dataset.get_batch(step_idx)  # torch.Tensor (B, N, 3)
    """

    def __init__(
        self,
        mesh_root: str,
        num_points: int,
        use_multi: bool = True,
        num_workers: int = 8,
        device: str = "cpu",
    ):
        self.mesh_root = Path(mesh_root).expanduser().resolve()
        self.num_points = int(num_points)
        self.use_multi = bool(use_multi)
        self.num_workers = int(num_workers)
        self.device = torch.device(device)

        self.mesh_paths: List[Path] = []
        self.meshes: List[Tuple[np.ndarray, np.ndarray]] = []
        self.trees: List[igl.AABB] = []

        self._load_meshes_and_build_aabbs()

        # These are only populated after prepare_epoch
        self._epoch_batches_x0: List[torch.Tensor] = []
        self._epoch_batches_x1: List[torch.Tensor] = []
        self._steps_per_epoch: int = 0

    # ----------------- Loading + AABB -----------------

    def _load_meshes_and_build_aabbs(self):
        # ply_files = find_all_ply(self.mesh_root)
        ply_files = find_ply_files_from_all_csv(self.mesh_root, split="train")

        if not ply_files:
            raise RuntimeError(f"No .ply files found under: {self.mesh_root}")

        print(f"[MeshNearPointPairDataset] Found {len(ply_files)} .ply files, loading...")
        for p in tqdm(ply_files, desc="Loading meshes"):
            try:
                V, F = igl.read_triangle_mesh(str(p))
            except Exception as e:
                print(f"[WARN] Failed to read mesh {p}: {e}")
                continue

            if V.size == 0 or F.size == 0:
                print(f"[WARN] Empty mesh (V/F) in {p}, skip.")
                continue

            Vn = normalize_and_orient_vertices(V)

            tree = igl.AABB()
            tree.init(Vn, F)

            self.mesh_paths.append(p)
            self.meshes.append((Vn, F))
            self.trees.append(tree)

        if not self.meshes:
            raise RuntimeError("No valid meshes loaded after filtering.")

        print(f"[MeshNearPointPairDataset] Loaded {len(self.meshes)} meshes with AABB trees.")

    @property
    def num_meshes(self) -> int:
        return len(self.meshes)

    # ----------------- Per-epoch precomputation of (x0, x1) -----------------

    def compute_batch(
        self,
        idx_batch: np.ndarray,  # shape (B,)
        epoch: int,
        step: int,
    ):
        """
        Given a batch of mesh indices idx_batch for this rank, return (x0, x1):

            x0: (B, N, 3) uniform in [0,1]^3, float32
            x1: (B, N, 3) nearest point on mesh surface, float32

        epoch + step are only used as a deterministic RNG seed.
        """
        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        B = idx_batch.shape[0]
        N = self.num_points

        # Use an independent seed for each (epoch, step) to stay deterministic
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        x0_np = rng.random((B, N, 3), dtype=np.float64)
        # to [-1, 1]
        x0_np = x0_np * 2.0 - 1.0
        x1_np = np.empty_like(x0_np, dtype=np.float64)

        if self.use_multi and self.num_workers > 1:
            tasks = []
            for bi, mi in enumerate(idx_batch):
                V, F = self.meshes[mi]
                P = x0_np[bi]
                tasks.append((bi, V, F, P))

            results = Parallel(
                n_jobs=self.num_workers,
                backend="loky",
            )(
                delayed(nearest_one_job_proc)(bi, V, F, P)
                for (bi, V, F, P) in tasks
            )

            for bi, C in results:
                x1_np[bi] = C
        else:
            # Single process: reuse the prebuilt AABB
            for bi, mi in enumerate(idx_batch):
                V, F = self.meshes[mi]
                tree = self.trees[mi]
                P = x0_np[bi]
                _, _, C = tree.squared_distance(V, F, P)
                x1_np[bi] = C

        # Return numpy (float32); the caller wraps it into torch
        return x0_np.astype(np.float32), x1_np.astype(np.float32)

    @property
    def steps_per_epoch(self) -> int:
        return self._steps_per_epoch

    def get_batch(self, step_idx: int):
        """
        Return (x0, x1) as two torch.Tensors, both shape (B, N, 3), currently on CPU.
        The trainer can call .to(device, non_blocking=True) to move them to the GPU.
        """
        if step_idx < 0 or step_idx >= self._steps_per_epoch:
            raise IndexError(
                f"step_idx={step_idx} out of range [0, {self._steps_per_epoch})"
            )
        return self._epoch_batches_x0[step_idx], self._epoch_batches_x1[step_idx]


class MeshPoissonSphereDataset:
    """
    MeshPoissonSphereDataset

    Loads a mesh dataset, normalizes it to [-1,1]^3, and precomputes
    (x0, x1) pairs each epoch according to a given permutation.

    - x1: points sampled on each mesh via Open3D's sample_points_poisson_disk (B, N, 3)
    - x0: the nearest points of those x1 on the sphere centered at the origin with radius sqrt(2) (B, N, 3)

    Usage:
        dataset = MeshPoissonSphereDataset(
            mesh_root="path/to/meshes",
            num_points=2048,
            use_multi=True,
            num_workers=8,
            device="cpu",
        )

        # In the trainer, together with AsyncLoader:
        #   g = torch.Generator(...); global_perm = torch.randperm(dataset.num_meshes, generator=g)
        #   local_perm = global_perm[rank::world_size]
        #   async_loader = MeshPairAsyncLoader(dataset, batch_size, device, prefetch_batches=2)
        #   async_loader.start_epoch(local_perm=local_perm, epoch=epoch)
        #   x0, x1 = async_loader.next_batch()
    """

    def __init__(
        self,
        mesh_root: str,
        num_points: int,
        use_multi: bool = True,
        num_workers: int = 8,
        device: str = "cpu",
        sphere_radius: float = math.sqrt(2.0),
    ):
        self.mesh_root = Path(mesh_root).expanduser().resolve()
        self.num_points = int(num_points)
        self.use_multi = bool(use_multi)
        self.num_workers = int(num_workers)
        self.device = torch.device(device)
        self.sphere_radius = float(sphere_radius)

        # Stores vertices & faces after normalization to [-1,1]^3
        self.mesh_paths: List[Path] = []
        self.vertices_list: List[np.ndarray] = []
        self.faces_list: List[np.ndarray] = []

        self._load_meshes()

        # Kept for interface compatibility (unused unless you use the synchronous prepare_epoch)
        self._epoch_batches_x0: List[torch.Tensor] = []
        self._epoch_batches_x1: List[torch.Tensor] = []
        self._steps_per_epoch: int = 0

    # ----------------- Loading & normalization -----------------

    def _load_meshes(self):
        # ply_files = sorted(self.mesh_root.rglob("*.ply"))
        ply_files = find_ply_files_from_all_csv(self.mesh_root, split="train")

        if not ply_files:
            raise RuntimeError(f"No .ply files found under: {self.mesh_root}")

        print(f"[MeshPoissonSphereDataset] Found {len(ply_files)} .ply files, loading...")
        for p in tqdm(ply_files, desc="Loading meshes"):
            try:
                mesh = o3d.io.read_triangle_mesh(str(p))
            except Exception as e:
                print(f"[WARN] Failed to read mesh {p}: {e}")
                continue

            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                print(f"[WARN] Empty mesh in {p}, skip.")
                continue

            V = np.asarray(mesh.vertices, dtype=np.float64)
            F = np.asarray(mesh.triangles, dtype=np.int32)

            Vn = normalize_vertices_minus1_1(V)

            self.mesh_paths.append(p)
            self.vertices_list.append(Vn)
            self.faces_list.append(F)

        if not self.vertices_list:
            raise RuntimeError("No valid meshes loaded after filtering.")

        print(f"[MeshPoissonSphereDataset] Loaded {len(self.vertices_list)} normalized meshes.")

    @property
    def num_meshes(self) -> int:
        return len(self.vertices_list)

    # ----------------- Per-epoch batch computation -----------------

    def compute_batch(
        self,
        idx_batch: np.ndarray,  # shape (B,)
        epoch: int,
        step: int,
    ):
        """
        Given a batch of mesh indices idx_batch for this rank, return (x0, x1):

            x1: (B, N, 3) Poisson-disk sample on mesh surface, float32
            x0: (B, N, 3) the corresponding nearest points on the sphere, float32

        epoch + step are currently only used as a seed (in case you want to use it in numpy).
        Open3D's Poisson sampling has no explicit seed, so strict determinism is not guaranteed.
        """
        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        B = idx_batch.shape[0]
        N = self.num_points

        # You can apply some numpy-level randomness control based on epoch/step,
        # though Open3D's internal RNG may not be affected (the seed is kept here for future extension).
        seed = 0x123456 + int(epoch) * 100000 + int(step)
        np.random.seed(seed)

        x0_np = np.empty((B, N, 3), dtype=np.float64)
        x1_np = np.empty((B, N, 3), dtype=np.float64)

        if self.use_multi and self.num_workers > 1:
            tasks = []
            for bi, mi in enumerate(idx_batch):
                V = self.vertices_list[mi]
                F = self.faces_list[mi]
                tasks.append((bi, V, F, N, self.sphere_radius))

            results = Parallel(
                n_jobs=self.num_workers,
                backend="loky",
            )(
                delayed(poisson_sphere_one_mesh)(bi, V, F, n_pts, radius)
                for (bi, V, F, n_pts, radius) in tasks
            )

            for bi, x0, x1 in results:
                x0_np[bi] = x0
                x1_np[bi] = x1
        else:
            # Single-process version
            for bi, mi in enumerate(idx_batch):
                V = self.vertices_list[mi]
                F = self.faces_list[mi]
                _, x0, x1 = poisson_sphere_one_mesh(
                    bi,
                    V,
                    F,
                    num_points=N,
                    radius=self.sphere_radius,
                )
                x0_np[bi] = x0
                x1_np[bi] = x1

        return x0_np.astype(np.float32), x1_np.astype(np.float32)

    # ----------------- Interfaces matching MeshNearPointPairDataset -----------------

    @property
    def steps_per_epoch(self) -> int:
        """
        Kept only for interface compatibility; unused if you use the AsyncLoader.
        """
        return self._steps_per_epoch

    def get_batch(self, step_idx: int):
        """
        Compatibility interface: returns (x0, x1) as torch.Tensors.
        Only meaningful if you implement a synchronous prepare_epoch that fills
        _epoch_batches_x0/x1. If you only use compute_batch with MeshPairAsyncLoader,
        this function is not needed.
        """
        if step_idx < 0 or step_idx >= self._steps_per_epoch:
            raise IndexError(
                f"step_idx={step_idx} out of range [0, {self._steps_per_epoch})"
            )
        return self._epoch_batches_x0[step_idx], self._epoch_batches_x1[step_idx]



def _natural_key(s: str):
    # Natural ordering (pts_2.txt < pts_10.txt)
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(s))]

def _read_one_txt(path: str) -> np.ndarray:
    """
    Read one pts_*.txt:
      first line: N or 'N dims' (this dataset usually has just N)
      following N lines: dims floats per line (default 2D)
    Returns: float32 (N, dims)
    """
    with open(path, "r") as f:
        header = f.readline().strip()
        toks = header.split()
        if len(toks) == 1:
            N = int(toks[0])
            dims = 2
        else:
            # Support the "N dims" format
            N = int(toks[0])
            dims = int(toks[1])

    # Read the remaining N lines with numpy
    arr = np.loadtxt(path, dtype=np.float32, skiprows=1)
    arr = np.atleast_2d(arr)
    if arr.shape[0] != N:
        raise ValueError(f"{path}: number of rows does not match declared N (got {arr.shape[0]} vs {N})")
    if arr.shape[1] != dims:
        raise ValueError(f"{path}: dimensionality does not match declared dims (got {arr.shape[1]} vs {dims})")
    return arr  # (N, dims), float32, values should be in [0,1)


class UniGBNSampler(nn.Module, Sampleable):
    """
    Samples a batch of GBN point sets from a directory (e.g. dataset/1024_1000_original).
    - Each .txt file is one sample (usually N=1024, dims=2)
    - Unlabeled, returns (x, None)
    - When preload=True, reads all files into memory (CPU) up front, then moves them to device at sampling time
    - Optionally maps coordinates from [0,1] to [-1,1]
    """
    def __init__(
        self,
        data_dir: str,
        rotate4: bool = False,
        preload: bool = False,
        random_shuffle: bool = False,
        map_to_neg1_1: bool = True,
        pattern: str = "pts_*.txt",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.rotate4 = rotate4
        self.preload = preload
        self.random_shuffle = random_shuffle
        self.map_to_neg1_1 = map_to_neg1_1

        if data_dir.endswith(".npz"):
            npz_paths = data_dir.split()
            first_N, first_D = None, None
            total_B = 0
            self._data: List[torch.Tensor] = []
            for path in npz_paths:
                d = np.load(path, allow_pickle=True)
                P = np.asarray(d["points"])
                if P.ndim != 3:
                    raise ValueError(f"Expect 3D points (B,N,D), got shape {P.shape}. "
                                    f"Use the converter or the 3D merge script.")
                B, N, D = P.shape
                if first_N is None:
                    first_N, first_D = N, D
                else:
                    if (N, D) != (first_N, first_D):
                        raise ValueError(
                            f"Shape mismatch in {path}: {P.shape} vs "
                            f"(B, {first_N}, {first_D})"
                        )

                # Consistent with directory mode: self._data is List[Tensor(N,D)]
                # self._data = [torch.from_numpy(P[i]).to(dtype=torch.float32) for i in range(B)]
                for i in range(B):
                    t = torch.from_numpy(P[i]).to(dtype=torch.float32)  # (N,D)
                    # If 3D, apply per-sample unit-box scaling
                    if D == 3:
                        t = normalize_3d_unit_box(t, D)

                    self._data.append(t)
                total_B += B

            self.length = total_B
            self.N, self.dims = first_N, first_D
            self.files = [f"npz:{i}" for i in range(self.length)]  # placeholder
            self.preload = True  # force the preload branch
            self.register_buffer("dummy", torch.zeros(1))
            return

        # Find files & sort
        files = glob.glob(os.path.join(data_dir, pattern))
        if not files:
            raise FileNotFoundError(f"No files matched {pattern} in {data_dir}")
        self.files: List[str] = sorted(files, key=_natural_key)
        self.length: int = len(self.files)

        # Read the first file to determine N and dims
        first = _read_one_txt(self.files[0])
        self.N = first.shape[0]
        self.dims = first.shape[1]

        # Register a buffer to obtain the target device
        self.register_buffer("dummy", torch.zeros(1))

        # Preload
        if self.preload:
            self._data: List[torch.Tensor] = []
            self._data.append(torch.from_numpy(first))  # (N,dims) float32, CPU
            for p in self.files[1:]:
                arr = _read_one_txt(p)
                if arr.shape != (self.N, self.dims):
                    raise ValueError(f"Shape mismatch in {p}: {arr.shape} vs ({self.N},{self.dims})")
                self._data.append(torch.from_numpy(arr))
        else:
            self._data = None  # read on demand

    @torch.no_grad()
    def sample(self, num_samples: int, rotate4: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise RuntimeError("Not implemented!")
    
    def __len__(self):
        if not self.rotate4:
            return self.length
        else:
            return self.length // 4

    @torch.no_grad()
    def get_batch(self, indices: Union[torch.Tensor, List[int]], 
                rotate4: bool = False, zorder: bool = False):
        """
        indices: 1D LongTensor or Python list (base indices: 0..B-1)
        If rotate4=True, self._data should have shape (4B, N, dims), with every 4 forming a group:
        [i*4+0, i*4+1, i*4+2, i*4+3] correspond to the four rotated versions of the same original sample
        (rotation + Hilbert sort done offline, range [0,1])
        Returns: (x, None) where x:(B, N, dims), mapped to [-1,1]
        """
        # Normalize indices to a python list[int]
        if torch.is_tensor(indices):
            idx_list = indices.detach().cpu().tolist()
        else:
            idx_list = list(map(int, indices))

        device = self.dummy.device

        # ---- Key: when rotate4=True, first map base indices to actual (4B) indices ----
        if rotate4:
            B = len(idx_list)
            base_idx = torch.tensor(idx_list, dtype=torch.long)
            # Randomly pick an offset in {0,1,2,3} for each base sample
            offsets = torch.randint(0, 4, (B,), dtype=torch.long)
            eff_idx = (base_idx * 4 + offsets).tolist()   # actual sampling indices, still length B
        else:
            eff_idx = idx_list

        # ---- Fetch the batch along the original path (supports preload / non-preload) ----
        # print(len(self._data))
        batch = []
        if self.preload:
            # Here self._data can be:
            # - non-rotate4: shape (B, N, d)
            # - rotate4:     shape (4B, N, d) (the augmented set you preloaded)
            for i in eff_idx:
                # print(i)
                batch.append(self._data[i])  # CPU tensor (N,dims) in [0,1]
        else:
            # If not preloaded, still read from files; if rotate4=True, self.files must correspond to 4B files
            for i in eff_idx:
                arr = _read_one_txt(self.files[i])
                batch.append(torch.from_numpy(arr))  # CPU tensor (N,dims) in [0,1]

        x = torch.stack(batch, 0).to(device=device, dtype=torch.float32)  # (B,N,d), range is still [0,1]

        # ---- No rotation / Hilbert sort during the online stage; optional zorder is kept (usually unnecessary) ----
        if zorder:
            x, _ = z_order_sort(x, p=16, in_unit_square=True)  # (B,N,2), (B,N)

        # Consistent with the original logic: finally map to [-1,1]
        x = x * 2.0 - 1.0

        # Random point order (keeping the original logic)
        if self.random_shuffle:
            B, N, D = x.shape
            perm_pts = torch.argsort(torch.rand(B, N, device=x.device), dim=1)
            x = x.gather(1, perm_pts.unsqueeze(-1).expand(-1, -1, D))

        return x, None

    @property
    def num_meshes(self) -> int:
        """Attribute name for compatibility with the MeshPairAsyncLoader interface"""
        return len(self)

    @property
    def num_points(self) -> int:
        """Number of points per sample"""
        return self.N

    def compute_batch(
        self,
        idx_batch: np.ndarray,
        epoch: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute_batch method compatible with the MeshSortDataset interface.

        Returns (x0, x1):
          x0: (B, N, dims) uniform in [-1,1]^dims, float32
          x1: (B, N, dims) point cloud sampled from the dataset, float32

        epoch + step are used as a deterministic RNG seed.
        """
        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        B = idx_batch.shape[0]
        N = self.N
        D = self.dims

        # deterministic seed
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        # x0: uniform in [-1, 1]^D
        x0_np = rng.random((B, N, D), dtype=np.float32) * 2.0 - 1.0

        # x1: load from the dataset
        # Handle rotate4 mode
        if self.rotate4:
            base_idx = idx_batch
            offsets = rng.integers(0, 4, size=(B,))
            eff_idx = (base_idx * 4 + offsets).tolist()
        else:
            eff_idx = idx_batch.tolist()

        # Load data
        x1_list = []
        if self.preload:
            for i in eff_idx:
                x1_list.append(self._data[i].numpy())  # (N, dims)
        else:
            for i in eff_idx:
                arr = _read_one_txt(self.files[i])
                x1_list.append(arr)

        x1_np = np.stack(x1_list, axis=0)  # (B, N, dims) in [0, 1]

        # Map to [-1, 1]
        if self.map_to_neg1_1:
            x1_np = x1_np * 2.0 - 1.0

        # Random shuffle (point order)
        if self.random_shuffle:
            for i in range(B):
                perm = rng.permutation(N)
                x1_np[i] = x1_np[i][perm]

        return x0_np.astype(np.float32), x1_np.astype(np.float32)


def _compute_cost_row(i: int, x0_flat: np.ndarray, x1_flat: np.ndarray, B: int) -> Tuple[int, np.ndarray]:
    """
    Compute row i of the cost matrix: C[i, :] = ||x0[i] - x1[j]||^2 for all j
    x0_flat, x1_flat: (B, N*D) flattened arrays
    Returns: (i, row) where row is a (B,) array
    """
    diff = x0_flat[i] - x1_flat  # (B, N*D)
    row = np.sum(diff ** 2, axis=1)  # (B,)
    return i, row.astype(np.float32)


class PointSetMiniBatchOTDataset:
    """
    2D point cloud dataset with sample-level minibatch OT pairing.

    OT matching logic:
    - Generate B noise point sets x0 (N points each)
    - Load B data point sets x1
    - Compute a B×B cost matrix (using flattened L2 distance)
    - Find the optimal pairing with greedy/hungarian OT
    - Return the paired (x0, x1[perm])
    """

    def __init__(
        self,
        data_dir: str,
        rotate4: bool = False,
        preload: bool = True,
        random_shuffle: bool = False,
        map_to_neg1_1: bool = True,
        pattern: str = "pts_*.txt",
        ot_solver: str = "hungarian",  # "greedy" or "hungarian"
        use_multi: bool = True,
        num_workers: int = 8,
    ):
        self.data_dir = data_dir
        self.rotate4 = rotate4
        self.preload = preload
        self.random_shuffle = random_shuffle
        self.map_to_neg1_1 = map_to_neg1_1
        self.ot_solver = ot_solver
        self.use_multi = use_multi
        self.num_workers = num_workers

        # Data-loading logic reuses the UniGBNSampler pattern
        if data_dir.endswith(".npz"):
            npz_paths = data_dir.split()
            first_N, first_D = None, None
            total_B = 0
            self._data: List[np.ndarray] = []
            for path in npz_paths:
                d = np.load(path, allow_pickle=True)
                P = np.asarray(d["points"])
                if P.ndim != 3:
                    raise ValueError(f"Expect 3D points (B,N,D), got shape {P.shape}.")
                B, N, D = P.shape
                if first_N is None:
                    first_N, first_D = N, D
                else:
                    if (N, D) != (first_N, first_D):
                        raise ValueError(
                            f"Shape mismatch in {path}: {P.shape} vs (B, {first_N}, {first_D})"
                        )

                for i in range(B):
                    arr = P[i].astype(np.float32)  # (N, D)
                    self._data.append(arr)
                total_B += B

            self.length = total_B
            self.N, self.dims = first_N, first_D
            self.files = [f"npz:{i}" for i in range(self.length)]
            self.preload = True
        else:
            # Load txt files from a directory
            files = glob.glob(os.path.join(data_dir, pattern))
            if not files:
                raise FileNotFoundError(f"No files matched {pattern} in {data_dir}")
            self.files: List[str] = sorted(files, key=_natural_key)
            self.length: int = len(self.files)

            # Read the first file to determine N and dims
            first = _read_one_txt(self.files[0])
            self.N = first.shape[0]
            self.dims = first.shape[1]

            # Preload
            if self.preload:
                self._data: List[np.ndarray] = []
                self._data.append(first)  # (N, dims) float32
                for p in self.files[1:]:
                    arr = _read_one_txt(p)
                    if arr.shape != (self.N, self.dims):
                        raise ValueError(f"Shape mismatch in {p}: {arr.shape} vs ({self.N},{self.dims})")
                    self._data.append(arr)
            else:
                self._data = None

    def __len__(self):
        if not self.rotate4:
            return self.length
        else:
            return self.length // 4

    @property
    def num_meshes(self) -> int:
        """Compatibility with the async loader interface"""
        return len(self)

    @property
    def num_points(self) -> int:
        """Number of points per sample"""
        return self.N

    def compute_batch(
        self,
        idx_batch: np.ndarray,
        epoch: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute_batch with OT matching.

        1. Generate x0: (B, N, dims) uniform in [-1,1]^dims
        2. Load x1: (B, N, dims) read from the dataset
        3. Compute cost matrix C[i,j] = ||x0[i].flatten() - x1[j].flatten()||^2
        4. Get perm via solve_ot_assignment(C, mode=self.ot_solver)
        5. Return (x0, x1[perm])
        """
        from .utils import solve_ot_assignment

        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        B = idx_batch.shape[0]
        N = self.N
        D = self.dims

        # deterministic seed
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        # x0: uniform in [-1, 1]^D
        x0_np = rng.random((B, N, D), dtype=np.float32) * 2.0 - 1.0

        # x1: load from the dataset
        # Handle rotate4 mode
        if self.rotate4:
            base_idx = idx_batch
            offsets = rng.integers(0, 4, size=(B,))
            eff_idx = (base_idx * 4 + offsets).tolist()
        else:
            eff_idx = idx_batch.tolist()

        # Load data
        x1_list = []
        if self.preload:
            for i in eff_idx:
                x1_list.append(self._data[i])  # (N, dims)
        else:
            for i in eff_idx:
                arr = _read_one_txt(self.files[i])
                x1_list.append(arr)

        x1_np = np.stack(x1_list, axis=0)  # (B, N, dims) in [0, 1]

        # Map to [-1, 1]
        if self.map_to_neg1_1:
            x1_np = x1_np * 2.0 - 1.0

        # Random shuffle (point order)
        if self.random_shuffle:
            for i in range(B):
                perm = rng.permutation(N)
                x1_np[i] = x1_np[i][perm]

        # Convert to float32 and flatten
        x0_f32 = x0_np.astype(np.float32)
        x1_f32 = x1_np.astype(np.float32)

        # Flatten: (B, N, D) -> (B, N*D)
        x0_flat = x0_f32.reshape(B, -1)
        x1_flat = x1_f32.reshape(B, -1)

        # Compute the B×B cost matrix (flattened L2 distance)
        C = np.zeros((B, B), dtype=np.float32)

        if self.use_multi and self.num_workers > 1 and B > 1:
            # Use joblib multiprocessing for speedup
            results = Parallel(n_jobs=self.num_workers, backend="loky")(
                delayed(_compute_cost_row)(i, x0_flat, x1_flat, B)
                for i in range(B)
            )
            for i, row in results:
                C[i] = row
        else:
            # Single-process computation
            for i in range(B):
                diff = x0_flat[i] - x1_flat  # (B, N*D)
                C[i] = np.sum(diff ** 2, axis=1)  # (B,)

        # OT matching
        perm = solve_ot_assignment(C, mode=self.ot_solver)

        # Apply the permutation
        x1_matched = x1_f32[perm]

        return x0_f32, x1_matched


# ============== Helper functions for PointSetEqOTFMDataset ==============

def _compute_point_level_ot(x0_i: np.ndarray, x1_j: np.ndarray, solver: str = "hungarian"):
    """
    Solve point-level OT between two point sets.

    Args:
        x0_i: (N, D) source point set
        x1_j: (N, D) target point set
        solver: "hungarian" or "greedy"

    Returns:
        (cost, perm) where x1_j[perm] aligns with x0_i
    """
    from scipy.spatial.distance import cdist
    from .utils import solve_ot_assignment

    cost_matrix = cdist(x0_i, x1_j, metric='sqeuclidean')  # (N, N)
    perm = solve_ot_assignment(cost_matrix, mode=solver)
    total_cost = cost_matrix[np.arange(len(perm)), perm].sum()
    return total_cost, perm


def _compute_point_ot_pair(i: int, j: int, x0: np.ndarray, x1: np.ndarray, solver: str):
    """
    Worker function for parallel point-level OT computation.

    Args:
        i: index in x0
        j: index in x1
        x0: (B, N, D) source batch
        x1: (B, N, D) target batch
        solver: "hungarian" or "greedy"

    Returns:
        (i, j, cost, perm)
    """
    cost, perm = _compute_point_level_ot(x0[i], x1[j], solver)
    return i, j, cost, perm


def _compute_point_ot_pair_slim(i: int, j: int, x0_i: np.ndarray, x1_j: np.ndarray, solver: str):
    """
    Slim version: only receives the needed slices to avoid pickling entire batch arrays.

    Args:
        i: index for result tracking
        j: index for result tracking
        x0_i: (N, D) single source point set
        x1_j: (N, D) single target point set
        solver: "hungarian" or "greedy"

    Returns:
        (i, j, cost, perm)
    """
    cost, perm = _compute_point_level_ot(x0_i, x1_j, solver)
    return i, j, cost, perm


class PointSetEqOTFMDataset:
    """
    2D/3D point cloud dataset with two-level OT matching for Equivariant OT flow matching.

    Difference from PointSetMiniBatchOTDataset:
    - PointSetMiniBatchOTDataset uses flattened L2 distance for batch-level OT
    - This class first aligns at the point level via permutation, then does batch-level OT

    Two-level OT matching logic:
    1. For each pair (x0[i], x1[j]), find the point-level permutation:
       s*[i,j] = argmin_{s in S(N)} ||x0[i] - x1[j][s]||^2
    2. Build the cost matrix: M[i,j] = cost after alignment
    3. Batch-level OT: sigma* = argmin sum_i M[i, sigma(i)]
    4. Return (x0, aligned_x1[sigma*])
    """

    def __init__(
        self,
        data_dir: str,
        rotate4: bool = False,
        preload: bool = True,
        random_shuffle: bool = False,
        map_to_neg1_1: bool = True,
        pattern: str = "pts_*.txt",
        point_ot_solver: str = "greedy",  # point-level OT solver
        batch_ot_solver: str = "hungarian",     # batch-level OT solver
        use_multi: bool = True,
        num_workers: int = 8,
    ):
        self.data_dir = data_dir
        self.rotate4 = rotate4
        self.preload = preload
        self.random_shuffle = random_shuffle
        self.map_to_neg1_1 = map_to_neg1_1
        self.point_ot_solver = point_ot_solver
        self.batch_ot_solver = batch_ot_solver
        self.use_multi = use_multi
        self.num_workers = num_workers

        # Data-loading logic reuses the PointSetMiniBatchOTDataset pattern
        if data_dir.endswith(".npz"):
            npz_paths = data_dir.split()
            first_N, first_D = None, None
            total_B = 0
            self._data: List[np.ndarray] = []
            for path in npz_paths:
                d = np.load(path, allow_pickle=True)
                P = np.asarray(d["points"])
                if P.ndim != 3:
                    raise ValueError(f"Expect 3D points (B,N,D), got shape {P.shape}.")
                B, N, D = P.shape
                if first_N is None:
                    first_N, first_D = N, D
                else:
                    if (N, D) != (first_N, first_D):
                        raise ValueError(
                            f"Shape mismatch in {path}: {P.shape} vs (B, {first_N}, {first_D})"
                        )

                for i in range(B):
                    arr = P[i].astype(np.float32)  # (N, D)
                    self._data.append(arr)
                total_B += B

            self.length = total_B
            self.N, self.dims = first_N, first_D
            self.files = [f"npz:{i}" for i in range(self.length)]
            self.preload = True
        else:
            # Load txt files from a directory
            files = glob.glob(os.path.join(data_dir, pattern))
            if not files:
                raise FileNotFoundError(f"No files matched {pattern} in {data_dir}")
            self.files: List[str] = sorted(files, key=_natural_key)
            self.length: int = len(self.files)

            # Read the first file to determine N and dims
            first = _read_one_txt(self.files[0])
            self.N = first.shape[0]
            self.dims = first.shape[1]

            # Preload
            if self.preload:
                self._data: List[np.ndarray] = []
                self._data.append(first)  # (N, dims) float32
                for p in self.files[1:]:
                    arr = _read_one_txt(p)
                    if arr.shape != (self.N, self.dims):
                        raise ValueError(f"Shape mismatch in {p}: {arr.shape} vs ({self.N},{self.dims})")
                    self._data.append(arr)
            else:
                self._data = None

    def __len__(self):
        if not self.rotate4:
            return self.length
        else:
            return self.length // 4

    @property
    def num_meshes(self) -> int:
        """Compatibility with the async loader interface"""
        return len(self)

    @property
    def num_points(self) -> int:
        """Number of points per sample"""
        return self.N

    def compute_batch(
        self,
        idx_batch: np.ndarray,
        epoch: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute_batch with Equivariant OT matching.

        1. Generate x0: (B, N, dims) uniform in [-1,1]^dims
        2. Load x1: (B, N, dims) read from the dataset
        3. Point-level OT: for each pair (i,j), find the permutation aligning x1[j] to x0[i]
        4. Build the batch cost matrix M[i,j]
        5. Batch-level OT: find the optimal sample pairing sigma*
        6. Return (x0, x1_aligned) where x1_aligned[i] = x1[sigma*(i)][point_perm[i, sigma*(i)]]
        """
        from .utils import solve_ot_assignment

        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        B = idx_batch.shape[0]
        N = self.N
        D = self.dims

        # deterministic seed
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        # x0: uniform in [-1, 1]^D
        x0_np = rng.random((B, N, D), dtype=np.float32) * 2.0 - 1.0

        # x1: load from the dataset
        # Handle rotate4 mode
        if self.rotate4:
            base_idx = idx_batch
            offsets = rng.integers(0, 4, size=(B,))
            eff_idx = (base_idx * 4 + offsets).tolist()
        else:
            eff_idx = idx_batch.tolist()

        # Load data
        x1_list = []
        if self.preload:
            for i in eff_idx:
                x1_list.append(self._data[i])  # (N, dims)
        else:
            for i in eff_idx:
                arr = _read_one_txt(self.files[i])
                x1_list.append(arr)

        x1_np = np.stack(x1_list, axis=0)  # (B, N, dims) in [0, 1]

        # Map to [-1, 1]
        if self.map_to_neg1_1:
            x1_np = x1_np * 2.0 - 1.0

        # Random shuffle (point order)
        if self.random_shuffle:
            for i in range(B):
                perm = rng.permutation(N)
                x1_np[i] = x1_np[i][perm]

        # Convert to float32
        x0_f32 = x0_np.astype(np.float32)
        x1_f32 = x1_np.astype(np.float32)

        # ============== Two-level OT matching ==============
        # Step 1: Point-level OT, compute the B x B cost matrix and the corresponding permutations
        M = np.zeros((B, B), dtype=np.float32)
        perms = {}  # (i, j) -> point permutation array

        if self.use_multi and self.num_workers > 1 and B > 1:
            # Use joblib multiprocessing for speedup, passing slices to avoid pickling whole arrays
            results = Parallel(n_jobs=self.num_workers, backend="loky")(
                delayed(_compute_point_ot_pair_slim)(i, j, x0_f32[i], x1_f32[j], self.point_ot_solver)
                for i in range(B) for j in range(B)
            )
            for i, j, cost, perm in results:
                M[i, j] = cost
                perms[(i, j)] = perm
        else:
            # Single-process computation
            for i in range(B):
                for j in range(B):
                    cost, perm = _compute_point_level_ot(x0_f32[i], x1_f32[j], self.point_ot_solver)
                    M[i, j] = cost
                    perms[(i, j)] = perm

        # Step 2: Batch-level OT
        batch_perm = solve_ot_assignment(M, mode=self.batch_ot_solver)

        # Step 3: Build the aligned x1
        x1_aligned = np.zeros_like(x0_f32)
        for i in range(B):
            j = batch_perm[i]
            point_perm = perms[(i, j)]
            x1_aligned[i] = x1_f32[j][point_perm]

        return x0_f32, x1_aligned


class MeshEqOTFMDataset:
    """
    PCD point cloud dataset with two-level OT matching for Equivariant OT flow matching.

    Combines the PCD loading logic of MeshOTDataset with the two-level OT matching of PointSetEqOTFMDataset.
    Assumes the dataset is always in PCD format (not mesh), handling only point cloud data.

    Two-level OT matching logic:
    1. For each pair (x0[i], x1[j]), find the point-level permutation:
       s*[i,j] = argmin_{s in S(N)} ||x0[i] - x1[j][s]||^2
    2. Build the cost matrix: M[i,j] = cost after alignment
    3. Batch-level OT: sigma* = argmin sum_i M[i, sigma(i)]
    4. Return (x0, aligned_x1[sigma*])
    """

    def __init__(
        self,
        mesh_root: str,
        num_points: int,
        split: str = "train",
        use_multi: bool = True,
        num_workers: int = 8,
        # x0 initialization mode
        use_sphere: bool = False,
        use_shell: bool = False,
        # Normalization mode (pick one of three, mutually exclusive)
        normalize_globally: bool = False,
        recenter_per_shape: bool = False,
        normalize_per_shape_maxabs: bool = False,
        # Global parameters for normalize_globally (must be passed in for val/test)
        all_points_mean: np.ndarray = None,
        all_points_std: np.ndarray = None,
        global_scale: float = None,
        # OT solver parameters
        point_ot_solver: str = "greedy",  # point-level OT solver
        batch_ot_solver: str = "hungarian",     # batch-level OT solver
    ):
        self.mesh_root = Path(mesh_root).expanduser().resolve()
        self._num_points = int(num_points)
        self.split = split
        self.use_multi = bool(use_multi)
        self.num_workers = int(num_workers)
        self.use_sphere = use_sphere
        self.use_shell = use_shell
        self.point_ot_solver = point_ot_solver
        self.batch_ot_solver = batch_ot_solver

        # Normalization parameters
        self.normalize_globally = normalize_globally
        self.recenter_per_shape = recenter_per_shape
        self.normalize_per_shape_maxabs = normalize_per_shape_maxabs
        self._all_points_mean = all_points_mean
        self._all_points_std = all_points_std
        self._global_scale = global_scale

        # Ensure the three normalization modes are mutually exclusive
        norm_flags = [normalize_globally, recenter_per_shape, normalize_per_shape_maxabs]
        if sum(norm_flags) > 1:
            raise ValueError(
                "Only one of normalize_globally, recenter_per_shape, normalize_per_shape_maxabs can be True."
            )
        # If all are False, default to normalize_per_shape_maxabs
        if sum(norm_flags) == 0:
            self.normalize_per_shape_maxabs = True

        # Load point cloud data
        self.pointclouds_list: List[np.ndarray] = []
        self._load_pointclouds()

        if use_sphere:
            print("[MeshEqOTFMDataset] Using sphere init")
        if use_shell:
            print("[MeshEqOTFMDataset] Using shell init")
        if not use_sphere and not use_shell:
            print("[MeshEqOTFMDataset] Using unit box init")

    def _load_pointclouds(self):
        """
        Load point cloud data from a preprocessed NPZ file.

        NPZ file path: {mesh_root}/{split}.npz
        NPZ structure: {'pointclouds': (B, N, 3) array}
        """
        npz_path = self.mesh_root / f"{self.split}.npz"
        if not npz_path.exists():
            raise RuntimeError(f"Point cloud NPZ file not found: {npz_path}")

        print(f"[MeshEqOTFMDataset] Loading {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        raw_pcds = data['pointclouds']  # (B, N, 3)
        print(f"[MeshEqOTFMDataset] Found {len(raw_pcds)} point clouds, shape per pcd: {raw_pcds[0].shape}")

        # Normalization: three mutually exclusive modes
        if self.normalize_globally:
            # Mode 1: Global normalization (LION style: global mean + global std) + scale to [-1,1]
            if self._all_points_mean is not None and self._all_points_std is not None:
                # Use the provided mean/std (val/test scenario)
                all_points_mean = self._all_points_mean
                all_points_std = self._all_points_std
                global_scale = self._global_scale
                if global_scale is None:
                    raise ValueError("[MeshEqOTFMDataset] normalize_globally requires global_scale for val/test. "
                                     "Please pass the global_scale from the training dataset.")
            else:
                # Compute the global mean/std (train scenario)
                all_points = np.stack(raw_pcds, axis=0)  # (M, N, 3)
                all_points_mean = all_points.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
                all_points_std = all_points.reshape(-1).std().reshape(1, 1, 1)

                # First apply (x - mean) / std to all points, then compute the global max_abs
                all_points_norm = (all_points - all_points_mean) / all_points_std
                global_scale = np.abs(all_points_norm).max()
                global_scale = max(global_scale, 1e-8)  # avoid division by zero

                # Save for external use
                self.all_points_mean = all_points_mean
                self.all_points_std = all_points_std
                self.global_scale = global_scale
                print(f"[MeshEqOTFMDataset] Global normalization: mean={all_points_mean.flatten()}, "
                      f"std={all_points_std.item():.6f}, global_scale={global_scale:.6f}")

            # Apply global normalization: ((x - mean) / std) / global_scale -> [-1, 1]
            for pcd in raw_pcds:
                pcd_norm = (pcd - all_points_mean.reshape(1, 3)) / all_points_std.item() / global_scale
                self.pointclouds_list.append(pcd_norm.astype(np.float32))

        elif self.recenter_per_shape:
            # Mode 2: Per-shape recenter (LION default: bbox center + bbox half-extent)
            print("[MeshEqOTFMDataset] Using recenter_per_shape normalization (bbox center + half-extent)")
            for pcd in raw_pcds:
                pcd = pcd.astype(np.float64)
                p_min = pcd.min(axis=0, keepdims=True)  # (1, 3)
                p_max = pcd.max(axis=0, keepdims=True)  # (1, 3)
                p_center = (p_max + p_min) / 2.0       # (1, 3)
                p_half_extent = ((p_max - p_min) / 2.0).max()  # scalar
                p_half_extent = max(p_half_extent, 1e-8)  # avoid division by zero
                pcd_norm = (pcd - p_center) / p_half_extent
                self.pointclouds_list.append(pcd_norm.astype(np.float32))

        elif self.normalize_per_shape_maxabs:
            # Mode 3: Per-shape max-abs normalization
            print("[MeshEqOTFMDataset] Using normalize_per_shape_maxabs normalization")
            for pcd in raw_pcds:
                pcd_norm = normalize_vertices_minus1_1(pcd)
                self.pointclouds_list.append(pcd_norm)
        else:
            raise RuntimeError("No normalization mode selected")

        print(f"[MeshEqOTFMDataset] Loaded {len(self.pointclouds_list)} pointclouds.")

    @property
    def num_meshes(self) -> int:
        """Compatibility with the async loader interface"""
        return len(self.pointclouds_list)

    @property
    def num_points(self) -> int:
        """Number of points per sample"""
        return self._num_points

    def compute_batch(
        self,
        idx_batch: np.ndarray,
        epoch: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute_batch with Equivariant OT matching.

        1. Generate x0: (B, N, 3) according to use_sphere/use_shell/box
        2. Load x1: (B, N, 3) read from the dataset and subsampled
        3. Point-level OT: for each pair (i,j), find the permutation aligning x1[j] to x0[i]
        4. Build the batch cost matrix M[i,j]
        5. Batch-level OT: find the optimal sample pairing sigma*
        6. Return (x0, x1_aligned) where x1_aligned[i] = x1[sigma*(i)][point_perm[i, sigma*(i)]]
        """
        from .utils import solve_ot_assignment

        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        B = idx_batch.shape[0]
        N = self._num_points

        # deterministic seed
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        # Recommended to be mutually exclusive (optional)
        if self.use_sphere and self.use_shell:
            raise ValueError("use_sphere and use_shell cannot both be True.")

        # ============ Generate x0 ============
        r_min = np.sqrt(2.0)
        r_max = float(1.7)

        if self.use_shell:
            # Shell distribution
            eps = np.float32(1e-12)

            # direction: normalized random
            dir_np = rng.standard_normal((B, N, 3), dtype=np.float32)
            norm2 = (dir_np * dir_np).sum(axis=-1, keepdims=True)
            np.maximum(norm2, eps, out=norm2)
            np.sqrt(norm2, out=norm2)
            dir_np /= norm2

            # radius: uniform in volume between shells
            rmin = np.float32(r_min)
            rmax = np.float32(r_max)
            rmin3 = rmin * rmin * rmin
            rmax3 = rmax * rmax * rmax
            span3 = rmax3 - rmin3

            r = rng.random((B, N, 1), dtype=np.float32)
            r *= span3
            r += rmin3
            np.cbrt(r, out=r)

            x0_np = dir_np * r  # (B, N, 3)

        elif self.use_sphere:
            # Sphere surface
            x0_np = rng.normal(size=(B, N, 3)).astype(np.float32, copy=False)
            norm = np.linalg.norm(x0_np, axis=-1, keepdims=True)
            x0_np = x0_np / np.maximum(norm, 1e-12)
            x0_np = x0_np * np.sqrt(2.0)

        else:
            # Box: uniform in [-1,1]^3
            x0_np = rng.random((B, N, 3), dtype=np.float32)
            x0_np = x0_np * 2.0 - 1.0

        # ============ Load x1: subsample from pointclouds ============
        x1_list = []
        for mi in idx_batch:
            pcd = self.pointclouds_list[int(mi)]  # (N_total, 3)
            # subsample N points
            indices = rng.choice(pcd.shape[0], size=N, replace=False)
            x1_list.append(pcd[indices])

        x1_np = np.stack(x1_list, axis=0)  # (B, N, 3)

        # Convert to float32
        x0_f32 = x0_np.astype(np.float32)
        x1_f32 = x1_np.astype(np.float32)

        # ============== Two-level OT matching ==============
        # Step 1: Point-level OT, compute the B x B cost matrix and the corresponding permutations
        M = np.zeros((B, B), dtype=np.float32)
        perms = {}  # (i, j) -> point permutation array

        if self.use_multi and self.num_workers > 1 and B > 1:
            # Use joblib multiprocessing for speedup, passing slices to avoid pickling whole arrays
            results = Parallel(n_jobs=self.num_workers, backend="loky")(
                delayed(_compute_point_ot_pair_slim)(i, j, x0_f32[i], x1_f32[j], self.point_ot_solver)
                for i in range(B) for j in range(B)
            )
            for i, j, cost, perm in results:
                M[i, j] = cost
                perms[(i, j)] = perm
        else:
            # Single-process computation
            for i in range(B):
                for j in range(B):
                    cost, perm = _compute_point_level_ot(x0_f32[i], x1_f32[j], self.point_ot_solver)
                    M[i, j] = cost
                    perms[(i, j)] = perm

        # Step 2: Batch-level OT
        batch_perm = solve_ot_assignment(M, mode=self.batch_ot_solver)

        # Step 3: Build the aligned x1
        x1_aligned = np.zeros_like(x0_f32)
        for i in range(B):
            j = batch_perm[i]
            point_perm = perms[(i, j)]
            x1_aligned[i] = x1_f32[j][point_perm]

        return x0_f32, x1_aligned


class MeshMiniBatchOTDataset:
    """
    PCD point cloud dataset with sample-level minibatch OT pairing.

    Difference from MeshEqOTFMDataset:
    - MeshEqOTFMDataset uses two-level OT (point-level alignment first, then batch-level OT)
    - This class does only batch-level OT (using flattened L2 distance), no point-level permutation

    OT matching logic:
    - Generate B noise point sets x0 (N points each)
    - Load B data point sets x1
    - Compute a B×B cost matrix (using flattened L2 distance)
    - Find the optimal pairing with greedy/hungarian OT
    - Return the paired (x0, x1[perm])
    """

    def __init__(
        self,
        mesh_root: str,
        num_points: int,
        split: str = "train",
        use_multi: bool = True,
        num_workers: int = 8,
        # x0 initialization mode
        use_sphere: bool = False,
        use_shell: bool = False,
        # Normalization mode (pick one of three, mutually exclusive)
        normalize_globally: bool = False,
        recenter_per_shape: bool = False,
        normalize_per_shape_maxabs: bool = False,
        # Global parameters for normalize_globally (must be passed in for val/test)
        all_points_mean: np.ndarray = None,
        all_points_std: np.ndarray = None,
        global_scale: float = None,
        # OT solver parameters (only batch-level needed)
        ot_solver: str = "greedy",  # "greedy" or "hungarian"
    ):
        self.mesh_root = Path(mesh_root).expanduser().resolve()
        self._num_points = int(num_points)
        self.split = split
        self.use_multi = bool(use_multi)
        self.num_workers = int(num_workers)
        self.use_sphere = use_sphere
        self.use_shell = use_shell
        self.ot_solver = ot_solver

        # Normalization parameters
        self.normalize_globally = normalize_globally
        self.recenter_per_shape = recenter_per_shape
        self.normalize_per_shape_maxabs = normalize_per_shape_maxabs
        self._all_points_mean = all_points_mean
        self._all_points_std = all_points_std
        self._global_scale = global_scale

        # Ensure the three normalization modes are mutually exclusive
        norm_flags = [normalize_globally, recenter_per_shape, normalize_per_shape_maxabs]
        if sum(norm_flags) > 1:
            raise ValueError(
                "Only one of normalize_globally, recenter_per_shape, normalize_per_shape_maxabs can be True."
            )
        # If all are False, default to normalize_per_shape_maxabs
        if sum(norm_flags) == 0:
            self.normalize_per_shape_maxabs = True

        # Load point cloud data
        self.pointclouds_list: List[np.ndarray] = []
        self._load_pointclouds()

        if use_sphere:
            print("[MeshMiniBatchOTDataset] Using sphere init")
        if use_shell:
            print("[MeshMiniBatchOTDataset] Using shell init")
        if not use_sphere and not use_shell:
            print("[MeshMiniBatchOTDataset] Using unit box init")

    def _load_pointclouds(self):
        """
        Load point cloud data from a preprocessed NPZ file.

        NPZ file path: {mesh_root}/{split}.npz
        NPZ structure: {'pointclouds': (B, N, 3) array}
        """
        npz_path = self.mesh_root / f"{self.split}.npz"
        if not npz_path.exists():
            raise RuntimeError(f"Point cloud NPZ file not found: {npz_path}")

        print(f"[MeshMiniBatchOTDataset] Loading {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        raw_pcds = data['pointclouds']  # (B, N, 3)
        print(f"[MeshMiniBatchOTDataset] Found {len(raw_pcds)} point clouds, shape per pcd: {raw_pcds[0].shape}")

        # Normalization: three mutually exclusive modes
        if self.normalize_globally:
            # Mode 1: Global normalization (LION style: global mean + global std) + scale to [-1,1]
            if self._all_points_mean is not None and self._all_points_std is not None:
                # Use the provided mean/std (val/test scenario)
                all_points_mean = self._all_points_mean
                all_points_std = self._all_points_std
                global_scale = self._global_scale
                if global_scale is None:
                    raise ValueError("[MeshMiniBatchOTDataset] normalize_globally requires global_scale for val/test. "
                                     "Please pass the global_scale from the training dataset.")
            else:
                # Compute the global mean/std (train scenario)
                all_points = np.stack(raw_pcds, axis=0)  # (M, N, 3)
                all_points_mean = all_points.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
                all_points_std = all_points.reshape(-1).std().reshape(1, 1, 1)

                # First apply (x - mean) / std to all points, then compute the global max_abs
                all_points_norm = (all_points - all_points_mean) / all_points_std
                global_scale = np.abs(all_points_norm).max()
                global_scale = max(global_scale, 1e-8)  # avoid division by zero

                # Save for external use
                self.all_points_mean = all_points_mean
                self.all_points_std = all_points_std
                self.global_scale = global_scale
                print(f"[MeshMiniBatchOTDataset] Global normalization: mean={all_points_mean.flatten()}, "
                      f"std={all_points_std.item():.6f}, global_scale={global_scale:.6f}")

            # Apply global normalization: ((x - mean) / std) / global_scale -> [-1, 1]
            for pcd in raw_pcds:
                pcd_norm = (pcd - all_points_mean.reshape(1, 3)) / all_points_std.item() / global_scale
                self.pointclouds_list.append(pcd_norm.astype(np.float32))

        elif self.recenter_per_shape:
            # Mode 2: Per-shape recenter (LION default: bbox center + bbox half-extent)
            print("[MeshMiniBatchOTDataset] Using recenter_per_shape normalization (bbox center + half-extent)")
            for pcd in raw_pcds:
                pcd = pcd.astype(np.float64)
                p_min = pcd.min(axis=0, keepdims=True)  # (1, 3)
                p_max = pcd.max(axis=0, keepdims=True)  # (1, 3)
                p_center = (p_max + p_min) / 2.0       # (1, 3)
                p_half_extent = ((p_max - p_min) / 2.0).max()  # scalar
                p_half_extent = max(p_half_extent, 1e-8)  # avoid division by zero
                pcd_norm = (pcd - p_center) / p_half_extent
                self.pointclouds_list.append(pcd_norm.astype(np.float32))

        elif self.normalize_per_shape_maxabs:
            # Mode 3: Per-shape max-abs normalization
            print("[MeshMiniBatchOTDataset] Using normalize_per_shape_maxabs normalization")
            for pcd in raw_pcds:
                pcd_norm = normalize_vertices_minus1_1(pcd)
                self.pointclouds_list.append(pcd_norm)
        else:
            raise RuntimeError("No normalization mode selected")

        print(f"[MeshMiniBatchOTDataset] Loaded {len(self.pointclouds_list)} pointclouds.")

    @property
    def num_meshes(self) -> int:
        """Compatibility with the async loader interface"""
        return len(self.pointclouds_list)

    @property
    def num_points(self) -> int:
        """Number of points per sample"""
        return self._num_points

    def compute_batch(
        self,
        idx_batch: np.ndarray,
        epoch: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute_batch with minibatch OT matching.

        1. Generate x0: (B, N, 3) according to use_sphere/use_shell/box
        2. Load x1: (B, N, 3) read from the dataset and subsampled
        3. Compute cost matrix C[i,j] = ||x0[i].flatten() - x1[j].flatten()||^2
        4. Get perm via solve_ot_assignment(C, mode=self.ot_solver)
        5. Return (x0, x1[perm])
        """
        from .utils import solve_ot_assignment

        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        B = idx_batch.shape[0]
        N = self._num_points

        # deterministic seed
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        # Recommended to be mutually exclusive (optional)
        if self.use_sphere and self.use_shell:
            raise ValueError("use_sphere and use_shell cannot both be True.")

        # ============ Generate x0 ============
        r_min = np.sqrt(2.0)
        r_max = float(1.7)

        if self.use_shell:
            # Shell distribution
            eps = np.float32(1e-12)

            # direction: normalized random
            dir_np = rng.standard_normal((B, N, 3), dtype=np.float32)
            norm2 = (dir_np * dir_np).sum(axis=-1, keepdims=True)
            np.maximum(norm2, eps, out=norm2)
            np.sqrt(norm2, out=norm2)
            dir_np /= norm2

            # radius: uniform in volume between shells
            rmin = np.float32(r_min)
            rmax = np.float32(r_max)
            rmin3 = rmin * rmin * rmin
            rmax3 = rmax * rmax * rmax
            span3 = rmax3 - rmin3

            r = rng.random((B, N, 1), dtype=np.float32)
            r *= span3
            r += rmin3
            np.cbrt(r, out=r)

            x0_np = dir_np * r  # (B, N, 3)

        elif self.use_sphere:
            # Sphere surface
            x0_np = rng.normal(size=(B, N, 3)).astype(np.float32, copy=False)
            norm = np.linalg.norm(x0_np, axis=-1, keepdims=True)
            x0_np = x0_np / np.maximum(norm, 1e-12)
            x0_np = x0_np * np.sqrt(2.0)

        else:
            # Box: uniform in [-1,1]^3
            x0_np = rng.random((B, N, 3), dtype=np.float32)
            x0_np = x0_np * 2.0 - 1.0

        # ============ Load x1: subsample from pointclouds ============
        x1_list = []
        for mi in idx_batch:
            pcd = self.pointclouds_list[int(mi)]  # (N_total, 3)
            # subsample N points
            indices = rng.choice(pcd.shape[0], size=N, replace=False)
            x1_list.append(pcd[indices])

        x1_np = np.stack(x1_list, axis=0)  # (B, N, 3)

        # Convert to float32 and flatten
        x0_f32 = x0_np.astype(np.float32)
        x1_f32 = x1_np.astype(np.float32)

        # Flatten: (B, N, 3) -> (B, N*3)
        x0_flat = x0_f32.reshape(B, -1)
        x1_flat = x1_f32.reshape(B, -1)

        # Compute the B×B cost matrix (flattened L2 distance)
        C = np.zeros((B, B), dtype=np.float32)

        if self.use_multi and self.num_workers > 1 and B > 1:
            # Use joblib multiprocessing for speedup
            results = Parallel(n_jobs=self.num_workers, backend="loky")(
                delayed(_compute_cost_row)(i, x0_flat, x1_flat, B)
                for i in range(B)
            )
            for i, row in results:
                C[i] = row
        else:
            # Single-process computation
            for i in range(B):
                diff = x0_flat[i] - x1_flat  # (B, N*3)
                C[i] = np.sum(diff ** 2, axis=1)  # (B,)

        # OT matching
        perm = solve_ot_assignment(C, mode=self.ot_solver)

        # Apply the permutation
        x1_matched = x1_f32[perm]

        return x0_f32, x1_matched


class MeshOTDataset:
    """
    MeshOTDataset

    Similar to MeshSortDataset, but adds an OT-matching step:

      - x0: (B, N, 3), uniformly random samples in [-1,1]^3
      - x1: (B, N, 3), Poisson-disk samples on the mesh surface
      - For each batch element i:
          * First build a Hermite-style cost matrix C(x0[i], x1[i], n1[i])
          * Then do a linear assignment (OT / approximate OT),
            using the resulting perm to reorder x1[i],
            so that the output x1[i] is already in one-to-one correspondence with x0[i].

    Usage example:

        dataset = MeshOTDataset(
            mesh_root="path/to/meshes",
            num_points=2048,
            use_multi=True,
            num_workers=8,
            device="cpu",
            poisson_init_factor=4.0,
            ot_solver="greedy",      # or "hungarian"
            lambda_orient=0.2,
        )

        x0_np, x1_np = dataset.compute_batch(idx_batch, epoch, step)

    Note:
      - The output x0_np, x1_np are all (B, N, 3) float32 point clouds.
      - Normals are only used internally to build the cost; they are not concatenated into the output feature dimension.
    """

    def __init__(
        self,
        mesh_root: str,
        num_points: int,
        use_multi: bool = True,
        num_workers: int = 8,
        device: str = "cpu",
        # Kept for "interface similarity": the hilbert_p parameter is unused in this Dataset
        hilbert_p: int = 10,
        poisson_init_factor: float = 3.0,
        # New: OT / Hermite-related parameters
        use_normal: bool = True,
        use_poisson: bool = False,
        use_sphere: bool = False,
        use_shell: bool = False,
        zero_t0: bool = False,
        ot_solver: str = "greedy",     # "greedy" or "hungarian"
        hermite_degree: str = "quadratic",
        lambda_orient: float = 0.2,    # orientation penalty weight in the cost
        # Normalization mode (pick one of three, mutually exclusive)
        normalize_globally: bool = False,
        recenter_per_shape: bool = False,
        normalize_per_shape_maxabs: bool = False,
        all_points_mean: np.ndarray = None,
        all_points_std: np.ndarray = None,
        global_scale: float = None,  # global scale for normalize_globally (must be passed in for val/test)
        # Point cloud dataset mode
        split: str = "train",
        dataset_is_pcd: bool = False,
        dataset_is_mesh_npz: bool = False,  # load mesh from a preprocessed NPZ file
    ):
        self.mesh_root = Path(mesh_root).expanduser().resolve()
        self.num_points = int(num_points)
        self.use_multi = bool(use_multi)
        self.num_workers = int(num_workers)
        self.use_normal = use_normal
        self.use_poisson = use_poisson
        self.use_sphere = use_sphere
        self.use_shell = use_shell
        self.device = torch.device(device)
        self.hilbert_p = int(hilbert_p)          # unused in this class, just a placeholder
        self.poisson_init_factor = float(poisson_init_factor)
        self.ot_solver = ot_solver
        self.hermite_degree= hermite_degree
        self.lambda_orient = float(lambda_orient)
        self.zero_t0 = zero_t0
        self.split = split
        self.dataset_is_pcd = dataset_is_pcd
        self.dataset_is_mesh_npz = dataset_is_mesh_npz

        # Normalization parameters
        self.normalize_globally = normalize_globally
        self.recenter_per_shape = recenter_per_shape
        self.normalize_per_shape_maxabs = normalize_per_shape_maxabs
        self._all_points_mean = all_points_mean
        self._all_points_std = all_points_std
        self._global_scale = global_scale  # global scale for normalize_globally

        # Ensure the three normalization modes are mutually exclusive
        norm_flags = [normalize_globally, recenter_per_shape, normalize_per_shape_maxabs]
        if sum(norm_flags) > 1:
            raise ValueError(
                "Only one of normalize_globally, recenter_per_shape, normalize_per_shape_maxabs can be True."
            )
        # If all are False, default to normalize_per_shape_maxabs
        if sum(norm_flags) == 0:
            self.normalize_per_shape_maxabs = True

        # Load data according to the mode
        if self.dataset_is_pcd:
            # PCD mode: point cloud datasets never have normal info; no hermite curve, no poisson sample
            self.use_normal = False
            self.use_poisson = False
            self.pointclouds_list: List[np.ndarray] = []
            self._load_pointclouds()
        elif self.dataset_is_mesh_npz:
            # Mesh NPZ mode: load mesh from a preprocessed NPZ file
            self.mesh_paths: List[Path] = []
            self.vertices_list: List[np.ndarray] = []
            self.faces_list: List[np.ndarray] = []
            self._load_meshes_from_npz()
        else:
            # Raw PLY file mode: stores vertices & faces after normalization to [-1,1]^3
            self.mesh_paths: List[Path] = []
            self.vertices_list: List[np.ndarray] = []
            self.faces_list: List[np.ndarray] = []
            self._load_meshes()

        # For compatibility with the MeshNearPointPairDataset interface (optional)
        self._epoch_batches_x0: List[torch.Tensor] = []
        self._epoch_batches_x1: List[torch.Tensor] = []
        self._steps_per_epoch: int = 0

        if use_sphere:
            print("Using sphere init")
        if use_shell:
            print("Using shell init")
        if not use_sphere and not use_shell:
            print("Using unit box init")
    # ----------------- Loading & normalization -----------------

    def _load_meshes(self):
        # Same as MeshSortDataset: filter meshes using the split column in all.csv
        ply_files = find_ply_files_from_all_csv(self.mesh_root, split=self.split)

        if not ply_files:
            raise RuntimeError(f"No .ply files found under: {self.mesh_root}")

        print(f"[MeshOTDataset] Found {len(ply_files)} .ply files, loading...")

        # First collect all raw vertices and faces
        raw_vertices = []
        raw_faces = []
        raw_paths = []

        for p in tqdm(ply_files, desc="Loading meshes (OT)"):
            try:
                mesh = o3d.io.read_triangle_mesh(str(p))
            except Exception as e:
                print(f"[WARN] Failed to read mesh {p}: {e}")
                continue

            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                print(f"[WARN] Empty mesh in {p}, skip.")
                continue

            # clean mesh
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_unreferenced_vertices()
            mesh.remove_non_manifold_edges()
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                print(f"[WARN] Mesh in {p} became empty after cleaning, skip.")
                continue

            if self.use_normal:
                # try:
                #     a = mesh.orient_triangles()
                # except Exception as e:
                #     print(f"[WARN] orient_triangles() failed on {p}: {e}")
                mesh.compute_vertex_normals()

            V = np.asarray(mesh.vertices, dtype=np.float64)
            F = np.asarray(mesh.triangles, dtype=np.int32)

            raw_vertices.append(V)
            raw_faces.append(F)
            raw_paths.append(p)

        if not raw_vertices:
            raise RuntimeError("[MeshOTDataset] No valid meshes loaded after filtering.")

        # Normalization: three mutually exclusive modes
        if self.normalize_globally:
            # Mode 1: Global normalization (LION style: global mean + global std) + scale to [-1,1]
            if self._all_points_mean is not None and self._all_points_std is not None:
                all_points_mean = np.asarray(self._all_points_mean).reshape(1, 3)
                all_points_std = float(np.asarray(self._all_points_std))
                # Use the provided global_scale if present; otherwise it must be recomputed
                global_scale = getattr(self, '_global_scale', None)
                if global_scale is None:
                    # global_scale must be provided in the val/test scenario
                    raise ValueError("[MeshOTDataset] normalize_globally requires global_scale for val/test. "
                                     "Please pass the global_scale from the training dataset.")
            else:
                all_verts = np.concatenate(raw_vertices, axis=0)  # (total_verts, 3)
                all_points_mean = all_verts.reshape(-1, 3).mean(axis=0).reshape(1, 3)
                all_points_std = all_verts.reshape(-1).std()

                # First apply (x - mean) / std to all vertices, then compute the global max_abs
                all_verts_norm = (all_verts - all_points_mean) / all_points_std
                global_scale = np.abs(all_verts_norm).max()
                global_scale = max(global_scale, 1e-8)  # avoid division by zero

                self.all_points_mean = all_points_mean  # (1,3)
                self.all_points_std = all_points_std    # scalar
                self.global_scale = global_scale        # scalar: used to scale to [-1,1]
                print(f"[MeshOTDataset] Global normalization: mean={all_points_mean.flatten()}, std={all_points_std:.6f}, global_scale={global_scale:.6f}")

            # Apply global normalization: ((x - mean) / std) / global_scale -> [-1, 1]
            for V, F, p in zip(raw_vertices, raw_faces, raw_paths):
                Vn = (V - all_points_mean) / all_points_std / global_scale
                self.mesh_paths.append(p)
                self.vertices_list.append(Vn.astype(np.float32))
                self.faces_list.append(F)

        elif self.recenter_per_shape:
            # Mode 2: Per-shape recenter (LION default: bbox center + bbox half-extent)
            # mean = (max + min) / 2, std = max(max - min) / 2
            print("[MeshOTDataset] Using recenter_per_shape normalization (bbox center + half-extent)")
            for V, F, p in zip(raw_vertices, raw_faces, raw_paths):
                V = V.astype(np.float64)
                v_min = V.min(axis=0, keepdims=True)  # (1, 3)
                v_max = V.max(axis=0, keepdims=True)  # (1, 3)
                v_center = (v_max + v_min) / 2.0     # (1, 3)
                v_half_extent = ((v_max - v_min) / 2.0).max()  # scalar: half of the largest axis
                v_half_extent = max(v_half_extent, 1e-8)  # avoid division by zero
                Vn = (V - v_center) / v_half_extent
                self.mesh_paths.append(p)
                self.vertices_list.append(Vn.astype(np.float32))
                self.faces_list.append(F)

        elif self.normalize_per_shape_maxabs:
            # Mode 3: Per-shape max-abs normalization (original logic: normalize_vertices_minus1_1)
            print("[MeshOTDataset] Using normalize_per_shape_maxabs normalization")
            for V, F, p in zip(raw_vertices, raw_faces, raw_paths):
                Vn = normalize_vertices_minus1_1(V)
                self.mesh_paths.append(p)
                self.vertices_list.append(Vn)
                self.faces_list.append(F)
        else:
            raise RuntimeError("No normalization mode selected")

        print(f"[MeshOTDataset] Loaded {len(self.vertices_list)} normalized meshes.")

    def _load_meshes_from_npz(self):
        """
        Load mesh data from a preprocessed NPZ file.
        The NPZ stores raw (un-normalized) vertices; normalization is applied after loading.

        NPZ file structure:
            {
                'vertices': object array of (n_i, 3) float32 arrays
                'faces': object array of (m_i, 3) int32 arrays
            }
        """
        npz_path = self.mesh_root / f"{self.split}_mesh.npz"
        if not npz_path.exists():
            raise RuntimeError(f"Mesh NPZ file not found: {npz_path}. "
                             f"Please run scripts/convert_mesh_to_npz.py to convert your dataset.")

        print(f"[MeshOTDataset-MeshNPZ] Loading {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)

        # object arrays need special handling
        vertices_arr = data['vertices']  # object array
        faces_arr = data['faces']        # object array

        print(f"[MeshOTDataset-MeshNPZ] Found {len(vertices_arr)} meshes in NPZ file")

        # Collect raw data
        raw_vertices = []
        raw_faces = []
        for i in range(len(vertices_arr)):
            V = vertices_arr[i].astype(np.float64)  # convert to float64 for normalization computations
            F = faces_arr[i].astype(np.int32)
            raw_vertices.append(V)
            raw_faces.append(F)

        if not raw_vertices:
            raise RuntimeError("[MeshOTDataset-MeshNPZ] No valid meshes loaded from NPZ file.")

        # Normalization: three mutually exclusive modes (same logic as _load_meshes)
        if self.normalize_globally:
            # Mode 1: Global normalization (LION style: global mean + global std) + scale to [-1,1]
            if self._all_points_mean is not None and self._all_points_std is not None:
                all_points_mean = np.asarray(self._all_points_mean).reshape(1, 3)
                all_points_std = float(np.asarray(self._all_points_std))
                global_scale = getattr(self, '_global_scale', None)
                if global_scale is None:
                    raise ValueError("[MeshOTDataset-MeshNPZ] normalize_globally requires global_scale for val/test. "
                                     "Please pass the global_scale from the training dataset.")
            else:
                all_verts = np.concatenate(raw_vertices, axis=0)  # (total_verts, 3)
                all_points_mean = all_verts.reshape(-1, 3).mean(axis=0).reshape(1, 3)
                all_points_std = all_verts.reshape(-1).std()

                # First apply (x - mean) / std to all vertices, then compute the global max_abs
                all_verts_norm = (all_verts - all_points_mean) / all_points_std
                global_scale = np.abs(all_verts_norm).max()
                global_scale = max(global_scale, 1e-8)  # avoid division by zero

                self.all_points_mean = all_points_mean  # (1,3)
                self.all_points_std = all_points_std    # scalar
                self.global_scale = global_scale        # scalar: used to scale to [-1,1]
                print(f"[MeshOTDataset-MeshNPZ] Global normalization: mean={all_points_mean.flatten()}, std={all_points_std:.6f}, global_scale={global_scale:.6f}")

            # Apply global normalization: ((x - mean) / std) / global_scale -> [-1, 1]
            for V, F in zip(raw_vertices, raw_faces):
                Vn = (V - all_points_mean) / all_points_std / global_scale
                self.vertices_list.append(Vn.astype(np.float32))
                self.faces_list.append(F)

        elif self.recenter_per_shape:
            # Mode 2: Per-shape recenter (LION default: bbox center + bbox half-extent)
            print("[MeshOTDataset-MeshNPZ] Using recenter_per_shape normalization (bbox center + half-extent)")
            for V, F in zip(raw_vertices, raw_faces):
                V = V.astype(np.float64)
                v_min = V.min(axis=0, keepdims=True)  # (1, 3)
                v_max = V.max(axis=0, keepdims=True)  # (1, 3)
                v_center = (v_max + v_min) / 2.0     # (1, 3)
                v_half_extent = ((v_max - v_min) / 2.0).max()  # scalar: half of the largest axis
                v_half_extent = max(v_half_extent, 1e-8)  # avoid division by zero
                Vn = (V - v_center) / v_half_extent
                self.vertices_list.append(Vn.astype(np.float32))
                self.faces_list.append(F)

        elif self.normalize_per_shape_maxabs:
            # Mode 3: Per-shape max-abs normalization (original logic: normalize_vertices_minus1_1)
            print("[MeshOTDataset-MeshNPZ] Using normalize_per_shape_maxabs normalization")
            for V, F in zip(raw_vertices, raw_faces):
                Vn = normalize_vertices_minus1_1(V)
                self.vertices_list.append(Vn)
                self.faces_list.append(F)
        else:
            raise RuntimeError("No normalization mode selected")

        print(f"[MeshOTDataset-MeshNPZ] Loaded {len(self.vertices_list)} normalized meshes.")

    def _load_pointclouds(self):
        """
        Load the point cloud dataset:
        - mesh_root/{split}.npz (contains the 'pointclouds' key, shape: (B, N, 3))
        """
        npz_path = self.mesh_root / f"{self.split}.npz"
        if not npz_path.exists():
            raise RuntimeError(f"NPZ file not found: {npz_path}. "
                             f"Please run scripts/convert_npy_to_npz.py to convert your dataset.")

        print(f"[MeshOTDataset-PCD] Loading {npz_path}...")
        data = np.load(npz_path)
        all_pcds = data['pointclouds'].astype(np.float32)  # (B, N, 3)
        print(f"[MeshOTDataset-PCD] Loaded {all_pcds.shape[0]} pointclouds, shape per cloud: {all_pcds.shape[1:]}")

        # Convert to a list to be compatible with the downstream normalization logic
        raw_pcds = [all_pcds[i] for i in range(all_pcds.shape[0])]

        # 2. Normalization: three mutually exclusive modes
        if self.normalize_globally:
            # Mode 1: Global normalization (LION style: global mean + global std) + scale to [-1,1]
            if self._all_points_mean is not None and self._all_points_std is not None:
                # Use the provided mean/std (val/test scenario)
                all_points_mean = self._all_points_mean
                all_points_std = self._all_points_std
                # Use the provided global_scale if present; otherwise it must be recomputed
                global_scale = getattr(self, '_global_scale', None)
                if global_scale is None:
                    # global_scale must be provided in the val/test scenario
                    raise ValueError("[MeshOTDataset-PCD] normalize_globally requires global_scale for val/test. "
                                     "Please pass the global_scale from the training dataset.")
            else:
                # Compute the global mean/std (train scenario)
                all_points = np.stack(raw_pcds, axis=0)  # (M, 15000, 3)
                all_points_mean = all_points.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
                all_points_std = all_points.reshape(-1).std().reshape(1, 1, 1)

                # First apply (x - mean) / std to all points, then compute the global max_abs
                all_points_norm = (all_points - all_points_mean) / all_points_std
                global_scale = np.abs(all_points_norm).max()
                global_scale = max(global_scale, 1e-8)  # avoid division by zero

                # Save for external use
                self.all_points_mean = all_points_mean
                self.all_points_std = all_points_std
                self.global_scale = global_scale  # scalar: used to scale to [-1,1]
                print(f"[MeshOTDataset-PCD] Global normalization: mean={all_points_mean.flatten()}, std={all_points_std.item():.6f}, global_scale={global_scale:.6f}")

            # Apply global normalization: ((x - mean) / std) / global_scale -> [-1, 1]
            for pcd in raw_pcds:
                pcd_norm = (pcd - all_points_mean.reshape(1, 3)) / all_points_std.item() / global_scale
                self.pointclouds_list.append(pcd_norm.astype(np.float32))

        elif self.recenter_per_shape:
            # Mode 2: Per-shape recenter (LION default: bbox center + bbox half-extent)
            # mean = (max + min) / 2, std = max(max - min) / 2
            print("[MeshOTDataset-PCD] Using recenter_per_shape normalization (bbox center + half-extent)")
            for pcd in raw_pcds:
                pcd = pcd.astype(np.float64)
                p_min = pcd.min(axis=0, keepdims=True)  # (1, 3)
                p_max = pcd.max(axis=0, keepdims=True)  # (1, 3)
                p_center = (p_max + p_min) / 2.0       # (1, 3)
                p_half_extent = ((p_max - p_min) / 2.0).max()  # scalar: half of the largest axis
                p_half_extent = max(p_half_extent, 1e-8)  # avoid division by zero
                pcd_norm = (pcd - p_center) / p_half_extent
                self.pointclouds_list.append(pcd_norm.astype(np.float32))

        elif self.normalize_per_shape_maxabs:
            # Mode 3: Per-shape max-abs normalization (original logic: normalize_vertices_minus1_1)
            print("[MeshOTDataset-PCD] Using normalize_per_shape_maxabs normalization")
            for pcd in raw_pcds:
                pcd_norm = normalize_vertices_minus1_1(pcd)
                self.pointclouds_list.append(pcd_norm)
        else:
            raise RuntimeError("No normalization mode selected")

        print(f"[MeshOTDataset-PCD] Loaded {len(self.pointclouds_list)} pointclouds.")

    @property
    def num_meshes(self) -> int:
        if self.dataset_is_pcd:
            return len(self.pointclouds_list)
        return len(self.vertices_list)

    # ----------------- Core interface: compute_batch -----------------

    def compute_batch(
        self,
        idx_batch: np.ndarray,  # shape (B,)
        epoch: int,
        step: int,
    ):
        """
        Given a batch of mesh indices idx_batch for this rank, return (x0, x1):

            x0: (B, N, 3) uniform in [-1,1]^3, float32
            x1: (B, N, 3) Poisson-disk samples on each mesh in this batch,
                reordered after OT/greedy matching with a Hermite-style cost, float32

        OT part:
          - Default ot_solver="greedy", O(N^2);
          - If set to "hungarian", uses SciPy's Hungarian algorithm, O(N^3); watch the cost for large N.
        """
        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        B = idx_batch.shape[0]
        N = self.num_points

        # x0: use an independent seed per epoch/step to stay deterministic at the numpy level
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        r_min = np.sqrt(2.0)
        r_max = float(1.7)
        if r_max < r_min:
            raise ValueError(f"shell_r_max must be >= sqrt(2). Got {r_max} < {r_min}")

        # Recommended to be mutually exclusive (optional)
        if self.use_sphere and self.use_shell:
            raise ValueError("use_sphere and use_shell cannot both be True.")

        if self.use_shell:
            # --- constants ---
            eps = np.float32(1e-12)

            # 1) direction: float32 normal
            dir_np = rng.standard_normal((B, N, 3), dtype=np.float32)

            # norm2 in (B,N,1)
            norm2 = (dir_np * dir_np).sum(axis=-1, keepdims=True)   # float32
            np.maximum(norm2, eps, out=norm2)                       # in-place clamp
            np.sqrt(norm2, out=norm2)                               # now norm in-place

            # normalize in-place
            dir_np /= norm2

            # 2) radius: reuse u buffer as r buffer to avoid extra allocations
            rmin = np.float32(r_min)
            rmax = np.float32(r_max)
            rmin3 = rmin * rmin * rmin
            rmax3 = rmax * rmax * rmax
            span3 = rmax3 - rmin3

            r = rng.random((B, N, 1), dtype=np.float32)  # r := u
            r *= span3                                   # r := u*(rmax3-rmin3)
            r += rmin3                                   # r := base
            np.cbrt(r, out=r)                             # r := cbrt(base)

            # 3) final
            x0_np = dir_np * r                            # (B,N,3) float32

        elif self.use_sphere:
            x0_np = rng.normal(size=(B, N, 3)).astype(np.float32, copy=False)
            norm = np.linalg.norm(x0_np, axis=-1, keepdims=True)
            x0_np = x0_np / np.maximum(norm, 1e-12)
            x0_np = x0_np * np.sqrt(2.0)   # sphere radius fixed at sqrt(2)

        else:
            x0_np = rng.random((B, N, 3), dtype=np.float32)
            x0_np = x0_np * 2.0 - 1.0  # [-1,1]^3

        # ============ PCD mode: linear OT (no normal) ============
        if self.dataset_is_pcd:
            # x1: uniform subsample from preloaded points + linear OT matching
            x1_matched = np.empty((B, N, 3), dtype=np.float32)

            # Generate subsample indices for each sample (must be generated in the main process to stay deterministic)
            pcd_list = self.pointclouds_list
            subsample_indices = [
                rng.choice(pcd_list[int(mi)].shape[0], size=N, replace=False)
                for mi in idx_batch
            ]

            # Use joblib for parallel processing: subsample + linear OT
            results = Parallel(n_jobs=self.num_workers, backend="loky")(
                delayed(_process_one_pcd_ot)(
                    int(bi),
                    x0_np[bi],                      # (N,3)
                    pcd_list[int(mi)],
                    subsample_indices[bi],
                    str(self.ot_solver),
                )
                for bi, mi in enumerate(idx_batch)
            )

            for bi, x1_b in results:
                x1_matched[bi] = x1_b

            # In PCD mode, return only coordinates (B,N,3); no normal needed
            return x0_np.astype(np.float32), x1_matched.astype(np.float32)

        # ------------------------------------------------------------------
        # Mesh mode: x1 + Hermite OT; split into two complete branches based on use_poisson
        #    Both branches call Parallel(...) exactly once
        # ------------------------------------------------------------------
        x1_matched = np.empty((B, N, 3), dtype=np.float32)
        if self.hermite_degree == "cubic" and not self.zero_t0:
            n0_raw     = np.empty((B, N, 3), dtype=np.float32)  # (B,N,3)
        if not self.hermite_degree == "linear":
            n1_matched = np.empty((B, N, 3), dtype=np.float32)  # (B,N,3)


        if self.use_poisson:
            # -------------------- Poisson + Hermite OT branch --------------------
            results = Parallel(
                n_jobs=self.num_workers,
                backend="loky",
            )(
                delayed(poisson_points_one_mesh_noclean_and_hermite_ot_one_batch)(
                    bi=int(bi),
                    x0=x0_np[bi],                      # (N,3)
                    V=self.vertices_list[int(mi)],
                    F=self.faces_list[int(mi)],
                    num_points=N,
                    init_factor=float(self.poisson_init_factor),
                    lambda_orient=float(self.lambda_orient),
                    ot_solver=str(self.ot_solver),
                    hermite_degree=str(self.hermite_degree),
                    mesh_idx=int(mi),
                    zero_t0=self.zero_t0
                )
                for bi, mi in enumerate(idx_batch)
            )
            for bi, x1_b, n0_b, n1_b in results:
                if x1_b.shape != (N, 3):
                    raise RuntimeError(
                        f"Unexpected matched x1 shape {x1_b.shape}, expected {(N,3)}"
                    )
                x1_matched[bi] = x1_b
                if self.hermite_degree == "cubic" and not self.zero_t0:
                    n0_raw[bi]     = n0_b
                n1_matched[bi] = n1_b
        else:
            # ----------------- Uniform surface + Hermite OT branch -----------------
            results = Parallel(
                n_jobs=self.num_workers,
                backend="loky",
            )(
                delayed(random_points_one_mesh_noclean_and_hermite_ot_one_batch)(
                    bi=int(bi),
                    x0=x0_np[bi],                      # (N,3)
                    V=self.vertices_list[int(mi)],
                    F=self.faces_list[int(mi)],
                    num_points=N,
                    lambda_orient=float(self.lambda_orient),
                    ot_solver=str(self.ot_solver),
                    hermite_degree=str(self.hermite_degree),
                    mesh_idx=int(mi),
                    zero_t0=self.zero_t0
                )
                for bi, mi in enumerate(idx_batch)
            )
            for bi, x1_b, n0_b, n1_b in results:
                if x1_b.shape != (N, 3):
                    raise RuntimeError(
                        f"Unexpected matched x1 shape {x1_b.shape}, expected {(N,3)}"
                    )
                x1_matched[bi] = x1_b
                if self.hermite_degree == "cubic" and not self.zero_t0:
                    n0_raw[bi]     = n0_b
                if not self.hermite_degree == "linear":
                    n1_matched[bi] = n1_b

        # 4) Decide the output shape based on use_normal
        if not self.use_normal:
            # Compatible with the original logic: return only coordinates (B,N,3)
            return x0_np.astype(np.float32), x1_matched.astype(np.float32)

        # use_normal=True: concatenate [xyz, normal] -> (B,N,6)
        if self.hermite_degree == "cubic":
            x1_feat = np.concatenate([x1_matched, n1_matched], axis=-1)  # (B,N,6)
            if not self.zero_t0:
                x0_feat = np.concatenate([x0_np, n0_raw], axis=-1)       # (B,N,6)
                return x0_feat.astype(np.float32), x1_feat.astype(np.float32)
            else:
                return x0_np.astype(np.float32), x1_feat.astype(np.float32)
        else:
            x1_feat = np.concatenate([x1_matched, n1_matched], axis=-1)  # (B,N,6)
            return x0_np.astype(np.float32), x1_feat.astype(np.float32)


def _process_one_pcd_hilbert(bi: int, pcd_full: np.ndarray, indices: np.ndarray, hilbert_p: int):
    """
    Process the subsample + Hilbert sort for a single point cloud (for joblib parallelism)
    """
    pts = pcd_full[indices]  # (N, 3)
    pts_sorted = hilbert_sort_xyz_numba(pts, p_bits=hilbert_p)
    return bi, pts_sorted


def _process_one_pcd_ot(
    bi: int,
    x0: np.ndarray,           # (N, 3)
    pcd_full: np.ndarray,     # (M, 3) full point cloud
    indices: np.ndarray,      # (N,) subsample indices
    ot_solver: str,
):
    """
    Process the subsample + linear OT matching for a single point cloud (for joblib parallelism)

    In PCD mode:
      - No normal (point clouds have no normal information)
      - Use linear cost (pure straight-line distance)
      - Return the OT-matched point cloud
    """
    x1 = pcd_full[indices].astype(np.float32)  # (N, 3)

    # Use linear cost (pure straight-line distance)
    C = build_linear_cost_matrix(x0=x0, x1=x1)

    # Perform OT assignment
    perm = solve_ot_assignment(C, mode=ot_solver)

    x1_matched = x1[perm]  # (N, 3)
    return bi, x1_matched


class MeshSortDataset:

    def __init__(
        self,
        mesh_root: str,
        num_points: int,
        use_multi: bool = True,
        num_workers: int = 8,
        device: str = "cpu",
        hilbert_p: int = 10,
        poisson_init_factor: float = 3.0,
        use_normal: bool = False,
        use_normal_in_sort: bool = False,
        use_poisson: bool = False,
        use_sphere: bool = False,
        use_shell: bool = False,
        hermite_degree: str = "linear",
        zero_t0: bool = False,
        split: str = "train",
        dataset_is_pcd: bool = False,
        dataset_is_mesh_npz: bool = False,  # load mesh from a preprocessed NPZ file
        normalize_globally: bool = False,
        recenter_per_shape: bool = False,
        normalize_per_shape_maxabs: bool = False,
        all_points_mean: np.ndarray = None,
        all_points_std: np.ndarray = None,
        global_scale: float = None,  # global scale for normalize_globally (must be passed in for val/test)
        no_sort: bool = False,  # if True, skip Hilbert sorting
        is_single_mesh: bool = False,  # single-mesh mode
        is_single_pcd: bool = False,  # single point cloud mode (with RGB)
        linear_6d: bool = False,  # 6D linear mode: x0 in [-1,1]^6, x1 = xyz+normals
    ):
        self.mesh_root = Path(mesh_root).expanduser().resolve()
        self.num_points = int(num_points)
        self.use_multi = bool(use_multi)
        self.num_workers = int(num_workers)
        self.device = torch.device(device)
        if is_single_mesh or is_single_pcd:
            self.hilbert_p = 10
        else:
            self.hilbert_p = int(hilbert_p)
        self.poisson_init_factor = float(poisson_init_factor)
        self.use_normal = use_normal
        self.use_normal_in_sort = use_normal_in_sort
        self.linear_6d = linear_6d
        self.use_poisson = use_poisson
        self.use_sphere = use_sphere
        self.use_shell = use_shell
        self.hermite_degree = hermite_degree
        self.zero_t0 = zero_t0
        self.split = split
        self.dataset_is_pcd = dataset_is_pcd
        self.dataset_is_mesh_npz = dataset_is_mesh_npz
        self.normalize_globally = normalize_globally
        self.recenter_per_shape = recenter_per_shape
        self.normalize_per_shape_maxabs = normalize_per_shape_maxabs
        self._all_points_mean = all_points_mean
        self._all_points_std = all_points_std
        self._global_scale = global_scale  # global scale for normalize_globally
        self.no_sort = no_sort
        self.is_single_mesh = is_single_mesh
        self.is_single_pcd = is_single_pcd

        # Validation: is_single_mesh and is_single_pcd are mutually exclusive
        if self.is_single_mesh and self.is_single_pcd:
            raise ValueError("is_single_mesh and is_single_pcd cannot both be True")

        # Ensure the three normalization modes are mutually exclusive
        norm_flags = [normalize_globally, recenter_per_shape, normalize_per_shape_maxabs]
        if sum(norm_flags) > 1:
            raise ValueError(
                "Only one of normalize_globally, recenter_per_shape, normalize_per_shape_maxabs can be True."
            )
        # If all are False, default to normalize_per_shape_maxabs
        if sum(norm_flags) == 0:
            self.normalize_per_shape_maxabs = True

        if self.use_normal:
            print("Using Normal!")

        # Load data according to the mode
        if self.is_single_mesh:
            # Single-mesh mode: mesh_root points to a single mesh file
            self.mesh_paths: List[Path] = []
            self.vertices_list: List[np.ndarray] = []
            self.faces_list: List[np.ndarray] = []
            self._load_single_mesh()
        elif self.is_single_pcd:
            # Single point cloud mode: mesh_root points to a single .ply point cloud file (with RGB)
            self.mesh_paths: List[Path] = []
            self.vertices_list: List[np.ndarray] = []
            self.faces_list: List[np.ndarray] = []
            self._load_single_pcd()
        elif self.dataset_is_pcd:
            self.pointclouds_list: List[np.ndarray] = []
            self._load_pointclouds()
        elif self.dataset_is_mesh_npz:
            # Mesh NPZ mode: load mesh from a preprocessed NPZ file
            self.mesh_paths: List[Path] = []
            self.vertices_list: List[np.ndarray] = []
            self.faces_list: List[np.ndarray] = []
            self._load_meshes_from_npz()
        else:
            # Raw PLY file mode
            self.mesh_paths: List[Path] = []
            self.vertices_list: List[np.ndarray] = []
            self.faces_list: List[np.ndarray] = []
            self._load_meshes()

        # Kept for compatibility with the MeshNearPointPairDataset interface
        self._epoch_batches_x0: List[torch.Tensor] = []
        self._epoch_batches_x1: List[torch.Tensor] = []
        self._steps_per_epoch: int = 0

        if use_sphere:
            print("Using sphere init")
        if use_shell:
            print("Using shell init")
        if not use_sphere and not use_shell:
            print("Using unit box init")

    # ----------------- Loading & normalization -----------------

    def _load_single_mesh(self):
        """
        Load a single mesh file and normalize it to the [-1, 1]^3 unit box.
        mesh_root should be a path to a mesh file (e.g. .ply, .obj, .stl).
        """
        mesh_path = self.mesh_root

        if not mesh_path.is_file():
            raise RuntimeError(f"[MeshSortDataset-SingleMesh] Single mesh file not found: {mesh_path}")

        print(f"[MeshSortDataset-SingleMesh] Loading single mesh: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))

        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            raise RuntimeError(f"[MeshSortDataset-SingleMesh] Empty mesh: {mesh_path}")

        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.triangles, dtype=np.int32)

        # Normalize to the [-1, 1]^3 unit box (using bbox center + half-extent)
        v_min = V.min(axis=0, keepdims=True)
        v_max = V.max(axis=0, keepdims=True)
        v_center = (v_max + v_min) / 2.0
        v_half_extent = ((v_max - v_min) / 2.0).max()
        v_half_extent = max(v_half_extent, 1e-8)
        Vn = (V - v_center) / v_half_extent

        # Store the single mesh (use a list to keep the interface consistent)
        self.vertices_list = [Vn.astype(np.float32)]
        self.faces_list = [F]
        self.mesh_paths = [mesh_path]

        print(f"[MeshSortDataset-SingleMesh] Vertices: {V.shape[0]}, Faces: {F.shape[0]}")
        print(f"[MeshSortDataset-SingleMesh] Normalized to [-1, 1]^3 unit box")

        # ===== Presample 1M points + normals =====
        # This way compute_batch can randomly pick from these presampled points instead of resampling the mesh every time
        PRESAMPLE_COUNT = 32_000_000

        # Rebuild the mesh for sampling (using the normalized vertices)
        mesh_normalized = o3d.geometry.TriangleMesh()
        mesh_normalized.vertices = o3d.utility.Vector3dVector(Vn)
        mesh_normalized.triangles = o3d.utility.Vector3iVector(F)
        mesh_normalized.compute_vertex_normals()

        pcd = mesh_normalized.sample_points_uniformly(number_of_points=PRESAMPLE_COUNT)

        self.presampled_points = np.asarray(pcd.points, dtype=np.float32)
        # Note: negate the normals (consistent with the original logic)
        self.presampled_normals = -np.asarray(pcd.normals, dtype=np.float32)

        print(f"[MeshSortDataset-SingleMesh] Presampled {PRESAMPLE_COUNT} points with normals")

        # ===== Precompute Hilbert indices =====
        # Computed with a fixed bbox [-1, 1]^3 so that compute_batch only needs an argsort
        if not self.no_sort:
            # Check whether p_bits would make the Hilbert index overflow uint64
            if self.use_normal_in_sort:
                max_p_bits = 64 // 6  # 6D: max 10
                if self.hilbert_p > max_p_bits:
                    raise ValueError(
                        f"When use_normal_in_sort=True, hilbert_p is at most {max_p_bits} (6D x {max_p_bits} = {6*max_p_bits} bits); "
                        f"the current hilbert_p={self.hilbert_p} would cause uint64 overflow"
                    )
            else:
                max_p_bits = 64 // 3  # 3D: max 21
                if self.hilbert_p > max_p_bits:
                    raise ValueError(
                        f"hilbert_p is at most {max_p_bits} (3D x {max_p_bits} = {3*max_p_bits} bits); "
                        f"the current hilbert_p={self.hilbert_p} would cause uint64 overflow"
                    )
            if self.use_normal_in_sort:
                # 6D Hilbert index (coordinates + normals)
                pts_with_normals = np.concatenate([self.presampled_points, self.presampled_normals], axis=-1)
                # Normalize to [0, 1]^6 (coordinates in [-1,1]^3, normals in [-1,1]^3)
                norm_pts = (pts_with_normals + 1.0) / 2.0
                grid_max = (1 << self.hilbert_p) - 1
                coords = np.floor(norm_pts * grid_max).astype(np.uint64)
                coords = np.clip(coords, 0, grid_max)
                self.presampled_hilbert_indices = hilbert_indices_int_nd(coords, self.hilbert_p)
            else:
                # 3D Hilbert index (coordinates only)
                # Coordinates are already normalized to [-1, 1]^3; convert to [0, 1]^3
                norm_pts = (self.presampled_points + 1.0) / 2.0
                grid_max = (1 << self.hilbert_p) - 1
                coords = np.floor(norm_pts * grid_max).astype(np.int64)
                coords = np.clip(coords, 0, grid_max)
                self.presampled_hilbert_indices = hilbert_indices_int_3d(coords, self.hilbert_p)

            print(f"[MeshSortDataset-SingleMesh] Precomputed Hilbert indices")

    def _load_single_pcd(self):
        """
        Load a single point cloud file (.ply) with RGB colors.
        xyz is normalized to [-1, 1]^3, and RGB is normalized to [-1, 1]^3.
        mesh_root should be a path to a .ply point cloud file.
        """
        pcd_path = self.mesh_root

        if not pcd_path.is_file():
            raise RuntimeError(f"[MeshSortDataset-SinglePCD] Point cloud file not found: {pcd_path}")

        print(f"[MeshSortDataset-SinglePCD] Loading single point cloud: {pcd_path}")
        pcd = o3d.io.read_point_cloud(str(pcd_path))

        if len(pcd.points) == 0:
            raise RuntimeError(f"[MeshSortDataset-SinglePCD] Empty point cloud: {pcd_path}")

        # Extract xyz
        P = np.asarray(pcd.points, dtype=np.float64)

        # Normalize xyz to the [-1, 1]^3 unit box (using bbox center + half-extent)
        p_min = P.min(axis=0, keepdims=True)
        p_max = P.max(axis=0, keepdims=True)
        p_center = (p_max + p_min) / 2.0
        p_half_extent = ((p_max - p_min) / 2.0).max()
        p_half_extent = max(p_half_extent, 1e-8)
        Pn = (P - p_center) / p_half_extent

        # Extract RGB colors (Open3D colors are floats in the [0,1] range)
        if not pcd.has_colors():
            raise RuntimeError(f"[MeshSortDataset-SinglePCD] Point cloud has no RGB colors: {pcd_path}")

        colors = np.asarray(pcd.colors, dtype=np.float64)  # shape (N, 3), range [0, 1]
        # Normalize to [-1, 1]^3: [0,1] -> [-1,1]
        colors_normalized = colors * 2.0 - 1.0

        # Set the presampled data (no actual presampling; use all points)
        self.presampled_points = Pn.astype(np.float32)
        self.presampled_normals = colors_normalized.astype(np.float32)

        # Verify the normalized results are within the [-1, 1]^3 range
        xyz_min, xyz_max = self.presampled_points.min(), self.presampled_points.max()
        rgb_min, rgb_max = self.presampled_normals.min(), self.presampled_normals.max()

        if xyz_min < -1.0 - 1e-6 or xyz_max > 1.0 + 1e-6:
            raise RuntimeError(
                f"[MeshSortDataset-SinglePCD] xyz normalization failed: "
                f"min={xyz_min:.6f}, max={xyz_max:.6f}, expected [-1, 1]"
            )
        if rgb_min < -1.0 - 1e-6 or rgb_max > 1.0 + 1e-6:
            raise RuntimeError(
                f"[MeshSortDataset-SinglePCD] RGB normalization failed: "
                f"min={rgb_min:.6f}, max={rgb_max:.6f}, expected [-1, 1]"
            )

        print(f"[MeshSortDataset-SinglePCD] Points: {P.shape[0]}")
        print(f"[MeshSortDataset-SinglePCD] xyz range: [{xyz_min:.4f}, {xyz_max:.4f}]")
        print(f"[MeshSortDataset-SinglePCD] RGB range: [{rgb_min:.4f}, {rgb_max:.4f}]")

        # ===== Precompute Hilbert indices =====
        if not self.no_sort:
            # Check whether p_bits would make the Hilbert index overflow uint64
            if self.use_normal_in_sort:
                max_p_bits = 64 // 6  # 6D: max 10
                if self.hilbert_p > max_p_bits:
                    raise ValueError(
                        f"When use_normal_in_sort=True, hilbert_p is at most {max_p_bits} (6D x {max_p_bits} = {6*max_p_bits} bits); "
                        f"the current hilbert_p={self.hilbert_p} would cause uint64 overflow"
                    )
            else:
                max_p_bits = 64 // 3  # 3D: max 21
                if self.hilbert_p > max_p_bits:
                    raise ValueError(
                        f"hilbert_p is at most {max_p_bits} (3D x {max_p_bits} = {3*max_p_bits} bits); "
                        f"the current hilbert_p={self.hilbert_p} would cause uint64 overflow"
                    )

            if self.use_normal_in_sort:
                # 6D Hilbert index (xyz + RGB)
                pts_with_colors = np.concatenate([self.presampled_points, self.presampled_normals], axis=-1)
                # Normalize to [0, 1]^6 (coordinates in [-1,1]^3, RGB in [-1,1]^3)
                norm_pts = (pts_with_colors + 1.0) / 2.0
                grid_max = (1 << self.hilbert_p) - 1
                coords = np.floor(norm_pts * grid_max).astype(np.uint64)
                coords = np.clip(coords, 0, grid_max)
                self.presampled_hilbert_indices = hilbert_indices_int_nd(coords, self.hilbert_p)
            else:
                # 3D Hilbert index (xyz only)
                # Coordinates are already normalized to [-1, 1]^3; convert to [0, 1]^3
                norm_pts = (self.presampled_points + 1.0) / 2.0
                grid_max = (1 << self.hilbert_p) - 1
                coords = np.floor(norm_pts * grid_max).astype(np.int64)
                coords = np.clip(coords, 0, grid_max)
                self.presampled_hilbert_indices = hilbert_indices_int_3d(coords, self.hilbert_p)

            print(f"[MeshSortDataset-SinglePCD] Precomputed Hilbert indices")

    def _load_meshes(self):
        # As with the other datasets, filter meshes using the train split in all.csv
        ply_files = find_ply_files_from_all_csv(self.mesh_root, split=self.split)

        if not ply_files:
            raise RuntimeError(f"No .ply files found under: {self.mesh_root}")

        print(f"[MeshSortDataset] Found {len(ply_files)} .ply files, loading...")

        # First collect all raw vertices and faces
        raw_vertices = []
        raw_faces = []
        raw_paths = []

        for p in tqdm(ply_files, desc="Loading meshes"):
            try:
                mesh = o3d.io.read_triangle_mesh(str(p))
            except Exception as e:
                print(f"[WARN] Failed to read mesh {p}: {e}")
                continue

            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                print(f"[WARN] Empty mesh in {p}, skip.")
                continue

            V = np.asarray(mesh.vertices, dtype=np.float64)
            F = np.asarray(mesh.triangles, dtype=np.int32)
            raw_vertices.append(V)
            raw_faces.append(F)
            raw_paths.append(p)

        if not raw_vertices:
            raise RuntimeError("[MeshSortDataset] No valid meshes loaded after filtering.")

        # Normalization: three mutually exclusive modes
        if self.normalize_globally:
            # Mode 1: Global normalization (LION style: global mean + global std) + scale to [-1,1]
            if self._all_points_mean is not None and self._all_points_std is not None:
                all_points_mean = np.asarray(self._all_points_mean).reshape(1, 3)
                all_points_std = float(np.asarray(self._all_points_std))
                # Use the provided global_scale if present; otherwise it must be recomputed
                global_scale = getattr(self, '_global_scale', None)
                if global_scale is None:
                    # global_scale must be provided in the val/test scenario
                    raise ValueError("[MeshSortDataset] normalize_globally requires global_scale for val/test. "
                                     "Please pass the global_scale from the training dataset.")
            else:
                all_verts = np.concatenate(raw_vertices, axis=0)  # (total_verts, 3)
                all_points_mean = all_verts.reshape(-1, 3).mean(axis=0).reshape(1, 3)
                all_points_std = all_verts.reshape(-1).std()

                # First apply (x - mean) / std to all vertices, then compute the global max_abs
                all_verts_norm = (all_verts - all_points_mean) / all_points_std
                global_scale = np.abs(all_verts_norm).max()
                global_scale = max(global_scale, 1e-8)  # avoid division by zero

                self.all_points_mean = all_points_mean  # (1,3)
                self.all_points_std = all_points_std    # scalar
                self.global_scale = global_scale        # scalar: used to scale to [-1,1]
                print(f"[MeshSortDataset] Global normalization: mean={all_points_mean.flatten()}, std={all_points_std:.6f}, global_scale={global_scale:.6f}")

            # Apply global normalization: ((x - mean) / std) / global_scale -> [-1, 1]
            for V, F, p in zip(raw_vertices, raw_faces, raw_paths):
                Vn = (V - all_points_mean) / all_points_std / global_scale
                self.mesh_paths.append(p)
                self.vertices_list.append(Vn.astype(np.float32))
                self.faces_list.append(F)

        elif self.recenter_per_shape:
            # Mode 2: Per-shape recenter (LION default: bbox center + bbox half-extent)
            # mean = (max + min) / 2, std = max(max - min) / 2
            print("[MeshSortDataset] Using recenter_per_shape normalization (bbox center + half-extent)")
            for V, F, p in zip(raw_vertices, raw_faces, raw_paths):
                V = V.astype(np.float64)
                v_min = V.min(axis=0, keepdims=True)  # (1, 3)
                v_max = V.max(axis=0, keepdims=True)  # (1, 3)
                v_center = (v_max + v_min) / 2.0     # (1, 3)
                v_half_extent = ((v_max - v_min) / 2.0).max()  # scalar: half of the largest axis
                v_half_extent = max(v_half_extent, 1e-8)  # avoid division by zero
                Vn = (V - v_center) / v_half_extent
                self.mesh_paths.append(p)
                self.vertices_list.append(Vn.astype(np.float32))
                self.faces_list.append(F)

        elif self.normalize_per_shape_maxabs:
            # Mode 3: Per-shape max-abs normalization (original logic: normalize_vertices_minus1_1)
            print("[MeshSortDataset] Using normalize_per_shape_maxabs normalization")
            for V, F, p in zip(raw_vertices, raw_faces, raw_paths):
                Vn = normalize_vertices_minus1_1(V)
                self.mesh_paths.append(p)
                self.vertices_list.append(Vn)
                self.faces_list.append(F)
        else:
            raise RuntimeError("No normalization mode selected")
        print(f"[MeshSortDataset] Loaded {len(self.vertices_list)} normalized meshes.")

    def _load_meshes_from_npz(self):
        """
        Load mesh data from a preprocessed NPZ file.
        The NPZ stores raw (un-normalized) vertices; normalization is applied after loading.

        NPZ file structure:
            {
                'vertices': object array of (n_i, 3) float32 arrays
                'faces': object array of (m_i, 3) int32 arrays
            }
        """
        npz_path = self.mesh_root / f"{self.split}_mesh.npz"
        if not npz_path.exists():
            raise RuntimeError(f"Mesh NPZ file not found: {npz_path}. "
                             f"Please run scripts/convert_mesh_to_npz.py to convert your dataset.")

        print(f"[MeshSortDataset-MeshNPZ] Loading {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)

        # object arrays need special handling
        vertices_arr = data['vertices']  # object array
        faces_arr = data['faces']        # object array

        print(f"[MeshSortDataset-MeshNPZ] Found {len(vertices_arr)} meshes in NPZ file")

        # Collect raw data
        raw_vertices = []
        raw_faces = []
        for i in range(len(vertices_arr)):
            V = vertices_arr[i].astype(np.float64)  # convert to float64 for normalization computations
            F = faces_arr[i].astype(np.int32)
            raw_vertices.append(V)
            raw_faces.append(F)

        if not raw_vertices:
            raise RuntimeError("[MeshSortDataset-MeshNPZ] No valid meshes loaded from NPZ file.")

        # Normalization: three mutually exclusive modes (same logic as _load_meshes)
        if self.normalize_globally:
            # Mode 1: Global normalization (LION style: global mean + global std) + scale to [-1,1]
            if self._all_points_mean is not None and self._all_points_std is not None:
                all_points_mean = np.asarray(self._all_points_mean).reshape(1, 3)
                all_points_std = float(np.asarray(self._all_points_std))
                global_scale = getattr(self, '_global_scale', None)
                if global_scale is None:
                    raise ValueError("[MeshSortDataset-MeshNPZ] normalize_globally requires global_scale for val/test. "
                                     "Please pass the global_scale from the training dataset.")
            else:
                all_verts = np.concatenate(raw_vertices, axis=0)  # (total_verts, 3)
                all_points_mean = all_verts.reshape(-1, 3).mean(axis=0).reshape(1, 3)
                all_points_std = all_verts.reshape(-1).std()

                # First apply (x - mean) / std to all vertices, then compute the global max_abs
                all_verts_norm = (all_verts - all_points_mean) / all_points_std
                global_scale = np.abs(all_verts_norm).max()
                global_scale = max(global_scale, 1e-8)  # avoid division by zero

                self.all_points_mean = all_points_mean  # (1,3)
                self.all_points_std = all_points_std    # scalar
                self.global_scale = global_scale        # scalar: used to scale to [-1,1]
                print(f"[MeshSortDataset-MeshNPZ] Global normalization: mean={all_points_mean.flatten()}, std={all_points_std:.6f}, global_scale={global_scale:.6f}")

            # Apply global normalization: ((x - mean) / std) / global_scale -> [-1, 1]
            for V, F in zip(raw_vertices, raw_faces):
                Vn = (V - all_points_mean) / all_points_std / global_scale
                self.vertices_list.append(Vn.astype(np.float32))
                self.faces_list.append(F)

        elif self.recenter_per_shape:
            # Mode 2: Per-shape recenter (LION default: bbox center + bbox half-extent)
            print("[MeshSortDataset-MeshNPZ] Using recenter_per_shape normalization (bbox center + half-extent)")
            for V, F in zip(raw_vertices, raw_faces):
                V = V.astype(np.float64)
                v_min = V.min(axis=0, keepdims=True)  # (1, 3)
                v_max = V.max(axis=0, keepdims=True)  # (1, 3)
                v_center = (v_max + v_min) / 2.0     # (1, 3)
                v_half_extent = ((v_max - v_min) / 2.0).max()  # scalar: half of the largest axis
                v_half_extent = max(v_half_extent, 1e-8)  # avoid division by zero
                Vn = (V - v_center) / v_half_extent
                self.vertices_list.append(Vn.astype(np.float32))
                self.faces_list.append(F)

        elif self.normalize_per_shape_maxabs:
            # Mode 3: Per-shape max-abs normalization (original logic: normalize_vertices_minus1_1)
            print("[MeshSortDataset-MeshNPZ] Using normalize_per_shape_maxabs normalization")
            for V, F in zip(raw_vertices, raw_faces):
                Vn = normalize_vertices_minus1_1(V)
                self.vertices_list.append(Vn)
                self.faces_list.append(F)
        else:
            raise RuntimeError("No normalization mode selected")

        print(f"[MeshSortDataset-MeshNPZ] Loaded {len(self.vertices_list)} normalized meshes.")

    def _load_pointclouds(self):
        """
        Load the point cloud dataset:
        - mesh_root/{split}.npz (contains the 'pointclouds' key, shape: (B, N, 3))
        """
        npz_path = self.mesh_root / f"{self.split}.npz"
        if not npz_path.exists():
            raise RuntimeError(f"NPZ file not found: {npz_path}. "
                             f"Please run scripts/convert_npy_to_npz.py to convert your dataset.")

        print(f"[MeshSortDataset-PCD] Loading {npz_path}...")
        data = np.load(npz_path)
        all_pcds = data['pointclouds'].astype(np.float32)  # (B, N, 3)
        print(f"[MeshSortDataset-PCD] Loaded {all_pcds.shape[0]} pointclouds, shape per cloud: {all_pcds.shape[1:]}")

        # Convert to a list to be compatible with the downstream normalization logic
        raw_pcds = [all_pcds[i] for i in range(all_pcds.shape[0])]

        # Normalization: three mutually exclusive modes
        if self.normalize_globally:
            # Mode 1: Global normalization (LION style: global mean + global std) + scale to [-1,1]
            if self._all_points_mean is not None and self._all_points_std is not None:
                # Use the provided mean/std (val/test scenario)
                all_points_mean = self._all_points_mean
                all_points_std = self._all_points_std
                # Use the provided global_scale if present; otherwise it must be recomputed
                global_scale = getattr(self, '_global_scale', None)
                if global_scale is None:
                    # global_scale must be provided in the val/test scenario
                    raise ValueError("[MeshSortDataset-PCD] normalize_globally requires global_scale for val/test. "
                                     "Please pass the global_scale from the training dataset.")
            else:
                # Compute the global mean/std (train scenario)
                all_points = np.stack(raw_pcds, axis=0)  # (M, 15000, 3)
                all_points_mean = all_points.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
                all_points_std = all_points.reshape(-1).std().reshape(1, 1, 1)

                # First apply (x - mean) / std to all points, then compute the global max_abs
                all_points_norm = (all_points - all_points_mean) / all_points_std
                global_scale = np.abs(all_points_norm).max()
                global_scale = max(global_scale, 1e-8)  # avoid division by zero

                # Save for external use
                self.all_points_mean = all_points_mean
                self.all_points_std = all_points_std
                self.global_scale = global_scale  # scalar: used to scale to [-1,1]
                print(f"[MeshSortDataset-PCD] Global normalization: mean={all_points_mean.flatten()}, std={all_points_std.item():.6f}, global_scale={global_scale:.6f}")

            # Apply global normalization: ((x - mean) / std) / global_scale -> [-1, 1]
            for pcd in raw_pcds:
                pcd_norm = (pcd - all_points_mean.reshape(1, 3)) / all_points_std.item() / global_scale
                self.pointclouds_list.append(pcd_norm.astype(np.float32))

        elif self.recenter_per_shape:
            # Mode 2: Per-shape recenter (LION default: bbox center + bbox half-extent)
            # mean = (max + min) / 2, std = max(max - min) / 2
            print("[MeshSortDataset-PCD] Using recenter_per_shape normalization (bbox center + half-extent)")
            for pcd in raw_pcds:
                pcd = pcd.astype(np.float64)
                p_min = pcd.min(axis=0, keepdims=True)  # (1, 3)
                p_max = pcd.max(axis=0, keepdims=True)  # (1, 3)
                p_center = (p_max + p_min) / 2.0       # (1, 3)
                p_half_extent = ((p_max - p_min) / 2.0).max()  # scalar: half of the largest axis
                p_half_extent = max(p_half_extent, 1e-8)  # avoid division by zero
                pcd_norm = (pcd - p_center) / p_half_extent
                self.pointclouds_list.append(pcd_norm.astype(np.float32))

        elif self.normalize_per_shape_maxabs:
            # Mode 3: Per-shape max-abs normalization (original logic: normalize_vertices_minus1_1)
            print("[MeshSortDataset-PCD] Using normalize_per_shape_maxabs normalization")
            for pcd in raw_pcds:
                pcd_norm = normalize_vertices_minus1_1(pcd)
                self.pointclouds_list.append(pcd_norm)
        else:
            raise RuntimeError("No normalization mode selected")

        print(f"[MeshSortDataset-PCD] Loaded {len(self.pointclouds_list)} pointclouds.")

    @property
    def num_meshes(self) -> int:
        if self.is_single_mesh or self.is_single_pcd:
            return 1
        if self.dataset_is_pcd:
            return len(self.pointclouds_list)
        return len(self.vertices_list)

    # ----------------- Core interface: compute_batch -----------------

    def compute_batch(
        self,
        idx_batch: np.ndarray,  # shape (B,)
        epoch: int,
        step: int,
    ):

        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        B = idx_batch.shape[0]
        N = self.num_points

        # x0: use an independent seed per epoch/step to stay deterministic at the numpy level
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        # 1) x0: uniform in [-1,1]^3
        r_min = np.sqrt(2.0)
        r_max = float(1.7)
        if r_max < r_min:
            raise ValueError(f"shell_r_max must be >= sqrt(2). Got {r_max} < {r_min}")

        if self.use_sphere and self.use_shell:
            raise ValueError("use_sphere and use_shell cannot both be True.")

        if self.use_shell:
            # --- constants ---
            eps = np.float32(1e-12)

            # 1) direction: float32 normal
            dir_np = rng.standard_normal((B, N, 3), dtype=np.float32)

            # norm2 in (B,N,1)
            norm2 = (dir_np * dir_np).sum(axis=-1, keepdims=True)   # float32
            np.maximum(norm2, eps, out=norm2)                       # in-place clamp
            np.sqrt(norm2, out=norm2)                               # now norm in-place

            # normalize in-place
            dir_np /= norm2

            # 2) radius: reuse u buffer as r buffer to avoid extra allocations
            rmin = np.float32(r_min)
            rmax = np.float32(r_max)
            rmin3 = rmin * rmin * rmin
            rmax3 = rmax * rmax * rmax
            span3 = rmax3 - rmin3

            r = rng.random((B, N, 1), dtype=np.float32)  # r := u
            r *= span3                                   # r := u*(rmax3-rmin3)
            r += rmin3                                   # r := base
            np.cbrt(r, out=r)                             # r := cbrt(base)

            # 3) final
            x0_np = dir_np * r                            # (B,N,3) float32

        elif self.use_sphere:
            x0_np = rng.normal(size=(B, N, 3)).astype(np.float32, copy=False)
            norm = np.linalg.norm(x0_np, axis=-1, keepdims=True)
            x0_np = x0_np / np.maximum(norm, 1e-12)
            x0_np = x0_np * np.sqrt(2.0)   # sphere radius fixed at sqrt(2)

        else:
            if self.linear_6d:
                # 6D linear mode: x0 ~ Uniform([-1,1]^6)
                x0_np = rng.random((B, N, 6), dtype=np.float32)
                x0_np = x0_np * 2.0 - 1.0  # [-1,1]^6
            else:
                x0_np = rng.random((B, N, 3), dtype=np.float32)
                x0_np = x0_np * 2.0 - 1.0  # [-1,1]^3

        # ============ Single Mesh / Single PCD mode ============
        if self.is_single_mesh or self.is_single_pcd:
            feat_dim_x1 = 6 if self.use_normal else 3
            x1_np = np.empty((B, N, feat_dim_x1), dtype=np.float32)

            # Use the presampled point cloud
            pts = self.presampled_points     # (32M, 3)
            norms = self.presampled_normals if self.use_normal else None  # (32M, 3) or None
            M = pts.shape[0]

            if self.no_sort:
                # No sorting; pick randomly from the presampled point cloud directly
                for bi in range(B):
                    indices = rng.choice(M, size=N, replace=False)
                    if self.use_normal:
                        x1_np[bi] = np.concatenate([pts[indices], norms[indices]], axis=-1)
                    else:
                        x1_np[bi] = pts[indices]
            else:
                # Precompute optimization: random point selection -> fetch precomputed Hilbert indices -> argsort -> take points in order
                # Using Hilbert indices over a fixed bbox [-1,1]^3, the 0th point stays at the absolute bottom-left of the mesh
                h_indices = self.presampled_hilbert_indices
                for bi in range(B):
                    indices = rng.choice(M, size=N, replace=False)
                    order = np.argsort(h_indices[indices])
                    sorted_indices = indices[order]
                    if self.use_normal:
                        x1_np[bi] = np.concatenate([pts[sorted_indices], norms[sorted_indices]], axis=-1)
                    else:
                        x1_np[bi] = pts[sorted_indices]

            if self.zero_t0 or self.hermite_degree != "cubic":
                return x0_np.astype(np.float32), x1_np.astype(np.float32)
            else:
                eps = 1e-8
                d = x1_np[..., :3] - x0_np
                norm = np.linalg.norm(d, axis=-1, keepdims=True)
                d_unit = d / np.maximum(norm, eps)
                x0_6D = np.concatenate([x0_np, d_unit], axis=-1)
                return x0_6D.astype(np.float32), x1_np.astype(np.float32)

        # ============ PCD mode ============
        if self.dataset_is_pcd:
            # x1: uniform subsample from preloaded points + Hilbert sort (parallelized with joblib)
            x1_np = np.empty((B, N, 3), dtype=np.float32)

            # Generate subsample indices for each sample (must be generated in the main process to stay deterministic)
            pcd_list = self.pointclouds_list
            subsample_indices = [
                rng.choice(pcd_list[int(mi)].shape[0], size=N, replace=False)
                for mi in idx_batch
            ]

            if self.no_sort:
                # no_sort=True: no multithreading needed; subsample directly
                for bi, mi in enumerate(idx_batch):
                    x1_np[bi] = pcd_list[int(mi)][subsample_indices[bi]]
            else:
                # Original logic: hilbert sort (multithreaded)
                hilbert_p = int(self.hilbert_p)
                results = Parallel(n_jobs=self.num_workers, backend="loky")(
                    delayed(_process_one_pcd_hilbert)(
                        int(bi), pcd_list[int(mi)], subsample_indices[bi], hilbert_p
                    )
                    for bi, mi in enumerate(idx_batch)
                )
                for bi, pts_sorted in results:
                    x1_np[bi] = pts_sorted

            return x0_np.astype(np.float32), x1_np.astype(np.float32)

        # ============ Original Mesh mode ============
        # 2) x1: Poisson disk on mesh surface + Hilbert sort
        feat_dim_x1 = 6 if self.use_normal else 3
        x1_np = np.empty((B, N, feat_dim_x1), dtype=np.float32)

        V_list = self.vertices_list
        F_list = self.faces_list
        N_ = int(N)
        hilbert_p = int(self.hilbert_p)
        init_factor = float(self.poisson_init_factor)
        use_normal = self.use_normal
        use_normal_in_sort = self.use_normal_in_sort

        if self.no_sort:
            # no_sort=True: skip Hilbert sorting
            if self.use_poisson:
                # Poisson sampling is expensive, so still use multithreading
                results = Parallel(
                    n_jobs=self.num_workers,
                    backend="loky",
                )(
                    delayed(poisson_nosort_one_mesh_noclean)(
                        int(bi), V_list[int(mi)], F_list[int(mi)], N_, init_factor, use_normal
                    )
                    for bi, mi in enumerate(idx_batch)
                )
                for bi, pts in results:
                    if pts.shape != (N_, feat_dim_x1):
                        raise RuntimeError(
                            f"Unexpected pts shape {pts.shape}, expected {(N_, feat_dim_x1)}"
                        )
                    x1_np[bi] = pts
            else:
                # Uniform sampling + no sorting; no multithreading needed
                for bi, mi in enumerate(idx_batch):
                    _, pts = random_nosort_one_mesh_noclean(
                        int(bi), V_list[int(mi)], F_list[int(mi)], N_, use_normal
                    )
                    if pts.shape != (N_, feat_dim_x1):
                        raise RuntimeError(
                            f"Unexpected pts shape {pts.shape}, expected {(N_, feat_dim_x1)}"
                        )
                    x1_np[bi] = pts
        else:
            # Original logic: Hilbert sorting
            if self.use_poisson:
                results = Parallel(
                    n_jobs=self.num_workers,
                    backend="loky",
                )(
                    delayed(poisson_hilbert_one_mesh_noclean)(
                        int(bi), V_list[int(mi)], F_list[int(mi)], N_, hilbert_p, init_factor, use_normal, use_normal_in_sort
                    )
                    for bi, mi in enumerate(idx_batch)
                )
            else:
                results = Parallel(
                    n_jobs=self.num_workers,
                    backend="loky",
                )(
                    delayed(random_hilbert_one_mesh_noclean)(
                        int(bi), V_list[int(mi)], F_list[int(mi)], N_, hilbert_p, init_factor, use_normal, use_normal_in_sort
                    )
                    for bi, mi in enumerate(idx_batch)
                )

            for bi, pts_sorted in results:
                if pts_sorted.shape != (N, feat_dim_x1):
                    raise RuntimeError(
                        f"Unexpected pts_sorted shape {pts_sorted.shape}, "
                        f"expected {(N, feat_dim_x1)}"
                    )
                x1_np[bi] = pts_sorted


        if self.zero_t0 or self.hermite_degree != "cubic":
            return x0_np.astype(np.float32), x1_np.astype(np.float32)
        else:
            eps = 1e-8
            d = x1_np[..., :3] - x0_np
            norm = np.linalg.norm(d, axis=-1, keepdims=True)
            d_unit = d / np.maximum(norm, eps)

            x0_6D = np.concatenate([x0_np, d_unit], axis=-1)
            return x0_6D.astype(np.float32), x1_np.astype(np.float32)

class MeshPairAsyncLoader:
    """
    Asynchronous batch loader:
      - A background thread continuously calls MeshNearPointPairDataset.compute_batch(...)
      - Puts results into a bounded queue (prefetch_k batches)
      - The foreground training loop calls next_batch() to get (x0, x1) torch.Tensors

    Characteristics:
      - The CPU computes the next batch (x0,x1) in the background
      - The GPU trains the current batch in the foreground, achieving CPU-GPU overlap
    """

    def __init__(
        self,
        dataset: MeshNearPointPairDataset,
        batch_size: int,
        device: torch.device,
        prefetch_batches: int = 2,
        output_x0: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.device = device
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.output_x0 = bool(output_x0)


        # runtime state
        self._local_perm: Optional[np.ndarray] = None
        self._epoch: int = 0
        self._num_steps: int = 0

        self._queue: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
        self._worker: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._started = False
        self._next_step_to_consume = 0
        self._end_of_epoch = False
        self._exception_in_worker: Optional[BaseException] = None

    # ---------- epoch lifecycle ----------

    def start_epoch(self, local_perm: torch.Tensor, epoch: int):
        """
        Called at the start of each epoch.
        local_perm: the indices for this rank (len = local_len)
        """
        if self._started:
            raise RuntimeError("AsyncLoader.start_epoch called but loader already started; "
                               "call finish_epoch() first.")

        np_perm = local_perm.detach().cpu().numpy().astype(np.int64)
        local_len = len(np_perm)
        if local_len == 0:
            self._num_steps = 0
            self._end_of_epoch = True
            return

        self._local_perm = np_perm
        self._epoch = int(epoch)
        self._num_steps = int(np.ceil(local_len / self.batch_size))
        self._next_step_to_consume = 0
        self._end_of_epoch = False
        self._exception_in_worker = None
        self._stop_flag.clear()

        # Clear the queue (to prevent leftovers from the previous epoch)
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Start the background thread
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"MeshPairAsyncLoader-epoch{epoch}",
            daemon=True,
        )
        self._worker.start()
        self._started = True

    def finish_epoch(self):
        """
        Called at the end of an epoch to ensure the worker exits.
        """
        self._stop_flag.set()
        if self._worker is not None:
            self._worker.join()
        self._worker = None
        self._started = False
        self._local_perm = None
        self._num_steps = 0
        self._next_step_to_consume = 0
        self._end_of_epoch = True
        self._exception_in_worker = None

    @property
    def num_steps(self) -> int:
        return self._num_steps

    # ---------- Foreground: fetch a batch ----------

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called by the foreground training loop:
          - If there is an exception, raise it
          - If the epoch is finished, raise StopIteration
          - Otherwise fetch one batch from the queue and move it to the device
        """
        if self._exception_in_worker is not None:
            # Re-raise the error from the worker
            exc = self._exception_in_worker
            self._exception_in_worker = None
            raise exc

        if self._end_of_epoch or self._next_step_to_consume >= self._num_steps:
            raise StopIteration

        # Blockingly fetch one item from the queue
        item = self._queue.get()
        if item is None:
            # End-of-epoch marker placed by the worker
            self._end_of_epoch = True
            raise StopIteration

        step_idx, x0_np, x1_np = item
        if step_idx != self._next_step_to_consume:
            # In theory this should never be out of order; if it is, just assert
            raise RuntimeError(
                f"AsyncLoader got batch step_idx={step_idx} but expected {self._next_step_to_consume}"
            )

        self._next_step_to_consume += 1

        # Convert to torch and move to the device (can use non_blocking=True)
        x0 = torch.from_numpy(x0_np).to(device=self.device, non_blocking=True)
        x1 = torch.from_numpy(x1_np).to(device=self.device, non_blocking=True)
        return x0, x1

    # ---------- Background thread main loop ----------

    def _worker_loop(self):
        try:
            assert self._local_perm is not None
            local_perm = self._local_perm
            epoch = self._epoch
            local_len = len(local_perm)

            # ============ New: prepare a (#meshes, N, 3) buffer for the whole epoch ============
            save_x0 = self.output_x0
            epoch_x0 = None
            filled_mask = None
            if save_x0:
                # The dataset must expose num_meshes and num_points
                num_meshes = getattr(self.dataset, "num_meshes", None)
                num_points = getattr(self.dataset, "num_points", None)
                if num_meshes is None or num_points is None:
                    raise RuntimeError(
                        "output_x0=True requires dataset to have 'num_meshes' and 'num_points' attributes."
                    )

                epoch_x0 = np.empty((num_meshes, num_points, 3), dtype=np.float32)
                # Use a mask to track which meshes were actually sampled this epoch (handy for debugging)
                filled_mask = np.zeros(num_meshes, dtype=bool)
            # ====================================================================


            for step in range(self._num_steps):
                if self._stop_flag.is_set():
                    break

                start = step * self.batch_size
                end = min((step + 1) * self.batch_size, local_len)
                if end <= start:
                    continue

                idx_batch = local_perm[start:end]
                # Call the dataset's single-batch computation
                x0_np, x1_np = self.dataset.compute_batch(
                    idx_batch=idx_batch,
                    epoch=epoch,
                    step=step,
                )

                # ============ New: fill this batch's x0 into (#meshes, N, 3) ============
                if save_x0 and epoch_x0 is not None:
                    # idx_batch holds the global indices of the meshes in this batch
                    for row_idx, mesh_idx in enumerate(idx_batch):
                        epoch_x0[mesh_idx] = x0_np[row_idx]
                        filled_mask[mesh_idx] = True
                # =================================================================


                # Blockingly put into the queue, with at most prefetch_batches in the queue
                self._queue.put((step, x0_np, x1_np))

            # Tell the foreground: no more batches
            self._queue.put(None)

            if save_x0 and epoch_x0 is not None:

                mesh_root = getattr(self.dataset, "mesh_root", None)
                if mesh_root is None:
                    raise RuntimeError(
                        "dataset must have 'mesh_root' attribute when output_x0=True "
                        "(used to construct mesh_root/sphere_x0)."
                    )

                mesh_root = Path(mesh_root)
                out_dir = mesh_root / "0sphere_x0"
                out_dir.mkdir(parents=True, exist_ok=True)  # equivalent to os.makedirs(..., exist_ok=True)

                out_path = out_dir / f"x0_epoch_{epoch:04d}.npz"
                # points: (# of meshes, N, 3)
                np.savez(out_path, points=epoch_x0)

        except BaseException as e:
            # Pass the exception to the main thread
            self._exception_in_worker = e
            # Finally, push an end marker to avoid the main thread blocking forever
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass


class PointSetPairAsyncLoader:
    """
    Asynchronous batch loader for 2D point sets:
      - A background thread calls dataset.compute_batch(...)
      - Puts results into a bounded queue
      - The foreground training loop calls next_batch() to get (x0, x1) torch.Tensors

    Compatible with UniGBNSampler (which has compute_batch) and PointSetMiniBatchOTDataset
    """

    def __init__(
        self,
        dataset,  # UniGBNSampler or PointSetMiniBatchOTDataset
        batch_size: int,
        device: torch.device,
        prefetch_batches: int = 2,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.device = device
        self.prefetch_batches = max(1, int(prefetch_batches))

        # runtime state
        self._local_perm: Optional[np.ndarray] = None
        self._epoch: int = 0
        self._num_steps: int = 0

        self._queue: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
        self._worker: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._started = False
        self._next_step_to_consume = 0
        self._end_of_epoch = False
        self._exception_in_worker: Optional[BaseException] = None

    # ---------- epoch lifecycle ----------

    def start_epoch(self, local_perm: torch.Tensor, epoch: int):
        """
        Called at the start of each epoch.
        local_perm: the indices for this rank (len = local_len)
        """
        if self._started:
            raise RuntimeError("AsyncLoader.start_epoch called but loader already started; "
                               "call finish_epoch() first.")

        np_perm = local_perm.detach().cpu().numpy().astype(np.int64)
        local_len = len(np_perm)
        if local_len == 0:
            self._num_steps = 0
            self._end_of_epoch = True
            return

        self._local_perm = np_perm
        self._epoch = int(epoch)
        self._num_steps = int(np.ceil(local_len / self.batch_size))
        self._next_step_to_consume = 0
        self._end_of_epoch = False
        self._exception_in_worker = None
        self._stop_flag.clear()

        # Clear the queue (to prevent leftovers from the previous epoch)
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Start the background thread
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"PointSetPairAsyncLoader-epoch{epoch}",
            daemon=True,
        )
        self._worker.start()
        self._started = True

    def finish_epoch(self):
        """
        Called at the end of an epoch to ensure the worker exits.
        """
        self._stop_flag.set()
        if self._worker is not None:
            self._worker.join()
        self._worker = None
        self._started = False
        self._local_perm = None
        self._num_steps = 0
        self._next_step_to_consume = 0
        self._end_of_epoch = True
        self._exception_in_worker = None

    @property
    def num_steps(self) -> int:
        return self._num_steps

    # ---------- Foreground: fetch a batch ----------

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called by the foreground training loop:
          - If there is an exception, raise it
          - If the epoch is finished, raise StopIteration
          - Otherwise fetch one batch from the queue and move it to the device
        """
        if self._exception_in_worker is not None:
            # Re-raise the error from the worker
            exc = self._exception_in_worker
            self._exception_in_worker = None
            raise exc

        if self._end_of_epoch or self._next_step_to_consume >= self._num_steps:
            raise StopIteration

        # Blockingly fetch one item from the queue
        item = self._queue.get()
        if item is None:
            # End-of-epoch marker placed by the worker
            self._end_of_epoch = True
            raise StopIteration

        step_idx, x0_np, x1_np = item
        if step_idx != self._next_step_to_consume:
            raise RuntimeError(
                f"AsyncLoader got batch step_idx={step_idx} but expected {self._next_step_to_consume}"
            )

        self._next_step_to_consume += 1

        # Convert to torch and move to the device
        x0 = torch.from_numpy(x0_np).to(device=self.device, non_blocking=True)
        x1 = torch.from_numpy(x1_np).to(device=self.device, non_blocking=True)
        return x0, x1

    # ---------- Background thread main loop ----------

    def _worker_loop(self):
        try:
            assert self._local_perm is not None
            local_perm = self._local_perm
            epoch = self._epoch
            local_len = len(local_perm)

            for step in range(self._num_steps):
                if self._stop_flag.is_set():
                    break

                start = step * self.batch_size
                end = min((step + 1) * self.batch_size, local_len)
                if end <= start:
                    continue

                idx_batch = local_perm[start:end]
                # Call the dataset's single-batch computation
                x0_np, x1_np = self.dataset.compute_batch(
                    idx_batch=idx_batch,
                    epoch=epoch,
                    step=step,
                )

                # Blockingly put into the queue
                self._queue.put((step, x0_np, x1_np))

            # Tell the foreground: no more batches
            self._queue.put(None)

        except BaseException as e:
            # Pass the exception to the main thread
            self._exception_in_worker = e
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass


class SphereCoordThomsonDataset:
    """
    Loads the spherical-coordinate dataset for the Thomson problem.

    Data format:
    - The NPZ file contains ["points"], shape (num_samples, num_points, 3)
    - Each point is in (theta, phi, r) format, already scaled:
      - theta, phi: [-1, 1]
      - r: [0, 1]

    Output:
    - x0: (B, N, 2) - uniform random in [-1,1]^2 (theta, phi space)
    - x1: (B, N, 4) - (theta, phi, nx, ny)
      where (nx, ny) points in the (x1-x0) direction and nx^2 + ny^2 = r^2
    """

    def __init__(
        self,
        data_path: str,
        num_points: int = 384,
        num_workers: int = 8,
        hilbert_p: int = 10,
        presorted: bool = False,
    ):
        """
        Args:
            data_path: path to the NPZ file containing the ["points"] field
            num_points: number of points per point cloud
            num_workers: number of workers for multithreaded processing
            hilbert_p: Hilbert sort precision (2^p grid resolution)
            presorted: whether the data is already presorted. If True, skip Hilbert sorting
        """
        self.data_path = Path(data_path).expanduser().resolve()
        self.num_points = int(num_points)
        self.num_workers = int(num_workers)
        self.hilbert_p = int(hilbert_p)
        self.presorted = bool(presorted)

        # Load data
        self._load_data()

        sort_status = "presorted" if self.presorted else f"hilbert_p={self.hilbert_p}"
        print(f"[SphereCoordThomsonDataset] Loaded {self.num_meshes} samples, "
              f"each with {self.num_points} points ({sort_status})")

    def _load_data(self):
        """Load the points data from the NPZ file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = np.load(self.data_path)
        if 'points' not in data:
            raise KeyError(f"NPZ file must contain 'points' field. Found: {list(data.keys())}")

        self.points = data['points']  # (num_samples, num_points, 3) = (theta, phi, r)

        if self.points.ndim != 3 or self.points.shape[2] != 3:
            raise ValueError(f"Expected points shape (N, M, 3), got {self.points.shape}")

        if self.points.shape[1] != self.num_points:
            print(f"[WARN] Data has {self.points.shape[1]} points per sample, "
                  f"but num_points={self.num_points}. Using data's value.")
            self.num_points = self.points.shape[1]

    @property
    def num_meshes(self) -> int:
        """Compatibility with the MeshPairAsyncLoader interface"""
        return self.points.shape[0]

    def compute_batch(
        self,
        idx_batch: np.ndarray,
        epoch: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute one batch of (x0, x1)

        Args:
            idx_batch: array of sample indices, shape (B,)
            epoch: current epoch
            step: current step

        Returns:
            x0: (B, N, 2) - uniform random in [-1,1]^2
            x1: (B, N, 4) - (theta, phi, nx, ny)
        """
        B = idx_batch.shape[0]
        N = self.num_points

        # Deterministic random seed
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        # 1) x0: uniform random in [-1,1]^2
        x0_np = rng.random((B, N, 2), dtype=np.float32) * 2.0 - 1.0

        # 2) x1: take points from the data, apply Hilbert sorting, and compute the normal vector components
        x1_np = np.empty((B, N, 4), dtype=np.float32)

        # Multithreaded processing
        results = Parallel(n_jobs=self.num_workers, backend="loky")(
            delayed(self._process_one_sample)(
                bi, int(idx_batch[bi]), x0_np[bi]
            )
            for bi in range(B)
        )

        for bi, x1_sample in results:
            x1_np[bi] = x1_sample

        return x0_np, x1_np

    def _process_one_sample(
        self,
        bi: int,
        sample_idx: int,
        x0_2d: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        """
        Process a single sample

        Args:
            bi: batch index
            sample_idx: sample index in the dataset
            x0_2d: starting point (theta, phi), shape (N, 2)

        Returns:
            (bi, x1_4d): the 4D representation of x1 (theta, phi, nx, ny), shape (N, 4)
        """
        # Get the raw data (N, 3) = (theta, phi, r)
        pts = self.points[sample_idx]  # (N, 3)

        if self.presorted:
            # Data is already presorted; use it directly
            pts_sorted = pts
        else:
            # Use 3D Hilbert sorting (based on theta, phi, r)
            pts_sorted = hilbert_sort_xyz_numba(pts.copy(), p_bits=self.hilbert_p)

        theta_phi_sorted = pts_sorted[:, :2]  # (N, 2)
        r_sorted = pts_sorted[:, 2]           # (N,)

        # Compute (nx, ny): pointing in the (x1-x0) direction with magnitude r
        d = theta_phi_sorted - x0_2d  # (N, 2)
        d_norm = np.linalg.norm(d, axis=-1, keepdims=True)
        d_norm = np.maximum(d_norm, 1e-8)
        d_unit = d / d_norm  # unit direction vector

        # nx, ny = d_unit * r
        n = d_unit * r_sorted[:, np.newaxis]  # (N, 2)

        # x1 = (theta, phi, nx, ny)
        x1_4d = np.concatenate([theta_phi_sorted, n], axis=-1)  # (N, 4)

        return bi, x1_4d.astype(np.float32)


class XYZDataset:
    """
    A simple XYZ point cloud dataset (for 3D flow matching)

    Data format:
    - The NPZ file contains ["points"], shape (num_samples, num_points, 3)
    - xyz coordinates, already in the [-1,1]^3 range, already presorted

    Output:
    - x0: (B, N, 3) - uniform random in [-1,1]^3
    - x1: (B, N, 3) - the xyz coordinates from the data
    """

    def __init__(
        self,
        data_path: str,
        num_points: int = 384,
    ):
        """
        Args:
            data_path: path to the NPZ file containing the ["points"] field
            num_points: number of points per point cloud
        """
        self.data_path = Path(data_path).expanduser().resolve()
        self.num_points = int(num_points)

        # Load data
        self._load_data()

        print(f"[XYZDataset] Loaded {self.num_meshes} samples, "
              f"each with {self.num_points} points")

    def _load_data(self):
        """Load the points data from the NPZ file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = np.load(self.data_path)
        if 'points' not in data:
            raise KeyError(f"NPZ file must contain 'points' field. Found: {list(data.keys())}")

        self.points = data['points']  # (num_samples, num_points, 3) = xyz

        if self.points.ndim != 3 or self.points.shape[2] != 3:
            raise ValueError(f"Expected points shape (N, M, 3), got {self.points.shape}")

        if self.points.shape[1] != self.num_points:
            print(f"[WARN] Data has {self.points.shape[1]} points per sample, "
                  f"but num_points={self.num_points}. Using data's value.")
            self.num_points = self.points.shape[1]

    @property
    def num_meshes(self) -> int:
        """Compatibility with the MeshPairAsyncLoader interface"""
        return self.points.shape[0]

    def compute_batch(
        self,
        idx_batch: np.ndarray,
        epoch: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute one batch of (x0, x1)

        Args:
            idx_batch: array of sample indices, shape (B,)
            epoch: current epoch
            step: current step

        Returns:
            x0: (B, N, 3) - uniform random in [-1,1]^3
            x1: (B, N, 3) - the xyz coordinates from the data
        """
        B = idx_batch.shape[0]
        N = self.num_points

        # Deterministic random seed
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        # x0: uniform random in [-1,1]^3
        x0_np = rng.random((B, N, 3), dtype=np.float32) * 2.0 - 1.0

        # x1: take points directly from the data
        x1_np = self.points[idx_batch].astype(np.float32)  # (B, N, 3)

        return x0_np, x1_np


class XYZDataset2D:
    """
    2D point cloud dataset: takes xy after sorting 3D data by z

    Input: NPZ with ["points"] shape (N, M, 3)
    Processing: sort each sample's M points by z value, then drop the z dimension
    Output:
    - x0: (B, N, 2) - uniform random in [-1,1]^2
    - x1: (B, N, 2) - sorted xy coordinates
    """

    def __init__(
        self,
        data_path: str,
        num_points: int = 2048,
    ):
        """
        Args:
            data_path: path to the NPZ file containing the ["points"] field, shape (N, M, 3)
            num_points: number of points per point cloud
        """
        self.data_path = Path(data_path).expanduser().resolve()
        self.num_points = int(num_points)

        # Load data
        self._load_data()

        print(f"[XYZDataset2D] Loaded {self.num_meshes} samples, "
              f"each with {self.num_points} points (2D)")

    def _load_data(self):
        """Load the NPZ file, then take xy after sorting by z"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = np.load(self.data_path)
        if 'points' not in data:
            raise KeyError(f"NPZ file must contain 'points' field. Found: {list(data.keys())}")

        points_3d = data['points']  # (num_samples, num_points, 3) = xyz

        if points_3d.ndim != 3 or points_3d.shape[2] != 3:
            raise ValueError(f"Expected points shape (N, M, 3), got {points_3d.shape}")

        if points_3d.shape[1] != self.num_points:
            print(f"[WARN] Data has {points_3d.shape[1]} points per sample, "
                  f"but num_points={self.num_points}. Using data's value.")
            self.num_points = points_3d.shape[1]

        # Sort by z and drop the z dimension
        num_samples = points_3d.shape[0]
        self.points = np.empty((num_samples, self.num_points, 2), dtype=np.float32)

        for i in range(num_samples):
            z_vals = points_3d[i, :, 2]  # (M,)
            sort_idx = np.argsort(z_vals)  # sort by z in ascending order
            self.points[i] = points_3d[i, sort_idx, :2]  # take the sorted xy

        print(f"[XYZDataset2D] Sorted by z and kept xy, final shape: {self.points.shape}")

    @property
    def num_meshes(self) -> int:
        """Compatibility with the MeshPairAsyncLoader interface"""
        return self.points.shape[0]

    def compute_batch(
        self,
        idx_batch: np.ndarray,
        epoch: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute one batch of (x0, x1)

        Args:
            idx_batch: array of sample indices, shape (B,)
            epoch: current epoch
            step: current step

        Returns:
            x0: (B, N, 2) - uniform random in [-1,1]^2
            x1: (B, N, 2) - sorted xy coordinates
        """
        B = idx_batch.shape[0]
        N = self.num_points

        # Deterministic random seed
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        # x0: uniform random in [-1,1]^2
        x0_np = rng.random((B, N, 2), dtype=np.float32) * 2.0 - 1.0

        # x1: take points directly from the data (already sorted xy)
        x1_np = self.points[idx_batch].astype(np.float32)  # (B, N, 2)

        return x0_np, x1_np


class XYZMiniBatchOTDataset:
    """
    XYZ point cloud dataset with sample-level minibatch OT pairing.

    Based on XYZDataset, adding batch-level OT matching.

    OT matching logic:
    - Generate B noise point sets x0 (N points each, uniform in [-1,1]^3)
    - Load B data point sets x1
    - Compute a B×B cost matrix (using flattened L2 distance)
    - Find the optimal pairing with greedy/hungarian OT
    - Return the paired (x0, x1[perm])
    """

    def __init__(
        self,
        data_path: str,
        num_points: int = 384,
        ot_solver: str = "hungarian",
        use_multi: bool = True,
        num_workers: int = 8,
    ):
        """
        Args:
            data_path: path to the NPZ file containing the ["points"] field
            num_points: number of points per point cloud
            ot_solver: OT solver ("greedy" or "hungarian")
            use_multi: whether to compute the cost matrix with multiprocessing
            num_workers: number of multiprocessing workers
        """
        self.data_path = Path(data_path).expanduser().resolve()
        self.num_points = int(num_points)
        self.ot_solver = ot_solver
        self.use_multi = use_multi
        self.num_workers = num_workers

        # Load data
        self._load_data()

        print(f"[XYZMiniBatchOTDataset] Loaded {self.num_meshes} samples, "
              f"each with {self.num_points} points, ot_solver={ot_solver}")

    def _load_data(self):
        """Load the points data from the NPZ file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = np.load(self.data_path)
        if 'points' not in data:
            raise KeyError(f"NPZ file must contain 'points' field. Found: {list(data.keys())}")

        self.points = data['points']  # (num_samples, num_points, 3) = xyz

        if self.points.ndim != 3 or self.points.shape[2] != 3:
            raise ValueError(f"Expected points shape (N, M, 3), got {self.points.shape}")

        if self.points.shape[1] != self.num_points:
            print(f"[WARN] Data has {self.points.shape[1]} points per sample, "
                  f"but num_points={self.num_points}. Using data's value.")
            self.num_points = self.points.shape[1]

    @property
    def num_meshes(self) -> int:
        """Compatibility with the MeshPairAsyncLoader interface"""
        return self.points.shape[0]

    def compute_batch(
        self,
        idx_batch: np.ndarray,
        epoch: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute_batch with OT matching.

        1. Generate x0: (B, N, 3) uniform in [-1,1]^3
        2. Load x1: (B, N, 3) read from the dataset
        3. Compute cost matrix C[i,j] = ||x0[i].flatten() - x1[j].flatten()||^2
        4. Get perm via solve_ot_assignment(C, mode=self.ot_solver)
        5. Return (x0, x1[perm])
        """
        from .utils import solve_ot_assignment

        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        B = idx_batch.shape[0]
        N = self.num_points

        # deterministic seed
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        # x0: uniform in [-1, 1]^3
        x0_np = rng.random((B, N, 3), dtype=np.float32) * 2.0 - 1.0

        # x1: load from the dataset
        x1_np = self.points[idx_batch].astype(np.float32)  # (B, N, 3)

        # Flatten: (B, N, 3) -> (B, N*3)
        x0_flat = x0_np.reshape(B, -1)
        x1_flat = x1_np.reshape(B, -1)

        # Compute the B×B cost matrix (flattened L2 distance)
        C = np.zeros((B, B), dtype=np.float32)

        if self.use_multi and self.num_workers > 1 and B > 1:
            # Use joblib multiprocessing for speedup
            results = Parallel(n_jobs=self.num_workers, backend="loky")(
                delayed(_compute_cost_row)(i, x0_flat, x1_flat, B)
                for i in range(B)
            )
            for i, row in results:
                C[i] = row
        else:
            # Single-process computation
            for i in range(B):
                diff = x0_flat[i] - x1_flat  # (B, N*3)
                C[i] = np.sum(diff ** 2, axis=1)  # (B,)

        # OT matching
        perm = solve_ot_assignment(C, mode=self.ot_solver)

        # Apply the permutation
        x1_matched = x1_np[perm]

        return x0_np, x1_matched


class XYZEqOTFMDataset:
    """
    XYZ point cloud dataset with two-level OT matching for Equivariant OT flow matching.

    Based on XYZDataset, adding point-level + batch-level OT matching.

    Difference from XYZMiniBatchOTDataset:
    - XYZMiniBatchOTDataset uses flattened L2 distance for batch-level OT
    - This class first aligns at the point level via permutation, then does batch-level OT

    Two-level OT matching logic:
    1. For each pair (x0[i], x1[j]), find the point-level permutation:
       s*[i,j] = argmin_{s in S(N)} ||x0[i] - x1[j][s]||^2
    2. Build the cost matrix: M[i,j] = cost after alignment
    3. Batch-level OT: sigma* = argmin sum_i M[i, sigma(i)]
    4. Return (x0, aligned_x1[sigma*])
    """

    def __init__(
        self,
        data_path: str,
        num_points: int = 384,
        point_ot_solver: str = "greedy",
        batch_ot_solver: str = "hungarian",
        use_multi: bool = True,
        num_workers: int = 8,
    ):
        """
        Args:
            data_path: path to the NPZ file containing the ["points"] field
            num_points: number of points per point cloud
            point_ot_solver: point-level OT solver ("greedy" or "hungarian")
            batch_ot_solver: batch-level OT solver ("greedy" or "hungarian")
            use_multi: whether to use multiprocessing for the computation
            num_workers: number of multiprocessing workers
        """
        self.data_path = Path(data_path).expanduser().resolve()
        self.num_points = int(num_points)
        self.point_ot_solver = point_ot_solver
        self.batch_ot_solver = batch_ot_solver
        self.use_multi = use_multi
        self.num_workers = num_workers

        # Load data
        self._load_data()

        print(f"[XYZEqOTFMDataset] Loaded {self.num_meshes} samples, "
              f"each with {self.num_points} points, "
              f"point_ot={point_ot_solver}, batch_ot={batch_ot_solver}")

    def _load_data(self):
        """Load the points data from the NPZ file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = np.load(self.data_path)
        if 'points' not in data:
            raise KeyError(f"NPZ file must contain 'points' field. Found: {list(data.keys())}")

        self.points = data['points']  # (num_samples, num_points, 3) = xyz

        if self.points.ndim != 3 or self.points.shape[2] != 3:
            raise ValueError(f"Expected points shape (N, M, 3), got {self.points.shape}")

        if self.points.shape[1] != self.num_points:
            print(f"[WARN] Data has {self.points.shape[1]} points per sample, "
                  f"but num_points={self.num_points}. Using data's value.")
            self.num_points = self.points.shape[1]

    @property
    def num_meshes(self) -> int:
        """Compatibility with the MeshPairAsyncLoader interface"""
        return self.points.shape[0]

    def compute_batch(
        self,
        idx_batch: np.ndarray,
        epoch: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute_batch with Equivariant OT matching.

        1. Generate x0: (B, N, 3) uniform in [-1,1]^3
        2. Load x1: (B, N, 3) read from the dataset
        3. Point-level OT: for each pair (i,j), find the permutation aligning x1[j] to x0[i]
        4. Build the batch cost matrix M[i,j]
        5. Batch-level OT: find the optimal sample pairing sigma*
        6. Return (x0, x1_aligned) where x1_aligned[i] = x1[sigma*(i)][point_perm[i, sigma*(i)]]
        """
        from .utils import solve_ot_assignment

        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        B = idx_batch.shape[0]
        N = self.num_points

        # deterministic seed
        seed = 0xABCDEF + int(epoch) * 100000 + int(step)
        rng = np.random.default_rng(seed=seed)

        # x0: uniform in [-1, 1]^3
        x0_np = rng.random((B, N, 3), dtype=np.float32) * 2.0 - 1.0

        # x1: load from the dataset
        x1_np = self.points[idx_batch].astype(np.float32)  # (B, N, 3)

        # ============== Two-level OT matching ==============
        # Step 1: Point-level OT, compute the B x B cost matrix and the corresponding permutations
        M = np.zeros((B, B), dtype=np.float32)
        perms = {}  # (i, j) -> point permutation array

        if self.use_multi and self.num_workers > 1 and B > 1:
            # Use joblib multiprocessing for speedup, passing slices to avoid pickling whole arrays
            results = Parallel(n_jobs=self.num_workers, backend="loky")(
                delayed(_compute_point_ot_pair_slim)(i, j, x0_np[i], x1_np[j], self.point_ot_solver)
                for i in range(B) for j in range(B)
            )
            for i, j, cost, perm in results:
                M[i, j] = cost
                perms[(i, j)] = perm
        else:
            # Single-process computation
            for i in range(B):
                for j in range(B):
                    cost, perm = _compute_point_level_ot(x0_np[i], x1_np[j], self.point_ot_solver)
                    M[i, j] = cost
                    perms[(i, j)] = perm

        # Step 2: Batch-level OT
        batch_perm = solve_ot_assignment(M, mode=self.batch_ot_solver)

        # Step 3: Build the aligned x1
        x1_aligned = np.zeros_like(x0_np)
        for i in range(B):
            j = batch_perm[i]
            point_perm = perms[(i, j)]
            x1_aligned[i] = x1_np[j][point_perm]

        return x0_np, x1_aligned