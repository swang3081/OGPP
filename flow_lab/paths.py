from abc import ABC, abstractmethod
import torch
import math
from torch.func import vmap, jacrev
from .distributions import Sampleable, IsotropicGaussian
import torch.nn as nn
from typing import Optional, List, Type, Tuple, Dict, Iterator, Any
from .sort_numba import hilbert_sort_xy_fast
from .dynamics import wrap_coords, minimal_image
import ot
import os
import numpy as np
import time
import concurrent.futures as cf
import lap
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import linear_sum_assignment
from torch.utils.data import IterableDataset, get_worker_info

try:
    import lap  # pip install lap
    _HAS_LAP = True
except Exception:
    _HAS_LAP = False

def pairwise_torus_dist2_matrix(xy: np.ndarray) -> np.ndarray:
    """
    Return D2[i,j] = torus squared distance between point i and j (float32).
    O(N^2) once; N=1024 is fine.
    """
    x = xy[:,0][:,None] - xy[:,0][None,:]
    y = xy[:,1][:,None] - xy[:,1][None,:]
    x = x - np.round(x)
    y = y - np.round(y)
    D2 = (x*x + y*y).astype(np.float32)
    return D2

def knn_candidates(D2: np.ndarray, k: int) -> np.ndarray:
    """
    From D2 (with diagonal set to inf), pick k nearest neighbors per node.
    Return (N,k) int32 array of neighbor node indices.
    """
    N = D2.shape[0]
    D2 = D2.copy()
    np.fill_diagonal(D2, np.inf)
    # argpartition: O(Nk)
    idx = np.argpartition(D2, kth=min(k, N-1)-1, axis=1)[:, :k]
    # optional: sort each row by distance
    row_ar = np.arange(N)[:, None]
    dist_rows = D2[row_ar, idx]
    order_in_row = np.argsort(dist_rows, axis=1)
    idx_sorted = idx[row_ar, order_in_row]
    return idx_sorted.astype(np.int32)

def two_opt_fast(xy: np.ndarray,
                 init_order: np.ndarray,
                 D2: np.ndarray,
                 cand: np.ndarray,
                 passes: int = 2,
                 first_improvement: bool = True,
                 *,
                 forbid_cross: bool = True,
                 cross: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
    """
    Fast 2-opt using a candidate adjacency list + precomputed D2 (squared distances).
    Additionally supports "forbid wrap-around crossings":
      - If forbid_cross=True, only accept 2-opt moves that do not create new wrap-around edges.
      - Requires the cross matrix (computed by torus_cross_matrix). If not provided and forbid_cross=True, it is computed on the fly (slow).

    Parameters
    ----
    xy : (N,2)  coordinates (used as a fallback for on-the-fly cross computation)
    init_order : (N,) initial tour
    D2 : (N,N)  squared edge "length" (recommended to use the shortest squared distance under the torus metric)
    cand : (N,k) candidate neighbor indices for each vertex
    passes : number of outer iterations (each pass scans to a local optimum)
    first_improvement : True=first-improvement (take the first improving move); False=steepest-descent
    forbid_cross : whether to forbid edges that create a new wrap-around crossing
    cross : (N,N) boolean matrix; cross[i,j]=True means the i->j edge uses wrap (crosses the seam)

    Returns
    ----
    order, info
    """
    N = len(init_order)
    order = np.array(init_order, dtype=np.int64).copy()
    pos = np.empty(N, dtype=np.int32)
    pos[order] = np.arange(N, dtype=np.int32)

    if forbid_cross and cross is None:
        cross = torus_cross_matrix(xy)  # compute only when needed

    total_moves = 0
    for _ in range(max(1, passes)):
        improved_any = False
        changed = True
        while changed:
            changed = False
            for i in range(N):
                a = order[i]
                b = order[(i + 1) % N]
                ab = D2[a, b]

                # Candidate nodes = neighbors(a) ∪ neighbors(b)
                ca = cand[a]
                cb = cand[b]
                cs = np.unique(np.concatenate([ca, cb]))

                # Positions of candidates on the tour
                kpos = pos[cs]
                # Filter illegal positions: must not share an endpoint with (i,i+1), and avoid (k,k+1) overlapping i
                mask = (kpos != i) & (kpos != (i + 1) % N) & (((kpos + 1) % N) != i)
                cs = cs[mask]
                kpos = kpos[mask]
                if cs.size == 0:
                    continue

                dpos = (kpos + 1) % N
                c_nodes = cs
                d_nodes = order[dpos]

                # Vectorized improvement amount: Δ = (a,c)+(b,d) - (a,b)-(c,d)
                delta = (D2[a, c_nodes] + D2[b, d_nodes]) - (ab + D2[c_nodes, d_nodes])

                # If forbidding wrap-around crossings: filter out candidates that would create a new crossing
                if forbid_cross:
                    # Whether the new edges cross the seam
                    new_cross = cross[a, c_nodes] | cross[b, d_nodes]
                    # Only allow new connections that do not cross the seam at all
                    keep_nc = ~new_cross
                    if not np.any(keep_nc):
                        continue
                    # Pick improvements only within the non-crossing set
                    idx_improve = np.where((delta < -1e-12) & keep_nc)[0]
                else:
                    idx_improve = np.where(delta < -1e-12)[0]

                if idx_improve.size == 0:
                    continue

                if first_improvement:
                    idx_best = int(idx_improve[0])
                else:
                    idx_best = int(idx_improve[np.argmin(delta[idx_improve])])

                k = int(kpos[idx_best])

                # Reverse [i+1 .. k] (circular interval)
                i2 = (i + 1) % N
                if i2 <= k:
                    order[i2:k + 1] = order[i2:k + 1][::-1]
                    pos[order[i2:k + 1]] = np.arange(i2, k + 1, dtype=np.int32)
                else:
                    seg = np.concatenate([order[i2:], order[:k + 1]])[::-1]
                    L = N - i2
                    order[i2:] = seg[:L]
                    order[:k + 1] = seg[L:]
                    pos[order] = np.arange(N, dtype=np.int32)

                total_moves += 1
                changed = True
                improved_any = True

        if not improved_any:
            break

    info = {
        "two_opt_fast_moves": int(total_moves),
        "two_opt_fast_passes": int(passes),
        "forbid_cross": bool(forbid_cross),
    }
    return order, info

@torch.no_grad()
def validate_batched_perm(perm: torch.Tensor, n: Optional[int] = None, raise_on_error: bool = False):
    """
    Validate a batched permutation:
      - Dimensions must be [B, N]
      - Value range in [0, n) (default n=N)
      - Each column appears exactly once (no duplicates, no missing values)
      - -1 is not allowed

    Returns:
      ok_mask: [B] bool, whether each batch is valid
      report:  dict, containing the duplicate columns / missing columns / out-of-range positions for each batch (non-empty only when invalid)
    """
    assert perm.dim() == 2, "perm must be [B, N]"
    B, N = perm.shape
    if n is None:
        n = N
    if perm.dtype not in (torch.int64, torch.long, torch.int32):
        raise TypeError(f"perm dtype must be integer, got {perm.dtype}")

    # 1) Range check (including -1)
    bad_range_mask = (perm < 0) | (perm >= n)         # [B,N]
    has_bad_range  = bad_range_mask.any(dim=1)        # [B]

    # 2) Count occurrences per column (batched scatter_add)
    ones   = torch.ones_like(perm, dtype=torch.int32)
    counts = torch.zeros((B, n), dtype=torch.int32, device=perm.device)
    # Note: clamp_min(0) here to avoid crashing on negative indices; out-of-range is already flagged True in has_bad_range
    counts.scatter_add_(1, perm.clamp_min(0), ones)   # [B, n]

    dup_mask     = counts > 1                         # [B, n]  >1 means duplicate
    missing_mask = counts == 0                        # [B, n]  ==0 means missing

    has_dup     = dup_mask.any(dim=1)                 # [B]
    has_missing = missing_mask.any(dim=1)             # [B]

    ok_mask = ~(has_bad_range | has_dup | has_missing)

    # Build the report (only for failed batches)
    report = {}
    if (~ok_mask).any():
        bad_bs = torch.nonzero(~ok_mask, as_tuple=False).flatten().tolist()
        details = []
        for b in bad_bs:
            entry = {
                "batch": int(b),
                "bad_range_pos": torch.nonzero(bad_range_mask[b], as_tuple=False).flatten().tolist(),
                "dup_cols": torch.nonzero(dup_mask[b], as_tuple=False).flatten().tolist(),
                "missing_cols": torch.nonzero(missing_mask[b], as_tuple=False).flatten().tolist(),
            }
            details.append(entry)
        report["invalid_batches"] = details
        report["counts"] = counts  # can be used for further debugging (Tensor)

        if raise_on_error:
            msg = "Permutation check failed for batches: " + ", ".join(map(str, bad_bs))
            raise ValueError(msg)

    return ok_mask, report
def _lap_row2col(C_np: np.ndarray, *, extend_cost: bool = False) -> np.ndarray:
    """
    Return a row2col of shape (N,): the column index corresponding to row i.
    Compatible with the different return signatures of lap.lapjv; falls back to SciPy if necessary.
    """
    N = C_np.shape[0]
    if _HAS_LAP:
        try:
            # Prefer not requesting cost; many versions return two items (row2col, col2row)
            res = lap.lapjv(C_np, extend_cost=extend_cost, return_cost=False)
            if isinstance(res, (tuple, list)):
                row2col = res[0]
            else:
                arr = np.asarray(res)
                # Some implementations may directly return a single 1D or a 2xN array
                if arr.ndim == 1 and arr.shape == (N,):
                    row2col = arr
                elif arr.ndim == 2 and arr.shape == (2, N):
                    row2col = arr[0]
                else:
                    raise ValueError(f"Unexpected lapjv output shape {arr.shape}")
        except TypeError:
            # Some versions require return_cost=True (returning three items)
            cost, row2col, col2row = lap.lapjv(C_np, extend_cost=extend_cost, return_cost=True)
        row2col = np.asarray(row2col, dtype=np.int64)
        # Safety fallback: if there are -1 values (should not occur for a square matrix), fall back to SciPy
        if row2col.shape != (N,) or (row2col < 0).any():
            r, c = linear_sum_assignment(C_np, maximize=False)
            row2col = np.empty(N, dtype=np.int64); row2col[r] = c
        return row2col

    # ---- Fallback when lap is unavailable ----
    r, c = linear_sum_assignment(C_np, maximize=False)
    perm = np.empty(N, dtype=np.int64); perm[r] = c
    return perm

def _hungarian_one(C_np: np.ndarray) -> np.ndarray:
    # Our case is a square matrix (N==N), so extend_cost is not needed
    return _lap_row2col(C_np, extend_cost=False)

def _solve_batch_LAP(C_batched: torch.Tensor, threads: int = 0) -> torch.Tensor:
    """
    C_batched: (B, N, N) CPU tensor
    Returns perm: (B, N) long, each row is a row2col
    """
    B, N, _ = C_batched.shape
    C_np = C_batched.detach().cpu().numpy()
    out = np.empty((B, N), dtype=np.int64)

    if threads and threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            perms = list(ex.map(_hungarian_one, (C_np[i] for i in range(B))))
        for i, p in enumerate(perms):
            # Here p is guaranteed to be (N,)
            out[i] = p
    else:
        for i in range(B):
            out[i] = _hungarian_one(C_np[i])

    return torch.from_numpy(out)  # (B, N)

@torch.no_grad()
def harden_transport_scipy_mt(T: torch.Tensor, topk: Optional[int] = None, threads: Optional[int] = None) -> torch.Tensor:
    """
    Use the SciPy Hungarian algorithm on CPU to harden T:[B,N,N] -> perm:[B,N], processing each batch in parallel with multiple threads.
    - topk: keep only the top-k largest candidates per row; fill the rest with a large cost (rather than inf) to preserve feasibility.
    - threads: number of parallel threads; defaults to min(B, number of CPU cores).
    """
    assert T.dim() == 3 and T.shape[1] == T.shape[2], "T must be [B,N,N]"
    B, N, _ = T.shape
    device = T.device

    # Prepare the "profit" matrix on CPU / float64; non-topk entries use a very large negative profit (=> large positive cost)
    BIG_NEG = -1e30
    if topk is not None and topk < N:
        # Do topk on T's device first, then move the needed data to CPU
        vals, idxs = T.topk(topk, dim=-1, largest=True, sorted=False)  # [B,N,K]
        vals_cpu = vals.to(dtype=torch.float64, device='cpu', non_blocking=True)
        idxs_cpu = idxs.to(device='cpu', non_blocking=True)

        profit = torch.full((B, N, N), BIG_NEG, dtype=torch.float64)   # initialize all to a tiny profit
        profit.scatter_(2, idxs_cpu, vals_cpu)                          # write real profit only on the topk edges
    else:
        profit = T.detach().to('cpu', dtype=torch.float64, non_blocking=True)

    # SciPy minimizes cost => cost = -profit (float64 is more stable)
    cost_np = (-profit).contiguous().numpy()  # shape [B,N,N], dtype=float64

    # Single-batch solver (releases the GIL at the C level, allowing multithreading)
    def solve_one(b: int) -> np.ndarray:
        cmat = cost_np[b]
        # Safety: replace any non-finite values with a large positive cost
        np.nan_to_num(cmat, copy=False, nan=1e30, posinf=1e30, neginf=1e30)
        r, c = linear_sum_assignment(cmat)
        # Square matrix assumed here; if rectangular in the future, c can still be returned directly
        return c.astype(np.int64, copy=False)

    # Run the B problems concurrently
    # print(B, os.cpu_count())
    n_workers = threads or min(B, (os.cpu_count() or 1))
    if n_workers <= 1 or B == 1:
        cols = [solve_one(b) for b in range(B)]
    else:
        cols = [None] * B
        # print("Solving LAP with thread #:", n_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(solve_one, b): b for b in range(B)}
            for fut in as_completed(futs):
                b = futs[fut]
                cols[b] = fut.result()

    perm = torch.from_numpy(np.stack(cols, axis=0)).to(device=device, dtype=torch.long)  # [B,N]
    return perm
@torch.no_grad()
def assign_from_cost_scipy_mt(C: torch.Tensor, k_smallest: Optional[int] = None, threads: Optional[int] = None) -> torch.Tensor:
    """
    Solve the minimum linear assignment from the cost matrix C:[B,N,N] (smaller is better), returning perm:[B,N] (row->column).
    - k_smallest: keep only the k smallest candidates per row, filling the rest with a large cost; None means no pruning.
    - threads: number of parallel threads; defaults to min(B, number of CPU cores).
    """
    assert C.dim() == 3 and C.shape[1] == C.shape[2], "C must be [B,N,N]"
    B, N, _ = C.shape
    device = C.device

    BIG_POS = 1e30
    # Move to CPU / float64 (SciPy is more stable)
    C_cpu = C.detach().to('cpu', dtype=torch.float64)

    if k_smallest is not None and k_smallest < N:
        # Select the k smallest entries per row
        vals, idxs = torch.topk(C_cpu, k=k_smallest, dim=-1, largest=False, sorted=False)
        cost = torch.full((B, N, N), BIG_POS, dtype=torch.float64)
        cost.scatter_(2, idxs, vals)
    else:
        cost = C_cpu.contiguous()

    cost_np = cost.numpy()

    def solve_one(b: int) -> np.ndarray:
        cmat = cost_np[b]
        np.nan_to_num(cmat, copy=False, nan=BIG_POS, posinf=BIG_POS, neginf=BIG_POS)
        r, c = linear_sum_assignment(cmat, maximize=False)
        perm = np.empty(N, dtype=np.int64)
        perm[r] = c
        return perm

    n_workers = threads or min(B, (os.cpu_count() or 1))
    # print("num of workers:", n_workers)
    if n_workers > 1 and B > 1:
        cols = [None] * B
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(solve_one, b): b for b in range(B)}
            for fut in as_completed(futs):
                cols[futs[fut]] = fut.result()
    else:
        cols = [solve_one(b) for b in range(B)]

    perm = torch.from_numpy(np.stack(cols, axis=0)).to(device=device, dtype=torch.long)  # [B,N]
    return perm


@torch.no_grad()
def pairwise_sqdist_torus_batched(x0: torch.Tensor, z: torch.Tensor, L: float) -> torch.Tensor:
    """
    x0, z: [B,N,2] or [N,2], with coordinates in the range [-L/2, L/2)
    Returns C: [B,N,N], the squared shortest-torus-distance cost matrix
    """
    if x0.dim() == 2:  # [N,2] => [1,N,2]
        x0 = x0.unsqueeze(0); z = z.unsqueeze(0)
    B, N, _ = x0.shape
    # (B,N,1), (B,1,N)
    dx = (x0[..., 0].unsqueeze(-1) - z[..., 0].unsqueeze(-2)).abs()
    dy = (x0[..., 1].unsqueeze(-1) - z[..., 1].unsqueeze(-2)).abs()
    dx = torch.minimum(dx, L - dx)
    dy = torch.minimum(dy, L - dy)
    C = dx * dx + dy * dy
    return C  # [B,N,N]



@torch.no_grad()
def batched_sinkhorn_uniform(
    C: torch.Tensor,
    reg: float = 0.05,
    iters: int = 200,
    solver: str = "log_sinkhorn",   # 'log_sinkhorn' or 'sinkhorn'
    tol: float = 1e-3,
):
    """
    Run Sinkhorn on (B,N,N) using POT's batched solver, explicitly passing the uniform marginals a,b on the GPU
    to avoid mixing CPU/GPU.
    """
    assert C.dim() == 3, "C must be [B,N,N]"
    B, N, _ = C.shape
    device, dtype = C.device, C.dtype

    # Explicitly construct uniform marginals on the same device/dtype
    a = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    b = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)

    res = ot.batch.solve_batch(
        C, reg,
        a=a, b=b,                 # <== key: pass a,b in (on the GPU)
        max_iter=iters, tol=tol,
        solver=("log_sinkhorn" if solver in ("log_sinkhorn", "sinkhorn_log") else "sinkhorn"),
        reg_type="entropy",
        grad="detach",
    )
    T = res.plan
    return T.to(device=device, dtype=dtype)

@torch.no_grad()
def gather_by_perm(points: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    points: [B,N,2], perm: [B,N], returns points[b, perm[b], :]
    """
    B, N, D = points.shape
    idx = perm.unsqueeze(-1).expand(B, N, D)
    return torch.gather(points, dim=1, index=idx)

class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x), (num_samples, c, h, w)
        """
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(num_samples) # (num_samples, c, h, w)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # (num_samples, c, h, w)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        pass
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        pass
        
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """ 
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """ 
        pass

class Alpha(ABC):
    def __init__(self):
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.zeros(1,1,1,1)
        )
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.ones(1,1,1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)
    
class Beta(ABC):
    def __init__(self):
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.ones(1,1,1,1)
        )
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pass 

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)

class Gamma(ABC):
    def __init__(self):
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pass 

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)


class LinearAlpha(Alpha):    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)
        
class LinearBeta(Beta):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1-t
        
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return - torch.ones_like(t)

class LinearGamma(Gamma):
    def __init__(self, lamda: float = 1.0):
        self.lamda = lamda
        assert torch.allclose(self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1))


    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.lamda * (1-t)
        
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return - torch.ones_like(t) * self.lamda
    
    
class SquareRootBeta(Beta):
    def __call__(self, t: torch.Tensor) -> torch.Tensor: return torch.sqrt(1.0 - t)
    def dt(self, t: torch.Tensor) -> torch.Tensor: return -0.5 / (torch.sqrt(1 - t) + 1e-4)

class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape = p_simple_shape, std = 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        # x_near = self.alpha(t) * z + self.beta(t) * (torch.randn_like(z) * 0.6 - 0.3 + z)
        # x_far = self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
        # mask = (torch.rand(z.size(0), 1, 1, device=z.device) < 0.5).float() 
        # return mask * x_near + (1 - mask) * x_far

        # original sample conditional path
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)

        
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """ 
        alpha_t = self.alpha(t) # (num_samples, 1, 1, 1)
        beta_t = self.beta(t) # (num_samples, 1, 1, 1)
        dt_alpha_t = self.alpha.dt(t) # (num_samples, 1, 1, 1)
        dt_beta_t = self.beta.dt(t) # (num_samples, 1, 1, 1)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """ 
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2

class GaussianConditionalProbabilityPathSort(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape = p_simple_shape, std = 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        # x_near = self.alpha(t) * z + self.beta(t) * (torch.randn_like(z) * 0.6 - 0.3 + z)
        # x_far = self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
        # mask = (torch.rand(z.size(0), 1, 1, device=z.device) < 0.5).float() 
        # return mask * x_near + (1 - mask) * x_far

        # sort sample conditional path
        noise_sorted = torch.from_numpy(
            hilbert_sort_xy_fast(
                torch.randn_like(z).detach().cpu().numpy()
            )
        ).to(z.device)
        return self.alpha(t) * z + self.beta(t) * noise_sorted


       
        
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """ 
        alpha_t = self.alpha(t) # (num_samples, 1, 1, 1)
        beta_t = self.beta(t) # (num_samples, 1, 1, 1)
        dt_alpha_t = self.alpha.dt(t) # (num_samples, 1, 1, 1)
        dt_beta_t = self.beta.dt(t) # (num_samples, 1, 1, 1)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """ 
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2

class LinearConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__(p_simple, p_data)
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return self.p_data.sample(num_samples)[0]
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        num_samples = z.shape[0]; 
        x0, _ = self.p_simple.sample(num_samples)
        return (1 - t) * x0 + t * z
    def sample_conditional_path_inputx0(
        self,
        x0: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return (1.0 - t) * x0 + t * z
    def conditional_vector_field_inputx0(self, x0: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return z - x0

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (z - x) / (1 - t)
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("No closed-form conditional score for linear path.")

class GenVPConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    Generalized Variance Preserving (VP) conditional probability path.

    Uses trigonometric interpolation:
        alpha_t = sin(0.5 * pi * t)   # 0 at t=0, 1 at t=1
        sigma_t = cos(0.5 * pi * t)   # 1 at t=0, 0 at t=1

    Path: x_t = alpha_t * z + sigma_t * x0
    Vector field: v = dot_alpha_t * z + dot_sigma_t * x0

    where z is x1 (target) and x0 is from the simple distribution.
    """
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__(p_simple, p_data)

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return self.p_data.sample(num_samples)[0]

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from p_t(x|z) by interpolating between noise x0 and target z.
        """
        num_samples = z.shape[0]
        x0, _ = self.p_simple.sample(num_samples)
        alpha_t = torch.sin(0.5 * math.pi * t)
        sigma_t = torch.cos(0.5 * math.pi * t)
        return alpha_t * z + sigma_t * x0

    def sample_conditional_path_inputx0(
        self,
        x0: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample x_t given x0 and z (x1).

        x_t = alpha_t * z + sigma_t * x0
        """
        alpha_t = torch.sin(0.5 * math.pi * t)
        sigma_t = torch.cos(0.5 * math.pi * t)
        return alpha_t * z + sigma_t * x0

    def conditional_vector_field_inputx0(
        self,
        x0: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the reference velocity field given x0 and z (x1).

        v_ref = dot_alpha_t * z + dot_sigma_t * x0

        where:
            dot_alpha_t = 0.5 * pi * cos(0.5 * pi * t)
            dot_sigma_t = -0.5 * pi * sin(0.5 * pi * t)
        """
        dot_alpha_t = 0.5 * math.pi * torch.cos(0.5 * math.pi * t)
        dot_sigma_t = -0.5 * math.pi * torch.sin(0.5 * math.pi * t)
        return dot_alpha_t * z + dot_sigma_t * x0

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("No closed-form conditional_vector_field for GenVP path without x0.")

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("No closed-form conditional score for GenVP path.")

class QuadraticHermiteNormalConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    Quadratic Hermite path:
      - Constraint x(0) = x0
      - Constraint x(1) = x1
      - Constraint x'(1) ~ n1 (endpoint tangent along the normal)

    Concrete form:
        d   = x1 - x0
        v1  ~ n1 (normalized, then multiplied by a tunable scale)
        α(t) = 2t - t^2
        β(t) = t^2 - t

        x(t) = x0 + α(t) * d + β(t) * v1
    """

    def __init__(self, p_simple: Sampleable, p_data: Sampleable, in_out_dim: int = None,
                 normalize_tangent: bool = True,
                 tangent_scale_mode: str = "unit",
                 lambda_orient: float = 0.2):
        """
        Args:
            p_simple: simple distribution
            p_data: data distribution
            in_out_dim: spatial dimension; defaults to auto-inferred or 3
            normalize_tangent: whether to normalize the tangent vector v1.
                - True (default): v1 = n1 / ||n1||, tangent is a unit vector
                - False: v1 = n1, use n1 directly as the tangent (retains magnitude information, e.g. r)
            tangent_scale_mode: endpoint velocity scaling mode (only takes effect when normalize_tangent=True)
                - "unit": v1 = n1_hat (unit vector, default)
                - "original": v1 = n1 (use the raw normal directly, no normalization)
                - "chord": v1 = ||x1-x0|| * n1_hat (chord-length scaling)
                - "arc_length": v1 = D * (1 + λ*(1-S)) * n1_hat (arc-length estimate scaling)
            lambda_orient: misalignment penalty coefficient for the arc-length estimate (arc_length mode only)
        """
        super().__init__(p_simple, p_data)
        # Auto-infer or use the passed-in value
        if in_out_dim is not None:
            self.in_out_dim = in_out_dim
        elif hasattr(p_simple, 'shape') and len(p_simple.shape) > 0:
            self.in_out_dim = p_simple.shape[-1]
        else:
            self.in_out_dim = 3  # default 3D for backward compatibility

        self.normalize_tangent = normalize_tangent
        self.tangent_scale_mode = tangent_scale_mode
        self.lambda_orient = lambda_orient

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return RuntimeError("Not implemented.")

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return RuntimeError("Not implemented.")

    @staticmethod
    def _broadcast_t(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        t: any of (B,), (B,1), (B,1,1), (B,1,1,1)
        ref: e.g. (B, 3) or (B, N, 3)
        Returns: t_b, which can be broadcast against ref
        """
        t_b = t
        while t_b.dim() < ref.dim():
            t_b = t_b.unsqueeze(-1)
        return t_b

    def _compute_v1(self, x0: torch.Tensor, x1: torch.Tensor, n1: torch.Tensor) -> torch.Tensor:
        """
        Compute the endpoint velocity v1, according to tangent_scale_mode.

        Args:
            x0: start position (..., dim)
            x1: end position (..., dim)
            n1: endpoint normal (..., dim)

        Returns:
            v1: endpoint velocity (..., dim)
        """
        eps = 1e-8

        # When normalize_tangent=False, return n1 directly (backward compatible)
        if not self.normalize_tangent:
            return n1

        # "original" mode: use the raw n1 directly, without any normalization
        if self.tangent_scale_mode == "original":
            return n1

        # Compute the unit normal
        n_norm = torch.linalg.norm(n1, dim=-1, keepdim=True).clamp_min(eps)
        n_hat = n1 / n_norm

        # "unit" mode: return the unit vector (original default behavior)
        if self.tangent_scale_mode == "unit":
            return n_hat

        # Need to compute chord information
        d = x1 - x0
        d_norm = torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(eps)  # D

        # "chord" mode: v1 = D * n_hat
        if self.tangent_scale_mode == "chord":
            return d_norm * n_hat

        # "arc_length" mode: v1 = D * (1 + λ*(1 - S)) * n_hat
        elif self.tangent_scale_mode == "arc_length":
            d_hat = d / d_norm
            S = (d_hat * n_hat).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            arc_scale = d_norm * (1.0 + self.lambda_orient * (1.0 - S))
            return arc_scale * n_hat

        # fallback: return the unit vector
        return n_hat

    def sample_conditional_path_inputx0(
        self,
        x0: torch.Tensor,   # (..., dim)
        z: torch.Tensor,    # (..., 2*dim) = [x1(dim), n1(dim)]
        t: torch.Tensor,    # (...,) or (B,) / (B,1,1,1), all acceptable
    ) -> torch.Tensor:
        """
        Use a quadratic Hermite curve:
          x(t) = x0 + α(t) * (x1 - x0) + β(t) * v1

        where:
          - α(t) = 2t - t^2
          - β(t) = t^2 - t
          - v1 is computed by _compute_v1, with the scaling determined by tangent_scale_mode
        Returns: position (..., dim)
        """
        # Split z
        dim = self.in_out_dim
        x1 = z[..., :dim]      # end position
        n1 = z[..., dim:]      # endpoint normal

        # Compute v1 (using the unified method)
        v1 = self._compute_v1(x0, x1, n1)

        # Broadcast t to match x0's last dimension
        t_b = self._broadcast_t(t, x0)   # shape like (..., 1)
        t2 = t_b * t_b

        # α(t), β(t)
        alpha = 2.0 * t_b - t2           # 2t - t^2
        beta  = t2 - t_b                 # t^2 - t

        d = x1 - x0                      # (..., dim)

        # x(t) = x0 + α*d + β*v1
        x_t = x0 + alpha * d + beta * v1
        return x_t

    def conditional_vector_field(
        self,
        x: torch.Tensor,    # (..., dim) current position x_t
        z: torch.Tensor,    # (..., 2*dim) = [x1(dim), n1(dim)]
        t: torch.Tensor,    # (...,) or (B,) / (B,1,1,1)
    ) -> torch.Tensor:
        """
        Give the analytic vector field u(x,z,t) on the quadratic Hermite curve, satisfying:
          if x = φ_t(x0, z), then u(x,z,t) = d/dt φ_t(x0,z).

        Using the decomposition:
          x(t) = A(t) * x0 + B(t)
        where:
          α(t) = 2t - t^2
          β(t) = t^2 - t
          A(t) = 1 - α(t) = 1 - 2t + t^2
          B(t) = α(t) * x1 + β(t) * v1

        Then:
          u(x,z,t) = (A'(t)/A(t)) * (x - B(t)) + B'(t)

        Note: for chord/arc_length modes, x0 must first be recovered from x;
        to simplify the implementation, these modes use an approximation (assuming x0 ≈ (x - α*x1) / (1-α))
        """
        # Split z
        dim = self.in_out_dim
        x1 = z[..., :dim]
        n1 = z[..., dim:]

        # Broadcast t to match x's dimensions
        t_b = self._broadcast_t(t, x)   # (..., 1)
        t2 = t_b * t_b

        # α, β
        alpha = 2.0 * t_b - t2          # 2t - t^2
        beta  = t2 - t_b                # t^2 - t

        # A(t) = 1 - α(t)
        A = 1.0 - alpha                 # = 1 - 2t + t^2
        A_safe = A.clamp_min(1e-4)

        # For chord/arc_length modes, estimate x0 first
        if self.tangent_scale_mode in ["chord", "arc_length"] and self.normalize_tangent:
            # Use unit v1 to roughly estimate x0 first
            eps = 1e-8
            n_norm = torch.linalg.norm(n1, dim=-1, keepdim=True).clamp_min(eps)
            n_hat = n1 / n_norm
            v1_unit = n_hat

            # x ≈ x0 + α*(x1-x0) + β*v1_unit
            # x0 ≈ (x - α*x1 - β*v1_unit) / (1-α)
            x0_approx = (x - alpha * x1 - beta * v1_unit) / A_safe

            # Use the estimated x0 to compute the true v1
            v1 = self._compute_v1(x0_approx, x1, n1)
        else:
            # In "unit" or "original" mode, v1 does not depend on x0
            v1 = self._compute_v1(x, x1, n1)  # x is a placeholder and does not affect the result

        # A'(t) = 2t - 2
        A_dot = 2.0 * t_b - 2.0

        # B(t) = α * x1 + β * v1
        B = alpha * x1 + beta * v1

        # α'(t) = 2 - 2t,  β'(t) = 2t - 1
        alpha_dot = 2.0 - 2.0 * t_b
        beta_dot  = 2.0 * t_b - 1.0

        # B'(t) = α'(t) * x1 + β'(t) * v1
        B_dot = alpha_dot * x1 + beta_dot * v1

        # u(x,z,t) = (A'/A) * (x - B) + B'
        u = (A_dot / A_safe) * (x - B) + B_dot
        return u

    def conditional_vector_field_inputx0(
        self,
        x0: torch.Tensor,   # (..., dim) start point x0
        z: torch.Tensor,    # (..., 2*dim) = [x1(dim), n1(dim)]
        t: torch.Tensor,    # (...,) or (B,1,1,1)
    ) -> torch.Tensor:
        """
        Use the same quadratic Hermite curve as sample_conditional_path_inputx0:
            x(t) = x0 + α(t)*(x1 - x0) + β(t)*v1
        where v1 is computed by _compute_v1, with the scaling determined by tangent_scale_mode.

        Returns the analytic velocity on that curve:
            u_t = d/dt x(t)
                = α'(t)*(x1 - x0) + β'(t)*v1
        Note: this u_t is the "true" velocity only when x = x(t).
        """
        dim = self.in_out_dim
        x1 = z[..., :dim]
        n1 = z[..., dim:]

        # Compute v1 (using the unified method)
        v1 = self._compute_v1(x0, x1, n1)

        # Broadcast t
        t_b = self._broadcast_t(t, x0)   # (..., 1)

        # α'(t) = 2 - 2t, β'(t) = 2t - 1
        alpha_dot = 2.0 - 2.0 * t_b
        beta_dot  = 2.0 * t_b - 1.0

        d = x1 - x0

        # u_t = α'(t) * d + β'(t) * v1
        u = alpha_dot * d + beta_dot * v1
        return u

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("No closed-form conditional score for this path.")


class CubicHermiteNormalConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__(p_simple, p_data)
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return RuntimeError("Not implemented.")
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return RuntimeError("Not implemented.")
    def conditional_vector_field(self,x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return RuntimeError("Not implemented.")
    @staticmethod
    def _broadcast_t(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        t: any of (B,), (B,1), (B,1,1), (B,1,1,1)
        ref: e.g. (B, N, 3)
        Returns: t_b, which can be broadcast against ref
        """
        t_b = t
        while t_b.dim() < ref.dim():
            t_b = t_b.unsqueeze(-1)
        return t_b
    def sample_conditional_path_inputx0_zeron0(
        self,
        x0: torch.Tensor,   # (B, 3)
        z: torch.Tensor,    # (B, 6) = [x1(3), n1(3)]
        t: torch.Tensor,    # (B,) or (B,1,1,1), both acceptable
    ) -> torch.Tensor:
        """
        Use a cubic Hermite curve:
          x(t) = H0(t) * x0 + H1(t) * x1 + H3(t) * v1
        where:
          - H0, H1, H2, H3 are the standard Hermite basis functions
          - here the start tangent is set to 0 (smooth start), and the end tangent v1 follows the normal
        Only the 3D position (B,3) is returned.
        """
        # Split z
        x1 = z[..., :3]      # end position
        n1 = z[..., 3:]      # endpoint normal direction

        # Normalize the normal to guard against zero vectors
        eps = 1e-8
        n_norm = torch.linalg.norm(n1, dim=-1, keepdim=True).clamp_min(eps)
        n_hat = n1 / n_norm

        # End tangent v1: direction = normal, length is tunable (here we use the constant 1.0; adjust as you like)
        tangent_scale = 1.0
        v1 = tangent_scale * n_hat    # (B,3)

        # Broadcast t to match x0's dimensions
        t_b = self._broadcast_t(t, x0)       # (B, 1, 1)
        t2 = t_b * t_b
        t3 = t2 * t_b

        # Hermite basis functions (scalar, one set per sample)
        H1 = -2.0 * t3 + 3.0 * t2          # H1(t)
        H3 = t3 - t2                       # H3(t)

        # Use the equivalent form: x(t) = x0 + H1(t) * (x1 - x0) + H3(t) * v1
        d = x1 - x0                        # (B,3)
        x_t = x0 + H1 * d + H3 * v1        # (B,3)

        return x_t
    def conditional_vector_field_zeron0(
        self,
        x: torch.Tensor,    # (B, 3) current position x_t
        z: torch.Tensor,    # (B, 6) = [x1(3), n1(3)]
        t: torch.Tensor,    # (B,) or (B,1,1,1)
    ) -> torch.Tensor:
        """
        Use the same Hermite curve as sample_conditional_path_inputx0, and give
        the analytic vector field u(x,z,t) on that curve, satisfying:
          if x = φ_t(x0, z), then u(x,z,t) = d/dt φ_t(x0,z)
        """
        # Split z
        x1 = z[..., :3]      # end position
        n1 = z[..., 3:]      # endpoint normal

        eps = 1e-8
        n_norm = torch.linalg.norm(n1, dim=-1, keepdim=True).clamp_min(eps)
        n_hat = n1 / n_norm

        tangent_scale = 1.0
        v1 = tangent_scale * n_hat    # (B,3)

        # broadcast t
        t_b = self._broadcast_t(t, x)       # (B,1,1)
        t2 = t_b * t_b
        t3 = t2 * t_b

        # Hermite basis functions
        H0 = 2.0 * t3 - 3.0 * t2 + 1.0   # A(t)
        H1 = -2.0 * t3 + 3.0 * t2
        H3 = t3 - t2

        # Derivatives
        H0_dot = 6.0 * t2 - 6.0 * t_b    # A'(t)
        H1_dot = -6.0 * t2 + 6.0 * t_b
        H3_dot = 3.0 * t2 - 2.0 * t_b

        A = H0                            # (B,1,...) broadcast
        A_dot = H0_dot

        # B(t) = H1 * x1 + H3 * v1
        B = H1 * x1 + H3 * v1             # (B,3)

        # B'(t) = H1' * x1 + H3' * v1
        B_dot = H1_dot * x1 + H3_dot * v1 # (B,3)

        # Avoid precision blow-up of A(t) as t → 1 (same as 1 - t in the linear path)
        A_safe = A.clamp_min(1e-4)

        # u(x,z,t) = (A'/A) * (x - B) + B'
        u = (A_dot / A_safe) * (x - B) + B_dot
        return u
    def conditional_vector_field_inputx0_zeron0(
        self,
        x0: torch.Tensor,   # (B, 3) start point x0
        z: torch.Tensor,    # (B, 6) = [x1(3), n1(3)]
        t: torch.Tensor,    # (B,) or (B,1,1,1)
    ) -> torch.Tensor:
        """
        Use the same Hermite curve as sample_conditional_path_inputx0:
            x_t = H0(t) * x0 + H1(t) * x1 + H3(t) * v1
        where m0 = 0, m1 = v1 ~ n1

        Returns the analytic velocity on that curve:
            u_t = d/dt x_t
                = H0'(t) * x0 + H1'(t) * x1 + H3'(t) * v1
        Note: this u_t is the "true" velocity only when x = x_t.
        """

        # Split z
        x1 = z[..., :3]      # end position
        n1 = z[..., 3:]      # endpoint normal

        # Normalize the normal and use it as the end tangent v1
        eps = 1e-8
        n_norm = torch.linalg.norm(n1, dim=-1, keepdim=True).clamp_min(eps)
        n_hat = n1 / n_norm

        tangent_scale = 1.0          # adjust this coefficient to change the curvature
        v1 = tangent_scale * n_hat    # (B,3)

        # Flatten t to (B,1) for easier broadcasting
        t_b = self._broadcast_t(t, x0)      # (B,1,1)
        t2 = t_b * t_b
        t3 = t2 * t_b


        # Hermite basis functions (the m0=0 set)
        # H0(t) =  2t^3 - 3t^2 + 1     (weight of x0)
        # H1(t) = -2t^3 + 3t^2         (weight of x1)
        # H3(t) =  t^3 - t^2           (weight of v1, from h11)
        H0_dot = 6.0 * t2 - 6.0 * t_b        # d/dt H0 = 6t^2 - 6t
        H1_dot = -6.0 * t2 + 6.0 * t_b       # d/dt H1 = -6t^2 + 6t
        H3_dot = 3.0 * t2 - 2.0 * t_b        # d/dt H3 = 3t^2 - 2t

        # u_t = H0'(t) * x0 + H1'(t) * x1 + H3'(t) * v1
        u = H0_dot * x0 + H1_dot * x1 + H3_dot * v1   # (B,3)

        return u

    def sample_conditional_path_inputx0_withn0(
        self,
        x0: torch.Tensor,   # (..., 6) = [x0(3), n0(3)]
        z: torch.Tensor,    # (..., 6) = [x1(3), n1(3)]
        t: torch.Tensor,    # (...,) or broadcastable
    ) -> torch.Tensor:
        """
        Use the full cubic Hermite curve:
          x(t) = h00(t)*x0 + h01(t)*x1 + h10(t)*m0 + h11(t)*m1
        where:
          - x0, x1 are the start and end positions
          - m0 ~ n0, m1 ~ n1 are the start / end tangents respectively
        Returns: 3D position (...,3)
        """
        # Split position and normal
        x0_pos = x0[..., :3]   # (...,3)
        n0     = x0[..., 3:]   # (...,3)

        x1 = z[..., :3]        # (...,3)
        n1 = z[..., 3:]        # (...,3)

        eps = 1e-8
        n0_norm = torch.linalg.norm(n0, dim=-1, keepdim=True).clamp_min(eps)
        n1_norm = torch.linalg.norm(n1, dim=-1, keepdim=True).clamp_min(eps)
        n0_hat = n0 / n0_norm
        n1_hat = n1 / n1_norm

        tangent_scale = 1.0
        m0 = tangent_scale * n0_hat   # (...,3)
        m1 = tangent_scale * n1_hat   # (...,3)

        # Broadcast t to match x0_pos's dimensions (only the last dim matters)
        t_b = self._broadcast_t(t, x0_pos)   # shape like (...,1)
        t2 = t_b * t_b
        t3 = t2 * t_b

        # Standard Hermite basis functions
        h00 =  2.0 * t3 - 3.0 * t2 + 1.0    # (...,1)
        h01 = -2.0 * t3 + 3.0 * t2          # (...,1)
        h10 =        t3 - 2.0 * t2 + t_b    # (...,1)
        h11 =        t3 -        t2         # (...,1)

        # x(t) = h00*x0 + h01*x1 + h10*m0 + h11*m1
        x_t = h00 * x0_pos + h01 * x1 + h10 * m0 + h11 * m1   # (...,3)
        return x_t

    def conditional_vector_field_inputx0_withn0(
        self,
        x0: torch.Tensor,   # (..., 6) = [x0(3), n0(3)]
        z: torch.Tensor,    # (..., 6) = [x1(3), n1(3)]
        t: torch.Tensor,    # (...,)
    ) -> torch.Tensor:
        """
        Use the same Hermite curve as sample_conditional_path_inputx0_withn0,
        and return the analytic velocity u(t) = dx/dt:
          u(t) = h00'(t)*x0 + h01'(t)*x1 + h10'(t)*m0 + h11'(t)*m1
        """
        x0_pos = x0[..., :3]
        n0     = x0[..., 3:]

        x1 = z[..., :3]
        n1 = z[..., 3:]

        eps = 1e-8
        n0_norm = torch.linalg.norm(n0, dim=-1, keepdim=True).clamp_min(eps)
        n1_norm = torch.linalg.norm(n1, dim=-1, keepdim=True).clamp_min(eps)
        n0_hat = n0 / n0_norm
        n1_hat = n1 / n1_norm

        tangent_scale = 1.0
        m0 = tangent_scale * n0_hat   # (...,3)
        m1 = tangent_scale * n1_hat   # (...,3)

        t_b = self._broadcast_t(t, x0_pos)   # (...,1)
        t2 = t_b * t_b

        h00_dot = 6.0 * t2 - 6.0 * t_b          # d/dt h00
        h01_dot = -6.0 * t2 + 6.0 * t_b         # d/dt h01
        h10_dot = 3.0 * t2 - 4.0 * t_b + 1.0    # d/dt h10
        h11_dot = 3.0 * t2 - 2.0 * t_b          # d/dt h11

        # u(t) = h00'*x0 + h01'*x1 + h10'*m0 + h11'*m1
        u = h00_dot * x0_pos + h01_dot * x1 + h10_dot * m0 + h11_dot * m1   # (...,3)
        return u

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("No closed-form conditional score for this path.")
    
class LinearConditionalProbabilityTorusPath(ConditionalProbabilityPath):
    """
    Linear path on a torus:
      x_t = wrap( x0 + t * mi_L(z - x0) )
    and conditional vector field:
      v(x,z,t) = mi_L(z - x) / (1 - t)
    """
    def __init__(self, p_simple, p_data, periodic_L=2.0, eps=1e-6,
                 use_ot=False, ot_eps=0.05, ot_iters=100, ot_microbatch=128,
                 # New optional arguments (all have defaults; backward compatible)
                 ot_topk: Optional[int] = None,     # row top-k sparsification (None/int)
                 ot_harden: str = "hungarian",      # 'hungarian' | 'greedy'
                 ot_method: str = "sinkhorn_log",    # POT method: 'sinkhorn' | 'sinkhorn_log' | 'sinkhorn_stabilized'
                 ot_backend: str = "lap",

                 use_best_hilbert_shuffle: bool = False,
                 shift_chunk: int = 128,

                 use_random_hilbert_shuffle: bool = False,
                 random_shuffle_mode: str = "roll",  # 'roll' or 'perm'
                 random_seed: Optional[int] = None,

                 use_improve_two_opt: bool = False,
                 two_opt_k: int = 24,
                 two_opt_passes: int = 2,
                 two_opt_first_improvement: bool = True):
        super().__init__(p_simple, p_data)
        self.L = float(periodic_L)
        self.eps = eps
        self.use_ot = use_ot
        self.ot_eps = ot_eps
        self.ot_iters = ot_iters
        self.ot_microbatch = ot_microbatch
        self.ot_topk = ot_topk
        self.ot_harden = ot_harden
        self.ot_method = ot_method
        self.ot_backend   = ot_backend
        self.use_best_hilbert_shuffle = use_best_hilbert_shuffle
        self.shift_chunk = int(max(1, shift_chunk))
        # print("OT Backend: ", self.ot_backend)
        self.cpu_threads  = getattr(self, "cpu_threads", os.cpu_count() or 4)
        # print("OT thread: ", self.cpu_threads)
        self.ot_microbatch = min(self.ot_microbatch, self.cpu_threads)
        # print("ot_microbatch: ", self.ot_microbatch)
        print("self L:", self.L)

        self.use_random_hilbert_shuffle = use_random_hilbert_shuffle
        assert random_shuffle_mode in ("roll", "perm")
        self.random_shuffle_mode = random_shuffle_mode
        # Reproducible experiments (torch generator)
        self._rng = torch.Generator()
        if random_seed is not None:
            self._rng.manual_seed(int(random_seed))
        self.use_improve_two_opt = bool(use_improve_two_opt)
        self.two_opt_k = int(max(1, two_opt_k))
        self.two_opt_passes = int(max(1, two_opt_passes))
        self.two_opt_first_improvement = bool(two_opt_first_improvement)

    @torch.no_grad()
    def _improve_two_opt(self, z_sorted: torch.Tensor) -> torch.Tensor:
        """
        Input z_sorted: (B,N,2), already in Hilbert order.
        Run 2-opt on each batch in a thread pool (wrap-around allowed),
        and roll "the point at original index 0" back to the front of the sequence as a fixed start point.
        """
        import numpy as _np
        device = z_sorted.device
        dtype  = z_sorted.dtype
        B, N, _ = z_sorted.shape
        if N <= 3 or B == 0:
            return z_sorted.clone()

        z_np = z_sorted.detach().cpu().numpy()  # (B,N,2)
        L = float(self.L)
        u_np = ((z_np + 0.5 * L) / L) % 1.0     # normalize to [0,1)

        out_np = _np.empty_like(z_np)

        def _solve_one(b_idx: int):
            xy = u_np[b_idx]                              # (N,2)
            D2 = pairwise_torus_dist2_matrix(xy)          # (N,N)
            k  = min(self.two_opt_k, max(1, N-1))
            cand = knn_candidates(D2, k=k)                # (N,k)
            init_order = _np.arange(N, dtype=_np.int64)   # keep the initial Hilbert order

            order, _info = two_opt_fast(
                xy, init_order, D2, cand,
                passes=self.two_opt_passes,
                first_improvement=self.two_opt_first_improvement,
                forbid_cross=False,   # allow wrap-around
                cross=None
            )
            # Fixed start point: roll the position of "original index 0" to the front
            pos0 = int(_np.where(order == 0)[0][0])
            order = _np.roll(order, -pos0)

            return z_np[b_idx, order]

        # Thread-pool parallelism (number of threads does not exceed the batch size)
        max_workers = min(int(self.cpu_threads), int(B))
        if max_workers <= 1:
            for b in range(B):
                out_np[b] = _solve_one(b)
        else:
            print("MAX Workers:", max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_solve_one, range(B)))
            for b, arr in enumerate(results):
                out_np[b] = arr

        return torch.from_numpy(out_np).to(device=device, dtype=dtype)

    def _compute_perm_cpu(self, x0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x0, z: (B, N, 2) on CPU
        perm: (B, N) long
        """
        #  (B,N,N) on CPU
        C = pairwise_sqdist_torus_batched(x0, z, L=self.L).to(torch.float64).cpu()
        perm = _solve_batch_LAP(C, threads=self.cpu_threads)  # (B,N)
        return perm

    @torch.no_grad()
    def _random_hilbert_shuffle(self, z_sorted: torch.Tensor) -> torch.Tensor:
        """
        Apply a random shuffle to the already Hilbert-sorted z_sorted (B,N,2):
        - mode='roll': randomly choose M∈[0,N-1] and cyclically shift, preserving order (recommended)
        - mode='perm': random full permutation (does not preserve local order)
        Sample with a CPU generator, then move indices to the same device as z_sorted when needed.
        """
        B, N, _ = z_sorted.shape
        device = z_sorted.device

        if self.random_shuffle_mode == "roll":
            # Use self._rng on CPU to generate B random shifts, then convert to Python ints
            Ms_cpu = torch.randint(low=0, high=N, size=(B,), generator=self._rng, device="cpu")
            z_out = torch.empty_like(z_sorted)
            for b, M in enumerate(Ms_cpu.tolist()):        # Python int, no device involved
                z_out[b] = torch.roll(z_sorted[b], shifts=int(M), dims=0)
            return z_out

        else:  # 'perm'
            z_out = torch.empty_like(z_sorted)
            for b in range(B):
                # Generate a random permutation on CPU, then move to the target device for indexing
                perm = torch.randperm(N, generator=self._rng, device="cpu").to(device)
                z_out[b] = z_sorted[b, perm]
            return z_out


    @torch.no_grad()
    def _best_shift_argmin(self, x0: torch.Tensor, z_sorted: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given x0 (B,N,2) and the already Hilbert-sorted z_sorted (B,N,2),
        over cyclic shifts M∈{0..N-1}, find the M* that minimizes sum_i || mi_L(z_{i+M}-x0_i) ||^2 (per batch).
        Returns:
        M_star: (B,) long
        z_tgt : (B,N,2)  where each batch has been rolled by its own M_star
        Complexity: O(B * N * N), but chunked by shift_chunk; fast at N=1024 and can run on GPU.
        """
        device = x0.device
        B, N, _ = x0.shape
        best_cost = torch.full((B,), float("inf"), device=device, dtype=x0.dtype)
        best_M    = torch.zeros((B,), dtype=torch.long, device=device)

        # Scan shifts in chunks to avoid constructing (N,B,N,2) all at once
        for m0 in range(0, N, self.shift_chunk):
            m1 = min(N, m0 + self.shift_chunk)
            S  = m1 - m0
            # Compute cost for each shift (the small loop is within the chunk, avoiding a large N-iteration loop)
            for off in range(S):
                M = m0 + off
                zM = torch.roll(z_sorted, shifts=int(M), dims=1)  # (B,N,2)
                delta = zM - x0
                # minimal image on torus
                delta = (delta + 0.5 * self.L) % self.L - 0.5 * self.L
                cost  = (delta * delta).sum(dim=(1, 2))  # (B,)

                # Update the best M for each batch
                improved = cost < best_cost
                if torch.any(improved):
                    best_cost = torch.where(improved, cost, best_cost)
                    best_M    = torch.where(improved, torch.tensor(M, device=device).long(), best_M)

        # Roll z_sorted by each batch's best_M to assemble z_tgt
        z_tgt = torch.empty_like(z_sorted)
        for b in range(B):
            z_tgt[b] = torch.roll(z_sorted[b], shifts=int(best_M[b].item()), dims=0)

        return best_M, z_tgt


    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        z, _ = self.p_data.sample(num_samples)
        return z


    @torch.no_grad()
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, N, _ = z.shape
        x0, _ = self.p_simple.sample(B)  # [B,N,2] (ensure on the same device)

        if self.use_improve_two_opt:
            z_tgt = self._improve_two_opt(z)
        elif not self.use_ot and not self.use_best_hilbert_shuffle and not self.use_random_hilbert_shuffle:
            z_tgt = z
        elif self.use_best_hilbert_shuffle:
            _, z_tgt = self._best_shift_argmin(x0, z)
        elif self.use_random_hilbert_shuffle:
            z_tgt = self._random_hilbert_shuffle(z)
        else:
            z_tgt = torch.empty_like(z)
            mb = max(1, int(self.ot_microbatch))

            for s in range(0, B, mb):
                e = min(B, s + mb)
                x0_mb = x0[s:e]   # [mb,N,2]
                z_mb  = z[s:e]    # [mb,N,2]

                t0 = time.time()
                if self.ot_backend == "cpu_emd":
                    x0_cpu = x0_mb.detach().cpu()
                    z_cpu  = z_mb.detach().cpu()
                    C_cpu  = pairwise_sqdist_torus_batched(x0_cpu, z_cpu, L=self.L)  # (mb,N,N), torch on CPU
                    m_cur, N, _ = C_cpu.shape
                    a = np.full(N, 1.0 / N, dtype=np.float64)  # uniform marginal
                    b = a
                    def solve_one(i):
                        Ci = C_cpu[i].numpy().astype(np.float64, copy=False)
                        Ti = ot.emd(a, b, Ci)  # (N,N), float64
                        return Ti  # numpy
                    with cf.ThreadPoolExecutor(max_workers=self.cpu_threads) as ex:
                        T_list = list(ex.map(solve_one, range(C_cpu.shape[0])))
                    T_mb = torch.from_numpy(np.stack(T_list, axis=0)).to(z_mb.device, dtype=z_mb.dtype)  # (mb,N,N)
                    perm_mb = harden_transport_scipy_mt(T_mb, topk=8, threads=self.cpu_threads if self.ot_backend == "cpu_emd" else None)
                elif self.ot_backend == "gpu_sinkhorn":
                    C_mb = pairwise_sqdist_torus_batched(x0_mb, z_mb, L=self.L)  # (mb,N,N)
                    T_mb = batched_sinkhorn_uniform(
                        C_mb, reg=self.ot_eps, iters=self.ot_iters, solver=self.ot_method
                    )
                    perm_mb = harden_transport_scipy_mt(T_mb, topk=8, threads=self.cpu_threads if self.ot_backend == "cpu_emd" else None)
                
                elif self.ot_backend == "lap":
                    C_mb = pairwise_sqdist_torus_batched(x0_mb, z_mb, L=self.L)  # (mb,N,N)
                    perm_mb = assign_from_cost_scipy_mt(C_mb, k_smallest=8, threads=self.cpu_threads)

                t1 = time.time()
                # ok_mask, info = validate_batched_perm(perm_mb, raise_on_error=False)
                # if not ok_mask.all():
                t2 = time.time()
                # print(f"[OT backend={self.ot_backend}] solve={t1-t0:.3f}s, harden={t2-t1:.3f}s")
                z_tgt[s:e] = gather_by_perm(z_mb, perm_mb)   # [mb,N,2]
                print("ot time cpu:", t1 - t0)
                # print("harden_transport time:", t2 - t1)

        delta = minimal_image(z_tgt - x0, self.L)    # [B,N,2]
        x_t = wrap_coords(x0 + t * delta, self.L)
        self._cached_z_tgt = z_tgt
        return x_t
    
    def conditional_vector_field(self, x, z, t):
        z_tgt = getattr(self, "_cached_z_tgt", z)
        num = minimal_image(z_tgt - x, self.L)
        denom = torch.clamp(1.0 - t, min=self.eps)
        return num / denom

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("No closed-form conditional score for linear torus path.")


class OTPairDataset(IterableDataset):

    def __init__(
        self,
        path,                          # LinearConditionalProbabilityTorusPath
        batch_size: int,
        torus_L: float = 2.0,
        cpu_threads: int = 32,
        rotate4: bool = False
    ):
        super().__init__()
        self.path = path
        self.batch_size = int(batch_size)
        self.L = float(torus_L)
        self.cpu_threads = int(cpu_threads)
        self.rotate4 = rotate4

    @torch.no_grad()
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
            # Single worker: use the current initial seed
            base_seed = torch.initial_seed()
        else:
            worker_id, num_workers = info.id, info.num_workers
            # PyTorch convention: worker_seed = base_seed + worker_id
            base_seed = info.seed - worker_id

        g = torch.Generator()
        g.manual_seed(base_seed)

        N_total = len(self.path.p_data)
        perm = torch.randperm(N_total, generator=g)

        local_perm = perm[worker_id::num_workers]

        B = self.batch_size
        for start in range(0, local_perm.numel(), B):
            idx = local_perm[start:start+B]
            bsz = idx.numel()
            if bsz == 0:
                break

            z, _ = self.path.p_data.get_batch(idx, rotate4=self.rotate4)   # (B,N,2) on CPU / or p_data.device
            z = z.contiguous().cpu()
            x0, _ = self.path.p_simple.sample(bsz)
            x0 = x0.contiguous().cpu()

            z_tgt = torch.empty_like(z)
            mb = min(self.batch_size, self.cpu_threads)

            for s in range(0, self.batch_size, mb):
                e = min(self.batch_size, s + mb)
                x0_mb = x0[s:e]   # [mb,N,2]
                z_mb  = z[s:e]    # [mb,N,2]

                C_mb = pairwise_sqdist_torus_batched(x0_mb, z_mb, L=self.L)  # (mb,N,N)
                perm_mb = assign_from_cost_scipy_mt(C_mb, k_smallest=1024, threads=mb)

                z_tgt[s:e] = gather_by_perm(z_mb, perm_mb)   # [mb,N,2]

            yield {"x0": x0, "z_tgt_perm": z_tgt}

class LinearConditionalProbabilityTorusPath_EqM(ConditionalProbabilityPath):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable, 
                 gamma: Gamma, periodic_L: float = 2.0, eps: float = 1e-6):
        super().__init__(p_simple, p_data)
        self.L = float(periodic_L)
        self.gamma = gamma
        self.eps = eps
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        z, _ = self.p_data.sample(num_samples)
        return z
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        x0, _ = self.p_simple.sample(B)

        delta = minimal_image(z - x0, self.L)      # (B, N, 2) in [-L/2, L/2)
        x_t = x0 + t * delta
        x_t = wrap_coords(x_t, self.L)
        return x_t

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        v(x,z,t) = mi_L(z - x) / (1 - t)
        Safe near t→1 with small epsilon.
        """

        num = minimal_image(x - z, self.L)
        c_gamma = self.gamma(t)
        return num * c_gamma

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("No closed-form conditional score for linear torus path.")



def dct_ortho_matrix(n: int, device=None, dtype=None) -> torch.Tensor:
    """
    Orthonormal DCT-II matrix of size [n, n].
    D[k,n] = sqrt(2/N) * cos(pi*(n+0.5)*k/N), k=0..N-1; row 0 scaled by 1/sqrt(2).
    """
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # [n,1]
    n_idx = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)  # [1,n]
    mat = torch.cos(math.pi * (n_idx + 0.5) * k / n)                # [n,n]
    mat *= math.sqrt(2.0 / n)
    mat[0, :] *= 1.0 / math.sqrt(2.0)
    return mat  # [n,n]

def kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Kronecker product (for small sizes)"""
    a1, a2 = A.shape
    b1, b2 = B.shape
    return (A.unsqueeze(-1).unsqueeze(-3) * B.unsqueeze(0).unsqueeze(2)).reshape(a1*b1, a2*b2)



class FixedEigenSigmaRoot(ABC):
    def __init__(self, U: torch.Tensor):
        self.U = U

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        ...

    def sigma_dt(self, t: torch.Tensor) -> torch.Tensor:
        h = 1e-3
        return (self.sigma(torch.clamp(t + h, 0.0, 1.0)) -
                self.sigma(torch.clamp(t - h, 0.0, 1.0))) / (2*h)

    def root(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.size(-1) == 1: t = t.view(-1)
        N = t.shape[0]; U = self.U
        sig = self.sigma(t)                    # [N, D]
        D = sig.shape[1]
        return U @ torch.diag_embed(sig) @ U.T  # broadcast: [N,D,D]

    def cov(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.size(-1) == 1: t = t.view(-1)
        sig = self.sigma(t)                    # [N, D]
        return self.U @ torch.diag_embed(sig**2) @ self.U.T


class LP2HP_DCT_Root(FixedEigenSigmaRoot):
    def __init__(self,
                 shape,
                 device=None, dtype=None,
                 amp_power: float = 1.0,
                 gamma_eta: float = 1.3,
                 tau_lp: float = 0.25,
                 tau_hp: float = 0.25,
                 eps0: float = 0.05,
                 eps1: float = 0.01,
                 floor_power: float = 2.0):

        self.device = device
        self.dtype = dtype
        self.eps0 = eps0
        self.eps1 = eps1
        self.floor_power = floor_power

        self.amp_power = amp_power
        self.gamma_eta = gamma_eta
        self.tau_lp = tau_lp
        self.tau_hp = tau_hp

        if isinstance(shape, int):
            L = shape
            U = dct_ortho_matrix(L, device=device, dtype=dtype)          # [L,L]
            k = torch.arange(L, device=device, dtype=dtype)
            r = (k / max(L-1, 1)).view(1, L)                             # [1,D]
        else:
            H, W = shape
            Uh = dct_ortho_matrix(H, device=device, dtype=dtype)         # [H,H]
            Uw = dct_ortho_matrix(W, device=device, dtype=dtype)         # [W,W]
            U = kron(Uw, Uh)
            #: r = sqrt((kh/(H-1))^2 + (kw/(W-1))^2) / sqrt(2) ∈ [0,1]
            kh = torch.arange(H, device=device, dtype=dtype) / max(H-1, 1)
            kw = torch.arange(W, device=device, dtype=dtype) / max(W-1, 1)
            Kh, Kw = torch.meshgrid(kh, kw, indexing='ij')
            r2d = torch.sqrt(Kh**2 + Kw**2) / math.sqrt(2.0)            # [H,W]
            r = r2d.reshape(1, H*W)                                      # [1,D]

        self.r = r                  # [1, D]
        super().__init__(U=U)

    def _a(self, t):
        c = torch.cos(0.5 * math.pi * t)
        return torch.clamp(c, min=0.0) ** self.amp_power
    def _a_dt(self, t):
        if self.amp_power == 0.0: return torch.zeros_like(t)
        c = torch.cos(0.5 * math.pi * t); s = torch.sin(0.5 * math.pi * t)
        c = torch.clamp(c, min=0.0)
        return - self.amp_power * (math.pi/2.0) * s * (c ** max(self.amp_power - 1.0, 0.0))

    # def _gamma(self, t):  return t ** self.gamma_eta
    # def _gamma_dt(self, t): 
    #     if self.gamma_eta == 0.0: return torch.zeros_like(t)
    #     return self.gamma_eta * (t ** (self.gamma_eta - 1.0))
    def _gamma(self, t):
        t0, k = 0.1, 0.1
        return torch.sigmoid(k*(t - t0))
    def _gamma_dt(self, t):
        g = self._gamma(t); return g*(1-g)*0.1


    def _floor(self, t):      # [N,1]
        return (self.eps1 + (self.eps0 - self.eps1) * (1.0 - t) ** self.floor_power).unsqueeze(1)
    def _floor_dt(self, t):   # [N,1]
        return (-(self.eps0 - self.eps1) * self.floor_power * (1.0 - t) ** (self.floor_power - 1.0)).unsqueeze(1)

    def _gamma(self, t):  # γ(t) = t^{eta}
        return t ** self.gamma_eta
    def _gamma_dt(self, t):
        if self.gamma_eta == 0.0:
            return torch.zeros_like(t)
        return self.gamma_eta * (t ** (self.gamma_eta - 1.0))

    def _LP(self):
        r = self.r
        return torch.exp(- (r / self.tau_lp)**2)  # [1,D]
    def _HP(self):
        r = self.r
        return torch.exp(- ((1.0 - r) / self.tau_hp)**2)  # [1,D]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim()==2 and t.size(-1)==1: t=t.view(-1)
        t = t.clamp(0.0, 1.0)
        a  = self._a(t).unsqueeze(1)
        g  = self._gamma(t).unsqueeze(1)
        LP = self._LP().to(t.device, t.dtype); HP = self._HP().to(t.device, t.dtype)
        blend = (1.0 - g) * LP + g * HP       # [N,D]
        return self._floor(t) + a * blend

    def sigma_dt(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim()==2 and t.size(-1)==1: t=t.view(-1)
        t = t.clamp(0.0, 1.0)
        a    = self._a(t).unsqueeze(1)
        a_dt = self._a_dt(t).unsqueeze(1)
        g    = self._gamma(t).unsqueeze(1)
        g_dt = self._gamma_dt(t).unsqueeze(1)
        LP = self._LP().to(t.device, t.dtype); HP = self._HP().to(t.device, t.dtype)
        blend = (1.0 - g) * LP + g * HP
        return self._floor_dt(t) + a_dt * blend + a * g_dt * (HP - LP)

class AnisotropicGaussianConditionalPathCommuting(ConditionalProbabilityPath):
    def __init__(self, p_data, alpha, root_schedule: FixedEigenSigmaRoot, jitter: float = 1e-6):
        U = root_schedule.U
        if not isinstance(U, torch.Tensor):
            raise ValueError("root_schedule.U must be a torch.Tensor")

        D = U.shape[0]                      # use U's dimension as the reference
        device, dtype = U.device, U.dtype
        t0 = torch.zeros(1, device=device, dtype=dtype)

        # Σ0: prefer cov(t), otherwise use U diag(σ0^2) U^T
        sigma0 = root_schedule.sigma(t0).squeeze(0)        # [D]
        Sigma0 = U @ torch.diag(sigma0**2) @ U.T           # [D,D]

        Sigma0 = 0.5 * (Sigma0 + Sigma0.T)
        I = torch.eye(D, device=device, dtype=dtype)
        Sigma0 = Sigma0 + jitter * I

        mean0 = torch.zeros(D, device=device, dtype=dtype)
        p_simple = Gaussian(mean0, Sigma0)
        print(mean0, Sigma0)

        if hasattr(p_data, "dim") and p_data.dim != D:
            raise ValueError(f"Dimension mismatch: root dim={D}, p_data.dim={p_data.dim}.")

        super().__init__(p_simple, p_data)

        self.register_buffer("U", U.detach().clone())
        with torch.no_grad():
            self.register_buffer("sigma0", root_schedule.sigma(t0).squeeze(0))  # [D]

        self.U = U
        self.Sigma0 = Sigma0
        self.sigma0 = sigma0
        self.Sigma0_root = U @ torch.diag(sigma0) @ U.T
        self.dim = D
        self.alpha = alpha
        self.root_schedule = root_schedule
        self.jitter = jitter
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x = μ_t + S_t ε
        mu = (self.alpha(t) * z)                               # [N,D]
        S_t = self.root_schedule.root(t)                     # [N,D,D]
        eps = torch.randn_like(z)                              # [N,D]
        return mu + (S_t @ eps.unsqueeze(-1)).squeeze(-1)

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # u = (α̇ - α S) z + S x,  S = U diag(σ̇/σ) U^T
        N, D = x.shape
        alpha = self.alpha(t).view(N, 1)
        dalpha = self.alpha.dt(t).view(N, 1)
        sig = self.root_schedule.sigma(t)                     # [N,D]
        sig_dt = self.root_schedule.sigma_dt(t)               # [N,D]
        ratio = sig_dt / (sig + 1e-12)                        # [N,D]
        U = self.U
        # Sx:
        Sx = ((x @ U) * ratio) @ U.T
        Sz = ((z @ U) * ratio) @ U.T
        return dalpha * z - alpha * Sz + Sx

    def A(self, t: torch.Tensor) -> torch.Tensor:
        # A_t = U diag(σ_t / σ_0) U^T
        sig = self.root_schedule.sigma(t)       # [N,D]
        ratio = sig / (self.sigma0 + 1e-12)     # [N,D]
        return self.U @ torch.diag_embed(ratio) @ self.U.T

    def A_dotA_inv(self, t: torch.Tensor) -> torch.Tensor:
        # Ȧ_t A_t^{-1} = U diag(σ̇_t / σ_t) U^T
        sig = self.root_schedule.sigma(t)       # [N,D]
        sig_dt = self.root_schedule.sigma_dt(t) # [N,D]
        ratio = sig_dt / (sig + 1e-12)
        return self.U @ torch.diag_embed(ratio) @ self.U.T
