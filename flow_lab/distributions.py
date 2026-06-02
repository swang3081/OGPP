from abc import ABC, abstractmethod
import numpy as np
import torch, torch.distributions as D
from typing import Optional, List, Type, Tuple, Dict
import torch.nn as nn
import math
from typing import Optional
from .sort_numba import hilbert_sort_xy_fast

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """ 
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
            - labels: shape (batch_size, label_dim)
        """
        pass

class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """
    def __init__(self, shape: List[int], std: float = 0.2):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.register_buffer("dummy", torch.zeros(1)) # Will automatically be moved when self.to(...) is called...
        
    def sample(self, num_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device), None


class Uniform(nn.Module, Sampleable):
    """
    Distribution which can be sampled from
    """ 
    def __init__(self, shape: List[int], a: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.register_buffer("dummy", torch.zeros(1))
        self.register_buffer("scale", torch.tensor(float(a)))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        uniform_sample_0_1 = torch.rand(num_samples, *self.shape, device=self.dummy.device)
        samples = (uniform_sample_0_1 * 2.0 - 1.0) * self.scale
        return samples, None

class UniformSort(nn.Module, Sampleable):
    """
    Distribution which can be sampled from
    """ 
    def __init__(self, shape: List[int]):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.register_buffer("dummy", torch.zeros(1))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        uniform_sample_0_1 = torch.rand(num_samples, *self.shape).to(self.dummy.device)
        return torch.from_numpy(hilbert_sort_xy_fast((uniform_sample_0_1 * 2 - 1).detach().cpu().numpy())).to(uniform_sample_0_1.device), None


class JitterHilbertGridSample(nn.Module, Sampleable):
    """
    Hilbert-Jittered Grid sampler on [0,1]^2.

    - grid_size: G (default 32) so N = G*G points
    - jitter: 'uniform' (in-cell uniform) or 'gaussian' (around cell center)
    - sigma: gaussian jitter std; if None, uses cell_size/sqrt(12) (matching uniform variance)
    - periodic: if True, wrap to [0,1) after jitter; else clamp to [0,1]
    - dtype: output dtype
    - seed: optional manual seed for reproducibility (torch.Generator)
    """
    def __init__(
        self,
        grid_size: int = 32,
        jitter: str = "uniform",
        sigma: Optional[float] = None,
        periodic: bool = False,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert grid_size > 0 and ((grid_size & (grid_size - 1)) == 0), \
            "grid_size must be a power of two (e.g., 32)."
        assert jitter in ("uniform", "gaussian", "none")
        self.G = grid_size
        self.N = grid_size * grid_size
        self.jitter = jitter
        self.periodic = periodic
        self.dtype = dtype

        cell = 1.0 / self.G
        # default sigma matches uniform variance in a cell (per-dim)
        self.sigma = sigma if sigma is not None else (cell / math.sqrt(12))

        self.base_seed = seed  # the original seed
        self._gens = {}        # device -> Generator cache ('cpu'/'cuda')

        # torch.Generator for reproducibility (optional)
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        # Precompute Hilbert-ordered cell centers on CPU, then register as buffer
        centers = self._make_hilbert_centers(self.G)  # (N,2) in [0,1]
        self.register_buffer("centers", centers.to(dtype), persistent=False)

        # Also keep cell size as buffer for device/broadcast convenience
        self.register_buffer("cell_size_tensor", torch.tensor(cell, dtype=dtype), persistent=False)

    # 2) Bind precisely to the device + cache
    def _get_gen(self, device: torch.device, per_call_seed: Optional[int] = None):
        key = str(device)  # e.g. 'cuda:0' / 'cpu'
        if per_call_seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(per_call_seed)
            return g
        g = self._gens.get(key)
        if g is None:
            g = torch.Generator(device=device)
            if self.base_seed is not None:
                g.manual_seed(self.base_seed)
            self._gens[key] = g
        return g

    # ---------- Public API ----------
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            samples: (num_samples, N, 2) in [0,1] (wrapped or clamped)
            labels : None
        """
        device = self.centers.device
        cell = self.cell_size_tensor.item()
        gen = self._get_gen(device=self.centers.device)  # or pass per_call_seed to fix this call's output

        if self.jitter == "uniform":
            # in-cell uniform jitter: center +/- 0.5*cell in each dim
            # shape (B,N,2)
            eps = (torch.rand((num_samples, self.N, 2), device=device, dtype=self.dtype, generator=gen) - 0.5) * (cell)
            out = self.centers.unsqueeze(0) + eps
        elif self.jitter == "gaussian":
            eps = torch.randn((num_samples, self.N, 2), device=device, dtype=self.dtype, generator=gen) * self.sigma
            out = self.centers.unsqueeze(0) + eps
        else:  # 'none'
            out = self.centers.unsqueeze(0).expand(num_samples, -1, -1).clone()

        if self.periodic:
            out = out - torch.floor(out)                 # wrap to [0,1)
        else:
            out = out.clamp_(0.0, 1.0)                   # hard clamp
        
        out = out * 2 - 1.0

        return out, None

    # ---------- Helpers ----------
    @staticmethod
    # 1) Fix the argument passing inside _hilbert_xy2d
    def _hilbert_xy2d(n: int, x: int, y: int) -> int:
        d = 0
        s = n // 2

        def rot(n_, x_, y_, rx_, ry_):
            if ry_ == 0:
                if rx_ == 1:
                    x_ = n_ - 1 - x_
                    y_ = n_ - 1 - y_
                x_, y_ = y_, x_
            return x_, y_

        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            x, y = rot(s, x, y, rx, ry)   # <<-- changed to s here
            s //= 2
        return d


    def _make_hilbert_centers(self, G: int) -> torch.Tensor:
        """
        Build (N,2) cell centers in [0,1] sorted by Hilbert order.
        This is computed once at init; jitter happens at sample-time.
        """
        # If you prefer to use the numba implementation, replace this entire function with:
        #   grid = np.stack(np.meshgrid(np.arange(G), np.arange(G), indexing="xy"), axis=-1).reshape(-1,2)
        #   xy = (grid + 0.5) / G
        #   xy_sorted = hilbert_sort_xy_fast(xy, p=int(math.log2(G)))  # the function you provided
        #   return torch.from_numpy(xy_sorted).to(torch.float32)

        # Pure Python version (fast enough at the N=G*G=1024 scale)
        coords = []
        for iy in range(G):
            for ix in range(G):
                d = self._hilbert_xy2d(G, ix, iy)
                coords.append((d, ix, iy))
        coords.sort(key=lambda t: t[0])  # sort by Hilbert index
        # cell center = (ix+0.5)/G, (iy+0.5)/G
        centers = torch.empty((self.N, 2), dtype=torch.float32)
        invG = 1.0 / G
        for k, (_, ix, iy) in enumerate(coords):
            centers[k, 0] = (ix + 0.5) * invG
            centers[k, 1] = (iy + 0.5) * invG
        return centers