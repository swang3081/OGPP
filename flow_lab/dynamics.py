from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import torch.nn as nn
def wrap_coords(x: torch.Tensor, L: float) -> torch.Tensor:
    """Wrap coordinates to the periodic box of length L (centered): [-L/2, L/2)."""
    return (x + 0.5 * L) % L - 0.5 * L

def minimal_image(dx: torch.Tensor, L: float) -> torch.Tensor:
    """Minimal-image displacement in [-L/2, L/2)."""
    return (dx + 0.5 * L) % L - 0.5 * L

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - diffusion_coefficient: shape (bs, c, h, w)
        """
        pass


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
            - dt: time, shape (bs, 1, 1, 1)
        Returns:
            - nxt: state at time t + dt (bs, c, h, w)
        """
        pass
    
    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, periodic: bool = False, periodic_L: float = 2.0, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - x_final: final state at time ts[-1], shape (bs, c, h, w)
        """
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, periodic, periodic_L, **kwargs)
        return x
    
    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, periodic: bool = False, periodic_L: float = 2.0, return_velocity: bool = False, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
            - return_velocity: if True, also return velocity at each timestep
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, h, w)
            - vs (optional): velocity at each timestep, shape (batch_size, nts, c, h, w)
        """
        xs = [x.clone()]
        vs = [] if return_velocity else None
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            if return_velocity:
                x, v = self.step(x, t, h, periodic, periodic_L, return_velocity=True, **kwargs)
                vs.append(v.clone())
            else:
                x = self.step(x, t, h, periodic, periodic_L, **kwargs)
            xs.append(x.clone())

        if return_velocity:
            # The final step requires computing the velocity at the final position
            t_final = ts[:, -1]
            v_final = self.ode.drift_coefficient(x, t_final, **kwargs)
            vs.append(v_final.clone())
            return torch.stack(xs, dim=1), torch.stack(vs, dim=1)
        return torch.stack(xs, dim=1)

class RK4Simulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, periodic: bool = False, periodic_L: float = 2.0, return_velocity: bool = False, **kwargs):
        f = self.ode.drift_coefficient

        k1 = f(xt, t, **kwargs)
        x2 = xt + 0.5 * h * k1
        if periodic:
            x2 = wrap_coords(x2, periodic_L)

        k2 = f(x2, t + 0.5 * h, **kwargs)
        x3 = xt + 0.5 * h * k2
        if periodic:
            x3 = wrap_coords(x3, periodic_L)

        k3 = f(x3, t + 0.5 * h, **kwargs)
        x4 = xt + h * k3
        if periodic:
            x4 = wrap_coords(x4, periodic_L)

        k4 = f(x4, t + h, **kwargs)
        x_final = xt + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if periodic:
            x_final = wrap_coords(x_final, periodic_L)

        if return_velocity:
            # Return k1 as the velocity at this time (i.e. the velocity at xt)
            return x_final, k1
        return x_final


class MidpointSimulator(Simulator):
    """
    Midpoint Method:
    1. k1 = f(t_n, y_n)
    2. y_{n+1/2} = y_n + (h/2) * k1   # half-step Euler prediction
    3. k2 = f(t_n + h/2, y_{n+1/2})   # slope at midpoint
    4. y_{n+1} = y_n + h * k2         # full step using midpoint slope
    """
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, periodic: bool = False, periodic_L: float = 2.0, return_velocity: bool = False, **kwargs):
        f = self.ode.drift_coefficient

        # k1 = f(t_n, y_n)
        k1 = f(xt, t, **kwargs)

        # y_{n+1/2} = y_n + (h/2) * k1
        x_mid = xt + 0.5 * h * k1
        if periodic:
            x_mid = wrap_coords(x_mid, periodic_L)

        # k2 = f(t_n + h/2, y_{n+1/2})
        k2 = f(x_mid, t + 0.5 * h, **kwargs)

        # y_{n+1} = y_n + h * k2
        x_final = xt + h * k2
        if periodic:
            x_final = wrap_coords(x_final, periodic_L)

        if return_velocity:
            # Return k1 as the velocity at this time (i.e. the velocity at xt)
            return x_final, k1
        return x_final


class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, periodic: bool = False, periodic_L: float = 2.0, return_velocity: bool = False, **kwargs):
        v = self.ode.drift_coefficient(xt, t, **kwargs)
        if not periodic:
            x_next = xt + v * h
        else:
            x_next = wrap_coords(xt + v * h, periodic_L)

        if return_velocity:
            return x_next, v
        return x_next

class EulerSimulator_VoroRGB(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, periodic: bool = False, periodic_L: float = 2.0, **kwargs):
        xt_1 = xt
        xt_1[..., 2:] += self.ode.drift_coefficient(xt,t, **kwargs) * h
        if periodic:
            xt_1[..., 2:] = wrap_coords(xt_1[..., 2:], periodic_L)
        return xt_1


class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, periodic: bool = False, periodic_L: float = 2.0, **kwargs):
        if not periodic:
            return xt + self.sde.drift_coefficient(xt,t, **kwargs) * h + self.sde.diffusion_coefficient(xt,t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt)
        else:
            return wrap_coords(xt + self.sde.drift_coefficient(xt,t, **kwargs) * h + self.sde.diffusion_coefficient(xt,t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt), periodic_L)


def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )



class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass

class CFGVectorFieldODE(ODE):
    def __init__(self, net: ConditionalVectorField, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        """
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.ones_like(y) * 10
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field



class UnconditionalVectorField(nn.Module, ABC):
    """
    Learned vector field u_t^theta(x) without labels.
    forward(x, t) -> (B, C, H, W)
    """
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (B, C, H, W)
        - t: (B, 1, 1, 1)
        Returns:
        - u_t^theta(x): (B, C, H, W)
        """
        pass
class VectorFieldODE(ODE):
    def __init__(self, net: UnconditionalVectorField, use_magnitude: bool = False):
        self.net = net
        self.use_magnitude = use_magnitude

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
        - x: (B, C, H, W)
        - t: (B, 1, 1, 1)
        - y: ignored
        """
        if not self.use_magnitude:
            return self.net(x, t)
        else:
            magnitude =  self.net(x, t)
            norm = torch.linalg.norm(x, dim=-1, keepdim=True)

            eps = 1e-8
            direction = -x / (norm + eps)  # no NaN when x=0, since 0 / eps = 0

            # magnitude * direction: broadcast to the shape of x
            drift = magnitude * direction
            return drift



class EqMGradientFieldWrapper:
    """
    Wrap an unconditional vector field f(x[, t]) to expose f(x).
    Many of your nets still accept t, so we feed a zero t by default.
    """
    def __init__(self, net: UnconditionalVectorField, t_value: float = 0.0):
        self.net = net
        self.t_value = t_value

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t = torch.zeros(B, 1, 1, device=x.device, dtype=x.dtype) + self.t_value
        return self.net(x, t)


class EqMNAGSampler:
    """
    Algorithm 2: NAG-based gradient descent with adaptive compute for EqM.
    - Supports per-sample early stop on ||grad||_2 <= g_thresh
    - Supports periodic wrap (torus) after each update
    - Uses minimal-image displacement for lookahead under periodic box

    x_{k+1} = x_k - eta * grad_k
    grad_{k+1} = f( x_{k+1} + mu * (x_{k+1} - x_k) )
    """
    def __init__(
        self,
        grad_field: EqMGradientFieldWrapper,
        eta: float = 0.05,
        mu: float = 0.9,
        g_thresh: float = 1e-3,
        max_steps: int = 200,
        periodic: bool = False,
        periodic_L: float = 2.0,
        use_minimal_image: bool = True,
        record_trajectory: bool = False,
    ):
        self.f = grad_field
        self.eta = float(eta)
        self.mu = float(mu)
        self.g_thresh = float(g_thresh)
        self.max_steps = int(max_steps)
        self.periodic = bool(periodic)
        self.periodic_L = float(periodic_L)
        self.use_minimal_image = bool(use_minimal_image)
        self.record_trajectory = bool(record_trajectory)

    @staticmethod
    def _per_sample_norm(g: torch.Tensor) -> torch.Tensor:
        # g: (B, ..., d) -> norms: (B,)
        B = g.shape[0]
        return g.reshape(B, -1).pow(2).sum(dim=1).sqrt()

    @torch.no_grad()
    def sample(self, st: torch.Tensor):
        """
        Args:
          st: initial state, shape (B, N, 2)
        Returns:
          x: final samples (B, N, 2)
          steps: number of steps actually taken (int)
          traj (optional): (B, T, N, 2) if record_trajectory=True
        """
        x = st.clone()
        x_last = st.clone()

        # initial grad at st
        grad = self.f(x)
        norms = self._per_sample_norm(grad)
        active = norms > self.g_thresh

        traj = [x.clone()] if self.record_trajectory else None

        steps = 0
        while active.any() and steps < self.max_steps:
            steps += 1
            # --- update only the active batch items ---
            if active.any():
                xa = x[active]
                ga = grad[active]
                xa = xa - self.eta * ga

                if self.periodic:
                    xa = wrap_coords(xa, self.periodic_L)

                x[active] = xa

            # record
            if self.record_trajectory:
                traj.append(x.clone())

            # compute lookahead point for next grad
            dx = x - x_last
            if self.periodic and self.use_minimal_image:
                dx = minimal_image(dx, self.periodic_L)

            lookahead = x + self.mu * dx
            if self.periodic:
                lookahead = wrap_coords(lookahead, self.periodic_L)

            x_last = x  # update history
            grad = self.f(lookahead)

            norms = self._per_sample_norm(grad)
            active = norms > self.g_thresh

        if self.record_trajectory:
            return x, steps, torch.stack(traj, dim=1)
        return x, steps