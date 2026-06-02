from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from .paths import GaussianConditionalProbabilityPath, OTPairDataset
from .dynamics import ConditionalVectorField, wrap_coords, minimal_image
from .utils import save_checkpoint, load_checkpoint
import os, json, time, tempfile, shutil, uuid
from datetime import timedelta
import numpy as np
import math
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
import contextlib
from torch import Tensor
from torch.utils.data import DataLoader
import torch.distributed as dist
import mlflow
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from pathlib import Path
import matplotlib
import math
from torch.optim.lr_scheduler import LambdaLR
import sys  # used if you want to redirect tqdm output to stdout
from .datasets import MeshPairAsyncLoader

class SimpleTorchWriter:
    """
    A minimal log writer:
    - add_text:    writes to <log_dir>/text/<tag>.txt
    - add_scalar:  writes to <log_dir>/metrics/<tag>/step_XXXXXXXX.json
    - add_scalars: writes to <log_dir>/metrics/_batch/step_XXXXXXXX.json
    - add_histogram: by default writes <log_dir>/hist/<tag>/step_XXXXXXXX.npz (counts/edges);
                     optionally, histogram_mode="png" plots to .png
    """
    def __init__(self, log_dir: str, histogram_mode: str = "json"):
        self.log_dir = Path(log_dir)
        self.histogram_mode = histogram_mode  # "json" | "png"
        self.log_dir.mkdir(exist_ok=True)
        # Pre-create first-level subdirectories so later mkdir calls don't need parents=True
        (self.log_dir / "text").mkdir(exist_ok=True)
        (self.log_dir / "metrics").mkdir(exist_ok=True)
        (self.log_dir / "hist").mkdir(exist_ok=True)

    # --- internal: atomic write (cloud-storage compatible version) ---
    def _atomic_write_bytes(self, out_path: Path, data: bytes):
        """
        Atomically write byte data to a file.
        Optimized for cloud storage such as Databricks Unity Catalog Volumes:
        - Uses a UUID temporary filename to avoid NamedTemporaryFile compatibility issues
        - Adds a fallback: if the atomic rename fails, write directly (overwrite)
        """
        out_path.parent.mkdir(exist_ok=True)
        tmp_path = None
        try:
            # Option 1: UUID temporary file + atomic rename
            tmp_name = f".tmp_{uuid.uuid4().hex}"
            tmp_path = out_path.parent / tmp_name
            tmp_path.write_bytes(data)
            tmp_path.replace(out_path)
        except (PermissionError, OSError) as e:
            # Option 2: cloud-storage fallback - write directly to the target file
            try:
                out_path.write_bytes(data)
            except Exception as e2:
                print(f"[WARNING] Failed to write {out_path}: {e2}", file=sys.stderr, flush=True)
            finally:
                # Clean up any leftover temporary file
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

    def _atomic_write_text(self, out_path: Path, text: str, encoding="utf-8"):
        self._atomic_write_bytes(out_path, text.encode(encoding))

    # --- public APIs (matching/similar to your current call names) ---
    # All methods include exception guards to avoid interrupting training if log writing fails
    def add_text(self, tag: str, text: str, step=None):
        try:
            safe_tag = tag.replace("/", "_")
            out = self.log_dir / "text" / f"{safe_tag}.txt"
            # Do not append; overwrite directly (you may also choose per-step files)
            self._atomic_write_text(out, text)
        except Exception as e:
            print(f"[WARNING] SimpleTorchWriter.add_text failed for '{tag}': {e}", file=sys.stderr, flush=True)

    def add_scalar(self, tag: str, value: float, step: int):
        try:
            safe_tag = tag.replace("/", "_")
            out = self.log_dir / "metrics" / safe_tag / f"step_{step:08d}.json"
            payload = {"step": int(step), "tag": tag, "value": float(value), "ts": time.time()}
            self._atomic_write_text(out, json.dumps(payload))
        except Exception as e:
            print(f"[WARNING] SimpleTorchWriter.add_scalar failed for '{tag}': {e}", file=sys.stderr, flush=True)

    def add_scalars(self, scalars: dict, step: int):
        try:
            # Write one file at a time: <log_dir>/metrics/_batch/step_XXXXXXXX.json
            out = self.log_dir / "metrics" / "_batch" / f"step_{step:08d}.json"
            # Convert everything to float to avoid JSON serialization errors
            ser = {k: float(v) for k, v in scalars.items()}
            payload = {"step": int(step), "scalars": ser, "ts": time.time()}
            self._atomic_write_text(out, json.dumps(payload))
        except Exception as e:
            print(f"[WARNING] SimpleTorchWriter.add_scalars failed: {e}", file=sys.stderr, flush=True)

    def add_histogram(self, tag: str, tensor, step: int, bins: int = 50):
        try:
            import numpy as np
            safe_tag = tag.replace("/", "_")
            arr = tensor.detach().float().cpu().numpy().ravel()

            if self.histogram_mode.lower() == "png":
                # Plotting version (if you want images)
                fig_path = self.log_dir / "hist" / safe_tag / f"step_{step:08d}.png"
                fig_path.parent.mkdir(exist_ok=True)
                matplotlib.use("Agg")
                plt.figure()
                plt.hist(arr, bins=bins)
                plt.title(f"{tag} @ step {step}")
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()
            else:
                # Default: save counts / bin_edges to npz; lightweight and no plotting dependency
                counts, edges = np.histogram(arr, bins=bins)
                out = self.log_dir / "hist" / safe_tag / f"step_{step:08d}.npz"
                out.parent.mkdir(exist_ok=True)
                # Use a UUID temporary file + atomic replace (cloud-storage compatible)
                tmp_name = f".tmp_{uuid.uuid4().hex}.npz"
                tmp_path = out.parent / tmp_name
                try:
                    np.savez(tmp_path, counts=counts, edges=edges, step=int(step), tag=tag)
                    tmp_path.replace(out)
                except (PermissionError, OSError):
                    # Fallback: write directly
                    np.savez(out, counts=counts, edges=edges, step=int(step), tag=tag)
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[WARNING] SimpleTorchWriter.add_histogram failed for '{tag}': {e}", file=sys.stderr, flush=True)

    def flush(self):  # kept for interface compatibility
        pass

    def close(self):  # kept for interface compatibility
        pass

def make_ot_dataloader(path, batch_size, workers=2, threads_per_worker=4):
    ds = OTPairDataset(
        path=path,
        batch_size=batch_size,
        torus_L=path.L,
        cpu_threads=threads_per_worker,
    )
    # Recommended to set persistent_workers + prefetch_factor; pin_memory strongly recommended
    loader = DataLoader(
        ds,
        batch_size=None,                 # IterableDataset: each yield is one batch
        num_workers=workers,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True
    )
    return loader


MiB = 1024 ** 2

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(self, model: nn.Module, start_epoch: int = 0, overwrite_lr: bool = False, rotate4: bool = False):
        super().__init__()
        self.model = model
        self.start_epoch = start_epoch
        self.overwrite_lr = overwrite_lr
        self.rotate4 = rotate4
        self.writer = None
        self.global_step = 0

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, **kwargs) -> torch.Tensor:
        ckpt_dir = kwargs.get("ckpt_dir", "./checkpoints")
        arch_name = "UncondUniGBNTransformer"

        args = kwargs.get("args", None)
        lr = getattr(args, "lr", 1e-3)
        use_tb = getattr(args, "use_tb", True)  # keep your switch name
        exp_name = getattr(args, "exp_name", os.path.join("runs", time.strftime("%Y%m%d-%H%M%S")))
        log_every = getattr(args, "log_every", 10)
        embed_dim = getattr(args, "embed_dim", 256)
        depth = getattr(args, "depth", 6)
        num_heads = getattr(args, "num_heads", 4)
        mlp_ratio = getattr(args, "mlp_ratio", 4.0)

        log_path = getattr(args, "log_path", "log")
        log_hist_every = getattr(args, "log_hist_every", 100)
        batch_size = getattr(args, "batch_size", 64)
        histogram_mode = getattr(args, "histogram_mode", "json")  # "json" or "png"
        only_load_model_weight    = getattr(args, "only_load_model_weight", False)
        use_warmup    = getattr(args, "use_warmup", False)
        use_cos_decay = getattr(args, "use_cos_decay", False)
        min_lr         = float(getattr(args, "min_lr", 1e-6))   # minimum learning rate
        warmup_epochs  = int(getattr(args, "warmup_epochs", 1000))
        use_schedule = bool(use_warmup and use_cos_decay)
        load_schedule = bool(getattr(args, "load_schedule", True))

        self.overwrite_lr = getattr(args, "overwrite_lr", False)

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1

        # Only rank 0 creates directories, to avoid multi-process write contention
        if rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
        if is_dist:
            dist.barrier()

        if rank == 0:
            print("Batch Size: ", batch_size)
            print("Learning Rate: ", lr)
            if not load_schedule:
                print("Using new scheduler")

        # Log directory: still placed at the location you specified (e.g. a UC Volume POSIX path /Volumes/...)
        log_dir = os.path.join(log_path, exp_name, "ts_train_log_dir", f"runs_{time.strftime('%Y%m%d-%H%M%S')}")
        if use_tb and rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            # Use our own Writer (no append)
            self.writer = SimpleTorchWriter(log_dir=log_dir, histogram_mode="png")
            # Write an hparams text entry
            self.writer.add_text("hparams", f"exp_name = {exp_name}, lr={lr}, batch_size={batch_size}, embed_dim = {embed_dim}, depth = {depth}, num_heads = {num_heads}, mlp_ratio = {mlp_ratio}")
        else:
            self.writer = None

        # Report model size
        self.total_epochs = num_epochs
        size_b = model_size_b(self.model.module if hasattr(self.model, "module") else self.model)
        if rank == 0:
            print(f"Training model with size: {size_b / MiB:.3f} MiB")

        self.model.to(device)
        if use_schedule:
            # Optimizer base lr = (lr - min_lr); min_lr is added back each step to recover the true lr
            base_lr = max(lr - min_lr, 1e-12)
            opt = self.get_optimizer(base_lr)
        else:
            # Fixed lr version: keep exactly as is
            opt = self.get_optimizer(lr)

        self.opt = opt
        self.model.train()

        dataset_len = len(self.path.p_data)

        local_len_est = math.ceil(dataset_len / world_size)
        steps_per_epoch_est = math.ceil(local_len_est / batch_size)
        total_steps  = max(1, num_epochs * steps_per_epoch_est)
        warmup_steps = max(1, min(warmup_epochs, num_epochs) * steps_per_epoch_est)

        scheduler = None
        if use_schedule:
            _EPS = 1e-8
            def lr_factor(global_step: int) -> float:
                """f in [0,1], lr(t) = min_lr + f*(lr - min_lr)"""
                T = max(1, int(total_steps))
                W = min(max(1, int(warmup_steps)), T - 1)  # ensure W < T
                s = float(global_step)
                if s < W:
                    # Linear warmup
                    return (s + _EPS) / float(W)
                # Cosine decay (1->0)
                progress = (s - W) / float(max(1, T - W))
                progress = min(1.0, max(0.0, progress))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = LambdaLR(opt, lr_lambda=lr_factor)
            self.scheduler = scheduler

        if rank == 0:
            if use_schedule:
                print(f"[LR Scheduler] epochs={num_epochs}, warmup_epochs={warmup_epochs}, "
                    f"steps/epoch≈{steps_per_epoch_est}, total_steps≈{total_steps}, "
                    f"mode=warmup+cosine, base_lr={lr}, min_lr={min_lr}")
            else:
                print("[LR Scheduler] disabled (fixed LR mode)")

        resume_path = getattr(args, "resume_path", None)
        if resume_path:
            if rank == 0:
                print(f"[resume] loading from {resume_path}")
            ep, gs = load_checkpoint(
                model=self.model,
                ckpt_path=resume_path,
                map_location="cpu",
                strict=True,
                optimizer=opt,
                scheduler=scheduler if (use_schedule and load_schedule) else None,        # None in fixed-LR mode
                add_min_lr_back=True,       # you used the base_lr + min_lr implementation
                only_load_model_weight=only_load_model_weight,
                min_lr=min_lr,              # consistent with scheduler initialization
            )
            self.start_epoch = ep
            self.global_step = gs
            if self.overwrite_lr:
                for pg in self.opt.param_groups:
                    pg["lr"] = lr
            if is_dist:
                dist.barrier()


        epoch_iter = range(self.start_epoch, num_epochs)
        pbar = tqdm(epoch_iter, desc="Training") if rank == 0 else epoch_iter

        for epoch in (pbar if rank == 0 else epoch_iter):
            self.start_epoch = epoch

            g = torch.Generator(device='cpu').manual_seed(0xC0FFEE + epoch)
            # Single-mesh / single-pcd mode: generate all-zero indices (all pointing to the same mesh/pcd)
            if getattr(self.mesh_dataset, 'is_single_mesh', False) or getattr(self.mesh_dataset, 'is_single_pcd', False):
                global_perm = torch.zeros(dataset_len, dtype=torch.long)
            else:
                global_perm = torch.randperm(dataset_len, generator=g)
            local_perm = global_perm[rank::world_size]
            local_len = len(local_perm)
            steps_per_epoch = math.ceil(local_len / batch_size)

            running_loss = 0.0
            for step in range(steps_per_epoch):
                start = step * batch_size
                end   = min((step + 1) * batch_size, local_len)
                if end <= start:
                    continue
                idx   = local_perm[start:end]

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                    loss = self.get_train_loss(
                        batch_size=end - start,
                        current_epoch=epoch,
                        indices=idx,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                if scheduler is not None:
                    scheduler.step()
                    if min_lr > 0:
                        for pg in opt.param_groups:
                            pg["lr"] = pg["lr"] + min_lr

                if rank == 0 and self.writer and (self.global_step % log_every == 0):
                    self.writer.add_scalars(
                        {
                            "train/loss": float(loss.item()),
                            "train/lr": float(opt.param_groups[0]["lr"]),
                        },
                        step=self.global_step
                    )

                # if self.writer and (self.global_step % log_hist_every == 0):
                #     base_model = self.model.module if hasattr(self.model, "module") else self.model
                #     for name, p in base_model.named_parameters():
                #         self.writer.add_histogram(f"params/{name}", p.data, self.global_step)
                #         if p.grad is not None:
                #             self.writer.add_histogram(f"grads/{name}", p.grad.data, self.global_step)

                running_loss += loss.item()
                self.global_step += 1

            # Only rank0 reports / saves ckpt
            if rank == 0:
                avg_loss = running_loss / max(steps_per_epoch, 1)
                if isinstance(pbar, tqdm):
                    try:
                        pbar.set_description(f"Epoch {epoch:03d}, avg_loss: {avg_loss:.4f}")
                    except OSError:
                        pass  # Ignore IO errors such as insufficient disk space
                if self.writer:
                    self.writer.add_scalar("epoch/avg_loss", avg_loss, epoch)

                if (epoch) % 80 == 0 or epoch == num_epochs - 1:
                    ckpt_path = save_checkpoint(
                        model=self.model,
                        args=args,
                        save_dir=ckpt_dir,
                        arch_name=arch_name,
                        optimizer=opt,
                        scheduler=scheduler,       # may be None (fixed LR mode)
                        epoch=self.start_epoch,    # 0-based
                        global_step=self.global_step,
                    )
                    try:
                        tqdm.write(f"[Rank 0] Saved checkpoint at: {ckpt_path}")
                    except OSError:
                        pass  # Ignore IO errors such as insufficient disk space

            if is_dist:
                dist.barrier()

        self.model.eval()
        if self.writer:
            self.writer.flush()
            self.writer.close()
        return self.model


class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(self, path, model, **kwargs):
        super().__init__(model); self.path = path
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size, 1, device=z.device)
        x = self.path.sample_conditional_path(z, t)
        u_ref = self.path.conditional_vector_field(x, z, t)
        u_theta = self.model(x, t)
        return ((u_theta - u_ref).pow(2).sum(dim=1)).mean()

class ConditionalFlowMatchingTrainer_NonUniformTime(Trainer):
    def __init__(self, path, model, **kwargs):
        super().__init__(model); self.path = path
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)
        t = torch.distributions.Beta(5, 2).sample((batch_size, 1)).to(z.device)
        x = self.path.sample_conditional_path(z, t)
        u_ref = self.path.conditional_vector_field(x, z, t)
        u_theta = self.model(x, t)
        return ((u_theta - u_ref).pow(2).sum(dim=1)).mean()
    
class ConditionalScoreMatchingTrainer(Trainer):
    def __init__(self, path, model, **kwargs):
        super().__init__(model); self.path = path
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size, 1, device=z.device)
        x = self.path.sample_conditional_path(z, t)
        s_ref = self.path.conditional_score(x, z, t)
        s_theta = self.model(x, t)
        return ((s_theta - s_ref).pow(2).sum(dim=1)).mean()


class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size) # (bs, c, h, w), (bs,1)
        y = y.long()
        
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        xi = torch.rand(y.shape[0]).to(y.device)
        y[xi < self.eta] = 10.0
        
        # Step 3: Sample t and x
        t = torch.rand(batch_size,1,1,1).to(z) # (bs, 1, 1, 1)
        x = self.path.sample_conditional_path(z,t) # (bs, 1, 32, 32)

        # Step 4: Regress and output loss
        ut_theta = self.model(x,t,y) # (bs, 1, 32, 32)
        ut_ref = self.path.conditional_vector_field(x,z,t) # (bs, 1, 32, 32)
        error = torch.einsum('bchw -> b', torch.square(ut_theta - ut_ref)) # (bs,)
        return torch.mean(error)



class UniGBNUnconditionalTrainer(Trainer):
    def __init__(
        self,
        path,
        model,
        rotate4=False,
        zorder=False,
        use_async_loader: bool = False,
        mesh_dataset=None,
        prefetch_batches: int = 2,
        **kwargs
    ):
        start_epoch = kwargs.pop("start_epoch", 0)
        super().__init__(model, start_epoch=start_epoch)
        self.path = path
        self.rotate4 = rotate4
        self.zorder = zorder
        self.use_async_loader = use_async_loader
        self.mesh_dataset = mesh_dataset
        self.prefetch_batches = prefetch_batches
        self._async_loader = None

    @torch.no_grad()
    def _rand_t(self, b, device):
        t = torch.rand(b, 1, 1, device=device)
        # t = torch.distributions.Beta(5.0, 2.0).sample((b, 1, 1)).to(device)
        return t


    def get_train_loss(
        self,
        batch_size: int,
        current_epoch: int,
        indices: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        x1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the training loss.

        Supports two modes:
        1. indices mode (original logic): load z from p_data via indices
        2. x0/x1 mode (async loader): directly use precomputed x0, x1
        """
        if x0 is not None and x1 is not None:
            # Async loader mode: x0 is noise, x1 is the target
            z = x1  # target point cloud
            # Directly use x0 and x1 to compute the flow matching loss
            t = self._rand_t(x0.shape[0], device=x0.device)  # (B, 1, 1)

            # Linear interpolation: x_t = (1-t) * x0 + t * x1
            x_t = (1.0 - t) * x0 + t * z  # (B, N, D)

            out = self.model(x_t, t)  # (B, N, D)
            ut_theta = out
            ut_ref = z - x0  # velocity field: dx/dt = x1 - x0

            return ((ut_theta - ut_ref) ** 2).flatten(1).mean()
        else:
            # Original indices mode
            if indices is not None:
                z, _ = self.path.p_data.get_batch(indices, rotate4=self.rotate4, zorder=self.zorder)  # (B, N, 2)
            else:
                assert batch_size is not None
                z, _ = self.path.p_data.sample(batch_size)

            t = self._rand_t(batch_size, device=z.device)  # (B, 1, 1)
            x = self.path.sample_conditional_path(z, t)  # (B, N, 2)

            out = self.model(x, t)  # (B, N, 2)
            ut_theta = out
            ut_ref = self.path.conditional_vector_field(x, z, t)

            return ((ut_theta - ut_ref) ** 2).flatten(1).mean()

    def train(self, num_epochs: int, device: torch.device, **kwargs) -> torch.Tensor:
        """
        Training method, supporting two modes:
        1. Synchronous mode (original): call the parent class train directly
        2. Asynchronous mode (use_async_loader=True): use PointSetPairAsyncLoader
        """
        if not self.use_async_loader:
            # Original synchronous mode
            return super().train(num_epochs, device, **kwargs)

        # ===== Asynchronous mode =====
        from .datasets import PointSetPairAsyncLoader

        ckpt_dir = kwargs.get("ckpt_dir", "./checkpoints")
        arch_name = "UncondUniGBNTransformer"

        args = kwargs.get("args", None)
        lr = getattr(args, "lr", 1e-3)
        use_tb = getattr(args, "use_tb", True)
        exp_name = getattr(args, "exp_name", os.path.join("runs", time.strftime("%Y%m%d-%H%M%S")))
        log_every = getattr(args, "log_every", 10)
        embed_dim = getattr(args, "embed_dim", 256)
        depth = getattr(args, "depth", 6)
        num_heads = getattr(args, "num_heads", 4)
        mlp_ratio = getattr(args, "mlp_ratio", 4.0)

        log_path = getattr(args, "log_path", "log")
        batch_size = getattr(args, "batch_size", 64)
        only_load_model_weight = getattr(args, "only_load_model_weight", False)
        use_warmup = getattr(args, "use_warmup", False)
        use_cos_decay = getattr(args, "use_cos_decay", False)
        min_lr = float(getattr(args, "min_lr", 1e-6))
        warmup_epochs = int(getattr(args, "warmup_epochs", 1000))
        use_schedule = bool(use_warmup and use_cos_decay)
        load_schedule = bool(getattr(args, "load_schedule", True))

        self.overwrite_lr = getattr(args, "overwrite_lr", False)

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1

        if rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
        if is_dist:
            dist.barrier()

        if rank == 0:
            print("Batch Size: ", batch_size)
            print("Learning Rate: ", lr)
            print("[Async Loader Mode] Using PointSetPairAsyncLoader")

        # Log directory
        log_dir = os.path.join(
            log_path,
            exp_name,
            "ts_train_log_dir",
            f"runs_{time.strftime('%Y%m%d-%H%M%S')}",
        )
        if use_tb and rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SimpleTorchWriter(log_dir=log_dir, histogram_mode="png")
            self.writer.add_text(
                "hparams",
                f"exp_name = {exp_name}, lr={lr}, batch_size={batch_size}, "
                f"embed_dim = {embed_dim}, depth = {depth}, num_heads = {num_heads}, mlp_ratio = {mlp_ratio}",
            )
        else:
            self.writer = None

        self.total_epochs = num_epochs
        self.model.to(device)

        if use_schedule:
            base_lr = max(lr - min_lr, 1e-12)
            opt = self.get_optimizer(base_lr)
        else:
            opt = self.get_optimizer(lr)

        self.opt = opt
        self.model.train()

        # dataset_len uses mesh_dataset.num_meshes
        # Single-mesh / single-pcd mode: use batch_size * 12 as a virtual data count, ensuring 12 steps per epoch
        if getattr(self.mesh_dataset, 'is_single_mesh', False) or getattr(self.mesh_dataset, 'is_single_pcd', False):
            dataset_len = batch_size * 12
        else:
            dataset_len = self.mesh_dataset.num_meshes

        local_len_est = math.ceil(dataset_len / world_size)
        steps_per_epoch_est = math.ceil(local_len_est / batch_size)
        total_steps = max(1, num_epochs * steps_per_epoch_est)
        warmup_steps = max(1, min(warmup_epochs, num_epochs) * steps_per_epoch_est)

        scheduler = None
        if use_schedule:
            _EPS = 1e-8

            def lr_factor(global_step: int) -> float:
                T = max(1, int(total_steps))
                W = min(max(1, int(warmup_steps)), T - 1)
                s = float(global_step)
                if s < W:
                    return (s + _EPS) / float(W)
                progress = (s - W) / float(max(1, T - W))
                progress = min(1.0, max(0.0, progress))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = LambdaLR(opt, lr_lambda=lr_factor)
            self.scheduler = scheduler

        if rank == 0:
            if use_schedule:
                print(
                    f"[LR Scheduler] epochs={num_epochs}, warmup_epochs={warmup_epochs}, "
                    f"steps/epoch≈{steps_per_epoch_est}, total_steps≈{total_steps}, "
                    f"mode=warmup+cosine, base_lr={lr}, min_lr={min_lr}"
                )
            else:
                print("[LR Scheduler] disabled (fixed LR mode)")

        resume_path = getattr(args, "resume_path", None)
        if resume_path:
            if rank == 0:
                print(f"[resume] loading from {resume_path}")
            ep, gs = load_checkpoint(
                model=self.model,
                ckpt_path=resume_path,
                map_location="cpu",
                strict=True,
                optimizer=opt,
                scheduler=scheduler if (use_schedule and load_schedule) else None,
                add_min_lr_back=True,
                only_load_model_weight=only_load_model_weight,
                min_lr=min_lr,
            )
            self.start_epoch = ep
            self.global_step = gs
            if self.overwrite_lr:
                for pg in self.opt.param_groups:
                    pg["lr"] = lr
            if is_dist:
                dist.barrier()

        epoch_iter = range(self.start_epoch, num_epochs)
        pbar = tqdm(epoch_iter, desc="Training") if rank == 0 else epoch_iter

        # Create the async loader
        if self._async_loader is None:
            self._async_loader = PointSetPairAsyncLoader(
                dataset=self.mesh_dataset,
                batch_size=batch_size,
                device=device,
                prefetch_batches=self.prefetch_batches,
            )

        # Main training loop
        for epoch in (pbar if rank == 0 else epoch_iter):
            self.start_epoch = epoch

            g = torch.Generator(device="cpu").manual_seed(0xC0FFEE + epoch)
            # Single-mesh / single-pcd mode: generate all-zero indices (all pointing to the same mesh/pcd)
            if getattr(self.mesh_dataset, 'is_single_mesh', False) or getattr(self.mesh_dataset, 'is_single_pcd', False):
                global_perm = torch.zeros(dataset_len, dtype=torch.long)
            else:
                global_perm = torch.randperm(dataset_len, generator=g)
            local_perm = global_perm[rank::world_size]

            self._async_loader.start_epoch(local_perm=local_perm, epoch=epoch)
            steps_per_epoch = self._async_loader.num_steps

            running_loss = 0.0
            for step in range(steps_per_epoch):
                opt.zero_grad(set_to_none=True)

                try:
                    x0, x1 = self._async_loader.next_batch()
                except StopIteration:
                    break

                with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                    loss = self.get_train_loss(
                        batch_size=x0.shape[0],
                        current_epoch=epoch,
                        x0=x0,
                        x1=x1,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                if scheduler is not None:
                    scheduler.step()
                    if min_lr > 0:
                        for pg in opt.param_groups:
                            pg["lr"] = pg["lr"] + min_lr

                if rank == 0 and self.writer and (self.global_step % log_every == 0):
                    self.writer.add_scalars(
                        {
                            "train/loss": float(loss.item()),
                            "train/lr": float(opt.param_groups[0]["lr"]),
                        },
                        step=self.global_step,
                    )

                running_loss += loss.item()
                self.global_step += 1

            # End of epoch
            self._async_loader.finish_epoch()

            if rank == 0:
                avg_loss = running_loss / max(steps_per_epoch, 1)
                if isinstance(pbar, tqdm):
                    try:
                        pbar.set_description(f"Epoch {epoch:03d}, avg_loss: {avg_loss:.4f}")
                    except OSError:
                        pass  # Ignore IO errors such as insufficient disk space
                if self.writer:
                    self.writer.add_scalar("epoch/avg_loss", avg_loss, epoch)

                if (epoch) % 80 == 0 or epoch == num_epochs - 1:
                    ckpt_path = save_checkpoint(
                        model=self.model,
                        args=args,
                        save_dir=ckpt_dir,
                        arch_name=arch_name,
                        optimizer=opt,
                        scheduler=scheduler,
                        epoch=self.start_epoch,
                        global_step=self.global_step,
                    )
                    try:
                        tqdm.write(f"[Rank 0] Saved checkpoint at: {ckpt_path}")
                    except OSError:
                        pass  # Ignore IO errors such as insufficient disk space

            if is_dist:
                dist.barrier()

        self.model.eval()
        if self.writer:
            self.writer.flush()
            self.writer.close()
        return self.model


class UnconditionalTrainer_OT_Pair(Trainer):
    def __init__(self, path, model: nn.Module, rotate4=False, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
        self.rotate4 = rotate4

    @torch.no_grad()
    def _rand_t(self, b, device):
        t = torch.rand(b, 1, 1, device=device)
        # t = torch.distributions.Beta(5.0, 2.0).sample((b, 1, 1)).to(device)
        return t

    def get_train_loss(
        self,
        batch_size: int,
        current_epoch: int,
        prefetch: Optional[dict] = None,
        **kwargs
    ) -> torch.Tensor:
        x0 = prefetch["x0"]
        z_tgt_perm = prefetch["z_tgt_perm"]
        t = self._rand_t(batch_size, device=x0.device) # (B, 1, 1)

        delta = minimal_image(z_tgt_perm - x0, self.path.L)    # [B,N,2]
        x_t = wrap_coords(x0 + t * delta, self.path.L)

        u_theta = self.model(x_t, t)
        ut_ref   = self.path.conditional_vector_field(x_t, z_tgt_perm, t)
        return ((u_theta - ut_ref).pow(2).flatten(1).mean(dim=1)).mean()




class UniGBNConditionalTrainer(UniGBNUnconditionalTrainer):
    def __init__(self, *args, img_npz_list=None, p_drop=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_npz_list = img_npz_list
        self.p_drop = float(p_drop)

    def _load_img_tokens_batch(self, npz_paths, device):
        toks = []
        for p in npz_paths:
            z = np.load(p, allow_pickle=True)
            pt = torch.from_numpy(z["patch_tokens"]).to(device)
            toks.append(pt.float())
        return torch.stack(toks, dim=0)  # (B,Lc,Cimg)

    def get_train_loss(self, batch_size: int, current_epoch: int, indices: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        B = int(batch_size)

        # 1) Fetch data: target point set y
        y, _ = self.path.p_data.get_batch(indices.tolist())   # (B,N,2) or (B,N,D)
        y = y.float().to(device)
        B, N, D = y.shape

        # 2) Linear path samples (flow matching)
        t = (torch.rand(B, 1, 1, device=device) * 0.999 + 1e-3)
        x0, _ = self.path.p_simple.sample(B)                  # (B,N,D)
        x0 = x0.to(device).float()
        x_t = (1.0 - t) * x0 + t * y
        v_star = (y - x0)

        # 3) Image condition (batch-load ViT tokens)
        npz_paths = [self.img_npz_list[i] for i in indices.tolist()]
        img_tokens = self._load_img_tokens_batch(npz_paths, device=device)  # (B,Lc,Cimg)

        # 4) Condition dropout (CFG training)
        cond_drop_flag = (torch.rand((), device=device) < self.p_drop).item()

        # 5) Forward pass + MSE
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            v_pred = self.model(x_t, t, img_cond=img_tokens, cond_drop=cond_drop_flag)
            loss   = F.mse_loss(v_pred, v_star)

        return loss



class MeshUncondTrainer(Trainer):
    def __init__(
        self,
        path,
        mesh_dataset,
        model,
        rotate4: bool = False,
        zorder: bool = False,
        output_x0: bool = False,
        prefetch_batches: int = 2,
        cond_vec_use_x0: bool = False,
        cond_vec_use_x0_with_n0: bool = False,
        sample_cond_use_n0: bool = False,
        zero_t0: bool = False,
        overwrite_lr: bool = False,
        **kwargs,
    ):
        start_epoch = kwargs.pop("start_epoch", 0)
        super().__init__(model, start_epoch=start_epoch, overwrite_lr = overwrite_lr)

        self.path = path
        self.mesh_dataset = mesh_dataset
        self.rotate4 = rotate4
        self.zorder = zorder
        self.output_x0 = output_x0
        self.prefetch_batches = prefetch_batches
        self.cond_vec_use_x0 = cond_vec_use_x0
        self.cond_vec_use_x0_with_n0 = cond_vec_use_x0_with_n0
        self.sample_cond_use_n0 = sample_cond_use_n0
        self.zero_t0 = zero_t0

        self.writer = None
        self._async_loader: Optional[MeshPairAsyncLoader] = None

    @torch.no_grad()
    def _rand_t(self, b: int, device: torch.device):
        return torch.rand(b, 1, 1, device=device)

    def get_train_loss(
        self,
        x0: torch.Tensor,   # (B, N, 3/6)
        x1: torch.Tensor,   # (B, N, 3/6)
        current_epoch: int,
    ) -> torch.Tensor:
        B = x0.shape[0]
        device = x0.device

        t = self._rand_t(B, device=device)  # (B,1,1)

        if self.sample_cond_use_n0:
            if not self.zero_t0:
                xt = self.path.sample_conditional_path_inputx0_withn0(x0, x1, t)  # (B,N,3)
            else:
                xt = self.path.sample_conditional_path_inputx0_zeron0(x0, x1, t)  # (B,N,3)
        else:
            # must be quadratic
            xt = self.path.sample_conditional_path_inputx0(x0, x1, t)  # (B,N,3)

        out = self.model(xt, t)  # (B,N,3) or if use_magnitude : (B,N,1)
        ut_theta = out
        # if not self.cond_vec_use_x0:
        #     ut_ref = self.path.conditional_vector_field(xt, x1, t)
        # else:
        #     ut_ref = self.path.conditional_vector_field_inputx0(x0, x1, t)

        if self.cond_vec_use_x0_with_n0:
            if not self.zero_t0:
                ut_ref = self.path.conditional_vector_field_inputx0_withn0(x0, x1, t)
            else:
                ut_ref = self.path.conditional_vector_field_inputx0_zeron0(x0, x1, t)
        elif self.cond_vec_use_x0:
            ut_ref = self.path.conditional_vector_field_inputx0(x0, x1, t)
        else:
            ut_ref = self.path.conditional_vector_field(xt, x1, t)

        if ut_theta.shape[-1] == 1:
            # (B,N,3) -> (B,N,1)
            target = ut_ref.norm(dim=-1, keepdim=True)
        else:
            target = ut_ref

        return ((ut_theta - target) ** 2).flatten(1).mean()

    # ----------------- Override train -----------------

    def train(self, num_epochs: int, device: torch.device, **kwargs) -> torch.Tensor:
        ckpt_dir = kwargs.get("ckpt_dir", "./checkpoints")
        arch_name = "UncondUniGBNTransformer"

        args = kwargs.get("args", None)
        lr = getattr(args, "lr", 1e-3)
        use_tb = getattr(args, "use_tb", True)
        exp_name = getattr(args, "exp_name", os.path.join("runs", time.strftime("%Y%m%d-%H%M%S")))
        log_every = getattr(args, "log_every", 10)
        embed_dim = getattr(args, "embed_dim", 256)
        depth = getattr(args, "depth", 6)
        num_heads = getattr(args, "num_heads", 4)
        mlp_ratio = getattr(args, "mlp_ratio", 4.0)

        log_path = getattr(args, "log_path", "log")
        log_hist_every = getattr(args, "log_hist_every", 100)
        batch_size = getattr(args, "batch_size", 64)
        histogram_mode = getattr(args, "histogram_mode", "json")
        only_load_model_weight = getattr(args, "only_load_model_weight", False)
        use_warmup = getattr(args, "use_warmup", False)
        use_cos_decay = getattr(args, "use_cos_decay", False)
        min_lr = float(getattr(args, "min_lr", 1e-6))
        warmup_epochs = int(getattr(args, "warmup_epochs", 1000))
        use_schedule = bool(use_warmup and use_cos_decay)
        load_schedule = bool(getattr(args, "load_schedule", True))

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1

        # Only rank 0 creates directories, to avoid multi-process write contention
        if rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
        if is_dist:
            dist.barrier()

        if rank == 0:
            print("Batch Size: ", batch_size)
            print("Learning Rate: ", lr)
            if not load_schedule:
                print("Using new scheduler")

        # Log directory
        log_dir = os.path.join(
            log_path,
            exp_name,
            "ts_train_log_dir",
            f"runs_{time.strftime('%Y%m%d-%H%M%S')}",
        )
        if use_tb and rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SimpleTorchWriter(log_dir=log_dir, histogram_mode="png")
            self.writer.add_text(
                "hparams",
                f"exp_name = {exp_name}, lr={lr}, batch_size={batch_size}, "
                f"embed_dim = {embed_dim}, depth = {depth}, num_heads = {num_heads}, mlp_ratio = {mlp_ratio}",
            )
        else:
            self.writer = None

        # Model size
        self.total_epochs = num_epochs
        size_b = model_size_b(self.model.module if hasattr(self.model, "module") else self.model)
        if rank == 0:
            print(f"Training model with size: {size_b / MiB:.3f} MiB")

        self.model.to(device)
        if use_schedule:
            base_lr = max(lr - min_lr, 1e-12)
            opt = self.get_optimizer(base_lr)
        else:
            opt = self.get_optimizer(lr)

        self.opt = opt
        self.model.train()

        # ----------------- Here dataset_len is set to mesh_dataset.num_meshes -----------------
        # Single-mesh / single-pcd mode: use batch_size * 12 as a virtual data count, ensuring 12 steps per epoch
        if getattr(self.mesh_dataset, 'is_single_mesh', False) or getattr(self.mesh_dataset, 'is_single_pcd', False):
            dataset_len = batch_size * 12
        else:
            dataset_len = self.mesh_dataset.num_meshes

        local_len_est = math.ceil(dataset_len / world_size)
        steps_per_epoch_est = math.ceil(local_len_est / batch_size)
        total_steps = max(1, num_epochs * steps_per_epoch_est)
        warmup_steps = max(1, min(warmup_epochs, num_epochs) * steps_per_epoch_est)

        scheduler = None
        if use_schedule:
            _EPS = 1e-8

            def lr_factor(global_step: int) -> float:
                T = max(1, int(total_steps))
                W = min(max(1, int(warmup_steps)), T - 1)
                s = float(global_step)
                if s < W:
                    return (s + _EPS) / float(W)
                progress = (s - W) / float(max(1, T - W))
                progress = min(1.0, max(0.0, progress))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = LambdaLR(opt, lr_lambda=lr_factor)
            self.scheduler = scheduler

        if rank == 0:
            if use_schedule:
                print(
                    f"[LR Scheduler] epochs={num_epochs}, warmup_epochs={warmup_epochs}, "
                    f"steps/epoch≈{steps_per_epoch_est}, total_steps≈{total_steps}, "
                    f"mode=warmup+cosine, base_lr={lr}, min_lr={min_lr}"
                )
            else:
                print("[LR Scheduler] disabled (fixed LR mode)")

        resume_path = getattr(args, "resume_path", None)
        if resume_path:
            if rank == 0:
                print(f"[resume] loading from {resume_path}")
            ep, gs = load_checkpoint(
                model=self.model,
                ckpt_path=resume_path,
                map_location="cpu",
                strict=True,
                optimizer=opt,
                scheduler=scheduler if (use_schedule and load_schedule) else None,
                add_min_lr_back=True,
                only_load_model_weight=only_load_model_weight,
                min_lr=min_lr,
            )
            self.start_epoch = ep
            self.global_step = gs
            if self.overwrite_lr:
                for pg in self.opt.param_groups:
                    pg["lr"] = lr
            if is_dist:
                dist.barrier()

        epoch_iter = range(self.start_epoch, num_epochs)
        pbar = tqdm(epoch_iter, desc="Training") if rank == 0 else epoch_iter

        if self._async_loader is None:
            self._async_loader = MeshPairAsyncLoader(
                dataset=self.mesh_dataset,
                batch_size=batch_size,
                device=device,
                prefetch_batches=self.prefetch_batches,
                output_x0=self.output_x0,
            )

        # ----------------- Main training loop -----------------
        for epoch in (pbar if rank == 0 else epoch_iter):
            self.start_epoch = epoch

            # Same global_perm / local_perm as before
            g = torch.Generator(device="cpu").manual_seed(0xC0FFEE + epoch)
            # Single-mesh / single-pcd mode: generate all-zero indices (all pointing to the same mesh/pcd)
            if getattr(self.mesh_dataset, 'is_single_mesh', False) or getattr(self.mesh_dataset, 'is_single_pcd', False):
                global_perm = torch.zeros(dataset_len, dtype=torch.long)
            else:
                global_perm = torch.randperm(dataset_len, generator=g)
            local_perm = global_perm[rank::world_size]
            local_len = len(local_perm)

            self._async_loader.start_epoch(local_perm=local_perm, epoch=epoch)
            steps_per_epoch = self._async_loader.num_steps

            running_loss = 0.0
            for step in range(steps_per_epoch):
                opt.zero_grad(set_to_none=True)

                # Key: fetch the already-computed (x0,x1) from the queue here
                try:
                    x0, x1 = self._async_loader.next_batch()
                except StopIteration:
                    break  # In theory this should not end early

                with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                    loss = self.get_train_loss(
                        x0=x0,
                        x1=x1,
                        current_epoch=epoch,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                if scheduler is not None:
                    scheduler.step()
                    if min_lr > 0:
                        for pg in opt.param_groups:
                            pg["lr"] = pg["lr"] + min_lr

                if rank == 0 and self.writer and (self.global_step % log_every == 0):
                    self.writer.add_scalars(
                        {
                            "train/loss": float(loss.item()),
                            "train/lr": float(opt.param_groups[0]["lr"]),
                        },
                        step=self.global_step,
                    )

                running_loss += loss.item()
                self.global_step += 1

            # End of epoch: shut down the loader's worker threads
            self._async_loader.finish_epoch()

            # Multi-GPU all-reduce loss
            avg_loss_local = running_loss / max(steps_per_epoch, 1)
            if is_dist:
                # Reduce both the local avg_loss and the step count to compute the global average
                loss_tensor = torch.tensor([avg_loss_local * steps_per_epoch, float(steps_per_epoch)], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                total_loss_sum, total_steps = loss_tensor[0].item(), loss_tensor[1].item()
                avg_loss = total_loss_sum / max(total_steps, 1)
            else:
                avg_loss = avg_loss_local

            # rank0 writes logs / saves ckpt
            if rank == 0:
                if isinstance(pbar, tqdm):
                    try:
                        pbar.set_description(f"Epoch {epoch:03d}, avg_loss: {avg_loss:.4f}")
                    except OSError:
                        pass  # Ignore IO errors such as insufficient disk space
                if self.writer:
                    self.writer.add_scalar("epoch/avg_loss", avg_loss, epoch)

                if (epoch) % 80 == 0 or epoch == num_epochs - 1:
                    ckpt_path = save_checkpoint(
                        model=self.model,
                        args=args,
                        save_dir=ckpt_dir,
                        arch_name=arch_name,
                        optimizer=opt,
                        scheduler=scheduler,
                        epoch=self.start_epoch,
                        global_step=self.global_step,
                    )
                    try:
                        tqdm.write(f"[Rank 0] Saved checkpoint at: {ckpt_path}")
                    except OSError:
                        pass  # Ignore IO errors such as insufficient disk space

            if is_dist:
                dist.barrier()

        self.model.eval()
        if self.writer:
            self.writer.flush()
            self.writer.close()
        return self.model
