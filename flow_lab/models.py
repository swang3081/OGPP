from typing import List, Type, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import math
from .dynamics import ConditionalVectorField, UnconditionalVectorField, wrap_coords, minimal_image
import torch.nn.functional as F
import numpy as np
import functools
# from torch_cluster import fps
from einops import rearrange, repeat
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint

# PVCNN modules - lazy loaded to avoid compilation when not needed
_HAS_PVCNN = None  # Will be set on first import attempt
_pvcnn_modules = {}  # Cache for imported modules

def _lazy_import_pvcnn():
    """Lazily import PVCNN modules only when needed (triggers CUDA compilation)."""
    global _HAS_PVCNN, _pvcnn_modules
    if _HAS_PVCNN is not None:
        return _HAS_PVCNN

    try:
        import sys
        import os
        _pvcnn_path = os.path.join(os.path.dirname(__file__), '..', 'external', 'PVD-main')
        _pvcnn_path = os.path.abspath(_pvcnn_path)
        if _pvcnn_path not in sys.path:
            sys.path.insert(0, _pvcnn_path)
        from modules import SharedMLP, PVConv, PointNetSAModule, PointNetAModule, PointNetFPModule, Attention
        _pvcnn_modules['SharedMLP'] = SharedMLP
        _pvcnn_modules['PVConv'] = PVConv
        _pvcnn_modules['PointNetSAModule'] = PointNetSAModule
        _pvcnn_modules['PointNetAModule'] = PointNetAModule
        _pvcnn_modules['PointNetFPModule'] = PointNetFPModule
        _pvcnn_modules['Attention'] = Attention
        _HAS_PVCNN = True
    except ImportError as e:
        print(f"Warning: PVCNN modules not available: {e}")
        _HAS_PVCNN = False

    return _HAS_PVCNN

try:
    # PyTorch 2.1+ generally
    from torch.nn.attention import sdpa_kernel, SDPBackend
    print("Has SPDA Kernel")
    _HAS_SDPA_KERNEL = True
except Exception:
    sdpa_kernel, SDPBackend = None, None
    _HAS_SDPA_KERNEL = False

class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1) # (bs, 1)
        freqs = t * self.weights * 2 * math.pi # (bs, half_dim)
        sin_embed = torch.sin(freqs) # (bs, half_dim)
        cos_embed = torch.cos(freqs) # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # (bs, dim)
    
class ResidualLayer(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        # Converts (bs, time_embed_dim) -> (bs, channels)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels)
        )
        # Converts (bs, y_embed_dim) -> (bs, channels)
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, channels)
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        res = x.clone() # (bs, c, h, w)

        # Initial conv block
        x = self.block1(x) # (bs, c, h, w)

        # Add time embedding
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1) # (bs, c, 1, 1)
        x = x + t_embed

        # Add y embedding (conditional embedding)
        y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1) # (bs, c, 1, 1)
        x = x + y_embed

        # Second conv block
        x = self.block2(x) # (bs, c, h, w)

        # Add back residual
        x = x + res # (bs, c, h, w)

        return x
        
class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_in, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c_in, h, w) -> (bs, c_in, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Downsample: (bs, c_in, h, w) -> (bs, c_out, h // 2, w // 2)
        x = self.downsample(x)

        return x

class Midcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c, h, w) -> (bs, c, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
            
        return x
        
class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1))
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_out, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Upsample: (bs, c_in, h, w) -> (bs, c_out, 2 * h, 2 * w) 
        x = self.upsample(x)
        
        # Pass through residual blocks: (bs, c_out, h, w) -> (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x
        
class MNISTUNet(ConditionalVectorField):
    def __init__(self, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_embed_dim: int): 
        super().__init__()
        # Initial convolution: (bs, 1, 32, 32) -> (bs, c_0, 32, 32)
        self.init_conv = nn.Sequential(nn.Conv2d(1, channels[0], kernel_size=3, padding=1), nn.BatchNorm2d(channels[0]), nn.SiLU())

        # Initialize time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)

        # Initialize y embedder
        self.y_embedder = nn.Embedding(num_embeddings = 11, embedding_dim = y_embed_dim)

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)
            
        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, 1, 32, 32)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, 1, 32, 32)
        """
        # Embed t and y
        t_embed = self.time_embedder(t) # (bs, time_embed_dim)
        y_embed = self.y_embedder(y) # (bs, y_embed_dim)
        
        # Initial convolution
        x = self.init_conv(x) # (bs, c_0, 32, 32)

        residuals = []
        
        # Encoders
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed) # (bs, c_i, h, w) -> (bs, c_{i+1}, h // 2, w //2)
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoders
        for decoder in self.decoders:
            res = residuals.pop() # (bs, c_i, h, w)
            x = x + res
            x = decoder(x, t_embed, y_embed) # (bs, c_i, h, w) -> (bs, c_{i-1}, 2 * h, 2 * w)

        # Final convolution
        x = self.final_conv(x) # (bs, 1, 32, 32)

        return x


class RGBUNet(ConditionalVectorField):
    """
    UNet for RGB inputs/outputs on 32x32 images.
    Drop-in replacement for MNISTUNet, but with configurable in/out channels (defaults 3->3).
    forward(x, t, y) -> (B, out_channels, 32, 32)
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: List[int] = [32, 64, 128],
        num_residual_layers: int = 2,
        t_embed_dim: int = 40,
        y_embed_dim: int = 40,
        num_y_classes: int = 11,  # keep consistent with the original implementation (10 classes + 1 CFG null label)
        FourierEncoderCls=None,   # for custom injection if needed
        EncoderCls=None,
        MidcoderCls=None,
        DecoderCls=None,
    ):
        super().__init__()

        # Allow injecting module classes from outside; default to the same-named classes
        FourierEncoderLocal = FourierEncoder if FourierEncoderCls is None else FourierEncoderCls
        EncoderLocal = Encoder if EncoderCls is None else EncoderCls
        MidcoderLocal = Midcoder if MidcoderCls is None else MidcoderCls
        DecoderLocal = Decoder if DecoderCls is None else DecoderCls

        # Initial convolution: (B, in_channels, 32, 32) -> (B, c0, 32, 32)
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )

        # Time / label embeddings
        self.time_embedder = FourierEncoderLocal(t_embed_dim)
        self.y_embedder = nn.Embedding(num_embeddings=num_y_classes, embedding_dim=y_embed_dim)

        # Downsampling encoder and upsampling decoder
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(EncoderLocal(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(DecoderLocal(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        # Middle block
        self.midcoder = MidcoderLocal(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)

        # Final output to out_channels (keep color)
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        x: (B, in_channels, 32, 32)
        t: (B, 1, 1, 1)
        y: (B,)
        -> (B, out_channels, 32, 32)
        """
        t_embed = self.time_embedder(t)   # (B, t_embed_dim)
        y_embed = self.y_embedder(y)      # (B, y_embed_dim)

        x = self.init_conv(x)             # (B, c0, 32, 32)

        residuals = []
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed)  # downsample & channel up
            residuals.append(x.clone())

        x = self.midcoder(x, t_embed, y_embed)

        for decoder in self.decoders:
            res = residuals.pop()
            x = x + res                      # skip
            x = decoder(x, t_embed, y_embed) # upsample & channel down

        x = self.final_conv(x)               # (B, out_channels, 32, 32)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, t_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_c)
        self.norm2 = nn.GroupNorm(groups, out_c)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * out_c))
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # temb: (B, t_dim)
        h = self.conv1(self.act(self.norm1(x)))
        gamma, beta = self.time_mlp(temb).chunk(2, dim=-1)  # (B,out_c),(B,out_c)
        h = h * (1 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.down = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1)
    def forward(self, x): return self.down(x)


class Up(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    def forward(self, x): 
        x = self.up(x)
        return self.conv(x)


class UncondRGBUNet(UnconditionalVectorField):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels = [32, 64, 128],
        t_embed_dim: int = 40,
        num_residual_layers: int = 2,
    ):
        super().__init__()
        self.t_embed = FourierEncoder(t_embed_dim)

        c0, c1, c2 = channels
        self.init = nn.Conv2d(in_channels, c0, kernel_size=3, padding=1)

        # Encoder
        self.e0 = ResBlock(c0, c0, t_embed_dim)
        self.down0 = Down(c0)           # 32 -> 16
        self.e1 = ResBlock(c0, c1, t_embed_dim)
        self.down1 = Down(c1)           # 16 -> 8
        self.e2 = ResBlock(c1, c2, t_embed_dim)

        # Bottleneck
        self.mid1 = ResBlock(c2, c2, t_embed_dim)
        self.mid2 = ResBlock(c2, c2, t_embed_dim)

        # Decoder
        self.up1 = Up(c2, c1)           # 8 -> 16
        self.d1  = ResBlock(c1 + c1, c1, t_embed_dim)
        self.up0 = Up(c1, c0)           # 16 -> 32
        self.d0  = ResBlock(c0 + c0, c0, t_embed_dim)

        self.final = nn.Conv2d(c0, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: (B,1,1,1) or (B,1)
        temb = self.t_embed(t)  # (B, t_embed_dim)

        x0 = self.init(x)               # (B,c0,32,32)
        x1 = self.e0(x0, temb)
        d0 = self.down0(x1)             # (B,c0,16,16)

        x2 = self.e1(d0, temb)          # (B,c1,16,16)
        d1 = self.down1(x2)             # (B,c1,8,8)

        x3 = self.e2(d1, temb)          # (B,c2,8,8)

        m  = self.mid1(x3, temb)
        m  = self.mid2(m, temb)

        u1 = self.up1(m)                # (B,c1,16,16)
        u1 = torch.cat([u1, x2], dim=1) # skip
        u1 = self.d1(u1, temb)

        u0 = self.up0(u1)               # (B,c0,32,32)
        u0 = torch.cat([u0, x1], dim=1) # skip
        u0 = self.d0(u0, temb)

        out = self.final(u0)            # (B,3,32,32)
        return out

class UncondRGBUNet_N(UnconditionalVectorField):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels = [32, 64, 128],
        t_embed_dim: int = 40,
        target_points: int = 512,    # fixed pad to 512
        hw: tuple[int, int] = (16, 32),  # 512 = 16×32
    ):
        super().__init__()
        self.t_embed = FourierEncoder(t_embed_dim)

        self.target_points = target_points
        self.H, self.W = hw
        assert self.H * self.W == target_points, f"H×W must equal {target_points}"

        c0, c1, c2 = channels
        self.init = nn.Conv2d(in_channels, c0, kernel_size=3, padding=1)

        # Encoder
        self.e0   = ResBlock(c0, c0, t_embed_dim)
        self.down0= Down(c0)           # H,W /2
        self.e1   = ResBlock(c0, c1, t_embed_dim)
        self.down1= Down(c1)           # H,W /2
        self.e2   = ResBlock(c1, c2, t_embed_dim)

        # Bottleneck
        self.mid1 = ResBlock(c2, c2, t_embed_dim)
        self.mid2 = ResBlock(c2, c2, t_embed_dim)

        # Decoder
        self.up1  = Up(c2, c1)         # ×2
        self.d1   = ResBlock(c1 + c1, c1, t_embed_dim)
        self.up0  = Up(c1, c0)         # ×2
        self.d0   = ResBlock(c0 + c0, c0, t_embed_dim)

        self.final= nn.Conv2d(c0, out_channels, kernel_size=3, padding=1)

    # ----------------------------
    def _unet_bchw(self, x_bchw: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Standard U-Net backbone: both input and output are (B,C,H,W)"""
        x0 = self.init(x_bchw)
        x1 = self.e0(x0, temb)
        d0 = self.down0(x1)

        x2 = self.e1(d0, temb)
        d1 = self.down1(x2)

        x3 = self.e2(d1, temb)

        m  = self.mid1(x3, temb)
        m  = self.mid2(m, temb)

        u1 = self.up1(m)
        u1 = torch.cat([u1, x2], dim=1)  # skip
        u1 = self.d1(u1, temb)

        u0 = self.up0(u1)
        u0 = torch.cat([u0, x1], dim=1)  # skip
        u0 = self.d0(u0, temb)

        out = self.final(u0)              # (B, out_channels, H, W)
        return out

    # ----------------------------
    def forward(self, x_bnC: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_bnC: (B, 500, C_in)
        t:     (B,1)
        return: (B, 500, 3)
        """
        assert x_bnC.dim() == 3, f"expect (B,N,C), got {tuple(x_bnC.shape)}"
        B, N, Cin = x_bnC.shape
        pad_len = self.target_points - N
        assert pad_len >= 0, f"N={N} cannot exceed target {self.target_points}"

        # pad to 512 points
        if pad_len > 0:
            pad_tensor = torch.zeros(B, pad_len, Cin, device=x_bnC.device, dtype=x_bnC.dtype)
            x_bnC = torch.cat([x_bnC, pad_tensor], dim=1)  # (B,512,C)

        # (B,N,C) → (B,C,H,W)
        x_bchw = x_bnC.permute(0, 2, 1).contiguous().view(B, Cin, self.H, self.W)

        temb = self.t_embed(t)
        y_bchw = self._unet_bchw(x_bchw, temb)  # (B,3,H,W)

        # (B,3,H,W) → (B,N,3)
        y_bn3 = y_bchw.view(B, 3, self.target_points).permute(0, 2, 1).contiguous()

        # remove padding, return the original 500 points
        if pad_len > 0:
            y_bn3 = y_bn3[:, :N, :]

        return y_bn3



class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.p = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                     # (B,D,Hp,Wp)
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)     # (B,N,D)
        return x, Hp, Wp

class TFVectorFieldBuiltin(ConditionalVectorField):
    def __init__(self,
        in_channels=3, out_channels=3, patch_size=4,
        embed_dim=256, depth=8, num_heads=8, mlp_ratio=4.0,
        t_embed_dim=40, y_embed_dim=40, num_y_classes=11
    ):
        super().__init__()
        self.patch = PatchEmbed(in_channels, embed_dim, patch_size)
        n_tokens = (32//patch_size)*(32//patch_size)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.y_embedder = nn.Embedding(num_y_classes, y_embed_dim)
        self.cond_proj = nn.Linear(t_embed_dim + y_embed_dim, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim*mlp_ratio),
            batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, patch_size*patch_size*out_channels)
        self.patch_size = patch_size

    def _unpatchify(self, tokens, Hp, Wp):
        B, N, PP_C = tokens.shape
        P = self.patch_size
        Cout = PP_C // (P*P)
        x = tokens.view(B, Hp, Wp, P, P, Cout).permute(0,5,1,3,2,4).contiguous()
        return x.view(B, Cout, Hp*P, Wp*P)

    def forward(self, x, t, y):
        # Condition vector: added to each token (simple and effective)
        t_emb = self.time_embedder(t)          # (B, t_dim)
        y_emb = self.y_embedder(y.long())      # (B, y_dim)
        cond  = self.cond_proj(torch.cat([t_emb, y_emb], dim=-1))  # (B, D)

        tok, Hp, Wp = self.patch(x)            # (B,N,D)
        tok = tok + self.pos[:, :tok.size(1), :]
        tok = tok + cond.unsqueeze(1)          # condition injection

        tok = self.encoder(tok)                # (B,N,D)
        tok = self.head(tok)                   # (B,N,P*P*C)
        return self._unpatchify(tok, Hp, Wp)   # (B,C,32,32)



class UncondTransFormer(UnconditionalVectorField):
    """
    Unconditional Transformer-based Vector Field for 32x32 RGB images.
    Similar to ViT, but injects time embeddings as condition.
    """
    def __init__(self,
        in_channels=3, out_channels=3, patch_size=4,
        embed_dim=256, depth=8, num_heads=8, mlp_ratio=4.0,
        t_embed_dim=40
    ):
        super().__init__()
        self.patch = PatchEmbed(in_channels, embed_dim, patch_size)
        n_tokens = (32 // patch_size) * (32 // patch_size)

        # Positional embeddings
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))

        # Time embedding only (no class embedding here)
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.cond_proj = nn.Linear(t_embed_dim, embed_dim)

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

        # Head: project token to patch pixels
        self.head = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.patch_size = patch_size
        self.out_channels = out_channels

    def _unpatchify(self, tokens, Hp, Wp):
        B, N, PP_C = tokens.shape
        P = self.patch_size
        Cout = PP_C // (P * P)
        x = tokens.view(B, Hp, Wp, P, P, Cout).permute(0,5,1,3,2,4).contiguous()
        return x.view(B, Cout, Hp * P, Wp * P)

    def forward(self, x, t, y=None):
        """
        Args:
            x: (B,3,32,32)
            t: (B,1,1,1)
            y: ignored (kept for API compatibility)
        Returns:
            (B,3,32,32)
        """
        # --- Patchify ---
        tok, Hp, Wp = self.patch(x)              # (B,N,D)
        tok = tok + self.pos[:, :tok.size(1), :] # add pos emb

        # --- Condition (time only) ---
        t_emb = self.time_embedder(t)            # (B, t_dim)
        cond = self.cond_proj(t_emb)             # (B,D)
        tok = tok + cond.unsqueeze(1)

        # --- Transformer ---
        tok = self.encoder(tok)                  # (B,N,D)
        tok = self.head(tok)                     # (B,N,P*P*C)

        return self._unpatchify(tok, Hp, Wp)     # (B,C,H,W)


class XYFourierEncoder(nn.Module):
    def __init__(self, num_freqs=6):
        super().__init__()
        self.freqs = 2 ** torch.arange(num_freqs).float() * math.pi
        self.out_dim = 2 * 2 * num_freqs

    def forward(self, xy):
        # xy: (B, N, 2)
        freqs = self.freqs.to(xy.device)
        x_proj = xy[..., None] * freqs  # (B, N, 2, F)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).flatten(-2)


class UncondVoronoiTransformer(UnconditionalVectorField):
    def __init__(self, n_points=500, in_dim=5, out_dim = 5, embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0, t_embed_dim=40, use_posEmbed=False):
        super().__init__()

        self.pos = nn.Parameter(torch.zeros(1, n_points, embed_dim))
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.t_proj = nn.Linear(t_embed_dim, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim*mlp_ratio),
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Linear(embed_dim, out_dim)

        self.xy_encoder = XYFourierEncoder()
        self.use_posEmbed = use_posEmbed
        if not self.use_posEmbed:
            self.in_proj = nn.Linear(in_dim, embed_dim)
        else:
            self.in_proj = nn.Linear(3 + 24, embed_dim)

    def forward(self, x, t):
        """
        x: (B, N, 5)
        t: (B,1,1)
        """
        B, N, D = x.shape
        if self.use_posEmbed:
            xy_embed = self.xy_encoder(x[..., :2])        # (B, N, 2F)
            x_cat = torch.cat([xy_embed, x[..., 2:]], dim=-1)
            x = self.in_proj(x_cat)
        else:
            x = self.in_proj(x)                       # (B,N,E)
        x = x + self.pos[:, :N]
        t_emb = self.t_proj(self.time_embedder(t)) # (B,E)
        x = x + t_emb.unsqueeze(1)                # broadcast
        x = self.encoder(x)                       # (B,N,E)
        return self.out_proj(x)                   # (B,N,5)


class UncondUniGBNTransformer_PE(UnconditionalVectorField):
    def __init__(self, n_points=500, in_dim=5, out_dim=5, embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0, t_embed_dim=40):
        super().__init__()

        self.in_proj = nn.Linear(in_dim, embed_dim)

        # Key change: no longer use learnable positional encoding; register a constant-zero buffer (not trained)
        self.register_buffer("pos", torch.zeros(1, 1, embed_dim))

        self.time_embedder = FourierEncoder(t_embed_dim)
        self.t_proj = nn.Linear(t_embed_dim, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Linear(embed_dim, out_dim)

        self.xy_encoder = XYFourierEncoder()

    def forward(self, x, t):
        """
        x: (B, N, 5)
        t: (B, 1, 1)
        """
        # Per-point linear projection (permutation equivariant)
        x = self.in_proj(x)                        # (B, N, E)

        # Key change: only add a constant-zero pos (broadcast), introducing no ordering information
        x = x + self.pos                           # (B, N, E)

        # Time embedding applies the same bias to all points (equivariant)
        t_emb = self.t_proj(self.time_embedder(t)) # (B, E)
        x = x + t_emb.unsqueeze(1)                 # (B, N, E)

        # The Transformer without positional encoding stays permutation equivariant
        x = self.encoder(x)                        # (B, N, E)
        return self.out_proj(x)                    # (B, N, 5)



class UncondUniGBNTransformer(UnconditionalVectorField):
    def __init__(self, n_points=500, in_dim=5, out_dim = 5, embed_dim=128, 
                 depth=6, num_heads=8, mlp_ratio=4.0, t_embed_dim=40,
                use_ckpt_fwd: bool = False, force_no_math_sdp: bool = True,):
        super().__init__()
        self.use_ckpt_fwd = bool(use_ckpt_fwd)
        self.force_no_math_sdp = bool(force_no_math_sdp)

        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, n_points, embed_dim))
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.t_proj = nn.Linear(t_embed_dim, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim*mlp_ratio),
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Linear(embed_dim, out_dim)

        self.xy_encoder = XYFourierEncoder()

    def _run_encoder_layers(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder.layers:
            if self.training and self.use_ckpt_fwd:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        norm = getattr(self.encoder, "norm", None)
        if norm is not None:
            x = norm(x)
        return x


    def forward(self, x, t):
        """
        x: (B, N, 3)
        """
        B, N, D = x.shape
        # xy_embed = self.xy_encoder(x[..., :2])        # (B, N, 2F)
        # x_cat = torch.cat([x, xy_embed], dim=-1)
        # x = self.in_proj(x_cat)

        x = self.in_proj(x)                       # (B,N,E)
        x = x + self.pos[:, :N]
        t_emb = self.t_proj(self.time_embedder(t)) # (B,E)
        x = x + t_emb.unsqueeze(1)                # broadcast
        # x = self.encoder(x)                       # (B,N,E)

        if self.force_no_math_sdp and _HAS_SDPA_KERNEL:
            backends = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
            with sdpa_kernel(backends):
                x = self._run_encoder_layers(x)
        else:
            x = self._run_encoder_layers(x)

        return self.out_proj(x)                   # (B,N,3)


class UncondUniGBNTransformer_EqM(UnconditionalVectorField):
    def __init__(self, n_points=500, in_dim=5, out_dim = 5, 
                embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0,):
        super().__init__()

        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, n_points, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim*mlp_ratio),
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Linear(embed_dim, out_dim)

        self.xy_encoder = XYFourierEncoder()

    def forward(self, x, t):
        """
        x: (B, N, 5)
        t: (B,1,1)
        """
        B, N, D = x.shape
        # xy_embed = self.xy_encoder(x[..., :2])        # (B, N, 2F)
        # x_cat = torch.cat([x, xy_embed], dim=-1)
        # x = self.in_proj(x_cat)

        x = self.in_proj(x)                       # (B,N,E)
        x = x + self.pos[:, :N]
        x = self.encoder(x)                       # (B,N,E)
        return self.out_proj(x)                   # (B,N,5)


class UncondVoronoiTransformerFPS(UnconditionalVectorField):
    def __init__(self, n_points=500, in_dim=5, embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0, t_embed_dim=40,
                 # >>> New: FPS options <<<
                 use_fps: bool = True,
                 fps_ratio: float = 1.0,   # e.g. 0.5 keeps only half the points
                 fps_k: int = 8,           # number of KNN used in interpolation
                 fps_tau: float = 0.1      # soft-interpolation temperature (smaller is closer to nearest neighbor)
                 ):
        super().__init__()

        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, n_points, embed_dim))
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.t_proj = nn.Linear(t_embed_dim, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim*mlp_ratio),
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Linear(embed_dim, in_dim)

        self.xy_encoder = XYFourierEncoder()

        # FPS configuration
        self.use_fps  = use_fps
        self.fps_ratio = float(fps_ratio)
        self.fps_k     = int(fps_k)
        self.fps_tau   = float(fps_tau)

    @staticmethod
    def _knn_interp(full_xy: torch.Tensor, anchor_xy: torch.Tensor,
                    anchor_feat: torch.Tensor, k: int, tau: float) -> torch.Tensor:
        """
        full_xy:   (B,N,2)
        anchor_xy: (B,M,2)
        anchor_feat: (B,M,E)
        return: (B,N,E)
        """
        B, N, _ = full_xy.shape
        M = anchor_xy.shape[1]
        k = min(k, M)
        # distances (B,N,M)
        d = torch.cdist(full_xy, anchor_xy, p=2)
        dval, didx = torch.topk(d, k=k, dim=-1, largest=False)          # (B,N,k)
        b = torch.arange(B, device=full_xy.device)[:, None, None]
        neigh = anchor_feat[b, didx]                                     # (B,N,k,E)
        w = torch.softmax(-dval / max(tau, 1e-6), dim=-1).unsqueeze(-1)  # (B,N,k,1)
        return (neigh * w).sum(dim=-2)                                   # (B,N,E)

    def _fps_indices_torch_cluster(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Perform batched FPS with torch_cluster.fps, returning (B,M) indices (local indices relative to each batch).
        xy: (B,N,2)
        """
        B, N, _ = xy.shape
        device = xy.device
        # Compute a fixed M to ensure a consistent count across batches
        M = max(1, int(math.ceil(N * self.fps_ratio)))

        # Flatten + batch indices
        flat = xy.reshape(B * N, 2).contiguous()                 # (B*N,2)
        batch = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)

        # fps returns concatenated global indices
        idx_flat = fps(flat, batch, ratio=float(M) / float(N), random_start=False)  # (B*M,)
        # Split the global indices back into per-batch local indices
        out_idx = torch.empty(B, M, dtype=torch.long, device=device)
        for b in range(B):
            mask = (batch[idx_flat] == b)
            sel = idx_flat[mask]
            # Convert to local [0..N-1]
            sel = sel - b * N
            # Align to M: truncate if too many, repeat the last one if too few
            if sel.numel() >= M:
                out_idx[b] = sel[:M]
            else:
                pad = sel.new_full((M - sel.numel(),), int(sel[-1].item()) if sel.numel() > 0 else 0)
                out_idx[b] = torch.cat([sel, pad], dim=0)
        return out_idx  # (B,M)

    def forward(self, x, t):
        """
        x: (B, N, 5)  where x[..., :2] is (x,y)
        t: (B,1,1)
        """
        B, N, D = x.shape

        # --- Input projection + learnable position + time condition (keep original logic) ---
        h = self.in_proj(x)                        # (B,N,E)
        h = h + self.pos[:, :N]
        t_emb = self.t_proj(self.time_embedder(t)) # (B,E)
        h = h + t_emb.unsqueeze(1)                 # (B,N,E)

        if not self.use_fps or self.fps_ratio >= 0.999:
            # Original path
            h = self.encoder(h)                    # (B,N,E)
        else:
            # Run the encoder only on the subsample, then interpolate back
            xy = x[..., :2].contiguous()           # (B,N,2)

            # 1) FPS to pick M representative indices (same M guaranteed per batch)
            idx = self._fps_indices_torch_cluster(xy)  # (B,M)
            b = torch.arange(B, device=x.device)[:, None]

            # 2) Subset features / coordinates
            h_sub  = h[b, idx]                     # (B,M,E)
            xy_sub = xy[b, idx]                    # (B,M,2)

            # 3) Run the original TransformerEncoder only on the subsample
            h_sub = self.encoder(h_sub)            # (B,M,E)

            # 4) KNN soft interpolation to restore the subsample features back to N points
            h = self._knn_interp(xy, xy_sub, h_sub, k=self.fps_k, tau=self.fps_tau)  # (B,N,E)

        return self.out_proj(h)                    # (B,N,5)


def exists(x): return x is not None
def default(val, d): return val if exists(val) else d

def cache_fn(f):
    cache = None
    def cached_fn(*args, _cache=True, **kwargs):
        nonlocal cache
        if not _cache:
            return f(*args, **kwargs)
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None
    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            ctx = kwargs['context']
            kwargs.update(context=self.norm_context(ctx))
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, drop_path_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    """Attention matching the reference: to_q / to_kv linear + configurable (heads, dim_head); shared by Cross / Self"""
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q  = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            max_neg = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg)

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.drop_path(self.to_out(out))

class PointEmbed(nn.Module):
    """Matching the reference: 3D frequency basis (2^k * pi), apply sin/cos to (x,y,z), concat the raw coordinates, then MLP to dim"""
    def __init__(self, hidden_dim=48, dim=512):
        super().__init__()
        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e, torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6), e]),
        ])  # (3, hidden_dim//2)
        self.register_buffer('basis', e)  # 3 x (hidden_dim//2)
        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        # input: (B,N,3) ; basis: (3, F)
        proj = torch.einsum('bnd,de->bne', input, basis)    # (B,N,F)
        emb  = torch.cat([proj.sin(), proj.cos()], dim=2)   # (B,N,2F) ; 2F = hidden_dim
        return emb

    def forward(self, xyz):  # (B,N,3)
        emb = self.embed(xyz, self.basis)
        return self.mlp(torch.cat([emb, xyz], dim=2))  # (B,N,dim)

class DiagonalGaussianDistribution:
    """Diagonal Gaussian + KL matching the reference"""
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            z = torch.zeros_like(self.mean, device=self.mean.device)
            self.std = z; self.var = z

    def sample(self):
        eps = torch.randn_like(self.std)
        return self.mean + self.std * eps

    def kl(self, other=None):
        if self.deterministic:
            return torch.zeros((), device=self.mean.device)
        if other is None:
            kl = 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar)      # (..., C0)
            # Matching the reference: average over (M,C0); the caller may then average over the batch
            return kl.mean(dim=[-1, -2])  # (B,)
        else:
            # KL between distributions, unused; interface kept for compatibility
            kl = 0.5 * ((self.mean - other.mean).pow(2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar)
            return kl.mean(dim=[-1, -2, -3])

# ===================== Adapted version: Flow (B,N,5) → (x,y,0) → AE/KL-AE =====================

def xy_to_xyz(x5: torch.Tensor) -> torch.Tensor:
    """(B,N,5) -> (B,N,3): take (x,y) + z=0, used for PointEmbed/FPS"""
    xy = x5[..., :2]
    z  = torch.zeros_like(xy[..., :1])
    return torch.cat([xy, z], dim=-1)

class VoronoiSetKLAE_FPS(nn.Module):
    """
    Fully equivalent to the reference (structure / hyperparameters / implementation), except:
      - the input becomes x5(B,N,5), internally converted to xyz=(x,y,0)
      - the output becomes a 5-dim per-point vector field
    Everything else stays the same: FPS + CrossAttn encoding, latent SelfAttn (PreNorm+DropPath+GEGLU), CrossAttn decoding, KL compression.
    """
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=5,         # changed to 5
        num_inputs=2048,      # expected N (used for assertions)
        num_latents=512,      # M
        latent_dim=32,        # C0
        heads=8,              # latent self-attn heads
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False
    ):
        super().__init__()
        self.depth = depth
        self.num_inputs  = num_inputs
        self.num_latents = num_latents

        # Encoding CrossAttn (heads=1, dim_head=dim, same as the reference)
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads=1, dim_head=dim)),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)  # for (x,y,0)

        # latent self-attn: if weight tying is enabled, use cache_fn, matching the reference
        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1))
        get_latent_ff   = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))
        cache_args = {'_cache': weight_tie_layers}

        self.layers = nn.ModuleList([
            nn.ModuleList([get_latent_attn(**cache_args), get_latent_ff(**cache_args)])
            for _ in range(depth)
        ])

        # Decoding CrossAttn (heads=1, dim_head=dim, same as the reference)
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads=1, dim_head=dim))
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_outputs = nn.Linear(queries_dim, output_dim)

        # KL compression: dim -> latent_dim(C0) sample -> proj back to dim
        self.proj = nn.Linear(latent_dim, dim)
        self.mean_fc   = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    # ------- Encoding (FPS + CrossAttn) -------
    def encode(self, x5: torch.Tensor):
        """
        x5: (B,N,5); only (x,y) is used for geometry, with z=0 appended
        Returns: kl(B,) and the sampled latent (B,M,latent_dim)
        """
        B, N, _ = x5.shape
        assert N == self.num_inputs, f"N={N} does not match num_inputs={self.num_inputs}"

        pc = xy_to_xyz(x5)                       # (B,N,3)

        # torch_cluster.fps requires a flattened tensor + batch indices
        flattened = pc.reshape(B * N, 3)         # (B*N, 3)
        batch = torch.arange(B, device=pc.device).repeat_interleave(N)  # (B*N,)
        ratio = float(self.num_latents) / float(self.num_inputs)
        idx = fps(flattened, batch, ratio=ratio) # (~ B*M,)

        # Sampled anchors (restored per batch)
        sampled_pc = flattened[idx].view(B, -1, 3)  # (B,M,3)

        # Positional encoding to dim
        sampled_emb = self.point_embed(sampled_pc)   # (B,M,dim)
        pc_emb      = self.point_embed(pc)           # (B,N,dim)

        # CrossAttn(Q=PosEmb(X0), K=V=PosEmb(X))
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(sampled_emb, context=pc_emb) + sampled_emb
        x = cross_ff(x) + x                          # (B,M,dim)

        # KL: generate mean/logvar for each latent vector and sample down to the low-dim C0
        mean   = self.mean_fc(x)                     # (B,M,C0)
        logvar = self.logvar_fc(x)                   # (B,M,C0)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        z = posterior.sample()                       # (B,M,C0)
        kl = posterior.kl()                          # (B,)

        return kl, z

    # ------- Decoding (latent SelfAttn → CrossAttn(queries, latent)) -------
    def decode(self, z: torch.Tensor, queries_xyz: torch.Tensor):
        """
        z: (B,M,C0)  -> proj to dim, then stack L layers of SelfAttn
        queries_xyz: (B,N,3) as decoding queries (per-point output)
        """
        x = self.proj(z)                             # (B,M,dim)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x                       # (B,M,dim)

        queries_emb = self.point_embed(queries_xyz)  # (B,N,queries_dim=dim)
        latents = self.decoder_cross_attn(queries_emb, context=x)  # (B,N,dim)

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        return self.to_outputs(latents)              # (B,N,5)

    # ------- Forward -------
    def forward(self, x5: torch.Tensor, t=None):
        """
        x5: (B,N,5)  -> y: (B,N,5) , kl: (B,)  (same style as the reference; the trainer side weights it and averages over the batch)
        """
        kl, z = self.encode(x5)                      # (B,), (B,M,C0)
        queries = xy_to_xyz(x5)                      # (B,N,3)
        y = self.decode(z, queries)                  # (B,N,5)
        return y, kl

# ============ Factory functions (same style as the create_* you provided, only output_dim=5) ============

def flow_kl_d512_m512_l32(N=2048):
    return VoronoiSetKLAE_FPS(depth=24, dim=512, queries_dim=512,
                              output_dim=5, num_inputs=N,
                              num_latents=512, latent_dim=32,
                              heads=8, dim_head=64, weight_tie_layers=False, decoder_ff=False)

def flow_kl_d512_m256_l32(N=2048):
    return VoronoiSetKLAE_FPS(depth=24, dim=512, queries_dim=512,
                              output_dim=5, num_inputs=N,
                              num_latents=256, latent_dim=32,
                              heads=8, dim_head=64)

def flow_kl_d256_m256_l32(N=2048):
    return VoronoiSetKLAE_FPS(depth=24, dim=256, queries_dim=256,
                              output_dim=5, num_inputs=N,
                              num_latents=256, latent_dim=32,
                              heads=8, dim_head=64)





# -*- coding: utf-8 -*-
"""
Permutation-equivariant transformer for point sets on a 2D torus.
- Inputs:  x ∈ [-1, 1]^2, shape (B, N, 2)   (period length L = 2.0 by default)
- Outputs: per-point vectors, shape (B, N, out_dim)

Key properties:
1) Permutation equivariant: F(Πx) = ΠF(x).  (no absolute/positional ids)
2) Periodic boundary (toroidal) support via minimal-image relative displacement.
3) Relative-geometry aware: multi-head attention logits + messages are biased by
   an MLP over Fourier features of pairwise relative displacements Δx_ij.
4) Optional global conditioning (e.g., time t) via broadcast embedding (does not break equivariance).

Usage:
    model = EquivariantTorusTransformer(in_dim=2, out_dim=2, embed_dim=128, depth=6, num_heads=8)
    y = model(x)  # x in [-1,1], shape (B,N,2), y in (B,N,2)

Equivariance check:
    with torch.no_grad():
        B,N = 2, 256
        x = torch.rand(B, N, 2)*2-1
        y1 = model(x)
        perm = torch.randperm(N)
        y2 = model(x[:,perm,:])
        err = (y1[:,perm,:]-y2).abs().max().item()
        print('max equivariance err:', err)

Notes:
- Complexity is O(N^2) per layer due to full attention; for large N consider
  windowed/landmark attention or block-sparse variants.
- If your data is in [0,1), either rescale to [-1,1] before feeding the model
  or set periodic_L=1.0 and disable the internal rescale flags.
"""



def pairwise_delta(xy: torch.Tensor, periodic_L: float = 2.0) -> torch.Tensor:
    """Compute pairwise relative displacement Δx_ij under periodic boundary.
    xy: (B, N, 2) in the same unit as 'periodic_L' (default [-1,1] → L=2)
    return: (B, N, N, 2) in [-L/2, L/2)
    """
    B, N, _ = xy.shape
    dx = xy[:, :, None, :] - xy[:, None, :, :]  # (B,N,N,2)
    dx[..., 0] = minimal_image(dx[..., 0], periodic_L)
    dx[..., 1] = minimal_image(dx[..., 1], periodic_L)
    return dx


# ------------------------------
# Relative positional bias module
# ------------------------------
class RBFRelBias(nn.Module):
    """
    2D RBF relative bias: bias_ij^h = sum_k W[h,k] * exp(-||dx_ij||^2 / (2 sigma_k^2))
    Memory usage is only (B,K,N,N) + (B,H,N,N)
    """
    def __init__(self, num_heads: int, num_k: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.num_k = num_k
        # log σ initialized to uniformly cover the scales (adjust as needed)
        self.log_sigma = nn.Parameter(torch.linspace(-2.0, 0.5, num_k))
        self.weight = nn.Parameter(torch.zeros(num_heads, num_k))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, dx: torch.Tensor) -> torch.Tensor:
        # dx: (B,N,N,2)  -> dist2: (B,N,N)
        dist2 = (dx ** 2).sum(dim=-1)
        sig2 = (self.log_sigma.exp() ** 2).clamp_min(1e-6)  # (K,)
        # bases: (B,K,N,N)
        bases = torch.stack([(-0.5 / s2) * dist2 for s2 in sig2], dim=1).exp()
        # (B,H,N,N) = (B,K,N,N) × (H,K)
        bias = torch.einsum('bknm,hk->bhmn', bases, self.weight)
        return bias


# -----------------------------------
# Equivariant self-attention block
# -----------------------------------
class EquivariantAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 128, num_heads: int = 8, mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0,
                 fourier_dims: int = 16, rel_hidden: int = 64, periodic_L: float = 2.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.H = num_heads
        self.periodic_L = periodic_L
        self.d = d_model // num_heads

        self.q = nn.Linear(d_model, d_model, bias=True)
        self.k = nn.Linear(d_model, d_model, bias=True)
        self.v = nn.Linear(d_model, d_model, bias=True)
        self.o = nn.Linear(d_model, d_model, bias=True)

        self.rel_bias = RBFRelBias(num_heads, num_k=4)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """x:  (B,N,E)
           dx: (B,N,N,2) minimal-image displacements for this batch
        Return: (B,N,E)
        """
        B, N, E = x.shape
        dx = pairwise_delta(xy, periodic_L=self.periodic_L).detach()
        h = self.norm1(x)
        q = self.q(h).view(B, N, self.H, self.d).transpose(1, 2)  # (B,H,N,d)
        k = self.k(h).view(B, N, self.H, self.d).transpose(1, 2)
        v = self.v(h).view(B, N, self.H, self.d).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)     # (B,H,N,N)
        logits = logits + self.rel_bias(dx)                        # add relative bias
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ v                                              # (B,H,N,d)
        y = y.transpose(1, 2).contiguous().view(B, N, E)
        y = self.proj_dropout(self.o(y))
        x = x + y

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

class TorusRelBiasEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model=128, nhead=8, mlp_ratio=4.0, periodic_L=2.0, num_k=4, **kw):
        super().__init__(d_model=d_model, nhead=nhead, batch_first=True,
                         dim_feedforward=int(d_model*mlp_ratio), **kw)
        self.periodic_L = periodic_L
        self.rel_bias = RBFRelBias(nhead, num_k=num_k)  # still uses your RBF version

    def forward(self, src, xy, src_key_padding_mask=None):
        # xy: (B,N,2) used for the geometric bias; src: (B,N,E)
        B, N, _ = src.shape
        # Computed within the layer for the shortest lifetime; detach since we don't need to backprop to coordinates through the bias path
        dx = pairwise_delta(xy, periodic_L=self.periodic_L).detach()
        # The bias branch can use half precision to save memory
        bias = self.rel_bias(dx)                     # (B,H,N,N)
        attn_bias = bias.to(src.dtype)
        attn_bias = attn_bias.reshape(B * self.self_attn.num_heads, N, N).contiguous()


        # Call the parent implementation, passing mask (internally it maps src_mask -> self_attn(attn_mask=...))
        return super().forward(src, src_mask=attn_bias, src_key_padding_mask=src_key_padding_mask)


# --------------------------------------------------
# Main model: permutation-equivariant torus transformer
# --------------------------------------------------
class EquivariantTorusTransformer(nn.Module):
    def __init__(self,
                 in_dim: int = 2,
                 out_dim: int = 2,
                 embed_dim: int = 128,
                 depth: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 periodic_L: float = 2.0,          # L=2.0 ↔ inputs in [-1,1]
                 fourier_dims: int = 16,
                 rel_hidden: int = 64,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 t_embed_dim: Optional[int] = None # optional global conditioning
                 ):
        super().__init__()
        self.periodic_L = float(periodic_L)

        # Per-point input projection (permutation-equivariant because it is row-wise)
        self.in_proj = nn.Linear(in_dim, embed_dim)

        # Optional global/broadcast conditioning (e.g., time t)
        if t_embed_dim is not None:
            self.t_proj = nn.Sequential(
                FourierEncoder(t_embed_dim),
                nn.Linear(t_embed_dim, embed_dim)
            )
        else:
            self.t_proj = None

        self.blocks = nn.ModuleList([
            TorusRelBiasEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                    mlp_ratio=mlp_ratio, periodic_L=self.periodic_L, num_k=4,
                                    dropout=attn_dropout, activation='gelu')
            for _ in range(depth)
        ])
        self.out_proj = nn.Linear(embed_dim, out_dim)

        # Small init for stability (optional)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B,N,in_dim), coordinates in [-1,1] if periodic_L=2.0
           t_embed: (B, t_embed_dim) optional global conditioning
        return: (B,N,out_dim)
        """
        B, N, D = x.shape
        h = self.in_proj(x)  # (B,N,E)

        if self.t_proj is not None:
            assert t_embed is not None, "t_embed_dim was set but no t_embed passed"
            tb = self.t_proj(t_embed).unsqueeze(1)  # (B,1,E)
            h = h + tb  # broadcast; does not break permutation equivariance

        # Compute pairwise minimal-image displacements once per forward
        # dx = pairwise_delta(x, periodic_L=self.periodic_L)  # (B,N,N,2)

        for blk in self.blocks:
            h = blk(h, x)

        y = self.out_proj(h)
        return y

class PeriodicFourier(nn.Module):
    def __init__(self, K=8, L=2.0):
        super().__init__()
        self.L = L
        # Use fixed frequencies (or logspace) to avoid learnable params breaking the periodic alignment
        freqs = torch.logspace(0, math.log2(64.0), steps=K, base=2.0)  # example
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, xy):  # (B,N,2) in [-L/2,L/2)
        # 2π f x / L
        ang = (2*math.pi * xy[..., None, :] * (self.freqs[None,None,:,None] / self.L))  # (B,N,K,2)
        s = torch.sin(ang); c = torch.cos(ang)
        # Concatenate into per-point features: [(sin_x,cos_x),(sin_y,cos_y)] over K
        return torch.cat([s[...,0], c[...,0], s[...,1], c[...,1]], dim=-1)  # (B,N, 4K)

class EquivariantTorusTransformerLite(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, embed_dim=128, depth=6, num_heads=8,
                 mlp_ratio=4.0, L=2.0, K=8, t_embed_dim=None, dropout=0.0):
        super().__init__()
        self.L = L
        self.per_feat = PeriodicFourier(K=K, L=L)   # unpaired periodic features
        per_dim = 4*K
        self.in_proj = nn.Linear(in_dim + per_dim, embed_dim)

        self.t_proj = None
        if t_embed_dim is not None:
            self.t_proj = nn.Sequential(FourierEncoder(t_embed_dim),
                                        nn.Linear(t_embed_dim, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True,
            dim_feedforward=int(embed_dim*mlp_ratio),
            activation="gelu", dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x, t=None):  # x: (B,N,2) in [-L/2,L/2)
        per = self.per_feat(x)                      # (B,N,4K)
        h = self.in_proj(torch.cat([x, per], dim=-1))
        if self.t_proj is not None:
            h = h + self.t_proj(t).unsqueeze(1)     # broadcast; does not break equivariance
        h = self.encoder(h)                         # no relbias; uses SDPA
        return self.out_proj(h)




class RFFEncoder(nn.Module):
    """
    Non-periodic random Fourier features for 2D coords.
    x ∈ R^2  ->  phi(x) ∈ R^{2*F}
    z = x @ W  + b,  W ~ N(0, sigma^{-2} I), b ~ U[0, 2π)
    feat = [sin(z), cos(z)]
    """
    def __init__(self, num_freq: int = 32, sigma: float = 10.0, learnable: bool = False):
        super().__init__()
        self.F = num_freq
        W = torch.randn(2, num_freq) / sigma           # (2, F)
        b = torch.rand(num_freq) * (2 * math.pi)       # (F,)
        if learnable:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("W", W, persistent=False)
            self.register_buffer("b", b, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, 2) in Euclidean space (no wrapping)
        return: (B, N, 2F)
        """
        z = x @ self.W + self.b          # (B, N, F)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)


# --------- Lite equivariant transformer (no rel-bias, no torus) ---------
class EquivariantEuclidTransformerLite(nn.Module):
    """
    Permutation-equivariant transformer for point sets in R^2 (non-periodic).
    - No pairwise relative bias / dx
    - Per-point RFF features to help model spatial patterns
    - Uses nn.TransformerEncoder (SDPA/FlashAttention under the hood)
    """
    def __init__(self,
                 in_dim: int = 2,
                 out_dim: int = 2,
                 embed_dim: int = 128,
                 depth: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 rff_freqs: int = 32,
                 rff_sigma: float = 10.0,
                 rff_learnable: bool = False,
                 center_per_set: bool = False,   # optional: center each point set to enhance translation invariance
                 t_embed_dim: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        self.center_per_set = center_per_set

        self.rff = RFFEncoder(num_freq=rff_freqs, sigma=rff_sigma, learnable=rff_learnable)
        per_dim = 2 * rff_freqs
        self.in_proj = nn.Linear(in_dim + per_dim, embed_dim)

        self.t_proj = None
        if t_embed_dim is not None:
            # Reuse your existing FourierEncoder(t_embed_dim)
            self.t_proj = nn.Sequential(
                FourierEncoder(t_embed_dim),
                nn.Linear(t_embed_dim, embed_dim)
            )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            dropout=dropout,
            norm_first=True,   # recommended to enable; usually more stable
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.out_proj = nn.Linear(embed_dim, out_dim)

        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, 2)   (no wrapping; any real-valued coordinates work, recommended to pre-normalize to ~[-1,1])
        t: optional scalar conditioning; supports (B,), (B,1) or (B,1,1)
        return: (B, N, out_dim)
        """
        if self.center_per_set:
            x_in = x - x.mean(dim=1, keepdim=True)   # does not change permutation equivariance
        else:
            x_in = x

        per = self.rff(x_in)                         # (B, N, 2F)
        h = self.in_proj(torch.cat([x_in, per], dim=-1))  # (B, N, E)

        if self.t_proj is not None:
            tb = self.t_proj(t).unsqueeze(1)         # (B, 1, E)  broadcast to every point
            h = h + tb

        h = self.encoder(h)                          # SDPA/FlashAttention, no rel-bias
        return self.out_proj(h)
    
# ----------------------
# Optional: unit tests
# ----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 2, 128
    x = torch.rand(B, N, 2) * 2 - 1  # [-1,1]
    model = EquivariantTorusTransformer(in_dim=2, out_dim=2, embed_dim=64, depth=3, num_heads=4)
    with torch.no_grad():
        y1 = model(x)
        perm = torch.randperm(N)
        y2 = model(x[:, perm, :])
        err = (y1[:, perm, :] - y2).abs().max().item()
        print("max |perm-equivariance error| =", err)

def periodic_embed_box(x: torch.Tensor, a: float = -1.0, b: float = 1.0) -> torch.Tensor:
    """
    x: (..., C)  coordinates of periodic dimensions; here C=2 (x,y)
    The interval is [a,b), with period length L=b-a. Returns the (cos, sin) embedding of shape (..., 2C).
    """
    L = (b - a)
    theta = 2 * torch.pi * (x - a) / L
    return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)  # (..., 2C)


# from torch_cluster import fps
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        hidden = int(embed_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv):
        # cross-attention: Q = q (sub_X), K/V = kv (full X)
        attn_out, _ = self.attn(q, kv, kv)   # (B, M, E)
        q = q + self.drop(attn_out)
        q = self.norm1(q)

        ff_out = self.ff(q)
        q = q + self.drop(ff_out)
        q = self.norm2(q)
        return q

class CrossAttentionBlockSDPA(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, attn_dropout=0.0, resid_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate Q/K/V projections, convenient for caching or precomputing kv
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden = int(embed_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.attn_dropout = attn_dropout
        self.resid_drop = nn.Dropout(resid_dropout)

    def _shape(self, x):  # (B, T, E) -> (B, H, T, Hd)
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, q, kv, *, kv_proj=None, attn_mask=None):
        """
        q:  (B, Tq, E)
        kv: (B, Tk, E) raw sequence (if kv_proj=None, K/V projection is done within this layer)
        kv_proj: optional, a (k, v) tuple with shapes (B,H,Tk,Hd) respectively
        attn_mask: optional, (B, 1, Tq, Tk) or (Tq, Tk), compatible with F.sdpa
        """
        # PreNorm is more stable
        q = self.norm1(q)

        # ---- Q/K/V projection (supports externally provided pre-projected kv) ----
        Q = self._shape(self.q_proj(q))  # (B,H,Tq,Hd)
        if kv_proj is None:
            K = self._shape(self.k_proj(kv))  # (B,H,Tk,Hd)
            V = self._shape(self.v_proj(kv))  # (B,H,Tk,Hd)
        else:
            K, V = kv_proj  # already (B,H,Tk,Hd)

        # ---- Use native SDPA, automatically choosing the Flash/efficient kernel ----
        # During training dropout_p>0 may be used; set to 0 for inference
        # Note: SDPA expects the shape (B,H,T,Hd)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,       # may be None
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False
            )  # (B,H,Tq,Hd)

        # Merge heads
        B, H, Tq, Hd = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Tq, H * Hd)
        attn_out = self.out_proj(attn_out)

        # Residual
        x = q + self.resid_drop(attn_out)

        # FFN + residual
        x = x + self.resid_drop(self.ff(self.norm2(x)))
        return x

    @torch.no_grad()
    def preproject_kv(self, kv):
        """Pre-project and cache kv, returning (K,V) with shape (B,H,Tk,Hd)."""
        K = self._shape(self.k_proj(kv))
        V = self._shape(self.v_proj(kv))
        return K, V

class UncondUniGBNTransformerFPS(UnconditionalVectorField):
    def __init__(
        self,
        n_points=500,
        in_dim=5,
        out_dim=5,
        embed_dim=128,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        t_embed_dim=40,
        num_latents=128,
        use_periodic: bool = False,
        random_start: bool = False,
        box_min: float = -1.0,
        box_max: float =  1.0,
        direct_stride: bool = False,
        ascending_fps: bool = False
    ):
        super().__init__()

        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, n_points, embed_dim))
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.t_proj = nn.Linear(t_embed_dim, embed_dim)

        if num_latents is None:
            num_latents = max(128, n_points // 4)
        self.num_inputs = n_points
        self.num_latents = num_latents

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlockSDPA(embed_dim, num_heads, mlp_ratio=mlp_ratio)#, dropout=0.1)
            for _ in range(depth)
        ])

        self.final_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            dropout=0.1,
        )

        self.out_proj = nn.Linear(embed_dim, out_dim)

        self.xy_encoder = XYFourierEncoder()  

        self.use_periodic = use_periodic
        self.random_start = random_start
        self.box_min = float(box_min)
        self.box_max = float(box_max)
        self.direct_stride = bool(direct_stride)
        self.stride = int(n_points // num_latents)

        self.ascending_fps = ascending_fps

        if self.direct_stride:
            print(f"Use direct stride with stride: {self.stride}")


    def forward(self, x, t):
        """
        x: (B, N, 2)
        t: (B,1,1)
        Returns: (B, N, 2)
        """
        B, N, D = x.shape
        assert N <= self.num_inputs, "N cannot exceed the n_points set at init (the length of pos)"

        h = self.in_proj(x)                     # (B, N, E)
        h = h + self.pos[:, :N]                 # positional encoding

        t_emb = self.time_embedder(t)           # (B, t_embed_dim)
        t_emb = self.t_proj(t_emb)              # (B, E)
        h = h + t_emb.unsqueeze(1)              # time encoding added to all points

        coord_dim = min(3, D)
        coords = x[..., :coord_dim]             # (B, N, C)


        if self.direct_stride:
            # --- Direct strided sampling: each batch takes [0, stride, 2*stride, ...]
            base_idx = torch.arange(0, N, self.stride, device=x.device)  # (M,)
            if base_idx.numel() == 0:
                base_idx = torch.tensor([0], device=x.device)
            M = base_idx.numel()

            # Flatten into (B*M,) global indices
            offsets = (torch.arange(B, device=x.device) * N).unsqueeze(1)  # (B,1)
            idx = (offsets + base_idx.unsqueeze(0)).reshape(-1)            # (B*M,)
        else:
            if self.use_periodic:
                coords_emb = periodic_embed_box(coords, a=self.box_min, b=self.box_max)  # (B, N, 4)
                pos = coords_emb.reshape(B * N, -1)                                      # (B*N, 4)
            else:
                pos = coords.reshape(B * N, coord_dim)                                          # (B*N, 2)

            batch = torch.arange(B, device=x.device).repeat_interleave(N)

            ratio = float(self.num_latents) / float(N)
            idx = fps(pos, batch=batch, ratio=ratio, random_start=self.random_start)   # (B * M,)
            M = self.num_latents

        h_flat = h.reshape(B * N, -1)           # (B*N, E)
        sub = h_flat[idx].view(B, M, -1)        # (B, M, E)

        if self.ascending_fps and not self.direct_stride:
            idx_view = idx.view(B, M)                         # (B, M)
            perm = torch.argsort(idx_view, dim=1)             # (B, M)  each row is the ascending permutation
            # idx_view = torch.gather(idx_view, 1, perm)        # ascending idx (global)
            sub = torch.gather(sub, 1, perm.unsqueeze(-1).expand(-1, -1, sub.size(-1)))  # reorder the subset tokens


        # ====== 2) Multi-layer cross-attention: sub_X <- full X ======
        for block in self.cross_blocks:
            sub = block(sub, h)                 # Q=sub, KV=h

        # ====== 3) Broadcast latent information back to all points: X <- sub_X ======
        # Let all N points attend to the M latents
        h_updated, _ = self.final_attn(h, sub, sub)   # Q=h, KV=sub   -> (B, N, E)

        # ====== 4) Output projection, keep (B, N, 5) ======
        out = self.out_proj(h_updated)
        return out
    
class PointEmbed(nn.Module):
    """
    Input (B, N, 2) or (B, N, 3); if 2D, z=0 is appended automatically
    Frequency-domain positional encoding + linear projection -> (B, N, dim)
    """
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()
        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim

        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])  # (3, hidden_dim//2)
        self.register_buffer('basis', e)
        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(inp, basis):
        # inp: (B, N, 3), basis: (3, hidden_dim//2)
        proj = torch.einsum('bnd,de->bne', inp, basis)   # (B, N, hidden_dim//2)
        emb  = torch.cat([proj.sin(), proj.cos()], dim=2) # (B, N, hidden_dim)
        return emb

    def forward(self, inp):
        if inp.size(-1) == 2:
            zeros = torch.zeros_like(inp[..., :1])
            inp = torch.cat([inp, zeros], dim=-1)  # -> (B, N, 3)
        emb = self.embed(inp, self.basis)
        return self.mlp(torch.cat([emb, inp], dim=2))


class FourierEncoder_fixed(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, t):
        # t: (B,1) or (B,1,1)
        t = t.view(t.size(0), -1)                               # (B,1)
        freqs = torch.arange(self.embed_dim // 2, device=t.device, dtype=t.dtype)
        ang = t * (2.0 ** freqs)                                # (B, embed_dim//2)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(emb)                                   # (B, embed_dim)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class BroadcastBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ff   = FeedForward(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, H, sub):
        # Q = H (N points), KV = sub (M latents)
        q  = self.norm_q(H)
        kv = self.norm_kv(sub)
        y, _ = self.attn(q, kv, kv)
        H = H + y
        H = H + self.ff(H)
        return H

class SubSelfBlock(nn.Module):
    """Pre-Norm Self-Attention + FFN for the latent set (B, M, E)."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm_qkv = nn.LayerNorm(embed_dim)
        self.sa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ff = FeedForward(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, sub):        # sub: (B, M, E)
        x = self.norm_qkv(sub)
        y, _ = self.sa(x, x, x)    # self-attn on sub
        sub = sub + y
        sub = sub + self.ff(sub)
        return sub


# -------------------- Main Model --------------------

class UncondUniGBNTransformerFPS2(nn.Module):
    """
    encode(pc): FPS sampling -> shared point_embed -> separate pos for sub/full ->
                CrossAttentionBlockSDPA(sub <- full) + FF residual -> bottleneck.pre(x)
    forward(x, t): (optional) time condition -> broadcast latent back to N points -> out_proj
    """
    def __init__(
        self,
        n_points=500,
        in_dim=2,
        out_dim=2,
        embed_dim=128,
        t_embed_dim=40,
        num_latents=128,
        num_heads=8,
        mlp_ratio=4.0,
        random_start: bool = False,
        ascending_fps: bool = False,
        point_embed_hidden: int = 48,
        attn_dropout: float = 0.0,   # can be ignored if your CrossAttentionBlockSDPA does not use it
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        assert in_dim == 2, "This model assumes 2D point inputs (x, y)"
        self.num_inputs = n_points
        self.num_latents = num_latents
        self.random_start = random_start
        self.ascending_fps = ascending_fps
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        # Shared point embedding (used by both full and sub)
        self.point_embed = PointEmbed(hidden_dim=point_embed_hidden, dim=embed_dim)

        # Absolute positional encoding for full; sub's pos is sliced from here by the sampling indices
        self.pos_full = nn.Parameter(torch.zeros(1, n_points, embed_dim))

        # A "block" composed of cross-attention + FF, matching the style of your encode
        self.cross_attend_blocks = nn.ModuleList([
            CrossAttentionBlockSDPA(embed_dim, num_heads, mlp_ratio=mlp_ratio),
            FeedForward(embed_dim, mlp_ratio=mlp_ratio, dropout=ff_dropout),
        ])

        # bottleneck (provides a .pre interface in your style; defaults to Identity, replaceable)
        class _Bottleneck(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.pre = nn.Identity()
        self.bottleneck = _Bottleneck(embed_dim)

        # (Optional) time condition: used in forward
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.t_proj = nn.Linear(t_embed_dim, embed_dim)

        # how many self-attn layers on sub before broadcasting
        self.sub_self_depth = 2              # recommended 1~2 layers; start with 1
        self.sub_self_blocks = nn.ModuleList([
            SubSelfBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=0.0)
            for _ in range(self.sub_self_depth)
        ])


        self.final_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            dropout=0.1,
        )
        
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def _fps_subsample_with_idx(self, pc):
        """
        pc: (B, N, 2)
        returns:
        sampled_pc: (B, M, 2)         # M is the actual sample count (dynamic)
        idx_bm_local: (B, M)          # within-batch local indices, range [0, N)
        """
        B, N, D = pc.shape
        assert N == self.num_inputs, f"N({N}) must equal the initialized n_points({self.num_inputs})"

        ratio = float(self.num_latents) / float(N)
        flat  = pc.reshape(B * N, D)                             # (B*N, 2)
        batch = torch.arange(B, device=pc.device).repeat_interleave(N)

        idx = fps(flat, batch=batch, ratio=ratio, random_start=self.random_start)  # (B*M',) global indices
        M = idx.numel() // B
        assert idx.numel() == B * M and M >= 1

        # Treat as (B, M) global indices
        idx_bm = idx.view(B, M)                                  # (B, M)

        # Convert to within-batch local indices [0..N-1]
        base = (torch.arange(B, device=pc.device) * N).view(B, 1)   # (B,1)
        idx_bm_local = idx_bm - base                                  # (B, M)

        # Sample points are still taken from flat using global indices, then reshaped back to (B, M, D)
        sampled_pc = flat[idx].view(B, M, D)                       # (B, M, 2)

        if self.ascending_fps:
            perm = torch.argsort(idx_bm_local, dim=1)
            idx_bm_local = torch.gather(idx_bm_local, 1, perm)
            sampled_pc = torch.gather(sampled_pc, 1, perm.unsqueeze(-1).expand(-1, -1, D))

        # Strict validation
        assert idx_bm_local.dtype == torch.long
        assert int(idx_bm_local.min()) >= 0
        assert int(idx_bm_local.max()) < N, f"idx_bm_local.max()={int(idx_bm_local.max())} >= N={N}"

        return sampled_pc, idx_bm_local



    def encode(self, pc):
        """
        pc: (B, N, 2)
        Flow: FPS query (dynamic M) -> point_embed for both paths -> separate pos for full/sub (sub sliced from full) -> CrossAttentionBlockSDPA + FF -> bottleneck
        """
        B, N, _ = pc.shape
        assert N == self.num_inputs, f"N({N}) must equal the initialized n_points({self.num_inputs})"

        # 1) Sampling: dynamic M
        sampled_pc, idx_bm = self._fps_subsample_with_idx(pc)   # (B, M, 2), (B, M)
        M = idx_bm.size(1)

        # 2) Embedding
        x = self.point_embed(sampled_pc)                         # (B, M, E)
        H = self.point_embed(pc)                                 # (B, N, E)

        # 3) Positional encoding
        E = H.size(-1)
        assert E == self.embed_dim, f"embed_dim mismatch: H.last_dim={E}, self.embed_dim={self.embed_dim}"
        pos_full_N  = self.pos_full[:, :N]                       # (1, N, E)
        H = H + pos_full_N

        pos_full_BNE = pos_full_N.expand(B, -1, -1)              # (B, N, E)
        # -- Validate again to avoid out-of-bounds gather --
        assert int(idx_bm.max()) < N and int(idx_bm.min()) >= 0
        sub_pos = torch.gather(
            pos_full_BNE, 1,
            idx_bm.unsqueeze(-1).expand(-1, -1, E)
        )                                                        # (B, M, E)
        x = x + sub_pos

        # 4) SDPA + FF residual
        cross_attn, cross_ff = self.cross_attend_blocks
        # -- Check for NaN/Inf before feeding into LN/SDPA --
        if not torch.isfinite(x).all() or not torch.isfinite(H).all():
            bad = "x" if not torch.isfinite(x).all() else "H"
            ix  = torch.nonzero(~torch.isfinite(x if bad=="x" else H), as_tuple=False)
            raise RuntimeError(f"[NaN/Inf detected] in {bad} at {ix[:5]} (showing first few)")
        # Shape/dimension checks to avoid LN normalized_shape mismatch
        assert x.dim() == 3 and H.dim() == 3, f"rank should be 3: x={tuple(x.shape)}, H={tuple(H.shape)}"
        assert x.size(-1) == self.embed_dim and H.size(-1) == self.embed_dim, \
            f"LN shape mismatch: x_last={x.size(-1)}, H_last={H.size(-1)}, embed_dim={self.embed_dim}"

        x = cross_attn(x, H) + x
        x = cross_ff(x) + x

        # 5) bottleneck
        return self.bottleneck.pre(x)


    # ---------- Full forward (optional time condition + broadcast back to N points) ----------
    def forward(self, x, t=None):
        """
        x: (B, N, 2)
        t: (B,1) or (B,1,1) or None
        Returns: (B, N, out_dim)
        """
        B, N, _ = x.shape
        assert N == self.num_inputs, "N must equal the initialized n_points"

        # full tokens
        H = self.point_embed(x) + self.pos_full[:, :N]                 # (B, N, E)

        # Optional time
        if t is not None:
            t_emb = self.t_proj(self.time_embedder(t))                 # (B, E)
            H = H + t_emb.unsqueeze(1)

        # queries from encode (already includes cross-attn/ff)
        bottleneck = self.encode(x)                                    # (B, M, E)
        sub = bottleneck
        if t is not None:
            sub = sub + t_emb.unsqueeze(1)

        for blk in self.sub_self_blocks:
            sub = blk(sub)   
        h_updated, _ = self.final_attn(H, sub, sub)   # Q=h, KV=sub   -> (B, N, E)

        # ====== 4) Output projection, keep (B, N, 5) ======
        out = self.out_proj(h_updated)
        return out


# ============================================================================
# PVCNN Models for Flow Matching (Comparison with UncondUniGBNTransformer)
# ============================================================================

def _linear_gn_swish(in_channels, out_channels):
    """Linear + GroupNorm + Swish for MLP components."""
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.GroupNorm(8, out_channels),
        nn.SiLU()  # Swish
    )


def _create_pvcnn_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    """
    Create MLP components for PVCNN.
    """
    r = width_multiplier
    if not _lazy_import_pvcnn():
        raise ImportError("PVCNN modules not available. Please compile CUDA backend first.")

    SharedMLP = _pvcnn_modules['SharedMLP']

    if dim == 1:
        block = _linear_gn_swish
    else:
        block = SharedMLP

    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_swish(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def _create_pvcnn_sa_components(sa_blocks_cfg, extra_feature_channels, embed_dim=64, use_att=False,
                                 dropout=0.1, with_se=False, normalize=True, eps=0,
                                 width_multiplier=1, voxel_resolution_multiplier=1):
    """
    Create Set Abstraction (SA) layers for PVCNN.
    """
    if not _lazy_import_pvcnn():
        raise ImportError("PVCNN modules not available.")

    SharedMLP = _pvcnn_modules['SharedMLP']
    PVConv = _pvcnn_modules['PVConv']
    PointNetSAModule = _pvcnn_modules['PointNetSAModule']
    PointNetAModule = _pvcnn_modules['PointNetAModule']

    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3

    sa_layers, sa_in_channels = [], []
    c = 0
    for conv_configs, sa_configs in sa_blocks_cfg:
        k = 0
        sa_in_channels.append(in_channels)
        sa_blocks = []

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c+1) % 2 == 0 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                                              dropout=dropout,
                                              with_se=with_se, with_se_relu=True,
                                              normalize=normalize, eps=eps)

                if c == 0:
                    sa_blocks.append(block(in_channels, out_channels))
                elif k == 0:
                    sa_blocks.append(block(in_channels + embed_dim, out_channels))
                in_channels = out_channels
                k += 1
            extra_feature_channels = in_channels
        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                                      num_neighbors=num_neighbors)
        sa_blocks.append(block(in_channels=extra_feature_channels + (embed_dim if k == 0 else 0), out_channels=out_channels,
                               include_coordinates=True))
        c += 1
        in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))

    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def _create_pvcnn_fp_modules(fp_blocks_cfg, in_channels, sa_in_channels, embed_dim=64, use_att=False,
                              dropout=0.1, with_se=False, normalize=True, eps=0,
                              width_multiplier=1, voxel_resolution_multiplier=1):
    """
    Create Feature Propagation (FP) layers for PVCNN.
    """
    if not _lazy_import_pvcnn():
        raise ImportError("PVCNN modules not available.")

    SharedMLP = _pvcnn_modules['SharedMLP']
    PVConv = _pvcnn_modules['PVConv']
    PointNetFPModule = _pvcnn_modules['PointNetFPModule']

    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    c = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks_cfg):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim, out_channels=out_channels)
        )
        in_channels = out_channels[-1]

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c+1) % 2 == 0 and c < len(fp_blocks_cfg) - 1 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                                              dropout=dropout,
                                              with_se=with_se, with_se_relu=True,
                                              normalize=normalize, eps=eps)

                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))

        c += 1

    return fp_layers, in_channels


class UncondPVCNN(UnconditionalVectorField):
    """
    PVCNN-based unconditional vector field for flow matching.

    This is the **permutation equivariant** version - the output does NOT depend
    on the ordering of input points. It uses the original PVCNN architecture
    without any learnable positional embeddings.

    Input:  x: (B, N, 3) - xyz coordinates
            t: (B,) or (B, 1) - time step
    Output: (B, N, 3) - velocity vectors
    """

    # SA blocks configuration: (conv_configs, sa_configs)
    # conv_configs = (out_channels, num_blocks, voxel_resolution) or None
    # sa_configs = (num_centers, radius, num_neighbors, out_channels)
    # Same as PVD original (test_generation.py PVCNN2)
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),     # N -> 1024
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),     # 1024 -> 256
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),     # 256 -> 64
        (None, (16, 0.8, 32, (256, 256, 512))),       # 64 -> 16 (global aggregation)
    ]

    # FP blocks configuration: (fp_configs, conv_configs)
    # fp_configs = out_channels tuple for PointNetFPModule
    # conv_configs = (out_channels, num_blocks, voxel_resolution) or None
    fp_blocks = [
        ((256, 256), (256, 3, 8)),    # 16 -> 64
        ((256, 256), (256, 3, 8)),    # 64 -> 128
        ((256, 128), (128, 2, 16)),   # 128 -> 256
        ((128, 128, 64), (64, 2, 32)), # 256 -> N
    ]

    def __init__(self, n_points=500, embed_dim=64, use_att=True, dropout=0.1,
                 width_multiplier=1.0, voxel_resolution_multiplier=1.0):
        super().__init__()

        if not _lazy_import_pvcnn():
            raise ImportError("PVCNN modules not available. Please compile CUDA backend in PVD-main/modules/functional/")

        Attention = _pvcnn_modules['Attention']

        self.embed_dim = embed_dim
        self.n_points = n_points

        # Use learnable Fourier encoder for time embedding (same as other models)
        self.time_embedder = FourierEncoder(embed_dim)

        # extra_feature_channels = 0 because input is pure xyz (no extra features)
        self.in_channels = 3  # xyz only

        # Build SA layers
        sa_layers, sa_in_channels, channels_sa_features, _ = _create_pvcnn_sa_components(
            sa_blocks_cfg=self.sa_blocks, extra_feature_channels=0, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        # Optional global attention
        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        # Adjust sa_in_channels[0] for FP modules (no extra features at input level)
        sa_in_channels[0] = 0

        # Build FP layers
        fp_layers, channels_fp_features = _create_pvcnn_fp_modules(
            fp_blocks_cfg=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels,
            with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        # Output classifier: channels_fp_features -> 128 -> dropout -> 3
        layers, _ = _create_pvcnn_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, 3],  # output 3D velocity
            classifier=True, dim=2, width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)

        # Time embedding MLP
        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, t):
        """
        Args:
            x: (B, N, 3) - point cloud xyz coordinates
            t: (B,) or (B, 1) or (B, 1, 1) - time step in [0, 1]
        Returns:
            (B, N, 3) - velocity vectors for each point
        """
        B, N, D = x.shape
        assert D == 3, f"Expected input dim 3, got {D}"

        # Time embedding using learnable Fourier features: (B, embed_dim) -> (B, embed_dim, N)
        temb = self.embedf(self.time_embedder(t))[:, :, None].expand(-1, -1, N)

        # Convert to PVCNN format: (B, N, 3) -> (B, 3, N)
        coords = x.permute(0, 2, 1).contiguous()  # (B, 3, N)
        features = coords.clone()  # (B, 3, N) - use coords as features initially

        # Store intermediate features and coords for FP layers
        coords_list, in_features_list = [], []

        # SA layers (encoder)
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, temb = sa_blocks((features, coords, temb))
            else:
                features, coords, temb = sa_blocks((torch.cat([features, temb], dim=1), coords, temb))

        # Input level features for FP: no extra features
        in_features_list[0] = torch.zeros(B, 0, N, device=x.device)  # empty tensor for no extra features

        # Global attention
        if self.global_att is not None:
            features = self.global_att(features)

        # FP layers (decoder)
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords, temb = fp_blocks((
                coords_list[-1-fp_idx],  # original coords at this level
                coords,                   # current coords
                torch.cat([features, temb], dim=1),  # features + time
                in_features_list[-1-fp_idx],  # skip connection features
                temb
            ))

        # Output: (B, 3, N) -> (B, N, 3)
        out = self.classifier(features)  # (B, 3, N)
        return out.permute(0, 2, 1).contiguous()  # (B, N, 3)


class UncondPVCNN_PosEmbed(UnconditionalVectorField):
    """
    PVCNN-based unconditional vector field with learnable positional embeddings.

    This version adds learnable positional embeddings to each point index, similar
    to UncondUniGBNTransformer. This makes the model aware of the point ordering
    and breaks permutation equivariance.

    Input:  x: (B, N, 3) - xyz coordinates
            t: (B,) or (B, 1) - time step
    Output: (B, N, 3) - velocity vectors
    """

    # Same architecture as PVD original (test_generation.py PVCNN2)
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]

    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, n_points=500, embed_dim=64, pos_embed_dim=32, use_att=True, dropout=0.1,
                 width_multiplier=1.0, voxel_resolution_multiplier=1.0):
        super().__init__()

        if not _lazy_import_pvcnn():
            raise ImportError("PVCNN modules not available. Please compile CUDA backend in PVD-main/modules/functional/")

        Attention = _pvcnn_modules['Attention']

        self.embed_dim = embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.n_points = n_points

        # Use learnable Fourier encoder for time embedding (same as other models)
        self.time_embedder = FourierEncoder(embed_dim)

        # Learnable positional embedding for each point index
        # This breaks permutation equivariance!
        self.pos = nn.Parameter(torch.zeros(1, n_points, pos_embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # Project positional embedding to feature space (will be added to input features)
        self.pos_proj = nn.Conv1d(pos_embed_dim, 3, 1)  # pos_embed_dim -> 3 to match xyz

        # Input: xyz(3) + pos_embed projected(3) = 6 channels
        # But we'll use extra_feature_channels = 3 (the pos embed projection)
        self.in_channels = 3 + 3  # xyz + pos_embed_proj

        # Build SA layers with extra_feature_channels = 3 (pos embedding)
        sa_layers, sa_in_channels, channels_sa_features, _ = _create_pvcnn_sa_components(
            sa_blocks_cfg=self.sa_blocks, extra_feature_channels=3, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        # For FP layers, sa_in_channels[0] should be 3 (extra features at input = pos_embed_proj)
        sa_in_channels[0] = 3

        fp_layers, channels_fp_features = _create_pvcnn_fp_modules(
            fp_blocks_cfg=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels,
            with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = _create_pvcnn_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, 3],
            classifier=True, dim=2, width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)

        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, t):
        """
        Args:
            x: (B, N, 3) - point cloud xyz coordinates
            t: (B,) or (B, 1) or (B, 1, 1) - time step in [0, 1]
        Returns:
            (B, N, 3) - velocity vectors for each point
        """
        B, N, D = x.shape
        assert D == 3, f"Expected input dim 3, got {D}"

        # Time embedding using learnable Fourier features
        temb = self.embedf(self.time_embedder(t))[:, :, None].expand(-1, -1, N)  # (B, embed_dim, N)

        # Get positional embeddings for current N points
        pos_embed = self.pos[:, :N, :].expand(B, -1, -1)  # (B, N, pos_embed_dim)

        # Project positional embedding: (B, N, pos_embed_dim) -> (B, pos_embed_dim, N) -> (B, 3, N)
        pos_feat = self.pos_proj(pos_embed.permute(0, 2, 1))  # (B, 3, N)

        # Prepare PVCNN format
        coords = x.permute(0, 2, 1).contiguous()  # (B, 3, N)

        # Concatenate xyz with positional features: (B, 6, N)
        # features = [coords (xyz), pos_feat]
        features = torch.cat([coords, pos_feat], dim=1)  # (B, 6, N)

        coords_list, in_features_list = [], []

        # SA layers
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, temb = sa_blocks((features, coords, temb))
            else:
                features, coords, temb = sa_blocks((torch.cat([features, temb], dim=1), coords, temb))

        # For FP: in_features_list[0] should be the extra features (pos_feat)
        in_features_list[0] = pos_feat  # (B, 3, N) - the positional embeddings

        if self.global_att is not None:
            features = self.global_att(features)

        # FP layers
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords, temb = fp_blocks((
                coords_list[-1-fp_idx],
                coords,
                torch.cat([features, temb], dim=1),
                in_features_list[-1-fp_idx],
                temb
            ))

        out = self.classifier(features)  # (B, 3, N)
        return out.permute(0, 2, 1).contiguous()  # (B, N, 3)