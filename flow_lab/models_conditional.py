# cond_transformer_sdpa.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== The SDPA Cross-Attn block you provided =====
class CrossAttentionBlockSDPA(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, attn_dropout=0.0, resid_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

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
        # q: (B,Tq,E), kv: (B,Tk,E)
        q = self.norm1(q)
        Q = self._shape(self.q_proj(q))  # (B,H,Tq,Hd)
        if kv_proj is None:
            K = self._shape(self.k_proj(kv))
            V = self._shape(self.v_proj(kv))
        else:
            K, V = kv_proj  # already (B,H,Tk,Hd)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False
            )  # (B,H,Tq,Hd)

        B, H, Tq, Hd = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Tq, H * Hd)
        attn_out = self.out_proj(attn_out)
        x = q + self.resid_drop(attn_out)
        x = x + self.resid_drop(self.ff(self.norm2(x)))
        return x

    @torch.no_grad()
    def preproject_kv(self, kv):
        K = self._shape(self.k_proj(kv))
        V = self._shape(self.v_proj(kv))
        return K, V

# ===== Time encoding =====
class FourierEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 2), requires_grad=False)
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(t.shape[0], -1)
        wt = t * self.freqs[None, :]
        return torch.cat([torch.sin(wt), torch.cos(wt)], dim=-1)

# ===== Conditional Transformer body (point self-attention + SDPA cross-modal cross-attention) =====
class CondPointTransformerSDPA(nn.Module):
    """
    v_theta(x,t | c): the point sequence only does self-attn; the image serves as K/V for cross-attn (more efficient).
    - img_cond: (B, Lc, Cimg) or (B, Cimg)
    - cross_every: insert a cross-attn layer every few layers
    - Reuse the KV of the same img_tokens across all cross layers (a single pre-projection, memory-friendly)
    """
    def __init__(
        self,
        n_points=1024, in_dim=5, out_dim=5,
        embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0,
        t_embed_dim=40, img_dim=768,
        cross_every=2, max_img_tokens=256,
        attn_dropout=0.0, resid_dropout=0.1,
    ):
        super().__init__()
        self.n_points = n_points
        self.in_proj  = nn.Linear(in_dim, embed_dim)
        self.pos      = nn.Parameter(torch.zeros(1, n_points, embed_dim))

        self.time_embedder = FourierEncoder(t_embed_dim)
        self.t_proj        = nn.Linear(t_embed_dim, embed_dim)

        # Image condition projection + positional encoding (only added onto the tokens)
        self.img_proj = nn.Linear(img_dim, embed_dim)
        self.img_pos  = nn.Parameter(torch.zeros(1, max_img_tokens, embed_dim))

        # Point self-attn backbone
        self.self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                batch_first=True, activation="gelu", dropout=0.1
            ) for _ in range(depth)
        ])

        # cross-attn layers (Q=points, K/V=image)
        self.cross_every = max(1, int(cross_every))
        num_cross = max(1, depth // self.cross_every)
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlockSDPA(embed_dim, num_heads,
                                    mlp_ratio=mlp_ratio,
                                    attn_dropout=attn_dropout,
                                    resid_dropout=resid_dropout)
            for _ in range(num_cross)
        ])

        self.out_proj = nn.Linear(embed_dim, out_dim)

    def _make_img_tokens(self, img_cond: torch.Tensor, E: int) -> torch.Tensor:
        if img_cond.dim() == 2:          # (B, Cimg) -> (B,1,E)
            tok = self.img_proj(img_cond).unsqueeze(1)
            tok = tok + self.img_pos[:, :1]
            return tok
        elif img_cond.dim() == 3:        # (B, Lc, Cimg)
            tok = self.img_proj(img_cond)
            Lc  = tok.shape[1]
            if Lc > self.img_pos.shape[1]:
                k = (Lc + self.img_pos.shape[1] - 1) // self.img_pos.shape[1]
                img_pos = self.img_pos.repeat(1, k, 1)[:, :Lc]
            else:
                img_pos = self.img_pos[:, :Lc]
            return tok + img_pos
        else:
            raise ValueError("img_cond must be (B,Cimg) or (B,Lc,Cimg)")

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                img_cond: torch.Tensor | None = None,
                cond_drop: bool = False) -> torch.Tensor:
        """
        x: (B,N,in_dim), t: (B,1,1)
        img_cond: None / (B,Cimg) / (B,Lc,Cimg)
        cond_drop=True ignores the condition (used for CFG training)
        """
        B, N, _ = x.shape
        h = self.in_proj(x) + self.pos[:, :N]
        t_emb = self.t_proj(self.time_embedder(t))
        h = h + t_emb.unsqueeze(1)

        # If the condition is dropped / absent: only do point self-attn; otherwise prepare image KV (one-shot pre-projection)
        kv_proj = None
        img_tokens = None
        if (img_cond is not None) and (not (self.training and cond_drop)):
            img_tokens = self._make_img_tokens(img_cond, h.size(-1))      # (B,Lc,E) or (B,1,E)
            # Pre-project and cache KV, reused across all cross layers
            # Note: grad to K/V is not needed here (frozen / used only as a condition), but backprop is allowed and harmless
            K = self.cross_layers[0].k_proj(img_tokens)
            V = self.cross_layers[0].v_proj(img_tokens)
            # (B,Lc,E) -> (B,H,Tk,Hd)
            H = self.cross_layers[0].num_heads
            Hd = self.cross_layers[0].head_dim
            B_, Tk, E = K.shape
            K = K.view(B_, Tk, H, Hd).transpose(1, 2).contiguous()
            V = V.view(B_, Tk, H, Hd).transpose(1, 2).contiguous()
            kv_proj = (K, V)

        cidx = 0
        for i, lyr in enumerate(self.self_layers, start=1):
            h = lyr(h)                            # point self-attn
            if (img_tokens is not None) and (i % self.cross_every == 0):
                h = self.cross_layers[cidx](h, img_tokens, kv_proj=kv_proj, attn_mask=None)  # Q=points, KV=image
                cidx = min(cidx + 1, len(self.cross_layers) - 1)

        return self.out_proj(h)                   # (B,N,out_dim)


# ===== PE Variant (No Learnable Positional Encoding) =====
class CondPointTransformerSDPA_PE(nn.Module):
    """
    Same as CondPointTransformerSDPA but without learnable positional encoding.
    Uses register_buffer with zeros instead of nn.Parameter.
    """
    def __init__(
        self,
        n_points=1024, in_dim=5, out_dim=5,
        embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0,
        t_embed_dim=40, img_dim=768,
        cross_every=2, max_img_tokens=256,
        attn_dropout=0.0, resid_dropout=0.1,
    ):
        super().__init__()
        self.n_points = n_points
        self.in_proj  = nn.Linear(in_dim, embed_dim)

        # Key change: use register_buffer instead of nn.Parameter (not learnable)
        self.register_buffer("pos", torch.zeros(1, 1, embed_dim))

        self.time_embedder = FourierEncoder(t_embed_dim)
        self.t_proj        = nn.Linear(t_embed_dim, embed_dim)

        # Image/anchor condition projection (no learnable positional encoding)
        self.img_proj = nn.Linear(img_dim, embed_dim)
        self.register_buffer("img_pos", torch.zeros(1, 1, embed_dim))

        # Point self-attention layers
        self.self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                batch_first=True, activation="gelu", dropout=0.1
            ) for _ in range(depth)
        ])

        # Cross-attention layers
        self.cross_every = max(1, int(cross_every))
        num_cross = max(1, depth // self.cross_every)
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlockSDPA(embed_dim, num_heads,
                                    mlp_ratio=mlp_ratio,
                                    attn_dropout=attn_dropout,
                                    resid_dropout=resid_dropout)
            for _ in range(num_cross)
        ])

        self.out_proj = nn.Linear(embed_dim, out_dim)

    def _make_img_tokens(self, img_cond: torch.Tensor, E: int) -> torch.Tensor:
        if img_cond.dim() == 2:          # (B, Cimg) -> (B,1,E)
            tok = self.img_proj(img_cond).unsqueeze(1)
            # No positional encoding added (buffer is zeros)
            return tok
        elif img_cond.dim() == 3:        # (B, Lc, Cimg)
            tok = self.img_proj(img_cond)
            # No positional encoding added
            return tok
        else:
            raise ValueError("img_cond must be (B,Cimg) or (B,Lc,Cimg)")

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                img_cond: torch.Tensor | None = None,
                cond_drop: bool = False) -> torch.Tensor:
        """
        x: (B,N,in_dim), t: (B,1,1)
        img_cond: None / (B,Cimg) / (B,Lc,Cimg)
        cond_drop=True ignores the condition (used for CFG training)
        """
        B, N, _ = x.shape
        h = self.in_proj(x)  # No positional encoding added (pos buffer is zeros)
        t_emb = self.t_proj(self.time_embedder(t))
        h = h + t_emb.unsqueeze(1)

        kv_proj = None
        img_tokens = None
        if (img_cond is not None) and (not (self.training and cond_drop)):
            img_tokens = self._make_img_tokens(img_cond, h.size(-1))
            K = self.cross_layers[0].k_proj(img_tokens)
            V = self.cross_layers[0].v_proj(img_tokens)
            H = self.cross_layers[0].num_heads
            Hd = self.cross_layers[0].head_dim
            B_, Tk, E = K.shape
            K = K.view(B_, Tk, H, Hd).transpose(1, 2).contiguous()
            V = V.view(B_, Tk, H, Hd).transpose(1, 2).contiguous()
            kv_proj = (K, V)

        cidx = 0
        for i, lyr in enumerate(self.self_layers, start=1):
            h = lyr(h)
            if (img_tokens is not None) and (i % self.cross_every == 0):
                h = self.cross_layers[cidx](h, img_tokens, kv_proj=kv_proj, attn_mask=None)
                cidx = min(cidx + 1, len(self.cross_layers) - 1)

        return self.out_proj(h)


# ===== Variable Anchor Conditioning (Pad-to-max + Mask + Missing Embedding) =====
class CondPointTransformerSDPAVariable(nn.Module):
    """
    Conditional Transformer for variable number of anchors (3-8).

    Key features:
    - Pad-to-max: Anchors padded to max_anchors (8) with NaN
    - Mask: Attention mask to ignore padded positions
    - Missing embedding: Learnable embedding for padded anchor slots

    Each anchor is treated as a separate token for cross-attention.
    """
    def __init__(
        self,
        n_points=256, in_dim=2, out_dim=2,
        embed_dim=256, depth=6, num_heads=4, mlp_ratio=4.0,
        t_embed_dim=40,
        max_anchors=8,   # max number of anchor points
        anchor_dim=2,    # each anchor is 2D coordinate
        cross_every=2,
        attn_dropout=0.0, resid_dropout=0.1,
    ):
        super().__init__()
        self.n_points = n_points
        self.max_anchors = max_anchors
        self.anchor_dim = anchor_dim

        # Point input projection + positional encoding
        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, n_points, embed_dim))

        # Time embedding
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.t_proj = nn.Linear(t_embed_dim, embed_dim)

        # Anchor conditioning
        self.anchor_proj = nn.Linear(anchor_dim, embed_dim)
        self.anchor_pos = nn.Parameter(torch.zeros(1, max_anchors, embed_dim))

        # Missing embedding for padded anchor positions
        self.missing_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Point self-attention layers
        self.self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                batch_first=True, activation="gelu", dropout=0.1
            ) for _ in range(depth)
        ])

        # Cross-attention layers (Q=points, K/V=anchors)
        self.cross_every = max(1, int(cross_every))
        num_cross = max(1, depth // self.cross_every)
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlockSDPA(embed_dim, num_heads,
                                    mlp_ratio=mlp_ratio,
                                    attn_dropout=attn_dropout,
                                    resid_dropout=resid_dropout)
            for _ in range(num_cross)
        ])

        self.out_proj = nn.Linear(embed_dim, out_dim)

    def _make_anchor_tokens(self, anchors: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
        """
        Create anchor tokens with missing embedding for padded positions.

        Args:
            anchors: (B, max_anchors, 2) - coordinates, NaN for missing
            anchor_mask: (B, max_anchors) - True for valid, False for missing

        Returns:
            tokens: (B, max_anchors, embed_dim)
        """
        B, M, D = anchors.shape

        # Replace NaN with 0 for projection (will be replaced by missing_embed)
        anchors_clean = torch.nan_to_num(anchors, nan=0.0)
        tok = self.anchor_proj(anchors_clean)  # (B, M, E)
        tok = tok + self.anchor_pos[:, :M]

        # Replace invalid positions with missing embedding
        mask_expanded = anchor_mask.unsqueeze(-1)  # (B, M, 1)
        missing = self.missing_embed.expand(B, M, -1)  # (B, M, E)
        tok = torch.where(mask_expanded, tok, missing)

        return tok

    def _make_attn_mask(self, anchor_mask: torch.Tensor, Tq: int, num_heads: int) -> torch.Tensor:
        """
        Create attention mask for cross-attention.

        Args:
            anchor_mask: (B, max_anchors) - True for valid, False for padded
            Tq: number of query tokens (n_points)
            num_heads: number of attention heads

        Returns:
            attn_mask: (B, num_heads, Tq, max_anchors) or (B, 1, 1, max_anchors)
        """
        B, M = anchor_mask.shape
        # SDPA expects: True = attend, so we invert (~) for padding mask
        # Actually SDPA attn_mask: negative values = ignore, 0 = attend
        # We create float mask: 0 for valid, -inf for padded
        mask = torch.zeros(B, 1, 1, M, device=anchor_mask.device, dtype=torch.float32)
        mask = mask.masked_fill(~anchor_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        return mask  # (B, 1, 1, M) - broadcasts to (B, H, Tq, M)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                anchors: torch.Tensor | None = None,
                anchor_mask: torch.Tensor | None = None,
                cond_drop: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim) - point coordinates
            t: (B, 1, 1) - time
            anchors: (B, max_anchors, 2) - anchor coordinates (NaN for missing)
            anchor_mask: (B, max_anchors) - True for valid anchors
            cond_drop: if True, ignore condition (for CFG training)

        Returns:
            v: (B, N, out_dim) - predicted velocity
        """
        B, N, _ = x.shape
        h = self.in_proj(x) + self.pos[:, :N]
        t_emb = self.t_proj(self.time_embedder(t))
        h = h + t_emb.unsqueeze(1)

        # Prepare anchor conditioning
        kv_proj = None
        anchor_tokens = None
        attn_mask = None

        use_cond = (anchors is not None) and (not (self.training and cond_drop))

        if use_cond:
            # Create anchor tokens
            if anchor_mask is None:
                # Infer mask from NaN values
                anchor_mask = ~torch.isnan(anchors[:, :, 0])  # (B, max_anchors)

            anchor_tokens = self._make_anchor_tokens(anchors, anchor_mask)  # (B, M, E)
            attn_mask = self._make_attn_mask(anchor_mask, N, self.cross_layers[0].num_heads)

            # Pre-project K, V for all cross-attention layers
            K = self.cross_layers[0].k_proj(anchor_tokens)
            V = self.cross_layers[0].v_proj(anchor_tokens)
            H = self.cross_layers[0].num_heads
            Hd = self.cross_layers[0].head_dim
            B_, Tk, E = K.shape
            K = K.view(B_, Tk, H, Hd).transpose(1, 2).contiguous()
            V = V.view(B_, Tk, H, Hd).transpose(1, 2).contiguous()
            kv_proj = (K, V)

        cidx = 0
        for i, lyr in enumerate(self.self_layers, start=1):
            h = lyr(h)  # Point self-attention
            if (anchor_tokens is not None) and (i % self.cross_every == 0):
                h = self.cross_layers[cidx](h, anchor_tokens, kv_proj=kv_proj, attn_mask=attn_mask)
                cidx = min(cidx + 1, len(self.cross_layers) - 1)

        return self.out_proj(h)  # (B, N, out_dim)