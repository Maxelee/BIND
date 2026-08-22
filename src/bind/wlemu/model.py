"""Field-level UNet + conditional flow matching for ``θ → κ``.

``FieldUNet`` is a resolution-aware sibling of ``bind.model.UNet`` for the WL
emulator: a **single-channel** generator (noisy κ → velocity, no DMO input to
concatenate) conditioned on the 30 SB35 astro params + the source redshift.  It
reuses BIND's validated building blocks (``ResBlock``/``AdaGroupNorm`` param
conditioning, sinusoidal time + redshift embeddings, ``SelfAttention``) so the
conditioning machinery is identical to the painter — only the input/output
plumbing differs (no condition concat, ``in_ch == out_ch == 1``, attention
placement follows the chosen ``resolution``).

``FieldCFM`` is OT conditional flow matching: ``x_t = (1-t)·noise + t·κ``,
velocity target ``κ - noise``; sampling is Euler ODE integration from noise, with
optional classifier-free guidance to amplify the (intrinsically weak) feedback
response.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from bind.model import (
    AdaGroupNorm,  # noqa: F401  (re-exported convenience)
    ParamEncoder,
    ResBlock,
    SelfAttention,
    SinusoidalEmbedding,
)

__all__ = ["FieldUNet", "FieldCFM", "default_arch"]


def default_arch(resolution: int):
    """Pick (ch_mult, attn_resolutions) so the bottleneck lands at 32² for any
    input size — self-attention is *only* ever applied at ≤32² (≤1024 tokens),
    which is what makes native 1024² tractable.  Each extra factor-of-2 in
    resolution adds one (cheap, conv-only) outer stage.

        128  → (1,2,4,8)         3 downsamples → 16² bottleneck
        256  → (1,2,4,8)         3 downsamples → 32² bottleneck
        512  → (1,2,2,4,8)       4 downsamples → 32² bottleneck
        1024 → (1,1,2,2,4,8)     5 downsamples → 32² bottleneck
    """
    table = {
        128: (1, 2, 4, 8),
        256: (1, 2, 4, 8),
        512: (1, 2, 2, 4, 8),
        1024: (1, 1, 2, 2, 4, 8),
    }
    if resolution not in table:
        raise ValueError(f"no default arch for resolution {resolution}; "
                         "pass ch_mult explicitly")
    return table[resolution], (16, 32)


class _Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class _Up(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class FieldUNet(nn.Module):
    """Param + redshift conditioned UNet predicting a flow-matching velocity.

    Parameters
    ----------
    resolution : input/output map size; controls where self-attention is placed
        (``attn_resolutions``) and the number of feasible downsamplings.
    n_params : conditioning parameter dimension (30 SB35 astro params).
    condition_redshift : add a sinusoidal→MLP embedding of the scale factor
        ``a = 1/(1+z_s)`` (summed into the AdaGroupNorm conditioning), so one
        model spans all source planes.  On by default for the WL emulator.
    """

    def __init__(self, resolution=256, base_ch=128, ch_mult=None,
                 n_blocks=2, emb_dim=256, attn_resolutions=None,
                 dropout=0.0, n_params=30, condition_redshift=True,
                 in_ch=1, out_ch=1, grad_checkpoint=False):
        super().__init__()
        self.resolution = resolution
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.grad_checkpoint = grad_checkpoint
        # Derive a depth that keeps attention at ≤32² unless overridden.
        if ch_mult is None or attn_resolutions is None:
            d_mult, d_attn = default_arch(resolution)
            ch_mult = ch_mult or d_mult
            attn_resolutions = attn_resolutions or d_attn
        self.ch_mult = tuple(ch_mult)
        self.attn_resolutions = tuple(attn_resolutions)

        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(base_ch),
            nn.Linear(base_ch, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.param_emb = ParamEncoder(n_params, emb_dim)
        self.condition_redshift = condition_redshift
        self.redshift_emb = nn.Sequential(
            SinusoidalEmbedding(base_ch),
            nn.Linear(base_ch, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        ) if condition_redshift else None

        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [base_ch]
        ch = base_ch
        res = resolution
        for i, mult in enumerate(ch_mult):
            out = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(n_blocks):
                blocks.append(ResBlock(ch, out, emb_dim, dropout=dropout))
                if res in attn_resolutions:
                    blocks.append(SelfAttention(out))
                ch = out
            self.encoders.append(blocks)
            channels.append(ch)
            if i < len(ch_mult) - 1:
                self.downsamples.append(_Down(ch))
                res //= 2
            else:
                self.downsamples.append(nn.Identity())

        # Bottleneck
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, emb_dim, dropout=dropout),
            SelfAttention(ch),
            ResBlock(ch, ch, emb_dim, dropout=dropout),
        ])

        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in enumerate(reversed(ch_mult)):
            out = base_ch * mult
            blocks = nn.ModuleList()
            for j in range(n_blocks + 1):
                skip_ch = channels.pop() if j == 0 else 0
                blocks.append(ResBlock(ch + skip_ch, out, emb_dim, dropout=dropout))
                if res in attn_resolutions:
                    blocks.append(SelfAttention(out))
                ch = out
            self.decoders.append(blocks)
            if i < len(ch_mult) - 1:
                self.upsamples.append(_Up(ch))
                res *= 2
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _run_block(self, block, h, emb):
        """Run a block, optionally under activation checkpointing (training only).

        Trades recompute for memory — essential at native 1024² where the outer
        stages hold ~GB activation tensors.  Self-attention takes no ``emb``.
        """
        is_res = isinstance(block, ResBlock)
        if self.grad_checkpoint and self.training:
            if is_res:
                return checkpoint(block, h, emb, use_reentrant=False)
            return checkpoint(block, h, use_reentrant=False)
        return block(h, emb) if is_res else block(h)

    def forward(self, x, t, params, scale_factor=None):
        """x: (B,1,R,R); t: (B,); params: (B,30); scale_factor: (B,) a=1/(1+z_s)."""
        emb = self.time_emb(t) + self.param_emb(params)
        if self.redshift_emb is not None:
            if scale_factor is None:
                scale_factor = torch.ones(x.shape[0], device=x.device)
            emb = emb + self.redshift_emb(scale_factor)

        h = self.input_conv(x)
        skips = [h]
        for blocks, down in zip(self.encoders, self.downsamples):
            for block in blocks:
                h = self._run_block(block, h, emb)
            skips.append(h)
            h = down(h)

        for block in self.mid:
            h = self._run_block(block, h, emb)

        for blocks, up in zip(self.decoders, self.upsamples):
            for i, block in enumerate(blocks):
                if isinstance(block, ResBlock) and i == 0:
                    h = torch.cat([h, skips.pop()], dim=1)
                h = self._run_block(block, h, emb)
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))


class FieldCFM:
    """Optimal-transport conditional flow matching for ``θ → κ`` (no DMO input).

    Forward:  ``x_t = (1-t)·noise + t·x1``;  velocity target ``v = x1 - noise``.
    Sampling: Euler ODE integration from noise at ``t=0`` to data at ``t=1``,
    with optional classifier-free guidance (``cfg_scale``) toward the params.
    """

    def __init__(self, model: FieldUNet, cfg_dropout=0.1):
        self.model = model
        self.cfg_dropout = cfg_dropout

    def loss(self, x1, params, scale_factor=None):
        """x1: (B,1,R,R) normalised κ; params: (B,30); scale_factor: (B,)."""
        B = x1.shape[0]
        t = torch.rand(B, device=x1.device)
        t4 = t[:, None, None, None]
        noise = torch.randn_like(x1)
        x_t = (1 - t4) * noise + t4 * x1
        velocity_target = x1 - noise

        if self.cfg_dropout > 0 and self.model.training:
            mask = torch.rand(B, device=x1.device) < self.cfg_dropout
            params = params.clone()
            params[mask] = 0.0

        v_pred = self.model(x_t, t, params, scale_factor)
        return F.mse_loss(v_pred, velocity_target)

    @torch.no_grad()
    def sample(self, params, scale_factor=None, n_steps=50, cfg_scale=1.0,
               resolution=None, generator=None):
        """Generate κ maps.

        params : (B,30); scale_factor : (B,) or None; returns (B,1,R,R) normalised.
        ``cfg_scale > 1`` sharpens the parameter dependence via guidance against
        the param-dropped (zeroed) conditioning.
        """
        self.model.eval()
        B = params.shape[0]
        device = params.device
        R = resolution or self.model.resolution
        x = torch.randn(B, self.model.out_ch, R, R, device=device, generator=generator)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            if cfg_scale != 1.0:
                v_c = self.model(x, t, params, scale_factor)
                v_u = self.model(x, t, torch.zeros_like(params), scale_factor)
                v = v_u + cfg_scale * (v_c - v_u)
            else:
                v = self.model(x, t, params, scale_factor)
            x = x + v * dt
        return x
