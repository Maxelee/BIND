"""3D U-Net + Flow Matching for conditional baryonic field generation.

Mirrors model.py but operates on volumetric inputs (B, C, D, H, W).
Differences from the 2D version:
  - Conv2d -> Conv3d everywhere; AdaGroupNorm broadcasts over 3 spatial axes.
  - SelfAttention reshapes (B, C, D, H, W) -> (B, DHW, C). Attention is
    enabled only at the deepest level by default (token count grows as
    res**3 in 3D — 32**3 = 32768 tokens is impractical).
  - Optional gradient checkpointing per ResBlock / attention layer.
    Necessary at 128**3 with base_ch=128 to fit on a 40GB A100.
  - No `large_scale` channel: the 3D dataset doesn't ship one. Default
    in_ch = 3 (state) + 1 (condition) = 4; out_ch = 3.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class ParamEncoder(nn.Module):
    def __init__(self, n_params=35, emb_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, p):
        return self.net(p)


class AdaGroupNorm3d(nn.Module):
    def __init__(self, channels, emb_dim, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.proj = nn.Linear(emb_dim, channels * 2)

    def forward(self, x, emb):
        x = self.norm(x)
        scale, shift = self.proj(emb)[:, :, None, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class ResBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, num_groups=32, dropout=0.0):
        super().__init__()
        self.norm1 = AdaGroupNorm3d(in_ch, emb_dim, num_groups)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGroupNorm3d(out_ch, emb_dim, num_groups)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.act(self.norm1(x, emb))
        h = self.conv1(h)
        h = self.act(self.norm2(h, emb))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention3d(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.norm(x).reshape(B, C, D * H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, D, H, W)


class Downsample3d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet3d(nn.Module):
    """3D U-Net for flow matching velocity prediction.

    Default: state (3) + condition (1) = 4 input channels, 3 output channels.
    """

    def __init__(self, in_ch=4, out_ch=3, base_ch=128,
                 ch_mult=(1, 2, 4, 8), n_blocks=2, emb_dim=256,
                 attn_resolutions=(16,), dropout=0.0, n_params=35,
                 input_resolution=128, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(base_ch),
            nn.Linear(base_ch, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.param_emb = ParamEncoder(n_params, emb_dim)

        self.input_conv = nn.Conv3d(in_ch, base_ch, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [base_ch]
        ch = base_ch
        res = input_resolution
        for i, mult in enumerate(ch_mult):
            out = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(n_blocks):
                blocks.append(ResBlock3d(ch, out, emb_dim, dropout=dropout))
                if res in attn_resolutions:
                    blocks.append(SelfAttention3d(out))
                ch = out
            self.encoders.append(blocks)
            channels.append(ch)
            if i < len(ch_mult) - 1:
                self.downsamples.append(Downsample3d(ch))
                res //= 2
            else:
                self.downsamples.append(nn.Identity())

        self.mid = nn.ModuleList([
            ResBlock3d(ch, ch, emb_dim, dropout=dropout),
            SelfAttention3d(ch),
            ResBlock3d(ch, ch, emb_dim, dropout=dropout),
        ])

        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in enumerate(reversed(ch_mult)):
            out = base_ch * mult
            blocks = nn.ModuleList()
            for j in range(n_blocks + 1):
                skip_ch = channels.pop() if j == 0 else 0
                blocks.append(ResBlock3d(ch + skip_ch, out, emb_dim, dropout=dropout))
                if res in attn_resolutions:
                    blocks.append(SelfAttention3d(out))
                ch = out
            self.decoders.append(blocks)
            if i < len(ch_mult) - 1:
                self.upsamples.append(Upsample3d(ch))
                res *= 2
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv3d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _run_block(self, block, x, emb):
        if isinstance(block, ResBlock3d):
            if self.use_checkpoint and self.training:
                return checkpoint(block, x, emb, use_reentrant=False)
            return block(x, emb)
        else:
            if self.use_checkpoint and self.training:
                return checkpoint(block, x, use_reentrant=False)
            return block(x)

    def forward(self, x, t, params):
        """
        x: (B, in_ch, D, H, W) — state + condition concatenated on channel axis
        t: (B,) in [0, 1]
        params: (B, n_params)
        """
        emb = self.time_emb(t) + self.param_emb(params)

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
                if isinstance(block, ResBlock3d):
                    if i == 0:
                        h = torch.cat([h, skips.pop()], dim=1)
                    h = self._run_block(block, h, emb)
                else:
                    h = self._run_block(block, h, emb)
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))


class FlowMatching3d:
    """Optimal-transport conditional flow matching, 3D variant.

    Forward:        x_t = (1 - t) * noise + t * x_1
    Velocity tgt:   v   = x_1 - noise

    Supports the two-head Stars layout (out_channels=4): targets are
    [DM_hydro, Gas, occupancy, conditional_density] and the
    star_occ_weight upweights both stellar channels on occupied voxels.
    """

    def __init__(self, model, cfg_dropout=0.0, star_occ_weight=1.0,
                 star_zero_norm=None, out_channels=3):
        self.model = model
        self.cfg_dropout = cfg_dropout
        self.star_occ_weight = star_occ_weight
        self.star_zero_norm = star_zero_norm
        self.out_channels = out_channels

    def loss(self, x1, condition, params):
        """
        x1:        (B, C, D, H, W)  with C = self.out_channels
        condition: (B, 1, D, H, W)
        params:    (B, n_params)
        """
        B = x1.shape[0]
        t = torch.rand(B, device=x1.device)
        t5 = t[:, None, None, None, None]
        noise = torch.randn_like(x1)

        x_t = (1 - t5) * noise + t5 * x1
        velocity_target = x1 - noise

        if self.cfg_dropout > 0 and self.model.training:
            mask = torch.rand(B, device=x1.device) < self.cfg_dropout
            params = params.clone()
            params[mask] = 0.0

        model_input = torch.cat([x_t, condition], dim=1)
        v_pred = self.model(model_input, t, params)

        per_voxel = (v_pred - velocity_target) ** 2

        if self.star_occ_weight != 1.0:
            w = torch.ones_like(per_voxel)
            if self.out_channels == 4:
                # Two-head Stars: ch2=occupancy (binary, normalized),
                # ch3=conditional density. Upweight both on truly-occupied
                # voxels (raw occupancy=1 => occ_norm > -mean/std + 0.5).
                star_occ = (x1[:, 2:3] > 0.5).float()
                w[:, 2:3] = 1.0 + (self.star_occ_weight - 1.0) * star_occ
                w[:, 3:4] = 1.0 + (self.star_occ_weight - 1.0) * star_occ
            elif self.star_zero_norm is not None:
                star_occ = (x1[:, 2:3] > self.star_zero_norm + 0.1).float()
                w[:, 2:3] = 1.0 + (self.star_occ_weight - 1.0) * star_occ
            per_voxel = per_voxel * w

        return per_voxel.mean()

    @torch.no_grad()
    def sample(self, condition, params, n_steps=50, cfg_scale=1.0):
        """
        condition: (B, 1, D, H, W)
        params:    (B, n_params)
        Returns:   (B, out_channels, D, H, W)
        """
        self.model.eval()
        B, _, D, H, W = condition.shape
        device = condition.device

        x = torch.randn(B, self.out_channels, D, H, W, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            inp = torch.cat([x, condition], dim=1)

            if cfg_scale != 1.0:
                v_cond = self.model(inp, t, params)
                v_uncond = self.model(inp, t, torch.zeros_like(params))
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.model(inp, t, params)

            x = x + v * dt

        return x
