"""3D U-Net + Flow Matching for volumetric conditional generation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 32) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if groups <= max_groups and channels % groups == 0:
            return groups
    return 1


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class ParamEncoder(nn.Module):
    """Encode cosmological parameters into embedding space."""

    def __init__(self, n_params: int = 36, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)


class AdaGroupNorm3D(nn.Module):
    """GroupNorm with adaptive scale/shift from conditioning embedding."""

    def __init__(self, channels: int, emb_dim: int, max_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels, max_groups), channels)
        self.proj = nn.Linear(emb_dim, channels * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale, shift = self.proj(emb).chunk(2, dim=1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        return x * (1 + scale) + shift


class ResBlock3D(nn.Module):
    """Residual block with AdaGroupNorm conditioning for 3D tensors."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = AdaGroupNorm3D(in_ch, emb_dim)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGroupNorm3D(out_ch, emb_dim)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x, emb))
        h = self.conv1(h)
        h = self.act(self.norm2(h, emb))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for flow-matching velocity prediction."""

    def __init__(
        self,
        in_ch: int = 7,
        out_ch: int = 3,
        base_ch: int = 16,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        n_blocks: int = 2,
        emb_dim: int = 256,
        dropout: float = 0.0,
        n_params: int = 36,
    ):
        super().__init__()

        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(base_ch),
            nn.Linear(base_ch, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.param_emb = ParamEncoder(n_params, emb_dim)

        self.input_conv = nn.Conv3d(in_ch, base_ch, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        channels = [base_ch]
        ch = base_ch

        for i, mult in enumerate(ch_mult):
            out = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(n_blocks):
                blocks.append(ResBlock3D(ch, out, emb_dim, dropout=dropout))
                ch = out
            self.encoders.append(blocks)
            channels.append(ch)
            if i < len(ch_mult) - 1:
                self.downsamples.append(Downsample3D(ch))
            else:
                self.downsamples.append(nn.Identity())

        self.mid = nn.ModuleList(
            [
                ResBlock3D(ch, ch, emb_dim, dropout=dropout),
                ResBlock3D(ch, ch, emb_dim, dropout=dropout),
            ]
        )

        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i, mult in enumerate(reversed(ch_mult)):
            out = base_ch * mult
            blocks = nn.ModuleList()
            for j in range(n_blocks + 1):
                skip_ch = channels.pop() if j == 0 else 0
                blocks.append(ResBlock3D(ch + skip_ch, out, emb_dim, dropout=dropout))
                ch = out
            self.decoders.append(blocks)
            if i < len(ch_mult) - 1:
                self.upsamples.append(Upsample3D(ch))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(_group_count(ch), ch)
        self.out_conv = nn.Conv3d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        emb = self.time_emb(t) + self.param_emb(params)

        h = self.input_conv(x)

        skips = [h]
        for blocks, down in zip(self.encoders, self.downsamples):
            for block in blocks:
                h = block(h, emb)
            skips.append(h)
            h = down(h)

        for block in self.mid:
            h = block(h, emb)

        for blocks, up in zip(self.decoders, self.upsamples):
            for i, block in enumerate(blocks):
                if i == 0:
                    h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, emb)
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))


def _expand_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
    return t.view(t.shape[0], *([1] * (ndim - 1)))


class StochasticInterpolant3D:
    """Stochastic interpolant bridge from DMO to hydro in 3D."""

    def __init__(
        self,
        model: nn.Module,
        sigma: float = 0.5,
        cfg_dropout: float = 0.0,
        star_occ_weight: float = 5.0,
        star_zero_norm: float | None = None,
    ):
        self.model = model
        self.sigma = sigma
        self.cfg_dropout = cfg_dropout
        self.star_occ_weight = star_occ_weight
        self.star_zero_norm = star_zero_norm

    @staticmethod
    def _x0_from_dmo(condition: torch.Tensor) -> torch.Tensor:
        return condition.expand(-1, 3, *([-1] * (condition.ndim - 2)))

    def loss(
        self,
        x1: torch.Tensor,
        condition: torch.Tensor,
        large_scale: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device)
        t_view = _expand_t(t, x1.ndim)

        x0 = self._x0_from_dmo(condition)
        z = torch.randn_like(x1)

        x_t = (1 - t_view) * x0 + t_view * x1 + self.sigma * t_view * (1 - t_view) * z
        velocity_target = (x1 - x0) + self.sigma * (1 - 2 * t_view) * z

        if self.cfg_dropout > 0 and self.model.training:
            mask = torch.rand(bsz, device=x1.device) < self.cfg_dropout
            params = params.clone()
            params[mask] = 0.0

        model_input = torch.cat([x_t, condition, large_scale], dim=1)
        v_pred = self.model(model_input, t, params)

        per_voxel = (v_pred - velocity_target) ** 2

        if self.star_occ_weight != 1.0 and self.star_zero_norm is not None:
            star_occ = (x1[:, 2:3] > self.star_zero_norm + 0.1).float()
            weights = torch.ones_like(per_voxel)
            weights[:, 2:3] = 1.0 + (self.star_occ_weight - 1.0) * star_occ
            per_voxel = per_voxel * weights

        return per_voxel.mean()

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        large_scale: torch.Tensor,
        params: torch.Tensor,
        n_steps: int = 50,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        self.model.eval()
        bsz = condition.shape[0]
        device = condition.device

        x = self._x0_from_dmo(condition)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((bsz,), i * dt, device=device)
            inp = torch.cat([x, condition, large_scale], dim=1)

            if cfg_scale != 1.0:
                v_cond = self.model(inp, t, params)
                v_uncond = self.model(inp, t, torch.zeros_like(params))
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.model(inp, t, params)

            x = x + v * dt

        return x


class FlowMatching3D:
    """Optimal-transport conditional flow matching in 3D."""

    def __init__(
        self,
        model: nn.Module,
        cfg_dropout: float = 0.0,
        star_occ_weight: float = 5.0,
        star_zero_norm: float | None = None,
    ):
        self.model = model
        self.cfg_dropout = cfg_dropout
        self.star_occ_weight = star_occ_weight
        self.star_zero_norm = star_zero_norm

    def loss(
        self,
        x1: torch.Tensor,
        condition: torch.Tensor,
        large_scale: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device)
        t_view = _expand_t(t, x1.ndim)
        noise = torch.randn_like(x1)

        x_t = (1 - t_view) * noise + t_view * x1
        velocity_target = x1 - noise

        if self.cfg_dropout > 0 and self.model.training:
            mask = torch.rand(bsz, device=x1.device) < self.cfg_dropout
            params = params.clone()
            params[mask] = 0.0

        model_input = torch.cat([x_t, condition, large_scale], dim=1)
        v_pred = self.model(model_input, t, params)

        per_voxel = (v_pred - velocity_target) ** 2

        if self.star_occ_weight != 1.0 and self.star_zero_norm is not None:
            star_occ = (x1[:, 2:3] > self.star_zero_norm + 0.1).float()
            weights = torch.ones_like(per_voxel)
            weights[:, 2:3] = 1.0 + (self.star_occ_weight - 1.0) * star_occ
            per_voxel = per_voxel * weights

        return per_voxel.mean()

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        large_scale: torch.Tensor,
        params: torch.Tensor,
        n_steps: int = 50,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        self.model.eval()
        bsz = condition.shape[0]
        device = condition.device
        spatial = condition.shape[2:]

        x = torch.randn(bsz, 3, *spatial, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((bsz,), i * dt, device=device)
            inp = torch.cat([x, condition, large_scale], dim=1)

            if cfg_scale != 1.0:
                v_cond = self.model(inp, t, params)
                v_uncond = self.model(inp, t, torch.zeros_like(params))
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.model(inp, t, params)

            x = x + v * dt

        return x
