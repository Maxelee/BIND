"""U-Net + Flow Matching for conditional baryonic field generation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- building blocks ----------

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class ParamEncoder(nn.Module):
    """Encode 35-dim cosmological parameters into embedding space."""
    def __init__(self, n_params=35, emb_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, p):
        return self.net(p)


class AdaGroupNorm(nn.Module):
    """GroupNorm with adaptive scale and shift from conditioning embedding."""
    def __init__(self, channels, emb_dim, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.proj = nn.Linear(emb_dim, channels * 2)

    def forward(self, x, emb):
        x = self.norm(x)
        scale, shift = self.proj(emb)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class ResBlock(nn.Module):
    """Residual block with AdaGroupNorm conditioning."""
    def __init__(self, in_ch, out_ch, emb_dim, num_groups=32, dropout=0.0):
        super().__init__()
        self.norm1 = AdaGroupNorm(in_ch, emb_dim, num_groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGroupNorm(out_ch, emb_dim, num_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.act(self.norm1(x, emb))
        h = self.conv1(h)
        h = self.act(self.norm2(h, emb))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with GroupNorm."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ---------- U-Net ----------

class UNet(nn.Module):
    """U-Net for flow matching velocity prediction.

    Input:  noisy_target (3) + condition (1) + large_scale (3) = 7 channels
    Output: predicted velocity (3 channels)
    """

    def __init__(self, in_ch=7, out_ch=3, base_ch=128,
                 ch_mult=(1, 2, 4, 8), n_blocks=2, emb_dim=256,
                 attn_resolutions=(32, 16), dropout=0.0, n_params=35):
        super().__init__()

        # Time + param embeddings
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(base_ch),
            nn.Linear(base_ch, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.param_emb = ParamEncoder(n_params, emb_dim)

        # Input projection
        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [base_ch]
        ch = base_ch
        res = 128  # input resolution
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
                self.downsamples.append(Downsample(ch))
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
            for j in range(n_blocks + 1):  # +1 for skip connection processing
                skip_ch = channels.pop() if j == 0 else 0
                blocks.append(ResBlock(ch + skip_ch, out, emb_dim, dropout=dropout))
                if res in attn_resolutions:
                    blocks.append(SelfAttention(out))
                ch = out
            self.decoders.append(blocks)
            if i < len(ch_mult) - 1:
                self.upsamples.append(Upsample(ch))
                res *= 2
            else:
                self.upsamples.append(nn.Identity())

        # Output
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, params):
        """
        x: (B, 7, 128, 128) — concat of [noisy_target, condition, large_scale]
        t: (B,) — timestep in [0, 1]
        params: (B, 35) — normalized cosmological parameters
        """
        emb = self.time_emb(t) + self.param_emb(params)

        h = self.input_conv(x)

        # Encoder with skip connections
        skips = [h]
        for blocks, down in zip(self.encoders, self.downsamples):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, emb)
                else:
                    h = block(h)
            skips.append(h)
            h = down(h)

        # Bottleneck
        for block in self.mid:
            if isinstance(block, ResBlock):
                h = block(h, emb)
            else:
                h = block(h)

        # Decoder
        for blocks, up in zip(self.decoders, self.upsamples):
            for i, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    if i == 0:  # first ResBlock gets skip connection
                        h = torch.cat([h, skips.pop()], dim=1)
                    h = block(h, emb)
                else:
                    h = block(h)
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))


# ---------- Flow Matching ----------

class StochasticInterpolant:
    """Stochastic interpolant bridging DMO → hydro fields.

    Interpolant:     x_t = (1-t)*x0 + t*x1 + gamma(t)*z
    Velocity target: dx_t/dt = (x1 - x0) + sigma*(1 - 2t)*z

    where x0 = DMO field replicated to 3 channels, x1 = hydro target,
    z ~ N(0,I), and gamma(t) = sigma*t*(1-t) (zero at both endpoints).

    With sigma=0 this is a deterministic bridge from DMO to hydro.
    Sampling starts from x0 = DMO (no random noise needed).
    """

    def __init__(self, model, sigma=0.5, cfg_dropout=0.0,
                 star_occ_weight=5.0, star_zero_norm=None):
        self.model = model
        self.sigma = sigma
        self.cfg_dropout = cfg_dropout
        self.star_occ_weight = star_occ_weight
        self.star_zero_norm = star_zero_norm

    def _x0_from_dmo(self, condition):
        """Replicate 1-channel DMO to 3-channel starting point."""
        return condition.expand(-1, 3, -1, -1)

    def loss(self, x1, condition, large_scale, params):
        """Compute stochastic interpolant loss.

        Args:
            x1: (B, 3, H, W) — normalized target fields (hydro)
            condition: (B, 1, H, W) — normalized DMO condition
            large_scale: (B, 3, H, W) — normalized large-scale context
            params: (B, 35) — normalized parameters
        """
        B = x1.shape[0]
        t = torch.rand(B, device=x1.device)
        t4 = t[:, None, None, None]

        x0 = self._x0_from_dmo(condition)
        z = torch.randn_like(x1)

        # gamma(t) = sigma * t * (1-t), gamma_dot = sigma * (1 - 2t)
        x_t = (1 - t4) * x0 + t4 * x1 + self.sigma * t4 * (1 - t4) * z
        velocity_target = (x1 - x0) + self.sigma * (1 - 2 * t4) * z

        if self.cfg_dropout > 0 and self.model.training:
            mask = torch.rand(B, device=x1.device) < self.cfg_dropout
            params = params.clone()
            params[mask] = 0.0

        if large_scale is not None:
            model_input = torch.cat([x_t, condition, large_scale], dim=1)
        else:
            model_input = torch.cat([x_t, condition], dim=1)
        v_pred = self.model(model_input, t, params)

        per_pixel = (v_pred - velocity_target) ** 2

        if self.star_occ_weight != 1.0 and self.star_zero_norm is not None:
            star_occ = (x1[:, 2:3] > self.star_zero_norm + 0.1).float()
            w = torch.ones_like(per_pixel)
            w[:, 2:3] = 1.0 + (self.star_occ_weight - 1.0) * star_occ
            per_pixel = per_pixel * w

        return per_pixel.mean()

    def sample(self, condition, large_scale=None, params=None, n_steps=50, cfg_scale=1.0, grad=False):
        """Generate samples via Euler ODE integration starting from DMO.

        Args:
            condition: (B, 1, H, W)
            large_scale: (B, 3, H, W) or None when no large-scale conditioning
            params: (B, 35)
            n_steps: number of Euler steps
            cfg_scale: classifier-free guidance scale (1.0 = no guidance)
            grad: if True, enable gradients so d(output)/d(params) can be computed
        Returns:
            (B, 3, H, W) generated fields
        """
        self.model.eval()
        B = condition.shape[0]
        device = condition.device

        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            x = self._x0_from_dmo(condition)
            dt = 1.0 / n_steps

            for i in range(n_steps):
                t = torch.full((B,), i * dt, device=device)
                if large_scale is not None:
                    inp = torch.cat([x, condition, large_scale], dim=1)
                else:
                    inp = torch.cat([x, condition], dim=1)

                if cfg_scale != 1.0:
                    v_cond = self.model(inp, t, params)
                    v_uncond = self.model(inp, t, torch.zeros_like(params))
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = self.model(inp, t, params)

                x = x + v * dt

        return x


class FlowMatching:
    """Optimal-transport conditional flow matching.

    Forward: x_t = (1 - t) * noise + t * x_1   (linear interpolation)
    Velocity target: v = x_1 - noise
    Sampling: Euler ODE integration from t=0 (noise) to t=1 (data)
    """

    def __init__(self, model, cfg_dropout=0.0, star_occ_weight=5.0,
                 star_zero_norm=None, out_channels=3):
        self.model = model
        self.cfg_dropout = cfg_dropout
        self.star_occ_weight = star_occ_weight
        # Normalised value of a zero-density stellar pixel; anything above this is "occupied".
        # Set from norm_stats: (0 - target_mean[2]) / target_std[2].
        # In two-head Stars mode this knob is unused (channel 2 is occupancy,
        # not stars density) — keep star_occ_weight=1.0 in that case.
        self.star_zero_norm = star_zero_norm
        self.out_channels = out_channels

    def loss(self, x1, condition, large_scale, params):
        """Compute flow matching loss.

        Args:
            x1: (B, C, H, W) — normalized target fields (C = self.out_channels)
            condition: (B, 1, H, W) — normalized DMO condition
            large_scale: (B, 3, H, W) — normalized large-scale context
            params: (B, 35) — normalized parameters
        """
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

        if large_scale is not None:
            model_input = torch.cat([x_t, condition, large_scale], dim=1)
        else:
            model_input = torch.cat([x_t, condition], dim=1)
        v_pred = self.model(model_input, t, params)

        per_pixel = (v_pred - velocity_target) ** 2

        if self.star_occ_weight != 1.0:
            w = torch.ones_like(per_pixel)
            if self.out_channels == 4:
                # Two-head Stars mode: ch2=occupancy, ch3=conditional density.
                # Upweight both stellar channels on pixels that are truly occupied
                # (occupancy target > 0.5, i.e. the binary label is 1).
                star_occ = (x1[:, 2:3] > 0.5).float()
                w[:, 2:3] = 1.0 + (self.star_occ_weight - 1.0) * star_occ
                w[:, 3:4] = 1.0 + (self.star_occ_weight - 1.0) * star_occ
            elif self.star_zero_norm is not None:
                # Single-head mode: ch2 is stars density; upweight occupied pixels.
                star_occ = (x1[:, 2:3] > self.star_zero_norm + 0.1).float()
                w[:, 2:3] = 1.0 + (self.star_occ_weight - 1.0) * star_occ
            per_pixel = per_pixel * w

        return per_pixel.mean()

    def sample(self, condition, large_scale=None, params=None, n_steps=50, cfg_scale=1.0, grad=False):
        """Generate samples via Euler ODE integration.

        Args:
            condition: (B, 1, H, W)
            large_scale: (B, 3, H, W) or None when no large-scale conditioning
            params: (B, 35)
            n_steps: number of Euler steps
            cfg_scale: classifier-free guidance scale (1.0 = no guidance)
            grad: if True, enable gradients so d(output)/d(params) can be computed
        Returns:
            (B, self.out_channels, H, W) generated fields
        """
        self.model.eval()
        B = condition.shape[0]
        device = condition.device

        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            x = torch.randn(B, self.out_channels,
                            condition.shape[2], condition.shape[3], device=device)
            dt = 1.0 / n_steps

            for i in range(n_steps):
                t = torch.full((B,), i * dt, device=device)
                if large_scale is not None:
                    inp = torch.cat([x, condition, large_scale], dim=1)
                else:
                    inp = torch.cat([x, condition], dim=1)

                if cfg_scale != 1.0:
                    v_cond = self.model(inp, t, params)
                    v_uncond = self.model(inp, t, torch.zeros_like(params))
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = self.model(inp, t, params)

                x = x + v * dt

        return x
