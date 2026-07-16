"""``WLEmulator`` — load a trained checkpoint and generate κ maps from params.

    >>> import bind
    >>> from bind.wlemu import WLEmulator
    >>> em = WLEmulator.load("runs/fm_kappa1024/best.pt")
    >>> kappa = em.generate(bind.fiducial_params(), z_s=1.0, n=64)   # (64, R, R) raw κ

The checkpoint is self-contained: it carries the arch config, the EMA weights,
and the per-z κ normalisation, so no cache or training data is needed at predict
time.  ``z_s`` may be any value spanned by the training source planes — the
normalisation (λ, μ, σ) is linearly interpolated in ``z_s`` and the model is
conditioned on the continuous scale factor ``a = 1/(1+z_s)``.
"""

from __future__ import annotations

import numpy as np
import torch

from bind.emulator.dataset import params_to_unit
from bind.wlemu.model import FieldCFM, FieldUNet

__all__ = ["WLEmulator"]


class WLEmulator:
    def __init__(self, model, norm, source_redshifts, param_names, config):
        self.model = model
        self.fm = FieldCFM(model, cfg_dropout=config.get("cfg_dropout", 0.0))
        self.norm = norm                      # dict: lam, mu, sigma (len n_z)
        self.source_redshifts = np.asarray(source_redshifts, dtype=float)
        self.param_names = list(param_names)
        self.config = config
        self.resolution = config["resolution"]
        self.device = next(model.parameters()).device

    # ── construction ──────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path, device=None, weights="ema"):
        """Load from a ``.pt`` checkpoint. ``weights`` ∈ {"ema", "model"}."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = FieldUNet(
            resolution=cfg["resolution"], base_ch=cfg["base_ch"],
            n_blocks=cfg["n_blocks"], emb_dim=cfg["emb_dim"],
            n_params=cfg["n_params"], condition_redshift=cfg["condition_redshift"],
            ch_mult=cfg.get("ch_mult"))
        state = ckpt.get(weights) or ckpt["model"]
        model.load_state_dict(state)
        model.to(device).eval()
        return cls(model, ckpt["norm"], ckpt["source_redshifts"],
                   ckpt.get("param_names", []), cfg)

    # ── normalisation interpolation ───────────────────────────────────────────
    def _interp_norm(self, z_s):
        zs = self.source_redshifts
        lam = np.interp(z_s, zs, self.norm["lam"])
        mu = np.interp(z_s, zs, self.norm["mu"])
        sigma = np.interp(z_s, zs, self.norm["sigma"])
        return float(lam), float(mu), float(sigma)

    def _unit_params(self, params, params_are_unit):
        params = np.asarray(params, dtype=float)
        if params.ndim == 1:
            params = params[None]
        u = params if params_are_unit else params_to_unit(params)  # → (...,30)
        return np.asarray(u, dtype=np.float32)

    # ── generation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, params, z_s=1.0, n=16, n_steps=50, cfg_scale=1.0,
                 seed=None, batch_size=None, params_are_unit=False,
                 return_norm=False):
        """Generate ``n`` κ maps for a single parameter point and source redshift.

        params : native SB35 vector (35,) or astro-only (30,) — or unit-cube if
            ``params_are_unit=True``.
        Returns ``(n, R, R)`` raw κ (float32); set ``return_norm=True`` for the
        arcsinh-normalised field instead.
        """
        u = self._unit_params(params, params_are_unit)        # (1, 30)
        if u.shape[0] != 1:
            raise ValueError("generate() takes one parameter point; use generate_batch")
        theta = torch.from_numpy(np.repeat(u, n, axis=0)).to(self.device)
        a = 1.0 / (1.0 + float(z_s))
        sf = torch.full((n,), a, device=self.device)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        bs = batch_size or n
        out = []
        for i in range(0, n, bs):
            sl = slice(i, min(i + bs, n))
            x = self.fm.sample(theta[sl], scale_factor=sf[sl], n_steps=n_steps,
                               cfg_scale=cfg_scale, generator=gen)
            out.append(x[:, 0].float().cpu().numpy())
        xnorm = np.concatenate(out, axis=0)                   # (n, R, R) normalised
        if return_norm:
            return xnorm
        lam, mu, sigma = self._interp_norm(z_s)
        return (lam * np.sinh(sigma * xnorm + mu)).astype(np.float32)

    @torch.no_grad()
    def generate_batch(self, params, z_s=1.0, n_steps=50, cfg_scale=1.0,
                       seed=None, params_are_unit=False, return_norm=False):
        """One map per parameter row. ``params`` is ``(B, 35|30)``; ``z_s`` is a
        scalar or length-``B`` array.  Returns ``(B, R, R)`` raw κ."""
        u = self._unit_params(params, params_are_unit)        # (B, 30)
        B = u.shape[0]
        z_s = np.broadcast_to(np.asarray(z_s, dtype=float), (B,))
        theta = torch.from_numpy(u).to(self.device)
        a = 1.0 / (1.0 + z_s)
        sf = torch.from_numpy(a.astype(np.float32)).to(self.device)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        x = self.fm.sample(theta, scale_factor=sf, n_steps=n_steps,
                           cfg_scale=cfg_scale, generator=gen)
        xnorm = x[:, 0].float().cpu().numpy()
        if return_norm:
            return xnorm
        lam = np.interp(z_s, self.source_redshifts, self.norm["lam"])
        mu = np.interp(z_s, self.source_redshifts, self.norm["mu"])
        sigma = np.interp(z_s, self.source_redshifts, self.norm["sigma"])
        return (lam[:, None, None] * np.sinh(
            sigma[:, None, None] * xnorm + mu[:, None, None])).astype(np.float32)
