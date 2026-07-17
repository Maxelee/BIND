"""Numpy-only inference for the WP-A4 gas-observable emulator.

``GasEmulator`` maps the 30 SB35 astro parameters (cosmology fixed at the
TNG fiducial) and a snapshot to the painted gas observables of the A3 gate
frame: per-mass-bin cylindrical f_gas medians and scatters, the Y-M relation
(slope, normalization, intrinsic scatter), and stacked tau_CAP kSZ profiles
in two mass bins. Direct port of the ``bind.wlemu`` construction (per-plane
PCA + one exact ARD Matern-5/2 GP per PCA coefficient, plain-numpy
posterior); fitting lives in :mod:`fit` (needs gpytorch).

Predictions are in the *painted-observable* frame — apply :mod:`forward`
for anything compared to data (CylToSph, kSZ normalization/decorrelation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import params_meta as pm

DEFAULT_ARTIFACT = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator/gasemu_gp.npz")

LOG_BLOCKS = {"fgas_med", "fgas_scat", "ksz0", "ksz1"}  # strictly positive, emulated as log10

_SQRT5 = np.sqrt(5.0)


def _matern25(dist: np.ndarray) -> np.ndarray:
    """Matern-5/2 correlation as a function of the ARD-scaled distance."""
    return (1.0 + _SQRT5 * dist + (5.0 / 3.0) * dist**2) * np.exp(-_SQRT5 * dist)


class _SnapModel:
    """Exact-GP posterior for one snapshot, rebuilt from stored
    hyperparameters (per-PCA-coefficient batch of ARD Matern-5/2 GPs)."""

    def __init__(self, X: np.ndarray, d: dict):
        self.X = X                                     # (n, 30) train inputs
        self.valid = d["valid"].astype(bool)           # (D_tot,) emulated dims
        self.components = d["pca_components"]          # (P, Dv)
        self.pca_mean = d["pca_mean"]                  # (Dv,)
        self.c_sd = d["c_sd"]                          # (P,)
        self.ls = d["lengthscale"]                     # (P, 30)
        self.outputscale = d["outputscale"]            # (P,)
        self.noise = d["noise"]                        # (P,)
        self.mean_const = d["mean_const"]              # (P,)
        self.alpha = d["alpha"]                        # (P, n) = K^-1 (y - m)
        self.y_mu = d["y_mu"]                          # (Dv,) transform means
        self.y_sd = d["y_sd"]                          # (Dv,)
        self._chol = None

    def _cholesky(self) -> np.ndarray:
        if self._chol is None:
            P, n = self.alpha.shape
            L = np.empty((P, n, n))
            for b in range(P):
                diff = (self.X[:, None, :] - self.X[None, :, :]) / self.ls[b]
                K = self.outputscale[b] * _matern25(np.sqrt((diff**2).sum(-1)))
                K[np.diag_indices(n)] += self.noise[b]
                L[b] = np.linalg.cholesky(K)
            self._chol = L
        return self._chol

    def posterior(self, Xq: np.ndarray, return_std: bool, chunk: int = 512):
        """Posterior mean (and predictive std) of the scaled PCA coefficients
        at query points ``Xq (m, 30)`` -> ``(m, P)`` arrays."""
        P, _ = self.alpha.shape
        m = len(Xq)
        mean = np.empty((m, P))
        std = np.empty((m, P)) if return_std else None
        L = self._cholesky() if return_std else None
        for i0 in range(0, m, chunk):
            q = Xq[i0:i0 + chunk]
            for b in range(P):
                diff = (q[:, None, :] - self.X[None, :, :]) / self.ls[b]
                k_star = self.outputscale[b] * _matern25(np.sqrt((diff**2).sum(-1)))
                mean[i0:i0 + chunk, b] = self.mean_const[b] + k_star @ self.alpha[b]
                if return_std:
                    v = np.linalg.solve(L[b], k_star.T)
                    var_b = self.outputscale[b] + self.noise[b] - (v**2).sum(0)
                    std[i0:i0 + chunk, b] = np.sqrt(np.maximum(var_b, 0.0))
        return mean, std


class GasEmulator:
    """Predict painted gas observables from the 30 SB35 astro parameters.

    Inputs: unit-cube arrays ``(30,)``/``(N, 30)`` or dicts of physical
    values keyed by parameter name (missing entries -> TNG fiducial).
    ``snap`` is one of the emulated snapshot labels (:attr:`snaps`).
    """

    def __init__(self, arrays: dict):
        self._raw = arrays
        self.provenance = json.loads(str(arrays["provenance"]))
        self.block_names = [str(b) for b in arrays["block_names"]]
        sizes = np.asarray(arrays["block_sizes"], int)
        offs = np.concatenate([[0], np.cumsum(sizes)])
        self.block_slices = {b: slice(int(offs[i]), int(offs[i + 1]))
                             for i, b in enumerate(self.block_names)}
        self.n_obs = int(offs[-1])

        self.snaps = [str(s) for s in arrays["snaps"]]
        self.snap_z = {s: float(z) for s, z in zip(self.snaps, arrays["snap_z"])}
        self.radii_arcmin = np.asarray(arrays["radii_arcmin"], float)
        self.logm500_bin_edges = np.asarray(arrays["logm500_bin_edges"], float)

        self.param_names = [str(p) for p in arrays["param_names"]]
        self._name_to_idx = {n: i for i, n in enumerate(self.param_names)}
        self.n_params = len(self.param_names)

        X = np.asarray(arrays["X_train"], float)
        self._models: dict[str, _SnapModel] = {}
        for s in self.snaps:
            d = {k.split("_", 1)[1]: np.asarray(arrays[k])
                 for k in arrays if k.startswith(f"snap{s}_")}
            self._models[s] = _SnapModel(X, d)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT) -> "GasEmulator":
        with np.load(Path(path), allow_pickle=False) as f:
            return cls({k: f[k] for k in f.files})

    # ---------------------------------------------------------- parameters --

    def _resolve_params(self, params) -> tuple[np.ndarray, bool]:
        if isinstance(params, dict):
            phys = pm.ASTRO_FIDUCIAL.copy()
            for name, val in params.items():
                if name not in self._name_to_idx:
                    raise KeyError(f"unknown parameter '{name}'. Valid: {self.param_names}")
                phys[self._name_to_idx[name]] = float(val)
            return pm.astro_physical_to_unit(phys)[None], True
        u = np.asarray(params, float)
        single = u.ndim == 1
        u = np.atleast_2d(u)
        if u.shape[1] != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {u.shape[1]}")
        if (u < -0.05).any() or (u > 1.05).any():
            import warnings
            warnings.warn("unit-cube parameters outside [0, 1]: extrapolating "
                          "beyond the training box", stacklevel=3)
        return u, single

    # ----------------------------------------------------------- prediction --

    def predict_vector(self, params, snap: str, return_std: bool = True):
        """Full observable vector ``(N, D_tot)`` in physical units, NaN at
        dims not emulated at this snapshot (empty high-mass bins)."""
        u, single = self._resolve_params(params)
        if snap not in self._models:
            raise KeyError(f"snap {snap!r} not emulated; have {self.snaps}")
        m = self._models[snap]

        mean_s, std_s = m.posterior(u, return_std)
        C = mean_s * m.c_sd
        Y_std = C @ m.components + m.pca_mean
        Y_t = Y_std * m.y_sd + m.y_mu                     # transformed frame

        n = len(u)
        vec = np.full((n, self.n_obs), np.nan)
        sd_full = np.full((n, self.n_obs), np.nan) if return_std else None
        if return_std:
            sd_C = std_s * m.c_sd
            sd_t = np.sqrt((sd_C**2) @ (m.components**2)) * m.y_sd

        for b in self.block_names:
            s = self.block_slices[b]
            bm = m.valid[s]                                # per-block valid mask
            cols = np.flatnonzero(m.valid)                 # transformed-frame index
            # map block dims -> positions in the compact (valid-only) vector
            pos = {c: j for j, c in enumerate(cols)}
            idx = [pos[c] for c in range(s.start, s.stop) if m.valid[c]]
            vals = Y_t[:, idx]
            if b in LOG_BLOCKS:
                out_vals = 10.0 ** vals
            else:
                out_vals = vals
            vec[:, np.flatnonzero(bm) + s.start] = out_vals
            if return_std:
                sd = sd_t[:, idx]
                if b in LOG_BLOCKS:
                    sd = np.abs(out_vals) * np.log(10.0) * sd
                sd_full[:, np.flatnonzero(bm) + s.start] = sd
        if single:
            vec = vec[0]
            if return_std:
                sd_full = sd_full[0]
        return (vec, sd_full) if return_std else vec

    def predict(self, params, snap: str, return_std: bool = True) -> dict:
        """Per-block dict (``fgas_med``, ``fgas_scat``, ``ym``, ``ksz0``,
        ``ksz1``; plus ``<block>_std`` when requested)."""
        res = self.predict_vector(params, snap, return_std=return_std)
        vec, sd = res if return_std else (res, None)
        out = {}
        for b in self.block_names:
            s = self.block_slices[b]
            out[b] = vec[..., s]
            if return_std:
                out[b + "_std"] = sd[..., s]
        return out
