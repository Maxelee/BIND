"""The BIND weak-lensing statistics emulator (numpy-only inference).

``WLEmulator`` maps a 30-dim astrophysical parameter vector (the CAMELS SB35
astro parameters; cosmology fixed at the IllustrisTNG fiducial) and a source
redshift to the full set of weak-lensing convergence summary statistics —
power spectrum, one-point PDF, peak counts, minima counts, Minkowski
functionals, wavelet-scattering coefficients, and amplitude moments — with
calibrated per-bin uncertainties, plus the statistic covariance of a single
5x5 deg field.

The training data are convergence maps raytraced through BIND-baryonified
IllustrisTNG-DMO lightcones at 253 Sobol points of the 30-dim astro parameter
space (50 map realizations each, 5 source redshifts). All statistics are
measured from the *same* maps, and each redshift's statistics vector is
emulated jointly (one PCA + GP posterior), so predictions are mutually
self-consistent: the joint response across statistics is the response of the
underlying maps, not of independently fitted curves.

Design: per source redshift, PCA compression of the standardized statistics
vector followed by one exact Gaussian process per PCA coefficient
(ARD Matern-5/2), the classic cosmology-emulator construction. Fitting lives
in :mod:`bind.wlemu.fit` (needs gpytorch); this module only needs numpy and
the artifact file, which bundles everything: GP hyperparameters, training
inputs, PCA bases, normalizations, parameter metadata, bin coordinates, and
covariance matrices.

Example
-------
>>> from bind.wlemu import WLEmulator
>>> emu = WLEmulator.load()                       # packaged artifact
>>> pred = emu.predict({"WindEnergyIn1e51erg": 7.2}, z_source=1.0)
>>> pred["Cl"], pred["Cl_std"], emu.ell
>>> cov = emu.covariance(z_source=1.0, blocks=("Cl", "peak"))
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BLOCKS = ("Cl", "pdf", "peak", "min", "V0", "V1", "V2", "scat", "moments")
LOG_BLOCKS = {"Cl", "scat"}          # emulated as log10
NONNEG_BLOCKS = {"pdf", "peak", "min", "V0", "V1"}  # clipped at 0 in predict
COORD_KEYS = ("ell", "pdf_x", "peak_x", "mink_thr")

DEFAULT_ARTIFACT = Path(__file__).parent.parent / "assets" / "wlemu_gp.npz"

_SQRT5 = np.sqrt(5.0)


def _matern25(dist: np.ndarray) -> np.ndarray:
    """Matern-5/2 correlation as a function of the ARD-scaled distance."""
    return (1.0 + _SQRT5 * dist + (5.0 / 3.0) * dist**2) * np.exp(-_SQRT5 * dist)


class _ZModel:
    """Exact-GP posterior for one source redshift, rebuilt from stored
    hyperparameters (per-PCA-coefficient batch of ARD Matern-5/2 GPs)."""

    def __init__(self, X: np.ndarray, d: dict):
        self.X = X                                     # (n, 30) train inputs
        self.components = d["pca_components"]          # (P, D)
        self.pca_mean = d["pca_mean"]                  # (D,)
        self.c_sd = d["c_sd"]                          # (P,)
        self.ls = d["lengthscale"]                     # (P, 30)
        self.outputscale = d["outputscale"]            # (P,)
        self.noise = d["noise"]                        # (P,)
        self.mean_const = d["mean_const"]              # (P,)
        self.alpha = d["alpha"]                        # (P, n) = K^-1 (y - m)
        self.cov_phys = d.get("cov_phys")              # (D, D) or None
        self._chol = None                              # lazy, for variances

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
        at query points ``Xq (m, 30)`` -> ``(m, P)`` arrays. Queries are
        processed in chunks to bound the kernel-matrix memory."""
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
                    v = np.linalg.solve(L[b], k_star.T)          # (n, q); L lower
                    var_b = self.outputscale[b] + self.noise[b] - (v**2).sum(0)
                    std[i0:i0 + chunk, b] = np.sqrt(np.maximum(var_b, 0.0))
        return mean, std


class WLEmulator:
    """Predict WL convergence summary statistics from astro parameters.

    Load with :meth:`WLEmulator.load` (defaults to the packaged artifact).
    Inputs are the 30 SB35 astrophysical parameters, either on the unit cube
    (arrays) or as physical values (dicts keyed by parameter name; missing
    entries default to the IllustrisTNG fiducial). Source redshift must be one
    of :attr:`source_redshifts` (the raytraced source planes).
    """

    def __init__(self, arrays: dict):
        self._raw = arrays
        self.provenance = json.loads(str(arrays["provenance"]))
        self.block_names = [str(b) for b in arrays["block_names"]]
        sizes = np.asarray(arrays["block_sizes"], int)
        offs = np.concatenate([[0], np.cumsum(sizes)])
        self.block_slices = {b: slice(int(offs[i]), int(offs[i + 1]))
                             for i, b in enumerate(self.block_names)}
        self.n_stats = int(offs[-1])

        self.source_redshifts = np.asarray(arrays["source_redshifts"], float)
        self.fov_deg = float(arrays["fov_deg"])
        self.smoothing_arcmin = float(arrays["smoothing_arcmin"])
        self.n_real = int(arrays["n_real"])
        for k in COORD_KEYS:
            setattr(self, k, np.asarray(arrays[k], float))

        self.param_names = [str(p) for p in arrays["param_names"]]
        self.param_min = np.asarray(arrays["param_min"], float)
        self.param_max = np.asarray(arrays["param_max"], float)
        self.param_log = np.asarray(arrays["param_log"], bool)
        self.param_fiducial = np.asarray(arrays["param_fiducial"], float)
        self.n_params = len(self.param_names)
        self._name_to_idx = {n: i for i, n in enumerate(self.param_names)}

        self.y_mu = np.asarray(arrays["y_mu"], float)
        self.y_sd = np.asarray(arrays["y_sd"], float)
        X = np.asarray(arrays["X_train"], float)
        self._z_models = []
        for i in range(len(self.source_redshifts)):
            d = {k.split("_", 1)[1]: np.asarray(arrays[k], float)
                 for k in arrays if k.startswith(f"z{i}_")}
            self._z_models.append(_ZModel(X, d))

    # ------------------------------------------------------------- loading --

    @classmethod
    def load(cls, path: str | Path | None = None) -> "WLEmulator":
        """Load an emulator artifact (default: the packaged one)."""
        path = Path(path) if path is not None else DEFAULT_ARTIFACT
        if not path.exists():
            raise FileNotFoundError(
                f"emulator artifact not found: {path}. The packaged artifact ships "
                "with the repository; to build one from a statistics cache see "
                "bind.wlemu.fit.")
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        return cls(arrays)

    # ---------------------------------------------------------- parameters --

    def param_table(self):
        """Parameter metadata as a pandas DataFrame (name, bounds, log flag,
        fiducial, description)."""
        import pandas as pd
        desc = [str(s) for s in self._raw.get("param_descriptions", [""] * self.n_params)]
        return pd.DataFrame({
            "ParamName": self.param_names, "MinVal": self.param_min,
            "MaxVal": self.param_max, "LogFlag": self.param_log.astype(int),
            "FiducialVal": self.param_fiducial, "Description": desc,
        })

    def fiducial_params(self, physical: bool = False) -> np.ndarray:
        """The IllustrisTNG fiducial astro parameters (unit cube by default)."""
        if physical:
            return self.param_fiducial.copy()
        return self.physical_to_unit(self.param_fiducial)

    def physical_to_unit(self, params: np.ndarray) -> np.ndarray:
        """Map physical parameter values ``(..., 30)`` to the unit cube."""
        p = np.asarray(params, float)
        lo, hi = self.param_min, self.param_max
        with np.errstate(divide="ignore", invalid="ignore"):
            u_log = (np.log10(p) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        u_lin = (p - lo) / (hi - lo)
        return np.where(self.param_log, u_log, u_lin)

    def unit_to_physical(self, u: np.ndarray) -> np.ndarray:
        """Map unit-cube values ``(..., 30)`` to physical parameter values."""
        u = np.asarray(u, float)
        lo, hi = self.param_min, self.param_max
        with np.errstate(divide="ignore", invalid="ignore"):
            p_log = 10.0 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))
        p_lin = lo + u * (hi - lo)
        return np.where(self.param_log, p_log, p_lin)

    def _resolve_params(self, params) -> tuple[np.ndarray, bool]:
        """Accept a unit-cube array (30,)/(N, 30) or a dict of physical values
        (missing -> fiducial). Returns (unit-cube (N, 30), was_single)."""
        if isinstance(params, dict):
            phys = self.param_fiducial.copy()
            for name, val in params.items():
                if name not in self._name_to_idx:
                    raise KeyError(f"unknown parameter '{name}'. "
                                   f"Valid names: {self.param_names}")
                phys[self._name_to_idx[name]] = float(val)
            return self.physical_to_unit(phys)[None], True
        u = np.asarray(params, float)
        single = u.ndim == 1
        u = np.atleast_2d(u)
        if u.shape[1] != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {u.shape[1]}")
        if (u < -0.05).any() or (u > 1.05).any():
            import warnings
            warnings.warn("some unit-cube parameters lie outside [0, 1]; the "
                          "emulator is extrapolating beyond its training box",
                          stacklevel=3)
        return u, single

    def _resolve_z(self, z_source: float | None, z_idx: int | None) -> int:
        if (z_source is None) == (z_idx is None):
            raise ValueError("pass exactly one of z_source= or z_idx=")
        if z_idx is not None:
            if not 0 <= int(z_idx) < len(self.source_redshifts):
                raise IndexError(f"z_idx {z_idx} out of range")
            return int(z_idx)
        diff = np.abs(self.source_redshifts - float(z_source))
        i = int(np.argmin(diff))
        if diff[i] > 1e-3:
            raise ValueError(
                f"z_source={z_source} is not one of the emulated source planes "
                f"{self.source_redshifts.tolist()}; pass one of those (the maps "
                "were raytraced at these discrete source redshifts)")
        return i

    # ----------------------------------------------------------- prediction --

    def predict(self, params, z_source: float | None = None, z_idx: int | None = None,
                return_std: bool = True, clip: bool = True) -> dict:
        """Predict all statistic blocks in physical units.

        Parameters
        ----------
        params : (30,) or (N, 30) unit-cube array, or dict of physical values
            (missing parameters default to the fiducial).
        z_source / z_idx : one of the emulated source planes (value or index).
        return_std : also return per-bin GP uncertainties as ``<block>_std``.
        clip : clip count-like blocks (pdf, peak, min, V0, V1) at zero.

        Returns
        -------
        dict with one array per block (single input -> (D,), batched ->
        (N, D)), plus ``<block>_std`` if requested. Bin coordinates live on
        the emulator (:attr:`ell`, :attr:`pdf_x`, :attr:`peak_x`,
        :attr:`mink_thr`).
        """
        u, single = self._resolve_params(params)
        zi = self._resolve_z(z_source, z_idx)
        m = self._z_models[zi]

        mean_s, std_s = m.posterior(u, return_std)          # (N, P) scaled PCs
        C = mean_s * m.c_sd
        Y_std = C @ m.components + m.pca_mean
        Y_t = Y_std * self.y_sd + self.y_mu

        Yp = Y_t.copy()
        for b in LOG_BLOCKS:
            s = self.block_slices[b]
            Yp[:, s] = 10.0 ** Y_t[:, s]

        out = {}
        if return_std:
            sd_C = std_s * m.c_sd
            Y_std_sd = np.sqrt((sd_C**2) @ (m.components**2))
            sd_t = Y_std_sd * self.y_sd
        for b in self.block_names:
            s = self.block_slices[b]
            v = Yp[:, s]
            if clip and b in NONNEG_BLOCKS:
                v = np.maximum(v, 0.0)
            out[b] = v[0] if single else v
            if return_std:
                if b in LOG_BLOCKS:
                    sd = np.abs(Yp[:, s]) * np.log(10.0) * sd_t[:, s]
                else:
                    sd = sd_t[:, s]
                out[b + "_std"] = sd[0] if single else sd
        return out

    def predict_vector(self, params, z_source: float | None = None,
                       z_idx: int | None = None, return_std: bool = True,
                       clip: bool = True):
        """Like :meth:`predict` but returns the concatenated statistics vector
        ``(N, D)`` (and its std), aligned with :attr:`block_slices` — the shape
        needed for likelihoods against :meth:`covariance`."""
        out = self.predict(params, z_source, z_idx, return_std=return_std, clip=clip)
        vec = np.concatenate([np.atleast_2d(out[b]) for b in self.block_names], axis=1)
        if not return_std:
            return vec
        sd = np.concatenate([np.atleast_2d(out[b + "_std"]) for b in self.block_names],
                            axis=1)
        return vec, sd

    def covariance(self, z_source: float | None = None, z_idx: int | None = None,
                   blocks: tuple | list | None = None) -> np.ndarray:
        """Statistic covariance of a **single 5x5 deg field** (physical units),
        estimated from the 50 map realizations per parameter point and averaged
        over parameter points.

        ``blocks`` selects a subset (e.g. ``("Cl", "peak")``) and returns the
        corresponding joint sub-matrix, rows/cols ordered like the blocks'
        concatenation. Estimated from ``n_real=50`` fields — restrict to a
        data-vector subset well below 50 dimensions before inverting (and
        apply your favorite Hartlap-style correction).
        """
        zi = self._resolve_z(z_source, z_idx)
        cov = self._z_models[zi].cov_phys
        if cov is None:
            raise ValueError("this artifact carries no covariance matrices")
        if blocks is None:
            return cov.copy()
        idx = np.concatenate([np.arange(self.n_stats)[self.block_slices[b]]
                              for b in blocks])
        return cov[np.ix_(idx, idx)]

    def coords_for(self, block: str) -> np.ndarray:
        """Bin coordinates for a block (ell for Cl; S/N bin centers/thresholds
        for pdf/peak/min/V*; coefficient index for scat/moments)."""
        return {
            "Cl": self.ell, "pdf": self.pdf_x, "peak": self.peak_x,
            "min": self.peak_x, "V0": self.mink_thr, "V1": self.mink_thr,
            "V2": self.mink_thr,
            "scat": np.arange(self.block_slices["scat"].stop
                              - self.block_slices["scat"].start),
            "moments": np.arange(4),
        }[block]
