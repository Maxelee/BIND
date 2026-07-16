"""Per-statistic compression: transform → impute → standardize → PCA.

Each statistic block (e.g. ``cl_kappa`` ``(N, 5, 5, 724)``) is flattened to
``(N, F)`` and reduced to a handful of PCA coefficients that the regression
backends actually learn.  The :class:`StatCompressor` is the invertible bridge
between physical statistic space and the latent the backend predicts, and it
records the PCA-truncation residual as a per-feature noise floor so predictive
errors never claim to be tighter than the compression allows.

Transforms (set per statistic in ``dataset.STAT_SPECS``):

* ``log``   — positive quantities (power spectra, Y, T): ``log10(clip(y))``.
* ``log1p`` — counts / PDFs (sign-preserving): ``sign(y) log10(1+|y|)``.
* ``raw``   — signed quantities (cross-spectra, Minkowski, moments): identity.

Non-finite entries (sparse mass bins, the odd divide-by-zero) are imputed with the
training per-feature median, so a statistic with a few empty bins still compresses
cleanly.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-30


def _forward(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.log10(np.clip(y, _TINY, None))
    if transform == "log1p":
        return np.sign(y) * np.log10(1.0 + np.abs(y))
    return y


def _inverse(x: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return 10.0 ** x
    if transform == "log1p":
        return np.sign(x) * (10.0 ** np.abs(x) - 1.0)
    return x


class StatCompressor:
    """Invertible transform + standardize + PCA for one statistic block."""

    def __init__(self, transform: str = "raw", n_components: int = 20,
                 var_threshold: float = 0.9999):
        self.transform = transform
        self.n_components = n_components
        self.var_threshold = var_threshold
        self.shape: tuple[int, ...] = ()
        self.fill_: np.ndarray | None = None       # (F,) per-feature impute value
        self.mean_: np.ndarray | None = None       # (F,)
        self.std_: np.ndarray | None = None        # (F,)
        self.components_: np.ndarray | None = None  # (k, F)
        self.k = 0
        self.resid_std_: np.ndarray | None = None  # (F,) truncation residual floor

    # ── fit / transform ───────────────────────────────────────────────────────
    def fit(self, Y: np.ndarray) -> "StatCompressor":
        Y = np.asarray(Y, dtype=float)
        self.shape = Y.shape[1:]
        Xf = _forward(Y.reshape(len(Y), -1), self.transform)         # (N, F)
        # impute non-finite with per-feature median over finite entries
        import warnings
        with warnings.catch_warnings():                              # all-NaN columns
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.fill_ = np.nanmedian(np.where(np.isfinite(Xf), Xf, np.nan), axis=0)
        self.fill_ = np.where(np.isfinite(self.fill_), self.fill_, 0.0)
        Xf = np.where(np.isfinite(Xf), Xf, self.fill_)
        self.mean_ = Xf.mean(0)
        self.std_ = Xf.std(0)
        self.std_ = np.where(self.std_ > _TINY, self.std_, 1.0)
        Xs = (Xf - self.mean_) / self.std_
        # PCA via SVD (centered already).  Cap components both by n_components and
        # by cumulative explained variance, so we keep the signal-carrying PCs and
        # don't hand the regressor a tail of pure realization-noise components
        # (which is what makes an MLP collapse to the mean).
        kmax = int(max(1, min(self.n_components, Xs.shape[0] - 1, Xs.shape[1])))
        _, sv, Vt = np.linalg.svd(Xs, full_matrices=False)
        cumvar = np.cumsum(sv ** 2) / (np.sum(sv ** 2) + _TINY)
        k = int(min(kmax, np.searchsorted(cumvar, self.var_threshold) + 1))
        k = max(k, 1)
        self.components_ = Vt[:k]                                     # (k, F)
        self.k = k
        # truncation residual (per-feature std of reconstruction error)
        Z = Xs @ self.components_.T
        recon = Z @ self.components_
        self.resid_std_ = (Xs - recon).std(0)
        return self

    def transform_to_latent(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        Xf = _forward(Y.reshape(len(Y), -1), self.transform)
        Xf = np.where(np.isfinite(Xf), Xf, self.fill_)
        Xs = (Xf - self.mean_) / self.std_
        return Xs @ self.components_.T                               # (N, k)

    # ── inverse ───────────────────────────────────────────────────────────────
    def inverse_from_latent(self, Z: np.ndarray, Z_std: np.ndarray | None = None
                            ) -> tuple[np.ndarray, np.ndarray | None]:
        """Latent (+ optional latent std) → physical statistic (+ 1σ)."""
        Z = np.asarray(Z, dtype=float)
        Xs = Z @ self.components_                                     # (m, F)
        Xf = Xs * self.std_ + self.mean_
        Y = _inverse(Xf, self.transform).reshape((-1, *self.shape))
        err = None
        if Z_std is not None:
            # latent error → feature error in std units, plus the truncation floor
            var = (np.asarray(Z_std, dtype=float) ** 2) @ (self.components_ ** 2)
            std_units = np.sqrt(var + self.resid_std_ ** 2)
            Xf_hi = Xf + std_units * self.std_
            Y_hi = _inverse(Xf_hi, self.transform)
            err = np.abs(Y_hi - _inverse(Xf, self.transform)).reshape((-1, *self.shape))
        return Y, err

    # ── (de)serialization ─────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {
            "transform": self.transform, "n_components": self.n_components,
            "shape": np.array(self.shape), "fill": self.fill_, "mean": self.mean_,
            "std": self.std_, "components": self.components_, "k": self.k,
            "resid_std": self.resid_std_,
        }

    @classmethod
    def from_state(cls, s: dict) -> "StatCompressor":
        c = cls(transform=str(s["transform"]), n_components=int(s["n_components"]))
        c.shape = tuple(int(x) for x in s["shape"])
        c.fill_, c.mean_, c.std_ = s["fill"], s["mean"], s["std"]
        c.components_, c.resid_std_ = s["components"], s["resid_std"]
        c.k = int(s["k"])
        return c
