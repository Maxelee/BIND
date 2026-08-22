"""Per-statistic compression: transform → impute → standardize → PCA.

Each statistic block (e.g. ``cl_kappa`` ``(N, 5, 5, 724)``) is flattened to
``(N, F)`` and reduced to a handful of PCA coefficients that the regression
backends actually learn.  The :class:`StatCompressor` is the invertible bridge
between physical statistic space and the latent the backend predicts, and it
records the PCA-truncation residual as a per-feature noise floor so predictive
errors never claim to be tighter than the compression allows.

Transforms (set per statistic in ``dataset.STAT_SPECS``, or overridden by
``Emulator.transform_overrides`` for heads whose serialized dataset transform
predates a convention change -- see ``emulator.core``):

* ``log``       — positive quantities (power spectra, Y, T): ``log10(clip(y))``.
* ``log1p``     — counts / PDFs (sign-preserving): ``sign(y) log10(1+|y|)``.
* ``raw``       — signed quantities (Minkowski, moments): identity.
* ``asinh_std`` — signed, wide-dynamic-range quantities (cross-spectra):
  ``asinh(y / s)`` with ``s`` a per-feature scale fit from the TRAIN block
  (1.4826 x MAD, falling back to std when MAD is degenerate).  Unlike the
  other three transforms this one carries fitted state (``s_``), so it is
  handled as a first-class citizen of :class:`StatCompressor` rather than a
  stateless function -- see the WHY note below.

WHY ``asinh_std`` exists (measured, not assumed --
``papers/01_pipeline/audits/spectrum_head_experiment.md``): the raw signed
cross-spectra (kappa-y, kappa-tau, y-tau) span >2 decades of |value| across
ell even after dropping the aliased tail, so a plain per-feature
center/scale (what every other transform gets for free after this module's
``_forward``) leaves the fit numerically degenerate -- median |frac err| on
the order of 1e7% in the controlled experiment.  ``asinh(y/s)`` compresses
that dynamic range while staying (anti)symmetric through zero (unlike
``log``, which requires positivity, or ``log1p``, which is calibrated for
count-like magnitudes near O(1)); the per-feature ``s`` sets the linear/log
crossover at the bin's own typical scatter, and the *subsequent* generic
center/scale this class always applies afterwards is exactly the
"``- mu_bin) / sig_bin``" of the write-up -- ``asinh_std`` only owns the
``s`` scale, not the standardization.

Non-finite entries (sparse mass bins, the odd divide-by-zero) are imputed with the
training per-feature median, so a statistic with a few empty bins still compresses
cleanly.

Per-feature masking (``mask=`` in :meth:`StatCompressor.fit`) excludes chosen
features from the transform/standardize/PCA pipeline entirely -- e.g. the
CIC/pixelization-contaminated ell > ELL_MASK_ABOVE tail of the released
spectra (``emulator.core.ELL_MASK_ABOVE``, 3.0e4 as of 2026-08-04, was 1.5e4
-- see that module for the measurement provenance), which the same
write-up shows pollutes the PCA and roughly doubles the held-out error of
the trusted bins if left in.  Masked features are never seen by the PCA or
the regression backend; :meth:`StatCompressor.inverse_from_latent` fills
them back in at the TRAIN mean (physical units) with an inflated error bar
(``mask_err_inflate`` x the TRAIN std, physical units) so a caller that
naively trusts the returned 1-sigma is warned off, not silently misled --
the full released grid shape is always returned (see ``emulator.core``'s
``Emulator.predict`` docstring for the public contract).
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-30


def _forward(y: np.ndarray, transform: str, s: np.ndarray | None = None) -> np.ndarray:
    if transform == "log":
        return np.log10(np.clip(y, _TINY, None))
    if transform == "log1p":
        return np.sign(y) * np.log10(1.0 + np.abs(y))
    if transform == "asinh_std":
        return np.arcsinh(y / s)
    return y


def _inverse(x: np.ndarray, transform: str, s: np.ndarray | None = None) -> np.ndarray:
    if transform == "log":
        return 10.0 ** x
    if transform == "log1p":
        return np.sign(x) * (10.0 ** np.abs(x) - 1.0)
    if transform == "asinh_std":
        return np.sinh(x) * s
    return x


def _nanmad_scale(Y: np.ndarray) -> np.ndarray:
    """Per-column 1.4826*MAD over ``Y`` ``(N, F)``, falling back to std when MAD==0.

    Matches the spectrum-head experiment's ``s_bin`` convention exactly
    (``spectrum_head_experiment.md`` §4(ii)/§5.2).  ``1.4826`` makes the MAD a
    consistent estimator of sigma for normally-distributed data; it is a
    convention choice (comparable scale to std for well-behaved columns), not
    a fitted parameter.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(Y, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(Y - med), axis=0)
        std = np.nanstd(Y, axis=0)
    s = np.where(mad > _TINY, mad, np.where(std > _TINY, std, 1.0))
    return np.where(np.isfinite(s), s, 1.0)


class StatCompressor:
    """Invertible transform + standardize + PCA for one statistic block.

    ``mask`` (passed to :meth:`fit`, not the constructor) is an optional
    boolean array broadcastable to the per-run shape (``Y.shape[1:]``); ``True``
    = kept (compressed/fit normally), ``False`` = excluded from the whole
    transform/PCA/backend pipeline and reconstructed at the TRAIN mean with an
    inflated error bar on ``inverse_from_latent``.  ``None`` (default) keeps
    every feature, reproducing the pre-masking behaviour exactly.
    """

    def __init__(self, transform: str = "raw", n_components: int = 20,
                 var_threshold: float = 0.9999, mask_err_inflate: float = 1e3):
        self.transform = transform
        self.n_components = n_components
        self.var_threshold = var_threshold
        self.mask_err_inflate = mask_err_inflate
        self.shape: tuple[int, ...] = ()
        self.fill_: np.ndarray | None = None       # (Fk,) per-feature impute value
        self.mean_: np.ndarray | None = None        # (Fk,)
        self.std_: np.ndarray | None = None          # (Fk,)
        self.components_: np.ndarray | None = None  # (k, Fk)
        self.k = 0
        self.resid_std_: np.ndarray | None = None  # (Fk,) truncation residual floor
        self.s_: np.ndarray | None = None            # (Fk,) asinh_std scale, else None
        # full-F (unmasked) bookkeeping, always populated, used for the
        # masked-passthrough reconstruction and to invert the mask indices:
        self.mask_: np.ndarray | None = None         # (F,) bool, True = kept
        self.kept_idx_: np.ndarray | None = None      # (Fk,) indices into flat F
        self.full_mean_: np.ndarray | None = None    # (F,) physical-space TRAIN mean
        self.full_std_: np.ndarray | None = None      # (F,) physical-space TRAIN std

    # ── mask bookkeeping ─────────────────────────────────────────────────────
    def _resolve_mask(self, F: int, mask) -> np.ndarray:
        if mask is None:
            return np.ones(F, dtype=bool)
        m = np.broadcast_to(np.asarray(mask, dtype=bool), self.shape).reshape(-1)
        if m.shape[0] != F:
            raise ValueError(f"mask flattens to {m.shape[0]} features, expected {F}")
        if not m.any():
            raise ValueError("mask excludes every feature of this statistic")
        return m

    # ── fit / transform ───────────────────────────────────────────────────────
    def fit(self, Y: np.ndarray, mask: np.ndarray | None = None) -> "StatCompressor":
        Y = np.asarray(Y, dtype=float)
        self.shape = Y.shape[1:]
        Yflat = Y.reshape(len(Y), -1)                                  # (N, F)
        F = Yflat.shape[1]

        # physical-space TRAIN mean/std for EVERY feature (masked passthrough
        # uses these regardless of whether the feature is kept or masked).
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.full_mean_ = np.nanmean(Yflat, axis=0)
            self.full_std_ = np.nanstd(Yflat, axis=0)
        self.full_mean_ = np.where(np.isfinite(self.full_mean_), self.full_mean_, 0.0)
        self.full_std_ = np.where(np.isfinite(self.full_std_) & (self.full_std_ > _TINY),
                                  self.full_std_, _TINY)

        self.mask_ = self._resolve_mask(F, mask)
        self.kept_idx_ = np.flatnonzero(self.mask_)
        Yk = Yflat[:, self.kept_idx_]                                  # (N, Fk)

        self.s_ = _nanmad_scale(Yk) if self.transform == "asinh_std" else None
        Xf = _forward(Yk, self.transform, self.s_)                     # (N, Fk)
        # impute non-finite with per-feature median over finite entries
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
        self.components_ = Vt[:k]                                     # (k, Fk)
        self.k = k
        # truncation residual (per-feature std of reconstruction error)
        Z = Xs @ self.components_.T
        recon = Z @ self.components_
        self.resid_std_ = (Xs - recon).std(0)
        return self

    def transform_to_latent(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        Yflat = Y.reshape(len(Y), -1)
        Yk = Yflat[:, self.kept_idx_]
        Xf = _forward(Yk, self.transform, self.s_)
        Xf = np.where(np.isfinite(Xf), Xf, self.fill_)
        Xs = (Xf - self.mean_) / self.std_
        return Xs @ self.components_.T                               # (N, k)

    # ── inverse ───────────────────────────────────────────────────────────────
    def inverse_from_latent(self, Z: np.ndarray, Z_std: np.ndarray | None = None
                            ) -> tuple[np.ndarray, np.ndarray | None]:
        """Latent (+ optional latent std) → physical statistic (+ 1σ).

        Reconstructs the FULL released shape (``self.shape``): kept features
        come from the PCA/backend as before; masked features (if any) are
        filled with the physical-space TRAIN mean, and -- when an error is
        requested -- an inflated 1sigma (``mask_err_inflate`` x the physical
        TRAIN std) so the untrusted bins never look more confident than the
        trusted ones. See the module docstring / ``emulator.core`` for the
        public-contract framing ("untrusted aliased bins carry no response;
        returned at train-mean with inflated err").
        """
        Z = np.asarray(Z, dtype=float)
        N = Z.shape[0]
        Xs = Z @ self.components_                                     # (N, Fk)
        Xf = Xs * self.std_ + self.mean_
        Yk = _inverse(Xf, self.transform, self.s_)                    # (N, Fk) physical

        Yfull = np.tile(self.full_mean_, (N, 1))                      # (N, F) default
        Yfull[:, self.kept_idx_] = Yk

        err = None
        if Z_std is not None:
            # latent error → feature error in std units, plus the truncation floor
            var = (np.asarray(Z_std, dtype=float) ** 2) @ (self.components_ ** 2)
            std_units = np.sqrt(var + self.resid_std_ ** 2)
            Xf_hi = Xf + std_units * self.std_
            Yk_hi = _inverse(Xf_hi, self.transform, self.s_)
            errk = np.abs(Yk_hi - Yk)                                  # (N, Fk)
            err_full = np.tile(self.mask_err_inflate * self.full_std_, (N, 1))
            err_full[:, self.kept_idx_] = errk
            err = err_full.reshape((-1, *self.shape))

        Y = Yfull.reshape((-1, *self.shape))
        return Y, err

    # ── (de)serialization ─────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {
            "transform": self.transform, "n_components": self.n_components,
            "mask_err_inflate": self.mask_err_inflate,
            "shape": np.array(self.shape), "fill": self.fill_, "mean": self.mean_,
            "std": self.std_, "components": self.components_, "k": self.k,
            "resid_std": self.resid_std_, "s": self.s_,
            "mask": self.mask_, "kept_idx": self.kept_idx_,
            "full_mean": self.full_mean_, "full_std": self.full_std_,
        }

    @classmethod
    def from_state(cls, s: dict) -> "StatCompressor":
        c = cls(transform=str(s["transform"]), n_components=int(s["n_components"]),
                mask_err_inflate=float(s.get("mask_err_inflate", 1e3)))
        c.shape = tuple(int(x) for x in s["shape"])
        c.fill_, c.mean_, c.std_ = s["fill"], s["mean"], s["std"]
        c.components_, c.resid_std_ = s["components"], s["resid_std"]
        c.k = int(s["k"])
        c.s_ = s.get("s")
        F = int(np.prod(c.shape)) if c.shape else 1
        c.mask_ = s.get("mask")
        if c.mask_ is None:
            c.mask_ = np.ones(F, dtype=bool)
        c.kept_idx_ = s.get("kept_idx")
        if c.kept_idx_ is None:
            c.kept_idx_ = np.flatnonzero(c.mask_)
        c.full_mean_ = s.get("full_mean")
        c.full_std_ = s.get("full_std")
        if c.full_mean_ is None:
            c.full_mean_ = np.zeros(F)
        if c.full_std_ is None:
            c.full_std_ = np.ones(F)
        return c
