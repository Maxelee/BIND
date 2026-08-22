"""``Emulator`` — params (+ redshift) → every lightcone statistic, instantly.

Ties a per-statistic :class:`~bind.emulator.transforms.StatCompressor` to a
regression backend (:mod:`bind.emulator.backends`).  ``fit`` learns, for each
statistic, ``30 params → PCA latent`` on the runs where that statistic is present;
``predict`` runs them all forward and de-compresses to physical statistic space
with a propagated 1σ.  Statistics with a source-plane axis are interpolated to an
arbitrary source redshift ``z_s``; the WL suppression ``S(ℓ)`` is derived from the
emulated auto-spectrum and the stored DMO trace.

    >>> em = Emulator(backend="mlp").fit(EmulatorDataset.load("emulator_dataset.npz"))
    >>> out = em.predict(bind.fiducial_params(), z_s=1.0)
    >>> out["cl_kappa"], out["suppression"], out["peak_counts"], out["scaling_Y"]

**Public prediction shapes never change** (this is the release contract): every
head is always returned on its full grid (e.g. ``cl_yy`` ``(724,)``, ``cl_kappa``
``(5,5,724)``, ``cl_kappa_y`` ``(5,724)``) whether or not any of its bins were
masked out of training. Masked bins (currently: ell > ``mask_ell_above`` on the
six ell-domain spectrum heads -- the CIC-aliased tail, see
``papers/01_pipeline/audits/spectrum_head_experiment.md``) never reach the PCA
or the regression backend; ``predict`` fills them back in at the per-bin TRAIN
mean (physical units) with an inflated error (``mask_err_inflate`` x the bin's
TRAIN std) so a masked bin is always distinguishable from a genuinely-fit one by
its error bar, never by a shape or key difference. See
``transforms.StatCompressor`` for the mechanism.

**``cl_kappa`` is a COMPOSED head, not a directly-fit one.** Measured on the
released dataset (``emulator_dataset_xpkfix.npz``, 256 runs): the diagonal
(same-plane) blocks satisfy ``cl_kappa[:, i, i, :] == suppression[:, i, :] *
cl_dmo[i, :]`` to float precision (max relative deviation ~1e-15, this is the
*definition* of ``suppression`` in ``dataset.assemble``) -- so the diagonal is
composed for free from the already-well-behaved ``suppression`` head, never
fit directly. The off-diagonal (cross-plane) blocks were measured against
three candidate compositions (``sqrt(S_i*S_j)``, ``0.5*(S_i+S_j)``, ``S_i``)
against the DMO run's own off-diagonal trace and found NOT run-invariant
(coefficient of variation across the 256 runs ~9x the mean for every
candidate and every pair tested, vs ~1e-15 for the diagonal identity) -- so
they are fit directly, with a dedicated small PCA+GP over the 10 unique
``(i<j)`` pairs, in the ``asinh_std`` transform + ell-masked convention (they
are signed: e.g. pair (0,4) ranges down to -1.2e-14 in the raw dataset).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bind.emulator.backends import BACKEND_REGISTRY, make_backend
from bind.emulator.dataset import EmulatorDataset, params_to_unit
from bind.emulator.transforms import StatCompressor

_DEFAULT_BACKEND_KW = {
    "mlp": dict(n_models=8, hidden=128, depth=2, epochs=1500, weight_decay=1e-3),
    "gp": dict(n_restarts=2),
    "gpgpu": dict(epochs=400, lr=0.1),
    "flow": dict(transforms=4, hidden=128, epochs=1500),
}
_TORCH_BACKENDS = {"mlp", "gpgpu", "flow"}      # take a `device` kwarg (GPU-capable)

# The six ell-domain spectrum heads masked to ell <= ELL_MASK_ABOVE (the
# CIC/pixelization-contaminated tail; papers/01_pipeline/audits/
# spectrum_head_experiment.md). `suppression` is deliberately excluded: it is
# a ratio and the contamination artifact cancels in it (repo finding
# "Lightcone kappa upturn = CIC aliasing"; the audit's own suppression anchor
# used the full, unmasked grid and matched the production number). This is
# the Emulator-level default that applies whenever a head's own
# `Target.mask_ell_above` (dataset-serialized, see `dataset.StatSpec`) is
# None -- i.e. for every dataset assembled before that field existed, which
# as of this writing is all of them.
# 2026-08-04 (author ruling, in lockstep with papers/01_pipeline/
# _build_figures_nb.py's ELL_TRUST): moved from 1.5e4 to 3.0e4, the MEASURED
# contamination onset -- (a) the raw C_ell^kappakappa log-slope keeps
# steepening smoothly through 1.5-3e4 and only hardens in the 3e4-3.7e4 band;
# (b) the Sobol response-ratio 16-84 half-widths grow smoothly with no cliff
# near 1.5e4 on the corrected dataset. See the notebook setup cell for the
# full provenance. build_emulator.py mirrors this constant deliberately (kept
# in lockstep, not imported, to keep that script self-contained) -- update
# both. emulator_cache.py's fingerprint hashes this constant by live import
# (`_emulator_config_fingerprint`), so a bundle fit under the old value is
# automatically refused and refit, no manual cache-busting needed.
DEFAULT_ELL_MASK_HEADS = frozenset(
    {"cl_kappa", "cl_kappa_y", "cl_yy", "cl_kappa_tau", "cl_tt", "cl_yt"})
ELL_MASK_ABOVE = 3.0e4

# Per-head transform corrections that must apply regardless of what a
# previously-assembled dataset's manifest says (`Target.transform` is read
# verbatim from the serialized npz -- see `dataset.EmulatorDataset.load` --
# so changing `dataset.STAT_SPECS` alone has no effect on an already-built
# `emulator_dataset*.npz`).  Signed cross-spectra move from `raw` (numerically
# broken here, ~1e7% median frac err in the controlled experiment) to
# `asinh_std` (well-posed, error-to-response ratio ~0.7): see the module
# docstring and `spectrum_head_experiment.md` finding (ii).
DEFAULT_TRANSFORM_OVERRIDES = {
    "cl_kappa_y": "asinh_std", "cl_kappa_tau": "asinh_std", "cl_yt": "asinh_std",
}

# cl_kappa's 10 unique off-diagonal (i<j) tomographic pairs, i,j in [0,5).
_CL_KAPPA_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


class Emulator:
    # GP is the default: with ~100-256 design points in 30-d, its smooth kernel
    # prior interpolates the feedback response cleanly, where an MLP ensemble
    # over-fits/extrapolates and washes the response toward the mean.  Use
    # backend="gpgpu" (or "auto") for the GPU-accelerated exact GP.
    def __init__(self, backend: str = "gp", n_components: int = 12,
                 backend_kwargs: dict | None = None, min_valid: int = 20,
                 device: str | None = None,
                 ell_mask_above: float | None = ELL_MASK_ABOVE,
                 ell_mask_heads: frozenset | set | None = None,
                 transform_overrides: dict[str, str] | None = None,
                 mask_err_inflate: float = 1e3):
        if backend.lower() == "auto":
            import torch
            backend = "gpgpu" if torch.cuda.is_available() else "gp"
        self.backend = backend.lower()
        self.device = device
        self.n_components = n_components
        kw = {**_DEFAULT_BACKEND_KW.get(self.backend, {}), **(backend_kwargs or {})}
        if device is not None and self.backend in _TORCH_BACKENDS:
            kw.setdefault("device", device)
        self.backend_kwargs = kw
        self.min_valid = min_valid
        # ell-domain masking + transform-override config (see the module
        # docstring / DEFAULT_ELL_MASK_HEADS / DEFAULT_TRANSFORM_OVERRIDES):
        # `ell_mask_above=None` disables masking entirely; a per-head
        # `Target.mask_ell_above` (dataset-serialized) always overrides this
        # default WHEN SET (fit()), so this constructor default only governs
        # heads/datasets that don't carry their own opinion.
        self.ell_mask_above = ell_mask_above
        self.ell_mask_heads = frozenset(ell_mask_heads) if ell_mask_heads is not None \
            else DEFAULT_ELL_MASK_HEADS
        self.transform_overrides = {**DEFAULT_TRANSFORM_OVERRIDES,
                                    **(transform_overrides or {})}
        self.mask_err_inflate = mask_err_inflate
        self.compressors: dict[str, StatCompressor] = {}
        self.backends: dict[str, object] = {}
        self.src_axis: dict[str, int | None] = {}
        self.bin_axis: dict[str, str | None] = {}     # stat -> interpolatable last-axis name
        self.bin_values: dict[str, np.ndarray] = {}   # stat -> that axis's values (per-stat)
        self.axes: dict[str, np.ndarray] = {}
        self.param_names: list[str] = []
        self.source_redshifts = np.array([0.5, 1.0, 1.5, 2.0, 2.44])
        self.cl_dmo: np.ndarray | None = None
        self.cl_dmo_full: np.ndarray | None = None    # (5,5,L) DMO tensor, provenance only
        self.composed: dict[str, dict] = {}            # name -> composition metadata
        self.skipped: dict[str, int] = {}

    # ── masking helper ────────────────────────────────────────────────────────
    def _ell_mask_for(self, name: str, t, value_shape: tuple[int, ...]):
        """Boolean mask (broadcastable to ``value_shape``, or None) for head ``name``.

        A dataset-serialized ``t.mask_ell_above`` wins when set (including an
        explicit ``False``/``0`` meaning "never mask this head"); otherwise the
        Emulator-level default (``self.ell_mask_above`` + ``self.ell_mask_heads``)
        applies whenever the head actually has an ``ell`` axis.
        """
        ell = t.axes.get("ell") if hasattr(t, "axes") else None
        if ell is None:
            return None
        thresh = t.mask_ell_above if getattr(t, "mask_ell_above", None) is not None \
            else (self.ell_mask_above if name in self.ell_mask_heads else None)
        if not thresh:
            return None
        keep = np.asarray(ell, float) <= float(thresh)
        return np.broadcast_to(keep, value_shape)

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, ds: EmulatorDataset, stats: list[str] | None = None,
            verbose: bool = True, cl_dmo_full: np.ndarray | None = None) -> "Emulator":
        """Fit every requested statistic.

        ``cl_dmo_full`` (optional) is the raw DMO run's full ``(5,5,L)``
        tomographic tensor (``bind.emulator.dataset.DEFAULT_DMO /
        "Cl_kappa.npz"``'s ``"cl"`` array) -- stored verbatim in the bundle for
        provenance (it is what the ``cl_kappa`` composed-head off-diagonal
        measurement in this module's docstring was checked against), NOT used
        operationally: the measurement found the off-diagonal blocks are not a
        simple run-independent function of it, so they are fit directly instead
        (see ``_fit_cl_kappa_composed`` below).
        """
        self.param_names = list(ds.param_names)
        self.source_redshifts = np.asarray(ds.source_redshifts)
        self.cl_dmo = ds.cl_dmo
        if cl_dmo_full is not None:
            self.cl_dmo_full = np.asarray(cl_dmo_full)
        X = ds.X_unit
        names = stats or list(ds.targets.keys())
        # cl_kappa is COMPOSED (diag = suppression x cl_dmo, off-diag = a small
        # dedicated fit) -- fit it LAST so `suppression`'s backend already
        # exists regardless of where "cl_kappa" sits in the caller's `stats`
        # list order (build_emulator.py / the notebook happen to list
        # suppression first, but this makes that ordering a convenience, not a
        # correctness requirement).
        generic_names = [n for n in names if n != "cl_kappa"]
        composed_names = [n for n in names if n == "cl_kappa"]
        for name in generic_names:
            t = ds.targets[name]
            mask = t.valid_mask()
            n_ok = int(mask.sum())
            if n_ok < self.min_valid:
                self.skipped[name] = n_ok
                if verbose:
                    print(f"  [skip] {name}: only {n_ok} valid runs (< {self.min_valid})")
                continue
            transform = self.transform_overrides.get(name, t.transform)
            value = t.value[mask]
            ell_mask = self._ell_mask_for(name, t, value.shape[1:])
            comp = StatCompressor(transform=transform, n_components=self.n_components,
                                  mask_err_inflate=self.mask_err_inflate
                                  ).fit(value, mask=ell_mask)
            Z = comp.transform_to_latent(value)
            bk = make_backend(self.backend, **self.backend_kwargs).fit(X[mask], Z)
            self.compressors[name] = comp
            self.backends[name] = bk
            self.src_axis[name] = t.src_axis
            # the interpolatable bin axis = the single 1-D axis matching the last
            # data dimension (ell / nu / log_mass_bins / ...), if any
            bn = None
            if len(t.axes) == 1:
                ((an, av),) = t.axes.items()
                if np.ndim(av) == 1 and av.shape[0] == t.value.shape[-1]:
                    bn = an
                    self.bin_values[name] = np.asarray(av)
            self.bin_axis[name] = bn
            for ax, arr in t.axes.items():
                self.axes.setdefault(ax, arr)
            if verbose:
                masked_note = f", {(~ell_mask.reshape(-1)).sum()}/{ell_mask.size} " \
                              "bins masked (ell)" if ell_mask is not None else ""
                print(f"  [fit]  {name}: {n_ok} runs, k={comp.k} PCs, "
                      f"transform={transform}, {self.backend} backend{masked_note}")
        for name in composed_names:
            self._fit_cl_kappa_composed(ds, X, verbose)
        return self

    def _fit_cl_kappa_composed(self, ds: EmulatorDataset, X: np.ndarray,
                               verbose: bool) -> None:
        """Fit the ``cl_kappa`` composed head: diag from suppression, off-diag direct.

        See the module docstring for the measurement that motivates this split.
        """
        name = "cl_kappa"
        t = ds.targets.get(name)
        if t is None:
            return
        if "suppression" not in self.backends or self.cl_dmo is None:
            self.skipped[name] = 0
            if verbose:
                print(f"  [skip] {name}: composed head needs a fitted 'suppression' "
                      "head + cl_dmo (include 'suppression' in `stats`)")
            return
        mask = t.valid_mask()
        n_ok = int(mask.sum())
        if n_ok < self.min_valid:
            self.skipped[name] = n_ok
            if verbose:
                print(f"  [skip] {name}: only {n_ok} valid runs (< {self.min_valid})")
            return
        offdiag = np.stack([t.value[mask][:, i, j, :] for i, j in _CL_KAPPA_PAIRS],
                           axis=1)                                    # (n_ok, 10, L)
        transform = self.transform_overrides.get(name, "asinh_std")
        ell_mask = self._ell_mask_for(name, t, offdiag.shape[1:])
        comp = StatCompressor(transform=transform, n_components=self.n_components,
                              mask_err_inflate=self.mask_err_inflate
                              ).fit(offdiag, mask=ell_mask)
        Z = comp.transform_to_latent(offdiag)
        bk = make_backend(self.backend, **self.backend_kwargs).fit(X[mask], Z)
        self.compressors[name] = comp
        self.backends[name] = bk
        self.src_axis[name] = None       # full (5,5,L) block, no z_s interpolation
        self.composed[name] = {"pairs": _CL_KAPPA_PAIRS, "diag_from": "suppression"}
        ell = t.axes.get("ell")
        if ell is not None:
            self.bin_axis[name] = "ell"
            self.bin_values[name] = np.asarray(ell)
            self.axes.setdefault("ell", ell)
        if verbose:
            masked_note = f", {(~ell_mask.reshape(-1)).sum()}/{ell_mask.size} " \
                          "bins masked (ell)" if ell_mask is not None else ""
            print(f"  [fit]  {name}: {n_ok} runs, COMPOSED (diag = suppression x "
                  f"cl_dmo; off-diag k={comp.k} PCs, transform={transform}, "
                  f"{self.backend} backend over {len(_CL_KAPPA_PAIRS)} pairs{masked_note})")

    # ── predict ───────────────────────────────────────────────────────────────
    def _interp_zs(self, Y: np.ndarray, src_axis: int, z_s) -> np.ndarray:
        """Interpolate Y (m, *shape) along its source-plane axis to z_s (scalar/array)."""
        zs = self.source_redshifts
        ax = 1 + src_axis
        z = np.atleast_1d(np.asarray(z_s, float))
        Ym = np.moveaxis(Y, ax, -1)                          # (..., n_src)
        zc = np.clip(z, zs[0], zs[-1])
        j = np.clip(np.searchsorted(zs, zc) - 1, 0, len(zs) - 2)
        w = (zc - zs[j]) / (zs[j + 1] - zs[j])
        out = Ym[..., j] * (1 - w) + Ym[..., j + 1] * w      # (..., len(z))
        out = np.moveaxis(out, -1, ax)
        return np.squeeze(out, ax) if np.isscalar(z_s) or np.ndim(z_s) == 0 else out

    @staticmethod
    def _interp_axis(Y: np.ndarray, src_x: np.ndarray, tgt_x, logx: bool) -> np.ndarray:
        """Interpolate Y along its LAST axis from ``src_x`` onto ``tgt_x`` (no extrapolation)."""
        Y = np.asarray(Y, float)
        src_x = np.asarray(src_x, float)
        tgt = np.clip(np.asarray(tgt_x, float), src_x.min(), src_x.max())
        x = np.log10(src_x) if logx else src_x
        xt = np.log10(tgt) if logx else tgt
        order = np.argsort(x)
        x = x[order]
        flat = Y.reshape(-1, Y.shape[-1])[:, order]
        res = np.stack([np.interp(xt, x, row) for row in flat])
        return res.reshape(*Y.shape[:-1], len(xt))

    def _predict_cl_kappa_composed(self, Xu: np.ndarray, return_std: bool
                                   ) -> tuple[np.ndarray, np.ndarray | None]:
        """diag = suppression(Xu) x cl_dmo (exact identity); off-diag = the
        dedicated small fit. Returns ``(Y, Yerr)`` each ``(m, 5, 5, L)``."""
        comp_off, bk_off = self.compressors["cl_kappa"], self.backends["cl_kappa"]
        Zmean, Zstd = bk_off.predict(Xu)
        Yoff, Yoff_err = comp_off.inverse_from_latent(Zmean, Zstd if return_std else None)
        Zs_mean, Zs_std = self.backends["suppression"].predict(Xu)
        Sdiag, Sdiag_err = self.compressors["suppression"].inverse_from_latent(
            Zs_mean, Zs_std if return_std else None)                  # (m,5,L)
        cl_dmo = self.cl_dmo                                          # (5,L)
        diag_val = Sdiag * cl_dmo[None]
        diag_err = None if Sdiag_err is None else np.abs(Sdiag_err) * cl_dmo[None]
        m, L = Xu.shape[0], cl_dmo.shape[-1]
        Y = np.zeros((m, 5, 5, L))
        Yerr = np.zeros((m, 5, 5, L)) if return_std else None
        for i in range(5):
            Y[:, i, i, :] = diag_val[:, i, :]
            if Yerr is not None:
                Yerr[:, i, i, :] = 0.0 if diag_err is None else diag_err[:, i, :]
        for k, (i, j) in enumerate(self.composed["cl_kappa"]["pairs"]):
            Y[:, i, j, :] = Yoff[:, k, :]
            Y[:, j, i, :] = Yoff[:, k, :]
            if Yerr is not None:
                e = Yoff_err[:, k, :] if Yoff_err is not None else 0.0
                Yerr[:, i, j, :] = e
                Yerr[:, j, i, :] = e
        return Y, Yerr

    def predict(self, params, z_s=None, stats: list[str] | None = None,
                grids: dict | None = None, return_std: bool = True) -> dict:
        """Predict statistics for ``params`` (35- or 30-vector, or a batch).

        * ``z_s`` (scalar/array) interpolates the source-plane statistics to that
          source redshift; ``None`` returns the full 5-plane blocks.
        * ``stats`` selects which statistics to return (default: all fitted).  Names
          are :attr:`statistics`; the derived ``suppression`` / ``cl_kappa_auto`` are
          included when ``suppression``/``cl_kappa`` is selected.
        * ``grids`` maps a bin-axis name to a target grid, e.g.
          ``{"ell": [...], "nu": [...], "log_mass_bins": [...]}`` — every statistic
          on that axis is interpolated onto it (ell in log-space; no extrapolation).

        Returns a dict of statistic arrays, their ``*_err`` 1σ, the bin ``axes``
        (updated to any requested grids), and ``source_redshifts``.
        """
        native = np.asarray(params, float)
        single = native.ndim == 1
        Xu = params_to_unit(np.atleast_2d(native))           # (m, 30)
        requested = set(stats) if stats is not None else set(self.backends)
        # composed heads (currently only cl_kappa) are handled by their own
        # block below, not the generic per-name loop -- the loop would
        # otherwise hand its offdiag-only latent straight to inverse_from_latent
        # and return a (m,10,L) array under the "cl_kappa" key instead of the
        # released (m,5,5,L) shape.
        names = [n for n in self.backends if n in requested and n not in self.composed]
        out: dict = {"axes": dict(self.axes), "source_redshifts": self.source_redshifts,
                     "z_s": z_s}
        out_axis: dict[str, str] = {}                        # output key -> bin-axis name
        out_src: dict[str, np.ndarray] = {}                  # output key -> source axis values
        ell_vals = self.bin_values.get("suppression",
                                       self.bin_values.get("cl_kappa", self.axes.get("ell")))
        for name in names:
            Zmean, Zstd = self.backends[name].predict(Xu)
            Y, Yerr = self.compressors[name].inverse_from_latent(
                Zmean, Zstd if return_std else None)
            sa = self.src_axis[name]
            if z_s is not None and sa is not None:
                Y = self._interp_zs(Y, sa, z_s)
                if Yerr is not None:
                    Yerr = self._interp_zs(Yerr, sa, z_s)
            out[name] = Y[0] if single else Y
            out_axis[name] = self.bin_axis.get(name)
            out_src[name] = self.bin_values.get(name)
            if return_std and Yerr is not None:
                out[f"{name}_err"] = Yerr[0] if single else Yerr
                out_axis[f"{name}_err"] = self.bin_axis.get(name)
                out_src[f"{name}_err"] = self.bin_values.get(name)

        # COMPOSED heads: cl_kappa = diag (suppression x cl_dmo, exact) + a
        # dedicated small off-diagonal fit -- see the module docstring and
        # _fit_cl_kappa_composed. Always full (m,5,5,L), never z_s-interpolated
        # (unchanged from the pre-composed contract: two source-plane axes).
        if "cl_kappa" in requested and "cl_kappa" in self.composed \
                and self.cl_dmo is not None and "suppression" in self.backends:
            Yck, Yck_err = self._predict_cl_kappa_composed(Xu, return_std)
            out["cl_kappa"] = Yck[0] if single else Yck
            out_axis["cl_kappa"] = "ell"
            out_src["cl_kappa"] = self.bin_values.get("cl_kappa")
            if return_std and Yck_err is not None:
                out["cl_kappa_err"] = Yck_err[0] if single else Yck_err
                out_axis["cl_kappa_err"] = "ell"
                out_src["cl_kappa_err"] = self.bin_values.get("cl_kappa")

        # WL auto-spectrum + suppression.  Prefer the DIRECTLY emulated S(ell)
        # (clean baryon ratio) and reconstruct C_auto = S * C_DMO; fall back to
        # deriving S from the (now possibly composed) C_kappa cube when no
        # direct S head was requested (checks `out`, not `names`/`requested`:
        # the composed block above populates `out["cl_kappa"]` independently of
        # generic-loop membership).
        if "suppression" in names and self.cl_dmo is not None:
            cl_dmo = self.cl_dmo if z_s is None else self._interp_zs(
                self.cl_dmo[None], 0, z_s)[0]
            out["cl_kappa_auto"] = out["suppression"] * cl_dmo
            out_axis["cl_kappa_auto"] = "ell"
            out_src["cl_kappa_auto"] = ell_vals
        elif "cl_kappa" in out and self.cl_dmo is not None:
            ck = out["cl_kappa"]
            ck = ck[None] if single else ck
            auto = np.moveaxis(np.diagonal(ck, axis1=1, axis2=2), -1, 1)  # (m,5,L)
            S = auto / np.where(self.cl_dmo > 0, self.cl_dmo, np.nan)
            if z_s is not None:
                auto = self._interp_zs(auto, 0, z_s)
                S = self._interp_zs(S, 0, z_s)
            out["cl_kappa_auto"] = auto[0] if single else auto
            out["suppression"] = S[0] if single else S
            out_axis["cl_kappa_auto"] = out_axis["suppression"] = "ell"
            out_src["cl_kappa_auto"] = out_src["suppression"] = ell_vals

        # optional: resample any statistic onto a user grid for its bin axis,
        # interpolating each from ITS OWN source values (axis names can collide,
        # e.g. peak_counts nu has 68 bins while peak_R nu has 14).
        if grids:
            for key, arr in list(out.items()):
                an = out_axis.get(key)
                src = out_src.get(key)
                if an and an in grids and src is not None:
                    out[key] = self._interp_axis(arr, src, grids[an], logx=(an == "ell"))
            for an, g in grids.items():
                if an in out["axes"]:
                    out["axes"][an] = np.asarray(g, float)
        return out

    # ── (de)serialization ─────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        import torch
        bundle = {
            "backend": self.backend, "n_components": self.n_components,
            "backend_kwargs": self.backend_kwargs, "min_valid": self.min_valid,
            "param_names": self.param_names,
            "source_redshifts": self.source_redshifts, "cl_dmo": self.cl_dmo,
            "cl_dmo_full": self.cl_dmo_full,
            "ell_mask_above": self.ell_mask_above,
            "ell_mask_heads": sorted(self.ell_mask_heads),
            "transform_overrides": self.transform_overrides,
            "mask_err_inflate": self.mask_err_inflate,
            "composed": self.composed,
            "axes": self.axes, "src_axis": self.src_axis, "bin_axis": self.bin_axis,
            "bin_values": self.bin_values, "skipped": self.skipped,
            "compressors": {k: c.state_dict() for k, c in self.compressors.items()},
            "backends": {k: b.state_dict() for k, b in self.backends.items()},
        }
        torch.save(bundle, path)

    @classmethod
    def load(cls, path: str | Path) -> "Emulator":
        import torch
        b = torch.load(path, map_location="cpu", weights_only=False)
        em = cls(backend=b["backend"], n_components=b["n_components"],
                 backend_kwargs=b["backend_kwargs"], min_valid=b["min_valid"],
                 ell_mask_above=b.get("ell_mask_above", ELL_MASK_ABOVE),
                 ell_mask_heads=(set(b["ell_mask_heads"])
                                if b.get("ell_mask_heads") is not None else None),
                 transform_overrides=b.get("transform_overrides"),
                 mask_err_inflate=b.get("mask_err_inflate", 1e3))
        em.param_names = list(b["param_names"])
        em.source_redshifts = np.asarray(b["source_redshifts"])
        em.cl_dmo = b["cl_dmo"]
        em.cl_dmo_full = b.get("cl_dmo_full")
        em.composed = b.get("composed", {})
        em.axes = b["axes"]
        em.src_axis = b["src_axis"]
        em.bin_axis = b.get("bin_axis", {})
        em.bin_values = b.get("bin_values", {})
        em.skipped = b["skipped"]
        em.compressors = {k: StatCompressor.from_state(s) for k, s in b["compressors"].items()}
        em.backends = {k: BACKEND_REGISTRY[s["kind"]].from_state(s)
                       for k, s in b["backends"].items()}
        return em

    @property
    def statistics(self) -> list[str]:
        return list(self.backends.keys())

    def describe(self) -> str:
        """Human-readable list of available statistics, their bin axis and native range."""
        lines = [f"Emulator ({self.backend}): {len(self.backends)} statistics, "
                 f"z_s={list(np.round(self.source_redshifts, 2))}"]
        for name in self.statistics:
            if name in self.bin_values:
                ax = self.bin_values[name]
                lines.append(f"  {name:16s} axis={self.bin_axis[name]} "
                             f"[{ax.min():.4g}, {ax.max():.4g}] (n={len(ax)})")
            else:
                lines.append(f"  {name:16s} (no resamplable bin axis)")
        if "suppression" in self.backends or "cl_kappa" in self.backends:
            lines.append("  derived: cl_kappa_auto, suppression (axis=ell)")
        return "\n".join(lines)
