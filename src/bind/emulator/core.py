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


class Emulator:
    # GP is the default: with ~100-256 design points in 30-d, its smooth kernel
    # prior interpolates the feedback response cleanly, where an MLP ensemble
    # over-fits/extrapolates and washes the response toward the mean.  Use
    # backend="gpgpu" (or "auto") for the GPU-accelerated exact GP.
    def __init__(self, backend: str = "gp", n_components: int = 12,
                 backend_kwargs: dict | None = None, min_valid: int = 20,
                 device: str | None = None):
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
        self.compressors: dict[str, StatCompressor] = {}
        self.backends: dict[str, object] = {}
        self.src_axis: dict[str, int | None] = {}
        self.bin_axis: dict[str, str | None] = {}     # stat -> interpolatable last-axis name
        self.bin_values: dict[str, np.ndarray] = {}   # stat -> that axis's values (per-stat)
        self.axes: dict[str, np.ndarray] = {}
        self.param_names: list[str] = []
        self.source_redshifts = np.array([0.5, 1.0, 1.5, 2.0, 2.44])
        self.cl_dmo: np.ndarray | None = None
        self.skipped: dict[str, int] = {}

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, ds: EmulatorDataset, stats: list[str] | None = None,
            verbose: bool = True) -> "Emulator":
        self.param_names = list(ds.param_names)
        self.source_redshifts = np.asarray(ds.source_redshifts)
        self.cl_dmo = ds.cl_dmo
        X = ds.X_unit
        names = stats or list(ds.targets.keys())
        for name in names:
            t = ds.targets[name]
            mask = t.valid_mask()
            n_ok = int(mask.sum())
            if n_ok < self.min_valid:
                self.skipped[name] = n_ok
                if verbose:
                    print(f"  [skip] {name}: only {n_ok} valid runs (< {self.min_valid})")
                continue
            comp = StatCompressor(transform=t.transform,
                                  n_components=self.n_components).fit(t.value[mask])
            Z = comp.transform_to_latent(t.value[mask])
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
                print(f"  [fit]  {name}: {n_ok} runs, k={comp.k} PCs, "
                      f"{self.backend} backend")
        return self

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
        names = [n for n in self.backends if n in requested]
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

        # WL auto-spectrum + suppression.  Prefer the DIRECTLY emulated S(ell)
        # (clean baryon ratio) and reconstruct C_auto = S * C_DMO; fall back to
        # deriving S from the raw C_kappa cube when no direct S head is present.
        if "suppression" in names and self.cl_dmo is not None:
            cl_dmo = self.cl_dmo if z_s is None else self._interp_zs(
                self.cl_dmo[None], 0, z_s)[0]
            out["cl_kappa_auto"] = out["suppression"] * cl_dmo
            out_axis["cl_kappa_auto"] = "ell"
            out_src["cl_kappa_auto"] = ell_vals
        elif "cl_kappa" in names and self.cl_dmo is not None:
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
                 backend_kwargs=b["backend_kwargs"], min_valid=b["min_valid"])
        em.param_names = list(b["param_names"])
        em.source_redshifts = np.asarray(b["source_redshifts"])
        em.cl_dmo = b["cl_dmo"]
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
