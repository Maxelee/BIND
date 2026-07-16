"""Assemble the lightcone-statistics emulator training table from a run suite.

Walks a suite of completed lightcone runs (e.g. the SB35 Sobol suite at
``/mnt/home/mlee1/ceph/bind_sb35/runs``) and packs every per-run summary
statistic into a single cached :class:`EmulatorDataset`:

* ``X_native`` ``(N, 30)`` the SB35 astro parameters, ``X_unit`` ``(N, 30)`` the
  same mapped to ``[0, 1]`` in the native (log10 where ``LogFlag``) sampling space
  — the emulator input (same convention as :mod:`bind.inference.design`).
* a named ``targets`` dict; each :class:`Target` holds ``value`` ``(N, *shape)``
  (the realization mean), ``err`` (cosmic-variance 1-sigma), the bin axes, and the
  source-plane axis (for z_s interpolation).

Power spectra (κκ, yy, ττ) are positive → emulated in ``log``; the cross spectra
(κy, κτ, yτ) can change sign → emulated raw.  The DMO convergence trace (fixed by
cosmology) is stored once as ``cl_dmo`` so the suppression ``S(ℓ)`` is a derived
output, not a separate fit.

The Y–M / f_gas–M / T–M relations are read from the pre-reduced SB35
``integrated.parquet`` cache (per ``run`` at one ``snap``) via
:func:`bind.inference.stats.scaling_relations`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bind.inference.design import ASTRO_PARAM_INDICES
from bind.inference.stats import scaling_relations
from bind.params import PARAM_LOG_FLAG, PARAM_MAX, PARAM_MIN, PARAM_NAMES

_ASTRO = np.asarray(ASTRO_PARAM_INDICES)
ASTRO_PARAM_NAMES = [PARAM_NAMES[i] for i in _ASTRO]
SOURCE_REDSHIFTS = np.array([0.5, 1.0, 1.5, 2.0, 2.44])

DEFAULT_RUNS = Path("/mnt/home/mlee1/ceph/bind_sb35/runs")
DEFAULT_DMO = Path("/mnt/home/mlee1/ceph/bind_science/runs/dmo/run_0000")
DEFAULT_PARQUET = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet")


def params_to_unit(native: np.ndarray, idx: np.ndarray = _ASTRO) -> np.ndarray:
    """Map native SB35 parameter values to ``[0, 1]`` (inverse of design._unit_to_native).

    Log-flagged params are mapped in log10, matching the Sobol prior.  ``native``
    is ``(..., 35)`` (full vector) or ``(..., 30)`` (already astro-only); the
    ``idx`` columns are normalized and returned as ``(..., 30)``.
    """
    native = np.asarray(native, dtype=float)
    cols = native[..., idx] if native.shape[-1] == len(PARAM_NAMES) else native
    log_mask = PARAM_LOG_FLAG[idx] == 1
    lo, hi = PARAM_MIN[idx], PARAM_MAX[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_u = (np.log10(np.where(cols > 0, cols, np.nan)) - np.log10(lo)) / (
            np.log10(hi) - np.log10(lo))
    lin_u = (cols - lo) / (hi - lo)
    return np.where(log_mask, log_u, lin_u)


# ── statistic registry ────────────────────────────────────────────────────────
# transform: "log" (log of positive), "log1p" (counts/pdf), "raw" (signed/standardize)
# src_axis: axis of the loaded array that indexes the 5 source planes (None = no z_s)

@dataclass
class StatSpec:
    file: str
    key: str
    transform: str = "raw"
    src_axis: int | None = None
    err_key: str | None = None
    axes: tuple[str, ...] = ()          # bin-axis keys in the same npz


STAT_SPECS: dict[str, StatSpec] = {
    # WL tomographic power (full 5x5xL cube; auto+cross packed together)
    "cl_kappa":     StatSpec("Cl_kappa.npz", "cl", "log", None, "cl_err", ("ell",)),
    # WL x tSZ and tSZ auto
    "cl_kappa_y":   StatSpec("Cl_kappa_y.npz", "cl_ky", "raw", 0, "cl_ky_err", ("ell",)),
    "cl_yy":        StatSpec("Cl_kappa_y.npz", "cl_yy", "log", None, "cl_yy_err", ("ell",)),
    # WL x tau, tau auto, y x tau
    "cl_kappa_tau": StatSpec("Cl_tau.npz", "cl_kt", "raw", 0, "cl_kt_err", ("ell",)),
    "cl_tt":        StatSpec("Cl_tau.npz", "cl_tt", "log", None, "cl_tt_err", ("ell",)),
    "cl_yt":        StatSpec("Cl_tau.npz", "cl_yt", "raw", None, "cl_yt_err", ("ell",)),
    # peaks / minima vs S/N nu
    "peak_counts":  StatSpec("peak_counts.npz", "peak_counts", "log1p", 0,
                             "peak_counts_err", ("nu",)),
    "minima_counts": StatSpec("peak_counts.npz", "minima_counts", "log1p", 0,
                              "minima_counts_err", ("nu",)),
    # convergence PDF + Minkowski functionals
    "pdf":          StatSpec("nongaussian_stats.npz", "pdf", "log1p", 0, None, ("pdf_bins",)),
    "mf_v0":        StatSpec("nongaussian_stats.npz", "V0", "raw", 0, None, ("mf_nu",)),
    "mf_v1":        StatSpec("nongaussian_stats.npz", "V1", "raw", 0, None, ("mf_nu",)),
    "mf_v2":        StatSpec("nongaussian_stats.npz", "V2", "raw", 0, None, ("mf_nu",)),
    "moments":      StatSpec("nongaussian_stats.npz", "_moments", "raw", 0, None,
                             ("smoothing_scales_arcmin",)),
    # wavelet scattering transform
    "wst":          StatSpec("wst.npz", "wst", "log", 0, "wst_err", ()),
    # dispersion measure
    "dm_pdf":       StatSpec("dm_stats.npz", "dm_pdf", "log1p", 0, None, ("dm_bins",)),
    "dm_moments":   StatSpec("dm_stats.npz", "_dm_moments", "raw", 0, None, ()),
    # tSZ specific-thermal-energy at WL peaks (R(nu)) and DM/y profiles
    "peak_R":       StatSpec("peak_cross.npz", "R", "raw", 0, None, ("nu",)),
    "peak_y":       StatSpec("peak_cross.npz", "y_peak", "raw", 0, None, ("nu",)),
}


@dataclass
class Target:
    value: np.ndarray                       # (N, *shape) realization mean (NaN where absent)
    err: np.ndarray | None = None           # (N, *shape) cosmic-variance 1sigma
    transform: str = "raw"
    src_axis: int | None = None             # axis of *shape* indexing source planes
    axes: dict[str, np.ndarray] = field(default_factory=dict)
    valid: np.ndarray | None = None         # (N,) bool: run has this statistic

    def valid_mask(self) -> np.ndarray:
        if self.valid is not None:
            return self.valid
        flat = self.value.reshape(len(self.value), -1)
        return np.isfinite(flat).all(axis=1)


def _load_stat(run: Path, spec: StatSpec) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Load one statistic block from a run; returns (value, err, axes) or (None,..)."""
    p = run / spec.file
    if not p.exists():
        return None, None, {}
    d = np.load(p)
    axes = {a: d[a] for a in spec.axes if a in d.files}
    # synthetic composite keys
    if spec.key == "_moments":
        if not all(k in d.files for k in ("variance", "skewness", "kurtosis")):
            return None, None, {}
        val = np.stack([d["variance"], d["skewness"], d["kurtosis"]], axis=1)  # (n_src,3,n_scale)
        return val, None, axes
    if spec.key == "_dm_moments":
        keys = ("sigma_dm", "F", "skewness", "kurtosis")
        if not all(k in d.files for k in keys):
            return None, None, {}
        return np.stack([d[k] for k in keys], axis=-1), None, axes   # (n_src, 4)
    if spec.key not in d.files:
        return None, None, {}
    val = d[spec.key]
    err = d[spec.err_key] if (spec.err_key and spec.err_key in d.files) else None
    return val, err, axes


@dataclass
class EmulatorDataset:
    param_names: list[str]
    X_native: np.ndarray                    # (N, 30)
    X_unit: np.ndarray                      # (N, 30)
    run_ids: np.ndarray                     # (N,)
    source_redshifts: np.ndarray            # (5,)
    targets: dict[str, Target]
    cl_dmo: np.ndarray | None = None        # (n_src, n_ell) DMO auto, for S(ell)
    meta: dict = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return len(self.run_ids)

    def subset(self, idx: np.ndarray) -> "EmulatorDataset":
        """A view restricted to run rows ``idx`` (for cross-validation / hold-out)."""
        idx = np.asarray(idx)
        tg = {k: Target(t.value[idx], None if t.err is None else t.err[idx],
                        t.transform, t.src_axis, t.axes,
                        None if t.valid is None else t.valid[idx])
              for k, t in self.targets.items()}
        return EmulatorDataset(
            param_names=list(self.param_names), X_native=self.X_native[idx],
            X_unit=self.X_unit[idx], run_ids=self.run_ids[idx],
            source_redshifts=self.source_redshifts, targets=tg,
            cl_dmo=self.cl_dmo, meta=dict(self.meta))

    def summary(self) -> str:
        lines = [f"EmulatorDataset: {self.n_runs} runs x {self.X_unit.shape[1]} params, "
                 f"{len(self.targets)} statistics, z_s={list(self.source_redshifts)}"]
        for name, t in self.targets.items():
            sh = "x".join(map(str, t.value.shape[1:]))
            lines.append(f"  {name:14s} ({sh})  transform={t.transform}"
                         + (f" src_axis={t.src_axis}" if t.src_axis is not None else ""))
        if self.cl_dmo is not None:
            lines.append(f"  cl_dmo {self.cl_dmo.shape} (for suppression S)")
        return "\n".join(lines)

    # ── (de)serialization: numpy-native npz + a JSON manifest ────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        blob: dict[str, np.ndarray] = {
            "X_native": self.X_native, "X_unit": self.X_unit,
            "run_ids": self.run_ids, "source_redshifts": self.source_redshifts,
            "param_names": np.array(self.param_names),
        }
        if self.cl_dmo is not None:
            blob["cl_dmo"] = self.cl_dmo
        manifest = {"targets": {}, "meta": self.meta}
        for name, t in self.targets.items():
            blob[f"t__{name}__value"] = t.value
            blob[f"t__{name}__valid"] = t.valid_mask()
            if t.err is not None:
                blob[f"t__{name}__err"] = t.err
            for ax, arr in t.axes.items():
                blob[f"a__{name}__{ax}"] = arr
            manifest["targets"][name] = {
                "transform": t.transform, "src_axis": t.src_axis,
                "has_err": t.err is not None, "axes": list(t.axes.keys()),
            }
        blob["manifest"] = np.array(json.dumps(manifest))
        np.savez_compressed(path, **blob)

    @classmethod
    def load(cls, path: str | Path) -> "EmulatorDataset":
        d = np.load(Path(path), allow_pickle=False)
        manifest = json.loads(str(d["manifest"]))
        targets = {}
        for name, m in manifest["targets"].items():
            axes = {ax: d[f"a__{name}__{ax}"] for ax in m["axes"]}
            targets[name] = Target(
                value=d[f"t__{name}__value"],
                err=d[f"t__{name}__err"] if m["has_err"] else None,
                transform=m["transform"], src_axis=m["src_axis"], axes=axes,
                valid=d[f"t__{name}__valid"] if f"t__{name}__valid" in d.files else None)
        return cls(
            param_names=list(d["param_names"]),
            X_native=d["X_native"], X_unit=d["X_unit"], run_ids=d["run_ids"],
            source_redshifts=d["source_redshifts"],
            targets=targets,
            cl_dmo=d["cl_dmo"] if "cl_dmo" in d.files else None,
            meta=manifest.get("meta", {}))


def _scaling_from_parquet(parquet: Path, runs: list[int], snap: int,
                          mass_bins: np.ndarray) -> dict[str, np.ndarray] | None:
    """Per-run binned Y-M / f_gas-M / f_star-M / T-M from the integrated cache."""
    try:
        import pandas as pd
    except ImportError:
        return None
    if not parquet.exists():
        return None
    cols = ["run", "snap", "M200", "Y_500", "f_gas_500", "f_star_500", "T_mw_500"]
    df = pd.read_parquet(parquet, columns=cols)
    df = df[df.snap == snap]
    out: dict[str, list] = {}
    for r in runs:
        g = df[df.run == r]
        sr = scaling_relations(g.M200.values, Y=g.Y_500.values, f_gas=g.f_gas_500.values,
                               f_star=g.f_star_500.values, T=g.T_mw_500.values,
                               mass_bins=mass_bins)
        for k, v in sr.items():
            out.setdefault(k, []).append(np.asarray(v))
    return {k: np.asarray(v) for k, v in out.items()} if out else None


def assemble(
    runs_dir: str | Path = DEFAULT_RUNS,
    *,
    dmo_dir: str | Path | None = DEFAULT_DMO,
    parquet: str | Path | None = DEFAULT_PARQUET,
    scaling_snap: int = 96,
    require: str = "Cl_kappa.npz",
    mass_bins: np.ndarray | None = None,
    verbose: bool = True,
) -> EmulatorDataset:
    """Build the :class:`EmulatorDataset` from every run with the ``require`` stat present."""
    runs_dir = Path(runs_dir)
    run_dirs = sorted(p for p in runs_dir.glob("run_*") if (p / require).exists())
    if not run_dirs:
        raise FileNotFoundError(f"no runs with {require} under {runs_dir}")
    run_ids = np.array([int(p.name.split("_")[1]) for p in run_dirs])
    X_native = np.array([np.load(p / "params.npy") for p in run_dirs])      # (N, 35)
    X_unit = params_to_unit(X_native)                                       # (N, 30)

    # collect each statistic block over runs (only runs where it's present)
    raw: dict[str, dict] = {}
    for name, spec in STAT_SPECS.items():
        vals, errs, axes_ref, present = [], [], None, []
        for p in run_dirs:
            v, e, ax = _load_stat(p, spec)
            if v is None:
                continue
            present.append(int(p.name.split("_")[1]))
            vals.append(v)
            errs.append(e if e is not None else np.full_like(v, np.nan, dtype=float))
            axes_ref = axes_ref or ax
        if not vals:
            if verbose:
                print(f"  [skip] {name}: not present in any run")
            continue
        raw[name] = dict(value=np.asarray(vals), err=np.asarray(errs),
                         ids=np.asarray(present), axes=axes_ref or {}, spec=spec)
        if verbose:
            print(f"  [ok]   {name}: {len(vals)} runs  value{np.asarray(vals).shape}")

    # Global run set = every run with the `require` stat.  Each statistic is
    # aligned to it and NaN-padded where absent (with a per-stat valid mask), so a
    # partially-backfilled stat (e.g. WST mid-array) never shrinks the others.
    sel_ids = run_ids
    targets: dict[str, Target] = {}
    for name, r in raw.items():
        spec: StatSpec = r["spec"]
        shape = r["value"].shape[1:]
        value = np.full((len(sel_ids), *shape), np.nan, dtype=float)
        err = np.full((len(sel_ids), *shape), np.nan, dtype=float)
        valid = np.zeros(len(sel_ids), dtype=bool)
        pos = {i: k for k, i in enumerate(r["ids"])}
        for gi, rid in enumerate(sel_ids):
            if rid in pos:
                value[gi] = r["value"][pos[rid]]
                err[gi] = r["err"][pos[rid]]
                valid[gi] = True
        targets[name] = Target(
            value=value, err=None if np.all(np.isnan(err)) else err,
            transform=spec.transform, src_axis=spec.src_axis, axes=r["axes"], valid=valid)

    # Y-M family from parquet (optional).  The TNG300 halo sample populates
    # ~10^13–10^14.7 Msun/h; bins outside that are empty for every run and only
    # contribute imputation artifacts, so the default range is the populated band.
    if mass_bins is None:
        mass_bins = np.linspace(13.0, 14.75, 8)
    if parquet is not None:
        scal = _scaling_from_parquet(Path(parquet), sel_ids.tolist(), scaling_snap, mass_bins)
        if scal is not None:
            cent = scal["log_mass_bins"][0]
            for q in ("Y", "f_gas", "f_star", "T"):
                mk = f"{q}_median"
                if mk in scal:
                    val = scal[mk]
                    targets[f"scaling_{q}"] = Target(
                        value=val, err=scal.get(f"{q}_scatter"),
                        transform=("log" if q in ("Y", "T") else "raw"),
                        src_axis=None, axes={"log_mass_bins": cent},
                        valid=np.isfinite(val).any(axis=tuple(range(1, val.ndim))))
            if verbose:
                print(f"  [ok]   scaling_* (Y/f_gas/f_star/T) from parquet snap {scaling_snap}")
        elif verbose:
            print("  [skip] scaling_*: parquet unavailable")

    # DMO trace for suppression S(ell) (cosmology fixed -> a single vector)
    cl_dmo = None
    if dmo_dir is not None and (Path(dmo_dir) / "Cl_kappa.npz").exists():
        dmo = np.load(Path(dmo_dir) / "Cl_kappa.npz")
        if "cl" in dmo.files:
            cl = dmo["cl"]
            dmo_auto = np.array([cl[i, i] for i in range(cl.shape[0])])     # (n_src, n_ell)
            cl_self = targets.get("cl_kappa")
            if cl_self is not None and dmo_auto.shape[-1] == cl_self.value.shape[-1]:
                cl_dmo = dmo_auto
                # Emulate the suppression S(ell)=C_auto/C_DMO DIRECTLY, not derived
                # from raw C_kappa: the raw spectrum is dominated by the fixed-cosmo
                # LCDM shape and PCA buries the few-% baryon ratio.  S is O(1) and
                # cosmic-variance-cancelled (shared DMO trace, r~0.98).
                auto = np.moveaxis(np.diagonal(cl_self.value, axis1=1, axis2=2), -1, 1)
                S = auto / np.where(cl_dmo > 0, cl_dmo, np.nan)      # (N, 5, L)
                targets["suppression"] = Target(
                    value=S, err=None, transform="raw", src_axis=0,
                    axes={"ell": cl_self.axes.get("ell")}, valid=cl_self.valid)
                if verbose:
                    print(f"  [ok]   cl_dmo {cl_dmo.shape} + direct suppression target S")
            elif verbose:
                print("  [skip] cl_dmo: ell length mismatch vs cl_kappa")

    return EmulatorDataset(
        param_names=ASTRO_PARAM_NAMES,
        X_native=X_native, X_unit=X_unit, run_ids=sel_ids,
        source_redshifts=SOURCE_REDSHIFTS, targets=targets, cl_dmo=cl_dmo,
        meta={"runs_dir": str(runs_dir), "scaling_snap": scaling_snap,
              "mass_bins": mass_bins.tolist()})
