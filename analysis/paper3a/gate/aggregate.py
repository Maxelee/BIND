"""A3 gate aggregation: 317 x 6 operator tables -> envelopes, overlays, memo inputs.

Consumes the per-(design point, snapshot) npz written by `gate_operators`
(under /mnt/ceph/users/mlee1/paper3/A/wp3_gate/operator_tables/) and builds:

- **kSZ envelopes**: per snapshot and mass bin, the SB35 Sobol 16-84% and
  min-max bands of the stacked tau_CAP profile, the 60 twobound 1P-extreme
  trajectories (paired lower/upper per parameter), and the fiducial curve
  when its tables exist.
- **f_gas(M) envelopes**: per snapshot, median cylindrical f_gas in log-M500
  bins per design point -> the same band structure, with the truth-calibrated
  CylToSph factor applied for comparison against the (spherical) eROSITA
  values.
- **Y-M envelopes**: per design point power-law fits (slope, normalization at
  the pivot, intrinsic scatter) -> band structure. Model-only at gate level
  (no Y-M vector was frozen in A1).

Convention notes baked into every output (see `ANNOTATIONS`):
- Painted compton_y is in the PROPER-pixel-area convention (physically
  correct dimensionless y) — confirmed by the 2026-07-17 a-factor audit
  (wp2 REPORT ## BLOCKED); no z-dependent rescale is applied or needed here.
- CylToSph factors are the Popeye wp2 session-3 re-run values (job 2451211),
  quoted from the wp2 REPORT until the npz artifacts arrive on rusty
  (WS2 rsync pending); swap `CYLTOSPH_INTERIM` for the npz load then.
- kSZ tau_CAP -> T_kSZ uses a NOMINAL v_rms/c; the per-bin sigma_true
  extraction is an A5 to-do, so the kSZ overlay carries an explicit
  normalization-freedom annotation rather than a hidden choice.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis.paper3a.observables.constants import T_CMB_UK, TNG_H

TABLES = Path("/mnt/ceph/users/mlee1/paper3/A/wp3_gate/operator_tables_v2")

# Popeye wp2 truth validation, session-3 re-run (job 2451211), snaps
# 096/071/067/063 — provenance: projectA/wp2-observable-matching/REPORT.md
# "Session 3" table (private plans repo). INTERIM quote pending the
# wp2_validation npz pull (WS2); the mean factor is stable across z.
CYLTOSPH_INTERIM = {"096": 0.528, "071": 0.507, "067": 0.518, "063": 0.513}
CYLTOSPH_SCATTER = 0.13  # representative per-halo rms (0.11-0.15 across snaps)


def cyltosph_for_snap(snap: str) -> tuple[float, str]:
    """(factor, provenance) — validated snaps directly; others take the
    nearest-z validated factor (defensible at gate level: the factor is
    stable, 0.507-0.528, over z = 0.03-0.60), flagged as extrapolated."""
    if snap in CYLTOSPH_INTERIM:
        return CYLTOSPH_INTERIM[snap], "validated (wp2 session 3)"
    validated = {"096": 0.034, "071": 0.420, "067": 0.503, "063": 0.599}
    all_z = {"096": 0.034, "071": 0.420, "067": 0.503, "063": 0.599,
             "056": 0.791, "049": 1.036, "059": 0.700, "052": 0.923}
    z = all_z.get(snap)
    if z is None:
        raise KeyError(f"snap {snap}: no redshift on record for CylToSph extrapolation")
    nearest = min(validated, key=lambda s: abs(validated[s] - z))
    return CYLTOSPH_INTERIM[nearest], f"EXTRAPOLATED from snap {nearest} (nearest validated z)"

NOMINAL_VRMS_OVER_C = 1.06e-3  # placeholder normalization; sigma_true(z) = A5 to-do

ANNOTATIONS = {
    "y_convention": "painted y = proper-pixel-area convention (audit 2026-07-17); no rescale applied",
    "cyltosph": "interim wp2 session-3 values (see CYLTOSPH_INTERIM provenance); swap for npz on arrival",
    "ksz_norm": f"tau_CAP -> T_kSZ via NOMINAL v_rms/c = {NOMINAL_VRMS_OVER_C}; per-bin sigma_true pending (A5)",
    "ksz_outer_radii": "outer radii (>=4.75') at z>0.4 are Sigma_model-dominated (wp2 session 3)",
    "ksz_NOT_DATA_COMPARABLE": (
        "2026-07-17 dry-run finding: kSZ support stats are NOT meaningful yet — "
        "the composite gas map has no diffuse background between apertures (CAP "
        "annulus compensates against zeros) and no velocity decorrelation of the "
        "LOS column / 2-halo term. See wp3 DECISION_MEMO.md axis 3. f_gas and "
        "Y-M axes are unaffected."
    ),
    "erosita_masses": "eRASS1 point-estimate masses biased high below 1e14 Msun (freeze doc 6d) — gate-level overlay only",
}

LOGM500_BIN_EDGES = np.array([13.0, 13.4, 13.8, 14.2, 14.6, 15.2])  # log10 Msun/h


@dataclass
class GateTable:
    """One design point at one snapshot, thinly wrapping the npz payload."""

    bundle: str
    run: str
    snap: str
    z_snap: float
    data: dict = field(repr=False)

    @classmethod
    def load(cls, path: Path) -> "GateTable":
        bundle, run = path.stem.split("_run_")[0], "run_" + path.stem.split("_run_")[1].split("_snap")[0]
        d = dict(np.load(path, allow_pickle=False))
        return cls(bundle=bundle, run=run, snap=path.stem.split("snap")[-1], z_snap=float(d["z_snap"]), data=d)

    def fgas_median_by_massbin(self, cyltosph: float) -> np.ndarray:
        """Median SPHERICAL-equivalent f_gas per log-M500 bin (NaN where empty)."""
        logm500 = np.log10(self.data["m500_msunh"])
        fgas_sph = self.data["fgas_cyl_r500"] * cyltosph
        out = np.full(len(LOGM500_BIN_EDGES) - 1, np.nan)
        for i in range(len(out)):
            sel = (logm500 >= LOGM500_BIN_EDGES[i]) & (logm500 < LOGM500_BIN_EDGES[i + 1])
            if sel.sum() >= 5:
                out[i] = np.median(fgas_sph[sel])
        return out

    def ym_fit(self):
        from analysis.paper3a.observables import fit_ym_relation

        return fit_ym_relation(self.data["m500_msunh"], self.data["y_cyl_r500_mpc2"])


def load_all(tables_dir: Path = TABLES, snaps: tuple[str, ...] | None = None) -> dict:
    """{snap: {bundle: [GateTable, ...]}} for every table on disk."""
    out: dict[str, dict[str, list[GateTable]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(tables_dir.glob("*.npz")):
        t = GateTable.load(path)
        if snaps is None or t.snap in snaps:
            out[t.snap][t.bundle].append(t)
    return {s: dict(b) for s, b in out.items()}


def band(values: np.ndarray, axis: int = 0) -> dict:
    """16-84% and min-max envelope bands along `axis`, NaN-aware.

    Bins empty in every design point (e.g. the top mass bin at high z) come
    out NaN across the band — expected, so the all-NaN warning is silenced.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return {
            "p16": np.nanpercentile(values, 16, axis=axis),
            "p50": np.nanpercentile(values, 50, axis=axis),
            "p84": np.nanpercentile(values, 84, axis=axis),
            "min": np.nanmin(values, axis=axis),
            "max": np.nanmax(values, axis=axis),
            "n": values.shape[axis],
        }


def ksz_envelope(tables: dict, snap: str, mass_bin: int, in_tksz: bool = False) -> dict:
    """Sobol band + twobound trajectories + fiducial for one kSZ stack bin.

    in_tksz=True multiplies tau_CAP by T_CMB * NOMINAL_VRMS_OVER_C (explicitly
    nominal — see ANNOTATIONS['ksz_norm']).
    """
    key = f"ksz_bin{mass_bin}_stack"
    scale = T_CMB_UK * NOMINAL_VRMS_OVER_C if in_tksz else 1.0
    sobol = np.array([t.data[key] for t in tables[snap].get("sb35", [])]) * scale
    result = {
        "radii_arcmin": tables[snap]["sb35"][0].data["radii_arcmin"],
        "mass_bin_range": tables[snap]["sb35"][0].data[f"ksz_bin{mass_bin}_range"],
        "sobol_band": band(sobol),
        "units": "muK arcmin^2 (NOMINAL norm)" if in_tksz else "tau_CAP arcmin^2",
    }
    result["twobound"] = {
        t.run: t.data[key] * scale for t in tables[snap].get("twobound", [])
    }
    fid = tables[snap].get("fiducial", [])
    result["fiducial"] = fid[0].data[key] * scale if fid else None
    return result


def fgas_envelope(tables: dict, snap: str) -> dict:
    """Spherical-equivalent f_gas(M500) bands across the design, + bin centers."""
    cyl2sph, cyl2sph_src = cyltosph_for_snap(snap)
    sobol = np.array([t.fgas_median_by_massbin(cyl2sph) for t in tables[snap].get("sb35", [])])
    result = {
        "logm500_centers_msunh": 0.5 * (LOGM500_BIN_EDGES[1:] + LOGM500_BIN_EDGES[:-1]),
        "cyltosph_applied": cyl2sph,
        "cyltosph_source": cyl2sph_src,
        "sobol_band": band(sobol),
    }
    result["twobound"] = {
        t.run: t.fgas_median_by_massbin(cyl2sph) for t in tables[snap].get("twobound", [])
    }
    fid = tables[snap].get("fiducial", [])
    result["fiducial"] = fid[0].fgas_median_by_massbin(cyl2sph) if fid else None
    return result


def ym_envelope(tables: dict, snap: str) -> dict:
    """Distribution of (slope, log-normalization, intrinsic scatter) across the design."""
    fits = [t.ym_fit() for t in tables[snap].get("sb35", [])]
    arr = np.array([[f.alpha, f.beta, f.scatter_dex] for f in fits])
    result = {"param_names": ["alpha", "beta_at_1e14", "scatter_dex"], "sobol_band": band(arr)}
    result["twobound"] = {}
    for t in tables[snap].get("twobound", []):
        f = t.ym_fit()
        result["twobound"][t.run] = np.array([f.alpha, f.beta, f.scatter_dex])
    fid = tables[snap].get("fiducial", [])
    if fid:
        f = fid[0].ym_fit()
        result["fiducial"] = np.array([f.alpha, f.beta, f.scatter_dex])
    else:
        result["fiducial"] = None
    return result


def support_fraction(model_curves: np.ndarray, data_values: np.ndarray, data_errors: np.ndarray) -> dict:
    """Plan task 5: fraction of design points within 1/2 sigma of the data.

    Per-point chi over the provided bins with diagonal errors (gate-level;
    the A5 likelihood does this properly with full covariances + Sigma_model).
    """
    resid = (model_curves - data_values[None, :]) / data_errors[None, :]
    rms = np.sqrt(np.nanmean(resid**2, axis=1))
    return {
        "frac_within_1sig": float(np.mean(rms <= 1.0)),
        "frac_within_2sig": float(np.mean(rms <= 2.0)),
        "best_point_rms": float(np.nanmin(rms)),
        "rms_per_point": rms,
    }


def save_envelopes(out_dir: Path, snap: str, envelopes: dict) -> None:
    """Persist one snapshot's envelope set + annotations as npz + json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    flat: dict[str, np.ndarray] = {}

    def _flatten(prefix: str, obj) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(f"{prefix}__{k}", v)
        elif obj is not None:
            flat[prefix] = np.asarray(obj)

    _flatten(f"snap{snap}", envelopes)
    np.savez(out_dir / f"envelopes_snap{snap}.npz", **flat)
    (out_dir / f"envelopes_snap{snap}.annotations.json").write_text(
        json.dumps(ANNOTATIONS, indent=2)
    )
