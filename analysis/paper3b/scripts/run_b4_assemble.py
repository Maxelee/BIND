"""WP-B4 assembly: grid npzs -> model grid table + R2 validation + MC budget.

Serial, fast (reads the per-unit summaries, not the maps). Run AFTER the grid
job. Outputs to <out>:

  model_grid.npz  — per-θ: params (35,), y_mean (nbin, nrad), y_mc_err,
                    n_per_bin, abundance_per_deg2, delta_ln_mgas, delta_ln_t
                    (halo-matched vs the bind fiducial, mass-weighted over
                    M > MASS_MIN — provisional definition, flagged in
                    REPORT.md pending the Paper-2 §5 exact form).
  b4_summary.json — R2 truth-vs-bind ratios + pass/fail vs the ±10%
                    tolerance, the MC-error budget vs the frozen σ_stat
                    (ERROR BARS ONLY — the frozen central values are never
                    compared to the model here; blinding discipline), run
                    inventory, and the weight-scheme provenance.

The (Δln M_gas, Δln T) coordinates use the per-run halo_scaling.npz, which
shares one DMO halo skeleton across all runs (verified: identical halo count
and masses), so the coordinates are halo-matched log-ratios.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RUNS_ROOT = Path("/mnt/home/mlee1/ceph/bind_science/runs")
DEFAULT_OUT = Path("/mnt/home/mlee1/ceph/paper3/B/wp4_mocks")
MASS_MIN = 10 ** 13.5          # Msun; peak-relevant halo mass floor (provisional)
PATCH_AREA_DEG2 = 25.0         # 5 deg x 5 deg
R2_TOLERANCE = 0.10            # plan task 4: ±5-10% -> use the outer bound


def halo_coordinates(run_dir: Path, fid: dict) -> tuple[float, float]:
    """(delta_ln_mgas, delta_ln_t) halo-matched, mass-weighted over M > MASS_MIN."""
    d = np.load(run_dir / "halo_scaling.npz")
    m = np.asarray(d["halo_mass"], dtype=np.float64)
    if m.shape != fid["mass"].shape or not np.allclose(m, fid["mass"]):
        raise RuntimeError(f"{run_dir}: halo skeleton differs from fiducial")
    sel = (m > MASS_MIN) & (fid["f_gas"] > 0) & (d["f_gas_500c"] > 0) \
        & (fid["T_mw"] > 0) & (d["T_mw_500c"] > 0)
    w = m[sel]
    dlnm = np.average(np.log(np.asarray(d["f_gas_500c"])[sel] / fid["f_gas"][sel]), weights=w)
    dlnt = np.average(np.log(np.asarray(d["T_mw_500c"])[sel] / fid["T_mw"][sel]), weights=w)
    return float(dlnm), float(dlnt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--grid-subdir", default="grid")
    args = ap.parse_args()
    out = Path(args.out)
    grid_dir = out / args.grid_subdir
    # products carry the grid variant in their name (grid_tfwiener -> _tfwiener)
    tag = "" if args.grid_subdir == "grid" else "_" + args.grid_subdir.removeprefix("grid_")
    model_grid_path = out / f"model_grid{tag}.npz"
    summary_path = out / f"b4_summary{tag}.json"

    files = sorted(grid_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no grid bundles under {grid_dir} — run the grid job first")

    fid_hs = np.load(RUNS_ROOT / "bind/run_0000/halo_scaling.npz")
    fid = {"mass": np.asarray(fid_hs["halo_mass"], dtype=np.float64),
           "f_gas": np.asarray(fid_hs["f_gas_500c"], dtype=np.float64),
           "T_mw": np.asarray(fid_hs["T_mw_500c"], dtype=np.float64)}

    rows: dict[str, dict] = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        cat, run = str(d["category"]), str(d["run"])
        npatch = int(d["counts"].shape[0])
        entry = {
            "params": np.asarray(d["params"], dtype=float),
            "y_mean": np.asarray(d["y_mean"], dtype=float),
            "y_mc_err": np.asarray(d["y_mc_err"], dtype=float),
            "n_per_bin": np.asarray(d["n_per_bin"], dtype=float),
            "abundance_per_deg2": np.asarray(d["n_per_bin"], dtype=float)
                                  / (npatch * PATCH_AREA_DEG2),
            "n_patches": npatch,
        }
        if cat == "twobound":
            entry["coords"] = halo_coordinates(RUNS_ROOT / cat / run, fid)
        rows[f"{cat}/{run}"] = entry

    # ── model grid table (twobound runs) ─────────────────────────────────────
    tb = sorted(k for k in rows if k.startswith("twobound/"))
    if tb:
        first = rows[tb[0]]
        np.savez(model_grid_path,
                 run_names=np.array(tb),
                 params=np.stack([rows[k]["params"] for k in tb]),
                 y_mean=np.stack([rows[k]["y_mean"] for k in tb]),
                 y_mc_err=np.stack([rows[k]["y_mc_err"] for k in tb]),
                 n_per_bin=np.stack([rows[k]["n_per_bin"] for k in tb]),
                 abundance_per_deg2=np.stack([rows[k]["abundance_per_deg2"] for k in tb]),
                 delta_ln_mgas=np.array([rows[k]["coords"][0] for k in tb]),
                 delta_ln_t=np.array([rows[k]["coords"][1] for k in tb]),
                 mass_min=MASS_MIN,
                 nbin=first["y_mean"].shape[0], nrad=first["y_mean"].shape[1])

    summary: dict = {"n_units": len(rows),
                     "categories": sorted({k.split("/")[0] for k in rows})}

    # ── R2: truth vs bind through the identical chain ────────────────────────
    if "truth/run_0000" in rows and "bind/run_0000" in rows:
        t, b = rows["truth/run_0000"], rows["bind/run_0000"]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = b["y_mean"] / t["y_mean"]
            rel_err = np.sqrt((b["y_mc_err"] / b["y_mean"]) ** 2
                              + (t["y_mc_err"] / t["y_mean"]) ** 2)
        occupied = (t["n_per_bin"] > 50) & (b["n_per_bin"] > 50)
        dev = np.abs(ratio[occupied] - 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dev_sigma = np.abs(ratio - 1.0) / rel_err
        summary["r2_selection_validation"] = {
            # significance-aware companion to the fixed tolerance: on low-count
            # chains (tf grids) the per-cell MC error (~9%) makes the fixed 10%
            # trip statistically; report both, never silently relax either
            "max_dev_sigma_occupied": float(np.nanmax(dev_sigma[occupied]))
                                      if occupied.any() else None,
            "ratio_bind_over_truth": np.where(np.isfinite(ratio), ratio, None).tolist(),
            "ratio_mc_rel_err": np.where(np.isfinite(rel_err), rel_err, None).tolist(),
            "occupied_bins": occupied.tolist(),
            "max_abs_dev_occupied": float(dev.max()) if dev.size else None,
            "tolerance": R2_TOLERANCE,
            "pass": bool(dev.size and (dev <= R2_TOLERANCE).all()),
            "note": ("bind fiducial vs TNG300-hydro truth lightcone through the "
                     "IDENTICAL survey-realism + frozen-measurement chain; the "
                     "residual is archived as the B4 selection-model error"),
        }

    # ── MC budget vs frozen σ_stat (error bars only — no central values) ─────
    try:
        from analysis.paper3b.stack.frozen import load_frozen
        sigma_stat = np.sqrt(np.diag(load_frozen("wiener").cov_stat))
        worst = 0.0
        for k in tb or [k for k in rows]:
            mc = rows[k]["y_mc_err"][1:5, 2]              # B5 bins, fiducial radius
            ok = np.isfinite(mc)
            if ok.any():
                worst = max(worst, float((mc[ok] / sigma_stat[ok]).max()))
        summary["mc_budget"] = {
            "worst_mc_over_sigma_stat": worst,
            "target": "<= 0.2 (MC error subdominant to the frozen statistical error)",
            "pass": bool(worst <= 0.2),
        }
    except Exception as e:                                # frozen archive absent
        summary["mc_budget"] = {"error": str(e)}

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", model_grid_path, "and", summary_path)


if __name__ == "__main__":
    main()
