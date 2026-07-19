"""JOINT_AB_PLAN steps 1-2: the statsemu mirror of B4's decomposition
coordinates + the pre-registered twobound cross-frame validation gate.

B4's definition (`run_b4_assemble.py::halo_coordinates`, read from the
paper3b branch): per run, the HALO-MASS-weighted average over matched
halos with M > 10^13.5 Msun of ln(f_gas_500c / f_gas_500c^fid) and
ln(T_mw_500c / T_mw_500c^fid).

The statsemu mirror: the sb35 scaling targets give per-mass-bin
f_gas / T at snap 096 per theta. The mirror coordinate is

    Delta ln X(theta) = sum_b w_b [ln X_b(theta) - ln X_b(theta_fid)]

with w_b = sum of halo masses in bin b (m > 10^13.5, halos above the top
bin edge assigned to the top bin), taken from the shared-DMO fiducial
snap-096 catalog (`wp3_gate/operator_tables_v3/fiducial_run_0000_snap096`,
logm200 column — the same mass frame the composite halo_scaling uses).

Documented frame differences (the gate MEASURES their net effect):
- B4: lightcone composites, multiple slabs/redshifts along the cone,
  circular apertures at R500c; A: the snap-096 box scalings.
- Mass definitions: composite `halo_masses` vs the gate table's m200.
- statsemu bins are 0.25-dex aggregates, not per-halo.

Gate criteria (pre-registered in JOINT_AB_PLAN.md): >= 80% of the 60
twobound runs within 1.5x the per-run error (propagated GP std + the
CV floor, in ln space) per coordinate, and no sign disagreements among
runs with |measured| > 2x the error. FAIL -> stop, reconcile frames.

Run: python analysis/paper3a/scripts/run_ab_gate.py
Out: wp6_propagation/ab_gate.json (+ figures/ab_gate.png)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.emulator.statsemu import WP6, StatsEmulator  # noqa: E402

GRID = Path("/mnt/ceph/users/mlee1/paper3/B/wp4_mocks/model_grid_tfwiener.npz")
FID_TABLE = Path("/mnt/ceph/users/mlee1/paper3/A/wp3_gate/operator_tables_v3/"
                 "fiducial_run_0000_snap096.npz")
MASS_MIN_MSUN = 10**13.5          # B4's peak-relevant floor
CV_LN = {"scaling_f_gas": np.sqrt(2) * 0.023,   # 8-fold CV, ln-frame floor
         "scaling_T": np.sqrt(2) * 0.017}


def bin_weights(edges: np.ndarray) -> np.ndarray:
    d = np.load(FID_TABLE, allow_pickle=True)
    m200 = 10.0 ** np.asarray(d["logm200"], float)        # Msun-frame masses
    m200 = m200[m200 > MASS_MIN_MSUN]
    idx = np.clip(np.digitize(np.log10(m200), edges) - 1, 0, len(edges) - 2)
    w = np.zeros(len(edges) - 1)
    for i, mm in zip(idx, m200):
        w[i] += mm
    return w / w.sum()


def delta_coords(emu: StatsEmulator, U: np.ndarray, u_fid: np.ndarray,
                 w: np.ndarray) -> dict:
    """Weighted Delta-ln coordinates; only bins with w > 0 enter (the
    sub-13.5 bins have zero weight, and evaluating log there would poison
    the weighted sum via 0 * nan). GP overshoot into <= 0 predictions in
    a WEIGHTED bin (possible for raw-transform scaling_f_gas at the
    extreme-evacuation corner) is floored at 1e-4 and counted."""
    out = {}
    sel = w > 0
    ws = w[sel]
    for tgt, key in (("scaling_f_gas", "dln_mgas"), ("scaling_T", "dln_t")):
        pred, sd = emu.predict(U, tgt, return_std=True)
        fid, sdf = emu.predict(u_fid, tgt, return_std=True)
        pred, sd = np.atleast_2d(pred)[:, sel], np.atleast_2d(sd)[:, sel]
        fid, sdf = fid[sel], sdf[sel]
        n_bad = int(np.sum(pred <= 0))
        pred = np.maximum(pred, 1e-4)
        dln = np.log(pred) - np.log(fid)[None, :]
        sln = np.sqrt((sd / pred) ** 2 + ((sdf / fid) ** 2)[None, :])
        out[key] = dln @ ws
        out[key + "_err"] = np.sqrt(((sln * ws[None, :]) ** 2).sum(axis=1)
                                    + CV_LN[tgt] ** 2)
        out[key + "_n_floored"] = n_bad
    return out


def main() -> None:
    g = np.load(GRID, allow_pickle=True)
    names = [str(n) for n in g["run_names"]]
    tb = [i for i, n in enumerate(names) if "twobound" in n or "run_" in n]
    # twobound rows: everything that is not the bind fiducial / truth entry
    tb = [i for i, n in enumerate(names) if n not in ("bind", "truth")
          and "bind/run_0000" not in n]
    params = np.asarray(g["params"], float)[tb]
    meas_m = np.asarray(g["delta_ln_mgas"], float)[tb]
    meas_t = np.asarray(g["delta_ln_t"], float)[tb]

    U = np.array([pm.astro_physical_to_unit(p[pm.ASTRO_IDX]) for p in params])
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)

    emu = StatsEmulator.load()
    raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz", allow_pickle=False)
    man = json.loads(str(raw["manifest"]))
    edges = np.asarray(man["meta"]["mass_bins"], float)
    w = bin_weights(edges)

    pred = delta_coords(emu, U, u_fid, w)

    res = {}
    for key, meas in (("dln_mgas", meas_m), ("dln_t", meas_t)):
        p, e = pred[key], pred[key + "_err"]
        within = np.abs(p - meas) <= 1.5 * e
        strong = np.abs(meas) > 2 * e
        signs_ok = np.all(np.sign(p[strong]) == np.sign(meas[strong])) \
            if strong.any() else True
        slope = float(np.polyfit(meas, p, 1)[0])
        res[key] = {
            "frac_within_1p5err": float(np.mean(within)),
            "n_strong": int(strong.sum()),
            "signs_ok": bool(signs_ok),
            "slope_pred_vs_meas": slope,
            "rms_resid": float(np.sqrt(np.mean((p - meas) ** 2))),
            "median_err": float(np.median(e)),
            "meas_range": [float(meas.min()), float(meas.max())],
            "PASS": bool(np.mean(within) >= 0.80 and signs_ok),
        }
    verdict = bool(all(res[k]["PASS"] for k in res))
    out = {"gate": res, "PASS": verdict,
           "n_twobound": len(tb),
           "weights_per_bin": w.tolist(),
           "mass_floor_msun": MASS_MIN_MSUN,
           "criteria": "JOINT_AB_PLAN.md step 2 (pre-registered)"}
    (WP6 / "ab_gate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, meas, lab in ((axes[0], "dln_mgas", meas_m,
                                r"$\Delta \ln M_{\rm gas}$"),
                               (axes[1], "dln_t", meas_t,
                                r"$\Delta \ln T$")):
        p, e = pred[key], pred[key + "_err"]
        ax.errorbar(meas, p, yerr=e, fmt="o", ms=4, color="#0072B2",
                    elinewidth=0.8)
        lim = np.array([min(meas.min(), p.min()), max(meas.max(), p.max())])
        ax.plot(lim, lim, "k:", lw=0.8)
        ax.set_xlabel(f"B4 measured {lab}")
        ax.set_ylabel(f"statsemu mirror {lab}")
        ax.set_title(f"{key}: {res[key]['frac_within_1p5err']:.0%} within "
                     f"1.5err, slope {res[key]['slope_pred_vs_meas']:.2f} "
                     f"[{'PASS' if res[key]['PASS'] else 'FAIL'}]", fontsize=9)
    fig.suptitle("JOINT_AB_PLAN step-2 gate: cross-frame coordinate validation")
    fig.tight_layout()
    (WP6 / "figures").mkdir(exist_ok=True)
    fig.savefig(WP6 / "figures" / "ab_gate.png", dpi=150)
    print(f"wrote {WP6}/ab_gate.json + figures/ab_gate.png")


if __name__ == "__main__":
    main()
