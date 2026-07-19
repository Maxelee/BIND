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

GATE v2 (amendment registered in JOINT_AB_PLAN.md 2026-07-18, after the
sobol absolute-frame FAIL): B4 coordinates are ln-ratios against the
bind PRODUCTION fiducial's halo_scaling, and that reference is offset
from a same-theta re-run by ~4% in gas mass (measured directly: the
twobound grid's fiducial-theta node run_0018 sits at (-0.0387, +0.0143),
not (0,0)). v2 re-applies the same criteria after removing the per-suite
reference offset: twobound subtracts run_0018's measured coords (a
measurement, not a fit; run_0018 excluded from the pass fraction); sobol
fits one constant per coordinate (slope NOT fitted; 2 dof / 209 runs,
disclosed) pending Popeye's re-measurement of the reference. Both
verdicts (v1 absolute, v2 re-referenced) are emitted.

Run: python analysis/paper3a/scripts/run_ab_gate.py [--sobol]
Out: wp6_propagation/ab_gate[_sobol].json (+ figures/ab_gate[_sobol].png)
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


SOBOL_COORDS = Path("/mnt/ceph/users/mlee1/paper3/B/wp4_mocks/sobol/"
                    "sb35_coords.npz")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sobol", action="store_true",
                    help="use the 253-point Sobol coordinate table "
                         "(Popeye session-9 gate upgrade) instead of the "
                         "60 twobound runs")
    args = ap.parse_args()

    if args.sobol:
        if not SOBOL_COORDS.exists():
            raise SystemExit(f"{SOBOL_COORDS} not rsync'd yet (⛔ Max, "
                             "few KB from Popeye ceph wp4_mocks/sobol/)")
        g = np.load(SOBOL_COORDS, allow_pickle=True)
        params = np.asarray(g["params"] if "params" in g.files
                            else g["theta"], float)
        meas_m = np.asarray(g["delta_ln_mgas"], float)
        meas_t = np.asarray(g["delta_ln_t"], float)
        run_names = [str(r) for r in g["run"]]
        label = f"sobol ({len(params)} runs)"
    else:
        g = np.load(GRID, allow_pickle=True)
        names = [str(n) for n in g["run_names"]]
        tb = [i for i, n in enumerate(names) if n not in ("bind", "truth")
              and "bind/run_0000" not in n]
        params = np.asarray(g["params"], float)[tb]
        meas_m = np.asarray(g["delta_ln_mgas"], float)[tb]
        meas_t = np.asarray(g["delta_ln_t"], float)[tb]
        run_names = [names[i] for i in tb]
        label = f"twobound ({len(tb)} runs)"
    print(f"gate frame: {label}")

    U = np.array([pm.astro_physical_to_unit(p[pm.ASTRO_IDX]) for p in params])
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)

    # v2 reference offset (JOINT_AB_PLAN amendment 2026-07-18): B4 coords
    # are ratios against the bind PRODUCTION fiducial, which is offset from
    # a same-theta re-run. twobound: measured directly at its fiducial-theta
    # node; sobol: fit one constant per coordinate (slope not fitted).
    fid_node = None
    if not args.sobol:
        d_fid = np.linalg.norm(U - u_fid, axis=1)
        i_fid = int(np.argmin(d_fid))
        if d_fid[i_fid] < 1e-8:
            fid_node = i_fid
            print(f"v2 reference node: {run_names[i_fid]} "
                  f"offset=({meas_m[i_fid]:+.4f}, {meas_t[i_fid]:+.4f})")

    emu = StatsEmulator.load()
    raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz", allow_pickle=False)
    man = json.loads(str(raw["manifest"]))
    edges = np.asarray(man["meta"]["mass_bins"], float)
    w = bin_weights(edges)

    pred = delta_coords(emu, U, u_fid, w)

    def criteria(p, e, meas, keep):
        within = np.abs(p - meas) <= 1.5 * e
        strong = np.abs(meas) > 2 * e
        sel_s = strong & keep
        signs_ok = np.all(np.sign(p[sel_s]) == np.sign(meas[sel_s])) \
            if sel_s.any() else True
        return {
            "frac_within_1p5err": float(np.mean(within[keep])),
            "n_strong": int(sel_s.sum()),
            "signs_ok": bool(signs_ok),
            "slope_pred_vs_meas": float(np.polyfit(meas, p, 1)[0]),
            "rms_resid": float(np.sqrt(np.mean((p - meas)[keep] ** 2))),
            "median_err": float(np.median(e)),
            "meas_range": [float(meas.min()), float(meas.max())],
            "PASS": bool(np.mean(within[keep]) >= 0.80 and signs_ok),
        }

    keep_all = np.ones(len(params), bool)
    keep_v2 = keep_all.copy()
    if fid_node is not None:
        keep_v2[fid_node] = False        # the reference defines the offset

    # v3 table (Popeye 2026-07-18): the fiducial-theta node's measured
    # delta is embedded -> a fit-free offset for the sobol frame too
    v3_off = None
    if args.sobol and "fiducial_theta_node_twobound_run_0018_delta" in g.files:
        v3_off = np.asarray(
            g["fiducial_theta_node_twobound_run_0018_delta"], float)

    res, res_v2, res_ff, ref_offset = {}, {}, {}, {}
    for ki, (key, meas) in enumerate((("dln_mgas", meas_m),
                                      ("dln_t", meas_t))):
        p, e = pred[key], pred[key + "_err"]
        res[key] = criteria(p, e, meas, keep_all)
        if fid_node is not None:
            off = float(meas[fid_node])
            ref_offset[key] = {"value": off, "method": "measured at "
                               + run_names[fid_node]}
        else:
            off = float(np.mean(meas - p))       # slope fixed at 1; 1 dof
            ref_offset[key] = {"value": off,
                               "method": "constant fit (slope=1), pending "
                                         "Popeye reference re-measurement"}
        res_v2[key] = criteria(p, e, meas - off, keep_v2)
        if v3_off is not None:
            res_ff[key] = criteria(p, e, meas - v3_off[ki], keep_v2)
    verdict = bool(all(res[k]["PASS"] for k in res))
    verdict_v2 = bool(all(res_v2[k]["PASS"] for k in res_v2))
    out = {"gate": res, "PASS": verdict,
           "gate_v2_dereferenced": res_v2, "PASS_v2": verdict_v2,
           "reference_offset": ref_offset,
           "gate_frame": label,
           "n_runs": int(len(params)),
           "weights_per_bin": w.tolist(),
           "mass_floor_msun": MASS_MIN_MSUN,
           "criteria": "v1: JOINT_AB_PLAN.md step 2 (pre-registered); "
                       "v2: reference-offset amendment (2026-07-18)"}
    if v3_off is not None:
        out["gate_v2_fitfree_run0018"] = res_ff
        out["PASS_v2_fitfree"] = bool(all(r["PASS"] for r in res_ff.values()))
        out["fitfree_offset"] = v3_off.tolist()
    tag = "_sobol" if args.sobol else ""
    (WP6 / f"ab_gate{tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, meas, lab in ((axes[0], "dln_mgas", meas_m,
                                r"$\Delta \ln M_{\rm gas}$"),
                               (axes[1], "dln_t", meas_t,
                                r"$\Delta \ln T$")):
        p, e = pred[key], pred[key + "_err"]
        off = ref_offset[key]["value"]
        ax.errorbar(meas - off, p, yerr=e, fmt="o", ms=4, color="#0072B2",
                    elinewidth=0.8)
        lim = np.array([min(meas.min() - off, p.min()),
                        max(meas.max() - off, p.max())])
        ax.plot(lim, lim, "k:", lw=0.8)
        ax.set_xlabel(f"B4 measured {lab} (ref-offset {off:+.3f} removed)")
        ax.set_ylabel(f"statsemu mirror {lab}")
        ax.set_title(f"{key} v2: {res_v2[key]['frac_within_1p5err']:.0%} "
                     f"within 1.5err, slope "
                     f"{res_v2[key]['slope_pred_vs_meas']:.2f} "
                     f"[{'PASS' if res_v2[key]['PASS'] else 'FAIL'}] "
                     f"(v1 abs: "
                     f"{'PASS' if res[key]['PASS'] else 'FAIL'})", fontsize=8)
    fig.suptitle("JOINT_AB_PLAN step-2 gate v2: cross-frame validation, "
                 "reference offset removed")
    fig.tight_layout()
    (WP6 / "figures").mkdir(exist_ok=True)
    fig.savefig(WP6 / "figures" / f"ab_gate{tag}.png", dpi=150)
    print(f"wrote {WP6}/ab_gate{tag}.json + figures/ab_gate{tag}.png")


if __name__ == "__main__":
    main()
