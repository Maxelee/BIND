"""SYSTEMATICS-HUNT: is the sobol/twobound reference-offset disagreement a
genuine per-suite paint difference, or slope leakage from the fit?

FINDINGS_b_model.md candidate 2. The twobound grid's fiducial-theta node
(run_0018) MEASURES a reference offset of (-0.0387, +0.0143). The Sobol
suite has no fiducial-theta node, so `run_ab_gate.py --sobol` FITS one
constant per coordinate with the slope pinned at 1:

    off = mean(meas - p)          # run_ab_gate.py, sobol branch

giving (-0.05474, +0.02760). The 0.016 / 0.013 difference is carried as
`bblock.OFFSET_SYS = [0.016, 0.014]`. Popeye was asked to paint one
sobol-pipeline run at fiducial theta to settle which it is; that paint was
never executed.

The zero-GPU rusty proxy: re-fit the sobol offset with the SLOPE FREE.

    fixed-slope (current):   meas = p + c          ->  c = mean(meas - p)
    free-slope  (this test): meas = a + b*p        ->  OLS

`p` is the statsemu mirror's Delta-ln coordinate, which is ZERO at
fiducial theta by construction (delta_coords is a ratio against u_fid).
So the free-slope intercept `a` is exactly the suite's own extrapolation
of what a fiducial-theta paint would measure -- the same quantity
run_0018 measures directly for twobound. That makes `a` the rusty-local
prediction of the un-executed Popeye paint, not merely a fit diagnostic.

The arithmetic is transparent:

    a - c = (1 - b) * mean(p)

so leakage requires BOTH a slope departure from 1 AND a design whose
mirror coordinates are off-centre.

PRE-REGISTERED CRITERION (fixed before running, from FINDINGS_b_model.md):
    |a - c| >= 0.010  =>  the 0.016 is substantially SLOPE LEAKAGE; the
                          Popeye fiducial-theta paint drops in priority.
    |a - c| <  0.010  =>  the offset is a genuine PER-SUITE PAINT
                          DIFFERENCE; OFFSET_SYS stands at 0.016 pending
                          the paint.

Calibration, stated so the result is not overread: the whole
reference-offset question is worth 1.7-2.9% in Y. NEITHER outcome changes
the B deficit. This decides only whether a requested GPU round is needed.

Run: python analysis/paper3a/scripts/run_syshunt_sobol_offset.py
Out: systematics_hunt/sobol_offset_slope_leakage.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.emulator.statsemu import WP6, StatsEmulator  # noqa: E402
from analysis.paper3a.scripts.run_ab_gate import (  # noqa: E402
    SOBOL_COORDS, bin_weights, delta_coords)

OUT = Path("/mnt/ceph/users/mlee1/paper3/A/systematics_hunt")
THRESH = 0.010          # pre-registered intercept-movement threshold


def ols(x, y):
    """y = a + b x. Returns a, b, sigma_a, sigma_b, resid rms."""
    n = len(x)
    xb, yb = x.mean(), y.mean()
    sxx = np.sum((x - xb) ** 2)
    b = np.sum((x - xb) * (y - yb)) / sxx
    a = yb - b * xb
    r = y - (a + b * x)
    s2 = np.sum(r**2) / (n - 2)
    return (float(a), float(b),
            float(np.sqrt(s2 * (1.0 / n + xb**2 / sxx))),
            float(np.sqrt(s2 / sxx)),
            float(np.sqrt(np.mean(r**2))))


def tls(x, y):
    """Total least squares (errors on both axes), intercept at x=0."""
    xb, yb = x.mean(), y.mean()
    u, v = x - xb, y - yb
    sxx, syy, sxy = np.sum(u * u), np.sum(v * v), np.sum(u * v)
    b = ((syy - sxx) + np.sqrt((syy - sxx) ** 2 + 4 * sxy**2)) / (2 * sxy)
    return float(yb - b * xb), float(b)


def main() -> None:
    g = np.load(SOBOL_COORDS, allow_pickle=True)
    params = np.asarray(g["params"], float)
    meas = {"dln_mgas": np.asarray(g["delta_ln_mgas"], float),
            "dln_t": np.asarray(g["delta_ln_t"], float)}
    node = np.asarray(g["fiducial_theta_node_twobound_run_0018_delta"], float)

    U = np.array([pm.astro_physical_to_unit(p[pm.ASTRO_IDX]) for p in params])
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)

    emu = StatsEmulator.load()
    raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz",
                  allow_pickle=False)
    man = json.loads(str(raw["manifest"]))
    w = bin_weights(np.asarray(man["meta"]["mass_bins"], float))
    pred = delta_coords(emu, U, u_fid, w)

    res = {
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "coords": str(SOBOL_COORDS),
            "n_runs": int(len(params)),
            "mirror": "statsemu delta_coords (run_ab_gate.delta_coords), "
                      "identical call to the sobol gate",
            "twobound_fiducial_theta_node_run_0018": node.tolist(),
            "note": "mirror prediction p is 0 at fiducial theta by "
                    "construction, so the free-slope intercept a is the "
                    "suite's prediction of a fiducial-theta paint.",
        },
        "preregistered_criterion": {
            "threshold": THRESH,
            "ge": "slope leakage; Popeye fiducial-theta paint lower priority",
            "lt": "genuine per-suite paint difference; OFFSET_SYS stands",
        },
        "coordinates": {},
    }

    verdicts = {}
    for i, key in enumerate(("dln_mgas", "dln_t")):
        m, p, e = meas[key], pred[key], pred[key + "_err"]
        c = float(np.mean(m - p))                        # current, slope=1
        a, b, sa, sb, rms = ols(p, m)                    # slope free
        a_t, b_t = tls(p, m)
        # reverse orientation (regress p on m), implied offset at p = 0
        a_r, b_r, sar, sbr, _ = ols(m, p)
        a_rev = -a_r / b_r
        wts = 1.0 / e**2
        cw = float(np.sum(wts * (m - p)) / np.sum(wts))
        aw = float(np.sum(wts * (m - b * p)) / np.sum(wts))
        move = a - c
        v = "SLOPE_LEAKAGE" if abs(move) >= THRESH else "PER_SUITE_PAINT"
        verdicts[key] = v
        res["coordinates"][key] = {
            "fixed_slope_offset_c": c,
            "fixed_slope_offset_c_recorded_in_ab_gate_sobol_json":
                (-0.05474079367222181 if i == 0 else 0.027603275877059907),
            "free_slope_intercept_a": a,
            "free_slope_intercept_sigma": sa,
            "free_slope_slope_b": b,
            "free_slope_slope_sigma": sb,
            "slope_deviation_b_minus_1": b - 1.0,
            "slope_deviation_in_sigma": (b - 1.0) / sb,
            "mean_mirror_pred_p": float(p.mean()),
            "identity_check_a_minus_c": {
                "a_minus_c": move,
                "one_minus_b_times_mean_p": float((1.0 - b) * p.mean()),
            },
            "resid_rms_free_slope": rms,
            "resid_rms_fixed_slope": float(np.sqrt(np.mean((m - p - c) ** 2))),
            "abs_move": abs(move),
            "verdict": v,
            "cross_checks": {
                "tls_intercept": a_t, "tls_slope": b_t,
                "tls_move_vs_c": a_t - c,
                "reverse_regression_implied_offset": float(a_rev),
                "reverse_move_vs_c": float(a_rev - c),
                "inverse_variance_weighted_c": cw,
                "inverse_variance_weighted_a": aw,
            },
            "vs_twobound_node": {
                "twobound_node": float(node[i]),
                "c_minus_node": c - float(node[i]),
                "a_minus_node": a - float(node[i]),
                "carried_OFFSET_SYS": [0.016, 0.014][i],
            },
        }

    # Cross-check: the twobound suite's offset is MEASURED at its
    # fiducial-theta node (run_0018), so slope leakage cannot touch it --
    # the leakage question is sobol-only. Recorded here because the two
    # suites' mirror-vs-B4 slopes depart from 1 in OPPOSITE directions,
    # which rules out a single shared mirror miscalibration.
    tb = json.loads((WP6 / "ab_gate.json").read_text())
    res["twobound_cross_check"] = {
        "source": str(WP6 / "ab_gate.json"),
        "offset_method": {k: tb["reference_offset"][k]["method"]
                          for k in ("dln_mgas", "dln_t")},
        "slope_pred_vs_meas_twobound": {
            k: tb["gate"][k]["slope_pred_vs_meas"]
            for k in ("dln_mgas", "dln_t")},
        "slope_pred_vs_meas_sobol": {
            k: float(np.polyfit(meas[k], pred[k], 1)[0])
            for k in ("dln_mgas", "dln_t")},
        "note": "twobound < 1, sobol > 1 in dln_mgas: the slope departure "
                "is not a single shared mirror miscalibration. It cannot "
                "bias the twobound offset in any case, which is measured "
                "at a node rather than fitted.",
    }

    res["verdict_overall"] = (
        "SLOPE_LEAKAGE" if any(v == "SLOPE_LEAKAGE" for v in verdicts.values())
        else "PER_SUITE_PAINT")
    res["verdict_per_coordinate"] = verdicts

    OUT.mkdir(parents=True, exist_ok=True)
    pth = OUT / "sobol_offset_slope_leakage.json"
    pth.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["coordinates"], indent=2))
    print("OVERALL:", res["verdict_overall"])
    print(f"wrote {pth}")


if __name__ == "__main__":
    main()
