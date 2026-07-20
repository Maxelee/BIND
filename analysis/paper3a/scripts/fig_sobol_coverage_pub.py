"""Publication figure: does the SB35 design contain ANY unit that fits our data?

The Paper-2 house figure (f_gas envelope + DESI x ACT stars) invites the eye
to ask "do some Sobol curves pass through the data?". This figure asks that
question properly, for all three Paper-3 probes at once, and separates the
two things the eye conflates:

  top row    -- native units: the SB35 design family (thin curves) against
                the frozen data vector. This is the Paper-2 look.
  bottom row -- pull (model - data)/sigma on a SHARED axis. A unit "goes
                through the data" only if its curve stays inside the band
                across EVERY bin. None does, on any probe.

Frames (the traps this script gets right):
- `snap096_X_sb35` is ALREADY the unit cube. Passing it through
  `astro_physical_to_unit` a second time sends it to [-6, 1109] and the
  emulator silently extrapolates. Verified in-cube at runtime.
- A-side spaghetti is pushed through the SAME forward model as the
  likelihood (CylToSph + paint-bias for f_gas; CAP operator + satellite
  dilution for kSZ), not the raw painted-cylindrical targets.
- The kSZ f_sat nuisance is held at its midpoint 0.20; it is a <=12%
  amplitude knob at the innermost radius and <1% at the outermost, and
  minimising chi2 over it changes the best unit from 38.1 to 34.9 (/9).
- B-side uses the 253 raytraced SB35 units' MEASURED <Y(nu)> at the 4'
  headline radius (no GP), matching B5's frozen per-unit verdict recipe.

Run: python analysis/paper3a/scripts/fig_sobol_coverage_pub.py
Out: wp6_propagation/figures/sobol_coverage_pub.{pdf,png}
     wp6_propagation/sobol_coverage.json  (+ plans-repo copies)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.paper3a.inference.bblock import (  # noqa: E402
    RADIUS_IDX, SCI, BBlock, TWOBOUND_REF_OFFSET)
from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.ksz import KszBlock  # noqa: E402
from analysis.paper3a.style import (  # noqa: E402
    COL, W_DOUBLE, apply, condition_tag, data_points, fiducial_line, refline,
    spaghetti)

# The 253-run raytraced SB35 grid. NOT bblock.GRID, which defaults to the
# 60-unit twobound grid the joint fit's GP was trained on.
BSOBOL = Path("/mnt/home/mlee1/ceph/paper3/B/wp4_mocks/sobol")
BGRID = BSOBOL / "model_grid_tfwiener_sb35.npz"
B5_VERDICT = BSOBOL / "b5_sobol_verdict.json"

WP4 = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator")
WP6 = Path("/mnt/ceph/users/mlee1/paper3/A/wp6_propagation")
PLANS_FIG = Path("/mnt/home/mlee1/bind-paper3-plans/figures_AB")
F_SAT_MID_UNIT = 0.5


def _load_a_side():
    """512 SB35 units -> f_gas and kSZ predictions in the likelihood frame."""
    d = np.load(WP4 / "gasemu_dataset_v4.npz", allow_pickle=True)
    U = d["snap096_X_sb35"].astype(float)
    runs = [str(r) for r in d["snap096_runs_sb35"]]
    if not ((U >= 0) & (U <= 1)).all():
        raise SystemExit(
            "snap096_X_sb35 is not in the unit cube — the design convention "
            "changed; do NOT apply astro_physical_to_unit blindly")

    ksz = KszBlock()
    fgas = FgasBlock(emu=ksz.emu)

    # closure guard: the emulator must reproduce the measured targets at its
    # own design points, else the frame is wrong
    Y = d["snap096_Y_sb35"].astype(float)
    emu_f = np.atleast_2d(ksz.emu.predict(U, "096", return_std=False)["fgas_med"])
    frac = np.abs((emu_f - Y[:, 0:5]) / Y[:, 0:5])
    if np.median(frac) > 0.02:
        raise SystemExit(f"emulator/measured closure failed: median frac err "
                         f"{np.median(frac):.4f} (>2%) — frame mismatch")

    U31 = np.concatenate([U, np.full((len(U), 1), F_SAT_MID_UNIT)], axis=1)
    from analysis.paper3a.emulator import params_meta as pm
    u_fid = np.full(31, F_SAT_MID_UNIT)
    u_fid[:30] = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)

    pf = fgas.predict(U31)
    pk = ksz.predict(U31)
    return dict(
        runs=runs, closure=float(np.median(frac)),
        fgas=dict(runs=runs, x=np.asarray(fgas.data.bin_centers, float), pred=pf,
                  sig=fgas.sigma(pf), dat=np.asarray(fgas.data.values, float),
                  err=np.asarray(fgas.data.stat_err, float),
                  fid=fgas.predict(u_fid[None, :])[0],
                  chi2=fgas.chi2(U31)),
        ksz=dict(runs=runs, x=np.asarray(ksz.radii, float), pred=pk,
                 sig=np.sqrt(np.diagonal(ksz.cov(pk), axis1=1, axis2=2)),
                 dat=np.asarray(ksz.values, float),
                 err=np.sqrt(np.diag(np.asarray(ksz.cov_data, float))),
                 fid=ksz.predict(u_fid[None, :])[0],
                 chi2=ksz.chi2(U31)))


def _load_b_side():
    """253 raytraced SB35 units' measured <Y(nu)> at the 4' headline radius."""
    g = np.load(BGRID, allow_pickle=True)
    names = [str(n) for n in g["run_names"]]
    keep = [i for i, n in enumerate(names) if "truth" not in n]
    pred = np.asarray(g["y_mean"], float)[keep][:, SCI, RADIUS_IDX]   # (N, 4)
    coords = np.stack([g["delta_ln_mgas"], g["delta_ln_t"]], axis=1)[keep]

    bb = BBlock()
    sig = np.sqrt(bb.var_stat + np.diag(bb.cov_sys))                  # B5 recipe
    r = (pred - bb.data[None, :]) / sig[None, :]
    chi2 = np.sum(r ** 2, axis=1)

    # guard: this must reproduce B5's frozen, pre-registered per-unit verdict
    v = json.loads(B5_VERDICT.read_text())
    if not np.isclose(chi2.min(), v["chi2_min"], rtol=1e-3):
        raise SystemExit(f"B-side recipe drift: chi2_min {chi2.min():.2f} vs "
                         f"frozen B5 verdict {v['chi2_min']:.2f}")

    # reference unit: the SB35 member nearest the TNG fiducial-theta node
    i_fid = int(np.argmin(np.linalg.norm(coords - TWOBOUND_REF_OFFSET, axis=1)))
    return dict(runs=[names[i] for i in keep], x=np.arange(4), pred=pred,
                sig=np.broadcast_to(sig, pred.shape).copy(), dat=bb.data,
                err=np.sqrt(bb.var_stat), fid=pred[i_fid], chi2=chi2,
                fid_run=names[keep[i_fid]])


def _stats(p, label, dof):
    pred, dat, sig, chi2 = p["pred"], p["dat"], p["sig"], p["chi2"]
    lo, hi = pred.min(0), pred.max(0)
    covered = (hi >= dat - p["err"]) & (lo <= dat + p["err"])
    inside = np.abs(pred - dat[None, :]) <= sig
    b = int(np.argmin(chi2))
    return dict(
        probe=label, n_units=int(len(pred)), n_bins=int(pred.shape[1]),
        dof=dof,
        pointwise_bins_reached=int(covered.sum()),
        pointwise_per_bin=covered.astype(int).tolist(),
        n_units_all_bins_within_1sig=int(inside.all(1).sum()),
        n_units_all_bins_within_2sig=int(
            (np.abs(pred - dat[None, :]) <= 2 * sig).all(1).sum()),
        best_run=str(p["runs"][b]), best_chi2=float(chi2[b]),
        best_chi2_per_dof=float(chi2[b] / dof),
        best_residuals_sigma=((pred[b] - dat) / sig[b]).round(2).tolist(),
        median_chi2=float(np.median(chi2)))


def main() -> None:
    apply()
    A = _load_a_side()
    Bs = _load_b_side()

    panels = [
        (A["fgas"], r"$\log_{10} M_{500c}\ [M_\odot/h]$",
         r"$f_{\rm gas}(<R_{500})$", "X-ray $f_{\\rm gas}$ · eRASS1",
         "SB35 (512)", 5, 1.0),
        (A["ksz"], r"$R\ [{\rm arcmin}]$",
         r"$T_{\rm kSZ}\ [\mu{\rm K}\,{\rm arcmin}^2]$", "kSZ CAP · DESI LRG",
         "SB35 (512)", 9, 1.0),
        (Bs, r"peak height bin $\nu$",
         r"$\langle Y(\nu)\rangle\ (4')\ /\ 10^{-4}$",
         r"$\kappa$-peaks · ACT $y$ × DES", "SB35 (253)", 4, 1e4),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(W_DOUBLE, 4.9),
                             gridspec_kw=dict(height_ratios=[1.35, 1.0]))
    out = []

    for j, (p, xlab, ylab, tag, sublab, dof, sc) in enumerate(panels):
        x = p["x"]
        ax = axes[0, j]
        spaghetti(ax, x, p["pred"] * sc, label=sublab)
        fiducial_line(ax, x, p["fid"] * sc,
                      label="nearest-TNG unit" if j == 2 else "BIND fiducial")
        data_points(ax, x, p["dat"] * sc, yerr=p["err"] * sc, label="data")
        ax.set_ylabel(ylab)
        condition_tag(ax, tag)
        if j == 2:
            ax.set_xticks(x)
            ax.set_xticklabels([r"$1$-$2$", r"$2$-$3$", r"$3$-$4$", r"$4+$"])
        if j == 0:
            ax.legend(loc="lower right", fontsize=6.0)

        # ---- pull row ----
        axp = axes[1, j]
        pull = (p["pred"] - p["dat"][None, :]) / p["sig"]
        for band_n, a in ((2, 0.12), (1, 0.28)):
            axp.axhspan(-band_n, band_n, color=COL["shade"], alpha=a, lw=0,
                        zorder=0)
        spaghetti(axp, x, pull)
        refline(axp, y=0.0)
        axp.set_xlabel(xlab)
        if j == 0:
            axp.set_ylabel(r"(model $-$ data)$/\sigma$")
        axp.set_ylim(-6, 26)
        if j == 2:
            axp.set_xticks(x)
            axp.set_xticklabels([r"$1$-$2$", r"$2$-$3$", r"$3$-$4$", r"$4+$"])

        s = _stats(p, tag, dof)
        out.append(s)
        axp.text(0.04, 0.93,
                 f"best {s['best_chi2']:.0f}/{dof} = "
                 f"{s['best_chi2_per_dof']:.1f} per dof\n"
                 f"0 of {s['n_units']} inside $2\\sigma$ everywhere",
                 transform=axp.transAxes, fontsize=6.0, va="top")

    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    fig.tight_layout(pad=0.4, w_pad=1.4)

    (WP6 / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(WP6 / "figures" / f"sobol_coverage_pub.{ext}")

    rec = dict(
        what="per-unit fit of the SB35 design to each frozen Paper-3 data "
             "vector; asks whether ANY single design member passes through "
             "the data, the question the Paper-2 envelope figure invites",
        emulator_measured_closure_median_frac_err=A["closure"],
        ksz_f_sat="held at midpoint 0.20 (<=12% amplitude knob; minimising "
                  "chi2 over it gives 34.9/9 vs 38.1/9)",
        b_side_recipe="measured <Y(nu)> at 4', B5 frozen diagonal-sigma recipe",
        probes=out)
    (WP6 / "sobol_coverage.json").write_text(json.dumps(rec, indent=2))

    PLANS_FIG.mkdir(parents=True, exist_ok=True)
    for f in ("figures/sobol_coverage_pub.pdf", "figures/sobol_coverage_pub.png",
              "sobol_coverage.json"):
        shutil.copy2(WP6 / f, PLANS_FIG / Path(f).name)

    for s in out:
        print(f"{s['probe']:32s} best {s['best_chi2']:7.1f}/{s['dof']} "
              f"({s['best_chi2_per_dof']:5.2f}/dof)  "
              f"pointwise {s['pointwise_bins_reached']}/{s['n_bins']}  "
              f"within2sig {s['n_units_all_bins_within_2sig']}/{s['n_units']}")
    print(f"\nwrote {WP6}/figures/sobol_coverage_pub.pdf (+ plans copy)")


if __name__ == "__main__":
    main()
