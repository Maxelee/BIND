#!/usr/bin/env python
"""fig02_validation_field -- field-level WL closure, BIND vs TNG300 hydro
truth, fiducial lightcone (Sec. 2.4).

Six columns, top = statistic (BIND solid vs hydro-truth dashed), bottom =
relative residual: C_ell^kappakappa, N_peak(nu), N_min(nu), and Minkowski
functionals V0/V1/V2(nu). Residuals are a percent ratio (B/H-1) for the
ratio-type statistics and (B-H) normalized by the peak |H| amplitude for the
Minkowski functionals (which cross zero, so a ratio residual is undefined
near the zero-crossing).

Data (cached, no re-derivation), z_s = 1.0 (index 1 of the 5 lux source
planes [0.5, 1.0, 1.5, 2.0, 2.44]):
  - /mnt/home/mlee1/ceph/bind_science/runs/bind/run_0000/{Cl_kappa.npz,
    peak_counts.npz, nongaussian_stats.npz}
  - /mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000/{Cl_kappa.npz,
    peak_counts.npz, nongaussian_stats.npz}

Source: examples/paper_lightcone_figs.ipynb (worktree wl-tsz-bridge), cell 6
(heading "Sec. 2.4 -- Validation against TNG300 hydro truth").
Placeholder this replaces: figs/fig02_validation_field.png (md5-identical to
figs_raw/paper_lightcone_figs/cell006_out1.png).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

RUNS = Path("/mnt/home/mlee1/ceph/bind_science/runs")
ZIDX = 1  # z_s = 1.0


def main() -> None:
    setup()
    fb, tr = RUNS / "bind/run_0000", RUNS / "truth/run_0000"
    clb, clt = np.load(fb / "Cl_kappa.npz"), np.load(tr / "Cl_kappa.npz")
    pkb, pkt = np.load(fb / "peak_counts.npz"), np.load(tr / "peak_counts.npz")
    ngb, ngt = np.load(fb / "nongaussian_stats.npz"), np.load(tr / "nongaussian_stats.npz")
    nu = pkb["nu"]
    mfn = ngb["mf_nu"]
    cB, cH = COLORS["bind"], COLORS["truth"]

    # (x, BIND, hydro, xlabel, ylabel, xscale, yscale, xlim, residual-mode, panel tag)
    cols = [
        (clb["ell"], clb["cl"][ZIDX, ZIDX], clt["cl"][ZIDX, ZIDX],
         r"$\ell$", r"$C_\ell^{\kappa\kappa}$", "log", "log", (90, 1e4), "ratio", "(a)_ur"),
        (nu, pkb["peak_counts"][ZIDX], pkt["peak_counts"][ZIDX],
         r"$\nu$", r"$N_{\rm peak}$", "linear", "log", (-1, 6), "ratio", "(b)_ur"),
        (nu, pkb["minima_counts"][ZIDX], pkt["minima_counts"][ZIDX],
         r"$\nu$", r"$N_{\rm min}$", "linear", "log", (-6, 1), "ratio", "(c)"),
        (mfn, ngb["V0"][ZIDX], ngt["V0"][ZIDX],
         r"$\nu$", r"$V_0$", "linear", "linear", (-3, 3), "norm", "(d)_ur"),
        (mfn, ngb["V1"][ZIDX], ngt["V1"][ZIDX],
         r"$\nu$", r"$V_1$", "linear", "linear", (-3, 3), "norm", "(e)"),
        (mfn, ngb["V2"][ZIDX], ngt["V2"][ZIDX],
         r"$\nu$", r"$V_2$", "linear", "linear", (-3, 3), "norm", "(f)"),
    ]

    fig, ax = plt.subplots(2, 6, figsize=(TWO_COL[0], 3.5),
                            gridspec_kw=dict(height_ratios=[2.4, 1]))
    for j, (x, B, H, xl, yl, xs, ys, xlim, mode, tag) in enumerate(cols):
        a, ar = ax[0, j], ax[1, j]
        a.plot(x, B, color=cB, lw=1.4)
        a.plot(x, H, color=cH, ls="--", lw=1.2)
        a.set_xscale(xs)
        a.set_yscale(ys)
        a.set_xlim(*xlim)
        a.set_ylabel(yl)
        a.set_xticklabels([])
        if tag.endswith("_ur"):  # dodge the log-axis top tick label ("$10^{-8}$")
            panel_label(a, tag[:-3], loc="upper right")
        else:
            panel_label(a, tag)
        with np.errstate(divide="ignore", invalid="ignore"):
            if mode == "ratio":
                res = 100 * (B / H - 1)
            else:  # MFs cross zero -> normalize by peak amplitude
                res = 100 * (B - H) / np.nanmax(np.abs(H))
        ar.plot(x, res, color=cB, lw=1.1)
        ar.axhline(0, color=cH, lw=0.7)
        ar.axhspan(-5, 5, color="0.7", alpha=0.25, lw=0)
        ar.set_xscale(xs)
        ar.set_xlim(*xlim)
        ar.set_ylim(-25, 25)
        ar.set_xlabel(xl)
        ar.set_ylabel("resid. [\\%]" if j == 0 else "")

    ax[0, 0].plot([], [], color=cB, lw=1.4, label="BIND")
    ax[0, 0].plot([], [], color=cH, ls="--", lw=1.2, label="TNG300 hydro")
    ax[0, 0].legend(loc="lower left", fontsize=6)

    fig.tight_layout(w_pad=0.5, h_pad=0.3)
    save(fig, "figs/fig02_validation_field")


if __name__ == "__main__":
    main()
