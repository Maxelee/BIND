"""fig10_bridge_compton_y.pdf — is the integrated Compton-Y a better single-quantity
predictor of the WL suppression S(ell) than the group gas fraction f_gas alone?

Panel (a): S(ell~1500) vs. log10 Y_500c^group across the 57-run 1P feedback suite, r=0.93.
Panel (b): Y_500c^group vs. group f_gas — the two are nearly degenerate at fixed mass/z.
Panel (c): raw correlations of f_gas, log Y, log T with S(ell), plus the partial
correlations r[Y,S|f_gas] and r[f_gas,S|Y] (linear, control variable regressed out) that
show Y subsumes essentially all the f_gas predictive power while retaining a residual
"heating leg" beyond it.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz
    (keys: M_fof, m_gas_500c_bg, m_tot_500c_bg, Y_500c, T_mw_500c)
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_stats.npz (per-run "{run}_clk")
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json
  - /mnt/home/mlee1/ceph/bind_science/runs/{bind,dmo}/run_0000/Cl_kappa.npz (fiducial S(ell))

Original source: examples/fgas_bridge_explore.ipynb / examples/_build_fgas_bridge_explore_nb.py
(worktree analysis/wl-tsz-bridge), section 3 "Is Compton-Y a better lever than f_gas?"
(builder lines ~409-484). Placeholder figs/fig10_bridge_compton_y.pdf is byte-identical
(md5) to examples/figures_lightcone/fig_bridge_compton_y.pdf.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

SCI = Path("/mnt/home/mlee1/ceph/bind_science")
CACHE = SCI / "dashboard_cache"
ATLAS = SCI / "halo_atlas"
RUNS = SCI / "runs"
ZIDX = 1
SNAP = 96


def _family(name: str) -> str:
    s = name.lower()
    if any(k in s for k in ("blackhole", "quasar", "radio", "agn")):
        return "AGN"
    if any(k in s for k in ("wind", "snii", "snia", "imf", "supernov", "sfr", "eqs")):
        return "SN / wind"
    return "other"


FAMC = {"AGN": COLORS["highlight"], "SN / wind": COLORS["bind"], "other": COLORS["dmo"]}


def group_q(run, key, mlo=1e13, mhi=2e13, snap=SNAP, reduce=np.nanmedian, nmin=5):
    """Median over halos with M_fof in [mlo,mhi); key='fgas' -> M_gas/M_tot (R500c, bg-sub)."""
    f = ATLAS / f"{run}_snap{snap:03d}.npz"
    if not f.exists():
        return np.nan
    d = np.load(f)
    s = (d["M_fof"] >= mlo) & (d["M_fof"] < mhi) & (d["m_tot_500c_bg"] > 0)
    if s.sum() < nmin:
        return np.nan
    q = (d["m_gas_500c_bg"] / d["m_tot_500c_bg"])[s] if key == "fgas" else d[key][s].astype(float)
    return reduce(q)


def pcorr(a, b, c):
    """Partial correlation of a,b controlling for c (linear)."""
    ra = a - np.polyval(np.polyfit(c, a, 1), c)
    rb = b - np.polyval(np.polyfit(c, b, 1), c)
    return np.corrcoef(ra, rb)[0, 1]


def main():
    setup()

    design = [
        dict(run=r, name=name)
        for r, _i, name, _val, _fid in json.load(open(CACHE / "g1_design.json"))
    ]
    for d in design:
        d["fam"] = _family(d["name"])
    GS = dict(np.load(CACHE / "g1_stats.npz", allow_pickle=True))
    ELL = GS["ell"]
    LI = int(np.argmin(np.abs(ELL - 1500)))

    CL_BIND = np.load(RUNS / "bind/run_0000/Cl_kappa.npz")["cl"]
    CL_DMO = np.load(RUNS / "dmo/run_0000/Cl_kappa.npz")["cl"]
    S_FID = CL_BIND[ZIDX, ZIDX] / CL_DMO[ZIDX, ZIDX]

    def S_of(run, l_idx=LI):
        return (1 + GS[f"{run}_clk"][l_idx]) * S_FID[l_idx]

    runs_ok = [d for d in design if f"{d['run']}_clk" in GS]

    fg, lY, lT, Sg, fmc = [], [], [], [], []
    for d in runs_ok:
        g = group_q(d["run"], "fgas")
        y = group_q(d["run"], "Y_500c")
        t = group_q(d["run"], "T_mw_500c")
        if np.isfinite(g) and np.isfinite(y) and np.isfinite(t):
            fg.append(g)
            lY.append(np.log10(y))
            lT.append(np.log10(t))
            Sg.append(S_of(d["run"]))
            fmc.append(FAMC[d["fam"]])
    fg, lY, lT, Sg = (np.asarray(x) for x in (fg, lY, lT, Sg))
    fmc = np.array(fmc)

    r_fg = np.corrcoef(fg, Sg)[0, 1]
    r_Y = np.corrcoef(lY, Sg)[0, 1]
    r_T = np.corrcoef(lT, Sg)[0, 1]
    pr_Y = pcorr(lY, Sg, fg)
    pr_fg = pcorr(fg, Sg, lY)
    aY, bY = np.polyfit(lY, Sg, 1)
    rfgY = np.corrcoef(fg, lY)[0, 1]

    fig, ax = plt.subplots(1, 3, figsize=TWO_COL)

    # (a) the Compton-Y bridge
    ax[0].scatter(lY, Sg, s=22, c=fmc, edgecolor="k", lw=0.3, zorder=3)
    xg = np.linspace(lY.min(), lY.max(), 20)
    ax[0].plot(xg, aY * xg + bY, color=COLORS["truth"], lw=1.5, label=f"$r={r_Y:.2f}$")
    for fam in FAMC:
        ax[0].scatter([], [], s=22, color=FAMC[fam], edgecolor="k", lw=0.3, label=fam)
    ax[0].set_xlabel(r"$\log_{10} Y_{500c}^{\rm group}$")
    ax[0].set_ylabel(r"$S(\ell\!\sim\!1500)$")
    ax[0].legend(fontsize=5.8, loc="lower right")
    panel_label(ax[0], "(a)")

    # (b) f_gas-Y degeneracy at group scale
    ax[1].scatter(fg, lY, s=22, c=fmc, edgecolor="k", lw=0.3, zorder=3)
    ax[1].set_xlabel(r"group $f_{\rm gas}$")
    ax[1].set_ylabel(r"$\log_{10} Y_{500c}^{\rm group}$")
    ax[1].text(0.95, 0.05, f"$r={rfgY:.2f}$", transform=ax[1].transAxes, ha="right",
               va="bottom", fontsize=7)
    panel_label(ax[1], "(b)")

    # (c) raw vs. partial correlation bars
    labels = [r"$f_{\rm gas}$", r"$\log Y$", r"$\log T$",
              r"$Y\,|\,f_{\rm gas}$", r"$f_{\rm gas}\,|\,Y$"]
    vals = [r_fg, r_Y, r_T, pr_Y, pr_fg]
    barc = [COLORS["bind"], COLORS["highlight"], "#e08214", COLORS["highlight"], COLORS["bind"]]
    hatch = ["", "", "", "//", "//"]
    xb = np.arange(len(vals))
    for i, (v, c, h) in enumerate(zip(vals, barc, hatch)):
        ax[2].bar(i, v, color=c, edgecolor="k", lw=0.5, hatch=h, alpha=0.9)
        ax[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xticks(xb)
    ax[2].set_xticklabels(labels, fontsize=6.6, rotation=20)
    ax[2].set_ylabel(r"corr with $S(\ell)$")
    ax[2].set_ylim(-0.15, 1.0)
    panel_label(ax[2], "(c)", loc="upper right")

    fig.tight_layout(w_pad=1.2)
    save(fig, "figs/fig10_bridge_compton_y")

    print(f"r(f_gas, S)        = {r_fg:+.3f}")
    print(f"r(logY,  S)        = {r_Y:+.3f}")
    print(f"r(logT,  S)        = {r_T:+.3f}")
    print(f"r(f_gas, logY)     = {rfgY:+.3f}")
    print(f"partial r(Y, S | f_gas) = {pr_Y:+.3f}")
    print(f"partial r(f_gas, S | Y) = {pr_fg:+.3f}")


if __name__ == "__main__":
    main()
