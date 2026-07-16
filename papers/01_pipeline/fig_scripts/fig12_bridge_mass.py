"""fig12_bridge_mass.pdf — how the f_gas <-> S(ell) bridge changes from group to cluster
scale, fit separately in three M200c bins.

Panel (a): S(ell~1500) vs. group f_gas, one linear fit per mass bin (group / intermediate
/ cluster), across the 57-run 1P feedback suite. Panel (b): Pearson r and best-fit slope
alpha = dS/df_gas vs. mass-bin center. Panel (c): r[f_gas, S(ell)] vs. ell, per mass bin.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_stats.npz (per-run "{run}_clk")
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json
  - /mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz
    (keys: M_fof, m_gas_500c_bg, m_tot_500c_bg)
  - /mnt/home/mlee1/ceph/bind_science/runs/{bind,dmo}/run_0000/Cl_kappa.npz (fiducial S(ell))

Original source: examples/fgas_bridge_explore.ipynb / examples/_build_fgas_bridge_explore_nb.py
(worktree analysis/wl-tsz-bridge), section 2 "How the bridge changes from groups to
clusters" (builder lines ~209-271). Placeholder figs/fig12_bridge_mass.pdf is byte-identical
(md5) to examples/figures_lightcone/fig_bridge_mass.pdf.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import TWO_COL, panel_label, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

SCI = Path("/mnt/home/mlee1/ceph/bind_science")
CACHE = SCI / "dashboard_cache"
ATLAS = SCI / "halo_atlas"
RUNS = SCI / "runs"
ZIDX = 1
SNAP = 96


def group_q(run, key, mlo, mhi, snap=SNAP, reduce=np.nanmedian, nmin=5):
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


def main():
    setup()

    design = [dict(run=r) for r, _i, _name, _val, _fid in json.load(open(CACHE / "g1_design.json"))]
    GS = dict(np.load(CACHE / "g1_stats.npz", allow_pickle=True))
    ELL = GS["ell"]
    LI = int(np.argmin(np.abs(ELL - 1500)))

    CL_BIND = np.load(RUNS / "bind/run_0000/Cl_kappa.npz")["cl"]
    CL_DMO = np.load(RUNS / "dmo/run_0000/Cl_kappa.npz")["cl"]
    S_FID = CL_BIND[ZIDX, ZIDX] / CL_DMO[ZIDX, ZIDX]

    runs_ok = [d for d in design if f"{d['run']}_clk" in GS]

    mass_bins = [(1e13, 3e13, "group"), (3e13, 1e14, "intermediate"), (1e14, 3e14, "cluster")]
    cols = ["#1b7837", "#762a83", "#b35806"]

    fig, ax = plt.subplots(1, 3, figsize=TWO_COL)
    summ = []
    for (mlo, mhi, lab), c in zip(mass_bins, cols):
        fg, Svec = [], []
        for d in runs_ok:
            g = group_q(d["run"], "fgas", mlo, mhi)
            if np.isfinite(g):
                fg.append(g)
                Svec.append((1 + GS[f"{d['run']}_clk"]) * S_FID)
        fg = np.asarray(fg)
        Svec = np.asarray(Svec)  # (n_runs, n_ell)
        Sat = Svec[:, LI]
        a, b = np.polyfit(fg, Sat, 1)
        r = np.corrcoef(fg, Sat)[0, 1]
        summ.append((lab, mlo, mhi, len(fg), r, a))

        # (a) scatter + fit at l~1500
        ax[0].scatter(fg, Sat, s=14, color=c, edgecolor="k", lw=0.2, alpha=0.85)
        xg = np.linspace(fg.min(), fg.max(), 20)
        ax[0].plot(xg, a * xg + b, color=c, lw=1.6, label=fr"{lab}: $r={r:.2f}$, $\alpha={a:.2f}$")

        # (c) r(f_gas, S) vs l
        fz = fg - fg.mean()
        rl = (fz @ (Svec - Svec.mean(0))) / (
            np.sqrt((fz**2).sum()) * np.sqrt(((Svec - Svec.mean(0)) ** 2).sum(0)) + 1e-30
        )
        ax[2].plot(ELL, rl, color=c, lw=1.5, label=lab)

    ax[0].set_xlabel(r"group $f_{\rm gas}$")
    ax[0].set_ylabel(r"$S(\ell\!\sim\!1500)$")
    ax[0].legend(fontsize=5.8, loc="lower right")
    panel_label(ax[0], "(a)")

    # (b) slope & r vs mass
    mc = [np.sqrt(s[1] * s[2]) for s in summ]
    hr = ax[1].plot(mc, [s[4] for s in summ], "o-", color="k", label=r"Pearson $r$")
    ax[1].set_ylabel(r"Pearson $r$")
    ax[1].set_ylim(0.25, 1.0)  # headroom below the lowest point clears a legend band
    axb = ax[1].twinx()
    ha = axb.plot(mc, [s[5] for s in summ], "s--", color="0.55", label=r"slope $\alpha$")
    axb.set_ylabel(r"slope $\alpha = \partial S/\partial f_{\rm gas}$", color="0.45")
    axb.tick_params(axis="y", colors="0.45")
    axb.set_ylim(0.4, 1.3)  # headroom below the lowest point, matching ax[1] above
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$M_{200c}$ bin center $[M_\odot/h]$")
    # single combined legend, centered low where neither series has a marker
    ax[1].legend(hr + ha, [h.get_label() for h in hr + ha], fontsize=6.0,
                 loc="lower center", ncol=2)
    panel_label(ax[1], "(b)")

    ax[2].axvspan(1000, 2000, color="0.6", alpha=0.12, lw=0)
    ax[2].axhline(0, color="0.7", lw=0.6)
    ax[2].set_xscale("log")
    ax[2].set_xlim(150, 1e4)
    ax[2].set_ylim(0, 1)
    ax[2].set_xlabel(r"$\ell$")
    ax[2].set_ylabel(r"$r[\,f_{\rm gas},\,S(\ell)\,]$")
    ax[2].legend(fontsize=6.0, loc="lower left")
    panel_label(ax[2], "(c)")

    fig.tight_layout(w_pad=1.2)
    save(fig, "figs/fig12_bridge_mass")

    print(f"{'bin':12s} {'n':>3s} {'r':>7s} {'slope':>8s}")
    for lab, mlo, mhi, n, r, a in summ:
        print(f"{lab:12s} {n:3d} {r:7.3f} {a:8.3f}")


if __name__ == "__main__":
    main()
