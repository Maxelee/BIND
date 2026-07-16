#!/usr/bin/env python
"""fig08_vandaalen_analog -- reproducing van Daalen+2020 Fig.16 for weak lensing.

Panel (a): where the 57-run 1P TNG-CAMELS feedback suite sits on the
universal van Daalen+2020 curve, Delta-P/P_DMO(k=0.5 h/Mpc) as a function of
the renormalized baryon (gas+star) fraction f_bar-tilde,500c = f_bar /
(Omega_b/Omega_m) in M500c in [6e13, 2e14] Msun/h -- a rug of the 57 runs,
coloured by feedback family, plus the observationally-preferred group band.
Panel (b): the WL analogue -- Delta-C_ell/C_ell^DMO vs. the same
f_bar-tilde,500c, at ell=1000,3000,6000 (increasing depth into the
suppression curve).

van Daalen+2020 fit (hardcoded, as in the source notebook):
  Delta-P/P = -exp(-5.990 * f_bar_tilde - 0.5107)

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_stats.npz (ell, per-run "{run}_clk")
  - /mnt/home/mlee1/ceph/bind_science/runs/{bind,dmo}/run_0000/Cl_kappa.npz (fiducial absolute C_ell)
  - /mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz
      (keys: m_tot_500c_bg, m_gas_500c_bg, m_star_500c)
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json

Source: examples/_build_fgas_bridge_explore_nb.py, section 5 "Reproducing
van Daalen+2020 for weak lensing" (lines 651-724; builds
examples/fgas_bridge_explore.ipynb, worktree analysis/wl-tsz-bridge).
Placeholder figs/fig08_vandaalen_analog.pdf is byte-identical (md5) to the
on-disk examples/figures_lightcone/fig_bridge_vandaalen.pdf.
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
ZIDX = 1  # z_s = 1.0 reference plane
SNAP = 96
F_B = 0.0490 / 0.3089  # cosmic baryon fraction Omega_b/Omega_m

# van Daalen+2020 Fig.16 fit (3D, k=0.5 h/Mpc)
vd = lambda ft: -np.exp(-5.990 * ft - 0.5107)  # noqa: E731


def _family(name):
    """Feedback family (matches _build_fgas_bridge_explore_nb.py's own classifier,
    used for §5 of that notebook -- distinct from bind_bridge.py's name+desc version
    used for fig06/fig07; both exist in the source tree, see fig13's provenance note)."""
    s = name.lower()
    if any(k in s for k in ("blackhole", "quasar", "radio", "agn")):
        return "AGN"
    if any(k in s for k in ("wind", "snii", "snia", "imf", "supernov", "sfr", "eqs")):
        return "SN / wind"
    return "other"


FAMC = {"AGN": COLORS["highlight"], "SN / wind": COLORS["bind"], "other": COLORS["dmo"]}


def fbar_tilde(run, mlo=6e13, mhi=2e14, snap=SNAP, nmin=5):
    """Renormalized baryon (gas+star) fraction in M500c bin, van Daalen definition."""
    f = ATLAS / f"{run}_snap{snap:03d}.npz"
    if not f.exists():
        return np.nan
    d = np.load(f)
    M5 = d["m_tot_500c_bg"]
    s = (M5 >= mlo) & (M5 < mhi) & (M5 > 0)
    if s.sum() < nmin:
        return np.nan
    return np.nanmedian(((d["m_gas_500c_bg"] + d["m_star_500c"]) / M5)[s]) / F_B


def main():
    setup()

    DESIGN = []
    for run, _i, name, val, fid in json.load(open(CACHE / "g1_design.json")):
        DESIGN.append(dict(run=run, name=name, fam=_family(name)))

    GS = dict(np.load(CACHE / "g1_stats.npz", allow_pickle=True))
    ELL = GS["ell"]
    CL_BIND = np.load(RUNS / "bind/run_0000/Cl_kappa.npz")["cl"]
    CL_DMO = np.load(RUNS / "dmo/run_0000/Cl_kappa.npz")["cl"]
    S_FID = CL_BIND[ZIDX, ZIDX] / CL_DMO[ZIDX, ZIDX]

    RUNS_OK = [d for d in DESIGN if f"{d['run']}_clk" in GS]
    print(f"{len(RUNS_OK)} of {len(DESIGN)} 1P runs carry WL statistics")

    ft = np.array([fbar_tilde(d["run"]) for d in RUNS_OK])
    fcol = np.array([FAMC[d["fam"]] for d in RUNS_OK])
    ok = np.isfinite(ft)
    ft, fcol = ft[ok], fcol[ok]
    DCm = np.array([(1 + GS[f"{d['run']}_clk"]) * S_FID
                    for d, o in zip(RUNS_OK, ok) if o]) - 1

    fig, ax = plt.subplots(1, 2, figsize=TWO_COL)

    # (a) the universal plane: where TNG-CAMELS lives on the van Daalen curve
    fgrid = np.linspace(0.2, 1.05, 200)
    ax[0].plot(fgrid, vd(fgrid), color=COLORS["truth"], ls="--", lw=1.4,
              label="van Daalen+20")
    ax[0].fill_between(fgrid, vd(fgrid) - 0.01, vd(fgrid) + 0.01, color=COLORS["dmo"],
                       alpha=0.3, lw=0)
    ax[0].axhline(0, ls=":", color="0.5", lw=0.6)
    ax[0].axvspan(0.55, 0.76, color=COLORS["secondary"], alpha=0.15, lw=0,
                 label="obs. group")
    ax[0].axvspan(ft.min(), ft.max(), color=COLORS["bind"], alpha=0.12, lw=0,
                 label="TNG-CAMELS")
    for x, c in zip(ft, fcol):
        ax[0].plot([x, x], [0.004, 0.018], color=c, lw=0.7, alpha=0.85)
    ax[0].set_xlim(0.2, 1.05)
    ax[0].set_ylim(-0.13, 0.035)
    ax[0].set_xlabel(r"$\tilde f_{\rm bar,500c}=f_{\rm bar}/(\Omega_b/\Omega_m)$")
    ax[0].set_ylabel(r"$\Delta P/P_{\rm DMO}$")
    ax[0].legend(fontsize=6.0, loc="lower right")
    ax[0].annotate("flat top\n(saturated)", xy=(0.9, -0.002), xytext=(0.55, -0.06),
                   fontsize=6, ha="center", arrowprops=dict(arrowstyle="->", lw=0.6, color="0.4"))
    panel_label(ax[0], "(a)")

    # (b) the WL analog: Delta-C_ell/C_ell vs f_bar-tilde at several ell
    cmap = plt.get_cmap("cividis")
    for lt, c in [(1000, cmap(0.15)), (3000, cmap(0.55)), (6000, cmap(0.92))]:
        li = int(np.argmin(np.abs(ELL - lt)))
        y = DCm[:, li]
        a, b = np.polyfit(ft, y, 1)
        r = np.corrcoef(ft, y)[0, 1]
        xg = np.linspace(ft.min(), ft.max(), 12)
        ax[1].scatter(ft, y, s=10, color=c, edgecolor="k", lw=0.2, alpha=0.9)
        ax[1].plot(xg, a * xg + b, color=c, lw=1.4, label=fr"$\ell={lt}$ ($r={r:.2f}$)")
    ax[1].axhline(0, ls=":", color="0.5", lw=0.6)
    txt_bg = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0)
    ax[1].text(0.80, 0.012, "enhancement", fontsize=7, color="0.3", bbox=txt_bg)
    ax[1].text(0.80, -0.085, "suppression", fontsize=7, color="0.3", bbox=txt_bg)
    ax[1].set_xlim(ft.min() - 0.01, ft.max() + 0.01)
    ax[1].set_xlabel(r"$\tilde f_{\rm bar,500c}$")
    ax[1].set_ylabel(r"$\Delta C_\ell/C_\ell^{\rm DMO}$ (WL)")
    ax[1].legend(fontsize=6.2, loc="lower right")
    panel_label(ax[1], "(b)")

    fig.tight_layout(w_pad=1.6)
    save(fig, "figs/fig08_vandaalen_analog")
    print(f"TNG-CAMELS f_bar~ range: {ft.min():.3f}-{ft.max():.3f} (median {np.median(ft):.3f}); "
          f"van Daalen DP/P there = {vd(np.median(ft)):+.4f}")


if __name__ == "__main__":
    main()
