"""fig09_cl_envelope.pdf — the f_gas <-> S(ell) bridge, 1P corners vs. the full Sobol cloud.

Left panel: S(ell) = C_ell^bind / C_ell^DMO envelopes for the fiducial run, the 57-run
1P (one-parameter-at-a-time) feedback design, and the 253-run Sobol quasi-random design
(5-95% envelope), with LSST-Y10 / Euclid Gaussian shape-noise error bands on the fiducial.
Right panel: Pearson r[f_gas^group, S(ell)] vs. ell for both designs, with an inset
showing the raw Sobol scatter at the peak-correlation multipole (ell~1037).

Data (cached, no engine re-run):
  - CACHE = /mnt/home/mlee1/ceph/bind_science/dashboard_cache/{g1_design.json,g1_stats.npz}
  - RUNS  = /mnt/home/mlee1/ceph/bind_science/runs/{bind,dmo}/run_0000/Cl_kappa.npz
  - ATLAS = /mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz   (1P halo atlas)
  - Sobol cache: /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz (t__suppression__{value,valid},
    run_ids, param_names) + /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/halo_atlas/run_%04d_snap096.npz

Original source: examples/paper_lightcone_figs2.ipynb (worktree analysis/sobol-sb35) cell 12,
"# section 4.1 fig_cl -- now spanning all three datasets: the fiducial, the 1P (twobound)
spokes, and the prior-filling Sobol sweep." (NOT examples/_build_paper_nb.py, which lacks
the Sobol dataset.) Placeholder figs/fig09_cl_envelope.pdf is byte-identical (md5) to
examples/figures_lightcone/fig_cl.pdf, confirmed against this cell's source.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
CACHE = SCI / "dashboard_cache"
ATLAS = SCI / "halo_atlas"
RUNS = SCI / "runs"
SB35 = CEPH / "bind_sb35"
SOBOL_ATLAS = SB35 / "analysis_cache/halo_atlas"
ZIDX = 1  # z_s = 1.0 reference plane


def _family(name: str) -> str:
    s = name.lower()
    if any(k in s for k in ("blackhole", "quasar", "radio", "agn")):
        return "AGN"
    if any(k in s for k in ("wind", "snii", "snia", "imf", "supernov", "sfr", "eqs")):
        return "SN / wind"
    return "other"


FAMC = {"AGN": COLORS["highlight"], "SN / wind": COLORS["bind"], "other": COLORS["dmo"]}


def fgas_group(atlas_dir, run, mlo=1e13, mhi=3e13, snap=96):
    """Median background-subtracted f_gas (R500c) in a group M200c bin."""
    f = atlas_dir / f"{run}_snap{snap:03d}.npz"
    if not f.exists():
        return np.nan
    d = np.load(f)
    if "m_gas_500c_bg" not in d.files:
        return np.nan
    s = (d["M_fof"] >= mlo) & (d["M_fof"] < mhi) & (d["m_tot_500c_bg"] > 0)
    return np.nanmedian((d["m_gas_500c_bg"] / d["m_tot_500c_bg"])[s]) if s.sum() >= 5 else np.nan


def main():
    setup()

    # ---- 1P design + field statistics ----
    design = [
        dict(run=r, name=name)
        for r, _i, name, _val, _fid in json.load(open(CACHE / "g1_design.json"))
    ]
    for d in design:
        d["fam"] = _family(d["name"])
    GS = dict(np.load(CACHE / "g1_stats.npz", allow_pickle=True))
    ELL = GS["ell"]
    runs_ok = [d for d in design if f"{d['run']}_clk" in GS]

    CL_BIND = np.load(RUNS / "bind/run_0000/Cl_kappa.npz")["cl"]
    CL_DMO = np.load(RUNS / "dmo/run_0000/Cl_kappa.npz")["cl"]
    S_FID = CL_BIND[ZIDX, ZIDX] / CL_DMO[ZIDX, ZIDX]

    # ---- Sobol design (253 runs): S(ell) at z_s=1 + group f_gas from its own atlas ----
    SOBOL = np.load(SB35 / "emulator_dataset.npz", allow_pickle=True)
    sv = np.asarray(SOBOL["t__suppression__valid"], bool)
    sob_rid = np.asarray(SOBOL["run_ids"])[sv]
    S_SOBOL = np.asarray(SOBOL["t__suppression__value"])[sv][:, ZIDX, :]  # (n,724)
    fgas_sob = np.array([fgas_group(SOBOL_ATLAS, f"run_{int(r):04d}") for r in sob_rid])
    mS = np.isfinite(fgas_sob)

    # ---- survey Gaussian relative error on C_l^kk (Knox formula, z_s=1) ----
    A2SR = (180 * 60 / np.pi) ** 2

    def cl_relerr(cl, ngal, sige, fsky, dlnl=0.15):
        Nl = sige**2 / (ngal * A2SR)
        dl = np.maximum(ELL * dlnl, 1.0)
        return np.sqrt(2.0 / ((2 * ELL + 1) * dl * fsky)) * (1 + Nl / cl)

    cl_sig = CL_BIND[ZIDX, ZIDX]
    err_lsst = cl_relerr(cl_sig, 27.0, 0.26, 0.44)
    err_euc = cl_relerr(cl_sig, 30.0, 0.30, 0.36)

    def rcurve(fg, S):
        fz = fg - fg.mean()
        return (fz @ (S - S.mean(0))) / (
            np.sqrt((fz**2).sum()) * np.sqrt(((S - S.mean(0)) ** 2).sum(0)) + 1e-30
        )

    slo, shi = np.nanpercentile(S_SOBOL, [5, 95], axis=0)

    fig, ax = plt.subplots(1, 2, figsize=TWO_COL)

    # --- (a) S(ell) envelope: Sobol 5-95%, 1P family spokes, fiducial, survey bands ---
    ax[0].fill_between(ELL, slo, shi, color=COLORS["dmo"], alpha=0.30, lw=0, zorder=0,
                        label="Sobol 5-95%")
    S_runs, fg_all = [], []
    for d in runs_ok:
        S = (1 + GS[f"{d['run']}_clk"]) * S_FID
        fg = fgas_group(ATLAS, d["run"])
        ax[0].plot(ELL, S, color=FAMC[d["fam"]], lw=0.5, alpha=0.35)
        if np.isfinite(fg):
            S_runs.append(S)
            fg_all.append(fg)
    ax[0].plot(ELL, S_FID, color=COLORS["truth"], lw=2.0, zorder=6, label="fiducial")
    ax[0].fill_between(ELL, S_FID * (1 - err_lsst), S_FID * (1 + err_lsst),
                        color="0.5", alpha=0.25, lw=0, label=r"LSST-Y10 $\sigma(C_\ell)$")
    ax[0].plot(ELL, S_FID * (1 + err_euc), color="0.4", ls=":", lw=0.8)
    ax[0].plot(ELL, S_FID * (1 - err_euc), color="0.4", ls=":", lw=0.8,
               label=r"Euclid $\sigma(C_\ell)$")
    ax[0].axhline(1, color="0.7", lw=0.7)
    ax[0].axvspan(1000, 2000, color="0.6", alpha=0.10, lw=0)
    for fam in FAMC:
        ax[0].plot([], [], color=FAMC[fam], lw=1.2, label=fam)
    ax[0].set_xscale("log")
    ax[0].set_xlim(80, 1e4)
    ax[0].set_ylim(0.80, 1.06)
    ax[0].set_xlabel(r"$\ell$")
    ax[0].set_ylabel(r"$S(\ell)=C_\ell^{\rm bind}/C_\ell^{\rm DMO}$")
    ax[0].legend(loc="lower left", fontsize=5.2, ncol=2)
    panel_label(ax[0], "(a)")

    # --- (b) r[f_gas, S(ell)] vs ell, 1P (n=57) vs Sobol (n=253), inset scatter ---
    S_runs = np.array(S_runs)
    fg_all = np.array(fg_all)
    rl_1p = rcurve(fg_all, S_runs)
    Ss, fgs = S_SOBOL[mS], fgas_sob[mS]
    rl_sob = rcurve(fgs, Ss)
    pk = (ELL >= 1000) & (ELL <= 2000)
    lp = np.where(pk)[0][np.argmax(rl_sob[pk])]

    ax[1].plot(ELL, rl_1p, color=COLORS["dmo"], lw=1.0, ls="--", label=f"1P (n={len(fg_all)})")
    ax[1].plot(ELL, rl_sob, color=COLORS["truth"], lw=1.8, label=f"Sobol (n={mS.sum()})")
    ax[1].axvspan(1000, 2000, color="0.6", alpha=0.12, lw=0)
    ax[1].plot(ELL[lp], rl_sob[lp], "o", color=COLORS["highlight"], ms=6, zorder=6)
    ax[1].set_xscale("log")
    ax[1].set_xlim(150, 1e4)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel(r"$\ell$")
    ax[1].set_ylabel(r"Pearson $r\,[\,f_{\rm gas}^{\rm group},\,S(\ell)\,]$")
    ax[1].legend(loc="upper left", fontsize=6.0)
    panel_label(ax[1], "(b)", loc="upper right")

    axin = ax[1].inset_axes([0.56, 0.13, 0.40, 0.40])
    Sp = Ss[:, lp]
    axin.scatter(fgs, Sp, s=6, color=COLORS["secondary"], alpha=0.45, lw=0, label="Sobol")
    axin.scatter(fg_all, S_runs[:, lp], s=16, color=COLORS["highlight"], alpha=0.75, lw=0,
                 marker="^", label="1P")
    zf = np.polyfit(fgs, Sp, 1)
    xg = np.linspace(fgs.min(), fgs.max(), 20)
    axin.plot(xg, np.polyval(zf, xg), color=COLORS["truth"], ls="--", lw=1.0)
    axin.legend(fontsize=6.5, loc="upper left", handletextpad=0.2, borderpad=0.15,
                labelspacing=0.2)
    axin.tick_params(labelsize=6.5)
    axin.set_xlabel(r"$f_{\rm gas}^{\rm group}$", fontsize=6.5, labelpad=1)
    axin.set_ylabel(r"$S(\ell)$", fontsize=6.5, labelpad=1)
    con = ConnectionPatch(xyA=(ELL[lp], rl_sob[lp]), coordsA=ax[1].transData,
                           xyB=(0.5, 1.0), coordsB=axin.transAxes, color="0.5", lw=0.7, ls="-")
    fig.add_artist(con)

    fig.tight_layout(w_pad=1.0)
    save(fig, "figs/fig09_cl_envelope")

    band = (ELL >= 1000) & (ELL <= 2000)
    print(f"f_gas-S(l): 1P peak r={np.nanmax(rl_1p[band]):.3f}  |  "
          f"Sobol peak r={rl_sob[lp]:.3f} at l={int(ELL[lp])} (n={mS.sum()})")


if __name__ == "__main__":
    main()
