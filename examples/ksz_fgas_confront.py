"""P4 / kSZ confrontation, Step 1 (velocity-free leg): BIND f_gas(M500, z) for
DESI-tracer-like halo samples vs eROSITA X-ray gas fractions, across the full
256-node SB35 Sobol grid.

This is the density-weighted (kSZ-tau / f_gas) leg of the (kappa, tau, y)
decomposition -- see docs/wl_tsz_plan.md and docs/ksz_desi_act_plan.md. It needs
*no gas velocity*: the kSZ surveys report the tau (electron-column) profile after
velocity reconstruction, which integrates to f_gas. Here we confront the
*integrated* f_gas; per-radius tau/y GNFW profiles are the next deliverable.

Data: bind_sb35/analysis_cache/integrated.parquet -- the per-halo integrated
catalogue (8.6M halo-instances x 256 runs x 20 snaps), columns f_gas_500,
M_tot_500, Y_500, T_mw_500, M_star_500, z, + the 30 astro params per row.

DESI tracer windows (host-halo mass x redshift, matched to Hadzhiyska+26 /
Ried Guachalla+25 DESI x ACT samples; primary axis is M500 because that is what
both the kSZ stacks and eROSITA bin on):

    BGS  z~0.1-0.4   logM500 ~ 13.0-13.8   (low-z, gas-poor / strong AGN)
    LRG  z~0.4-0.9   logM500 ~ 13.1-13.7
    ELG  z~0.8-1.6   logM500 ~ 12.7-13.2   (gas-rich; MASS-FLOOR LIMITED, see note)

NOTE: the TNG300 lightcone halo floor is M200 >= 1e13, so the ELG-host regime
(~10^12.5) sits at the very edge of the sample -- BGS/LRG are the statistically
robust legs. ELG points are shown but flagged.

    python examples/ksz_fgas_confront.py            # both figures + npz
    python examples/ksz_fgas_confront.py --tracer bgs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

CEPH = Path(os.environ.get("CEPH", "/mnt/home/mlee1/ceph"))
PARQUET = Path(os.environ.get("INTEGRATED_PARQUET",
                              CEPH / "bind_sb35/analysis_cache/integrated.parquet"))
OUT = Path(os.environ.get("OUTPUT_ROOT", CEPH / "bind_science")) / "ksz_confront"

F_B_COSMIC = 0.0490 / 0.3089          # Omega_b/Omega_m (TNG/Planck) ~ 0.1586

# ── observational X-ray gas-fraction edges (from examples/halo_atlas.py) ───────
def popesso_fgas(M_Msun):
    """Popesso+24 eROSITA hot-gas fraction, f_gas,500 = LOW / strong-feedback edge."""
    return 2.23e-7 * M_Msun ** 0.39


def eckert_fgas(M_Msun):
    """Eckert+19 mainstream X-ray f_gas,500 = HIGH edge (0.131 @ 2e14, slope 0.21)."""
    return 0.131 * (M_Msun / 2e14) ** 0.21


# ── DESI tracer windows: (z_lo, z_hi, logM500_lo, logM500_hi) ──────────────────
TRACERS = {
    "bgs": (0.08, 0.45, 13.0, 13.8),
    "lrg": (0.45, 0.90, 13.1, 13.7),
    "elg": (0.80, 1.60, 12.7, 13.2),
}
# AGN / SN feedback proxies among the 30 Sobol params (CAMELS-TNG naming).
A_AGN = "BlackHoleFeedbackFactor"
A_SN = "WindEnergyIn1e51erg"


def _load():
    cols = ["run", "snap", "z", "M_tot_500", "f_gas_500", "M_star_500", A_AGN, A_SN]
    df = pd.read_parquet(PARQUET, columns=cols)
    df = df[(df.M_tot_500 > 0) & (df.f_gas_500 > 0) & np.isfinite(df.f_gas_500)]
    df["logM500"] = np.log10(df.M_tot_500)
    return df


def _per_run_relation(df, zwin, mbins):
    """Median f_gas_500 in M500 bins, per Sobol run -> (centers, stack[n_run, n_bin])."""
    lo, hi = zwin
    sel = df[(df.z >= lo) & (df.z < hi)]
    centers = 0.5 * (mbins[:-1] + mbins[1:])
    runs = np.sort(sel.run.unique())
    stack = np.full((len(runs), len(centers)), np.nan)
    sel = sel.assign(mbin=np.digitize(sel.logM500, mbins) - 1)
    g = sel.groupby(["run", "mbin"]).f_gas_500.median()
    for i, r in enumerate(runs):
        for b in range(len(centers)):
            if (r, b) in g.index:
                stack[i, b] = g.loc[(r, b)]
    return centers, stack, runs


def fig_erosita(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, tr in zip(axes, ("bgs", "lrg")):
        zlo, zhi, mlo, mhi = TRACERS[tr]
        mbins = np.linspace(13.0, 14.4, 8)
        c, stack, runs = _per_run_relation(df, (zlo, zhi), mbins)
        Mc = 10 ** c
        med = np.nanmedian(stack, axis=0)
        lo16, hi84 = np.nanpercentile(stack, [16, 84], axis=0)
        strong = np.nanpercentile(stack, 2.5, axis=0)        # strongest-feedback edge

        ax.fill_between(Mc, lo16, hi84, color="tab:blue", alpha=0.25,
                        label="BIND Sobol 16-84%")
        ax.plot(Mc, med, "o-", color="tab:blue", lw=2, label="BIND Sobol median")
        ax.plot(Mc, strong, ":", color="navy", lw=1.6, label="BIND strongest-fb (2.5%)")
        ax.plot(Mc, eckert_fgas(Mc), color="tab:green", lw=2,
                label="Eckert+19 (X-ray, weak-fb)")
        ax.plot(Mc, popesso_fgas(Mc), color="tab:red", lw=2,
                label="Popesso+24 eROSITA (strong-fb)")
        ax.axhline(F_B_COSMIC, color="k", ls="--", lw=1, label=r"$\Omega_b/\Omega_m$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$M_{500c}\ [M_\odot/h]$")
        ax.set_ylabel(r"$f_{\rm gas,500}$")
        ax.set_title(f"{tr.upper()}  ($z\\in[{zlo},{zhi}]$, {len(runs)} runs)")
        ax.set_ylim(0, F_B_COSMIC * 1.15)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("BIND f_gas(M500) across SB35 Sobol vs eROSITA X-ray "
                 "(velocity-free kSZ leg)", y=1.01)
    fig.tight_layout()
    p = OUT / "fgas_M500_vs_erosita.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"[ksz] wrote {p}")

    # quick verdict at group scale (logM500 ~ 13.5, BGS window)
    gi = np.argmin(np.abs(c - 13.5))
    print(f"[ksz] BGS @ logM500~{c[gi]:.2f}: BIND median f_gas={med[gi]:.3f} "
          f"(16-84 [{lo16[gi]:.3f},{hi84[gi]:.3f}]); "
          f"Eckert={eckert_fgas(10**c[gi]):.3f}  Popesso/eROSITA={popesso_fgas(10**c[gi]):.3f}")
    n_reach = np.mean(stack[:, gi] <= popesso_fgas(10 ** c[gi]))
    print(f"[ksz] fraction of Sobol nodes reaching the eROSITA strong-fb band "
          f"at this mass: {n_reach:.1%}")


def fig_dichotomy(df):
    """BGS/ELG f_gas inversion test (spec Fig 6c): do the same Sobol nodes that
    make BGS gas-poor keep ELG gas-rich, and is that controlled by A_AGN?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def run_med(tr):
        zlo, zhi, mlo, mhi = TRACERS[tr]
        s = df[(df.z >= zlo) & (df.z < zhi) & (df.logM500 >= mlo) & (df.logM500 < mhi)]
        return s.groupby("run").agg(fgas=("f_gas_500", "median"),
                                    a_agn=(A_AGN, "first"), a_sn=(A_SN, "first"))

    bgs, elg = run_med("bgs"), run_med("elg")
    j = bgs.join(elg, lsuffix="_bgs", rsuffix="_elg", how="inner").dropna()
    print(f"[ksz] dichotomy: {len(j)} runs with both BGS & ELG halos in-window")

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    sc = ax.scatter(j.fgas_bgs, j.fgas_elg, c=np.log10(j.a_agn_bgs),
                    cmap="viridis", s=26, edgecolor="k", lw=0.3)
    ax.axline((0.1, 0.1), slope=1, color="grey", ls=":", lw=1, label="equal f_gas")
    fig.colorbar(sc, label=r"$\log_{10}$ BlackHoleFeedbackFactor ($A_{\rm AGN}$)")
    ax.set_xlabel(r"$f_{\rm gas,500}$  BGS-mass, $z<0.45$")
    ax.set_ylabel(r"$f_{\rm gas,500}$  ELG-mass, $z\sim1$")
    ax.set_title("BGS/ELG gas-fraction inversion vs AGN feedback\n"
                 "(ELG leg mass-floor limited -- see note)")
    r = np.corrcoef(np.log10(j.a_agn_bgs), j.fgas_bgs)[0, 1]
    ax.text(0.03, 0.96, f"corr(logA_AGN, f_gas^BGS) = {r:+.2f}",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / "bgs_elg_dichotomy.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"[ksz] wrote {p}")
    np.savez(OUT / "dichotomy.npz", run=j.index.values,
             fgas_bgs=j.fgas_bgs.values, fgas_elg=j.fgas_elg.values,
             a_agn=j.a_agn_bgs.values, a_sn=j.a_sn_bgs.values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracer", choices=["bgs", "lrg", "elg", "all"], default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[ksz] loading {PARQUET}")
    df = _load()
    print(f"[ksz] {len(df):,} halo-instances, {df.run.nunique()} runs, "
          f"z {df.z.min():.2f}-{df.z.max():.2f}")
    fig_erosita(df)
    fig_dichotomy(df)


if __name__ == "__main__":
    main()
