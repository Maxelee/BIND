"""scatter/fig_fgas_tension.py

Forward-mapping the eROSITA / kSZ gas-fraction tension with the BIND emulator.

Science question (Angle 1 + 3 of the tension memo): observations (eROSITA X-ray,
DESI+ACT kSZ) find group-scale hot-gas fractions a factor ~2-3 below fiducial
IllustrisTNG. Can the BIND forward model reach the observed low f_gas by dialing
up AGN kinetic feedback, and at what cost to the stellar content (SHMR)?

This script needs NO new generation. It reuses the already-generated feedback
cube produced by scatter/scatter_decomposition.py:

    outputs/scatter_diagnostics/scatter_decomposition_cube_cv.npz
      cube_AGN : (5 levels, 1154 halos, 12 noise draws, 16 obs)   params [3,5]
      cube_SN  : (5 levels, 1154 halos, 12 noise draws, 16 obs)   params [2,4]
      levels   : theta_norm in {0.15, 0.325, 0.5(=fiducial), 0.675, 0.85}
      obs_names: M_dm, M_gas, M_star, f_b, ...  (f_gas = M_gas/(M_dm+M_gas+M_star))
      masses / log_mass : per-halo M200c [M_sun/h]

The AGN axis varies A_AGN1 (RadioFeedbackFactor, kinetic-mode energy) and
A_AGN2 (RadioFeedbackReorientationFactor, kinetic-mode burstiness) *jointly*.

Aperture caveat: f_gas here is the TOTAL gas fraction within R200c (the native
BIND observable aperture), not hot X-ray gas within R500c. The robust result is
the forward-model RESPONSE of f_gas to feedback; the observed band is an
illustrative overlay (see OBS_* block below — swap in published values).

Usage:  python scatter/fig_fgas_tension.py
Output: paper_figures/fig_fgas_tension.{pdf,png}  and  ..._data.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CUBE = ROOT / "outputs/scatter_diagnostics/scatter_decomposition_cube_cv.npz"
OUT_DIR = ROOT / "paper_figures"

# Cosmology of the CV / fiducial-TNG simulations
OMEGA_M, OMEGA_B = 0.30, 0.049
F_B_COSMIC = OMEGA_B / OMEGA_M               # 0.163

# SB35 raw ranges for the two AGN kinetic params (both LogFlag=1), for axis labels
AGN1_LO, AGN1_HI = 0.25, 4.0   # RadioFeedbackFactor   (fiducial 1.0  -> theta_norm 0.5)
AGN2_LO, AGN2_HI = 10.0, 40.0  # ReorientationFactor   (fiducial 20.0 -> theta_norm 0.5)


def lvl_to_raw(lvl, lo, hi):
    """theta_norm level -> raw parameter value for a log-flagged SB35 param."""
    return 10.0 ** (np.log10(lo) + lvl * (np.log10(hi) - np.log10(lo)))


# ---------------------------------------------------------------------------
# Illustrative observational target band.  *** NOT a fit to data ***
# Encodes the reported eROSITA/kSZ group-scale gas deficit (factor ~2-3 below
# fiducial TNG, shrinking toward cluster scales). Replace `obs_fgas_band` with
# published f_gas(M500c) measurements (e.g. eRASS/eFEDS, DESI+ACT kSZ) when
# making the final paper figure.
def obs_fgas_band(log_m200c):
    """Return (lo, hi) illustrative observed total-gas-fraction envelope vs M200c.

    Anchored so that at group scale (1e13.3) observations sit a factor ~2.5
    below the fiducial-TNG f_gas, relaxing to ~1.2x by 1e14.5. Expressed as a
    fraction of the cosmic baryon fraction for a smooth, monotone shape.
    """
    # deficit factor relative to cosmic: rises with mass (more gas retained)
    frac_lo = np.interp(log_m200c, [13.0, 13.5, 14.0, 14.5], [0.18, 0.27, 0.40, 0.55])
    frac_hi = np.interp(log_m200c, [13.0, 13.5, 14.0, 14.5], [0.30, 0.42, 0.58, 0.75])
    return frac_lo * F_B_COSMIC, frac_hi * F_B_COSMIC


# Observed group-scale stellar-mass-to-halo-mass (aperture incl. satellites/ICL),
# illustrative band for the feasibility panel.
OBS_SHMR_LO, OBS_SHMR_HI = 0.008, 0.020


def fgas_per_halo(cube, obs):
    """f_gas = M_gas/(M_dm+M_gas+M_star), median over K -> (n_levels, n_halos)."""
    iM = [obs.index(x) for x in ("M_dm", "M_gas", "M_star")]
    Mdm, Mg, Ms = cube[..., iM[0]], cube[..., iM[1]], cube[..., iM[2]]
    with np.errstate(divide="ignore", invalid="ignore"):
        fg = Mg / (Mdm + Mg + Ms)
    return np.nanmedian(fg, axis=2)


def mstar_frac_per_halo(cube, obs, m200c):
    """M_star(aperture)/M200c, median over K -> (n_levels, n_halos)."""
    iMs = obs.index("M_star")
    Ms = np.nanmedian(cube[..., iMs], axis=2)        # (levels, halos)
    return Ms / m200c[None, :]


def binned_curve(x, y, edges):
    """Median and 16/84 percentiles of y in bins of x. Returns centers, med, lo, hi."""
    centers, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b) & np.isfinite(y)
        if m.sum() < 5:
            continue
        centers.append(0.5 * (a + b))
        med.append(np.nanmedian(y[m]))
        lo.append(np.nanpercentile(y[m], 16))
        hi.append(np.nanpercentile(y[m], 84))
    return map(np.asarray, (centers, med, lo, hi))


def main():
    d = np.load(CUBE)
    obs = list(d["obs_names"])
    levels = d["levels"]
    logm = d["log_mass"]
    masses = d["masses"]
    fid = int(np.argmin(np.abs(levels - 0.5)))      # fiducial level index

    fg_agn = fgas_per_halo(d["cube_AGN"], obs)       # (5, 1154)
    fg_sn = fgas_per_halo(d["cube_SN"], obs)
    msf_agn = mstar_frac_per_halo(d["cube_AGN"], obs, masses)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    cmap = matplotlib.colormaps["viridis"]
    cols = [cmap(i / (len(levels) - 1)) for i in range(len(levels))]

    edges = np.arange(13.0, 14.61, 0.2)

    # ---- Panel A: f_gas(M_halo) family of AGN-feedback curves -----------------
    for li, lvl in enumerate(levels):
        c, med, lo, hi = binned_curve(logm, fg_agn[li], edges)
        a1 = lvl_to_raw(lvl, AGN1_LO, AGN1_HI)
        label = (rf"$A_{{\rm AGN1}}$={a1:.2f}" + (" (fiducial)" if li == fid else ""))
        lw = 3.0 if li == fid else 1.8
        axA.plot(c, med, "-o", color=cols[li], lw=lw, ms=4, label=label, zorder=3)
        if li == fid:
            axA.fill_between(c, lo, hi, color=cols[li], alpha=0.15, zorder=1,
                             label=r"halo-to-halo 16-84% (fiducial)")

    # cosmic ceiling + illustrative observed band
    axA.axhline(F_B_COSMIC, ls=":", color="k", lw=1.3)
    axA.text(14.45, F_B_COSMIC + 0.003, r"cosmic $\Omega_b/\Omega_m$",
             ha="right", va="bottom", fontsize=9)
    lm = np.linspace(13.0, 14.5, 50)
    o_lo, o_hi = obs_fgas_band(lm)
    axA.fill_between(lm, o_lo, o_hi, color="crimson", alpha=0.18, zorder=0)
    axA.plot(lm, 0.5 * (o_lo + o_hi), color="crimson", lw=1.5, ls="--",
             label="observed target (illustrative)")

    axA.set_xlabel(r"$\log_{10} M_{\rm 200c}\ [M_\odot/h]$")
    axA.set_ylabel(r"$f_{\rm gas}(<R_{\rm 200c}) = M_{\rm gas}/M_{\rm tot}$")
    axA.set_title("(A)  BIND forward map: AGN kinetic feedback vs. gas fraction")
    axA.legend(fontsize=8, loc="lower right", framealpha=0.9)
    axA.grid(alpha=0.25)

    # ---- Panel B: f_gas - M*/M_halo trade-off at group scale ------------------
    grp = (logm >= 13.0) & (logm < 13.5)
    fg_path = np.array([np.nanmedian(fg_agn[li, grp]) for li in range(len(levels))])
    ms_path = np.array([np.nanmedian(msf_agn[li, grp]) for li in range(len(levels))])

    axB.plot(fg_path, ms_path, "-", color="0.4", lw=1.5, zorder=2)
    for li, lvl in enumerate(levels):
        a1 = lvl_to_raw(lvl, AGN1_LO, AGN1_HI)
        axB.scatter(fg_path[li], ms_path[li], s=120, color=cols[li],
                    edgecolor="k", lw=0.8, zorder=4)
        axB.annotate(rf"$A_{{\rm AGN1}}$={a1:.2f}", (fg_path[li], ms_path[li]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
    # arrow indicating the feedback direction (weak -> strong AGN)
    axB.annotate("", xy=(fg_path[-1], ms_path[-1]), xytext=(fg_path[0], ms_path[0]),
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=1.2,
                                 connectionstyle="arc3,rad=0.0"), zorder=1)

    # observed target windows (group scale)
    o_lo, o_hi = obs_fgas_band(np.array([13.25]))
    axB.axvspan(float(o_lo), float(o_hi), color="crimson", alpha=0.15,
                label=r"observed $f_{\rm gas}$ target")
    axB.axhspan(OBS_SHMR_LO, OBS_SHMR_HI, color="steelblue", alpha=0.15,
                label="observed SHMR band")

    axB.set_xlabel(r"$f_{\rm gas}(<R_{\rm 200c})$")
    axB.set_ylabel(r"aperture $M_\star/M_{\rm 200c}$ (incl. satellites)")
    axB.set_title(r"(B)  Cost of gas removal at group scale ($10^{13}$-$10^{13.5}$)")
    axB.legend(fontsize=8, loc="best", framealpha=0.9)
    axB.grid(alpha=0.25)

    fig.suptitle(
        "Forward-modelling the eROSITA / kSZ gas-fraction tension with BIND",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig_fgas_tension.pdf"
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"[saved] {pdf}")
    print(f"[saved] {pdf.with_suffix('.png')}")

    # ---- numeric summary + data dump -----------------------------------------
    print("\n=== Group-scale (1e13-1e13.5) AGN feedback response ===")
    print(f"  {'A_AGN1':>8} {'A_AGN2':>8} {'f_gas':>8} {'M*/Mh':>8}")
    for li, lvl in enumerate(levels):
        print(f"  {lvl_to_raw(lvl, AGN1_LO, AGN1_HI):8.2f} "
              f"{lvl_to_raw(lvl, AGN2_LO, AGN2_HI):8.1f} "
              f"{fg_path[li]:8.4f} {ms_path[li]:8.5f}")
    drop = 1 - fg_path[-1] / fg_path[fid]
    print(f"\n  fiducial f_gas = {fg_path[fid]:.4f}; strongest-AGN f_gas = "
          f"{fg_path[-1]:.4f}  ({drop*100:.0f}% removed)")
    o_lo, o_hi = obs_fgas_band(np.array([13.25]))
    print(f"  illustrative observed target = {float(o_lo):.3f}-{float(o_hi):.3f}; "
          f"strongest sampled AGN still {'ABOVE' if fg_path[-1] > o_hi else 'within'} it.")

    np.savez_compressed(
        OUT_DIR / "fig_fgas_tension_data.npz",
        levels=levels, log_mass=logm, masses=masses,
        fgas_agn=fg_agn, fgas_sn=fg_sn, mstar_frac_agn=msf_agn,
        agn1_raw=np.array([lvl_to_raw(l, AGN1_LO, AGN1_HI) for l in levels]),
        agn2_raw=np.array([lvl_to_raw(l, AGN2_LO, AGN2_HI) for l in levels]),
        f_b_cosmic=F_B_COSMIC,
    )
    print(f"[saved] {OUT_DIR / 'fig_fgas_tension_data.npz'}")


if __name__ == "__main__":
    main()
