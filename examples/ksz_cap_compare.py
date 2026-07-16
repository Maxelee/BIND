"""P4 / D4 (proper observable): confront BIND with the Hadzhiyska+26 kSZ stacks in
THEIR estimator -- the compensated-aperture (CAP, disk-minus-ring) signal -- not the
local tau or the GNFW. This is what resolves the spurious ~30x "tension": the data
are T_kSZ^CAP (tau integrated over a compensated aperture, units arcmin^2), a
different quantity from the local tau(R) we first compared to the GNFW.

CAP at aperture theta_d:  tau^CAP(theta_d) = ∮_{r<theta_d} tau dΩ - ∮_{theta_d<r<√2 theta_d} tau dΩ
(equal-area disk minus ring). Apply it to BIND's stacked tau(R/r200) profile (M*-
matched bins, BIND SHMR -> r200), convert R/r200 -> theta via the angular diameter
distance, and overlay the digitized DESI x ACT points (Hadzhiyska+26 Fig. with the
M*-split BGS stacks). No GNFW, no velocity -- the tau side only.

Normalization (confirmed from the paper): (r/r_fid) is the velocity-reconstruction
FIDELITY coefficient r=<v_true v_rec>/(sig_true sig_rec), =1 at the fiducial (BGS
r=0.64, ELG r=0.55) -> NO radial scaling on the plotted points. CAP filter = +1 on
the disk, -1 on the equal-area ring [theta_d, sqrt2 theta_d]. tau^CAP =
T_kSZ^CAP/(T_CMB sigma_v/c), sigma_v^true~300 km/s (used throughout). Result: BIND
sits ~2-6x above the data at small/mid aperture, converging to ~1x at the largest
aperture -> BIND's gas is too CENTRALLY concentrated; the data show it pushed out
(TNG feedback too weak), ~2-6x (inner), consistent with D1 f_gas + Siegel/Bigwood.
(The earlier spurious 30x was local-tau-vs-GNFW, the wrong observable.)

    python examples/ksz_cap_compare.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CEPH = Path(os.environ.get("CEPH", "/mnt/home/mlee1/ceph"))
OUT = Path(os.environ.get("OUTPUT_ROOT", CEPH / "bind_science")) / "ksz_confront"

Om, OL, h = 0.3089, 0.6911, 0.6774
C_KMS, H0 = 299792.458, 100 * 0.6774
ARCMIN = 180 * 60 / np.pi
# T_CMB * sigma_v/c with sigma_v^true ~ 300 km/s (paper, used throughout); (r/r_fid)=1
# at fiducial (it is a velocity-reconstruction fidelity factor, NOT a radial scaling).
SIGMA_V_KMS = 300.0
T_CMB_SIGV = 2.7255e6 * SIGMA_V_KMS / 299792.458         # uK

# digitized BGS BRIGHT-20.2 (z=0.26) CAP points: theta[arcmin], T^CAP*(r/r_fid)[uK arcmin^2]
# (Hadzhiyska+26, M*-split panel; clean mid-aperture points + ~35% errors)
DATA = {
    "M*>11.0":  ([2.5, 3.4, 4.2, 5.0, 5.9, 6.7, 7.5], [2.5, 4.5, 6.5, 9.0, 11.0, 13.5, 17.0]),
    "M*>11.25": ([2.5, 3.4, 4.2, 5.0, 5.9, 6.7, 7.5], [4.5, 8.0, 17.0, 22.0, 30.0, 37.0, 40.0]),
}
CUT_IDX = {"M*>11.0": 0, "M*>11.25": 1}


def _Ez(z):
    return np.sqrt(Om * (1 + z) ** 3 + OL)


def _DA(z, n=3000):
    zz = np.linspace(0, z, n)
    return (C_KMS / H0) * np.trapezoid(1 / _Ez(zz), zz) / (1 + z) / h     # phys Mpc


def _r200phys(logM200, z):
    M = 10 ** logM200 / h
    rho = 2.775e11 * h ** 2 * _Ez(z) ** 2
    return (3 * M / (4 * np.pi * 200 * rho)) ** (1 / 3)


def cap(theta_d, theta, tau):
    """Compensated aperture photometry of a radial tau(theta) profile -> arcmin^2."""
    f = np.linspace(theta.min(), max(theta.max(), 1.5 * theta_d), 5000)
    tt = np.interp(f, theta, tau, left=tau[0], right=0.0)
    disk = 2 * np.pi * np.trapezoid(np.where(f <= theta_d, tt * f, 0.0), f)
    ring = 2 * np.pi * np.trapezoid(np.where((f > theta_d) & (f <= np.sqrt(2) * theta_d), tt * f, 0.0), f)
    return disk - ring


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.load(OUT / "bind_mstar_xprof_snap085.npz")
    x, z = c["x"], 0.26
    da = _DA(z)
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(13, 5.2))
    for lbl, col in [("M*>11.0", "tab:green"), ("M*>11.25", "tab:orange")]:
        ci = CUT_IDX[lbl]
        lM = np.nanmedian(c["logM200"][:, ci]); r2 = _r200phys(lM, z)
        theta = x * r2 / da * ARCMIN
        tau = np.nanmedian(c["tau"][:, ci, :], axis=0)
        th_d = np.array(DATA[lbl][0])
        bind_cap = np.array([cap(t, theta, tau) for t in th_d])
        data_cap = np.array(DATA[lbl][1]) / T_CMB_SIGV
        ax.errorbar(th_d, data_cap, yerr=0.35 * data_cap, fmt="o", color=col,
                    capsize=3, label=f"DESIxACT {lbl}")
        ax.plot(th_d, bind_cap, "-s", color=col, mfc="white", lw=2,
                label=f"BIND {lbl} (logM200={lM:.2f})")
        axr.plot(th_d, bind_cap / data_cap, "-o", color=col, lw=2, label=lbl)
    ax.set_xlabel(r"$\theta$ [arcmin]"); ax.set_ylabel(r"$\tau^{\rm CAP}$ [arcmin$^2$]")
    ax.set_yscale("log"); ax.set_title("M*-matched CAP kSZ: BIND vs DESI$\\times$ACT")
    ax.legend(fontsize=7)
    axr.axhline(1, color="k", ls=":"); axr.axhspan(0.5, 2, color="gray", alpha=0.12)
    axr.set_xlabel(r"$\theta$ [arcmin]"); axr.set_ylabel("BIND / data")
    axr.set_title("BIND ~2-6x high (inner) -> ~1x (outer): gas too central")
    axr.set_ylim(0, 4); axr.legend(fontsize=8)
    fig.suptitle(r"Compensated-aperture kSZ: BIND vs DESI$\times$ACT "
                 r"($\sigma_v{=}300$ km/s, $r/r_{\rm fid}{=}1$)", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "ksz_cap_matched.png", dpi=150, bbox_inches="tight")
    print(f"[cap] wrote {OUT}/ksz_cap_matched.png")


if __name__ == "__main__":
    main()
