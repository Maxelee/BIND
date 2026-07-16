"""P4 / D2-D3: confront BIND stacked tau (electron-column) AND y (tSZ pressure)
profiles with the Hadzhiyska+26 (DESI DR2 x ACT DR6, arXiv:2604.19745) kSZ GNFW
density fits, across the 256-node SB35 Sobol grid.

Their kSZ density profile (Eq. 26-27), r normalized to r200c, fixed gamma=-0.5,
x_c=0.7, free {rho0, alpha, beta}:

    rho_gas(r) = f_b rho_cr(z) * rho0 * (r/(x_c r200c))^gamma
                 * [1 + (r/(x_c r200c))^alpha]^(-(beta+gamma)/alpha)

Outer log-slope = -beta. BIND tau = sigma_T int n_e dl (no velocity needed); y is
the tSZ Compton parameter (already physical). One pass over the patches produces
both legs; tau/y ~ k_B T_e gives the temperature decomposition (spec Eq. 9).

Comparison is in x = R_proj/r200c. The "clean alpha" fit restricts to the
data-sensitive, BIND-resolution-reliable band x in [FIT_XMIN, FIT_XMAX] (drops the
pixel-limited inner core that biased alpha high, and the bg-subtracted outskirts).

SCOPE / caveats (see docs/ksz_desi_act_plan.md):
  * BIND halo floor M200>=1e13 -> ELG hosts (~10^12.5) below it: ELG GNFW shown as
    data target, BIND uses logM200~13.2 halos. Massive BGS / LRG mass overlaps BIND.
  * Matched by HALO MASS (M200c); BIND M_star_500 is total-halo not central.

    # serial:
    python examples/ksz_tau_gnfw.py --reduce --snaps 85,46 --nodes 0-255
    # MPI (openmpi + mpi4py in the venv): see run_ksz_tau.sh
    mpirun -n 64 python examples/ksz_tau_gnfw.py --reduce --snaps 85,46 --nodes 0-255 --mpi
    python examples/ksz_tau_gnfw.py --plot
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

CEPH = Path(os.environ.get("CEPH", "/mnt/home/mlee1/ceph"))
RUNS = CEPH / "bind_sb35/runs"
OUT = Path(os.environ.get("OUTPUT_ROOT", CEPH / "bind_science")) / "ksz_confront"

PIX_MPCH = 6.25 / 128.0                 # comoving patch pixel, Mpc/h
SIGMA_T = 6.6524e-25                     # cm^2
M_P = 1.6726e-24                         # g
MSUN_G = 1.989e33
MPC_CM = 3.0857e24
X_E_PER_MASS = 0.88 / M_P                # free electrons per gram
H = 0.6774
M_E_C2_OVER_KB = 5.93e9                  # m_e c^2 / k_B  [K]

# clean-alpha fit band (x = R/r200c): data-sensitive AND BIND-reliable overlap
FIT_XMIN, FIT_XMAX = 0.3, 1.5
BG_X = (2.5, 3.0)                        # LOS background annulus in x
RHO_CR0 = 2.775e11 * H ** 2             # Msun/Mpc^3 physical
F_B = 0.0490 / 0.3089                   # Omega_b/Omega_m

# mean halo mass (log10 M200c [Msun/h]) per Hadzhiyska sample. ELG≈12.2 (paper;
# below BIND's 1e13 floor); massive BGS bins are the BIND-overlapping ones.
# ⚠ BGS M200c are PLACEHOLDERS pending the paper's per-sample halo masses.
HADZ_LOGM200 = {"ELG_all": 12.2, "ELG_Ms9.0": 12.2, "ELG_Ms9.5": 12.25,
                "BGS_all": 12.9, "BGS_Ms10.0": 12.9, "BGS_Ms10.5": 13.0,
                "BGS_Ms11.0": 13.3, "BGS_Ms11.25": 13.5}


def _Ez2(z):
    return 0.3089 * (1 + z) ** 3 + 0.6911


def project_tau_absolute(R_over_r200, logM200c, z, logrho0, alpha, beta, l_max=8.0, n_l=600):
    """Hadzhiyska GNFW -> ABSOLUTE tau at projected R/r200c, given M200c [Msun/h], z.
    rho_gas = f_b rho_cr(z) rho0 GNFW(r/r200c); tau = sigma_T x_e/m_p ∫ rho_gas dl."""
    M = 10 ** logM200c / H                                  # Msun
    rho_cr = RHO_CR0 * _Ez2(z)                              # Msun/Mpc^3
    r200 = (3 * M / (4 * np.pi * 200 * rho_cr)) ** (1 / 3)  # Mpc physical
    amp = F_B * rho_cr * 10 ** logrho0                      # Msun/Mpc^3
    xp = np.atleast_1d(R_over_r200).astype(float)
    l = np.linspace(0.0, l_max, n_l)
    r = np.sqrt(xp[:, None] ** 2 + l[None, :] ** 2)
    col = 2.0 * np.trapezoid(amp * gnfw_density(r, alpha, beta), l, axis=1) * r200  # Msun/Mpc^2
    sigma_gcm2 = col * MSUN_G / MPC_CM ** 2
    return SIGMA_T * X_E_PER_MASS * sigma_gcm2

# Halo-mass bins where BIND overlaps the data (log10 M200c, Msun/h).
MBINS = np.array([13.0, 13.4, 13.8, 14.2, 15.0])
GAMMA, X_C = -0.5, 0.7                   # fixed in Hadzhiyska+26

# ── Hadzhiyska+26 Table II (user-supplied) ────────────────────────────────────
HADZ = {
    "ELG_all":     dict(z=1.17, logrho0=(6.59, 0.90), alpha=(0.215, 0.026), beta=(4.55, 1.30), logA2h=(-0.72, 0.23), snr=7.50),
    "ELG_Ms9.0":   dict(z=1.17, logrho0=(6.60, 0.94), alpha=(0.212, 0.026), beta=(4.54, 1.28), logA2h=(-0.71, 0.24), snr=7.21),
    "ELG_Ms9.5":   dict(z=1.17, logrho0=(6.48, 0.88), alpha=(0.212, 0.027), beta=(4.39, 1.22), logA2h=(-0.72, 0.25), snr=6.11),
    "BGS_all":     dict(z=0.26, logrho0=(9.33, 1.25), alpha=(0.172, 0.020), beta=(7.27, 1.24), logA2h=(0.74, 0.22), snr=7.36),
    "BGS_Ms10.0":  dict(z=0.26, logrho0=(9.06, 1.33), alpha=(0.175, 0.021), beta=(7.08, 1.32), logA2h=(0.72, 0.24), snr=7.15),
    "BGS_Ms10.5":  dict(z=0.26, logrho0=(9.18, 1.41), alpha=(0.176, 0.021), beta=(7.04, 1.39), logA2h=(0.63, 0.28), snr=7.83),
    "BGS_Ms11.0":  dict(z=0.26, logrho0=(7.06, 1.62), alpha=(0.204, 0.029), beta=(4.80, 1.57), logA2h=(0.06, 0.50), snr=8.38),
    "BGS_Ms11.25": dict(z=0.26, logrho0=(7.57, 1.64), alpha=(0.200, 0.028), beta=(5.17, 1.70), logA2h=(0.33, 0.47), snr=8.99),
}
# snap -> tracer label for plotting
SNAP_TRACER = {85: "BGS", 46: "ELG"}


# ── GNFW density + projection ─────────────────────────────────────────────────
def gnfw_density(x, alpha, beta, gamma=GAMMA, xc=X_C):
    """rho_GNFW(x)/rho0, x = r/r200c (shape; rho0 factored out)."""
    u = x / xc
    return u ** gamma * (1.0 + u ** alpha) ** (-(beta + gamma) / alpha)


def project_tau_shape(xp, alpha, beta, l_max=8.0, n_l=400):
    """LOS-projected GNFW shape at projected xp=R/r200c (units of rho0*r200c)."""
    xp = np.atleast_1d(xp).astype(float)
    l = np.linspace(0.0, l_max, n_l)
    r = np.sqrt(xp[:, None] ** 2 + l[None, :] ** 2)
    return 2.0 * np.trapezoid(gnfw_density(r, alpha, beta), l, axis=1)


# ── BIND tau+y stacking (by halo mass, across Sobol nodes) ────────────────────
def _scale_factor(snap):
    for p in (CEPH / "bind_lightcone_tng" / f"snap_{snap:03d}" / "stage1").glob("*.json"):
        d = json.loads(p.read_text())
        for k in ("scale_factor", "a"):
            if k in d:
                return float(d[k])
    raise RuntimeError(f"no scale factor for snap {snap}")


def tau_from_gas(gas_patch, a):
    """Gas-mass patch [Msun/h]/pixel -> tau/pixel (electron column)."""
    area_cm2 = (PIX_MPCH * a / H * MPC_CM) ** 2
    return gas_patch.astype(np.float64) * (SIGMA_T * X_E_PER_MASS * MSUN_G / (H * area_cm2))


def stack_node_snap(run, snap, x_edges, a):
    """Stack tau(x) and y(x) per M200c bin for one run x snap, bg-subtracted.
    Returns (tau_prof, y_prof) each (nb, nr), and counts (nb,)."""
    rd = RUNS / f"run_{run:04d}" / f"snap_{snap:03d}"
    files = sorted(rd.glob("composite_slab*.npz"))
    if not files:
        return None
    nb, nr = len(MBINS) - 1, len(x_edges) - 1
    st = np.zeros((nb, nr)); sy = np.zeros((nb, nr)); sn = np.zeros((nb, nr)); cnt = np.zeros(nb, int)
    for f in files:
        d = np.load(f)
        if int(d["n_halos"]) == 0:
            continue
        gas = d["generated_patches"][:, 1]              # Msun/h
        ymap = d["thermo_patches"][:, 0]                # Compton-y (physical)
        M, r200 = d["halo_masses"], d["halo_r200"]
        P = gas.shape[-1]; cen = P // 2
        yy, xx = np.mgrid[0:P, 0:P]
        rr = np.hypot(xx - cen, yy - cen) * PIX_MPCH    # comoving Mpc/h
        lM = np.log10(M)
        for h in range(len(M)):
            b = int(np.digitize(lM[h], MBINS)) - 1
            if b < 0 or b >= nb:
                continue
            t = tau_from_gas(gas[h], a)
            yv = ymap[h].astype(np.float64)
            x = rr / r200[h]
            mbg = (x > BG_X[0]) & (x < BG_X[1])
            t = t - (t[mbg].mean() if mbg.any() else 0.0)
            yv = yv - (yv[mbg].mean() if mbg.any() else 0.0)
            idx = np.digitize(x.ravel(), x_edges) - 1
            ok = (idx >= 0) & (idx < nr)
            np.add.at(st[b], idx[ok], t.ravel()[ok])
            np.add.at(sy[b], idx[ok], yv.ravel()[ok])
            np.add.at(sn[b], idx[ok], 1.0)
            cnt[b] += 1
    den = np.where(sn > 0, sn, 1.0)
    tau_prof = np.where(sn > 0, st / den, np.nan)
    y_prof = np.where(sn > 0, sy / den, np.nan)
    return tau_prof, y_prof, cnt


SHARDS = OUT / "shards_tauy"
X_EDGES = np.geomspace(0.08, 3.0, 18)
XC = np.sqrt(X_EDGES[1:] * X_EDGES[:-1])


def do_reduce(snaps, nodes, use_mpi=False, force=False):
    """Stack each (snap, node) into a small shard npz. Restart-safe: existing
    shards are skipped unless force. MPI partitions tasks across ranks."""
    if use_mpi:
        from mpi4py import MPI
        rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    else:
        rank, size = 0, 1
    SHARDS.mkdir(parents=True, exist_ok=True)
    afac = {s: _scale_factor(s) for s in snaps}
    tasks = [(s, n) for s in snaps for n in nodes]
    mine = tasks[rank::size]
    for k, (s, n) in enumerate(mine):
        shard = SHARDS / f"tauy_snap{s:03d}_node{n:04d}.npz"
        if shard.exists() and not force:
            continue
        r = stack_node_snap(n, s, X_EDGES, afac[s])
        if r is None:
            continue
        np.savez(shard, tau=r[0], y=r[1], cnts=r[2], a=afac[s], node=n, snap=s)
        if rank == 0 and k % 16 == 0:
            print(f"[d2] rank0 {k+1}/{len(mine)} (snap {s} node {n})", flush=True)


# ── stellar-mass-matched stacking (match Hadzhiyska central-galaxy M* bins) ────
MSTAR_AP_KPCH = 50.0                     # central aperture (~1 pixel) for galaxy M*
MSTAR_CUTS = [11.0, 11.25]              # cumulative log10 M* thresholds BIND can host
CUT_SAMPLE = {11.0: "BGS_Ms11.0", 11.25: "BGS_Ms11.25"}
SHARDS_MS = OUT / "shards_mstar"


def central_mstar(stars_patch, rr_kpch):
    """Central-galaxy M* proxy = Stars mass within MSTAR_AP_KPCH of the centre."""
    return stars_patch[rr_kpch < MSTAR_AP_KPCH].sum()


def stack_node_snap_mstar(run, snap, x_edges, a):
    """Stack ABSOLUTE tau(x) + y(x) for cumulative central-M* thresholds; also
    return mean log10 M200 per threshold (to set r200c for the GNFW projection)."""
    rd = RUNS / f"run_{run:04d}" / f"snap_{snap:03d}"
    files = sorted(rd.glob("composite_slab*.npz"))
    if not files:
        return None
    nc, nr = len(MSTAR_CUTS), len(x_edges) - 1
    st = np.zeros((nc, nr)); sy = np.zeros((nc, nr)); sn = np.zeros((nc, nr))
    smg = np.zeros((nc, nr))                                # cumulative bg-sub gas mass [Msun/h]
    smtot = np.zeros((nc, nr))                              # bg-sub TOTAL mass/pixel [Msun/h] (kappa leg)
    cnt = np.zeros(nc, int); sM = np.zeros(nc)
    for f in files:
        d = np.load(f)
        if int(d["n_halos"]) == 0:
            continue
        gp = d["generated_patches"]
        dm, gas, stars = gp[:, 0], gp[:, 1], gp[:, 2]
        ymap = d["thermo_patches"][:, 0]; M, r200 = d["halo_masses"], d["halo_r200"]
        P = gas.shape[-1]; cc = P // 2
        yy, xx = np.mgrid[0:P, 0:P]
        rr = np.hypot(xx - cc, yy - cc) * PIX_MPCH                  # Mpc/h comoving
        rr_kpch = rr * 1e3
        gphys = gas.astype(np.float64)                     # gas mass [Msun/h] per pixel
        mtot = (dm + gas + stars).astype(np.float64)       # total matter [Msun/h] per pixel (kappa)
        for h in range(len(M)):
            lms = np.log10(max(central_mstar(stars[h], rr_kpch), 1.0))
            t = tau_from_gas(gas[h], a); yv = ymap[h].astype(np.float64)
            x = rr / r200[h]; mbg = (x > BG_X[0]) & (x < BG_X[1])
            t = t - (t[mbg].mean() if mbg.any() else 0.0)
            yv = yv - (yv[mbg].mean() if mbg.any() else 0.0)
            gbg = gphys[h] - (gphys[h][mbg].mean() if mbg.any() else 0.0)   # bg-sub gas/pixel
            mtb = mtot[h] - (mtot[h][mbg].mean() if mbg.any() else 0.0)     # bg-sub total mass/pixel
            idx = np.digitize(x.ravel(), x_edges) - 1
            ok = (idx >= 0) & (idx < nr)
            # cumulative bg-sub gas mass within each aperture x_edges[k+1]
            gper = np.zeros(nr); np.add.at(gper, idx[ok], gbg.ravel()[ok])
            mgcum = np.cumsum(gper)
            for ci, cut in enumerate(MSTAR_CUTS):
                if lms >= cut:
                    np.add.at(st[ci], idx[ok], t.ravel()[ok])
                    np.add.at(sy[ci], idx[ok], yv.ravel()[ok])
                    np.add.at(smtot[ci], idx[ok], mtb.ravel()[ok])
                    np.add.at(sn[ci], idx[ok], 1.0)
                    smg[ci] += mgcum
                    cnt[ci] += 1; sM[ci] += np.log10(M[h])
    den = np.where(sn > 0, sn, 1.0)
    tau = np.where(sn > 0, st / den, np.nan); y = np.where(sn > 0, sy / den, np.nan)
    mtot_prof = np.where(sn > 0, smtot / den, np.nan)      # mean bg-sub total mass/pixel [Msun/h]
    cden = np.where(cnt > 0, cnt, 1)[:, None]
    mgas_cum = np.where(cnt[:, None] > 0, smg / cden, np.nan)   # mean cumulative gas [Msun/h]
    logM200 = np.where(cnt > 0, sM / np.where(cnt > 0, cnt, 1), np.nan)
    return tau, y, cnt, logM200, mgas_cum, mtot_prof


def do_reduce_mstar(snaps, nodes, use_mpi=False, force=False):
    if use_mpi:
        from mpi4py import MPI
        rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    else:
        rank, size = 0, 1
    SHARDS_MS.mkdir(parents=True, exist_ok=True)
    afac = {s: _scale_factor(s) for s in snaps}
    tasks = [(s, n) for s in snaps for n in nodes][rank::size]
    for k, (s, n) in enumerate(tasks):
        shard = SHARDS_MS / f"ms_snap{s:03d}_node{n:04d}.npz"
        if shard.exists() and not force:
            continue
        r = stack_node_snap_mstar(n, s, X_EDGES, afac[s])
        if r is None:
            continue
        np.savez(shard, tau=r[0], y=r[1], cnts=r[2], logM200=r[3], mgas_cum=r[4],
                 mtot_prof=r[5], a=afac[s], node=n, snap=s)
        if rank == 0 and k % 16 == 0:
            print(f"[ms] rank0 {k+1}/{len(tasks)} (snap {s} node {n})", flush=True)


def do_consolidate_mstar(snaps, nodes):
    OUT.mkdir(parents=True, exist_ok=True)
    for s in snaps:
        rows = [(n, np.load(SHARDS_MS / f"ms_snap{s:03d}_node{n:04d}.npz"))
                for n in nodes if (SHARDS_MS / f"ms_snap{s:03d}_node{n:04d}.npz").exists()]
        if not rows:
            print(f"[ms] snap {s}: no shards"); continue
        ns = [n for n, _ in rows]
        out = OUT / f"bind_mstar_xprof_snap{s:03d}.npz"
        np.savez(out, x=XC, cuts=np.array(MSTAR_CUTS), nodes=np.array(ns),
                 a=float(rows[0][1]["a"]),
                 tau=np.stack([d["tau"] for _, d in rows]),
                 y=np.stack([d["y"] for _, d in rows]),
                 cnts=np.stack([d["cnts"] for _, d in rows]),
                 mgas_cum=np.stack([d["mgas_cum"] for _, d in rows]),
                 mtot_prof=np.stack([d["mtot_prof"] for _, d in rows]),
                 logM200=np.stack([d["logM200"] for _, d in rows]))
        print(f"[ms] wrote {out} ({len(ns)} nodes)")


def do_plot_mstar(snap=85, zdat=0.26):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = OUT / f"bind_mstar_xprof_snap{snap:03d}.npz"
    if not f.exists():
        print("[ms] no cache -- run --reduce-mstar"); return
    c = np.load(f); x = c["x"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ci, cut in enumerate(MSTAR_CUTS):
        col = ["tab:blue", "navy"][ci]
        tau = c["tau"][:, ci, :]
        med = np.nanmedian(tau, axis=0); lo, hi = np.nanpercentile(tau, [16, 84], axis=0)
        lM = np.nanmedian(c["logM200"][:, ci])
        ok = med > 0
        axes[0].fill_between(x[ok], lo[ok], hi[ok], color=col, alpha=0.18)
        axes[0].plot(x[ok], med[ok], "o-", color=col, lw=2,
                     label=f"BIND M*>{cut} (logM200={lM:.2f})")
        h = HADZ[CUT_SAMPLE[cut]]
        td = project_tau_absolute(x, lM, zdat, h["logrho0"][0], h["alpha"][0], h["beta"][0])
        axes[0].plot(x, td, "--", color=col, lw=2, label=f"{CUT_SAMPLE[cut]} GNFW")
        # ratio at clean band
        i = (x >= FIT_XMIN) & (x <= FIT_XMAX)
        axes[1].plot(x[i], (med / td)[i], "o-", color=col, lw=2, label=f"M*>{cut}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_ylim(1e-7, 2e-3)
    axes[0].axvspan(FIT_XMIN, FIT_XMAX, color="gray", alpha=0.08)
    axes[0].set_xlabel(r"$R/r_{200c}$"); axes[0].set_ylabel(r"absolute $\tau$")
    axes[0].set_title("M*-matched absolute tau: BIND vs Hadzhiyska BGS"); axes[0].legend(fontsize=7)
    axes[1].axhline(1, color="k", ls=":"); axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$R/r_{200c}$"); axes[1].set_ylabel("BIND / Hadzhiyska")
    axes[1].set_title("amplitude ratio (clean band)"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "tau_mstar_matched.png", dpi=150, bbox_inches="tight")
    print(f"[ms] wrote {OUT}/tau_mstar_matched.png")
    for ci, cut in enumerate(MSTAR_CUTS):
        i = np.argmin(np.abs(x - 0.5))
        med = np.nanmedian(c["tau"][:, ci, :], axis=0)
        lM = np.nanmedian(c["logM200"][:, ci]); h = HADZ[CUT_SAMPLE[cut]]
        td = project_tau_absolute([x[i]], lM, zdat, h["logrho0"][0], h["alpha"][0], h["beta"][0])[0]
        print(f"[ms] M*>{cut} (logM200={lM:.2f}): BIND tau(0.5)={med[i]:.2e}  Hadz={td:.2e}  ratio={med[i]/td:.1f}")


def do_consolidate(snaps, nodes):
    """Merge per-(snap,node) shards into one per-snap npz the plotter reads."""
    OUT.mkdir(parents=True, exist_ok=True)
    for s in snaps:
        rows = []
        for n in nodes:
            shard = SHARDS / f"tauy_snap{s:03d}_node{n:04d}.npz"
            if shard.exists():
                rows.append((n, np.load(shard)))
        if not rows:
            print(f"[d2] snap {s}: no shards"); continue
        ns = [n for n, _ in rows]
        tau = np.stack([d["tau"] for _, d in rows])
        y = np.stack([d["y"] for _, d in rows])
        cnt = np.stack([d["cnts"] for _, d in rows])
        a = float(rows[0][1]["a"])
        out = OUT / f"bind_tauy_xprof_snap{s:03d}.npz"
        np.savez(out, x=XC, mbins=MBINS, nodes=np.array(ns), a=a, tau=tau, y=y, cnts=cnt)
        print(f"[d2] wrote {out}  ({len(ns)} nodes, a={a:.3f})")


# ── fitting / plotting ────────────────────────────────────────────────────────
def fit_gnfw_shape(x, prof, xmin=FIT_XMIN, xmax=FIT_XMAX):
    """Fit (alpha, beta) of projected GNFW to tau(x) in the clean band [xmin,xmax]."""
    from scipy.optimize import least_squares
    ok = np.isfinite(prof) & (prof > 0) & (x >= xmin) & (x <= xmax)
    if ok.sum() < 4:
        return np.nan, np.nan
    lx, ly = x[ok], np.log(prof[ok])

    def resid(p):
        a, b, amp = p
        if not (0.05 < a < 8 and 1.0 < b < 20):
            return np.full_like(ly, 1e3)
        return amp + np.log(np.maximum(project_tau_shape(lx, a, b), 1e-300)) - ly
    try:
        r = least_squares(resid, [0.3, 5.0, ly.mean()], max_nfev=3000)
        return r.x[0], r.x[1]
    except Exception:
        return np.nan, np.nan


def _load_caches():
    return {s: np.load(OUT / f"bind_tauy_xprof_snap{s:03d}.npz")
            for s in (85, 46) if (OUT / f"bind_tauy_xprof_snap{s:03d}.npz").exists()}


def do_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caches = _load_caches()
    if not caches:
        print("[d2] no caches -- run --reduce first"); return

    # ---- Fig A: clean tau shape, BIND envelope vs Hadzhiyska GNFW --------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = [("BGS-mass (z~0.18)", 85, ["BGS_all", "BGS_Ms11.0", "BGS_Ms11.25"], 1),
              ("ELG-mass (z~1.16)", 46, ["ELG_all", "ELG_Ms9.5"], 0)]
    for ax, (title, snap, samples, mbin) in zip(axes, panels):
        if snap in caches:
            c = caches[snap]; x = c["x"]; p = c["tau"][:, mbin, :]
            xn = int(np.argmin(np.abs(x - 0.5)))
            pn = p / p[:, xn][:, None]
            med = np.nanmedian(pn, axis=0)
            lo, hi = np.nanpercentile(pn, [16, 84], axis=0)
            ax.fill_between(x, lo, hi, color="tab:blue", alpha=0.25, label="BIND Sobol 16-84%")
            ax.plot(x, med, "o-", color="tab:blue", lw=2,
                    label=f"BIND median (logM200~{0.5*(MBINS[mbin]+MBINS[mbin+1]):.1f}, N={c['nodes'].size})")
        for s in samples:
            h = HADZ[s]
            ty = project_tau_shape(x, h["alpha"][0], h["beta"][0]); ty = ty / ty[xn]
            ax.plot(x, ty, "--", lw=2, label=f"{s} (β={h['beta'][0]:.1f})")
        ax.axvspan(FIT_XMIN, FIT_XMAX, color="gray", alpha=0.08)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$R_{\rm proj}/r_{200c}$"); ax.set_ylabel(r"$\tau$ (norm. at $x{=}0.5$)")
        ax.set_title(title); ax.legend(fontsize=7); ax.set_ylim(1e-3, 5)
    fig.suptitle("BIND stacked tau shape vs Hadzhiyska+26 GNFW (clean band shaded)", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "tau_shape_vs_hadzhiyska.png", dpi=150, bbox_inches="tight")
    print(f"[d2] wrote {OUT}/tau_shape_vs_hadzhiyska.png")

    # ---- Fig B: clean alpha-beta plane ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    col = {85: "tab:blue", 46: "tab:green"}
    for snap, c in caches.items():
        x = c["x"]
        for mbin in range(len(MBINS) - 1):
            ab = np.array([fit_gnfw_shape(x, c["tau"][i, mbin]) for i in range(c["tau"].shape[0])])
            ab = ab[np.isfinite(ab).all(1)]
            if len(ab) < 3:
                continue
            ax.scatter(ab[:, 0], ab[:, 1], s=9, alpha=0.35, color=col[snap],
                       label=f"BIND {SNAP_TRACER[snap]} logM200 {MBINS[mbin]:.1f}-{MBINS[mbin+1]:.1f}")
    for s, h in HADZ.items():
        mk = "s" if s.startswith("BGS") else "^"
        ax.errorbar(h["alpha"][0], h["beta"][0], xerr=h["alpha"][1], yerr=h["beta"][1],
                    fmt=mk, ms=7, capsize=3, label=s)
    ax.set_xlabel(r"$\alpha$ (GNFW transition)"); ax.set_ylabel(r"$\beta$ (outer slope)")
    ax.set_title(f"Clean shape plane (fit x in [{FIT_XMIN},{FIT_XMAX}])")
    ax.legend(fontsize=6, ncol=2); ax.set_ylim(1, 12); ax.set_xlim(0, 4.5)
    fig.tight_layout(); fig.savefig(OUT / "alpha_beta_plane.png", dpi=150, bbox_inches="tight")
    print(f"[d2] wrote {OUT}/alpha_beta_plane.png")

    # ---- Fig C: y (pressure) profile + tau/y temperature proxy ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for snap, c in caches.items():
        x = c["x"]; mbin = 1
        yv = c["y"][:, mbin, :]; tv = c["tau"][:, mbin, :]
        ymed = np.nanmedian(yv, axis=0)
        ylo, yhi = np.nanpercentile(yv, [16, 84], axis=0)
        axes[0].fill_between(x, np.abs(ylo), np.abs(yhi), color=col[snap], alpha=0.2)
        axes[0].plot(x, np.abs(ymed), "o-", color=col[snap], lw=2,
                     label=f"BIND {SNAP_TRACER[snap]} y (logM200~13.6)")
        # tau/y ~ 1/(k_B T_e) -> T_e proxy = (y/tau) * m_e c^2/k_B
        with np.errstate(divide="ignore", invalid="ignore"):
            Te = (yv / tv) * M_E_C2_OVER_KB
        Tm = np.nanmedian(Te, axis=0)
        axes[1].plot(x, Tm, "o-", color=col[snap], lw=2, label=f"BIND {SNAP_TRACER[snap]}")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel(r"$R_{\rm proj}/r_{200c}$")
        ax.axvspan(FIT_XMIN, FIT_XMAX, color="gray", alpha=0.08); ax.legend(fontsize=8)
    axes[0].set_yscale("log"); axes[0].set_ylabel(r"$\langle y\rangle$"); axes[0].set_title("tSZ pressure profile")
    axes[1].set_yscale("log"); axes[1].set_ylabel(r"$k_B T_e$ proxy $\propto y/\tau$  [K]")
    axes[1].set_title(r"temperature decomposition (spec Eq. 9)")
    fig.tight_layout(); fig.savefig(OUT / "y_and_temperature.png", dpi=150, bbox_inches="tight")
    print(f"[d2] wrote {OUT}/y_and_temperature.png")

    # ---- Fig D: amplitude leg -- absolute tau(R) vs feedback (how much gas) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (title, snap) in zip(axes, [("BGS-mass (z~0.18)", 85), ("ELG-mass (z~1.16)", 46)]):
        if snap not in caches:
            continue
        c = caches[snap]; x = c["x"]
        for mbin, cc in [(0, "tab:cyan"), (1, "tab:blue"), (2, "navy")]:
            p = c["tau"][:, mbin, :]
            med = np.nanmedian(p, axis=0); lo, hi = np.nanpercentile(p, [16, 84], axis=0)
            ok = med > 0
            ax.fill_between(x[ok], lo[ok], hi[ok], color=cc, alpha=0.2)
            ax.plot(x[ok], med[ok], "o-", color=cc, lw=1.8, ms=4,
                    label=f"logM200 {MBINS[mbin]:.1f}-{MBINS[mbin+1]:.1f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axvspan(FIT_XMIN, FIT_XMAX, color="gray", alpha=0.08)
        ax.set_xlabel(r"$R_{\rm proj}/r_{200c}$"); ax.set_ylabel(r"absolute $\langle\tau\rangle$")
        ax.set_title(title + " -- kSZ amplitude (Sobol 16-84%)"); ax.legend(fontsize=7)
    fig.suptitle("Amplitude leg: absolute BIND tau(R) -- feedback modulates 'how much gas'", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "tau_amplitude.png", dpi=150, bbox_inches="tight")
    print(f"[d2] wrote {OUT}/tau_amplitude.png")

    # ---- Fig E: ABSOLUTE tau, BIND vs Hadzhiyska GNFW (confirmed form) ---------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = [("BGS-mass (z~0.18)", 85, 0.26, ["BGS_Ms11.0", "BGS_Ms11.25"]),
              ("ELG-mass (z~1.16)", 46, 1.17, ["ELG_all", "ELG_Ms9.5"])]
    for ax, (title, snap, zdat, samples) in zip(axes, panels):
        if snap in caches:
            c = caches[snap]; x = c["x"]
            for mbin, cc in [(0, "tab:cyan"), (1, "tab:blue")]:
                lo_m, hi_m = MBINS[mbin], MBINS[mbin + 1]
                p = c["tau"][:, mbin, :]
                med = np.nanmedian(p, axis=0); plo, phi = np.nanpercentile(p, [16, 84], axis=0)
                ok = med > 0
                ax.fill_between(x[ok], plo[ok], phi[ok], color=cc, alpha=0.18)
                ax.plot(x[ok], med[ok], "-", color=cc, lw=2,
                        label=f"BIND logM200 {lo_m:.1f}-{hi_m:.1f}")
        for s in samples:
            h = HADZ[s]; lM = HADZ_LOGM200[s]
            ty = project_tau_absolute(x, lM, zdat, h["logrho0"][0], h["alpha"][0], h["beta"][0])
            ax.plot(x, ty, "--", lw=2, label=f"{s} GNFW (logM200~{lM})")
        ax.axvspan(FIT_XMIN, FIT_XMAX, color="gray", alpha=0.08)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$R_{\rm proj}/r_{200c}$"); ax.set_ylabel(r"absolute $\tau$")
        ax.set_title(title); ax.legend(fontsize=7); ax.set_ylim(1e-7, 1e-3)
    fig.suptitle("ABSOLUTE tau: BIND vs Hadzhiyska+26 GNFW (BGS M200c placeholders; ELG below floor)", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "tau_absolute_vs_hadzhiyska.png", dpi=150, bbox_inches="tight")
    print(f"[d2] wrote {OUT}/tau_absolute_vs_hadzhiyska.png")


def _parse_list(s, cast=int):
    if "-" in s and "," not in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [cast(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduce", action="store_true")
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--reduce-mstar", dest="reduce_mstar", action="store_true")
    ap.add_argument("--consolidate-mstar", dest="consolidate_mstar", action="store_true")
    ap.add_argument("--plot-mstar", dest="plot_mstar", action="store_true")
    ap.add_argument("--snaps", type=str, default="85,46")
    ap.add_argument("--nodes", type=str, default="0-255")
    ap.add_argument("--mpi", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.reduce:
        do_reduce(_parse_list(a.snaps), _parse_list(a.nodes), use_mpi=a.mpi, force=a.force)
    if a.consolidate:
        do_consolidate(_parse_list(a.snaps), _parse_list(a.nodes))
    if a.plot:
        do_plot()
    if a.reduce_mstar:
        do_reduce_mstar(_parse_list(a.snaps), _parse_list(a.nodes), use_mpi=a.mpi, force=a.force)
    if a.consolidate_mstar:
        do_consolidate_mstar(_parse_list(a.snaps), _parse_list(a.nodes))
    if a.plot_mstar:
        do_plot_mstar()
    if not (a.reduce or a.consolidate or a.plot or a.reduce_mstar or a.consolidate_mstar or a.plot_mstar):
        ap.print_help()


if __name__ == "__main__":
    main()
