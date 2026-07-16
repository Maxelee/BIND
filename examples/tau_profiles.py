"""Per-halo Compton-depth tau profiles from composite gas patches (BIND vs truth).

The kSZ observable surveys report after velocity reconstruction is the tau
profile (Schaan+21; Ried Guachalla+25 DESI x ACT) — no velocities needed:
``tau(theta) = sigma_T int n_e dl = sigma_T x_e Sigma_gas / m_p`` with
``x_e = X + Y_He/2 = 0.88`` electrons per nucleon mass for ionized H+He.
This is the f_gas (density-weighted) leg of the (kappa, tau, y) thermodynamic
decomposition — see docs/wl_tsz_plan.md §2.

Reads ``composite_slab*.npz`` per-halo ``generated_patches`` (ch 1 = gas mass
per pixel, Msun/h) for BIND and the truth extraction, stacks tau(r) in mass
bins, and writes a comparison npz + png.

    python examples/tau_profiles.py --snap 96
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SIGMA_T = 6.6524e-25            # cm^2
M_P = 1.6726e-24                # g
MSUN_G = 1.989e33
MPC_CM = 3.0857e24
X_E_PER_MASS = 0.88 / M_P       # free electrons per gram (X=0.76, Y_He=0.24)
H = 0.6774
PIX_MPCH = 6.25 / 128.0         # comoving patch pixel, Mpc/h


def _scale_factor(snap_dir: Path) -> float:
    """Scale factor from the stage-1 manifest if present, else a=1."""
    for mf in list(snap_dir.glob("stage1/*.json")) + [snap_dir / "truth_manifest.json"]:
        if mf.exists():
            d = json.loads(mf.read_text())
            for k in ("scale_factor", "a"):
                if isinstance(d, dict) and k in d:
                    return float(d[k])
    return 1.0


def load_halos(snap_dir: Path):
    """Concatenate per-halo gas patches + masses + r200 over the slabs."""
    gas, M, r200 = [], [], []
    for p in sorted(snap_dir.glob("composite_slab*.npz")):
        d = np.load(p)
        if "generated_patches" not in d.files or int(d["n_halos"]) == 0:
            continue
        gas.append(d["generated_patches"][:, 1])          # (n, P, P) Msun/h
        M.append(d["halo_masses"]); r200.append(d["halo_r200"])
    return np.concatenate(gas), np.concatenate(M), np.concatenate(r200)


def tau_patches(gas_patches: np.ndarray, a: float) -> np.ndarray:
    """Gas-mass patches [Msun/h] -> Compton depth tau per pixel."""
    area_cm2 = (PIX_MPCH * a / H * MPC_CM) ** 2           # physical pixel area
    # scalar factor first: float32 patches * 2e33 would overflow
    return gas_patches.astype(np.float64) * (SIGMA_T * X_E_PER_MASS * MSUN_G / (H * area_cm2))


def stack_profiles(tau: np.ndarray, M: np.ndarray, r200: np.ndarray,
                   mass_bins: np.ndarray, r_edges_r500: np.ndarray,
                   r500_over_r200: float = 0.659):
    """Mean tau(r/r500) per mass bin + aperture tau within 1x/2x r500."""
    P = tau.shape[-1]
    cen = P // 2
    yy, xx = np.mgrid[0:P, 0:P]
    rr = np.hypot(xx - cen, yy - cen) * PIX_MPCH          # Mpc/h comoving
    n_b, n_r = len(mass_bins) - 1, len(r_edges_r500) - 1
    prof = np.full((n_b, n_r), np.nan)
    ap1 = np.full(n_b, np.nan); ap2 = np.full(n_b, np.nan); cnt = np.zeros(n_b, int)
    for b in range(n_b):
        sel = (M >= mass_bins[b]) & (M < mass_bins[b + 1])
        cnt[b] = sel.sum()
        if cnt[b] < 10:
            continue
        s_prof = np.zeros(n_r); s_n = np.zeros(n_r)
        s1 = s2 = 0.0
        for t, r5 in zip(tau[sel], r500_over_r200 * r200[sel]):
            idx = np.digitize(rr.ravel() / r5, r_edges_r500) - 1
            ok = (idx >= 0) & (idx < n_r)
            np.add.at(s_prof, idx[ok], t.ravel()[ok])
            np.add.at(s_n, idx[ok], 1.0)
            s1 += t[rr <= r5].mean()                       # mean tau in aperture
            s2 += t[rr <= 2 * r5].mean()
        prof[b] = s_prof / np.where(s_n > 0, s_n, 1.0)
        ap1[b], ap2[b] = s1 / cnt[b], s2 / cnt[b]
    return prof, ap1, ap2, cnt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bind_root", type=Path,
                   default=Path("/mnt/home/mlee1/ceph/bind_lightcone_tng"))
    p.add_argument("--truth_root", type=Path,
                   default=Path("/mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000"))
    p.add_argument("--snap", type=int, default=96)
    p.add_argument("--out", type=Path,
                   default=Path("/mnt/home/mlee1/ceph/bind_science/tau_profiles"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd_b = args.bind_root / f"snap_{args.snap:03d}"
    sd_t = args.truth_root / f"snap_{args.snap:03d}"
    a = _scale_factor(sd_b)
    print(f"[tau] snap {args.snap}: a={a:.4f}")

    mass_bins = np.array([1e13, 3e13, 1e14, 3e14, 1e15])
    r_edges = np.geomspace(0.1, 6.0, 16)                  # r / r500
    r_c = np.sqrt(r_edges[1:] * r_edges[:-1])

    res = {}
    for tag, sd in (("bind", sd_b), ("truth", sd_t)):
        gas, M, r200 = load_halos(sd)
        tau = tau_patches(gas, a)
        prof, ap1, ap2, cnt = stack_profiles(tau, M, r200, mass_bins, r_edges)
        res[tag] = dict(prof=prof, ap1=ap1, ap2=ap2, cnt=cnt)
        print(f"[tau] {tag}: {len(M)} halos")

    print(f"\n{'mass bin':>12s} {'N':>6s} {'tau(<r500) B/T':>15s} {'tau(<2r500) B/T':>16s}")
    for b in range(len(mass_bins) - 1):
        r1 = res["bind"]["ap1"][b] / res["truth"]["ap1"][b]
        r2 = res["bind"]["ap2"][b] / res["truth"]["ap2"][b]
        print(f"{np.sqrt(mass_bins[b]*mass_bins[b+1]):12.2e} {res['bind']['cnt'][b]:6d} "
              f"{r1:15.3f} {r2:16.3f}")

    np.savez(args.out / f"tau_profiles_snap{args.snap:03d}.npz",
             r_r500=r_c, mass_bins=mass_bins, scale_factor=a,
             **{f"{t}_{k}": v for t, d in res.items() for k, v in d.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for b in range(len(mass_bins) - 1):
        if res["bind"]["cnt"][b] < 10:
            continue
        l, = ax[0].loglog(r_c, res["bind"]["prof"][b],
                          label=f"{mass_bins[b]:.0e}-{mass_bins[b+1]:.0e}")
        ax[0].loglog(r_c, res["truth"]["prof"][b], "--", color=l.get_color())
        ax[1].semilogx(r_c, res["bind"]["prof"][b] / res["truth"]["prof"][b],
                       color=l.get_color())
    ax[0].set_xlabel(r"$r/r_{500c}$"); ax[0].set_ylabel(r"$\tau$")
    ax[0].set_title("stacked tau profile (BIND solid, truth dashed)"); ax[0].legend(fontsize=7)
    ax[1].axhline(1, c="gray", ls=":"); ax[1].axhspan(.95, 1.05, color="gray", alpha=.15)
    ax[1].set_xlabel(r"$r/r_{500c}$"); ax[1].set_ylabel("BIND / truth"); ax[1].set_ylim(.7, 1.3)
    fig.tight_layout()
    fig.savefig(args.out / f"tau_profiles_snap{args.snap:03d}.png", dpi=150)
    print(f"[tau] wrote {args.out}/tau_profiles_snap{args.snap:03d}.{{npz,png}}")


if __name__ == "__main__":
    main()
