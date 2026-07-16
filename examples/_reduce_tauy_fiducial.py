"""Reduce the FIDUCIAL painted lightcone to a stacked tau(x) & y(x) profile per M200c bin,
in the exact (x, MBINS) layout of `ksz_tau_gnfw.py`'s 256-node `bind_tauy_xprof_snap{snap}.npz`.

The node cache stores only the 256 Sobol runs (no fiducial), so the §3 response-fan figure had
to draw the per-node MEDIAN as its reference line. This adds the true TNG FIDUCIAL profile (the
painted `bind_lightcone_tng/snap_{snap}` run) so §3 can show the fiducial as the reference, exactly
as §6/§6b/§6d already do. Fast: the fiducial is a single run (~4 slabs/snapshot).

    python examples/_reduce_tauy_fiducial.py            # snaps 85 (BGS) + 46 (ELG)
"""
import glob
from pathlib import Path
import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
FID = CEPH / "bind_lightcone_tng"
OUT = CEPH / "bind_science/ksz_confront"

PIX_MPCH = 6.25 / 128.0
SIGMA_T = 6.6524e-25; M_P = 1.6726e-24; MSUN_G = 1.989e33; MPC_CM = 3.0857e24
X_E_PER_MASS = 0.88 / M_P; H = 0.6774
MBINS = np.array([13.0, 13.4, 13.8, 14.2, 15.0])
X_EDGES = np.geomspace(0.08, 3.0, 18); XC = np.sqrt(X_EDGES[1:] * X_EDGES[:-1])
BG_X = (2.5, 3.0)


def tau_from_gas(gas_patch, a):
    area_cm2 = (PIX_MPCH * a / H * MPC_CM) ** 2
    return gas_patch.astype(np.float64) * (SIGMA_T * X_E_PER_MASS * MSUN_G / (H * area_cm2))


def stack_fiducial(snapdir, a):
    """Same bg-subtracted radial stack as ksz_tau_gnfw.stack_node_snap, fiducial run."""
    files = sorted(glob.glob(str(snapdir) + "/composite_slab*.npz"))
    nb, nr = len(MBINS) - 1, len(X_EDGES) - 1
    st = np.zeros((nb, nr)); sy = np.zeros((nb, nr)); sn = np.zeros((nb, nr)); cnt = np.zeros(nb, int)
    for f in files:
        d = np.load(f)
        if int(d["n_halos"]) == 0:
            continue
        gas = d["generated_patches"][:, 1]; ymap = d["thermo_patches"][:, 0]
        M, r200 = d["halo_masses"], d["halo_r200"]
        P = gas.shape[-1]; cen = P // 2
        yy, xx = np.mgrid[0:P, 0:P]; rr = np.hypot(xx - cen, yy - cen) * PIX_MPCH
        lM = np.log10(M)
        for h in range(len(M)):
            b = int(np.digitize(lM[h], MBINS)) - 1
            if b < 0 or b >= nb:
                continue
            t = tau_from_gas(gas[h], a); yv = ymap[h].astype(np.float64)
            x = rr / r200[h]; mbg = (x > BG_X[0]) & (x < BG_X[1])
            t = t - (t[mbg].mean() if mbg.any() else 0.0)
            yv = yv - (yv[mbg].mean() if mbg.any() else 0.0)
            idx = np.digitize(x.ravel(), X_EDGES) - 1
            ok = (idx >= 0) & (idx < nr)
            np.add.at(st[b], idx[ok], t.ravel()[ok])
            np.add.at(sy[b], idx[ok], yv.ravel()[ok])
            np.add.at(sn[b], idx[ok], 1.0); cnt[b] += 1
    den = np.where(sn > 0, sn, 1.0)
    return np.where(sn > 0, st / den, np.nan), np.where(sn > 0, sy / den, np.nan), cnt


def main():
    for snap in (85, 46):
        snapdir = FID / f"snap_{snap:03d}"
        if not snapdir.exists():
            print(f"snap {snap}: {snapdir} missing, skip"); continue
        a = float(np.load(OUT / f"bind_tauy_xprof_snap{snap:03d}.npz")["a"])  # match node-cache a
        tau, y, cnt = stack_fiducial(snapdir, a)
        out = OUT / f"bind_tauy_fiducial_snap{snap:03d}.npz"
        np.savez(out, x=XC, mbins=MBINS, a=a, tau=tau, y=y, cnts=cnt)
        print(f"wrote {out}  (a={a:.3f}, counts/bin={cnt.tolist()})")


if __name__ == "__main__":
    main()
