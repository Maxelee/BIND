"""BIND CMB-lensing kappa(theta) stacked on BGS halos, per DESI M* cut -- the MASS anchor.

kappa = Sigma_tot / Sigma_crit (CMB lensing, z_s=1100). The painted total-mass patch
(DM+Gas+Stars) gives the projected surface density; we (a) build the kappa map, (b) smooth
to the ACT DR6 lensing L<3000 cut (Gaussian sigma_theta ~ 1/L), (c) background-subtract the
stacked profile (outer annulus, mimicking the random-point subtraction), (d) stack per M*
cut (BIND painted central M*, same selection as the f~gas). Confronts the DESI BGS x ACT
kappa(theta) (Part II Fig.7, desact_zenodo/Fig7_bgs_y3_logm{cut}.npz). kappa ~ TOTAL mass,
so a match validates BIND's halo masses -> the f~gas tension is GAS, not mass.

    python examples/_reduce_kappa_bgs.py            # fiducial + 24 nodes
"""
import glob, time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

CEPH = Path("/mnt/home/mlee1/ceph")
FID = CEPH / "bind_lightcone_tng/snap_085"
SB35 = CEPH / "bind_sb35/runs"
OUT = CEPH / "bind_science/ksz_confront/kappa_bgs_snap085.npz"
h = 0.6774; Om, OL = 0.3089, 0.6911; C_KMS = 299792.458; H0 = 100 * h; ARCMIN = 180 * 60 / np.pi
G = 4.301e-9                              # Mpc Msun^-1 (km/s)^2
PIX = 0.048828125                        # Mpc/h comoving
ZL, ZS = 0.18, 1100.0                    # BGS lens z (snap85), CMB source
CUTS = ["10.00", "10.50", "11.00", "11.25"]   # DESI BGS M* cuts with kappa data
THB = np.array([0.75, 2.25, 3.75, 5.25])      # data theta bins (arcmin), 1.5' wide
NODES = list(range(0, 256, 11))


def _cosmo():
    def Ez(z): return np.sqrt(Om * (1 + z) ** 3 + OL)
    def DC(z, n=4000): zz = np.linspace(0, z, n); return (C_KMS / H0) * np.trapezoid(1 / Ez(zz), zz)
    DA = lambda z: DC(z) / (1 + z)
    DAls = (DC(ZS) - DC(ZL)) / (1 + ZS)
    Sig_crit = C_KMS ** 2 / (4 * np.pi * G) * DA(ZS) / (DA(ZL) * DAls)   # Msun/Mpc^2 physical
    pix_phys = PIX / h / (1 + ZL)                                        # Mpc proper / pixel
    arcmin_per_pix = pix_phys / DA(ZL) * ARCMIN
    return Sig_crit, pix_phys, arcmin_per_pix


def kappa_profiles(snap_dir, Sig_crit, parea, apix):
    gp, hm = [], []
    for f in sorted(glob.glob(str(snap_dir) + "/composite_slab*.npz")):
        d = np.load(f); gp.append(d["generated_patches"]); hm.append(d["halo_masses"])
    if not gp:
        return None
    gp = np.concatenate(gp); hm = np.concatenate(hm); P = gp.shape[-1]
    rr = np.hypot(*(np.mgrid[0:P, 0:P] - P / 2 + 0.5))
    th = rr * apix                                                       # arcmin per pixel
    mstar = (gp[:, 2] * (rr < 1.0)).sum((1, 2))
    sig_L = (1.0 / ZS if False else 1.0)                                # placeholder
    sig_px = (1.0 / 3000.0 * ARCMIN) / apix                             # L<3000 -> sigma_theta~1/L rad
    bg = (th > 6.0) & (th < 8.0)                                        # outer annulus = background
    out = np.full((len(CUTS), len(THB)), np.nan)
    for c, cut in enumerate(CUTS):
        sel = np.log10(mstar + 1) > float(cut)
        if sel.sum() < 20:
            continue
        kap = (gp[sel].sum(1) / h) / parea / Sig_crit                  # kappa map per halo (Msun/h->Msun)
        kap = gaussian_filter(kap, (0, sig_px, sig_px))                # L<3000 smoothing (per-halo)
        prof = np.full((sel.sum(), len(THB)), np.nan)
        for i in range(sel.sum()):
            k = kap[i]
            bgv = k[bg].mean() if bg.sum() else 0.0
            for j, t in enumerate(THB):
                m = (th > t - 0.75) & (th < t + 0.75)
                prof[i, j] = k[m].mean() - bgv                          # bg-subtracted local kappa
        out[c] = np.nanmedian(prof, 0)
    return out


def main():
    t0 = time.time()
    Sig_crit, pix_phys, apix = _cosmo()
    parea = pix_phys ** 2
    print(f"Sigma_crit={Sig_crit:.3e} Msun/Mpc^2 ; {apix:.3f} arcmin/pix  [{time.time()-t0:.0f}s]", flush=True)
    fid = kappa_profiles(FID, Sig_crit, parea, apix)
    print("fiducial kappa(theta) per cut:\n", np.round(fid, 4), flush=True)
    sb, ok = [], []
    for n in NODES:
        d = SB35 / f"run_{n:04d}/snap_085"
        if not d.exists():
            continue
        k = kappa_profiles(d, Sig_crit, parea, apix)
        if k is not None:
            sb.append(k); ok.append(n)
            if n % 33 == 0:
                print(f"  run_{n:04d} [{time.time()-t0:.0f}s]", flush=True)
    sb = np.array(sb)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, theta=THB, cuts=np.array([float(c) for c in CUTS]), fiducial=fid, sb35=sb, node_ids=np.array(ok))
    print(f"\nwrote {OUT}  (fiducial + {len(ok)} nodes)  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
