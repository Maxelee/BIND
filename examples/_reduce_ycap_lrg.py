"""BIND Compton-y CAP at the DESI LRG mass+redshift -- the PRESSURE leg (tSZ).

Confronts the ACT x DESI photometric-LRG tSZ y-CAP (arXiv:2502.08850, Zenodo 14706729,
fig3 = fiducial cumulative-CAP Compton-y). LRG: logM200~13.3, z̄~0.47-0.92; ⚠ the Zenodo
per-z columns are duplicated (export bug) so we confront ONE representative profile at a
single LRG snapshot. BIND y from the painted thermo patches (y = thermo_patches[:,0]),
ACT-beam-convolved (~1.6'), compensated-aperture (disk - equal-area ring). With f̃_gas
(density) + κ (mass), y (pressure ∝ f̃_gas·M·T_e) gives the TEMPERATURE: pressure ~3× =
gas 2× × mass 1.3× × T_e ~1.1× -> "too much gas, not too hot".

    python examples/_reduce_ycap_lrg.py            # snap 067 (z=0.50), fiducial + 24 nodes
"""
import glob, time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

CEPH = Path("/mnt/home/mlee1/ceph")
SNAP = "067"; ZL = 0.503                  # LRG redshift (snap 67); mean LRG z̄~0.47-0.92
FID = CEPH / f"bind_lightcone_tng/snap_{SNAP}"
SB35 = CEPH / "bind_sb35/runs"
OUT = CEPH / "bind_science/ksz_confront/ycap_lrg_snap067.npz"
h = 0.6774; Om, OL = 0.3089, 0.6911; C_KMS = 299792.458; H0 = 100 * h; ARCMIN = 180 * 60 / np.pi
PIX = 0.048828125
# DESI LRG host mass: log10(M200c/[Msun/h]) ~ 13.18 from ACT CMB-lensing (Sailer+24); the OLD band
# [13.2,13.5] Msun/h was ~0.2 dex too high (an Msun-vs-Msun/h slip) -> Y inflated ~2.6x (Y~M^5/3).
MLO, MHI = 13.0, 13.35                    # LRG host band (brackets the lensing mean 13.18, Msun/h)
M_LENS = 13.18                            # lensing mean host mass; y-CAP is mass-systematic-dominated
BEAM = 1.6                                # ACT DR6 FWHM arcmin
RAP = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.125, 4.75, 5.375, 6.0])   # data fig3 apertures
NODES = list(range(256))                  # full Sobol set (the paper shows all 256 run profiles)


def _apix():
    def Ez(z): return np.sqrt(Om * (1 + z) ** 3 + OL)
    def DC(z, n=4000): zz = np.linspace(0, z, n); return (C_KMS / H0) * np.trapezoid(1 / Ez(zz), zz)
    DA = DC(ZL) / (1 + ZL)
    return (PIX / h / (1 + ZL)) / DA * ARCMIN


def ycap(snap_dir, apix, mlo=MLO, mhi=MHI):
    """MEAN-stacked (matches the data stack) y-CAP for halos in [mlo,mhi] log Msun/h."""
    yy, hm = [], []
    for f in sorted(glob.glob(str(snap_dir) + "/composite_slab*.npz")):
        d = np.load(f); yy.append(d["thermo_patches"][:, 0]); hm.append(d["halo_masses"])
    if not yy:
        return None
    Y = np.concatenate(yy); hm = np.concatenate(hm)
    sel = (np.log10(hm) > mlo) & (np.log10(hm) < mhi)
    if sel.sum() < 20:
        return None
    ymap = gaussian_filter(np.nanmean(Y[sel], 0), BEAM / 2.355 / apix)     # MEAN stack + ACT beam
    P = ymap.shape[-1]; rr = np.hypot(*(np.mgrid[0:P, 0:P] - P / 2 + 0.5)) * apix
    out = np.full(len(RAP), np.nan)
    for j, td in enumerate(RAP):
        disk = rr < td; ring = (rr > td) & (rr < np.sqrt(2) * td)
        out[j] = ymap[disk].sum() * apix ** 2 - ymap[ring].sum() * apix ** 2 * (disk.sum() / max(ring.sum(), 1))
    return out


_APIX = None      # set in main() before the Pool; forked workers inherit (Linux fork)


def _node_task(n):
    """Worker: one Sobol node -> (n, y-CAP profile | None). Uses the global _APIX."""
    d = SB35 / f"run_{n:04d}/snap_{SNAP}"
    if not d.exists():
        return n, None
    try:
        return n, ycap(d, _APIX)
    except Exception:
        return n, None


def main():
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=1, help="parallelise the 256-node loop across this many cores")
    a = ap.parse_args()
    t0 = time.time(); apix = _apix()
    print(f"snap{SNAP} z={ZL}, {apix:.3f} arcmin/pix, ACT beam {BEAM}'; LRG band [{MLO},{MHI}] Msun/h (lens mean {M_LENS}); nproc={a.nproc}", flush=True)
    fid = ycap(FID, apix); print(f"fiducial y-CAP: {np.array2string(fid,formatter={'float':lambda x:f'{x:.2e}'})}  [{time.time()-t0:.0f}s]", flush=True)
    # mass-systematic band: shift the host band by +-0.1 dex (Y~M^5/3 -> the dominant uncertainty)
    fid_lo = ycap(FID, apix, MLO - 0.1, MHI - 0.1); fid_hi = ycap(FID, apix, MLO + 0.1, MHI + 0.1)
    print(f"mass-syst (band -+0.1 dex) y-CAP@3.5': {np.interp(3.5,RAP,fid_lo)*1e6:.2f} / {np.interp(3.5,RAP,fid)*1e6:.2f} / {np.interp(3.5,RAP,fid_hi)*1e6:.2f} e-6", flush=True)
    global _APIX; _APIX = apix                                  # share with forked workers
    if a.nproc > 1:
        with Pool(a.nproc) as pool:
            results = pool.map(_node_task, NODES, chunksize=1)
    else:
        results = [_node_task(n) for n in NODES]
    ok, sb = [], []
    for n, r in results:
        if r is not None:
            ok.append(n); sb.append(r)
    sb = np.array(sb)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, R=RAP, fiducial=fid, fid_msys_lo=fid_lo, fid_msys_hi=fid_hi, sb35=sb,
             node_ids=np.array(ok), z=ZL, logM=M_LENS, mass_band=np.array([MLO, MHI]))
    print(f"\nwrote {OUT}  (fiducial + {len(ok)} nodes, mean-stack, LRG mass-corrected)  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
