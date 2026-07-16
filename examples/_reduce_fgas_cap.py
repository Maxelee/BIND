"""Reduce per-halo painted patches -> CAP-ratio f~gas(theta_d/r200), the EXACT observable
Ried Guachalla+25 (DESI x ACT, arXiv:2604.19745) report.

Their gas fraction is a COMPENSATED-APERTURE-PHOTOMETRY ratio, not a cumulative or 3D
fraction: from sec III.2.1, they compare "the value of the CAP filter for the gas at the
virial radius with that of the CAP filter for the matter". The CAP filter at aperture
theta_d is +1 inside the disk (r<theta_d) and -1 in an EQUAL-AREA ring (theta_d<r<sqrt2
theta_d). So the apples-to-apples BIND prediction is

    CAP_X(theta_d) = sum_{disk} X  -  (A_disk/A_ring) * sum_{ring} X
    f~gas_CAP(theta_d) = CAP_gas(theta_d) / CAP_tot(theta_d) / (Omega_b/Omega_m),

with tot = DM+Gas+Stars (the CMB-lensing matter leg). This CANCELS the ~50 Mpc/h
line-of-sight that the cumulative aperture (`_reduce_fgas_radial.py`) wrongly keeps -- the
cumulative inflates f~gas toward cosmic (0.86) and manufactures a spurious "0/256" tension;
the CAP ratio is lower (~0.6 fiducial) and is what the data actually measure.

Same M*-matched selection + output layout as `_reduce_fgas_radial.py` (drop-in for the
notebook): central M* = Stars within ~1px, cuts M*>11.0/11.25, FIDUCIAL + all 256 nodes.

    python examples/_reduce_fgas_cap.py            # fiducial + 256 nodes (~20 min I/O)
"""
import glob, time
from pathlib import Path
import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
FID = CEPH / "bind_lightcone_tng/snap_085"
SB35 = CEPH / "bind_sb35/runs"
OUT = CEPH / "bind_science/ksz_confront/fgas_cap_mstar_snap085.npz"
F_B = 0.0490 / 0.3089
PIX = 0.048828125                         # Mpc/h native (128*PIX = 6.25)
MSTAR_AP = 1.0                            # central-M* aperture in px (~50 kpc/h)
MSTAR_CUTS = [11.0, 11.25]               # log10 M*  (DESI BGS stellar-mass cuts)
XB = np.linspace(0.3, 3.0, 18)            # CAP aperture theta_d / r200  (data spans ~0.3-3.6)
SQ2 = np.sqrt(2.0)
NODES = list(range(256))


def cap_profiles(snapdir):
    """Per-M*-cut median CAP-ratio f~gas(theta_d/r200) + mean logM200, for one snapshot."""
    gp, hm, hr = [], [], []
    for f in sorted(glob.glob(str(snapdir) + "/composite_slab*.npz")):
        d = np.load(f, allow_pickle=True)
        gp.append(d["generated_patches"]); hm.append(d["halo_masses"]); hr.append(d["halo_r200"])
    if not gp:
        return None, None
    gp = np.concatenate(gp); hm = np.concatenate(hm); hr = np.concatenate(hr)
    npx = gp.shape[-1]
    rr = np.hypot(*(np.mgrid[0:npx, 0:npx] - npx / 2 + 0.5))
    order = np.argsort(rr.ravel()); rs = rr.ravel()[order]
    mstar = (gp[:, 2] * (rr < MSTAR_AP)).sum((1, 2))          # painted central stellar mass
    out = np.full((len(MSTAR_CUTS), len(XB)), np.nan); lm = np.full(len(MSTAR_CUTS), np.nan)
    for k, cut in enumerate(MSTAR_CUTS):
        sel = np.log10(mstar + 1) > cut
        if sel.sum() == 0:
            continue
        lm[k] = np.log10(hm[sel]).mean()
        prof = np.full((int(sel.sum()), len(XB)), np.nan)
        for i, idx in enumerate(np.where(sel)[0]):
            DM, Gas, St = gp[idx]; tot = DM + Gas + St; r200px = hr[idx] / PIX
            gcs = np.cumsum(Gas.ravel()[order]); tcs = np.cumsum(tot.ravel()[order])
            for j, xd in enumerate(XB):
                Rpx = xd * r200px
                i_d = int(np.searchsorted(rs, Rpx)); i_r = int(np.searchsorted(rs, SQ2 * Rpx))
                if i_d < 2 or i_r <= i_d or i_r > len(rs) - 1:      # need disk + a valid equal-area ring inside the patch
                    continue
                nd, nr = i_d, i_r - i_d; w = nd / nr                # ring area-match factor (~1)
                capg = gcs[i_d - 1] - (gcs[i_r - 1] - gcs[i_d - 1]) * w
                capt = tcs[i_d - 1] - (tcs[i_r - 1] - tcs[i_d - 1]) * w
                if capt > 0:
                    prof[i, j] = (capg / capt) / F_B
        out[k] = np.nanmedian(prof, 0)
    return out, lm


def main():
    t0 = time.time()
    fid, lm = cap_profiles(FID)
    print(f"fiducial CAP f~gas(r200) M*>11.0={np.interp(1,XB,fid[0]):.2f} M*>11.25={np.interp(1,XB,fid[1]):.2f} "
          f"(logM200 {lm[0]:.2f}/{lm[1]:.2f})  [{time.time()-t0:.0f}s]", flush=True)
    sb, ok = [], []
    for n in NODES:
        p, _ = cap_profiles(SB35 / f"run_{n:04d}/snap_085")
        if p is not None:
            sb.append(p); ok.append(n)
            if n % 16 == 0:
                print(f"  run_{n:04d}: CAP f~gas(r200) M*>11.25={np.interp(1,XB,p[1]):.2f}  [{time.time()-t0:.0f}s]", flush=True)
    sb = np.array(sb)                                          # (Nnodes, 2 cuts, 18)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, xb=XB, fiducial=fid, sb35=sb, node_ids=np.array(ok), logM200=lm,
             cuts=np.array(MSTAR_CUTS), pix=PIX, mstar_ap_px=MSTAR_AP)
    print(f"\nwrote {OUT}  (fiducial + {len(ok)} nodes, 2 M* cuts, CAP ratio)  [{time.time()-t0:.0f}s]")
    for k, cut in enumerate(MSTAR_CUTS):
        lo, hi = np.nanpercentile(sb[:, k], [16, 84], 0); mn = np.nanmin(sb[:, k], 0)
        print(f"M*>{cut} CAP f~gas(r200): fid {np.interp(1,XB,fid[k]):.2f}  SB[16-84] "
              f"{np.interp(1,XB,lo):.2f}-{np.interp(1,XB,hi):.2f}  min(256) {np.interp(1,XB,mn):.2f}")


if __name__ == "__main__":
    main()
