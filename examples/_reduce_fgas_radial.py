"""Reduce per-halo painted patches -> projected f~gas(R/r200), matched to the DESI
stellar-mass cuts (apples-to-apples with the kSZ data).

The DESI x ACT kSZ stacks select galaxies by STELLAR mass (M*>11.0, M*>11.25), not
halo mass. We reproduce that selection on BIND's painted patches and compute the
cumulative projected gas fraction directly:

    central M*  = sum of the Stars patch within ~50 kpc/h (1 px) of the centre,
    f~gas(<R)   = [ sum_{<R} M_gas^pix / sum_{<R} M_tot^pix ] / (Omega_b/Omega_m),

for each cut, for the FIDUCIAL TNG run (bind_lightcone_tng) and ALL 256 SB35 Sobol
nodes (the feedback band / extremes). Patches are 128px @ native 0.048828125 Mpc/h
(128*pix = 6.25 Mpc/h); r200 in pixels = halo_r200 / pix. The M* cuts recover mean
logM200 = 13.60 / 13.82 (cf. the mstar-npz / CAP-panel 13.56 / 13.81), so BIND and the
data use the identical selection. Output cache feeds the paper notebook f~gas panel.

    python examples/_reduce_fgas_radial.py            # fiducial + 256 nodes (~20 min I/O)
"""
import glob, time
from pathlib import Path
import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
FID = CEPH / "bind_lightcone_tng/snap_085"
SB35 = CEPH / "bind_sb35/runs"
OUT = CEPH / "bind_science/ksz_confront/fgas_radial_mstar_snap085.npz"
F_B = 0.0490 / 0.3089
PIX = 0.048828125                         # Mpc/h native (128*PIX = 6.25)
MSTAR_AP = 1.0                            # central-M* aperture in px (~50 kpc/h)
MSTAR_CUTS = [11.0, 11.25]               # log10 M*  (DESI BGS stellar-mass cuts)
XB = np.linspace(0.15, 3.0, 18)           # R/r200 grid
NODES = list(range(256))                  # full Sobol set


def profiles(snapdir):
    """Per-M*-cut median projected f~gas(R/r200) + mean logM200, for one snapshot."""
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
            gcum = np.cumsum(Gas.ravel()[order]); tcum = np.cumsum(tot.ravel()[order])
            prof[i] = np.interp(XB, rs / r200px, (gcum / tcum) / F_B)
        out[k] = np.nanmedian(prof, 0)
    return out, lm


def main():
    t0 = time.time()
    fid, lm = profiles(FID)
    print(f"fiducial: f~gas(r200) M*>11.0={np.interp(1,XB,fid[0]):.2f} M*>11.25={np.interp(1,XB,fid[1]):.2f} "
          f"(logM200 {lm[0]:.2f}/{lm[1]:.2f})  [{time.time()-t0:.0f}s]", flush=True)
    sb, ok = [], []
    for n in NODES:
        p, _ = profiles(SB35 / f"run_{n:04d}/snap_085")
        if p is not None:
            sb.append(p); ok.append(n)
            if n % 16 == 0:
                print(f"  run_{n:04d}: f~gas(r200) M*>11.25={np.interp(1,XB,p[1]):.2f}  [{time.time()-t0:.0f}s]", flush=True)
    sb = np.array(sb)                                          # (Nnodes, 2 cuts, 18)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, xb=XB, fiducial=fid, sb35=sb, node_ids=np.array(ok), logM200=lm,
             cuts=np.array(MSTAR_CUTS), pix=PIX, mstar_ap_px=MSTAR_AP)
    print(f"\nwrote {OUT}  (fiducial + {len(ok)} nodes, 2 M* cuts)  [{time.time()-t0:.0f}s]")
    for k, cut in enumerate(MSTAR_CUTS):
        lo, hi = np.nanpercentile(sb[:, k], [16, 84], 0); mn = np.nanmin(sb[:, k], 0)
        print(f"M*>{cut} f~gas(r200): fid {np.interp(1,XB,fid[k]):.2f}  SB[16-84] "
              f"{np.interp(1,XB,lo):.2f}-{np.interp(1,XB,hi):.2f}  min(256) {np.interp(1,XB,mn):.2f}")


if __name__ == "__main__":
    main()
