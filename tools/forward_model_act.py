#!/usr/bin/env python
"""ACT-tuned forward-model reduction for the field-vs-profile demonstration, with
a CAMELS hydro-truth observation.

Observable = Compensated Aperture Photometry (CAP), the ACT x CMASS estimator of
Schaan et al. 2021 / Amodeo et al. 2021:
    CAP(theta) = <map>_{disk r<theta} - <map>_{ring theta<r<sqrt(2) theta}
on the tSZ map (Compton-Y) and the kSZ map (projected gas / electron density).
The instrument is the ACT f150 beam (1.6' FWHM) at the CMASS effective redshift
z=0.55, plus galaxy mis-centering; CAP nulls the uniform background by construction.

For each source we build, per object, TWO versions of every CAP observable:
  * PROFILE version  -- the clean field is azimuthally symmetrised (centred,
    axisymmetric) and beam-convolved, then CAP'd.  This is the steelmanned
    profile-emulator prediction: it KNOWS the beam, but cannot know the per-halo
    2D morphology or the mis-centering.
  * FIELD version    -- the 2D field is mis-centered and beam-convolved, then
    CAP'd.  Only a field-level emulator can produce this.
The two differ ONLY by the irreducibly-2D operations (mis-centering + asphericity).

Sources:
  --source bind   : 256 BIND Sobol feedback designs (the training set / the
                    256-design leave-one-out demonstration).  Output per design:
                    mass-function-weighted, flux-selected stacked CAP (profile and
                    field) for tSZ and kSZ, and the parent f_b.
  --source truth  : the 26-sim CV hydro TRUTH (one feedback point) -> per-halo CAP
                    (profile and field), per-halo f_b, mass, flux, so the notebook
                    can stack with the same pipeline and bootstrap over halos.

Output -> <ceph>/sobol_ss_cv/forward_model_act_{bind,truth}.npz
Run:
    python tools/forward_model_act.py --source bind
    python tools/forward_model_act.py --source truth
Smoke:  --end 3 (bind) / --nsim 2 (truth) with --out _smoke.npz
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.cosmology import FlatLambdaCDM

S = Path('/mnt/home/mlee1/ceph/sobol_ss_cv')
CV = Path('/mnt/home/mlee1/ceph/fm_testsuite/CV')

# ---- geometry / channels ---------------------------------------------------
PIX_KPC = 48.828125
H = 128
CEN = H / 2 - 0.5
CH = {'DM': 0, 'Gas': 1, 'Stars': 2, 'Y': 3, 'T': 4, 'S': 5, 'P': 6}   # BIND maps

# ---- ACT instrument + cosmology -------------------------------------------
Z_OBS = 0.55                                          # CMASS effective redshift
COSMO = FlatLambdaCDM(H0=67.11, Om0=0.3049, Ob0=0.0479)   # CAMELS fiducial
H_LITTLE = COSMO.H0.value / 100.0
KPC_H_PER_ARCMIN = float(COSMO.kpc_comoving_per_arcmin(Z_OBS).value) * H_LITTLE
BEAM_FWHM_ARCMIN = 1.6                                # ACT f150
SIGMA_MC_KPC = 150.0                                 # galaxy mis-centering (1-sigma)
CAP_THETA_ARCMIN = np.arange(1.0, 5.01, 0.5)         # ACT CAP apertures (9)
FLUX_SCATTER = 0.30                                  # fractional flux measurement scatter
SEL_Q = 0.30                                         # global flux limit quantile
MF_SLOPE = 1.0                                       # target dn/dlnM ~ M^-slope
SEED = 20250607

EDGES = [13.5, 14.0]
# truth file layout
SNAP = 'snap_090'; MTAG = 'mass_threshold_1p000e13'; NPIX = 1024; BOX = 50.0; HC = 64


def _act_kernels():
    """Precompute radius grid, symmetrisation index, and CAP disk/ring masks."""
    yy, xx = np.mgrid[0:H, 0:H]
    rr_pix = np.hypot(yy - CEN, xx - CEN)
    rr_kpc = rr_pix * PIX_KPC
    rint = np.round(rr_pix).astype(int)                 # symmetrisation bins
    sig_beam = BEAM_FWHM_ARCMIN / 2.35482 * KPC_H_PER_ARCMIN / PIX_KPC
    sig_mc = SIGMA_MC_KPC / PIX_KPC
    theta_pix = CAP_THETA_ARCMIN * KPC_H_PER_ARCMIN / PIX_KPC
    disks = [(rr_pix <= t) for t in theta_pix]
    rings = [(rr_pix > t) & (rr_pix <= np.sqrt(2) * t) for t in theta_pix]
    return dict(rr_kpc=rr_kpc, rint=rint, sig_beam=sig_beam, sig_mc=sig_mc,
                theta_pix=theta_pix, disks=disks, rings=rings,
                ndisk=[d.sum() for d in disks], nring=[r.sum() for r in rings])


def cap(maps2d, K):
    """CAP(theta) for a stack maps2d[...,H,H] -> [...,ntheta]."""
    out = np.empty(maps2d.shape[:-2] + (len(K['disks']),))
    for a, (d, g) in enumerate(zip(K['disks'], K['rings'])):
        out[..., a] = maps2d[..., d].mean(-1) - maps2d[..., g].mean(-1)
    return out


def symmetrise(m, rint, nbin):
    """Replace each pixel by the azimuthal average at its radius (centred)."""
    flat = m.ravel()
    prof = np.bincount(rint.ravel(), flat, nbin) / np.bincount(rint.ravel(), None, nbin)
    return prof[rint]


def per_object_caps(Ymap, Gasmap, R200, K, rng, nbin):
    """For one object's 2D Y and Gas maps return profile-CAP and field-CAP for both
    channels plus the field-observed aperture flux (for selection)."""
    sb = K['sig_beam']; sm = K['sig_mc']
    # profile (axisymmetrised + beam), centred
    pY = gaussian_filter(symmetrise(Ymap, K['rint'], nbin), sb, mode='nearest')
    pG = gaussian_filter(symmetrise(Gasmap, K['rint'], nbin), sb, mode='nearest')
    # field (miscenter + beam)
    dy, dx = np.round(rng.normal(0, sm, 2)).astype(int)
    fY = gaussian_filter(np.roll(np.roll(Ymap, dy, 0), dx, 1), sb, mode='nearest')
    fG = gaussian_filter(np.roll(np.roll(Gasmap, dy, 0), dx, 1), sb, mode='nearest')
    ap = K['rr_kpc'] <= R200
    flux = fY[ap].sum() if ap.any() else 0.0
    return (cap(pY, K), cap(pG, K), cap(fY, K), cap(fG, K), flux)


def reduce_objects(Y, Gas, R200, K, seed):
    """Loop objects -> per-object CAP arrays. Y,Gas: (n,128,128)."""
    n = Y.shape[0]; nt = len(K['disks']); nbin = K['rint'].max() + 1
    rng = np.random.default_rng(seed)
    cpY = np.empty((n, nt)); cpG = np.empty((n, nt))
    cfY = np.empty((n, nt)); cfG = np.empty((n, nt)); flux = np.empty(n)
    for i in range(n):
        cpY[i], cpG[i], cfY[i], cfG[i], flux[i] = per_object_caps(
            Y[i].astype(np.float64), Gas[i].astype(np.float64), R200[i], K, rng, nbin)
    return cpY, cpG, cfY, cfG, flux


def stack_design(cpY, cpG, cfY, cfG, flux, fb_halo, w_mf, rng):
    """Mass-function-weighted, flux-selected stacks (profile & field) + parent f_b."""
    obs_flux = flux * (1.0 + rng.normal(0, FLUX_SCATTER, flux.shape))
    limit = np.quantile(flux[flux > 0], SEL_Q) if (flux > 0).any() else 0.0
    sel = obs_flux > limit
    ws = w_mf * sel
    def wstack(c): return np.average(c, 0, weights=ws)
    fb_parent = np.average(fb_halo, weights=w_mf)        # parent (unselected) target
    return (np.average(cpY, 0, weights=w_mf), np.average(cpG, 0, weights=w_mf),  # profile: parent
            wstack(cfY), wstack(cfG), fb_parent, sel.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['bind', 'truth'], required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=256)
    ap.add_argument('--nsim', type=int, default=27)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    np.seterr(invalid='ignore', divide='ignore')   # empty centre bin in symmetrise (unused)
    K = _act_kernels()
    nt = len(CAP_THETA_ARCMIN)
    print(f'ACT: {KPC_H_PER_ARCMIN:.1f} h^-1 kpc/arcmin @ z={Z_OBS}; '
          f'beam sigma={K["sig_beam"]:.2f} pix; mc={K["sig_mc"]:.2f} pix; '
          f'CAP theta[pix]={np.round(K["theta_pix"],1)}', flush=True)

    cube = np.load(S / 'cube.npz', allow_pickle=True)
    M200 = np.asarray(cube['M200'], float); R200 = np.asarray(cube['R200'], float)
    w_mf = M200 ** (-MF_SLOPE); w_mf /= w_mf.sum()

    if args.source == 'bind':
        ex = np.load(S / 'obs_fb_extra.npz', allow_pickle=True)
        fb_halo = ex['extra'][:, :, list(ex['extra_names']).index('f_b')].astype(float)
        rng = np.random.default_rng(SEED)
        nd = args.end - args.start
        out = {k: np.empty((nd, nt)) for k in ['cpY', 'cpG', 'cfY', 'cfG']}
        fb_parent = np.empty(nd); frac_sel = np.empty(nd)
        for j, d in enumerate(range(args.start, args.end)):
            gen = np.load(S / 'maps' / f'gen_design{d:04d}.npz')['generated']
            cpY, cpG, cfY, cfG, flux = reduce_objects(
                gen[:, CH['Y']], gen[:, CH['Gas']], R200, K, SEED + d)
            r = stack_design(cpY, cpG, cfY, cfG, flux, fb_halo[d], w_mf, rng)
            out['cpY'][j], out['cpG'][j], out['cfY'][j], out['cfG'][j] = r[:4]
            fb_parent[j], frac_sel[j] = r[4], r[5]
            print(f'design {d:3d}  f_sel={frac_sel[j]:.2f}  fb={fb_parent[j]:.3f}', flush=True)
        outf = S / (args.out or 'forward_model_act_bind.npz')
        np.savez(outf, theta_arcmin=CAP_THETA_ARCMIN, fb_parent=fb_parent,
                 frac_sel=frac_sel, **out, meta=_meta())
        print('wrote', outf)

    else:  # truth: per-halo CAP so the notebook can stack + bootstrap
        sims = sorted(p for p in CV.iterdir() if p.is_dir() and p.name.startswith('sim_'))
        allcpY = []; allcpG = []; allcfY = []; allcfG = []
        allflux = []; allfb = []; allM = []
        for sd in sims[:args.nsim]:
            md = sd / SNAP / MTAG
            cat_f = md / 'halo_catalog.npz'; th_f = md / 'truth_thermo_patches.npz'
            fm_f = sd / SNAP / 'full_maps.npz'
            if not (cat_f.exists() and th_f.exists() and fm_f.exists()):
                continue
            cat = np.load(cat_f)
            if 'radii' not in cat.files or len(cat['radii']) == 0:
                continue
            centers = cat['centers']; Mh = np.asarray(cat['masses'], float)
            Rh = np.asarray(cat['radii'], float)
            tmaps = np.load(fm_f)['truth_maps']                 # (3,1024,1024)
            thermo = np.load(th_f)['truth_thermo'].astype(np.float32)  # (n,4,128,128)
            Yt = thermo[:, 0]                                   # compton-y
            n = len(centers)
            Gas = np.empty((n, H, H), np.float32); DM = np.empty_like(Gas); Star = np.empty_like(Gas)
            for i, c in enumerate(centers):
                cx = int(c[0] / BOX * NPIX); cy = int(c[1] / BOX * NPIX)
                ix = (cx - HC + np.arange(H)) % NPIX; iy = (cy - HC + np.arange(H)) % NPIX
                DM[i] = tmaps[0][np.ix_(ix, iy)]; Gas[i] = tmaps[1][np.ix_(ix, iy)]
                Star[i] = tmaps[2][np.ix_(ix, iy)]
            cpY, cpG, cfY, cfG, flux = reduce_objects(Yt, Gas, Rh, K, SEED + 99999)
            apm = K['rr_kpc'][None] <= Rh[:, None, None]
            bary = (Gas + Star); tot = (DM + Gas + Star)
            fb = np.array([bary[i][apm[i]].sum() / max(tot[i][apm[i]].sum(), 1e-30)
                           for i in range(n)])
            allcpY.append(cpY); allcpG.append(cpG); allcfY.append(cfY); allcfG.append(cfG)
            allflux.append(flux); allfb.append(fb); allM.append(Mh)
            print(f'[{sd.name}] n={n}', flush=True)
        cat = lambda L: np.concatenate(L, 0)
        outf = S / (args.out or 'forward_model_act_truth.npz')
        np.savez(outf, theta_arcmin=CAP_THETA_ARCMIN,
                 cpY=cat(allcpY), cpG=cat(allcpG), cfY=cat(allcfY), cfG=cat(allcfG),
                 flux=cat(allflux), fb=cat(allfb), M200=cat(allM), meta=_meta())
        print('wrote', outf, 'n_halo=', len(cat(allfb)))


def _meta():
    return dict(z_obs=Z_OBS, kpc_h_per_arcmin=KPC_H_PER_ARCMIN,
                beam_fwhm_arcmin=BEAM_FWHM_ARCMIN, sigma_mc_kpc=SIGMA_MC_KPC,
                cap_theta_arcmin=list(CAP_THETA_ARCMIN), flux_scatter=FLUX_SCATTER,
                sel_q=SEL_Q, mf_slope=MF_SLOPE, seed=SEED)


if __name__ == '__main__':
    main()
