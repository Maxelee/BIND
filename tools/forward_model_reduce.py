#!/usr/bin/env python
"""Forward-model reduction for the field-vs-profile demonstration.

For each of the 256 Sobol feedback designs (the SAME 1111 CV halos repainted by
BIND under different feedback), stream the per-halo 2D fields and produce, per
design:

  * the CLEAN mass-function-weighted stacked profiles Y(r), SX(r)
        -- what a *profile-level* emulator can output (azimuthal average of the
           parent population, no instrument, no selection);
  * the PROCESSED mass-function-weighted stacked profiles Y(r), SX(r)
        -- what a real survey measures: each halo's 2D field is miscentered,
           beam-convolved, core-masked and noised; halos enter the stack only if
           their observed aperture flux passes an S/N cut (Eddington selection),
           weighted by the target mass function;
  * the parent (unselected) mass-function-weighted baryon fraction fb_parent
        -- the science target both estimators try to recover.

Only a *field-level* emulator can produce the PROCESSED stack (the operators act
on the 2D field and do not commute with the azimuthal average). The notebook then
trains one regression on CLEAN stacks and one on PROCESSED stacks, applies both to
the processed mock observation, and reports the bias of the profile path.

Output: <ceph>/sobol_ss_cv/forward_model_stacks.npz  (small; resumable).

Run (full, ~streams 256 x 510 MB):
    python tools/forward_model_reduce.py
Smoke test (first 3 designs):
    python tools/forward_model_reduce.py --end 3 --out _fm_smoke.npz
Slurm array friendly: --start/--end select a design range.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

S = Path('/mnt/home/mlee1/ceph/sobol_ss_cv')

# ---- geometry / channels ---------------------------------------------------
PIX_KPC = 48.828125
CH = {'DM': 0, 'Gas': 1, 'Stars': 2, 'Y': 3, 'T': 4, 'S': 5, 'P': 6}
H = 128
CEN = H / 2 - 0.5                                    # image centre (observer's guess)

# ---- forward-model parameters (documented knobs) ---------------------------
SIGMA_MC_KPC = 150.0                                 # centroiding error (1-sigma)
BEAM_SIGMA_KPC = 53.0                                # 1.6' ACT-equivalent beam sigma
RMASK_FRAC = 0.15                                    # core excision radius / R200 (AGN/BCG)
FLUX_SCATTER = 0.30                                  # fractional flux measurement scatter
SEL_Q = 0.30                                         # global flux limit = this quantile of flux
MF_SLOPE = 1.0                                       # target dn/dlnM ~ M^-MF_SLOPE (steep HMF)
SEED = 20250607

EDGES = [13.5, 14.0]                                 # mass-bin edges (for per-bin output)


def radial_operator(r_kpc):
    """Build masks for the 12 stacked-profile radial bins on the 128^2 grid,
    centred on the image centre (the observer's assumed centre)."""
    yy, xx = np.mgrid[0:H, 0:H]
    rr = np.hypot(yy - CEN, xx - CEN) * PIX_KPC
    logr = np.log10(r_kpc)
    edges = np.concatenate([[10 ** (logr[0] - (logr[1] - logr[0]) / 2)],
                            10 ** ((logr[1:] + logr[:-1]) / 2),
                            [10 ** (logr[-1] + (logr[-1] - logr[-2]) / 2)]])
    binidx = np.digitize(rr, edges) - 1
    masks = [(binidx == b) for b in range(len(r_kpc))]
    return rr, masks


def azimuthal(stack2d, masks):
    """Azimuthal average of a single 2D map over the radial-bin masks."""
    return np.array([stack2d[m].mean() if m.any() else np.nan for m in masks])


def process_design(gen, R200, rr, masks, rng):
    """Apply the instrument+selection forward model to one design's fields.

    Returns per-halo observed aperture flux (for selection) and the per-halo
    PROCESSED 2D Y and SX maps already reduced to radial profiles, plus the CLEAN
    per-halo radial profiles. Shapes: fluxY (n,), {clean,proc}{Y,SX} (n, nbin)."""
    n = gen.shape[0]
    Ymap = gen[:, CH['Y']].astype(np.float64)                       # (n,128,128)
    Gas = gen[:, CH['Gas']].astype(np.float64)
    T = np.clip(gen[:, CH['T']].astype(np.float64), 0, None)
    SXmap = Gas ** 2 * np.sqrt(T)                                   # X-ray SB proxy

    nbin = len(masks)
    cleanY = np.full((n, nbin), np.nan); cleanSX = np.full((n, nbin), np.nan)
    procY = np.full((n, nbin), np.nan); procSX = np.full((n, nbin), np.nan)
    fluxY = np.zeros(n)

    sig_mc = SIGMA_MC_KPC / PIX_KPC
    sig_beam = BEAM_SIGMA_KPC / PIX_KPC

    for i in range(n):
        ap = rr <= R200[i]
        # ---- clean profiles (parent, no instrument, no selection) ----------
        cleanY[i] = azimuthal(Ymap[i], masks)
        cleanSX[i] = azimuthal(SXmap[i], masks)
        # ---- processed: miscenter -> beam -> core-mask --------------------
        dy, dx = np.round(rng.normal(0, sig_mc, 2)).astype(int)
        py = np.roll(np.roll(Ymap[i], dy, 0), dx, 1)
        ps = np.roll(np.roll(SXmap[i], dy, 0), dx, 1)
        py = gaussian_filter(py, sig_beam, mode='nearest')
        ps = gaussian_filter(ps, sig_beam, mode='nearest')
        core = rr <= RMASK_FRAC * R200[i]
        py = py.copy(); ps = ps.copy()
        py[core] = np.nan; ps[core] = np.nan
        # noiseless observed aperture flux for selection (core excised)
        apm = ap & ~core
        fluxY[i] = np.nansum(py[apm]) if apm.any() else 0.0
        procY[i] = np.array([np.nanmean(py[m]) if m.any() else np.nan for m in masks])
        procSX[i] = np.array([np.nanmean(ps[m]) if m.any() else np.nan for m in masks])

    return cleanY, cleanSX, procY, procSX, fluxY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=256)
    ap.add_argument('--out', default='forward_model_stacks.npz')
    args = ap.parse_args()

    cube = np.load(S / 'cube.npz', allow_pickle=True)
    M200 = np.asarray(cube['M200'], float)
    R200_kpc = np.asarray(cube['R200'], float)
    lM = np.log10(M200)
    ex = np.load(S / 'obs_fb_extra.npz', allow_pickle=True)
    en = list(ex['extra_names'])
    fb_halo = ex['extra'][:, :, en.index('f_b')].astype(float)      # (256,1111)

    sp = np.load(S / 'stacked_profiles.npz', allow_pickle=True)
    r_kpc = np.asarray(sp['r_kpc'], float)
    rr, masks = radial_operator(r_kpc)
    nbin = len(r_kpc)

    # target mass-function weight: dn/dlnM ~ M^-slope (relative to suite sampling)
    w_mf = M200 ** (-MF_SLOPE)
    w_mf = w_mf / w_mf.sum()
    mb = np.digitize(lM, EDGES)
    NB = 3

    ndes = args.end - args.start
    Yclean = np.full((ndes, nbin), np.nan); SXclean = np.full((ndes, nbin), np.nan)
    Yproc = np.full((ndes, nbin), np.nan); SXproc = np.full((ndes, nbin), np.nan)
    fb_parent = np.zeros(ndes)
    # per-mass-bin (robustness)
    Yclean_b = np.full((ndes, NB, nbin), np.nan); Yproc_b = np.full((ndes, NB, nbin), np.nan)
    SXclean_b = np.full((ndes, NB, nbin), np.nan); SXproc_b = np.full((ndes, NB, nbin), np.nan)
    fb_parent_b = np.zeros((ndes, NB))
    frac_sel = np.zeros(ndes)

    np.seterr(invalid='ignore', divide='ignore')   # empty inner radial bins -> NaN (dropped later)
    for j, d in enumerate(range(args.start, args.end)):
        gen = np.load(S / 'maps' / f'gen_design{d:04d}.npz')['generated']
        rng = np.random.default_rng(SEED + d)
        cY, cSX, pY, pSX, fluxY = process_design(gen, R200_kpc, rr, masks, rng)

        # flux-limited selection with measurement scatter -> Eddington bias:
        # observed flux scatters about the noiseless value; a global flux limit
        # (per-design quantile) then preferentially admits up-scattered, gas-rich
        # (high-f_b) halos near the cut.
        obs_flux = fluxY * (1.0 + rng.normal(0, FLUX_SCATTER, fluxY.shape))
        limit = np.quantile(fluxY[fluxY > 0], SEL_Q) if (fluxY > 0).any() else 0.0
        sel = obs_flux > limit                                       # Eddington selection
        frac_sel[j] = sel.mean()
        # parent (unselected) mass-weighted target + clean stacks
        wp = w_mf
        fb_parent[j] = np.average(fb_halo[d], weights=wp)
        Yclean[j] = np.nansum(cY * wp[:, None], 0) / np.nansum(~np.isnan(cY) * wp[:, None], 0)
        SXclean[j] = np.nansum(cSX * wp[:, None], 0) / np.nansum(~np.isnan(cSX) * wp[:, None], 0)
        # processed (selected) mass-weighted stacks
        ws = w_mf * sel
        if ws.sum() > 0:
            Yproc[j] = np.nansum(pY * ws[:, None], 0) / np.nansum(~np.isnan(pY) * ws[:, None], 0)
            SXproc[j] = np.nansum(pSX * ws[:, None], 0) / np.nansum(~np.isnan(pSX) * ws[:, None], 0)
        for b in range(NB):
            mbk = (mb == b)
            wpb = w_mf * mbk
            wsb = w_mf * mbk * sel
            fb_parent_b[j, b] = np.average(fb_halo[d], weights=wpb)
            Yclean_b[j, b] = np.nansum(cY * wpb[:, None], 0) / np.nansum(~np.isnan(cY) * wpb[:, None], 0)
            SXclean_b[j, b] = np.nansum(cSX * wpb[:, None], 0) / np.nansum(~np.isnan(cSX) * wpb[:, None], 0)
            if wsb.sum() > 0:
                Yproc_b[j, b] = np.nansum(pY * wsb[:, None], 0) / np.nansum(~np.isnan(pY) * wsb[:, None], 0)
                SXproc_b[j, b] = np.nansum(pSX * wsb[:, None], 0) / np.nansum(~np.isnan(pSX) * wsb[:, None], 0)
        print(f'design {d:3d}  f_sel={frac_sel[j]:.2f}  fb_par={fb_parent[j]:.3f}', flush=True)

    out = S / args.out
    np.savez(out,
             r_kpc=r_kpc, design_range=np.array([args.start, args.end]),
             Yclean=Yclean, SXclean=SXclean, Yproc=Yproc, SXproc=SXproc,
             fb_parent=fb_parent, frac_sel=frac_sel,
             Yclean_b=Yclean_b, Yproc_b=Yproc_b, SXclean_b=SXclean_b,
             SXproc_b=SXproc_b, fb_parent_b=fb_parent_b, mass_lbl=sp['mass_lbl'],
             params=dict(sigma_mc_kpc=SIGMA_MC_KPC, beam_sigma_kpc=BEAM_SIGMA_KPC,
                         rmask_frac=RMASK_FRAC, flux_scatter=FLUX_SCATTER, sel_q=SEL_Q,
                         mf_slope=MF_SLOPE, seed=SEED))
    print('wrote', out)


if __name__ == '__main__':
    main()
