"""Box-level matter-power suppression for the Sobol feedback cube.

For each of the 256 Sobol design points we paste the generated halo patches back
into the full 50 Mpc/h DMO box (one composite per CV sim), measure the total-
matter power spectrum P_BIND(k), and form the suppression S(k)=P_BIND/P_DMO,
averaged over the CV sims.  We also build an "assembly-marginalised"
counterfactual box in which each halo's baryonic patch is replaced by the
mean patch of its mass bin (same design), isolating the assembly-correlated
part of the imprint.

Outputs analysis_physics_cache/box_supp_sobol.npz:
  k_box            (nk,)            shared wavenumber grid [h/Mpc]
  S_true           (256, nk)        box suppression, sim-averaged
  S_massonly       (256, nk)        assembly-marginalised counterfactual
  S_truth          (nk,)            hydro-truth box suppression (design-indep ref)
  design_id        (256,)
Run:
    python tools/box_supp_sobol.py --designs all          # full cache
    python tools/box_supp_sobol.py --designs 0,1,128      # smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import Pk_library as PKL

SOBOL_ROOT = Path('/mnt/home/mlee1/ceph/sobol_ss_cv')
CV_ROOT = Path('/mnt/home/mlee1/ceph/fm_testsuite/CV')
SNAP = 'snap_090'
MASS_TAG = 'mass_threshold_1p000e13'
BOX = 50.0
NPIX = 1024
PATCH = 128
OUT = Path('analysis_physics_cache/box_supp_sobol.npz')

# mass bins for the assembly-marginalised counterfactual (log10 Msun/h)
MASS_EDGES = np.arange(13.0, 15.01, 0.25)


# ---- pure-numpy composite primitives (mirrors bind.inference.pipeline) -------
def square_taper_weight(patch_size: int, taper_frac: float = 0.15) -> np.ndarray:
    t = max(1, int(patch_size * taper_frac))
    w1 = np.ones(patch_size, dtype=np.float32)
    ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(t) + 0.5) / t))
    w1[:t] = ramp
    w1[-t:] = ramp[::-1]
    return np.outer(w1, w1).astype(np.float32)


def paste_halos_2d(canvas_res, box_size, halos, patches, weight):
    canvas = np.zeros((3, canvas_res, canvas_res), dtype=np.float32)
    w_accum = np.zeros((canvas_res, canvas_res), dtype=np.float32)
    ppm = canvas_res / box_size
    w_half = weight.shape[0] // 2
    for halo, patch in zip(halos, patches):
        cx = int(halo['halo_center'][0] * ppm) % canvas_res
        cy = int(halo['halo_center'][1] * ppm) % canvas_res
        ix = (cx - w_half + np.arange(weight.shape[0])) % canvas_res
        iy = (cy - w_half + np.arange(weight.shape[0])) % canvas_res
        for ch in range(3):
            canvas[ch][np.ix_(ix, iy)] += patch[ch] * weight
        w_accum[np.ix_(ix, iy)] += weight
    safe = np.where(w_accum > 0, w_accum, 1.0)
    canvas /= safe[None]
    return canvas, w_accum


def build_bind_composite(dmo_fullbox, halos, generated_patches, halo_cutouts,
                         box_size, npix, patch_pix, patch_mass_match,
                         taper_frac):
    patches = []
    for patch, hc in zip(generated_patches, halo_cutouts):
        p = patch.copy()
        if patch_mass_match:
            s = float(hc['condition'].sum()) / (float(p.sum()) + 1e-30)
            p *= s
        patches.append(p)
    patches_np = np.asarray(patches, dtype=np.float32)
    taper = square_taper_weight(patch_pix, taper_frac=taper_frac)
    hydro_canvas, hydro_weights = paste_halos_2d(npix, box_size, halos,
                                                 patches_np, taper)
    alpha = np.clip(hydro_weights, 0.0, 1.0)
    comp = np.zeros((3, npix, npix), dtype=np.float32)
    comp[0] = (1 - alpha) * dmo_fullbox + alpha * hydro_canvas[0]
    comp[1] = alpha * hydro_canvas[1]
    comp[2] = alpha * hydro_canvas[2]
    comp *= float(dmo_fullbox.sum() / (comp.sum() + 1e-30))
    return {'composite': comp}


def power_spectrum_2d(field, box_size=BOX):
    delta = np.asarray(field, dtype=np.float32) / np.mean(field)
    Pk2D = PKL.Pk_plane(delta, box_size, 'CIC', 1)
    return Pk2D.k, Pk2D.Pk


def total_matter(comp3):
    """Sum DM_hydro+Gas+Stars -> total-matter 2D field."""
    return comp3[0] + comp3[1] + comp3[2]


def load_sim_static():
    """Per-sim DMO box, truth total-matter Pk, halos list, cutout conditions.

    Returns a list of per-sim dicts in the cube's halo ordering, plus the
    contiguous global-index block for each sim.
    """
    cube = np.load(SOBOL_ROOT / 'cube.npz', allow_pickle=True)
    sim_id = np.asarray([str(s) for s in cube['sim_id']])
    M200 = np.asarray(cube['M200'], float)

    sims = []
    off = 0
    # preserve order of first appearance == sorted CV dirs (build order)
    seen = []
    for s in sim_id:
        if s not in seen:
            seen.append(s)
    for s in seen:
        idx = np.where(sim_id == s)[0]
        assert (np.diff(idx) == 1).all(), f'{s}: non-contiguous block'
        sdir = CV_ROOT / s / SNAP
        fm = np.load(sdir / 'full_maps.npz')
        cat = np.load(sdir / MASS_TAG / 'halo_catalog.npz')
        cuts = np.load(sdir / MASS_TAG / 'halo_cutouts.npz')
        masses = np.asarray(cat['masses'], float)
        assert np.allclose(masses, M200[idx]), f'{s}: mass misalignment'
        halos = [{'halo_center': cat['centers'][i], 'r200': 0.0}
                 for i in range(len(masses))]
        cutlist = [{'condition': cuts['condition'][i]} for i in range(len(masses))]
        dmo = np.asarray(fm['dmo_fullbox'], np.float32)
        truth_tot = np.asarray(fm['truth_maps'], np.float32).sum(0)
        k_d, pk_dmo = power_spectrum_2d(dmo, box_size=BOX)
        _, pk_truth = power_spectrum_2d(truth_tot, box_size=BOX)
        sims.append(dict(name=s, block=idx, dmo=dmo, halos=halos, cuts=cutlist,
                         logM=np.log10(masses), k=k_d, pk_dmo=pk_dmo,
                         pk_truth=pk_truth))
        off = idx[-1] + 1
    return sims, off


def composite_pk(sim, patches3):
    """Box total-matter Pk for one sim from its generated mass-channel patches."""
    bundle = build_bind_composite(
        sim['dmo'], sim['halos'], patches3, sim['cuts'],
        box_size=BOX, npix=NPIX, patch_pix=PATCH,
        patch_mass_match=True, taper_frac=0.15,
    )
    _, pk = power_spectrum_2d(total_matter(bundle['composite']), box_size=BOX)
    return pk


def massonly_patches(patches3, logM):
    """Replace each halo patch by the mean patch of its mass bin (this design)."""
    out = patches3.copy()
    b = np.digitize(logM, MASS_EDGES)
    for bi in np.unique(b):
        m = b == bi
        if m.sum() >= 3:
            out[m] = patches3[m].mean(0, keepdims=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--designs', default='all')
    args = ap.parse_args()

    sims, _ = load_sim_static()
    k_box = sims[0]['k']
    # design-independent hydro-truth reference (sim-averaged ratio)
    S_truth = np.mean([s['pk_truth'] / s['pk_dmo'] for s in sims], axis=0)

    map_files = sorted((SOBOL_ROOT / 'maps').glob('gen_design*.npz'))
    n_design = len(map_files)
    if args.designs == 'all':
        dids = list(range(n_design))
    else:
        dids = [int(x) for x in args.designs.split(',')]

    S_true = np.full((n_design, len(k_box)), np.nan)
    S_mass = np.full((n_design, len(k_box)), np.nan)

    for n, d in enumerate(dids):
        t0 = time.time()
        gen = np.load(map_files[d])['generated'][:, :3].astype(np.float32)
        rt, rm = [], []
        for s in sims:
            p = gen[s['block']]
            rt.append(composite_pk(s, p) / s['pk_dmo'])
            rm.append(composite_pk(s, massonly_patches(p, s['logM'])) / s['pk_dmo'])
        S_true[d] = np.mean(rt, axis=0)
        S_mass[d] = np.mean(rm, axis=0)
        print(f'[{n + 1}/{len(dids)}] design {d:4d}  '
              f'S(k~1)={S_true[d][np.argmin(abs(k_box - 1))]:.3f}  '
              f'({time.time() - t0:.1f}s)', flush=True)

    if args.designs == 'all':
        OUT.parent.mkdir(exist_ok=True)
        np.savez_compressed(OUT, k_box=k_box, S_true=S_true, S_massonly=S_mass,
                            S_truth=S_truth, design_id=np.arange(n_design))
        print(f'wrote {OUT}')
    else:
        print('smoke test (not saved). k range:', k_box.min(), k_box.max(),
              'nk=', len(k_box))


if __name__ == '__main__':
    main()
