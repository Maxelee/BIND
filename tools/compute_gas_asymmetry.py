#!/usr/bin/env python
"""Compute a transparent gas-morphology (CAS rotational asymmetry) for every
(design, halo) in the Sobol single-snapshot suite, and cache a small demo pack
for the paper.ipynb same-halo figure.

Asymmetry (Conselice 2003), per halo, on the Gas channel within R200:

    A[x] = sum_ap |x - R_pi x| / (2 sum_ap |x|),

where R_pi is a 180-deg rotation about the gas centroid. A is *exactly* zero for
any azimuthally symmetric field, so it is orthogonal to the radial profile by
construction. Unlike the undocumented `morph` scalar in the profile shards, this
definition is explicit and reproducible.

Output: <S>/gas_asymmetry.npz
    A            (256,1111)  per-(design,halo) gas asymmetry (NaN if no gas)
    demo_halo    int         a mid-mass halo with a strong feedback response
    demo_logM    float
    demo_R200    float       [kpc/h]
    demo_param   int         parameter index used to split low/high feedback
    demo_gas     (256,128,128) demo halo's Gas map under every design
    param_names  (35,)

Run (Flatiron rusty, torch3 venv):  python tools/compute_gas_asymmetry.py
"""
from pathlib import Path
import numpy as np

S = Path('/mnt/home/mlee1/ceph/sobol_ss_cv')
PIX_KPC = 48.828125
H = 128
GAS = 1                       # channel order [DM,Gas,Stars,Y,T,S,P]
ND = 256

yy, xx = np.mgrid[0:H, 0:H]
rr = np.hypot(yy - H / 2 + 0.5, xx - H / 2 + 0.5) * PIX_KPC


def asym(img, R):
    """Centroid-centered CAS rotational asymmetry of img within radius R [kpc/h]."""
    ap = rr <= R
    w = np.clip(img, 0, None) * ap
    tot = w.sum()
    if tot <= 0:
        return np.nan
    yc = (w * yy).sum() / tot
    xc = (w * xx).sum() / tot
    im = np.roll(np.roll(img, int(round(H / 2 - yc)), 0), int(round(H / 2 - xc)), 1)
    rot = im[::-1, ::-1]
    return (np.abs(im - rot) * ap).sum() / (2.0 * (np.abs(im) * ap).sum())


def main():
    cube = np.load(S / 'cube.npz', allow_pickle=True)
    R200 = np.asarray(cube['R200'], float)
    lM = np.log10(np.asarray(cube['M200'], float))
    pn = list(cube['param_names'])
    Nh = R200.shape[0]

    # seed demo candidates: mid-mass halos with the largest stored-morph swing
    morph = np.stack([np.load(S / 'profile_shards' / f'prof_design{d:04d}.npz')
                      ['morph'][:, 0] for d in range(ND)])
    mid = np.where((lM > 13.4) & (lM < 14.0))[0]
    cand = mid[np.argsort(-morph.std(0)[mid])[:12]]
    cand_gas = np.zeros((len(cand), ND, H, H), np.float32)

    A = np.full((ND, Nh), np.nan)
    for d in range(ND):
        g = np.load(S / 'maps' / f'gen_design{d:04d}.npz')['generated'][:, GAS]
        g = g.astype(np.float64)
        for h in range(Nh):
            A[d, h] = asym(g[h], R200[h])
        cand_gas[:, d] = g[cand].astype(np.float32)
        if d % 25 == 0:
            print(f'  design {d:3d}/{ND}', flush=True)

    # choose the demo halo: candidate with the largest feedback response in A
    swing = np.nanstd(A[:, cand], axis=0)
    demo_i = int(np.argmax(swing))
    demo_halo = int(cand[demo_i])
    demo_param = pn.index('BlackHoleRadiativeEfficiency')

    out = S / 'gas_asymmetry.npz'
    np.savez(out, A=A.astype(np.float32),
             demo_halo=demo_halo, demo_logM=float(lM[demo_halo]),
             demo_R200=float(R200[demo_halo]), demo_param=demo_param,
             demo_gas=cand_gas[demo_i], param_names=np.array(pn))
    print(f'wrote {out}')
    print(f'  demo_halo={demo_halo} logM={lM[demo_halo]:.2f} '
          f'A-swing={swing[demo_i]:.3f}')


if __name__ == '__main__':
    main()
