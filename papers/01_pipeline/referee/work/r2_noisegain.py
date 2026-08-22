"""Split the noisy covariance into its shape-noise and cosmic-variance legs.

WHY. Two numbers in the R2 response need this and cannot be read off the cache:

  (a) the referee's core question -- "by how much does LSST-Y10 shape noise inflate
      the covariance of each nu-domain statistic?" -- at FIXED convention. The
      released noiseless cache (nu_grid.py) uses different smoothing and a per-map
      nu normalisation, so noisy-minus-noiseless there mixes three changes at once.

  (b) the correct measurement floor on the Sobol node-to-node spread. All Sobol
      runs share the ray-tracing seed ladder, so their common cosmic variance does
      NOT broaden the node-to-node spread; only the independent shape-noise draw
      does. Using the full 25 deg^2 covariance as the floor over-states it by
      1/sqrt(1 - f_noise).

HOW. For the first N realizations of the fiducial cube, at z_s = 1, compute the
noisy statistics TWICE with two independent noise streams on the SAME map. The
difference has covariance exactly 2 C_noise, with the map (cosmic-variance) leg
cancelled identically. C_cv = C_total - C_noise, with C_total from the released
550-realization shard.

The first stream is noisy_grid's own `noise_rng(target, r, plane)`, so this run
also re-derives cached rows bit-for-bit -- a spot-check of the cache itself.
"""
from __future__ import annotations

import json
import sys
import time
import zipfile

import numpy as np
import numpy.lib.format as npf

import r2_common as R
import noisy_grid as ng

sys.path.insert(0, str(R.P1))

N_REAL = 120
ZI = R.ZI
ALT_SEED = 90210


def stream_plane(path, n, plane):
    """Realizations 0..n-1 of one source plane, without materialising the cube."""
    with zipfile.ZipFile(path) as zf, zf.open("kappa.npy") as fh:
        version = npf.read_magic(fh)
        shape, _, dtype = (npf.read_array_header_1_0(fh) if version == (1, 0)
                           else npf.read_array_header_2_0(fh))
        n_avail, n_plane, ny, nx = shape
        assert n <= n_avail
        pb = ny * nx * dtype.itemsize
        out = np.empty((n, ny, nx), dtype=dtype)
        for r in range(n):
            for p in range(n_plane):
                buf = bytearray()
                while len(buf) < pb:
                    c = fh.read(pb - len(buf))
                    assert c
                    buf += c
                if p == plane:
                    out[r] = np.frombuffer(bytes(buf), dtype=dtype).reshape(ny, nx)
    return out


def stats_of(sm, kappa_rms):
    from bind.inference.stats import _local_extrema, minkowski_functionals
    nu = (sm - sm.mean()) / kappa_rms
    pk = _local_extrema(sm, True)
    mn = _local_extrema(sm, False)
    v0, v1, v2 = minkowski_functionals(nu, ng.NU)
    return dict(
        pdf=np.histogram(nu, bins=ng.NU_EDGES, density=True)[0],
        peak_counts=np.histogram(nu[pk], bins=ng.NU_EDGES)[0].astype(float),
        minima_counts=np.histogram(nu[mn], bins=ng.NU_EDGES)[0].astype(float),
        mf_v0=v0, mf_v1=v1, mf_v2=v2)


def main() -> None:
    t0 = time.time()
    krms = ng.load_kappa_rms()[ZI]
    K = stream_plane(ng.kappa_path("fid"), N_REAL, ZI)
    print(f"streamed {N_REAL} fid realizations, plane {ZI}, in {time.time()-t0:.0f}s")

    alt = np.random.default_rng(ALT_SEED)
    A = {k: np.empty((N_REAL, len(ng.NU))) for k in R.NU_KEYS}
    B = {k: np.empty((N_REAL, len(ng.NU))) for k in R.NU_KEYS}
    for r in range(N_REAL):
        m = K[r].astype(np.float64)
        s1 = ng.smooth(m + ng.noise_rng("fid", r, ZI).normal(0, ng.NOISE_SIGMA_PIX, m.shape))
        s2 = ng.smooth(m + alt.normal(0, ng.NOISE_SIGMA_PIX, m.shape))
        for k, v in stats_of(s1, krms).items():
            A[k][r] = v
        for k, v in stats_of(s2, krms).items():
            B[k][r] = v
        if r % 20 == 0:
            print(f"  real {r:3d}/{N_REAL}  ({time.time()-t0:.0f}s)")
    del K

    fid = R.load_noisy_target("fid")
    out = {}
    print(f"\n{'stat':>14s} {'cache repro':>12s} {'f_noise(var)':>13s} "
          f"{'infl sqrt(tot/cv)':>18s} {'floor_spread':>13s}")
    for k in R.NU_KEYS:
        cached = fid[f"{k}_real"][:N_REAL, ZI, :]
        repro = float(np.nanmax(np.abs(A[k] - cached) /
                                np.where(np.abs(cached) > 0, np.abs(cached), np.nan)))
        Cn = np.cov(A[k] - B[k], rowvar=False) / 2.0          # pure shape-noise cov
        Ct = np.cov(fid[f"{k}_real"][:, ZI, :], rowvar=False)  # total, 550 real
        dn, dt = np.diag(Cn), np.diag(Ct)
        good = dt > 0
        fn = float(np.nanmedian(dn[good] / dt[good]))
        infl = float(np.nanmedian(np.sqrt(dt[good] / np.clip(dt[good] - dn[good],
                                                             1e-300, None))))
        # node-to-node spread floor, in units of sigma_LSST: only the independent
        # shape-noise leg broadens the Sobol spread (shared ray seeds).
        floor = float(np.sqrt(np.nanmedian(dn[good] / dt[good]) / 50.0
                              / R.area_scale(R.AREA_LSST)))
        out[k] = dict(cache_max_rel_dev=repro, f_noise=fn, cov_inflation=infl,
                      spread_floor_sigma_lsst=floor,
                      f_noise_bins=[float(x) for x in (dn / np.where(dt > 0, dt, np.nan))])
        print(f"{k:>14s} {repro:>12.2e} {fn:>13.3f} {infl:>18.2f} {floor:>13.2f}")

    # the FULL measured shape-noise covariance matrices, plus the raw draws, so no
    # consumer has to re-run this (the per-bin diagonal fraction alone is not enough
    # to build a floor: element-wise scaling of a strongly correlated covariance
    # destroys its null-space alignment and the resulting trace blows up).
    np.savez_compressed(R.OUT / "noise_split_cov.npz",
                        **{f"Cn_{k}": np.cov(A[k] - B[k], rowvar=False) / 2.0
                           for k in R.NU_KEYS},
                        **{f"A_{k}": A[k] for k in R.NU_KEYS},
                        **{f"B_{k}": B[k] for k in R.NU_KEYS},
                        nu=ng.NU, n_real=np.array(N_REAL), plane=np.array(ZI))
    out["_meta"] = dict(n_real=N_REAL, plane=ZI, alt_seed=ALT_SEED,
                        kappa_rms=float(krms))
    with open(R.OUT / "noise_split.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {R.OUT/'noise_split.json'}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
