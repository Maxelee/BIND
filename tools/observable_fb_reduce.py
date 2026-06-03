"""Derive the extra observables (kSZ optical-depth proxy, X-ray surface brightness)
for the Observable -> f_b proof-of-concept, from the Sobol feedback-cube maps.

The released cube (`/mnt/home/mlee1/ceph/sobol_ss_cv/cube.npz`) already carries
``[Y200, T, S, P, f_gas, m_gen, supp_k10, supp_prof]`` per (design, halo) in a
fixed R200 circular aperture (see sobol_ss_generation.reduce_design).  This tool
adds four columns by reading the 7-channel maps ``[DM,Gas,Stars,Y,T,S,P]`` and
re-using the *identical* aperture:

  tau_ksz : sum_ap(Gas) * PIX_AREA_MPC2
            kSZ optical-depth / electron-column proxy = projected gas mass in
            the aperture.  NOTE this equals f_gas * m_gen (gas mass), so it is a
            near-deterministic function of the f_b target at fixed mass -- it is
            the *upper-bound anchor*, not an independent observable.
  SX      : sum_ap(Gas**2 * sqrt(T))
            soft-band X-ray surface-brightness proxy (emission measure ~ n_e^2,
            bremsstrahlung emissivity ~ sqrt(T)).  This carries genuinely new,
            density-squared + temperature information that Y (pressure ~ n_e T)
            does not -- the degeneracy-breaking observable in the PoC.
  f_star  : sum_ap(Stars) / sum_ap(DM+Gas+Stars)   (stellar mass fraction)
  f_b     : sum_ap(Gas+Stars) / sum_ap(DM+Gas+Stars)   (the f_b target)

A built-in self-check recomputes Y200 and f_gas from the maps and asserts they
match the cube to float tolerance, which validates the aperture is reproduced.

Run (whole design grid, ~3 min, single thread):
    python tools/observable_fb_reduce.py
SLURM-array friendly (each task does a contiguous chunk, then run --reduce):
    CHUNK_ID=$SLURM_ARRAY_TASK_ID N_CHUNKS=8 python tools/observable_fb_reduce.py
    python tools/observable_fb_reduce.py --reduce
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

SOBOL_ROOT = Path("/mnt/home/mlee1/ceph/sobol_ss_cv")
OUT = SOBOL_ROOT / "obs_fb_extra.npz"          # final merged (256, 1111, 2)
SHARD_DIR = SOBOL_ROOT / "obs_fb_shards"        # per-chunk shards for array mode

# geometry -- identical to sobol_ss_generation.py
PATCH_PIX = 128
PIX_MPC = 50.0 * PATCH_PIX / 1024 / PATCH_PIX   # 0.048828 Mpc/h per pixel
PIX_KPC = PIX_MPC * 1000.0                       # 48.83 kpc/h (R200 is kpc/h)
PIX_AREA_MPC2 = PIX_MPC ** 2

_yy, _xx = np.mgrid[0:PATCH_PIX, 0:PATCH_PIX]
_RR = np.sqrt((_xx - PATCH_PIX / 2.0) ** 2 + (_yy - PATCH_PIX / 2.0) ** 2)

EXTRA_NAMES = ["tau_ksz", "SX", "f_star", "f_b"]


def reduce_design(gen: np.ndarray, R200: np.ndarray):
    """gen: (N,7,128,128) physical. Returns extra (N,4) and check (N,2) [Y200,f_gas].

    extra cols: tau_ksz, SX, f_star=Stars/total, f_b=(Gas+Stars)/total -- all in
    the identical R200 aperture used by sobol_ss_generation.reduce_design."""
    n = gen.shape[0]
    extra = np.full((n, len(EXTRA_NAMES)), np.nan)
    check = np.full((n, 2), np.nan)
    for i in range(n):
        g = gen[i]
        r200_pix = max(R200[i] / PIX_KPC, 1.0)
        ap = _RR <= r200_pix
        gas = g[1][ap]
        star = g[2][ap]
        T = g[4][ap]
        tot = (g[0] + g[1] + g[2])[ap]
        sw = float(gas.sum())
        ssw = float(star.sum())
        mtot = float(tot.sum())
        extra[i, 0] = sw * PIX_AREA_MPC2                              # tau_ksz
        extra[i, 1] = float((np.clip(gas, 0, None) ** 2
                             * np.sqrt(np.clip(T, 0, None))).sum())   # SX
        if mtot > 0:
            extra[i, 2] = ssw / mtot                                  # f_star
            extra[i, 3] = (sw + ssw) / mtot                           # f_b
        check[i, 0] = float(g[3][ap].sum()) * PIX_AREA_MPC2          # Y200
        check[i, 1] = sw / mtot if mtot > 0 else np.nan              # f_gas
    return extra, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduce", action="store_true",
                    help="merge per-chunk shards into the final cube")
    args = ap.parse_args()

    map_files = sorted((SOBOL_ROOT / "maps").glob("gen_design*.npz"))
    n_design = len(map_files)
    cube = np.load(SOBOL_ROOT / "cube.npz", allow_pickle=True)
    R200 = np.asarray(cube["R200"], float)
    cube_Y = cube["obs"][:, :, 0]
    cube_fg = cube["obs"][:, :, 4]

    if args.reduce:
        shards = sorted(SHARD_DIR.glob("shard_*.npz"))
        merged = np.full((n_design, len(R200), len(EXTRA_NAMES)), np.nan)
        for s in shards:
            z = np.load(s)
            merged[z["dids"]] = z["extra"]
        assert not np.isnan(merged).all(axis=(1, 2)).any(), "some designs missing"
        np.savez_compressed(OUT, extra=merged, extra_names=np.array(EXTRA_NAMES),
                            design_id=np.arange(n_design))
        print(f"wrote {OUT}  shape {merged.shape}")
        return

    chunk_id = int(os.environ.get("CHUNK_ID", -1))
    n_chunks = int(os.environ.get("N_CHUNKS", 1))
    dids = (list(range(n_design)) if chunk_id < 0
            else list(range(n_design))[chunk_id::n_chunks])

    extra = np.full((len(dids), len(R200), len(EXTRA_NAMES)), np.nan)
    max_dY = max_dfg = 0.0
    for n, d in enumerate(dids):
        t0 = time.time()
        gen = np.load(map_files[d])["generated"].astype(np.float32)
        ex, ck = reduce_design(gen, R200)
        extra[n] = ex
        # self-check against the released cube (skip NaN-safe)
        m = np.isfinite(ck[:, 0]) & np.isfinite(cube_Y[d])
        dY = np.nanmax(np.abs(ck[m, 0] - cube_Y[d][m]) / (np.abs(cube_Y[d][m]) + 1e-30))
        dfg = np.nanmax(np.abs(ck[m, 1] - cube_fg[d][m]))
        max_dY, max_dfg = max(max_dY, dY), max(max_dfg, dfg)
        print(f"[{n+1}/{len(dids)}] design {d:4d}  relY={dY:.2e} dfg={dfg:.2e} "
              f"({time.time()-t0:.1f}s)", flush=True)

    print(f"self-check: max rel Y200 err={max_dY:.2e}  max f_gas err={max_dfg:.2e}")
    assert max_dY < 1e-3 and max_dfg < 1e-3, "aperture mismatch vs cube!"

    if chunk_id < 0:
        np.savez_compressed(OUT, extra=extra, extra_names=np.array(EXTRA_NAMES),
                            design_id=np.array(dids))
        print(f"wrote {OUT}  shape {extra.shape}")
    else:
        SHARD_DIR.mkdir(exist_ok=True)
        sp = SHARD_DIR / f"shard_{chunk_id:03d}.npz"
        np.savez_compressed(sp, extra=extra, dids=np.array(dids))
        print(f"wrote {sp}")


if __name__ == "__main__":
    main()
