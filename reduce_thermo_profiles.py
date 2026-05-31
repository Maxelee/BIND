"""reduce_thermo_profiles.py — radially-resolved thermo features for field-level susceptibility.

The Sobol cube stores only R200-integrated scalars. Project 1's field-level extension asks
*where* in the halo feedback acts: it needs each thermo field as a **radial profile**, per halo
per design, so we can measure susceptibility separately in the core vs the outskirts and ask
whether the assembly dependence (concentrated/early-forming halos → plastic pressure) is a
*core* effect.

This re-reduces the 256 generated-field files `maps/gen_design{d:04d}.npz` (each (1111,7,128,128)
physical units; channels 0=DM 1=gas 2=stars 3=compton_y 4=T 5=S 6=P) into per-halo radial
profiles, in r/R200 bins, aligned 1:1 with `cube.npz`. Mass-weighted (gas) T/S/P; y summed×area;
gas mass per bin. Also a pressure-field morphology scalar (centroid shift / R200).

Run (no GPU; ~130 GB read, a few minutes; chunk for SLURM parallelism):
    source /mnt/home/mlee1/venvs/torch3/bin/activate
    python reduce_thermo_profiles.py                       # all 256 designs -> shards
    python reduce_thermo_profiles.py --n_chunks 8 --chunk_id 0   # one shard
    python reduce_thermo_profiles.py --reduce              # concat shards -> thermo_profiles.npz
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from sobol_ss_generation import (load_cv_halos, OUT_ROOT_DEFAULT, PIX_KPC, PIX_AREA_MPC2,
                                  PATCH_PIX, _RR)

# r/R200 bin edges; bins are [lo, hi) in units of R200
RBIN_EDGES = np.array([0.0, 0.15, 0.30, 0.50, 0.75, 1.00, 1.50])
RBIN_MID = 0.5 * (RBIN_EDGES[:-1] + RBIN_EDGES[1:])
FIELD_NAMES = ["gas_mass", "T", "S", "P", "y"]            # per-bin reductions
MORPH_NAMES = ["P_centroid_shift"]                       # per-halo morphology scalars
SHARD_DIR = OUT_ROOT_DEFAULT / "profile_shards"
_YY, _XX = np.mgrid[0:PATCH_PIX, 0:PATCH_PIX]


def reduce_one_design(gen: np.ndarray, R200_kpc: np.ndarray):
    """gen (N,7,128,128) physical -> prof (N, n_field, n_bin), morph (N, n_morph)."""
    n = gen.shape[0]
    nb = len(RBIN_MID)
    prof = np.full((n, len(FIELD_NAMES), nb), np.nan)
    morph = np.full((n, len(MORPH_NAMES)), np.nan)
    for i in range(n):
        g = gen[i]
        r200_pix = max(R200_kpc[i] / PIX_KPC, 1.0)
        gas = np.maximum(g[1], 0.0)
        rr = _RR / r200_pix                                  # pixel radius in R200 units
        for b in range(nb):
            m = (rr >= RBIN_EDGES[b]) & (rr < RBIN_EDGES[b + 1])
            gm = float(gas[m].sum())
            prof[i, 0, b] = gm
            if gm > 0:
                prof[i, 1, b] = float((g[4][m] * gas[m]).sum() / gm)     # T (gas-weighted)
                prof[i, 2, b] = float((g[5][m] * gas[m]).sum() / gm)     # S
                prof[i, 3, b] = float((g[6][m] * gas[m]).sum() / gm)     # P
            prof[i, 4, b] = float(np.maximum(g[3][m], 0.0).sum()) * PIX_AREA_MPC2  # y
        # pressure-field centroid shift within R200 (disturbed-morphology proxy)
        ap = _RR <= r200_pix
        P = np.maximum(g[6], 0.0) * ap
        tot = P.sum()
        if tot > 0:
            cx = (P * _XX).sum() / tot; cy = (P * _YY).sum() / tot
            morph[i, 0] = np.hypot(cx - PATCH_PIX / 2.0, cy - PATCH_PIX / 2.0) / r200_pix
    return prof, morph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--reduce", action="store_true")
    args = ap.parse_args()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    if args.reduce:
        shards = sorted(glob.glob(str(SHARD_DIR / "prof_design*.npz")))
        ids = [int(Path(s).stem.split("design")[1]) for s in shards]
        order = np.argsort(ids)
        prof = np.stack([np.load(shards[k])["prof"] for k in order])    # (n_design, N, nf, nb)
        morph = np.stack([np.load(shards[k])["morph"] for k in order])
        out = OUT_ROOT_DEFAULT / "thermo_profiles.npz"
        np.savez(out, prof=prof, morph=morph,
                 field_names=np.array(FIELD_NAMES, dtype=object),
                 morph_names=np.array(MORPH_NAMES, dtype=object),
                 rbin_mid=RBIN_MID, rbin_edges=RBIN_EDGES,
                 design_id=np.array(sorted(ids)))
        print(f"wrote {out}  prof shape {prof.shape}  morph shape {morph.shape}")
        return

    halos = load_cv_halos()
    R200 = np.asarray(halos["R200"], float)
    maps = sorted(glob.glob(str(OUT_ROOT_DEFAULT / "maps" / "gen_design*.npz")))
    mine = maps[args.chunk_id::args.n_chunks]
    print(f"chunk {args.chunk_id}/{args.n_chunks}: {len(mine)} design files")
    for p in mine:
        d = int(Path(p).stem.split("design")[1])
        shard = SHARD_DIR / f"prof_design{d:04d}.npz"
        if shard.exists():
            continue
        zf = np.load(p)
        gen = zf["generated"] if "generated" in zf.files else zf[zf.files[0]]
        assert gen.shape[0] == len(R200), f"design {d}: halo count {gen.shape[0]} != {len(R200)}"
        prof, morph = reduce_one_design(gen.astype(np.float32), R200)
        np.savez(shard, prof=prof, morph=morph)
        print(f"  design {d:04d}  median P-core(<0.15R200)="
              f"{np.nanmedian(prof[:, 3, 0]):.3e}")


if __name__ == "__main__":
    main()
