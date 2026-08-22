"""Per-halo radial profiles in native r/R200c bins, fid + truth, snap 096.

The shipped profiles/{fid,truth}_snap096.npz store ALREADY-STACKED mean profiles
(4 mass bins x 18 radii, no per-halo axis), binned on the legacy r/(0.659*r200)
coordinate with an annulus background subtraction — so the fig-6 overhaul
(author 2026-08-05: all mass bins, per-halo medians with real percentile bands,
residual sub-panels) cannot be built from them. This script re-reduces the
composite slabs into a PER-HALO product:

  profiles/perhalo_{fid,truth}_snap096.npz
    prof_{DM,gas,star,y} : (n_halos, n_rbins)  mean pixel value per annulus
                           (RAW — no background subtraction; masses Msun/h per
                           px, y dimensionless), NaN where the annulus holds
                           no pixel (small halos, inner radii)
    npix                 : (n_halos, n_rbins)  pixels per annulus
    x_edges              : (n_rbins+1,) annulus edges in r/R200c (native — the
                           0.659 constant of the legacy grid is gone)
    M, r200              : per-halo DMO mass [Msun/h] and R200c [comoving Mpc/h]
    z, snap              : carried from the shipped stacked cache

Guards: fid and truth slabs must carry identical halo lists (paired reduction),
and the new per-halo means, restacked with the legacy recipe, are NOT expected
to equal the old cache (different radial coordinate + no bg subtraction), so no
bit-exact gate here — instead the halo count must match the atlas (2933).

    python papers/01_pipeline/build_perhalo_profiles.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
OUT = SCI / "profiles"
SRC = {"fid": CEPH / "bind_lightcone_tng", "truth": SCI / "runs/truth/run_0000"}
SNAP = 96

PIX = 6.25 / 128.0                       # comoving Mpc/h per patch pixel
X_EDGES = np.geomspace(0.05, 3.0, 19)    # annulus edges in r/R200c, 18 log bins
N_X = len(X_EDGES) - 1
CHANNELS = ("DM", "gas", "star", "y")    # gen[:,0..2] + thermo[:,0]


def reduce_slab(path: Path) -> dict | None:
    d = np.load(path)
    if "generated_patches" not in d.files or int(d["n_halos"]) == 0:
        return None
    gen = d["generated_patches"].astype(np.float64)      # (n,3,P,P) Msun/h
    thr = d["thermo_patches"].astype(np.float64)         # (n,4,P,P); [0]=y
    M = np.asarray(d["halo_masses"], float)
    r200 = np.asarray(d["halo_r200"], float)
    n, _, P, _ = gen.shape
    cen = P // 2
    yy, xx = np.mgrid[0:P, 0:P]
    rr = (np.hypot(xx - cen, yy - cen) * PIX).ravel()    # Mpc/h from centre

    maps = {"DM": gen[:, 0], "gas": gen[:, 1], "star": gen[:, 2], "y": thr[:, 0]}
    prof = {k: np.full((n, N_X), np.nan) for k in CHANNELS}
    npix = np.zeros((n, N_X), dtype=np.int64)
    for i in range(n):
        idx = np.digitize(rr / r200[i], X_EDGES) - 1
        ok = (idx >= 0) & (idx < N_X)
        cnt = np.bincount(idx[ok], minlength=N_X)
        npix[i] = cnt
        nz = cnt > 0
        for k in CHANNELS:
            s = np.bincount(idx[ok], weights=maps[k][i].ravel()[ok], minlength=N_X)
            prof[k][i, nz] = s[nz] / cnt[nz]
    out = {f"prof_{k}": prof[k] for k in CHANNELS}
    out.update(M=M, r200=r200, npix=npix)
    return out


def main() -> None:
    ref = np.load(OUT / f"fid_snap{SNAP:03d}.npz")        # z/snap carried over
    cats = {}
    for run, root in SRC.items():
        slabs = sorted((root / f"snap_{SNAP:03d}").glob("composite_slab*.npz"))
        assert slabs, f"no composite slabs under {root}/snap_{SNAP:03d}"
        parts = [r for r in (reduce_slab(s) for s in slabs) if r is not None]
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        assert len(cat["M"]) == 2933, f"{run}: {len(cat['M'])} halos != atlas 2933"
        cat["x_edges"] = X_EDGES
        cat["z"] = ref["z"]
        cat["snap"] = ref["snap"]
        cats[run] = cat

    # paired reduction: identical halo lists in identical order on both sides
    assert np.allclose(cats["fid"]["M"], cats["truth"]["M"]), "halo lists differ"
    assert np.allclose(cats["fid"]["r200"], cats["truth"]["r200"])
    assert np.array_equal(cats["fid"]["npix"], cats["truth"]["npix"])

    for run, cat in cats.items():
        f = OUT / f"perhalo_{run}_snap{SNAP:03d}.npz"
        tmp = f.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **cat)
        tmp.replace(f)
        fin = np.isfinite(cat["prof_y"]).mean(0)
        print(f"[{run}] wrote {f.name}: {len(cat['M'])} halos x {N_X} annuli "
              f"(x = {X_EDGES[0]:.2f}-{X_EDGES[-1]:.1f} r/R200c); "
              f"finite fraction per annulus min/med {fin.min():.2f}/{np.median(fin):.2f}")


if __name__ == "__main__":
    main()
