"""Build one shard of the per-realization field-statistics cache (see field_cache.py).

One invocation = one (product, side) pair = one independent disBatch task.

    python build_field_cache.py --product ky --side bind
    python build_field_cache.py --merge
    python build_field_cache.py --verify     # against independently-built caches
"""
import argparse
import time

import numpy as np

import field_cache as fc


def verify() -> int:
    """Check the recomputation against caches built by a DIFFERENT pipeline.

    Spectra: paired_perreal_fid.npz (BIND) holds per-realization clk produced by the
    lightcone pipeline, so it is a genuine independent reference rather than this code
    checking itself.

    Counts: the reference is peak_counts.npz on BOTH sides, because that is the
    nu_norm='map' convention fig 4 uses (and asserts against). paired_perreal_fid.npz
    is NOT a valid counts reference — its pk/min are the nu_norm='fixed' convention,
    matching peak_counts_nufid.npz to 0.0 and differing from the 'map' counts by up to
    2.2e-2. Checking against it would fail a correct cache.

    Deliberately NOT used as references: Cl_kappa_y.npz / Cl_tau.npz (old XPk_plane
    cross normalization, low by ~1.2e7 and ell-dependently) and, on the BIND side,
    nongaussian_stats.npz (its Minkowski functionals are stale — see mf_cache.py).
    """
    ok = True
    pr = np.load(fc.SCI / "runs/bind/run_0000/paired_perreal_fid.npz")

    got = fc.compute("kk", "bind")["cl"]
    for zi in (0, 1, 4):
        dev = np.max(np.abs(got[:, zi] - pr["clk"][:, zi])) / np.max(np.abs(pr["clk"][:, zi]))
        ok &= dev < 1e-6
        print(f"  kk bind z{zi} vs paired_perreal clk: "
              f"{'OK' if dev < 1e-6 else 'MISMATCH'} ({dev:.2e})")

    bind_bins = np.load(fc.SCI / "runs/bind/run_0000/nongaussian_stats.npz")["pdf_bins"]
    for side in ("bind", "truth"):
        c = fc.compute("counts", side)
        # the PDF grid must be the BIND one on both sides (the two runs' own grids
        # differ: +-0.20863 vs +-0.20745), else the paired residual compares
        # different bins. fig 4 asserts this too; check it before we ship a cache.
        same = np.allclose(c["pdf_bins"], bind_bins)
        ok &= same
        print(f"  counts {side:5s} pdf grid == BIND grid: {'OK' if same else 'MISMATCH'}")
        pcm = np.load(fc._run_dir(side) / "peak_counts.npz")
        assert str(pcm["nu_norm"]) == "map", "peak_counts.npz is no longer nu_norm='map'"
        for key, ref in (("pk", "peak_counts"), ("min", "minima_counts")):
            dev = np.max(np.abs(c[key].mean(0) - pcm[ref])) / np.max(np.abs(pcm[ref]))
            ok &= dev < 1e-6
            print(f"  counts {side:5s} {key} mean vs peak_counts.npz (nu_norm=map): "
                  f"{'OK' if dev < 1e-6 else 'MISMATCH'} ({dev:.2e})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=sorted({p for p, _ in fc.TASKS}))
    ap.add_argument("--side", choices=("bind", "truth"))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify:
        return verify()
    if a.merge:
        p = fc.merge()
        d = np.load(p)
        print(f"merged -> {p} ({p.stat().st_size/1e6:.1f} MB)")
        for k in sorted(d.files):
            print(f"   {k:14s} {d[k].shape}")
        return 0
    if a.product is None or a.side is None:
        ap.error("need --product and --side (or --merge / --verify)")
    if (a.product, a.side) not in fc.TASKS:
        ap.error(f"({a.product}, {a.side}) is not a task; "
                 f"tau products exist on the bind side only")

    fc.SHARD_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = fc.compute(a.product, a.side)
    out = fc.shard_path(a.product, a.side)
    np.savez_compressed(out, **res)
    shp = res["cl"].shape if "cl" in res else res["pk"].shape
    print(f"{a.product}/{a.side}: {shp} -> {out.name} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
