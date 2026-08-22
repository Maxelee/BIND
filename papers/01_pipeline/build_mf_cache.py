"""Build one shard of the extended-grid Minkowski cache (see mf_cache.py).

One invocation = one (side, source plane) pair = one independent task, which is what
makes the whole build trivially parallel under disBatch.

    python build_mf_cache.py --side bind  --plane 0
    python build_mf_cache.py --merge            # after all 10 shards exist
    python build_mf_cache.py --verify           # check against the released 29-bin grid
"""
import argparse
import time

import numpy as np

import mf_cache as mc


def verify() -> int:
    """The recomputation must reproduce the released V0/V1/V2 on their own grid.

    Without this, a convention drift in nongaussian_stats (smoothing scale, S/N
    normalisation) would silently move every MF curve in fig 4.

    Each side is checked against the reference that is actually current:
      truth -> nongaussian_stats.npz                   (reproduces bit-exactly)
      bind  -> paired_perreal_fid.npz realization mean (reproduces bit-exactly)
    runs/bind/run_0000/nongaussian_stats.npz is deliberately NOT the bind reference.
    Its V0/V1/V2 are stale with respect to the current bind kappa maps: this routine
    and the independently-built per-realization cache agree to 0.0 and both differ
    from that summary file by 3.6e-4 / 2.5e-3 / 5.6e-3 (V0/V1/V2, peak-relative).
    The discrepancy is reported below so it cannot be forgotten.
    """
    from bind.inference.stats import nongaussian_stats

    ok = True
    n29 = 29
    assert np.allclose(mc.MF_NU8[:n29], np.linspace(-3, 4, n29)), \
        "MF_NU8 no longer contains the released grid as a prefix"
    for side in mc.SIDES:
        ng = np.load(mc.SCI / f"runs/{side}/run_0000/nongaussian_stats.npz")
        a = np.load(mc.SCI / f"runs/{side}/run_0000/kappa_maps.npz")
        K = a["kappa"][:, 1].astype(np.float32)      # z_s = 1, the working plane
        del a
        out = nongaussian_stats(K[:, None], fov_deg=mc.FOV_DEG,
                                smoothing_scales_arcmin=(mc.SMOOTH_ARCMIN,),
                                mf_thresholds=ng["mf_nu"])
        ref, src = ng, "nongaussian_stats"
        if side == "bind":
            pr = np.load(mc.SCI / "runs/bind/run_0000/paired_perreal_fid.npz")
            ref = {k: pr[k][:, 1, :].mean(0) for k in mc.KEYS}
            src = "paired_perreal_fid"
        for k in mc.KEYS:
            got = np.asarray(out[k])[0]
            want = ref[k] if side == "bind" else ref[k][1]
            close = np.max(np.abs(got - want))/max(np.max(np.abs(want)), 1e-30) < 1e-6
            ok &= close
            print(f"  {side:5s} {k} (vs {src}): {'OK' if close else 'MISMATCH'} "
                  f"(max|diff| = {np.max(np.abs(got - want)):.3e})")
            if side == "bind":
                st = np.max(np.abs(want - ng[k][1]))/np.max(np.abs(ng[k][1]))
                print(f"        bind/nongaussian_stats.npz {k} SUPERSEDED "
                      f"(peak-relative offset {st:.2e}) -- not used")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=mc.SIDES)
    ap.add_argument("--plane", type=int)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify:
        return verify()

    if a.merge:
        p = mc.merge()
        d = np.load(p)
        print(f"merged -> {p} ({p.stat().st_size/1e3:.0f} kB)")
        print(f"  keys: {sorted(d.files)}")
        print(f"  bind_V0 {d['bind_V0'].shape}, truth_V0 {d['truth_V0'].shape}, "
              f"nu {d['mf_nu'][0]:.0f}..{d['mf_nu'][-1]:.0f} ({len(d['mf_nu'])})")
        return 0

    if a.side is None or a.plane is None:
        ap.error("need --side and --plane (or --merge / --verify)")
    mc.SHARD_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = mc.compute(a.side, a.plane)
    out = mc.shard_path(a.side, a.plane)
    np.savez_compressed(out, **res)
    print(f"{a.side} plane {a.plane}: {res['V0'].shape} -> {out.name} "
          f"in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
