"""Build the LSST-Y10 noisy-statistics cache (see noisy_grid.py for the recipe).

Targets:
    fid     bind_lightcone_tng/kappa_maps.npz               (550 real, fiducial BIND)
    truth   bind_science/runs/truth/run_0000/kappa_maps.npz (550 real, hydro truth)
    dmo     bind_science/runs/dmo/run_0000/kappa_maps.npz   (whatever it currently
                                                              holds -- read, not assumed)
    run_NNNN  bind_sb35/runs/run_NNNN/kappa_maps.npz         (256 Sobol runs, 50 real each)

Order of operations (fid/truth/dmo are heavy -- 550 realizations -- so they are
chunked by realization range; Sobol runs are cheap enough for one task each):

    # 1. ONCE, from the fiducial target only -- every other build requires this:
    python build_noisy_cache.py --kappa-rms

    # 2a. Sobol runs (one task each, no chunking needed):
    python build_noisy_cache.py --run run_0007

    # 2b. fid/truth/dmo, chunked (each chunk is one disBatch task, <=30 min):
    python build_noisy_cache.py --run fid --real-start 0 --real-end 138
    python build_noisy_cache.py --run fid --real-start 138 --real-end 276
    ...
    python build_noisy_cache.py --merge-chunks fid      # stitches the chunks back together

    # 3. after all 256 Sobol shards exist:
    python build_noisy_cache.py --assemble    # -> emulator_dataset_nu05n.npz
    python build_noisy_cache.py --verify

Smoke test (local, small): --max-real N caps the realization count of a --run build
(shorthand for --real-end N with --real-start 0); used for both fast --kappa-rms
passes and small fid/truth/dmo validation shards.

Nothing released, and nothing in nu05_shards/ (the NOISELESS cache), is touched.
Shards land in bind_sb35/nu05n_shards/; the assembled dataset is a NEW file.
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np

import noisy_grid as ng


def shard_path(target: str, real_start: int | None = None, real_end: int | None = None) -> Path:
    """Full-target shard name if no range is given, else a chunk-named shard."""
    if real_start is None and real_end is None:
        return ng.SHARD_DIR / f"{target}.npz"
    rs = 0 if real_start is None else real_start
    re_ = "end" if real_end is None else f"{real_end:05d}"
    return ng.SHARD_DIR / f"{target}_chunk_{rs:05d}_{re_}.npz"


def build_kappa_rms(n_real: int | None) -> Path:
    t0 = time.time()
    rms, nr = ng.compute_kappa_rms(n_real=n_real)
    out = ng.save_kappa_rms(rms, nr)
    print(f"kappa_rms (n_real={nr}): {np.array2string(rms, precision=5)} "
          f"-> {out} in {time.time() - t0:.0f}s")
    return out


def build_run(target: str, real_start: int, real_end: int | None) -> Path:
    full_range = real_start == 0 and real_end is None
    out = shard_path(target, None, None) if full_range else shard_path(target, real_start, real_end)
    if out.exists():
        print(f"{out.name}: shard exists, skipping")
        return out
    # Sobol runs never carry per-realization draws (256 of them -> space); the
    # fiducial trio (fid/truth/dmo) always do (needed for the Eq.-8 covariance).
    per_real = target in ng.TARGETS
    t0 = time.time()
    res = ng.compute(target, real_start=real_start, real_end=real_end, per_real=per_real)
    ng.SHARD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: v for k, v in res.items() if v is not None})
    print(f"{target} [{res['real_start']}:{res['real_end']}) "
          f"({res['n_real']} real) -> {out.name} in {time.time() - t0:.0f}s")
    return out


def merge_chunks(target: str) -> Path:
    """Concatenate a target's realization-range chunk shards into one full shard."""
    pattern = str(ng.SHARD_DIR / f"{target}_chunk_*.npz")
    chunks = glob.glob(pattern)
    if not chunks:
        raise FileNotFoundError(f"no chunk shards match {pattern}")
    loaded = [np.load(c) for c in chunks]
    # order (and sanity-check contiguity) by the real_start/real_end stored INSIDE each
    # shard, not the filename -- authoritative regardless of how the file was named.
    starts = [int(d["real_start"]) for d in loaded]
    ends = [int(d["real_end"]) for d in loaded]
    order = np.argsort(starts)
    starts = [starts[i] for i in order]; ends = [ends[i] for i in order]; loaded = [loaded[i] for i in order]
    for a, b in zip(ends[:-1], starts[1:]):
        if a != b:
            raise ValueError(f"{target}: chunk gap/overlap at real={a}->{b} "
                             f"(chunks: {list(zip(starts, ends))})")
    real_keys = [f"{k}_real" for k in ng.KEYS] + ["cl_kappa_real"]
    store: dict[str, np.ndarray] = {}
    for k in real_keys:
        store[k] = np.concatenate([d[k] for d in loaded], axis=0)
    store["nu"] = loaded[0]["nu"]
    store["cl_ell"] = loaded[0]["cl_ell"]
    store["kappa_rms"] = loaded[0]["kappa_rms"]
    for d in loaded[1:]:
        assert np.array_equal(d["kappa_rms"], store["kappa_rms"]), \
            f"{target}: chunks were built against DIFFERENT kappa_rms caches"
    store["real_start"] = np.array(starts[0]); store["real_end"] = np.array(ends[-1])
    store["n_real"] = np.array(ends[-1] - starts[0])
    store["target"] = np.array(target)
    # recompute the mean(+err) summary from the now-complete realization cube
    for k in ng.KEYS:
        store[k] = store[f"{k}_real"].mean(0)
        if k in ("peak_counts", "minima_counts"):
            store[f"{k}_err"] = store[f"{k}_real"].std(0) / np.sqrt(store["n_real"])
    store["cl_kappa"] = store["cl_kappa_real"].mean(0)
    store["cl_kappa_err"] = store["cl_kappa_real"].std(0) / np.sqrt(store["n_real"])

    out = shard_path(target)
    np.savez_compressed(out, **store)
    print(f"{target}: merged {len(chunks)} chunks ({starts[0]}..{ends[-1]}, "
          f"{store['n_real']} real) -> {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    return out


def _sobol_run_ids() -> list[int]:
    d = np.load(ng.DATASET_IN, allow_pickle=True)
    return [int(r) for r in d["run_ids"]]


def assemble() -> Path:
    """Six noisy nu statistics + noisy cl_kappa for the 256 Sobol runs -> one dataset.

    A small, SELF-CONTAINED dataset (not a copy-and-patch of the big noiseless
    emulator_dataset*.npz -- cl_kappa here is a per-plane AUTO spectrum, (n,5,724),
    a DIFFERENT shape from that dataset's (n,5,5,724) tomographic cl_kappa matrix,
    so reusing the same key inside a copied file would be actively misleading).
    run_ids/X_native/X_unit/param_names are copied from DATASET_IN so this dataset
    stays joinable against the noiseless one.
    """
    run_ids = _sobol_run_ids()
    n = len(run_ids)
    missing = [r for r in run_ids if not shard_path(f"run_{r:04d}").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{n} Sobol shards missing, e.g. "
            f"{[f'run_{r:04d}' for r in missing[:5]]}")

    src = np.load(ng.DATASET_IN, allow_pickle=True)
    store: dict[str, np.ndarray] = {
        "run_ids": src["run_ids"],
        "X_native": src["X_native"], "X_unit": src["X_unit"],
        "param_names": src["param_names"],
    }

    n_nu = len(ng.NU)
    for key in ng.KEYS:
        val = np.empty((n, ng.N_PLANES, n_nu))
        er = np.empty_like(val) if key in ("peak_counts", "minima_counts") else None
        valid = np.ones(n, bool)
        for i, r in enumerate(run_ids):
            s = np.load(shard_path(f"run_{r:04d}"))
            val[i] = s[key]
            if er is not None:
                er[i] = s[f"{key}_err"]
        store[f"t__{key}__value"] = val
        store[f"t__{key}__valid"] = valid
        if er is not None:
            store[f"t__{key}__err"] = er
    for key, axis in (("pdf", "pdf_bins"), ("peak_counts", "nu"), ("minima_counts", "nu"),
                      ("mf_v0", "mf_nu"), ("mf_v1", "mf_nu"), ("mf_v2", "mf_nu")):
        store[f"a__{key}__{axis}"] = ng.NU

    ell0 = np.load(shard_path(f"run_{run_ids[0]:04d}"))["cl_ell"]
    cl = np.empty((n, ng.N_PLANES, len(ell0)))
    cl_er = np.empty_like(cl)
    for i, r in enumerate(run_ids):
        s = np.load(shard_path(f"run_{r:04d}"))
        assert np.array_equal(s["cl_ell"], ell0), f"run_{r:04d}: ell grid mismatch"
        cl[i] = s["cl_kappa"]; cl_er[i] = s["cl_kappa_err"]
    store["t__cl_kappa__value"] = cl
    store["t__cl_kappa__err"] = cl_er
    store["t__cl_kappa__valid"] = np.ones(n, bool)
    store["a__cl_kappa__ell"] = ell0

    store["kappa_rms"] = ng.load_kappa_rms()
    store["noisy_grid_note"] = np.array(
        f"LSST-Y10 shape noise (Lee+22 arXiv:2201.08320 Eq.7: sigma_e={ng.SIGMA_E}, "
        f"n_gal={ng.NGAL_ARCMIN2}/arcmin^2, sigma_pix={ng.NOISE_SIGMA_PIX:.6f}) added "
        f"BEFORE Gaussian smoothing (Eq.6: theta_G={ng.THETA_G_ARCMIN}' 1/e radius -> "
        f"filter sigma={ng.SMOOTH_SIGMA_ARCMIN:.6f}'={ng.SMOOTH_SIGMA_PIX:.5f}px), on "
        "the noisy smoothed map, nu = (map-map.mean())/kappa_rms[plane] with "
        "kappa_rms measured once from the fiducial suite's own noisy smoothed maps "
        "(see kappa_rms array here). peak_counts/minima_counts/pdf are histograms on "
        "NU_EDGES (nu_grid.py); mf_v0/1/2 are threshold functionals AT NU; cl_kappa "
        "is the per-plane AUTO spectrum of the noisy smoothed map (NOT the "
        "tomographic (5,5,ell) matrix the noiseless emulator_dataset*.npz carries "
        "under the same name -- shapes differ, do not conflate the two files). "
        f"Noise/seeding is per-target-independent (see noisy_grid.noise_rng); this "
        "file covers the 256 Sobol runs only -- fid/truth/dmo per-realization noisy "
        "shards live separately in nu05n_shards/{fid,truth,dmo}.npz.")
    np.savez_compressed(ng.DATASET_OUT, **store)
    return ng.DATASET_OUT


def verify() -> int:
    """NaN-aware sanity + completeness checks (mirrors build_nu_cache.py's --verify)."""
    ok = True
    if not ng.DATASET_OUT.exists():
        print(f"  MISSING: {ng.DATASET_OUT}"); return 1
    b = np.load(ng.DATASET_OUT, allow_pickle=True)

    run_ids = _sobol_run_ids()
    n = len(run_ids)
    same_ids = np.array_equal(np.asarray(b["run_ids"]), np.asarray(run_ids))
    ok &= same_ids
    print(f"  run_ids matches DATASET_IN's 256 Sobol runs: {'OK' if same_ids else 'FAIL'}")

    for key in ng.KEYS:
        v = b[f"t__{key}__value"]
        fin = float(np.isfinite(v).mean())
        shape_ok = v.shape == (n, ng.N_PLANES, len(ng.NU))
        ok &= shape_ok
        print(f"  {key:15s} {str(v.shape):18s} finite {100*fin:6.2f}% "
              f"shape {'OK' if shape_ok else 'FAIL'}")
    cl = b["t__cl_kappa__value"]
    cl_fin = float(np.isfinite(cl).mean())
    cl_shape_ok = cl.shape[:2] == (n, ng.N_PLANES)
    ok &= cl_shape_ok
    print(f"  {'cl_kappa':15s} {str(cl.shape):18s} finite {100*cl_fin:6.2f}% "
          f"shape {'OK' if cl_shape_ok else 'FAIL'}")

    ax_ok = True
    for key, axis in (("pdf", "pdf_bins"), ("peak_counts", "nu"), ("mf_v0", "mf_nu")):
        ax_ok &= bool(np.allclose(b[f"a__{key}__{axis}"], ng.NU, equal_nan=True))
    ok &= ax_ok
    print(f"  all nu axes == 22 bin centres of NU_EDGES: {'OK' if ax_ok else 'FAIL'}")

    # spot-check: reloading a couple of raw shards must reproduce the assembled arrays exactly
    for r in run_ids[:3] + run_ids[-1:]:
        s = np.load(shard_path(f"run_{r:04d}"))
        i = run_ids.index(r)
        same = np.array_equal(s["peak_counts"], b["t__peak_counts__value"][i], equal_nan=True)
        ok &= same
        print(f"  run_{r:04d} peak_counts shard == assembled row: {'OK' if same else 'FAIL'}")

    krms_ok = "kappa_rms" in b.files and len(b["kappa_rms"]) == ng.N_PLANES
    ok &= krms_ok
    print(f"  kappa_rms metadata present ({ng.N_PLANES} planes): {'OK' if krms_ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="target: fid | truth | dmo | run_NNNN")
    p.add_argument("--real-start", type=int, default=0)
    p.add_argument("--real-end", type=int, default=None)
    p.add_argument("--max-real", type=int, default=None,
                   help="shorthand for --real-end (smoke tests); ignored if --real-end given")
    p.add_argument("--kappa-rms", action="store_true",
                   help="build/refresh the fiducial kappa_rms cache (run this FIRST)")
    p.add_argument("--merge-chunks", metavar="TARGET",
                   help="stitch a fid/truth/dmo target's realization chunks into one shard")
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()

    if a.kappa_rms:
        n_real = a.max_real if a.real_end is None else a.real_end
        build_kappa_rms(n_real)
        return 0
    if a.merge_chunks:
        merge_chunks(a.merge_chunks)
        return 0
    if a.assemble:
        t0 = time.time()
        out = assemble()
        print(f"assembled -> {out} ({out.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.0f}s")
        return 0
    if a.verify:
        return verify()
    if not a.run:
        p.error("need --run, --kappa-rms, --merge-chunks, --assemble or --verify")

    real_end = a.real_end if a.real_end is not None else a.max_real
    build_run(a.run, a.real_start, real_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
