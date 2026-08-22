"""Recompute every nu statistic on the canonical 0.5-wide-bin grid, then assemble a dataset.

    python build_nu_cache.py --run run_0007            # one Sobol shard
    python build_nu_cache.py --run bind --sci          # the fiducial pair lives elsewhere
    python build_nu_cache.py --assemble                # -> emulator_dataset_nu05.npz
    python build_nu_cache.py --verify                  # sanity-check the new dataset

Grid: NU_EDGES = 23 edges spaced 0.5 from -3 to 8; NU = the 22 bin centres (-2.75 .. 7.75).
See nu_grid.py's docstring for the edges-vs-centres distinction (histograms vs MF
thresholds). This supersedes the earlier linspace(-3, 8, 45) grid (nu8_shards/,
emulator_dataset_nu8.npz), which is left untouched alongside this one.

Nothing released is overwritten: shards go to bind_sb35/nu05_shards/, and --assemble writes a
NEW dataset file, copying every non-nu array from emulator_dataset_xpkfix.npz unchanged.
"""
import argparse
import time

import numpy as np

import nu_grid as ng


def shard(run: str) -> "ng.Path":
    return ng.SHARD_DIR / f"{run}.npz"


def kappa_path(run: str, sci: bool):
    return (ng.SCI / f"runs/{run}/run_0000/kappa_maps.npz") if sci else \
           (ng.SB35 / f"runs/{run}/kappa_maps.npz")


def assemble() -> "ng.Path":
    """Copy the released dataset, replacing ONLY the six nu statistics and their axes."""
    src = np.load(ng.DATASET_IN, allow_pickle=True)
    run_ids = src["run_ids"]
    assert len(run_ids) == 256, (
        f"expected 256 Sobol runs in {ng.DATASET_IN.name}, found {len(run_ids)}")
    store = {k: src[k] for k in src.files}

    missing = [f"run_{int(r):04d}" for r in run_ids
               if not shard(f"run_{int(r):04d}").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} shards missing, e.g. {missing[:5]}")

    n = len(run_ids)
    for key in ng.KEYS:
        val = np.empty((n, ng.N_PLANES, len(ng.NU)))
        er = np.empty_like(val) if key in ("peak_counts", "minima_counts") else None
        for i, r in enumerate(run_ids):
            s = np.load(shard(f"run_{int(r):04d}"))
            val[i] = s[key]
            if er is not None:
                er[i] = s[f"{key}_err"]
        store[f"t__{key}__value"] = val
        store[f"t__{key}__valid"] = np.ones(n, bool)
        if er is not None:
            store[f"t__{key}__err"] = er
    # axes: every nu-domain statistic now points at the SAME array
    for key, axis in (("pdf", "pdf_bins"), ("peak_counts", "nu"), ("minima_counts", "nu"),
                      ("mf_v0", "mf_nu"), ("mf_v1", "mf_nu"), ("mf_v2", "mf_nu")):
        store[f"a__{key}__{axis}"] = ng.NU
    store["nu_grid_note"] = np.array(
        "peak_counts/minima_counts/pdf recomputed as histograms on NU_EDGES "
        "(23 edges, width 0.5, -3..8); mf_v0/mf_v1/mf_v2 recomputed as threshold "
        "functionals evaluated AT the 22 bin centres NU (-2.75..7.75); all other "
        "arrays copied unchanged from emulator_dataset_xpkfix.npz (256 runs). "
        "peak_R/peak_y keep their own 14-bin nu axis. Supersedes the earlier "
        "linspace(-3,8,45) grid in emulator_dataset_nu8.npz.")
    np.savez_compressed(ng.DATASET_OUT, **store)
    return ng.DATASET_OUT


def verify() -> int:
    """The new dataset must differ ONLY where intended."""
    a = np.load(ng.DATASET_IN, allow_pickle=True)
    b = np.load(ng.DATASET_OUT, allow_pickle=True)
    changed = {f"t__{k}__value" for k in ng.KEYS} | {f"t__{k}__err" for k in
               ("peak_counts", "minima_counts")} | {f"t__{k}__valid" for k in ng.KEYS} | {
        "a__pdf__pdf_bins", "a__peak_counts__nu", "a__minima_counts__nu",
        "a__mf_v0__mf_nu", "a__mf_v1__mf_nu", "a__mf_v2__mf_nu", "nu_grid_note"}
    ok = True
    for k in a.files:
        if k in changed:
            continue
        if k not in b.files:
            print(f"  MISSING in new dataset: {k}"); ok = False; continue
        # NaN-aware: (x == y).all() is False for identical arrays holding NaN
        # (bit us 2026-08-03: wst/peak_R carry all-NaN rows for uncovered runs)
        same = (a[k].dtype == object or
                np.array_equal(a[k], b[k], equal_nan=(a[k].dtype.kind == "f")))
        if not same:
            print(f"  UNEXPECTEDLY CHANGED: {k}"); ok = False
    print(f"  untouched arrays identical: {'OK' if ok else 'FAIL'} "
          f"({len(set(a.files) - changed)} checked)")
    for k in ng.KEYS:
        v = b[f"t__{k}__value"]
        fin = np.isfinite(v).mean()
        print(f"  {k:15s} {str(v.shape):16s} finite {100*fin:6.2f}%")
        ok &= v.shape[-1] == len(ng.NU)
    ax_ok = True
    for key, axis in (("pdf", "pdf_bins"), ("peak_counts", "nu"), ("mf_v0", "mf_nu")):
        ax_ok &= bool(np.allclose(b[f"a__{key}__{axis}"], ng.NU))
    ok &= ax_ok
    print(f"  all nu axes == 22 bin centres of arange(-3,8.5,0.5): {'OK' if ax_ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run")
    p.add_argument("--sci", action="store_true",
                   help="run dir lives under bind_science/runs/<run>/run_0000")
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()

    if a.assemble:
        t0 = time.time()
        out = assemble()
        print(f"assembled -> {out} ({out.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.0f}s")
        return 0
    if a.verify:
        return verify()
    if not a.run:
        p.error("need --run, --assemble or --verify")

    ng.SHARD_DIR.mkdir(parents=True, exist_ok=True)
    name = a.run if not a.sci else f"sci_{a.run}"
    out = shard(name)
    if out.exists():
        print(f"{name}: shard exists, skipping"); return 0
    t0 = time.time()
    # the fiducial pair also needs per-realization draws (fig 4's paired bands)
    res = ng.compute(kappa_path(a.run, a.sci), per_real=a.sci)
    np.savez_compressed(out, **res)
    print(f"{name}: {res['peak_counts'].shape} -> {out.name} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
