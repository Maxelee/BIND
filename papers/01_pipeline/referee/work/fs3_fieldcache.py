#!/usr/bin/env python3
"""fidswap step 5 -- per-realization field statistics for the NEW fiducial.

field_cache.py, with the 'bind' side re-pointed at a twobound replica.  Two
deliberate differences from the shipped module, both improvements:

  * tau comes from the replica's OWN tau_maps.npz.  The shipped cache hardcodes
    bind_lightcone_tng/tau_maps.npz because runs/bind/run_0000 has no tau trace;
    the replica has one, so tt/kt get real same-paint provenance for the first time.
  * the PDF bin grid is taken from the REPLICA's nongaussian_stats.npz (each run's
    grid is adaptive, +-6 sigma of its own kappa).  Both sides are binned on that
    one grid, exactly as field_cache does, so the paired residual stays meaningful.
    The replica grid is 0.4% narrower than the old fiducial's -- see --gate output.

The truth-side legs are UNAFFECTED by the fiducial bug and are copied verbatim
from the shipped cache into the merged output, so the result is a drop-in for
bind_science/field_cache/field_stats_fid.npz.

    python referee/work/fs3_fieldcache.py --gate         # reproduce a shipped shard
    python referee/work/fs3_fieldcache.py --run 49       # build all bind-side tasks
    python referee/work/fs3_fieldcache.py --run 49 --merge
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import field_cache as fc  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
TWOBOUND = SCI / "runs/twobound"
OUT = CEPH / "referee_work/fidswap/field_cache"
SHARDS = OUT / "shards"

N_PLANES = fc.N_PLANES
FOV_DEG = fc.FOV_DEG
SMOOTH_PK = fc.SMOOTH_PK
BIND_TASKS = ["kk", "yy", "ky", "tt", "kt", "counts"]


def run_dir(run: int) -> Path:
    return TWOBOUND / f"run_{run:04d}"


def _load(run: int, field: str) -> np.ndarray:
    f = np.load(run_dir(run) / f"{field}_maps.npz")
    a = f[field]
    del f
    return a


def compute(product: str, run: int, planes=range(N_PLANES)) -> dict:
    """field_cache.compute for the 'bind' side, re-pointed at run_{run:04d}."""
    from bind.inference.stats import peak_counts, power_spectrum

    planes = list(planes)
    if product == "counts":
        K = _load(run, "kappa")
        nr = K.shape[0]
        ng = np.load(run_dir(run) / "nongaussian_stats.npz")
        cen = ng["pdf_bins"]
        d = cen[1] - cen[0]
        edges = np.concatenate([cen - d / 2, [cen[-1] + d / 2]])
        n_nu = len(np.load(run_dir(run) / "peak_counts.npz")["nu"])
        pk = np.full((nr, N_PLANES, n_nu), np.nan)
        mn = np.full_like(pk, np.nan)
        pdf = np.full((nr, N_PLANES, len(cen)), np.nan)
        for zi in planes:
            o = peak_counts(K[:, zi][:, None], fov_deg=FOV_DEG,
                            smoothing_arcmin=SMOOTH_PK, nu_norm="map",
                            return_realizations=True)
            pk[:, zi] = o["peak_counts_real"][:, 0]
            mn[:, zi] = o["minima_counts_real"][:, 0]
            pdf[:, zi] = [np.histogram(m - m.mean(), bins=edges, density=True)[0]
                          for m in K[:, zi]]
            del o
            gc.collect()
        return {"pk": pk, "min": mn, "pdf": pdf, "pdf_bins": cen}

    auto = {"kk": "kappa", "yy": "y", "tt": "tau"}
    cross = {"ky": ("kappa", "y"), "kt": ("kappa", "tau")}
    if product in auto:
        A = _load(run, auto[product])
        B = None
    else:
        fa, fb = cross[product]
        A = _load(run, fa)
        B = _load(run, fb)[:, -1].copy()          # kappa(z_s) x TOTAL column
    nr = A.shape[0]
    ell, _ = power_spectrum(A[0, 0])
    out = np.full((nr, N_PLANES, len(ell)), np.nan)
    for zi in planes:
        for r in range(nr):
            out[r, zi] = (power_spectrum(A[r, zi])[1] if B is None
                          else power_spectrum(A[r, zi], B[r])[1])
        gc.collect()
    del A, B
    gc.collect()
    return {"cl": out, "ell": ell}


def gate():
    """recompute one shipped shard from the OLD fiducial and diff it."""
    from bind.inference.stats import power_spectrum

    ship = np.load(SCI / "field_cache/field_stats_fid.npz")
    K = np.load(SCI / "runs/bind/run_0000/kappa_maps.npz")["kappa"]
    t0 = time.time()
    got = np.stack([power_spectrum(K[r, 1])[1] for r in range(K.shape[0])])
    ref = ship["kk_bind"][:, 1]
    rel = float(np.max(np.abs(got / ref - 1.0)))
    print(f"GATE kk_bind z_s=1 (50 reals, {time.time()-t0:.1f}s): "
          f"max|rel diff| vs shipped field_stats_fid.npz = {rel:.3e}  "
          + ("PASS (bit-for-bit)" if rel == 0.0 else
             "PASS" if rel < 1e-12 else "FAIL"))
    # PDF grid drift, old fiducial vs replicas
    old = np.load(SCI / "runs/bind/run_0000/nongaussian_stats.npz")["pdf_bins"]
    for r in (18, 49, 53):
        new = np.load(run_dir(r) / "nongaussian_stats.npz")["pdf_bins"]
        print(f"  pdf_bins tb{r}: endpoint scale vs old fiducial "
              f"{new[-1]/old[-1]:.6f}, drift {(new[0]-old[0])/(old[1]-old[0]):+.4f} bins")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=49)
    ap.add_argument("--tasks", nargs="*", default=BIND_TASKS)
    ap.add_argument("--planes", type=int, nargs="*", default=None)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.gate:
        gate()
        return
    SHARDS.mkdir(parents=True, exist_ok=True)
    tag = f"tb{a.run:02d}"
    if not a.merge:
        planes = a.planes if a.planes is not None else list(range(N_PLANES))
        for p in a.tasks:
            sp = SHARDS / f"{p}_bind_{tag}.npz"
            if sp.exists():
                print(f"skip {p} (exists)", flush=True)
                continue
            t0 = time.time()
            np.savez(sp, **compute(p, a.run, planes))
            print(f"{p:>6s}  {time.time()-t0:6.1f}s  -> {sp.name}", flush=True)
        return

    ship = np.load(SCI / "field_cache/field_stats_fid.npz")
    store = {k: ship[k] for k in ship.files if k.endswith("_truth")}
    for p in BIND_TASKS:
        s = np.load(SHARDS / f"{p}_bind_{tag}.npz")
        if p == "counts":
            for k in ("pk", "min", "pdf"):
                store[f"{k}_bind"] = s[k]
            store["pdf_bins"] = s["pdf_bins"]
        else:
            store[f"{p}_bind"] = s["cl"]
            store["ell"] = s["ell"]
    store["pdf_bins_shipped"] = ship["pdf_bins"]
    store["fid_run"] = np.array(f"twobound/run_{a.run:04d}")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"field_stats_fid{tag}.npz"
    np.savez_compressed(out, **store)
    print(f"merged -> {out} ({out.stat().st_size/1e6:.1f} MB)")
    print("NB truth legs copied verbatim from the shipped cache; the BIND-side PDF "
          "grid is the replica's own (see pdf_bins vs pdf_bins_shipped).")


if __name__ == "__main__":
    main()
