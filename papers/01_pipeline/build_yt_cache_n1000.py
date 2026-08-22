"""Compute-once C_l^{y tau} legs for the N=1000 campaign fiducial + truth.

The n1000 field cache (build_field_cache_n1000.py) carries per-realization
kk/yy/ky/tt/kt legs for both arms, but not the y x tau cross -- the one leg
fig06 (panel f) and fig10 (the cl_yt fiducial denominator) currently recompute
live from raw map cubes with a hardcoded 50-realization cap.  This builder
streams the y and tau cubes ONCE, per arm, and saves the per-realization
per-plane cross-spectra next to the field cache, so every figure re-render
afterwards is a cache read.

    yt_stats_{side}_n1000.npz : yt_real (n_real, 5, n_ell), ell, run, n_done

Checkpointed every 25 realizations (re-running resumes from n_done); single
process, ~1 h/side dominated by zip decompression.

    python3 papers/01_pipeline/build_yt_cache_n1000.py --side bind
    python3 papers/01_pipeline/build_yt_cache_n1000.py --side truth

BIND_FID_REPLICA=tb18|tb49|tb53 selects the bind-side replica (default tb49,
the paper fiducial); the output filename carries the replica tag so a tb18
build cannot masquerade as the tb49 cache.
"""
from __future__ import annotations

import argparse
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

from bind.inference.stats import power_spectrum

CEPH = Path("/mnt/home/mlee1/ceph")
N1K = CEPH / "bind_n1000"
CACHE_DIR = N1K / "field_cache"
REPLICA = os.environ.get("BIND_FID_REPLICA", "tb49")
_RUN = {"tb18": "run_0018", "tb49": "run_0049", "tb53": "run_0053"}[REPLICA]
CKPT_EVERY = 25


def _stream(path: Path, key: str):
    """Yield (n_real, n_plane) then one (n_plane, ny, nx) block per realization."""
    with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as fh:
        version = fmt.read_magic(fh)
        shape, _, dtype = (fmt.read_array_header_1_0(fh) if version == (1, 0)
                           else fmt.read_array_header_2_0(fh))
        n_real, n_plane, ny, nx = shape
        yield n_real, n_plane
        block = n_plane * ny * nx * dtype.itemsize
        for _ in range(n_real):
            buf = bytearray()
            while len(buf) < block:
                chunk = fh.read(block - len(buf))
                assert chunk, f"truncated {key}.npy in {path}"
                buf += chunk
            yield np.frombuffer(bytes(buf), dtype=dtype).reshape(n_plane, ny, nx)


def build(side: str) -> None:
    run = {"bind": f"twobound/{_RUN}", "truth": "truth/run_0000"}[side]
    rd = N1K / run
    tag = f"_{REPLICA}" if side == "bind" else ""
    dest = CACHE_DIR / f"yt_stats_{side}{tag}_n1000.npz"

    ys = _stream(rd / "y_maps.npz", "y")
    ts = _stream(rd / "tau_maps.npz", "tau")
    (n_real, n_plane), (n_real_t, n_plane_t) = next(ys), next(ts)
    assert (n_real, n_plane) == (n_real_t, n_plane_t), "y/tau cube shape mismatch"

    ell = None
    yt = None
    done = 0
    if dest.exists():
        # a kill during a checkpoint write can leave a truncated zip -- treat
        # any unreadable/mismatched file as "start over", never crash on it
        try:
            with np.load(dest) as d:
                if d["yt_real"].shape[0] == n_real and str(d["run"]) == run:
                    yt, ell, done = d["yt_real"].copy(), d["ell"].copy(), int(d["n_done"])
                    print(f"{dest.name}: resuming at {done}/{n_real}")
        except Exception as e:
            print(f"{dest.name}: unreadable ({e!r}) -- rebuilding from scratch")

    def _checkpoint():
        # atomic: np.savez TRUNCATES its target on open, so a mid-write kill
        # would otherwise destroy the previous good checkpoint too
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, yt_real=yt, ell=ell, run=run,
                            n_done=done, replica=(REPLICA if side == "bind" else ""))
        os.replace(tmp, dest)

    t0 = time.time()
    for r in range(n_real):
        ymaps, tmaps = next(ys), next(ts)
        if r < done:
            continue
        for zi in range(n_plane):
            l, cl = power_spectrum(ymaps[zi], tmaps[zi])
            if yt is None:
                ell = np.asarray(l, float)
                yt = np.full((n_real, n_plane, len(ell)), np.nan, np.float64)
            yt[r, zi] = cl
        done = r + 1
        if done % CKPT_EVERY == 0 or done == n_real:
            _checkpoint()
            print(f"  {side}: {done}/{n_real} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE {side} -> {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True, choices=("bind", "truth"))
    args = ap.parse_args()
    build(args.side)
