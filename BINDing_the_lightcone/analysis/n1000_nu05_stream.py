"""nu05-convention per-realization nu statistics for the n1000 fig-5 arms.

The paper's canonical nu convention (papers/01_pipeline/nu_grid.py):
  NU_EDGES = arange(-3, 8.5, 0.5)  (23 edges) ; NU = 22 centres, -2.75..7.75
  peaks/minima : 2' smoothing, nu_norm='map', histogrammed on NU_EDGES
  V0-V2        : 1' smoothing, threshold functionals evaluated AT NU
  PDF          : UNSMOOTHED map in its own S/N units, density on NU_EDGES

This streams the bind_n1000 kappa cubes and applies nu_grid.compute's exact
per-map calls, one realization at a time:
  bind  arm = twobound/run_0018, all 5 source planes
  truth arm = truth/run_0000, the z_s = 1 plane only (all fig-5 uses)

Checkpoints every 25 reals to analysis/n1000_nu05_stats.npz.

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python n1000_nu05_stream.py
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

from bind.inference.stats import nongaussian_stats, peak_counts

N1K = Path("/mnt/home/mlee1/ceph/bind_n1000")
HERE = Path(__file__).resolve().parent
DEST = HERE / "n1000_nu05_stats.npz"
NU_EDGES = np.arange(-3.0, 8.0 + 0.25, 0.5)
NU = 0.5 * (NU_EDGES[:-1] + NU_EDGES[1:])
assert len(NU) == 22
FOV, NREAL = 5.0, 1000
ARMS = {"bind": ("twobound/run_0018", [0, 1, 2, 3, 4]),
        "truth": ("truth/run_0000", [1])}
KEYS = ("pdf", "peaks", "minima", "v0", "v1", "v2")


def stream_realizations(npz, key):
    with zipfile.ZipFile(npz) as z, z.open(key + ".npy") as f:
        v = fmt.read_magic(f)
        shape, fortran, dt = (fmt.read_array_header_1_0(f) if v == (1, 0)
                              else fmt.read_array_header_2_0(f))
        assert not fortran
        per = int(np.prod(shape[1:])) * dt.itemsize
        for i in range(min(shape[0], NREAL)):
            buf = f.read(per)
            if len(buf) < per:
                return
            yield i, np.frombuffer(buf, dtype=dt).reshape(shape[1:])


def nu05_stats(planes):
    """(n_pl, 6, 22) for one realization's selected planes (nu_grid recipe)."""
    P = planes[None]                                    # (1, n_pl, N, N)
    pc = peak_counts(P, fov_deg=FOV, smoothing_arcmin=2.0, nu_bins=NU_EDGES,
                     nu_norm="map", return_realizations=True)
    ng = nongaussian_stats(P, fov_deg=FOV, smoothing_scales_arcmin=(1.0,),
                           mf_thresholds=NU, return_realizations=True)
    out = np.empty((planes.shape[0], 6, len(NU)))
    for s in range(planes.shape[0]):
        m = planes[s].astype(np.float64)
        out[s, 0] = np.histogram((m - m.mean()) / m.std(), bins=NU_EDGES,
                                 density=True)[0]
    out[:, 1] = np.asarray(pc["peak_counts_real"])[0]
    out[:, 2] = np.asarray(pc["minima_counts_real"])[0]
    for c in range(3):
        out[:, 3 + c] = np.asarray(ng[f"V{c}_real"])[0]
    return out


STATE = {"nu": NU}
if DEST.exists():
    _p = np.load(DEST)
    STATE = {k: _p[k] for k in _p.files}
    print("resuming: " + ", ".join(f"{a}={STATE.get(f'n_done_{a}', 0)}" for a in ARMS),
          flush=True)

for arm, (rel, planes) in ARMS.items():
    done = int(STATE.get(f"n_done_{arm}", 0))
    if done >= NREAL:
        print(f"[{arm}] already complete", flush=True)
        continue
    acc = {k: list(STATE[f"{arm}_{k}"][:done]) if done else [] for k in KEYS}
    t0 = time.time()
    for i, cube in stream_realizations(N1K / rel / "kappa_maps.npz", "kappa"):
        if i < done:
            continue
        v = nu05_stats(cube[planes])                   # (n_pl, 6, 22)
        for c, k in enumerate(KEYS):
            acc[k].append(v[:, c] if len(planes) > 1 else v[0, c])
        if (i + 1) % 25 == 0:
            for k in KEYS:
                STATE[f"{arm}_{k}"] = np.asarray(acc[k])
            STATE[f"n_done_{arm}"] = i + 1
            np.savez_compressed(DEST, **STATE)
            el = time.time() - t0
            nn = i + 1 - done
            print(f"[{arm}] {i+1}/{NREAL}  {el/nn:.2f}s/real  "
                  f"eta {(NREAL-i-1)*el/nn/60:.0f} min", flush=True)
    for k in KEYS:
        STATE[f"{arm}_{k}"] = np.asarray(acc[k])
    STATE[f"n_done_{arm}"] = len(acc["pdf"])
    np.savez_compressed(DEST, **STATE)
    print(f"[{arm}] DONE {len(acc['pdf'])}", flush=True)

print("ALL DONE", flush=True)
