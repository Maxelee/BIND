"""Minkowski functionals on an EXTENDED nu grid (-3..8) for the n1000 arms.

The campaign's stream_stats files carry V0-V2 on mf_nu = linspace(-3, 4, 29);
the peaks/minima grid reaches nu ~ 12. To draw the MF panels of the posterity
fig-5 out to nu = 8, this recomputes the MFs per realization on
mf_nu_ext = linspace(-3, 8, 45) (same 0.25 spacing), with conventions pinned
to examples/n1000_stats_stream.py: 1' Gaussian smoothing, per-map nu
normalization, float64, fov 5 deg.

Arms: bind (all 5 source planes; the figure draws every z_s) and truth
(z_s = 1 plane only; all the figure uses). Checkpoints every 25 realizations.

Writes analysis/n1000_mfs_ext.npz:
  mf_nu_ext (45,)  bind_V0/V1/V2 (n,5,45)  truth_V0/V1/V2 (n,45)  n_done_{arm}

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python n1000_mfs_ext.py
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

from bind.inference.stats import _gaussian_smooth, minkowski_functionals

N1K = Path("/mnt/home/mlee1/ceph/bind_n1000")
HERE = Path(__file__).resolve().parent
DEST = HERE / "n1000_mfs_ext.npz"
MF_NU_EXT = np.linspace(-3.0, 8.0, 45)
FOV, MF_SM = 5.0, 1.0
NREAL = 1000


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


def mfs(m):
    sm = _gaussian_smooth(m.astype(np.float64), MF_SM, FOV)
    nu = (sm - sm.mean()) / sm.std()
    return np.stack(minkowski_functionals(nu, MF_NU_EXT))  # (3, 45)


STATE = {}
if DEST.exists():
    _p = np.load(DEST)
    STATE = {k: _p[k] for k in _p.files}
    print(f"resuming from checkpoint: " +
          ", ".join(f"{a}={STATE.get('n_done_' + a, 0)}" for a in ("bind", "truth")),
          flush=True)

for arm, planes in (("bind", range(5)), ("truth", (1,))):
    done = int(STATE.get(f"n_done_{arm}", 0))
    if done >= NREAL:
        print(f"[{arm}] already complete", flush=True)
        continue
    acc = {f"V{c}": list(STATE[f"{arm}_V{c}"][:done]) if done else []
           for c in range(3)}
    t0 = time.time()
    for i, cube in stream_realizations(N1K / arm / "run_0000/kappa_maps.npz", "kappa"):
        if i < done:
            continue
        v = np.stack([mfs(cube[s]) for s in planes])   # (n_pl, 3, 45)
        for c in range(3):
            acc[f"V{c}"].append(v[:, c] if len(planes) > 1 else v[0, c])
        if (i + 1) % 25 == 0:
            for c in range(3):
                STATE[f"{arm}_V{c}"] = np.asarray(acc[f"V{c}"])
            STATE[f"n_done_{arm}"] = i + 1
            STATE["mf_nu_ext"] = MF_NU_EXT
            np.savez_compressed(DEST, **STATE)
            el = time.time() - t0
            nn = i + 1 - done
            print(f"[{arm}] {i+1}/{NREAL}  {el/nn:.2f}s/real  eta this arm "
                  f"{(NREAL-i-1)*el/nn/60:.0f} min", flush=True)
    for c in range(3):
        STATE[f"{arm}_V{c}"] = np.asarray(acc[f"V{c}"])
    STATE[f"n_done_{arm}"] = len(acc["V0"])
    STATE["mf_nu_ext"] = MF_NU_EXT
    np.savez_compressed(DEST, **STATE)
    print(f"[{arm}] DONE {len(acc['V0'])} reals", flush=True)

print("ALL DONE", flush=True)
