"""Per-realization stats for bind_n1000 twobound/run_0018 (the correct-fiducial
paint replica), for the posterity fig-5 at N = 1000.

Conventions pinned to examples/n1000_stats_stream.py (the campaign's stream):
  cl      (n, 5, 724)  Pylians auto power, fov 5 deg
  peaks   (n, 5, 68)   2' smoothing, per-map nu norm, nu_bins linspace(-5,12,69)
  minima  (n, 5, 68)
  V0-V2   (n, 5, 45)   1' smoothing, per-map nu norm -- on the EXTENDED
                       mf_nu grid linspace(-3, 8, 45); its first 29 points are
                       exactly the campaign's native linspace(-3, 4, 29)
  pdf     (n, 5, 41)   fixed edges from the production reference

Checkpoints every 25 reals to analysis/stream_stats_twobound_run_0018.npz.

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python stream_stats_tb18.py
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

from bind.inference.stats import (
    _gaussian_smooth,
    minkowski_functionals,
    peak_counts,
    power_spectrum,
)

SRC = Path("/mnt/home/mlee1/ceph/bind_n1000/twobound/run_0018/kappa_maps.npz")
PDF_REF = Path("/mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000/nongaussian_stats.npz")
HERE = Path(__file__).resolve().parent
DEST = HERE / "stream_stats_twobound_run_0018.npz"
FOV, PK_SM, MF_SM, NREAL = 5.0, 2.0, 1.0, 1000

NU_BINS = np.linspace(-5.0, 12.0, 69)
MF_NU = np.linspace(-3.0, 8.0, 45)
assert np.allclose(MF_NU[:29], np.linspace(-3.0, 4.0, 29))

_c = np.load(PDF_REF)["pdf_bins"]
_d = float(np.diff(_c).mean())
EDGES = np.concatenate([_c - 0.5 * _d, [_c[-1] + 0.5 * _d]])[:42]


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


done = 0
ACC = {k: [] for k in ("cl", "peaks", "minima", "V0", "V1", "V2", "pdf")}
if DEST.exists():
    _p = np.load(DEST)
    done = int(_p["n_done"])
    for k in ACC:
        ACC[k] = list(_p[k][:done])
    print(f"resuming at {done}", flush=True)


def checkpoint(n, ell):
    np.savez_compressed(DEST, ell=ell, nu=0.5 * (NU_BINS[1:] + NU_BINS[:-1]),
                        mf_nu=MF_NU, pdf_edges=EDGES, n_done=n,
                        **{k: np.asarray(v) for k, v in ACC.items()})


ell = None
t0 = time.time()
for i, cube in stream_realizations(SRC, "kappa"):
    if i < done:
        continue
    pc = peak_counts(cube[None], fov_deg=FOV, smoothing_arcmin=PK_SM,
                     nu_bins=NU_BINS, return_realizations=True)
    ACC["peaks"].append(pc["peak_counts_real"][0])
    ACC["minima"].append(pc["minima_counts_real"][0])
    cl_r, v_r, pdf_r = [], [], []
    for s in range(cube.shape[0]):
        m = cube[s].astype(np.float64)
        ell, cl = power_spectrum(m, fov_deg=FOV)
        cl_r.append(cl)
        sm = _gaussian_smooth(m, MF_SM, FOV)
        nu = (sm - sm.mean()) / sm.std()
        v_r.append(np.stack(minkowski_functionals(nu, MF_NU)))
        pdf_r.append(np.histogram(m, bins=EDGES, density=True)[0])
    ACC["cl"].append(np.asarray(cl_r))
    v = np.asarray(v_r)
    ACC["V0"].append(v[:, 0])
    ACC["V1"].append(v[:, 1])
    ACC["V2"].append(v[:, 2])
    ACC["pdf"].append(np.asarray(pdf_r))
    if (i + 1) % 25 == 0:
        checkpoint(i + 1, ell)
        el = time.time() - t0
        nn = i + 1 - done
        print(f"[tb18] {i+1}/{NREAL}  {el/nn:.2f}s/real  eta {(NREAL-i-1)*el/nn/60:.0f} min",
              flush=True)

checkpoint(len(ACC["cl"]), ell)
print(f"[tb18] ALL DONE {len(ACC['cl'])} reals in {(time.time()-t0)/60:.1f} min", flush=True)
