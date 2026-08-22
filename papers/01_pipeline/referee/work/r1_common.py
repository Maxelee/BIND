"""Shared I/O + estimator conventions for the R1 three-rung ladder (referee response).

ONE code path for every rung. Nothing here reads a released Cl_*.npz cache: the
cross-spectrum caches in the older run dirs carry the pre-fix XPk_plane
normalization (low by ~1.2e7, ell-dependently), and bind/run_0000's
nongaussian_stats.npz carries stale Minkowski functionals. Everything is
recomputed from the raw map cubes with the conventions documented in
papers/01_pipeline/field_cache.py and papers/01_pipeline/nu_grid.py:

  kappa autos   : power_spectrum(kappa[r, zi])
  y/tau autos   : power_spectrum(F[r, zi])            per-plane cumulative column
  crosses       : power_spectrum(kappa[r, zi], F[r, -1])   kappa(z_s) x TOTAL column
  peaks/minima  : peak_counts(kappa[:, zi, None], 2.0', nu_norm='map') on NU_EDGES
  MFs           : nongaussian_stats(..., 1.0', mf_thresholds=NU)
  PDF           : histogram((m - m.mean())/m.std()) on NU_EDGES, density=True

The rungs (all seed-paired, base seed 1992 + 7r; first 50 realizations used):
  bind        BIND-generated fiducial lightcone     ceph/bind_lightcone_tng (550 real)
  pasted      hydro-pasted "truth"                  bind_science/runs/truth/run_0000 (550)
  diffuse     pasted + f_b*rho_DMO diffuse gas      tng_full_validation/runs/diffuse (50)
  hydro_full  full TNG300 hydro ray trace           tng_full_validation/runs/hydro_full (50)
  dmo         seed-paired DMO                       bind_science/runs/dmo/run_0000 (550)
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as npf

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
LC = CEPH / "bind_lightcone_tng"
VAL = CEPH / "tng_full_validation"
WORK = CEPH / "referee_work/r1"

N_REAL = 50
N_PLANES = 5
FOV_DEG = 5.0
ZS = (0.5, 1.0, 1.5, 2.0, 2.44)
ZI = 1                       # working source plane, z_s = 1.0
ZI2 = 3                      # secondary source plane, z_s = 2.0 ("other z_s look similar")

# ell-domain constants, identical to the paper (_build_figures_nb.py setup cell)
ELL_LO = 100.0
ELL_TRUST = 3.0e4            # measured CIC/pixelization onset = 0.8 ell_Nyq
ELL_MAX_PLOT = 36864.0       # axis-Nyquist of 1024^2 over 5 deg

# rung -> directory holding {kappa,y,tau}_maps.npz
RUNS = {
    "bind": LC,
    "pasted": SCI / "runs/truth/run_0000",
    "diffuse": VAL / "runs/diffuse/run_0000",
    "hydro_full": VAL / "runs/hydro_full/run_0000",
    "dmo": SCI / "runs/dmo/run_0000",
}


def nu_grid():
    """The canonical Paper-I nu axis (papers/01_pipeline/nu_grid.py)."""
    sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline")
    import nu_grid as ng
    return ng


def load_prefix(path: Path, key: str, n: int = N_REAL, plane: int | None = None):
    """First n realizations of a (n_real, n_plane, ny, nx) map-cube npz, streamed.

    npz members are DEFLATE streams, so reading the seed-paired first-n prefix
    touches only n/n_avail of the file and never materializes the 550-real
    (11.5 GB) cubes. plane=None keeps all planes.
    """
    with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as fh:
        version = npf.read_magic(fh)
        shape, fortran, dtype = (npf.read_array_header_1_0(fh) if version == (1, 0)
                                 else npf.read_array_header_2_0(fh))
        assert not fortran
        n_avail, n_plane, ny, nx = shape
        assert n <= n_avail, f"{path.name}: {n_avail} reals < requested {n}"
        p_want = None if plane is None else plane % n_plane
        pbytes = ny * nx * dtype.itemsize
        out = np.empty((n, n_plane, ny, nx) if p_want is None else (n, ny, nx), dtype=dtype)
        for r in range(n):
            for p in range(n_plane):
                buf = bytearray()
                while len(buf) < pbytes:
                    chunk = fh.read(pbytes - len(buf))
                    assert chunk, f"truncated {key}.npy at real {r} plane {p}"
                    buf += chunk
                if p_want is None:
                    out[r, p] = np.frombuffer(bytes(buf), dtype=dtype).reshape(ny, nx)
                elif p == p_want:
                    out[r] = np.frombuffer(bytes(buf), dtype=dtype).reshape(ny, nx)
    return out


def cube_meta(path: Path) -> dict:
    with np.load(path) as f:
        return {k: np.asarray(f[k]).tolist() for k in f.files if k != path.stem.split("_")[0]}


def nu_stats(K: np.ndarray) -> dict:
    """The six nu-domain WL statistics for one (n_real, ny, nx) kappa stack.

    Byte-for-byte the body of nu_grid.compute() for a single source plane, but
    taking an in-memory stack (the paper's helper takes a path and loads the whole
    550-real cube, which does not fit the session).
    """
    from bind.inference.stats import nongaussian_stats, peak_counts
    ng = nu_grid()
    nr = K.shape[0]
    out = {}
    pc = peak_counts(K[:, None], fov_deg=FOV_DEG, smoothing_arcmin=ng.SMOOTH_PK,
                     nu_bins=ng.NU_EDGES, nu_norm="map", return_realizations=True)
    out["peak_counts"] = np.asarray(pc["peak_counts_real"])[:, 0]
    out["minima_counts"] = np.asarray(pc["minima_counts_real"])[:, 0]
    ngk = nongaussian_stats(K[:, None], fov_deg=FOV_DEG, smoothing_scales_arcmin=(1.0,),
                            mf_thresholds=ng.NU, return_realizations=True)
    for k in ("V0", "V1", "V2"):
        out[f"mf_v{k[-1]}"] = np.asarray(ngk[f"{k}_real"])[:, 0]
    pdf = np.empty((nr, len(ng.NU)))
    for r in range(nr):
        m = K[r].astype(float)
        pdf[r] = np.histogram((m - m.mean()) / m.std(), bins=ng.NU_EDGES, density=True)[0]
    out["pdf"] = pdf
    return out
