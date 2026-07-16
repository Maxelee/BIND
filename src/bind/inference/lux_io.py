"""Read lux ray-tracing output (kappa / gamma maps) for the BIND lightcone.

lux writes, per realization ``run<NNN>/``, a Fortran-record binary per source
plane: ``int32 N | float64 N*N | int32 N`` for ``kappa<plane>.dat`` and a
2-component (gamma1, gamma2) array for ``gamma<plane>.dat``.  The BIND fiducial
run uses ``output_planes = 26, 45, 59, 70, 78`` chi-matched to source redshifts
``z_s ~ 0.5, 1.0, 1.5, 2.0, 2.44`` (see ``lux_bind.ini``).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# lux output_planes -> nominal source redshift (lux_bind.ini)
PLANE_TO_ZS: dict[int, float] = {26: 0.5, 45: 1.0, 59: 1.5, 70: 2.0, 78: 2.44}


def read_lux_map(path: str | Path, n_fields: int = 1) -> np.ndarray:
    """Read one lux Fortran-record map. ``n_fields=2`` for gamma (-> (2, N, N))."""
    with open(path, "rb") as fp:
        n = struct.unpack("<i", fp.read(4))[0]
        data = np.frombuffer(fp.read(n * n * n_fields * 8), dtype="<f8")
    if n_fields == 1:
        return data.reshape(n, n).copy()
    return data.reshape(n, n, n_fields).transpose(2, 0, 1).copy()


def load_kappa_realizations(
    rt_root: str | Path,
    *,
    planes=(26, 45, 59, 70, 78),
    n_real: int | None = None,
    run_glob: str = "run*",
) -> tuple[np.ndarray, np.ndarray]:
    """Load all lux kappa realizations into ``(n_real, n_planes, N, N)``.

    Returns ``(kappa, source_redshifts)``; realizations are the ``run<NNN>``
    subdirs of ``rt_root`` (sorted), truncated to ``n_real`` if given.
    """
    rt_root = Path(rt_root)
    runs = sorted(d for d in rt_root.glob(run_glob) if d.is_dir())
    if n_real is not None:
        runs = runs[:n_real]
    if not runs:
        raise FileNotFoundError(f"no {run_glob} dirs under {rt_root}")
    zs = np.array([PLANE_TO_ZS.get(p, np.nan) for p in planes])
    out = []
    for rd in runs:
        out.append([read_lux_map(rd / f"kappa{p}.dat") for p in planes])
    return np.asarray(out), zs


def load_y_realizations(
    rt_root: str | Path,
    *,
    planes=(26, 45, 59, 70, 78),
    n_real: int | None = None,
    run_glob: str = "run*",
) -> tuple[np.ndarray, np.ndarray]:
    """Load lux tSZ y realizations into ``(n_real, n_planes, N, N)``.

    ``y{plane}.dat`` is the cumulative Compton-y integrated to that plane's source
    distance (so ``y78`` is the most complete, to z~2.44).  Only runs that have
    *all* requested ``y`` planes are included — handy while the lux rerun is still
    in flight.  Returns ``(y, source_redshifts)``.
    """
    rt_root = Path(rt_root)
    runs = sorted(d for d in rt_root.glob(run_glob) if d.is_dir())
    runs = [rd for rd in runs if all((rd / f"y{p}.dat").exists() for p in planes)]
    if n_real is not None:
        runs = runs[:n_real]
    if not runs:
        raise FileNotFoundError(f"no runs with complete y planes under {rt_root}")
    zs = np.array([PLANE_TO_ZS.get(p, np.nan) for p in planes])
    out = [[read_lux_map(rd / f"y{p}.dat") for p in planes] for rd in runs]
    return np.asarray(out), zs


def load_tau_realizations(
    rt_root: str | Path,
    *,
    planes=(26, 45, 59, 70, 78),
    n_real: int | None = None,
    run_glob: str = "run*",
) -> tuple[np.ndarray, np.ndarray]:
    """Load lux kSZ/FRB electron-column ``tau`` realizations into ``(n_real,
    n_planes, N, N)``.

    ``tau{plane}.dat`` (written by lux with ``compute_tau = true``) is the
    cumulative electron column integrated to that plane's source distance, the
    ray-traced counterpart of the Born ``tau`` from
    :func:`bind.inference.lightcone_maps.assemble_lightcone`.  Only runs that have
    *all* requested ``tau`` planes are included.  Returns ``(tau, source_redshifts)``.
    """
    rt_root = Path(rt_root)
    runs = sorted(d for d in rt_root.glob(run_glob) if d.is_dir())
    runs = [rd for rd in runs if all((rd / f"tau{p}.dat").exists() for p in planes)]
    if n_real is not None:
        runs = runs[:n_real]
    if not runs:
        raise FileNotFoundError(f"no runs with complete tau planes under {rt_root}")
    zs = np.array([PLANE_TO_ZS.get(p, np.nan) for p in planes])
    out = [[read_lux_map(rd / f"tau{p}.dat") for p in planes] for rd in runs]
    return np.asarray(out), zs
