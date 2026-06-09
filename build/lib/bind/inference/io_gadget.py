"""Generic Gadget/Arepo HDF5 readers (independent of CAMELS conventions).

These accept a snapshot path or directory and return raw arrays — no
``SimulationSpec`` required.  Position units returned are **Mpc/h**; mass units
are **Msun/h** (i.e. the standard 1e10 ``Msun/h`` factor is already applied).

Snapshot file conventions supported:

* Single-file:   ``snap_<NNN>.hdf5``
* Multi-chunk:   ``snap_<NNN>.<chunk>.hdf5``
* Direct path:   the user passes the exact file (or any glob)

For FOF/SUBFIND group catalogs the same conventions apply with the
``fof_subhalo_tab_<NNN>`` prefix.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_snap_files(path: Path | str, snapshot: int | None) -> list[str]:
    """Resolve a user-supplied path/glob to a sorted list of HDF5 chunk files.

    Accepts:
      - explicit file path
      - directory containing snap_<NNN>.*.hdf5 or snap_<NNN>.hdf5
      - glob pattern with wildcards
    """
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if any(c in str(p) for c in "*?["):
        return sorted(glob.glob(str(p)))
    if p.is_dir():
        if snapshot is None:
            raise ValueError(f"snapshot index required when path is a directory: {p}")
        chunked = sorted(glob.glob(str(p / f"snap_{snapshot:03d}.*.hdf5")))
        if chunked:
            return chunked
        single = p / f"snap_{snapshot:03d}.hdf5"
        if single.exists():
            return [str(single)]
        raise FileNotFoundError(f"No snap_{snapshot:03d}.* files in {p}")
    raise FileNotFoundError(str(p))


def _resolve_group_files(path: Path | str, snapshot: int | None) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if any(c in str(p) for c in "*?["):
        return sorted(glob.glob(str(p)))
    if p.is_dir():
        if snapshot is None:
            raise ValueError(f"snapshot index required when path is a directory: {p}")
        chunked = sorted(glob.glob(str(p / f"fof_subhalo_tab_{snapshot:03d}.*.hdf5")))
        if chunked:
            return chunked
        single = p / f"fof_subhalo_tab_{snapshot:03d}.hdf5"
        if single.exists():
            return [str(single)]
        raise FileNotFoundError(f"No fof_subhalo_tab_{snapshot:03d}.* files in {p}")
    raise FileNotFoundError(str(p))


def _infer_snapshot_index(snap_files: list[str]) -> int | None:
    m = re.search(r"snap[_a-zA-Z]*_(\d+)", Path(snap_files[0]).name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Snapshot readers
# ---------------------------------------------------------------------------

def read_box_size(snap_path: Path | str, snapshot: int | None = None) -> float:
    """Return the periodic box size in Mpc/h from the snapshot Header."""
    files = _resolve_snap_files(snap_path, snapshot)
    with h5py.File(files[0], "r") as h:
        box_kpch = float(h["Header"].attrs["BoxSize"])
    return box_kpch / 1000.0


def read_dmo_particles(
    snap_path: Path | str,
    snapshot: int | None = None,
) -> tuple[np.ndarray, float, float]:
    """Read all PartType1 (collisionless DM) particles from a Gadget snapshot.

    Returns
    -------
    positions : (N, 3) float32 array, Mpc/h
    particle_mass : float, Msun/h (uniform; from MassTable[1])
    box_size : float, Mpc/h
    """
    files = _resolve_snap_files(snap_path, snapshot)

    pos_chunks: list[np.ndarray] = []
    particle_mass: float | None = None
    box_kpch: float | None = None
    for fname in files:
        with h5py.File(fname, "r") as h:
            pos_chunks.append(h["PartType1/Coordinates"][:])
            if particle_mass is None:
                particle_mass = float(h["Header"].attrs["MassTable"][1]) * 1e10
                box_kpch = float(h["Header"].attrs["BoxSize"])
    positions = np.concatenate(pos_chunks).astype(np.float32) / 1000.0
    return positions, float(particle_mass), float(box_kpch) / 1000.0


def read_hydro_particles(
    snap_path: Path | str,
    snapshot: int | None = None,
    species: tuple[str, ...] = ("dm", "gas", "stars"),
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read hydro particles and return ``{species: (positions_mpch, masses_msunh)}``.

    Supported species: ``"dm"`` (PartType1), ``"gas"`` (PartType0), ``"stars"`` (PartType4).
    """
    PARTTYPE = {"dm": 1, "gas": 0, "stars": 4}
    for s in species:
        if s not in PARTTYPE:
            raise ValueError(f"unknown species {s!r}; expected one of {list(PARTTYPE)}")

    files = _resolve_snap_files(snap_path, snapshot)
    out: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {s: ([], []) for s in species}

    for fname in files:
        with h5py.File(fname, "r") as h:
            mt = h["Header"].attrs["MassTable"]
            for s in species:
                pt = PARTTYPE[s]
                key = f"PartType{pt}"
                if key not in h or "Coordinates" not in h[key]:
                    continue
                pos = h[f"{key}/Coordinates"][:]
                if "Masses" in h[key]:
                    masses = h[f"{key}/Masses"][:].astype(np.float32)
                else:
                    masses = np.full(len(pos), mt[pt], dtype=np.float32)
                out[s][0].append(pos)
                out[s][1].append(masses)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in species:
        pos_list, mass_list = out[s]
        if not pos_list:
            result[s] = (np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32))
        else:
            result[s] = (
                (np.concatenate(pos_list).astype(np.float32) / 1000.0),
                (np.concatenate(mass_list).astype(np.float32) * 1e10),
            )
    return result


# ---------------------------------------------------------------------------
# FOF / SUBFIND group catalog
# ---------------------------------------------------------------------------

def read_fof_catalog(
    group_path: Path | str,
    snapshot: int | None = None,
    halo_mass_min: float = 1e13,
    mass_field: str = "Group_M_Crit200",
) -> dict[str, np.ndarray]:
    """Read FoF/SUBFIND group catalog and return mass-cut halos.

    Parameters
    ----------
    mass_field
        Group dataset to use for the mass cut.  Default ``Group_M_Crit200``
        (M200c).  Set to ``"GroupMass"`` for total FoF mass.

    Returns
    -------
    dict with keys
        positions : (M, 3) float32 array, Mpc/h
        mass      : (M,)   float32 array, Msun/h
        r200      : (M,)   float32 array, Mpc/h (zeros if not present)
    """
    files = _resolve_group_files(group_path, snapshot)

    masses: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    r200s: list[np.ndarray] = []
    for fname in files:
        with h5py.File(fname, "r") as h:
            grp = h.get("Group")
            if grp is None or mass_field not in grp:
                continue
            masses.append(grp[mass_field][:])
            positions.append(grp["GroupPos"][:])
            if "Group_R_Crit200" in grp:
                r200s.append(grp["Group_R_Crit200"][:].astype(np.float32))
            else:
                r200s.append(np.zeros(len(grp[mass_field]), dtype=np.float32))

    if not masses:
        raise RuntimeError(f"No {mass_field} entries found in {group_path}")

    masses_arr = np.concatenate(masses) * 1e10
    positions_arr = np.concatenate(positions) / 1000.0
    r200s_arr = np.concatenate(r200s) / 1000.0

    keep = masses_arr > halo_mass_min
    return {
        "positions": positions_arr[keep].astype(np.float32),
        "mass": masses_arr[keep].astype(np.float32),
        "r200": r200s_arr[keep].astype(np.float32),
    }
