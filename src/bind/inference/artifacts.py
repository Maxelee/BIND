"""Artifact path management, provenance stamping, and cache I/O for runs."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schemas import SimulationSpec

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
# Generated maps are worthless as evidence unless the reader can tell what made
# them.  Every run stamps a provenance block into its summary.json and into each
# output .npz: the bind version, the checkpoint + norm_stats identities (path and
# sha256), and the *resolved* sampling/compositing settings actually used
# (n_steps, r200_factor, paste_mode, seed, ...) — resolved, because defaults have
# changed over the project's life and a cached directory that mixes n_steps and
# r200_factor values with no stamp cannot be untangled after the fact.

PROVENANCE_KEY = "provenance"

# (resolved path, size, mtime_ns) -> hex digest or None.  Keyed on the file's
# identity so a multi-GB checkpoint is read once per process, not once per
# simulation, but a file that changes on disk is re-hashed.
_SHA256_CACHE: dict[tuple[str, int, int], str | None] = {}


def file_sha256(path: Path | str | None) -> str | None:
    """SHA-256 hex digest of a file, cached per (path, size, mtime).

    Returns ``None`` — never raises — when the path is missing, unreadable, or
    not given.  A long suite run must not die because a checkpoint sat on a
    filesystem that hiccuped while being hashed; an unknown digest is recorded
    as null and the run continues.
    """
    if path is None:
        return None
    try:
        p = Path(path)
        st = p.stat()
        key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    except OSError:
        return None
    if key in _SHA256_CACHE:
        return _SHA256_CACHE[key]
    digest: str | None
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(16 * 1024 * 1024), b""):
                h.update(block)
        digest = h.hexdigest()
    except Exception:  # unreadable/vanished mid-hash — degrade, don't crash
        digest = None
    _SHA256_CACHE[key] = digest
    return digest


def bind_version() -> str | None:
    """``bind.__version__``, or None if it cannot be determined."""
    try:
        from bind import __version__  # local import: bind imports this module
        return str(__version__)
    except Exception:
        return None


def build_provenance(
    *,
    checkpoint_path: Path | str | None = None,
    norm_stats_path: Path | str | None = None,
    n_steps: int | None = None,
    r200_factor: float | None = None,
    paste_mode: str | None = None,
    seed: int | None = None,
    **extra,
) -> dict:
    """Build the provenance block stamped into summaries and output arrays.

    All of ``n_steps`` / ``r200_factor`` / ``paste_mode`` / ``seed`` must be the
    *resolved* values the run actually used, not the caller's defaults.  Extra
    keyword arguments (``batch_size``, ``taper_frac``, ``model``, ...) are merged
    in as-is, so each entry point can record what else matters to it.
    """
    prov = {
        "bind_version": bind_version(),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "norm_stats_path": str(norm_stats_path) if norm_stats_path is not None else None,
        "norm_stats_sha256": file_sha256(norm_stats_path),
        "n_steps": n_steps,
        "r200_factor": r200_factor,
        "paste_mode": paste_mode,
        "seed": seed,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    prov.update(extra)
    return prov


def provenance_array(provenance: dict | None) -> np.ndarray | None:
    """Pack a provenance dict as a 0-d numpy string array for ``np.savez``.

    npz files hold arrays only, so the block travels as one JSON string under the
    ``provenance`` key; read it back with :func:`read_provenance`.
    """
    if provenance is None:
        return None
    return np.asarray(json.dumps(to_jsonable(provenance), sort_keys=True))


def read_provenance(source) -> dict | None:
    """Read the provenance block from an npz path or a loaded ``NpzFile``.

    Returns ``None`` for artifacts written before stamping existed (which is
    itself the useful signal: an unstamped file predates this release).
    """
    if hasattr(source, "files"):
        return _extract_provenance(source)
    # Close the NpzFile explicitly: runner.py reads artifacts from a thread pool,
    # where relying on refcount timing to release the file handle is fragile.
    with np.load(source, allow_pickle=False) as loaded:
        return _extract_provenance(loaded)


def _extract_provenance(loaded) -> dict | None:
    if PROVENANCE_KEY not in loaded.files:
        return None
    try:
        return json.loads(str(loaded[PROVENANCE_KEY]))
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class ArtifactPaths:
    """All on-disk artifacts for one simulation run."""

    sim_base_dir: Path
    snap_dir: Path
    mass_threshold_dir: Path
    model_dir: Path
    full_maps_npz: Path
    halo_catalog_npz: Path
    halo_cutouts_npz: Path
    halo_cutouts_cube_npz: Path   # DMO condition patches (3D voxel method, cube model)
    truth_halos_cube_npz: Path    # hydro truth patches   (3D voxel method, cube model)
    truth_thermo_patches_npz: Path  # per-halo truth thermo patches (snapshot reprojection)
    generated_halos_npz: Path
    composite_npz: Path
    summary_json: Path


def _format_mass_threshold_tag(halo_mass_min: float) -> str:
    """Return a filesystem-safe mass-threshold tag with preserved significant digits."""
    # Example: 1e13 -> 1p000e13, 2.5e14 -> 2p500e14
    sci = f"{float(halo_mass_min):.3e}"
    return sci.replace(".", "p").replace("+", "").replace("-", "m")


def resolve_artifact_paths(output_root: Path, spec: SimulationSpec, model_name: str) -> ArtifactPaths:
    """Build the hierarchical output paths used by the runner."""
    suite_name = spec.suite
    if suite_name.upper() == "1P":
        sim_dir_name = str(spec.sim_id)
    else:
        sim_dir_name = f"sim_{spec.sim_id}"

    sim_base = output_root / suite_name / sim_dir_name
    snap_dir = sim_base / f"snap_{spec.snapshot:03d}"
    mass_tag = _format_mass_threshold_tag(spec.halo_mass_min)
    mass_dir = snap_dir / f"mass_threshold_{mass_tag}"
    model_dir = mass_dir / model_name

    return ArtifactPaths(
        sim_base_dir=sim_base,
        snap_dir=snap_dir,
        mass_threshold_dir=mass_dir,
        model_dir=model_dir,
        full_maps_npz=snap_dir / "full_maps.npz",
        halo_catalog_npz=mass_dir / "halo_catalog.npz",
        halo_cutouts_npz=mass_dir / "halo_cutouts.npz",
        halo_cutouts_cube_npz=mass_dir / "halo_cutouts_cube.npz",
        truth_halos_cube_npz=mass_dir / "truth_halos_cube.npz",
        truth_thermo_patches_npz=mass_dir / "truth_thermo_patches.npz",
        generated_halos_npz=model_dir / "generated_halos.npz",
        composite_npz=model_dir / "composite.npz",
        summary_json=model_dir / "summary.json",
    )


def ensure_dirs(paths: ArtifactPaths) -> None:
    """Create all directories for this simulation."""
    paths.sim_base_dir.mkdir(parents=True, exist_ok=True)
    paths.snap_dir.mkdir(parents=True, exist_ok=True)
    paths.mass_threshold_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)


def save_full_maps(path: Path, dmo_fullbox: np.ndarray, truth_maps: np.ndarray | None) -> None:
    """Save projected DMO and optional hydro truth maps."""
    if truth_maps is None:
        np.savez(path, dmo_fullbox=dmo_fullbox)
    else:
        np.savez(path, dmo_fullbox=dmo_fullbox, truth_maps=truth_maps)


def load_full_maps(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load projected DMO and optional hydro truth maps."""
    loaded = np.load(path)
    dmo = loaded["dmo_fullbox"]
    truth = loaded["truth_maps"] if "truth_maps" in loaded else None
    return dmo, truth


def save_halo_catalog(
    path: Path,
    halos: list[dict],
    halo_masses: np.ndarray,
    halo_r200s: np.ndarray,
    halo_positions: np.ndarray,
) -> None:
    """Save halo list and source arrays."""
    centers = np.asarray([h["halo_center"] for h in halos], dtype=np.float32)
    params = np.asarray([h["params"] for h in halos], dtype=np.float32)
    masses = np.asarray([h["halo_mass"] for h in halos], dtype=np.float32)
    r200s = np.asarray([h.get("r200", 0.0) for h in halos], dtype=np.float32)

    np.savez(
        path,
        centers=centers,
        params=params,
        masses=masses,
        r200s=r200s,
        halo_masses=halo_masses.astype(np.float32),
        halo_r200s=halo_r200s.astype(np.float32),
        halo_positions=halo_positions.astype(np.float32),
    )


def load_halo_catalog(path: Path) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Load halo list and source arrays from cache.

    Returns (halos, halo_masses, halo_r200s, halo_positions).  R200c (Mpc/h) is
    read from ``r200s`` (current format) or, for back-compatibility, from the
    legacy ``radii`` key (stored in kpc/h, converted here).  Cache files with
    neither return zeros, so circular pasting falls back to the FOF reload.
    """
    loaded = np.load(path)
    centers = loaded["centers"]
    params = loaded["params"]
    masses = loaded["masses"]
    if "r200s" in loaded:
        r200s = loaded["r200s"]
    elif "radii" in loaded:  # legacy catalogs stored R200c in kpc/h
        r200s = loaded["radii"].astype(np.float32) / 1e3
    else:
        r200s = np.zeros(len(centers), dtype=np.float32)
    if "halo_r200s" in loaded:
        halo_r200s = loaded["halo_r200s"]
    elif "radii" in loaded:
        halo_r200s = loaded["radii"].astype(np.float32) / 1e3
    else:
        halo_r200s = np.zeros(len(loaded["halo_masses"]), dtype=np.float32)

    halos = [
        {"halo_center": centers[i], "halo_mass": float(masses[i]), "r200": float(r200s[i]), "params": params[i]}
        for i in range(len(centers))
    ]

    return halos, loaded["halo_masses"], halo_r200s, loaded["halo_positions"]


def save_halo_cutouts(path: Path, halo_cutouts: list[dict]) -> None:
    """Save cutouts as packed arrays for faster reloads."""
    if not halo_cutouts:
        np.savez(
            path,
            condition=np.zeros((0, 0, 0), dtype=np.float32),
            large_scale=np.zeros((0, 0, 0, 0), dtype=np.float32),
        )
        return

    condition = np.stack([hc["condition"] for hc in halo_cutouts]).astype(np.float32)
    large_scale = np.stack([hc["large_scale"] for hc in halo_cutouts]).astype(np.float32)
    np.savez(path, condition=condition, large_scale=large_scale)


def load_halo_cutouts(path: Path) -> list[dict]:
    """Load packed cutout arrays back to list-of-dicts format."""
    loaded = np.load(path)
    cond = loaded["condition"]
    ls = loaded["large_scale"]
    return [{"condition": cond[i], "large_scale": ls[i]} for i in range(cond.shape[0])]


def save_generated_halos(
    path: Path, generated_halos: np.ndarray, provenance: dict | None = None
) -> None:
    """Save generated halo patches, stamped with the run provenance."""
    kw = {"generated": generated_halos.astype(np.float32)}
    prov = provenance_array(provenance)
    if prov is not None:
        kw[PROVENANCE_KEY] = prov
    np.savez(path, **kw)


def load_generated_halos(path: Path) -> np.ndarray:
    """Load generated halo patches."""
    loaded = np.load(path)
    return loaded["generated"]


def save_composite(
    path: Path, composite_bundle: dict, mass_stats: dict, provenance: dict | None = None
) -> None:
    """Save composite map products and summary diagnostics, stamped with provenance."""
    prov = provenance_array(provenance)
    np.savez(
        path,
        **({PROVENANCE_KEY: prov} if prov is not None else {}),
        composite=composite_bundle["composite"].astype(np.float32),
        alpha=composite_bundle["alpha"].astype(np.float32),
        hydro_canvas=composite_bundle["hydro_canvas"].astype(np.float32),
        hydro_weights=composite_bundle["hydro_weights"].astype(np.float32),
        patch_scales=composite_bundle["patch_scales"].astype(np.float64),
        scale_global=np.asarray([composite_bundle["scale_global"]], dtype=np.float64),
        coverage_pct=np.asarray([composite_bundle["coverage_pct"]], dtype=np.float64),
        mass_rel_err=mass_stats["rel_err"].astype(np.float64),
        dmo_halo_mass=mass_stats["dmo_halo_mass"].astype(np.float64),
        bind_halo_mass=mass_stats["bind_halo_mass"].astype(np.float64),
    )


def load_composite(path: Path) -> dict:
    """Load composite map products (plus the provenance block, if stamped)."""
    loaded = np.load(path)
    return {
        "provenance": read_provenance(loaded),
        "composite": loaded["composite"],
        "alpha": loaded["alpha"],
        "hydro_canvas": loaded["hydro_canvas"],
        "hydro_weights": loaded["hydro_weights"],
        "patch_scales": loaded["patch_scales"],
        "scale_global": float(loaded["scale_global"][0]),
        "coverage_pct": float(loaded["coverage_pct"][0]),
        "mass_rel_err": loaded["mass_rel_err"],
        "dmo_halo_mass": loaded["dmo_halo_mass"],
        "bind_halo_mass": loaded["bind_halo_mass"],
    }


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_summary_json(path: Path, summary: dict) -> None:
    """Write a JSON summary with numpy-safe conversion."""
    path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


def save_truth_halos_cube(path: Path, truth_halos: np.ndarray) -> None:
    """Save per-halo hydro truth patches from 3D cube voxelization.

    Args:
        truth_halos: (N_halos, 3, patch_pix, patch_pix) float32 array with
            channels [DM_hydro, Gas, Stars].
    """
    np.savez(path, truth_halos=truth_halos.astype(np.float32))


def load_truth_halos_cube(path: Path) -> np.ndarray:
    """Load per-halo hydro truth patches saved by save_truth_halos_cube.

    Returns (N_halos, 3, patch_pix, patch_pix) float32 array.
    """
    return np.load(path)["truth_halos"]


def save_truth_thermo_patches(path: Path, truth_thermo: np.ndarray) -> None:
    """Save per-halo truth thermo patches.

    Args:
        truth_thermo: (N_halos, N_THERMO, patch_pix, patch_pix) float32 array,
            channels in THERMO_KEYS order (compton_y, temperature, entropy, pressure).
    """
    np.savez(path, truth_thermo=truth_thermo.astype(np.float32))


def load_truth_thermo_patches(path: Path) -> np.ndarray:
    """Load per-halo truth thermo patches saved by save_truth_thermo_patches."""
    return np.load(path)["truth_thermo"]
