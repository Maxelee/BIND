"""Artifact path management and cache I/O for test-suite runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schemas import SimulationSpec


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


def save_halo_catalog(path: Path, halos: list[dict], halo_masses: np.ndarray, halo_positions: np.ndarray) -> None:
    """Save halo list and source arrays."""
    centers = np.asarray([h["halo_center"] for h in halos], dtype=np.float32)
    params = np.asarray([h["params"] for h in halos], dtype=np.float32)
    masses = np.asarray([h["halo_mass"] for h in halos], dtype=np.float32)

    np.savez(
        path,
        centers=centers,
        params=params,
        masses=masses,
        halo_masses=halo_masses.astype(np.float32),
        halo_positions=halo_positions.astype(np.float32),
    )


def load_halo_catalog(path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load halo list and source arrays from cache."""
    loaded = np.load(path)
    centers = loaded["centers"]
    params = loaded["params"]
    masses = loaded["masses"]

    halos = [
        {"halo_center": centers[i], "halo_mass": float(masses[i]), "params": params[i]}
        for i in range(len(centers))
    ]

    return halos, loaded["halo_masses"], loaded["halo_positions"]


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


def save_generated_halos(path: Path, generated_halos: np.ndarray) -> None:
    """Save generated halo patches."""
    np.savez(path, generated=generated_halos.astype(np.float32))


def load_generated_halos(path: Path) -> np.ndarray:
    """Load generated halo patches."""
    loaded = np.load(path)
    return loaded["generated"]


def save_composite(path: Path, composite_bundle: dict, mass_stats: dict) -> None:
    """Save composite map products and summary diagnostics."""
    np.savez(
        path,
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
    """Load composite map products."""
    loaded = np.load(path)
    return {
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
