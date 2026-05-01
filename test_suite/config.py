"""Suite configuration helpers for CV, 1P, and custom test manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import SimulationSpec


def parse_sim_ids(sim_ids: str | None) -> list[str] | None:
    """Parse a comma-separated simulation id string."""
    if not sim_ids:
        return None
    return [chunk.strip() for chunk in sim_ids.split(",") if chunk.strip()]


def build_cv_specs(
    sim_ids: Iterable[str] | None,
    param_file: Path,
    nbody_root: Path,
    hydro_root: Path,
    fof_root: Path,
    snapshot: int,
    box_size: float,
    npix: int,
    patch_pix: int,
    proj_frac: float,
    halo_mass_min: float,
) -> list[SimulationSpec]:
    """Build simulation specs for the CV suite."""
    df = pd.read_csv(param_file, sep=r"\s+", comment="#", header=None, skiprows=1)
    if sim_ids is None:
        ids = [str(i) for i in range(27)]
    else:
        ids = [str(int(s)) for s in sim_ids]

    specs: list[SimulationSpec] = []
    for sid in ids:
        sim_name = f"CV_{sid}"
        row = df[df[0] == sim_name]
        if row.empty:
            raise ValueError(f"No parameters found for {sim_name} in {param_file}")
        params = row.iloc[0, 1:36].values.astype(np.float32)
        # CV table lists VariableWindSpecMomentum=2000, but the CV box was
        # actually run at the fiducial value of 0 (matches 1P_p1_0 and the
        # SB35 minmax fiducial). Override here so conditioning matches the
        # true simulation value.
        params[14] = 0.0

        specs.append(
            SimulationSpec(
                suite="CV",
                sim_id=sid,
                snapshot=snapshot,
                nbody_path=nbody_root / sim_name,
                hydro_snapdir=hydro_root / sim_name / f"snapdir_{snapshot:03d}",
                group_catalog=fof_root / sim_name,
                params=params,
                box_size=box_size,
                npix=npix,
                patch_pix=patch_pix,
                proj_frac=proj_frac,
                halo_mass_min=halo_mass_min,
            )
        )
    return specs


def _resolve_1p_nbody_sim(sim_name: str) -> str:
    """Mirror CAMELS 1P N-body routing used in existing workflows."""
    parts = sim_name.split("_")
    if len(parts) < 3:
        return sim_name

    value = parts[-1]
    cosmological_param_ids = {1, 2, 7, 8, 9}
    try:
        v = int(value)
    except ValueError:
        return sim_name

    if v == 0:
        return "1P_p1_0"

    try:
        param_id = int(parts[1][1:])
    except (IndexError, ValueError):
        return sim_name

    if param_id in cosmological_param_ids:
        return sim_name
    return "1P_p1_0"


def build_1p_specs(
    sim_ids: Iterable[str] | None,
    param_file: Path,
    nbody_root: Path,
    hydro_root: Path,
    fof_root: Path,
    snapshot: int,
    box_size: float,
    npix: int,
    patch_pix: int,
    proj_frac: float,
    halo_mass_min: float,
) -> list[SimulationSpec]:
    """Build simulation specs for the 1P suite."""
    df = pd.read_csv(param_file, sep=r"\s+")
    all_names = df["#Name"].astype(str).to_list()
    names = list(sim_ids) if sim_ids is not None else all_names

    specs: list[SimulationSpec] = []
    for sim_name in names:
        row = df[df["#Name"] == sim_name]
        if row.empty:
            raise ValueError(f"No parameters found for {sim_name} in {param_file}")

        params = row.iloc[0, 1:-1].values.astype(np.float32)

        nbody_sim = _resolve_1p_nbody_sim(sim_name)
        fof_catalog = fof_root / sim_name
        if not fof_catalog.exists():
            fof_catalog = fof_root / "1P_p1_0"
        specs.append(
            SimulationSpec(
                suite="1P",
                sim_id=sim_name,
                snapshot=snapshot,
                nbody_path=nbody_root / nbody_sim,
                hydro_snapdir=hydro_root / sim_name / f"snapdir_{snapshot:03d}",
                group_catalog=fof_catalog,
                params=params,
                box_size=box_size,
                npix=npix,
                patch_pix=patch_pix,
                proj_frac=proj_frac,
                halo_mass_min=halo_mass_min,
            )
        )
    return specs


def build_sb35_specs(
    sim_ids: Iterable[str] | None,
    param_file: Path,
    nbody_root: Path,
    hydro_root: Path,
    fof_root: Path,
    snapshot: int,
    box_size: float,
    npix: int,
    patch_pix: int,
    proj_frac: float,
    halo_mass_min: float,
) -> list[SimulationSpec]:
    """Build simulation specs for the SB35 suite.

    FoF catalogs use the groups_NNN.*.hdf5 naming convention (handled
    automatically by load_halo_catalog's fallback glob).
    """
    df = pd.read_csv(param_file, sep=r"\s+", comment="#", header=None, skiprows=1)
    if sim_ids is None:
        ids = [str(i) for i in range(1024)]
    else:
        ids = [str(s) for s in sim_ids]

    specs: list[SimulationSpec] = []
    for sid in ids:
        sim_name = f"SB35_{sid}"
        row = df[df[0] == sim_name]
        if row.empty:
            raise ValueError(f"No parameters found for {sim_name} in {param_file}")
        params = row.iloc[0, 1:36].values.astype(np.float32)

        specs.append(
            SimulationSpec(
                suite="SB35",
                sim_id=sid,
                snapshot=snapshot,
                nbody_path=nbody_root / sim_name,
                hydro_snapdir=hydro_root / sim_name / f"snapdir_{snapshot:03d}",
                group_catalog=fof_root / sim_name,
                params=params,
                box_size=box_size,
                npix=npix,
                patch_pix=patch_pix,
                proj_frac=proj_frac,
                halo_mass_min=halo_mass_min,
            )
        )
    return specs


def build_test_specs_from_manifest(
    manifest_path: Path,
    snapshot: int,
    box_size: float,
    npix: int,
    patch_pix: int,
    proj_frac: float,
    halo_mass_min: float,
    sim_ids: Iterable[str] | None = None,
) -> list[SimulationSpec]:
    """Build test-suite specs from a JSON manifest.

    Manifest format:
    {
      "simulations": [
        {
          "suite": "Test",
          "sim_id": "SB35_12",
          "nbody_path": "...",
          "hydro_snapdir": "...",
          "group_catalog": "...",
          "params": [35 floats]
        }
      ]
    }
    """
    payload = json.loads(manifest_path.read_text())
    sims = payload.get("simulations", [])
    if not sims:
        raise ValueError(f"No simulations found in manifest {manifest_path}")

    if sim_ids is not None:
        selected = [str(s) for s in sim_ids]
        if not selected:
            return []
        selected_set = set(selected)
        sims = [item for item in sims if str(item.get("sim_id")) in selected_set]
        if not sims:
            raise ValueError(
                f"No simulations matched sim_ids={selected} in manifest {manifest_path}"
            )

    specs: list[SimulationSpec] = []
    for item in sims:
        if "params" not in item:
            raise ValueError(f"Manifest entry missing 'params': {item}")
        params = np.asarray(item["params"], dtype=np.float32)
        if params.shape[0] != 35:
            raise ValueError(
                f"Expected 35 parameters, got {params.shape[0]} for {item.get('sim_id', 'unknown')}"
            )
        specs.append(
            SimulationSpec(
                suite=str(item.get("suite", "Test")),
                sim_id=str(item["sim_id"]),
                snapshot=snapshot,
                nbody_path=Path(item["nbody_path"]),
                hydro_snapdir=Path(item["hydro_snapdir"]),
                group_catalog=Path(item["group_catalog"]),
                params=params,
                box_size=box_size,
                npix=npix,
                patch_pix=patch_pix,
                proj_frac=proj_frac,
                halo_mass_min=halo_mass_min,
            )
        )
    return specs


def _split_sim_ids_for_all_sb35(
    sim_ids: list[str] | None,
) -> tuple[list[str] | None, list[str] | None, list[str] | None, list[str] | None]:
    """Extend _split_sim_ids_for_all to also handle sb35: prefix."""
    if sim_ids is None:
        return None, None, None, None

    cv_ids: list[str] = []
    onep_ids: list[str] = []
    test_ids: list[str] = []
    sb35_ids: list[str] = []

    for raw in sim_ids:
        token = raw.strip()
        if not token:
            continue
        lower = token.lower()
        if lower.startswith("cv:"):
            cv_ids.append(token.split(":", 1)[1])
        elif lower.startswith("1p:") or lower.startswith("onep:"):
            onep_ids.append(token.split(":", 1)[1])
        elif lower.startswith("test:"):
            test_ids.append(token.split(":", 1)[1])
        elif lower.startswith("sb35:"):
            sb35_ids.append(token.split(":", 1)[1])
        elif token.isdigit():
            cv_ids.append(token)
        elif token.upper().startswith("CV_") and token[3:].isdigit():
            cv_ids.append(token[3:])
        else:
            onep_ids.append(token)
            test_ids.append(token)

    return (
        cv_ids or None,
        onep_ids or None,
        test_ids or None,
        sb35_ids or None,
    )


def build_suite_specs(
    suite: str,
    sim_ids: list[str] | None,
    snapshot: int,
    box_size: float,
    npix: int,
    patch_pix: int,
    proj_frac: float,
    halo_mass_min: float,
    cv_param_file: Path,
    cv_nbody_root: Path,
    cv_hydro_root: Path,
    cv_fof_root: Path,
    onep_param_file: Path,
    onep_nbody_root: Path,
    onep_hydro_root: Path,
    onep_fof_root: Path,
    test_manifest: Path | None,
    sb35_param_file: Path | None = None,
    sb35_nbody_root: Path | None = None,
    sb35_hydro_root: Path | None = None,
    sb35_fof_root: Path | None = None,
) -> list[SimulationSpec]:
    """Build full simulation list for a chosen suite."""
    suite_norm = suite.lower()
    specs: list[SimulationSpec] = []

    if suite_norm == "all":
        cv_ids, onep_ids, test_ids, sb35_ids = _split_sim_ids_for_all_sb35(sim_ids)
    else:
        cv_ids = sim_ids if suite_norm == "cv" else None
        onep_ids = sim_ids if suite_norm == "1p" else None
        test_ids = sim_ids if suite_norm == "test" else None
        sb35_ids = sim_ids if suite_norm == "sb35" else None

    if suite_norm in {"all", "cv"}:
        specs.extend(
            build_cv_specs(
                cv_ids,
                cv_param_file,
                cv_nbody_root,
                cv_hydro_root,
                cv_fof_root,
                snapshot,
                box_size,
                npix,
                patch_pix,
                proj_frac,
                halo_mass_min,
            )
        )

    if suite_norm in {"all", "1p"}:
        specs.extend(
            build_1p_specs(
                onep_ids,
                onep_param_file,
                onep_nbody_root,
                onep_hydro_root,
                onep_fof_root,
                snapshot,
                box_size,
                npix,
                patch_pix,
                proj_frac,
                halo_mass_min,
            )
        )

    if suite_norm in {"all", "test"}:
        if test_manifest is None:
            raise ValueError("--test_manifest is required for suite=test or suite=all")
        specs.extend(
            build_test_specs_from_manifest(
                test_manifest,
                snapshot,
                box_size,
                npix,
                patch_pix,
                proj_frac,
                halo_mass_min,
                sim_ids=test_ids,
            )
        )

    if suite_norm == "sb35":
        if sb35_param_file is None or sb35_nbody_root is None or sb35_hydro_root is None or sb35_fof_root is None:
            raise ValueError(
                "--sb35_param_file, --sb35_nbody_root, --sb35_hydro_root, and --sb35_fof_root "
                "are required for suite=sb35 or suite=all"
            )
        specs.extend(
            build_sb35_specs(
                sb35_ids,
                sb35_param_file,
                sb35_nbody_root,
                sb35_hydro_root,
                sb35_fof_root,
                snapshot,
                box_size,
                npix,
                patch_pix,
                proj_frac,
                halo_mass_min,
            )
        )

    return specs
