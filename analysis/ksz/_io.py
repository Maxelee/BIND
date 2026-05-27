"""Shared I/O for kSZ validation scripts.

Walks the test-suite output tree produced by run_test_suite_parallel.sh and
returns a uniform per-simulation payload.  Centralised so plots A/B/C/... all
agree on field semantics and unit conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Channel index of "Gas" in BIND's 3-channel output [DM_hydro, Gas, Stars].
GAS_CH = 1


def format_mass_tag(halo_mass_min: float) -> str:
    """Replicate test_suite.artifacts._format_mass_threshold_tag."""
    sci = f"{float(halo_mass_min):.3e}"
    return sci.replace(".", "p").replace("+", "").replace("-", "m")


def find_sim_dirs(testsuite_root: Path, suite: str) -> list[Path]:
    suite_dir = testsuite_root / suite
    if not suite_dir.is_dir():
        return []
    return sorted(p for p in suite_dir.iterdir() if p.is_dir())


@dataclass(frozen=True)
class SimArtifacts:
    suite: str
    sim_id: str
    snapshot: int
    patch_pix: int
    halo_masses: np.ndarray       # (N,)   Msun/h
    r200_mpc_h: np.ndarray        # (N,)   Mpc/h (0 where unknown)
    centers: np.ndarray           # (N, 2) Mpc/h (sky-plane only)
    params: np.ndarray            # (N, 35) raw CAMELS params (per halo)
    bind_gas: np.ndarray          # (N, P, P) gas-mass-surface-density-per-pixel (Msun/h/pixel)
    truth_gas: np.ndarray         # (N, P, P) same convention as bind_gas
    n_fallback_r200: int = 0      # halos whose r200 was missing


def _extract_truth_gas_from_full_maps(
    full_maps_path: Path,
    centers_mpc_h: np.ndarray,
    box_size: float,
    npix: int,
    patch_pix: int,
) -> np.ndarray:
    """Periodic gas-channel cutout from the saved full-box truth_maps."""
    if not full_maps_path.exists():
        return np.zeros((0, patch_pix, patch_pix), dtype=np.float32)
    loaded = np.load(full_maps_path)
    if "truth_maps" not in loaded.files:
        return np.zeros((0, patch_pix, patch_pix), dtype=np.float32)
    truth = loaded["truth_maps"]
    pixels_per_mpc = npix / box_size
    half = patch_pix // 2
    n = len(centers_mpc_h)
    out = np.zeros((n, patch_pix, patch_pix), dtype=np.float32)
    for i in range(n):
        cx = int(centers_mpc_h[i, 0] * pixels_per_mpc) % npix
        cy = int(centers_mpc_h[i, 1] * pixels_per_mpc) % npix
        ix = (cx - half + np.arange(patch_pix)) % npix
        iy = (cy - half + np.arange(patch_pix)) % npix
        out[i] = truth[GAS_CH][np.ix_(ix, iy)]
    return out


def load_sim(
    sim_dir: Path,
    suite: str,
    model_name: str,
    halo_mass_min: float,
    box_size: float,
    patch_size_mpc_h: float,
) -> SimArtifacts | None:
    """Load one simulation's BIND + truth gas patches, halo catalog, and params.

    Returns None when any required artifact is missing or empty.  R200 is read
    from ``r200s`` if present; otherwise from ``radii`` (legacy convention,
    kpc/h → Mpc/h); else zeros (caller decides fallback).
    """
    snap_dirs = sorted(
        p for p in sim_dir.iterdir() if p.is_dir() and p.name.startswith("snap_")
    )
    if not snap_dirs:
        return None
    snap_dir = snap_dirs[0]
    snap = int(snap_dir.name.removeprefix("snap_"))

    mass_dir = snap_dir / f"mass_threshold_{format_mass_tag(halo_mass_min)}"
    halo_catalog_path = mass_dir / "halo_catalog.npz"
    if not halo_catalog_path.exists():
        return None
    gen_path = mass_dir / model_name / "generated_halos.npz"
    if not gen_path.exists():
        return None

    cat = np.load(halo_catalog_path)
    centers = np.asarray(cat["centers"], dtype=np.float32)
    halo_masses = np.asarray(
        cat["halo_masses"] if "halo_masses" in cat.files else cat["masses"],
        dtype=np.float64,
    )
    if "r200s" in cat.files:
        r200_mpc_h = np.asarray(cat["r200s"], dtype=np.float64)
    elif "radii" in cat.files:
        # legacy convention: kpc/h
        r200_mpc_h = np.asarray(cat["radii"], dtype=np.float64) / 1000.0
    else:
        r200_mpc_h = np.zeros(len(centers), dtype=np.float64)
    params = (
        np.asarray(cat["params"], dtype=np.float32)
        if "params" in cat.files
        else np.zeros((len(centers), 0), dtype=np.float32)
    )

    if len(centers) == 0:
        return None

    generated = np.load(gen_path)["generated"]
    if generated.ndim != 4 or generated.shape[0] != len(centers):
        return None
    patch_pix = int(generated.shape[-1])
    bind_gas = np.asarray(generated[:, GAS_CH], dtype=np.float32)

    truth_cube_path = mass_dir / "truth_halos_cube.npz"
    if truth_cube_path.exists():
        truth = np.load(truth_cube_path)["truth_halos"]
        if truth.shape[0] != len(centers):
            return None
        truth_gas = np.asarray(truth[:, GAS_CH], dtype=np.float32)
    else:
        npix = int(round(patch_pix * box_size / patch_size_mpc_h))
        truth_gas = _extract_truth_gas_from_full_maps(
            snap_dir / "full_maps.npz",
            centers_mpc_h=centers,
            box_size=box_size,
            npix=npix,
            patch_pix=patch_pix,
        )
        if truth_gas.shape[0] == 0:
            return None

    n_fallback = int((r200_mpc_h <= 0).sum())
    return SimArtifacts(
        suite=suite,
        sim_id=sim_dir.name,
        snapshot=snap,
        patch_pix=patch_pix,
        halo_masses=halo_masses,
        r200_mpc_h=r200_mpc_h,
        centers=centers,
        params=params,
        bind_gas=bind_gas,
        truth_gas=truth_gas,
        n_fallback_r200=n_fallback,
    )
