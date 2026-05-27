"""Validation plot A — per-halo τ recovery: BIND vs simulation truth.

Walks an existing test-suite output tree and collects per-halo aperture-integrated
gas mass (≡ τ up to a constant) for both BIND-generated patches and the
simulation truth.  Writes a single npz with per-halo arrays so downstream
plotting (plot_validation_a.py) is decoupled from the (slow) I/O walk.

Layout expected (from test_suite.runner):

  <output_root>/<suite>/sim_<sim_id>/snap_<snap>/full_maps.npz
                                                 /mass_threshold_<tag>/halo_catalog.npz
                                                                       /truth_halos_cube.npz   (cube model)
                                                                       /truth_thermo_patches.npz
                                                                       /<model>/generated_halos.npz

Usage:
    python -m analysis.ksz.validation_a \\
        --testsuite_root /mnt/home/mlee1/ceph/fm_testsuite \\
        --model fm_two_head \\
        --suites cv 1p sb35 \\
        --aperture r200 --r200_factor 1.0 \\
        --out analysis_physics_cache/ksz_validation_a_fm_two_head.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .tau_utils import (
    aperture_gas_mass,
    gas_mass_to_tau_in_aperture,
    r200_to_pixels,
)


# Channel index of "Gas" in the BIND 3-channel output [DM_hydro, Gas, Stars]
GAS_CH = 1


# ---------------------------------------------------------------------------
# Filesystem walk
# ---------------------------------------------------------------------------
def _format_mass_tag(halo_mass_min: float) -> str:
    sci = f"{float(halo_mass_min):.3e}"
    return sci.replace(".", "p").replace("+", "").replace("-", "m")


def _find_sim_dirs(testsuite_root: Path, suite: str) -> list[Path]:
    suite_dir = testsuite_root / suite
    if not suite_dir.is_dir():
        return []
    # CV/SB35/test: sim_XXXX/, 1P: <name>/ without prefix
    candidates = [p for p in suite_dir.iterdir() if p.is_dir()]
    return sorted(candidates)


def _extract_truth_gas_from_full_maps(full_maps_path: Path, halos: list[dict],
                                      box_size: float, npix: int,
                                      patch_pix: int) -> np.ndarray:
    """Fallback for standard (large-scale) models: cut Gas channel from truth_maps.

    Mirrors the periodic cutout used in test_suite.pipeline.extract_periodic_cutout.
    """
    loaded = np.load(full_maps_path)
    if "truth_maps" not in loaded.files:
        return np.zeros((0, patch_pix, patch_pix), dtype=np.float32)
    truth = loaded["truth_maps"]  # (3, npix, npix)
    pixels_per_mpc = npix / box_size
    half = patch_pix // 2
    out = np.zeros((len(halos), patch_pix, patch_pix), dtype=np.float32)
    for i, halo in enumerate(halos):
        cx = int(halo["halo_center"][0] * pixels_per_mpc) % npix
        cy = int(halo["halo_center"][1] * pixels_per_mpc) % npix
        ix = (cx - half + np.arange(patch_pix)) % npix
        iy = (cy - half + np.arange(patch_pix)) % npix
        out[i] = truth[GAS_CH][np.ix_(ix, iy)]
    return out


def _aperture_radius_pixels(
    aperture: str,
    r200s_mpc_h: np.ndarray,
    r200_factor: float,
    patch_size_mpc_h: float,
    patch_pix: int,
    fixed_r_pix: float,
) -> tuple[np.ndarray, int]:
    """Per-halo aperture radius in pixels.

    Returns (r_pix, n_fallback) where n_fallback is the number of halos for
    which R200 was missing (≤ 0) and ``fixed_r_pix`` was substituted.  Legacy
    halo_catalog.npz files written before R200 storage was added contain all
    zeros for r200, so without this fallback every aperture would have zero
    area and τ would be NaN.
    """
    r200s = np.asarray(r200s_mpc_h, dtype=np.float64)
    if aperture == "r200":
        r = r200_to_pixels(r200s * r200_factor, patch_size_mpc_h, patch_pix)
        bad = r200s <= 0
        r[bad] = float(fixed_r_pix)
        return r, int(bad.sum())
    if aperture == "fixed":
        return np.full(len(r200s), float(fixed_r_pix), dtype=np.float64), 0
    raise ValueError(f"Unknown aperture mode {aperture!r}")


def _process_sim(
    sim_dir: Path,
    suite: str,
    model_name: str,
    halo_mass_min: float,
    aperture: str,
    r200_factor: float,
    fixed_r_pix: float,
    box_size: float,
    hubble: float,
    patch_size_mpc_h: float,
) -> dict | None:
    """Collect per-halo τ arrays for one simulation directory.

    Returns None if any required artifact is missing (sim is skipped silently).
    """
    snap_dirs = sorted([p for p in sim_dir.iterdir() if p.is_dir() and p.name.startswith("snap_")])
    if not snap_dirs:
        return None
    snap_dir = snap_dirs[0]  # only one snapshot per sim in current test-suite
    snap = int(snap_dir.name.removeprefix("snap_"))

    mass_dir = snap_dir / f"mass_threshold_{_format_mass_tag(halo_mass_min)}"
    halo_catalog_path = mass_dir / "halo_catalog.npz"
    if not halo_catalog_path.exists():
        return None
    gen_path = mass_dir / model_name / "generated_halos.npz"
    if not gen_path.exists():
        return None

    halo_cat = np.load(halo_catalog_path)
    centers = halo_cat["centers"]
    halo_masses = halo_cat["halo_masses"] if "halo_masses" in halo_cat.files else halo_cat["masses"]
    if "r200s" in halo_cat.files:
        r200s = halo_cat["r200s"]
    elif "radii" in halo_cat.files:
        # legacy: radii stored in kpc/h
        r200s = np.asarray(halo_cat["radii"], dtype=np.float64) / 1000.0
    else:
        r200s = np.zeros(len(centers))
    halos = [
        {"halo_center": centers[i], "halo_mass": float(halo_masses[i]),
         "r200": float(r200s[i])}
        for i in range(len(centers))
    ]
    if not halos:
        return None

    generated = np.load(gen_path)["generated"]  # (N, C, P, P)
    if generated.ndim != 4 or generated.shape[0] != len(halos):
        return None
    patch_pix = generated.shape[-1]
    gen_gas = generated[:, GAS_CH]  # (N, P, P)

    # Truth: prefer cube artifact, fallback to full-map cutout
    truth_cube_path = mass_dir / "truth_halos_cube.npz"
    if truth_cube_path.exists():
        truth = np.load(truth_cube_path)["truth_halos"]  # (N, 3, P, P)
        if truth.shape[0] != len(halos):
            return None
        truth_gas = truth[:, GAS_CH]
    else:
        truth_gas = _extract_truth_gas_from_full_maps(
            snap_dir / "full_maps.npz",
            halos,
            box_size=box_size,
            npix=int(generated.shape[-1] * box_size / patch_size_mpc_h),
            patch_pix=patch_pix,
        )
        if truth_gas.shape[0] == 0:
            return None

    r_pix, n_fallback = _aperture_radius_pixels(
        aperture=aperture,
        r200s_mpc_h=np.asarray([h["r200"] for h in halos]),
        r200_factor=r200_factor,
        patch_size_mpc_h=patch_size_mpc_h,
        patch_pix=patch_pix,
        fixed_r_pix=fixed_r_pix,
    )
    if n_fallback:
        print(f"[warn] {sim_dir.name}: {n_fallback}/{len(halos)} halos had r200=0; "
              f"using fixed_r_pix={fixed_r_pix} for those.")

    # NB: aperture_gas_mass uses a single scalar radius.  For per-halo radii
    # (R200-based) we loop; cost is negligible vs the SBI cost downstream.
    bind_mass = np.empty(len(halos), dtype=np.float64)
    truth_mass = np.empty(len(halos), dtype=np.float64)
    aperture_area = np.empty(len(halos), dtype=np.float64)
    pix_size_mpc_h = patch_size_mpc_h / patch_pix
    for i, r in enumerate(r_pix):
        bind_mass[i] = aperture_gas_mass(gen_gas[i:i + 1], float(r))[0]
        truth_mass[i] = aperture_gas_mass(truth_gas[i:i + 1], float(r))[0]
        aperture_area[i] = np.pi * (float(r) * pix_size_mpc_h) ** 2

    bind_tau = gas_mass_to_tau_in_aperture(bind_mass, aperture_area, hubble=hubble)
    truth_tau = gas_mass_to_tau_in_aperture(truth_mass, aperture_area, hubble=hubble)

    return {
        "suite": np.array([suite] * len(halos)),
        "sim_id": np.array([sim_dir.name] * len(halos)),
        "snapshot": np.array([snap] * len(halos), dtype=np.int32),
        "halo_mass_msun_h": np.asarray(halo_masses, dtype=np.float64),
        "r200_mpc_h": np.asarray([h["r200"] for h in halos], dtype=np.float64),
        "aperture_r_pix": r_pix,
        "aperture_area_mpc_h2": aperture_area,
        "bind_gas_mass_in_ap": bind_mass,
        "truth_gas_mass_in_ap": truth_mass,
        "bind_tau_ap": bind_tau,
        "truth_tau_ap": truth_tau,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testsuite_root", type=Path, required=True,
                   help="Root of test-suite outputs (contains <suite>/sim_*/...).")
    p.add_argument("--model", required=True,
                   help="Model name subdir under mass_threshold_<tag>/ (e.g. fm_two_head).")
    p.add_argument("--suites", nargs="+", default=["cv", "1p", "sb35"],
                   help="Suites to walk.")
    p.add_argument("--halo_mass_min", type=float, default=1e13,
                   help="Mass threshold tag used at runtime (default 1e13).")
    p.add_argument("--aperture", choices=["r200", "fixed"], default="r200")
    p.add_argument("--r200_factor", type=float, default=1.0,
                   help="Multiplier on R200c when --aperture r200.")
    p.add_argument("--fixed_r_pix", type=float, default=8.0,
                   help="Aperture radius in pixels when --aperture fixed.")
    p.add_argument("--box_size", type=float, default=50.0,
                   help="Simulation box size [Mpc/h] (CAMELS-25 default 50).")
    p.add_argument("--patch_size_mpc_h", type=float, default=6.25,
                   help="Physical side of each patch [Mpc/h]. Cube model default 6.25.")
    p.add_argument("--hubble", type=float, default=0.6711)
    p.add_argument("--out", type=Path, required=True,
                   help="Output npz file.")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[dict] = []
    for suite in args.suites:
        sim_dirs = _find_sim_dirs(args.testsuite_root, suite)
        if not sim_dirs:
            print(f"[skip] suite {suite}: no sim dirs under {args.testsuite_root / suite}")
            continue
        n_ok = 0
        for sd in sim_dirs:
            try:
                result = _process_sim(
                    sd, suite=suite, model_name=args.model,
                    halo_mass_min=args.halo_mass_min,
                    aperture=args.aperture, r200_factor=args.r200_factor,
                    fixed_r_pix=args.fixed_r_pix,
                    box_size=args.box_size, hubble=args.hubble,
                    patch_size_mpc_h=args.patch_size_mpc_h,
                )
            except Exception as exc:  # surface but don't kill the walk
                print(f"[err]  {suite}/{sd.name}: {exc}")
                continue
            if result is None:
                continue
            chunks.append(result)
            n_ok += 1
        print(f"[ok]   suite {suite}: {n_ok}/{len(sim_dirs)} sims processed")

    if not chunks:
        raise SystemExit("No simulations produced output; nothing to save.")

    merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0].keys()}
    meta = dict(
        model=args.model,
        aperture=args.aperture,
        r200_factor=args.r200_factor,
        fixed_r_pix=args.fixed_r_pix,
        halo_mass_min=args.halo_mass_min,
        hubble=args.hubble,
        box_size=args.box_size,
        patch_size_mpc_h=args.patch_size_mpc_h,
    )
    np.savez(
        args.out,
        meta_json=np.array(repr(meta)),
        **merged,
    )
    print(f"[save] {args.out}  ({len(merged['halo_mass_msun_h'])} halos)")


if __name__ == "__main__":
    main()
