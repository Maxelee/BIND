"""WP-B2 measurement archive: save/load PeakStackResult bundles (task 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .stacker import PeakStackResult

ARCHIVE_DIR = Path("/mnt/home/mlee1/ceph/paper3/B/wp2_measurement")


def save_result(res: PeakStackResult, out_dir: Path = ARCHIVE_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"stack_{res.label}.npz"
    payload = {
        "label": res.label, "nu_edges": res.nu_edges,
        "cap_radii_arcmin": res.cap_radii_arcmin, "n_per_bin": res.n_per_bin,
        "y_mean": res.y_mean, "y_cov": res.y_cov, "n_patches": res.n_patches,
        "profiles_r_arcmin": res.profiles_r_arcmin, "profiles": res.profiles,
        "stacked_thumbs": res.stacked_thumbs,
        "per_peak_y": res.per_peak_y, "per_peak_nu": res.per_peak_nu,
        "per_peak_patch8": res.per_peak_patch8,
    }
    for ns, sig in res.y_sigma_stability.items():
        payload[f"y_sigma_jk{ns}"] = sig
    np.savez_compressed(path, **payload)
    return path


def load_result(label: str, out_dir: Path = ARCHIVE_DIR) -> PeakStackResult:
    d = np.load(Path(out_dir) / f"stack_{label}.npz")
    stab = {k.removeprefix("y_sigma_jk"): d[k] for k in d.files if k.startswith("y_sigma_jk")}
    return PeakStackResult(
        label=str(d["label"]), nu_edges=d["nu_edges"],
        cap_radii_arcmin=d["cap_radii_arcmin"], n_per_bin=d["n_per_bin"],
        y_mean=d["y_mean"], y_cov=d["y_cov"], y_sigma_stability=stab,
        n_patches=d["n_patches"], profiles_r_arcmin=d["profiles_r_arcmin"],
        profiles=d["profiles"], stacked_thumbs=d["stacked_thumbs"],
        per_peak_y=d["per_peak_y"], per_peak_nu=d["per_peak_nu"],
        per_peak_patch8=d["per_peak_patch8"],
    )
