"""Assemble the WP-A4 emulator training set from the v2 gate tables.

One row per (design point, snapshot). The observable vector concatenates
five blocks, all in the *painted-observable* frame (no CylToSph, no kSZ
normalization — those are forward-model steps applied downstream in
:mod:`forward`, where their theta-dependence and systematics live):

- ``fgas_med``  (5): median cylindrical f_gas(<R500) per log-M500 bin
  (the A3 gate bins, ``LOGM500_BIN_EDGES``).
- ``fgas_scat`` (5): per-bin (p84 - p16)/2 of f_gas — BIND's per-halo
  stochasticity is an emulated output, not a nuisance.
- ``ym``        (3): Y-M power law (alpha, beta at 1e14, scatter_dex).
- ``ksz0/ksz1`` (9+9): stacked tau_CAP profiles, two mass bins (v2:
  diffuse-background composite; velocity decorrelation applied downstream).

Bins empty at a snapshot (top mass bins at high z) are empty for *every*
design point (the DMO halo catalog is shared across the design), so validity
is a per-snapshot column mask, not per-row raggedness.

Per-row noise floor: bootstrap over halos (fgas and ym blocks). The kSZ
stacks ship without per-halo profiles, so their floor is not estimable here
(NaN, flagged); the kSZ acceptance test uses the frozen measurement errors
instead. Note the same DMO halos underlie every design point, so bootstrap
noise is *shared* across rows — an emulator interpolating smooth theta-response
can legitimately sit below this floor.

Usage::

    python -m analysis.paper3a.emulator.dataset  # writes gasemu_dataset.npz
"""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

import numpy as np

from analysis.paper3a.gate.aggregate import LOGM500_BIN_EDGES, TABLES, _SNAP_Z
from analysis.paper3a.observables import fit_ym_relation

from . import params_meta as pm

OUT_DIR = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator")
DATASET = OUT_DIR / "gasemu_dataset.npz"

SNAPS = ("096", "071", "067", "063", "056", "049")
BLOCKS = ("fgas_med", "fgas_scat", "ym", "ksz0", "ksz1")
BLOCK_SIZES = {"fgas_med": 5, "fgas_scat": 5, "ym": 3, "ksz0": 9, "ksz1": 9}
# v3 tables additionally carry the own-patch CAP stacks and the stacked
# Sigma(R) profiles the decorrelation forward model consumes; when present
# they become extra emulated blocks (block name -> table key).
V3_BLOCK_KEYS = {
    "ksz0_own": "ksz_bin0_stack_own", "ksz1_own": "ksz_bin1_stack_own",
    "ksz0_sr": "ksz_bin0_sigma_r", "ksz1_sr": "ksz_bin1_sigma_r",
    "ksz0_sr_own": "ksz_bin0_sigma_r_own", "ksz1_sr_own": "ksz_bin1_sigma_r_own",
}
D_TOT = sum(BLOCK_SIZES.values())
N_BOOT = 100
MIN_HALOS_PER_BIN = 5


def detect_blocks(table: dict) -> tuple[tuple[str, ...], dict[str, int]]:
    """Block layout for a loaded table: the v2 base, plus the v3 columns
    when the table carries them."""
    blocks, sizes = list(BLOCKS), dict(BLOCK_SIZES)
    for name, key in V3_BLOCK_KEYS.items():
        if key in table:
            blocks.append(name)
            sizes[name] = len(np.atleast_1d(table[key]))
    return tuple(blocks), sizes


def block_slices(blocks: tuple[str, ...] = BLOCKS,
                 sizes: dict[str, int] | None = None) -> dict[str, slice]:
    sizes = sizes or BLOCK_SIZES
    out, d0 = {}, 0
    for b in blocks:
        out[b] = slice(d0, d0 + sizes[b])
        d0 += sizes[b]
    return out


def _fgas_bin_stats(logm500: np.ndarray, fgas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(median, half 16-84 width) of cylindrical f_gas per gate mass bin."""
    med = np.full(5, np.nan)
    scat = np.full(5, np.nan)
    for i in range(5):
        sel = (logm500 >= LOGM500_BIN_EDGES[i]) & (logm500 < LOGM500_BIN_EDGES[i + 1])
        if sel.sum() >= MIN_HALOS_PER_BIN:
            med[i] = np.median(fgas[sel])
            p16, p84 = np.percentile(fgas[sel], [16, 84])
            scat[i] = 0.5 * (p84 - p16)
    return med, scat


def _row_from_table(d: dict, rng: np.random.Generator | None,
                    blocks: tuple[str, ...] = BLOCKS,
                    sizes: dict[str, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(observable vector, bootstrap SEM) for one loaded table."""
    sl = block_slices(blocks, sizes)
    d_tot = sum((sizes or BLOCK_SIZES)[b] for b in blocks)
    y = np.full(d_tot, np.nan)
    sem = np.full(d_tot, np.nan)

    logm500 = np.log10(d["m500_msunh"])
    fgas = d["fgas_cyl_r500"]
    y[sl["fgas_med"]], y[sl["fgas_scat"]] = _fgas_bin_stats(logm500, fgas)
    f = fit_ym_relation(d["m500_msunh"], d["y_cyl_r500_mpc2"])
    y[sl["ym"]] = [f.alpha, f.beta, f.scatter_dex]
    y[sl["ksz0"]] = d["ksz_bin0_stack"]
    y[sl["ksz1"]] = d["ksz_bin1_stack"]
    for name in blocks:
        if name in V3_BLOCK_KEYS:
            y[sl[name]] = d[V3_BLOCK_KEYS[name]]

    if rng is not None:
        n = len(fgas)
        boots = np.full((N_BOOT, 13), np.nan)  # 5 med + 5 scat + 3 ym
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            m, s = _fgas_bin_stats(logm500[idx], fgas[idx])
            fb = fit_ym_relation(d["m500_msunh"][idx], d["y_cyl_r500_mpc2"][idx])
            boots[b] = np.concatenate([m, s, [fb.alpha, fb.beta, fb.scatter_dex]])
        sem[:13] = np.nanstd(boots, axis=0)
    return y, sem


def _twobound_identity(params35: np.ndarray) -> tuple[str, str]:
    """(varied param name, 'low'|'high') of a 1P-extreme design point."""
    u = pm.table_params_to_unit(params35)
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    i = int(np.argmax(np.abs(u - u_fid)))
    return pm.ASTRO_NAMES[i], "low" if u[i] < u_fid[i] else "high"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return "unknown"


def build(tables_dir: Path = TABLES, out_path: Path = DATASET,
          n_boot: int = N_BOOT, verbose: bool = True) -> dict:
    first = sorted(tables_dir.glob(f"sb35_run_*_snap{SNAPS[0]}.npz"))
    if not first:
        raise FileNotFoundError(f"no sb35 tables under {tables_dir}")
    blocks, sizes = detect_blocks(dict(np.load(first[0], allow_pickle=False)))
    d_tot = sum(sizes[b] for b in blocks)
    arrays: dict = {
        "block_names": np.array(blocks),
        "block_sizes": np.array([sizes[b] for b in blocks]),
        "snaps": np.array(SNAPS),
        "snap_z": np.array([_SNAP_Z[s] for s in SNAPS]),
        "logm500_bin_edges": LOGM500_BIN_EDGES,
        "param_names": np.array(pm.ASTRO_NAMES),
        "param_min": pm.ASTRO_MIN, "param_max": pm.ASTRO_MAX,
        "param_log": pm.ASTRO_LOG, "param_fiducial": pm.ASTRO_FIDUCIAL,
        "fixed_cosmology_names": np.array(list(pm.FIXED_COSMOLOGY)),
        "fixed_cosmology_values": np.array(list(pm.FIXED_COSMOLOGY.values())),
    }
    radii_written = False
    for snap in SNAPS:
        for bundle in ("sb35", "twobound", "fiducial"):
            paths = sorted(tables_dir.glob(f"{bundle}_run_*_snap{snap}.npz"))
            if not paths:
                raise FileNotFoundError(f"no {bundle} tables for snap {snap} under {tables_dir}")
            X, Y, SEM, runs = [], [], [], []
            tb_id = []
            for p in paths:
                d = dict(np.load(p, allow_pickle=False))
                if not radii_written:
                    arrays["radii_arcmin"] = d["radii_arcmin"]
                    arrays["ksz_bin0_range"] = d["ksz_bin0_range"]
                    arrays["ksz_bin1_range"] = d["ksz_bin1_range"]
                    if "sigma_r_centers_mpch" in d:
                        arrays["sigma_r_centers_mpch"] = d["sigma_r_centers_mpch"]
                    radii_written = True
                rng = np.random.default_rng(abs(hash((p.stem, "boot"))) % 2**32)
                yv, sem = _row_from_table(d, rng if n_boot else None, blocks, sizes)
                X.append(pm.table_params_to_unit(d["params"]))
                Y.append(yv)
                SEM.append(sem)
                runs.append(p.stem.split("_snap")[0].split(f"{bundle}_")[-1])
                if bundle == "twobound":
                    tb_id.append(_twobound_identity(d["params"]))
            arrays[f"snap{snap}_X_{bundle}"] = np.array(X)
            arrays[f"snap{snap}_Y_{bundle}"] = np.array(Y)
            arrays[f"snap{snap}_sem_{bundle}"] = np.array(SEM)
            arrays[f"snap{snap}_runs_{bundle}"] = np.array(runs)
            if bundle == "twobound":
                arrays[f"snap{snap}_twobound_param"] = np.array([t[0] for t in tb_id])
                arrays[f"snap{snap}_twobound_bound"] = np.array([t[1] for t in tb_id])
        # a column is emulated at this snapshot iff finite for every Sobol row
        arrays[f"snap{snap}_valid"] = np.isfinite(arrays[f"snap{snap}_Y_sb35"]).all(axis=0)
        if verbose:
            nv = int(arrays[f"snap{snap}_valid"].sum())
            print(f"snap {snap}: sb35 {arrays[f'snap{snap}_Y_sb35'].shape}, "
                  f"twobound {arrays[f'snap{snap}_Y_twobound'].shape}, "
                  f"valid dims {nv}/{d_tot}", flush=True)

    arrays["provenance"] = np.array(json.dumps({
        "created": datetime.date.today().isoformat(),
        "tables_dir": str(tables_dir),
        "table_version": 3 if any(b in V3_BLOCK_KEYS for b in blocks) else 2,
        "n_boot": n_boot, "git_sha": _git_sha(),
        "frame": ("painted observables: cylindrical f_gas (no CylToSph), "
                  "tau_CAP arcmin^2 (no velocity decorrelation / normalization); "
                  "forward-model corrections applied downstream"),
    }))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    if verbose:
        print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    return arrays


if __name__ == "__main__":
    build()
