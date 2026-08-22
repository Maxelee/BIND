"""Enhancement-branch closure test: weak-feedback CAMELS-TNG hydro vs paired DMO,
and whether BIND reproduces it.

Question: in the weak-feedback corner of the CAMELS IllustrisTNG L50n512 (50
Mpc/h, SB35) Sobol suite, does real hydro small-scale matter power *exceed*
the paired DMO-only run (baryons condensing into halo cores can outweigh
feedback-driven smoothing at k >~ 1-2 h/Mpc -- the "enhancement branch" seen
in van Daalen-style baryonification studies)? And does BIND, painted at the
same Sobol nodes with the released flow-matching emulator, reproduce that
enhancement (not just suppress it, which is the model's more common regime)?

Pipeline (reuses bind.inference machinery -- notebook-consistent, same code
path as `bind-camels-suite --suite sb35`):
  1. Resolve the SB35 raw-data roots (recorded in run_test_suite.sh; see the
     provenance note below) and print a per-sim FOUND/MISSING table *before*
     any GPU work is requested.
  2. For sims with an already-existing evaluation (e.g. a prior
     `run_test_suite.sh` SUITE=sb35/all pass under --existing_eval_root),
     reuse the cached full_maps.npz + composite.npz and skip painting.
  3. For the rest, build a SimulationSpec per sim (config.build_sb35_specs,
     same box_size/npix/patch_pix/halo_mass_min the suite CLI uses) and run
     the paint via bind.inference.runner.run_suite with the model selected by
     RUN_DIR/MODEL_NAME/CHECKPOINT_PATH (see PROVENANCE below).
  4. Measure 2D auto power (bind.metrics.power_spectrum_pylians_2d, CIC
     deconvolved) of truth-total (=DM_hydro+Gas+Stars), DMO fullbox, and BIND
     composite-total; band-average the truth/DMO and BIND/DMO ratios in
     k in [0.5,1),[1,2),[2,4),[4,8) h/Mpc; save results + a diagnostic PNG.

PROVENANCE / decisions worth knowing before running this:
  * Data roots (SB35_PARAM_FILE/DM_ROOT/HYDRO_ROOT/GROUP_ROOT) are copied
    from run_test_suite.sh, which is the only place they are recorded in this
    repo; nothing else pins them down. They are env-overridable with the
    *same* variable names run_test_suite.sh uses, so a corrected path can be
    injected at submit time without editing this file.
  * *** THIS JOB MUST RUN ON RUSTY, NOT POPEYE. *** The CAMELS raw data
    (Sims/, FOF_Subfind/) lives on Rusty ceph; the authoring/dev session for
    this script ran on Popeye (sdceph) where none of these paths resolve, so
    the verification step below is UNVERIFIED against real data and is the
    first thing the job does, before any GPU work is requested.
  * Model: RUN_DIR/MODEL_NAME/CHECKPOINT_PATH env overrides, mirroring
    run_test_suite.sh exactly. run_test_suite.sh's own recorded default is
    RUN_DIR=.../fm_runs/fm_two_head, MODEL_NAME=fm_two_head (NOT the
    redshift-conditioned fm_redshift_thermo the lightcone suite used) -- so
    that is this script's default too, pointed at the repo's released
    weights (`weights/fm_two_head`, populated via `bind-download-weights
    --run fm_two_head`). As of the authoring session, `weights/` in this
    checkout only actually contains `fm_redshift_thermo` (no `fm_two_head`)
    -- run bind-download-weights first, or override:
    RUN_DIR=weights/fm_redshift_thermo MODEL_NAME=fm_redshift_thermo. Either
    checkpoint works with this pipeline (extra thermo/redshift channels are
    simply unused by the composite-mass measurement here; a redshift model
    defaults the missing scale_factor to a=1, i.e. z=0, exactly what SB35
    snap_090 is).
  * Released weights use a *flat* layout (weights/<run>/last.ckpt); training
    run-directories use a *nested* one (<run_dir>/checkpoints/last.ckpt, per
    bind.cli.camels_suite). --checkpoint_path/CHECKPOINT_PATH probes both.
  * Normalization: DMO fullbox and truth-total are both CIC-projected surface
    density in the same units (Msun/h per pixel; see
    bind.inference.pipeline.load_dmo_projection / load_truth_maps) -- their
    *raw* P(k) ratio carries both the clustering-shape enhancement AND any
    mean-surface-density mismatch (baryon bookkeeping: the truth sum omits
    the black-hole particle channel; in a periodic box total matter is
    otherwise conserved, so this mismatch should be small for truth/DMO but
    is exactly where BIND's known ~+5-7% composite mass bias shows up for
    BIND/DMO). Both conventions are computed and reported: "raw" (as
    projected) and "delta" (each field divided by its own mean before the
    FFT, i.e. true overdensity contrast, which cancels any such offset).
    power_spectrum_pylians_2d's as_overdensity flag toggles exactly this.

Usage:
    python examples/enhancement_closure.py --help          # no data touched
    python examples/enhancement_closure.py --verify_only    # just the table
    python examples/enhancement_closure.py                  # full run (GPU)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_PARAM_FILE = REPO_ROOT / "src" / "bind" / "assets" / "CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt"

# ---------------------------------------------------------------------------
# Sim selection -- see the module docstring / recon report for how these were
# picked from src/bind/assets/CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt.
# Column layout (0-based index into the 35-param vector): Omega0=0, sigma8=1,
# WindEnergyIn1e51erg=2, VariableWindVelFactor=4, IMFslope=11,
# BlackHoleRadiativeEfficiency=25. Fiducial cosmology is Omega0=0.3, sigma8=0.8.
#
# Weak-feedback corner (low VariableWindVelFactor, low/steep IMFslope near its
# -2.8 end, high BlackHoleRadiativeEfficiency -- i.e. weak SN wind coupling +
# efficient-but-radiative-not-kinetic BH feedback): near-fiducial-cosmology
# picks first, per the recon report.
WEAK_FEEDBACK_SIMS = [
    "SB35_1170", "SB35_5", "SB35_347", "SB35_263",   # cosmo_dist(fiducial) < 0.12
    "SB35_1949", "SB35_596", "SB35_778",
]
# Strong-wind / strong-feedback contrast corner (opposite: high
# VariableWindVelFactor, high/shallow IMFslope near its -1.8 end, low
# BlackHoleRadiativeEfficiency), ranked by a z-scored composite score over the
# same three params and filtered to near-fiducial cosmology
# (sqrt((Om0-0.3)^2+(s8-0.8)^2) < 0.06) the same way the weak corner was --
# see the selection script referenced in the WORKLOG/session notes.
STRONG_FEEDBACK_SIMS = ["SB35_1004", "SB35_1426", "SB35_1501", "SB35_690"]
# Mid-range controls: near-fiducial cosmology (dist < 0.05) AND near-zero on
# the same feedback score (neither corner) -- i.e. unremarkable astro params.
CONTROL_SIMS = ["SB35_194", "SB35_2018"]

DEFAULT_SIMS = WEAK_FEEDBACK_SIMS + STRONG_FEEDBACK_SIMS + CONTROL_SIMS
SIM_GROUP = {
    **{s: "weak" for s in WEAK_FEEDBACK_SIMS},
    **{s: "strong" for s in STRONG_FEEDBACK_SIMS},
    **{s: "control" for s in CONTROL_SIMS},
}
GROUP_COLOR = {"weak": "tab:red", "strong": "tab:blue", "control": "tab:gray", "custom": "tab:purple"}

PARAM_IDX = {
    "Omega0": 0, "sigma8": 1, "WindEnergyIn1e51erg": 2,
    "VariableWindVelFactor": 4, "IMFslope": 11, "BlackHoleRadiativeEfficiency": 25,
}

K_BANDS = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]  # h/Mpc

# ---------------------------------------------------------------------------
# Raw-data root candidates. Primary = recorded in run_test_suite.sh. A
# fallback (non-sdceph -- this job runs on Rusty, where the canonical
# /mnt/ceph/users/camels/... roots should resolve directly) is only added
# where the primary is NOT already under /mnt/ceph/users/camels (i.e. the
# DM root + param file, which live under the ~/Sims symlink).
DM_ROOT_CANDIDATES = [
    "/mnt/home/mlee1/Sims/IllustrisTNG_DM/L50n512/SB35",
    "/mnt/ceph/users/camels/Sims/IllustrisTNG_DM/L50n512/SB35",
]
HYDRO_ROOT_CANDIDATES = [
    "/mnt/ceph/users/camels/Sims/IllustrisTNG_extras/L50n512/SB35",
]
GROUP_ROOT_CANDIDATES = [
    "/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/SB35",
]
PARAM_FILE_CANDIDATES = [
    "/mnt/home/mlee1/Sims/IllustrisTNG_DM/L50n512/SB35/CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt",
    "/mnt/ceph/users/camels/Sims/IllustrisTNG_DM/L50n512/SB35/CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt",
    str(BUNDLED_PARAM_FILE),  # bundled package asset -- identical content, always present
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enhancement-branch closure test (weak-feedback CAMELS-TNG vs DMO vs BIND).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sims", type=str, default=",".join(DEFAULT_SIMS),
                   help="Comma-separated SB35_<id> list")

    p.add_argument("--snapshot", type=int, default=90)
    p.add_argument("--box_size", type=float, default=50.0)
    p.add_argument("--npix", type=int, default=1024)
    p.add_argument("--patch_pix", type=int, default=128)
    p.add_argument("--proj_frac", type=float, default=1.0)
    p.add_argument("--halo_mass_min", type=float, default=1e13)

    p.add_argument("--sb35_param_file", type=str, default=os.environ.get("SB35_PARAM_FILE"))
    p.add_argument("--sb35_dm_root", type=str, default=os.environ.get("SB35_DM_ROOT"))
    p.add_argument("--sb35_hydro_root", type=str, default=os.environ.get("SB35_HYDRO_ROOT"))
    p.add_argument("--sb35_group_root", type=str, default=os.environ.get("SB35_GROUP_ROOT"))

    p.add_argument("--existing_eval_root", type=str,
                   default=os.environ.get("EXISTING_EVAL_ROOT", "/mnt/home/mlee1/ceph/fm_testsuite"),
                   help="Prior bind-camels-suite output root to reuse full_maps.npz/composite.npz from")
    p.add_argument("--output_root", type=str,
                   default=os.environ.get("OUTPUT_ROOT", "/mnt/home/mlee1/ceph/enhancement_closure"),
                   help="Where freshly-painted artifacts + this script's results/figure land")

    p.add_argument("--run_dir", type=str,
                   default=os.environ.get("RUN_DIR", str(REPO_ROOT / "weights" / "fm_two_head")))
    p.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME", "fm_two_head"))
    p.add_argument("--checkpoint_path", type=str, default=os.environ.get("CHECKPOINT_PATH") or None)

    p.add_argument("--n_steps", type=int, default=int(os.environ.get("N_STEPS", 50)))
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 16)))
    p.add_argument("--device", type=str, default=os.environ.get("DEVICE", "auto"))
    p.add_argument("--regenerate_all", action="store_true",
                   help="Force repaint even if cached artifacts exist under --output_root")

    p.add_argument("--threads", type=int, default=4, help="Pk_library OpenMP threads")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify_only", action="store_true",
                   help="Print the sim/param/data-availability tables and exit (no painting, no measurement)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Root resolution (probe candidates, print which one resolved)
# ---------------------------------------------------------------------------

@dataclass
class ResolvedRoot:
    chosen: Path
    tried: list[tuple[Path, bool]]
    explicit: bool


def _dir_nonempty(p: Path) -> bool:
    try:
        return p.is_dir() and next(p.iterdir(), None) is not None
    except OSError:
        return False


def _file_nonempty(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def resolve_root(explicit: str | None, candidates: list[str], checker) -> ResolvedRoot:
    if explicit:
        p = Path(explicit)
        return ResolvedRoot(chosen=p, tried=[(p, checker(p))], explicit=True)
    tried: list[tuple[Path, bool]] = []
    chosen: Path | None = None
    for c in candidates:
        p = Path(c)
        ok = checker(p)
        tried.append((p, ok))
        if ok and chosen is None:
            chosen = p
    if chosen is None:
        chosen = Path(candidates[0])
    return ResolvedRoot(chosen=chosen, tried=tried, explicit=False)


def print_root_resolution(label: str, rr: ResolvedRoot) -> None:
    src = "explicit (--flag/env)" if rr.explicit else "auto-probed"
    print(f"  {label} [{src}]:")
    for path, ok in rr.tried:
        mark = "FOUND  " if ok else "MISSING"
        chosen_mark = " <- chosen" if path == rr.chosen and ok else ""
        print(f"    [{mark}] {path}{chosen_mark}")
    if not any(ok for _, ok in rr.tried):
        print(f"    -> none resolved; proceeding with {rr.chosen} as a placeholder (all downstream checks will read MISSING)")


# ---------------------------------------------------------------------------
# Per-sim raw-data existence (mirrors bind.inference.pipeline's snapshot
# resolution logic exactly, without touching bind.inference at import time)
# ---------------------------------------------------------------------------

def _dmo_snap_files(dm_root: Path, sim_name: str, snapshot: int) -> list[Path]:
    sim_dir = dm_root / sim_name
    single = sim_dir / f"snap_{snapshot:03d}.hdf5"
    if single.is_file():
        return [single]
    pattern = sim_dir / f"snapdir_{snapshot:03d}" / f"snap_{snapshot:03d}.*.hdf5"
    return sorted(Path(x) for x in glob.glob(str(pattern)))


def _hydro_snap_files(hydro_root: Path, sim_name: str, snapshot: int) -> list[Path]:
    snapdir = hydro_root / sim_name / f"snapdir_{snapshot:03d}"
    pattern = snapdir / f"snap_{snapshot:03d}.*.hdf5"
    files = sorted(Path(x) for x in glob.glob(str(pattern)))
    if files:
        return files
    single = snapdir / f"snap_{snapshot:03d}.hdf5"
    return [single] if single.is_file() else []


def _fof_files(group_root: Path, sim_name: str, snapshot: int) -> list[Path]:
    sim_dir = group_root / sim_name
    if not sim_dir.is_dir():
        return []
    for pattern in (f"groups_{snapshot:03d}*", f"fof_subhalo_tab_{snapshot:03d}*"):
        hits = sorted(sim_dir.glob(pattern))
        if hits:
            return hits
    # lenient fallback: any file directly under the sim's group dir
    return [p for p in sorted(sim_dir.glob("*")) if p.is_file()]


def _nonempty_files(paths: list[Path]) -> bool:
    return bool(paths) and any(_file_nonempty(p) for p in paths)


@dataclass
class SimAvailability:
    sim_id: str
    group: str
    hydro_ok: bool
    hydro_path: Path
    dmo_ok: bool
    dmo_path: Path
    fof_ok: bool
    fof_path: Path
    reuse_ok: bool
    reuse_detail: str
    action: str  # REUSE | PAINT | SKIP-MISSING


def check_sim_availability(
    sim_id: str, spec, hydro_root: Path, dm_root: Path, group_root: Path,
    existing_eval_root: Path, model_name: str, snapshot: int,
) -> SimAvailability:
    from bind.inference.artifacts import resolve_artifact_paths  # lazy: heavy-ish import graph

    hydro_files = _hydro_snap_files(hydro_root, sim_id, snapshot)
    dmo_files = _dmo_snap_files(dm_root, sim_id, snapshot)
    fof_files = _fof_files(group_root, sim_id, snapshot)

    hydro_ok = _nonempty_files(hydro_files)
    dmo_ok = _nonempty_files(dmo_files)
    fof_ok = _nonempty_files(fof_files)
    raw_ok = hydro_ok and dmo_ok and fof_ok

    ap = resolve_artifact_paths(existing_eval_root, spec, model_name)
    fm_ok = _file_nonempty(ap.full_maps_npz)
    comp_ok = _file_nonempty(ap.composite_npz)
    truth_ok = False
    if fm_ok:
        try:
            with np.load(ap.full_maps_npz) as d:
                truth_ok = "truth_maps" in d.files
        except Exception:
            truth_ok = False
    reuse_ok = fm_ok and comp_ok and truth_ok
    reuse_detail = (
        f"full_maps={'OK' if fm_ok else 'missing'}"
        f"(truth_maps={'OK' if truth_ok else 'missing/absent'}) "
        f"composite={'OK' if comp_ok else 'missing'} @ {ap.model_dir}"
    )

    action = "REUSE" if reuse_ok else ("PAINT" if raw_ok else "SKIP-MISSING")

    return SimAvailability(
        sim_id=sim_id, group=SIM_GROUP.get(sim_id, "custom"),
        hydro_ok=hydro_ok, hydro_path=hydro_root / sim_id / f"snapdir_{snapshot:03d}",
        dmo_ok=dmo_ok, dmo_path=dm_root / sim_id,
        fof_ok=fof_ok, fof_path=group_root / sim_id,
        reuse_ok=reuse_ok, reuse_detail=reuse_detail, action=action,
    )


def print_availability_table(avail: list[SimAvailability]) -> None:
    print()
    print("=" * 100)
    print("PER-SIM DATA AVAILABILITY (checked before any GPU work is requested)")
    print("=" * 100)
    for a in avail:
        print(f"[{a.sim_id}] group={a.group}")
        print(f"    hydro snapdir  {'FOUND  ' if a.hydro_ok else 'MISSING'}  {a.hydro_path}")
        print(f"    dmo (nbody)    {'FOUND  ' if a.dmo_ok else 'MISSING'}  {a.dmo_path}")
        print(f"    fof catalog    {'FOUND  ' if a.fof_ok else 'MISSING'}  {a.fof_path}")
        print(f"    existing eval  {'FOUND  ' if a.reuse_ok else 'MISSING'}  {a.reuse_detail}")
        print(f"    -> action: {a.action}")
    print("-" * 100)
    n_reuse = sum(a.action == "REUSE" for a in avail)
    n_paint = sum(a.action == "PAINT" for a in avail)
    n_skip = sum(a.action == "SKIP-MISSING" for a in avail)
    print(f"Summary: {n_reuse} reuse, {n_paint} paint, {n_skip} skip-missing (of {len(avail)} requested)")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Parameter table
# ---------------------------------------------------------------------------

def load_sim_params(param_file: Path, sim_ids: list[str]) -> dict[str, np.ndarray]:
    df = pd.read_csv(param_file, sep=r"\s+", comment="#", header=None, skiprows=1)
    out: dict[str, np.ndarray] = {}
    for sim_id in sim_ids:
        row = df[df[0] == sim_id]
        if row.empty:
            raise ValueError(f"No parameters found for {sim_id} in {param_file}")
        out[sim_id] = row.iloc[0, 1:36].to_numpy(dtype=np.float32)
    return out


def print_param_table(sim_ids: list[str], params: dict[str, np.ndarray]) -> None:
    print()
    print("=" * 100)
    print("CHOSEN SIMS + PARAMETERS")
    print("=" * 100)
    header = f"{'sim':<12}{'group':<9}{'Omega0':>8}{'sigma8':>8}{'VarWindVel':>12}{'IMFslope':>10}{'BHRadEff':>10}{'WindE1e51':>11}"
    print(header)
    for sim_id in sim_ids:
        p = params[sim_id]
        g = SIM_GROUP.get(sim_id, "custom")
        print(
            f"{sim_id:<12}{g:<9}"
            f"{p[PARAM_IDX['Omega0']]:>8.3f}{p[PARAM_IDX['sigma8']]:>8.3f}"
            f"{p[PARAM_IDX['VariableWindVelFactor']]:>12.3f}{p[PARAM_IDX['IMFslope']]:>10.3f}"
            f"{p[PARAM_IDX['BlackHoleRadiativeEfficiency']]:>10.4f}{p[PARAM_IDX['WindEnergyIn1e51erg']]:>11.3f}"
        )
    print("=" * 100)


# ---------------------------------------------------------------------------
# Checkpoint resolution (released weights/ is FLAT: <run>/last.ckpt; a
# training-output run_dir is NESTED: <run_dir>/checkpoints/last.ckpt, per
# bind.cli.camels_suite -- probe both, released layout first)
# ---------------------------------------------------------------------------

def resolve_checkpoint(run_dir: Path, checkpoint_path_arg: str | None) -> Path:
    if checkpoint_path_arg:
        return Path(checkpoint_path_arg)
    flat = run_dir / "last.ckpt"
    if flat.is_file():
        return flat
    return run_dir / "checkpoints" / "last.ckpt"


# ---------------------------------------------------------------------------
# Power spectrum measurement
# ---------------------------------------------------------------------------

def band_average(k: np.ndarray, values: np.ndarray, nmodes: np.ndarray, lo: float, hi: float) -> float:
    mask = (k >= lo) & (k < hi) & np.isfinite(values)
    if not np.any(mask):
        return float("nan")
    w = nmodes[mask].astype(np.float64)
    v = values[mask].astype(np.float64)
    if w.sum() <= 0:
        return float(np.mean(v))
    return float(np.sum(v * w) / np.sum(w))


def measure_pk(field: np.ndarray, box_size: float, threads: int) -> dict:
    """Both normalization conventions for one 2D field; see module docstring."""
    from bind.metrics import power_spectrum_pylians_2d

    k, pk_delta, nmodes = power_spectrum_pylians_2d(
        field, box_size=box_size, MAS="CIC", threads=threads, as_overdensity=True
    )
    _, pk_raw, _ = power_spectrum_pylians_2d(
        field, box_size=box_size, MAS="CIC", threads=threads, as_overdensity=False
    )
    return {"k": k, "pk_delta": pk_delta, "pk_raw": pk_raw, "nmodes": nmodes, "mean": float(np.mean(field))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    sim_ids = [s.strip() for s in args.sims.split(",") if s.strip()]
    sim_ids = [s if s.startswith("SB35_") else f"SB35_{s}" for s in sim_ids]  # normalize bare ids
    if not sim_ids:
        raise SystemExit("--sims produced an empty list")

    output_root = Path(args.output_root)
    existing_eval_root = Path(args.existing_eval_root)
    run_dir = Path(args.run_dir)
    results_dir = output_root / "enhancement_closure"

    print("=" * 100)
    print("ENHANCEMENT-BRANCH CLOSURE TEST")
    print("=" * 100)
    print(f"Sims requested ({len(sim_ids)}): {sim_ids}")
    print(f"Snapshot: {args.snapshot}  box_size: {args.box_size} Mpc/h  npix: {args.npix}  patch_pix: {args.patch_pix}")
    print(f"halo_mass_min: {args.halo_mass_min:.2e}")
    print(f"Model: run_dir={run_dir}  model_name={args.model_name}")
    print(f"Output root: {output_root}")
    print(f"Existing-eval reuse root: {existing_eval_root}")
    print("=" * 100)

    # ── Root resolution ────────────────────────────────────────────────────
    print("\nRAW-DATA ROOT RESOLUTION")
    dm_root_r = resolve_root(args.sb35_dm_root, DM_ROOT_CANDIDATES, _dir_nonempty)
    print_root_resolution("SB35_DM_ROOT", dm_root_r)
    hydro_root_r = resolve_root(args.sb35_hydro_root, HYDRO_ROOT_CANDIDATES, _dir_nonempty)
    print_root_resolution("SB35_HYDRO_ROOT", hydro_root_r)
    group_root_r = resolve_root(args.sb35_group_root, GROUP_ROOT_CANDIDATES, _dir_nonempty)
    print_root_resolution("SB35_GROUP_ROOT", group_root_r)
    param_file_r = resolve_root(args.sb35_param_file, PARAM_FILE_CANDIDATES, _file_nonempty)
    print_root_resolution("SB35_PARAM_FILE", param_file_r)

    dm_root, hydro_root, group_root, param_file = (
        dm_root_r.chosen, hydro_root_r.chosen, group_root_r.chosen, param_file_r.chosen,
    )

    # ── Params + spec construction (only needs the param file, which always
    #    resolves thanks to the bundled-asset fallback) ─────────────────────
    bare_ids = [s.split("SB35_", 1)[-1] for s in sim_ids]
    params = load_sim_params(param_file, sim_ids)
    print_param_table(sim_ids, params)

    from bind.inference.config import build_sb35_specs

    specs = build_sb35_specs(
        sim_ids=bare_ids, param_file=param_file, nbody_root=dm_root, hydro_root=hydro_root,
        fof_root=group_root, snapshot=args.snapshot, box_size=args.box_size, npix=args.npix,
        patch_pix=args.patch_pix, proj_frac=args.proj_frac, halo_mass_min=args.halo_mass_min,
    )
    specs_by_sim = {f"SB35_{s.sim_id}": s for s in specs}

    # ── Verification ─────────────────────────────────────────────────────
    avail = [
        check_sim_availability(
            sim_id, specs_by_sim[sim_id], hydro_root, dm_root, group_root,
            existing_eval_root, args.model_name, args.snapshot,
        )
        for sim_id in sim_ids
    ]
    print_availability_table(avail)

    runnable = [a for a in avail if a.action != "SKIP-MISSING"]
    if not runnable:
        print()
        print("!" * 100)
        print("!!! ERROR: SB35 raw data is NOT at the recorded location, and no reusable prior")
        print("!!! evaluation artifacts were found either. Every selected sim is unreachable.")
        print(f"!!! Checked DM_ROOT={dm_root}  HYDRO_ROOT={hydro_root}  GROUP_ROOT={group_root}")
        print(f"!!! and EXISTING_EVAL_ROOT={existing_eval_root}.")
        print("!!! This job must run on Rusty (data locality) -- see the module docstring.")
        print("!!! Override paths via SB35_DM_ROOT/SB35_HYDRO_ROOT/SB35_GROUP_ROOT/SB35_PARAM_FILE")
        print("!!! or the matching --sb35_* flags, and/or EXISTING_EVAL_ROOT.")
        print("!" * 100)
        sys.exit(2)

    if args.verify_only:
        print("\n--verify_only: stopping after the availability check.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Paint the sims that need it ─────────────────────────────────────
    paint_ids = [a.sim_id for a in avail if a.action == "PAINT"]
    if paint_ids:
        from bind.inference.runner import run_suite
        from bind.inference.schemas import RunConfig

        checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint_path)
        norm_stats_path = run_dir / "norm_stats.npz"
        if not checkpoint_path.is_file() or not norm_stats_path.is_file():
            print()
            print("!" * 100)
            print(f"!!! ERROR: model checkpoint/norm_stats not found for run_dir={run_dir}")
            print(f"!!!   checkpoint: {checkpoint_path} ({'OK' if checkpoint_path.is_file() else 'MISSING'})")
            print(f"!!!   norm_stats: {norm_stats_path} ({'OK' if norm_stats_path.is_file() else 'MISSING'})")
            print(f"!!! Run: bind-download-weights --run {args.model_name}")
            print("!!! or override RUN_DIR/CHECKPOINT_PATH (env or --run_dir/--checkpoint_path).")
            print("!" * 100)
            sys.exit(2)

        print(f"\nPainting {len(paint_ids)} sim(s) with checkpoint={checkpoint_path}")
        run_cfg = RunConfig(
            run_dir=run_dir, checkpoint_path=checkpoint_path, output_root=output_root,
            model_name=args.model_name, n_steps=args.n_steps, batch_size=args.batch_size,
            device=args.device, regenerate_all=args.regenerate_all,
        )
        paint_specs = [specs_by_sim[s] for s in paint_ids]
        summaries = run_suite(paint_specs, run_cfg, load_truth=True, max_workers=1)
        print(f"Painting done: {len(summaries)}/{len(paint_specs)} sims completed")

    # ── Load full_maps.npz + composite.npz for every runnable sim ───────
    from bind.inference.artifacts import load_composite, load_full_maps, resolve_artifact_paths

    measured: list[dict] = []
    for a in avail:
        if a.action == "SKIP-MISSING":
            continue
        root = existing_eval_root if a.action == "REUSE" else output_root
        spec = specs_by_sim[a.sim_id]
        paths = resolve_artifact_paths(root, spec, args.model_name)
        try:
            dmo_fullbox, truth_maps = load_full_maps(paths.full_maps_npz)
            if truth_maps is None:
                raise ValueError("full_maps.npz has no truth_maps (was it produced with --skip_truth?)")
            composite = load_composite(paths.composite_npz)["composite"]
        except Exception as exc:
            print(f"[{a.sim_id}] WARNING: could not load artifacts from {root} ({a.action}): {exc} -- skipping")
            continue
        measured.append({
            "sim_id": a.sim_id, "group": a.group, "action": a.action, "root": str(root),
            "dmo_fullbox": dmo_fullbox, "truth_total": truth_maps.sum(axis=0),
            "bind_total": composite.sum(axis=0),
        })

    if not measured:
        print("\nERROR: no sim produced usable full_maps.npz/composite.npz -- nothing to measure.")
        sys.exit(2)

    # ── Power spectra + band ratios ──────────────────────────────────────
    print(f"\nMeasuring power spectra for {len(measured)} sim(s) (box_size={args.box_size} Mpc/h, MAS=CIC)...")
    sim_results: list[dict] = []
    k_ref: np.ndarray | None = None
    stacked = {key: [] for key in (
        "pk_truth_delta", "pk_dmo_delta", "pk_bind_delta",
        "pk_truth_raw", "pk_dmo_raw", "pk_bind_raw", "nmodes",
    )}
    stacked_sim_ids: list[str] = []
    stacked_groups: list[str] = []
    stacked_means: list[list[float]] = []

    for m in measured:
        pk_truth = measure_pk(m["truth_total"], args.box_size, args.threads)
        pk_dmo = measure_pk(m["dmo_fullbox"], args.box_size, args.threads)
        pk_bind = measure_pk(m["bind_total"], args.box_size, args.threads)
        k = pk_truth["k"]
        if k_ref is None:
            k_ref = k

        s_box_delta = pk_truth["pk_delta"] / pk_dmo["pk_delta"]
        s_box_raw = pk_truth["pk_raw"] / pk_dmo["pk_raw"]
        s_bind_delta = pk_bind["pk_delta"] / pk_dmo["pk_delta"]
        s_bind_raw = pk_bind["pk_raw"] / pk_dmo["pk_raw"]

        bands = {}
        for lo, hi in K_BANDS:
            key = f"{lo:g}-{hi:g}"
            bands[key] = {
                "S_box_delta": band_average(k, s_box_delta, pk_truth["nmodes"], lo, hi),
                "S_box_raw": band_average(k, s_box_raw, pk_truth["nmodes"], lo, hi),
                "S_bind_delta": band_average(k, s_bind_delta, pk_truth["nmodes"], lo, hi),
                "S_bind_raw": band_average(k, s_bind_raw, pk_truth["nmodes"], lo, hi),
            }

        mean_ratio_truth_dmo_sq = (pk_truth["mean"] / pk_dmo["mean"]) ** 2
        mean_ratio_bind_dmo_sq = (pk_bind["mean"] / pk_dmo["mean"]) ** 2

        sim_results.append({
            "sim_id": m["sim_id"], "group": m["group"], "action": m["action"], "root": m["root"],
            "params": {name: float(params[m["sim_id"]][idx]) for name, idx in PARAM_IDX.items()},
            "mean_truth": pk_truth["mean"], "mean_dmo": pk_dmo["mean"], "mean_bind": pk_bind["mean"],
            "mean_ratio_truth_dmo_sq": mean_ratio_truth_dmo_sq,
            "mean_ratio_bind_dmo_sq": mean_ratio_bind_dmo_sq,
            "k_bands": bands,
        })

        stacked["pk_truth_delta"].append(pk_truth["pk_delta"])
        stacked["pk_dmo_delta"].append(pk_dmo["pk_delta"])
        stacked["pk_bind_delta"].append(pk_bind["pk_delta"])
        stacked["pk_truth_raw"].append(pk_truth["pk_raw"])
        stacked["pk_dmo_raw"].append(pk_dmo["pk_raw"])
        stacked["pk_bind_raw"].append(pk_bind["pk_raw"])
        stacked["nmodes"].append(pk_truth["nmodes"])
        stacked_sim_ids.append(m["sim_id"])
        stacked_groups.append(m["group"])
        stacked_means.append([pk_truth["mean"], pk_dmo["mean"], pk_bind["mean"]])

        print(
            f"  [{m['sim_id']:<10} {m['group']:<7} {m['action']:<6}] "
            f"mean(truth/dmo)^2={mean_ratio_truth_dmo_sq:.4f}  mean(bind/dmo)^2={mean_ratio_bind_dmo_sq:.4f}  "
            f"S_box_delta[2-4]={bands['2-4']['S_box_delta']:.3f}  S_bind_delta[2-4]={bands['2-4']['S_bind_delta']:.3f}"
        )

    # ── Save results npz ──────────────────────────────────────────────
    results_npz = results_dir / "pk_results.npz"
    np.savez(
        results_npz,
        k=k_ref,
        sim_ids=np.asarray(stacked_sim_ids),
        groups=np.asarray(stacked_groups),
        means_truth_dmo_bind=np.asarray(stacked_means, dtype=np.float64),
        **{key: np.stack(val) for key, val in stacked.items()},
    )
    print(f"\nSaved {results_npz}")

    # ── Group-level summary ──────────────────────────────────────────
    group_summary: dict[str, dict] = {}
    for group in sorted(set(r["group"] for r in sim_results)):
        rows = [r for r in sim_results if r["group"] == group]
        gs: dict[str, dict] = {}
        for lo, hi in K_BANDS:
            key = f"{lo:g}-{hi:g}"
            gs[key] = {
                metric: float(np.nanmean([r["k_bands"][key][metric] for r in rows]))
                for metric in ("S_box_delta", "S_box_raw", "S_bind_delta", "S_bind_raw")
            }
        group_summary[group] = {"n_sims": len(rows), "bands": gs}

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "box_size_mpch": args.box_size, "npix": args.npix, "snapshot": args.snapshot,
        "halo_mass_min": args.halo_mass_min,
        "run_dir": str(run_dir), "model_name": args.model_name,
        "k_bands_hmpc": K_BANDS,
        "normalization_note": (
            "delta = each field divided by its own mean before the FFT (overdensity contrast; "
            "cancels absolute mean-density offsets, i.e. baryon-bookkeeping mismatches). "
            "raw = as-projected surface density (no per-field mean division); its ratio also "
            "carries (mean_num/mean_den)^2, which is where BIND's known composite mass bias shows up."
        ),
        "sims": sim_results,
        "group_summary": group_summary,
    }
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False))
    print(f"Saved {summary_path}")

    # ── Figure ──────────────────────────────────────────────────────
    fig_path = results_dir / "enhancement_closure.png"
    make_figure(k_ref, stacked, stacked_sim_ids, stacked_groups, group_summary, fig_path)
    print(f"Saved {fig_path}")

    print("\nDone.")


def make_figure(
    k: np.ndarray, stacked: dict, sim_ids: list[str], groups: list[str],
    group_summary: dict, out_path: Path,
) -> None:
    """Quick diagnostic PNG (not part of the paper figure package)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_pk, ax_txt) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.4, 1.0]})

    seen_labels = set()
    for i, sim_id in enumerate(sim_ids):
        group = groups[i]
        color = GROUP_COLOR.get(group, "black")
        s_box = stacked["pk_truth_delta"][i] / stacked["pk_dmo_delta"][i]
        s_bind = stacked["pk_bind_delta"][i] / stacked["pk_dmo_delta"][i]
        label = group if group not in seen_labels else None
        seen_labels.add(group)
        ax_pk.plot(k, s_box, color=color, lw=1.3, alpha=0.85, label=label)
        ax_pk.plot(k, s_bind, color=color, lw=1.3, alpha=0.85, ls="--")

    ax_pk.axhline(1.0, color="k", lw=0.8, ls=":")
    for lo, hi in K_BANDS:
        ax_pk.axvspan(lo, hi, color="gray", alpha=0.05)
    ax_pk.set_xscale("log")
    ax_pk.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]")
    ax_pk.set_ylabel(r"$S(k) = P(k) / P_\mathrm{DMO}(k)$   (delta convention)")
    ax_pk.set_title("Solid: truth/DMO $S_\\mathrm{box}(k)$.  Dashed: BIND/DMO $S_\\mathrm{bind}(k)$")
    ax_pk.legend(loc="best", fontsize=9)
    ax_pk.grid(alpha=0.2)

    ax_txt.axis("off")
    lines = ["k-band summary (nmodes-weighted mean over sims in group)", ""]
    header = f"{'band':<9}{'group':<9}{'S_box':>8}{'S_bind':>9}{'n':>4}"
    lines.append(header)
    lines.append("-" * len(header))
    for lo, hi in K_BANDS:
        key = f"{lo:g}-{hi:g}"
        for group in ("weak", "strong", "control"):
            if group not in group_summary:
                continue
            b = group_summary[group]["bands"][key]
            n = group_summary[group]["n_sims"]
            lines.append(f"{key:<9}{group:<9}{b['S_box_delta']:>8.3f}{b['S_bind_delta']:>9.3f}{n:>4}")
        lines.append("")
    ax_txt.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9, va="top", ha="left",
                transform=ax_txt.transAxes)

    fig.suptitle("Enhancement-branch closure: weak-feedback CAMELS-TNG vs DMO vs BIND (SB35, L50n512)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
