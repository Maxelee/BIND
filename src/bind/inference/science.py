"""Orchestration for the BIND astro-parameter science runs (1a / 1b).

The expensive, parameter-*independent* work — projecting the TNG300-Dark box
into z-slabs and extracting per-halo DMO cutouts (stage 1) — is done **once per
snapshot** and shared across every parameter run.  Only stage 2 (the flow-matching
``model.generate`` + composite) depends on the astro parameters, so the science
sweep is a fan-out of stage-2 jobs over (run x snapshot) that all read the same
shared stage-1 directories.

:func:`plan_generate` lays out the run tree, writes a per-run ``params.npy``, and
emits a `disBatch <https://github.com/flatironinstitute/disBatch/>`_ task file —
one ``bind-paint-generate`` command per (run, snapshot) — plus a ``plan.json``
manifest.  Submit the task file inside a GPU allocation with disBatch; nothing
here launches Slurm jobs.

Layout under ``output_root``::

    design/                         # bind-design output (param matrices)
    runs/<design>/run_<i:04d>/
        params.npy                  # the 35-dim vector for this run
        snap_<NNN>/composite_slab*.npz
    generate_tasks_<design>.db      # disBatch task file
    plan_<design>.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Lightcone snapshot order (low-z first), matching run_lightcone_*.sh.
LIGHTCONE_SNAPSHOTS: tuple[int, ...] = (
    96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29,
)


def run_dir(output_root: Path | str, design: str, i: int) -> Path:
    return Path(output_root) / "runs" / design / f"run_{i:04d}"


def _stage1_dir(shared_stage1_root: Path | str, snap: int) -> Path:
    return Path(shared_stage1_root) / f"snap_{snap:03d}" / "stage1"


def plan_generate(
    *,
    output_root: Path | str,
    design: str,
    params_matrix: np.ndarray,
    shared_stage1_root: Path | str,
    weights: str = "weights/fm_redshift_thermo",
    snapshots: tuple[int, ...] = LIGHTCONE_SNAPSHOTS,
    r200_factor: float = 4.0,
    n_steps: int = 50,
    batch_size: int = 16,
    save_patches: bool = False,
    python: str = "python -u -m bind.cli.paint_generate",
) -> dict:
    """Write per-run params + a disBatch task file for the stage-2 fan-out.

    Parameters
    ----------
    params_matrix : (n_runs, 35) or (35,) — one row per run.
    shared_stage1_root : root holding ``snap_<NNN>/stage1`` for each snapshot
        (e.g. the existing ``/ceph/bind_lightcone_tng``).
    save_patches : keep per-halo generated/thermo patches in each
        ``composite_slab*.npz``.  Off by default to bound disk on the 256-run
        Sobol sweep; turn on for fiducial/1P where per-halo diagnostics are wanted.

    Returns the ``plan.json`` dict (also written to disk).  Skips (run, snapshot)
    pairs whose composite already exists so the task file is restart-safe.
    """
    output_root = Path(output_root)
    # Absolutise a relative weights path (resolved against the cwd at plan time,
    # i.e. the repo root) so disBatch tasks are independent of their run cwd.
    wp = Path(weights)
    if not wp.is_absolute() and wp.exists():
        weights = str(wp.resolve())
    pm = np.atleast_2d(np.asarray(params_matrix, dtype=np.float64))
    if pm.shape[1] != 35:
        raise ValueError(f"params_matrix must have 35 columns, got {pm.shape}")
    n_runs = pm.shape[0]

    # Validate the shared stage-1 dirs up front — a missing one is a silent
    # data-loss footgun across thousands of tasks.
    missing = [s for s in snapshots
               if not (_stage1_dir(shared_stage1_root, s) / "stage1_manifest.json").exists()]
    if missing:
        raise FileNotFoundError(
            f"shared stage-1 manifest missing for snapshots {missing} under "
            f"{shared_stage1_root}")

    extra = []
    if not save_patches:
        extra.append("--no_save_patches")

    task_lines: list[str] = []
    n_done = 0
    for i in range(n_runs):
        rd = run_dir(output_root, design, i)
        rd.mkdir(parents=True, exist_ok=True)
        params_path = rd / "params.npy"
        np.save(params_path, pm[i])
        for s in snapshots:
            out = rd / f"snap_{s:03d}"
            # restart-safe: skip pairs already composited
            if (out / "composite_slab00.npz").exists():
                n_done += 1
                continue
            cmd = (
                f"{python} "
                f"--stage1_dir {_stage1_dir(shared_stage1_root, s)} "
                f"--params {params_path} "
                f"--run_dir {weights} "
                f"--output_dir {out} "
                f"--r200_factor {r200_factor} "
                f"--n_steps {n_steps} --batch_size {batch_size} --device auto"
                + (" " + " ".join(extra) if extra else "")
            )
            task_lines.append(cmd)

    task_file = output_root / f"generate_tasks_{design}.db"
    header = (
        f"# disBatch task file — BIND science stage 2 ({design})\n"
        f"# {len(task_lines)} tasks ({n_runs} runs x {len(snapshots)} snapshots, "
        f"{n_done} already done & skipped)\n"
        f"# Submit inside a GPU allocation, e.g.:\n"
        f"#   disBatch -p {output_root}/db_logs_{design} {task_file.name}\n"
    )
    task_file.write_text(header + "\n".join(task_lines) + ("\n" if task_lines else ""))

    plan = {
        "design": design,
        "n_runs": int(n_runs),
        "snapshots": list(snapshots),
        "n_tasks": len(task_lines),
        "n_already_done": int(n_done),
        "shared_stage1_root": str(shared_stage1_root),
        "weights": weights,
        "r200_factor": float(r200_factor),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "save_patches": bool(save_patches),
        "task_file": str(task_file),
        "run_dir_template": str(run_dir(output_root, design, 0)).replace("0000", "<i:04d>"),
    }
    (output_root / f"plan_{design}.json").write_text(json.dumps(plan, indent=1))
    return plan


def plan_runs(
    *,
    output_root: Path | str,
    design: str,
    params_matrix: np.ndarray,
) -> dict:
    """Write per-run ``params.npy`` for the canonical (ray-traced) per-run driver.

    Each run is executed end-to-end by ``run_science_run.sh`` (generate halos →
    composite → lensplanes+y-planes → lux → collect → stats → delete transients),
    submitted as a SLURM array over ``run_<i>``.  This is the production path;
    the Born ``plan_maps_stats`` below is retained only for quick diagnostics.
    """
    output_root = Path(output_root)
    pm = np.atleast_2d(np.asarray(params_matrix, dtype=np.float64))
    if pm.shape[1] != 35:
        raise ValueError(f"params_matrix must have 35 columns, got {pm.shape}")
    n_runs = pm.shape[0]
    n_todo = 0
    for i in range(n_runs):
        rd = run_dir(output_root, design, i)
        rd.mkdir(parents=True, exist_ok=True)
        np.save(rd / "params.npy", pm[i])
        if not (rd / "Cl_kappa.npz").exists():
            n_todo += 1
    plan = {"design": design, "n_runs": int(n_runs), "n_todo": int(n_todo),
            "driver": "run_science_run.sh",
            "submit": f"sbatch --array=0-{n_runs-1} --export=ALL,DESIGN={design},"
                      f"OUTPUT_ROOT={output_root} run_science_run.sh"}
    (output_root / f"plan_runs_{design}.json").write_text(json.dumps(plan, indent=1))
    return plan


def plan_maps_stats(
    *,
    output_root: Path | str,
    design: str,
    n_runs: int,
    shared_stage1_root: Path | str,
    source_redshifts=(0.5, 1.0, 1.5, 2.0),
    n_real: int = 8,
    fov_deg: float = 5.0,
    npix: int = 1024,
    r200_factor: float = 4.0,
    with_dmo: bool = True,
    halo_scaling: bool = True,
    python_maps: str = "python -u -m bind.cli.lightcone_maps",
    python_stats: str = "python -u -m bind.cli.lightcone_stats",
) -> dict:
    """Stage 2(maps)+3 disBatch task file: one task per run (assemble maps -> stats).

    Each task assembles ``kappa_maps.npz``/``y_maps.npz`` for a run (reading its
    composites from ``runs/<design>/run_i`` and the shared manifests from
    ``shared_stage1_root``) then runs the stats.  Restart-safe: runs whose
    ``Cl_kappa.npz`` already exists are skipped.
    """
    output_root = Path(output_root)
    zs = " ".join(str(z) for z in source_redshifts)
    lines: list[str] = []
    n_done = 0
    for i in range(n_runs):
        rd = run_dir(output_root, design, i)
        if (rd / "Cl_kappa.npz").exists():
            n_done += 1
            continue
        maps = (
            f"{python_maps} --snap_root {rd} --manifest_root {shared_stage1_root} "
            f"--output_dir {rd} --source_redshifts {zs} --n_real {n_real} "
            f"--fov_deg {fov_deg} --npix {npix} --r200_factor {r200_factor}"
            + (" --with_dmo" if with_dmo else "")
        )
        stats = f"{python_stats} --run_dir {rd}" + ("" if halo_scaling else " --no_halo_scaling")
        lines.append(f"{maps} && {stats}")

    task_file = output_root / f"maps_stats_tasks_{design}.db"
    header = (
        f"# disBatch task file — BIND maps+stats ({design})\n"
        f"# {len(lines)} tasks (1 per run; {n_done} already done & skipped)\n"
        f"# Run AFTER generate_tasks_{design}.db completes.\n"
    )
    task_file.write_text(header + "\n".join(lines) + ("\n" if lines else ""))
    plan = {
        "design": design, "n_runs": int(n_runs), "n_tasks": len(lines),
        "n_already_done": int(n_done), "source_redshifts": list(source_redshifts),
        "n_real": int(n_real), "fov_deg": float(fov_deg), "npix": int(npix),
        "with_dmo": bool(with_dmo), "task_file": str(task_file),
    }
    (output_root / f"plan_maps_stats_{design}.json").write_text(json.dumps(plan, indent=1))
    return plan
