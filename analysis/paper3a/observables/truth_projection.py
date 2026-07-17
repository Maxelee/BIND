"""TNG300-hydro truth projection in the stage-1 frame (WP-A2 Popeye half, task 5).

Builds the *truth* side of the painted-vs-truth validation: projects the
TNG300-hydro gas particles into the **same transformed z-slab frame** the BIND
conditions were built in (``frame_transforms``), so a halo's truth cutout and
its painted cutout are spatially registered and integrate the same LOS depth.

Per-particle thermodynamics (temperature, pressure, entropy, compton_y) are
imported *directly* from ``bind.inference.pipeline`` — the exact functions and
constants that built the training-target thermo maps — so any painted-vs-truth
residual is a real model error, not a re-derivation mismatch (constants.py in
this package documents the same rule). Two extra channels the training maps do
not carry are added for WP-A2:

* ``gas_sigma`` — gas surface mass density [Msun/h per pixel]; the kSZ operator
  applies the ``x_e = 1.158`` fully-ionized assumption to it (``tau_map_from_gas``),
  exactly as it does to a painted gas map.
* ``gas_sigma_xe`` — x_e-weighted gas surface density Σ(x_e·M_gas) [Msun/h per
  pixel] using the *actual* per-particle ``ElectronAbundance``. The true kSZ
  optical depth is ``tau_map_from_gas(gas_sigma_xe, x_e=1.0)``; comparing it to
  the assumption ``tau_map_from_gas(gas_sigma, x_e=1.158)`` isolates the
  ionization-assumption error (plan step 4, "x_e residual"). Stored as a mass
  (not a raw electron count, which overflows float32).

Unlike ``pipeline.project_thermo_fullbox`` (axis-aligned, whole box, no slabs)
this applies the frame transform, decomposes into z-slabs, and streams over the
snapshot's ~600 files so peak memory is one chunk + the slab accumulators
(~7 GB for 4 slabs × 6 channels at 4198²), not the whole box in RAM.

This module imports the bind/h5py/Pylians chain, so — unlike the rest of the
observables package — it is NOT pure numpy and is deliberately not re-exported
from ``observables/__init__.py``; import it explicitly on Popeye.
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

# 'spawn', not the Linux-default 'fork': the workers use native libraries
# (Pylians/MAS_library, HDF5, and the torch that bind.inference.pipeline pulls
# in at import) that are not fork-safe — forking mid-state segfaults the worker
# ("BrokenProcessPool"). spawn re-imports cleanly in each worker.
_MP = mp.get_context("spawn")


def _default_workers(n_workers: int | None) -> int:
    """Resolve the worker count: explicit > SLURM_CPUS_PER_TASK > cpu_count."""
    if n_workers is not None:
        return max(1, int(n_workers))
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    return max(1, int(slurm) if slurm else (os.cpu_count() or 1))


def mpi_comm():
    """COMM_WORLD when running as a multi-rank MPI job, else None.

    Detection is env-based (mpirun sets ``OMPI_COMM_WORLD_SIZE``; srun sets
    ``PMI_SIZE`` / ``SLURM_STEP_NUM_TASKS``) so single-process runs never
    import mpi4py and the ProcessPool path stays the default off-MPI. Set
    ``BIND_NO_MPI=1`` to force the single-process path even under mpirun.

    A multi-task launch whose MPI world failed to wire up (each rank a size-1
    singleton) raises rather than letting N ranks each run the full job and
    race on the outputs — launch with ``mpirun`` (or ``srun --mpi=pmix``).
    """
    if os.environ.get("BIND_NO_MPI"):
        return None
    n_env = (os.environ.get("OMPI_COMM_WORLD_SIZE")
             or os.environ.get("PMI_SIZE")
             or os.environ.get("SLURM_STEP_NUM_TASKS"))
    if n_env is None or int(n_env) < 2:
        return None
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_size() < int(n_env):
        raise RuntimeError(
            f"launched with {n_env} tasks but MPI world size is {comm.Get_size()} — "
            "MPI did not wire up; launch with mpirun (or srun --mpi=pmix)")
    return comm

from bind.inference.pipeline import (
    GAMMA,
    K_B_J_PER_K,
    KEV_IN_J,
    KPC_IN_M,
    MPC_IN_M,
    MSUN_KG,
    M_PROTON_KG,
    SIGMA_T_M2,
    X_H,
    _safe_divide,
    compton_y_integrand_per_particle,
    extract_periodic_cutout,
    gas_temperature_K,
    pixelize_z_projection,
)

from .frame_transforms import Stage1Manifest, apply_frame_transform, assign_slabs

# Channels carried through the accumulator. Mass-weighted means (temperature,
# pressure, entropy) store a *_m = sum(field*mass) numerator and are divided by
# gas_sigma at finalize; compton_y and ne_true_column are direct sums.
_WEIGHTED = ("temperature_m", "pressure_m", "entropy_m")
_DIRECT = ("gas_sigma", "compton_y", "gas_sigma_xe")


@dataclass
class TruthMapAccumulator:
    """Per-slab running sums for one snapshot's transformed truth projection."""

    n_slabs: int
    npix: int
    box_size: float
    slab_depth: float
    pixel_area_m2: float  # comoving pixel area [m^2] (matches project_thermo_fullbox)
    maps: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.maps:
            shape = (self.n_slabs, self.npix, self.npix)
            for key in (*_WEIGHTED, *_DIRECT):
                self.maps[key] = np.zeros(shape, dtype=np.float32)


def _hydro_unit_factors(h: float) -> tuple[float, float]:
    """(density code→kg/m^3, mass 1e10 Msun/h→kg) — verbatim project_thermo_fullbox."""
    density_to_kg_m3 = 1e10 * MSUN_KG * h ** 2 / KPC_IN_M ** 3
    mass_code_to_kg = 1e10 * MSUN_KG / h
    return density_to_kg_m3, mass_code_to_kg


def accumulate_gas_chunk(
    acc: TruthMapAccumulator,
    pos_mpch: np.ndarray,      # (N,3) Mpc/h, original frame
    mass_code: np.ndarray,     # (N,) 1e10 Msun/h
    density_code: np.ndarray,  # (N,) code density
    u_code: np.ndarray,        # (N,) InternalEnergy (km/s)^2
    xe: np.ndarray,            # (N,) ElectronAbundance
    sfr: np.ndarray,           # (N,) StarFormationRate
    manifest: Stage1Manifest,
    h: float,
) -> None:
    """Project one gas chunk into the transformed z-slab accumulators.

    Applies the SFR<=0 hot-gas cut, computes per-particle thermo (identical to
    ``project_thermo_fullbox``), transforms positions into the stage-1 frame,
    assigns z-slabs, and CIC-projects each channel per slab (summed in place —
    CIC into a fresh grid then summing per chunk is exact, since each particle's
    cloud stays within its slab's transverse grid).
    """
    hot = np.asarray(sfr) <= 0.0
    if not hot.any():
        return
    pos = np.asarray(pos_mpch)[hot]
    mass = np.asarray(mass_code)[hot].astype(np.float64)
    dens = np.asarray(density_code)[hot]
    u = np.asarray(u_code)[hot]
    xe_h = np.asarray(xe)[hot]

    density_to_kg_m3, mass_code_to_kg = _hydro_unit_factors(h)

    T = gas_temperature_K(u, xe_h)                                     # [K]
    rho_phys = dens.astype(np.float64) * density_to_kg_m3             # [kg/m^3]
    u_phys = u.astype(np.float64) * 1e6                              # [(m/s)^2]
    P = ((GAMMA - 1.0) * rho_phys * u_phys)                          # [Pa]
    n_e = (xe_h.astype(np.float64) * X_H * rho_phys / M_PROTON_KG * 1e-6)  # [cm^-3]
    n_e_safe = np.where(n_e > 0, n_e, 1.0)
    K = (K_B_J_PER_K * T / KEV_IN_J) / n_e_safe ** (2.0 / 3.0)        # [keV cm^2]
    K = np.where(n_e > 0, K, 0.0)
    y_int = compton_y_integrand_per_particle(u, xe_h, mass, mass_code_to_kg)  # [m^2]
    # gas mass per particle in Msun/h (map convention: painted maps are Msun/h)
    mass_msunh = mass * 1e10

    tpos = apply_frame_transform(pos, manifest.proj_dir, manifest.disp, manifest.flip, acc.box_size)
    slab = assign_slabs(tpos[:, 2], acc.box_size, acc.n_slabs)

    weights = {
        "gas_sigma": mass_msunh.astype(np.float32),
        "temperature_m": (T * mass_msunh).astype(np.float32),
        "pressure_m": (P * mass_msunh).astype(np.float32),
        "entropy_m": (K * mass_msunh).astype(np.float32),
        "compton_y": (y_int / acc.pixel_area_m2).astype(np.float32),
        "gas_sigma_xe": (xe_h.astype(np.float64) * mass_msunh).astype(np.float32),
    }
    for si in range(acc.n_slabs):
        m = slab == si
        if not m.any():
            continue
        p_si = np.ascontiguousarray(tpos[m])
        for key, w in weights.items():
            acc.maps[key][si] += pixelize_z_projection(p_si, w[m], acc.box_size, acc.npix)


def finalize_truth_maps(acc: TruthMapAccumulator) -> dict[str, np.ndarray]:
    """Divide mass-weighted numerators by gas_sigma → physical per-slab maps.

    Returns dict of (n_slabs, npix, npix) float32 maps: ``gas_sigma`` [Msun/h/pix],
    ``temperature`` [K], ``pressure`` [Pa], ``entropy`` [keV cm^2], ``compton_y``
    [dimensionless y/pix], ``gas_sigma_xe`` [Msun/h/pix, x_e-weighted].
    """
    m_map = acc.maps["gas_sigma"]
    out = {
        "gas_sigma": m_map.copy(),
        "compton_y": acc.maps["compton_y"].copy(),
        "gas_sigma_xe": acc.maps["gas_sigma_xe"].copy(),
        "temperature": _safe_divide(acc.maps["temperature_m"], m_map).astype(np.float32),
        "pressure": _safe_divide(acc.maps["pressure_m"], m_map).astype(np.float32),
        "entropy": _safe_divide(acc.maps["entropy_m"], m_map).astype(np.float32),
    }
    return out


def _pixel_area_m2(manifest: Stage1Manifest, h: float) -> float:
    """PROPER pixel area [m^2] — the physical Compton-y normalization.

    y is dimensionless: sum(per-particle y_int [m^2]) / (PROPER transverse
    pixel area). The z~0 builder (`add_gas_thermo_maps.py` /
    `pipeline.project_thermo_fullbox`) uses the comoving side, correct only
    because a(z~0)=1; the multi-z training-map builder
    (`data_generation/process_simulations_multiz.py`, feature/redshift — the
    fm_redshift_thermo targets) uses the proper side explicitly
    (``pixel_side * a``). Using the comoving area at z>0 under-normalizes
    truth y by a^2 — the exact (1+z)^2 painted/truth excess the 2026-07-16
    validation measured (A/wp2 REPORT ## BLOCKED; audit 2026-07-17). Only the
    ``compton_y`` channel consumes this area.
    """
    return ((manifest.box_size / manifest.npix) / h * MPC_IN_M * manifest.scale_factor) ** 2


def _snapshot_files(hydro_snapdir: str, snapshot: int) -> list[str]:
    d = Path(hydro_snapdir)
    files = sorted(glob.glob(str(d / f"snap_{snapshot:03d}.*.hdf5")))
    if not files and (d / f"snap_{snapshot:03d}.hdf5").exists():
        files = [str(d / f"snap_{snapshot:03d}.hdf5")]
    if not files:
        raise FileNotFoundError(f"No hydro snapshot files for snap {snapshot} in {hydro_snapdir}")
    return files


def project_truth_maps(
    manifest: Stage1Manifest,
    hydro_snapdir: str | None = None,
    snapshot: int | None = None,
    max_files: int | None = None,
    progress: bool = True,
    n_workers: int | None = None,
) -> dict[str, np.ndarray]:
    """Full transformed-frame truth projection for one snapshot (streamed over files).

    ``hydro_snapdir``/``snapshot`` default to the manifest's hydro twin and
    ``snapshot_index``. ``max_files`` truncates the file loop for smoke tests.
    ``n_workers`` sets the process-pool size (default SLURM_CPUS_PER_TASK or
    os.cpu_count(); 1 = serial) — the ~600-file loop is a map-reduce, so this
    scales near-linearly with cores. Returns the ``finalize_truth_maps`` dict.

    Under a multi-rank MPI launch (``mpi_comm()``) this is COLLECTIVE: every
    rank must call it; ranks stride the file list and the raw maps are
    MPI-Reduced to rank 0, which returns the finalized dict (other ranks
    return None). No ProcessPool is used in that mode — one rank per core.
    """
    hydro_snapdir = hydro_snapdir or manifest.hydro_snapdir()
    snapshot = manifest.snapshot_index if snapshot is None else snapshot
    files = _snapshot_files(hydro_snapdir, snapshot)
    if max_files is not None:
        files = files[:max_files]

    h = _read_hubble(files[0])
    pixel_area_m2 = _pixel_area_m2(manifest, h)

    comm = mpi_comm()
    if comm is not None:
        return _project_truth_maps_mpi(comm, files, manifest, snapshot, pixel_area_m2, h, progress)

    W = _default_workers(n_workers)

    def _new_acc():
        return TruthMapAccumulator(
            n_slabs=manifest.n_slabs, npix=manifest.npix, box_size=manifest.box_size,
            slab_depth=manifest.slab_depth, pixel_area_m2=pixel_area_m2,
        )

    if W <= 1 or len(files) <= 1:
        acc = _new_acc()
        _accumulate_files(acc, files, manifest, h, progress, f"truth snap_{snapshot:03d}")
        return finalize_truth_maps(acc)

    # Parallel map-reduce: split files into W chunks, each worker projects its
    # chunk into a local accumulator (raw *_m numerators + direct sums, all
    # additive), then the parent sums the chunk maps and finalizes ONCE. Peak
    # memory ~ W * (1.7 GB accumulator + ~1 GB transient/file) — size W to the
    # node (default SLURM_CPUS_PER_TASK; ~32 fits a 192 GB node).
    chunks = [list(c) for c in np.array_split(np.array(files, dtype=object), W) if len(c)]
    combined: dict[str, np.ndarray] | None = None
    done = 0
    with ProcessPoolExecutor(max_workers=W, mp_context=_MP) as ex:
        futs = {ex.submit(_project_chunk_worker, ch, manifest, manifest.npix, manifest.box_size,
                          manifest.n_slabs, manifest.slab_depth, pixel_area_m2, h): i
                for i, ch in enumerate(chunks)}
        for fut in as_completed(futs):
            m = fut.result()
            if combined is None:
                combined = m
            else:
                for k in combined:
                    combined[k] += m[k]
                del m
            done += 1
            if progress:
                print(f"[truth snap_{snapshot:03d}] reduced {done}/{len(chunks)} chunks", flush=True)
    acc = _new_acc()
    acc.maps = combined
    return finalize_truth_maps(acc)


def _project_truth_maps_mpi(comm, files, manifest, snapshot, pixel_area_m2, h, progress):
    """MPI map-reduce: ranks stride the file list, raw maps Reduce-sum to rank 0.

    Collective over ``comm`` — called by ``project_truth_maps`` on every rank
    of an mpirun launch. Each rank accumulates its files serially (parallelism
    is one rank per core, no nested pool), so per-rank memory is one ~1.7 GB
    accumulator + one file transient — the same footprint as a pool worker.
    Returns the finalized maps dict on rank 0, None on the other ranks.
    """
    from mpi4py import MPI
    rank, size = comm.Get_rank(), comm.Get_size()
    acc = TruthMapAccumulator(
        n_slabs=manifest.n_slabs, npix=manifest.npix, box_size=manifest.box_size,
        slab_depth=manifest.slab_depth, pixel_area_m2=pixel_area_m2,
    )
    my_files = files[rank::size]
    if progress and rank == 0:
        print(f"[truth snap_{snapshot:03d}] MPI: {size} ranks x ~{len(my_files)} files each", flush=True)
    _accumulate_files(acc, my_files, manifest, h,
                      progress=progress and rank == 0, desc=f"truth snap_{snapshot:03d} rank0")
    # Fixed channel order — collectives must be issued identically on all ranks.
    for key in (*_WEIGHTED, *_DIRECT):
        recv = np.empty_like(acc.maps[key]) if rank == 0 else None
        comm.Reduce(acc.maps[key], recv, op=MPI.SUM, root=0)
        acc.maps[key] = recv                      # rank 0: combined; workers: free
    if rank != 0:
        return None
    if progress:
        print(f"[truth snap_{snapshot:03d}] MPI reduce done", flush=True)
    return finalize_truth_maps(acc)


def _accumulate_files(acc, files, manifest, h, progress=False, desc=""):
    """Serial: read each file's PartType0 gas fields into the accumulator."""
    it = files
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(files, desc=desc)
        except Exception:
            pass
    for fname in it:
        with h5py.File(fname, "r") as f:
            if "PartType0" not in f:
                continue
            g = f["PartType0"]
            accumulate_gas_chunk(
                acc,
                pos_mpch=g["Coordinates"][:].astype(np.float32) / 1000.0,
                mass_code=g["Masses"][:], density_code=g["Density"][:],
                u_code=g["InternalEnergy"][:], xe=g["ElectronAbundance"][:],
                sfr=g["StarFormationRate"][:], manifest=manifest, h=h,
            )


def _project_chunk_worker(files, manifest, npix, box_size, n_slabs, slab_depth, pixel_area_m2, h):
    """ProcessPool worker: accumulate a file chunk -> raw (pre-finalize) maps dict."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # 1 thread/proc — parallelism is process-level
    acc = TruthMapAccumulator(n_slabs=n_slabs, npix=npix, box_size=box_size,
                              slab_depth=slab_depth, pixel_area_m2=pixel_area_m2)
    _accumulate_files(acc, files, manifest, h, progress=False)
    return acc.maps


def _read_hubble(fname: str) -> float:
    with h5py.File(fname, "r") as f:
        return float(f["Header"].attrs["HubbleParam"])


# ---------------------------------------------------------------------------
# Hydro halo catalog (M500c/R500c for f_gas/Y apertures; the matched-halo join)
# ---------------------------------------------------------------------------

@dataclass
class HydroHaloCatalog:
    """Transformed-frame TNG300-hydro FoF halos above the mass floor.

    ``centers_xy`` (transverse, Mpc/h) and ``los`` (Mpc/h) are in the stage-1
    frame — directly comparable to the condition ``halo_centers``. Masses in
    Msun/h, radii in Mpc/h. ``centers_orig`` is the SAME halos' positions in
    the ORIGINAL (untransformed) frame — the frame the snapshot particle
    coordinates are in; ``spherical_gas_mass_from_particles`` must be fed
    these, never the transformed ones.
    """

    centers_xy: np.ndarray    # (M, 2) stage-1 frame
    los: np.ndarray           # (M,)   stage-1 frame
    centers_orig: np.ndarray  # (M, 3) original frame, Mpc/h
    m200c: np.ndarray
    m500c: np.ndarray
    r200c: np.ndarray
    r500c: np.ndarray
    slab: np.ndarray          # (M,) slab index


def load_hydro_halo_catalog(
    manifest: Stage1Manifest,
    group_catalog: str | None = None,
    snapshot: int | None = None,
    mass_min_msunh: float = 1e13,
) -> HydroHaloCatalog:
    """Load hydro FoF groups, apply the transform + M200c mass floor.

    Reads ``Group_M_Crit200/500``, ``Group_R_Crit200/500``, ``GroupPos`` from
    the ``fof_subhalo_tab`` files (same glob as ``pipeline.load_halo_catalog``),
    converts units, applies the stage-1 frame transform to ``GroupPos``, and
    keeps M200c >= ``mass_min_msunh`` (the 1e13 training floor, SHARED_CONTEXT
    caveat 1).
    """
    group_catalog = group_catalog or manifest.hydro_group_catalog()
    snapshot = manifest.snapshot_index if snapshot is None else snapshot
    d = Path(group_catalog)
    files = sorted(glob.glob(str(d / f"fof_subhalo_tab_{snapshot:03d}.*.hdf5")))
    if not files and (d / f"fof_subhalo_tab_{snapshot:03d}.hdf5").exists():
        files = [str(d / f"fof_subhalo_tab_{snapshot:03d}.hdf5")]
    if not files:
        raise FileNotFoundError(f"No FoF group files in {group_catalog}")

    m200_l, m500_l, r200_l, r500_l, pos_l = [], [], [], [], []
    for fname in files:
        with h5py.File(fname, "r") as handle:
            if "Group/Group_M_Crit200" not in handle:
                continue
            grp = handle["Group"]
            n = len(grp["Group_M_Crit200"])
            m200_l.append(grp["Group_M_Crit200"][:])
            m500_l.append(grp["Group_M_Crit500"][:] if "Group_M_Crit500" in grp else np.zeros(n, np.float32))
            r200_l.append(grp["Group_R_Crit200"][:] if "Group_R_Crit200" in grp else np.zeros(n, np.float32))
            r500_l.append(grp["Group_R_Crit500"][:] if "Group_R_Crit500" in grp else np.zeros(n, np.float32))
            pos_l.append(grp["GroupPos"][:])
    if not m200_l:
        raise RuntimeError(f"No Group_M_Crit200 datasets in {group_catalog}")

    m200 = np.concatenate(m200_l) * 1e10
    m500 = np.concatenate(m500_l) * 1e10
    r200 = np.concatenate(r200_l) / 1e3   # kpc/h -> Mpc/h
    r500 = np.concatenate(r500_l) / 1e3
    pos = np.concatenate(pos_l, axis=0) / 1e3  # kpc/h -> Mpc/h

    keep = m200 >= mass_min_msunh
    tpos = apply_frame_transform(pos[keep], manifest.proj_dir, manifest.disp, manifest.flip, manifest.box_size)
    slab = assign_slabs(tpos[:, 2], manifest.box_size, manifest.n_slabs)
    return HydroHaloCatalog(
        centers_xy=tpos[:, :2].astype(np.float64),
        los=tpos[:, 2].astype(np.float64),
        centers_orig=(pos[keep].astype(np.float64) % manifest.box_size),
        m200c=m200[keep],
        m500c=m500[keep],
        r200c=r200[keep],
        r500c=r500[keep],
        slab=slab,
    )


def match_condition_halos(
    cond_centers_xy: np.ndarray,   # (K, 2) Mpc/h, transverse, from stage1 slab npz
    cond_m200: np.ndarray,         # (K,) Msun/h
    hydro: HydroHaloCatalog,
    box_size: float,
    max_sep_mpch: float = 0.5,
    mass_ratio_tol: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Match DMO condition halos to hydro FoF halos by transverse position.

    DMO and hydro share initial conditions, so a condition halo's transverse
    position lands within a fraction of R200 of its hydro counterpart. Matches
    to the nearest hydro halo within ``max_sep_mpch`` (periodic) whose M200c is
    within a factor ``mass_ratio_tol`` (guards against grabbing a nearby
    different-mass halo — the ordering caveat the plan flags for cross-catalog
    joins). Returns ``(hydro_idx, matched_mask)`` aligned to the condition halos;
    ``hydro_idx[i] = -1`` where no match survives.

    O(K·M) brute force; K,M ~ few×10^3 per slab, so ~10^7 ops — fine, and no
    scipy dependency (keeps this callable in the minimal venv).
    """
    cond = np.asarray(cond_centers_xy, dtype=np.float64)
    hxy = hydro.centers_xy
    out = np.full(len(cond), -1, dtype=np.int64)
    matched = np.zeros(len(cond), dtype=bool)
    half = box_size / 2.0
    for i in range(len(cond)):
        dx = np.abs(hxy[:, 0] - cond[i, 0]); dx = np.minimum(dx, box_size - dx)
        dy = np.abs(hxy[:, 1] - cond[i, 1]); dy = np.minimum(dy, box_size - dy)
        d2 = dx * dx + dy * dy
        j = int(np.argmin(d2))
        if d2[j] > max_sep_mpch ** 2:
            continue
        ratio = hydro.m200c[j] / max(cond_m200[i], 1.0)
        if ratio > mass_ratio_tol or ratio < 1.0 / mass_ratio_tol:
            continue
        out[i] = j
        matched[i] = True
    return out, matched


def spherical_gas_mass_from_particles(
    hydro_snapdir: str,
    snapshot: int,
    centers_xyz_mpch: np.ndarray,   # (M, 3) ORIGINAL-frame halo positions, Mpc/h
    r500c_mpch: np.ndarray,         # (M,) Mpc/h
    box_size: float,
    max_files: int | None = None,
    progress: bool = True,
    n_workers: int | None = None,
) -> np.ndarray:
    """3D spherical gas mass M_gas(<R500c) [Msun/h] per halo, from particles.

    The numerator of the CylToSph correction (plan step 4): the *spherical*
    aperture gas mass that the projected cylinder over-counts. Streamed over the
    snapshot files; per file a periodic ``cKDTree`` ball query gathers the gas
    particles inside each halo's R500c and their masses are summed (hot +
    star-forming both count — a spherical gas mass is the total gas, no SFR cut).
    Frame-independent, so it uses the *original* (untransformed) halo positions
    and particle coordinates.

    Parallel over files (each returns a per-halo mass vector, summed) —
    ``n_workers`` defaults to SLURM_CPUS_PER_TASK / cpu_count. Intended for a
    bounded halo subset (a few hundred); a KDTree per file over ~2.6e7 particles
    is the cost driver.

    Under a multi-rank MPI launch (``mpi_comm()``) this is COLLECTIVE: every
    rank must call it with the SAME centers/r500 (broadcast them first); ranks
    stride the files and the per-halo vector is Reduce-summed to rank 0, which
    returns it (other ranks return None).
    """
    centers = np.ascontiguousarray(np.asarray(centers_xyz_mpch, dtype=np.float64) % box_size)
    r500 = np.asarray(r500c_mpch, dtype=np.float64)
    files = _snapshot_files(hydro_snapdir, snapshot)
    if max_files is not None:
        files = files[:max_files]
    W = _default_workers(n_workers)
    m_gas = np.zeros(len(centers), dtype=np.float64)

    comm = mpi_comm()
    if comm is not None:
        from mpi4py import MPI
        rank, size = comm.Get_rank(), comm.Get_size()
        for fname in files[rank::size]:
            m_gas += _sph_mass_one_file(fname, centers, r500, box_size)
        recv = np.empty_like(m_gas) if rank == 0 else None
        comm.Reduce(m_gas, recv, op=MPI.SUM, root=0)
        if rank == 0 and progress:
            print(f"[sph M_gas snap_{snapshot:03d}] MPI reduce over {len(files)} files done", flush=True)
        return recv if rank == 0 else None

    if W <= 1 or len(files) <= 1:
        for fname in files:
            m_gas += _sph_mass_one_file(fname, centers, r500, box_size)
        return m_gas

    with ProcessPoolExecutor(max_workers=W, mp_context=_MP) as ex:
        futs = [ex.submit(_sph_mass_one_file, fn, centers, r500, box_size) for fn in files]
        done = 0
        for fut in as_completed(futs):
            m_gas += fut.result()
            done += 1
            if progress and done % 50 == 0:
                print(f"[sph M_gas snap_{snapshot:03d}] {done}/{len(files)} files", flush=True)
    return m_gas


def _sph_mass_one_file(fname, centers, r500, box_size):
    """Per-halo gas mass [Msun/h] inside R500c from ONE snapshot file (periodic)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from scipy.spatial import cKDTree
    with h5py.File(fname, "r") as f:
        if "PartType0" not in f:
            return np.zeros(len(centers), dtype=np.float64)
        g = f["PartType0"]
        ppos = (g["Coordinates"][:].astype(np.float64) / 1000.0) % box_size
        pmass = g["Masses"][:].astype(np.float64) * 1e10  # Msun/h
    tree = cKDTree(ppos, boxsize=box_size)
    out = np.zeros(len(centers), dtype=np.float64)
    for i, idx in enumerate(tree.query_ball_point(centers, r500)):
        if idx:
            out[i] = pmass[idx].sum()
    return out


def cut_truth_cutout(
    slab_map: np.ndarray,       # (npix, npix) one channel, one slab
    center_xy_mpch: tuple[float, float],
    box_size: float,
    npix: int,
    cutout_pix: int,
) -> np.ndarray:
    """Periodic cutout of a truth slab map at a halo's transverse center.

    Uses the same ``cx = int(x * npix/box) % npix`` convention as
    ``paint.extract_halo_cutouts`` / ``pipeline.extract_halo_thermo_cutouts`` so
    truth and painted cutouts share the halo-center pixel.
    """
    ppm = npix / box_size
    cx = int(center_xy_mpch[0] * ppm) % npix
    cy = int(center_xy_mpch[1] * ppm) % npix
    return extract_periodic_cutout(slab_map, cx, cy, cutout_pix)
