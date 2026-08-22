"""Two-stage paint: CPU/MPI projection+cutouts, then GPU generation.

The one-shot :func:`bind.paint` loads *all* DMO particles on a single process
(``io_gadget.read_dmo_particles`` concatenates every snapshot chunk) and then
masks them per z-slab.  For a large box like TNG300-1-Dark (~2.7e9 particles,
75 chunks, ~33 GB of positions) the concatenate peak alone OOMs a single node.

This module splits the work so neither stage needs the whole particle set in
memory at once:

* **Stage 1 — :func:`project_and_extract` (CPU, MPI-parallel).**  Each rank
  reads only *its* subset of snapshot chunks and accumulates them into the
  small z-slab maps (a 205 Mpc/h box at native resolution is only ~70 MB per
  slab), then the partial maps are ``MPI.SUM``-reduced onto rank 0.  Particle
  positions are streamed one chunk at a time and freed immediately, so peak
  memory is ~(one chunk + the slab maps) regardless of box size.  Rank 0 then
  reads the FoF catalog, extracts per-halo DMO cutouts, and writes a
  ``stage1_slab{NN}.npz`` per slab plus a ``stage1_manifest.json``.

  Run serially (1 rank) this still fixes the OOM — it just doesn't get the
  multi-node speedup.

* **Stage 2 — :func:`generate_from_stage1` (GPU).**  Loads the intermediate,
  runs the flow-matching sampler on the saved cutouts, composites the painted
  patches back into per-slab maps, and writes the same ``composite_slab{NN}.npz``
  / ``summary.json`` artifacts as :func:`bind.paint`.

The two stages communicate only through the on-disk intermediate, so the heavy
particle I/O (CPU, many nodes) and the sampling (one GPU) run as independent
SLURM jobs — see ``run_paint_tng_project.sh`` and ``run_paint_tng_generate.sh``.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from bind.data import N_THERMO, THERMO_KEYS

from . import io_gadget
from .artifacts import (
    PROVENANCE_KEY,
    bind_version,
    build_provenance,
    provenance_array,
    read_provenance,
    to_jsonable,
)
from .lightcone_transforms import LightconeTransforms
from .paint import (
    NATIVE_PIXEL_SIZE_MPCH,
    NATIVE_SLAB_DEPTH_MPCH,
    PATCH_PIX,
    Model,
    PaintResult,
    _assign_halos_to_slabs,
    _project_zslabs,
    _round_n_slabs,
    _round_npix,
    _save_empty_slab,
    _validate_params,
    extract_halo_cutouts,
)
from .pipeline import build_bind_composite, derive_seed, interlace_combine_2d

MANIFEST_NAME = "stage1_manifest.json"


#: Relative tolerance for "the conditioning cosmology matches the substrate".
COSMOLOGY_RTOL = 1e-3


#: Parameter index -> snapshot-header attribute, for every cosmology entry the
#: header can actually adjudicate.  sigma8 (idx 1) and n_s (idx 8) are properties
#: of the initial conditions and are NOT in the snapshot header, so they cannot be
#: checked here.
_HEADER_COSMO = ((0, "Omega0"), (6, "OmegaBaryon"), (7, "HubbleParam"))


def _check_cosmology(params: np.ndarray, header: dict | float, *, strict: bool,
                     where: str) -> None:
    """Guard: the conditioned cosmology must match the snapshot's own.

    Painting a box with a parameter vector whose cosmology block belongs to a
    *different* simulation suite silently mis-conditions the generative model —
    e.g. conditioning a TNG300 substrate on the CAMELS fiducial makes
    Omega_b/Omega_m 3.8% too high and inflates the painted gas/tau power by
    ~7.7%.  The header values come from the snapshot HDF5, so they are ground
    truth for the substrate; ``params`` is what the model will see.

    **All three checkable entries are compared, not just Omega_m.**  The damage in
    the released lightcone was driven by Omega_b/Omega_m, not by Omega_m: a vector
    carrying the right Omega_m=0.3089 with CAMELS' Omega_b=0.049 reproduces the
    exact same +7.7% bug, and an Omega_m-only guard would wave it through.  The
    error message therefore reports Omega_b/Omega_m explicitly.

    ``header`` is a mapping of header attributes (missing or non-positive entries
    are skipped); a bare float is accepted as ``{"Omega0": value}`` for backwards
    compatibility.

    Stage 1 raises (the substrate's cosmology is a fact).  Painting from an
    existing stage 1 only warns, since parameter sweeps legitimately paint
    off-substrate cosmologies.
    """
    if not isinstance(header, dict):
        header = {"Omega0": float(header)}

    bad: list[str] = []
    for idx, key in _HEADER_COSMO:
        ref = header.get(key, float("nan"))
        try:
            ref = float(ref)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ref) or ref <= 0:
            continue          # header did not carry it — nothing to check against
        got = float(params[idx])
        if abs(got - ref) / ref >= COSMOLOGY_RTOL:
            bad.append(f"{key}: params[{idx}]={got:.5f} vs header {ref:.5f}")
    if not bad:
        return

    om_p, ob_p = float(params[0]), float(params[6])
    fb_p = ob_p / om_p if om_p else float("nan")
    om_h = float(header.get("Omega0", float("nan")))
    ob_h = float(header.get("OmegaBaryon", float("nan")))
    fb_h = ob_h / om_h if om_h else float("nan")
    msg = (f"[{where}] COSMOLOGY MISMATCH ({len(bad)} entr"
           f"{'y' if len(bad) == 1 else 'ies'}): " + "; ".join(bad)
           + f". Omega_b/Omega_m would be {fb_p:.6f} against the substrate's "
             f"{fb_h:.6f} (ratio {fb_p / fb_h:.6f} — the painted gas/tau amplitude "
             f"tracks this ~linearly and the plane power as its square). "
             f"The model would be conditioned on a cosmology the substrate does not "
             f"have (see bind.tng300_params for IllustrisTNG substrates; "
             f"bind.fiducial_params is the CAMELS one).")
    if strict:
        raise ValueError(
            msg + " Pass allow_cosmology_mismatch=True "
                  "(--allow_cosmology_mismatch) if this is deliberate.")
    warnings.warn(msg, stacklevel=2)


def _stage1_slab_path(stage1_dir: Path, si: int) -> Path:
    return Path(stage1_dir) / f"stage1_slab{si:02d}.npz"


# ---------------------------------------------------------------------------
# Stage 1: projection + halo cutouts (CPU, MPI-parallel)
# ---------------------------------------------------------------------------

def _accumulate_chunk_into_slabs(
    local_slabs: np.ndarray,
    positions_mpch: np.ndarray,
    particle_mass: float,
    box_size: float,
    n_slabs: int,
) -> None:
    """CIC-project one chunk's particles into the per-slab maps, summing in place.

    Each chunk is projected into a *fresh* set of slab maps via
    ``_project_zslabs`` (the same primitive the one-shot ``paint`` uses) and the
    result is numpy-added onto ``local_slabs``.  CIC mass assignment into a zero
    field is linear in the particles, and each particle's mass lands in exactly
    one slab, so summing per-chunk projections is identical to projecting the
    whole box at once — but never holds more than one chunk in memory.

    NB: Pylians ``MASL.MA`` does *not* accumulate additively onto a pre-filled
    field (it blends), so we must deposit into zeros and add ourselves rather
    than call ``MA`` repeatedly on the same buffer.
    """
    npix = local_slabs.shape[1]
    masses = np.full(len(positions_mpch), np.float32(particle_mass), dtype=np.float32)
    local_slabs += _project_zslabs(positions_mpch, masses, box_size, npix, n_slabs)


def project_and_extract(
    snapshot: Path | str,
    group_catalog: Path | str,
    *,
    params: np.ndarray,
    output_dir: Path | str,
    snapshot_index: int | None = None,
    halo_mass_min: float = 1e13,
    halo_mass_field: str = "Group_M_Crit200",
    pixel_size: float = NATIVE_PIXEL_SIZE_MPCH,
    slab_depth: float = NATIVE_SLAB_DEPTH_MPCH,
    patch_pix: int = PATCH_PIX,
    transforms: LightconeTransforms | None = None,
    transforms_snap_idx: int | None = None,
    mas_correct: bool = False,
    allow_cosmology_mismatch: bool = False,
    comm: Any | None = None,
    progress: bool = True,
) -> Path | None:
    """Stage 1: project DMO into z-slabs and extract per-halo cutouts.

    Parameters
    ----------
    snapshot, group_catalog
        Gadget/Arepo HDF5 snapshot + FoF catalog (file, directory, or glob).
    params
        35-dim parameter vector the model will condition on (saved for stage 2).
    output_dir
        Intermediate directory.  Rank 0 writes ``stage1_slab{NN}.npz`` per slab,
        ``params.npy``, and ``stage1_manifest.json``.
    transforms : LightconeTransforms, optional
        Lightcone geometric transforms (rotation/translation/flip).  When provided,
        each chunk's particle positions are transformed with
        ``transforms.apply(pos, transforms_snap_idx, box_size)`` **before**
        projection.  The transform parameters are stored in the manifest so the
        lensplane step can reproduce the geometry.
    transforms_snap_idx : int, optional
        Index into *transforms* for this snapshot (0 = lowest-z snapshot).
        Required when *transforms* is provided.
    mas_correct : bool
        If True, additionally accumulate a half-cell-shifted CIC deposit and
        store an **anti-aliased** DMO map (``dmo_aa``, interlaced + CIC-window
        deconvolved) alongside the raw ``dmo`` in each slab npz.  The raw ``dmo``
        is left untouched and remains the source for the model condition cutouts
        (the generative model was trained on raw-CIC inputs); ``dmo_aa`` is for
        the absolute lensing comparison against an anti-aliased reference
        (e.g. kappaTNG).  Default False (no behaviour change).
    allow_cosmology_mismatch : bool
        By default stage 1 REFUSES to write a params vector whose ``Omega0``,
        ``OmegaBaryon`` or ``HubbleParam`` disagrees with the snapshot header by
        more than :data:`COSMOLOGY_RTOL` (the check that would have caught the
        CAMELS-cosmology-on-TNG300 bug on 20/20 snapshots — note it was
        ``OmegaBaryon``/``Omega0``, not ``Omega0`` alone, that did the damage).
        ``sigma8`` and ``n_s`` are not in the snapshot header and cannot be
        checked.  Set True to paint a deliberately off-substrate cosmology.
    comm
        An ``mpi4py`` communicator, or ``None`` for a single-process run.  Each
        rank reads ``files[rank::size]``; partial slab maps are reduced to rank 0.

    Returns
    -------
    Path to ``output_dir`` on rank 0, ``None`` on other ranks.
    """
    params = _validate_params(params)
    rank = comm.rank if comm is not None else 0
    size = comm.size if comm is not None else 1

    snap_files = io_gadget._resolve_snap_files(snapshot, snapshot_index)
    if snapshot_index is None:
        snapshot_index = io_gadget._infer_snapshot_index(snap_files)

    if transforms is not None and transforms_snap_idx is None:
        raise ValueError("transforms_snap_idx is required when transforms is provided")

    # Box size, cosmology, scale factor, and DM particle mass from the first chunk header.
    with h5py.File(snap_files[0], "r") as h:
        box_size = float(h["Header"].attrs["BoxSize"]) / 1000.0
        particle_mass = float(h["Header"].attrs["MassTable"][1]) * 1e10
        scale_factor = float(h["Header"].attrs.get("Time", 1.0))
        Omega_m = float(h["Header"].attrs.get("Omega0", float("nan")))
        # Every cosmology entry the header can adjudicate.  TNG300-Dark carries
        # all three (verified on snapdir_096: Omega0=0.3089, OmegaBaryon=0.0486,
        # HubbleParam=0.6774); absent/non-positive entries are simply skipped.
        header_cosmo = {k: float(h["Header"].attrs.get(k, float("nan")))
                        for _, k in _HEADER_COSMO}

    # Refuse to project (and, later, to hand stage 2) a vector whose cosmology is
    # not the substrate's.  Checked here, on every rank, BEFORE the hours-long
    # projection — not after it.
    _check_cosmology(params, header_cosmo, strict=not allow_cosmology_mismatch,
                     where="stage1")

    npix = _round_npix(box_size, pixel_size)
    n_slabs = _round_n_slabs(box_size, slab_depth)

    redshift = 1.0 / scale_factor - 1.0

    if rank == 0:
        print(f"[stage1] box={box_size:.3f} Mpc/h  npix={npix}  n_slabs={n_slabs}")
        print(f"[stage1] scale_factor={scale_factor:.4f}  z={redshift:.4f}  Omega_m={Omega_m:.4f}")
        print(f"[stage1] {len(snap_files)} snapshot chunk(s) across {size} rank(s)")
        print(f"[stage1] particle_mass={particle_mass:.3e} Msun/h")
        if transforms is not None:
            pd = int(transforms.proj_dirs[transforms_snap_idx])
            print(f"[stage1] lightcone transform: snap_idx={transforms_snap_idx}  "
                  f"proj_dir={pd}  disp={transforms.disp[transforms_snap_idx].tolist()}")

    # --- project this rank's chunks into local slab maps -------------------
    local_slabs = np.zeros((n_slabs, npix, npix), dtype=np.float32)
    # Interlacing: a second deposit shifted by half a transverse pixel.  The
    # LOS (z) slab assignment is unchanged; only x,y are shifted before deposit.
    local_slabs_s = np.zeros((n_slabs, npix, npix), dtype=np.float32) if mas_correct else None
    half_pix = 0.5 * box_size / npix
    my_files = snap_files[rank::size]
    iterator = tqdm(my_files, desc=f"[rank {rank}] projecting") if (progress and my_files) else my_files
    t0 = time.time()
    for fname in iterator:
        with h5py.File(fname, "r") as h:
            pos = h["PartType1/Coordinates"][:].astype(np.float32) / 1000.0
        if transforms is not None:
            pos = transforms.apply(pos, transforms_snap_idx, box_size)
        _accumulate_chunk_into_slabs(local_slabs, pos, particle_mass, box_size, n_slabs)
        if mas_correct:
            pos_s = pos.copy()
            pos_s[:, 0] = (pos_s[:, 0] + half_pix) % box_size
            pos_s[:, 1] = (pos_s[:, 1] + half_pix) % box_size
            _accumulate_chunk_into_slabs(local_slabs_s, pos_s, particle_mass, box_size, n_slabs)
            del pos_s
        del pos

    # --- reduce partial maps onto rank 0 (slab-by-slab to cap message size) -
    if comm is not None and size > 1:
        from mpi4py import MPI

        if not my_files:
            # Ranks with no chunks never wrote their calloc'd buffers; UCX CMA
            # (process_vm_readv) aborts on such unfaulted pages during Reduce.
            local_slabs.fill(0.0)
            if local_slabs_s is not None:
                local_slabs_s.fill(0.0)

        global_slabs = np.zeros_like(local_slabs) if rank == 0 else None
        global_slabs_s = (np.zeros_like(local_slabs_s)
                          if (mas_correct and rank == 0) else None)
        for si in range(n_slabs):
            recv = global_slabs[si] if rank == 0 else None
            comm.Reduce(local_slabs[si], recv, op=MPI.SUM, root=0)
            if mas_correct:
                recv_s = global_slabs_s[si] if rank == 0 else None
                comm.Reduce(local_slabs_s[si], recv_s, op=MPI.SUM, root=0)
        comm.Barrier()
    else:
        global_slabs = local_slabs
        global_slabs_s = local_slabs_s

    if rank != 0:
        return None

    # Anti-aliased DMO maps (interlace + deconvolve); raw `global_slabs` is kept
    # intact as the model-condition source.
    aa_slabs = None
    if mas_correct:
        aa_slabs = np.stack([
            interlace_combine_2d(global_slabs[si], global_slabs_s[si], deconvolve=True)
            for si in range(n_slabs)
        ]).astype(np.float32)

    print(f"[stage1] projection done in {time.time() - t0:.1f}s; reading halos...")

    # --- rank 0: FoF catalog, cutouts, save -------------------------------
    cat = io_gadget.read_fof_catalog(
        group_catalog, snapshot=snapshot_index,
        halo_mass_min=halo_mass_min, mass_field=halo_mass_field,
    )
    halo_pos = cat["positions"]        # (M, 3) in original frame [Mpc/h]
    halo_mass = cat["mass"]
    halo_r200 = cat["r200"]

    # Apply the same transform to halo centres so they align with the projected slabs.
    if transforms is not None:
        halo_pos = transforms.apply(halo_pos, transforms_snap_idx, box_size)

    halo_slab_idx = _assign_halos_to_slabs(halo_pos[:, 2], box_size, n_slabs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "params.npy", params.astype(np.float64))

    n_halos_total = int(len(halo_mass))
    print(f"[stage1] {n_halos_total} halos with M>{halo_mass_min:.1e} Msun/h")

    per_slab_meta = []
    for si in range(n_slabs):
        slab_map = global_slabs[si]
        aa_kw = {"dmo_aa": aa_slabs[si]} if aa_slabs is not None else {}
        in_slab = np.where(halo_slab_idx == si)[0]
        n = int(len(in_slab))
        slab_path = _stage1_slab_path(output_dir, si)

        if n == 0:
            np.savez(
                slab_path, dmo=slab_map, n_halos=0, slab_idx=si,
                n_slabs=n_slabs, box_size=box_size, npix=npix, **aa_kw,
            )
            per_slab_meta.append({"slab_idx": si, "n_halos": 0})
            print(f"[stage1] slab {si}: 0 halos")
            continue

        halo_xy = halo_pos[in_slab, :2]
        halo_m = halo_mass[in_slab]
        halo_r = halo_r200[in_slab]

        cutouts = extract_halo_cutouts(
            slab_map, halo_xy, box_size=box_size, patch_pix=patch_pix
        )
        # condition: (128,128); large_scale: (3,128,128)  -> stack over halos
        cond = np.stack([c["condition"] for c in cutouts]).astype(np.float32)
        large_scale = np.stack([c["large_scale"] for c in cutouts]).astype(np.float32)

        # Plain np.savez (not compressed): the cutout arrays are multi-GB and
        # this intermediate is transient; speed matters more than disk here.
        np.savez(
            slab_path,
            dmo=slab_map,
            condition=cond,
            large_scale=large_scale,
            halo_centers=halo_xy.astype(np.float32),
            halo_masses=halo_m.astype(np.float32),
            halo_r200=halo_r.astype(np.float32),
            n_halos=n,
            slab_idx=si,
            n_slabs=n_slabs,
            box_size=box_size,
            npix=npix,
            **aa_kw,
        )
        per_slab_meta.append({"slab_idx": si, "n_halos": n})
        print(f"[stage1] slab {si}: {n} halos -> {slab_path.name}")

    manifest = {
        "bind_version": bind_version(),
        "box_size": box_size,
        "npix": npix,
        "n_slabs": n_slabs,
        "mas_correct": bool(mas_correct),
        "pixel_size": pixel_size,
        "slab_depth": slab_depth,
        "patch_pix": patch_pix,
        "n_halos": n_halos_total,
        "halo_mass_min": halo_mass_min,
        "halo_mass_field": halo_mass_field,
        "snapshot": str(snapshot),
        "group_catalog": str(group_catalog),
        "snapshot_index": snapshot_index,
        "particle_mass": particle_mass,
        "scale_factor": scale_factor,
        "redshift": redshift,
        "Omega_m": Omega_m,
        "params_file": "params.npy",
        "per_slab": per_slab_meta,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if transforms is not None:
        manifest["transforms_snap_idx"] = transforms_snap_idx
        manifest["proj_dir"] = int(transforms.proj_dirs[transforms_snap_idx])
        manifest["disp"] = transforms.disp[transforms_snap_idx].tolist()
        manifest["flip"] = transforms.flip[transforms_snap_idx].tolist()
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"[stage1] wrote manifest -> {output_dir / MANIFEST_NAME}")
    return output_dir


# ---------------------------------------------------------------------------
# Stage 2: generation + compositing (GPU)
# ---------------------------------------------------------------------------

def _save_composite_slab(
    output_dir: Path | str,
    si: int,
    *,
    n_slabs: int,
    box_size: float,
    dmo: np.ndarray,
    bundle: dict,
    halo_centers: np.ndarray,
    halo_masses: np.ndarray,
    halo_r200: np.ndarray,
    generated_patches: np.ndarray | None = None,
    thermo_patches: np.ndarray | None = None,
    condition_sums: np.ndarray | None = None,
    provenance: dict | None = None,
) -> Path:
    """Write one ``composite_slab{NN}.npz`` (shared by generate + recomposite).

    ``condition_sums`` (per-halo DMO cutout mass) is saved so the result can be
    re-composited later with new blend settings without re-reading the stage-1
    cutouts. ``generated_patches`` (and ``thermo_patches``) are carried so the
    output is itself re-compositable.
    """
    slab_path = Path(output_dir) / f"composite_slab{si:02d}.npz"
    kw: dict[str, Any] = dict(
        dmo=dmo,
        composite=bundle["composite"],
        alpha=bundle["alpha"],
        patch_scales=bundle["patch_scales"],
        scale_global=bundle["scale_global"],
        coverage_pct=bundle["coverage_pct"],
        n_halos=int(len(halo_masses)),
        slab_idx=si,
        n_slabs=n_slabs,
        box_size=box_size,
        halo_centers=np.asarray(halo_centers, np.float32),
        halo_masses=np.asarray(halo_masses, np.float32),
        halo_r200=np.asarray(halo_r200, np.float32),
    )
    if "composite_thermo" in bundle:
        kw["composite_thermo"] = bundle["composite_thermo"]
    if generated_patches is not None:
        kw["generated_patches"] = generated_patches
    if thermo_patches is not None:
        kw["thermo_patches"] = thermo_patches
    if condition_sums is not None:
        kw["condition_sums"] = np.asarray(condition_sums, np.float32)
    prov = provenance_array(provenance)
    if prov is not None:
        kw[PROVENANCE_KEY] = prov
    np.savez_compressed(slab_path, **kw)
    return slab_path


def generate_from_stage1(
    stage1_dir: Path | str,
    model: Model,
    *,
    output_dir: Path | str,
    params: np.ndarray | None = None,
    redshift: float | None = None,
    scale_factor: float | None = None,
    n_steps: int = 50,
    batch_size: int = 16,
    use_amp: bool = True,
    patch_mass_match: bool = True,
    taper_frac: float = 0.15,
    r200_factor: float = 4.0,
    paste_mode: str = "shared",
    save_per_halo_patches: bool = True,
    progress: bool = True,
    seed: int | None = None,
) -> PaintResult:
    """Stage 2: run the sampler on stage-1 cutouts and composite per slab.

    Reads the ``stage1_manifest.json`` + ``stage1_slab{NN}.npz`` written by
    :func:`project_and_extract`, generates hydro patches on the GPU, and writes
    the same ``composite_slab{NN}.npz`` / ``summary.json`` as :func:`bind.paint`.

    For a redshift-conditioned model (``fm_redshift_thermo``), the scale factor
    ``a=1/(1+z)`` is read from the manifest (written by stage 1 from the snapshot
    Header/Time attribute) and forwarded to ``model.generate``.  Pass ``redshift``
    or ``scale_factor`` to override the manifest value.

    ``seed`` (optional) makes the sampling reproducible: each slab draws from its
    own sub-seed derived from it, so no two slabs replay the same noise.  ``None``
    keeps the historical unseeded behaviour (global RNG, different every run).
    The resolved settings — including the seed and ``batch_size``, which the noise
    stream depends on — are stamped into every output as ``provenance``.
    """
    if redshift is not None and scale_factor is not None:
        raise ValueError("pass only one of redshift= / scale_factor=")

    stage1_dir = Path(stage1_dir)
    manifest = json.loads((stage1_dir / MANIFEST_NAME).read_text())

    box_size = float(manifest["box_size"])
    npix = int(manifest["npix"])
    n_slabs = int(manifest["n_slabs"])
    patch_pix = int(manifest["patch_pix"])
    pixel_size = float(manifest["pixel_size"])
    slab_depth = float(manifest["slab_depth"])

    # Resolve redshift: explicit override > manifest > default z=0
    if scale_factor is not None:
        _redshift = 1.0 / scale_factor - 1.0
        _scale_factor = float(scale_factor)
    elif redshift is not None:
        _redshift = float(redshift)
        _scale_factor = 1.0 / (1.0 + _redshift)
    elif "scale_factor" in manifest:
        _scale_factor = float(manifest["scale_factor"])
        _redshift = float(manifest.get("redshift", 1.0 / _scale_factor - 1.0))
    else:
        _scale_factor = None
        _redshift = None

    if params is None:
        params = np.load(stage1_dir / manifest.get("params_file", "params.npy"))
    params = _validate_params(params)
    # The stage-1 manifest carries only Omega_m; OmegaBaryon/HubbleParam are
    # skipped automatically when absent, so this stays a one-entry check here and
    # the full three-entry check lives where the header is actually readable
    # (project_and_extract).
    _check_cosmology(params, {"Omega0": float(manifest.get("Omega_m", float("nan")))},
                     strict=False, where="stage2")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = model.provenance(
        n_steps=n_steps,
        r200_factor=r200_factor,
        paste_mode=paste_mode,
        seed=seed,
        batch_size=batch_size,
        taper_frac=taper_frac,
        patch_mass_match=patch_mass_match,
        use_amp=use_amp,
        npix=npix,
        n_slabs=n_slabs,
        box_size=box_size,
        patch_pix=patch_pix,
        stage1_dir=str(stage1_dir),
        entry_point="bind-paint-generate",
    )

    print(f"[stage2] {model!r}")
    if _scale_factor is not None:
        print(f"[stage2] scale_factor={_scale_factor:.4f}  z={_redshift:.4f}")
    print(f"[stage2] box={box_size:.3f} Mpc/h  npix={npix}  n_slabs={n_slabs}  "
          f"{manifest['n_halos']} halos")

    composite_paths: list[Path] = []
    per_slab: list[dict] = []

    for si in range(n_slabs):
        d = np.load(_stage1_slab_path(stage1_dir, si))
        # Composite background: prefer the anti-aliased DMO map when stage 1
        # produced one (mas_correct), so the composite -> lensplane path matches
        # an anti-aliased reference.  Model conditions still come from the raw
        # `condition` cutouts below, so generative fidelity is unaffected.
        bg_map = d["dmo_aa"] if "dmo_aa" in d.files else d["dmo"]
        n = int(d["n_halos"])

        if n == 0:
            composite_paths.append(_save_empty_slab(output_dir, si, n_slabs, box_size, bg_map,
                                                    provenance=provenance))
            per_slab.append({"slab_idx": si, "n_halos": 0})
            continue

        cond = d["condition"]
        large_scale = d["large_scale"]
        halo_xy = d["halo_centers"]
        halo_m = d["halo_masses"]
        halo_r = d["halo_r200"]
        cutouts = [
            {"condition": cond[i], "large_scale": large_scale[i]} for i in range(n)
        ]

        slab_seed = derive_seed(seed, f"slab{si}")
        gen = model.generate(
            cutouts, params,
            n_steps=n_steps, batch_size=batch_size, use_amp=use_amp,
            progress=progress, scale_factor=_scale_factor, seed=slab_seed,
        )

        halos_dicts = [
            {"halo_center": halo_xy[i], "halo_mass": float(halo_m[i]),
             "r200": float(halo_r[i]), "params": params.astype(np.float32)}
            for i in range(n)
        ]
        thermo = (gen[:, 3:3 + N_THERMO]
                  if (model.predict_thermo and gen.shape[1] >= 3 + N_THERMO) else None)
        bundle = build_bind_composite(
            bg_map, halos_dicts, gen[:, :3], cutouts,
            box_size=box_size, npix=npix, patch_pix=patch_pix,
            patch_mass_match=patch_mass_match, taper_frac=taper_frac,
            r200_factor=r200_factor, paste_mode=paste_mode, thermo_patches=thermo,
        )
        slab_path = _save_composite_slab(
            output_dir, si, n_slabs=n_slabs, box_size=box_size,
            dmo=bg_map, bundle=bundle,
            halo_centers=halo_xy, halo_masses=halo_m, halo_r200=halo_r,
            generated_patches=(gen[:, :3] if save_per_halo_patches else None),
            thermo_patches=(thermo if save_per_halo_patches else None),
            condition_sums=cond.sum(axis=(1, 2)),
            provenance={**provenance, "slab_idx": si, "slab_seed": slab_seed},
        )

        composite_paths.append(slab_path)
        per_slab.append({
            "slab_idx": si,
            "n_halos": n,
            "coverage_pct": float(bundle["coverage_pct"]),
            "scale_global": float(bundle["scale_global"]),
        })
        print(f"[stage2] slab {si}: {n} halos, coverage={bundle['coverage_pct']:.1f}%")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "box_size": box_size,
        "pixel_size": pixel_size,
        "slab_depth": slab_depth,
        "npix": npix,
        "n_slabs": n_slabs,
        "n_halos": int(manifest["n_halos"]),
        "scale_factor": _scale_factor,
        "redshift": _redshift,
        "model": repr(model),
        "n_steps": n_steps,
        "patch_mass_match": patch_mass_match,
        "taper_frac": taper_frac,
        "r200_factor": r200_factor,
        "paste_mode": paste_mode,
        "predict_thermo": model.predict_thermo,
        "thermo_keys": list(THERMO_KEYS) if model.predict_thermo else [],
        "stage1_dir": str(stage1_dir),
        "seed": seed,
        "provenance": to_jsonable(provenance),
        "per_slab": per_slab,
    }, indent=2))

    return PaintResult(
        n_halos=int(manifest["n_halos"]),
        n_slabs=n_slabs,
        box_size=box_size,
        npix=npix,
        output_dir=output_dir,
        summary_path=summary_path,
        composite_paths=composite_paths,
        per_slab=per_slab,
        predict_thermo=model.predict_thermo,
        thermo_keys=THERMO_KEYS if model.predict_thermo else (),
    )


def generate_halos(
    stage1_dir: Path | str,
    model: Model,
    *,
    output_dir: Path | str,
    params: np.ndarray | None = None,
    redshift: float | None = None,
    scale_factor: float | None = None,
    n_steps: int = 50,
    batch_size: int = 16,
    use_amp: bool = True,
    progress: bool = True,
) -> Path:
    """GPU generation **only** (no compositing) — the portable, GPU-heavy half.

    Runs the flow-matching sampler on the stage-1 cutouts and saves just the
    per-halo patches as ``composite_slab{NN}.npz`` (``generated_patches`` +
    ``thermo_patches`` + halo metadata, **no full-box maps and no DMO**).  This
    needs only the cutouts (``condition`` / ``large_scale``), so a stripped
    stage-1 (cutouts + params) can be shipped to a GPU-rich machine, generated
    there, and the halos shipped back; compositing (which needs the DMO
    background) then runs on the CPU side via :func:`recomposite_from_saved`.
    """
    stage1_dir = Path(stage1_dir)
    manifest = json.loads((stage1_dir / MANIFEST_NAME).read_text())
    n_slabs = int(manifest["n_slabs"])
    box_size = float(manifest["box_size"])
    if scale_factor is not None:
        _sf = float(scale_factor)
    elif redshift is not None:
        _sf = 1.0 / (1.0 + float(redshift))
    else:
        _sf = float(manifest.get("scale_factor")) if "scale_factor" in manifest else None
    if params is None:
        params = np.load(stage1_dir / manifest.get("params_file", "params.npy"))
    params = _validate_params(params)
    # The stage-1 manifest carries only Omega_m; OmegaBaryon/HubbleParam are
    # skipped automatically when absent, so this stays a one-entry check here and
    # the full three-entry check lives where the header is actually readable
    # (project_and_extract).
    _check_cosmology(params, {"Omega0": float(manifest.get("Omega_m", float("nan")))},
                     strict=False, where="generate_halos")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for si in range(n_slabs):
        d = np.load(_stage1_slab_path(stage1_dir, si))
        n = int(d["n_halos"])
        slab_path = Path(output_dir) / f"composite_slab{si:02d}.npz"
        if n == 0:
            np.savez(slab_path, n_halos=0, slab_idx=si, n_slabs=n_slabs, box_size=box_size)
            continue
        cutouts = [{"condition": d["condition"][i], "large_scale": d["large_scale"][i]}
                   for i in range(n)]
        gen = model.generate(cutouts, params, n_steps=n_steps, batch_size=batch_size,
                             use_amp=use_amp, progress=progress, scale_factor=_sf)
        thermo = (gen[:, 3:3 + N_THERMO]
                  if (model.predict_thermo and gen.shape[1] >= 3 + N_THERMO) else None)
        kw = dict(generated_patches=gen[:, :3].astype(np.float32),
                  halo_centers=d["halo_centers"], halo_masses=d["halo_masses"],
                  halo_r200=d["halo_r200"], condition_sums=d["condition"].sum(axis=(1, 2)),
                  n_halos=n, slab_idx=si, n_slabs=n_slabs, box_size=box_size)
        if thermo is not None:
            kw["thermo_patches"] = thermo.astype(np.float32)
        np.savez_compressed(slab_path, **kw)
        print(f"[generate-halos] slab {si}: {n} halos -> {slab_path.name}")
    return output_dir


# ---------------------------------------------------------------------------
# Stage 3: re-composite already-generated patches (CPU, no GPU / no regen)
# ---------------------------------------------------------------------------

def recomposite_slab(
    stage1_npz: Path | str,
    generated_npz: Path | str,
    *,
    params: np.ndarray | None = None,
    patch_mass_match: bool = True,
    taper_frac: float = 0.15,
    r200_factor: float = 4.0,
    paste_mode: str = "shared",
) -> dict | None:
    """Re-composite ONE slab's already-generated patches with new blend settings.

    The flow-matching sampler is the expensive part of stage 2 and its output is
    already saved (``generated_patches`` in each ``composite_slab{NN}.npz``). This
    reloads those patches + the DMO map and ``condition`` cutouts from the stage-1
    npz and re-runs only :func:`build_bind_composite` — **no model, no GPU** — so
    you can sweep ``taper_frac`` / ``r200_factor`` / ``patch_mass_match`` cheaply
    (e.g. interactively in a notebook).

    Returns the :func:`build_bind_composite` bundle (``composite`` (3, npix, npix),
    ``alpha``, ``coverage_pct``, ``scale_global``, ``patch_scales``), or ``None``
    for an empty slab.
    """
    s = np.load(stage1_npz)
    g = np.load(generated_npz)
    if int(s["n_halos"]) == 0:
        return None
    if "generated_patches" not in g.files:
        raise ValueError(
            f"{generated_npz} has no 'generated_patches' — stage 2 was run with "
            "--no_save_patches, so there is nothing to re-composite."
        )

    dmo = s["dmo_aa"] if "dmo_aa" in s.files else s["dmo"]
    box_size = float(s["box_size"])
    npix = int(s["npix"])
    gen = g["generated_patches"]                  # (N, 3, patch, patch)
    n = gen.shape[0]
    patch_pix = gen.shape[-1]
    cond = s["condition"]
    centers, mass, r200 = s["halo_centers"], s["halo_masses"], s["halo_r200"]
    p = (np.zeros(35, np.float32) if params is None
         else np.asarray(params, np.float32).reshape(-1))

    thermo = g["thermo_patches"] if "thermo_patches" in g.files else None
    cutouts = [{"condition": cond[i]} for i in range(n)]
    halos = [
        {"halo_center": centers[i], "halo_mass": float(mass[i]),
         "r200": float(r200[i]), "params": p}
        for i in range(n)
    ]
    return build_bind_composite(
        dmo, halos, gen, cutouts,
        box_size=box_size, npix=npix, patch_pix=patch_pix,
        patch_mass_match=patch_mass_match, taper_frac=taper_frac,
        r200_factor=r200_factor, paste_mode=paste_mode, thermo_patches=thermo,
    )


def recomposite_from_saved(
    stage1_dir: Path | str,
    generated_dir: Path | str,
    output_dir: Path | str,
    *,
    params: np.ndarray | None = None,
    patch_mass_match: bool = True,
    taper_frac: float = 0.15,
    r200_factor: float = 4.0,
    paste_mode: str = "shared",
    save_per_halo_patches: bool = True,
    progress: bool = True,
    seed: int | None = None,
) -> PaintResult:
    """Re-composite ALL slabs from saved patches with new settings (no GPU).

    Reuses the generated patches in ``generated_dir/composite_slab*.npz`` and the
    stage-1 cutouts; writes fresh ``composite_slab*.npz`` + ``summary.json`` to
    ``output_dir`` (use a *different* directory so the originals are preserved).
    The generated (and thermo) patches are carried through unchanged, so the new
    output is itself re-compositable.

    Re-compositing is deterministic — it draws no noise — so ``seed`` changes
    nothing about the result; it only lets a caller record the seed of the
    generation being re-composited when the source predates provenance stamping.
    The source run's own provenance (checkpoint identity, n_steps, its seed) is
    inherited from the generated npz and carried into the new outputs.
    """
    stage1_dir = Path(stage1_dir)
    generated_dir = Path(generated_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    man = json.loads((stage1_dir / MANIFEST_NAME).read_text())
    n_slabs = int(man["n_slabs"])
    box_size = float(man["box_size"])
    npix = int(man["npix"])
    if params is None:
        params = np.load(stage1_dir / man.get("params_file", "params.npy"))
    params = _validate_params(params)

    # Inherit the generation's identity from the first stamped source slab: the
    # patches were sampled there, so its checkpoint sha / n_steps / seed are the
    # honest record — re-hashing a path that may no longer hold that file is not.
    source_prov: dict | None = None
    for si in range(n_slabs):
        gp = generated_dir / f"composite_slab{si:02d}.npz"
        if gp.exists():
            source_prov = read_provenance(gp)
            if source_prov is not None:
                break
    src = source_prov or {}
    provenance = build_provenance(
        n_steps=src.get("n_steps"),
        r200_factor=r200_factor,
        paste_mode=paste_mode,
        seed=seed if seed is not None else src.get("seed"),
        taper_frac=taper_frac,
        patch_mass_match=patch_mass_match,
        npix=npix,
        n_slabs=n_slabs,
        box_size=box_size,
        stage1_dir=str(stage1_dir),
        recomposited_from=str(generated_dir),
        entry_point="bind-paint-recomposite",
        source_provenance=source_prov,
    )
    for key in ("bind_version", "checkpoint_path", "checkpoint_sha256",
                "norm_stats_path", "norm_stats_sha256", "batch_size", "model"):
        if key in src:
            provenance[f"source_{key}"] = src[key]

    print(f"[recomposite] {generated_dir} -> {output_dir}  "
          f"(taper_frac={taper_frac}, r200_factor={r200_factor}, "
          f"paste_mode={paste_mode}, patch_mass_match={patch_mass_match})")

    composite_paths: list[Path] = []
    per_slab: list[dict] = []
    for si in range(n_slabs):
        s1 = _stage1_slab_path(stage1_dir, si)
        gp = generated_dir / f"composite_slab{si:02d}.npz"
        s = np.load(s1)
        if int(s["n_halos"]) == 0:
            composite_paths.append(_save_empty_slab(output_dir, si, n_slabs, box_size, s["dmo"],
                                                    provenance=provenance))
            per_slab.append({"slab_idx": si, "n_halos": 0})
            continue
        if not gp.exists():
            raise FileNotFoundError(f"missing generated composite for slab {si}: {gp}")

        bundle = recomposite_slab(
            s1, gp, params=params, patch_mass_match=patch_mass_match,
            taper_frac=taper_frac, r200_factor=r200_factor, paste_mode=paste_mode,
        )
        g = np.load(gp)
        slab_path = _save_composite_slab(
            output_dir, si, n_slabs=n_slabs, box_size=box_size,
            dmo=s["dmo"], bundle=bundle,
            halo_centers=s["halo_centers"], halo_masses=s["halo_masses"],
            halo_r200=s["halo_r200"],
            generated_patches=(g["generated_patches"]
                               if (save_per_halo_patches and "generated_patches" in g.files) else None),
            thermo_patches=(g["thermo_patches"]
                            if (save_per_halo_patches and "thermo_patches" in g.files) else None),
            condition_sums=s["condition"].sum(axis=(1, 2)),
            provenance={**provenance, "slab_idx": si},
        )
        composite_paths.append(slab_path)
        per_slab.append({
            "slab_idx": si, "n_halos": int(s["n_halos"]),
            "coverage_pct": float(bundle["coverage_pct"]),
            "scale_global": float(bundle["scale_global"]),
        })
        if progress:
            print(f"[recomposite] slab {si}: {int(s['n_halos'])} halos, "
                  f"coverage={bundle['coverage_pct']:.1f}%")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "box_size": box_size,
        "pixel_size": float(man["pixel_size"]),
        "slab_depth": float(man["slab_depth"]),
        "npix": npix,
        "n_slabs": n_slabs,
        "n_halos": int(man["n_halos"]),
        "recomposited_from": str(generated_dir),
        "stage1_dir": str(stage1_dir),
        "patch_mass_match": patch_mass_match,
        "taper_frac": taper_frac,
        "r200_factor": r200_factor,
        "paste_mode": paste_mode,
        "seed": provenance["seed"],
        "provenance": to_jsonable(provenance),
        "per_slab": per_slab,
    }, indent=2))

    return PaintResult(
        n_halos=int(man["n_halos"]),
        n_slabs=n_slabs,
        box_size=box_size,
        npix=npix,
        output_dir=output_dir,
        summary_path=summary_path,
        composite_paths=composite_paths,
        per_slab=per_slab,
    )
