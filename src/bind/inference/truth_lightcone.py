"""Truth lightcone: per-halo TNG300 **hydro** patches in BIND's geometry.

Apples-to-apples comparison for the fiducial BIND lightcone.  For each lightcone
snapshot this projects the actual IllustrisTNG **hydro** fields (DM_hydro, Gas,
Stars + gas-thermo compton_y/T/entropy/P_e) at the **same M>=10^13 DMO-FoF halos**
BIND paints, under the **same** lightcone transform, and writes them in the BIND
``composite_slab{NN}.npz`` patch format.  The result is a drop-in for the normal
recomposite -> lensplane -> lux -> stats pipeline, so the truth lightcone shares
BIND's exact slabs / transforms / ray-tracing geometry.

Per-particle thermo physics and the comoving->physical ``a``-factors match the
training-data generator (``data_generation/process_simulations_multiz.py``), so
the truth ``compton_y`` is in the same physical units BIND learned.
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from . import io_gadget
from .lightcone_transforms import LightconeTransforms
from .paint import (
    NATIVE_PIXEL_SIZE_MPCH,
    NATIVE_SLAB_DEPTH_MPCH,
    PATCH_PIX,
    _assign_halos_to_slabs,
    _project_zslabs,
    _round_n_slabs,
    _round_npix,
    extract_halo_cutouts,
)
from .pipeline import (
    GAMMA,
    K_B_J_PER_K,
    KEV_IN_J,
    KPC_IN_M,
    M_PROTON_KG,
    MPC_IN_M,
    MSUN_KG,
    X_H,
    _safe_divide,
    compton_y_integrand_per_particle,
    gas_temperature_K,
)

# THERMO_KEYS order: compton_y, temperature, entropy, pressure
N_MASS = 3       # DM_hydro, Gas, Stars
N_CH = 8         # hdm, gas, star, y_sum, m_hot, Tm, Pm, Km  (slab accumulators)


def _hydro_snap_files(snapshot: Path | str, snapshot_index: int) -> list[str]:
    p = Path(snapshot)
    if p.is_file():
        return [str(p)]
    pat = p / f"snapdir_{snapshot_index:03d}" / f"snap_{snapshot_index:03d}.*.hdf5"
    files = sorted(glob.glob(str(pat)))
    if not files:
        files = sorted(glob.glob(str(p / f"snap_{snapshot_index:03d}.*.hdf5")))
    if not files:
        raise FileNotFoundError(f"no hydro snapshot files for {snapshot} idx {snapshot_index}")
    return files


def _accumulate_chunk(local, fname, transforms, snap_idx, box_size, n_slabs,
                      a, h, dm_particle_mass):
    """Project one hydro chunk's PartType0/1/4 into the 8 per-slab accumulators.

    ``local`` has shape ``(N_CH, n_slabs, npix, npix)``; each ``_project_zslabs``
    call returns ``(n_slabs, npix, npix)`` and is added onto a channel.
    """
    npix = local.shape[2]
    density_to_kg_m3 = 1e10 * MSUN_KG * h ** 2 / KPC_IN_M ** 3 / a ** 3
    mass_code_to_kg = 1e10 * MSUN_KG / h
    # divide the (huge) per-particle compton-y integrand by the physical pixel area
    # BEFORE casting to float32 (matches process_simulations_multiz; avoids overflow)
    pixel_area_phys = ((box_size / npix) / h * MPC_IN_M * a) ** 2

    def _tx(pos):
        pos = pos.astype(np.float32) / 1000.0   # ckpc/h -> cMpc/h
        return transforms.apply(pos, snap_idx, box_size) if transforms is not None else pos

    with h5py.File(fname, "r") as f:
        # DM (PartType1): uniform mass from MassTable
        if "PartType1/Coordinates" in f:
            pos = _tx(f["PartType1/Coordinates"][:])
            m = np.full(len(pos), dm_particle_mass, np.float32)        # Msun/h
            local[0] += _project_zslabs(pos, m, box_size, npix, n_slabs)
        # Stars (PartType4)
        if "PartType4/Coordinates" in f:
            pos = _tx(f["PartType4/Coordinates"][:])
            m = f["PartType4/Masses"][:].astype(np.float32) * 1e10     # Msun/h
            local[2] += _project_zslabs(pos, m, box_size, npix, n_slabs)
        # Gas (PartType0): mass map (all gas) + thermo (hot gas only)
        if "PartType0/Coordinates" in f:
            gpos = f["PartType0/Coordinates"][:]
            gm = f["PartType0/Masses"][:].astype(np.float64)           # code (1e10 Msun/h)
            dens = f["PartType0/Density"][:].astype(np.float64)
            u = f["PartType0/InternalEnergy"][:].astype(np.float64)
            xe = f["PartType0/ElectronAbundance"][:].astype(np.float64)
            sfr = f["PartType0/StarFormationRate"][:] if "PartType0/StarFormationRate" in f \
                else np.zeros(len(gm))
            pos_all = _tx(gpos)
            local[1] += _project_zslabs(pos_all, (gm * 1e10).astype(np.float32),
                                        box_size, npix, n_slabs)            # Gas mass [Msun/h]
            hot = np.asarray(sfr) <= 0.0
            if hot.any():
                ph = pos_all[hot]
                gmh = gm[hot]
                T = gas_temperature_K(u[hot], xe[hot])                       # [K]
                rho = dens[hot] * density_to_kg_m3                           # [kg/m^3]
                P = ((GAMMA - 1.0) * rho * u[hot] * 1e6)                     # [Pa]
                ne = (xe[hot] * X_H * rho / M_PROTON_KG * 1e-6)              # [cm^-3]
                ne_safe = np.where(ne > 0, ne, 1.0)
                K = (K_B_J_PER_K * T / KEV_IN_J) / ne_safe ** (2.0 / 3.0)    # [keV cm^2]
                K[ne <= 0] = 0.0
                yint = compton_y_integrand_per_particle(u[hot], xe[hot], gmh, mass_code_to_kg)
                wm = gmh.astype(np.float32)
                local[3] += _project_zslabs(ph, (yint / pixel_area_phys).astype(np.float32),
                                            box_size, npix, n_slabs)
                local[4] += _project_zslabs(ph, wm, box_size, npix, n_slabs)
                local[5] += _project_zslabs(ph, (T * gmh).astype(np.float32), box_size, npix, n_slabs)
                local[6] += _project_zslabs(ph, (P * gmh).astype(np.float32), box_size, npix, n_slabs)
                local[7] += _project_zslabs(ph, (K * gmh).astype(np.float32), box_size, npix, n_slabs)


def extract_truth_halos(
    hydro_snapshot: Path | str,
    dmo_group_catalog: Path | str,
    *,
    output_dir: Path | str,
    snapshot_index: int,
    transforms: LightconeTransforms | None = None,
    transforms_snap_idx: int | None = None,
    halo_mass_min: float = 1e13,
    halo_mass_field: str = "Group_M_Crit200",
    pixel_size: float = NATIVE_PIXEL_SIZE_MPCH,
    slab_depth: float = NATIVE_SLAB_DEPTH_MPCH,
    patch_pix: int = PATCH_PIX,
    comm: Any | None = None,
    progress: bool = True,
) -> Path | None:
    """Project TNG300 hydro truth patches at the DMO halos -> composite_slab npz.

    Mirrors ``project_and_extract`` (same transform, slabs, halos) but on the
    hydro sim, writing per-halo ``generated_patches`` (DM_hydro, Gas, Stars) +
    ``thermo_patches`` (compton_y, T, entropy, P_e) — drop-in for recomposite.
    """
    rank = comm.rank if comm is not None else 0
    size = comm.size if comm is not None else 1
    snap_files = _hydro_snap_files(hydro_snapshot, snapshot_index)

    with h5py.File(snap_files[0], "r") as hf:
        hdr = hf["Header"].attrs
        box_size = float(hdr["BoxSize"]) / 1000.0
        a = float(hdr.get("Time", 1.0))
        h = float(hdr.get("HubbleParam", 0.6774))
        Omega_m = float(hdr.get("Omega0", float("nan")))
        dm_particle_mass = float(hdr["MassTable"][1]) * 1e10              # Msun/h
    npix = _round_npix(box_size, pixel_size)
    n_slabs = _round_n_slabs(box_size, slab_depth)
    if rank == 0:
        print(f"[truth] box={box_size:.2f} npix={npix} n_slabs={n_slabs} "
              f"a={a:.4f} z={1/a-1:.3f} Om={Omega_m:.4f}")

    local = np.zeros((N_CH, n_slabs, npix, npix), dtype=np.float32)
    my_files = snap_files[rank::size]
    t0 = time.time()
    for k, fname in enumerate(my_files):
        _accumulate_chunk(local, fname, transforms, transforms_snap_idx,
                          box_size, n_slabs, a, h, dm_particle_mass)
        if progress and rank == 0:
            print(f"[truth] chunk {k+1}/{len(my_files)} ({time.time()-t0:.0f}s)", flush=True)

    if comm is not None and size > 1:
        from mpi4py import MPI
        if not my_files:
            # Ranks with no chunks never wrote their calloc'd buffer; UCX CMA
            # (process_vm_readv) aborts on unfaulted pages during Reduce.
            local.fill(0.0)
        g = np.zeros_like(local) if rank == 0 else None
        comm.Reduce(local, g, op=MPI.SUM, root=0)
        comm.Barrier()
    else:
        g = local
    if rank != 0:
        return None

    # ---- halos (DMO FoF) + transform + slab assignment --------------------
    cat = io_gadget.read_fof_catalog(dmo_group_catalog, snapshot=snapshot_index,
                                     halo_mass_min=halo_mass_min, mass_field=halo_mass_field)
    hpos = cat["positions"]
    if transforms is not None:
        hpos = transforms.apply(hpos, transforms_snap_idx, box_size)
    hidx = _assign_halos_to_slabs(hpos[:, 2], box_size, n_slabs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_slab = []
    for si in range(n_slabs):
        in_slab = np.where(hidx == si)[0]
        n = int(len(in_slab))
        path = output_dir / f"composite_slab{si:02d}.npz"
        if n == 0:
            np.savez(path, n_halos=0, slab_idx=si, n_slabs=n_slabs, box_size=box_size)
            per_slab.append({"slab_idx": si, "n_halos": 0})
            continue
        xy = hpos[in_slab, :2]

        def cut(field):
            return np.stack([c["condition"] for c in
                             extract_halo_cutouts(field, xy, box_size=box_size, patch_pix=patch_pix)])

        hdm = cut(g[0, si])
        gas = cut(g[1, si])
        star = cut(g[2, si])
        y = cut(g[3, si])   # already in Compton-y units (divided by pixel area per particle)
        mh = cut(g[4, si])
        T = _safe_divide(cut(g[5, si]), mh)
        P = _safe_divide(cut(g[6, si]), mh)
        Kentr = _safe_divide(cut(g[7, si]), mh)
        mass_patches = np.stack([hdm, gas, star], axis=1).astype(np.float32)
        thermo_patches = np.stack([y, T, Kentr, P], axis=1).astype(np.float32)
        np.savez_compressed(
            path, generated_patches=mass_patches, thermo_patches=thermo_patches,
            halo_centers=xy.astype(np.float32),
            halo_masses=cat["mass"][in_slab].astype(np.float32),
            halo_r200=cat["r200"][in_slab].astype(np.float32),
            n_halos=n, slab_idx=si, n_slabs=n_slabs, box_size=box_size, npix=npix)
        per_slab.append({"slab_idx": si, "n_halos": n})
        print(f"[truth] slab {si}: {n} halos -> {path.name}")

    (output_dir / "truth_manifest.json").write_text(json.dumps({
        "box_size": box_size, "npix": npix, "n_slabs": n_slabs, "scale_factor": a,
        "redshift": 1 / a - 1, "Omega_m": Omega_m, "halo_mass_min": halo_mass_min,
        "hydro_snapshot": str(hydro_snapshot), "dmo_group_catalog": str(dmo_group_catalog),
        "per_slab": per_slab}, indent=2))
    return output_dir
