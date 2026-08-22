"""``python -m bind.cli.paint_project_hydro``: full-hydro lightcone slab projection.

Projects one FULL hydro snapshot (all matter + gas Compton-y) into the lightcone
slab geometry, writing ``composite_slab{NN}.npz`` files that are drop-in inputs
for ``bind.cli.paint_lensplane`` / ``paint_yplane`` / ``paint_tauplane`` — i.e. a
ground-truth alternative to the pasted-patch composites, with gas EVERYWHERE
along the line of sight instead of only inside halo paste apertures.

Output npz keys (the only ones stage 3 reads, plus bookkeeping):
  composite        (3, npix, npix) float32  Msun/h per pixel
                   ch0 = DM (+BH), ch1 = Gas (ALL gas, incl. star-forming — the
                   tau convention), ch2 = Stars.  ``paint_lensplane`` sums all
                   three; ``paint_tauplane`` converts ch1 to tau with the
                   pipeline's fixed x_e=0.88 fully-ionized convention.
  composite_thermo (4, npix, npix) float32  THERMO_KEYS order; only ch0
                   (compton_y) is populated — computed with the pipeline's
                   per-particle physics (T from InternalEnergy+ElectronAbundance,
                   per-particle x_e, SFR>0 gas EXCLUDED).  By default the
                   **legacy comoving-pixel-area** convention is used so the maps
                   are directly comparable to the existing pasted composites;
                   physical (proper-area) y = ch0 / scale_factor**2 under
                   ``--y_convention legacy`` (and = ch0 under ``physical``).

Streaming MPI design mirrors ``bind.inference.paint_stages.project_and_extract``:
each rank reads ``files[rank::size]`` chunk-by-chunk, applies the lightcone
transform, CIC-deposits into local (n_slabs, npix, npix) buffers, and the
buffers are reduced to rank 0 slab-by-slab.  Memory is independent of the total
particle count (~2 GB/rank at TNG300 scale).

    srun -n 64 python -m bind.cli.paint_project_hydro \
        --hydro_snapdir .../L205n2500TNG/output/snapdir_096 --snapshot_index 96 \
        --stage1_dir .../bind_lightcone_tng/snap_096/stage1 \
        --transforms .../lightcone_transforms.json --lc_snap_idx 0 \
        --output_dir .../tng_full_validation/composites/snap_096
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from bind.data import N_THERMO
from bind.inference import io_gadget
from bind.inference.lightcone_transforms import LightconeTransforms
from bind.inference.paint import _project_zslabs
from bind.inference.pipeline import (
    MPC_IN_M,
    MSUN_KG,
    compton_y_integrand_per_particle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hydro_snapdir", type=Path, required=True,
                   help="snapdir_NNN with snap_NNN.M.hdf5 chunks (full hydro box)")
    p.add_argument("--snapshot_index", type=int, required=True)
    p.add_argument("--stage1_dir", type=Path, required=True,
                   help="existing stage1 dir for THIS snapshot (geometry contract; "
                        "box/npix/n_slabs/scale_factor are asserted to match)")
    p.add_argument("--transforms", type=Path, required=True,
                   help="lightcone_transforms.json (shared fiducial geometry)")
    p.add_argument("--lc_snap_idx", type=int, required=True,
                   help="index into the transforms for this snapshot (0 = lowest-z)")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--y_convention", choices=["legacy", "physical"], default="physical",
                   help="'physical' = proper (a-corrected) pixel area — the convention "
                        "the truth lightcone thermo uses (truth_lightcone.py, multiz); "
                        "'legacy' = comoving pixel area (CAMELS z~0 code only). "
                        "legacy_y = physical_y * a^2.")
    return p.parse_args()


def _load_manifest(stage1_dir: Path) -> dict:
    with open(stage1_dir / "stage1_manifest.json") as f:
        return json.load(f)


def main() -> None:  # noqa: PLR0915
    args = parse_args()
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.rank, comm.size
    except ImportError:
        comm, rank, size = None, 0, 1

    man = _load_manifest(args.stage1_dir)
    box = float(man["box_size"])
    npix = int(man["npix"])
    n_slabs = int(man["n_slabs"])
    a_l = float(man["scale_factor"])

    snap_files = io_gadget._resolve_snap_files(args.hydro_snapdir, args.snapshot_index)
    with h5py.File(snap_files[0], "r") as f:
        hdr = f["Header"].attrs
        box_snap = float(hdr["BoxSize"]) / 1000.0
        h = float(hdr["HubbleParam"])
        a_snap = float(hdr.get("Time", 1.0))
        dm_mass = float(hdr["MassTable"][1]) * 1e10          # Msun/h
    assert abs(box_snap - box) < 1e-6, f"box mismatch: snapshot {box_snap} vs manifest {box}"
    assert abs(a_snap - a_l) < 1e-3, f"scale_factor mismatch: snapshot {a_snap} vs manifest {a_l}"

    transforms = LightconeTransforms.load(args.transforms)

    # Compton-y pixel area (see module docstring; matches pipeline.project_thermo_fullbox
    # for 'legacy', with the multiz proper-area correction for 'physical').
    pixel_side_m = (box / npix) / h * MPC_IN_M
    if args.y_convention == "physical":
        pixel_side_m *= a_l
    pixel_area_m2 = pixel_side_m ** 2
    mass_code_to_kg = 1e10 * MSUN_KG / h

    if rank == 0:
        print(f"[hydro-proj] snap {args.snapshot_index}: box={box} npix={npix} "
              f"n_slabs={n_slabs} a={a_l:.5f} y_convention={args.y_convention}")
        print(f"[hydro-proj] {len(snap_files)} chunks across {size} ranks")

    # local accumulators: 3 mass channels + y
    mass_local = np.zeros((3, n_slabs, npix, npix), dtype=np.float32)
    y_local = np.zeros((n_slabs, npix, npix), dtype=np.float32)
    counts = np.zeros(4, dtype=np.int64)     # gas, dm, stars, bh
    t0 = time.time()

    for fname in snap_files[rank::size]:
        with h5py.File(fname, "r") as f:
            # ---- gas: mass channel (ALL gas) + y weight (SFR<=0 only) ----
            if "PartType0" in f:
                g = f["PartType0"]
                pos = g["Coordinates"][:].astype(np.float32) / 1000.0
                pos = transforms.apply(pos, args.lc_snap_idx, box)
                mass = g["Masses"][:].astype(np.float32)             # code 1e10 Msun/h
                mass_local[1] += _project_zslabs(pos, mass * np.float32(1e10),
                                                 box, npix, n_slabs)
                u = g["InternalEnergy"][:]
                xe = g["ElectronAbundance"][:]
                sfr = g["StarFormationRate"][:]
                hot = sfr <= 0.0
                y_w = compton_y_integrand_per_particle(u[hot], xe[hot], mass[hot],
                                                       mass_code_to_kg)
                y_local += _project_zslabs(pos[hot],
                                           (y_w / pixel_area_m2).astype(np.float32),
                                           box, npix, n_slabs)
                counts[0] += len(pos)
                del pos, mass, u, xe, sfr, y_w
            # ---- DM (constant mass from MassTable) ----
            if "PartType1" in f:
                pos = f["PartType1/Coordinates"][:].astype(np.float32) / 1000.0
                pos = transforms.apply(pos, args.lc_snap_idx, box)
                w = np.full(len(pos), np.float32(dm_mass), dtype=np.float32)
                mass_local[0] += _project_zslabs(pos, w, box, npix, n_slabs)
                counts[1] += len(pos)
                del pos, w
            # ---- stars (incl. wind particles) ----
            if "PartType4" in f and "Masses" in f["PartType4"]:
                g = f["PartType4"]
                pos = g["Coordinates"][:].astype(np.float32) / 1000.0
                pos = transforms.apply(pos, args.lc_snap_idx, box)
                mass_local[2] += _project_zslabs(
                    pos, g["Masses"][:].astype(np.float32) * np.float32(1e10),
                    box, npix, n_slabs)
                counts[2] += len(pos)
                del pos
            # ---- BH (dynamical mass, into the DM channel; ~1e-4 of total) ----
            if "PartType5" in f and "Masses" in f["PartType5"]:
                g = f["PartType5"]
                pos = g["Coordinates"][:].astype(np.float32) / 1000.0
                pos = transforms.apply(pos, args.lc_snap_idx, box)
                mass_local[0] += _project_zslabs(
                    pos, g["Masses"][:].astype(np.float32) * np.float32(1e10),
                    box, npix, n_slabs)
                counts[3] += len(pos)
                del pos

    # ---- reduce to rank 0, slab-by-slab to cap message size ----
    if comm is not None and size > 1:
        from mpi4py import MPI
        mass_g = np.zeros_like(mass_local) if rank == 0 else None
        y_g = np.zeros_like(y_local) if rank == 0 else None
        counts_g = np.zeros_like(counts) if rank == 0 else None
        for ch in range(3):
            for si in range(n_slabs):
                recv = mass_g[ch, si] if rank == 0 else None
                comm.Reduce(mass_local[ch, si], recv, op=MPI.SUM, root=0)
        for si in range(n_slabs):
            recv = y_g[si] if rank == 0 else None
            comm.Reduce(y_local[si], recv, op=MPI.SUM, root=0)
        comm.Reduce(counts, counts_g, op=MPI.SUM, root=0)
        comm.Barrier()
    else:
        mass_g, y_g, counts_g = mass_local, y_local, counts

    if rank != 0:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for si in range(n_slabs):
        thermo = np.zeros((N_THERMO, npix, npix), dtype=np.float32)
        thermo[0] = y_g[si]
        np.savez_compressed(
            args.output_dir / f"composite_slab{si:02d}.npz",
            composite=mass_g[:, si],
            composite_thermo=thermo,
            box_size=np.float64(box), slab_idx=np.int64(si), n_slabs=np.int64(n_slabs),
            scale_factor=np.float64(a_l), y_convention=str(args.y_convention),
        )
        print(f"[hydro-proj] wrote composite_slab{si:02d}.npz  "
              f"(mass sum {mass_g[:, si].sum():.4e} Msun/h, y max {y_g[si].max():.3e})")

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump({
            "kind": "full_hydro_projection",
            "hydro_snapdir": str(args.hydro_snapdir),
            "snapshot_index": args.snapshot_index,
            "stage1_dir": str(args.stage1_dir),
            "lc_snap_idx": args.lc_snap_idx,
            "box_size": box, "npix": npix, "n_slabs": n_slabs,
            "scale_factor": a_l, "y_convention": args.y_convention,
            "particles": {"gas": int(counts_g[0]), "dm": int(counts_g[1]),
                          "stars": int(counts_g[2]), "bh": int(counts_g[3])},
            "channels": {"0": "DM+BH", "1": "Gas (all, incl. SFR>0)", "2": "Stars"},
            "sfr_cut_applied_to_y": True,
            "elapsed_s": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"[hydro-proj] snap {args.snapshot_index} done in {time.time() - t0:.0f}s  "
          f"gas={counts_g[0]:,} dm={counts_g[1]:,} stars={counts_g[2]:,} bh={counts_g[3]:,}")


if __name__ == "__main__":
    main()
