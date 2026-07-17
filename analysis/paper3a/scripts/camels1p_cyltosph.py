"""CylToSph feedback dependence from the CAMELS L50n512 TNG 1P suite.

The gate call promoted a feedback-dependent CylToSph treatment to a
mandatory A4/A5 component: the wp2 factor (~0.51-0.53) is calibrated at the
fiducial theta only, and its theta-dependence is unquantified. This script
measures, per 1P simulation and halo (M500c >= 1e13 Msun/h, the BIND floor),

    CylToSph_i = M_gas(< R500c sphere) / M_gas(< R500c cylinder, full box)

whose per-mass-bin median tracks exactly the factor the f_gas forward model
applies (identical M500c denominators cancel). The L50n512 1P suite is the
*same 30-parameter, +-bound design as the painted twobound bundle* (and the
volume behind the SB35 training set), so d ln CylToSph / d theta maps 1:1
onto the emulator's parameter axes. Absolute values are not comparable to
wp2 (50 Mpc/h box-depth cylinder vs the 51.25 Mpc/h painted slab is close,
but selection and resolution differ) — the fiducial-normalized trend is the
deliverable.

One task = one simulation directory (disBatch; camels1p_cyltosph.disbatch).
Snapshot 090 = z ~ 0, the eRASS1 gate axis. Chunked snapshots are streamed
(one HDF5 chunk in memory at a time), so a 1-core disBatch slot suffices.

Usage:  python analysis/paper3a/scripts/camels1p_cyltosph.py <sim_name> [snap]
Output: /mnt/ceph/users/mlee1/paper3/A/wp4_emulator/cyltosph_1p/<sim>_snap<S>.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

SIM_ROOT = Path("/mnt/ceph/users/camels/Sims/IllustrisTNG/L50n512/1P")
OUT = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator/cyltosph_1p")
DEFAULT_SNAP = "090"                      # z = 0 (the eRASS1 gate axis)
M500_MIN_MSUNH = 1e13                     # the BIND halo-mass floor
LOGM500_BIN_EDGES = np.array([13.0, 13.4, 13.8, 14.2, 14.6, 15.2])  # gate bins


def _read_groups(sim_dir: Path, snap: str):
    pos, r500, m500, z, box = [], [], [], None, None
    for p in sorted((sim_dir / f"groups_{snap}").glob("*.hdf5")):
        with h5py.File(p, "r") as f:
            if z is None:
                z = float(f["Header"].attrs["Redshift"])
                box = float(f["Header"].attrs["BoxSize"])          # ckpc/h
            if "Group" in f and "GroupPos" in f["Group"]:
                pos.append(np.asarray(f["Group"]["GroupPos"], np.float64))
                r500.append(np.asarray(f["Group"]["Group_R_Crit500"], np.float64))
                m500.append(np.asarray(f["Group"]["Group_M_Crit500"], np.float64) * 1e10)
    return np.concatenate(pos), np.concatenate(r500), np.concatenate(m500), z, box


def measure(sim: str, snap: str = DEFAULT_SNAP) -> dict:
    sim_dir = SIM_ROOT / sim
    pos, r500, m500, z, box = _read_groups(sim_dir, snap)
    sel = np.where((m500 >= M500_MIN_MSUNH) & (r500 > 0))[0]
    if len(sel) == 0:
        raise SystemExit(f"{sim}: no halos above {M500_MIN_MSUNH:.0e} Msun/h")
    c = np.mod(pos[sel], box)                                       # (H, 3)
    r = r500[sel]

    m_sph = np.zeros(len(sel))
    m_cyl = np.zeros(len(sel))
    chunks = sorted((sim_dir / f"snapdir_{snap}").glob(f"snapshot_{snap}.*.hdf5"))
    if not chunks:
        raise SystemExit(f"{sim}: no snapshot chunks under snapdir_{snap}")
    for p in chunks:
        with h5py.File(p, "r") as f:
            if "PartType0" not in f:
                continue
            gp = np.mod(np.asarray(f["PartType0"]["Coordinates"], np.float64), box)
            gm = np.asarray(f["PartType0"]["Masses"], np.float64) * 1e10
        for j in range(len(sel)):
            d = gp - c[j]
            d -= box * np.round(d / box)                            # periodic
            d2_xy = d[:, 0] ** 2 + d[:, 1] ** 2
            in_cyl = d2_xy < r[j] ** 2
            m_cyl[j] += gm[in_cyl].sum()
            in_sph = in_cyl & (d2_xy + d[:, 2] ** 2 < r[j] ** 2)
            m_sph[j] += gm[in_sph].sum()

    ratio = m_sph / np.maximum(m_cyl, 1e-30)
    logm = np.log10(m500[sel])
    med = np.full(len(LOGM500_BIN_EDGES) - 1, np.nan)
    nbin = np.zeros(len(LOGM500_BIN_EDGES) - 1, int)
    for b in range(len(med)):
        s = (logm >= LOGM500_BIN_EDGES[b]) & (logm < LOGM500_BIN_EDGES[b + 1])
        nbin[b] = s.sum()
        if nbin[b] >= 3:
            med[b] = np.median(ratio[s])

    return {
        "sim": sim, "snap": snap, "z": z, "box_ckpch": box,
        "m500_msunh": m500[sel], "r500_ckpch": r,
        "mgas_sph_msunh": m_sph, "mgas_cyl_msunh": m_cyl,
        "cyltosph_per_halo": ratio,
        "logm500_bin_edges": LOGM500_BIN_EDGES,
        "cyltosph_median_by_bin": med, "n_by_bin": nbin,
        "depth_note": ("cylinder depth = one 50 Mpc/h box (painted slab: 51.25); "
                       "fiducial-normalized trends vs theta are the deliverable"),
    }


def main() -> None:
    sim = sys.argv[1]
    snap = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SNAP
    out = OUT / f"{sim}_snap{snap}.npz"
    if out.exists():
        print(f"{out.name}: exists, skipping")
        return
    r = measure(sim, snap)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **r)
    ok = np.isfinite(r["cyltosph_median_by_bin"])
    print(f"{sim}: {len(r['m500_msunh'])} halos >= 1e13, medians "
          f"{np.round(r['cyltosph_median_by_bin'][ok], 3)} in bins {np.nonzero(ok)[0]}, "
          f"n_by_bin {r['n_by_bin']}")


if __name__ == "__main__":
    main()
