#!/usr/bin/env python
"""shell_straddle.py -- shell/slab-straddling halo fraction for the BIND lightcone.

TODO stretch item (feeds paper Sec 5 caveat 10, "shell-straddling"). No such
bookkeeping existed anywhere in the repo before this script. BIND paints the
lightcone per-slab: each snapshot's simulation box (205 Mpc/h) is cut into
n_slabs=4 non-overlapping slab_depth=box_size/4=51.25 Mpc/h deep slices along
a per-snapshot randomly-drawn line-of-sight (LOS) axis
(`bind.inference.lightcone_transforms.LightconeTransforms`), and each halo's
condition/patch is extracted from ONLY the slab containing its (transformed)
center (`bind.inference.paint._assign_halos_to_slabs`,
`bind.inference.truth_lightcone.extract_truth_halos`). Particles are hard-cut
into slabs by exact LOS coordinate (`_project_zslabs`: `(z >= lo) & (z < hi)`),
so a halo whose 3D extent straddles a slab cut has part of its real mass
literally projected into the WRONG slab's density/condition map -- the
network sees a truncated halo. This script quantifies how often that happens.

Two physically distinct boundary types are reported separately:
  - "straddle" (any of the n_slabs-1 = 3 cuts *within* one snapshot's single,
    continuous 205 Mpc/h box): mass is real, just mis-binned between two
    slabs of the SAME snapshot/realization.
  - "shell" (snapshot) boundary -- the box's own periodic seam at
    z=0/box_size: each snapshot's box is used exactly ONCE per 205 Mpc/h
    lightcone shell (its own random disp/flip/proj_dir draw); slab 3 of
    snapshot k is immediately followed in comoving distance by slab 0 of the
    DIFFERENT snapshot k+1 (its own, unrelated realization), not by slab 0 of
    snapshot k's own periodic image. A halo straddling z=0/box_size is
    therefore split across two entirely unrelated realizations/redshifts --
    the physically worse case. Its count is a SUBSET of "straddle".

Halo LOS positions are not cached anywhere on disk: truth_lightcone.py /
paint.py keep only the transverse (x, y) `halo_centers` in
`composite_slab*.npz` (see stage1/stage1_slabNN.npz: halo_centers is (N,2));
the z-coordinate used by `_assign_halos_to_slabs` is computed on the fly and
discarded. This script recomputes it directly: for each of the 20 lightcone
snapshots it reads the raw DMO FoF catalog (GroupPos / Group_M_Crit200 /
Group_R_Crit200) named in that snapshot's `stage1/stage1_manifest.json`, and
re-applies the SAME `LightconeTransforms` (`bind_lightcone_tng/
lightcone_transforms.json`) used to build the lightcone -- exactly mirroring
`truth_lightcone.extract_truth_halos`'s halo-assignment block.

Runtime: a catalog cross-reference only (GroupPos/mass/r200 columns of ~80
FoF sub-files x 20 snapshots), no map arrays are touched. ~1-2 min.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bind.inference.io_gadget import read_fof_catalog
from bind.inference.lightcone_transforms import LightconeTransforms

LIGHTCONE = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng")
TRANSFORMS_JSON = LIGHTCONE / "lightcone_transforms.json"
MASS_CUT = 1e13   # M200c, Msun/h -- "painted halos" (matches stage1 halo_mass_min)

OUT_DIR = Path("/mnt/home/mlee1/BIND/papers/01_pipeline/audits")
OUT_JSON = OUT_DIR / "shell_straddle_data.json"


def main():
    transforms = LightconeTransforms.from_dict(json.loads(TRANSFORMS_JSON.read_text()))

    snap_dirs = sorted(LIGHTCONE.glob("snap_*"))
    per_snap = []
    all_straddle, all_shell, all_mass = [], [], []

    print(f"{'snapshot':>10s} {'z':>6s} {'N(M>1e13)':>10s} {'straddle':>9s} "
          f"{'frac':>7s} {'mass-wt':>8s} {'shell':>6s} {'frac':>7s} {'mass-wt':>8s}")
    for d in snap_dirs:
        mf = d / "stage1" / "stage1_manifest.json"
        if not mf.exists():
            print(f"  [skip] {d.name}: no stage1_manifest.json")
            continue
        man = json.loads(mf.read_text())
        box_size = float(man["box_size"])
        n_slabs = int(man["n_slabs"])
        slab_depth = box_size / n_slabs             # 205/4 = 51.25 Mpc/h (native)
        snap_idx = int(man["snapshot_index"])
        t_idx = int(man["transforms_snap_idx"])
        group_catalog = man["group_catalog"]
        mass_field = man.get("halo_mass_field", "Group_M_Crit200")

        cat = read_fof_catalog(group_catalog, snapshot=snap_idx,
                                halo_mass_min=MASS_CUT, mass_field=mass_field)
        pos, mass, r200 = cat["positions"], cat["mass"], cat["r200"]   # Mpc/h, Msun/h, Mpc/h
        n = len(mass)

        tpos = transforms.apply(pos, t_idx, box_size)   # cols: x,y transverse; z = LOS
        z = tpos[:, 2].astype(np.float64)

        z_lo, z_hi = z - r200, z + r200
        # slab index of each edge; z_lo/z_hi are allowed to fall outside
        # [0, box_size) -- that excursion IS the shell-boundary signal below.
        slab_lo = np.floor(z_lo / slab_depth).astype(np.int64)
        slab_hi = np.floor(z_hi / slab_depth).astype(np.int64)

        crosses_any = slab_lo != slab_hi                       # any of the 3 internal cuts, or the shell seam
        crosses_shell = (z_lo < 0.0) | (z_hi >= box_size)       # wraps past the box's own edge -> next snapshot

        n_straddle, n_shell = int(crosses_any.sum()), int(crosses_shell.sum())
        frac_straddle = n_straddle / n if n else float("nan")
        frac_shell = n_shell / n if n else float("nan")
        mtot = mass.sum()
        mass_frac_straddle = float(mass[crosses_any].sum() / mtot) if n else float("nan")
        mass_frac_shell = float(mass[crosses_shell].sum() / mtot) if n else float("nan")

        per_snap.append(dict(
            snap=d.name, snapshot_index=snap_idx, redshift=man["redshift"],
            box_size=box_size, n_slabs=n_slabs, slab_depth=slab_depth,
            n_halos=n, n_straddle=n_straddle, frac_straddle=frac_straddle,
            mass_frac_straddle=mass_frac_straddle,
            n_shell=n_shell, frac_shell=frac_shell, mass_frac_shell=mass_frac_shell,
        ))
        all_straddle.append(crosses_any); all_shell.append(crosses_shell); all_mass.append(mass)
        print(f"{d.name:>10s} {man['redshift']:6.3f} {n:10d} {n_straddle:9d} "
              f"{frac_straddle:7.2%} {mass_frac_straddle:8.2%} {n_shell:6d} "
              f"{frac_shell:7.2%} {mass_frac_shell:8.2%}")

    straddle_cat = np.concatenate(all_straddle)
    shell_cat = np.concatenate(all_shell)
    mass_cat = np.concatenate(all_mass)
    n_tot = len(mass_cat)

    overall = dict(
        n_halos_total=n_tot,
        n_straddle_total=int(straddle_cat.sum()),
        frac_straddle_overall=float(straddle_cat.sum() / n_tot),
        mass_frac_straddle_overall=float(mass_cat[straddle_cat].sum() / mass_cat.sum()),
        n_shell_total=int(shell_cat.sum()),
        frac_shell_overall=float(shell_cat.sum() / n_tot),
        mass_frac_shell_overall=float(mass_cat[shell_cat].sum() / mass_cat.sum()),
        n_internal_only=int((straddle_cat & ~shell_cat).sum()),
        frac_internal_only=float((straddle_cat & ~shell_cat).sum() / n_tot),
    )
    print("\n=== OVERALL (20 snapshots, 80 slabs total, M200c > 1e13 Msun/h) ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    OUT_DIR.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(mass_cut=MASS_CUT, per_snapshot=per_snap,
                                         overall=overall), indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
