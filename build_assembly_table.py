"""build_assembly_table.py — per-halo 3D assembly history for the Sobol-cube CV halos.

Project 1 (assembly-dependent feedback susceptibility) needs each cube halo's *assembly*
history attached. The structure is baryon-free and design-independent, read from the CAMELS
DMO N-body Subfind catalogs (same physics as `scatter/assembly_3d.py`, `analysis/2d`):

  c_V      = Vmax / V200            concentration proxy / formation-time tracer (Prada+2012)
  lambda   = Bullock spin           |j| / (sqrt(2) V200 R200)
  veldisp  = subhalo vel. dispersion
  rhalf    = SubhaloHalfmassRad / R200
  z_form   = z at which the main progenitor first reached 0.5 M200(z=0)  (tree-free trace)

Rows are aligned 1:1 with `cube.npz` (same `load_cv_halos()` ordering: sorted CV sim dirs,
all halos per sim in catalog order). Each cube halo is matched to its DMO group by position
(periodic kd-match); unmatched halos get NaN. We assert the hydro M200 column equals the cube's.

Run (no GPU; z_form tracing reads ~56 snapshots/sim, a few minutes):
    source /mnt/home/mlee1/venvs/torch3/bin/activate
    python build_assembly_table.py                 # -> <OUT_ROOT>/assembly_table.npz
    python build_assembly_table.py --no-formation  # skip the slow z_form trace
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sobol_ss_generation import CV_ROOT, SNAP, MASS_TAG, OUT_ROOT_DEFAULT
from scatter.assembly_3d import dmo_assembly, formation_redshift, _match, DMO_ROOT

ASM_KEYS = ["c_V", "lambda", "veldisp", "rhalf"]      # from the z=0 subfind catalog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m200_min", type=float, default=5e12,
                    help="DMO group mass floor for the assembly catalog (Msun/h)")
    ap.add_argument("--match_tol_kpc", type=float, default=300.0)
    ap.add_argument("--form_tol_kpc", type=float, default=400.0)
    ap.add_argument("--snap_min", type=int, default=33)
    ap.add_argument("--no-formation", dest="no_formation", action="store_true")
    args = ap.parse_args()

    cube = np.load(OUT_ROOT_DEFAULT / "cube.npz", allow_pickle=True)
    cube_M200 = np.asarray(cube["M200"], float)
    cube_sid = [str(s) for s in cube["sim_id"]]
    n_total = len(cube_M200)

    cols = {k: np.full(n_total, np.nan) for k in ASM_KEYS}
    cols["z_form"] = np.full(n_total, np.nan)
    matched = np.zeros(n_total, bool)

    off = 0
    for sd in sorted(p for p in CV_ROOT.iterdir() if p.is_dir()):
        # block of cube rows belonging to this sim (contiguous, in catalog order)
        idx = [k for k in range(off, n_total) if cube_sid[k] == sd.name]
        if not idx:
            continue
        block = np.array(idx)
        assert (np.diff(block) == 1).all(), f"{sd.name}: cube rows not contiguous"
        off = block[-1] + 1

        cat = np.load(sd / SNAP / MASS_TAG / "halo_catalog.npz", allow_pickle=True)
        pos_mpc = np.asarray(cat["halo_positions"], float)        # (n_sim, 3) Mpc/h
        masses = np.asarray(cat["masses"], float)
        assert len(masses) == len(block), f"{sd.name}: catalog/cube count mismatch"
        assert np.allclose(masses, cube_M200[block]), f"{sd.name}: M200 misaligned with cube"

        cv_i = int(sd.name.split("_")[1])
        dmo = dmo_assembly(f"{DMO_ROOT}/CV_{cv_i}/fof_subhalo_tab_090.hdf5", args.m200_min)
        if not dmo:
            print(f"  {sd.name}: DMO catalog missing — leaving NaN")
            continue

        j = _match(pos_mpc * 1000.0, dmo["pos"], args.match_tol_kpc)   # (n_sim,) dmo idx or -1
        ok = j >= 0
        for k in ASM_KEYS:
            cols[k][block[ok]] = dmo[k][j[ok]]
        matched[block[ok]] = True

        if not args.no_formation and ok.any():
            mj = j[ok]                                              # matched dmo indices
            zf = formation_redshift(f"{DMO_ROOT}/CV_{cv_i}", dmo["pos"][mj], dmo["M200"][mj],
                                    snap_min=args.snap_min, tol_kpc=args.form_tol_kpc)
            cols["z_form"][block[ok]] = zf
        print(f"  {sd.name}: matched {ok.sum()}/{len(block)}"
              + ("" if args.no_formation else
                 f"  z_form med={np.nanmedian(cols['z_form'][block]):.2f}"))

    feat_names = ASM_KEYS + ["z_form"]
    feats = np.column_stack([cols[k] for k in feat_names])
    out = OUT_ROOT_DEFAULT / "assembly_table.npz"
    np.savez(out, feats=feats, feat_names=np.array(feat_names, dtype=object),
             matched=matched, M200=cube_M200, sim_id=np.array(cube_sid, dtype=object))
    print(f"\nwrote {out}   ({matched.sum()}/{n_total} halos matched to DMO assembly)")
    for j2, name in enumerate(feat_names):
        v = feats[:, j2]
        print(f"  {name:9s} median={np.nanmedian(v):+.3f}  "
              f"[{np.nanpercentile(v, 16):+.3f}, {np.nanpercentile(v, 84):+.3f}]  "
              f"({np.isfinite(v).sum()}/{n_total} finite)")


if __name__ == "__main__":
    main()
