#!/usr/bin/env python
"""Build thermo_cv_fd_fiducial.npz (Ffid_Y200, Ffid_SX) from the pre-generated
fm_thermo EMA fiducial fields — no GPU run needed.

The CV fiducial generation already exists at
  CV/sim_i/snap_090/mass_threshold_1p000e13/fm_thermo_ema/generated_halos.npz
  -> 'generated' (n_i, 7, 128, 128) = [DM,Gas,Stars,Y,T,S,P].
We iterate the SAME sorted sim dirs that fd_jacobian_thermo.load_cv_halos uses
(require halo_catalog.npz + halo_cutouts.npz), compute the tSZ Y200 + X-ray SX
aperture observables with the SAME aperture convention as the FD, concatenate, and
verify the halo masses match the merged FD Jacobian. The result feeds synthesis §D
(local-FD log-Jacobian dlnY/dtheta = J / Ffid).

NB: these EMA fiducials pair with the non-EMA FD Jacobian -> a small per-observable
EMA/non-EMA normalisation offset (a constant factor, ~few %), negligible for the
relative Fisher reduction. For the exact non-EMA fiducial use
`fd_jacobian_thermo.py --fiducial_only` instead.

Run:  python tools/build_thermo_fiducial.py
"""
from pathlib import Path
import numpy as np

CV = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
AC = Path("analysis_physics_cache")
SNAP, MTAG = "snap_090", "mass_threshold_1p000e13"
MPC_PER_PIX = 50.0 / 1024.0                         # = BOX / N_PIX_FULL
RHO_CRIT = 2.775e11                                 # M_sun/h per (Mpc/h)^3
PATCH = 128


def r200c_mpc_h(m200):                              # FD-loader fallback when 'radii' absent
    return (3.0 * m200 / (4.0 * np.pi * 200.0 * RHO_CRIT)) ** (1.0 / 3.0)
_yy, _xx = np.mgrid[:PATCH, :PATCH] - PATCH / 2     # centred at index 64 (FD convention)
_RR_PIX = np.sqrt(_xx ** 2 + _yy ** 2)


def thermo_obs(phys7, r200_pix):
    r_aper = max(min(r200_pix, PATCH / 2 - 2), 4.0)
    m = _RR_PIX < r_aper
    gas, Y, T = phys7[1], phys7[3], phys7[4]
    sx = np.maximum(gas, 0.0) ** 2 * np.sqrt(np.clip(T, 0.0, None))
    return float(np.maximum(Y, 0.0)[m].sum()), float(sx[m].sum())


def main():
    Y, SX, masses, sids = [], [], [], []
    for d in sorted(p for p in CV.iterdir() if p.is_dir()):
        md = d / SNAP / MTAG
        cat_p, cut_p = md / "halo_catalog.npz", md / "halo_cutouts.npz"
        gen_p = md / "fm_thermo_ema" / "generated_halos.npz"
        if not (cat_p.exists() and cut_p.exists()):          # match FD loader gate
            continue
        if not gen_p.exists():
            raise FileNotFoundError(f"{d.name}: FD-loader sim lacks fm_thermo_ema generation")
        cat = np.load(cat_p)
        gen = np.load(gen_p)["generated"]                    # (n,7,128,128)
        m = np.asarray(cat["masses"], float)
        if "radii" in cat.files:                             # match FD loader exactly
            radii_pix = np.asarray(cat["radii"], float) / 1000.0 / MPC_PER_PIX
        else:
            radii_pix = r200c_mpc_h(m) / MPC_PER_PIX
        assert len(gen) == len(m), f"{d.name}: {len(gen)} gen vs {len(m)} halos"
        for i in range(len(m)):
            y, sx = thermo_obs(gen[i].astype(np.float64), radii_pix[i])
            Y.append(y); SX.append(sx)
        masses.append(m); sids.append(np.full(len(m), d.name))
    Y = np.array(Y); SX = np.array(SX)
    masses = np.concatenate(masses); sids = np.concatenate(sids)
    print(f"computed Ffid for {len(Y)} halos over {len(np.unique(sids))} sims")

    # verify alignment with the merged FD Jacobian
    fd_p = AC / "thermo_cv_fd_fm_thermo.npz"
    if fd_p.exists():
        fd = np.load(fd_p, allow_pickle=True)
        ok = (len(masses) == len(fd["masses_use"])) and np.allclose(masses, fd["masses_use"])
        print(f"halo alignment with FD Jacobian: {'OK' if ok else 'MISMATCH'}"
              f" ({len(masses)} vs {len(fd['masses_use'])})")
        if not ok:
            raise SystemExit("masses do not match FD masses_use — ordering differs, aborting")

    out = AC / "thermo_cv_fd_fiducial.npz"
    np.savez_compressed(out, Ffid_Y200=Y.astype(np.float32), Ffid_SX=SX.astype(np.float32),
                        masses_use=masses, sim_id_use=sids)
    print("wrote", out, "| Ffid_Y200>0:", bool(np.all(Y > 0)), "Ffid_SX>0:", bool(np.all(SX > 0)))


if __name__ == "__main__":
    main()
