#!/usr/bin/env python
"""Thermo gas observables for the FD-Jacobian WL chain (extends fd_jacobian_cv.py).

These are the NEW per-halo observables needed to put the gas->S(k) Fisher chain on
the same rigorous local finite-difference footing as the mass/scaling-relation
Jacobian: the tSZ aperture signal Y200 and the X-ray surface-brightness proxy SX,
computed from the 7-channel physical map [DM,Gas,Stars,Y,T,S,P] the fm_thermo model
emits.

Validated against the Sobol cube (maps/gen_design0000.npz vs cube Y200 / extra SX):
    Y200 aperture  corr = 1.0000  (overall units constant ~417 -> irrelevant: a
                                    Jacobian of log Y is invariant to it)
    SX   aperture  corr = 1.0000  ratio 0.997

Integration into a thermo copy of fd_jacobian_cv.py:
  1. point --run_dir at the fm_thermo weights (out_channels = 7);
  2. extend PER_HALO_KEYS with THERMO_KEYS and have observables_from_phys also accept
     the 7-channel map, calling thermo_observables() below;
  3. the rest (fixed-noise sampling, central differences, sharding, merge) is unchanged.

NOTE: S(k) suppression is a FULL-BOX quantity (paste patches into the 50 Mpc/h box ->
P_hydro/P_DMO), so it does NOT belong in this per-halo file. The rigorous local-FD
dS(k)/dtheta is a separate generate-at-fiducial+-Delta -> box_supp_sobol pipeline
(fm_two_head only; no thermo weights needed). See the run plan in the session notes.
"""
import numpy as np

PATCH_PIX = 128
PIX_KPC = 48.828125
_yy, _xx = np.mgrid[0:PATCH_PIX, 0:PATCH_PIX]
_RR_KPC = np.hypot(_yy - PATCH_PIX / 2 + 0.5, _xx - PATCH_PIX / 2 + 0.5) * PIX_KPC

THERMO_KEYS = ["Y200", "SX"]   # extend PER_HALO_KEYS with these in the thermo FD script


def _aperture_sum(field_2d, r200_kpc):
    """Sum of a (clipped-positive) field within the R200c circular aperture."""
    return float(np.maximum(field_2d, 0.0)[_RR_KPC <= r200_kpc].sum())


def thermo_observables(phys_7HW, r200_kpc):
    """tSZ + X-ray aperture observables from a 7-channel physical map.

    Args:
        phys_7HW : (7, 128, 128) = [DM, Gas, Stars, Y, T, S, P] in physical units.
        r200_kpc : R200c in h^-1 kpc.
    Returns dict over THERMO_KEYS. SX proxy = sum(Gas^2 * sqrt(T)) (n_e^2 sqrt(T)).
    """
    gas, Y, T = phys_7HW[1], phys_7HW[3], phys_7HW[4]
    SXmap = np.maximum(gas, 0.0) ** 2 * np.sqrt(np.clip(T, 0.0, None))
    return {"Y200": _aperture_sum(Y, r200_kpc), "SX": _aperture_sum(SXmap, r200_kpc)}


def _selftest(n=200):
    """Validate against the Sobol cube using the stored generated maps."""
    from pathlib import Path
    S = Path("/mnt/home/mlee1/ceph/sobol_ss_cv")
    cube = np.load(S / "cube.npz", allow_pickle=True); on = list(cube["obs_names"])
    ex = np.load(S / "obs_fb_extra.npz", allow_pickle=True); en = list(ex["extra_names"])
    R200 = np.asarray(cube["R200"], float)
    g = np.load(S / "maps" / "gen_design0000.npz")["generated"]
    mine = [thermo_observables(g[i], R200[i]) for i in range(n)]
    Ym = np.array([m["Y200"] for m in mine]); SXm = np.array([m["SX"] for m in mine])
    Yc = cube["obs"][0, :n, on.index("Y200")]; SXc = ex["extra"][0, :n, en.index("SX")]
    for nm, a, b in [("Y200", Ym, Yc), ("SX", SXm, SXc)]:
        ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
        print(f"  {nm}: corr={np.corrcoef(a[ok], b[ok])[0,1]:.4f}  "
              f"median(mine/cube)={np.median(a[ok]/b[ok]):.3f}")


if __name__ == "__main__":
    _selftest()
