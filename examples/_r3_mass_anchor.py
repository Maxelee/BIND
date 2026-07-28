#!/usr/bin/env python
"""R3 (docs/p4c_referee_hardening_plan.md): mass-anchor uncertainty template.

The logM=13.18 anchor (Sailer+24) carries sigma_logM ~ 0.1 dex. A shifted
anchor moves BOTH sides coherently:
  data:  the xb apertures rescale (theta200 ∝ M^(1/3)) — computable from
         the measured xb curve by interpolation at xb' = xb * (theta200
         ratio), no re-measurement;
  model: the mock population's mass window shifts — two extra fiducial
         stacks at delta = ∓0.1 dex (same satellite mix f_eff=0.08, same
         kappa-calibrated shift), fractional response assumed node-uniform
         (the mass scaling of y is far more node-uniform than the satellite
         boost; stated assumption).
The net residual template T_mass = d(model−data)/dlogM * sigma_logM enters
the covariance as a fully-correlated outer product. Writes
R3_mass_template.npz (consumed by the R5 budget assembly).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")

# Products root (Round-2 T3, docs/paper_improvement_plan.md): env-var
# override only (straight-line script, no argparse); behavior with
# $BIND_KSZ_PRODUCTS unset is byte-identical to before.
KS = Path(os.environ.get("BIND_KSZ_PRODUCTS", "/mnt/home/mlee1/ceph/bind_science/ksz_confront"))
LC = KS / "lightcone"
SIGMA_LOGM = 0.1
F_EFF = 0.08


def main():
    from lightcone_hod_stack import (_context, _population, _theta_grid,
                                     stream_cube, FID_Y, CONFIG_NPZ, XB, BEAM)
    from lightcone_cap_stack import apply_beam, compute_stack

    cfg = np.load(CONFIG_NPZ, allow_pickle=True)
    d08 = float(cfg["deltas"][list(cfg["f_grid"]).index(F_EFF)])
    ctx = _context()
    cube = apply_beam(stream_cube(FID_Y, "y"), BEAM)

    A = 0.29296875 ** 2
    stacks = {}
    for tag, dshift in [("m_minus", +SIGMA_LOGM), ("m_center", 0.0),
                        ("m_plus", -SIGMA_LOGM)]:
        # anchor UP means the window moves UP: delta_total = d08 - shift_up
        pi, pj, pp = _population(ctx, F_EFF, d08 + dshift)
        sr = compute_stack(cube, ctx["geom"], pi, pj, pp,
                           _theta_grid(ctx, len(pi)))[0]
        stacks[tag] = np.nanmean(sr, axis=0) * A
        print(f"[R3] {tag}: done", flush=True)

    # model-side fractional response per 0.1 dex (window down = anchor up)
    dmodel_frac = (stacks["m_plus"] - stacks["m_minus"]) / (2 * stacks["m_center"])

    # data-side: aperture rescale theta200 ∝ M^(1/3)
    dd = np.load(LC / "act_ycap_lrg_real.npz", allow_pickle=True)
    d_xb = dd["mean_xb_cib17"]
    fac = (10 ** SIGMA_LOGM) ** (1.0 / 3.0)
    d_up = np.interp(XB * fac, XB, d_xb)
    d_dn = np.interp(XB / fac, XB, d_xb)
    ddata = (d_up - d_dn) / 2.0

    np.savez(LC / "R3_mass_template.npz", xb=XB,
             dmodel_frac_per_sigma=dmodel_frac, ddata_per_sigma=ddata,
             sigma_logM=SIGMA_LOGM,
             m_center=stacks["m_center"], m_plus=stacks["m_plus"],
             m_minus=stacks["m_minus"])
    print("dmodel frac per +0.1dex:", np.round(dmodel_frac[1:8], 3))
    print("ddata (y*arcmin2 1e6) per +0.1dex:", np.round(ddata[1:8] * 1e6, 3))


if __name__ == "__main__":
    main()
