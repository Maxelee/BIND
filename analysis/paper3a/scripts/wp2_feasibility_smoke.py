"""WP-A2 end-to-end feasibility smoke: SB35 painted patches -> gas composite ->
large cutouts -> stacked CAP T_kSZ profile at the frozen Qu et al. 2026 radii.

Provenance script for the numbers quoted in the WP-A2 REPORT (private plans
repo). Run from the repo root on rusty (torch3 venv, CPU, ~2 min):

    python analysis/paper3a/scripts/wp2_feasibility_smoke.py

A plumbing proof for the A3 gate, not a validation: it demonstrates that the
composite-cutout path required by the patch-size finding (CAP annuli at LRG
redshifts exceed the 6.25 h^-1 Mpc core patch) works on the real bundle.
"""
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from bind.inference.pipeline import (  # noqa: E402
    circular_taper_weight,
    extract_periodic_cutout,
    paste_halos_2d,
    square_taper_weight,
)
from analysis.paper3a.observables import (  # noqa: E402
    FlatLCDM,
    KSZOperatorConfig,
    PatchGeometry,
    ksz_cap_profile,
    minimum_cutout_extent_hmpc,
    stack_profiles,
)
from analysis.paper3a.data_vectors.data_vectors import load_ksz_qu2026_lrg_fiducial  # noqa: E402

SLAB = "/mnt/ceph/users/mlee1/bind_sb35/runs/run_0000/snap_063/composite_slab00.npz"
Z_SNAP = 0.5985432881875667  # stage1_manifest.json, snap 063
BOX, NPIX, PATCH_PIX = 205.0, 4198, 128
PIX_MPCH = BOX / NPIX  # 0.01% off the manifest's 6.25/128 (4198 = round(4198.4)); negligible

d = np.load(SLAB)
patches = d["generated_patches"].astype(np.float32).copy()  # (N, 3, 128, 128)
centers = d["halo_centers"]
r200 = d["halo_r200"]
masses = d["halo_masses"]
cond_sums = d["condition_sums"]
n = len(masses)

# mass-match each patch to its DMO condition sum (build_bind_composite convention)
for i in range(n):
    patches[i] *= cond_sums[i] / (patches[i].sum() + 1e-30)

halos = [{"halo_center": centers[i], "r200": float(r200[i])} for i in range(n)]
ppm = NPIX / BOX
weights_list = [
    circular_taper_weight(PATCH_PIX, r_pix=float(r200[i]) * ppm * 4.0, taper_frac=0.15)
    for i in range(n)
]
sq = square_taper_weight(PATCH_PIX, taper_frac=0.15)
hydro_canvas, w_accum = paste_halos_2d(NPIX, BOX, halos, patches, sq, weights_list=weights_list)
alpha = np.clip(w_accum, 0.0, 1.0)
gas_map = alpha * hydro_canvas[1]  # gas comes only from painted patches
print(f"gas composite: {gas_map.shape}, coverage {100 * (alpha > 0.01).mean():.1f}%, "
      f"total gas {gas_map.sum():.3e} Msun/h")

geom = PatchGeometry(pixel_mpch=PIX_MPCH, z=Z_SNAP, cosmology=FlatLCDM())
dv = load_ksz_qu2026_lrg_fiducial()
cfg = KSZOperatorConfig(radii_arcmin=dv.bins, beam_fwhm_arcmin=1.6, z_eff=Z_SNAP,
                        v_rms_over_c=1.06e-3)
print(f"min cutout extent needed: {minimum_cutout_extent_hmpc(cfg, geom):.1f} Mpc/h "
      f"(core patch is 6.25); arcmin/px = {geom.arcmin_per_pixel():.4f}")

cut = 361  # px = 17.6 Mpc/h, comfortably above the minimum extent
sel = np.where((masses >= 10**13.3) & (masses < 10**13.6))[0][:60]
print(f"stacking {len(sel)} halos with log10 M200c in [13.3, 13.6)")

profiles = []
for i in sel:
    cx = int(centers[i][0] * ppm) % NPIX
    cy = int(centers[i][1] * ppm) % NPIX
    profiles.append(ksz_cap_profile(extract_periodic_cutout(gas_map, cx, cy, cut), geom, cfg))
stack = stack_profiles(np.array(profiles))

print(f"\nstacked T_kSZ(theta_d) [muK arcmin^2] over {len(sel)} halos:")
for r, v in zip(dv.bins, stack):
    print(f"  theta_d = {r:5.2f}'  T_kSZ = {v:8.4f}")
print("\nQu 2026 measured values on the same grid (all-mass fiducial, scale reference only):")
print(" ", np.array2string(dv.values, precision=3))
