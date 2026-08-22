"""Radial-profile decomposition of the patch-level gas/y excess (Investigation 2).

Reproduces the exact aperture geometry of examples/halo_atlas.py (branch
analysis/sobol-sb35: PIX=6.25/128 Mpc/h, r measured from patch centre) and bins
the SAME `generated_patches`/`thermo_patches` (raw model output vs raw particle
projection, no compositing) into radial shells in TWO ways:
  (a) fixed physical radius bins [Mpc/h]  (unbiased by mass-mix across bins)
  (b) r/R200c bins (self-similar stack)   (checks the excess isn't just a
      mass-mix artifact of using a fixed physical radius against a varying halo
      mass distribution)
Reports FID/TRUTH ratio of the summed mass/y in each shell.
"""
import numpy as np
import gc

FID_DIR = "/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096"
TRUTH_DIR = "/mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000/snap_096"

PIX = 6.25 / 128.0  # Mpc/h per pixel
P = 128
cen = P // 2
yy, xx = np.mgrid[0:P, 0:P]
rr_mpc = np.hypot(xx - cen, yy - cen) * PIX  # (128,128) Mpc/h from patch centre

# (a) physical bins
r_edges_phys = np.array([0.0, 0.1, 0.2, 0.3, 0.44, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.13])
n_phys = len(r_edges_phys) - 1
gas_f_phys = np.zeros(n_phys); gas_t_phys = np.zeros(n_phys)
y_f_phys = np.zeros(n_phys); y_t_phys = np.zeros(n_phys)
npix_phys = np.zeros(n_phys)

# (b) r/R200c bins (self-similar)
r_edges_norm = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
n_norm = len(r_edges_norm) - 1
gas_f_norm = np.zeros(n_norm); gas_t_norm = np.zeros(n_norm)
y_f_norm = np.zeros(n_norm); y_t_norm = np.zeros(n_norm)
npix_norm = np.zeros(n_norm)

bin_idx_phys = np.digitize(rr_mpc.ravel(), r_edges_phys) - 1  # (16384,)

n_tot = 0
for si in range(4):
    f = np.load(f"{FID_DIR}/composite_slab{si:02d}.npz", allow_pickle=True)
    t = np.load(f"{TRUTH_DIR}/composite_slab{si:02d}.npz", allow_pickle=True)
    n = int(f["n_halos"])
    if n == 0:
        continue
    assert np.allclose(f["halo_masses"], t["halo_masses"])
    gp_f = f["generated_patches"][:, 1]  # (n,128,128) gas
    gp_t = t["generated_patches"][:, 1]
    y_pf = f["thermo_patches"][:, 0]
    y_pt = t["thermo_patches"][:, 0]
    r200 = f["halo_r200"].astype(np.float64)  # (n,) Mpc/h, identical to truth (verified)
    n_tot += n

    # ---- (a) physical bins: same bin_idx for every halo ----
    for b in range(n_phys):
        sel = bin_idx_phys == b
        gas_f_phys[b] += gp_f.reshape(n, -1)[:, sel].sum()
        gas_t_phys[b] += gp_t.reshape(n, -1)[:, sel].sum()
        y_f_phys[b] += y_pf.reshape(n, -1)[:, sel].sum()
        y_t_phys[b] += y_pt.reshape(n, -1)[:, sel].sum()
        npix_phys[b] += sel.sum() * n

    # ---- (b) r/R200c bins: per-halo normalized radius ----
    rnorm = rr_mpc[None, :, :] / r200[:, None, None]  # (n,128,128)
    bidx = np.digitize(rnorm.ravel(), r_edges_norm) - 1
    gf_flat = gp_f.ravel(); gt_flat = gp_t.ravel()
    yf_flat = y_pf.ravel(); yt_flat = y_pt.ravel()
    for b in range(n_norm):
        sel = bidx == b
        gas_f_norm[b] += gf_flat[sel].sum()
        gas_t_norm[b] += gt_flat[sel].sum()
        y_f_norm[b] += yf_flat[sel].sum()
        y_t_norm[b] += yt_flat[sel].sum()
        npix_norm[b] += sel.sum()

    del f, t, gp_f, gp_t, y_pf, y_pt, rnorm, bidx
    gc.collect()

print(f"Total halos: {n_tot}\n")
print("=== (a) Fixed physical radius shells [Mpc/h] ===")
print(f"{'r_lo':>6s} {'r_hi':>6s} {'gas ratio':>10s} {'y ratio':>10s}")
for b in range(n_phys):
    gr = gas_f_phys[b] / gas_t_phys[b] if gas_t_phys[b] != 0 else np.nan
    yr = y_f_phys[b] / y_t_phys[b] if y_t_phys[b] != 0 else np.nan
    print(f"{r_edges_phys[b]:6.2f} {r_edges_phys[b+1]:6.2f} {gr:10.4f} {yr:10.4f}")

cum_gas_f = np.cumsum(gas_f_phys); cum_gas_t = np.cumsum(gas_t_phys)
print("\ncumulative (< r_hi) gas ratio:")
for b in range(n_phys):
    print(f"  r<{r_edges_phys[b+1]:.2f}: {cum_gas_f[b]/cum_gas_t[b]:.4f}")

print("\n=== (b) r/R200c shells (self-similar stack) ===")
print(f"{'x_lo':>6s} {'x_hi':>6s} {'gas ratio':>10s} {'y ratio':>10s}")
for b in range(n_norm):
    gr = gas_f_norm[b] / gas_t_norm[b] if gas_t_norm[b] != 0 else np.nan
    yr = y_f_norm[b] / y_t_norm[b] if y_t_norm[b] != 0 else np.nan
    print(f"{r_edges_norm[b]:6.2f} {r_edges_norm[b+1]:6.2f} {gr:10.4f} {yr:10.4f}")

cum_gas_f_n = np.cumsum(gas_f_norm); cum_gas_t_n = np.cumsum(gas_t_norm)
print("\ncumulative (<x r200) gas ratio:")
for b in range(n_norm):
    print(f"  x<{r_edges_norm[b+1]:.2f}: {cum_gas_f_n[b]/cum_gas_t_n[b]:.4f}")

np.savez(
    "/tmp/claude-2107/-mnt-home-mlee1-BIND/eaad8798-3159-4ecc-ad3b-687c176db4cc/scratchpad/inv2/radial_profiles.npz",
    r_edges_phys=r_edges_phys, gas_f_phys=gas_f_phys, gas_t_phys=gas_t_phys,
    y_f_phys=y_f_phys, y_t_phys=y_t_phys,
    r_edges_norm=r_edges_norm, gas_f_norm=gas_f_norm, gas_t_norm=gas_t_norm,
    y_f_norm=y_f_norm, y_t_norm=y_t_norm,
)
