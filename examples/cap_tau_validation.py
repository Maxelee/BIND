"""Is CAP-on-halos still valid with the diffuse-tau maps?

Stacks Compensated Aperture Photometry (disk r<theta minus equal-area ring
theta<r<sqrt2*theta -- the exact convention of examples/_reduce_fgas_lightcone.py
::cap_on_map) on tau at halo positions, for three constructions of the SAME
lightcone (identical seeds, identical positions):

  pasted  = old-tree truth tau (gas only inside paste apertures)
  diffuse = campaign tau (pasted + f_b*DMO*(1-alpha) at T=1e4K)
  full    = full-hydro TNG300 trace  <- GROUND TRUTH

Halo positions are kappa peaks above nu_min (the proxy the pipeline already uses
for its cross-stacks); peaks are found once, in the pasted kappa map, and reused
for all three tau maps so selection is identical by construction.
"""
import zipfile

import numpy as np
import numpy.lib.format as fmt
from scipy.ndimage import gaussian_filter, maximum_filter

VAL = "/mnt/home/mlee1/ceph/tng_full_validation/runs"
SCI = "/mnt/home/mlee1/ceph/bind_science/runs"
FOV_DEG, NPIX = 5.0, 1024
DTHETA = FOV_DEG * 60.0 / NPIX          # arcmin per pixel (0.293')
APERTURES = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
N_REAL, ZI, NU_MIN = 12, 4, 3.0         # z_s = 2.44 (deepest tau plane)


def load(npz, key, n, zi):
    with zipfile.ZipFile(npz) as z, z.open(key + ".npy") as f:
        v = fmt.read_magic(f)
        s, _, dt = (fmt.read_array_header_1_0(f) if v == (1, 0)
                    else fmt.read_array_header_2_0(f))
        per = int(np.prod(s[1:])) * dt.itemsize
        out = np.empty((n, s[2], s[3]), np.float32)
        for i in range(n):
            out[i] = np.frombuffer(f.read(per), dtype=dt).reshape(s[1:])[zi]
        return out


def cap_stamps(theta_pix):
    """(disk, ring, weight) boolean stamps for one aperture."""
    w = int(np.ceil(np.sqrt(2) * theta_pix)) + 2
    yy, xx = np.mgrid[-w:w + 1, -w:w + 1]
    r = np.hypot(xx, yy)
    disk = r < theta_pix
    ring = (r >= theta_pix) & (r < np.sqrt(2) * theta_pix)
    return disk, ring, disk.sum() / ring.sum(), w


kap = load(f"{SCI}/truth/run_0000/kappa_maps.npz", "kappa", N_REAL, ZI)
taus = {
    "pasted": load(f"{SCI}/truth/run_0000/tau_maps.npz", "tau", N_REAL, ZI),
    "diffuse": load(f"{VAL}/diffuse/run_0000/tau_maps.npz", "tau", N_REAL, ZI),
    "full": load(f"{VAL}/hydro_full/run_0000/tau_maps.npz", "tau", N_REAL, ZI),
}
stamps = [cap_stamps(a / DTHETA) for a in APERTURES]
acc = {k: np.zeros(len(APERTURES)) for k in taus}
n_used = 0

for r in range(N_REAL):
    sm = gaussian_filter(kap[r].astype(np.float64), (2.0 / DTHETA) / 2.355)
    nu = (sm - sm.mean()) / sm.std()
    pk = (maximum_filter(sm, size=3) == sm) & (nu > NU_MIN)
    ys, xs = np.nonzero(pk)
    for k, m in taus.items():
        mm = m[r].astype(np.float64)
        for ai, (disk, ring, wgt, w) in enumerate(stamps):
            tot = 0.0
            cnt = 0
            for y, x in zip(ys, xs):
                if y - w < 0 or y + w + 1 > NPIX or x - w < 0 or x + w + 1 > NPIX:
                    continue
                sub = mm[y - w:y + w + 1, x - w:x + w + 1]
                tot += sub[disk].sum() - sub[ring].sum() * wgt
                cnt += 1
            acc[k][ai] += tot
            if k == "full" and ai == 0:
                n_used += cnt
    if r == 0:
        print(f"  peaks/real above nu={NU_MIN}: {len(ys)} (usable {n_used})", flush=True)

for k in acc:
    acc[k] /= max(n_used, 1)

print(f"\n=== CAP(tau) stacked at {n_used} halo positions ({N_REAL} reals, z_s=2.44) ===")
print(f"{'aperture':>9s} {'pasted':>11s} {'diffuse':>11s} {'full(truth)':>12s} "
      f"{'pasted/full':>12s} {'diffuse/full':>13s}")
for i, a in enumerate(APERTURES):
    print(f"{a:7.1f}'  {acc['pasted'][i]:11.4e} {acc['diffuse'][i]:11.4e} "
          f"{acc['full'][i]:12.4e} {acc['pasted'][i]/acc['full'][i]:12.3f} "
          f"{acc['diffuse'][i]/acc['full'][i]:13.3f}")
