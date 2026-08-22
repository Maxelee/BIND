"""Test the retransformation-bias (Jensen's-gap) hypothesis for the gas amplitude excess.

_denormalize_to_physical (src/bind/inference/pipeline.py) applies mass = 10**x - 1
per pixel (x = normalized-space model output, de-standardized). If the model's
per-pixel log10(1+gas) residual (fid - truth, in normalized/physical log space) is
~zero-mean but has nonzero VARIANCE that grows toward the halo centre (where the
gradient/dynamic range is steepest and hardest to match sample-by-sample), then the
convexity of x -> 10**x - 1 alone would produce a positive bias in the LINEAR sum
that (a) grows with local variance (hence with radius, hence "core excess"), and
(b) is present with zero log-space mean bias.

This script computes, pixel-wise (same location grid, same 2933 halos), the
log10(1+gas) residual delta = log10(1+gas_fid) - log10(1+gas_truth) in radial
shells: its MEAN (log-space bias) and STD (log-space scatter), plus the
Jensen's-gap prediction exp(ln10 * mean_delta + 0.5*(ln10*std_delta)^2) for the
*linear*-space ratio it would produce under a log-normal-residual approximation,
to check whether it roughly reproduces the measured linear ratio (a positive
control for the retransformation-bias mechanism).
"""
import numpy as np
import gc

FID_DIR = "/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096"
TRUTH_DIR = "/mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000/snap_096"

PIX = 6.25 / 128.0
P = 128
cen = P // 2
yy, xx = np.mgrid[0:P, 0:P]
rr = np.hypot(xx - cen, yy - cen) * PIX
r_edges = np.array([0.0, 0.1, 0.2, 0.3, 0.44, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.13])
nb = len(r_edges) - 1
bidx = np.digitize(rr.ravel(), r_edges) - 1

sum_d = np.zeros(nb); sum_d2 = np.zeros(nb); cnt = np.zeros(nb)
sum_lin_f = np.zeros(nb); sum_lin_t = np.zeros(nb)

for si in range(4):
    f = np.load(f"{FID_DIR}/composite_slab{si:02d}.npz", allow_pickle=True)
    t = np.load(f"{TRUTH_DIR}/composite_slab{si:02d}.npz", allow_pickle=True)
    n = int(f["n_halos"])
    if n == 0:
        continue
    gf = f["generated_patches"][:, 1].astype(np.float64).reshape(n, -1)
    gt = t["generated_patches"][:, 1].astype(np.float64).reshape(n, -1)
    lf = np.log10(1.0 + np.clip(gf, 0, None))
    lt = np.log10(1.0 + np.clip(gt, 0, None))
    delta = lf - lt  # (n, 16384)
    for b in range(nb):
        sel = bidx == b
        d = delta[:, sel]
        sum_d[b] += d.sum()
        sum_d2[b] += (d ** 2).sum()
        cnt[b] += d.size
        sum_lin_f[b] += gf[:, sel].sum()
        sum_lin_t[b] += gt[:, sel].sum()
    del f, t, gf, gt, lf, lt, delta
    gc.collect()

ln10 = np.log(10.0)
print(f"{'r_lo':>5s} {'r_hi':>5s} {'mean_dlog':>10s} {'std_dlog':>9s} "
      f"{'jensen_pred':>11s} {'measured_lin':>12s}")
for b in range(nb):
    mu = sum_d[b] / cnt[b]
    var = sum_d2[b] / cnt[b] - mu ** 2
    std = np.sqrt(max(var, 0))
    # lognormal-ish prediction for E[10^X]/E[10^Y] with X=Y+delta, delta~N(mu,var):
    # this predicts the ratio of MEANS of 10^delta type multiplicative factor
    # under the crude approximation that delta is ~Gaussian & pixel means dominate the +1:
    jensen = np.exp(ln10 * mu + 0.5 * (ln10 * std) ** 2)
    meas = sum_lin_f[b] / sum_lin_t[b]
    print(f"{r_edges[b]:5.2f} {r_edges[b+1]:5.2f} {mu:10.5f} {std:9.5f} {jensen:11.4f} {meas:12.4f}")
