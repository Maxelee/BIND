"""fig07_tsz_pressure.py

The tSZ "pressure leg" consistency check (Sec. 6d): (a) y-CAP amplitude for
all 256 Sobol nodes, the subset of 49 nodes that are chi2-consistent with
the kSZ (density) data, and the BIND fiducial run at the corrected
DESI-LRG lensing host mass, against the real ACT x DESI-LRG y-CAP data,
with a +-0.1 dex host-mass systematic band (Y ~ M^5/3) that dominates the
visual amplitude spread; (b) the same y-CAP shape, normalized at 2.25
arcmin, showing BIND's painted pressure profile is more radially extended
than the (compact) data. Panel (a) also reports the Spearman rank
correlation between the per-node kSZ chi2 and tSZ chi2 -- the same
strong-feedback nodes fit both legs.

Data (pre-cached, read-only):
  - KS/ycap_lrg_snap067.npz
      keys: R, fiducial, fid_msys_lo, fid_msys_hi, sb35, node_ids, z,
            logM, mass_band
  - KS/tsz_zenodo/fig3.csv          (real ACT x DESI-LRG y-CAP data, plain CSV)
  - KS/fgas_cap_mstar_snap085.npz   (reused: identifies the 49 kSZ-consistent
                                      node IDs via the same chi2 cut as fig05)
  - KS/desact_zenodo/Fig8_BGS_BRIGHT-20.2_logm11.00.npz  (reused: kSZ data +
                                      covariance for the chi2 cross-match)
  KS = /mnt/home/mlee1/ceph/bind_science/ksz_confront

Source: examples/paper_ksz_desi_act.ipynb (branch analysis/ksz_project,
worktree ksz-desi-act), cell 22, built by
examples/_build_ksz_paper_nb.py lines ~999-1058. Cosmology helpers
(r200phys, DA, ARCMIN) ported from the notebook's cell 2 (Sec. 0 setup),
needed to convert the kSZ-cache's theta(r200) exactly as in cell 14/fig05.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import setup, save, panel_label, COLORS

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")

# --- cosmology helpers, ported from notebook cell 2 ("Sec 0 setup") ---
Om, OL, h = 0.3089, 0.6911, 0.6774
C_KMS, H0, ARCMIN = 299792.458, 100 * 0.6774, 180 * 60 / np.pi


def Ez(z):
    return np.sqrt(Om * (1 + z) ** 3 + OL)


def DA(z, n=3000):
    zz = np.linspace(0, z, n)
    return (C_KMS / H0) * np.trapezoid(1 / Ez(zz), zz) / (1 + z)


def r200phys(logM200, z):
    M = 10 ** logM200 / h
    rho = 2.775e11 * h ** 2 * Ez(z) ** 2
    return (3 * M / (4 * np.pi * 200 * rho)) ** (1 / 3)


setup()

yl = np.load(KS / "ycap_lrg_snap067.npz")
R, yfid, ysb, idy = yl["R"], yl["fiducial"], yl["sb35"], yl["node_ids"]
ylo, yhi = yl["fid_msys_lo"], yl["fid_msys_hi"]
M_LENS = float(yl["logM"])

rows = list(csv.reader(open(KS / "tsz_zenodo/fig3.csv")))
Rd = np.array([float(r[1]) for r in rows[1:] if r[1]])
yd = np.array([float(r[2]) for r in rows[1:] if r[1]])
yde = np.array([float(r[3]) for r in rows[1:] if r[1]])
d35 = np.interp(3.5, Rd, yd)
yr = float(np.interp(3.5, R, yfid) / d35)
rlo = float(np.interp(3.5, R, ylo) / d35)
rhi = float(np.interp(3.5, R, yhi) / d35)

# --- which nodes fit the kSZ data (same chi2 cut as fig05)? highlight them here ---
fr = np.load(KS / "fgas_cap_mstar_snap085.npz")
sbk, idk = fr["sb35"][:, 0, :], fr["node_ids"]
lmk, zk = float(fr["logM200"][0]), 0.26
thx = fr["xb"] * r200phys(lmk, zk) / DA(zk) * ARCMIN
thr = r200phys(lmk, zk) / DA(zk) * ARCMIN
dk = np.load(KS / "desact_zenodo/Fig8_BGS_BRIGHT-20.2_logm11.00.npz")
m1 = dk["th"] <= 1.4 * thr
Cinv = np.linalg.inv(dk["cov_ksz"][np.ix_(m1, m1)])
pnk = np.array([np.interp(dk["th"][m1], thx, sbk[i]) for i in range(len(idk))])
chi2k = np.einsum("ni,ij,nj->n", pnk - dk["ratio"][m1], Cinv, pnk - dk["ratio"][m1])
ndof = int(m1.sum())
kfit_ids = set(idk[chi2k < ndof + 2 * np.sqrt(2 * ndof)].tolist())

# per-node tSZ amplitude chi2 (R<=2.5') vs the kSZ chi2, aligned by node id
myA = R <= 2.5
dyA, eyA = np.interp(R[myA], Rd, yd), np.interp(R[myA], Rd, yde)
chi2y = (((ysb[:, myA] - dyA) / eyA) ** 2).sum(1)
cy = {int(n): chi2y[i] for i, n in enumerate(idy)}
ck = {int(n): chi2k[i] for i, n in enumerate(idk)}
common = sorted(set(cy) & set(ck))
rho, _ = spearmanr([cy[n] for n in common], [ck[n] for n in common])

fig, (A, B) = plt.subplots(2, 1, figsize=(3.5, 5.1), constrained_layout=True)

# --- (a) amplitude: mass-selection-dominated ---
A.fill_between(R, ylo * 1e6, yhi * 1e6, color="tab:purple", alpha=0.16, zorder=0,
                label=r"$\pm0.1$ dex mass syst")
for i, row in enumerate(ysb):
    if np.all(np.isfinite(row)):
        if int(idy[i]) in kfit_ids:
            A.plot(R, row * 1e6, "-", color="tab:orange", lw=0.8, alpha=0.55, zorder=2)
        else:
            A.plot(R, row * 1e6, "-", color=COLORS["bind"], lw=0.4, alpha=0.10, zorder=1)
A.plot([], [], "-", color=COLORS["bind"], alpha=0.5, label=f"SB35 ({len(ysb)})")
A.plot([], [], "-", color="tab:orange", label=f"kSZ-consistent ({len(kfit_ids)})")
A.plot(R, yfid * 1e6, "-o", color="tab:purple", lw=2.0, ms=3.5, zorder=4,
       label=f"BIND fid (logM={M_LENS:.2f})")
A.errorbar(Rd, yd * 1e6, yerr=yde * 1e6, fmt="s", color="k", mfc="white", ms=4,
           capsize=2, zorder=5, label="ACT$\\times$DESI LRG")
A.text(0.96, 0.98,
       f"BIND $\\approx${yr:.1f}$\\times$ data (mass syst. {rlo:.1f}$-${rhi:.1f}$\\times$)"
       f"\nkSZ \\& tSZ same nodes: $\\rho{{=}}{rho:.2f}$",
       transform=A.transAxes, ha="right", va="top", fontsize=7,
       bbox=dict(boxstyle="round", fc="0.96", ec="0.7", lw=0.4))
A.set_xlabel(r"$R$ [arcmin]")
A.set_ylabel(r"$y$-CAP [$10^{-6}\,y\cdot$arcmin$^2$]")
A.legend(loc="upper left")
panel_label(A, "(a)", loc="lower right")

# --- (b) shape: normalized at R=2.25' ---
i0 = int(np.argmin(np.abs(R - 2.25)))
j0 = int(np.argmin(np.abs(Rd - 2.25)))
B.plot(R, yfid / yfid[i0], "-o", color="tab:purple", lw=1.8, ms=3, label="BIND fiducial")
B.errorbar(Rd, yd / yd[j0], yerr=yde / yd[j0], fmt="s", color="k", mfc="white", ms=4,
           capsize=2, label="ACT$\\times$DESI LRG")
B.axhline(1, color="0.6", ls=":", lw=0.6)
B.axvline(2.25, color="0.6", ls=":", lw=0.6)
B.annotate("data compact,\nBIND extended", xy=(5.4, yd[-3] / yd[j0]), xytext=(3.7, 1.68),
           fontsize=7, color="0.25", arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))
B.set_xlabel(r"$R$ [arcmin]")
B.set_ylabel(r"$y$-CAP / $y$-CAP($2.25'$)")
B.legend(loc="lower left")
panel_label(B, "(b)")

save(fig, "figs/fig07_tsz_pressure")
