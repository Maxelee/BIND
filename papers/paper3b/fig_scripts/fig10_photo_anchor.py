"""fig10_photo_anchor.pdf — the external absolute-photometry check: our chain's
stacked cumulative CAP profiles for the DESI DR9 Main-LRG sample (four photo-z
bins, 120k galaxies each) on the same ACT DR6 NILC y map, in the exact axes
convention of Liu et al. 2025 figs 3/4 (linear y x 1e-6, R in arcmin). The one
genuine series recoverable from their bugged Zenodo release is shown for
reference; it is hump-shaped (differential, not cumulative) and matches no bin
— the caption carries the identification-test framing. Published fig-3 panel
amplitudes (~2-4e-6 at 6') agree with ours to well inside 2x.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/paper3/B/wp5_inference/photo_anchor/photo_anchor_summary.json

Original source: wp5 REPORT ladder item 5 (executed 2026-07-19).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL, save, setup  # noqa: E402

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

s = json.loads(Path("/mnt/home/mlee1/ceph/paper3/B/wp5_inference/photo_anchor/"
                    "photo_anchor_summary.json").read_text())
r = np.array(s["radii_arcmin"])
d1 = s["bins"]["pz1"]
liu = np.array(d1["y_cap"]) / np.array(d1["ratio_to_liu_series"])

setup()
fig, ax = plt.subplots(figsize=ONE_COL)
cmap = mpl.cm.cividis
for k, b in enumerate((1, 2, 3, 4)):
    d = s["bins"][f"pz{b}"]
    ax.errorbar(r + 0.015 * k, 1e6 * np.array(d["y_cap"]),
                1e6 * np.array(d["y_cap_err"]), fmt="o-", ms=2.5, lw=0.9,
                capsize=1.5, color=cmap(k / 4.2), label=f"pz{b}")
ax.plot(r, 1e6 * liu, "s--", ms=3, mfc="none", color="k", lw=0.8,
        label="Zenodo series")
ax.axhline(0, color=COLORS["dmo"], lw=0.5)
ax.set_xlabel(r"$\theta_d$ [arcmin]")
ax.set_ylabel(r"$\langle y_{\rm CAP}\rangle\ \ [10^{-6}\,{\rm arcmin^2}]$")
ax.legend(loc="upper left", fontsize=6, ncols=2)

save(fig, "figs/fig10_photo_anchor")
