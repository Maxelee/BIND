"""fig04_null_battery.pdf — the pre-registered null battery + covariate tests in
one consistency panel (Jeffrey-Fig-11 convention: one row per test, excursion
in units of the statistical error, criterion threshold marked per family).
Left of the line = pass; the worst excursion anywhere is 2.4 sigma (tercile
family, criterion 2), and every excision-family excursion is <= 0.71.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/paper3/B/wp3_nulls/b3_scorecard.json (S1-S6)
  - /mnt/home/mlee1/ceph/paper3/B/wp3_nulls/star_dust_summary.json (S7/S8)

Original source: WP-B3 NULL_CRITERIA.md battery (pre-registered) + the
session-8 star/PSF+dust follow-up; all numbers independently recomputed
bit-identically in the 2026-07-19 gap-3 closure.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL_SQ, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

B = Path("/mnt/home/mlee1/ceph/paper3/B/wp3_nulls")
sc = json.loads((B / "b3_scorecard.json").read_text())
sd = json.loads((B / "star_dust_summary.json").read_text())


def sd_worst(block, what):
    out = 0.0
    for var in ("wiener", "glimpse"):
        blk = sd.get(f"{var}_{block}_{what}", {})
        for k, vv in blk.items():
            if isinstance(vv, dict):
                z = vv.get("sigma", vv.get("delta_over_sigma_stat"))
                if z is not None:
                    out = max(out, abs(float(z)))
    return out


def s5_worst():
    out = 0.0
    for var, blk in sc["S5_mask_proximity"]["detail"].items():
        for b, vv in blk.items():
            out = max(out, abs(float(vv["sigma"])))
    return out


rows = [
    # (label, worst |z| wiener-side, worst |z| glimpse-side, threshold)
    ("S1 random ensemble",
     2 * np.max(np.abs(sc["S1_ensemble_wiener"]["mean_over_2err"])),
     2 * np.max(np.abs(sc["S1_ensemble_glimpse"]["mean_over_2err"])), 2.0),
    ("S2 RA-shift coherence",
     np.max(np.abs(sc["S2_shifts"]["bin_coherence_mean_z"])), np.nan, 2.0),
    ("S3 B-mode peaks",
     np.max(np.abs(sc["S3_bmode"]["significance"])), np.nan, 2.0),
    ("S5 mask proximity", s5_worst(), np.nan, 2.0),
    ("star/PSF terciles",
     max(sd_worst("star", "tercile"), sd_worst("psf_dresid", "tercile")), np.nan, 2.0),
    ("dust terciles", sd_worst("ebv", "tercile"), np.nan, 2.0),
    ("star/PSF excision",
     max(sd_worst("star", "excision"), sd_worst("psf_dresid", "excision")), np.nan, 0.5),
    ("dust excision", sd_worst("ebv", "excision"), np.nan, 0.5),
]

setup()
fig, ax = plt.subplots(figsize=ONE_COL_SQ)
yy = np.arange(len(rows))[::-1]
for y0, (label, zw, _zg, thr) in zip(yy, rows):
    ax.plot([thr, thr], [y0 - 0.32, y0 + 0.32], color=COLORS["highlight"], lw=1.0)
    ax.plot([0, zw], [y0, y0], color=COLORS["dmo"], lw=0.6)
    ax.plot(zw, y0, "o", ms=3.5, color="k")
ax.plot([], [], "o", ms=3.5, color="k", label="worst excursion (both variants)")
ax.plot([], [], color=COLORS["highlight"], lw=1.0, label="pre-registered criterion")
ax.set_yticks(yy, [r[0] for r in rows], fontsize=6.5)
ax.set_xlabel(r"$|\Delta|/\sigma_{\rm stat}$")
ax.set_xlim(0, 2.9)
ax.legend(loc="lower right", fontsize=5.5)

save(fig, "figs/fig04_null_battery")
