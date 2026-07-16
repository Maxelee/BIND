#!/usr/bin/env python
"""Fig 10 -- SUPERSEDED pre-correction Step-4 tomographic self-calibration
result (retained only as a methodological cautionary figure; the paper's
actual self-calibration result is Fig. 7 / fig07_selfcal, which this figure
does NOT reproduce -- see main text Sec. 3.6 for the correction).

3 panels, single-amplitude-per-source-plane collapse of Delta_S8(z_s):
  (a) per-run Delta_S8(z_s) spaghetti (99 runs) + ensemble mean.
  (b) same, normalized to each run's z_s=1 value -- "z-shape universality".
  (c) SVD variance-fraction spectrum of the raw (99x5) Delta_S8(z_s) matrix;
      PC1 approx 99.7% of the variance is the (flawed) rank-1 evidence that
      was originally read as "baryons act as one nuisance amplitude across
      redshift" -- since corrected (Fig. 7).

Data (already-cached; the CCL Fisher derivative that produced Delta_S8(z_s)
already ran once and its final numbers are saved -- no pyccl call here):
  /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/bias_tomography.npz --
  dS8z(99,5), zs(5,), var_frac_raw(5,).

Source: examples/lightcone_bias_tomography.py (wl-cosmo-bias worktree),
main() plot block lines ~99-137.
Placeholder this replaces: figs/fig10_bias_tomography_superseded.png
(md5-identical to examples/figures_lightcone/bias_tomography.png).

Note: the original panel (a) additionally colored each run by group f_gas,
which required a second cached file (cosmo_bias.npz) not listed in this
figure's DATA_MAP entry and not referenced by the figure's caption text;
that coloring is dropped here (plain ensemble lines + mean) as a
non-content-bearing simplification -- all data values (dS8z, zs, var_frac_raw)
are unchanged.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import BAND_ALPHA, COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

CACHE = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache")


def main() -> None:
    setup()
    d = np.load(CACHE / "bias_tomography.npz", allow_pickle=True)
    dS8z, zs = d["dS8z"], d["zs"]
    var_frac_raw = d["var_frac_raw"]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(TWO_COL[0], 2.6))

    for r in range(dS8z.shape[0]):
        a1.plot(zs, dS8z[r], "-", lw=0.5, alpha=0.35, color=COLORS["dmo"])
    a1.plot(zs, dS8z.mean(0), color=COLORS["truth"], lw=1.6, label="mean")
    a1.axhline(0, color="0.6", lw=0.7)
    a1.set_xlabel(r"source redshift $z_s$")
    a1.set_ylabel(r"$\Delta S_8(z_s)$")
    a1.legend(loc="upper right")
    panel_label(a1, "(a)")

    ref = dS8z[:, 1]  # z_s = 1.0 (second of the 5 planes)
    good = np.abs(ref) > 1e-4
    shp = dS8z[good] / ref[good][:, None]
    for r in range(shp.shape[0]):
        a2.plot(zs, shp[r], "-", lw=0.4, alpha=0.3, color=COLORS["dmo"])
    med = np.median(shp, 0)
    a2.plot(zs, med, color=COLORS["highlight"], lw=1.6, label="median")
    a2.fill_between(zs, np.percentile(shp, 16, 0), np.percentile(shp, 84, 0),
                     color=COLORS["highlight"], alpha=BAND_ALPHA)
    a2.axhline(1, color="0.6", lw=0.7)
    a2.set_xlabel(r"source redshift $z_s$")
    a2.set_ylabel(r"$\Delta S_8(z_s)\,/\,\Delta S_8(z_s{=}1)$")
    a2.set_ylim(-1, 3)
    a2.legend(loc="upper right")
    panel_label(a2, "(b)")

    a3.bar(range(1, 6), var_frac_raw, color=COLORS["bind"])
    a3.set_xlabel("SVD component")
    a3.set_ylabel("variance fraction")
    a3.set_yscale("log")
    panel_label(a3, "(c)")

    fig.tight_layout()
    save(fig, "figs/fig10_bias_tomography_superseded")


if __name__ == "__main__":
    main()
