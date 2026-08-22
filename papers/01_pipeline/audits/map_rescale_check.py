#!/usr/bin/env python
"""Investigation 5 (small_items_2026-08-03.md, item 5) -- MAP-RESCALING CONSISTENCY CHECK.

Fig 3 of the Paper I pipeline draft reports a halo-level Y_500c offset of
BIND/truth = 1.054 +/- 0.007 (8 sigma). Question: if the BIND y maps were
uniformly rescaled by 1/1.054, would the map-level C_ell^yy residual flatten?

Rescaling a map by a constant c scales its power spectrum by c^2 at ALL ell,
so a uniform normalization offset predicts a FLAT (ell-independent) fractional
excess of c^2 - 1 = 1.054^2 - 1 = +11.09% in C_ell^yy at every ell. The real
question is how much of the MEASURED, ell-DEPENDENT C_yy residual is
consistent with that flat +11.09% prediction, and what is left after dividing
it out (i.e. after rescaling the BIND map by 1/1.054, equivalently dividing
its power by 1.054^2).

Data: bind_science/field_cache/field_stats_fid.npz -- yy_bind / yy_truth,
shape (50 realizations, 5 planes, 724 ell), built by papers/01_pipeline/
field_cache.py. Per that module's own documented convention (and reproduced
verbatim in papers/01_pipeline/_build_figures_nb.py's fig-4 cell, ~line 1163):
y / tau autos are PER-PLANE CUMULATIVE columns, and the released convention
used throughout the paper is the TOTAL column, i.e. index [:, -1] (NOT the
z_s=1 index used for kappa) -- using [:, 1] here would silently redefine the
statistic (moves its diagonal chi2/dof from 177 to 63, per the notebook's own
comment at that line).

Conventions reproduced exactly from the notebook's `paired()` helper
(papers/01_pipeline/_build_figures_nb.py, fig-4 cell, ~line 1297) and the
shared ELL_LO / ELL_TRUST constants (~line 143-154), so the diagonal
chi2/dof computed here is a direct check against the notebook's own printed
number (quoted in the fig-4 markdown as 177 for yy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE = Path("/mnt/home/mlee1/ceph/bind_science/field_cache/field_stats_fid.npz")

ELL_LO = 100.0      # matches _build_figures_nb.py fig-4 cell
# 2026-08-04: matches _build_figures_nb.py setup cell's ELL_TRUST, moved from
# 1.5e4 to the MEASURED CIC/pixelization onset 3.0e4 (author ruling -- see
# that cell's provenance comment for the two measurements behind the move).
ELL_TRUST = 3.0e4

Y_OFFSET = 1.054                 # fig-3 halo-level Y_500c BIND/truth ratio
Y_OFFSET_ERR = 0.007
C2 = Y_OFFSET ** 2                # predicted flat multiplicative excess in C_yy


def paired(bR: np.ndarray, tR: np.ndarray, mask: np.ndarray):
    """Reproduces _build_figures_nb.py's paired(): mean curves, %-residual of
    means, paired +-1sigma band on the residual (std of per-realization ratio
    / sqrt(N)), and the diagonal chi2/dof over `mask`."""
    rat = bR / np.where(tR > 0, tR, np.nan)
    res = 100 * (np.nanmean(bR, 0) / np.nanmean(tR, 0) - 1)
    band = 100 * np.nanstd(rat, 0) / np.sqrt(bR.shape[0])
    chi2 = np.nanmean((res[mask] / np.where(band[mask] > 0, band[mask], np.nan)) ** 2)
    return res, band, chi2


def main() -> None:
    fc = np.load(CACHE)
    ell = fc["ell"]
    yy_b = fc["yy_bind"][:, -1]     # (50, 724) -- TOTAL column, released convention
    yy_t = fc["yy_truth"][:, -1]
    nr = yy_b.shape[0]
    print(f"loaded {CACHE}")
    print(f"  {nr} realizations, {ell.size} ell bins, ell in [{ell.min():.1f}, {ell.max():.1f}]")

    mt = (ell >= ELL_LO) & (ell <= ELL_TRUST)
    print(f"  trusted range ell in [{ELL_LO:.0f}, {ELL_TRUST:.0f}]: {mt.sum()} bins")

    res, band, chi2_before = paired(yy_b, yy_t, mt)
    print(f"\ndiagonal chi2/dof, full trusted range, AS MEASURED: {chi2_before:.1f}"
          "   (NOTE 2026-08-04: this was a sanity check against the notebook's "
          "OLD printed fig-4 yy number, 177, computed under the pre-migration "
          "ELL_TRUST=1.5e4. Under the widened ELL_TRUST=3.0e4 the two are no "
          "longer expected to match the old figure -- this script's own number, "
          "computed from the same field_stats_fid.npz cache with the same "
          "paired() convention, IS the projection for what a re-rendered fig 4 "
          "will print; fig 4 itself has not been re-rendered in this pass "
          "(OOMs on this node -- see audits/elltrust3e4_migration.md).)")

    print(f"\nY_500c halo-level offset: {Y_OFFSET} +/- {Y_OFFSET_ERR}"
          f"  ->  predicted flat C_yy excess c^2 = {C2:.6f}  (= {100*(C2-1):+.2f}%)")

    # Rescale the BIND map by 1/Y_OFFSET  <=>  divide its power by C2.
    # ratio_after = (1 + res/100) / C2  (exact; NOT a linear res/C2 approximation)
    res_after = 100 * ((1 + res / 100) / C2 - 1)
    band_after = band / C2      # the additive %-band on a ratio scales the same way
    chi2_after = np.nanmean((res_after[mt] / np.where(band_after[mt] > 0, band_after[mt], np.nan)) ** 2)
    print(f"diagonal chi2/dof, full trusted range, AFTER dividing out c^2={C2:.4f}: "
          f"{chi2_after:.1f}")

    frac_improved = np.mean(np.abs(res_after[mt]) < np.abs(res[mt]))
    print(f"fraction of trusted ell-bins where |residual| SHRINKS after the rescale: "
          f"{100*frac_improved:.0f}%")

    print(f"\nfull-range residual (BIND/truth - 1, %): median={np.nanmedian(res[mt]):+.2f}%  "
          f"mean={np.nanmean(res[mt]):+.2f}%  min={np.nanmin(res[mt]):+.2f}%  "
          f"max={np.nanmax(res[mt]):+.2f}%")

    print("\n--- windowed medians ---")
    windows = [(300, 3000), (5000, int(ELL_TRUST))]   # "high ell" edge = ELL_TRUST (was 15000)
    summary = {}
    for lo, hi in windows:
        m = (ell >= lo) & (ell <= hi)
        med_before = np.nanmedian(res[m])
        med_after = np.nanmedian(res_after[m])
        c2w_before = np.nanmean((res[m] / np.where(band[m] > 0, band[m], np.nan)) ** 2)
        c2w_after = np.nanmean((res_after[m] / np.where(band_after[m] > 0, band_after[m], np.nan)) ** 2)
        summary[(lo, hi)] = (med_before, med_after, c2w_before, c2w_after)
        print(f"ell in [{lo:>5d}, {hi:>5d}]  (n={m.sum():3d} bins):  "
              f"median residual BEFORE = {med_before:+7.2f}%   "
              f"AFTER /c^2 = {med_after:+7.2f}%   "
              f"chi2/dof BEFORE={c2w_before:8.1f}  AFTER={c2w_after:8.1f}")

    # where does the residual change sign? (texture signature: should NOT be a
    # single low->high monotonic curve if this were a pure amplitude offset)
    sgn = np.sign(res[mt])
    xchg = np.where(np.diff(sgn) != 0)[0]
    ell_t = ell[mt]
    print(f"\nresidual sign changes at ell ~ {np.round(ell_t[xchg], 0).tolist()}")

    print("\n=== VERDICT ===")
    lo_med_b, lo_med_a, lo_c2_b, lo_c2_a = summary[(300, 3000)]
    hi_med_b, hi_med_a, hi_c2_b, hi_c2_a = summary[(5000, int(ELL_TRUST))]
    print(
        "A constant rescale by the halo-level Y-offset (c^2 = 1.054^2 = "
        f"{C2:.3f}, i.e. a flat +{100*(C2-1):.1f}% prediction for C_yy at every ell) "
        "does NOT flatten the measured C_ell^yy residual. At mid ell (300-3000) the "
        f"measured residual is itself NEGATIVE (median {lo_med_b:+.1f}%, opposite sign "
        f"from the offset's prediction), so dividing it out makes it WORSE "
        f"(median {lo_med_a:+.1f}%, chi2/dof {lo_c2_b:.0f} -> {lo_c2_a:.0f}). At high ell "
        f"(5000-{int(ELL_TRUST)}) the measured excess (median {hi_med_b:+.1f}%) is partially but not "
        f"fully accounted for: dividing by c^2 OVERSHOOTS past zero to "
        f"{hi_med_a:+.1f}% rather than flattening to it (chi2/dof {hi_c2_b:.0f} -> "
        f"{hi_c2_a:.0f}). Over the full trusted range the diagonal chi2/dof actually "
        f"INCREASES after the rescale ({chi2_before:.0f} -> {chi2_after:.0f}), and only "
        f"{100*frac_improved:.0f}% of trusted ell-bins see |residual| shrink. The "
        "halo-integrated Y-normalization offset is therefore NOT the dominant driver of "
        "the map-level C_yy discrepancy; a scale-dependent (texture) effect dominates "
        "instead, consistent with the previously-identified over-textured-interior / "
        "under-textured-outskirts mechanism."
    )


if __name__ == "__main__":
    main()
