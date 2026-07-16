# notes: fig03_bias_scatter

**Panels**: unchanged — (a) `(dOmega_m, dS8)` scatter at `ell_max=3000`
(99 runs), colored by group `f_gas`; (b) histogram of `dS8/sigma_S8` at 3
`ell_max` cuts (2000/3000/5000). Tagged `(a)`/`(b)` via `panel_label`
instead of the placeholder's `ax.set_title(...)` banners
("Baryonic cosmology bias (ell_max=3000), 99 runs" /
"Significance of the unmodelled baryon bias") — that context now belongs
in the caption. The `ell_max=3000` callout is preserved as in-axes text in
panel (a) (`panel_label(ax, "(a) ell_max=3000", loc="lower right")`).

**IMPORTANT — one component of the reference marker could not be
reproduced from cache**: the placeholder's panel (a) draws a red "cross"
at the origin with BOTH a horizontal error bar (`sigma_Omega_m`, LSST-Y10)
and a vertical one (`sigma_S8`, LSST-Y10). `cosmo_bias.npz` was verified to
contain `run_idx, fgas, ell_maxes, dS8_{2000,3000,5000},
dOm_{2000,3000,5000}, sigS8_{2000,3000,5000}` — there is **no cached
`sigOm`** (only `sigS8`). The original script computes `sigOm` from the
same per-run 2x2 Fisher-matrix inverse as `sigS8` but that intermediate
value was never written to the npz. Recomputing it requires re-running the
pyccl Fisher derivative pipeline, which is out of scope (forbidden
re-derivation per the task rules).

Resolution taken: panel (a) keeps only the **vertical** `1-sigma S8`
reference bar (`errorbar(0, 0, yerr=median(sigS8), ...)`, labeled
"$1\sigma\,S_8$ LSST-Y10"); the horizontal `sigma_Omega_m` bar is omitted.
No plotted data point was altered or fabricated — the full 99-run
`(dOmega_m, dS8, f_gas)` scatter and the panel (b) histograms are
byte-for-byte the same cached values as the placeholder. If the caption
described a "cross" at the origin, it should be updated to describe a
single vertical bar (S8 precision only); if `sigma_Omega_m` is needed for
the paper, it requires a fresh pyccl Fisher run outside this figure task.

**Style**: `COLORS["highlight"]` (`#FF2C00`) used for the reference bar
(same red as the placeholder's `crimson`).
