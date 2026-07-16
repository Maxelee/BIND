# DATA_MAP — Paper II (02_cosmo_bias)

Data-provenance map for every `\includegraphics` in `main.tex`. For each
figure: placeholder file, what it shows, the cached data file(s) + keys that
back it, where the original plotting code lives, and a regeneration-
difficulty note. All cached files listed below were verified to exist and to
contain the listed keys (`np.load(..., allow_pickle=True).keys()`), read-only,
no engine/analysis/Slurm re-run performed.

Two source locations recur:
- **Cache dir** `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/` (explicit
  ceph path named directly in the engine scripts below — reading it is
  in-scope per the task's "explicit path named in a source" allowance).
- **Engine scripts** live on the `wl-cosmo-bias` topic-branch worktree at
  `/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/wt/wl-cosmo-bias/examples/`
  (current checkout's `examples/` does not have these `lightcone_*.py`
  files — they were never merged to the branch this worktree is on). The
  rescaling-appendix engine (fig09) lives on the sibling `cosmo-rescale`
  worktree at the analogous path.

**Headline finding**: for 8 of the 10 figures, the exact PNG bytes already
in `papers/02_cosmo_bias/figs/` are byte-identical (`md5sum` match) to the
corresponding file the dossier cites in `examples/figures_lightcone/` (or
`examples/rescale_validation.png`). These are not fabricated placeholders —
they are the real, already-rendered outputs of the named engine scripts,
simply copied into the paper tree. The job below is (a) confirming an
on-disk array of the *exact plotted numbers* exists so a script can
reproduce the same figure in the house style (`paper_style.py`), and (b) for
the two figures where no such final-product array exists, saying so plainly
per `FIGURE_STYLE.md` rule 9 rather than re-deriving new numbers.

---

## fig01_transfer_ensemble.png

- **main.tex**: line 335, `\label{fig:transfer_ensemble}`, Section 3.1.
- **Shows**: 5-panel spaghetti plot of `S(ell, z_s)` per source plane, 99
  SB35 Sobol runs colored by group `f_gas`, median + 95% envelope overlaid.
- **md5 vs source**: identical to `examples/figures_lightcone/transfer_function_ensemble.png`.
- **Data file(s)+keys**:
  - `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/transfer_Sell.npz` —
    `ell(724,)`, `S(99,5,724)`, `run_idx(99,)`, `source_redshifts(5,)`,
    `sobol_params(256,35)`, `param_names(35,)`, `astro_idx(30,)`,
    `param_lo/hi/log(35,)`.
  - `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz`
    (needed only for the `f_gas` color axis via `group_fgas()`: uses
    `M_fof`, `sobol_m_gas_500c`, `sobol_m_dm_500c`, `sobol_m_star_500c`).
- **Original plotting code**: `examples/lightcone_transfer.py` (wl-cosmo-bias
  worktree), `main()`, "Fig 1" block, lines ~92–120 (`build_cube()` +
  `group_fgas()` feed it; no notebook, this is a flat script).
- **Regeneration difficulty**: Easy. Pure numpy + matplotlib on two already-
  cached arrays, no re-derivation, no external cosmology library.

## fig02_transfer_tomography.png

- **main.tex**: line 354, `\label{fig:transfer_tomography}`, Section 3.1.
- **Shows**: 2-panel — median `S(ell)` (16–84% band) per `z_s` (left);
  run-to-run `sigma[S(ell)]` vs `ell` per `z_s` (right).
- **md5 vs source**: identical to `examples/figures_lightcone/transfer_function_tomography.png`.
- **Data file(s)+keys**: same `transfer_Sell.npz` as fig01 (`ell`, `S`,
  `source_redshifts`); no `f_gas` needed for this panel pair.
- **Original plotting code**: `examples/lightcone_transfer.py`, "Fig 2"
  block, lines ~124–148 (same file/function as fig01, second half of `main()`).
- **Regeneration difficulty**: Easy. Same single cached array as fig01.

## fig03_bias_scatter.png

- **main.tex**: line 396, `\label{fig:bias_scatter}`, Section 3.2.
- **Shows**: 2-panel — `(dOmega_m, dS8)` bias cloud at `ell_max=3000` (99
  runs) colored by `f_gas`, with the 1-sigma LSST-Y10 error cross (left);
  histogram of `dS8/sigma` at 3 `ell_max` cuts (right).
- **md5 vs source**: identical to `examples/figures_lightcone/cosmo_bias.png`.
- **Data file(s)+keys**:
  `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/cosmo_bias.npz` —
  `run_idx(99,)`, `fgas(99,)`, `ell_maxes(3,)`, `dS8_2000/3000/5000(99,)`,
  `dOm_2000/3000/5000(99,)`, `sigS8_2000/3000/5000(99,)`.
- **Original plotting code**: `examples/lightcone_cosmo_bias.py`, `main()`,
  lines ~176–209. Note: building this npz originally required a `pyccl`
  Fisher-derivative computation (`build_theory()`), but the **final** numbers
  that the figure plots are already fully cached — no `pyccl` needed to
  regenerate the figure itself.
- **Regeneration difficulty**: Easy. All plotted quantities are directly in
  `cosmo_bias.npz`; no cosmology-library call needed at plot time.

## fig04_template_ladder.png

- **main.tex**: line 473, `\label{fig:template_ladder}`, Section 3.3
  (headline result; also backs Table 1, `tab:ladder`).
- **Shows**: 2-panel headline — residual `|dS8|/sigma_S8` vs N templates
  (median + 95th pct, 3 `ell_max` curves, log-y, N=2 marked) (left);
  `sigma_S8` inflation vs N templates (right); 216 runs.
- **md5 vs source**: identical to `examples/figures_lightcone/cosmo_bias_capstone.png`.
- **Data file(s)+keys**:
  `/mnt/home/mlee1/BIND/examples/figures_lightcone/cosmo_bias_capstone.npz`
  (already inside the main BIND checkout, not ceph) — `ell_maxes(3,)`,
  `n_templates(7,)=[0,1,2,3,4,6,10]`, `ratio_{ellmax}_{N}(216,)` for every
  `ellmax`×`N` combination, `sig_{ellmax}_{N}()` scalars, `sig0_{ellmax}()`
  no-baryon floor scalars, `evr_{ellmax}(6,)` PCA explained-variance ratios.
  This single file also backs every number in Table 1.
- **Original plotting code**: `examples/cosmo_bias_capstone.py` (worktree
  `wl-cosmo-bias/examples/`), `main()`, lines ~121–153.
- **Regeneration difficulty**: Easy. One self-contained npz has every curve
  and every table entry already computed.

## fig05_beyond2pt_manifold.png

- **main.tex**: line 557, `\label{fig:beyond2pt}`, Section 3.5.
- **Shows**: 2-panel — cumulative feedback-manifold variance vs N template
  modes for 10 WL/SZ statistics (left); 10×10 principal-angle subspace-
  alignment matrix relative to kappa `C_ell` (right); 237 runs.
- **md5 vs source**: identical to `examples/figures_lightcone/beyond2pt_manifold.png`.
- **Data file(s)+keys**:
  `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` (104 MB, 87 keys) —
  `X_unit`, and per statistic `t__{stat}__value` / `t__{stat}__valid` for
  `stat in {cl_kappa, peak_counts, minima_counts, pdf, mf_v0, mf_v1, mf_v2,
  moments, peak_y, cl_kappa_y}`; `cl_kappa` additionally needs
  `a__cl_kappa__ell` (for the `ELL_TRUST` cap) and `cl_kappa_y` needs
  `a__cl_kappa_y__ell`. All verified present.
- **Original plotting code**: `examples/lightcone_beyond2pt.py`, `extract()`
  (raw-array selection), `manifold()`/`subspace_overlap()` (the linearised-
  active-subspace computation, Constantine 2015 — plain `numpy.linalg.lstsq`
  + `numpy.linalg.svd`, no external library), and the plotting block in
  `main()`, lines ~163–211. The script's docstring states explicitly: "this
  prototype needs NO cosmology derivative — it is pure manifold geometry
  from the suite." (The one place it *does* call `pyccl`,
  `kappa_cost_anchor()`, only produces a printed console cross-check number,
  not anything drawn in the figure — safely ignorable for regeneration.)
- **Regeneration difficulty**: Moderate. No cached final-product npz exists
  for this figure specifically (no `savez` call in the script), but the
  entire computation is closed-form linear algebra (SVD/lstsq) on the single
  raw `emulator_dataset.npz` cache — no simulation, no MPI, no external
  physics library. This is squarely "plotting/postprocessing from a cached
  array," not "re-running an analysis engine," so it is feasible: a
  regenerator script can re-implement `extract()` + `manifold()` +
  `subspace_overlap()` verbatim against the cached dataset.

## fig06_bias_sensitivity.png

- **main.tex**: line 593, `\label{fig:bias_sensitivity}`, Section 3.6.
- **Shows**: 2-panel — ranked total/first-order Sobol sensitivity indices of
  `dS8` over top 12 of 30 astro params (left); scatter of `dS8` vs top driver
  (IMFslope), colored by `f_gas` (right); 99 runs, `ell_max=3000`.
- **md5 vs source**: identical to `examples/figures_lightcone/bias_sensitivity.png`.
- **Data file(s)+keys**:
  - `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/bias_sensitivity.npz` —
    `names(30,)`, `ell_max()`, `S_total(30,)`, `S_first_surr(30,)`,
    `S_first_bin(30,)`, `perm(30,)`, `cv_r2()`. This has everything for the
    **left** panel bars (`S_total`/`S_first_bin`) directly.
  - For the **right** panel scatter, raw per-run values are needed:
    `transfer_Sell.npz` (`sobol_params`, `astro_idx`, `param_names`,
    `param_lo/hi/log`, `run_idx`) to reconstruct the normalized top-driver
    column `X[:, j0]`, plus `cosmo_bias.npz` (`dS8_3000`, `fgas`) for the y
    and color axes — both already verified above (fig01/fig03).
- **Original plotting code**: `examples/lightcone_bias_sensitivity.py`,
  `load_design()` (lines ~29–44) for the raw-param reconstruction, `main()`
  plotting block lines ~95–141.
- **Regeneration difficulty**: Easy. No gradient-boosted surrogate refit
  needed — `S_total`/`S_first_bin` (what's actually drawn) are cached
  directly; the right panel only needs already-cached per-run arrays.

## fig07_selfcal.png

- **main.tex**: line 646, `\label{fig:selfcal}`, Section 3.6 ("Corrected
  tomographic self-calibration").
- **Shows**: 2-panel CORRECTED result — per-plane bias amplitude declining
  with `z_s` vs a flat pure-`S8`-shift reference (left); `sigma(S8)`
  degradation vs number of tomographic bins, dropping to ~1 by 2 bins
  (right).
- **md5 vs source**: identical to `examples/figures_lightcone/selfcal.png`.
- **Data file(s)+keys**: **none with the final plotted numbers.** The script
  only reads the raw `transfer_Sell.npz` cube and then computes the joint
  `(S8, Omega_m, A_bary)` Fisher forecast itself, at plot time, by calling
  `lightcone_cosmo_bias.build_theory()` / `covariance()` — i.e. fresh
  `pyccl` (`ccl.Cosmology`, `ccl.WeakLensingTracer`, `ccl.angular_cl`) calls
  for the cosmology derivatives that feed the degradation-vs-bins curve.
  `lightcone_selfcal.py` has **no `np.savez` call at all** — only
  `fig.savefig(OUT / "selfcal.png", ...)` — so the final numbers this
  figure draws exist nowhere on disk as an array, only baked into this PNG.
- **Original plotting code**: `examples/lightcone_selfcal.py` (worktree
  `wl-cosmo-bias/examples/`), `derivatives()` lines ~32–37, main Fisher/
  tomography-bin loop, plot block lines ~150–174.
- **Regeneration difficulty**: **Not cache-feasible under FIGURE_STYLE.md
  rule 9.** Reproducing the exact curves requires re-running a `pyccl`
  cosmology-derivative + Fisher computation — this is "computing science,"
  not loading a precomputed array, and re-running analysis engines is
  explicitly forbidden by this task's hard rules regardless of how fast the
  computation is. **UNMAPPED for a from-cache regeneration script.** The
  existing PNG (`examples/figures_lightcone/selfcal.png`, byte-identical to
  the current placeholder) is itself the only faithful on-disk artifact of
  this result; it is not fabricated, but it cannot be re-plotted in house
  style from a cached numeric array. Recommend keeping the placeholder as-is
  and flagging in `fig_scripts/UNREGENERATED.md` per the style guide, rather
  than inventing a `pyccl` re-run.

## fig10_bias_tomography_superseded.png

- **main.tex**: line 662, `\label{fig:bias_tomography}`, Section 3.6
  (explicitly superseded footnote figure, width `0.85\textwidth`).
- **Shows**: 3-panel — per-plane `dS8(z_s)` spaghetti with mean (left);
  z-shape universality normalized to `z_s=1` (middle); SVD rank bar chart,
  PC1≈99.7% (right). Pre-correction Step-4 result, retained only as a
  methodological cautionary figure.
- **md5 vs source**: identical to `examples/figures_lightcone/bias_tomography.png`.
- **Data file(s)+keys**:
  `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/bias_tomography.npz` —
  `dS8z(99,5)`, `zs(5,)`, `slope(99,)`, `ampl(99,)`, `var_frac(5,)`,
  `var_frac_raw(5,)`, `rho_slope(30,)`, `names(30,)`, `run_idx(99,)`. All
  three panels' content (spaghetti, normalized shape, SVD variance bars) is
  fully contained here.
- **Original plotting code**: `examples/lightcone_bias_tomography.py`
  (worktree `wl-cosmo-bias/examples/`), `main()`, plot block lines ~99–137.
  Unlike `selfcal.py`, this earlier (pre-correction) script *does* call
  `np.savez(CACHE / "bias_tomography.npz", ...)` before plotting — its
  final numbers are on disk.
- **Regeneration difficulty**: Easy. Fully cached; no `pyccl` needed to
  redraw — the Fisher-derived `dS8z` values were already computed once and
  saved before this (now-superseded) script's own plotting step. Caption
  must retain the "SUPERSEDED — see Fig. 7" framing already in `main.tex`.

## fig08_model_comparison.png

- **main.tex**: line 712, `\label{fig:model_comparison}`, Section 3.7.
- **Shows**: 2-panel (`z_s=0.5, 1.0`) — BIND `S(ell)` median + 95% envelope
  vs BCM (Schneider15), van Daalen 2019, and BCemu analytic baryon-model
  spans, all CCL-projected through the same WL kernel.
- **md5 vs source**: identical to `examples/figures_lightcone/model_comparison.png`.
- **Data file(s)+keys**: the **BIND envelope only** is cached
  (`transfer_Sell.npz`: `ell`, `S` — same file as fig01/fig02, sliced to
  `zs in {0.5,1.0}` i.e. `i in {0,1}`). The three **analytic model spans**
  (`bcm_span()`, `vandaalen_span()`, `bcemu_span()`) are computed fresh at
  plot time from scratch every run: `bcm_span`/`vandaalen_span` call
  `pyccl.baryons.BaryonsSchneider15`/`BaryonsvanDaalen19` +
  `ccl.angular_cl` over a random/linspace parameter scan (250 and 60 draws
  respectively); `bcemu_span` calls the external `BCemu` package
  (`BCemu.BCemu2025().get_boost(...)`) over 12 `log10Mc` values, then
  projects through CCL again. **No `np.savez` in this script at all** — it
  only ever writes `figs/model_comparison.png` (confirmed: grep for
  `savez`/`savefig` in `lightcone_model_comparison.py` shows only one
  `savefig` line). None of the three model-span arrays exist anywhere on
  disk.
- **Original plotting code**: `examples/lightcone_model_comparison.py`
  (worktree `wl-cosmo-bias/examples/`), `bcm_span()`/`vandaalen_span()`/
  `bcemu_span()` lines ~57–99, plot block lines ~132–159.
- **Regeneration difficulty**: **Not cache-feasible under FIGURE_STYLE.md
  rule 9.** The scientifically interesting content of this figure — the
  three analytic-model spans BIND is being compared against — has no cached
  array anywhere; regenerating it exactly means re-running `pyccl` (halofit
  + BCM/van-Daalen boosts) and the external `BCemu` emulator, i.e.
  re-executing the Step-5 analysis engine, which this task forbids. Only the
  BIND envelope half of the figure could be redrawn from cache in isolation
  — insufficient to reproduce the placeholder's actual content (three-way
  model comparison). **UNMAPPED for a from-cache regeneration script.**
  The existing byte-identical PNG (`examples/figures_lightcone/model_comparison.png`)
  is the only faithful on-disk artifact; recommend keeping the placeholder
  and flagging in `fig_scripts/UNREGENERATED.md`.

## fig09_rescale_validation.png

- **main.tex**: line 998, `\label{fig:rescale_validation}`, Appendix A.
- **Shows**: 3×2 grid — `P(k)`/halofit ratio and `N(>M)`/Tinker08 ratio for
  mild / low-`Omega_m` / extreme-corner AW10-rescaled TNG300-3-Dark targets;
  control (unrescaled-vs-own-theory) vs rescaled (vs-target-theory) curves
  per row.
- **md5 vs source**: identical to `examples/rescale_validation.png` (note:
  this one lives directly under `examples/`, not `examples/figures_lightcone/`).
- **Data file(s)+keys**:
  `/mnt/home/mlee1/BIND/examples/rescale_validation.npz` (in the main BIND
  checkout) — for each of the three target rows, prefix `{mild, low-Om,
  corner-hi}_`: `s()`, `z_star()`, `snap()`, `rms()`, `mass_factor()`,
  `k_ctrl/ratio_ctrl(124,)`, `k_resc/ratio_resc(124,)`,
  `M_ctrl/hmf_ctrl(24,)`, `M_resc/hmf_resc(24,)`, `n_ctrl/n_resc(24,)`,
  `pk_err_k/pk_err(~122-123,)`. Every number quoted in the Appendix-A text
  (3.4% mild max error, 26% low-Om BAO spike, 50%+ corner-hi breakdown,
  HMF ratios, halofit `D(k)` point-by-point predictions) traces to these
  arrays.
- **Original plotting code**: `examples/rescale_validation.py` (worktree
  `cosmo-rescale/examples/`), `_plot()` function, lines ~189–217 (the
  `main()`/data-generation half of this script does require re-running the
  AW10 rescaling solver against the real TNG300-3-Dark snapshot — but that
  step already ran once and its output is what's cached in
  `rescale_validation.npz`; only `_plot()` is needed for the figure).
- **Regeneration difficulty**: Easy. `_plot(results)` takes exactly the
  dict this npz's flattened keys reconstruct (`{name}_{key}` →
  `results[name][key]`); no simulation re-run needed, pure matplotlib on
  cached arrays.

---

## Summary table

| Figure | Placeholder≡source PNG? | Final-product cache exists? | Feasible from cache alone? |
|---|---|---|---|
| fig01_transfer_ensemble | yes | `transfer_Sell.npz` + `atlas_cube_snap096.npz` | yes |
| fig02_transfer_tomography | yes | `transfer_Sell.npz` | yes |
| fig03_bias_scatter | yes | `cosmo_bias.npz` | yes |
| fig04_template_ladder | yes | `cosmo_bias_capstone.npz` | yes |
| fig05_beyond2pt_manifold | yes | `emulator_dataset.npz` (raw; needs numpy-only postprocessing) | yes |
| fig06_bias_sensitivity | yes | `bias_sensitivity.npz` + `transfer_Sell.npz`/`cosmo_bias.npz` | yes |
| fig07_selfcal | yes | **none** — needs fresh `pyccl` Fisher calc | **no** |
| fig08_model_comparison | yes | **none** — needs fresh `pyccl`+`BCemu` calc | **no** |
| fig09_rescale_validation | yes | `rescale_validation.npz` | yes |
| fig10_bias_tomography_superseded | yes | `bias_tomography.npz` | yes |

8/10 figures are straightforwardly regenerable from an on-disk cached array
in house style. fig07 and fig08 have no cached final-product array — their
existing PNGs are real (not fabricated) but were produced by scripts that
compute fresh cosmology-model curves (`pyccl`/`BCemu`) at plot time rather
than reading them from disk; per this task's hard rule against re-running
analysis engines and `FIGURE_STYLE.md` rule 9, these two should stay as the
extracted placeholder with an `UNREGENERATED.md` entry rather than be
re-derived here.
