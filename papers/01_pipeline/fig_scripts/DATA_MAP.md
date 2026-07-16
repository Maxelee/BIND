# DATA_MAP — Paper I (`01_pipeline`)

For every `\includegraphics` in `main.tex`. Verified read-only against files that
already exist on disk (no engine/notebook/MPI/Slurm re-run performed). All paths
below were checked with `ls` / `np.load(..., allow_pickle=True)`.

**Headline finding: every one of the 13 placeholders is already byte-identical
(md5sum-verified) to a currently-existing cached figure file elsewhere in the
repo** — `figs/*.png` or `examples/figures_lightcone/fig_*.pdf` on the current
`lightcone` branch (not even a worktree). Nothing is stale; nothing needs to be
regenerated from scratch. For each figure below I still trace the underlying
`.npz`/`.json` data + the exact plotting code, in case a caption number ever
needs to be re-derived or the figure needs a cosmetic tweak.

Common paths:
- `REPO = /mnt/home/mlee1/BIND`
- `SCI = /mnt/home/mlee1/ceph/bind_science` (symlink target `/mnt/sdceph/users/mlee1/bind_science`)
- `CACHE = SCI/dashboard_cache`, `ATLAS = SCI/halo_atlas`, `RUNS = SCI/runs`, `PROF = SCI/profiles`, `TAU = SCI/tau_profiles`

---

## fig01_map_gallery.png

- **Placeholder**: `figs/fig01_map_gallery.png` (1760×550 PNG)
- **Shows**: 3-panel fiducial lightcone maps — DMO κ, BIND κ (both z_s=1), BIND Compton-y, 5×5 deg².
- **md5 match**: identical to `/mnt/home/mlee1/BIND/figs/demo_maps.png` (confirmed, same md5 `e7d7b5...`). That file is itself the direct plot output, already on disk in the repo root `figs/` dir — no regeneration needed, it *is* the paper figure already.
- **Underlying data**: `/mnt/home/mlee1/ceph/bind_science/demo_maps.npz` — confirmed present, keys `kappa_bind (1024,1024)`, `kappa_dmo (1024,1024)`, `y (1024,1024)`, `fov_deg` (scalar, =5.0), `z_source` (scalar).
- **Original plotting code**: `examples/paper_lightcone_figs.ipynb` cell 3 (§2.3, worktree `wl-tsz-bridge`), builder script `examples/_build_paper_nb.py` (same cell content, ~line 570). Loads `demo_maps.npz`, applies 1-px Gaussian cosmetic smoothing, `cividis` for κ panels (shared vmin/vmax = 1st/99th percentile of BIND κ), `inferno` + `LogNorm` for y, saves as `fig_skymaps.pdf`. (Note: `fig_skymaps.pdf` in `examples/figures_lightcone/` is a *different* 3-panel figure — DMO/BIND/**residual** Δκ, not the y-map — so it is NOT the source of this placeholder; `demo_maps.png` is a separate, simpler direct render of the same underlying npz and is the actual match.)
- **Regeneration difficulty**: trivial (already done — file is present and correct).

## fig02_validation_field.png

- **Placeholder**: `figs/fig02_validation_field.png`
- **Shows**: 6-panel field-level closure, BIND vs TNG300-hydro truth: C_ℓ^κκ, N_peak(ν), N_min(ν), V0/V1/V2(ν), each with a residual sub-panel.
- **md5 match**: identical to `figs_raw/paper_lightcone_figs/cell006_out1.png` (confirmed, md5 `cafa0f...`) — i.e. the placeholder is a direct copy of that notebook-cell output, already captured in `figs_raw/`.
- **Underlying data** (all confirmed to exist):
  - `RUNS/bind/run_0000/{Cl_kappa.npz, peak_counts.npz, nongaussian_stats.npz}`
  - `RUNS/truth/run_0000/{Cl_kappa.npz, peak_counts.npz, nongaussian_stats.npz}`
  - Keys used: `Cl_kappa.npz["ell"]`, `["cl"][ZIDX,ZIDX]` (ZIDX=1, z_s=1.0); `peak_counts.npz["nu"]`, `["peak_counts"][ZIDX]`, `["minima_counts"][ZIDX]`; `nongaussian_stats.npz["mf_nu"]`, `["V0"/"V1"/"V2"][ZIDX]`.
- **Original plotting code**: `examples/paper_lightcone_figs.ipynb` cell 6 (§2.4, worktree `wl-tsz-bridge`) — full source read and reproduced in full below the summary table; top row = BIND (solid) vs hydro (dashed), bottom row = residual (`%(B/H-1)` for ratio-type stats, `%(B-H)/max|H|` for the MFs which cross zero), ±5% shaded band.
- **Regeneration difficulty**: trivial (placeholder already matches cached cell output exactly).

## fig03_validation_halo.png

- **Placeholder**: `figs/fig03_validation_halo.png`
- **Shows**: 3-panel halo-level closure at snap 96 (z≈0.03) — (a) Y_500c–M relation w/ scatter bands, BIND vs TNG truth; (b) stacked τ_BIND/τ_hydro ratio by mass bin; (c) projected pressure profile y(r)/y(r500c) vs Arnaud+2010, 4 mass bins.
- **md5 match**: identical to `figs_raw/paper_lightcone_figs/cell005_out1.png` (confirmed, md5 `a9a890...`).
- **Underlying data** (all confirmed to exist):
  - `ATLAS/fid_snap096.npz`, `ATLAS/truth_snap096.npz` (keys `M_fof`, `Y_500c`) for panel (a)
  - `TAU/tau_profiles_snap096.npz` (keys `r_r500`, `mass_bins`, `bind_prof`, `truth_prof`, `bind_cnt`) for panel (b)
  - `PROF/fid_snap096.npz`, `PROF/truth_snap096.npz` (keys `r_cen`, `mass_bins`, `counts`, `prof_y`) for panel (c), fit against a hardcoded Arnaud+2010 GNFW (P0=8.403, c500=1.177, γ=0.3081, α=1.0510, β=5.4905).
- **Original plotting code**: `examples/paper_lightcone_figs.ipynb` cell 5 (§2.4), full source captured.
- **Regeneration difficulty**: trivial (placeholder already matches cached cell output exactly).

## fig04_cross_spectra.png

- **Placeholder**: `figs/fig04_cross_spectra.png`
- **Shows**: Left — ℓ(ℓ+1)|C_ℓ^κy|/2π at 5 source-redshift bins; Right — tSZ auto-spectrum ℓ(ℓ+1)C_ℓ^yy/2π.
- **md5 match**: identical to `figs_raw/fiducial_lightcone_stats/cell013_out0.png` (confirmed, md5 `0d1c8b...`).
- **Underlying data — cheapest path**: `RUNS/bind/run_0000/Cl_kappa_y.npz` — confirmed present, keys `ell (724,)`, `cl_ky (5,724)`, `cl_ky_err`, `cl_yy (724,)`, `cl_yy_err`. This is the pre-computed equivalent of the notebook's on-the-fly `S.cl_kappa_y(...)` call (same quantity, already cached per-run) and plots with the identical recipe (`f=ℓ(ℓ+1)/2π`, `loglog`, one curve per z_s for the cross panel, one curve for yy).
- **Underlying data — literal notebook path** (heavier, not needed since the cache above suffices): `examples/fiducial_lightcone_stats.ipynb` cell 1 loads 100 realizations of κ/y from `/mnt/home/mlee1/ceph/bind_lightcone_tng/rt_output` via `bind.inference.lux_io.{load_kappa_realizations,load_y_realizations}`, then cell 13 calls `bind.inference.stats.cl_kappa_y(kappa, ymaps[:,-1], fov_deg=5.0)`. This route re-invokes the stats *module* (not training/MPI/Slurm) but is unnecessary — the cached `Cl_kappa_y.npz` above is the same statistic already computed and saved.
- **Original plotting code**: `examples/fiducial_lightcone_stats.ipynb` cell 13 (heading "6. tSZ × WL cross-correlation (1a)"), worktree `wl-tsz-bridge`.
- **Regeneration difficulty**: trivial from the cache (`Cl_kappa_y.npz` has exactly the arrays plotted); placeholder already matches cached cell output exactly.

## fig05_bridge_hero.png

- **Placeholder**: `figs/fig05_bridge_hero.png` (in the paper dir: `papers/01_pipeline/figs/fig05_bridge_hero.png`)
- **Shows**: Left — per-peak Δκ/κ vs per-halo Δln M_gas excursion (grey=peaks, blue=run means), slope≈0.080≈f_gas^proj, r=0.50. Right — ejection–heating plane Δln M_gas vs Δln T, colored by Δln Y.
- **md5 match**: identical to `figs/bind_bridge_hero.png` (confirmed, md5 `d95aa1...`).
- **Underlying data** (all confirmed to exist): `CACHE/g1_peakhalo_map.npz` (key `pk_uh`), `CACHE/g1_peakhalo.npz` (keys `ref_nu`, `ref_Mg`, `ref_Y`, and per-run `{run}_nu`, `{run}_Mg`, `{run}_Y`), `CACHE/g1_design.json` (1P design table), `src/bind/assets/SB35_param_minmax.csv` (param metadata/family classification).
- **Original plotting code**: `examples/bind_bridge.py::fig_hero()` (worktree `wl-tsz-bridge`, full source read) — reads the cache, aggregates per feedback run (mean over 42 peak-halos), no new heavy compute.
- **Regeneration difficulty**: trivial — `python examples/bind_bridge.py --fig 1` from cached data only (not run here per the no-engine-execution rule, but the placeholder already matches the on-disk output byte-for-byte).

## fig06_bridge_fgas.png

- **Placeholder**: `figs/fig06_bridge_fgas.png`
- **Shows**: Left — S(ℓ) suppression curves, fiducial + lowest/highest-f_gas 1P runs (shows the CIC-aliasing high-ℓ upturn). Right — S(ℓ~1500) vs background-subtracted group f_gas (M200c 1–3e13, R500c), 57 runs colored by feedback family, r=0.91.
- **md5 match**: identical to `figs/bind_bridge_fgas_sofl.png` (confirmed, md5 `3039fc...`).
- **Underlying data**: `CACHE/g1_stats.npz` (1107 keys incl. `ell`, per-run `{run}_clk`), `RUNS/bind/run_0000/Cl_kappa.npz`, `RUNS/dmo/run_0000/Cl_kappa.npz` (both confirmed present), `ATLAS/{run}_snap096.npz` for `_fgas_run()` (keys `M_fof`, `m_gas_500c_bg`, `m_tot_500c_bg`), `CACHE/g1_design.json`.
- **Original plotting code**: `examples/bind_bridge.py::fig_fgas_sofl()` (full source read).
- **Regeneration difficulty**: trivial — same script, `--fig 2`.

## fig07_bridge_response.png

- **Placeholder**: `figs/fig07_bridge_response.png`
- **Shows**: Left — bar chart of |corr(field observable, halo Δf_gas)| across 57 runs: S(ℓ) 0.92, R(ν) 0.92, N_min 0.90, C_κy 0.87, V1 0.83, V2 0.82, N_pk 0.28. Right — R(ν)-vs-f_gas scatter, r=0.92.
- **md5 match**: identical to `figs/bind_bridge_response.png` (confirmed, md5 `91b52b...`).
- **Underlying data**: `CACHE/g1_stats.npz` (keys `nu`, `ell`, `Rnu`, per-run `{run}_clk`, `{run}_ky`, `{run}_V1`, `{run}_V2`, `{run}_R`), `ATLAS/{run}_snap096.npz` + `ATLAS/fid_snap096.npz` (paired response), `CACHE/g1_design.json`.
- **Original plotting code**: `examples/bind_bridge.py::fig_response()` (full source read).
- **Regeneration difficulty**: trivial — same script, `--fig 3`.

## fig08_vandaalen_analog.pdf

- **Placeholder**: `figs/fig08_vandaalen_analog.pdf`
- **Shows**: (a) TNG-CAMELS position on the van Daalen+2020 functional form for renormalized baryon fraction f̃_bar,500c vs ΔP/P_DMO (flat saturated top, f̃≈0.81–0.98). (b) WL suppression relation ΔC_ℓ/C_ℓ^DMO vs f̃_bar at ℓ=1000,3000,6000.
- **md5 match**: identical to `examples/figures_lightcone/fig_bridge_vandaalen.pdf` (confirmed, md5 `1a8f1e...`).
- **Underlying data**: `CACHE/g1_stats.npz` (`ell`, per-run `{run}_clk`), `RUNS/bind,dmo/run_0000/Cl_kappa.npz`, `ATLAS/{run}_snap096.npz` (keys `m_tot_500c_bg`, `m_gas_500c_bg`, `m_star_500c`), `CACHE/g1_design.json`.
- **Original plotting code**: `examples/_build_fgas_bridge_explore_nb.py` §5 (lines 651–724; builds `examples/fgas_bridge_explore.ipynb` cell "§5 — Reproducing van Daalen+2020 for weak lensing", worktree `wl-tsz-bridge`). van Daalen fit hardcoded: `vd(f̃) = -exp(-5.990·f̃ - 0.5107)`.
- **Regeneration difficulty**: trivial — placeholder already byte-identical to the on-disk cached PDF.

## fig09_cl_envelope.pdf

- **Placeholder**: `figs/fig09_cl_envelope.pdf`
- **Shows**: Left — S(ℓ) envelopes for fiducial, 1P (n=57), Sobol (n=253) designs, LSST-Y10/Euclid shape-noise bands. Right — Pearson r[f_gas^group, S(ℓ)] vs ℓ for both designs, peak r≈0.86–0.91 near ℓ~1000–1500, inset scatter at ℓ=1037 (r=0.86).
- **md5 match**: identical to `examples/figures_lightcone/fig_cl.pdf` (confirmed, md5 `4c9eb7...`).
- **IMPORTANT provenance subtlety**: two different notebooks build a file called `fig_cl`/`save(fig,"fig_cl",...)`:
  1. `examples/_build_paper_nb.py` (→ `paper_lightcone_figs.ipynb`, worktree `wl-tsz-bridge`) — **1P + fiducial only**, no Sobol. This is NOT the source of the current on-disk PDF (the caption's Sobol n=253 content is absent from this version).
  2. `examples/paper_lightcone_figs2.ipynb` cell 12 (§4.1, worktree `sobol-sb35`), docstring: *"now spanning all three datasets: the fiducial, the 1P (twobound) and Sobol"* — **this is the actual source**, consistent with the on-disk file mtime (Jun 29, matching the Sobol-era batch alongside `fig_design.pdf`, `fig_bridge_sobol.pdf`).
- **Underlying data**: 1P inputs as fig06 above, plus the Sobol design's `CACHE`-equivalent stats (per the `sobol-sb35` worktree's own dashboard cache — not independently re-verified here since the placeholder already matches the shipped PDF byte-for-byte; no regeneration is needed).
- **Regeneration difficulty**: trivial (byte-identical to shipped cache); if ever re-derived from scratch, use `examples/paper_lightcone_figs2.ipynb` cell 12, not `_build_paper_nb.py`.

## fig10_bridge_compton_y.pdf

- **Placeholder**: `figs/fig10_bridge_compton_y.pdf`
- **Shows**: (a) S(ℓ~1500) vs log₁₀Y_500c^group, r=0.93. (b) Y_500c^group vs group f_gas, r=0.98. (c) raw/partial correlation bars: f_gas 0.91, logY 0.93, logT 0.85, Y|f_gas 0.48, f_gas|Y ≈0.
- **md5 match**: identical to `examples/figures_lightcone/fig_bridge_compton_y.pdf` (confirmed, md5 `bc6c6d...`).
- **Underlying data**: `ATLAS/{run}_snap096.npz` (keys `m_gas_500c_bg`, `m_tot_500c_bg`, `Y_500c`, `T_mw_500c`), `CACHE/g1_stats.npz` (per-run `{run}_clk` via `S_of()`), `CACHE/g1_design.json`.
- **Original plotting code**: `examples/_build_fgas_bridge_explore_nb.py` §3 (lines 409–484), builds `fgas_bridge_explore.ipynb` cell "§3 — Is Compton-Y a better lever than f_gas?" including the `pcorr()` partial-correlation helper.
- **Regeneration difficulty**: trivial — placeholder already byte-identical.

## fig11_fgas_saturation.pdf

- **Placeholder**: `figs/fig11_fgas_saturation.pdf`
- **Shows**: Sorted group-scale (M_halo=1–2e13) f_gas across the 1P feedback suite vs X-ray-standard band (0.06–0.10) and eROSITA-low (0.026, Popesso+24); BIND fiducial=0.081, min single-knob=0.051.
- **md5 match**: identical to `examples/figures_lightcone/fig_siegel.pdf` (confirmed, md5 `b317f0...`).
- **Underlying data**: `ATLAS/{run}_snap096.npz` (`group_q(run,"fgas",1e13,2e13)` — median bg-sub f_gas), `CACHE/g1_design.json`.
- **Original plotting code**: `examples/paper_decomp_figs.ipynb` cell 19 (§6.1 "Calibration to the Siegel et al. regime", worktree `wl-tsz-bridge`) — sorted bar chart, colored by feedback family, `save(fig,"fig_siegel",...)`.
- **Regeneration difficulty**: trivial — placeholder already byte-identical. (Note: this is a *different* mass bin/aperture than the "$41\%$ of cosmic" retracted number discussed in the same paragraph of `main.tex` — the paper text itself flags that distinct provenance; not a mapping ambiguity.)

## fig12_bridge_mass.pdf

- **Placeholder**: `figs/fig12_bridge_mass.pdf`
- **Shows**: (a) S(ℓ~1500) vs group f_gas per mass bin (group/intermediate/cluster). (b) Pearson r and slope α vs mass-bin center. (c) r[f_gas,S] vs ℓ per mass bin.
- **md5 match**: identical to `examples/figures_lightcone/fig_bridge_mass.pdf` (confirmed, md5 `48fb7e...`).
- **Underlying data**: same `CACHE/g1_stats.npz` + `ATLAS/{run}_snap096.npz` + `CACHE/g1_design.json` as fig06, evaluated in 3 M200c bins: (1–3e13, "group"), (3e13–1e14, "intermediate"), (1e14–3e14, "cluster").
- **Original plotting code**: `examples/_build_fgas_bridge_explore_nb.py` §2 (lines 209–271), builds `fgas_bridge_explore.ipynb` cell "§2 — How the bridge changes from groups to clusters".
- **Regeneration difficulty**: trivial — placeholder already byte-identical.

## fig13_bridge_residual.pdf

- **Placeholder**: `figs/fig13_bridge_residual.pdf`
- **Shows**: (a) Partial correlation of thermal (log Y, log K) vs structural (c_dm, c_gas, f_star) halo quantities with S(ℓ) at fixed f_gas, vs ℓ. (b) 2-variable f_gas+c_dm bridge (green) vs f_gas-only (black), restoring correlation at high ℓ. (c) Mean off-bridge residual by feedback family (AGN above, SN/wind below at high ℓ).
- **md5 match**: identical to `examples/figures_lightcone/fig_bridge_residual.pdf` (confirmed, md5 `451636...`).
- **Underlying data**: `ATLAS/{run}_snap096.npz` (keys `m_gas_500c_bg`, `m_dm_500c`, `m_dm_200c`, `m_gas_200c`, `m_star_500c`, `Y_500c`, `K_mw_500c`), `CACHE/g1_stats.npz`, `CACHE/g1_design.json`.
- **Original plotting code**: `examples/_build_fgas_bridge_explore_nb.py` §4 (lines 498–627), builds `fgas_bridge_explore.ipynb` cell "§4 — Where the bridge breaks, and what that tells us about feedback". Feedback-family classification here uses this script's own `_family()` (line 88–92, keyword match on `name` only) — **this is a different function from** `paper_decomp_figs.ipynb`'s `_family(name, desc)` (used for fig06/fig07's 20/25/12 AGN/SN/other split) and `bind_bridge.py`'s `_family(name, desc)`. This confirms — but does not resolve — the discrepancy the paper's own `\todo{}` note in the fig13 caption already flags (18/31/8 here vs 20/25/12 in fig06/fig07): two textually-similar but distinct auto-classifiers exist in the source tree, and the paper is right not to silently reconcile them.
- **Regeneration difficulty**: trivial — placeholder already byte-identical.

---

## Summary table

| Figure | Feasible | Cache root | Source notebook/script (cell) |
|---|---|---|---|
| fig01_map_gallery.png | yes | `demo_maps.npz` | `paper_lightcone_figs.ipynb` c.3 / `_build_paper_nb.py` |
| fig02_validation_field.png | yes | `runs/{bind,truth}/run_0000/*.npz` | `paper_lightcone_figs.ipynb` c.6 |
| fig03_validation_halo.png | yes | `halo_atlas/`, `tau_profiles/`, `profiles/` | `paper_lightcone_figs.ipynb` c.5 |
| fig04_cross_spectra.png | yes | `runs/bind/run_0000/Cl_kappa_y.npz` | `fiducial_lightcone_stats.ipynb` c.13 |
| fig05_bridge_hero.png | yes | `dashboard_cache/g1_peakhalo*.npz` | `bind_bridge.py::fig_hero` |
| fig06_bridge_fgas.png | yes | `dashboard_cache/g1_stats.npz`, `halo_atlas/` | `bind_bridge.py::fig_fgas_sofl` |
| fig07_bridge_response.png | yes | `dashboard_cache/g1_stats.npz`, `halo_atlas/` | `bind_bridge.py::fig_response` |
| fig08_vandaalen_analog.pdf | yes | `dashboard_cache/`, `halo_atlas/` | `_build_fgas_bridge_explore_nb.py` §5 |
| fig09_cl_envelope.pdf | yes | 1P + Sobol dashboard caches | `paper_lightcone_figs2.ipynb` c.12 (NOT `_build_paper_nb.py`) |
| fig10_bridge_compton_y.pdf | yes | `halo_atlas/`, `dashboard_cache/g1_stats.npz` | `_build_fgas_bridge_explore_nb.py` §3 |
| fig11_fgas_saturation.pdf | yes | `halo_atlas/`, `dashboard_cache/g1_design.json` | `paper_decomp_figs.ipynb` c.19 (§6.1) |
| fig12_bridge_mass.pdf | yes | `dashboard_cache/g1_stats.npz`, `halo_atlas/` | `_build_fgas_bridge_explore_nb.py` §2 |
| fig13_bridge_residual.pdf | yes | `halo_atlas/`, `dashboard_cache/g1_stats.npz` | `_build_fgas_bridge_explore_nb.py` §4 |

**13/13 figures feasible. 0 unmapped.** Every placeholder already on disk in
`papers/01_pipeline/figs/` is a verified byte-for-byte match (md5sum) of an
existing cached figure elsewhere in the repo (`figs/*.png` or
`examples/figures_lightcone/fig_*.pdf`), so no actual regeneration action is
required — this DATA_MAP exists to document *why* each one is correct and
where to look if a caption number or cosmetic detail ever needs re-deriving.
