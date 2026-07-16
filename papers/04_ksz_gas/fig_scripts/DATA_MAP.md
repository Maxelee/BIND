# DATA_MAP — Paper IV (04_ksz_gas)

Provenance map from every `\includegraphics` in `main.tex` to cached data on
disk. Source notebooks live in the `ksz-desi-act` worktree (branch
`analysis/ksz_project`, checked out at
`/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/wt/ksz-desi-act`):
`examples/paper_ksz_desi_act.ipynb` (built by `examples/_build_ksz_paper_nb.py`)
for figs 01–08, and `examples/paper_ksz_field.ipynb` (built by
`examples/_build_ksz_field_nb.py`) for figs 09–10. All data-file paths below
were confirmed to exist and to contain the expected array keys (checked with
`numpy.load(..., allow_pickle=True).keys()` / `pandas.read_parquet(...).columns`,
read-only, 2026-07-16).

Common constants cell (both notebooks, first code cell): cosmology
(`Om,OL,h`), `DA(z)`/`r200phys` helpers, `PARAMS` (30 CAMELS-TNG astro param
names, filtered against the parquet schema), `PMETA` (physical ranges/log-flags
from the **in-repo** `bind.assets/SB35_param_minmax.csv`, not a cache), the
`DESI` host-mass dict (literature, hardcoded), and the Eckert19/Popesso24
`f_gas(M)` fit functions (hardcoded). None of these need external cache files
beyond the ones listed per figure.

Base cache directories:
- `KS` = `/mnt/home/mlee1/ceph/bind_science/ksz_confront/` (confirmed to exist, `ls` read-only)
- `PARQUET` = `/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet` (1.2 GB, per-halo table, 55 columns incl. `run,snap,z,M200,M_tot_500,M_tot_200,f_gas_500,M_star_500,M_gas_500,M_gas_200,VariableWindVelFactor,WindEnergyIn1e51erg,...`)
- `DS` = `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` (104 MB, ray-traced field-level Sobol dataset, 87 keys)

---

## fig01_lightcone_design_hmf.png

- **Placeholder**: `figs/fig01_lightcone_design_hmf.png`, caption Fig.\ref{fig:design}, 3-panel (a) halo counts/snapshot vs z, (b) 256-node Sobol design in physical wind-energy/wind-speed units w/ fiducial star, (c) HMF at BGS+ELG slices vs DESI host bands + BIND 1e13 floor + patch-reuse band.
- **Dossier match**: `figs_raw/paper_ksz_desi_act/cell004_out0.png` (on-disk equiv `examples/figures_ksz/f1_lightcone.pdf`), §1 DATA.
- **Original plotting code**: `examples/_build_ksz_paper_nb.py` lines ~260–310 (notebook `paper_ksz_desi_act.ipynb`, cell 4, heading "§1 · DATA — the continuous, painted feedback lightcone").
- **Data needed**: `PARQUET` columns `snap,z` (panel a, grouped counts), `run,VariableWindVelFactor,WindEnergyIn1e51erg` (panel b), `snap,M200` (panel c) — all confirmed present. `PMETA`/`DESI` from in-repo assets (no cache).
- **Feasibility**: FEASIBLE — straightforward re-plot from the parquet alone.

## fig02_methods_cap_schematic.png

- **Placeholder**: `figs/fig02_methods_cap_schematic.png`, Fig.\ref{fig:methods}, 3-panel (a) fiducial τ(R/r200c) stack by mass bin, (b) SHMR M*→M200 2D histogram w/ the two DESI M* cuts, (c) CAP filter schematic on a model profile.
- **Dossier match**: `cell006_out0.png` (equiv `f2_pipeline.pdf`), §2 METHODS.
- **Original plotting code**: `_build_ksz_paper_nb.py` lines ~348–388 (cell 6).
- **Data needed**:
  - `KS/bind_tauy_fiducial_snap085.npz` — keys `['x','mbins','a','tau','y','cnts']` ✓
  - `KS/bind_mstar_xprof_snap085.npz` — keys `['x','cuts','nodes','a','tau','y','cnts','mgas_cum','mtot_prof','logM200']` ✓
  - `PARQUET` columns `snap,M200,M_star_500` ✓
  - Panel (c) is a purely synthetic illustrative profile (no data file — analytic `(1+(θ/1.5)²)^-1`).
- **Feasibility**: FEASIBLE.

## fig03_fgas_erosita.png

- **Placeholder**: `figs/fig03_fgas_erosita.png`, Fig.\ref{fig:erosita} — the paper's headline "0% coverage" figure: f_gas,500 vs M500c, 16–84% Sobol band + median + 2.5% strongest-feedback edge vs Eckert19/Popesso24 X-ray curves.
- **Dossier match**: `cell012_out0.png` (equiv `f5_fgas_erosita.pdf`), §5 "Confrontation I". Note: main.tex places this as figure 3 (early, headline result) even though it is notebook cell 12 / dossier position 5 — main.tex figure numbering is by document/narrative order, not notebook cell order (per the comment at main.tex:989).
- **Original plotting code**: `_build_ksz_paper_nb.py` lines ~612–634 (cell 12).
- **Data needed**: `PARQUET` columns `run,snap,z,M_tot_500,f_gas_500` only ✓. `eckert_fgas`/`popesso_fgas` are hardcoded analytic fits (no cache).
- **Feasibility**: FEASIBLE — single parquet query + two analytic curves. This is the most important figure in the paper (§ Results 1) and has the simplest data dependency.

## fig04_response_fan.png

- **Placeholder**: `figs/fig04_response_fan.png`, Fig.\ref{fig:responsefan} — stacked τ(R)/y(R) for all 256 Sobol nodes (BGS mass bin) colored by f_gas,500, fiducial TNG overlay, bottom row = node-spread/fiducial ratio vs radius.
- **Dossier match**: `cell008_out0.png` (equiv `f3_response_fan.pdf`), §3.
- **Original plotting code**: `_build_ksz_paper_nb.py` lines ~417–458 (cell 8).
- **Data needed**:
  - `KS/bind_tauy_xprof_snap085.npz` — keys `['x','mbins','nodes','a','tau','y','cnts']` ✓ (per-node τ/y profile cube)
  - `KS/bind_tauy_fiducial_snap085.npz` (fiducial reference) ✓
  - `PARQUET` columns `run,snap,M_tot_500,f_gas_500` (for the color-by-f_gas normalization) ✓
- **Feasibility**: FEASIBLE. Uses `matplotlib.collections.LineCollection` for the 256-curve fan — straightforward to reproduce.

## fig05_cap_ratio_confront.png

- **Placeholder**: `figs/fig05_cap_ratio_confront.png`, Fig.\ref{fig:capconfront} — the corrected CAP-ratio f̃_gas(θ) fiducial curves (2 DESI M* cuts) + 256-node band + best-fit node vs real DESI×ACT data points.
- **Dossier match**: `cell014_out1.png` (equiv `f6_ksz_confront.pdf`), §6 "Confrontation II" — this is the paper's corrected, final kSZ-amplitude statement (supersedes the retracted cumulative-aperture numbers discussed in the text around main.tex:483-496).
- **Original plotting code**: `_build_ksz_paper_nb.py` lines ~685–717 (cell 14; note dossier file is `cell014_out1.png`, i.e. the *second* output of that cell — the cell prints text then draws the figure, or the plt.show() emits a duplicate output; the plotting code itself is the single block below).
- **Data needed**:
  - `KS/fgas_cap_mstar_snap085.npz` — keys `['xb','fiducial','sb35','node_ids','logM200','cuts','pix','mstar_ap_px']` ✓
  - `KS/desact_zenodo/Fig8_BGS_BRIGHT-20.2_logm11.00.npz` and `...logm11.25.npz` — keys `['th','ratio','yerr','prof_kappa_err','cov_ksz']` ✓ (real DESI×ACT Zenodo data, both M* cuts confirmed present)
- **Feasibility**: FEASIBLE. Chi-square best-fit-node selection uses only these two files plus the `r200phys`/`DA` helper functions (in-repo, no cache).

## fig06_money_forecast.png

- **Placeholder**: `figs/fig06_money_forecast.png`, Fig.\ref{fig:moneyplot} — forecast corner plot of the 2 constrained feedback directions under kSZ-only/tSZ-only/joint SO/CMB-S4×DESI forecast posteriors.
- **Dossier match**: `cell028_out0.png` (equiv `f8_money_forecast.pdf`), §8 MONEY PLOT.
- **Original plotting code**: `_build_ksz_paper_nb.py` lines ~1307–1330 (cell 28). Uses the `corner` package (confirmed importable in `BIND_env`).
- **Data needed**: `KS/ksz_posterior_multiprobe_future.npz` (193 MB) — keys `['params','log_params','prior_lo','prior_hi','chain_ksz','chain_tsz','chain_joint','ratio_ksz','ratio_tsz','ratio_joint']` ✓ — contains the pre-computed MCMC chains for all three forecast scenarios; the figure is a pure re-plot (prior-whitening + eigendecomposition of `chain_joint`, then `corner.corner` on the projected chains), no re-sampling needed.
- **Feasibility**: FEASIBLE. Large file (193 MB) but a direct load; `corner` plotting of ~3×N-sample chains may take a little wall time but is not a re-run of any analysis engine.

## fig07_tsz_pressure.png

- **Placeholder**: `figs/fig07_tsz_pressure.png`, Fig.\ref{fig:tsz} — 2-panel: (a) y-CAP amplitude for all 256+49-consistent nodes + fiducial (corrected DESI-LRG lensing mass) vs ACT×DESI-LRG data with ±0.1-dex mass-systematic band; (b) y-CAP shape normalized at 2.25′.
- **Dossier match**: `cell022_out0.png` (equiv `f6d_tsz_pressure.pdf`), §6d "pressure leg".
- **Original plotting code**: `_build_ksz_paper_nb.py` lines ~999–1058 (cell 22).
- **Data needed**:
  - `KS/ycap_lrg_snap067.npz` — keys `['R','fiducial','fid_msys_lo','fid_msys_hi','sb35','node_ids','z','logM','mass_band']` ✓
  - `KS/tsz_zenodo/fig3.csv` — plain CSV, confirmed present (7276 bytes) ✓ (real ACT×DESI-LRG y-CAP data)
  - `KS/fgas_cap_mstar_snap085.npz` (reused, for identifying the 49 kSZ-consistent node IDs) ✓
  - `KS/desact_zenodo/Fig8_BGS_BRIGHT-20.2_logm11.00.npz` (reused, for the kSZ χ² cross-match) ✓
- **Feasibility**: FEASIBLE. Slightly more involved (cross-matches node IDs between the kSZ and tSZ caches + Spearman correlation via `scipy.stats.spearmanr`), but every input is a cached array — no re-computation of the underlying stacks.

## fig08_latent_2d.png

- **Placeholder**: `figs/fig08_latent_2d.png`, Fig.\ref{fig:latent} — 4-panel: (a) scree plot (2 latents = 97% variance), (b) parameter loadings on the 2 axes, (c)/(d) 2-D latent plane colored by inner-gas/outer-gas f_gas respectively.
- **Dossier match**: `cell010_out0.png` (equiv `f4_latent.pdf`), §4 "the idea — 2-d latent manifold".
- **Original plotting code**: `_build_ksz_paper_nb.py` lines ~508–589 (cell 10). This cell **depends on in-notebook state** from cell 8 (`x, tau, y, nodes, fg` — the response-fan arrays) — i.e. it is not fully self-contained; a regeneration script must first reload `KS/bind_tauy_xprof_snap085.npz` + recompute `fg` from `PARQUET` exactly as cell 8 does (see fig04 above) before running the SVD/rotation logic.
- **Data needed**:
  - `KS/bind_tauy_xprof_snap085.npz` (same file as fig04) ✓
  - `PARQUET` columns `run,snap,M_tot_500,f_gas_500` (for `fg`, as in cell 8) and `run,snap,M200,M_gas_500,M_gas_200,M_tot_500,M_tot_200` (for the inner/outer gas-fraction labels, cell 10 itself) ✓
- **Feasibility**: FEASIBLE, but the regenerator must port ~15 lines from cell 8 (loading `c=np.load(...)`, computing `fg`) in addition to cell 10's own body — flag this dependency in the script header so it isn't silently dropped.

## fig09_field_1d_latent.png

- **Placeholder**: `figs/fig09_field_1d_latent.png`, Fig.\ref{fig:field1d} — field-level companion: (a) scree plot showing component 1 ≈0.95 already exceeds 95%-variance line (component 2 flat), (b) dominant latent vs weak 2nd latent colored by f_gas, (c) parameter loadings.
- **Dossier match**: `figs_raw/paper_ksz_field/cell010_out0.png`, §4 "field response compresses to ~1 latent" — this is `g4_latent.pdf` in the source builder (filenames there are `g1..g8`, mapped to cell numbers via `figs_raw/paper_ksz_field/manifest.json`: cell010 → §4 → `g4_latent.pdf`).
- **Original plotting code**: `examples/_build_ksz_field_nb.py` lines ~292–330 (notebook `paper_ksz_field.ipynb`, cell 10). Depends on the constants-cell load of `E = np.load(DS, allow_pickle=True)` and the `stat()` helper (cell ~90-124), plus `fgas` computed from `E["t__scaling_f_gas__value"]`.
- **Data needed**: `DS` = `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` — confirmed present with all needed keys: `source_redshifts, param_names, run_ids, X_unit, t__scaling_f_gas__value, t__scaling_f_gas__valid, t__suppression__value/valid, a__suppression__ell, t__cl_kappa_y__value/valid, a__cl_kappa_y__ell` (used via the `stat("suppression")`/`stat("cl_kappa_y")` accessor) ✓.
- **Feasibility**: FEASIBLE. Single-file dependency; the constants-cell `stat()` helper and `F_XPK` normalization constant must be ported into the regen script (documented, in-repo code, no external data).

## fig10_field_desY3_act.png

- **Placeholder**: `figs/fig10_field_desY3_act.png`, Fig.\ref{fig:field_desy3} — ray-traced BIND shear×y ξ_γy(θ) (16–84% Sobol band, 2.4′-beam-convolved solid + no-beam dotted) vs real DES Y3×ACT measurement, DES source bins 3 and 4.
- **Dossier match**: `figs_raw/paper_ksz_field/cell012_out0.png`, §5 "Confrontation — BIND shear×y vs the real DES Y3×ACT" — `g5_desact_confront.pdf` in the source builder (cell012 → §5 → `g5_desact_confront.pdf` per the field-notebook manifest).
- **Original plotting code**: `examples/_build_ksz_field_nb.py` lines ~359–392 (cell 12). Reuses `E`/`ZS`/`ell` from the constants cell (same `DS` load as fig09).
- **Data needed**:
  - `DS` = `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` (for `E["a__cl_kappa_y__ell"]` and `stat("cl_kappa_y")`) ✓
  - `KS/desact_data.npz` — keys `['cs_bin1','cs_ang','cs_value','cs_angbin','xip_value','xip_ang','xip_bin1','xip_bin2','xim_value','xim_ang','covmat','nz_z','nz_bins','dv_order']` ✓ (real DES Y3×ACT Gatti/Pandey cross-correlation + n(z) + joint covariance)
- **Feasibility**: FEASIBLE. Needs `scipy.special.jv` for the Hankel transform (`xi_gy`, a short helper defined in-cell, not a cache dependency) and the `F_XPK` constant from the constants cell.

---

## Summary

All 10 figures referenced by `main.tex` are **FEASIBLE** to regenerate from
existing cached data — no placeholder needs to stay as a gap. None require
re-running any analysis engine, notebook, MPI job, or Slurm job; every input
is a pre-computed `.npz`/`.parquet`/`.csv` already on `ceph` (paths above,
all under `/mnt/home/mlee1/ceph/bind_science/ksz_confront/` or
`/mnt/home/mlee1/ceph/bind_sb35/`) or an in-repo asset
(`bind.assets/SB35_param_minmax.csv`).

Two figures (fig08, fig10; also fig09 to a lesser extent) have an
**in-notebook state dependency** on an earlier cell (fig08 needs cell 8's
`fg` computation; fig09/fig10 need the shared constants cell's `E`/`stat()`
helper) rather than being fully self-contained — the regeneration scripts
must port those few lines rather than assume the cached `.npz` alone is
sufficient. This is a script-authoring note, not a data-availability gap.

No figure requires fabricated or substituted data; all data files were
verified to exist and to contain the exact array keys the original plotting
code reads, by direct read-only inspection (`numpy.load(...).keys()` /
`pandas.read_parquet(...).columns`), not by assumption.
