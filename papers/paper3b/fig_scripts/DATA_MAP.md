# Paper 3B figure scripts — data map

House rules: `papers/_tools/FIGURE_STYLE.md` (one script per figure; cached
arrays only; style via `paper_style.setup()`; PDF via `save()`). Run each as
`/mnt/home/mlee1/venvs/BIND_env/bin/python fig_scripts/figNN_slug.py` from
`papers/paper3b/`. Field-style conventions follow the 2026-07-19 survey of
Schaan+21 / Liu+25 / Pandey+25 / Coulton+23 / Jeffrey+21 / Zürcher+22 /
Bigwood+25 / Siegel+25 (data = black points; models = lines; parameter
families = colorbar-coded curves; bands = envelopes/posteriors only; grey
shading = correlated/low-weight regions; per-model χ² in legends).

All inputs live under `B = /mnt/home/mlee1/ceph/paper3/B` (frozen artifacts
are sha-pinned in `analysis/paper3b/stack/frozen.py` and the release-bundle
MANIFEST; nothing here recomputes science).

| fig | shows | inputs (under B) |
|---|---|---|
| fig01_peak_catalogs | ν histograms, Wiener/GLIMPSE + smoothing set | wp1_maps/peaks_{wiener,glimpse}_sm{1,2,5,8}am.npz |
| fig02_stack_gallery | stacked y thumbnails per ν bin + CAP aperture | wp2_measurement/stack_wiener_sm2am_fid.npz |
| fig03_measurement | the frozen ⟨Y_CAP(4′)⟩ vector, W+G, σ_total | wp2_measurement/stack_*_fid.npz + wp3_nulls/sigma_sys_*_decomposed.npz |
| fig04_null_battery | null battery + covariates, worst \|Δ\|/σ vs criteria | wp3_nulls/{b3_scorecard,star_dust_summary}.json |
| fig05_deficit_hero | data vs the whole model manifold (grid colorbar + Sobol hull) | stack + sigma_sys + wp4_mocks/model_grid_tfwiener.npz + wp4_mocks/sobol/model_grid_tfwiener_sb35.npz + grid_tfwiener/bind_run_0000.npz; χ² from wp5_inference/b5_chi2_supplement.json + sobol/b5_sobol_verdict.json |
| fig06_radius_ratio | CAP-radius-resolved data vs model + ratio panels | stack (per-bin radius cov) + grid_tfwiener/bind_run_0000.npz |
| fig07_plane_posterior | (Δln M_gas, Δln T) plane: manifold + posterior contours | model grids + wp5_inference/b5_posterior_wiener{,_pg1024}.npz |
| fig08_sobol_chi2 | pre-registered item-8 verdict: per-unit χ² + thresholds | sobol/b5_sobol_verdict.json + model grids + frozen vectors |
| fig09_ladder_budget | ladder rungs as absorbable deficit factors vs the 3.2–6.6× band | wp5_inference/{transfer_movement,b5_item6_cosmology_bracket}.json + validation_closures/gap6_mask_hole_asymmetry.json (+ constants with artifact provenance in-script) |
| fig10_photo_anchor | DESI LRG × DR6 cumulative CAP profiles vs the (bugged) Zenodo series | wp5_inference/photo_anchor/photo_anchor_summary.json |

**Discussion figures (fig11–fig14, 2026-07-19).** Drafts for the Discussion
section (see the plans repo `projectB/discussion-plans/`). All frozen
post-processing; each was executed + independently verified (workflow Wave-1).
Lead with the data/model ratio; absolute keV/f_gas are prior-conditional.

| fig | shows | inputs (under B) |
|---|---|---|
| fig11_ykappa_specific_energy | Y/κ specific energy: grid response a_g/a_T(ν,r) + deficit + f_gas/T split | wp5_inference/p1_ykappa_specific_energy.json (+ p1_ykappa_figdata.npz) |
| fig12_selfsimilar_entropy_baseline | self-similar/entropy baseline: Δln(f_gas·T) displacement + non-thermal (masking) test | wp5_inference/p4_selfsimilar_entropy_baseline.json |
| fig13_two_component_feedback_mode | two-component decomposition: outer-amplitude floor + extra-central below the full 313-unit envelope | wp5_inference/two_component_feedback_mode.json |
| fig14_latent_radial_zones | (inner,outer) f_gas latent plane: SB35 cloud + twobound diagonal + data (inner ~9× deficit; outer needs τ) | wp5_inference/b5_latent_radial_zones.json |
