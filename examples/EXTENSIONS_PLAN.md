# Paper IV (04_ksz_gas) — reviewer-driven extensions plan

Actionable follow-ups to the two headline figures, derived from a review of
`fig08_latent_2d` (the 2-D feedback latent) and `f6b_lowmass` (the two-redshift
`f̃_gas(M200)` confrontation). Each task is self-contained and scoped so a
**sonnet** (reasoning-heavy) or **haiku** (mechanical) agent can enact it without
further discovery. Read this whole preamble first — it lists every path and
constant the tasks assume.

> Convention (repo): new analysis goes on a **topic branch**, not `main`. Do all
> work below on a branch such as `analysis/ksz-extensions`. Generated artifacts
> (`*.npz/*.npy/*.pdf/*.png/*.log`) are gitignored — do **not** commit them.
> Only commit scripts + this plan + short notes.

---

## 0. Shared context (paths, constants, provenance)

### Source code
- **Fig scripts (workspace, editable):** `papers/04_ksz_gas/fig_scripts/`
  - `fig08_latent_2d.py`  — the latent (SVD + rotation). **START HERE for F1.**
  - `fig05_cap_ratio_confront.py` — kSZ CAP-ratio confront (cosmology helpers
    `r200phys`, `DA`, `ARCMIN`, `F_B` are defined at the top; copy them).
  - `notes_figNN_*.md` — per-figure provenance notes; write a sibling
    `notes_*.md` for every new figure you add.
- **Source notebook builder (read-only, on the ksz worktree):**
  `/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/wt/ksz-desi-act/examples/_build_ksz_paper_nb.py`
  - cell 8 (lines ~417–458): loads `x, tau, y, nodes`; computes `fg`.
  - cell 10 (lines ~508–589): the SVD / latent rotation (mirrored in `fig08`).
  - cell 14 (lines ~685–717): kSZ CAP-ratio confront (mirrored in `fig05`).
  - cell 18 (lines ~855–892): the two-redshift `f6b_lowmass` panel.
  - cell 24 (lines ~1120–1200): **`f6e_latent_data`** — real data projected into
    the latent plane with kSZ-only / tSZ-only / joint χ² posteriors. This is the
    reference implementation for anything involving forward-mapping the latent to
    a data likelihood.
- **Reduction scripts (read-only, same worktree `examples/`):**
  `_reduce_fgas_lowmass.py` (the reuse `f̃_gas(M)`), `_reduce_fgas_cap.py`
  (the CAP-ratio observable), `_reduce_ycap_lrg.py` (the tSZ y-CAP),
  plus `bind.inference.lightcone_transforms`, `bind.inference.paint`,
  `bind.inference.io_gadget`, and `examples/lightcone_lowmass_reuse.py`.

### Cached data
- `KS = /mnt/home/mlee1/ceph/bind_science/ksz_confront/`
  - `bind_tauy_xprof_snap085.npz` — keys `x, mbins, nodes, a, tau, y, cnts`.
    `tau`/`y` are `[node, massbin, radius]`; **`mbins` has all mass bins**
    (BGS bin = index `MB=1`, `logM200∈[13.4,13.8]`).
  - `fgas_cap_mstar_snap085.npz` — CAP-ratio observable (`xb, fiducial, sb35,
    node_ids, logM200, cuts`).
  - `fgas_lowmass_snap085.npz`, `fgas_lowmass_snap046.npz` — the reuse
    `f̃_gas(M)` (`logM, fiducial, truth, sb35, node_ids, mass_edges`).
  - `ycap_lrg_snap067.npz` — tSZ y-CAP (`R, fiducial, sb35, node_ids, z, logM, ...`).
  - `desact_zenodo/Fig8_{BGS_BRIGHT-20.2,ELG_LOPnotqso}_logm*.npz` — digitized
    kSZ data (`th, ratio, yerr, cov_ksz`).
  - `tsz_zenodo/fig3.csv` — digitized Liu+2025 tSZ y-CAP.
- `PARQUET = /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet`
  — per-halo table (`run, snap, M200, M_tot_500, M_tot_200, M_gas_500,
  M_gas_200, f_gas_500`, + 30 astro params).
- **Lightcones (per Sobol node):** `/mnt/home/mlee1/ceph/bind_sb35/runs/run_NNNN/`
  - `snap_XXX/composite_slabMM.npz` — keys `generated_patches [N,3,128,128]`
    (DM/Gas/Stars), `thermo_patches [N,4,128,128]` (compton_y, T, entropy, P_e),
    `halo_centers, halo_masses, halo_r200, box_size, n_slabs`.
  - `tau_maps.npz` — projected **kSZ** lightcone `tau [50, 5, 1024, 1024]`,
    `source_redshifts[5], fov_deg, npix`. **Standalone product** (parallel to
    `y_maps`); use it directly for kSZ stacking.
  - `y_maps.npz` — projected **tSZ** lightcone `y [50, 5, 1024, 1024]`,
    `source_redshifts[5], fov_deg, npix`.
  - `Cl_tau.npz` — `ell, cl_kt, cl_tt, cl_yt` (power spectra; the maps live in
    `tau_maps.npz`).
- **Fiducial lightcone:** `/mnt/home/mlee1/ceph/bind_science/runs/bind/run_0000/`
  (`y_maps.npz, kappa_maps.npz, Cl_kappa_y.npz`, …). **NOTE:** the fiducial run
  has `y_maps` + `kappa_maps` but **no `tau_maps.npz`/`Cl_tau`** — for the kSZ/τ
  fiducial you must either project it from the fiducial `composite_slab*` gas
  channel, or regenerate `tau_maps` for `runs/bind/run_0000`. Truth in
  `.../runs/truth/`.
- **Real DESI/ACT downloads:** `/mnt/home/mlee1/ceph/paper3/B/downloads/`
  - `liu2025_tsz_desi/` — official Liu+2025 tSZ release: `fig3.csv, fig8.csv,
    fig10-13.csv`, `dr9_lrg_pzbins.fits`, `fig2_*_dndz_*.txt`, `quality_cuts.py`.
  - `act_dr6_planck_ymap/ilc_actplanck_ymap_deproj_cib_*.fits` — real y-maps.
  - `act_dr6_cmb/` — real ACT DR6 CMB (kSZ source).
  - `desi_dr1_lrg_spec/` — DESI DR1 LRG spectroscopic catalog.

### Constants (copy verbatim from `fig05`)
```python
F_B = 0.0490 / 0.3089                    # Omega_b/Omega_m
Om, OL, h = 0.3089, 0.6911, 0.6774
C_KMS, H0, ARCMIN = 299792.458, 100*0.6774, 180*60/np.pi
# Ez, DA(z), r200phys(logM200, z) — see fig05 top.
```

### Plotting
- Use `papers/_tools/paper_style.py` (`setup, save, panel_label, COLORS,
  ONE_COL, TWO_COL, TWO_COL_TALL`). No titles (use `panel_label`); BIND=blue,
  data=black/`COLORS["truth"]`, sequential maps = `cividis`.
- Save to `papers/04_ksz_gas/figs/<name>.pdf` (+ preview PNG in `figs_preview/`).

### Definitions (reuse everywhere)
- `f̃_gas ≡ f_gas / F_B`. Inner: `M_gas_500/M_tot_500/F_B`. Outer (shell):
  `(M_gas_200-M_gas_500)/(M_tot_200-M_tot_500)/F_B`.
- CAP filter: disk `r<r200` minus **equal-area** ring `r200<r<√2 r200`, applied
  per channel; `f̃_gas = CAP_gas / Σ_c CAP_c / F_B`.

---

## Task A — kSZ-only latent variant  [F1.1]  ·  agent: **haiku**  ·  difficulty: LOW

**Goal.** Reproduce `fig08` but building the latent from τ (kSZ) alone, to test
whether the 2-D inner/outer-gas structure is a kSZ-only result.

**Method.**
1. Copy `fig_scripts/fig08_latent_2d.py` → `fig_scripts/figA_latent_ksz_only.py`.
2. Change the response matrix to drop the `y` block:
   ```python
   # was: R = np.hstack([log10(tau[:,MB,band]), log10(y[:,MB,band])])
   R = np.log10(np.clip(tau[:, MB, band], 1e-30, None))
   ```
   Leave the standardization, SVD, rotation, and labeling code unchanged.
3. Save as `figs/figA_latent_ksz_only.pdf`. Print `lam[:4]`, the cumulative at 2,
   `r_in_e1`, `r_out_e2`, `r_io`, `R^2`.

**Outputs.** `figA_latent_ksz_only.pdf` + `notes_figA_latent_ksz_only.md`
recording the numbers vs the τ+y baseline (fig08: λ=0.51/0.46, 97%, r=0.95/0.95,
r_io=0.24, R²=0.90/0.96).

**Acceptance.** Script runs top-to-bottom; note states whether (i) still ≥90%
variance in 2 comps, (ii) inner/outer r each still ≥ ~0.85. If a correlation
drops materially, report it (that is a real finding, not a bug).

---

## Task B — mass-resolved latent  [F1.2]  ·  agent: **sonnet**  ·  difficulty: MED

**Goal.** Replace the single BGS mass bin with a mass-resolved treatment so the
latent connects to the multi-mass data points of Fig. 2.

**Method.**
1. New script `fig_scripts/figB_latent_massbins.py`, starting from `fig08`.
2. Inspect `mbins` in `bind_tauy_xprof_snap085.npz` and identify the bins that
   bracket the 5 BGS data-point host masses (`logM200 ≈ 13.36–13.82`).
3. **Variant B1 (robustness):** loop over each mass bin, build the latent
   independently, rotate, and report `r_in_e1`, `r_out_e2` per bin. Plot the two
   correlations vs mass-bin center → shows axis stability across mass.
4. **Variant B2 (joint):** build ONE latent whose response matrix concatenates
   the clean-band profiles across all mass bins
   (`R = hstack over bins of [log τ | log y]`), standardized per column. Re-run
   SVD/rotation. This ties each multi-mass data point to the manifold at its mass.
5. Overlay (as points) where the 5 BGS `f̃_gas` data values fall along the
   `f̃_gas`-calibrated `ê1` axis (reuse the `fg_of_e1` calibration idiom from
   cell 24).

**Outputs.** `figB_latent_massbins.pdf` (2-panel: B1 stability + B2 joint plane)
+ `notes_figB_*.md`.

**Acceptance.** B1 shows whether inner/outer r stays ≥0.85 across bins; B2 runs
and the joint scree is reported. Document any mass bin where the structure breaks.

---

## Task C — non-circular re-presentation of the latent  [F1.3]  ·  agent: **sonnet**  ·  difficulty: LOW-MED

**Goal.** Present the inner/outer-gas result without the "rotate-to-align then
measure-alignment" appearance of circularity.

**Method (no rotation to the gas gradient).**
1. New script `fig_scripts/figC_latent_noncircular.py` from `fig08`, but stop
   after the **raw** SVD (`Z = U*S`, keep `Z[:,:2]`); do **not** rotate to `e1/e2`.
2. Regress each physical quantity on the raw PC1–PC2 plane and report the
   captured-variance `R²`:
   `R²_inner = R²(f_in ~ PC1+PC2)`, `R²_outer = R²(f_out ~ PC1+PC2)`.
   High R² ⇒ the gas fraction lives in the observable plane (the non-trivial,
   non-circular content).
3. Compute the two gradient directions in the raw plane
   (`g_in = [cov(PC1,f_in), cov(PC2,f_in)]`, same for `f_out`) and report the
   **angle between them** (`arccos(ĝ_in·ĝ_out)`); ~90° ⇒ independent DOF.
4. Panel: raw PC1–PC2 scatter with the two gradient arrows drawn; annotate
   `R²_inner`, `R²_outer`, angle, and the inner/outer correlation `r=0.24`.

**Outputs.** `figC_latent_noncircular.pdf` + `notes_figC_*.md` that states, in one
paragraph, the argument: dimensionality (SVD) and the *angle + R²* are
rotation-free and falsifiable; only the "ê1↔inner" label is definitional, and
even it is bounded below 1 unless inner gas lies in the plane.

**Acceptance.** Numbers reproduce the physics (expect `R²`≈0.9, angle near 90°,
consistent with fig08's rotated `r=0.95`, `r_io=0.24`).

---

## Task D — lightcone-native `f̃_gas(M)` / y-CAP(M)  [F2.1]  ·  agent: **sonnet**  ·  difficulty: HIGH

**Goal.** Re-derive the Fig. 2 confrontation by CAP-stacking the **projected
lightcone** at halo positions (same geometry as ACT), replacing the per-halo +
reuse-annulus method.

**Method.**
1. **Get the projected maps.** kSZ: use `tau_maps.npz` (`tau [50,5,1024,1024]`)
   directly — it is a standalone product parallel to `y_maps.npz`, no assembly
   needed. tSZ: use `y_maps.npz` directly. (The `composite_slab*` gas /
   `thermo_patches[:,0]` channels are only needed for a per-slab / off-lightcone
   cross-check, or to build the **fiducial** τ map, which is missing — see
   §0 note.)
2. **Stack + CAP.** For a halo/galaxy sample (start with the FoF centrals used in
   `_reduce_fgas_lowmass.py`, then a DESI-like `M_star` selection), place each on
   the projected map at its `(ra,dec)` pixel, apply the **CAP filter** at
   `θ(r200)=r200phys(logM,z)/DA(z)·ARCMIN`, and record CAP amplitude.
3. **Bin by host mass** to recover `f̃_gas(M)` (kSZ/τ) and `Y-CAP(M)` (tSZ/y);
   compute the fiducial, the 256-node band, and (from `runs/truth`) the truth.
4. **Validate** against the existing `fgas_lowmass_snap085.npz` (per-halo method):
   the two should agree on the trend; document amplitude offsets (aperture/LOS).

**Outputs.** A reduction script `examples/_reduce_fgas_lightcone.py` (new, on the
worktree or repo `data_generation/` style) that writes
`KS/fgas_lightcone_snap085.npz` + `..._snap046.npz`, and a fig script
`fig_scripts/figD_lightcone_confront.py` reproducing Fig. 2's layout from it.
Plus `notes_figD_*.md`.

**Acceptance.** Lightcone `f̃_gas(M)` reproduces the per-halo trend within the
documented aperture offset; the 256-node band brackets the same DESI×ACT points.
**Checkpoint before scaling to 256 nodes:** get it working on node 0 + fiducial +
truth first, print the arrays, then parallelize.

**Gotchas.** `tau_maps`/`y_maps` are `[real, z_source, H, W]` — pick the
source-redshift slice matching the snapshot; CAMELS `p14` bug does not apply here
(SB35, not CV). The **fiducial** run lacks `tau_maps.npz` (§0 note) — handle the
fiducial τ separately. Do not recompute anything that already exists as a cache;
verify keys with `np.load(...).files` first (load big maps with
`mmap_mode='r'`).

---

## Task E — node ↔ line ↔ latent-point cross-check  [F2.2]  ·  agent: **haiku**  ·  difficulty: LOW

**Goal.** Verify and annotate that each SB35 line in the Fig. 2 BGS panel is the
same Sobol node as one point in the latent scatter.

**Method.**
1. Small script `fig_scripts/figE_node_correspondence.py`.
2. Load `fgas_lowmass_snap085.npz` (`node_ids`, `sb35`) and the latent
   `nodes[good]` from the fig08 pipeline.
3. Print: `len(node_ids)`, `len(nodes[good])`, the set difference (nodes dropped
   by the clean-band finiteness cut), and confirm the intersection colors match
   (`f̃_gas` per node from the parquet).
4. Optional: a 2-panel figure — left = Fig. 2 BGS lines, right = latent scatter —
   with 3 example nodes highlighted in the same color on both.

**Outputs.** `figE_node_correspondence.pdf` (optional) + `notes_figE_*.md` stating
the exact node counts and how many (if any) drop out.

**Acceptance.** Note reports the precise correspondence and the dropped-node count.

---

## Task F — upgrade DESI/ACT data to the official release  [F2.3]  ·  agent: **sonnet**  ·  difficulty: MED-HIGH

**Goal.** Replace hand-digitized Zenodo curves with published Liu+2025 products,
and (stretch) an independent tSZ measurement from the raw y-map.

**Method.**
- **F-Tier1 (do first).** Inspect `paper3/B/downloads/liu2025_tsz_desi/fig3.csv`,
  `fig8.csv`, `dr9_lrg_pzbins.fits`, `fig2_*_dndz_*.txt`. Map them to the arrays
  the tSZ leg currently reads from `tsz_zenodo/fig3.csv` (R, y, yerr) and, if a
  covariance is provided, use it. Produce a drop-in replacement npz
  `KS/tsz_liu2025_official.npz` and a variant of `fig07`/cell-24 tSZ leg that
  reads it. Compare digitized vs official (should be close; document deltas).
- **F-Tier2 (stretch, tSZ only).** CAP-stack a real y-map
  (`act_dr6_planck_ymap/ilc_actplanck_ymap_deproj_cib_1.7_10.7.fits`, the fiducial
  CIB-deprojection) on the DESI DR1 LRG positions (`desi_dr1_lrg_spec/`), using
  the `dndz`/pz bins and `quality_cuts.py` from the Liu release. Apply the same
  CAP aperture; produce `Y-CAP(θ)` for the LRG stack. This is an independent
  reproduction of the tSZ data vector.
- **kSZ: OUT OF SCOPE.** A from-scratch kSZ measurement needs per-galaxy velocity
  reconstruction — keep the published `desact_zenodo` kSZ product. State this
  explicitly in the notes.

**Outputs.** `KS/tsz_liu2025_official.npz` (+ optional `KS/ycap_actplanck_lrg.npz`
from F-Tier2), an updated tSZ-leg fig script, `notes_figF_*.md`.

**Acceptance.** Tier1 official points overlay the digitized ones within their
errors; any shift documented. Tier2 (if attempted) yields a Y-CAP profile whose
amplitude is consistent with Liu+2025 fig3.

**Gotchas.** FITS y-maps are large (GB); read with `astropy.io.fits`, work on a
cutout around the DESI footprint, do not load all deprojection variants. Confirm
map WCS/units (dimensionless Compton-y) before stacking.

---

## Task G — tSZ twin of Fig. 2 (Y-CAP vs M, two redshifts)  [F2.4]  ·  agent: **sonnet**  ·  difficulty: MED

**Goal.** The direct tSZ analog of `f6b_lowmass`: `Y-CAP` amplitude vs `M200` at
`z=0.18` (BGS) and `z=1.16` (ELG), band + fiducial + truth vs the tSZ data points.

**Method.**
1. New `fig_scripts/figG_tsz_confront_2z.py`, cloning the two-panel layout of the
   `f6b_lowmass` cell (build script lines ~855–892).
2. y source: either `ycap_lrg_snap067.npz` extended, or (preferred) the
   lightcone/reuse y-CAP by mass from Task D's `thermo_patches[:,0]` / `y_maps`.
   Measure `Y-CAP(M)` for fiducial, 256-node band, and truth at snap 085 and 046.
3. Data: overlay the Liu+2025 tSZ points (Task F official arrays if available,
   else `tsz_zenodo/fig3.csv`) at their host `logM200`, at the matched aperture
   (`r200` for BGS, `~2.8 r200` innermost for ELG — mirror the aperture logic in
   the kSZ `f6b` cell).
4. **Physics caveat in the caption/notes:** `y ∝ n_e T_e` is *pressure*, so the
   y-axis is `Y-CAP` (integrated pressure), NOT `f̃_gas`. Do not relabel it as a
   gas fraction without an explicit temperature model.

**Outputs.** `figG_tsz_confront_2z.pdf` + `notes_figG_*.md`.

**Acceptance.** Two-panel figure renders; band brackets (or is offset from) the
tSZ points with the offset discussed. Note explicitly flags the pressure-vs-gas
distinction.

---

## Suggested order & dependencies

```
A (haiku)  ─┐
C (sonnet) ─┼─ independent, do first (latent-side, cheap)
E (haiku)  ─┘
B (sonnet) ── uses the latent; after A
F-Tier1 (sonnet) ── data upgrade; independent
D (sonnet) ── heavy; unblocks G's preferred y source
G (sonnet) ── after D (or after F for the data), uses tSZ
F-Tier2 (sonnet, stretch) ── after F-Tier1
```

Start every task by (1) branching, (2) `np.load(...).files` on each input to
confirm keys, (3) getting node 0 / fiducial working before scaling to 256.
Write a `notes_*.md` beside every new figure (match the existing style in
`fig_scripts/notes_fig*.md`). Never commit generated `.npz/.pdf/.png`.
