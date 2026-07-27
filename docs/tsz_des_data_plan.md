# P4c — Real ACT DR6 tSZ measurement + DES Y3 vs the kSZ-selected feedback subspace

Created 2026-07-24 (branch `analysis/ksz-desi-act-v2`). Successor to
`docs/ksz_lightcone_map_plan.md` (P4b, complete through M6). Two streams:

- **Stream T (tSZ)**: measure stacked Compton-y ourselves from the ACT
  DR6+Planck y-map at real DESI LRG positions — replacing the buggy Liu+2025
  csv (bit-identical photo-z bins) as the data source for the M2 money plot.
- **Stream D (DES WL)**: test whether the kSZ-consistent SB35 nodes (M6's
  `ksz_consistent_nodes.npz`) are also consistent with DES Y3 weak-lensing
  statistics (2pt + mass-map Cl/peaks), i.e. whether the kSZ-selected feedback
  subspace survives an independent probe.
- **Capstone X**: one figure — the kSZ-selected node set confronted with BOTH
  new data vectors (real y-CAP, DES WL) — multi-probe consistency of the
  feedback subspace.

Enactment protocol identical to P4b (§6 there): each phase = one agent with §1
of this file + its own phase entry + the previous verdict JSON; verdicts to
`KS/lightcone/verdicts/` (KS=/mnt/home/mlee1/ceph/bind_science/ksz_confront);
figures to `KS/lightcone/figs/`; no Slurm execution by agents; no recursive
searches on ceph roots. Owners: S=Sonnet (engines/judgment), H=Haiku
(mechanical runs).

---

## 1. Verified inventory (recon 2026-07-24 — do not re-derive; cite this)

`DL = /mnt/home/mlee1/ceph/paper3/B/downloads` (resolves under
/mnt/sdceph/users/mlee1 — user-scoped; non-recursive ls only).

### 1.1 ACT DR6+Planck y-map release (`DL/act_dr6_planck_ymap/`)
- All maps: FITS **CAR** (CTYPE RA---CAR/DEC--CAR), 43200×10320 px,
  **0.5′/px** (CDELT ±0.0083333°), CRPIX (21601, 7561), RA 0–360°,
  Dec ≈ −63°…+23°. Compton-y dimensionless (no BUNIT — expected).
- `ilc_actplanck_ymap.fits` = baseline ILC. CIB-deprojected variants
  `..._deproj_cib_{1.0,1.2,1.4,1.6,1.7,1.8,2.0}_10.7.fits`, `_1.7_24.0`, and
  moment-expansion `_cib_cibdBeta_1.7_10.7`, `_cib_cibdBeta_cibdT_1.7_10.7`
  (1.78 GB each).
- `wide_mask_GAL070_apod_1.50_deg_wExtended.fits` — same grid, Galactic 70% +
  extended point-source mask, 1.5° apodized.
- `ilc_beam.txt` — b_ℓ table ℓ=0..19999. **Verified numerically ≈ 1.6′ FWHM
  Gaussian at all tested ℓ (<0.1%)** → matches BIND's `beam1.6am` shards
  exactly; no beam correction needed (optional exact-b_ℓ filter ≲0.3%).
- No noise/covariance files in the release.

### 1.2 Galaxy catalogs
- `DL/desi_dr1_lrg_spec/LRG_SGC_clustering.dat.fits` — DESI DR1 LRG LSS
  catalog, **SGC only**, 662,492 rows; cols TARGETID, Z (0.400–1.100, median
  0.751), RA, DEC, WEIGHT (=SYS·COMP·ZFAIL), WEIGHT_FKP, … .
  `LRG_SGC_0_clustering.ran.fits` — 4.96M randoms (only index 0 on disk; NGC
  absent — flag, don't fetch).
- `DL/liu2025_tsz_desi/dr9_lrg_pzbins.fits` — Liu's own DR9 photometric LRG
  catalog, 12.39M rows (Main; Extended 33.7M), cols incl. Z_PHOT_MEDIAN,
  EBV, lrg_mask, pz_bin (1–4) — useful for EBV cross-match by TARGETID and a
  later full-sample repeat; its README references subdirs NOT downloaded.
- `DL/act_dr5_szcluster_hilton2009.11043/DR5_cluster-catalog_v1.1.fits` —
  4195 clusters: RADeg, decDeg, SNR, fixed_y_c(+err), redshift, M500c(+err),
  M200m, footprint flags. + search-area mask (same CAR grid) — the
  **high-SNR pipeline-validation target**.
- `DL/dustmaps_data/` — SFD98 only (NGP/SGP ZEA, mag). No Planck HFI dust.

### 1.3 DES Y3 (`DL/des_y3_massmaps_jeffrey2105.13539/`, 2pt FITS)
- Mass maps (Jeffrey+21): HEALPix RING **NSIDE=1024**, single 'T' column.
  `glimpse_{full,tomo1..4}.fits` float32, DES footprint only (11.49% sky,
  UNSEEN elsewhere) + `glimpse_mask.fits` (0/1). `wiener_{full,tomo1..4}`
  float64 full-sky inpainted (variance ~8× lower outside footprint).
  `nullB_full.fits` = B-mode null (float64; NOT in SHA256SUMS — unverified).
  **These are filtered reconstructions (sparsity/Wiener), not raw KS maps —
  caps how quantitative map-level comparisons can be (see D-phases).**
- `DL/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits` — DES Y3
  3×2pt: COVMAT (1000×1000; blocks xip[0:200], xim[200:400], gammat[400:880],
  wtheta[880:1000]); xip/xim 4-bin source pairs × 20 θ bins (2.5–250′);
  nz_source (4 bins, z=0–3, dz=0.01) + **1000 nz_source realisations**;
  nz_lens (MagLim, 6 bins). This is the quantitative 2pt leg + the n(z) for
  weighting BIND's source planes.
- Also on disk from D10: `KS/../ksz_confront/DES_ACT_pandey.fits` +
  `desact_data.npz` (shear×y, joint cov) — optional Tier-X leg.

### 1.4 BIND side (P4b products, all validated)
- `ksz_consistent_nodes.npz` (KS/lightcone/) — being written by M6: node ID
  lists for DESI-precision consistent sets (31 bgs110 / 48 bgs1125) + χ²
  arrays. Node id k ↔ `bind_sb35/runs/run_{k:04d}` ↔ emulator_dataset row
  (run_ids verified aligned).
- Per-run kappa: `kappa_maps.npz` (50,5,1024,1024) f4, z_s={0.5,1,1.5,2,2.44},
  fov 5°, 0.29297′/px; precomputed `Cl_kappa.npz` (5,5,724), `peak_counts`,
  `nongaussian_stats`, `wst`. LRG mock + y-CAP envelope:
  `desi_mock_snap067.npz` (z=0.503, ⟨logM200⟩=13.181 = Sailer+24 anchor),
  `lrgy_beam_lightcone.npz` (fid + 253 nodes, θ grids: fixed-arcmin RAP =
  [1.0,1.625,…,6.0] + xb=linspace(0.3,3,18)·θ200).
- CAP filter conventions: `lightcone_cap_stack.py` (cap_batch; **pixel scale
  is an argument — the real map is 0.5′/px, NOT BIND's 0.29297′**);
  cosmology helpers r200phys/DA in `_build_ksz_paper2_nb.py` §0.

### 1.5 Environment (verified)
- BIND_env: astropy 7.0.1 + fitsio OK; **pixell NOT installed**; healpy
  1.18.0 **broken** (numpy 2.4.6 removed np.in1d). → T0/D0 fixes below.
- `module load python/3.11.11` gives a working healpy/numpy/astropy stack
  (no `bind`) — Stream D runs two-stage: module-python → plain npz →
  BIND_env. Never mix the two stacks in one process. pymaster/treecorr absent
  → no NaMaster mask deconvolution; use the patch-tiling strategy (D2).
- **mas_correct caveat**: strong evidence the SB35 kappa corpus was traced
  WITHOUT `--mas_correct` (CIC-aliasing upturn at high ℓ; fine for internal
  ratios, suspect for absolute DES comparison). D1 quantifies before
  anything absolute is trusted; ELL_TRUST≈1.5e4 cut is the fallback.

---

## 2. Stream T — real tSZ measurement (phases T0–T3)

### T0 (H, 15 min) — tooling
`pip install pixell` into BIND_env; verify `enmap.read_map` on the y-map +
`reproject.thumbnails` on 3 test positions vs astropy-WCS cross-check
(agreement <0.05 px). Do NOT touch healpy here. Verdict T0.json.

### T1 (S) — pipeline validation on ACT DR5 clusters (the decisive gate)
New `examples/act_ycap_measure.py` (the ONE measurement engine, reused by T2):
catalog in → pixell thumbnails (box ≥ 2·√2·θ_max) → repo CAP filter at
0.5′/px → mask handling (require mean mask over the √2·θ_d ring > 0.99;
keep a mask-weighted variant) → stack with weights → bootstrap + spatial
jackknife (~30 RA/DEC cells) covariances → npz.
Validate on `DR5_cluster-catalog_v1.1.fits` (SNR>5, in-footprint): stacked
y-CAP must be strongly detected (S/N ≫ 10), scale with fixed_y_c bins, and a
random-position stack must be null. **Fig V-T1**: cluster stack vs random
null + y_c-bin scaling. Gate: detection + null pass. This proves geometry,
CAP, mask, and covariance on a known signal before touching LRGs.

### T2 (S runs; H for variant sweeps) — the LRG measurement
Sample: DR1 SGC spec LRGs, primary window **z∈[0.4,0.6]** (~160k gals,
matches the mock's z=0.503); robustness window z∈[0.45,0.9]. EBV: cross-match
TARGETID→dr9_lrg_pzbins EBV, cut EBV<0.15 as a variant (record both; the LSS
catalog is already systematics-weighted).
Maps: baseline ILC + `deproj_cib_1.7_10.7` (fiducial bracket pair); H-agent
sweep over the remaining β variants + the two moment variants → CIB
systematic band. Apertures: BOTH grids of §1.4 (RAP fixed-arcmin — the
drop-in replacement for the Liu csv — and xb·θ200 with per-galaxy z,
logM200=13.18 anchor).
Nulls: random-position (from the randoms file, same z window, ~10× data
count) + RA-rotated positions (Dec-preserving, in-footprint redraw). Both
must be consistent with zero at every aperture.
Covariance: report bootstrap AND jackknife (the P6b two-variant lesson).
Output: `KS/lightcone/act_ycap_lrg_real.npz` (θ grids, mean, err, per-variant
axis, nulls, sample metadata). **Fig V-T2**: measurement + nulls + CIB band.
Gate: nulls pass; the baseline−deproj spread is reported, not hidden.

### T3 (S) — the corrected M2 + cross-probe verdict
Swap the Liu loader in `lightcone_m2_ycap_liu.py` for the T2 npz (keep the
old csv as a greyed "superseded (release bug)" series). Overlay: fiducial +
253-node envelope + **the kSZ-consistent subsets colored** (from
`ksz_consistent_nodes.npz`). Report: fid/data ratio per aperture; χ² per node
(data cov ⊕ node realization-mean cov) → tSZ-consistent counts; and the
cross-probe table: P(tSZ-consistent | kSZ-consistent) vs P(tSZ-consistent).
**Fig M2R** (replaces M2 as the money plot) + notebook §4 update + verdict.
Kill-gate KG-T: if the DR5-cluster validation passes but the LRG stack
disagrees with Liu's (buggy-binned but roughly-scaled) values by >3× in
amplitude, suspect the sample/weighting before blaming physics.

---

## 3. Stream D — DES Y3 vs the kSZ-selected subspace (phases D0–D4)

### D0 (H) — prerequisites
(a) Confirm `ksz_consistent_nodes.npz` exists (M6 writes it; else patch
`lightcone_m1_fgas_desiact.py` to save it — 5 lines). Decide set: **report
bgs110 (31), bgs1125 (48), and their intersection** throughout; never a
silent choice. (b) Provenance: two `emulator_dataset.npz` files exist
(bind_sb35/ 105MB newer vs bind_sb35/emulator/ 50MB older) — use the per-run
`Cl_kappa.npz` files directly to avoid the stale-dataset trap; record the
decision. Verdict D0.json.

### D1 (S) — mas_correct / aliasing bias check (BEFORE any absolute number)
On 3 nodes (fiducial + 1 consistent + 1 inconsistent): regenerate one
snapshot's stage1 with `--mas_correct`, rebuild that slab's kappa
contribution OR (cheaper, preferred) quantify in ℓ-space: compare Cl_kappa
against the aliasing model / the ELL_TRUST≈1.5e4 precedent; decide per
statistic the trusted ℓ/smoothing range. Deliverable: the ℓ-cut (expected
ℓ≲6000 for peaks at ≥2′ smoothing — verify) + a bias band. Full 256-node
regeneration is out of scope (describe the SLURM job only if the bias is
fatal). Verdict D1.json + Fig V-D1 (Cl with/without correction).

### D2 (S authors; runs under `module load python/3.11.11`) — DES-side stats
Two-stage design (module-python → npz → BIND_env; §1.5).
(a) **2pt leg (quantitative)**: extract ξ±, COVMAT blocks, nz_source (+1000
realizations) from the 2pt FITS → plain npz. No remeasurement needed — the
official data vector IS the measurement.
(b) **Map leg (exploratory, honest)**: from `glimpse_full/tomo*` (primary;
`wiener_*` as the filter-systematic bracket): tile the footprint interior
(glimpse_mask==1, edge-buffered) with non-overlapping **5°×5° gnomonic
patches at 1024²** (matching BIND geometry exactly) → save patch stacks as
npz. Expect ~150–200 patches. Compute per-patch Cl/peaks/MFs LATER in
BIND_env with bind.inference.stats VERBATIM (identical estimator on data and
sims — the whole point of patch-tiling; no pymaster needed). Patch-to-patch
scatter = the data covariance for map stats.
**Reconstruction-filter caveat (must appear on every map-leg figure)**: the
Jeffrey+21 maps are GLIMPSE/Wiener-filtered; BIND maps are not. Mitigations:
compare at smoothing scales ≥ the reconstruction scale (test ≥5′ and ≥10′),
use GLIMPSE−Wiener spread as a filter-systematic band, and treat the map leg
as consistency-check-grade, not likelihood-grade. Verdict D2.json + Fig V-D2
(patch mosaic + one patch vs one BIND realization side by side).

### D3 (S) — BIND-side forward model
In BIND_env: n(z)-weight the 5 source planes per DES source bin (weights =
interp of nz_source at z_s, normalized — the desact_sheary_realfit.py
pattern; flag the coarse-plane approximation, bracket with the 1000 n(z)
realizations at negligible cost for the 2pt leg). For the map leg: combine
planes per bin → add DES-level shape noise (literature n_eff/σ_e per bin —
Gatti+21 values as constants; **user to confirm or supply**) → smooth
identically to D2 → same estimators → per-node stacks for {all 253} and the
kSZ-consistent sets, 50 realizations each. For the 2pt leg: n(z)-weighted
Cl (from the (5,5,724) Cl_kappa matrices — analytic recombination, cheap) →
Hankel J0/J4 → ξ±(θ) with the D1 ℓ-cut + the P4b XPk normalization fix
heeded. Verdict D3.json.

### D4 (S) — consistency tests + the fixed-cosmology trap
Per statistic (ξ± [quantitative]; patch Cl, peaks, MFs [exploratory]):
three-curve figure — {kSZ-consistent} envelope, {all-node} envelope, DES
data + errors. χ² with cov_DES ⊕ cov_BIND/n_patch-rescaled (the M1 area
rescaling pattern: A_DES/25 deg² effective). **Two variants mandatory**
(SB35 cosmology is fixed at TNG300, σ8=0.816 vs DES-preferred lower S8 —
without this the test is a cosmology test, not a feedback test):
(i) raw fixed-cosmology; (ii) amplitude-marginalized (A~σ8Ωm^0.5 nuisance,
Cκκ∝A², the wl_sz_cosmo_anchor pattern). Report consistent-node counts per
variant per statistic. **Fig M7** (the Stream-D money plot): ξ± panel +
peaks panel, kSZ-consistent nodes colored. Kill-gate KG-D: if even the
amplitude-marginalized ξ± rejects ALL nodes, first suspect the D1 aliasing
treatment and the n(z)-plane approximation — quantify both before claiming a
physics result.

---

## 4. Capstone X (S) — the multi-probe subspace figure

One figure, three panels sharing the node coloring {kSZ-consistent ∩,
kSZ-only, rest}: (a) M2R y-CAP vs real ACT, (b) M7 ξ± (amplitude-marg.),
(c) the M6 parameter-space scatter (top-2 constrained params) with the
multi-probe-surviving nodes highlighted. Caption states the headline: does
the kSZ-selected feedback subspace survive tSZ and WL simultaneously, and
which parameter directions do the survivors occupy? + notebook §6 + WORKLOG.

---

## 5. Decision items (user)

1. D3 shape noise: adopt Gatti+21 n_eff/σ_e literature values? (default yes)
2. T2 EBV cut: report both (default) or hard-cut?
3. mas_correct: if D1 finds a fatal bias, authorize the 256-node re-trace?
   (separate SLURM campaign, ~256×5h — only if needed)
4. NGC LRG + full randoms + Liu imaging-weights subdirs are not on disk —
   fetch later only if SGC-only statistics prove limiting.

## 6. Token/cost notes

T0+T1 ≈ one Sonnet session; T2 sweep = H (one engine call per map variant,
idempotent shards); D2 patch extraction = one module-python script (H runs);
D3/D4 = S. No disBatch needed anywhere (largest loop = 253 nodes × light
stats, workstation-scale). Heavy figures only from reduced npz. Every phase:
verdict JSON ≤2 KB; figures carry their gate in the title.
