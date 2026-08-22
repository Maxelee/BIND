# The fiducial swap — decision, numbers, text deltas, release statement

**Date:** 2026-08-13 · **Branch:** `papers` · **Status:** decision made, caches + 8 figures rebuilt, 6 items deferred to the author

> **One-line summary.** The released fiducial lightcone was painted on the CAMELS
> cosmology and is retired. Its replacement is `bind_science/runs/twobound/run_0049`,
> one of three independent paints that already sit at fiducial astrophysics *and* the
> correct TNG300 cosmology. No repaint is or will be done. Every halo-level and
> gas-spectrum number in the paper improves; the map-level `S(ℓ)` closure against
> full-hydro TNG300 **gets worse by 4.3× above ℓ=5000**, because the old agreement was
> two errors cancelling. That last item is a science rewrite, not a rebuild.

Working products for everything below live in
`/mnt/home/mlee1/ceph/referee_work/fidswap/` (index: `FIGURES_INDEX.txt`,
manifest: `fidswap_manifest.json`). Scripts live in
`/mnt/home/mlee1/BIND/papers/01_pipeline/referee/work/fs*.py`. Nothing was written
into any campaign tree; no file in `imgs/` was overwritten; no git action was taken.

---

## 1. The decision and its basis

### 1.1 What is being retired, and why

`/mnt/home/mlee1/ceph/bind_lightcone_tng` — the released fiducial lightcone and every
`fid`/`bind` product derived from it — was painted with the **CAMELS SB35** cosmology
conditioning instead of **TNG300's**:

| | Ω_m | σ_8 | Ω_b | h | n_s |
|---|---|---|---|---|---|
| painted (CAMELS, `bind.fiducial_params()`) | 0.3000 | 0.8000 | 0.0490 | 0.6711 | 0.9624 |
| correct (TNG300, `bind.tng300_params()`) | 0.3089 | 0.8159 | 0.0486 | 0.6774 | 0.9667 |

Source: `src/bind/params.py:68` (`fiducial_params`, CAMELS) and `src/bind/params.py:87,96`
(`TNG300_COSMOLOGY`, `tng300_params`). Ω_b/Ω_m was **1.03814× high**, so the painted gas
column is high by that factor and gas/τ plane power by **(Ω_b/Ω_m)² = 1.0777**.

Root cause and permanent fix are already recorded in `docs/WORKLOG.md` (2026-08-13 entry)
and `docs/fiducial_repaint_plan.md`: `run_lightcone_project.sh` wrote
`bind.fiducial_params()` into stage 1; `paint_stages._check_cosmology` now raises on a
mismatch in Omega0/OmegaBaryon/HubbleParam. **The bug is confined to that one tree** —
`bind_science` and `bind_sb35` (Sobol + twobound) were always conditioned correctly,
which is the entire reason a drop-in replacement exists.

### 1.2 Why the twobound trio is a valid replacement (evidence, not assertion)

`bind_science/runs/twobound/run_{0018,0049,0053}` are the three members of the 60-run
twobound (1P) design whose parameter vector happens to *be* the fiducial — the known
"57-vs-60" fact. They are three **independent paints** at identical parameters.

| gate | result | evidence |
|---|---|---|
| **Parameters identical** | `np.array_equal(twobound[r].params, runs/fiducial/run_0000/params.npy)` **True** for exactly r ∈ {18, 49, 53}; max diff exactly 0 | `fidswap_equivalence_audit.npz` |
| **Same substrate** | same shared stage-1 halo cutouts (33 678 halos, 20 snapshots), same checkpoint `weights/fm_redshift_thermo`, same paste settings (taper_frac 0.15, r200_factor 4.0, patch_mass_match True) | run configs |
| **Seed pairing to the DMO arm** | corr(DMO, tb49) = **+0.99977** per realization at low ℓ (notebook guard `_build_figures_nb.py:334` requires > 0.99; tb18 0.99974, tb53 0.99972; retired fiducial 0.99981) | `numbers_fig05_field_tb49.txt` |
| **Seed ladder is realization-count-independent** | `bind/run_0000` κ[0] is **bit-identical** to `bind_lightcone_tng` κ[0], so the trio's 50 realizations *are* the first 50 of the released 550 — paired residuals and `S(ℓ)` work directly against the existing truth/DMO sets | verified in the depmap scope |
| **Convention equivalence** | **bit-identical** across `bind/run_0000`, tb18/49/53 and truth: ℓ grid (724 pts, 86.91–52133.57), `mf_nu` (29), peak ν (68), nu05 ν (22), peak_cross ν/r_arcmin, smoothing scales (1,2,5,8′), `halo_mass`/`r200` (33 678), z_s = [0.5,1,1.5,2,2.44], fov 5 deg, npix 1024, σ_e 0.26, `nu_norm='map'` | `fidswap_equivalence_audit.npz` |
| **Stats are drop-in** | identical keys/shapes for `Cl_kappa.npz`, `nongaussian_stats.npz`, `peak_counts.npz`, `nu05_stats.npz`, `Cl_kappa_y.npz`, `halo_scaling.npz`. The trio has **more**: it also ships `Cl_tau.npz` and `paired_stats.npz`, which `bind/run_0000` never had | directory listing + load |
| **Reduction path is the shipped path** | re-running `build_atlas_200c.reduce_file` on `runs/truth/run_0000` reproduces the shipped `halo_atlas/truth_snap096.npz` **bit-for-bit** on all 22 keys (max rel diff exactly 0); same for the recomposite against the released fiducial's stored `composite`/`alpha`/`patch_scales`/`scale_global` | `atlas_truth_CHECK_snap096.npz`, `composites/` |
| **Estimators reproduce the published paper** | running the identical pipelines on the **retired** fiducial gives Y/τ/M★ = 1.0374 / 1.0649 / 0.9429 (main.tex:502 quotes 1.037 / 1.065 / 0.943) and median \|resid\| kk/ky/yy = 0.65 / 1.36 / 5.60 % (abstract quotes 0.7 / 1.4 / 5.6 %) | `numbers_fig03_halo_tb49.txt`, `numbers_fig05_field_tb49.txt` |

The one grid that is *not* shared is `nongaussian_stats.npz` `pdf_bins`, which is
adaptive by construction (`src/bind/inference/stats.py:502`,
`edges = linspace(-6σ, +6σ, 42)` with σ = the run's own global κ std). Trio-internal
drift is 0.0008–0.0037 bin widths; trio-vs-`bind` 0.081; trio-vs-truth 0.185 — ~20×
smaller than the drift between extreme twobound bound pairs, and the paper's PDF panels
use the nu05 grid anyway. **Not a blocker**; state the convention and move on.

### 1.3 Which run is canonical — and why not the mean

**Canonical fiducial: `/mnt/home/mlee1/ceph/bind_science/runs/twobound/run_0049`.**
Runs 0018 and 0053 are quoted only as the paint-to-paint envelope. Every rebuild script
honours `BIND_FID_REPLICA=tb18|tb49|tb53`, so re-pointing is one environment variable.

*Why one paint and not the 3-mean.* What a user downloads is one paint, so a single
paint is the honest fiducial. A 3-mean would suppress stochastic texture by √3 and bias
every texture-sensitive statistic (PDF, peaks, MFs, C_ℓ^yy) toward the ensemble mean —
understating exactly the paint stochasticity the paper elsewhere invokes as an
irreducible floor — and the paper's covariance (Eq. 8, main.tex:558–563) is estimated
from 550 **single**-realization hydro-pasted truth sets, inside which a 3-averaged
fiducial would sit inconsistently.

*Why run_0049 specifically.* Ranking all three by RMS z-score against the trio mean over
21 statistic blocks (24 `S(ℓ)` bands, C_ℓ^κκ at 5 z_s, C_ℓ^yy, C_ℓ^κy, C_ℓ^ττ, C_ℓ^κτ,
the six nu05 statistics, three moment sets, halo medians, the 8 latents): run_0049 has
mean z **0.7377** vs 0.8431 (0018) and 0.8418 (0053), and is closest to the trio mean in
**14 of 21** blocks vs 3 and 4.

*The honesty check that decided it.* run_0018 gives by far the **best** family-model
fiducial closure (RMS 0.0020, trough residual +0.0022) while run_0049 gives the
**worst** (RMS 0.0056, trough residual −0.0085). Picking 0018 would be indistinguishable
from tuning. run_0049 is simultaneously the **most typical** and the **least favourable
of the three** for the paper's own headline closure claim, so the choice cannot be read
as cherry-picking. (`closure_fidswap.npz`, `lam_fidswap_v2.npz`.)

*One carve-out.* The mass-dependent **debias/calibration templates** (Fig 3
Δ_Y/Δ_fgas/Δ_M★; Fig 11 r(a)) are estimators of an expectation, not of a realization.
Average the three replicas there and say so.

### 1.4 What was rebuilt (all gates passed)

| product | path | cost (1 core) | gate |
|---|---|---|---|
| snap-096 recomposite, 3 runs × 4 slabs, 3 mass + 4 thermo + alpha | `composites/tb{18,49,53}_snap096_slab{00..03}.npz` (6.8 GB) | 31 s/run | bit-for-bit vs released fiducial's stored composite |
| halo atlas, run_0049 **all 20 snapshots** + tb18/tb53 snap096 | `halo_atlas/fidtb{RR}_snap{NNN}.npz` | 2 m 46 s | bit-for-bit vs shipped `truth_snap096.npz` |
| per-realization field cache (kk, yy, ky, tt, kt, peaks, minima, PDF) ×3 replicas | `field_cache/field_stats_fidtb{RR}.npz` | ~3 min/replica | bit-for-bit vs shipped `field_stats_fid.npz` |
| extended-ν MF cache (ν ∈ [−3,8], 45 thresholds) ×3 | `mf_cache/mf_nu8_snap096_fidtb{RR}.npz` | 2 m 15 s/replica | bit-for-bit vs shipped `mf_nu8_snap096.npz` |
| per-halo radial profiles, run_0049 | `profiles/perhalo_fidtb49_snap096.npz` | 8.6 s | paired guards vs truth PASS, 2933 halos |
| nu05 per-realization shard + ngal=10 peaks | `nu05_shards/{sci_bind_tb49,peak_counts_ngal10_tb49}.npz` | 172 s | — |
| R1 ladder bind rung | `r1/{kappa,y,tau}_bind.npz` | ~5 min | 4 unaffected rungs hard-linked from `referee_work/r1/` |

Total ≈ 45 min on one core. **No GPU, no Slurm, no repaint, no re-trace.**

Two provenance corrections the gates turned up, both worth keeping:

1. `figs_preview/amplitude_sets.npz`'s own `notes` string claims `v0/v1/v2__fid_measured`
   came from `mf_cache` interpolated onto the canonical grid. **They did not.** The
   mf_cache route misses the shipped curves by up to 1.4 % (v1) / 3.6 % (v2) of curve
   scale on 8–9 bins; the **nu05 shard** (`bind_sb35/nu05_shards/sci_bind.npz`)
   reproduces them bit-for-bit, and is also what the Sobol design targets use. Fix the
   `notes` string when the file is rebuilt.
2. Notebook `fig11` panel (b) is **broken for both fiducials** (pre-existing, unrelated
   to the swap): it divides `runs/bind/run_0000/Cl_kappa_y.npz` — which carries the
   pre-xpkfix `XPk_plane` normalization, measured 1.2402e7 low — by the correctly-normed
   truth cache, so it plots −100 % at every ℓ. Fix by reading the field cache's `ky` legs
   (what Figs 5 and 6 already do), not by swapping the fiducial.
   (`numbers_fig11_zclosure_numbers_tb49.txt`.)

---

## 2. What moved

All values below are for the canonical replica **run_0049** unless a replica spread is
quoted. "OLD" always means the retired `bind_lightcone_tng` fiducial, computed with the
*same* estimator in the *same* script, so the comparison is apples-to-apples.

### 2.1 Halo level — everything improves

Median BIND / hydro-pasted-truth over 2933 halos at snap 096 (z = 0.0337), 2000-sample
bootstrap SE on ln.
**Source:** `numbers_fig03_halo_tb49.txt`; caches `halo_atlas/fidtb49_snap096.npz` vs
`bind_science/halo_atlas/{fid,truth}_snap096.npz`. **Figure:** `imgs/fig03_halo_validation_fidswap.png`.

| quantity | OLD | **NEW (tb49)** | tb18 | tb53 | significance OLD → NEW |
|---|---|---|---|---|---|
| Y_200c | 1.0374 ± 0.0057 | **1.0062 ± 0.0062** | 1.0075 | 1.0173 | 6.4σ → **1.0σ** |
| τ_200c | 1.0649 ± 0.0023 | **1.0199 ± 0.0019** | 1.0205 | 1.0191 | 27.8σ → **10.6σ** |
| M★,200c | 0.9429 ± 0.0056 | **0.9656 ± 0.0056** | 0.9565 | 0.9571 | 10.5σ → **6.2σ** |
| f_gas,200c (bg-sub) | 1.0624 | **1.0154** | 1.0166 | 1.0145 | — |
| f_gas,500c (bg-sub) | 1.0739 | **1.0243** | 1.0222 | 1.0196 | — |
| Y_500c | 1.0542 | **1.0228** | 1.0224 | 1.0306 | — |
| T_mw,500c | 0.9871 | **0.9958** | 0.9935 | 1.0019 | — |
| P_e,mw,500c | 1.0974 | **1.0854** | 1.0698 | 1.0827 | — |

Mass-dependent calibration templates (linear in log M, pivot 13.5):

| template | OLD | **NEW** |
|---|---|---|
| Δ_Y | +3.42 % − 5.95 %/dex | **+0.51 % − 6.24 %/dex** |
| Δ_τ | +6.34 % − 4.01 %/dex | **+1.91 % − 1.97 %/dex** |
| Δ_M★ | −5.05 % + 5.96 %/dex | **−3.01 % + 3.47 %/dex** |
| Δ_fgas (bg-sub) | — | **+1.50 % − 3.40 %/dex** |

Post-correction binned-median residuals (NEW): max\|f_gas\| **2.0 %**, max\|Y\| **16.1 %**
(the 16 % is the 3-halo top bin — quote it with that caveat or restrict the fit range).

### 2.2 Radial profiles — the τ bias essentially disappears; the DM control gets *better*

Mean BIND/truth of the median profile over annuli inside R_200c, four mass bins.
**Source:** `numbers_fig04_profiles_tb49.txt`; cache `profiles/perhalo_fidtb49_snap096.npz`.
**Figure:** `imgs/fig04_radial_profiles_fidswap.png`.

| channel | 13.00–13.40 | 13.40–13.78 | 13.78–14.18 | 14.18–15.00 |
|---|---|---|---|---|
| τ OLD | 1.080 | 1.057 | 1.037 | 1.040 |
| **τ NEW** | **1.035** | **1.013** | **1.008** | **1.005** |
| y OLD | 1.062 | 1.016 | 0.990 | 1.024 |
| **y NEW** | **1.024** | **0.996** | **1.003** | **0.992** |
| M★ OLD | 0.686 | 0.758 | 0.804 | 0.886 |
| **M★ NEW** | **0.708** | **0.759** | **0.832** | **0.920** |
| DM control OLD | 0.993 | 0.990 | 0.995 | 0.994 |
| **DM control NEW** | **0.999** | **0.998** | **1.005** | **1.002** |

The DM control moving *toward* unity is a free correctness check on the whole rebuild:
nothing in the swap should have touched it, and it did not degrade.

### 2.3 Map level — mixed, and this is where the paper has to change

**Source:** `numbers_fig05_field_tb49.txt`, `fig05_numbers.npz`; caches
`field_cache/field_stats_fidtb49.npz`, `mf_cache/mf_nu8_snap096_fidtb49.npz`,
`nu05_shards/sci_bind_tb49.npz`. **Figure:** `imgs/fig05_field_validation_fidswap.png`.

Median \|BIND/truth − 1\| over ℓ ∈ [300, 5000], z_s = 1:

| statistic | OLD | **NEW** | direction |
|---|---|---|---|
| C_ℓ^κκ | 0.65 % | **1.18 %** | **worse** |
| C_ℓ^κy | 1.36 % | **1.78 %** | **worse** |
| C_ℓ^yy (total) | 5.60 % | **3.76 %** | better |
| C_ℓ^ττ | 11.22 % | **2.75 %** | **much better** |
| C_ℓ^κτ | 6.79 % | **2.70 %** | **much better** |
| C_ℓ^yτ | — | **1.0 %** | new (truth-validated for the first time) |

Seed-paired diagonal χ²/dof:

| | OLD | **NEW** |
|---|---|---|
| κκ, and S(ℓ) (identical by construction) | 11.8 | **140.8** |
| κy | 145.2 | **151.0** |
| yy | 1162.5 | **1180.6** |
| PDF | 3.93 | **7.06** |
| N_pk / N_min | 0.01 / 0.01 | **0.00 / 0.01** |
| V0 / V1 / V2 | 47.75 / 33.18 / 33.95 | **35.01 / 72.95 / 54.13** |
| ττ / κτ / yτ | — | **309 / 128 / 213** |

New full-covariance Hartlap χ²/dof: κκ 155, κy 96, yy 363. Unpaired T² : κκ 2.7,
κy 11.9, yy 49.5. ngal=10 shape-noise peaks median \|resid\| **2.1 %**.

> **Caveat on a stale stamp.** `FIGURE_NUMBERS.md` records "kk 19, S(ℓ) 19" for the OLD
> fiducial; today's code gives **11.8**. That stamp is from Slurm job 2454934
> (2026-07-30), predating the ELL_TRUST 1.5e4 → 3e4 move and the nu05 migration. The
> median-\|resid\| values *do* reproduce to the digit, so the estimator is right and the
> χ² stamp is superseded. Do not quote the old χ² numbers.

### 2.4 Gas spectra — the payoff, and the figure gains content

Because the replica ships its **own seed-paired τ cube** (`bind/run_0000` never had one)
and `runs/truth/run_0000/tau_maps.npz` exists, panels (c) C^ττ, (e) C^κτ, (f) C^yτ render
as **truth-validated closure panels** instead of Sobol-spread strips.
**Source:** `numbers_fig06_spectra_tb49.txt`, `fig06_numbers.npz`.
**Figure:** `imgs/fig06_spectra_validation_fidswap.png`.

Signed mean BIND/truth:

| leg | OLD 300–1000 / 300–5000 | **NEW 300–1000 / 300–5000** |
|---|---|---|
| ττ | 1.0830 / 1.1113 | **1.0012 / 1.0267** |
| κτ | 1.0444 / 1.0682 | **1.0053 / 1.0278** |
| yτ | — | **0.9559 / 0.9874** |
| yy (total) | — | **0.9179 / 0.9584** |
| κy | — | **0.9549 / 0.9898** |
| κκ | — | **1.0035 / 1.0114** |

**C_ℓ^ττ NEW/OLD = 0.9239** over ℓ 300–5000 and **0.9244** over 300–1000, against the
theory prediction **(Ω_b/Ω_m)² = 0.9279**. The cosmology fix lands where theory says it
should — this is the single cleanest confirmation that the diagnosis was right.

**Consequence for the text:** the τ and y legs must now be quoted **separately**. The
swap essentially removes the τ bias (+8.3 % → +0.1 % at ℓ 300–1000) but leaves the y
deficit (BIND/truth 0.918 at 300–1000), and it costs κ (0.65 % → 1.18 %).

### 2.5 `S(ℓ)` closure against full-hydro TNG300 — **this got worse**

24 canonical bands, z_s = 1, paired-50 DMO denominator
(`runs/dmo/run_0000/Cl_kappa_paired.npz`). **Source:** `validation_fidswap.npz`.

| ℓ | S OLD | **S NEW** | S truth | BIND−truth OLD | **BIND−truth NEW** |
|---|---|---|---|---|---|
| 1044 | 0.9974 | 0.9976 | 0.9916 | +0.0058 | **+0.0061** |
| 1265 | 0.9955 | 0.9972 | 0.9892 | +0.0063 | **+0.0080** |
| 3302 | 0.9590 | 0.9656 | 0.9526 | +0.0064 | **+0.0130** |
| 4847 | 0.9337 | 0.9442 | 0.9263 | +0.0074 | **+0.0180** |
| 8619 | 0.8935 | 0.9101 | 0.8870 | +0.0065 | **+0.0231** |
| **12651 (trough)** | **0.8812** | **0.9016** | **0.8765** | **+0.0047** | **+0.0251** |
| 22497 | 0.8986 | 0.9266 | 0.9018 | −0.0032 | **+0.0248** |

RMS \|S − S_truth\|: **0.0055 → 0.0093** for ℓ < 5000; **0.0057 → 0.0242** for ℓ ≥ 5000
(**4.3× worse**). Degradation sets in above ℓ ≈ 860 and grows monotonically.

**Mechanism.** The old +3.8 % excess Ω_b/Ω_m painted extra gas, whose extra small-scale
suppression masked BIND's *intrinsic* under-suppression. With the correct cosmology,
BIND under-suppresses by ≈ 2.5 % at the trough. **No cache rebuild recovers this.** The
paper's sub-percent `S(ℓ)` validation claim must be rewritten.

Paint-to-paint scatter on `S(ℓ)` is **0.016–0.094 %** per band (median 0.052 %,
`validation_fidswap.npz` `S_replicas`), i.e. ~50× smaller than the degradation. **The
degradation is real, not paint noise.**

### 2.6 R1 three-rung ladder — the response-capture number moves

**Source:** `r1_numbers_z1_tb49.json` vs the untouched shipped
`referee/work/r1_numbers_z1.json`. **Figure:** `imgs/fig06b_full_hydro_fidswap.png`.

| quantity | OLD | **NEW** |
|---|---|---|
| C_ℓ^ττ bind/pasted, ℓ<1000 | 1.0834 | **1.0017** |
| C_ℓ^ττ bind/pasted, trusted | 1.1562 | **1.0595** |
| C_ℓ^ττ bind/hydro_full, trusted | 0.6561 | **0.6022** |
| C_ℓ^κκ bind/pasted, trusted | 1.0021 | **1.0251** |
| C_ℓ^y bind/pasted, trusted | 1.0956 | **1.1259** |
| S(ℓ) trough, bind rung | 0.8767 | **0.8986** (pasted 0.8722, hydro_full 0.8235 — unchanged) |
| **response capture S(ℓ), bind** | **0.6918** | **0.5385** |
| response capture S(ℓ), **pasted** | **0.7035** | **0.7035 — UNCHANGED** |
| response capture pdf / peaks / minima, bind | 1.0963 / 0.6587 / 0.6029 | **0.9751 / 0.5908 / 0.4647** |

The memory-noted "**halo replacement = 70.4 % of full-hydro S(ℓ)**" is the *pasted* rung
and is **fiducial-free — it survives verbatim**. What moves is the *BIND* rung: BIND now
captures **54 %** rather than 69 % of full-hydro TNG300's `S(ℓ)` response.

### 2.7 The eight shipped latents — all move ≤ 0.31 σ_design

Reducer = `r5_common.measure_lambda` = `family_basis_all.py` L408–457 verbatim; it
reproduces the shipped `agnostic_lambda_results_obs.npz` `X_FID[chosen]` to 1.4e-17.
**Source:** `lam_fidswap_v2.npz`.

| latent | OLD | **tb49** | trio mean | σ_design | **Δ/σ (tb49)** | Δ/σ (trio) | design pct (tb49) |
|---|---|---|---|---|---|---|---|
| f̃_★[13.2–13.4] | 0.09824 | **0.10220** | 0.10000 | 0.08431 | **+0.047** | +0.021 | 67.2 % |
| log T̃[13.0–13.2] | 6.60332 | **6.60266** | 6.60514 | 0.02782 | **−0.023** | +0.065 | 29.7 % |
| log Ỹ[13.0–13.2] | −7.36031 | **−7.37936** | −7.37086 | 0.13588 | **−0.140** | −0.078 | 24.2 % |
| log P̃_e[14.0–14.3] | −12.48609 | **−12.50431** | −12.49215 | 0.18921 | **−0.096** | −0.032 | 23.8 % |
| c_gas[14.0–14.3] | 0.74916 | **0.74451** | 0.74217 | 0.02250 | **−0.207** | −0.310 | 30.5 % |
| log Ỹ_ss[13.4–13.6] | −29.05502 | **−29.06645** | −29.06687 | 0.07260 | **−0.157** | −0.163 | 31.2 % |
| c_gas[13.2–13.4] | 0.63028 | **0.62744** | 0.62700 | 0.05573 | **−0.051** | −0.059 | 23.8 % |
| log P̃_e[13.4–13.6] | −13.24779 | **−13.25723** | −13.25051 | 0.26410 | **−0.036** | −0.010 | 23.8 % |

Largest single-replica move is `c_gas[14.0–14.3]` at −0.45 σ on tb53. Design percentiles
for tb49 span **23.8–67.2 %**, so main.tex:888's "interior to the design cloud" survives
but the quoted **24th–65th** range becomes **24th–67th**.

### 2.8 Family-model fiducial closure — improves 5×, and the residual flips sign

Measured curves from the new caches; prediction from the shipped bundle.
**Source:** `closure_fidswap.npz`.

| | RMS(pred − meas), clk | trough measured | trough predicted | residual | closure err / suppression depth |
|---|---|---|---|---|---|
| **OLD fiducial** | 0.01049 | **0.8812** | **0.8950** | **+0.0139** | 11.7 % |
| **tb49 (canonical)** | **0.00555** | **0.9016** | **0.8932** | **−0.0085** | 8.6 % |
| tb18 | 0.00197 | 0.9017 | 0.9039 | +0.0022 | 2.2 % |
| tb53 | 0.00292 | 0.9027 | 0.8989 | −0.0038 | 3.9 % |
| trio-mean λ | — | 0.9020 | **0.8986** | **−0.0034** | 3.5 % |

Full-band RMS(pred − meas) improves for **every** statistic: clk 0.0105 → 0.0056,
pk 0.363 → 0.211, mn 0.398 → 0.148, v0 1.11e-4 → 7.3e-5, v1 2.24e-5 → 4.4e-6,
v2 1.90e-6 → 6.9e-7 (pdf flat at 0.023).

**The key structural statement:** the trough **prediction** spans 0.8932–0.9039 across
the three replicas (sd **0.0054**) while the **measurement** spans only 0.9016–0.9027
(sd 0.0006). The residual (−0.0085 for tb49, −0.0034 for the trio mean) is now
**comparable to or smaller than the paint noise in the model leg**. The family model
therefore closes on the out-of-design fiducial to within paint stochasticity — which
**retires the "trough miss = λ information ceiling" diagnosis** in project memory. The
surviving 0.025 map-level gap is cleanly a BIND-vs-hydro model bias, not a
latent-coverage limit.

> **Two anchors to reconcile before quoting anything.**
> (a) My old *measured* trough is **0.8812** (verified two ways, agreeing with the
> notebook's own `band_reals` estimator to 4.1e-4). Neither 0.8905 (task brief) nor
> 0.8765 reproduces — **0.8765 is the *truth* curve's trough**, so the brief conflated
> them. (b) My old *predicted* trough is **0.8950**, not main.tex:895's **0.898**. Either
> the text predates a cache refresh or a different bundle was used. Resolve (b) against
> the executed notebook's printed value before the sign-flip story is written up, or it
> rests on an unverified baseline.

### 2.9 Fig 10 covariation — spreads unchanged, levels move

A common denominator cancels out of a spread-of-ratios, so **every panel's 16–84 spread
is bit-identical**. What moves is the level.
**Source:** `numbers_fig10_covariation_tb49.txt`, `fig10_numbers.npz`.
**Figure:** `imgs/fig10_covariation_fidswap.png`.

| statistic | median curve/fid OLD | **NEW** | fiducial NEW/OLD |
|---|---|---|---|
| suppression | 1.0292 | **1.0121** | 1.0326 |
| C_ℓ^ττ | 1.0125 | **1.0967** | **0.9252** |
| C_ℓ^κτ | 1.1465 | **1.1894** | 0.9674 |
| C_ℓ^yτ | 1.1170 | **1.1726** | 0.9504 |
| C_ℓ^yy | 1.1196 | **1.1098** | 1.0007 |
| C_ℓ^κy | 1.1533 | **1.1519** | 1.0065 |
| six ν statistics | — | — | 1.0000–1.0048 |

**Narrative change worth a sentence.** For C_ℓ^ττ the fiducial used to sit essentially
*on* the Sobol median (1.01); it now sits **9.7 % below** it. That is a real, quotable
statement about where TNG's fiducial feedback lives inside the design, and it was
previously masked by the Ω_b/Ω_m error.

### 2.10 Hero-figure stamps and the z-ladder

Hero (`numbers_fig02_hero_tb49.txt`, `imgs/fig02_hero_fidswap.png`): mean τ (total
column) **1.8026e-03 → 1.7369e-03** (×0.9635); mean y **1.0447e-06 → 1.0277e-06**
(×0.9837); κ rms painted 0.01834 → 0.01841 (DMO 0.01870); Δκ rms weak/strong
9.32 %/9.78 % → **9.50 %/9.51 %** of κ rms. Pixel corr(new, old) = 0.9960 (κ) / 0.9955
(τ) — same sky, different paint.

z-ladder (`numbers_fig11_zclosure_numbers_tb49.txt`, `fig11_numbers.npz`; **figure
[PENDING]**, see §5): Y_500c BIND/truth **1.0542 → 1.0228** at z = 0.034, rising to
1.304 → **1.284** at z = 2.00; f_gas ladder range 1.068–1.142 → **1.020–1.090**;
a-factor exponent α = −0.184 → **−0.188** (a pure missing a-factor would give −1).
Refitted `quadratic_a` debias templates: Y c0/c1/c2 1.5628/−1.1351/0.6358 →
**1.5153/−1.0773/0.5957**; f_gas 1.2139/−0.2998/0.1558 → **1.1745/−0.3297/0.1799**.

### 2.11 Summary: what got worse

Stated plainly so it is not buried.

| got worse | OLD → NEW | source |
|---|---|---|
| **`S(ℓ)` closure vs full-hydro truth, ℓ ≥ 5000** | RMS 0.0057 → **0.0242** (4.3×) | `validation_fidswap.npz` |
| **`S(ℓ)` trough gap** | +0.0047 → **+0.0251** | `validation_fidswap.npz` |
| **BIND response capture of full-hydro S(ℓ)** | 0.6918 → **0.5385** | `r1_numbers_z1_tb49.json` |
| median \|resid\| C_ℓ^κκ | 0.65 % → **1.18 %** | `fig05_numbers.npz` |
| median \|resid\| C_ℓ^κy | 1.36 % → **1.78 %** | `fig05_numbers.npz` |
| paired χ²/dof κκ (and S(ℓ)) | 11.8 → **140.8** | `numbers_fig05_field_tb49.txt` |
| χ²/dof V1, V2 | 33.2 / 34.0 → **72.9 / 54.1** | `numbers_fig05_field_tb49.txt` |
| C_ℓ^yy in ℓ 300–1000 (signed) | 0.854 → **0.839** vs truth | `fig06_numbers.npz` |
| family-model trough residual, tb49 | +0.0139 → **−0.0085** (smaller, but sign-flipped) | `closure_fidswap.npz` |
| P_e,mw,500c significance | 7.1σ → **7.9σ** | `halo_atlas/fidtb49_snap096.npz` |

---

## 3. Text deltas for `main.tex`

`main.tex` was **not edited** (per instruction). Line numbers are as of this session's
read of `papers/01_pipeline/main.tex` (1053 lines). Each block gives the current text
verbatim, then a drafted replacement. Anything still gated on a deferred rebuild is
marked **[PENDING]**.

---

### 3.1 Abstract — realization count and the LSST-Y10 claim (l.200)

**Current (l.200, two clauses):**

> "…each with seed-matched convergence, Compton-$y$, and optical-depth map triplets at
> five source redshifts, and $1000$ realizations each. At the fiducial parameters, the
> generated maps reproduce their hydro-pasted truth within LSST-Y10-like precision for
> every WL statistic."

**Problem.** (a) The canonical fiducial carries **50** realizations, not 550 and not
1000; only the DMO and hydro-pasted truth arms carry 550. (b) The LSST-Y10
indistinguishability claim is now **weaker for C_ℓ^κκ**: paired χ²/dof went 11.8 → 140.8
and the median \|resid\| 0.65 % → 1.18 %.

**Replacement:**

> "…each with seed-matched convergence, Compton-$y$, and optical-depth map triplets at
> five source redshifts and fifty realizations each, with 550-realization DMO and
> hydro-pasted reference sets for covariance estimation. At the fiducial parameters the
> generated maps reproduce their hydro-pasted truth to $1.2\%$ in $C_\ell^{\kappa\kappa}$
> and to $1.8\%$ and $3.8\%$ in the $\kappa y$ and $yy$ spectra over $300 \le \ell \le
> 5000$, and match the $\nu$-domain weak lensing statistics to within LSST-Y10-like
> precision."

> **Author decision required.** If the "1000 realizations" claim is meant to survive, it
> refers to the `bind_n1000` campaign (see §4), whose **BIND arm inherits the bug** and
> would need re-painting. Either drop the number, or scope it explicitly to the truth/DMO
> arms.

---

### 3.2 Methods — Sec 2.3 five-set list, the Fiducial bullet (l.422)

**Current (l.422):**

> "\item Fiducial set: Each halo is generated with the fiducial TNG300 parameters to
> create the lightcone. This can then be validated against the hydro-pasted set."

**Replacement:**

> "\item Fiducial set: Each halo is generated with the 30 fiducial TNG300 astrophysical
> parameters, conditioned on the TNG300 cosmology ($\Omega_m = 0.3089$, $\sigma_8 =
> 0.8159$, $\Omega_b = 0.0486$, $h = 0.6774$, $n_s = 0.9667$), to create the lightcone.
> Because three of the thirty parameters have their fiducial value at a prior bound, this
> parameter vector is also a member of the 1P set below; the released fiducial is one of
> those members, so the 1P responses of Eq.~\ref{eq:response} are internally
> self-consistent by construction. This set is validated against the hydro-pasted set in
> \S~\ref{sec:validation}."

**Also update l.428**, which currently reads "In all of the above, the cosmological
parameters are fixed to the TNG300 fiducial values, and only the 30 astrophysical
parameters are varied." — that sentence is now *true as written*, and should stay, but
consider adding a footnote pointing at the conditioning guard
(`bind.inference.paint_stages._check_cosmology`) as the mechanism that enforces it.

---

### 3.3 Methods — the seed-sharing sentence, which is separately wrong (l.430)

**Current (l.430):**

> "Each halo has a unique random seed for flow matching, used to generate it a single
> time per parameter space location, and that seed is shared across all lightcones, such
> that differences between lightcones are purely parameter-driven."

**Problem.** This is **not true**. `src/bind/model.py:454` draws the flow-matching
initial state with an **unseeded** `torch.randn(...)`; no per-halo generator is
constructed anywhere in the sampling path. The three twobound replicas at *byte-identical*
parameters produce measurably different maps, which is direct experimental proof (§6).
Differences between lightcones are parameter-driven **plus** an irreducible paint
stochasticity, which we can now quote.

**Replacement:**

> "The lightcones share their dark matter substrate exactly: the same rotations,
> translations and projections of the snapshots, the same halo catalog, and the same
> replaced halos. The generative step, however, is not seeded --- BIND draws its
> flow-matching initial state from an unseeded normal --- so each lightcone carries an
> independent realization of the model's sampling noise in addition to its parameter
> shift. We measure that noise directly from three independent paints at identical
> parameters (\S~\ref{sec:val_map}): it contributes $0.02$--$0.09\%$ per band to
> $S(\ell)$, a median $0.4$ times our own 50-realization paired standard error, and
> $2$--$13\%$ of the design standard deviation to the eight halo latents of
> \S~\ref{sec:latents}. It is therefore subdominant to, but not negligible against, the
> parameter responses reported below. The result is 314 lightcones at different parameter
> space locations that use BIND, a single lightcone from the TNG300-Dark simulation, and
> a single lightcone that has replaced the TNG300-Dark halos of $M_{200c}\geq 10^{13}\,
> h^{-1}\,{\rm M}_\odot$ with their true hydrodynamical field counterparts."

---

### 3.4 Halo-level ratios (l.502)

**Current (l.502, the quantitative clause):**

> "The median per-halo ratios are $1.037\pm0.006$ for $Y_{200c}$, $1.065\pm0.002$ for
> $\tau_{200c}$, and $0.943\pm0.006$ for $M_{\star,200c}$, where the uncertainties are
> bootstrap errors on the median. We compute the significance of these deviations with
> bootstrap resampling and find them to be unambiguous --- $6.4\sigma$, $27.6\sigma$, and
> $10.4\sigma$, respectively --- however, the halo-to-halo scatter is far larger, at
> $23\%$, $10\%$, and $23\%$."

**Replacement:**

> "The median per-halo ratios are $1.006\pm0.006$ for $Y_{200c}$, $1.020\pm0.002$ for
> $\tau_{200c}$, and $0.966\pm0.006$ for $M_{\star,200c}$, where the uncertainties are
> bootstrap errors on the median. Bootstrap resampling gives significances of
> $1.0\sigma$, $10.6\sigma$ and $6.2\sigma$ respectively: the integrated Compton $Y$ is
> now consistent with the hydro-pasted halos, while the small $\tau$ excess and stellar
> deficit remain resolved. In all three cases the halo-to-halo scatter is far larger, at
> $23\%$, $10\%$ and $23\%$, so these biases are absorbed into the halo-to-halo scatter
> in the actual lightcones."

**Provenance:** `numbers_fig03_halo_tb49.txt`; the 23/10/23 % scatters were re-checked and
barely move (leave as is). Figure to swap in: `imgs/fig03_halo_validation_fidswap.png`.

---

### 3.5 Radial profiles (l.513)

**Current (l.513, three claims):**

> "…the $\tau$ profiles sit a few percent high at all radii, most significantly at small
> radii in the lowest mass bin, while the $y$ profiles show a comparable bias only in the
> inner regions of the lowest mass bin. The stellar channel is the clear exception:
> although its integrated mass is only $\sim6\%$ low, the $\Sigma_\star$ profiles sit
> $25$--$40\%$ low in individual annuli… As a control, the dark matter channel --- which
> BIND also generates --- agrees with the hydro-pasted halos at the [half-percent] level"

**Problems.** (a) The τ excess is now ≤ 3.5 % and **decreases monotonically with mass**
(1.035 / 1.013 / 1.008 / 1.005). (b) "$\sim6\%$ low" for the integrated stellar mass is
now **3.4 %** (median ratio 0.966), and the profile deficit is **8–29 %**, not 25–40 %,
on the corrected fiducial — the old "25–40%" does not reproduce with this estimator for
*either* fiducial and should be restated from the figure. (c) "[half-percent]" resolves:
the DM control is **0.998–1.005**, i.e. within 0.5 % in every bin.

**Replacement:**

> "…the $\tau$ profiles sit at most $3.5\%$ high, with the excess confined to the lowest
> mass bin and falling monotonically with halo mass ($1.035$, $1.013$, $1.008$, $1.005$
> from the lowest to the highest bin); the $y$ profiles are consistent with the
> hydro-pasted halos to $2.4\%$ or better in every bin. The stellar channel is the clear
> exception: although its integrated mass is only $3.4\%$ low, the $\Sigma_\star$ profiles
> sit $8$--$29\%$ low in individual annuli, meaning BIND misplaces stellar mass radially
> far more than it underproduces it… As a control, the dark matter channel --- which BIND
> also generates --- agrees with the hydro-pasted halos to better than $0.5\%$ in every
> mass bin ($0.998$--$1.005$), confirming that the offsets above are channel-specific
> rather than a property of the pipeline."

**Provenance:** `numbers_fig04_profiles_tb49.txt`. Figure:
`imgs/fig04_radial_profiles_fidswap.png`.

---

### 3.6 Map-level validation — the realization inventory (l.516)

**Current (l.516):**

> "…the fiducial and hydro-pasted sets contain 50 pseudo-independent maps each from which
> we extract summary statistics, with a further 550 realizations of the fiducial,
> hydro-pasted, and DMO sets available for covariance estimation."

**Problem.** The corrected fiducial carries **50** realizations only. Eq. 8's covariance
is estimated from the 550 **hydro-pasted truth** realizations, which are unaffected by
the bug — so the science is intact, but the sentence is now factually wrong.

**Replacement:**

> "…the fiducial and hydro-pasted sets contain 50 pseudo-independent maps each from which
> we extract summary statistics, with a further 550 realizations of the hydro-pasted and
> DMO sets available for covariance estimation. All covariances quoted below (Eq.~
> \ref{eq:cov}) are estimated from the 550 hydro-pasted realizations."

---

### 3.7 Map-level validation — the indistinguishability claim (l.574, l.576)

**Current (l.574):**

> "…we see clearly that the BIND-generated statistics match the hydro-pasted maps to
> within LSST-like precision for all weak lensing statistics."

**Current (l.576):**

> "This is an important finding: it represents, to our knowledge, the first set of
> generated lightcones that self-consistently produce weak lensing statistics that are
> statistically indistinguishable from their hydro-pasted truth at the precision of an
> LSST-Y10-like survey."

**Problem.** Paired χ²/dof for C_ℓ^κκ is **140.8** on the corrected fiducial (11.8 before);
the ν-domain statistics remain excellent (N_pk 0.00, N_min 0.01, PDF 7.06, V0 35.0) but
V1 and V2 rise to 72.9 and 54.1. The blanket "all weak lensing statistics" claim no
longer holds for the power spectrum.

**Replacement (l.574):**

> "…the BIND-generated $\nu$-domain statistics --- the convergence PDF, peak and minima
> counts, and the Minkowski functionals --- match the hydro-pasted maps to within
> LSST-Y10-like precision, with median residuals of $0.9\%$, $1.3\%$, $0.7\%$ and
> $0.25$--$0.94\%$ respectively. The convergence power spectrum agrees to a median
> $1.2\%$ over $300\le\ell\le5000$, which our seed-paired errors resolve
> ($\chi^2/{\rm dof}=141$ on 25 bins): at the paired precision of 50 shared-seed
> realizations the residual is a measured, scale-dependent offset rather than noise, and
> we characterize it in \S~\ref{sec:val_map} and Fig.~\ref{fig:s_ell_closure}."

**Replacement (l.576):**

> "This is, to our knowledge, the first set of generated lightcones that self-consistently
> reproduce the morphological weak lensing statistics of their hydro-pasted truth at
> LSST-Y10-like precision across a full 30-dimensional astrophysical parameter space. The
> power spectrum is reproduced to the percent level but not to our paired precision; we
> quantify that residual, and its relation to full-hydro TNG300, below."

> **[AUTHOR DECISION]** This is the single most consequential rewrite in the document.
> The honest framing is that BIND's *morphological* fidelity is what is
> survey-indistinguishable, and its *power-spectrum* fidelity is percent-level. Section
> §3.9 below carries the companion `S(ℓ)` rewrite.

---

### 3.8 Gas spectra — the `[QUANTIFY]` placeholder and the Y–M rescaling test (l.586, l.589)

**Current (l.586, tail):**

> "…we see that the thermodynamic spectra, both auto and cross, are more biased:
> [QUANTIFY from figure --- e.g., within $\sim$X\% at $\ell\lesssim$Y, rising to
> $\sim$Z\% approaching $0.8\,\ell_{\rm Ny}$]."

**Replacement:**

> "…we see that the thermodynamic spectra separate cleanly by channel. The optical-depth
> spectra are now unbiased in the trusted range: $C_\ell^{\tau\tau}$ agrees with the
> hydro-pasted truth to $0.1\%$ over $300\le\ell\le1000$ and $2.7\%$ over
> $300\le\ell\le5000$ (median $|{\rm resid}|$ $2.8\%$), and the $\kappa\tau$ cross
> spectrum to $0.5\%$ and $2.8\%$ respectively (median $2.7\%$). The Compton-$y$ spectra
> carry a genuine deficit: $C_\ell^{yy}$ sits at $0.918$ of truth over
> $300\le\ell\le1000$ and $0.958$ over $300\le\ell\le5000$ (median $|{\rm resid}|$
> $3.8\%$), and $C_\ell^{\kappa y}$ at $0.955$ and $0.990$ (median $1.8\%$). The $y\tau$
> cross spectrum, validated here against a hydro-pasted truth for the first time, agrees
> to $1.0\%$."

**Current (l.589):**

> "We test this directly: rescaling the $y$ maps by the measured $Y$--$M$ offset moves the
> mid-$\ell$ residual in the wrong direction ([$-6.2\%$] measured against a [$+11.1\%$]
> prediction) and worsens the fit everywhere ([$\chi^2/{\rm dof}$ of $46 \rightarrow
> 353$] in the mid-$\ell$ range)."

**Status: [PENDING] — this must be re-run, not re-numbered.** Both legs are
fiducial-derived: the $+11.1\%$ prediction is the halo-level $Y$ offset, which drops from
$+3.7\%$ to $+0.6\%$ ($Y_{200c}$ median 1.0374 → 1.0062), and the $-6.2\%$ is the
map-level mid-ℓ residual. With the halo-level $Y$ offset now consistent with zero, the
"two distinct systematics" argument may have to be reframed entirely: there is no longer
a meaningful halo-level amplitude offset for the map-level $y$ deficit to *fail* to
explain. Suggested skeleton once re-run:

> "A natural hypothesis is that this map-level deficit is the halo-level amplitude offset
> of \S~\ref{sec:val_halo} propagated to the maps. That hypothesis is now excluded by
> construction: the halo-level $Y_{200c}$ offset is $+0.6\pm0.6\%$, consistent with zero,
> while the map-level $C_\ell^{yy}$ deficit is $[8.2]\%$ at $\ell\sim500$. The two cannot
> be the same systematic. The map-scale deficit is instead a texture effect in how the
> generated gas is distributed across the maps, which we localize in \S~[…]."

Run `fsfig6_spectra.py` with the rescaling test re-enabled to fill the bracketed value.

---

### 3.9 The `5–10\%` gas claim (l.592, and the same claim at l.950)

**Current (l.592):**

> "Moreover, the biases --- [$5$--$10\%$] in the trusted regime --- are an order of
> magnitude below the feedback response range across the Sobol prior, which is of order
> unity for the gas spectra, so the maps resolve feedback responses at high contrast."

**Replacement:**

> "Moreover the biases are now channel-specific and small: $2.7\%$ for
> $C_\ell^{\tau\tau}$, $2.7\%$ for $C_\ell^{\kappa\tau}$, $1.0\%$ for $C_\ell^{y\tau}$,
> and $3.8\%$ for $C_\ell^{yy}$, as median absolute residuals over
> $300\le\ell\le5000$. These are one to two orders of magnitude below the feedback
> response range across the Sobol prior --- of order unity for the gas spectra, with
> $16$--$84$ spreads of $1.24$ ($yy$), $1.59$ ($\tau\tau$), $1.21$ ($\kappa y$) and $1.56$
> ($\kappa\tau$) --- so the maps resolve feedback responses at high contrast. We note that
> an earlier version of this release conditioned the fiducial paint on the CAMELS rather
> than the TNG300 cosmology, inflating $\Omega_b/\Omega_m$ by $3.8\%$ and hence the gas
> plane power by $(\Omega_b/\Omega_m)^2 = 7.8\%$; the maps analysed here use the correct
> TNG300 conditioning, and the measured $C_\ell^{\tau\tau}$ ratio between the two is
> $0.924$, matching the $0.928$ predicted by that factor."

**Provenance:** `numbers_fig06_spectra_tb49.txt`, `numbers_fig10_covariation_tb49.txt`.

> **Note.** The last sentence is optional but strongly recommended if any version of
> `bind_lightcone_tng` was ever circulated; see §4.

---

### 3.10 New paragraph: the `S(ℓ)` closure against full hydro (insert near l.594)

There is currently **no** sentence in `main.tex` reporting the BIND-vs-full-hydro `S(ℓ)`
comparison at the corrected fiducial, and one is now required — the referee R1 ladder
figure (`imgs/fig06b_full_hydro_fidswap.png`) is where the change is largest.

**Drafted insertion:**

> "Finally, we compare the suppression $S(\ell)$ of the BIND maps not only against the
> hydro-pasted maps but against a full-hydro TNG300 lightcone, which additionally carries
> the diffuse gas outside the replaced halos. The hydro-pasted set --- halo replacement
> alone --- recovers $70\%$ of full TNG300's suppression response, the remainder being
> the diffuse component our architecture does not paint (\S~\ref{sec:caveats}). BIND
> recovers $54\%$: it reproduces the halo-replacement geometry but under-suppresses
> within it, by $2.5\%$ at the trough ($S=0.902$ against the hydro-pasted $0.876$ at
> $\ell\simeq1.3\times10^4$). The under-suppression grows monotonically above
> $\ell\simeq10^3$, from $0.6\%$ at $\ell\sim10^3$ to $2.5\%$ at the trough, with an RMS
> of $0.009$ below $\ell=5000$ and $0.024$ above. Three independent paints at identical
> parameters agree on $S(\ell)$ to $0.05\%$ per band, so this is a systematic of the
> generative model and not sampling noise. Because it is smooth in $\ell$ and enters every
> statistic as a common factor, it is absorbed almost entirely by the ratio and rank
> statistics used throughout \S\S~\ref{sec:astro}--\ref{sec:emulator}, but it is the
> honest limit on absolute suppression predictions from this release."

---

### 3.11 Sec 5.4 — the fiducial-closure numbers (l.884–912)

**Current (l.886–888):**

> "The fiducial simulation is not a Sobol node --- three of its one-sided parameters sit
> at the very corner of the prior [verify count] --- yet its measured latents are interior
> to the design cloud (24th--65th percentile)."

**Replacement** (the count is confirmed as **3**: VariableWindSpecMomentum, UVBH0Deltaz,
UVBHepDeltaz — verified by `np.array_equal` against the twobound parameter table; and the
framing should be *strengthened*, not hedged, because the fiducial is now literally one
of the design's own members):

> "The fiducial simulation is not a Sobol node --- three of its parameters sit at a prior
> bound, which is why it appears as a member of the 1P set rather than as an independent
> design point --- yet its measured latents are interior to the design cloud
> (24th--67th percentile)."

**Current (l.889–893):**

> "Predicting from its eight measured numbers, the median absolute deviation is $0.7\%$
> for $S(\ell)$, $0.1$--$2.3\%$ for the counts and Minkowski functionals, and $11\%$ for
> the $\kappa$-PDF"

**Status: [PENDING]** — these come from the `*__fid_measured` legs of
`figs_preview/amplitude_sets.npz`, which must be refreshed (see §5, item 3). The full-band
RMS(pred − meas) improvements are already measured (`closure_fidswap.npz`): clk
0.0105 → 0.0056, pk 0.363 → 0.211, mn 0.398 → 0.148, v0 1.11e-4 → 7.3e-5, v1 2.24e-5 →
4.4e-6, v2 1.90e-6 → 6.9e-7, pdf 0.023 → 0.023. Expect every number in this sentence to
improve except the PDF, whose 11 % is a normalization-convention offset that the swap does
not touch.

**Current (l.894–905), the headline closure:**

> "At the deepest point of the $S(\ell)$ suppression trough the model predicts $S = 0.898$
> against a measured $0.881$ --- a $0.9\sigma$ draw of $\sigma_{\rm pred}$ there…, an error
> of $14\%$ of the suppression effect. For calibration, a hand-picked four-latent model…
> predicts $0.920$ at the same point, and a two-bin eight-latent variant we explored
> before the agnostic search predicts $0.908$: pushing the latent search fully agnostic…
> converts the trough from a $1.4\sigma$ miss into a sub-$1\sigma$ one."

**Replacement** (measured values from `closure_fidswap.npz`; the ladder rungs 0.920 /
0.908 are **[PENDING]** because `family_model_section_figs.py:354-356` hardcodes them as
literals and they must be re-derived from the new λ under
`figs_preview/family_model_bundle_{4lat,2bin8}.npz`):

> "At the deepest point of the $S(\ell)$ suppression trough the model predicts
> $S = 0.893$ against a measured $0.902$ --- an error of $8.6\%$ of the suppression
> effect, and a residual smaller than the $0.005$ scatter the model's own prediction shows
> across three independent paints of the same parameters. In other words the out-of-design
> closure is now limited by the generative model's paint stochasticity rather than by the
> latent map: averaging the three paints, the prediction is $0.899$ against a measured
> $0.902$, a residual of $0.003$ or $3.5\%$ of the suppression effect. Across the full
> $\ell$ range the RMS prediction error is $0.0056$, and it improves for every one of the
> seven statistics relative to a prediction made at the fiducial's previously published
> latents. For calibration, a hand-picked four-latent model (budget, partition,
> concentration, temperature in a single group bin --- the a-priori set our earlier
> sections would have suggested) predicts $[{\rm PENDING}]$ at the same point, and a
> two-bin eight-latent variant we explored before the agnostic search predicts
> $[{\rm PENDING}]$."

**Current (l.906–912), the diagnosis:**

> "The residual gap is consistent with a design-coverage limitation rather than a
> flexibility one… the indicated remedy is simulations near the fiducial's latent
> neighborhood, not a bigger model."

**Replacement** — this diagnosis is now **retired** by the measurement, and the honest
replacement is stronger:

> "The residual is no longer resolved against the model's own paint-to-paint scatter, so
> we do not attribute it to a coverage or a flexibility limitation: with the corrected
> fiducial the latent map closes on the single out-of-design point to within the
> stochasticity of the generative step itself. What remains resolved is the separate,
> larger gap between BIND and full-hydro TNG300 at the same point ($0.025$ at the trough,
> \S~\ref{sec:val_map}); that is a property of the painting model, not of the latent
> description of it. The corollary is that a sharper test of the latent map requires
> either more out-of-design points or repeated paints at each one, not a richer map."

> **[BLOCKER — resolve before writing this section.]** My re-derivation of the **old**
> prediction using `family_model_section_figs.py`'s own recipe against the current
> `figs_preview/{amplitude_sets,agnostic_lambda_results_obs}.npz` gives **0.8950**, not
> the **0.898** in l.895. The old *measurement* reproduces exactly (0.8812). Either the
> text predates a cache refresh or a different bundle was used. The sign-flip story rests
> on this baseline.

---

### 3.12 Conclusions (l.948, l.950)

**Current (l.948):**

> "The release comprises $314$ BIND lightcones (256 Sobol nodes, the 57-run 1P atlas, and
> the fiducial), the DMO and hydro-pasted references, $\kappa$, Compton-$y$, and $\tau$
> map triplets at five source redshifts and fifty realizations each, the 550-realization
> covariance sets…"

**Replacement:**

> "The release comprises $313$ BIND lightcones (256 Sobol nodes and the 57-run 1P atlas,
> of which the fiducial is itself a member), the DMO and hydro-pasted references, $\kappa$,
> Compton-$y$, and $\tau$ map triplets at five source redshifts and fifty realizations
> each, the 550-realization DMO and hydro-pasted covariance sets…"

> **Note on the arithmetic.** Under the swap the fiducial is no longer a 314th independent
> lightcone; it *is* one of the 1P members (`twobound/run_0049`). The notebook's own
> data-products table already says "the three bound-at-fiducial runs are excluded as
> duplicates of the fiducial" (`audits/_build_figures_nb.pre-edits0805.py:6417`) — that
> statement becomes **exactly** true instead of approximately true. Pick 313 or 314 and
> make the manifest agree; do not leave them inconsistent.

**Current (l.950):**

> "At the fiducial parameters, the generated maps reproduce the hydro-pasted truth to
> within the precision of an LSST-Y10-like survey for every weak lensing statistic, and to
> $[5$--$10\%]$ in the trusted regime for the gas spectra, whose residuals we show arise
> from a map-scale texture systematic distinct from the small halo-level amplitude
> offsets. …the families are mediated by four halo properties…"

**Replacement:**

> "At the fiducial parameters, the generated maps reproduce the hydro-pasted truth to
> within the precision of an LSST-Y10-like survey for the morphological weak lensing
> statistics, to $1.2\%$ for the convergence power spectrum, and to $1$--$4\%$ in the
> trusted regime for the gas spectra --- with the optical-depth channel now unbiased at
> the percent level and a residual $yy$ deficit that we show is a map-scale texture
> systematic and not a halo-level amplitude offset. …the families are mediated by eight
> halo properties measured in one aperture convention at one epoch…"

> **Separate stale item, independent of the swap:** l.950 says "**four** halo properties"
> while the shipped model uses **eight** (l.779, Table~\ref{tab:latents}). Fix regardless.

---

### 3.13 Numbers explicitly verified as **unaffected** — do not touch

| main.tex | claim | why it survives |
|---|---|---|
| l.642 | "$C_\ell^{\tau\tau}$ against VarWindVelFactor… $\rho = -0.68$" | rank correlation over 256 Sobol nodes; no fiducial denominator. (Its own separate `[verify]` flag about the trusted-range restriction still stands.) |
| l.775–779 | search scores 0.913 / 0.916 / 0.900 | "the fiducial simulation is never touched by the search" (l.777) — verified true |
| l.808, l.843–846, l.863–869 | 30-param CV R² ≈ 0.6 vs 0.96; σ_pred 0.003 / 0.019; Table 2 spans and CV R² | 256-run Sobol design at the correct TNG300 cosmology throughout |
| l.880–882 | median absolute residual 0.3 % for S(ℓ), 0.0–0.1 % for ν-domain, over 7 held-out Sobol nodes | leave-one-out on Sobol only |
| Eq. 8 covariance (l.558–563) | 550 hydro-pasted realizations | no BIND conditioning anywhere in that arm |
| Figs 1, 8, 9, 20g, 20i, pfig_s4a_cv_buildup, pfig_s4b_generality / app_zs / app_epoch | — | Sobol-only or shared-stage-1 products; ship as-is |

---

## 4. The release statement

### 4.1 What ships as the fiducial lightcone

**`bind_science/runs/twobound/run_0049`** — 50 realizations × 5 source planes × 1024²,
`kappa_maps.npz` / `y_maps.npz` / `tau_maps.npz` plus the full statistics set, already on
disk and already seed-paired to the truth and DMO arms. **Zero compute required.** It
also *supersedes* the old `runs/bind/run_0000` in content: it adds `Cl_tau.npz` and
`paired_stats.npz`, which that run never had.

Sizes (measured): `bind_lightcone_tng` **350.25 GB** (full composites, dmo, alpha,
lensplanes); `twobound/run_0018` **15.41 GB** (patches + maps + stats, **no** composites);
`bind/run_0000` **1.93 GB**.

**Option (A) — deprecate in place. Recommended.** Keep `bind_lightcone_tng` for
provenance, banner it, ship run_0049's maps as the fiducial. Cost: 0.

**Option (B) — re-release.** Promote run_0049 to a full lightcone tree, which requires
recompositing its 20 snapshots (patches only on disk) and re-tracing lensplanes. Hours to
days of cluster time, author-submitted. Only worth it if the release must ship composites
and lensplanes for the fiducial, which no figure in this paper needs.

### 4.2 The 350 GB mis-conditioned tree

Do **not** delete it — it is the provenance record for anything already circulated. Add a
banner at the tree root and in the manifest. Do not quietly re-use any `fid`-keyed
derived product from it (`halo_atlas/fid_snap*.npz`, `field_cache/field_stats_fid.npz`,
`mf_cache/mf_nu8_snap096.npz` bind leg, `nu05_shards/sci_bind.npz`, `nu05n_shards/fid.npz`,
`runs/bind/run_0000/*`) without the same banner.

### 4.3 The `bind_n1000` BIND arm

Measured sizes: `bind_n1000/bind` **56.43 GB**, `bind_n1000/truth` **56.40 GB**,
`bind_n1000/dmo` **19.40 GB**. The **truth and dmo arms are unaffected** (no conditioning
enters them). The **bind arm inherits the CAMELS conditioning** and must carry the same
deprecation notice or be re-painted. Because the seed ladder `1992 + 7r` is independent of
realization count (verified: `bind/run_0000` κ[0] bit-identical to `bind_lightcone_tng`
κ[0]), the first 50 of any n1000 BIND realization set are the same maps as the 50-real
product — so **one sentence covers both trees**.

### 4.4 Exact wording for the release README / manifest

```
FIDUCIAL LIGHTCONE — CONDITIONING CORRECTION (2026-08-13)

The fiducial BIND lightcone released as `bind_lightcone_tng` was conditioned on the
CAMELS SB35 cosmology (Omega_m 0.3000, sigma_8 0.8000, Omega_b 0.0490, h 0.6711,
n_s 0.9624) rather than TNG300's (0.3089, 0.8159, 0.0486, 0.6774, 0.9667). The
astrophysical parameters were correct; only the five cosmological entries of the
conditioning vector were wrong. The ratio Omega_b/Omega_m was high by a factor
1.038141, so the painted gas column is high by that factor and the gas and optical-
depth plane power by (Omega_b/Omega_m)^2 = 1.0777 (measured: +7.69 %). Convergence
is affected at the 0.03 % level below ell = 4000 and 0.9-1.9 % above.

The scope of the error is that one tree and the products derived from it. The DMO,
hydro-pasted truth, Sobol (256-node) and 1P (57-run) sets were conditioned correctly
and are unaffected. The 550-realization covariance sets, which are estimated from the
hydro-pasted truth, are unaffected.

THE RELEASED FIDUCIAL IS NOW `bind_science/runs/twobound/run_0049`. It is an
independent paint at the fiducial 30 astrophysical parameters and the correct TNG300
cosmology, on the identical dark-matter substrate (same stage-1 halo cutouts, 33,678
halos, 20 snapshots), the identical model checkpoint, and the identical paste
settings. Because three of the thirty parameters have their fiducial value at a prior
bound, this parameter vector is a member of the 1P design; run_0049 is therefore
simultaneously the fiducial and a 1P member, which makes the 1P responses internally
self-consistent. Its 50 realizations are seed-paired to the DMO and hydro-pasted truth
sets bin-for-bin (per-realization low-ell correlation with the DMO arm: 0.99977).

Two further independent paints at the same parameters, `twobound/run_0018` and
`twobound/run_0053`, are released alongside it. BIND's sampling step is not seeded, so
these three quantify the model's paint-to-paint reproducibility: 0.02-0.09 % per band
on S(ell), 0.1-0.5 % on C_ell^tautau, 1-4 % on C_ell^yy, and 2-13 % of the design
standard deviation on the eight halo latents. Users needing a paint-noise estimate
should use the trio; users needing "the fiducial" should use run_0049.

`bind_lightcone_tng` is RETAINED FOR PROVENANCE ONLY and must not be used for new
science. The same notice applies to the BIND arm of `bind_n1000` (its truth and DMO
arms are unaffected), and to every cached product keyed `fid` that was derived from
either: halo_atlas/fid_snap*.npz, field_cache/field_stats_fid.npz, the bind leg of
mf_cache/mf_nu8_snap096.npz, nu05_shards/sci_bind.npz, nu05n_shards/fid.npz, and
bind_science/runs/bind/run_0000/.

The conditioning bug cannot recur: bind.inference.paint_stages._check_cosmology now
raises if the stage-1 conditioning vector disagrees with the snapshot header on
Omega0, OmegaBaryon or HubbleParam, and bind.tng300_params() supplies the correct
vector for any IllustrisTNG substrate (src/bind/params.py).
```

---

## 5. Remaining work for the author, in order

Costs measured on this 1-core node unless marked "estimated". `PY=/mnt/home/mlee1/venvs/BIND_env/bin/python3`,
`P1=/mnt/home/mlee1/BIND/papers/01_pipeline`.

### 5.1 [~12 h, SUBMIT] `paired_stats` for the 60 twobound runs

**Measured, and it is 3× worse than originally scoped:** `_perreal` on one 50-realization
cube takes **712 s**; with load + σ₀, **~739 s (12.3 min) per task ⇒ ~12.3 h serial**
(measured under contention with two other jobs; ~7–10 min/task uncontended). Hand to a
60-task disBatch.

```bash
bind-paired-stats \
  --run_dir  /mnt/home/mlee1/ceph/bind_science/runs/twobound/run_NNNN \
  --fid_dir  /mnt/home/mlee1/ceph/bind_science/runs/twobound/run_0049 \
  --n_real 50 --out_name paired_stats_fidtb49.npz
```

**Two traps.** (a) The CLI writes `run_dir/<out_name>` and caches
`fid_dir/paired_perreal_fid.npz` — **both inside read-only campaign trees**; stage to
`referee_work/fidswap/` or get the trees temporarily writable. (b) `run_dir == fid_dir` is
a no-op, so run_0049 yields **59** files, not 60.

Then re-point: `paper_s3b_clusters.py:73`, `family_basis_all.py:110`,
`family_basis_model.py:81`, `family_basis_ship.py:93`, `imf_shape_clusters.py:68`,
`sr_kernels_v2.py:236`, `sr_kernels_v3.py:384`, `_build_mechanism_nb.py:120`; then
`$PY family_basis_all.py && $PY family_model_section_figs.py`.

**Unblocks:** `imgs/pfig_s3b_cl_clusters.png` (paper Fig 11), `pfig_s3b_pdf_clusters.png`,
and the three figures `main.tex` cites that **do not exist in `imgs/` at all** —
`pfig_fm_basis.png`, `pfig_fm_freeamp.png`, `pfig_fm_model_curves.png`.
**Will change:** ≤ 0.03 % below ℓ = 4×10³ and 0.9–1.9 % above (the ℓ-*shape* of C_fid
only; Eq. 20 max-normalises the constant part away). Family membership and basis shapes
are stable to ≲ 2 %, so **these can ship provisionally with a footnote** if the author
prefers not to wait.

### 5.2 [~10 min, inline] Noisy LSST-Y10 target for the new fiducial

```bash
# 1. add TARGETS['fidtb49'] = SCI/'runs/twobound/run_0049/kappa_maps.npz'
#    at papers/01_pipeline/noisy_grid.py:130
# 2. monkeypatch ng.SHARD_DIR -> /mnt/home/mlee1/ceph/referee_work/fidswap/nu05n_shards/
#    in a small driver (do NOT edit the constant: bind_sb35 is read-only and the
#    constant is shared with the untouched Sobol builds)
cd $P1 && $PY build_noisy_cache.py --run fidtb49
```

Then in `referee/work/r2_common.py` point `SHARDS_N` at the redirected dir and change the
target string `'fid'` → `'fidtb49'` in `r2_validation.py:52` and `r2_detect.py:58`.

**Reuse `_kappa_rms_fid.npz` unchanged** — the ν denominator moves by only **0.07–0.11 %**
between old and new fiducial (measured per-plane: [0.01600, 0.02059, 0.02500, 0.02878,
0.03161] → [0.01601, 0.02061, 0.02502, 0.02881, 0.03164]) — and **do not rebuild the 256
Sobol noisy shards** (that would be ~20 h serial for no gain).

**Unblocks:** `imgs/fig05n_field_validation_noisy`, `imgs/fig23n_s3_opener_noisy`.
**Will change:** the noisy fiducial curve and the χ² detectability reference. **Note** the
shard drops from 550 to 50 realizations, weakening that covariance leg by √11.

### 5.3 [~seconds once 5.1/5.2 land] Refresh `figs_preview/amplitude_sets.npz`

Rebuild `clk/pdf/pk/mn/v0/v1/v2__fid_measured`, `*__a_fid`, `lam_fid` (currently still the
**old 4-latent** vector `[0.801, 0.0968, 0.645, 6.833]`), and `theta_fid` (currently the
**CAMELS** vector — replace with `bind.tng300_params()`). Use the **nu05 shard** route for
v0/v1/v2, not `mf_cache` — and fix the `notes` string, which currently misstates the
provenance (§1.4). **Unblocks** the l.889–893 deviation numbers.

### 5.4 [~5 min] Re-derive the calibration ladder in Sec 5.4

`family_model_section_figs.py:354-356` hardcodes the 4-latent **0.920** and two-bin
8-latent **0.908** rungs as literals; only the third is computed live. Re-derive all three
from the new λ under `figs_preview/family_model_bundle_4lat.npz` and `_2bin8.npz`.
**Unblocks** the l.898–905 replacement in §3.11.

### 5.5 [~3 min + a port] `fig11_zclosure` panel (d)

Panels (a)/(b)/(c) inputs are ready and their numbers are already computed
(`fsfig11_zclosure_numbers.py`, `fig11_numbers.npz`). Panel (d) reads the **stacked**
`bind_science/profiles/fid_snap{029..096}.npz`, which exists only for the retired
fiducial; the builder `examples/halo_atlas.py` lives on branch `analysis/sobol-sb35`.

```bash
# port the stacked reducer from analysis/sobol-sb35:examples/halo_atlas.py, then
for SN in 029 031 033 035 038 041 043 046 049 052 056 059 063 067 071 076 080 085 090 096; do
  $PY $P1/build_perhalo_profiles.py --src twobound/run_0049 --snap $SN
done          # ~8 s/snap = ~3 min serial, plus the stack step
```

**Also fix panel (b) regardless of the swap** (§1.4, item 2). **Will change:** the Y_500c
z-ladder, the f_gas ladder, and the `quadratic_a` debias templates (all numbers in §2.10).

### 5.6 [~5 min] The §3c latent cell — Figs 20a/20c, `pfig_s4b_reconstruction`, `pfig_s3c_hinge_plane`

Not re-implemented standalone: it is one ~1500-line notebook cell
(`_build_figures_nb.py:4556-6012`) with a long internal dependency chain (`fa20`, `clb20`,
`S_fid_vd`, `S_fid_hi`, `S_fid_b24`, `pred_fid`), and re-deriving it outside the notebook
risks silent divergence.

```bash
# in _build_figures_nb.py:
#   fa20  = np.load(SCI/'halo_atlas/fid_snap096.npz')
#        -> /mnt/home/mlee1/ceph/referee_work/fidswap/halo_atlas/fidtb49_snap096.npz
#   clb20 = np.load(SCI/'runs/bind/run_0000/Cl_kappa.npz')
#        -> SCI/'runs/twobound/run_0049/Cl_kappa.npz'
cd $P1 && $PY _build_figures_nb.py && $PY _run_subset.py <cells>
```

All inputs already exist. **Will change:** the fiducial star's (f̄_bar, S) position,
`S_fid_b24` measured (trough 0.8812 → 0.9016 on the 24-band grid) and `pred_fid`
(0.8950 → 0.8932 for run_0049). The Sobol matrix, the kernels (fig20g) and the corner
(fig20i) are **unaffected**.

### 5.7 [0 compute — AUTHOR DECISION] `imgs/fig23_s3_opener.png` (paper Fig 7)

Blocked on a decision, not on compute. `main.tex:604`'s caption ("the 5–95 % spread of
$S(\ell)$… the fiducial (black)… LSST-Y10 and *Euclid* envelopes") describes notebook
**`fig12_survey_context`** (`_build_figures_nb.py:6644`) or the noiseless twin of
`referee/work/r2_detect.py` — **not** notebook `fig23_s3_opener`
(`_build_figures_nb.py:3367`), which is the S(5000)-vs-f_gas scatter. Neither PNG exists
in `imgs/`. Once the author names the cell, the swap is one line
(`S_fid_full = bcl0/dmo_paired_cl`, `_build_figures_nb.py:6573-6576` → the replica's
`Cl_kappa`) and ~5 min to render.

### 5.8 [3 min + 9 s each, only if the canonical pick changes] tb18/tb53 ladders

Only run_0049 has the 20-snapshot atlas ladder and the per-halo profiles. Adding another
replica is 3 min (atlas) + 9 s (profiles).

### 5.9 Not attempted, and not required by any current figure

- The full r3b-style texture decomposition across all 20 snapshots (~3–4 h; the
  codebase-wide grep found **exactly one** consumer of the recomposite path,
  `referee/r3b_texture_analysis.py` itself).
- `peak_counts_multi` / `peak_cross_multi` for the replica (~15 min; the `ngal10` variant
  *was* built).
- `/mnt/home/mlee1/BIND/lowmass_096_compare.npz` (notebook `fig14_completeness` panel b);
  builder is `examples/lightcone_lowmass_reuse.py` on an analysis branch, estimated
  30–90 min — hand to the author.
- A lux re-trace of run_0049 to 550 or 1000 realizations (GPU/Slurm campaign).

---

## 6. Reproducibility note — paints are not seeded

### 6.1 The fact

`src/bind/model.py:454` draws the flow-matching initial state as

```python
x = torch.randn(B, self.out_channels,
                condition.shape[2], condition.shape[3], device=device)
```

with **no generator argument**, and no `torch.manual_seed` is set anywhere in the sampling
path (`bind.inference.paint_stages` / `bind.paint`). The RT_SEED that *is* controlled
(1992 + 7r) governs the ray-tracing realization ladder, not the generative draw. This is
why `main.tex:430`'s "that seed is shared across all lightcones, such that differences
between lightcones are purely parameter-driven" is incorrect (§3.3).

### 6.2 The consequence, turned into a measurement

Runs 0018, 0049 and 0053 are three independent paints at **byte-identical** parameters,
substrate, checkpoint and paste settings. Their spread **is** the model's paint-to-paint
scatter — a number the paper could not previously quote. All values from
`fidswap_equivalence_audit.npz`, `validation_fidswap.npz`, `lam_fidswap_v2.npz`.

| observable | 1σ over 3 paints |
|---|---|
| **S(ℓ), z_s = 1, per band** | **0.016–0.094 %** (median **0.052 %**; absolute 0.00016–0.00087) |
| S(ℓ) relative to LSST-Y10 per-band error | 0.4–18 % (Knox, ngal 27, σ_e 0.26, f_sky 0.44, dlnℓ 0.15) |
| S(ℓ) relative to the paper's own 50-real paired SE | 0.07–1.29×, **median 0.39×** |
| C_ℓ^κκ | 0.011–0.022 % (ℓ<1000) → 0.037–0.178 % (ℓ>1.5e4) |
| C_ℓ^ττ | 0.112–0.487 % |
| C_ℓ^κτ | 0.102–0.348 % |
| C_ℓ^κy | 0.309–1.251 % |
| C_ℓ^yy | 1.064–3.994 % (bright-cluster dominated) |
| nu05 statistics (median / max over 22 bins) | pdf 0.082/1.554 %, peaks 0.864/5.916 %, minima 0.210/4.875 %, V0 0.026/0.439 %, V1 0.107/1.446 %, V2 0.133/1.385 % |
| κ moments (2′) | variance 0.002–0.024 %, skewness 0.039–0.087 %, kurtosis 0.132–0.161 % |
| halo-population medians | **0.040–0.138 %** — *below their own bootstrap SE* |
| the 8 model latents | **2.2–12.7 % of σ_design** (f̃_★ 2.32, log T̃ 7.71, log Ỹ 5.79, log P̃_e[14] 10.19, c_gas[14] 12.65, log Ỹ_ss 2.19, c_gas[13.2] 2.16, log P̃_e[13.4] 2.55) |

### 6.3 Why this matters for the error budget

`paper_error_budget.py` (`pfig_s4b`) gains **two rows**:

1. **σ_paint on the observable:** 0.0002–0.0009 absolute on `S(ℓ)` — subdominant to but
   **not negligible** against the paper's own paired error bars; it reaches **1.29×** the
   paired SE at ℓ ≈ 587 and sits at a median 0.39×.
2. **σ_paint propagated through λ:** 2.2–12.7 % of σ_design per latent maps to **0.0054**
   on the predicted `S(ℓ)` trough — **larger than the trio-mean closure residual
   (−0.0034)**. The family model's fiducial closure is therefore limited by paint
   stochasticity, not by model error. That is a *stronger* claim than the paper currently
   makes, and it is the reason the "λ information ceiling" diagnosis in project memory is
   retired.

Adopting the 3-mean instead would drop both rows by √3 — but would then describe a product
no user can download. That is the trade §1.3 declines.

### 6.4 The optional fix

Determinism is a one-line change if it is ever wanted: thread a
`torch.Generator(device=device).manual_seed(halo_seed)` into the `torch.randn` call at
`model.py:454` (and the sibling `randn_like` calls at `model.py:288,396` for training
paths). **Do not apply it retroactively** — it would invalidate the reproducibility
measurement above and every released map. If applied, it should be a new keyword
(`generator=` / `seed=`) that defaults to `None`, preserving current behaviour, so that
the released trio's status as an unseeded paint-noise sample remains well-defined.

---

## Appendix — provenance index

| this document says | file |
|---|---|
| halo medians, calibration templates | `referee_work/fidswap/numbers_fig03_halo_tb49.txt`, `halo_atlas/fidtb49_snap096.npz` |
| radial profiles | `numbers_fig04_profiles_tb49.txt`, `profiles/perhalo_fidtb49_snap096.npz` |
| field validation, χ², nu-domain | `numbers_fig05_field_tb49.txt`, `fig05_numbers.npz`, `field_cache/field_stats_fidtb49.npz`, `mf_cache/mf_nu8_snap096_fidtb49.npz`, `nu05_shards/sci_bind_tb49.npz` |
| gas spectra, (Ω_b/Ω_m)² check | `numbers_fig06_spectra_tb49.txt`, `fig06_numbers.npz` |
| covariation levels/spreads | `numbers_fig10_covariation_tb49.txt`, `fig10_numbers.npz` |
| S(ℓ) 24-band table, paint scatter | `validation_fidswap.npz` |
| family-model closure | `closure_fidswap.npz` |
| the 8 latents | `lam_fidswap_v2.npz` |
| R1 ladder | `r1_numbers_z1_tb49.json` vs `referee/work/r1_numbers_z1.json` |
| hero stamps | `numbers_fig02_hero_tb49.txt` |
| z-ladder + debias templates | `numbers_fig11_zclosure_numbers_tb49.txt`, `fig11_numbers.npz` |
| convention equivalence, reproducibility | `fidswap_equivalence_audit.npz` |
| recomposite + cache gates | `fidswap_manifest.json`, `composites/`, `atlas_truth_CHECK_snap096.npz` |
| figures written this session | `imgs/fig{02_hero,03_halo_validation,03_halo_scaling,04_radial_profiles,05_field_validation,06_spectra_validation,06b_full_hydro,10_covariation}_fidswap.{png,pdf}` |
| scripts | `papers/01_pipeline/referee/work/fs*.py` (`fsfig_common.py` carries the single swap; `BIND_FID_REPLICA` selects the replica) |
