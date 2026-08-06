# The analytic halo-latent model for BIND field statistics

Method + full procedure record for the fig 20d–f analysis (2026-08-05/06 session).
Companion to the figure code, which is the source of truth: the single §3c cell of
`_build_figures_nb.py` (search `Figs 20a-e`), which live-prints every number quoted
here. Figure toolchain committed at `03d1507`. All accuracy numbers below are
**out-of-sample** (cross-validated or leave-one-out) unless explicitly marked
in-sample.

**One-paragraph summary.** The van Daalen+20 relation says the matter-power
suppression at $k\lesssim0.5\,h/\mathrm{Mpc}$ is a one-dimensional function of the
group baryon fraction. We find its completion for the full WL suppression curve:
$S(\ell)$ over the entire 256-node TNG-prior Sobol suite is, to CV-$R^2\simeq0.94$,
a **linear** function of four halo-population numbers — budget $\tilde f_{\rm bar}$,
partition $\tilde f_\star$, gas concentration $c_{\rm gas}$, temperature
$\log\tilde T$ — while a linear model on all 30 raw feedback parameters reaches only
$\simeq$0.5–0.6. The latents *linearize* the response: all parameter-space
nonlinearity lives in the map params→latents (halo astrophysics), none in
latents→statistics. One low-$z$ latent vector per universe covers all five source
planes, the same latents carry the PDF/MF/Y–M/T–M sector self-consistently, and
individual $S(\ell)$ curves — including the out-of-design fiducial's — are
regenerated from 4 numbers at RMS $\simeq$ 0.006–0.026 against a cloud spread of
0.057.

---

## 0. Convention v2 — HARMONIZED (2026-08-06, author ruling; current canon)

Every latent is measured under **one** observer-reproducible convention: one
snapshot (096, $z=0.034$), one mass bin (hinge, $\log_{10}m^{\rm bg}_{\rm tot,500c}
\in[13.3,13.6)$), one aperture/background convention (projected cylinders,
2.5–3.0 $h^{-1}$Mpc transverse annulus, bg-subtracted totals/gas), one reducer
(median over bin halos, $\geq5$ halos):

$$\tilde f_{\rm bar} = \frac{1}{\Omega_b/\Omega_m}\,\mathrm{med}_{h\in\mathcal G}
\frac{m^{\rm bg}_{\rm gas,500c}+m_{\star,500c}}{m^{\rm bg}_{\rm tot,500c}},\qquad
\tilde f_\star = \frac{1}{\Omega_b/\Omega_m}\,\mathrm{med}_{h\in\mathcal G}
\frac{m_{\star,500c}}{m^{\rm bg}_{\rm tot,500c}}$$

$$c_{\rm gas} = \mathrm{med}_{h\in\mathcal G}
\frac{m^{\rm bg}_{\rm gas,500c}}{m^{\rm bg}_{\rm gas,200c}}
\;\;(\text{two-aperture concentration: } R_{500c}{=}0.659\,r_{200}\text{ vs }r_{200}),
\qquad
\log\tilde T = \log_{10}\,\mathrm{med}_{h\in\mathcal G}\,T_{\rm mw,500c},\;\;
T_{\rm mw} = \frac{\sum T\,\Sigma_{\rm gas}}{\sum\Sigma_{\rm gas}}$$

The snap085 kSZ-profile $c_\tau$ (v1's structure latent: different epoch, FoF mass
bin, per-halo $x$-annulus, pixel-weighted stack) is **retired to a cross-check**:
added to the harmonized set it moves the $S(\ell)$ CV-$R^2$ median by only
$+0.012$.

**Headline numbers (harmonized, live-printed by the fig 20 cell):**
- $S(\ell)$, $z_s=1$: 4-latent CV-$R^2$ = 0.95/0.96/0.91 at
  $\ell\sim10^3/5\times10^3/1.9\times10^4$ (median 0.94); 30-param linear
  0.51/0.52/0.61. All $z_s$: medians 0.95/0.94/0.93/0.93/0.92.
- Scorecard (4-latent CV-$R^2$): PDF 0.85, MF-$V_1$ 0.94, MF-$V_2$ 0.93,
  Y–M 0.94, $f_{\rm gas}$–M 0.97, $\kappa\tau$ 0.86, **T–M 0.77** (fixed by
  $\log\tilde T$), $C_\ell^{yy}$ 0.77 (stated boundary).
- Generation demo: LOO nodes RMS 0.0059/0.0092/0.0128; **fiducial
  (out-of-design) RMS 0.026 full set / 0.015 without the thermal kernel** —
  $\log\tilde T$ spans only $\sim$0.09 dex across the cloud, making its kernel
  the least out-of-design-robust leg (the fid's $\log\tilde T$ is at the cloud's
  47th percentile, not an outlier); both far below the cloud spread 0.057.
- Kernel table: $5\times24\times5$ numbers (`latent_model_coeffs.npz` v2,
  `predict_from_latents.py --fbar --fstar --cgas --logt`). Note the strong
  collinearity $r(c_{\rm gas},\tilde f_{\rm bar})=+0.94$: individual kernel
  *shapes* are jointly de-mixed partial slopes and less individually
  interpretable than in v1; the $1\sigma$-impact panel (fig 20g f) is the fair
  importance comparison.

Sections below describe the v1 (mixed-convention) analysis where numbers differ;
the fig 20 cell prints are the canon.

---

## 1. Context and question

- **van Daalen, McCarthy & Schaye (2020)** (MNRAS 491, 2424): across independent
  hydro simulations, $\Delta P/P$ at $k=0.5\,h/\mathrm{Mpc}$ collapses onto a single
  curve in the renormalized group baryon fraction — a 1-D latent space. Their fit
  (hardcoded through our v1 `fig_scripts/fig08_vandaalen_analog.py`):
  $\Delta P/P = -\exp(-5.990\,\tilde f_{\rm bar} - 0.5107)$.
- **The tension**: the twobound per-parameter response shapes
  (`imf_shape_clusters.py`, scratch fig `quick_deltacl_clusters.png`) cluster into
  ~4 families with *different* $\Delta C_\ell$ shapes (C1 BH/AGN, C2 wind-velocity,
  C3 RadioFdbk/IMF, C4 WindEnergy) — apparently more than one dimension.
- **The question** (author, 2026-08-05): can the families be reconciled with the
  1-D vD law through the halo-level $f_{\rm gas}$, giving an analytic model of the
  suppression instead of the GP emulator?
- **The answer**: yes. The vD budget is latent #1 (sets the large-scale amplitude);
  there is exactly **one** more mass-sector latent — the gas↔star **partition** of
  the budget (sets the small-scale shape); the families are *directions* in this
  2-D latent plane, not violations of the vD law. Adding gas **concentration** and
  **temperature** completes the set. Figures: 20d (the latent plane), 20e (the
  analytic model + generation demo), 20f (redshift + thermal).

## 2. Data

All from the 2026-08 nu05 state of the Sobol suite (`bind_sb35`,
`emulator_dataset_nu05.npz`, $N=256$ nodes after the 0114/0115/0117 repaint;
the fiducial is *not* a design node).

### 2.1 Statistics

- **Suppression**: $S_q(\ell) = C_\ell^{\kappa\kappa}[q] / C_\ell^{\kappa\kappa}[\mathrm{DMO}]$,
  both sides means over the same 50 **seed-paired** ray-tracing realizations
  (shared DMO lightcone; per-realization low-$\ell$ corr(DMO, BIND) = +0.9998), so
  cosmic variance cancels mode-by-mode. Fiducial-side denominators use the
  paired-prefix mean (`dmo_paired_cl()`, never the shipped 550-real mean — see the
  DMO-denominator memo).
- **Per-node measurement precision** (justifies treating sub-percent scatter as
  signal): from the 50 fiducial/DMO realization pairs, the per-node band-0
  uncertainty is $\sigma(\Delta S) = 5.0\times10^{-4}$ — the cloud's band-0 spread
  is $9\times$ larger.
- **Binning**: fine $\ell$ grid (724 bins) → $B=24$ logarithmic bands over the
  trusted range $\ell\in[300,\,3\times10^4]$ (ELL_TRUST); band value = mean of the
  fine-grid $S$ in the band. Source planes $z_s \in \{0.5, 1.0, 1.5, 2.0, 2.44\}$
  ($z_s=1$ is the working plane unless stated).
- **Other statistics** (cross-statistic scorecard): $\kappa$ PDF, Minkowski
  functionals $V_0,V_1,V_2$, peak/minima counts (all on the $\Delta\nu=0.5$ nu05
  grid), DM PDF, scaling relations $Y$–$M$, $T$–$M$, $f_{\rm gas}$–$M$,
  $f_\star$–$M$, and the spectra $C_\ell^{yy}$, $C_\ell^{\kappa y}$,
  $C_\ell^{\kappa\tau}$. Spectra are modeled in $\log_{10}$; runs 114/115/117 are
  excluded from all SZ-spectra legs (cross-norm mismatch memo).

### 2.2 The latents

Population medians over the shared 2933-halo sample, from the atlas cubes
(`bind_sb35/analysis_cache/atlas_cubes/`) and the stacked-profile cache
(`bind_science/ksz_confront/`). $\mathcal G$ = the "hinge" group mass bin
$\log_{10} M^{\rm bg}_{500c}/{\rm M_\odot} \in [13.3, 13.6)$ (the single bin that
predicts $S(\ell\!\sim\!10^3)$ at $r=0.98$; ≥5 halos required, all 256 nodes pass);
apertures are projected cylinders (500c transverse, $\pm25.6\,h^{-1}$Mpc slab,
annulus background-subtracted) — painted maps admit no 3-D measure.

| latent | definition | epoch / file |
|---|---|---|
| budget | $\tilde f_{\mathrm{bar},q} = \dfrac{1}{\Omega_b/\Omega_m}\,\mathrm{med}_{h\in\mathcal G}\dfrac{m^{\rm bg}_{\rm gas,500c}+m_{\star,500c}}{m^{\rm bg}_{\rm tot,500c}}$ | $z=0.034$, `atlas_cube_snap096.npz` |
| partition | $\tilde f_{\star,q} = \dfrac{1}{\Omega_b/\Omega_m}\,\mathrm{med}_{h\in\mathcal G}\dfrac{m_{\star,500c}}{m^{\rm bg}_{\rm tot,500c}}$ | same |
| structure | $c_{\tau,q} = \dfrac{\langle\tau(x)\rangle_{x<0.3}}{\langle\tau(x)\rangle_{0.3\le x<1}}$, $x=R/r_{200c}$, stacked profile, mass bin 13.0–13.4 | $z=0.18$, `bind_tauy_xprof_snap085.npz` |
| thermal | $\log\tilde T_q = \log_{10}\,\mathrm{med}_{h\in\mathcal G}\, T_{\rm mw,500c}(h)$ | $z=0.034$, `atlas_cube_snap096.npz` |

$\Omega_b/\Omega_m = 0.0486/0.3089$. Fiducial latents (never in any fit) come from
the identical reducers applied to `halo_atlas/fid_snap096.npz` and
`bind_tauy_fiducial_snap085.npz` ($c_{\tau,\rm fid}=2.15$).

### 2.3 Upstream computation chain (how the per-halo inputs are actually made)

Everything starts from the BIND per-halo patches stored in the runs'
`composite_slab*.npz` files: for every halo above $10^{13}\,M_\odot$, a
$6.25\times6.25\,(h^{-1}\mathrm{Mpc})^2$ projected cutout ($128^2$ px, pixel
$=48.8\,h^{-1}$kpc) with `generated_patches` = 3 mass channels
$[\Sigma_{\rm DM},\Sigma_{\rm gas},\Sigma_\star]$ (surface mass per pixel,
full-slab line of sight) and `thermo_patches` = 4 gas channels
$[y, T, K, P_e]$.

**$\tilde f_{\rm bar}$, $\tilde f_\star$, $\log\tilde T$ — the atlas chain**
(per-halo reducer: `examples/halo_atlas.py::reduce_file` on `analysis/sobol-sb35`,
mirrored in `papers/01_pipeline/build_atlas_200c.py`; the 256-node cubes are built
by `examples/sobol_atlas_mpi.py` via `run_sobol_atlas.sh`, viewed by
`_build_sobol_atlas_nb.py`):

1. circular projected aperture $R_{500c} = 0.659\,r_{200}$ around the patch
   center; $m_X = \sum_{r<R_{500c}} \Sigma_X$ per species $X$;
2. background: mean surface density in the fixed transverse annulus
   $2.5$–$3.0\,h^{-1}$Mpc (total and gas separately);
   $m^{\rm bg}_{\rm tot} = m_{\rm dm}+m_{\rm gas}+m_\star - \bar\Sigma^{\rm ann}_{\rm tot} N_{\rm pix}$,
   $m^{\rm bg}_{\rm gas} = m_{\rm gas} - \bar\Sigma^{\rm ann}_{\rm gas} N_{\rm pix}$.
   **The stellar mass is *not* background-subtracted** (no uniform stellar
   background at this depth);
3. thermodynamics: $T_{\rm mw,500c} = \sum (T\,\Sigma_{\rm gas})\,/\sum
   \Sigma_{\rm gas}$ over the same aperture — gas-mass-weighted with the *raw*
   (not bg-subtracted) gas weights; $Y_{500c} = \sum y\,\cdot$ pixel area; no
   background subtraction on the thermo channels;
4. the cube stacks these per-halo arrays over all 256 nodes at the shared
   2933-halo sample (`atlas_cube_snapNNN.npz`, `sobol_*` keys + `fid_*`/`tb_*`
   twins); the latent is then the hinge-bin median as defined above (mass bin on
   $m^{\rm bg}_{\rm tot,500c}$).

**$c_\tau$ — the kSZ-profile chain**
(node cache: `examples/ksz_tau_gnfw.py::stack_node_snap` on
`analysis/ksz-desi-act-v2`, writing `bind_tauy_xprof_snap{085,046}.npz`; fiducial
twin: `examples/_reduce_tauy_fiducial.py`):

1. per halo (snap 085, $z=0.18$), convert the gas patch to Thomson optical depth
   per pixel: $\tau = \Sigma_{\rm gas}\,\sigma_T\,(x_e/m_p)\,/\,A^{\rm phys}_{\rm pix}$
   with $x_e = 0.88$ electrons per proton mass and the pixel area converted to
   physical cm$^2$ at the snapshot's $a$;
2. radial coordinate $x = R/r_{200c}$ (each halo scaled by its own $r_{200}$);
   per-halo background = mean $\tau$ over the annulus $x\in(2.5, 3.0)$,
   subtracted;
3. stack: all halos in the $\log_{10}M \in [13.0, 13.4)$ bin (note: **FoF-mass
   binned** — a different bin convention from the atlas's cylinder-mass hinge
   bin), pixel values accumulated into 17 geometric $x$-bins over $[0.08, 3.0]$,
   profile = pixel mean per bin → `tau[massbin, x]` per node;
4. the latent is the inner/outer ratio of that stacked profile
   ($x<0.3$ vs $0.3$–$1.0$).

Convention differences worth stating in a methods section: the three latents do
**not** share a mass-bin convention ($\tilde f$'s and $\log\tilde T$: cylinder-mass
hinge bin 13.3–13.6 at $z=0.034$; $c_\tau$: FoF bin 13.0–13.4 at $z=0.18$), the
background treatments differ (fixed-Mpc annulus vs per-halo $x$-annulus; stars and
thermo channels unsubtracted), and $c_\tau$ is a pixel-weighted stack (halos enter
$\propto$ their pixel counts). None of this affects the model's internal
consistency — the kernels are fit to whatever the latents *are* — but an
observational reproduction must match these definitions, not idealized ones.

## 3. The two-latent decomposition (fig 20d)

Procedure, in order (all per-band operations use the 24-band $S$ matrix
$S\in\mathbb R^{256\times24}$ at $z_s=1$):

1. **Budget-only regression, per band**: OLS $S(:,\ell_b) \sim
   [\tilde f_{\rm bar}, 1]$. In-sample $R^2(\ell_b)$: 0.83–0.95 for
   $\ell\lesssim3000$, collapsing to **0.20** at $\ell\approx1.9\times10^4$. The vD
   latent owns the large scales and loses the small scales.
2. **Residual PCA**: residual matrix $R = S - \hat S_{\rm budget}$, columns
   standardized, SVD. Variance shares: **PC1 = 97.9%**, PC2 = 1.8%, PC3 = 0.2%.
   One extra latent, peaked at the smallest scales.
3. **Identify PC1** (sign fixed so PC1 anti-correlates with $\tilde f_\star$):
   $r(\mathrm{PC1}, \tilde f_\star) = -0.90$; $r(\mathrm{PC1}, c_\tau) = -0.44$;
   $r(\mathrm{PC1}, \tilde f_{\rm gas}) = +0.51$; $r(\mathrm{PC1},
   \tilde f_{\rm bar}) = 0.00$ (by construction). Crucially $\tilde f_\star$ is
   *invisible to the amplitude*: $r(\tilde f_\star, S(\ell\!\sim\!10^3)) = +0.08$.
   The second latent is the **gas↔star partition of the budget**.
4. **Nested predictors at $\ell\approx1.9\times10^4$** (in-sample here; CV version
   in §4): $\tilde f_{\rm bar}$ alone 0.20 → $+\tilde f_\star$ 0.85 →
   $+c_\tau$ 0.94.
5. **Lever attribution** (Spearman of the 30 unit-cube params against each axis):
   budget axis — IMFslope $-0.48$, WindEnergy $+0.35$, BHRadiativeEff $+0.28$;
   partition axis (residual PC1) — **VariableWindVelFactor $+0.64$**,
   WindFreeTravelDens $-0.30$, BHRadiativeEff $-0.28$. The C2 wind-velocity family
   of the $\Delta C_\ell$ shape clustering **is** the partition direction; the
   $S(5000)>1$ enhancement branch is its extreme (weak winds → gas retained and
   concentrated, stars formed). Also $r(c_\tau, \tilde f_{\rm bar}) = +0.69$
   (structure partially tracks budget; PC1 captures the independent part).

This is the halo-level restatement of the 2-D WL feedback latent of Lin et al. 2026
(arXiv:2509.01881): both abstract latents get measurable population names.

## 4. The analytic model (fig 20e)

### 4.1 Definition

$$S_q(\ell_b) \;=\; c_0(\ell_b) + c_1(\ell_b)\,\tilde f_{\mathrm{bar},q}
 + c_2(\ell_b)\,\tilde f_{\star,q} + c_3(\ell_b)\,c_{\tau,q} + \varepsilon_q(\ell_b)$$

- $c_1(\ell) = \partial S/\partial\tilde f_{\rm bar}$: the **budget kernel** (the
  vD response), dominant at $\ell\lesssim3\times10^3$.
- $c_2(\ell) = \partial S/\partial\tilde f_\star$: the **partition kernel**,
  growing toward small scales.
- $c_3(\ell)$: the concentration correction.
- Formally the first-order Taylor expansion of (population halo properties) →
  (statistic) about the cloud mean; empirically first order is sufficient over the
  whole prior (a quadratic $\tilde f_{\rm bar}^2$ term adds nothing beyond
  $\tilde f_\star$: CV 0.92 vs 0.90 at $\ell=5000$, vs 0.96 for $+c_\tau$).
- For tomography, coefficients gain a $z_s$ index, $c_i(\ell_b, z_s)$; the latent
  vector is **shared** (§6). The full model is the table $c_i(\ell_b,z_s)$:
  $4\times24$ numbers per source plane.

### 4.2 Fitting and validation

- Design matrix $A\in\mathbb R^{N\times4}$, rows $(\tilde f_{\rm bar},
  \tilde f_\star, c_\tau, 1)_q$; per band, OLS via SVD pseudoinverse
  (`numpy.linalg.lstsq`, 24-column RHS in one call). No regularization
  ($N=256$, 4 regressors).
- **Cross-validation**: 5 folds, *deterministic* assignment fold = node-index
  mod 5 (the figure cell is RNG-free by contract; a shuffled-fold scratch run gave
  the same numbers to ±0.01).
- Headline (CV-$R^2$ at $\ell\sim10^3 / 5\times10^3 / 1.9\times10^4$):

| model | $10^3$ | $5\times10^3$ | $1.9\times10^4$ |
|---|---|---|---|
| $\tilde f_{\rm bar}$ only | 0.95 | 0.80 | 0.19 |
| $+\tilde f_\star$ | 0.95 | 0.90 | 0.84–0.85 |
| $+c_\tau$ (**the model**) | **0.95** | **0.96** | **0.93** |
| all 30 raw params, linear | 0.51 | 0.52 | 0.61 |

  RMSE (3-latent): 0.0025 / 0.012 / 0.031 against cloud std 0.011 / 0.058 / 0.120.
- **The structural point**: the 3-number model beats the 30-parameter linear model
  everywhere. The map $\theta\to S$ is nonlinear (why the GP emulator exists), but
  it *factorizes* as $\theta \to \lambda$ (nonlinear; performed by the
  simulation/painting — or by observations) followed by $\lambda \to S$ (linear,
  four measured kernel functions, no ML).

### 4.3 The generation demo (fig 20e panel b) — exact protocol

1. **Node selection** (deterministic, no RNG): the nodes whose band value
   $S(\ell\approx5000)$ is nearest the 5th / 50th / 95th percentile of the cloud →
   run ids **81, 216, 245** (deep suppression / median / strong enhancement,
   $S\to1.35$).
2. **Leave-one-out per node** $g$: delete row $g$ from $A$ and from $S$; refit
   $\beta^{(-g)}(\ell_b)$ on the other 255 nodes; predict
   $\hat S_g(\ell_b) = (\tilde f_{\mathrm{bar},g}, \tilde f_{\star,g}, c_{\tau,g},
   1)\cdot\beta^{(-g)}(\ell_b)$. The predicted (dashed) curve never saw the node's
   own statistic. RMS over the 24 bands: **0.0078 / 0.0043 / 0.0097**.
3. **Fiducial (out-of-design)**: fit $\beta$ on all 256 Sobol rows — the fiducial
   is not a design node, so it contributes nothing. Latents measured independently
   (§2.2). Measured curve = fiducial lightcone $C_\ell$ over the seed-paired DMO
   denominator, band-averaged. RMS = **0.0128** vs cloud std 0.057 at
   $\ell\approx5000$.
4. **Panel (c)** repeats step 2 for the $\kappa$ PDF (22 nu05 bins): RMS
   $5\times10^{-4}$–$10^{-3}$ on values of order 0.6. (No trusted fiducial nu05
   PDF exists — pre-nu05 grid + stale-MF memo — so the fiducial appears in panel
   (b) only.)

## 5. Cross-statistic scorecard (fig 20e prints)

Same latents, same folds, per-bin CV-$R^2$ median. "Shared-latent $|r|$" =
correlation between the statistic's own residual-after-budget PC1 and the
$S(\ell)$ partition mode of §3 — the self-consistency test.

| statistic | CV-$R^2$ (3-latent) | shared-latent $\lvert r\rvert$ |
|---|---|---|
| $\kappa$ PDF | 0.85 | **0.99** |
| MF $V_1$ | 0.90 | 0.81 |
| MF $V_2$ | 0.94 | 0.86 |
| $Y$–$M$ | 0.92 | 0.90 |
| $f_{\rm gas}$–$M$ | 0.97 | 0.89 |
| $C_\ell^{\kappa\tau}$ | 0.83 | 0.68 |
| $C_\ell^{yy}$ | 0.76 | 0.05 |
| $T$–$M$ | 0.48 | 0.17 |

- The mass sector (PDF, MFs, scalings, $\kappa\tau$) is carried by the *same* two
  latents — one self-consistent model family across statistics.
- $C_\ell^{yy}$ and $T$–$M$ have a *different* residual mode → thermal sector (§7).
- **Peak/minima counts are excluded for cause**: at the $\Delta\nu=0.5$ binning the
  node-to-node spread per bin is only 0.4–0.5× the per-node measurement error
  (checked against the dataset's own `t__*__err` arrays) — noise-dominated; no
  model (the 30-param linear reaches $R^2=0.04$) can or should fit them per-bin.

## 6. Redshift dependence (fig 20f a–b)

1. **One latent triplet, five source planes**: with the latents *fixed* at their
   low-$z$ values, CV-$R^2$ median per $z_s$:
   0.96 (0.5), 0.95 (1.0), 0.93 (1.5), 0.92 (2.0), 0.91 (2.44).
   The tomographic dependence lives in the coefficients: amplitudes dilute
   monotonically — $\max|c_1|$: 0.38→0.22, $\max|c_2|$: 2.10→1.14 from
   $z_s=0.5$→2.44 (lensing-kernel dilution).
2. **Latent evolution** (epochs from the atlas cubes at snaps 085/067/052/041/033,
   i.e. $z=0.18/0.50/0.92/1.41/2.00$; the older cubes lack the `_bg` aperture, so
   this comparison uses plain-500c apertures *uniformly*, including snap096):
   - rank corr. vs $z\simeq0$: $\tilde f_\star$ = 1.00/1.00/0.99/0.97/**0.94** —
     the partition is set **early**;
   - $\tilde f_{\rm bar}$ = 0.99/0.92/0.79/0.61/**0.44** — the budget is built
     **late**.
3. **The z-matched test**: 2-latent CV-$R^2$ for $S(\ell, z_s{=}2.44)$ using
   latents measured at epoch $z$: 0.88 (z=0.03), 0.90 (0.18), 0.90 (0.50),
   0.87 (0.92), 0.70 (1.41), **0.47 (2.00)**. **Z-matched latents are worse, not
   better** — the end-state (low-$z$) budget integrates the entire feedback
   history, which is the same reason the vD relation itself works.

## 7. The thermal fourth latent (fig 20f c)

- Candidates tested (all group- or cluster-bin medians from `atlas_cube_snap096`):
  $\log T$ (group bin 13.3–13.6), $\log T$ (cluster bin 13.6+), $\log Y_{500c}$
  (both bins), $\log P_e$ (cluster). Selection criteria: budget-independence +
  what it fixes.
- **Winner**: group-bin $\log\tilde T$ ($T_{\rm mw,500c}$) — the most
  budget-independent ($r=-0.39$ vs $\tilde f_{\rm bar}$, $-0.14$ vs
  $\tilde f_\star$; cluster-bin $T$ has $r=-0.81$, cluster $P_e$ $r=+0.91$ — they
  mostly re-measure the budget).
- Effect of adding $\log\tilde T$ (CV-$R^2$): $T$–$M$ **0.48→0.78**; $Y$–$M$
  0.92→0.94; PDF 0.85→0.87; $S(\ell)$ 0.94→0.94 (unchanged, as it should be);
  $C_\ell^{yy}$ 0.76→0.77.
- **Stated boundary**: no population-median thermal latent lifts $C_\ell^{yy}$
  past ≈0.78 (cluster-bin variants included) — the $y$-auto residual carries
  profile-level pressure information beyond population medians.

**Full latent set**: $\lambda = (\tilde f_{\rm bar},\ \tilde f_\star,\ c_\tau,\
\log\tilde T)$ — budget, partition, structure, temperature.

## 8. Caveats (state these next to any use of the model)

1. **Mediation, not free-standing emulation**: the latents are measured from the
   same painted maps. Emulating from $\theta$ needs a $\theta\to\lambda$ stage
   (a cheap 30→4 regression) — or none at all when $\lambda$ is constrained
   directly by data (X-ray/kSZ gas fractions, stellar fractions), which is the
   point: data-constrained halo properties → every WL statistic, no TNG parameters
   involved.
2. **Prior-conditional**: coefficients are fit across the TNG-family Sobol prior
   (fixed cosmology); validity outside it is unestablished (same caveat as
   Lin+2026).
3. **Domain**: $\ell\in[300, 3\times10^4]$; $z_s\in[0.5, 2.44]$ with per-plane
   coefficients; group-bin latents at the stated epochs.
4. **Error model**: use the per-band CV residual std as the model-error term.
5. **Boundaries**: $C_\ell^{yy}$ (profile-level pressure), per-bin nu05
   peak/minima counts (realization noise).
6. **Aperture convention**: all $\tilde f$ are projected-cylinder quantities; the
   cylinder→3-D calibration (fig 20c: live truth anchor $\tilde f^{\rm cyl}=0.929$
   in the exact vD bin / Nelson+24 3-D 0.81 → ÷1.147, ±0.03 anchor sensitivity
   printed) applies only where literature 3-D comparisons are drawn — the model
   itself lives entirely in the self-consistent cylinder system.

## 9. Reproduction

```bash
cd papers/01_pipeline
/mnt/home/mlee1/venvs/BIND_env/bin/python _check_builder.py     # rebuild + syntax gate
/mnt/home/mlee1/venvs/BIND_env/bin/python _run_subset.py \
    fig20d_budget_partition fig20e_analytic_model fig20f_redshift_thermal
# any one fig20* name executes the whole §3c cell (all six figures + prints);
# fig03_halo_validation is auto-pulled as the OB_OM prerequisite (DEPS).
```

- Figures → `figs_v2/fig20{a..f}_*.pdf` + `figs_preview/*.png` (gitignored).
- Quoted numbers → live cell prints, recorded in `FIGURE_NUMBERS.md`.
- Session narrative → `docs/WORKLOG.md` (2026-08-05 entries).
- Data inputs: `bind_sb35/emulator_dataset_nu05.npz`,
  `bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap{029..096}.npz`,
  `bind_science/ksz_confront/bind_tauy_{xprof,fiducial}_snap085.npz`,
  `bind_science/halo_atlas/fid_snap096.npz`,
  `bind_science/runs/{bind,dmo}/run_0000/` (paired spectra).
- Planned (not yet implemented; design in WORKLOG): a `latents` mode for
  `bind.emulator` — `LinearBackend` in `BACKEND_REGISTRY` + an
  `inputs="latents"` X-swap in `Emulator` + a latent table in `EmulatorDataset`;
  blocked on the uncommitted emulator-module changes from the repaint campaign.
