# Math skeleton of the analytic-model narrative (arc v2, 2026-08-06)

The equations of the paper's flow, in the paper's order. Each beat: the math,
its figure slot (PAPER_FIGURE_MAP.md), and the live-printed numbers to quote.
Notation is uniform throughout: universes $q$, parameters $\theta \in
\mathbb{R}^{30}$, latents $\lambda \in \mathbb{R}^4$, multipole bands
$\ell_b$ ($b = 1..24$, $\ell \in [300, 3\times10^4]$), source plane $z_s$
(working plane $z_s = 1$).

---

## Beat 1–3 (locked): setup, halo validation, map validation

Only one definition is needed downstream. For every universe $q$:

$$S_q(\ell) \;=\; \frac{C_\ell^{\kappa\kappa}[q]}{C_\ell^{\kappa\kappa}[\rm DMO]}
\qquad\text{(seed-paired ratio; band means over the 24 log bands)}$$

and more generally a statistic vector $D_q$ = the concatenated bins of any of
the 13 statistics (PDF, peaks, minima, $V_{0,1,2}$, the three scaling
relations, the three SZ/$\tau$ spectra in $\log_{10}$, band-averaged like
$S$).

## Beat 4 (locked): parameter effects and the shape families

*Figure: `pfig_s3b_cl_clusters` (+ PDF appendix).*

Per-parameter gradient from the twobound pair (all else fixed):

$$g_j(\ell) = S\!\left(\theta^{(j,\rm hi)}\right)(\ell) - S\!\left(\theta^{(j,\rm lo)}\right)(\ell)
\;\approx\; \Delta\theta_j\,\frac{\partial S(\ell)}{\partial \theta_j},
\qquad
\hat g_j(\ell) = \frac{g_j(\ell)}{g_j(\ell_{\rm peak})}$$

Clustering: distance $d_{jk} = 1 - \mathrm{corr}(\hat g_j, \hat g_k)$,
average linkage, tree cut at $d = 0.15$, after a peak-S/N $\geq 3$ gate.
**Numbers**: four families + one singleton; $\bar r = 0.98/0.95/0.96/0.96$
(C1–C4); C3 contains IMFslope. *Why the families exist is deferred — the
narrative hook into the pivot.*

## Beat 5a — model construction (the pivot opens)

*Figure: `pfig_s4a_model_construction` (a–d).*

**The ansatz** (panel a: a universe is four numbers):

$$S_q(\ell) \;=\; c_0(\ell) \;+\; \sum_{i=1}^{4} c_i(\ell)\,\lambda_{q,i} \;+\; \varepsilon_q(\ell)$$

**What a coefficient is** (panel b): at one band, $c_1(\ell_b)$ is the
*partial* regression slope of $S$ on $\tilde f_{\rm bar}$ across the Sobol
cloud — one number ($c_1(\ell\!\simeq\!4847) = +0.20$). **The kernels**
(panel c): the per-band slopes strung across $\ell$,

$$\beta(\ell_b) = \big(c_1, c_2, c_3, c_4, c_0\big)(\ell_b)
= (A^{\top}A)^{-1} A^{\top} S(:, \ell_b),
\qquad A = \big[\lambda_{q,1}, \lambda_{q,2}, \lambda_{q,3}, \lambda_{q,4}, 1\big]_{q}$$

($A$ is the **design matrix** — nothing spectral: the plain data table with
one row per universe, columns = its four measured latent values and a
constant 1 for the intercept, shape $N \times 5$. E.g. row for run_0000:
$[0.980,\ 0.101,\ 0.772,\ 6.809,\ 1]$.)

Joint, not marginal: the latents are collinear
($r(c_{\rm gas}, \tilde f_{\rm bar}) = +0.94$), so one-variable slopes would
double-count. **The chain-rule preview** (panel d): for one knob,

$$\Delta S_j(\ell) \;=\; \sum_i c_i(\ell)\,\Delta\lambda_i^{(j)}
\qquad\text{(WindEnergy: } r = 0.99\text{, nothing fitted)}$$

## Beat 5b — the latent choice

*Text + `latent_ablation.py`; definitions in ANALYTIC_LATENT_MODEL.md §0.*

One convention (snapshot 096, hinge bin
$\log_{10} m^{\rm bg}_{\rm tot,500c} \in [13.3, 13.6)$, projected cylinders,
2.5–3.0 $h^{-1}$Mpc annulus, bin medians):

$$\tilde f_{\rm bar} = \frac{\mathrm{med}\big[(m^{\rm bg}_{\rm gas,500c} + m_{\star,500c})/m^{\rm bg}_{\rm tot,500c}\big]}{\Omega_b/\Omega_m},\qquad
\tilde f_\star = \frac{\mathrm{med}\big[m_{\star,500c}/m^{\rm bg}_{\rm tot,500c}\big]}{\Omega_b/\Omega_m}$$

$$c_{\rm gas} = \mathrm{med}\big[m^{\rm bg}_{\rm gas,500c}/m^{\rm bg}_{\rm gas,200c}\big],\qquad
\log\tilde T = \log_{10}\mathrm{med}\big[T_{\rm mw,500c}\big],\;\;
T_{\rm mw} = \tfrac{\sum T\,\Sigma_{\rm gas}}{\sum \Sigma_{\rm gas}}$$

Selection ablation (objective: mean over 9 statistics of median per-bin
CV-$R^2$): exhaustive over $\binom{15}{4} = 1365$ quads → ours ranks 50th
(top 3.7%), $0.007$ below optimum; $\tilde f_\star$ in every top-12 quad;
forward-selection scores $0.53/0.82/0.89/0.90/0.92/0.93$ for sizes 1–6
(elbow at 3–4).

## Beat 5c — coefficients from the Sobol set

*Figures: `fig20g_latent_kernels`, `pfig_s4a_cv_buildup`.*

The kernel table is the model: $\beta(\ell_b, z_s)$, $5 \times 24 \times 5$
numbers, with analytic standard errors
$\mathrm{SE}_i = [\hat\sigma^2_{\rm res}\,((A^\top A)^{-1})_{ii}]^{1/2}$.
Tomography: one latent vector per universe; all $z_s$ dependence in
$c_i(\ell, z_s)$ (amplitude dilution printed). Accuracy buildup
(deterministic 5-fold CV, folds = node index mod 5):

$$R^2_{\rm CV}(\ell_b) = 1 - \frac{\langle (S - \hat S_{\rm CV})^2 \rangle_q}{\mathrm{Var}_q(S)}$$

**Numbers**: medians $0.85 \to 0.91 \to 0.92 \to 0.94$ for
$\tilde f_{\rm bar} \to +\tilde f_\star \to +c_{\rm gas} \to +\log\tilde T$;
the 30-raw-parameter linear baseline: $0.53$. *The latents linearize the
response; the parameters do not.*

## Beat 5d — one model for every statistic

Same machinery per statistic $s$ with bins $x$:

$$y_{q,s}(x) = b_{0,s}(x) + \sum_i B_{s,i}(x)\,\lambda_{q,i} + \varepsilon$$

**Numbers** (4-latent median CV-$R^2$): PDF 0.85, MF $V_1$ 0.94, $V_2$ 0.93,
Y–M 0.94, $f_{\rm gas}$–M 0.97, T–M 0.77, $\kappa\tau$ 0.86, $yy$ 0.77
(stated boundary: profile-level pressure), peaks/minima per-bin
noise-dominated (spread $= 0.4$–$0.5\times$ measurement error).

## Beat 5e — validation

*Figures: `pfig_s4b_model_curves` (fig09c layout), `pfig_s4b_reconstruction`,
`pfig_s4b_generality`.*

Leave-one-out protocol: for shown node $g$, fit $\beta^{(-g)}$ on the other
nodes, predict $\hat y_g = [\lambda_g, 1] \cdot \beta^{(-g)}$ — the node's
own data never enters its prediction. Per-panel stamp:
$\mathrm{med}|\hat y - y| / \mathrm{med}|y| \times 100 = 0.1$–$1.0\%$ across
13 statistics. Out-of-design test: the fiducial (not a Sobol node; latents
measured from its own atlas) reproduces $S(\ell)$ at RMS $0.026$ full set /
$0.015$ without the thermal kernel ($\log\tilde T$ spans only 0.09 dex — the
least out-of-design-robust leg), vs cloud spread $0.057$.

## Beat 5f — tie back to the families

*Figure: `pfig_family_kernel_bridge`.*

Chain rule through the model, per parameter and per family:

$$\hat g_j(\ell) \;\propto\; \sum_i c_i(\ell)\,\Delta\lambda_i^{(j)},
\qquad \Delta\lambda^{(j)} = \text{the fingerprint, measured from halo catalogs}$$

Out-of-design closure (Sobol kernels × twobound fingerprints × twobound
shapes; zero fitted degrees of freedom): family means at
$r = 1.00/0.99/0.98/0.98$ (C1–C4), singleton $0.86$. **A family is a set of
parameters sharing a fingerprint; the families exist because there are only
four kernels.** Counting facts for the referee: no family is axis-aligned
(max $|\cos| = 0.77$); the shapes alone are $\sim$2-dimensional (SVD 77.5% +
21.9%) — the kernel basis is recoverable only with the halo data.

## Beat 6 — astrophysical constraints (the corner)

*Figure: `fig20i_latent_corner`.*

Prior and test are **measured** latents (no $\theta\to\lambda$ regression
anywhere). Score compression per statistic family $f$:

$$t_f = B_f^{\top} C_f^{-1}\,(D - b_{0,f}) \in \mathbb{R}^4
\qquad (C_f = \mathrm{diag[train\ resid\ var]}\text{: compression weighting only})$$

Sellentin–Heavens likelihood on the stacked scores ($q = 4k \leq 24$) with
their full empirical covariance $S_t$; with the linear model and flat prior
the posterior is an analytic 4-dim Student-$t$:

$$\hat\lambda = P^{-1} M^{\top} S_t^{-1} t,\quad P = M^{\top} S_t^{-1} M,\quad
\nu = n_{\rm tr} - 4,\quad
\Sigma_t = \frac{n_{\rm tr} - 1 + \chi^2_{\rm min}}{\nu}\, P^{-1}$$

No Hartlap, no temperature; LOO 68% coverage $= 0.73/0.73/0.69/0.66/0.66/0.64$
across the six stages (nominal $0.68 \pm 0.03$) is a *verification*.
Shrinkage: $S(\ell)$ alone $\sigma = 0.030/0.061/0.016/0.035$ → all
statistics $0.0035/0.0009/0.0040/0.0031$ (34×/89×/13×/5× below the prior;
$\tilde f_\star$ flagged near-circular via the $f_\star$–M scaling).
**Mandatory framing**: internal recovery test (held-out node's own
seed-paired measurements; no observational noise or systematics; same-suite
prior) — information content and calibration, not a survey forecast.
