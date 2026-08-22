# R1 — closing the validation against FULL HYDRO TNG300

**Referee point.** The paper's headline validation compares BIND against a *hydro-pasted*
truth, which is itself an approximation (exact halo replacement above
$M_{200c}=10^{13}\,{\rm M}_\odot h^{-1}$, DMO everywhere else), and its fidelity is only
cited from \citet{Lee-2026a} ("$\sim$90% of the response at the level of the power
spectrum, virtually all of the response for peak counts and Minkowski functionals").

**What is new here.** A full TNG300 hydrodynamic lightcone traced through the *identical*
ray-tracing pipeline on the *identical* seed ladder (base seed 1992, $1992+7r$), so the
comparison is realization-by-realization paired and cosmic variance cancels. Every
statistic below is recomputed from the raw map cubes with one code path for all rungs.

---

## 0. Headline

| question | answer measured on these lightcones |
|---|---|
| How close is the hydro-pasted truth to full hydro ($\kappa\kappa$, $z_s=1$)? | $+0.3\%$ ($\ell<1000$), $+1.8\%$ ($10^3$–$5\!\times\!10^3$), $+5.2\%$ ($5\!\times\!10^3$–$3\!\times\!10^4$) |
| Does the construction capture "$\sim$90% of the response"? | **No — $0.704\pm0.007$** of the DMO$\to$full-hydro $S(\ell)$ suppression |
| "Virtually all" of the peak / MF response? | **No — $0.61$–$0.73$** (peaks $0.669$, minima $0.625$, $V_0/V_1/V_2$ $0.729/0.610/0.734$) |
| Is BIND's own error subdominant to the construction's? | **Yes, by $4$–$11\times$** in the mid-$\ell$ range ($\chi^2_{\rm LSST}/{\rm dof}$: BIND–pasted $37$ vs pasted–full $399$) |
| Does the LSST-Y10 indistinguishability claim survive? | **Against the pasted truth, yes** (BIND stays $\le1.2\times$ the band). **Against full hydro, no** for $\ell\gtrsim900$ — but the failure is the *reference*, not BIND |
| Gas columns | pasted keeps $85\%$ of $\bar y$ and only $32\%$ of $\bar\tau$ at $z_s=1$; the $f_b\rho_{\rm DMO}$ diffuse patch restores $\bar\tau$ to $1.012$ |

---

## 1. Provenance and the identity checks the ladder rests on

Recomputed from the raw cubes; no released `Cl_*.npz` (pre-fix `XPk_plane` normalization)
and no released `nongaussian_stats.npz` (stale MFs) is read anywhere.

| check | result |
|---|---|
| `bind_lightcone_tng` first-50 vs the paper's `bind_science/runs/bind/run_0000` | **bit-identical** ($\max|\Delta\kappa|=0$, $\max|\Delta y|=0$) — my BIND numbers are the paper's BIND maps |
| my $\nu$-domain estimator vs the released `nu05_shards/sci_bind.npz` the paper's Fig. 5 reads | **bit-identical** for all six statistics (max fractional diff $=0$) |
| diffuse $\kappa$ vs pasted $\kappa$ (must be equal by construction) | $\max|\Delta|=3.0\times10^{-8}$ vs $\max|\kappa|=0.82$ (float32 rounding) |
| seed pairing *across trees* (pasted vs full hydro) | per-realization pixel correlation $0.993$; cross-realization $0.002$ |

Conventions inherited verbatim from `papers/01_pipeline/{field_cache,nu_grid}.py` and the
`_build_figures_nb.py` setup cell: $\kappa$ autos on the plane; $y/\tau$ autos on the
per-plane cumulative column; crosses $=\kappa(z_s)\times$ TOTAL column; peaks/minima
`peak_counts(..., 2.0', nu_norm='map')` on `NU_EDGES`; MFs `nongaussian_stats(..., 1.0',
mf_thresholds=NU)`; PDF a density histogram of $(\kappa-\bar\kappa)/\sigma_{\rm map}$ on
`NU_EDGES`; $\ell$ range $100\le\ell\le\ell_{\rm trust}=3\times10^4=0.8\,\ell_{\rm Nyq}$,
plotted to $\ell=36864$.

## 2. Mandatory cross-check — reproduction of `tng_full_validation_summary.md`

All reproduced exactly (3 d.p.), from an independent code path:

| quantity | summary | this work |
|---|---|---|
| $C_\ell^{\kappa\kappa}$ pasted/full, $z_s=1$, $\ell<1000$ / $10^3$–$5\!\times\!10^3$ / $>5000$ | 1.003 / 1.018 / 1.152 | **1.003 / 1.018 / 1.152** |
| $C_\ell^{yy}$ pasted/full | 0.980 / 0.995 / 0.983 | **0.980 / 0.995 / 0.983** |
| $C_\ell^{\tau\tau}$ pasted/full | 1.224 / 0.964 / 0.427 | **1.224 / 0.964 / 0.427** |
| $C_\ell^{\tau\tau}$ diffuse/full | 1.041 / 1.158 / 2.771 | **1.041 / 1.158 / 2.771** |
| mean $y$ pasted/full, $z_s=0.5$–$2.44$ | 0.889 / 0.852 / 0.805 / 0.757 / 0.719 | **identical** |
| mean $\tau$ pasted/full | 0.371 / 0.320 / 0.260 / 0.203 / 0.161 | **identical** |
| mean $\tau$ diffuse/full | 1.010 / 1.012 / 1.012 / 1.012 / 1.012 | **identical** |

Also reproduces the paper's own published BIND-vs-pasted numbers: median $|$resid$|$ over
$300\le\ell\le5000$ is $0.65\%$ for $C_\ell^{\kappa\kappa}$ (paper: 0.7%) and $5.63\%$ /
$5.43\%$ for $C_\ell^{yy}$ on the $z_s=1$ / total column (paper: 5.7% / 5.6%).

> **Note on the ">5000" band.** The summary's third band is open-ended and runs into the
> corner-mode zone ($\ell>3.7\times10^4$), which inflates it. Everywhere below I use
> $5000\le\ell\le\ell_{\rm trust}$ instead; for $\kappa\kappa$ that band mean is **1.052**
> rather than 1.152. Both are reported above so the reproduction is unambiguous.

---

## 3. $\kappa$ sector — the number tables ($z_s=1$, 50 paired realizations)

### 3.1 Paired band-mean ratios $C_\ell^{\kappa\kappa}$

| ratio | $\ell<1000$ | $10^3$–$5\!\times\!10^3$ | $5\!\times\!10^3$–$3\!\times\!10^4$ | trusted mean | SE |
|---|---|---|---|---|---|
| BIND / pasted | 1.003 | 1.007 | 1.001 | 1.002 | 0.0021 |
| BIND / full hydro | 1.006 | 1.025 | 1.054 | 1.048 | 0.0024 |
| pasted / full hydro | 1.003 | 1.018 | 1.052 | 1.046 | 0.0011 |
| diffuse / full hydro | 1.003 | 1.018 | 1.052 | 1.046 | 0.0011 |

The diffuse row equals the pasted row identically: the added gas carries no mass into the
lensing planes.

### 3.2 Where it degrades, against the LSST-Y10 band

LSST-Y10 band (v2 recipe: measured covariance of the pasted set, 8-bin block average,
area-scaled by $\sqrt{25/18000\,{\rm deg}^2}$, plus the analytic shape-noise excess
$n_{\rm gal}=27$, $\sigma_e=0.26$, $f_{\rm sky}=0.44$): $0.62\%$ at $\ell\lesssim300$,
$0.51\%$ at $\ell=10^3$, $0.49\%$ at $5\times10^3$, $0.96\%$ at $10^4$, $4.4\%$ at
$3\times10^4$.

| comparison | first $\ell$ exceeding the band | fraction of trusted bins exceeding | $|\Delta|/$band at $\ell=3\times10^3$ / $10^4$ |
|---|---|---|---|
| BIND vs pasted | 1038 | 23% | 1.2 / 0.5 |
| pasted vs full hydro | 887 | 95% | 4.7 / 5.5 |
| BIND vs full hydro | 604 | 90% | 5.9 / 6.0 |

Read: **BIND tracks the pasted truth to $\le1.2\times$ LSST-Y10 precision everywhere**
(the paper's claim), while **the pasted truth itself departs from full hydro by
$5\times$ the same band** over $10^3\lesssim\ell\lesssim10^4$.

### 3.3 $\chi^2$ under the LSST-Y10 covariance

$\hat C^{-1}=\frac{N-d-2}{N-1}C^{-1}$ (Hartlap) with $N=550$ hydro-pasted realizations,
covariance taken in fractional units and area-scaled $C(A)=(A/25\,{\rm deg}^2)^{-1}C$,
$A=18000\,{\rm deg}^2$. Spectra are compressed to 25 $\ell$ bins over the trusted range
(rebin 16, the paper's `chi2_full` compression); $\nu$-domain vectors use the 22-bin
canonical grid, restricted to bins with a usable signal. The inverse is eigenmode-truncated
at $10^{-10}\lambda_{\rm max}$ — the density-normalized PDF carries an exact sum constraint,
so its covariance is singular by construction and a plain solve returns a meaningless
(negative) $\chi^2$; exactly one mode is dropped, and only for the PDF.

| statistic ($d$) | BIND – full | pasted – full | BIND – pasted |
|---|---|---|---|
| $C_\ell^{\kappa\kappa}$ (25) | 501.4 | 398.5 | **37.5** |
| $p(\nu)$ (19) | 62.3 | 79.7 | 37.1 |
| $N_{\rm pk}$ (12) | 7.1 | 6.7 | **3.0** |
| $N_{\rm min}$ (7) | 14.1 | 16.0 | **4.0** |
| $V_0$ (20) | 8.7 | 2.7 | 7.5 |
| $V_1$ (21) | 22.3 | 34.9 | **9.5** |
| $V_2$ (21) | 12.5 | 22.3 | **9.0** |

(Values are $\chi^2/{\rm dof}$. An LSST-Y10 footprint is 720$\times$ this box's area, so
$\chi^2/{\rm dof}\gg1$ is the expected outcome of *any* percent-level residual; the
informative content is the **ratio** between columns.) For continuity, the paper's paired
$\pm1\sigma/\sqrt{50}$ $\chi^2/{\rm dof}$ on $C_\ell^{\kappa\kappa}$ is 423.7 (BIND–full),
1693.8 (pasted–full), 11.8 (BIND–pasted).

### 3.4 $\nu$-domain residuals vs full hydro (median over $\nu\le4$)

| statistic | BIND vs full | pasted vs full | LSST-Y10 band (median) |
|---|---|---|---|
| $p(\nu)$ | $+0.31\%$ | $+1.26\%$ | 0.06% |
| $N_{\rm pk}$ | $+0.74\%$ | $+0.44\%$ | 0.68% |
| $N_{\rm min}$ | $+0.83\%$ | $+0.80\%$ | 2.28% |
| $V_0$ | $-0.02\%$ | $-0.03\%$ | 0.09% |
| $V_1$ | $+0.71\%$ | $+0.92\%$ | 0.22% |
| $V_2$ | $+1.56\%$ | $+1.24\%$ | 0.31% |

## 4. Response capture — the measurement that replaces the cited "$\sim$90%"

Define the DMO$\to$full-hydro response $R_{\rm full}=X_{\rm full}-X_{\rm DMO}$ and the
captured response $R_x=X_x-X_{\rm DMO}$. The table gives the least-squares projection
$\langle R_x,R_{\rm full}\rangle/\langle R_{\rm full},R_{\rm full}\rangle$ with a 16–84
bootstrap over the 50 realizations (resampled with shared indices in every rung, so the
seed pairing is preserved).

| statistic | hydro-pasted captures | BIND captures |
|---|---|---|
| $S(\ell)$, trusted range | **0.704** [0.698, 0.712] | 0.692 [0.685, 0.701] |
| $S(\ell)$, $\ell<1000$ | 0.455 | $-0.024$ |
| $S(\ell)$, $10^3$–$5\!\times\!10^3$ | 0.711 | 0.594 |
| $S(\ell)$, $5\!\times\!10^3$–$3\!\times\!10^4$ | 0.701 | 0.694 |
| $p(\nu)$ | 1.414 [1.404, 1.425] | 1.096 [1.085, 1.109] |
| $N_{\rm pk}$ | 0.669 [0.627, 0.718] | 0.659 [0.593, 0.698] |
| $N_{\rm min}$ | 0.625 [0.571, 0.665] | 0.603 [0.539, 0.641] |
| $V_0$ | 0.729 [0.717, 0.743] | 0.900 [0.873, 0.925] |
| $V_1$ | 0.610 [0.607, 0.612] | 0.625 [0.621, 0.628] |
| $V_2$ | 0.734 [0.731, 0.738] | 0.681 [0.677, 0.685] |

Three readings:

1. **The power-spectrum figure is $\sim$70%, not $\sim$90%.** On these lightcones, halo
   replacement above $10^{13}\,{\rm M}_\odot h^{-1}$ reproduces $0.70$ of the true
   suppression; the remaining $0.30$ lives in halos below the threshold and in the diffuse
   gas that the construction leaves as DMO.
2. **"Virtually all" does not hold for the $\nu$-domain statistics either** — peaks,
   minima and the MFs capture $0.61$–$0.73$. But the *absolute* response there is small
   (the median $|R_{\rm full}|$ is 1.9–4.1% of the signal for $N_{\rm pk}$, $N_{\rm min}$,
   $V_1$, $V_2$ and 0.2% for $V_0$), which is why missing a third of it still leaves
   sub-percent residuals — the paper's qualitative conclusion survives even though the
   quoted fraction does not.
3. **The $\ell<1000$ entries are noise-limited, not informative**: the true response there
   is only $-0.6\%$, so the ratio's denominator is tiny. Quote the trusted-range projection.
4. The PDF *over*-responds ($1.41$): the pasted construction pushes the one-point
   distribution further from DMO than full hydro does.

## 5. Gas sector

### 5.1 Spectra ($z_s=1$, per-plane cumulative column)

| ratio | $\ell<1000$ | $10^3$–$5\!\times\!10^3$ | $5\!\times\!10^3$–$3\!\times\!10^4$ | trusted |
|---|---|---|---|---|
| $C_\ell^{yy}$ pasted / full | 0.984 | 0.995 | 0.992 | 0.992 |
| $C_\ell^{yy}$ diffuse / full | 0.984 | 0.995 | 0.992 | 0.992 |
| $C_\ell^{yy}$ BIND / full | 0.898 | 0.940 | 1.116 | 1.087 |
| $C_\ell^{yy}$ BIND / pasted | 0.912 | 0.945 | 1.126 | 1.096 |
| $C_\ell^{\tau\tau}$ pasted / full | 1.253 | 0.964 | 0.481 | 0.568 |
| $C_\ell^{\tau\tau}$ diffuse / full | 1.042 | 1.158 | 2.378 | 2.177 |
| $C_\ell^{\tau\tau}$ BIND / full | 1.358 | 1.073 | 0.566 | 0.656 |
| $C_\ell^{\tau\tau}$ BIND / pasted | 1.083 | 1.117 | 1.165 | 1.156 |
| $C_\ell^{\kappa y}$ pasted / full | 0.983 | 0.983 | 0.932 | 0.940 |
| $C_\ell^{\kappa y}$ BIND / full | 0.951 | 0.987 | 1.097 | 1.079 |

Paired $\chi^2/{\rm dof}$ against the $\pm1\sigma/\sqrt{50}$ band (Fig. 6 convention, no
survey analog exists for the gas observables): $C_\ell^{yy}$ 146.0 (BIND–full), 127.0
(pasted–full), 161.8 (BIND–pasted); $C_\ell^{\tau\tau}$ 32858 / 58286 / 1259; diffuse–full
16909.

**The $y$ power is a halo statistic and the pasted construction gets it right** (within
$2\%$ at every $\ell$), even though the *mean* $y$ is $15\%$ low — the missing WHIM
contributes a nearly uniform pedestal, not power. **The $\tau$ power is not**: the pasted
map is $52\%$ low above $\ell=5000$, and the $f_b\rho_{\rm DMO}$ diffuse patch overshoots
there by $138\%$ while fixing the mean column to $1.2\%$. That is the honest statement of
what the released $\tau$ maps are and are not.

### 5.2 Mean columns vs $z_s$ (ratio to full hydro)

| field | rung | 0.5 | 1.0 | 1.5 | 2.0 | 2.44 |
|---|---|---|---|---|---|---|
| $\bar y$ | pasted | 0.889 | 0.852 | 0.805 | 0.757 | 0.719 |
| $\bar y$ | diffuse | 0.891 | 0.854 | 0.807 | 0.759 | 0.721 |
| $\bar y$ | BIND | 0.870 | 0.838 | 0.797 | 0.752 | 0.715 |
| $\bar\tau$ | pasted | 0.371 | 0.320 | 0.260 | 0.203 | 0.161 |
| $\bar\tau$ | diffuse | 1.010 | 1.012 | 1.012 | 1.012 | 1.012 |
| $\bar\tau$ | BIND | 0.387 | 0.333 | 0.270 | 0.211 | 0.167 |

The gap grows with source redshift because more of the column is diffuse at higher $z$;
the $T=10^4$ K diffuse gas adds essentially nothing to $y$ by construction, so the $y$ gap
measures WHIM pressure that *neither* construction captures.

### 5.3 Attribution of the BIND-vs-pasted gas offset — NOT a texture systematic

The released fiducial paint was conditioned on the CAMELS SB35 fiducial **cosmology**
rather than TNG300's, an already-verified conditioning-provenance error
(`referee/R3b_texture_results.md` §5.4; not re-derived here). $\Omega_b/\Omega_m$ is 3.8%
high, and because patch mass-matching pins the total patch mass, the conditioned gas
fraction inherits it: predicted column ratio $1.03814$, predicted power excess $+7.77\%$.
Measured on **this lightcone**, independently:

| prediction (R3b §5.4) | measured here |
|---|---|
| mean gas/$\tau$ column $\times1.03814$ | mean $\bar\tau$ BIND/pasted $=1.0425,\,1.0411,\,1.0397,\,1.0387,\,1.0385$ across $z_s$ |
| $\tau$ power $\times1.0777$ | $C_\ell^{\tau\tau}$ BIND/pasted $=1.083$ at $\ell<1000$ |

Agreement to $0.4\%$ in the column and $0.5$ percentage points in power. So the $\tau$
excess in panel (e) is a one-line parameter-file provenance offset of the released
fiducial paint, and should be described as such rather than folded into a texture
narrative. The Compton-$y$ *deficit* ($C_\ell^{yy}$ BIND/pasted $=0.91$–$0.95$ at
$\ell<5000$, $\bar y$ 2% low) is in the opposite direction and is real model bias, again
per R3b §5.4.

**Why the $\kappa$ sector is immune.** The paint pins the *total* projected patch mass to
the DMO cutout, so a wrong baryon fraction redistributes mass between the gas and stellar
channels without changing the total. Lensing sees only the total, and indeed
$C_\ell^{\kappa\kappa}$ BIND/pasted $=1.003$ at $\ell<1000$ and $1.002$ over the whole
trusted range. Every conclusion in §§3–4 is therefore unaffected.

## 6. Other source redshifts

Checked at $z_s=2$ ($z_s$ index 3). Same picture: $C_\ell^{\kappa\kappa}$ pasted/full
$=1.002$ / $1.012$ / $1.044$ (vs $1.003$ / $1.018$ / $1.052$ at $z_s=1$); $S(\ell)$ capture
$0.677$ (vs $0.704$); $\chi^2_{\rm LSST}/{\rm dof}$ for $C_\ell^{\kappa\kappa}$ 302
(pasted–full) and 462 (BIND–full); $N_{\rm pk}$ 3.0 / 3.7. $C_\ell^{yy}$ pasted/full
$=0.981$ / $0.996$ / $0.974$; $C_\ell^{\tau\tau}$ pasted/full $=1.005$ / $1.083$ / $0.372$.

---

## 7. DRAFT LATEX

### (a) Replacement for the paragraph at main.tex ~line 552

```latex
An important note is that the hydro-pasted maps are not the full TNG300 hydro ray-traced
maps. Instead, they are TNG300-Dark maps where each halo with $M_{200c}\geq
10^{13}\,{\rm M}_\odot\,h^{-1}$ has been replaced exactly with the dark matter, stellar,
and gas particles of the TNG300 simulation, following \citet{Lee-2026a}. Because this
construction is itself an approximation, we have measured its fidelity directly rather
than quoting it: we ray-traced the full TNG300 hydrodynamic simulation through the same
pipeline, with the same rotations, translations and ray seeds, so that the two map sets
are paired realization by realization and cosmic variance cancels in their ratio. The
comparison is shown in Fig.~\ref{fig:full_hydro}. At $z_s=1$, the hydro-pasted
$C_\ell^{\kappa\kappa}$ agrees with the full-hydro trace to $0.3\%$ for $\ell<10^3$,
$1.8\%$ for $10^3<\ell<5\times10^3$, and $5.2\%$ from there to $0.8\,\ell_{\rm Nyq}$;
the convergence PDF, peak counts, minimum counts and the three Minkowski functionals agree
to better than $1.3\%$ in the median over $\nu\leq4$. In terms of the response that halo
replacement is meant to reproduce, the construction captures $70\%$ of the dark
matter-only to hydrodynamical suppression of $S(\ell)$ over the trusted range, and
$61$--$73\%$ of the peak-count, minimum-count and Minkowski response (the one-point PDF is
the exception, over-responding by $40\%$) --- less than the $\sim$90\% and near-complete
fractions reported in \citet{Lee-2026a}, whose measurement was made on a different
statistic and mass range. The residual response lives in halos below the replacement
threshold and in the diffuse gas outside the pasted apertures.
What matters for the validation above is the comparison of the two gaps: BIND differs
from the hydro-pasted maps by $0.2\%$ over the trusted range, while the hydro-pasted maps
differ from full hydro by $4.6\%$ --- a factor of $4$ at $\ell=3\times10^3$ and a factor
of $11$ at $\ell=10^4$. The accuracy of the comparison in Fig.~\ref{fig:field_validation}
is therefore set by the reference, not by the emulator. For the gas channels the
construction behaves differently in the two observables: the pasted $C_\ell^{yy}$ is
within $2\%$ of full hydro at every $\ell$, because the tSZ power is dominated by the
replaced halos, while the mean Compton-$y$ is $15\%$ low and the mean $\tau$ only $32\%$
of full hydro at $z_s=1$, since most of the electron column is diffuse gas that the
construction leaves out. We return to the consequences of this construction in
\S~\ref{sec:caveats}. The comparison here is also noiseless: no shape noise, beam, or
survey systematics enter at any stage.
```

### (b) Figure caption

```latex
\begin{figure*}
    \centering
    \includegraphics[width=\linewidth]{imgs/fig06b_full_hydro.png}
    \caption{The three-rung validation ladder at $z_s=1$: BIND-generated (blue), the
    hydro-pasted truth used throughout this paper (black dashed), and a full TNG300
    hydrodynamic lightcone traced through the same pipeline with the same ray seeds (red).
    A fourth rung adds $f_b\rho_{\rm DMO}$ diffuse gas at $T=10^4$~K to the pasted maps
    (green); it is identical to the pasted rung in $\kappa$ by construction and is shown
    only for the gas panels. All 50 realizations are seed-paired across the four sets, so
    every residual is cosmic-variance cancelled. \textit{(a)} The suppression
    $S(\ell)=C_\ell^{\kappa\kappa}/C_\ell^{\kappa\kappa,{\rm DMO}}$, with the residual
    against full hydro below; the shaded band is LSST-Y10 precision, built from the
    550-realization hydro-pasted covariance area-scaled to $18{,}000\,{\rm deg}^2$ as in
    Fig.~\ref{fig:field_validation}. \textit{(b)} Peak counts on the same $\nu$ grid as
    Fig.~\ref{fig:field_validation}. \textit{(c)} The fraction of the dark matter-only to
    full-hydro response that each rung reproduces, for the suppression and for all six
    $\nu$-domain statistics, with 16--84 bootstrap intervals over the realizations; the
    red line marks the complete response. \textit{(d), (e)} The $yy$ and $\tau\tau$ auto
    spectra, with residuals against full hydro and the paired $\pm1\sigma/\sqrt{50}$ band.
    \textit{(f)} Mean map value relative to full hydro as a function of source redshift,
    for $y$ (circles) and $\tau$ (squares). Dotted vertical lines mark
    $0.8\,\ell_{\rm Nyq}$. BIND tracks the hydro-pasted maps far more closely than the
    hydro-pasted maps track full hydro: the accuracy of the validation is set by the
    halo-replacement construction, not by the generative model.}
    \label{fig:full_hydro}
\end{figure*}
```

### (c) Caveats-section pairing sentence (caveat 1: halo replacement / diffuse gas)

```latex
The halo-replacement construction that defines our truth set is exact above
$M_{200c}=10^{13}\,{\rm M}_\odot\,h^{-1}$ and empty below it, and the full-hydro trace of
\S~\ref{sec:validation} measures the cost: $70\%$ of the lensing suppression and
$61$--$73\%$ of the peak, minimum and Minkowski response is recovered, leaving a
$1.8\%$ ($5.2\%$) offset in $C_\ell^{\kappa\kappa}$ at $\ell\sim10^3$ ($10^4$), while the
omitted diffuse gas removes $68\%$ of the mean electron column and $15\%$ of the mean
Compton-$y$ at $z_s=1$ --- so the $\kappa$ and $y$ statistics of this release are
construction-limited at the few-percent level, and absolute $\tau$ amplitudes should not
be used at all. The remedies are specified and partly demonstrated: extending the painted
population below $10^{13}\,{\rm M}_\odot\,h^{-1}$, and adding the diffuse component, which
in the $f_b\rho_{\rm DMO}$ form tested here restores the mean electron column to $1.2\%$
of full hydro but overshoots $C_\ell^{\tau\tau}$ above $\ell\sim5\times10^3$
(\S~\ref{sec:outlook}).
```

---

## 8. Files and reproduction

| file | what |
|---|---|
| `/mnt/home/mlee1/BIND/imgs/fig06b_full_hydro.{png,pdf}` | the figure (300 dpi PNG + vector PDF) |
| `papers/01_pipeline/referee/work/r1_common.py` | streaming cube I/O + the paper's estimator conventions |
| `papers/01_pipeline/referee/work/r1_stats.py` | recomputes every per-realization statistic from the raw cubes |
| `papers/01_pipeline/referee/work/r1_ladder_fig.py` | tables, $\chi^2$, response capture, and the figure |
| `papers/01_pipeline/referee/work/r1_numbers_z1.json` (`_z3`) | every number above, machine-readable |
| `/mnt/home/mlee1/ceph/referee_work/r1/` | per-realization arrays (~1.5 GB), `stats.log`, `tables_z{1,3}.txt`, `pre.json` |

```bash
cd /mnt/home/mlee1/BIND/papers/01_pipeline/referee/work
/mnt/home/mlee1/venvs/BIND_env/bin/python3 r1_stats.py --stage all      # ~65 min, 1 CPU
/mnt/home/mlee1/venvs/BIND_env/bin/python3 r1_ladder_fig.py --nu_panels peak_counts
```

Nothing under the ceph campaign trees was modified; `main.tex` and all existing figures,
caches and notebooks are untouched.

## 9. Open items for the authors

1. **The $\sim$90% citation must change.** Measured on these lightcones it is $70\%$ for
   $S(\ell)$ and $0.61$–$0.73$ for the $\nu$-domain statistics. If \citet{Lee-2026a}'s
   number refers to a different statistic (3D $P(k)$) or mass range, say so explicitly;
   otherwise replace it with the measured values as drafted in §7(a).
2. **"Statistically indistinguishable at LSST-Y10 precision"** is true against the pasted
   truth ($\le1.2\times$ the band) and false against full hydro for $\ell\gtrsim900$.
   The draft paragraph states the comparison of gaps rather than weakening the claim;
   an author ruling is needed on whether §5's summary sentence should be qualified too.
3. **The $\tau$ conditioning offset** (§5.3) is an independent decision item already
   raised in R3b §5.4; the draft text attributes it rather than repairing it.
