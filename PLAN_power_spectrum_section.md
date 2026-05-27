# Power Spectrum Results for `draft.tex`

## Figures to Include

Three figures from `full_map_power_spectrum.ipynb` belong in the paper, in the order listed below.

---

### Figure A — `paper_figures/fig5b_total_field_pk.png` ⭐ Primary

**Two-panel figure: CV suite, full box (50 Mpc/h, 1024×1024)**

- **Panel (a):** Mean ratio $P(k)/P_{\rm truth}(k)$ for BIND-composited (red), hydro-replace (green), and N-body DMO (grey), averaged over 27 CV sims.  A vertical dotted line and golden band mark $k_{R_{200c}}(M_{\rm thr} = 10^{13}\,M_\odot\,h^{-1}) \approx 18\,h\,\mathrm{Mpc}^{-1}$.
- **Panel (b):** Cross-correlation of the *baryonic correction* — $r(\Delta_X, \Delta_{\rm truth})$ where $\Delta_X \equiv X - \Sigma_{\rm DMO}$ — for BIND (red) and hydro-replace (green), with ±1σ bands.

**Suggested placement:** New subsection in §4 Validation, after the shape/morphology subsection:

> **§4.X Full-Box Power Spectrum**

**Suggested caption:**

> **(a)** Mean power spectrum ratio relative to the true hydrodynamical field (truth), averaged over 27 CV simulations.  BIND (red) and the hydro-replace ceiling (green, true hydro patches pasted at the same halo positions) are nearly coincident at all $k$, with both falling below unity above the characteristic wavenumber of the mass-threshold halos (gold dashed line, $k_{R_{200c}} \approx 18\,h\,\mathrm{Mpc}^{-1}$).  The N-body DMO baseline (grey) shows the deficit in the absence of any baryonification.
> **(b)** Mode-alignment of the baryonic correction field $\Delta_X = X - \Sigma_{\rm DMO}$.  The cross-correlation coefficient $r(\Delta_{\rm BIND},\Delta_{\rm truth})$ tracks the hydro-replace ceiling at $> 0.9$ for all $k$ below the mass threshold, confirming that BIND-generated halos sit on the same Fourier modes as the true baryonic imprint.

---

### Figure B — `paper_figures/fig7b_1p_response_pk_summary.png` ⭐ Primary

**Twin-axis bar chart: 1P suite parameter response on the full box**

- **Bars (left axis):** Pearson correlation of $\mathcal{R}_{\rm BIND}(k)$ and $\mathcal{R}_{\rm hydro\_replace}(k)$ with $\mathcal{R}_{\rm truth}(k)$, where $\mathcal{R}_X \equiv P_X/P_X^{\rm fid} - 1$, pooled over all non-fiducial 1P levels and $k \le k_{R_{200c}}(M_{\rm thr})$.  Parameters ordered by decreasing truth-response RMS.
- **Diamonds (right axis, blue):** Truth-response RMS $\sqrt{\langle \mathcal{R}_{\rm truth}^2 \rangle}$ per parameter.  Provides an amplitude context for the Pearson values — parameters with near-zero RMS produce meaningless correlations.

**Suggested placement:** Same subsection as Figure A.

**Suggested caption:**

> For each of the 35 SB35 parameters, bars show the Pearson correlation between the BIND-generated power spectrum response $\mathcal{R}_{\rm BIND}$ (red) and the hydro-replace ceiling (green) versus the true response $\mathcal{R}_{\rm truth}$, evaluated at $k \le k_{R_{200c}}(M_{\rm thr})$.  Blue diamonds give the truth-response RMS amplitude per parameter; correlations for parameters with RMS $\lesssim 0.01$ (right side of the panel) are dominated by noise.  For all parameters with an appreciable truth response ($A_{\rm SN1}$, $A_{\rm AGN1}$, $\Omega_m$, $\sigma_8$, $A_{\rm SN2}$, $A_{\rm AGN2}$, $\Omega_b$, and several tertiary parameters), BIND achieves Pearson $\rho \gtrsim 0.9$, indistinguishable from the hydro-replace ceiling.

---

### Figure C — `paper_figures/fig5_power_spectra.png` ⬜ Supporting / Supplementary

**Nine-panel figure: per-channel P(k) for CV full-box, SB35 full-box, and CV halo patches**

Compares BIND (orange) vs truth (black) for the three matter channels (DM_hydro, Gas, Stars) across two full-box test suites and the per-halo-patch level.  Sub-panels show the ratio BIND/truth.

**Suggested placement:** Appendix or supplementary material.  The information it conveys — that the per-channel field-level P(k) is reproduced at the halo-patch level and that the full-box deficit is in the ratio — is already captured more cleanly and with proper context in Figure A.  However, this figure shows per-channel accuracy (especially for gas and stars) which is not visible in the total-mass version.

---

## Procedure (for Methods / Validation text)

The following should appear in the §4.X subsection before the figures:

1. **Map construction.** For each simulation, four full-box ($L = 50\,h^{-1}\,\mathrm{Mpc}$, $1024 \times 1024$ pixels) projected total surface-density maps are formed by summing the three matter channels (DM, gas, stars):
   - **truth** — the full hydrodynamical projection.
   - **BIND** — the N-body DMO field with BIND-generated halo patches alpha-blended in at every halo above $M_{200c} \ge 10^{13}\,M_\odot\,h^{-1}$.
   - **hydro-replace** — the same composite recipe but using the *true* hydro halo patches; this is the methodological ceiling for any halo-paste scheme at this mass threshold.
   - **DMO** — the raw N-body field, the baseline.

2. **Power spectrum computation.** The 2D projected power spectrum $P(k)$ is computed for each map using the Pylians `Pk_plane` routine, with results cached per simulation.

3. **CV analysis.** Over 27 cosmic-variance simulations at the fiducial parameter point, the mean ratio $P(k)/P_{\rm truth}(k)$ and its standard deviation are computed for BIND, hydro-replace, and DMO.  The cross-correlation coefficient of the *baryonic correction field* $\Delta_X = X - \Sigma_{\rm DMO}$ is computed from cross-spectra $P_{\Delta_X \Delta_{\rm truth}}(k)$, directly isolating mode alignment of the painted mass without the large DMO-background correlation inflating the metric.

4. **1P parameter-response analysis.** For each of the 35 SB35 parameters the fractional power spectrum response $\mathcal{R}_X(k \mid p, \ell) = P_X(k \mid p, \ell) / P_X(k \mid \mathrm{fid}) - 1$ is computed.  Pearson correlations between $\mathcal{R}_{\rm BIND}$ and $\mathcal{R}_{\rm truth}$, pooled over all non-fiducial levels and $k \le k_{R_{200c}}(M_{\rm thr})$, quantify whether BIND preserves the parameter-induced modulation of large-scale structure.

5. **Mass-threshold diagnostic.** The characteristic wavenumber of the mass-threshold halos,
   $$k_{R_{200c}}(M_{\rm thr}) = \frac{2\pi}{R_{200c}(M_{\rm thr})}, \qquad R_{200c} = \left(\frac{3M_{\rm thr}}{800\pi\rho_{\rm crit}}\right)^{1/3}$$
   with $\rho_{\rm crit} = 2.775 \times 10^{11}\,M_\odot\,h^{-1}(\mathrm{Mpc}/h)^{-3}$, gives $k_{R_{200c}} \approx 18\,h\,\mathrm{Mpc}^{-1}$ for $M_{\rm thr} = 10^{13}\,M_\odot\,h^{-1}$.  Below this scale, sub-threshold halos that remain DMO contribute an unavoidable baryonification deficit in any halo-paste scheme at this mass floor.

---

## Importance of the Results

These results close the loop from halo-level to cosmological-volume validation:

1. **BIND matches the hydro-replace ceiling.** The ratio $P_{\rm BIND}/P_{\rm hydro\_replace} = 1 \pm O(1\%)$ across all $k$.  Any deficit in $P_{\rm BIND}/P_{\rm truth}$ that is also present in $P_{\rm hydro\_replace}/P_{\rm truth}$ is *not* a BIND error — it is the known limitation of any halo-paste approach that omits sub-threshold halos.  This is the key statement that separates the model's accuracy from the inherent constraint of the $M_{200c} \ge 10^{13}\,M_\odot\,h^{-1}$ mass floor.

2. **The deficit is localized at $k \gtrsim k_{R_{200c}}(M_{\rm thr})$.** This confirms the physical interpretation: sub-threshold halos have $R_{200c} < 0.35\,h^{-1}\,\mathrm{Mpc}$, so their absent baryonic signatures appear at $k \gtrsim 18\,h\,\mathrm{Mpc}^{-1}$.  Lowering the mass floor or combining with an isotropic transfer-function approach (e.g., Sharma et al. 2024) could close this gap; BIND naturally provides the former.

3. **BIND-painted halos sit on the same Fourier modes as the truth.**  The baryonic-correction cross-correlation $r(\Delta_{\rm BIND}, \Delta_{\rm truth}) > 0.9$ at $k < k_{R_{200c}}$ (Figure A, panel b), matching the hydro-replace ceiling.  This rules out the possibility that BIND has correct power statistics but wrong spatial structure — the painted mass lives in the right physical locations.

4. **Parameter sensitivity is faithfully reproduced.**  For the seven parameters with the largest power spectrum response ($A_{\rm SN1}$, $A_{\rm AGN1}$, $\Omega_m$, $\sigma_8$, $A_{\rm SN2}$, $A_{\rm AGN2}$, $\Omega_b$), BIND achieves Pearson $\rho \gtrsim 0.9$ with truth, matching the hydro-replace ceiling (Figure B).  This means BIND can be used as a forward model in parameter-inference pipelines that employ large-scale structure statistics derived from the total projected mass field, not just halo-level observables.

5. **Practical implication: forward-modeling for Stage IV surveys.**  By running BIND patch-by-patch over all halos above the mass floor in an N-body simulation, a single baryonified cosmological volume can be generated in the time needed to run the neural network inference — orders of magnitude faster than a full hydrodynamical simulation.  The fidelity established here (% level in $P(k)$ at $k < 18\,h\,\mathrm{Mpc}^{-1}$, correct parameter response) is sufficient for the power spectrum and cross-correlation analyses planned for LSST and *Euclid*.

---

## Suggested Draft Section Text (sketch)

```latex
%-------------------------------------------------------------
\subsection{Full-Box Power Spectrum}
\label{subsec:power_spectrum}
%-------------------------------------------------------------

\begin{figure*}[t]
    \centering
    \includegraphics[width=\linewidth]{paper_figures/fig5b_total_field_pk}
    \caption{...}
    \label{fig:full_box_pk}
\end{figure*}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{paper_figures/fig7b_1p_response_pk_summary}
    \caption{...}
    \label{fig:1p_pk_response}
\end{figure}

The preceding tests validate \textsc{BIND} at the level of individual halo patches.
We now ask whether the model reproduces the projected matter power spectrum when 
applied to baryonify an entire N-body box at $z=0$.

For each simulation in the CV and 1P suites we construct four full-box 
($L=50\,h^{-1}\,\mathrm{Mpc}$, $1024\times1024$) total projected surface-density 
maps: (i) the true hydrodynamical field (\textit{truth}), (ii) the BIND-composited 
field (\textit{bind}), in which \textsc{BIND}-generated halo patches are 
alpha-blended into the N-body canvas at every halo above 
$M_{200c} \ge 10^{13}\,M_\odot\,h^{-1}$, (iii) the \textit{hydro-replace} 
map, constructed identically to \textit{bind} but using the true hydro patches ---
the methodological ceiling for any halo-paste scheme at this mass threshold --- 
and (iv) the unmodified N-body DMO field.
2D power spectra are computed with Pylians...

Figure~\ref{fig:full_box_pk}a shows the mean ratio $P(k)/P_{\rm truth}(k)$ 
over the 27 CV simulations.  The BIND and hydro-replace curves are nearly 
coincident at all scales, confirming that \textsc{BIND} achieves the full 
accuracy permitted by the $M_{200c} \ge 10^{13}\,M_\odot\,h^{-1}$ mass floor...

Figure~\ref{fig:full_box_pk}b shows the cross-correlation coefficient of the 
baryonic correction field $\Delta_X = X - \Sigma_{\rm DMO}$...

Figure~\ref{fig:1p_pk_response} summarizes the 1P parameter response...
```
