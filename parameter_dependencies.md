# Quantified Physical Trends Across the IllustrisTNG 35-Parameter Space: A Reference for BIND Validation
## Overview
BIND operates on halo patches above \(10^{13}\ M_\odot\), generating 2D pixelized fields for stellar, gas, and dark matter mass as a function of 35 cosmological and astrophysical parameters drawn from the IllustrisTNG model. This report catalogs the *quantified, source-cited* physical trends that govern each major parameter group — the trends the network must learn, and that physical validation tests should recover. Parameters are drawn from the CAMELS SB35 suite, which extends the standard 4-parameter CAMELS set to a full 30 astrophysical parameters plus 5 cosmological ones, spanning the complete IllustrisTNG subgrid model.[1][2]

***
## 1. Cosmological Parameters (\(\Omega_m\), \(\sigma_8\), \(\Omega_b\), \(H_0\), \(n_s\))
### Physical meanings and expected trends
The CAMELS SB35/SB28 sets vary five cosmological parameters:[3]
- \(\Omega_m\) (0.1–0.5, fiducial 0.3), \(\sigma_8\) (0.6–1.0, fiducial 0.8), \(\Omega_b\) (0.029–0.069, fiducial 0.049), \(H_0\) (47–87 km/s/Mpc, fiducial 67.11), \(n_s\) (0.76–1.16, fiducial 0.9624)

**\(\Omega_m\) and \(\sigma_8\)** primarily affect structure formation. For fixed \(M_{200c}\), higher \(\sigma_8\) shifts the same halo mass to a less rare peak, meaning earlier formation times and systematically higher DM concentrations. For group-mass halos (\(10^{13}–10^{14.5}\ M_\odot\)), both \(\Omega_m\) and \(H_0\) produce the largest spread in stacked gas density, temperature, and Compton-\(y\) profiles, while \(\sigma_8\) and \(n_s\) show a more subtle effect. In the CAMELS-zoomGZ inference analysis, neural networks infer \(\Omega_m\) with correlation coefficient \(r \geq 0.99\) from stacked group/cluster profiles, compared to \(r \approx 0.97\) for \(\sigma_8\).[4]

**\(\Omega_b\)** directly sets the cosmic baryon budget available to each halo. Observations constrain the hot gas fraction at \(r_{500c}\) in \(10^{14}\ M_\odot\) clusters to \(f_{\rm gas} \approx 0.12–0.14\), and IllustrisTNG reproduces this at fiducial parameters. Increasing \(\Omega_b\) raises the halo baryon fraction floor, and its signal is well-encoded in the gas density profile.[5][4]

**What BIND's DM patches should show:** higher \(\sigma_8\) → more concentrated DM patches (inner \(\Sigma_{\rm DM}(<0.3R_{200c})/\Sigma_{\rm DM}(<R_{200c})\) increases). Higher \(\Omega_m\) → denser environment, larger apparent halo mass at fixed patch size. These cosmological imprints survive in the DM field even after the gas and stellar channels have added baryonic variance.

***
## 2. Stellar Feedback Parameters
### 2.1 \(A_{\rm SN1}\) — Galactic Wind Energy per Unit SFR (0.25–4, fiducial 1)
In the IllustrisTNG model, \(A_{\rm SN1}\) is a multiplicative prefactor to the available wind energy per unit star-forming mass, \(\bar{e}_w\), encoded in Equation 13 of Pillepich et al. (2018). The wind energy per gas cell is:[6]
\[e_w = A_{\rm SN1} \times \omega \times N_{\rm SNII} E_{\rm SNII,51}\ M_\odot^{-1}
\]
with a fiducial total wind energy of 3.6 \(\times 10^{51}\) erg, varied from 0.9 to 14.4 \(\times 10^{51}\) erg across the SB28 range.[7]

**Causal chain at \(M_{200c} > 10^{13}\ M_\odot\):**

The dominant and most non-linear effect of \(A_{\rm SN1}\) operates *indirectly* through black hole growth. Stronger stellar winds suppress star formation and ISM gas density, which reduces the Bondi accretion rate onto the central SMBH. This delays the SMBH reaching the \(\sim 10^8\ M_\odot\) threshold for kinetic-mode AGN feedback to higher halo masses. Consequently:[8][6]

- **High \(A_{\rm SN1}\):** Stellar mass decreases (direct wind ejection of star-forming gas); BH growth is suppressed; kinetic AGN mode is delayed; the CGM gas fraction \(f_{\rm CGM}/f_{b,\rm cosmic}\) is *higher* than fiducial at \(M_{200c} \gtrsim 10^{11.3}\ M_\odot\) (fewer AGN outflows); the closure radius (radius enclosing the cosmic baryon fraction) decreases.[9][6]
- **Low \(A_{\rm SN1}\):** Higher stellar mass; faster BH growth → earlier, stronger kinetic AGN feedback → lower \(f_{\rm CGM}\) and larger closure radius; total cumulative feedback energy (normalized by halo binding energy) can be *higher* at group scales because the dominant AGN mode kicks in earlier.[6]

This **counter-intuitive reversal** — increasing stellar feedback *reduces* total baryon ejection from group-scale halos — is one of the most important and observationally consequential results from the CAMELS literature. It means the network, to be physically correct, must learn that the effect of \(A_{\rm SN1}\) on the gas distribution in \(> 10^{13}\ M_\odot\) halos is *opposite* to a naive expectation.[10][9]

**Quantitative observational anchor:** IllustrisTNG at fiducial parameters predicts stellar masses within 30 kpc scaling as \(M_\star \propto (M_{500c})^{0.49}\) with \(\sim 0.12\) dex scatter across \(10^{13}–10^{15}\ M_\odot\). The color–mass bimodality transitions at \(M_\star \sim 10^{10.5}\ M_\odot\), setting the quenching mass scale that \(A_{\rm SN1}\) shifts. In groups (\(10^{13}\ M_\odot\)), the quenched fraction of centrals reaches 70–90%.[11][12][13]
### 2.2 \(A_{\rm SN2}\) — Galactic Wind Speed (0.5–2, fiducial 1)
\(A_{\rm SN2}\) is a prefactor to the wind velocity factor \(\kappa_w = 7.4\), with a velocity floor at \(v_{w,\min} = 350\) km/s:[6]
\[v_w = A_{\rm SN2} \times \max\!\left(\kappa_w \sigma_{\rm DM} \left(\frac{H_0}{H(z)}\right)^{1/3},\ v_{w,\min}\right)
\]

The wind speed determines whether wind particles escape the halo or recycle in the ISM/CGM. Higher wind speeds increase the kinetic energy carried per unit wind mass. The mass loading factor \(\eta_w \propto e_w/v_w^2\), so increasing \(A_{\rm SN2}\) at fixed \(A_{\rm SN1}\) *decreases* the mass loading for fixed energy, reducing the total gas mass ejected per unit SFR while increasing the velocity at which it is ejected.[14][6]

At group scales, the effect on halo baryon fractions is more moderate than \(A_{\rm SN1}\). However, \(A_{\rm SN2}\) is among the *best-constrained* parameters from cluster profiles in the CAMELS-zoomGZ inference analysis — neural network predictions for \(A_{\rm SN2}\) approach \(r \sim 1\). This is because the wind speed directly sets the spatial scale at which winds recouple to the CGM, imprinting a scale-dependent feature in the gas density profile. High \(A_{\rm SN2}\) → winds escape to larger radii → gas density profile is shallower in the inner CGM, steeper beyond the coupling radius.[4]

**The WindFreeTravelDensFac (W3)** parameter, which sets the density threshold below which wind particles recouple, governs the *location* of the transition between decoupled and recoupled wind gas and has a strong effect on gas density profiles in the CGM at \(\sim 0.1–0.3\ R_{200c}\).[4]

**The minimum wind velocity (W4, \(v_{w,\min}\) varied from 150 to 550 km/s)** sets a hard floor. In massive halos (\(M_{200c} > 10^{13}\ M_\odot\)), \(\sigma_{\rm DM}\) is already high enough that the minimum is not binding, so W4 has a weaker effect in BIND's mass range.
### 2.3 Equation of State: \(f_{\rm EQS}\) (FactorForSofterEQS, 0.1–0.9, fiducial 0.3)
This parameter adjusts the effective equation of state (eEOS) for star-forming gas in the Springel–Hernquist (2003) multiphase ISM model. A lower \(f_{\rm EQS}\) makes the ISM "softer" — gas in star-forming regions has lower effective pressure support, making it easier for both SN winds and AGN to displace it. A higher \(f_{\rm EQS}\) stiffens the ISM, suppressing wind-driven outflows. This parameter primarily affects the *central* stellar and gas morphology and the spatial compactness of the star-forming region, which then propagates to the BH accretion rate.[4]
### 2.4 IMF Slope — Initial Mass Function (−2.8 to −1.8, fiducial −2.3)
The IMF slope sets the relative abundance of high-mass stars and thus the total SN rate and total metal yield per unit stellar mass. A top-heavier IMF (less negative slope) increases both the SN energy budget and the metal enrichment rate. In the CAMELS-zoomGZ analysis, the IMF is among the best-constrained astrophysical parameters, with its signal most strongly encoded in the gas metallicity profile (which BIND does not directly produce, but which leaves imprints on the gas mass distribution through metallicity-dependent wind energy).[4]
### 2.5 \({\rm SNII\_MinMass}\ M_\odot\) (4–12 \(M_\odot\), fiducial 8)
This sets the minimum stellar mass for SNII explosions. Lowering this threshold increases the number of stars that end as SNII, boosting total SN feedback energy. At group scales, a lower SNII minimum mass delays AGN onset via the same \(A_{\rm SN1}\) causal chain (more SN feedback → slower BH growth), and it also increases ISM metal enrichment.[7]
### 2.6 ThermalWindFraction (W1, 0.025–0.4, fiducial 0.1)
The fraction of wind energy injected thermally versus kinetically. At fiducial values, 10% is thermal and 90% kinetic. A higher thermal fraction slows wind propagation (thermal energy dissipates without directed momentum) and reduces the effective mass loading of kinetic outflows. This affects the *spatial distribution* of CGM gas more than the total mass budget.
### 2.7 Wind Metallicity Coupling (W5–W7)
Parameters W5 (WindEnergyReductionFactor), W6 (WindEnergyReductionMetallicity), and W7 (WindEnergyReductionExponent) together set how much the wind energy is reduced in high-metallicity gas. At fiducial values, wind energy in high-metallicity gas is reduced by a factor \(f_{w,Z} = 0.25\). These parameters primarily affect star-forming satellite galaxies that enrich quickly, and at the group-halo mass scale their effect is secondary to \(A_{\rm SN1}\) and \(A_{\rm SN2}\).[7]

***
## 3. AGN Feedback Parameters
### 3.1 \(A_{\rm AGN1}\) — Kinetic Mode Energy per BH Accretion Rate (0.25–4, fiducial 1)
\(A_{\rm AGN1}\) is a prefactor to the kinetic feedback energy injection rate in the low-accretion (radio/kinetic) mode, as described in Weinberger et al. (2017):[6]
\[\dot{E}_{\rm kin} = A_{\rm AGN1} \times \epsilon_{f,\rm kin}\ \dot{M}_{\rm BH}\ c^2
\]
where the fiducial efficiency \(\epsilon_{f,\rm kin}\) is itself a function of BH mass. At each timestep, kinetic energy is accumulated until it exceeds a threshold proportional to the gas kinetic energy in the surrounding bubble; this threshold is scaled by \(A_{\rm AGN2}\).[15]

**Crucially**, because the BH is self-regulating, increasing \(A_{\rm AGN1}\) does not simply increase the total feedback energy. Instead, stronger kinetic feedback more efficiently quenches BH accretion, so the BH grows more slowly and injects energy less frequently. The net effect at group scales (\(10^{13}\ M_\odot\)) is:[10][6]

- Higher \(A_{\rm AGN1}\) → somewhat lower \(f_{\rm CGM}\) (stronger individual AGN events eject gas); the closure radius increases modestly; BH mass at fixed halo mass decreases due to self-regulation.
- The effect on *total* cumulative feedback energy is approximately flat across \(A_{\rm AGN1}\) variations because of self-regulation.

Observationally, the kinetic mode in TNG is responsible for quenching star formation at \(M_\star \gtrsim 10^{10.5}\ M_\odot\), and this mode activates once the SMBH exceeds \(\sim 10^8\ M_\odot\) and its Eddington ratio drops below the critical threshold \(\chi_{\rm crit}\). The BH kinetic winds simultaneously eject central gas (ejective) and increase CGM entropy from 10–100 Myr to 1–10 Gyr cooling times (preventative).[16][17]
### 3.2 \(A_{\rm AGN2}\) — Kinetic Mode Burstiness/Ejection Speed (0.5–2, fiducial 1)
\(A_{\rm AGN2}\) is a prefactor to the reorientation factor \(f_{\rm re} = 20\) that controls the energy accumulation threshold before kinetic mode feedback fires. When \(A_{\rm AGN2}\) doubles, so does the energy threshold, resulting in less frequent but more energetic AGN events. This is formally "burstiness":[15][14]

- **High \(A_{\rm AGN2}\):** Less frequent, more impulsive kinetic kicks → deeper gas evacuation per event → lower central gas density at \(r < 0.3 R_{200c}\); slightly larger closure radius at \(M_{200c} > 10^{12}\ M_\odot\).[6]
- **Low \(A_{\rm AGN2}\):** More frequent, weaker kicks → gas heated more continuously but less strongly displaced → higher central gas density; lower effective thermalization.

The key physical prediction here is that \(A_{\rm AGN2}\) imprints a **morphological** signature — the spatial profile of the central gas depression — rather than only an amplitude effect. The kinetic mode creates 3–15 kpc wide holes in gas discs, a feature detectable as a deficit in the central gas patch even in projection.[18]
### 3.3 Seed Black Hole Mass (BH1, \(2.5 \times 10^{-5}–2.5 \times 10^{-4} \times 10^{10}\ M_\odot h^{-1}\), fiducial \(8 \times 10^{-5}\))
This sets the initial mass of newly seeded SMBHs. Higher seed masses accelerate the path to the kinetic mode threshold (\(M_{\rm BH} \sim 10^8\ M_\odot\)), so halos that start with higher-mass seeds activate kinetic feedback earlier in their history, reach lower stellar masses at \(z = 0\), and have lower-density central gas patches.[7][4]
### 3.4 Black Hole Accretion Factor (BH2, Bondi rate multiplier, 0.25–4, fiducial 1)
A higher Bondi rate multiplier accelerates BH growth by directly boosting the Bondi accretion rate. At \(M_{200c} > 10^{13}\ M_\odot\), the central BH is already in the self-regulating kinetic mode, so the effect of BH2 on current accretion is limited — but its effect on *BH mass at fixed halo mass* (via the growth history) is strong. Higher BH2 → higher \(M_{\rm BH}\) at fixed \(M_{200c}\) → more total cumulative AGN energy injected → lower current stellar mass and lower current central gas density.[7]
### 3.5 Quasar Threshold (Q1) and Quasar Threshold Power (Q2)
Q1 is the Eddington ratio \(\chi_{\rm crit}\) at which the BH transitions from thermal (quasar) to kinetic (radio/kinetic) mode. The fiducial value is \(\chi_{\rm crit} = 0.002\), varied from \(6.3 \times 10^{-5}\) to \(0.063\). A higher \(\chi_{\rm crit}\) means the kinetic mode activates at higher Eddington ratios, which at fixed BH mass corresponds to earlier activation.[7]

Q2 sets the power-law steepness of the mass-dependent transition in Weinberger et al. (2017) Eq. 5. Together, Q1 and Q2 control *when* in the BH growth history the most impactful kinetic feedback begins, making them shift the stellar mass function and quenched fractions in a way that is qualitatively similar to \(A_{\rm AGN1}\) but via a different mechanism. At group scales, lowering Q1 (earlier kinetic mode) directly reduces the quenched fraction of satellite galaxies.[17][2]
### 3.6 Black Hole Feedback Factor (BH4) and Radiative Efficiency (BH5)
BH4 is the high-accretion (thermal) mode feedback efficiency, with the fiducial value set to \(\epsilon_{f,\rm therm} = 0.1\) (i.e., 10% of accreted energy injected thermally). BH5 is the radiative efficiency \(\epsilon_r = 0.2\), which determines how much rest-mass energy is converted to radiation. Higher BH4 or BH5 means more total energy injected per unit accreted mass in the early, high-accretion-rate phase. For group-mass halos at \(z=0\), the BH is predominantly in kinetic mode, so BH4/BH5 affect the *historical* growth path more than the current state.[19]

***
## 4. Star Formation ISM Parameters
### 4.1 MaxSfrTimescale (\(t_{\rm SFR}\), 1.135–4.54 Gyr, fiducial 2.27 Gyr)
This is the gas consumption timescale in the multiphase ISM model. A longer \(t_{\rm SFR}\) reduces the instantaneous SFR at fixed gas density, which reduces wind energy injection per unit time. This shifts the integrated stellar mass lower and delays BH growth. Its effect is similar in direction to high \(A_{\rm SN1}\) but operates through SFR suppression rather than wind energy scaling.
### 4.2 VariableWindSpecMomentum (W2, 0–4000 \({\rm km/s}\), fiducial 0)
A non-zero value adds a fixed isotropic momentum kick to all wind particles independent of local ISM conditions. This parameter allows for a scale-independent component to wind propagation, which is most important for low-mass galaxies at sub-group scales. At the \(>10^{13}\ M_\odot\) mass scale BIND targets, this parameter has minimal effect on the central halo mass distribution.
### 4.3 WindDumpFactor (W8, 0.2–1.0, fiducial 0.6)
Controls the fraction of metals *not* ejected into winds but instead deposited into nearby star-forming cells. A higher W8 means more metal retention in the ISM, increasing metallicity-driven wind energy reduction (via W5–W7), and enriching the CGM less. This parameter primarily encodes the local metal cycling and is most directly visible in the metallicity distribution of the stellar patches.

***
## 5. The ASN1–AAGN Coupling: The Most Important Physical Non-linearity
The strongest non-linear physics BIND must learn is the **anti-correlated feedback coupling** between stellar and AGN channels. As documented in Medlock et al. (2024), Delgado et al. (2023), and confirmed by the X-ray CAMELS emulator analysis:[9][20][6]

> "Stronger stellar feedback often results in *weaker* effects [on the matter power spectrum and baryon fraction] by suppressing black hole growth and therefore the impact of AGN feedback."

This means that in BIND's mass range (\(M_{200c} > 10^{13}\ M_\odot\)):

| Scenario | \(A_{\rm SN1}\) | BH growth | AGN kinetic mode | \(f_{\rm gas}(<R_{200c})\) | Gas distribution |
|---|---|---|---|---|---|
| SN-dominated | High (×4) | Suppressed, delayed | Weak or absent | Higher (less AGN ejection) | Extended, smooth, lower central density |
| AGN-dominated | Low (×0.25) | Enhanced, early | Strong | Lower (strong AGN ejection) | Centrally evacuated, extended tails |
| Dual feedback | Fiducial | Moderate | Moderate | Intermediate | Intermediate |

This is observable in the 2D patches: the spatial morphology of the gas channel distinguishes SN-dominated (shallow, extended suppression) from AGN-dominated (sharp central cavity, more concentrated outer gas) regimes. At fiducial parameters, TNG predicts black hole feedback dominates the overall mass flow throughout the halo while stellar feedback mainly affects the inner region within \(\sim 0.2\ R_{\rm vir}\).[21]

***
## 6. Dark Matter Back-Reaction
The DM channel in BIND should reflect the back-reaction of baryons on the dark matter profile. IllustrisTNG shows a transition from adiabatic contraction (inner DM density enhancement from cooling-dominated halos) to core formation and expansion (AGN-dominated feedback ejecting central baryons and reducing the DM potential):[22]

- Low-mass ETGs (\(M_\star \lesssim 10^{11}\ M_\odot\)): DM halos are *contracted* relative to DMO — steeper-than-NFW inner profiles
- High-mass ETGs/groups (\(M_\star \gtrsim 10^{11.5}\ M_\odot\)): DM halos are *expanded* — kinetic AGN kicks remove central baryons, reducing the gravitational potential, and the DM responds by expanding[23][22]

The total power-law slope of the density profile for ETGs in IllustrisTNG has a mean of \(\langle \gamma' \rangle = 2.011 \pm 0.007\) with scatter \(\sigma_{\gamma'} = 0.171\), and "black hole kinetic winds are crucial to lowering \(\gamma'\) and matching observed galaxy correlations". The DM halo shape (axis ratios) responds similarly: in the FP run, inner density slopes anti-correlate with halo mass, whereas in DMO they remain constant. The dominant axis ratio effect in BIND's mass range is AGN feedback (via \(A_{\rm AGN1}\), Q1), not stellar feedback.[22]

For the power spectrum: baryons suppress the total matter power spectrum by up to 20% at \(k \sim 10\ h\ {\rm Mpc}^{-1}\), with group-scale halos (\(\log M_{200m} \in [13,14]\)) contributing the largest fraction of this suppression at \(k \sim 2–30\ h\ {\rm Mpc}^{-1}\).[24][25]

***
## 7. Cosmological Parameter Effects on Halo Structure
At fixed \(M_{200c}\), the cosmological parameters produce structural changes that are distinct from — but degenerate with — baryonic effects in 2D projections:

| Parameter | Effect on DM patch | Effect on gas patch | Observable proxy |
|---|---|---|---|
| \(\sigma_8\) ↑ | Higher concentration, more compact DM core | Higher gas density (earlier formation, more cooling) | \(c_{\rm 2D}({\rm DM})\) |
| \(\Omega_m\) ↑ | Denser large-scale environment, more substructure | Higher gas fraction from richer accretion history | Subhalo abundance in patch |
| \(\Omega_b\) ↑ | Slightly more back-reaction | Higher absolute gas mass at fixed DM | \(f_b(< R_{200c})\) |
| \(H_0\) ↑ | Faster expansion → later structure formation | Lower halo masses at fixed angular scale | Profile shape vs. \(R_{200c}\) |
| \(n_s\) ↑ | More small-scale power → more concentration | More satellite substructure | Patch texture |

In the CAMELS-zoomGZ inference study, \(\Omega_m\) and \(H_0\) produce the most pronounced profile variation, while \(\sigma_8\) and \(n_s\) show subtler effects on thermodynamic profiles. The gas density profile is the most sensitive to \(\sigma_8\) among all profile types.[4]

***
## 8. Summary: Parameter Hierarchy for Group-Scale Halos
Based on CAMELS 1P scans and the CAMELS-zoomGZ emulator results, the parameters can be ranked by their impact on observables in the \(M_{200c} \in [10^{13}, 10^{14.5}]\ M_\odot\) range:[10][9][2][4]

**Dominant (largest effect on halo mass distributions):**
1. \(A_{\rm SN1}\) — strongest modulator of BH growth history and total baryon fraction via the SN–AGN coupling
2. \(\Omega_m\) — sets the matter density normalization
3. \(H_0\) / \(\Omega_b\) — overall normalization of gas content and expansion rate

**Strong (significant but secondary):**
4. \(A_{\rm AGN1}\) — directly modulates kinetic ejection efficiency
5. \(A_{\rm SN2}\) — wind speed governs CGM recoupling scale (best-constrained parameter in zoomGZ inference)
6. \(\sigma_8\) — concentration and formation history
7. BH2 (Bondi rate) / BH1 (seed mass) — shift the BH mass at fixed halo mass
8. Q1 (Quasar threshold) / IMF slope — shift quenched fractions and enrichment

**Moderate (detectable with high-quality profiles):**
9. \(A_{\rm AGN2}\) — burstiness of kinetic events, central gas morphology
10. \(f_{\rm EQS}\) — ISM pressure support, wind propagation
11. W3 (WindFreeTravelDensFac) — gas coupling radius in CGM
12. SNII\_MinMass — effective SN energy budget
13. \(n_s\) — small-scale power, substructure

**Weak at \(> 10^{13}\ M_\odot\) (stronger at lower masses):**
14–28+. W1 (ThermalWindFraction), W2 (VariableWindSpecMomentum), W4 (MinWindVel), W5–W7 (metallicity wind coupling), W8 (WindDumpFactor), BH3 (Eddington factor), BH4–BH5 (thermal efficiency), Q2 (quasar threshold power)

This hierarchy directly informs which BIND parameter response curves should be validated quantitatively versus qualitatively, and which parameters represent genuine physical tests versus second-order effects at the mass scale of interest.