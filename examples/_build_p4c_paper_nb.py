"""Assembles examples/paper_p4c_tsz_ksz.ipynb — the P4c PAPER notebook.

Builder (not a science module), same emit-only harness as
`_build_ksz_lightcone_nb.py` / `_build_ksz_paper2_nb.py`; execution is a
separate nbconvert step. Unlike the campaign notebooks (which mirror plan
phases), this notebook is structured like the PAPER itself:

  Abstract
  1  Introduction                     (narrative + literature)
  2  Data & methods
     2.1  The BIND lightcone suite
     2.2  ACT DR6 y x DESI DR1 LRG y-CAP measurement    -> Fig 1
     2.3  External validation (DR5 clusters, Liu+2025)  -> Fig 2
     2.4  The kSZ leg (Ried Guachalla+2025) + R6 fix
     2.5  Corrections & error budget                    -> Fig 3
  3  Results
     3.1  kSZ selection of the feedback space           -> Fig 4
     3.2  tSZ confrontation: the tension                -> Fig 5
     3.3  Cross-probe coherence                         -> Fig 6
     3.4  Posterior inference (GP+MCMC)                 -> Figs 7, 8
  4  Discussion
  5  Caveats & robustness                               -> Fig 9
  6  Conclusions
  7  Future work
  References / Reproducibility appendix

Every figure is REGENERATED from the merged products under
KS/lightcone/{*.npz, verdicts/*.json} — nothing is re-measured, no raw
maps are loaded, and every quoted number is printed from its verdict JSON
(single source of truth). Total data read ~250 MB; runs in a few minutes
on the workstation.

Run once:
    python examples/_build_p4c_paper_nb.py && jupyter nbconvert --to notebook \
        --execute --inplace examples/paper_p4c_tsz_ksz.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python (BIND_env)", "language": "python", "name": "bind_env"}
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ===================================================================== title
md(r"""
# Group-scale gas thermodynamics from kSZ + tSZ stacking versus the CAMELS-SB35 feedback space

**A map-level confrontation of 253 IllustrisTNG feedback variants with DESI $\times$ ACT
kinematic and thermal Sunyaev–Zel'dovich measurements**

*Paper-structured notebook: every figure regenerates from the archived campaign
products (no re-measurement); every quoted number is printed live from its
verdict JSON. Campaign: `docs/tsz_des_data_plan.md` +
`docs/p4c_referee_hardening_plan.md`, git tag `p4c-hardened-v1`,
branch `analysis/ksz-desi-act-v2`.*

---

## Abstract

Baryonic feedback redistributes gas in and around dark-matter halos, and is now
the dominant systematic for small-scale weak-lensing cosmology. We confront a
253-node Sobol design over the 30-dimensional IllustrisTNG astrophysics
parameter space — realized as ray-traced 25 deg$^2$ lightcones painted by the
BIND generative emulator on a shared TNG300 dark-matter-only base — with two
independent gas observables at the DESI galaxy population: **(i)** the published
DESI DR1 spectroscopic BGS $\times$ ACT kSZ aperture-photometry gas-fraction
profile (Ried Guachalla et al. 2025), and **(ii)** our **own** Compton-$y$ CAP
measurement on the ACT DR6 component-separated $y$-map at 160,150 DESI DR1 SGC
spectroscopic LRGs ($0.4<z<0.6$; S/N per aperture 4–15, CIB-deprojected primary,
validated against ACT DR5 clusters and Liu et al. 2025). After a
referee-hardening program — jackknife+Hartlap statistics, a measured
resampling-operator correction, per-node HOD satellite forward-modeling
calibrated on the lensing mass anchor, a correlated CIB systematic covariance,
a $\pm0.1$ dex mass-anchor template, a truth-lightcone painting-fidelity
correction ($+16.4\%$, divided out), an analytic two-halo floor
($A_{2h}\simeq1$), and a covariance-unit fix to the published kSZ release —
we find: **no node of the design fits the tSZ data** (best
$\chi^2=15.9/6$; TNG fiducial $\chi^2/{\rm dof}=4.2$, $p\approx3\times10^{-4}$,
over-predicting small-aperture $y$ by $2.3$–$5.0\times$), while the kSZ and tSZ
$\chi^2$ rankings across the design are strongly coherent (Spearman
$\rho=0.82$): both probes independently prefer the **strong-feedback,
gas-poor edge** of the TNG model space. A GP-emulated 33-parameter MCMC over
the design maps the two probes into a joint constraint on the halo gas latent:
$\tilde f_{\rm gas}(<R_{500})=0.45^{+0.06}_{-0.06}$ and
$\tilde f_{\rm gas}(R_{500}\!\to\!R_{200})=0.95^{+0.05}_{-0.05}$ of the cosmic
baryon fraction — groups that have expelled half their inner gas. Only one of
30 feedback parameters (`WindFreeTravelDensFac`) survives a look-elsewhere
null. The result is the map-level, population-stacked counterpart of the
eROSITA group gas-fraction tension: the data demand feedback stronger than any
variant TNG's model family can produce at fixed cosmology.
""")

# ===================================================================== setup
md(r"""
## Setup — products, verdicts, style

All inputs live under the campaign products directory (`LC`). The verdict
JSONs are the single source of truth for quoted numbers; figures regenerate
from the merged npz products.
""")
code(r"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
LC = KS / "lightcone"
ZEN = KS / "desact_zenodo"          # Ried Guachalla+2025 Zenodo release
VD = LC / "verdicts"
REPO = Path("/mnt/home/mlee1/BIND-ksz2")

def verdict(name):
    with open(VD / f"{name}.json") as f:
        return json.load(f)

V = {k: verdict(k) for k in ["T1", "T3", "R5", "R6", "R7", "R4", "R8", "L", "X"]}

# analysis constants (mirroring examples/lightcone_m2r_ycap_real.py)
BIND_PIX_AREA = 0.29296875 ** 2     # BIND CAP is a pixel SUM; data is a mean x area
XB_ONEHALO_MAX = 1.4                # 1-halo fit range (M1 kSZ convention)
def consistent(chi2, dof):          # M1 consistency rule
    return chi2 < dof + 2.0 * np.sqrt(2.0 * dof)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11,
    "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})
C = dict(data="k", nodes="#b0c4d8", cons="#1f77b4", fid="#d62728",
         best="#2ca02c", band="#ff7f0e", null="#7f7f7f")
FIGDIR = LC / "figs" / "paper"
FIGDIR.mkdir(exist_ok=True)
print("products:", LC)
print("verdicts loaded:", ", ".join(V))
""")

# ===================================================================== s1
md(r"""
## 1  Introduction

Roughly half the baryons associated with galaxy groups are not where
gravity-only physics would put them. Feedback from supernovae and active
galactic nuclei ejects and redistributes the circumgalactic medium, and the
resulting suppression of the small-scale matter power spectrum is now the
limiting astrophysical systematic for cosmic-shear cosmology (Chisari et al.
2019; van Daalen, McCarthy & Schaye 2020; Amon & Efstathiou 2022). van Daalen
et al. (2020) showed the suppression is tightly predicted by a single halo
property — the baryon fraction of $\sim10^{13-14}\,M_\odot$ groups — which
elevates *direct measurements of group gas* to the status of a cosmological
calibration.

The Sunyaev–Zel'dovich effects (Sunyaev & Zel'dovich 1972) are the sharpest
population-level tools we have for that measurement. Stacked on large
spectroscopic samples with a compensated aperture-photometry (CAP) filter, the
**kinematic** SZ effect measures the electron *column* (gas mass) around
galaxies (Schaan et al. 2021), while the **thermal** SZ effect measures the
line-of-sight electron *pressure* (Amodeo et al. 2021). The ACT + DESI era has
made both routine: photometric-LRG kSZ stacks (Hadzhiyska et al. 2024),
spectroscopic DESI DR1 kSZ profiles (Ried Guachalla et al. 2025), and
Compton-$y$ CAP profiles at DESI LRGs (Liu et al. 2025). A consistent theme is
emerging: the gas is **more extended and less bound than fiducial
hydrodynamical simulations predict** — reported as kSZ profiles favoring
stronger feedback than TNG/FLAMINGO fiducials (Hadzhiyska et al. 2024;
McCarthy et al. 2025), as weak-lensing + kSZ joint fits pulling the baryon
correction beyond standard ranges (Bigwood et al. 2024), and as eROSITA
group X-ray gas fractions falling below every fiducial simulation
(e.g. Popesso et al. 2024).

What has been missing is a **many-model confrontation**: not "does TNG fit?"
but *which region of a full feedback parameter space* survives the SZ data,
asked at map level with the same filter, beam, aperture rule, and stacking
population as the measurements. The CAMELS program (Villaescusa-Navarro et al.
2021; Ni et al. 2023) demonstrated that feedback parameter spaces can be
explored with suites of small boxes; our BIND emulator extends this to the
observational regime by *painting* baryons — including the gas thermodynamic
fields needed for SZ work — onto a single TNG300-scale dark-matter-only
simulation, for arbitrary points in the 30-dimensional IllustrisTNG
astrophysics space (35-dim including cosmology; cosmology held fixed here).
A 253-node Sobol design over that space, ray-traced into 50 lightcone
realizations per node, is the model bank this paper confronts with the data.

Three design choices define the analysis. **First**, we measure the tSZ data
vector ourselves — a CAP Compton-$y$ profile at 160k DESI DR1 SGC
spectroscopic LRGs on the ACT DR6 component-separated $y$-map (Coulton et al.
2024) — because published photometric-sample profiles (Liu et al. 2025) probe
a different population than our mock catalogs, and because we require the
measurement and the model to share every convention (pixelization, aperture
rule, footprint gates) exactly (§2.2). **Second**, every analysis choice is
registered as pre- or post-hoc with its trigger (`docs/p4c_decisions_table.md`),
and the error budget was hardened adversarially before the headline was
frozen (§2.5, §5). **Third**, model-space conclusions are drawn from a
GP-emulated MCMC over the design with nuisance marginalization and a
look-elsewhere null — not from raw consistency counts, which we show are
treatment-dependent (§3.4, §5).

Section 2 describes the data, our measurement, and the error budget; §3 the
kSZ selection, the tSZ tension, their coherence, and the posterior; §4–5
discussion and caveats; §6–7 conclusions and future work.
""")

# ===================================================================== s2.1
md(r"""
## 2  Data & methods

### 2.1  The BIND lightcone suite (model bank)

BIND is a conditional flow-matching generative model that paints hydrodynamic
fields — including Compton-$y$ and the electron column $\tau$ — onto
dark-matter-only halo cutouts, conditioned on the 35-dimensional CAMELS
cosmological+astrophysical parameter vector (trained on CAMELS-SB35,
IllustrisTNG model family; Villaescusa-Navarro et al. 2021; Ni et al. 2023).
For this campaign the model painted the halo population
($M_{200}\ge10^{13}\,h^{-1}M_\odot$) of the TNG300-Dark box at **253 Sobol
nodes** of the 30-d astrophysics subspace (fixed fiducial cosmology), which
was then ray-traced into **50 lightcone realizations of 25 deg$^2$ each per
node** ($\tau$, $y$, $\kappa$ maps at BIND's native $0.29296875'$ pixel), with
a shared random-rotation sequence across nodes so that node-to-node
differences are purely astrophysical. The same trace on the *hydrodynamic*
TNG300 box provides a truth lightcone used for the painting-fidelity closure
(§2.5). Mock DESI samples (BGS-like cuts at $\log M_{200}\ge13.10/13.25$ for
the kSZ leg; an LRG-like cut matched to the Sailer et al. 2024 lensing mass
anchor $\log M_{200}=13.18$ for the tSZ leg) are stacked with the identical
CAP estimator as the data. Products used here are the merged per-node CAP
matrices — no raw maps are touched by this notebook.
""")

# ===================================================================== s2.2
md(r"""
### 2.2  Our ACT DR6 $y$-CAP measurement at DESI DR1 spectroscopic LRGs

We stack the ACT DR6 night-time component-separated Compton-$y$ map (Coulton
et al. 2024) at **160,150 DESI DR1 SGC spectroscopic LRGs** with
$0.4<z<0.6$ (WEIGHT-weighted; mean $z\simeq0.50$ matching the mock shell),
using the CAP filter of Schaan et al. (2021): disk minus equal-area ring to
$\sqrt2\,\theta_d$. Apertures follow the **anchor rule**
$\theta_d = x_b\,\theta_{200}(z;\log M_{200}=13.18)$ on an 18-point
$x_b\in[0.3,3.0]$ grid — the same dimensionless rule applied to the mocks —
plus a fixed-arcmin grid for display and external comparison. Two estimator
subtleties, both caught by closure tests, are folded in (§2.5): the discrete
CAP filter is **pixel-scale dependent** at $\theta\lesssim1.6'$, so the map is
cubic-resampled to BIND's exact $0.29296875'$ pixel and the measured
resampling-operator bias is divided out; and the **CIB-deprojected** ILC
variant (deproj-CIB $\beta=1.7$) is the primary map, because the baseline ILC
is dust-contaminated at the innermost apertures ($-4$ to $-7\sigma$
negative), consistent with Liu et al. (2025), whose fiducial is likewise
deprojected. Errors are a 100-cell RA/Dec jackknife with Hartlap debiasing
(Hartlap et al. 2007). Two nulls gate the measurement: 1.18M randoms in the
same footprint, and an RA-rotated LRG catalog.

**Fig. 1** shows the data vector, the CIB-variant band, and the nulls.
""")
code(r"""
# ---- Fig 1: the y-CAP measurement -----------------------------------------
a = np.load(LC / "act_ycap_lrg_real.npz")
xb, m17, e17 = a["xb"], a["mean_xb_cib17"], a["err_jk_xb_cib17"]
sig_cib, valid = a["sig_cib_xb"], a["valid_cols"].astype(bool)
t200 = float(a["theta200_data_arcmin"])

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0))
# (a) xb grid, the analysis vector
ax[0].axvspan(xb[valid].min(), xb[valid].max(), color="gold", alpha=0.12,
              label=f"$\\chi^2$ fit range ($x_b\\leq{XB_ONEHALO_MAX}$)")
ax[0].fill_between(xb, m17 - sig_cib, m17 + sig_cib, color=C["band"], alpha=0.25,
                   label="CIB-variant systematic band (11 ILC variants)")
ax[0].errorbar(xb, m17, e17, fmt="o", ms=4, color=C["data"], lw=1.2, capsize=2,
               label="deproj-CIB 1.7 (primary), resample-corrected")
ax[0].plot(xb, a["mean_xb_baseline"], "s", ms=3.5, mfc="none", color=C["fid"],
           label="baseline ILC (dust-biased, secondary)")
ax[0].axhline(0, color="0.6", lw=0.7)
ax[0].set_xlabel(r"$x_b=\theta_d/\theta_{200}$  ($\theta_{200}=%.2f'$ anchor)" % t200)
ax[0].set_ylabel(r"CAP $y$ [arcmin$^2$]")
ax[0].set_title("(a) ACT DR6 $y$-CAP at 160k DESI DR1 SGC LRGs")
ax[0].legend(loc="upper left", frameon=False)
# (b) fixed-arcmin grid + nulls
tr = a["theta_rap_arcmin"]
ax[1].errorbar(tr, a["mean_rap"], a["err_jk_rap"], fmt="o", ms=4, color=C["data"],
               lw=1.2, capsize=2, label="LRG stack")
ax[1].errorbar(tr, a["random_null"], a["random_null_err"], fmt="^", ms=4,
               color=C["null"], lw=1, capsize=2, label="random null (1.18M)")
ax[1].plot(tr, a["rotated_null"], "v", ms=4, mfc="none", color=C["cons"],
           label="RA-rotated null")
ax[1].axhline(0, color="0.6", lw=0.7)
ax[1].set_xlabel(r"$\theta_d$ [arcmin]")
ax[1].set_title("(b) fixed apertures + nulls")
ax[1].legend(frameon=False)
fig.savefig(FIGDIR / "fig1_ycap_measurement.png")
plt.show()
snr = np.abs(a["mean_rap"] / a["err_jk_rap"])
print(f"per-aperture S/N: {snr.min():.1f}-{snr.max():.1f};  n_gal = {int(a['n_gal']):,}")
print(f"nulls: random |max| = {np.abs(a['random_null']).max():.2e}, "
      f"rotated |max| = {np.abs(a['rotated_null']).max():.2e} "
      f"(vs signal {a['mean_rap'].max():.2e})")
""")

# ===================================================================== s2.3
md(r"""
### 2.3  External validation of the measurement pipeline

Two end-to-end checks anchor the pipeline against published results
(**Fig. 2**). **(a)** Stacking ACT DR5 SZ clusters (Hilton et al. 2021,
S/N$>5$) yields a $y$-CAP profile at S/N $\simeq26$ per aperture whose
amplitude scales monotonically with the catalog's own central Compton-$y$
terciles — the classic first-light test of a CAP pipeline (cf. Schaan et al.
2021). **(b)** Rerunning our engine on the exact Liu et al. (2025)
photometric-LRG samples (DESI DR9 imaging, their $p_z$ bins and mask)
reproduces their measurement — and resolves a release bug: their public csv
ships the **same column duplicated for all four $p_z$ bins**, and that column
is their $p_z4$ ($z\sim0.9$) bin ($\chi^2/9=1.7$ against our $p_z4$; $46$
against $p_z1$). Our four bins separate at 2–4$\sigma$ per point with the
correct redshift ordering. The published profile is thus *reproduced*, the
csv identified as an export artifact, and the comparison retired from any
quantitative role (decision #21).
""")
code(r"""
# ---- Fig 2: pipeline validation -------------------------------------------
d5 = np.load(LC / "T1_dr5_snr5.npz")
t1m = V["T1"]["metrics"]

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0))
th5 = d5["theta_value"].astype(float)
ax[0].errorbar(th5, d5["mean"], d5["err_jk"], fmt="o", ms=4, color=C["data"],
               capsize=2, label=f"ACT DR5 clusters (S/N>5, n={int(d5['n_gal'])})")
ax[0].axhline(0, color="0.6", lw=0.7)
amps = t1m["yc_tercile_amps_3.5arcmin"]
ax[0].set_title("(a) DR5 cluster stack — S/N$_{3.5'}$=%.0f, $y_c$-tercile amps %.1e/%.1e/%.1e"
                % (t1m["stack_snr_jk_at_3.5arcmin"], *amps), fontsize=9.5)
ax[0].set_xlabel(r"$\theta_d$ [arcmin]"); ax[0].set_ylabel(r"CAP $y$ [arcmin$^2$]")
ax[0].legend(frameon=False)

liu = np.load(REPO / "examples/figures_ksz2/tsz_liu2025_official.npz")
th_l = liu["theta"].astype(float)
for i, c in zip(range(1, 5), ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]):
    d = np.load(LC / f"R5_liu_pz{i}_cib1.7.npz")
    rap = np.asarray(d["theta_kind"]).astype(str) == "rap"
    ax[1].errorbar(d["theta_value"][rap].astype(float), d["mean"][rap],
                   d["err_jk"][rap], fmt="o-", ms=3, lw=1, color=c, capsize=2,
                   label=f"our $p_z${i}")
ax[1].plot(th_l, liu["pz1_fiducial"], "k--", lw=2,
           label="Liu+2025 csv (identical all bins)")
ax[1].set_title("(b) Liu+2025 reproduction — csv $\\equiv p_z4$ "
                "($\\chi^2/9=%.1f$ vs %.0f for $p_z1$)"
                % (1.7, 46), fontsize=9.5)
ax[1].set_xlabel(r"$\theta_d$ [arcmin]"); ax[1].legend(frameon=False, ncol=2)
fig.savefig(FIGDIR / "fig2_validation.png")
plt.show()
r5c = V["R5"]["metrics"]["r5c_liu_reproduction"]
print("R5c:", r5c["bins_resolved"])
""")

# ===================================================================== s2.4
md(r"""
### 2.4  The kSZ leg: DESI DR1 BGS $\times$ ACT (Ried Guachalla et al. 2025)

The kSZ side uses the published DESI DR1 spectroscopic BGS stacked kSZ
gas-fraction profiles $\tilde f_{\rm gas}(\theta)$ of Ried Guachalla et al.
(2025) (their Fig. 8 release; cuts $\log M_\star>11.0$, primary, and $>11.25$,
variant), compared with the same quantity built from the BIND lightcones:
per-node stacked $\tau$-CAP converted through the per-node CAP mass matrix
and the cosmic baryon fraction. BIND paints the electron column; no velocity
field enters on either side because the published profiles are already
velocity-normalized. Two corrections were required (phase R6,
`examples/_r6_ksz_audit.py`): **(i)** the release's `cov_ksz` is
**byte-identical to the raw $T^{\rm CAP}$ amplitude covariance** of their
Fig. 6 — not the covariance of the $\tilde f_{\rm gas}$ ratio the $\chi^2$
is built on; we rescale it onto the release's own $\tilde f_{\rm gas}$ errors
by a correlation-preserving congruence transform; **(ii)** Hartlap inflation
of the BIND-side realization covariance. The primary cut is stable under the
fix (27 vs 31 consistent nodes, rank correlation $\rho=0.97$); the
$\log M_\star>11.25$ variant is not (48$\to$5) and is quarantined to variant
status (decision #15).
""")

# ===================================================================== s2.5
md(r"""
### 2.5  Corrections and the error budget

Seven effects beyond the jackknife enter the tSZ comparison; each was
diagnosed by a dedicated closure test (phases R0–R7), applied in a fixed
order, and registered in `docs/p4c_decisions_table.md`:

| # | effect | treatment | size at fit apertures |
|---|--------|-----------|----------------------|
| R0 | jackknife bias | 100 cells + Hartlap $(n-1)/(n-p-2)$ per sample-estimated block | $\chi^2$ deflation $\times1.32$ undone |
| R1 | 0.5$'\to$0.293$'$ resampling operator | measured on BIND $y$ at matched halos; divided out; 50% kept as systematic | $-1.9$ to $-4.5\%$ at $\theta<1.6'$ |
| R2 | satellites + aperture rule | per-node HOD population stacks ($f_{\rm eff}=0.08\pm0.04$), $\kappa$-stack-calibrated to the lensing anchor | node-dependent, $\pm35$–$39\%$ spread |
| R3 | mass anchor $\pm0.1$ dex | fully-correlated template (both sides) | $\sim\pm17\%$ model |
| R5 | CIB residual | **correlated** $\Sigma_{\rm CIB}$ from 11 ILC deprojection variants | dominant off-diagonal term |
| R7 | painting fidelity | truth-lightcone closure at the same halos/apertures; divided out of the model | $+16.4\%\pm2\%$ |
| R4 | 2-halo / unpainted gas | analytic GNFW (Battaglia et al. 2012) template, $A_{2h}\sim\mathcal N(1,0.3)$, MCMC nuisance | 16–30% of data |

The jackknife covariance, $\Sigma_{\rm CIB}$, the satellite span, the mass
template, and the BIND realization covariance (Hartlap-corrected) sum to the
full comparison covariance. **Fig. 3** decomposes the diagonal at the six fit
columns ($x_b\le1.4$, the 1-halo range; beyond it the $\ge10^{13}$-only
painting lacks the 2-halo floor that R4 quantifies).
""")
code(r"""
# ---- Fig 3: error-budget decomposition at the fit columns ------------------
r5c_ = np.load(LC / "R5_cib_cov.npz")
r3 = np.load(LC / "R3_mass_template.npz")
r1 = np.load(LC / "R1_resample_correction.npz")
hd = np.load(LC / "R2_hod_model_curves.npz")
r7cols = V["R7"]["metrics"]["fit_columns_xb_0.46_1.25"]
xb_r7 = np.array([c["xb"] for c in r7cols])
bias_r7 = np.array([c["frac_bias_pct"] for c in r7cols]) / 100.0

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty-realization NaN slices

iv = np.nonzero(valid)[0]
dof = len(iv)
n_jk = 100
h_data = (n_jk - 1) / (n_jk - dof - 2)
sig_jk = np.sqrt(h_data) * e17
sig_cibcorr = np.sqrt(np.diag(r5c_["cov_cib"]))
b_res = np.interp(xb * t200, r1["theta_arcmin"], r1["bias_pixwin"])
sig_res = 0.5 * np.abs(b_res) * np.abs(m17)
# model-side terms (median over nodes), R7-corrected like the analysis
fcorr = 1.0 + np.interp(xb, xb_r7, bias_r7)
nodes = hd["nodes_kcal_f0.08"] * BIND_PIX_AREA / fcorr[None, None, :]
sat = 0.5 * np.abs(hd["nodes_kcal_f0.12"].mean(1) - hd["nodes_kcal_f0.04"].mean(1)) \
    * BIND_PIX_AREA / fcorr[None, :]
sig_sat = np.nanmedian(sat, axis=0)
node_mean = np.nanmean(nodes, axis=1)
sig_real = np.nanmedian(np.nanstd(nodes, axis=1), axis=0) / np.sqrt(nodes.shape[1])
sig_mass = np.nanmedian(np.abs(r3["dmodel_frac_per_sigma"][None, :] * node_mean
                               - r3["ddata_per_sigma"][None, :]), axis=0)

fig, ax = plt.subplots(figsize=(7.2, 4.2))
terms = [("jackknife (Hartlap)", sig_jk, C["data"]),
         (r"correlated $\Sigma_{\rm CIB}$ (diag)", sig_cibcorr, C["band"]),
         ("resampling (50% of corr.)", sig_res, "#9467bd"),
         (r"satellite $f_{\rm eff}$ span (median node)", sig_sat, C["cons"]),
         (r"mass anchor $\pm0.1$ dex (median node)", sig_mass, C["best"]),
         ("BIND realization (median node)", sig_real, C["null"])]
for name, s, c in terms:
    ax.plot(xb[iv], s[iv], "o-", ms=4, lw=1.4, color=c, label=name)
ax.plot(xb[iv], np.abs(m17[iv]), "k--", lw=2, label="|data|")
ax.set_yscale("log")
ax.set_xlabel(r"$x_b=\theta_d/\theta_{200}$")
ax.set_ylabel(r"$\sigma$ contribution [arcmin$^2$]")
ax.set_title("Fig 3 — error budget at the fit columns (diagonal view)")
ax.legend(frameon=False, fontsize=8.5, ncol=2)
fig.savefig(FIGDIR / "fig3_error_budget.png")
plt.show()
print("R7 fidelity correction over fit range: mean %+.1f%% (max |%.1f|%%)" % (
    V["R7"]["metrics"]["mean_frac_bias_pct_fit_range"],
    V["R7"]["metrics"]["max_abs_frac_bias_pct_fit_range"]))
print("R4 2-halo contamination (analytic): %.0f%% of data (mean over fit cols); "
      "analytic A2h prediction = %s (empirical fit unusable, see R4.json; "
      "A2h enters R8 as an N(1,0.3) nuisance)" % (
          V["R4"]["metrics"]["mean_contamination_pct_analytic"],
          V["R4"]["metrics"]["A2h_analytic_prediction"]))
""")

# ===================================================================== s3.1
md(r"""
## 3  Results

### 3.1  The kSZ data select a strong-feedback subspace

**Fig. 4** confronts the 253-node design with the primary kSZ profile
(BGS, $\log M_\star>11.0$; R6-corrected covariance). The design brackets the
data, and the consistency cut ($\chi^2 < {\rm dof} + 2\sqrt{2\,{\rm dof}}$)
retains **27/253 nodes**. The TNG fiducial sits gas-rich of the data — the
same direction Hadzhiyska et al. (2024), Ried Guachalla et al. (2025) and the
FLAMINGO comparison of McCarthy et al. (2025) report against their fiducial
simulations. The consistent set defines the "kSZ-selected subspace" used as a
conditioning set below.
""")
code(r"""
# ---- Fig 4: kSZ confrontation (bgs110, R6-corrected) -----------------------
tau = np.load(LC / "taucap_lightcone.npz", allow_pickle=True)
capm = np.load(LC / "capmat_lightcone.npz", allow_pickle=True)
k6 = np.load(LC / "ksz_consistent_nodes_r6.npz", allow_pickle=True)
d8 = np.load(ZEN / "Fig8_BGS_BRIGHT-20.2_logm11.00.npz")

N_XB = 18
F_B = float(capm["F_B"])
thr = V["R6"]["metrics"]["per_sample"]["bgs110"]["theta200_fid_arcmin"]
xbk = tau["theta_value"][:N_XB].astype(float)
with np.errstate(invalid="ignore", divide="ignore"):
    fgas_nodes = tau["sb35_mean_bgs110"][:, :N_XB] / capm["sb35_mean_bgs110"][:, :N_XB] / F_B
    fgas_fid = tau["fid_mean_bgs110"][:N_XB] / capm["fid_mean_bgs110"][:N_XB] / F_B
cons_ids = set(int(i) for i in k6["node_ids_bgs110"])
node_ids_all = k6["node_ids_all"]

fig, ax = plt.subplots(figsize=(7.4, 4.6))
for i, nid in enumerate(node_ids_all):
    is_c = int(nid) in cons_ids
    ax.plot(xbk * thr, fgas_nodes[i], color=C["cons"] if is_c else C["nodes"],
            lw=1.3 if is_c else 0.5, alpha=0.9 if is_c else 0.35,
            zorder=3 if is_c else 1)
ax.plot(xbk * thr, fgas_fid, color=C["fid"], lw=2.4, zorder=4, label="TNG fiducial")
ax.errorbar(d8["th"], d8["ratio"], d8["yerr"], fmt="o", ms=5, color=C["data"],
            lw=1.5, capsize=2.5, zorder=5,
            label=r"DESI DR1 BGS $\times$ ACT kSZ (Ried Guachalla+25)")
ax.axvline(XB_ONEHALO_MAX * thr, color="0.7", ls=":", lw=1)
ax.text(XB_ONEHALO_MAX * thr, ax.get_ylim()[0], " 1-halo fit range ", fontsize=8,
        color="0.4", ha="left", va="bottom")
ax.plot([], [], color=C["cons"], lw=1.3, label=f"consistent nodes ({len(cons_ids)}/253)")
ax.plot([], [], color=C["nodes"], lw=0.8, label="other SB35 nodes")
ax.set_xlabel(r"$\theta_d$ [arcmin]")
ax.set_ylabel(r"$\tilde f_{\rm gas}(<\theta_d)\,/\,f_{\rm b,cosmic}$")
ax.set_title("Fig 4 — kSZ gas-fraction profile vs the 253-node feedback design")
ax.legend(frameon=False, loc="upper left")
fig.savefig(FIGDIR / "fig4_ksz_selection.png")
plt.show()
r6b = V["R6"]["metrics"]["per_sample"]["bgs110"]
print(f"consistent: {r6b['n_new_consistent']}/253 (was {r6b['n_old_consistent']} pre-fix; "
      f"rank rho old-new = {np.atleast_1d(r6b['spearman_old_vs_new_chi2'])[0]:.2f})")
""")

# ===================================================================== s3.2
md(r"""
### 3.2  The tSZ data reject the entire design: a strong-feedback tension

**Fig. 5** is the central result. Against our $y$-CAP measurement with the
complete correlated budget (§2.5), **no node of the design is consistent**:
the best node reaches $\chi^2=15.9$ for 6 degrees of freedom and the TNG
fiducial is rejected at $\chi^2/{\rm dof}=4.2$ ($p\approx3\times10^{-4}$),
over-predicting the smallest fit apertures by $5.0\times$ (declining to
$2.3\times$ at $x_b\simeq1.25$). The failure mode is *shape-coherent*: the
data fall faster toward small apertures than even the strongest-feedback
node, and the freedoms in the budget (mass anchor, satellites, CIB,
$A_{2h}$) are amplitude-like and cannot reproduce it. This is the map-level,
population-stacked analogue of the eROSITA group gas-fraction deficit
(Popesso et al. 2024) — reached with a completely independent instrument,
estimator, and systematics budget.
""")
code(r"""
# ---- Fig 5: tSZ money plot -------------------------------------------------
t3m = V["T3"]["metrics"]
node_mean_y = node_mean            # from Fig 3 cell: R2 kcal f0.08, R7-corrected
fid_y = np.nanmean(hd["fid_kcal_f0.08"], axis=0) * BIND_PIX_AREA / fcorr
sig_disp = np.sqrt((h_data * e17**2) + np.diag(r5c_["cov_cib"]))  # display error

chi2_tsz_nodes = np.load(LC / "latent_constraints.npz")["chi2_tsz"]
best = int(np.nanargmin(chi2_tsz_nodes))

fig, ax = plt.subplots(figsize=(7.4, 4.8))
for i in range(node_mean_y.shape[0]):
    is_c = int(node_ids_all[i]) in cons_ids
    ax.plot(xb, node_mean_y[i], color=C["cons"] if is_c else C["nodes"],
            lw=1.2 if is_c else 0.5, alpha=0.9 if is_c else 0.3,
            zorder=3 if is_c else 1)
ax.plot(xb, fid_y, color=C["fid"], lw=2.4, zorder=4,
        label=r"TNG fiducial ($\chi^2/{\rm dof}=%.1f$)" % t3m["chi2_fid_over_dof"])
ax.plot(xb, node_mean_y[best], color=C["best"], lw=2.2, zorder=4,
        label=r"best node ($\chi^2=%.1f/%d$)" % (t3m["chi2_nodes_min_med"][0], t3m["dof"]))
ax.errorbar(xb[iv], m17[iv], sig_disp[iv], fmt="o", ms=5.5, color=C["data"], lw=1.5,
            capsize=2.5, zorder=5, label="ACT DR6 $y$-CAP (fit cols, full budget)")
ax.errorbar(xb[~valid], m17[~valid], sig_disp[~valid], fmt="o", ms=4, mfc="none",
            color="0.5", lw=1, capsize=2, zorder=5, label="outside fit range")
ax.plot([], [], color=C["cons"], lw=1.2, label="kSZ-consistent nodes")
ax.axvline(XB_ONEHALO_MAX, color="0.7", ls=":", lw=1)
ax.set_yscale("symlog", linthresh=2e-8)
ax.set_xlabel(r"$x_b=\theta_d/\theta_{200}$")
ax.set_ylabel(r"CAP $y$ [arcmin$^2$]")
ax.set_title("Fig 5 — tSZ confrontation: the full design over-predicts the data")
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.savefig(FIGDIR / "fig5_tsz_tension.png")
plt.show()
print(f"consistent with tSZ under full budget: {t3m['counts']['n_tsz_consistent']}/253")
print("fiducial/data ratio over fit cols:",
      np.round(t3m["fid_over_data_ratio_valid"], 2))
""")

# ===================================================================== s3.3
md(r"""
### 3.3  The two probes agree on the ranking

A rejected model bank can still be *informative* if the probes order it the
same way. **Fig. 6**: the per-node kSZ and tSZ $\chi^2$ are strongly rank-
correlated (Spearman $\rho=0.82$, $p\sim10^{-63}$) — gas-poor nodes do better
on **both** probes, across a design in which 30 parameters vary
simultaneously. This coherence (not any consistency count) is the robust
qualitative claim: two SZ observables with disjoint systematics budgets
independently order the feedback space by the same physical property — how
much gas the feedback has removed from the group cores.
""")
code(r"""
# ---- Fig 6: ranking coherence ---------------------------------------------
lat = np.load(LC / "latent_constraints.npz", allow_pickle=True)
c_k, c_t = lat["chi2_ksz"], lat["chi2_tsz"]
from scipy.stats import spearmanr
ok_ = np.isfinite(c_k) & np.isfinite(c_t)
rho, pv = spearmanr(c_k[ok_], c_t[ok_])

fig, ax = plt.subplots(figsize=(5.8, 4.8))
is_cons = np.array([int(n) in cons_ids for n in lat["node_ids"]])
ax.scatter(c_k[~is_cons], c_t[~is_cons], s=14, color=C["nodes"], label="SB35 nodes")
ax.scatter(c_k[is_cons], c_t[is_cons], s=26, color=C["cons"], zorder=3,
           label="kSZ-consistent (27)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\chi^2_{\rm kSZ}$ (dof=%d)" %
              V["R6"]["metrics"]["per_sample"]["bgs110"]["n_dof"])
ax.set_ylabel(r"$\chi^2_{\rm tSZ}$ (dof=%d)" % t3m["dof"])
ax.set_title(r"Fig 6 — cross-probe coherence: $\rho_{\rm Spearman}=%.2f$" % rho)
ax.legend(frameon=False)
fig.savefig(FIGDIR / "fig6_coherence.png")
plt.show()
print(f"Spearman rho = {rho:.3f} (p = {pv:.1e}); "
      f"T3 verdict value: {t3m['counts']['spearman_ksz110_vs_tsz_chi2'][0]:.3f}")
""")

# ===================================================================== s3.4
md(r"""
### 3.4  Posterior inference: the gas plane and the parameter forest

Consistency counts depend on the error treatment (§5), so quantitative
statements come from a likelihood analysis (phase R8,
`examples/_r8_gp_mcmc.py`): per-column GP emulators over the design
(3-fold-validated, coverage-calibrated) feed a 33-dimensional `emcee` MCMC —
30 astro parameters + satellite fraction $f_{\rm sat}$ + mass-anchor offset
$\Delta\log M$ + two-halo amplitude $A_{2h}$ (Foreman-Mackey et al. 2013) —
with all chains run 740–1200 autocorrelation times (gate: 50).

**Fig. 7** projects the kSZ, tSZ, and joint posteriors into the physically
interpretable **gas plane** of the parent analysis — the 2-D latent that
captures 97% of the design's observable response, rotated so its axes track
$\tilde f_{\rm gas}(<R_{500})$ and the $R_{500}\!\to\!R_{200}$ shell gas
(both $r\simeq0.95$; the low dimensionality of feedback response echoes
Lin et al. 2026). The two probes **independently** land on the same
low-inner-gas region, and jointly:

$$\tilde f_{\rm gas}(<R_{500}) = 0.45^{+0.06}_{-0.06}, \qquad
  \tilde f_{\rm gas}(R_{500}\!\to\!R_{200}) = 0.95^{+0.05}_{-0.05}$$

of the cosmic baryon fraction — inner-halo gas at half cosmic, outskirts
near-cosmic, i.e. feedback that *redistributes* gas outward rather than
merely heating it in place. **Fig. 8** shows the 30-parameter forest: the
constrained directions are the wind-energy sector
(`WindEnergyIn1e51erg`, `VariableWindVelFactor` pulled low; `IMFslope`
pulled flat; `WindFreeTravelDensFac` high), but under an ESS-matched
Dirichlet look-elsewhere null **only `WindFreeTravelDensFac`** remains
individually significant — an honest statement of how much a 253-node
design can localize 30 parameters.
""")
code(r"""
# ---- Fig 7: gas-plane posterior corner ------------------------------------
from scipy.stats import gaussian_kde
p8 = np.load(LC / "r8_posterior.npz", allow_pickle=True)
rng = np.random.default_rng(11)

def kde2(fx, fy, nmax=30000, grid=130):
    n = len(fx)
    if n > nmax:
        s = rng.choice(n, nmax, replace=False)
        fx, fy = fx[s], fy[s]
    kde = gaussian_kde(np.vstack([fx, fy]))
    gx = np.linspace(fx.min(), fx.max(), grid)
    gy = np.linspace(fy.min(), fy.max(), grid)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    zs = np.sort(Z.ravel())[::-1]
    cz = np.cumsum(zs); cz /= cz[-1]
    levels = [zs[min(np.searchsorted(cz, q), len(zs) - 1)] for q in (0.95, 0.68)]
    return GX, GY, Z, levels

fig, ax = plt.subplots(figsize=(6.4, 5.6))
for leg, color, lw in [("ksz", C["cons"], 1.6), ("tsz", C["fid"], 1.6),
                       ("joint", "k", 2.4)]:
    GX, GY, Z, lv = kde2(p8[f"gas_fin_{leg}"], p8[f"gas_fout_{leg}"])
    ax.contour(GX, GY, Z, levels=lv, colors=[color], linewidths=[0.8 * lw, lw],
               linestyles=["--", "-"])
    ax.plot([], [], color=color, lw=lw, label={"ksz": "kSZ", "tsz": "tSZ",
                                               "joint": "joint"}[leg])
w = lat["w_joint"]; wf = w / w.sum()
ax.scatter(lat["f_in"], lat["f_out"], s=4 + 600 * wf, color="#8c564b", alpha=0.35,
           label="design nodes (area $\\propto$ Phase-L weight)")
ax.axvline(1.0, color="0.8", lw=0.7); ax.axhline(1.0, color="0.8", lw=0.7)
ax.set_xlabel(r"$\tilde f_{\rm gas}(<R_{500})\,/\,f_{\rm b,cosmic}$")
ax.set_ylabel(r"$\tilde f_{\rm gas}(R_{500}\to R_{200})\,/\,f_{\rm b,cosmic}$")
ax.set_title("Fig 7 — gas-plane posterior (68/95%)")
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.savefig(FIGDIR / "fig7_gas_plane.png")
plt.show()
gp = V["R8"]["metrics"]["gas_plane"]["joint"]
print("joint f_in 16/50/84:", np.round(gp["f_in_165084"], 3),
      " f_out:", np.round(gp["f_out_165084"], 3))
""")
code(r"""
# ---- Fig 8: 30-parameter forest -------------------------------------------
names = [str(n) for n in p8["names"]][:30]
prior_q = p8["prior_q"]
le = V["R8"]["metrics"]["look_elsewhere_null"]["per_param"]
beats = set(V["R8"]["metrics"]["look_elsewhere_null"]["beating_params"])

qs = {leg: np.percentile(p8[f"chain_{leg}"][:, :30], [16, 50, 84], axis=0)
      for leg in ("ksz", "tsz", "joint")}
wr = (qs["joint"][2] - qs["joint"][0]) / (prior_q[:, 2] - prior_q[:, 0])
order = np.argsort(wr)[::-1]

fig, ax = plt.subplots(figsize=(7.6, 10.2))
for row, j in enumerate(order):
    y = len(order) - 1 - row
    ax.barh(y, prior_q[j, 2] - prior_q[j, 0], left=prior_q[j, 0], height=0.62,
            color="0.92", zorder=1)
    for leg, color, dy in [("ksz", C["cons"], 0.16), ("tsz", C["fid"], 0.0),
                           ("joint", "k", -0.16)]:
        q = qs[leg][:, j]
        ax.plot([q[0], q[2]], [y + dy] * 2, color=color, lw=2 if leg == "joint" else 1.4,
                zorder=3)
        ax.plot(q[1], y + dy, "o", ms=3.5, color=color, zorder=4)
    star = names[j] in beats
    lbl = ("$\\bf{%s}$" % names[j].replace("_", r"\_") + " *") if star else names[j]
    ax.text(-0.02, y, lbl + f"  [wr {wr[j]:.2f}]", ha="right", va="center",
            fontsize=8, color=C["best"] if star else "k",
            transform=ax.get_yaxis_transform())
ax.axvline(0.5, color="0.75", lw=0.8)
ax.set_yticks([])
ax.set_xlim(0, 1)
ax.set_xlabel("prior-normalized parameter (16–50–84%); grey = prior 68%")
ax.set_title("Fig 8 — parameter forest (kSZ blue / tSZ red / joint black)\n"
             "* = beats the ESS-matched look-elsewhere null")
fig.savefig(FIGDIR / "fig8_forest.png")
plt.show()
print("beats null:", sorted(beats), " (of 30)")
print("A_2h posterior (joint):",
      np.round(V["R8"]["metrics"]["r4_two_halo_a2h_nuisance"]
               .get("a2h_posterior_165084", {}).get("joint", []), 3)
      if isinstance(V["R8"]["metrics"].get("r4_two_halo_a2h_nuisance"), dict) else "see R8.json")
""")

# ===================================================================== s4
md(r"""
## 4  Discussion

**The tension is physical, not instrumental.** The measurement pipeline
reproduces ACT DR5 clusters and the Liu et al. (2025) profiles (§2.3); the
random and rotated nulls are clean in the fit range; and the painting
fidelity of the emulator is *measured* against a hydrodynamic truth lightcone
at the exact stacking halos and apertures — BIND over-paints CAP-$y$ by
$+16.4\%\pm2\%$, which is divided out and explains only $\sim$1/6 of the
$2.3$–$5\times$ deficit. The unpainted two-halo term goes the *wrong way*:
adding the analytic floor ($A_{2h}\simeq1$, confirmed by its posterior)
*raises* the model further above the data. The freedoms that could mimic the
deficit — mass anchor, satellite fraction, CIB — are marginalized with
correlated templates and cannot reproduce its aperture dependence.

**Convergence with independent probes.** Three independent lines now point
at gas-poorer groups than fiducial simulations: X-ray gas fractions (eROSITA;
Popesso et al. 2024), kSZ profiles (Hadzhiyska et al. 2024; Ried Guachalla et
al. 2025; McCarthy et al. 2025), and weak-lensing + kSZ joint analyses
(Bigwood et al. 2024). Our contribution is the *model-space* statement: even
allowing all 30 IllustrisTNG feedback parameters to vary jointly across their
CAMELS-SB35 ranges, the thermal SZ data at DESI LRGs remain below the entire
family — the deficit is not a fiducial-tuning accident but a structural
property of the TNG feedback model at group scales. Within the design, the
data pull toward high `WindFreeTravelDensFac`, low wind energies, and flat
IMF slope — i.e. configurations that move gas out of $R_{500}$ efficiently
without over-heating what remains (Fig. 7's near-cosmic outskirt shell).

**What the gas plane means for lensing.** Through the van Daalen et al.
(2020) relation, $\tilde f_{\rm gas}(<R_{500})\simeq0.45$ at
$10^{13.2}\,M_\odot$ implies matter-power suppression at the strong end of
current cosmic-shear priors — consistent with the direction advocated by
Amon & Efstathiou (2022) and Bigwood et al. (2024) for resolving small-scale
lensing tensions. The 2-D gas plane (inner deficit + outskirt shell) is the
natural interface to pass to lensing analyses, echoing the low-dimensional
latent structure of feedback response found in Lin et al. (2026).
""")

# ===================================================================== s5
md(r"""
## 5  Caveats & robustness

**Consistency counts are treatment-dependent — the headline is not.**
Across defensible error treatments the number of "tSZ-consistent" nodes
swings between 0 and 119 (**Fig. 9**); with the complete correlated budget it
is 0. We therefore *never* lead with counts: the invariants are the deficit
direction/amplitude and the cross-probe ranking coherence
($\rho=0.82$–$0.90$ across all treatments). All 21 analysis decisions with
their pre/post-hoc status and triggers are in `docs/p4c_decisions_table.md`;
the headline moved *against* the initially-favorable framing (consistency
$\to$ tension) as the budget was completed, which is the opposite of
confirmation bias.

Other caveats, each quantified in its verdict:

- **Model family**: the design spans IllustrisTNG's 30 astro parameters at
  fixed cosmology (TNG300). The conclusion "no node fits" is a statement
  about this family; SIMBA/Astrid-family sweeps could differ (CAMELS shows
  their feedback modes are qualitatively distinct; Ni et al. 2023).
- **Painting floor**: only halos $\ge10^{13}\,h^{-1}M_\odot$ are painted;
  the missing diffuse/2-halo signal is handled by the R4 analytic template
  and the $x_b\le1.4$ fit range, not by the emulator itself.
- **Shared DMO base**: all nodes share one TNG300-Dark realization; sample
  variance is common-mode across nodes (helps ranking, limits absolute
  covariance realism). BIND realization scatter is propagated per node
  (Hartlap-corrected), but a single N-body box underlies everything.
- **kSZ external covariance**: the release's own sample-covariance estimator
  count is unpublished, so its Hartlap factor cannot be applied (flagged in
  R6.json); the primary cut is the one stable under our unit fix.
- **GP emulation**: likelihoods are GP-emulated over 253 nodes with
  coverage-calibrated uncertainties (temperature 1.2–1.5$\times$); posterior
  tails beyond the design hull are prior-dominated by construction.
- **Look-elsewhere honesty**: the earlier importance-sampling analysis
  (Phase L, ESS$\sim$6) produced a "stable constrained pair" that did NOT
  survive the Dirichlet null — retained here as a methodological warning
  that robustness checks are not null tests.
""")
code(r"""
# ---- Fig 9: counts by treatment (the honesty figure) -----------------------
cbt = V["R5"]["metrics"]["counts_by_treatment"]
t3c = V["T3"]["metrics"]["counts"]
labels = ["diagonal half-band CIB\n(pre-R5)", "correlated $\\Sigma_{CIB}$\n(final budget)",
          "dust-template variant"]
keys = ["diag_halfband", "correlated", "dust_template"]
ns = [cbt[k]["n"] for k in keys]
rhos = [cbt[k].get("P_given_ksz", np.nan) for k in keys]

fig, ax = plt.subplots(figsize=(6.8, 3.8))
bars = ax.bar(range(len(keys)), ns, color=[C["nodes"], C["cons"], C["null"]], width=0.6)
for i, (n, r) in enumerate(zip(ns, rhos)):
    ax.text(i, n + 2, f"n={n}\nP(tSZ|kSZ)={r:.2f}", ha="center", fontsize=9)
ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("tSZ-consistent nodes (of 253)")
ax.set_ylim(0, 140)
ax.set_title("Fig 9 — consistency counts are treatment-dependent;\n"
             "the deficit direction and $\\rho$(kSZ,tSZ) are not")
fig.savefig(FIGDIR / "fig9_treatment.png")
plt.show()
print("chi2_med by treatment:",
      {k: round(cbt[k]["chi2_med"], 1) for k in keys})
print(f"threshold scan (final budget): {t3c['threshold_scan']}")
""")

# ===================================================================== s6
md(r"""
## 6  Conclusions

1. **We measured** a new ACT DR6 Compton-$y$ CAP profile at 160,150 DESI DR1
   SGC spectroscopic LRGs (S/N 4–15 per aperture), validated against ACT DR5
   clusters and Liu et al. (2025) — identifying, along the way, release bugs
   in two external data products (the Liu et al. csv $p_z$-bin duplication;
   the Ried Guachalla et al. `cov_ksz` unit mix-up), both reported with
   byte-level evidence.
2. **No node of the 253-point IllustrisTNG feedback design fits the tSZ
   data** under the complete correlated error budget (best $\chi^2=15.9/6$;
   fiducial $p\approx3\times10^{-4}$, $2.3$–$5\times$ over-prediction) —
   while the kSZ and tSZ rankings agree at $\rho=0.82$: both probes demand
   the strong-feedback, gas-poor edge and then some.
3. **The joint posterior localizes the gas latent**, not the parameters:
   $\tilde f_{\rm gas}(<R_{500})=0.45\pm0.06$,
   $\tilde f_{\rm gas}(R_{500}\!\to\!R_{200})=0.95\pm0.05$ of cosmic —
   inner-gas-poor, outskirt-near-cosmic groups. Only one of 30 parameters
   (`WindFreeTravelDensFac`) survives a look-elsewhere null.
4. **Methodologically**, map-level emulation makes many-model SZ
   confrontation tractable (253 full forward models), but the error budget —
   satellites, CIB correlations, painting fidelity, external covariance
   audits — is where the conclusion is actually decided; consistency counts
   without that budget are close to meaningless (0 vs 119 on the same data).
""")

# ===================================================================== s7
md(r"""
## 7  Future work

- **Velocity-aware kSZ.** BIND currently paints the electron column; pairing
  it with a DMO-derived velocity surrogate would build $\Delta T_{\rm kSZ}$
  directly and unlock pairwise/velocity-weighted estimators.
- **Cross-family designs.** Repeating the confrontation on SIMBA- and
  Astrid-family CAMELS sweeps tests whether *any* current feedback model
  family reaches the data, or whether the deficit indicts subgrid physics
  more broadly.
- **Cosmology dependence.** The design holds cosmology fixed; a joint
  cosmo+astro design would let the SZ gas plane be marginalized directly in
  cosmic-shear analyses (the natural continuation of Amon & Efstathiou 2022;
  Bigwood et al. 2024).
- **WL integration.** The $\kappa\times y$ cross-correlation was the most
  feedback-sensitive WL statistic in our earlier suite work; folding it in
  (with the DES demotion lessons of decision #20) would close the loop
  between the gas plane and the matter power spectrum.
- **Deeper SZ data.** Advanced ACTPol/SO depth improves the small-aperture
  columns where the tension is largest; the aperture-resolved deficit shape
  is a falsifiable prediction for Simons Observatory stacking.
""")

# ===================================================================== refs
md(r"""
## References

*(Author-year citations used in the narrative. arXiv identifiers included
where certain; verify all entries against ADS before submission.)*

- Amodeo S., et al., 2021, PRD 103, 063514 (ACT DR5 thermodynamic profiles)
- Amon A., Efstathiou G., 2022, MNRAS 516, 5355
- Battaglia N., Bond J.R., Pfrommer C., Sievers J.L., 2012, ApJ 758, 75
- Bigwood L., et al., 2024, MNRAS (WL + kSZ baryon feedback; arXiv:2404.06098)
- Chisari N.E., et al., 2019, Open J. Astrophys. 2, 4
- Coulton W., et al., 2024, PRD 109, 063530 (ACT DR6 component-separated maps)
- Foreman-Mackey D., Hogg D.W., Lang D., Goodman J., 2013, PASP 125, 306
- Hadzhiyska B., et al., 2024 (DESI photometric LRG kSZ; arXiv:2407.07152)
- Hartlap J., Simon P., Schneider P., 2007, A&A 464, 399
- Hilton M., et al., 2021, ApJS 253, 3 (ACT DR5 cluster catalog)
- Lin et al., 2026 (feedback response latent; arXiv:2509.01881)
- Liu et al., 2025 (ACT DR6 y-CAP at DESI photometric LRGs)
- McCarthy I.G., et al., 2025 (FLAMINGO vs stacked kSZ)
- Ni Y., et al., 2023, ApJ (CAMELS extended parameter sweeps)
- Popesso P., et al., 2024 (eROSITA group gas fractions)
- Ried Guachalla B., et al., 2025 (DESI DR1 spectroscopic kSZ profiles)
- Sailer N., et al., 2024 (DESI LRG lensing mass calibration)
- Schaan E., et al., 2021, PRD 103, 063513 (ACT DR5 kSZ CAP)
- Sunyaev R.A., Zel'dovich Ya.B., 1972, Comm. Astrophys. Space Phys. 4, 173
- van Daalen M.P., McCarthy I.G., Schaye J., 2020, MNRAS 491, 2424
- Villaescusa-Navarro F., et al., 2021, ApJ 915, 71 (CAMELS)

## Reproducibility appendix

| item | location |
|------|----------|
| campaign plan / hardening plan | `docs/tsz_des_data_plan.md`, `docs/p4c_referee_hardening_plan.md` |
| decision registry (21 choices, pre/post-hoc) | `docs/p4c_decisions_table.md` |
| measurement engine | `examples/act_ycap_measure.py` |
| tSZ comparison chain | `examples/lightcone_m2r_ycap_real.py` |
| kSZ audit / covariance fix | `examples/_r6_ksz_audit.py` |
| satellite HOD stacks | `examples/lightcone_hod_stack.py` |
| fidelity closure / 2-halo | `examples/_r7_fidelity_closure.py`, `examples/_r4_twohalo.py` |
| GP+MCMC | `examples/_r8_gp_mcmc.py` (chains: `r8_state/`, posterior: `r8_posterior.npz`) |
| verdicts (all gates) | `KS/lightcone/verdicts/*.json` |
| git tag | `p4c-hardened-v1` on `analysis/ksz-desi-act-v2` |

This notebook: built by `examples/_build_p4c_paper_nb.py`; paper figures are
written to `KS/lightcone/figs/paper/`.
""")

nb["cells"] = cells
out = "examples/paper_p4c_tsz_ksz.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
