"""Build papers/01_pipeline/family_model_tutorial.ipynb — the user-facing
tutorial for the shipped family-basis model (FamilyModel, lambda route), plus
the paper validation figure pfig_family_model_curves (the pfig_s4b_model_curves
layout re-made with the family basis + lambda amplitudes).

2026-08-11 revision: (i) all four latents get a sweep panel (not just f_bar);
(ii) the van Daalen connection is PROVEN, not asserted — the model's f_bar
sweep is overlaid on the suite's measured S(ell_vD)–f_bar relation, with the
partial (others-fixed) vs conditional (along-the-design) distinction drawn;
(iii) PDF panels are log-y.

2026-08-12 EXTENDED-LAMBDA revision: promoted the bake-off-winning 8-latent
map (shipped 4 + group/cluster-bin log-pressure + group/cluster-bin
f_gas,200c) that beats the 4-latent map on all 7 statistics (clk e2e CV
0.96 vs 0.92, fiducial trough prediction 0.908 vs 0.920 vs measured 0.881).
measure_latents/LAT/lam_fid now carry 8 numbers; the sweep figure splits
into two 4-row panels (shipped-4, then extended-4) for readability.

2026-08-12 estimator revision: the per-run amplitude fit switched from
ridge (clk only, $\kappa=520$) to plain OLS everywhere. The two are
end-to-end CV equivalent (clk 0.9585 ridge vs 0.9593 OLS; fiducial trough
prediction 0.9112 ridge vs 0.908 OLS, both against 0.881 measured) — OLS
amplitudes just carry a large, curve-invisible component along clk's
near-null basis direction, an interpretability cost only. See
`FAMILY_BASIS_METHODS.md`.

2026-08-12 OBSERVABLE AGNOSTIC-LAMBDA revision (author-adopted): the
latent set is now the 8 chosen by the fully agnostic, observables-only
SFFS search (agnostic_lambda_search.py, OBS_ONLY=1; see
FAMILY_BASIS_METHODS.md): f_star[13.2-13.4], logT[13.0-13.2],
logY[13.0-13.2], logPe[14.0-14.3], c_gas[14.0-14.3], logY_ss[13.4-13.6],
c_gas[13.2-13.4], logPe[13.4-13.6]. Every latent is observationally
accessible; f_bar is no longer a latent (selected first by the search,
then evicted as redundant with the same bin's Y+T), so the van-Daalen
section now measures f_bar as an AUXILIARY variable and traces the
relation via the design-conditional latent path — a stronger demo: the
model contains the vD relation in a variable it never sees. Fiducial
trough closure 0.895 (vs 0.908 two-bin-8, 0.920 four-latent; measured
0.881, 0.7 sigma of sig_pred 0.019).

Rebuild the notebook:
    /mnt/home/mlee1/venvs/BIND_env/bin/python _build_family_tutorial_nb.py
Execute it:
    /mnt/home/mlee1/venvs/BIND_env/bin/python _run_family_tutorial_nb.py
"""
import nbformat as nbf

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip()))


def code(src: str) -> None:
    CELLS.append(("code", src.strip()))


# ═════════════════════════════════════════════════════════════════════════════
md(r'''
# The family-basis model: a tutorial

This notebook demonstrates the **shipped family-basis model** of the BIND
lightcone suite: every weak-lensing field statistic is written as

$$\mathrm{stat}(x;\lambda) \;=\; \overline{\mathrm{stat}}(x)
\;+\; \sum_k a_k(\lambda)\, B_k(x),
\qquad a_k(\lambda) = M_k\cdot(\lambda-\lambda_{\rm ref}) + a_{{\rm ref},k}$$

* the **anchor** $\overline{\mathrm{stat}}(x)$ is the *measured design-mean
  curve* — the pointwise mean of the 256 Sobol runs' statistic, shipped as
  data in the bundle (at $\lambda=\lambda_{\rm ref}$ the model returns
  exactly this curve). It cannot be absorbed into the basis: the $B_k$ are
  built from bound-to-bound response *differences*, so they span the
  tangent directions of the feedback-response manifold while the mean pins
  its location — for $S(\ell)$, 8.9% of $\overline S - 1$ lies outside
  $\mathrm{span}(B)$ (max $5\times10^{-3}$ in $S$ units, larger than the
  low-$\ell$ predictive $\sigma$), so an $S = 1 + a\cdot B$ rewrite would
  cost more than the model's own precision;
* the **basis vectors** $B_k(x)$ are *measured shape archetypes* — the
  amplitude-weighted mean response shapes of the twobound parameter families
  (C1, C2, …) found by clustering the 30 one-parameter bound-to-bound
  responses;
* the **amplitudes** $a_k$ are linear in **eight measured,
  observationally accessible halo numbers** $\lambda$, chosen by a fully
  **agnostic search** (2026-08-12): sequential floating forward selection
  over 90 candidate halo-population summaries (medians, widths, and mass
  trends of X-ray/SZ/optical-accessible quantities in seven
  $\log_{10}M_{500c,\rm bg}$ bins from 13.0 to 14.3+; dark-matter-only
  quantities excluded), scored by the pooled end-to-end CV accuracy over
  all seven statistics, with no a-priori anchor set and no cap on the
  latent count. The search stops at eight on its own. Its first pick is
  the group baryon fraction (the van Daalen variable, rediscovered in
  20/20 bootstrap re-selections), later evicted as redundant once the
  same bin's Compton-$Y$ and temperature enter ($Y \simeq$ gas mass
  $\times$ temperature):

| latent | $\log_{10}M_{500c,\rm bg}$ bin | definition |
|---|---|---|
| $\tilde f_\star$ | $[13.2,13.4)$ | median $M_{\star,500c}/M_{\rm tot,500c,bg} \,/\, (\Omega_b/\Omega_m)$ |
| $\log\tilde T$ | $[13.0,13.2)$ | $\log_{10}$ median $T_{\rm mw,500c}$ [K] |
| $\log\tilde Y$ | $[13.0,13.2)$ | $\log_{10}$ median $Y_{500c}$ ($Y>0$) |
| $\log\tilde P_e$ | $[14.0,14.3)$ | $\log_{10}$ median $P_{e,\rm mw,500c}$ ($P_e>0$) |
| $c_{\rm gas}$ | $[14.0,14.3)$ | median $M_{\rm gas,500c,bg}/M_{\rm gas,200c,bg}$ |
| $\log\tilde Y_{\rm ss}$ | $[13.4,13.6)$ | $\log_{10}$ median $Y_{500c}/M_{\rm tot,500c,bg}^{5/3}$ |
| $c_{\rm gas}$ | $[13.2,13.4)$ | median $M_{\rm gas,500c,bg}/M_{\rm gas,200c,bg}$ |
| $\log\tilde P_e$ | $[13.4,13.6)$ | $\log_{10}$ median $P_{e,\rm mw,500c}$ ($P_e>0$) |

(Bins require $\ge 5$ halos, else NaN; $\Omega_b/\Omega_m=0.0486/0.3089$;
row order = the search's selection order.)

There is **no black box anywhere**: no emulator weights, no GP, no CAMELS
parameters --- and every latent is a quantity X-ray, SZ, or optical
observations can deliver. Measure eight numbers in *your* simulation (or
constrain them from observations) and the model returns $S(\ell)$, the
$\kappa$-PDF, peak and minima counts, and the three Minkowski functionals.
Everything lives in one npz bundle
(`figs_preview/family_model_bundle.npz`, built by `family_basis_all.py`)
consumed by the numpy-only loader `family_model.py`. Prior bundles are
preserved at `figs_preview/family_model_bundle_2bin8.npz` (two-bin
eight-latent) and `figs_preview/family_model_bundle_4lat.npz` (original
four-latent).
''')

code(r'''
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup
setup()
import matplotlib.pyplot as plt

from family_model import FamilyModel

fm = FamilyModel()
print("statistics:", fm.stats)
print("latents:   ", fm.lat_names)
print("convention:", fm.lat_convention)
rows = []
for st in fm.stats:
    m = fm.metrics(st)
    rows.append(dict(stat=st, families=",".join(fm.family_names(st)),
                     span_R2=round(m["r2_span"], 4),
                     lambda_e2e_CV=round(m["r2_lambda_e2e_cv"], 4)))
pd.DataFrame(rows).set_index("stat")
''')

md(r'''
`span_R2` is the free-amplitude ceiling (how much of the 256-run Sobol
variation the basis *can* absorb); `lambda_e2e_CV` is the 5-fold
cross-validated accuracy of the full pipeline "measure $\lambda$ → predict
the statistic".
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 1. The basis vectors

Each $B_k$ is the (unit-peak) mean response shape of one twobound parameter
family — a measured curve, not a fit product.
''')

code(r'''
fig, (a1, a2) = plt.subplots(1, 2, figsize=(TWO_COL[0], 2.4))
FC = ["#2a78d6", "#eb6834", "#199e70", "#c98500", "#d55181"]
for ax, st, logx in ((a1, "clk", True), (a2, "pdf", False)):
    B = fm.basis(st)
    for k, nm in enumerate(fm.family_names(st)):
        ax.plot(fm.grid(st), B[k], color=FC[k % len(FC)], lw=1.4, label=nm)
    if logx:
        ax.set_xscale("log")
    ax.axhline(0, color="0.85", lw=0.5)
    ax.set_xlabel(r"$\ell$" if st == "clk" else r"$\nu=\kappa/\sigma_\kappa$")
    ax.set_ylabel(rf"$B_k$ ({st})")
    ax.legend(fontsize=6)
plt.tight_layout()
plt.show()
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 2. Predicting: turn the eight physical knobs

First we load the Sobol design cloud (the 256 runs' measured latents and
statistics) — it defines the range over which each knob may be swept, and
below it serves as the ground truth the model is checked against.
''')

code(r'''
OB_OM = 0.0486 / 0.3089
LAT_BINS = {"13.0-13.2": (13.0, 13.2), "13.2-13.4": (13.2, 13.4),
            "13.4-13.6": (13.4, 13.6), "14.0-14.3": (14.0, 14.3)}

def _bin_med(arr, mask, pos=False, log=False, min_n=5):
    """median of arr within a log10(Mtot) bin mask; optional positivity
    guard + log10 of the median; NaN if fewer than min_n finite values."""
    v = arr[mask]
    if pos:
        v = np.where(v > 0, v, np.nan)
    if np.isfinite(v).sum() < min_n:
        return np.nan
    m = np.nanmedian(v)
    return np.log10(m) if (log and m > 0) else m

def measure_latents(m_tot_500, m_gas_500, m_star_500, m_gas_200,
                    T_mw_500, Pe_mw_500, Y_500):
    """The eight model latents from per-halo arrays (any simulation) --
    the 2026-08-12 OBSERVABLE AGNOSTIC set, in selection order. Masses
    in Msun (any consistent unit), T in K, Pe and Y in the atlas
    conventions; *_500/*_200 are the bg-subtracted 500c/200c apertures
    (m_star has no bg convention). Mirrors family_basis_all.py exactly."""
    lm = np.log10(np.where(m_tot_500 > 0, m_tot_500, np.nan))
    b = {k: (lm >= lo) & (lm < hi) for k, (lo, hi) in LAT_BINS.items()}
    for k, m in b.items():
        assert m.sum() >= 5, f"need >=5 halos in the {k} bin"
    with np.errstate(divide="ignore", invalid="ignore"):
        c_gas = m_gas_500 / m_gas_200
        y_ss = Y_500 / np.where(m_tot_500 > 0, m_tot_500, np.nan) ** (5 / 3)
        return np.array([
            _bin_med(m_star_500 / m_tot_500, b["13.2-13.4"]) / OB_OM,
            _bin_med(T_mw_500, b["13.0-13.2"], pos=True, log=True),
            _bin_med(Y_500, b["13.0-13.2"], pos=True, log=True),
            _bin_med(Pe_mw_500, b["14.0-14.3"], pos=True, log=True),
            _bin_med(c_gas, b["14.0-14.3"]),
            _bin_med(y_ss, b["13.4-13.6"], pos=True, log=True),
            _bin_med(c_gas, b["13.2-13.4"]),
            _bin_med(Pe_mw_500, b["13.4-13.6"], pos=True, log=True),
        ])

cz = np.load("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/"
             "atlas_cube_snap096.npz")
dsn = np.load("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_nu05.npz",
              allow_pickle=True)
coef = np.load("latent_model_coeffs.npz")
EDGb = coef["ell_edges"]
_eld = np.asarray(dsn["a__suppression__ell"], float)

def sobol_Y(st):
    dk = {"clk": "suppression", "pdf": "pdf", "pk": "peak_counts",
          "mn": "minima_counts", "v0": "mf_v0", "v1": "mf_v1",
          "v2": "mf_v2"}[st]
    Y = np.asarray(dsn[f"t__{dk}__value"], float)
    Y = Y[:, 1, :] if Y.ndim == 3 else Y
    if st == "clk":
        Y = np.stack([np.nanmean(Y[:, (_eld >= EDGb[i]) & (_eld < EDGb[i+1])],
                                 1) for i in range(len(EDGb) - 1)], axis=1)
    return Y[:, :fm.basis(st).shape[1]]

rows_s = dsn["run_ids"]
LAT = np.stack([measure_latents(
    cz["sobol_m_tot_500c_bg"][r], cz["sobol_m_gas_500c_bg"][r],
    cz["sobol_m_star_500c"][r], cz["sobol_m_gas_200c_bg"][r],
    cz["sobol_T_mw_500c"][r], cz["sobol_Pe_mw_500c"][r],
    cz["sobol_Y_500c"][r]) for r in rows_s])
assert np.allclose(LAT.mean(0), fm._z["clk__lat_ref"], atol=1e-4), \
    "tutorial measure_latents disagrees with the shipped bundle"
Yclk = sobol_Y("clk")
print("design latent ranges (2nd-98th pct):")
for j, nm in enumerate(fm.lat_names):
    lo, hi = np.percentile(LAT[:, j], [2, 98])
    print(f"  {nm:>18s}: {lo:.3f} .. {hi:.3f}")
''')

md(r'''
Now sweep **each latent** one at a time across its design range, with the
other seven held at the design mean — the model's *partial* response to
each physical knob. Split into two 4-row figures for readability: the
search's first four picks (the stellar partition, the low-mass-group
temperature and Compton-$Y$, and the cluster electron pressure), then the
final four (the cluster and group gas concentrations, the
self-similar-scaled $Y$, and the mid-group pressure). (PDF panels are
log-$y$.)
''')

code(r'''
lam0 = fm._z["clk__lat_ref"].copy()
LN_ALL = [r"$\tilde f_\star[13.2]$", r"$\log\tilde T[13.0]$",
          r"$\log\tilde Y[13.0]$", r"$\log\tilde P_e[14.0]$",
          r"$c_{\rm gas}[14.0]$", r"$\log\tilde Y_{\rm ss}[13.4]$",
          r"$c_{\rm gas}[13.2]$", r"$\log\tilde P_e[13.4]$"]
cmap = plt.get_cmap("viridis")

def plot_sweep(idxs, fname):
    fig, AX = plt.subplots(len(idxs), 2, figsize=(TWO_COL[0], 2.05 * len(idxs)))
    for row, j in enumerate(idxs):
        lo, hi = np.percentile(LAT[:, j], [2, 98])
        grid = np.linspace(lo, hi, 6)
        for v in grid:
            lam = lam0.copy(); lam[j] = v
            c = cmap((v - lo) / (hi - lo))
            AX[row, 0].plot(fm.grid("clk"), fm.predict(lam, "clk")[0],
                            color=c, lw=1.2, label=f"{LN_ALL[j]}={v:.2f}")
            AX[row, 1].plot(fm.grid("pdf"), fm.predict(lam, "pdf")[0],
                            color=c, lw=1.2)
        AX[row, 0].set_xscale("log")
        AX[row, 0].axhline(1, color="0.85", lw=0.5)
        AX[row, 0].set_ylabel(r"$S(\ell)$", fontsize=7)
        AX[row, 0].legend(fontsize=4.8, loc="lower left", ncol=2)
        AX[row, 1].set_yscale("log")
        AX[row, 1].set_ylim(1e-4, 1.2)
        AX[row, 1].set_ylabel(r"PDF$(\nu)$", fontsize=7)
        AX[row, 1].text(0.97, 0.85, LN_ALL[j] + " swept",
                        transform=AX[row, 1].transAxes, ha="right", fontsize=7)
        if row == len(idxs) - 1:
            AX[row, 0].set_xlabel(r"$\ell$"); AX[row, 1].set_xlabel(r"$\nu$")
    fig.tight_layout()
    fig.savefig(f"figs_preview/{fname}.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_sweep([0, 1, 2, 3], "tutorial_latent_sweeps")
''')

md(r'''
And the search's final four picks:
''')

code(r'''
plot_sweep([4, 5, 6, 7], "tutorial_latent_sweeps_ext")
''')

md(r'''
**Stay inside the design.** The map is linear; latents far outside the Sobol
design extrapolate unphysically.

### 2b. Proof of the van Daalen connection

"Deeper suppression at lower baryon budget" is the weak-lensing analog of
the van Daalen et al. relation ($\Delta P/P$ vs the group baryon
fraction). Here the test is sharper than before, because **the baryon
fraction is not even an input to the model** — the agnostic search picked
it first, then evicted it as redundant with the group Compton-$Y$ and
temperature. If the model still traces the measured $S$--$f_{\rm bar}$
locus, it *contains* the van Daalen relation in a variable it never sees.

The demonstration: measure $\tilde f_{\rm bar}$ (low-mass-group bin,
$13.0\!\le\!\log_{10}M\!<\!13.2$ — the search's first pick) as an
*auxiliary* variable on every run; find the suite's van-Daalen scale (the
$\ell$ band where $S$ correlates most strongly with it); then sweep the
**eight latents along their design-conditional means given
$f_{\rm bar}$** and ask whether the model's prediction rides the measured
cloud. A *partial* sweep does not exist here — there is no budget knob to
turn — which is exactly the point: the budget acts on the statistics only
through the thermodynamic state it generates.
''')

code(r'''
# auxiliary variable: the low-mass-group baryon fraction (search's 1st pick)
def measure_fbar(m_tot_500, m_gas_500, m_star_500):
    lm = np.log10(np.where(m_tot_500 > 0, m_tot_500, np.nan))
    s = (lm >= 13.0) & (lm < 13.2)
    return np.nanmedian(((m_gas_500 + m_star_500) / m_tot_500)[s]) / OB_OM

FB = np.array([measure_fbar(cz["sobol_m_tot_500c_bg"][r],
                            cz["sobol_m_gas_500c_bg"][r],
                            cz["sobol_m_star_500c"][r]) for r in rows_s])

# the suite's van Daalen scale: the band where S correlates best with f_bar
r_b = np.array([np.corrcoef(FB, Yclk[:, b])[0, 1]
                for b in range(Yclk.shape[1])])
b_vd = int(np.nanargmax(np.abs(r_b)))
ell_vd = fm.grid("clk")[b_vd]
r_vd = r_b[b_vd]

fb_lo, fb_hi = np.percentile(FB, [2, 98])
fb_grid = np.linspace(fb_lo, fb_hi, 30)
# conditional sweep: ALL EIGHT latents at their design-conditional mean | f_bar
cond = np.stack([np.polyval(np.polyfit(FB, LAT[:, j], 1), fb_grid)
                 for j in range(LAT.shape[1])], axis=1)
S_cond = fm.predict(cond, "clk")[:, b_vd]

# quantify: model conditional line vs binned medians of the measured cloud
edges = np.linspace(fb_lo, fb_hi, 9)
ctr, med = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (FB >= lo) & (FB < hi)
    if s.sum() >= 8:
        ctr.append(0.5 * (lo + hi))
        med.append(np.median(Yclk[s, b_vd]))
rms = float(np.sqrt(np.mean((np.interp(ctr, fb_grid, S_cond)
                             - np.array(med)) ** 2)))

fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.scatter(FB, Yclk[:, b_vd], s=8, color="0.6", alpha=0.55,
           rasterized=True, label="256 Sobol runs (measured)")
ax.plot(ctr, med, "s", ms=5, color="k", label="binned medians")
ax.plot(fb_grid, S_cond, color=COLORS["bind"], lw=2.0,
        label=r"model, $\lambda$ at design-conditional means $|\,f_{\rm bar}$")
ax.set_xlabel(r"$\tilde f_{\rm bar}[13.0$--$13.2]$ (measured; NOT a model input)")
ax.set_ylabel(rf"$S(\ell \simeq {ell_vd:.0f})$")
ax.set_title(f"the suite's van Daalen relation: $r={r_vd:+.2f}$; "
             f"model line vs medians RMS={rms:.4f}", fontsize=7.5)
ax.legend(fontsize=6, loc="lower right")
fig.tight_layout()
fig.savefig("figs_preview/tutorial_vandaalen_proof.png", dpi=150,
            bbox_inches="tight")
plt.show()
print(f"van Daalen scale of the suite: ell ~ {ell_vd:.0f} "
      f"(|r| = {abs(r_vd):.2f}); conditional-model vs binned-median RMS "
      f"= {rms:.4f} (S units)")
''')

md(r'''
The conditional sweep rides the measured cloud: the model contains the
suite's van Daalen relation *without the baryon fraction among its
inputs*. The budget's information reaches the statistics entirely through
the latents that co-vary with it — chiefly the same bin's Compton-$Y$ and
temperature, the thermal-energy content the budget generates.
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 3. Bring your own simulation (here: the TNG300 fiducial) — vs the TRUTH

All the model needs from a simulation is a group catalog. We reuse the
`measure_latents` helper on the fiducial (TNG-calibrated) paint of the shared
2933-halo atlas — this is how you would run TNG, SIMBA, or any other box
through the model.

And because the fiducial is **out of the Sobol design**, we can close the
loop against the *measured* curves: the fiducial BIND paint's suppression
(50 seed-paired ray-trace realizations over the paired DMO trace — the
`Cl_kappa_paired` per-realization cache, immune to the 550-real denominator
mismatch) and the **full-hydro TNG300 truth** itself. The model line below
comes from *eight measured halo numbers*; the black line is the actual
hydrodynamical simulation.
''')

code(r'''
lam_fid = measure_latents(cz["fid_m_tot_500c_bg"], cz["fid_m_gas_500c_bg"],
                          cz["fid_m_star_500c"], cz["fid_m_gas_200c_bg"],
                          cz["fid_T_mw_500c"], cz["fid_Pe_mw_500c"],
                          cz["fid_Y_500c"])
print("fiducial (TNG300) lambda:", np.round(lam_fid, 3))

SCI = "/mnt/home/mlee1/ceph/bind_science"
fc = np.load(f"{SCI}/field_cache/field_stats_fid.npz")
dmo = np.load(f"{SCI}/runs/dmo/run_0000/Cl_kappa_paired.npz")

def band_reals(cl_real):
    """per-realization suppression ratio, band-averaged into the model
    bands (seed-paired ratio real-by-real, THEN band mean)."""
    S = cl_real[:, 1, :] / dmo["cl_real"][:, 1, :]
    return np.stack([np.nanmean(S[:, (_eld >= EDGb[i]) & (_eld < EDGb[i+1])],
                                1) for i in range(len(EDGb) - 1)], axis=1)

S_bind = band_reals(fc["kk_bind"])       # fiducial BIND paint (50 reals)
S_true = band_reals(fc["kk_truth"])      # full-hydro TNG300 truth

def pdf_to_nu(kb, p):
    """standardize a raw-kappa PDF onto the canonical nu grid (the model
    convention): PDF_nu(nu) = sigma * PDF_kappa(mu + sigma*nu)."""
    norm = np.trapezoid(p, kb)
    mu = np.trapezoid(kb * p, kb) / norm
    sig = np.sqrt(np.trapezoid((kb - mu) ** 2 * p, kb) / norm)
    return sig * np.interp(mu + sig * fm.grid("pdf"), kb, p, left=0, right=0)

pdf_bind = pdf_to_nu(fc["pdf_bins"], np.nanmean(fc["pdf_bind"][:, 1, :], 0))
pdf_true = pdf_to_nu(fc["pdf_bins"], np.nanmean(fc["pdf_truth"][:, 1, :], 0))

xg = fm.grid("clk")
pred_S = fm.predict(lam_fid, "clk")[0]
pred_P = fm.predict(lam_fid, "pdf")[0]
sig_S = fm.predictive_sigma("clk")
sig_P = fm.predictive_sigma("pdf")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(TWO_COL[0], 2.6))
m, e = S_true.mean(0), S_true.std(0) / np.sqrt(len(S_true))
a1.plot(xg, m, color="#111111", lw=1.5, label="TNG300 full hydro (truth)")
a1.fill_between(xg, m - e, m + e, color="#111111", alpha=0.18, lw=0)
mb = S_bind.mean(0)
a1.plot(xg, mb, color="0.55", lw=1.2, label="BIND fiducial paint (measured)")
a1.plot(xg, pred_S, color=COLORS["bind"], lw=1.8, ls="--",
        label=r"model: 8 measured numbers $\to S(\ell)$")
a1.fill_between(xg, pred_S - sig_S, pred_S + sig_S, color=COLORS["bind"],
                alpha=0.18, lw=0, label=r"model $\pm1\sigma$ predictive")
a1.set_xscale("log"); a1.axhline(1, color="0.85", lw=0.5)
a1.set_xlabel(r"$\ell$"); a1.set_ylabel(r"$S(\ell)$")
a1.legend(fontsize=5.5, loc="lower left")
a2.plot(fm.grid("pdf"), pdf_true, color="#111111", lw=1.5)
a2.plot(fm.grid("pdf"), pdf_bind, color="0.55", lw=1.2)
a2.plot(fm.grid("pdf"), pred_P, color=COLORS["bind"], lw=1.8, ls="--")
a2.fill_between(fm.grid("pdf"), pred_P - sig_P, pred_P + sig_P,
                color=COLORS["bind"], alpha=0.18, lw=0)
a2.set_yscale("log"); a2.set_ylim(1e-4, 1.2)
a2.set_xlabel(r"$\nu$"); a2.set_ylabel(r"PDF$(\nu)$")
plt.tight_layout()
fig.savefig("figs_preview/tutorial_fiducial_truth.png", dpi=150,
            bbox_inches="tight")
plt.show()

dev_b = 100 * np.abs(pred_S - mb) / mb
dev_t = 100 * np.abs(pred_S - m) / m
tro = int(np.argmin(mb))
print(f"S(ell) fiducial closure (out-of-design): median |pred - BIND fid| "
      f"= {np.median(dev_b):.2f}%, |pred - hydro truth| = "
      f"{np.median(dev_t):.2f}% (max {dev_t.max():.2f}%)")
print(f"at the trough (band {tro}, ell~{xg[tro]:.0f}): pred {pred_S[tro]:.3f}"
      f" vs measured {mb[tro]:.3f} -- a {abs(pred_S[tro]-mb[tro])/sig_S[tro]:.1f}"
      f" sigma draw of the model's own predictive scatter there "
      f"({sig_S[tro]:.3f}), which is {100*sig_S[tro]/(1-mb[tro]):.0f}% of "
      "the suppression effect")
''')

md(r'''
Reading the closure **honestly**: this is the **observable
agnostic-lambda** model (2026-08-12). The generations at the trough,
against a measured 0.881: the original hand-picked 4 group-bin latents
predicted 0.920 (recovering only $\sim$2/3 of the 0.119 suppression
there); the two-bin extended-lambda 8 brought it to $\sim$0.908; the
agnostic observable 8 bring it to $\sim$0.895 — a $0.7\sigma$ draw of
the model's own predictive scatter ($\sigma_{\rm pred} \simeq 0.019$
there), an error of $\sim$12\% of the effect. The gap is small, not
gone: the free-amplitude fit of the measured fiducial curve has a
residual of only 0.001 — the *basis* was never the limitation; every
improvement has lived in the $\lambda\to a$ map, and the residual is a
design-coverage limitation (richer maps predict this out-of-design
point *worse*). The $\pm1\sigma$ predictive band (drawn above) is
still wider at $\ell\gtrsim10^4$ than at the van Daalen scale — eight
measured halo numbers pin the large/mid-scale suppression almost
perfectly and the small-scale trough well but not perfectly.
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 4. Validation: leave-one-out predictions vs measurement, all 7 statistics

The `pfig_s4b_model_curves` layout, re-made with **this** model: for seven
Sobol runs spanning the design (including the strongest suppression and
enhancement), we refit the $\lambda\to a$ map with the shown run **held
out**, predict every statistic from that run's *measured* halo latents, and
overlay the measurement. The basis needs no holding out — it is built from
the independent twobound suite. (The design-mean curve uses all runs; the
1/256 effect of the shown run on it is negligible.) Stamps: the median
absolute residual over the seven shown runs, in % of the measurement (clk)
or of the per-run curve maximum (the $\nu$-domain statistics).
''')

code(r'''
Lc = LAT - LAT.mean(0)
key = (Yclk - Yclk.mean(0)).mean(1)
SHOW = [int(np.argsort(key)[int(q * (len(key) - 1))])
        for q in np.linspace(0, 1, 7)]
print("shown runs:", SHOW)

SPECN = {"clk": (r"$S(\ell)$", r"$\ell$", True),
         "pdf": (r"$P(\nu)$", r"$\nu$", False),
         "pk":  (r"$N_{\rm pk}$", r"$\nu$", False),
         "mn":  (r"$N_{\rm min}$", r"$\nu$", False),
         "v0":  (r"$V_0$", r"$\nu$", False),
         "v1":  (r"$V_1$", r"$\nu$", False),
         "v2":  (r"$V_2$", r"$\nu$", False)}

fig = plt.figure(figsize=(TWO_COL[0], 6.4))
gs = fig.add_gridspec(4, 4, height_ratios=[2.2, 1.0, 2.2, 1.0], hspace=0.45,
                      wspace=0.42)
for i, st in enumerate(fm.stats):
    r0, c0 = 2 * (i // 4), i % 4
    ax = fig.add_subplot(gs[r0, c0])
    rx = fig.add_subplot(gs[r0 + 1, c0], sharex=ax)
    slab, xlab, logx = SPECN[st]
    xg = fm.grid(st)
    Y = sobol_Y(st)
    B = fm.basis(st)
    mean = fm._z[f"{st}__mean"]
    A = fm._z[f"{st}__amps"]
    resids = []
    for r in SHOW:
        keep = np.arange(len(LAT)) != r          # leave the shown run out
        W, *_ = np.linalg.lstsq(np.c_[Lc[keep], np.ones(keep.sum())],
                                A[keep], rcond=None)
        a_pred = np.r_[Lc[r], 1.0] @ W
        model = mean + a_pred @ B
        meas = Y[r]
        ax.plot(xg, meas, color="k", lw=1.1)
        ax.plot(xg, model, color=COLORS["bind"], lw=1.1, ls="--")
        den = meas if st == "clk" else np.full_like(meas,
                                                    np.abs(meas).max())
        rel = 100 * (model - meas) / den
        rx.plot(xg, rel, color=COLORS["bind"], lw=0.7, alpha=0.8)
        resids.append(np.abs(rel))
    ax.text(0.05, 0.06, f"{np.median(np.concatenate(resids)):.1f}%",
            transform=ax.transAxes, fontsize=6.5, color=COLORS["bind"])
    if logx:
        ax.set_xscale("log")
        ax.axhline(1, color="0.9", lw=0.5, zorder=0)
    if st == "pdf":
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.2)
    rx.axhline(0, color="0.6", lw=0.6)
    rx.axhspan(-2, 2, color="0.92", zorder=0)
    ax.set_ylabel(slab, fontsize=7)
    rx.set_xlabel(xlab, fontsize=7)
    rx.set_ylabel(r"$\Delta$ [%]", fontsize=6)
    ax.tick_params(labelsize=6, labelbottom=False)
    rx.tick_params(labelsize=6)
    panel_label(ax, f"({'abcdefg'[i]})")
# legend in the free slot
axL = fig.add_subplot(gs[2:, 3])
axL.axis("off")
axL.plot([], [], color="k", lw=1.1, label="measured (7 held-out runs)")
axL.plot([], [], color=COLORS["bind"], lw=1.1, ls="--",
         label=r"family basis + $a(\hat\lambda)$, leave-one-out")
axL.plot([], [], color="0.8", lw=6, alpha=0.6,
         label=r"strips: (model$-$meas), $\pm2\%$ band")
axL.legend(fontsize=6.5, loc="center left", frameon=False)
save(fig, "figs_v2/pfig_family_model_curves")
plt.show()
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 5. Fine print

* **Accuracy.** The end-to-end 5-fold-CV $R^2$ per statistic is in the table
  of §0 (0.79–0.96 with the observable agnostic 8, up from 0.77–0.94
  with the original hand-picked 4); the free-amplitude ceilings
  (0.86–0.9997) say how much of the gap is the $\lambda\to a$ map vs the
  basis itself.
* **Observability.** Every latent is X-ray/SZ/optical accessible by
  construction (the search library excluded dark-matter-only
  quantities); the restriction costs nothing — the unrestricted search
  lands within fold noise of the same accuracy, swapping the two gas
  concentrations for gas-to-DM concentration ratios. The mass *binning*
  uses the true $M_{500c}$; observational use adds mass-proxy
  bias/scatter, hydrostatic bias, and projection effects not modeled
  here.
* **Range of validity.** Fixed cosmology (TNG-calibrated suite), $z_s = 1$
  source plane, the Sobol design's astrophysics range; the $\lambda$ map is
  linear — do not extrapolate far outside the design.
* **Amplitudes.** `fm.amplitudes(lam, stat)` exposes the family amplitudes
  themselves; their physical identifications are follow-up work — see
  `FAMILY_BASIS_METHODS.md`.
* **What each piece inherits.** $B_k$: the twobound suite's S/N gate and
  the clustering threshold (S/N $\ge$ 3, $t=0.15$). The per-run
  amplitudes: plain OLS (minimum-norm pinv) everywhere — as of 2026-08-12
  there is no ridge branch. clk's basis is still ill-conditioned
  ($\kappa=520$), so individual clk amplitudes carry a large, mostly
  free component along the basis's near-null direction (max $|a|$ grows
  from $\sim$0.25 under ridge to $\sim$1.9 under OLS); that direction's
  curve image is $\lesssim 0.005$ — observationally invisible — so it
  costs interpretability, not accuracy (e2e CV is ridge-vs-OLS
  equivalent, 0.9585 vs 0.9593). Compare individual clk amplitudes only
  through well-conditioned directions of $B_k$ (e.g. the pooled/rotated
  combinations used for the response fits), or re-project with an
  explicit ridge for display if a single-family reading is wanted. The
  other six statistics ($\kappa \le 91$) are unaffected. The
  $\lambda\to a$ map: the Sobol design's *coverage* — the residual
  fiducial trough gap is a coverage limitation, and the 2026-08-12
  bake-off showed more flexible mappings (polynomials, $\theta$-NNs, GPs)
  make it worse, not better. $\overline{\mathrm{stat}}$: the SB35 prior
  choice (it is the average feedback universe *of this design*).
* **Provenance.** Basis + amplitudes: `family_basis_all.py` (twobound
  families × 256-run Sobol suite); latent selection:
  `agnostic_lambda_search.py` (OBS_ONLY=1; log in `audits/`, results in
  `figs_preview/agnostic_lambda_results_obs.npz`); bundle:
  `figs_preview/family_model_bundle.npz`; loader: `family_model.py`.
''')

# ═════════════════════════════════════════════════════════════════════════════
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if t == "markdown" else
            nbf.v4.new_code_cell(s) for t, s in CELLS]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
out = "family_model_tutorial.ipynb"
nbf.write(nb, out)
print(f"wrote {out} with {len(nb.cells)} cells")
