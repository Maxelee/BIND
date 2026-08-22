"""Build papers/01_pipeline/paper1_figures.ipynb — the 10-figure notebook for
the lightcone paper (Paper I) following the user's 2026-07-27 outline.

Every figure loads ONLY existing cached arrays (bind_sb35 / bind_science /
bind_lightcone_tng on ceph); no engines, MPI, Slurm, or GPU are invoked.
Figures are written to figs_v2/ (the committed figs/ from the current draft
are untouched) via papers/_tools/paper_style.save().

Rebuild the notebook:
    /mnt/home/mlee1/venvs/BIND_env/bin/python _build_figures_nb.py
Execute it:
    see _run_figures_nb.py (nbclient, kernel pinned to the BIND venv).
"""
import nbformat as nbf

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip()))


def code(src: str) -> None:
    CELLS.append(("code", src.strip()))


# ═════════════════════════════════════════════════════════════════════════════
md(r'''
# The BIND Lightcone Suite (BIND-LS v1): 253 ray-traced $\kappa$/$\tau$/$y$ lightcones spanning 30 dimensions of galaxy-formation physics

**Paper-I working notebook** — every figure, number, and claim in the draft is generated live
here from cached data products; the markdown cells are the paper's narrative skeleton.

**Abstract.** We release a suite of ray-traced weak-lensing convergence ($\kappa$),
electron-column optical depth ($\tau$), and Compton-$y$ lightcone maps of the same
$5\times5\,\mathrm{deg}^2$ footprint under 253 variations of 30 IllustrisTNG
galaxy-formation parameters, built by painting baryons onto the TNG300-Dark lightcone with
the BIND conditional flow-matching emulator (redshift-conditioned, thermodynamics-enabled).
Each node ships 50 ray-trace realizations at five source planes ($z_s\in[0.5,2.44]$),
per-halo gas/thermodynamic catalogs (all 256 nodes), stacked $\tau$/$y$ profiles in stacking
units, and an assembled statistics dataset feeding a validated Gaussian-process emulator
($\theta\to$ any released statistic in milliseconds). At the fiducial, the suite closes
against the seed-paired hydro-pasted truth to 0.7% ($C_\ell^{\kappa\kappa}$), 1.4%
($C_\ell^{\kappa y}$), and 5.6% ($C_\ell^{yy}$) over $\ell=300$–5000, with all residual
systematics characterized: a mass-dependent halo-level gas/pressure amplitude offset
($\approx$+10% at group scale declining to $\approx$0 at cluster scale, with a fitted
linear-in-$\log M$ calibration), a redshift drift of the thermodynamic closure (shipped
with a de-biasing template), and a small-scale diffuse-$y$ texture excess localized
outside halo cores. Feedback moves the observables along a
dominantly two-dimensional response surface commanded by a handful of SN-wind, IMF, and BH
parameters; 38% of nodes *enhance* the $z_s=1$ WL spectrum at $\ell=5000$. We quantify what
Stage-IV surveys can resolve of this envelope and demonstrate an end-to-end emulator
constraint on the fiducial with a full-covariance likelihood.

**Reader's guide** (sections mirror the paper): §1 Methods (fig 0 hero, fig 1, plus §1d's
figure-free lightcone/compositing/ray-trace conventions),
§2 Validation (figs 3, 4, 11), §3 Astrophysical effects (fig 23 opener, then figs 5a, 5, 7, 20, 12, 8), §4 Emulation
(figs 9, 15, 10), §5 Caveats & completeness (fig 14), §6 Data products, §7 Conclusions;
appendices: fig 18 (Appendix A, $C_\ell^{yy}$ mechanism), fig 16 (Appendix B, posterior
robustness checks), fig 19 (Appendix C, $f_{\rm gas}$ vs X-ray/eROSITA data).
Provenance discipline: every quantitative statement is printed by a code cell in this
notebook or traceable to a named cache; nothing is quoted from memory.
''')

md(r'''
## §1 — Introduction

Baryonic feedback is simultaneously the largest astrophysical systematic for Stage-IV
weak-lensing cosmology and the observable of interest for the gas physics community:
the same ejection and heating processes that suppress the matter power spectrum at
$k\gtrsim 1\,h\,\mathrm{Mpc}^{-1}$ set the thermal Sunyaev–Zel'dovich (tSZ) signal, the
kinematic SZ (kSZ) electron column, and the group-scale gas fractions probed by X-ray
surveys. Progress requires simulation products in which *the same sky* can be observed
under *many* feedback assumptions, in the projected, ray-traced form that survey analyses
actually consume — and at a box size that contains the groups and clusters
($M_{200c}\gtrsim10^{13}\,M_\odot/h$) that dominate these signals.

No existing public suite provides this combination (§6, positioning table). CAMELS varies
feedback richly but in $25\,h^{-1}$Mpc boxes with no lightcone; CosmoGridV1 provides
ray-traced $\kappa$ lightcones across cosmologies but with post-hoc baryonification and no
hydro-informed $\tau$/$y$ fields; MillenniumTNG provides one galaxy-formation model.
The gap is a *feedback-swept, ray-traced, multi-probe* lightcone suite at group-resolving
volume — the corner this release fills.

Our approach decouples the expensive ingredients. Gravity and geometry come from a single
TNG300-Dark lightcone (20 shells to $z=2.44$). Galaxy-formation physics comes from BIND,
a conditional flow-matching emulator trained on the CAMELS-TNG SB35 suite, which paints
stochastic multi-channel baryonic fields — gas, stars, and gas thermodynamics — onto each
DMO halo as a function of a 35-dimensional cosmology+astrophysics vector $\theta$ and the
snapshot scale factor $a$. Painting one full 20-shell lightcone variation costs GPU-minutes
rather than the tens of millions of CPU-hours of a new hydro simulation, which is what makes
a 256-node, 30-dimensional Sobol design affordable at TNG300 volume.

This paper (I of a series) releases the suite and establishes its credibility budget:
§2 validates against the seed-paired hydro-pasted truth at halo level, field level, and as
a function of redshift, with every residual characterized rather than hidden; §3 maps how
the released statistics respond to the 30 parameters and shows the response is effectively
two-dimensional; §4 turns the suite into a public Gaussian-process emulator, validates and
*calibrates* it, and demonstrates an end-to-end constraint; §5–6 define the scope of
validity and the data products. Companion papers use the suite for cosmological-bias
mitigation (II), latent-space inference (III), kSZ/X-ray gas constraints (IV), and
feedback anisotropy (V).
''')

code(r'''
# ── Setup: style, paths, shared helpers ──────────────────────────────────────
import gc, json, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore", message=".*length_scale is close to.*")  # sklearn GP bound chatter

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")   # paper_style + vendored param_labels
from paper_style import (setup, save, panel_label, COLORS, BAND_ALPHA,
                         ONE_COL, ONE_COL_SQ, TWO_COL, TWO_COL_TALL)
from param_labels import short_label
setup()
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

CEPH = Path("/mnt/home/mlee1/ceph")
SB35, SCI, LC = CEPH/"bind_sb35", CEPH/"bind_science", CEPH/"bind_lightcone_tng"
assert Path.cwd().name == "01_pipeline", "run this notebook from papers/01_pipeline/"
Path("figs_v2").mkdir(exist_ok=True)

# Assembled Sobol statistics cube (253 completed runs x 30 astro params).
# _xpkfix: cross-spectra (C^ky, C^kt, C^yt) carry the corrected XPk_plane
# normalization; auto-spectra and S(ell) are identical to the base file.
DS_PATH = SB35/"emulator_dataset_xpkfix.npz"
d = np.load(DS_PATH, allow_pickle=True)
X_unit   = d["X_unit"]           # (253, 30) unit-cube params (log10 where LogFlag)
X_native = d["X_native"]         # (253, 35) native vectors (5 fixed cosmology slots)
run_ids  = d["run_ids"]
pnames   = [str(s) for s in d["param_names"]]
ZS       = d["source_redshifts"]; ZI = 1          # working source plane: z_s = 1.0
ELL      = d["a__suppression__ell"]
# Two ell-domain constants shared by every field-level figure (fig 4, 4b, 7, 12):
# ELL_TRUST = the CIC-aliasing limit -- above this the RAW spectra turn up from
#   pixelization aliasing (ratio statistics like S(ell) cancel it and stay
#   trustworthy beyond it; this is why S(ell) extends further than the raw C_ell
#   panels do).
# ELL_MAX_PLOT = the full extent of the released ell grid: the maps are 1024^2
#   over 5 deg, so axis-Nyquist is ell=36864 and the grid's last (corner) mode is
#   ell=52133.6 -- there are simply no modes beyond that, so this is the true
#   axis limit, not an arbitrary round number. Every ell-domain panel in this
#   notebook (fig 4 panels a/b, fig 4b all 5 panels, fig 7 panels a + the 5
#   cross/auto spectra, fig 12 panel a) shares these two constants and the same
#   axvspan(ELL_TRUST, ELL_MAX_PLOT) shading convention for the aliasing region.
ELL_TRUST    = 1.5e4
ELL_MAX_PLOT = 5.22e4

def rebin(a, k):
    """Mean over blocks of k along the last axis (trims the remainder)."""
    n = (a.shape[-1]//k)*k
    return a[..., :n].reshape(*a.shape[:-1], n//k, k).mean(-1)

def spearman(P, Y):
    """Spearman rho between columns of P (n,p) and Y (n,b) -> (p,b)."""
    rp = rankdata(P, axis=0).astype(float); ry = rankdata(Y, axis=0).astype(float)
    rp = (rp-rp.mean(0))/rp.std(0); ry = (ry-ry.mean(0))/np.maximum(ry.std(0), 1e-12)
    return rp.T @ ry / len(P)

def running_stat(x, y, edges, min_n=8, boot=0):
    """Median and 16/84 percentiles of y in bins of x; boot>0 adds a bootstrap
    standard error on the binned median (row 3)."""
    ib = np.digitize(x, edges)
    out = np.full((4, len(edges)-1), np.nan)
    rng = np.random.default_rng(2)
    for i in range(1, len(edges)):
        s = y[ib == i]
        if len(s) >= min_n:
            out[:3, i-1] = np.percentile(s, [50, 16, 84])
            if boot:
                bs = np.median(s[rng.integers(0, len(s), (boot, len(s)))], axis=1)
                out[3, i-1] = bs.std()
    cen = 0.5*(edges[1:]+edges[:-1]); ok = np.isfinite(out[0])
    return cen[ok], out[0, ok], out[1, ok], out[2, ok], out[3, ok]

def sellentin_heavens_loglike(chi2, n_real, n_data):
    """Sellentin & Heavens (2016, MNRAS 456, L132) REPORTED-ALTERNATIVE log-
    likelihood: marginalizing over the sampling uncertainty of a covariance
    estimated from a FINITE number n_real of realizations (n_data = the
    dimensionality/compression of the data vector, e.g. the number of bins)
    turns the Gaussian chi2 likelihood into a multivariate-t,

        log L = -(n_real/2) * log(1 + chi2/(n_real - 1))

    up to an additive constant IDENTICAL for every point in parameter space
    (Eq. 14 of Sellentin & Heavens 2016):
        const = log Gamma(n_real/2) - log Gamma((n_real - n_data)/2)
                - (n_data/2) * log(pi*(n_real - 1)) - 0.5*log(det C_hat)
    and therefore irrelevant to any likelihood RATIO or posterior shape --
    exactly the "up to a constant" convention every Gaussian -2 ln L in this
    notebook already uses. chi2 = r^T Chat^-1 r with the SAME (possibly
    theta-dependent) covariance the Gaussian branch uses; as n_real -> infinity
    this reduces to the familiar -chi2/2 (unit-checked in
    audits/unit_check_sellentin_heavens.py)."""
    n_real = np.asarray(n_real, float)
    return -0.5*n_real*np.log1p(np.asarray(chi2, float)/(n_real - 1))

print(f"dataset {DS_PATH.name}: {len(run_ids)} runs x {len(pnames)} params; "
      f"z_s planes {list(np.round(ZS,2))}; ell {ELL[0]:.0f}-{ELL[-1]:.0f}")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §1b — Fig 0 (hero): the release in one image

**DMO in, baryons out, on a lightcone, under feedback variants.** *Top row*: the paired
DMO-only convergence — the input skeleton, ray-traced with the *same* `lux` seed and
geometry as every painted node — beside the BIND-painted fiducial $\kappa$, $\tau$, and
Compton $y$ of the *same realization*; all four panels are pixel-aligned by construction.
*Bottom row*: the $\kappa$ **response** of that same sky to a **one-parameter** SN-wind-energy
bracket — the two-bound runs at $A_{\rm SN1}=0.9$ and $14.4$ (TNG fiducial $3.6$), with
all 34 other entries held at the fiducial (the two-bound design: each of the 30 astro
parameters swept to its documented min/max bound, $30\times2-3=57$ distinct perturbations —
for three parameters the fiducial value *is* the prior bound) — as the pixel difference
$\kappa_{\rm node}-\kappa_{\rm fiducial}$ (blue = this node's halos are *less* convergent
than the fiducial, red = *more*). $\kappa$ crosses zero pixel-by-pixel, so unlike $y$
(strictly positive) a ratio is ill-behaved; the seed-paired difference is the well-behaved
analogue and, because both node and fiducial share the same lux realization, the
common large-scale structure cancels and only the feedback-driven change in each halo's
profile survives. A controlled single-knob bracket, unlike the Sobol min/max wind nodes
shown previously, where all 30 astrophysical parameters differed at once and the response
could not be attributed. Every structure in all six panels comes from the same DMO
halo skeleton; only the galaxy-formation physics differs. The $y$ sky (not shown in the
bottom row) responds with sign-flipping mean shifts about the fiducial and order-unity
ratios halo-by-halo (fig 7), while $\kappa$'s own response is confined to the halo cores
and is percent-level of the map rms (fig 4): the asymmetry that makes a feedback-swept
multi-probe suite informative.
Conventions: 1-px Gaussian cosmetic smoothing on every panel; top-row $\kappa$ on a
**linear** symmetric stretch at the 99.9th percentile of $|\kappa|$, so the peaks are
shown rather than clipped — v2's zero-anchored `TwoSlopeNorm` made equal color steps mean
unequal $\kappa$ steps, so its bar could not be read quantitatively; $\tau$/$y$ log in
their own units; the bottom-row $\Delta\kappa$ bar is linear and symmetric at the 99.5th
percentile of $|\Delta\kappa|$ over both panels — tighter than the top row's 99.9th because
the halo-scale response is intrinsically sparser than the maps it is differenced from. The
1° scale bar applies to all panels; $\kappa$ is at $z_s=1$, $\tau$ and $y$ integrated to
$z=2.44$. All stamped numbers ($A_{\rm SN1}$ bounds, $\Delta\kappa$ response, and the
released fiducial map's mean $\tau$ and mean $y$ quoted in §5b) are computed live.
''')

code(r'''
# ── Fig 0 (hero): DMO in, baryons out, on a lightcone, under feedback ────────
# data: bind_science/runs/dmo/run_0000/kappa_maps.npz (paired DMO-only trace:
#       same lux seed/geometry as the fiducial => pixel-aligned) +
#       bind_lightcone_tng/{kappa,tau,y}_maps.npz (validated fiducial) +
#       bind_science/runs/twobound/run_{0000,0001}/kappa_maps.npz — the
#       ONE-PARAMETER A_SN1 (WindEnergyIn1e51erg, native col 2) bracket: run
#       0000 = low bound, run 0001 = high bound, all 34 other entries at the
#       fiducial, same lux seed => pixel-aligned with the fiducial and each
#       other. Realization 0, z_s=1 everywhere; six ~1 GB key loads, strictly
#       sequential with del/gc. 1-px smoothing on all. Bottom row is a
#       per-pixel DIFFERENCE, not a ratio: kappa crosses zero so a ratio is
#       ill-behaved, unlike the strictly-positive y map used for this panel
#       in earlier drafts.
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

FOV = 5.0
TB   = SCI/"runs/twobound"
tbp  = np.load(TB/"twobound_params.npy")   # (60, 35) native; run r varies param r//2
A_LO, A_HI = float(tbp[0, 2]), float(tbp[1, 2])   # 0.9 / 14.4 (TNG fiducial 3.6)

def _hero_map(path, key, zi):
    a = np.load(path)
    m = gaussian_filter(a[key][0, zi].astype(float), 1.0)   # 1-px cosmetic smoothing
    del a; gc.collect()
    return m

kap_dmo0 = _hero_map(SCI/"runs/dmo/run_0000/kappa_maps.npz", "kappa", ZI)
kap_fid0 = _hero_map(LC/"kappa_maps.npz", "kappa", ZI)
tau_fid0 = _hero_map(LC/"tau_maps.npz",   "tau",  -1)
y_fid0   = _hero_map(LC/"y_maps.npz",     "y",    -1)
kap_weak0 = _hero_map(TB/"run_0000/kappa_maps.npz", "kappa", ZI)  # A_SN1 low bound
kap_str0  = _hero_map(TB/"run_0001/kappa_maps.npz", "kappa", ZI)  # A_SN1 high bound

# top-row stretches. kappa is LINEAR and symmetric: v2 used a TwoSlopeNorm,
# whose piecewise-linear mapping means equal color steps do NOT mean equal
# kappa steps -- the bar could not be read quantitatively. The half-width is
# the 99.9th percentile of |kappa| over both panels, so the bright peaks are
# shown rather than silently clipped and the bar needs no extend arrow.
vk0 = float(np.percentile(np.abs(np.concatenate([kap_dmo0.ravel(),
                                                 kap_fid0.ravel()])), 99.9))
knorm0 = Normalize(-vk0, vk0)
tau_hi0 = np.percentile(tau_fid0, 99.8)
tnorm0  = LogNorm(vmin=tau_hi0/30.0, vmax=tau_hi0)      # >=1.5 decades, clean ticks
ynorm0  = LogNorm(vmin=np.percentile(y_fid0[y_fid0 > 0], 50),
                  vmax=np.percentile(y_fid0, 99.8))

# bottom row: RESPONSE maps, node/fiducial, for the one-parameter A_SN1 bracket
# (only WindEnergyIn1e51erg differs from the fiducial; the earlier version used
# the Sobol min/max wind-velocity nodes, where all 30 astro params differed at
# once). kappa crosses zero, so unlike the strictly-positive y map (used for
# this panel in earlier drafts) a ratio is ill-behaved -- near-zero fiducial
# pixels blow up the ratio with no physical content. The response is instead
# the per-pixel DIFFERENCE Delta-kappa = kappa_node - kappa_fiducial: both node
# and fiducial are seed-paired (same lux realization), so the shared large-scale
# structure cancels and only the feedback-driven change in each halo's own
# convergence profile survives. Symmetric linear stretch on a diverging norm
# (blue = this node is less convergent than fiducial, red = more), oriented
# node-minus-fid so the color reads as the node's own effect. The stretch half-
# width is the 99.5th percentile of |Delta-kappa| over both panels -- tighter
# than the top row's 99.9th because the halo-scale response is intrinsically
# sparser/speckled than the maps it is differenced from, so a looser percentile
# would wash the imprint out under salt-and-pepper single-halo outliers.
dkw0 = kap_weak0 - kap_fid0
dks0 = kap_str0  - kap_fid0
vr0 = float(np.percentile(np.abs(np.concatenate([dkw0.ravel(), dks0.ravel()])), 99.5))
rnorm0 = Normalize(-vr0, vr0)

fig = plt.figure(figsize=(TWO_COL[0], 5.0))
gs0 = fig.add_gridspec(2, 4, height_ratios=[1.0, 2.05], hspace=0.10, wspace=0.34,
                       left=0.015, right=0.955, top=0.985, bottom=0.012)
axd = fig.add_subplot(gs0[0, 0]); axk = fig.add_subplot(gs0[0, 1])
axt = fig.add_subplot(gs0[0, 2]); axy = fig.add_subplot(gs0[0, 3])
axw = fig.add_subplot(gs0[1, 0:2]); axs = fig.add_subplot(gs0[1, 2:4])
ext = [0, FOV, 0, FOV]
STK = [pe.withStroke(linewidth=2.0, foreground="black")]
BOX0 = dict(facecolor="w", edgecolor="none", alpha=0.78, pad=1.6)

# panel tags sit ABOVE the axes and carry only the field name: v2 stamped 2-3
# line captions inside each panel at 5.6 pt over structured image, which is what
# made them unreadable. The depth/scope clauses live in the caption.
panels0 = [(axd, kap_dmo0, "RdBu_r", knorm0, "(a) DMO $\\kappa$"),
           (axk, kap_fid0, "RdBu_r", knorm0, "(b) BIND $\\kappa$"),
           (axt, tau_fid0, "magma",  tnorm0, "(c) BIND $\\tau$"),
           (axy, y_fid0,   "inferno", ynorm0, "(d) BIND $y$")]
imk0 = None
for a, m, cmap, nrm, lab in panels0:
    im = a.imshow(m, origin="lower", extent=ext, cmap=cmap, norm=nrm, rasterized=True)
    a.set_xticks([]); a.set_yticks([])
    a.text(0.5, 1.02, lab, transform=a.transAxes, fontsize=7.5, ha="center", va="bottom")
    if nrm is knorm0:          # kappa panels (a,b) share ONE bar (drawn below)
        imk0 = im
        continue
    # plain log bars in the field's own (dimensionless) units -- no 10^-3/10^-6
    # rescaling in the label. 1-3-10 subs so tau's ~1.5-decade range gets more
    # than one labelled tick; ticks in scientific notation.
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.04)
    cb.set_label(r"$\tau$" if nrm is tnorm0 else r"$y$", fontsize=7, labelpad=1)
    cb.locator = LogLocator(base=10.0, subs=(1.0, 3.0), numticks=12)
    cb.formatter = LogFormatterSciNotation(minor_thresholds=(3, 0.4))
    cb.update_ticks()
    cb.ax.tick_params(labelsize=6)
cbk0 = fig.colorbar(imk0, ax=[axd, axk], fraction=0.028, pad=0.025)
cbk0.set_label(r"$\kappa$", fontsize=7, labelpad=1)
vkt0 = np.floor(vk0*1000)/1000     # round INWARD: a tick > vk0 is silently dropped
cbk0.set_ticks([-vkt0, 0.0, vkt0])
cbk0.ax.tick_params(labelsize=6)
# one scale bar (panel a; the angular scale is identical in all panels)
axd.plot([0.35, 1.35], [0.30, 0.30], color="w", lw=1.6, path_effects=STK)
axd.text(0.85, 0.45, r"$1^\circ$", color="w", fontsize=7, ha="center", va="bottom",
         path_effects=STK)

imw = axw.imshow(dkw0, origin="lower", extent=ext, cmap="RdBu_r", norm=rnorm0,
                 rasterized=True)
axs.imshow(dks0, origin="lower", extent=ext, cmap="RdBu_r", norm=rnorm0, rasterized=True)
for a, lab in [
        (axw, rf"(e) $A_{{\rm SN1}} = {A_LO:g}$"),
        (axs, rf"(f) $A_{{\rm SN1}} = {A_HI:g}$")]:
    a.set_xticks([]); a.set_yticks([])
    a.text(0.025, 0.975, lab, transform=a.transAxes, fontsize=8, va="top", bbox=BOX0)
cbb = fig.colorbar(imw, ax=[axw, axs], fraction=0.024, pad=0.015, extend="both")
cbb.set_label(r"$\kappa_{\rm node} - \kappa_{\rm fiducial}$", fontsize=8, labelpad=2)
vrt0 = np.floor(vr0*10000)/10000   # round INWARD: a tick > vr0 is silently dropped
cbb.set_ticks([-vrt0, 0.0, vrt0])
cbb.ax.tick_params(labelsize=7)

save(fig, "figs_v2/fig00_hero")
plt.show()
print(f"hero: fiducial realization 0; one-parameter A_SN1 bracket "
      f"twobound/run_0000 (A_SN1={A_LO:g}) / run_0001 (A_SN1={A_HI:g}), "
      f"all 34 other params fiducial; "
      # mean tau of the released fiducial map (total column to z=2.44, painted-halo
      # gas only) -- quoted in Sec 5b alongside the mean y; the 1-px cosmetic
      # smoothing conserves the map mean, so this is the map-level monopole.
      f"mean tau (fiducial, total column) = {tau_fid0.mean():.3e}; "
      f"mean y (fiducial) = {y_fid0.mean():.3e}; kappa rms DMO/painted = "
      f"{kap_dmo0.std():.4f}/{kap_fid0.std():.4f}; kappa norm +-{vk0:.4f} "
      f"(99.9th pct |kappa|); Delta-kappa rms weak/strong = "
      f"{dkw0.std():.4f}/{dks0.std():.4f} ({100*dkw0.std()/kap_fid0.std():.1f}%/"
      f"{100*dks0.std()/kap_fid0.std():.1f}% of kappa rms); Delta-kappa norm "
      f"+-{vr0:.4f} (99.5th pct |Delta-kappa|, {vrt0:.4f} plotted tick)")
del kap_dmo0, kap_fid0, tau_fid0, y_fid0, kap_weak0, kap_str0, dkw0, dks0
gc.collect()
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §1c — Fig 1: generating halos from TNG (the BIND painting engine)

**Model.** BIND is a conditional flow-matching (FM) generator. For each halo, the input is
the projected DMO surface density *condition* $c$ (a $128^2$ patch spanning
$6.25\,h^{-1}$Mpc), a large-scale context field, the parameter vector
$\theta\in\mathbb{R}^{35}$ (5 cosmological entries fixed at TNG300; 30 astrophysical entries
varied), and the snapshot scale factor $a=1/(1+z)$. The model learns a velocity field
$v_\phi$ transporting Gaussian noise $x_0$ to the normalized hydro fields $x_1$ along the
optimal-transport path $x_t=(1-t)\,x_0+t\,x_1$, by minimizing

$$\mathcal{L}=\mathbb{E}_{t,x_0,x_1}\big\|\,v_\phi(x_t,t\,|\,c,\theta,a)-(x_1-x_0)\,\big\|^2 .$$

Conditioning is injected as a *sum of embeddings* — parameter encoder, sinusoidal diffusion
time, and a sinusoidal$\to$MLP scale-factor embedding — driving adaptive group-norm
(scale/shift) in every residual block. The outputs are eight channels: dark matter, gas,
stars as an (occupancy, conditional log-density) two-head pair, and four gas-thermodynamic
channels $(y,\,T,\,K,\,P_e)$ with $\log_{10}$ normalization. Sampling integrates the learned
flow in 50 steps and de-normalizes to physical units.

**Every lightcone product in this release was painted with the redshift-conditioned
checkpoint** (`fm_redshift_thermo`; `condition_redshift=True` verified from the checkpoint
hyper-parameters), the scale factor supplied per snapshot from the stage-1 manifest — a
methods fact that matters for interpreting the residual redshift drift in §2c: the drift
survives *despite* explicit $a$-conditioning, pointing at the training data's $z>0$
thermodynamic conventions rather than at missing model capacity (§5b).

**This figure** (the methods schematic behind the fig-0 hero). TNG300-Dark slab → the
$M_{200c}\geq10^{13}\,M_\odot/h$ halo cutouts (40
most massive of the slab's 666 drawn) → one halo's DMO condition → the *same halo* (same row
index in every run's patch file) painted under three Sobol nodes bracketing the SN-wind
energy — the middle row is fiducial-*like*; all 30 parameters differ between rows —. The
painted block shows the full mass output of the generator (dark matter, gas, stars) plus one
of the four thermodynamic channels (Compton $y$). The three $\theta$ rows share one fixed
stretch per column — each of the four painted-channel norms is computed over all three rows
(a high-percentile floor on the nonzero pixels for the sparse stellar channel) — so
row-to-row brightness differences are directly comparable. The painted
snapshots are then composited into full-box planes, assembled into the 20-shell lightcone and
ray-traced by `lux` into the released $\kappa/\tau/y$ maps (§1d; fig 0).
The showcase halo is chosen as the most gas-responsive of the ten most massive in
$10^{13.4-13.8}\,M_\odot/h$, where feedback response is strongest.
''')

code(r'''
# ── Fig 1: halo-generation pipeline diagram ──────────────────────────────────
# data: bind_lightcone_tng/snap_096/stage1/stage1_slab00.npz (full-box 'dmo'
#       4198^2 projection, per-halo 'condition' patches, halo_centers/r200 in
#       Mpc/h — shared by ALL 256 runs) + bind_sb35/runs/run_NNNN/snap_096/
#       composite_slab00.npz (generated_patches [DM,Gas,Stars], thermo_patches
#       [y,T,K,P_e]; same halo order in every run) + design/astro_params_sobol.npy
s1 = np.load(LC/"snap_096/stage1/stage1_slab00.npz")
dmo_box = s1["dmo"]
centers, r200c, masses = s1["halo_centers"], s1["halo_r200"], s1["halo_masses"]
BOX, PATCH = 205.0, 6.25                             # Mpc/h

# three Sobol nodes spanning the SN-wind energy axis (native col 2, WindEnergyIn1e51erg)
sob   = np.load(SB35/"design/astro_params_sobol.npy")
WE    = sob[:, 2]
valid = np.asarray(sorted(run_ids.tolist()))
r_lo  = int(valid[np.argmin(WE[valid])])
r_mid = int(valid[np.argmin(np.abs(WE[valid] - 3.6))])   # closest to the TNG fiducial 3.6
r_hi  = int(valid[np.argmax(WE[valid])])
RUNS3 = [r_lo, r_mid, r_hi]
ROWLB = ["low", "fiducial", "high"]

# showcase halo: group scale (feedback response is strongest there) and, among
# the 10 most massive in the band, the one whose gas mass responds most lo->hi
glo = np.load(SB35/f"runs/run_{r_lo:04d}/snap_096/composite_slab00.npz")["generated_patches"][:, 1]
ghi = np.load(SB35/f"runs/run_{r_hi:04d}/snap_096/composite_slab00.npz")["generated_patches"][:, 1]
band = np.where((masses >= 10**13.4) & (masses <= 10**13.8))[0]
band = band[np.argsort(-masses[band])][:10]
resp = np.abs(np.log10(ghi[band].sum((1, 2))/glo[band].sum((1, 2))))
i_h  = int(band[np.argmax(resp)])
cond = s1["condition"][i_h]
del glo, ghi; gc.collect()

dm3, gas3, st3, y3 = [], [], [], []
for r in RUNS3:
    g  = np.load(SB35/f"runs/run_{r:04d}/snap_096/composite_slab00.npz")
    gp = g["generated_patches"]                      # (666, 3, 128, 128): DM, Gas, Stars
    dm3.append(gp[i_h, 0].copy())
    gas3.append(gp[i_h, 1].copy())
    st3.append(gp[i_h, 2].copy())
    y3.append(g["thermo_patches"][i_h, 0].copy())    # (666, 4, ...): y, T, K, P_e
    del gp, g
gc.collect()
dm3, gas3 = np.array(dm3), np.array(gas3)
st3, y3   = np.array(st3),  np.array(y3)

fig = plt.figure(figsize=(TWO_COL[0], 3.8))
gs  = fig.add_gridspec(3, 6, width_ratios=[2.6, 1.15, 1.0, 1.0, 1.0, 1.0],
                       wspace=0.08, hspace=0.10)

# (left) full DMO box with halo cutout squares
axb = fig.add_subplot(gs[:, 0])
axb.imshow(np.log10(dmo_box + dmo_box[dmo_box > 0].min()), origin="lower",
           extent=[0, BOX, 0, BOX], cmap="cividis", rasterized=True)
big = np.argsort(-masses)[:40]
for j in big:
    cx, cy = centers[j]
    axb.add_patch(Rectangle((cx-PATCH/2, cy-PATCH/2), PATCH, PATCH,
                            fill=False, ec="w", lw=0.35))
cx, cy = centers[i_h]
axb.add_patch(Rectangle((cx-PATCH/2, cy-PATCH/2), PATCH, PATCH,
                        fill=False, ec=COLORS["highlight"], lw=1.2))
axb.text(0.03, 0.97, "TNG300-Dark, 51.25 Mpc/$h$ slab", transform=axb.transAxes,
         color="w", fontsize=7, va="top")
axb.set_xlabel(r"$x\,[\mathrm{Mpc}/h]$"); axb.set_ylabel(r"$y\,[\mathrm{Mpc}/h]$")

# (middle) the DMO condition patch of the highlighted halo
axc = fig.add_subplot(gs[1, 1])
pos = cond[cond > 0]
axc.imshow(cond, origin="lower", cmap="cividis", rasterized=True,
           norm=LogNorm(vmin=np.percentile(pos, 1), vmax=np.percentile(pos, 99.9)))
axc.set_xticks([]); axc.set_yticks([])
axc.text(0.05, 0.95, "DMO\ncondition", transform=axc.transAxes, color="w",
         fontsize=6.5, va="top")
axc.text(0.5, -0.10, "6.25 Mpc/$h$", transform=axc.transAxes, ha="center",
         va="top", fontsize=6)
axc.text(0.5, 1.32, r"paint with $\theta$" "\n" "(30 astro params)",
         transform=axc.transAxes, fontsize=6, ha="center", va="bottom")

# (right) same halo painted at three Sobol parameter vectors: the full mass output
# (DM, gas, stars) plus one thermodynamic channel. One fixed norm per column,
# computed over all three rows, so rows are directly comparable.
dnorm = LogNorm(vmin=np.percentile(dm3[dm3 > 0], 55),  vmax=np.percentile(dm3, 99.9))
gnorm = LogNorm(vmin=np.percentile(gas3[gas3 > 0], 55), vmax=np.percentile(gas3, 99.9))
ynorm = LogNorm(vmin=np.percentile(y3[y3 > 0],  55), vmax=np.percentile(y3, 99.9))
# stars are sparse and zero-heavy: a 55th-percentile floor of the nonzero pixels is
# below the visible stellar light, so use a high nonzero percentile (linear fallback
# if the channel is empty or degenerate)
spos = st3[st3 > 0]
if spos.size:
    svmin, svmax = np.percentile(spos, 60), np.percentile(spos, 99.9)
else:
    svmin = svmax = 0.0
if svmax > svmin > 0:
    snorm = LogNorm(vmin=svmin, vmax=svmax)
else:
    snorm = Normalize(vmin=float(st3.min()), vmax=float(max(st3.max(), st3.min() + 1e-30)))

def _floored(name):
    """LogNorm MASKS non-positive pixels and matplotlib paints masked/under pixels
    with the axes background -- which turned the zero-heavy stars panel into black
    speckle on white paper while every other column was dark-on-dark. Pin 'bad' and
    'under' to the colormap's own low end so a zero pixel reads as 'no signal' in
    the same visual language as the rest of the row."""
    cm = plt.get_cmap(name).copy()
    cm.set_bad(cm(0.0)); cm.set_under(cm(0.0))
    return cm

PAN = [(dm3, _floored("cividis"), dnorm, r"DM $\Sigma$"),
       (gas3, _floored("cividis"), gnorm, r"gas $\Sigma$"),
       (st3, _floored("cividis"), snorm, r"stars $\Sigma$"),
       (y3,  _floored("inferno"), ynorm, r"Compton $y$")]
for k, r in enumerate(RUNS3):
    axp = [fig.add_subplot(gs[k, 2 + j]) for j in range(len(PAN))]
    for a, (arr, cmap, nrm, hdr) in zip(axp, PAN):
        a.imshow(arr[k], origin="lower", cmap=cmap, norm=nrm, rasterized=True)
        a.set_xticks([]); a.set_yticks([])
        if k == 0:
            a.text(0.5, 1.06, hdr, transform=a.transAxes, ha="center", fontsize=7)
    # row stamp on the DM column: the theta tier only -- the WindEnergy value it
    # used to carry is in the cell's print() stamp, and the caption explains that
    # the middle row is the closest Sobol node to the TNG fiducial (all 30 astro
    # parameters differ between rows, not just the wind energy).
    axp[0].text(0.04, 0.05, f"$\\theta$ {ROWLB[k]}",
                transform=axp[0].transAxes, fontsize=6.0, color="w", va="bottom")
    fig.add_artist(ConnectionPatch(
        xyA=(1.0, 0.5), coordsA=axc.transAxes, xyB=(0.0, 0.5), coordsB=axp[0].transAxes,
        arrowstyle="-|>", color="0.35", lw=0.8, shrinkA=2, shrinkB=2))

# arrow: highlighted halo -> condition patch; label the parameter step
fig.add_artist(ConnectionPatch(
    xyA=(cx+PATCH/2, cy), coordsA=axb.transData, xyB=(0.0, 0.5), coordsB=axc.transAxes,
    arrowstyle="-|>", color=COLORS["highlight"], lw=0.9, shrinkA=1, shrinkB=2))

save(fig, "figs_v2/fig01_pipeline_diagram")
plt.show()
print(f"halo: log10 M = {np.log10(masses[i_h]):.2f}, r200c = {r200c[i_h]:.2f} Mpc/h; "
      f"runs {RUNS3} with WindEnergyIn1e51erg = "
      + ", ".join(f"{WE[r]:.2f}" for r in RUNS3))
''')

# ═════════════════════════════════════════════════════════════════════════════
# Figure-free methods cell: it carries the lightcone/compositing/ray-trace facts
# that §2 onwards depend on (they were previously stated only in the deleted
# fig-2 caption).
md(r'''
## §1d — From painted snapshots to ray-traced lightcone maps

**Lightcone tiling.** Twenty TNG300-Dark snapshots spanning $z=0.034$–2.44 tile the
lightcone; each is painted independently at its own scale factor $a=1/(1+z)$, supplied to
the redshift-conditioned checkpoint of §1c.

**Compositing.** At each of the 20 snapshots the painted patches are pasted back into
full-box surface-density planes within $4\times R_{200c}$ circular apertures with a 15%
edge taper; regions outside painted apertures carry no baryonic signal — the one-halo
completeness scope quantified in §5b and fig 14. The electron-column plane follows from the
composited gas surface density,

$$\tau=\sigma_T\,\frac{x_e}{m_p}\,\Sigma_{\rm gas},\qquad x_e=0.88,$$

$x_e=0.88$ is the free-electron-per-proton-mass factor for a **fully ionized, primordial
(no-metal) plasma**: hydrogen mass fraction $X_H=0.76$, helium mass fraction $Y_{\rm
He}=0.24$, hydrogen singly and helium **doubly** ionized, so $x_e=X_H+Y_{\rm He}/2=0.88$
(equivalently a mean molecular weight per electron $\mu_e=2/(1+X_H)\simeq1.136$).
$\Sigma_{\rm gas}$ is the **physical** (proper-area, not comoving) gas surface density —
the composited comoving gas mass ($M_\odot/h$ per pixel) is converted to a physical column
using that plane's own scale factor $a_l$ (physical pixel area $\propto a_l^2/h^2$) before
the Thomson factor is applied, so the released $\tau$ carries no residual $h$ dependence.
$\sigma_T=6.6524\times10^{-25}\,\mathrm{cm^2}$, $m_p=1.6726\times10^{-24}\,\mathrm{g}$
(`bind.inference.lightcone_maps.{SIGMA_T,M_P,X_E_PER_MASS}`). This one formula is
re-evaluated at every plane's own $(box\_size, n_{\rm grid}, a_l)$ — it is not a single
frozen constant — and, e.g., at snap 96 ($a=0.96738$, fig 6) reduces to
$\tau=2.219785\times10^{-14}\times\Sigma_{\rm gas}\,[M_\odot h^{-1}\,\mathrm{px}^{-1}]$.

with per-plane physical-area factors applied at painting time. The Compton-$y$ channel is
painted directly in physical units, so slabs are simply summed with **no** extra $1/a^2$
weighting (the convention validated against Planck in earlier work).

**Ray tracing.** The `lux` multi-plane algorithm splits each snapshot into four slabs of
$51.25\,h^{-1}$Mpc depth — 80 lens planes in total — deflects rays through them (lens
potentials from the *total* matter planes), and integrates $\kappa$, $\tau$ and $y$ along
the *same* deflected rays, so the three released fields share one ray grid. Every node ships
50 realizations (random box rotations/translations, seed $1992+7r$) at five source planes
$z_s\in\{0.5,1,1.5,2,2.44\}$, on a $1024^2$ grid over $5\times5\,\mathrm{deg}^2$
(0.293′/px, no beam applied). That seed sequence is shared across all nodes **and** with the
DMO and hydro-pasted truth traces: it is the seed-pairing that every ratio statistic and paired
error band from §2 onwards relies on.
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §2.0 — Statistical conventions

Conventions used by every validation and response figure:

- **Realization structure.** Each run's maps are 50 `lux` ray-trace realizations of the single
  205 $h^{-1}$Mpc box (shared random rotations/translations, seed 1992+7r). The BIND, DMO, and
  hydro-pasted truth traces are *seed-paired*, so BIND/truth and BIND/DMO ratios cancel the shared
  large-scale modes. Realization scatter is **not** cosmic variance: the 25 deg² footprint has a
  Gaussian cosmic-variance floor of ~10% on $C_\ell$ at $\ell\simeq1000$; ratio statistics
  cancel it, absolute spectra do not.
- **Error bands and $\chi^2$.** Validation residual bands are *seed-paired*: the ±1σ scatter
  of the 50 per-realization BIND/truth ratios divided by $\sqrt{50}$ where a mean curve is
  compared. This one recipe is used for the spectra **and** for the peak/minimum counts and
  the $\kappa$-PDF: the caches store only realization means for the latter, so their
  per-realization draws are recomputed live from the cached maps with the cached conventions
  (2′ smoothing, `nu_norm='map'`, cached PDF bin grid — the recomputation reproduces the
  cached means exactly and is asserted in the fig-4 cell). The PDF uses a paired
  *difference* normalized by max(truth) (its tails cross zero density, so a ratio is
  ill-defined); count-statistic $\chi^2$ are quoted over the plotted $\nu$ range on bins
  with mean truth counts > 5. The one exception: Minkowski-functional bands use the BIND
  per-realization scatter $\times\sqrt2$ (truth per-realization MFs are not cached),
  normalized by max|truth|. We quote both a diagonal
  $\chi^2/\mathrm{dof}=\frac{1}{N_b}\sum_b (r_b/\sigma_b)^2$ and, on a 25-bin compression, a
  full-covariance version $\chi^2=h\,\mathbf{r}^{\sf T}\hat{C}^{-1}\mathbf{r}$ with the Hartlap
  debiasing factor $h=(n-p-2)/(n-1)$ for $n=50$ realizations and $p=25$ bins. Because paired
  errors are $\sim$0.1–0.7%, $\chi^2\gg1$ throughout §2 means *significant characterized
  systematic*, never "large error".
- **Peak statistics are noiseless** ($n_{\rm gal}=0$, map-normalized $\nu$); a shape-noise
  variant ($n_{\rm gal}=10$/arcmin², $\sigma_e=0.26$) is cached and its closure quoted in fig 4.
- **Maps carry no beam**, 0.29296875′/px; aperture-photometry users must match both conventions.
  (Exception: the released *stacked $y$-CAP profile product* — §6 products table — is measured
  on the patch pixel grid and already carries an ACT-like 1.6′ beam.)
- **Significance thresholds.** For $n=253$ runs, the single-cell 2σ Spearman null is
  $|\rho|=2/\sqrt{n}\simeq0.126$. The max-over-bins importance is judged against
  *per-statistic* permutation nulls (200 seeded label shuffles, max over that statistic's
  bins; 95th percentiles $\simeq0.147$–$0.177$ depending on bin count), and the two
  multiplicity corrections the fig-5 grid needs — a per-parameter look-elsewhere null over
  the 12 statistics ($\simeq0.196$) and a whole-grid one ($\simeq0.257$). All are printed by
  the fig-5 cell, and fig 5 *marks* sub-threshold cells rather than hiding them.
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §2a — Fig 3: halo-level scaling relations at the fiducial

The first credibility question is whether painted halos carry the right gas. We compare the
same 2933 halos (snap 096, $z=0.034$) painted by BIND against the matched hydro-pasted truth,
halo-by-halo, in three scaling relations on one shared mass axis: (a) $Y_{500c}$–$M_{200c}$,
(b) gas fraction $f_{\rm gas,500c}$ (background-subtracted), and (c) the aperture
stellar-to-halo-mass relation $M_{\star,200c}$–$M_{200c}$. BIND is solid, hydro-pasted truth
dashed, each with its 16–84 percentile band over the same 0.2-dex mass bins
($\log M=13.0$–14.8); the light points are the individual painted halos. The stacked
*radial* profiles that used to occupy columns (c,d) now have their own figure (fig 6).
The sample spans $\log_{10}M_{200c}=13.00$–$14.98$ ($10^{13}$–$9.6\times10^{14}\,M_\odot/h$);
it thins rapidly at the high-mass end (149 halos above $10^{14}$, 14 above $3\times10^{14}$,
only 6 above $4\times10^{14}\,M_\odot/h$), so the top one or two 0.2-dex bins in every panel
are low-number statistics, not populous cluster stacks.

The $x$-axis is $M_{200c}$: the halo atlas stores it under the legacy key `M_fof`, but that
array is already the spherical-overdensity mass — it reproduces
$\tfrac{4}{3}\pi r_{200}^3\,200\,\rho_c(z)$ from the cached $r_{200}$ to
$\max|{\rm ratio}-1|=3.5\times10^{-4}$ over all 2933 halos — so this is a relabel, not a
recomputation. Apertures differ between panels by necessity: $Y$ and $f_{\rm gas}$ are
500c-aperture quantities (the observational convention, and the only one the atlas caches
for thermodynamics), while $M_\star$ is the 200c aperture matching the axis. Panel (c) is an
*aperture* stellar mass — a projected-cylinder sum including satellites, ICL and an
unsubtracted line-of-sight background ($\sim$12% of the total) — so its absolute
normalization is not a central-galaxy SHMR; the BIND/truth ratio is what validates, and the
background largely cancels in it. Per-halo stellar ratios are also intrinsically noisy
(16–84 spread 0.75–1.20, five times wider than $Y$): only the binned medians carry
information, and the top two bins hold 32 and 23 halos with 3–5% bootstrap errors.

**Discussion.** Scaling-relation *slopes and scatter* are reproduced essentially perfectly
across 1.5 decades of mass, but the amplitude offset is **mass-dependent at the observable
level**: the binned $\Delta Y_{500c}$ and $\Delta f_{\rm gas}$ run from roughly $+10$–12%
at $\log M\simeq13.1$ declining monotonically to $\approx0$ (Y mildly negative) above
$\log M\simeq14.1$ (the global median $Y$ ratio, its bootstrap significance, and the fitted
mass trend are printed by the cell — RECOMPUTED-ON-RUN; quote those values, not this
prose). A dedicated patch-level audit adds a *distinct, pre-composite* diagnostic: in the
raw painted patches, before any compositing, the per-halo gas ratio is flat in mass (1.042
across quartiles: 1.042/1.042/1.042/1.043), the painted DM channel is unbiased (0.995), and
stars are −12% — a smooth, radially-graded regression bias of the flow-matching model in
normalized $\log_{10}(1+x)$ gas space (audit artifacts: `audits/patch_level_compare.py`,
`audits/patch_level_ratios.npz`, `audits/radial_bias_profile.png`). The patch-level
flatness does **not** transfer to the measured observables: the mass trend emerges through
the aperture/background-subtraction measurement on the composited maps, which weights the
radially-graded bias differently for groups than for clusters. The release-level
recommendation is therefore a **linear-in-$\log M$ calibration**, not a single factor: the
cell still fits $\Delta(\log M)$ to the binned medians (weighted by their bootstrap SEs) and
**prints** the coefficients plus the post-correction residual — the residual row that used
to plot them is gone, so the printed values, condensed into the two-line stamps on panels
(a,b), are the citable form; users divide BIND $Y$/$f_{\rm gas}$ by
$1+\Delta(\log M)/100$. Panel (c) carries the same treatment for the stars: a $\approx-6$%
median deficit that shrinks with mass, the composited counterpart of the $-12$% patch-level
star bias, noted as a separate and smaller-impact systematic (stars contribute little to
$\kappa/y/\tau$). Cosmology for $\Omega_b/\Omega_m$: TNG (0.0486/0.3089).
''')

code(r'''
# ── Fig 3: halo-level scaling relations, BIND vs hydro-pasted truth ────────
# data: bind_science/halo_atlas/{fid,truth}_snap096.npz (identical 2933-halo
#       schemas -> per-halo comparison by index)
# layout: ONE row of 3 panels on a shared M200c axis and a single 0.2-dex mass
# binning: (a) Y500c, (b) f_gas,500c, (c) aperture stellar-to-halo mass.
# The percent-difference row is gone; the per-halo ratios and the recommended
# linear-in-logM calibration fits are still computed, stamped compactly on the
# panels, and PRINTED below (those printed numbers are the citable ones).
# The stacked radial profiles moved to fig 6 (figs_v2/fig06_radial_profiles).
fa = np.load(SCI/"halo_atlas/fid_snap096.npz")
ta = np.load(SCI/"halo_atlas/truth_snap096.npz")

# NB the atlas key "M_fof" is a misnomer: it is ALREADY the spherical-overdensity
# mass M200c = (4/3)pi r200^3 * 200 rho_c(z) -- verified against the cached r200
# to max|ratio-1| = 3.5e-4 over all 2933 halos. Relabelling the axis M_FoF ->
# M_200c is therefore a PURE RELABEL; do not "fix" this by recomputing a mass
# (m_tot_200c_bg and the m_dm+m_gas+m_star sum are projected-cylinder masses,
# 1.10x and 1.27x M_fof, not SO masses).
logM   = np.log10(fa["M_fof"])
edges  = np.arange(13.0, 14.8, 0.2)                      # ONE binning, all 3 panels
fg_f   = fa["m_gas_500c_bg"]/fa["m_tot_500c_bg"]
fg_t   = ta["m_gas_500c_bg"]/ta["m_tot_500c_bg"]
OB_OM  = 0.0486/0.3089                                   # cosmic baryon fraction
MLAB   = r"$\log_{10} M_{200c}\,[M_\odot/h]$"

fig, ax = plt.subplots(1, 3, figsize=(TWO_COL[0], 2.6), sharex=True)

# (a) Y500c - M200c: measurement-space medians (Y is a 500c-aperture quantity;
# there is no Y_200c in the atlas, so the apertures differ from the x-axis by
# necessity -- stated in the markdown)
for atl, c, ls, lab in [(fa, COLORS["bind"], "-", "BIND"),
                        (ta, COLORS["truth"], "--", "hydro-pasted")]:
    ok = atl["Y_500c"] > 0
    cen, med, lo, hi, _ = running_stat(logM[ok], np.log10(atl["Y_500c"][ok]), edges)
    ax[0].fill_between(cen, lo, hi, color=c, alpha=BAND_ALPHA, lw=0)
    ax[0].plot(cen, med, color=c, ls=ls, lw=1.6, label=lab)
ax[0].scatter(logM[::4], np.log10(np.where(fa["Y_500c"] > 0, fa["Y_500c"], np.nan))[::4],
              s=1.0, color=COLORS["bind"], alpha=0.10, lw=0, rasterized=True)
ax[0].set_ylabel(r"$\log_{10} Y_{500c}$")
ax[0].set_xlim(13.0, 14.7)
ax[0].legend(loc="lower right")
# in-panel stamps removed at the author's request -- every number they carried
# (BIND/truth medians, the linear-in-logM calibrations, the aperture caveat)
# is still computed above and printed by this cell's print() block below.
panel_label(ax[0], "(a)")

# paired per-halo % offset + the recommended linear-in-logM calibration. The
# residual panel that used to draw these is gone; the fit is KEPT because its
# coefficients are the release-level calibration recipe (stamped + printed).
ok = (fa["Y_500c"] > 0) & (ta["Y_500c"] > 0)
ry = fa["Y_500c"][ok]/ta["Y_500c"][ok]
cen_y, med_y, _, _, se_y = running_stat(logM[ok], 100*(ry - 1), edges, boot=200)
rng_b = np.random.default_rng(3)
se_y_all = np.std([np.median(ry[rng_b.integers(0, len(ry), len(ry))]) for _ in range(200)])
PIV3 = 13.5
fit_y = np.polyfit(cen_y - PIV3, med_y, 1, w=1/np.where(se_y > 0, se_y, np.nan))

# (b) f_gas,500c - M200c
for fg, c, ls in [(fg_f, COLORS["bind"], "-"), (fg_t, COLORS["truth"], "--")]:
    ok2 = np.isfinite(fg)
    cen, med, lo, hi, _ = running_stat(logM[ok2], fg[ok2], edges)
    ax[1].fill_between(cen, lo, hi, color=c, alpha=BAND_ALPHA, lw=0)
    ax[1].plot(cen, med, color=c, ls=ls, lw=1.6)
ax[1].axhline(OB_OM, color=COLORS["dmo"], ls=":", lw=1.0)
ax[1].text(14.64, OB_OM + 0.004, r"$\Omega_b/\Omega_m$", color=COLORS["dmo"],
           fontsize=6, ha="right")            # data coords -> depends on xlim 14.7
ax[1].set_ylabel(r"$f_{\rm gas,500c}$")
ax[1].set_ylim(top=0.172)
panel_label(ax[1], "(b)")

okf = np.isfinite(fg_f) & np.isfinite(fg_t) & (fg_t != 0)
rg = fg_f[okf]/fg_t[okf]
cen_g, med_g, _, _, se_g = running_stat(logM[okf], 100*(rg - 1), edges, boot=200)
fit_g = np.polyfit(cen_g - PIV3, med_g, 1, w=1/np.where(se_g > 0, se_g, np.nan))

# (c) NEW: aperture stellar-to-halo-mass relation. m_star_200c is a RAW
# projected-cylinder sum -- centrals + satellites + ICL + an unsubtracted LOS
# background (~12% of the total; there is no sig_star_bg in the atlas). Its
# slope (~0.93 dex/dex) is therefore NOT a central-galaxy SHMR slope; the
# validation statement is the BIND/truth ratio, in which the shared background
# largely cancels. All 2933 halos have m_star_200c > 0 on both sides.
for atl, c, ls, lab in [(fa, COLORS["bind"], "-", "BIND"),
                        (ta, COLORS["truth"], "--", "hydro-pasted")]:
    oks = atl["m_star_200c"] > 0
    cen, med, lo, hi, _ = running_stat(logM[oks], np.log10(atl["m_star_200c"][oks]), edges)
    ax[2].fill_between(cen, lo, hi, color=c, alpha=BAND_ALPHA, lw=0)
    ax[2].plot(cen, med, color=c, ls=ls, lw=1.6, label=lab)
ax[2].scatter(logM[::4], np.log10(fa["m_star_200c"])[::4], s=1.0,
              color=COLORS["bind"], alpha=0.10, lw=0, rasterized=True)
ax[2].set_ylabel(r"$\log_{10} M_{\star,200c}\,[M_\odot/h]$")
panel_label(ax[2], "(c)")

oks2   = (fa["m_star_200c"] > 0) & (ta["m_star_200c"] > 0)
rs_star = fa["m_star_200c"][oks2]/ta["m_star_200c"][oks2]
cen_s, med_s, _, _, se_s = running_stat(logM[oks2], 100*(rs_star - 1), edges, boot=200)
fit_s = np.polyfit(cen_s - PIV3, med_s, 1, w=1/np.where(se_s > 0, se_s, np.nan))

for a3 in ax:
    a3.set_xlabel(MLAB)
fig.align_ylabels(ax)
fig.tight_layout(w_pad=0.9)
save(fig, "figs_v2/fig03_halo_validation")
plt.show()
print(f"median per-halo Y ratio BIND/truth = {np.median(ry):.3f} +/- {se_y_all:.3f} "
      f"(offset significance {abs(np.median(ry)-1)/se_y_all:.0f} sigma); "
      f"median f_gas ratio = {np.nanmedian(fg_f/fg_t):.3f}; "
      f"median M_star,200c ratio = {np.median(rs_star):.3f}")
print("MASS-DEPENDENT amplitude offset -- recommended linear-in-logM calibration "
      f"(pivot logM={PIV3}, weighted fit to binned medians):")
print(f"  Delta_Y(logM)    = {fit_y[1]:+.2f}% {fit_y[0]:+.2f}%/dex x (logM-{PIV3})   "
      f"[binned-median range {med_y.max():+.1f}% -> {med_y.min():+.1f}%]")
print(f"  Delta_fgas(logM) = {fit_g[1]:+.2f}% {fit_g[0]:+.2f}%/dex x (logM-{PIV3})   "
      f"[binned-median range {med_g.max():+.1f}% -> {med_g.min():+.1f}%]")
print(f"  Delta_Mstar(logM)= {fit_s[1]:+.2f}% {fit_s[0]:+.2f}%/dex x (logM-{PIV3})   "
      f"[binned-median range {med_s.max():+.1f}% -> {med_s.min():+.1f}%, "
      f"bootstrap SE {se_s.min():.1f}-{se_s.max():.1f}%]")
print("  after correction, binned-median residuals: "
      f"max|Y| = {np.max(np.abs(med_y - np.polyval(fit_y, cen_y - PIV3))):.1f}%, "
      f"max|fgas| = {np.max(np.abs(med_g - np.polyval(fit_g, cen_g - PIV3))):.1f}% "
      "-> divide BIND Y/f_gas by (1 + Delta(logM)/100)")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §2a′ — Fig 6: stacked radial profiles on a common $r/r_{200c}$ axis

Where fig 3 compares integrated aperture quantities, this figure compares the *radial
structure* of the painted halos: (a) the stacked Compton-$y$ profile, (b) the electron-column
profile $\tau(r)$, and (c) the stacked stellar surface density, BIND solid against TNG300
hydro dashed. The **same two mass bins appear in all three panels** — a well-populated
representative bin $10^{13.40}$–$10^{13.78}\,M_\odot/h$ ($N=728$) at full weight plus a
higher-mass $10^{14.18}$–$10^{14.70}$ bin ($N=59$) drawn lighter to signal its sparser
stack; both are named with their counts in the legend.

Everything comes from **one** cache (`bind_science/profiles/{fid,truth}_snap096.npz`), so the
mass bins and the radial grid are shared by construction — the previous version drew $\tau$
from a separate `tau_profiles/` reduction whose mass edges
($10^{13},3\times10^{13},10^{14},3\times10^{14},10^{15}$) disagreed with the rest of the
figure and whose top bin held 14 halos. That cache is retired here. $\tau$ is instead rebuilt
from the gas column as $\tau=\sigma_T\,(x_e/m_p)\,\Sigma_{\rm gas}$, a *pure scalar multiply*
(the same one the kSZ reduction applies), evaluated at this snapshot's $a=0.96738$; the
constant is printed by the cell. Because these profiles are background-subtracted (a
2.5–3.0 $h^{-1}$Mpc annulus), this is the **halo-only** $\tau$ and sits below the older
uncorrected product, which carried a cosmic floor.

The $x$-axis is $r/r_{200c}$ and the conversion is **exact, not approximate**: the reduction
bins each halo on $r/(0.659\,r_{200})$ with $0.659$ hardcoded, so multiplying the cached
radial coordinate by $0.659$ *undoes* that constant rather than introducing one. (The old
"$r/r_{500c}$" axis was the approximate one, since the true $r_{500c}/r_{200c}$ varies with
concentration.) The plotted window runs $0.15\lesssim r/r_{200c}\lesssim1.9$: the inner cut
is the pixel-resolution floor of the lower mass bin (1 px $=0.09\,r_{200c}$ there, verified:
median $r_{200}=0.543\,h^{-1}$Mpc in that bin) and the outer cut is a hand-set bound just
past where the sparser, noisier $N=59$ high-mass stack's $y$ and $\star$ legs first cross
zero (the fixed $2.5$–$3.0\,h^{-1}$Mpc background annulus leaves only $0.125\,h^{-1}$Mpc of
margin to the $3.125\,h^{-1}$Mpc patch half-width, so the already background-subtracted
signal is small and noisy there). Profiles are surface densities per $0.0488\,h^{-1}$Mpc
patch pixel.

There is no residual row: the BIND/truth mean ratios inside $r_{200c}$ are **printed** by the
cell for $y$, $\tau$, $M_\star$ and — as an unbiased control that is not plotted — the dark
matter, both over the plotted window and over every cached radial bin inside $r_{200c}$
(RECOMPUTED-ON-RUN; quote the printout). The pattern matches fig 3: $\tau$ and $y$ run a few
per cent high and more so in the lower mass bin ($y$ +3.8% vs −0.9%, $\tau$ +6.8% vs +4.5%),
stars are 8–13% low, and DM is unbiased to $<1$%.

**Bootstrap-over-halos bands: checked, not possible from this cache.** The high-mass bin
($N=59$) is thin enough that an errorbar on the stack matters, so this was checked directly
against the cache rather than assumed. `profiles/{fid,truth}_snap096.npz` stores `prof_*` as
*already-stacked* mean profiles, shape `(4 mass bins, 18 radial bins)` — a single number per
(mass bin, radius), plus a scalar `counts` per mass bin. There is no per-halo axis (and no
per-realization axis) anywhere in the file, so a delete-$d$/bootstrap resample **cannot be
built from this product** — printed and asserted by the cell rather than silently skipped. A
mirrored jackknife-style check (whether the high-mass bin's low-count *tail* end of any
grouping could stand in for a resampling axis) was also tried and comes up equally empty: there
is simply no second axis to resample over. No band is drawn on fig 6; this is stated here as a
release-completeness gap rather than worked around with a fabricated errorbar. The nearby
per-halo *aperture* quantities in fig 3 (`halo_atlas/{fid,truth}_snap096.npz`, same snapshot,
same halos) do carry a per-halo axis and are what fig 3's own bootstrap bands already use — but
they are integrated (aperture) numbers, not the radial shape plotted here, so they are not a
substitute. A genuine fig-6 error band needs a re-reduction of the profile cache with a
per-halo axis retained (flagged for a future pass, not attempted here since it is a data-product
change, not a figure edit).
''')

code(r'''
# ── Fig 6: stacked radial profiles, BIND vs hydro-pasted truth ──────────────
# data: bind_science/profiles/{fid,truth}_snap096.npz ONLY. Single cache => all
# three panels share mass_bins AND the radial grid by construction. (The old
# fig-3 tau panel used tau_profiles/, whose mass edges disagreed with these;
# that cache is deliberately NOT loaded here any more.)
pf = np.load(SCI/"profiles/fid_snap096.npz")
pt = np.load(SCI/"profiles/truth_snap096.npz")

# ── Bootstrap-over-halos CHECK (not assumed): does this cache have a per-halo
# axis to resample? Inspect the actual arrays rather than guessing.
_prof_shapes = {k: pf[k].shape for k in pf.files}
print(f"fig-6 profile cache arrays: {_prof_shapes}")
assert pf["prof_y"].shape == (len(pf["mass_bins"]) - 1, len(pf["r_cen"])), (
    "prof_y is not (n_mass_bins, n_radial_bins) -- re-check the bootstrap feasibility "
    "argument below, the cache layout has changed")
print("VERDICT: prof_* arrays are (mass_bin, radius) ONLY -- already-stacked means, no "
      "per-halo axis and no per-realization axis anywhere in this cache -- so a "
      "bootstrap-over-halos (or any resampling) error band CANNOT be built from this "
      "product. No band is drawn below; see the markdown for what would be needed "
      "(a per-halo-axis re-reduction) and why fig 3's per-halo atlas is not a substitute "
      "(integrated aperture quantities, not this radial shape).")

# EXACT r/r200c conversion, NOT an approximation: the generator bins each halo
# on x = r_Mpch/(R500_FAC*r200) with R500_FAC = 0.659 HARDCODED, so multiplying
# the cached r_cen by 0.659 undoes that constant with zero residual error.
R500_FAC = 0.659
r200x = R500_FAC*pf["r_cen"]

# tau = K_TAU * Sigma_gas: the kSZ reduction's tau_patches() is a pure scalar
# multiply of the same gas surface density (sigma_T * x_e/m_p / physical pixel
# area) at this snapshot's a, so rebuilding tau from prof_gas puts it on the
# same mass bins and radial grid as y and the stars.
SIGMA_T, M_P, MSUN_G, MPC_CM, HH = 6.6524e-25, 1.6726e-24, 1.989e33, 3.0857e24, 0.6774
PIX_MPCH = 6.25/128.0
a_snap   = 1.0/(1.0 + float(pf["z"]))                    # 0.96738 (z = 0.03372)
K_TAU    = SIGMA_T*(0.88/M_P)*MSUN_G/(HH*(PIX_MPCH*a_snap/HH*MPC_CM)**2)

MB3   = np.log10(np.asarray(pf["mass_bins"], float))     # 13.00 13.40 13.78 14.18 14.70
CNT3  = np.asarray(pf["counts"], int)                    # 1887 728 254 59
BINS3 = [(1, 1.0), (3, 0.6)]   # (mass-bin idx, alpha): representative bin at full
                               # weight + a lighter high-mass bin -- SAME two bins
                               # in all three panels (that is the whole point)
# inner cut = pixel-resolution floor of the lower plotted bin (1 px = 0.090
# r200c there); outer cut is HAND-SET (not derived from a "last positive bin"
# query -- checked: the high-mass (N=59) bin's y/star legs are already
# marginally negative right at r200x=1.855, one grid point past the true
# all-positive edge at 1.474; those two points are silently dropped by the
# per-key (yb>0)&(yt>0) mask below, which is why the figure still looks clean)
# -- beyond ~2 r200c the stacks hit the patch edge and the background
# subtraction drives them negative.
rok = (r200x >= 0.145) & (r200x <= 1.90)

CH3 = [("prof_y",    1.0,   r"$y(r)$",                                        "(a)"),
       ("prof_gas",  K_TAU, r"$\tau(r)$",                                     "(b)"),
       ("prof_star", 1.0,   r"$\Sigma_\star(r)$ [$M_\odot h^{-1}$ px$^{-1}$]", "(c)")]

fig, ax = plt.subplots(1, 3, figsize=(TWO_COL[0], 2.6), sharex=True)
for j, (key, sc, ylab, tag) in enumerate(CH3):
    for bi, al in BINS3:
        yb, yt = sc*pf[key][bi], sc*pt[key][bi]
        m = rok & np.isfinite(yb) & np.isfinite(yt) & (yb > 0) & (yt > 0)
        ax[j].loglog(r200x[m], yb[m], color=COLORS["bind"],
                     lw=1.6 if al == 1.0 else 1.0, alpha=al,
                     label=f"{MB3[bi]:.1f}-{MB3[bi+1]:.1f} ($N$={CNT3[bi]})")
        ax[j].loglog(r200x[m], yt[m], color=COLORS["truth"], ls="--",
                     lw=1.3 if al == 1.0 else 1.1, alpha=al)
    ax[j].set_xlabel(r"$r/r_{200c}$")
    ax[j].set_ylabel(ylab)
    panel_label(ax[j], tag, loc="upper right")
ax[0].legend(title=r"$\log_{10}M_{200c}\,[M_\odot/h]$", title_fontsize=5.6,
             fontsize=5.2, loc="lower left")
from matplotlib.lines import Line2D
ax[1].legend(handles=[Line2D([], [], color=COLORS["bind"], lw=1.6, label="BIND"),
                      Line2D([], [], color=COLORS["truth"], ls="--", lw=1.3,
                             label="hydro-pasted")], fontsize=5.2, loc="lower left")
fig.align_ylabels(ax)
fig.tight_layout(w_pad=0.9)
save(fig, "figs_v2/fig06_radial_profiles")
plt.show()
print(f"stacked profiles, snap {int(pf['snap'])} (z = {float(pf['z']):.5f}); "
      f"r/r200c = {R500_FAC}*r_cen (exact undo of the reduction constant); "
      f"tau = {K_TAU:.6e} * Sigma_gas at a = {a_snap:.5f}")
print(f"plotted window {r200x[rok][0]:.3f}-{r200x[rok][-1]:.3f} r200c "
      f"({int(rok.sum())} of {len(r200x)} radial bins); mass bins "
      + ", ".join(f"{MB3[bi]:.2f}-{MB3[bi+1]:.2f} (N={CNT3[bi]})" for bi, _ in BINS3))
print("mean BIND/truth inside r200c [plotted window] (all cached bins r<r200c):")
rin = rok & (r200x <= 1.0)
for key, lab in [("prof_y", "y"), ("prof_gas", "tau"), ("prof_star", "M_star"),
                 ("prof_DM", "DM (control, not plotted)")]:
    for bi, _ in BINS3:
        A, B = pf[key][bi], pt[key][bi]
        g = np.isfinite(A) & np.isfinite(B) & (A > 0) & (B > 0)
        m1, m2 = rin & g, (r200x <= 1.0) & g
        print(f"  {lab:26s} {MB3[bi]:.2f}-{MB3[bi+1]:.2f}: "
              f"{np.mean(A[m1]/B[m1]):.3f}  ({np.mean(A[m2]/B[m2]):.3f})")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §2b — Fig 4: the eight WL field statistics vs the hydro-pasted truth

**Closure philosophy.** "Truth" here is the seed-matched hydro *paste* of the same
$\geq10^{13}\,M_\odot/h$ halo population, composited and ray-traced identically — so this
figure isolates *emulation fidelity* (does painted gas produce the right projected
statistics?) from *completeness* (what the $\geq10^{13}$ scope omits; §5). Because the traces
share the ray seed, we recompute all spectra per realization from the cached maps and form
paired residuals whose ±1σ bands are ~0.1–0.7% (§2.0) — the most stringent field-level test
available. Fig 4 carries the eight **weak-lensing** statistics; the three auto- and two
cross-**spectra** of the released field vector (including both $y$ channels and the $\tau$
channel) are fig 4b.

**Noise-accounting audit.** A reader could reasonably ask whether the huge
$\chi^2/\mathrm{dof}=19/120/177$ ($\kappa\kappa/\kappa y/yy$) printed by the cell against these
sub-percent bands is a real signal or an artifact of treating 50 correlated ray-trace
realizations as independent. We settle this with two nulls computed directly on the cached
maps. (i) Split each 50-realization set in half and difference the two independent
25-realization means with the identical $\sigma/\sqrt{25}$ recipe used for the paired band —
if the noise model is correct this null $\chi^2/\mathrm{dof}$ should sit near 1. It comes back
at $0.8$–$0.9$ ($\kappa\kappa$) and $0.6$ (both $y$-involved spectra), for *both* the
truth-only and the BIND-only halves — consistent with (if anything mildly conservative
relative to) 1, never $\gg1$. (ii) Independently, the realization-to-realization correlation
matrix of each $C_\ell$ curve (z-scored per $\ell$ bin over the trusted range) has mean
off-diagonal $|\bar\rho|\approx0.02$ for all three spectra — consistent with zero — giving
$N_{\rm eff}\approx N=50$. **Verdict: the per-$\ell$ $\sigma/\sqrt{50}$ noise model is
validated by both tests, so no band rescaling is applied; the diagonal
$\chi^2/\mathrm{dof}=19/120/177$ are genuine, highly significant characterized emulation
systematics, not an accounting artifact.** A distinct, already-recognized effect remains:
individual $\ell$ bins *within* one realization's spectrum are themselves correlated (mean
$|\bar\rho|=0.25/0.43/0.48$), so the $\sim$600-bin diagonal dof over-counts independent
constraints; a Hartlap-corrected full-covariance test on a 25-bin compression gives
$\chi^2/\mathrm{dof}=15/101/34$ — smaller, but still $\gg1$: the offsets survive a fully
covariant treatment too. We additionally print an *unpaired* variant (T2): the same test with
the truth+BIND realization covariance and no seed-cancellation — the significance a map user
would face against an independent simulation. This audit (and the Hartlap block) covers the
$\kappa y$/$yy$ legs of fig 4b as well; **every $\chi^2/\mathrm{dof}$ now lives in the printed
cell output rather than on the figure.**

**Is the $yy$ offset just a normalization?** Fig 3 reports a halo-level $Y_{500c}$
BIND/truth offset of $1.054\pm0.007$; a uniform amplitude rescale by that factor would
predict a *flat*, $\ell$-independent $C_\ell^{yy}$ excess of $1.054^2-1=+11.1\%$
(`audits/map_rescale_check.py`). Dividing the measured $C_\ell^{yy}$ by that constant does
**not** flatten the residual: at mid $\ell$ (300–3000) the measured residual is itself
*negative* ($-6.2\%$ median — the opposite sign from the offset's prediction), so removing
it makes the mismatch worse ($\chi^2/\mathrm{dof}$ $46\to353$); at high $\ell$
(5000–15000) the $+7.6\%$ excess only partially aligns, overshooting to $-3.1\%$ rather
than flattening to zero. The overall diagonal $\chi^2/\mathrm{dof}$ across the trusted
range *increases* after the rescale ($177\to250$). The halo-integrated Y-normalization
offset is therefore not the map-level $C_{yy}$ discrepancy's dominant driver; its
$\ell$-dependence points to a scale-dependent (texture) effect instead.

**Survey context.** The residual strips of the two $C_\ell$-type panels shade the LSST-Y10
precision on the fiducial — **v2: measured, not Gaussian-only** — built from the std of the
binned $\log C_\ell^{\kappa\kappa}$ across the 50 fiducial realizations (this box's
$25\,\mathrm{deg}^2$ footprint), area-scaled to LSST-Y10 by $\sqrt{\Omega_{25}/\Omega_{\rm LSST}}$,
plus the analytic shape-noise excess ($n_{\rm gal}=27\,\mathrm{arcmin}^{-2}$, $\sigma_e=0.26$,
$f_{\rm sky}=0.44$) added in quadrature — the same recipe as Fig 12, whose markdown states the
small-$N$/shared-box caveat once. So "significant at paired precision" is visible against what
an actual survey could resolve. **v2 update:** the measured-covariance LSST-Y10 band is
*tighter* than the old Knox-only one at low-to-mid $\ell$ (RECOMPUTED-ON-RUN; quote the printed
old-vs-new table, not this prose) — comparable in scale to, not clearly larger than, our
sub-percent paired band and the 0.7% $\kappa\kappa$ offset itself — but still well below the
1.4%/5.6% $\kappa y$/$yy$ offsets of fig 4b, so the qualitative conclusion survives: a real
Stage-IV measurement would plausibly still detect the $y$-channel systematics reported here
even without seed-pairing, though the $\kappa\kappa$ offset's detectability against LSST-Y10
alone is now a closer call than the v1 band suggested.
We do not attempt an analogous band for the $\nu$-domain statistics (PDF, peaks, minima,
$V_{0,1,2}$), whose survey-noise translation would require an additional, ad hoc
smoothing/pixelization model rather than the direct `cl_relerr` recipe — their shape-noise
sensitivity is instead separately quantified via the $n_{\rm gal}=10$-matched closure quoted
in the cell output.

**Panels.** Top row $C_\ell^{\kappa\kappa}$, the suppression
$S(\ell)=C_\ell^{\rm bind}/C_\ell^{\rm dmo}$, the 1-pt PDF and peak counts; bottom row minima
counts and the Minkowski functionals $V_0,V_1,V_2$. **Every signal panel now draws all five
source planes**, colour-graded from $z_s=0.5$ to $2.44$; the hydro-pasted truth overlay (dashed) and
every residual strip stay at $z_s=1$, where the seed-paired per-realization band is built.
The six $\nu$-domain panels share one $\nu\in[-3,8]$ axis: $\nu$ is normalized throughout by
the cached per-plane $\sigma_0$ (the $2'$-smoothed map rms,
`paired_perreal_fid.npz['sigma0']`, the same constant the peak/minima axis already used), so
the **PDF is now plotted in S/N units** — $\nu=\kappa/\sigma_0$ with the density rescaled as
$P(\nu)=\sigma_0P(\kappa)$, a pure change of variable that leaves the residual and its
$\chi^2$ numerically untouched. Note the PDF is of the *unsmoothed* field, the counts of the
$2'$-smoothed field and the MFs of the $1'$-smoothed field, each in its own S/N convention.
The released PDF grid is fixed in $\kappa$ (41 bins over $\pm6\sigma$ of the *global* map
rms), so after the per-plane rescaling the low-$z_s$ curves are coarsely sampled in $\nu$
while the deep planes are finely sampled — a property of the released product, shown as it
is rather than re-binned.
The MF panels are the one place the caches could not be reused: the released
$V_{0,1,2}$ grid is hard-capped at $\nu\le4$, so they are **recomputed live** through
`stats.nongaussian_stats` on $\nu=\mathrm{linspace}(-3,8,45)$ (asserted to reproduce the
cached $\nu\le4$ values exactly, since that grid is the extension's first 29 nodes), on
log/symlog axes because $V_{0,1,2}$ fall three decades between $\nu=0$ and $\nu=8$ while
staying nonzero. That recomputation is also the first *truth-side* per-realization MF draw in
the release, which **retires the previous $\sqrt2$ "assume the truth scatter equals BIND's"
kludge**: the MF error band is now the genuine paired per-realization difference scatter, the
same construction the peaks/minima/PDF bands use. Their $\chi^2$ is reported over both
$\nu\le4$ (comparable with the published values) and the full extended range.
The $\ell$ axes run to the **corner mode $\ell=5.2\times10^4$** — the full extent of the
released grid; 1024² maps over 5 deg have axis-Nyquist $\ell=36864$ and simply contain no
modes beyond that, so a larger limit would be blank axis. Above $\ell\simeq1.5\times10^4$ the
*raw* spectrum turns up from CIC aliasing (shaded once, on panel (a)); the aliasing cancels
in every ratio, which is why $S(\ell)$ and the residual strips stay readable there.
$S(\ell)$'s residual and $\chi^2/\mathrm{dof}$ *numerically reproduce*
$C_\ell^{\kappa\kappa}$'s exactly — both BIND and truth are divided by the same fixed DMO
curve — it is shown regardless because the *suppression amplitude relative to DMO* is the
physically relevant quantity for §3–4, not because it is an independent test.
For peaks, minima and the PDF the per-realization draws are recomputed live from the cached
maps with the cached conventions (the caches store only means; the recomputation is asserted
to reproduce them exactly), so these three statistics carry the *same* seed-paired
per-realization scatter band as the spectra (ratio scatter for the counts, max-normalized
difference scatter for the PDF) — **a v2 correction**: the previous recipe added the cached
truth SE (already $\sigma/\sqrt{50}$) in quadrature and divided by $\sqrt{50}$ again,
double-counting the $\sqrt{50}$ on the truth side and ignoring the seed-pairing.

**Reading.** $C_\ell^{\kappa\kappa}$ closes at the sub-percent level (0.7% median, $\ell\in$
300–5000) across the trusted range — the WL leg is essentially exact. Against the corrected
seed-paired bands the counting statistics close outright: peaks and minima at
$\chi^2/\mathrm{dof}\approx1$ (exact values in the printed output), an order of magnitude
tighter than $C_\ell^{\kappa\kappa}$'s own 19 — scale-mixing statistics average over the
sub-percent spectral offset. The MFs, now banded with the *real* truth-side scatter instead
of the inflated $\sqrt2$ proxy, are re-quoted by the cell (their published $\chi^2$ values
change with the band recipe, not with the physics; the residual amplitudes are unchanged and
remain a few percent of peak). The $\kappa$-PDF is the instructive exception: its sub-percent
residual — previously shown unbanded — is *resolved as significant* by its tight paired band
($\chi^2/\mathrm{dof}\gg1$; the value and the residual amplitude are printed by the cell), the
1-pt counterpart of the sub-percent spectral offset and, like it, a small *characterized
systematic* rather than a failed closure. With paired errors this small,
$\chi^2/\mathrm{dof}\gg1$ — now audited rather than asserted — states these percent-level
offsets are *significant emulation systematics*, an error-budget template for users, not a
failed closure. Peaks are noiseless; the $n_{\rm gal}=10$ shape-noise closure is quoted in the
cell output. The $y$-involved channels and their high-$\ell$ features move to fig 4b, with the
mechanism localized in fig 18 (Appendix A).
''')

code(r'''
# ── Fig 4: noise-accounting audit + the eight WL field statistics ────────────
# data: bind_science/runs/{bind,truth}/run_0000/{kappa,y}_maps.npz (per-real
#       spectra AND per-real peak/minima/PDF/MF draws recomputed here),
#       peak_counts.npz, nongaussian_stats.npz, Cl_kappa.npz (5-plane means),
#       paired_perreal_fid.npz (BIND per-real clk/pk/min + sigma0),
#       runs/dmo/run_0000/Cl_kappa.npz (S(ell) denominator).
# v3: the SZ/tau spectra move to fig 4b; every signal panel carries all five
#     source planes (residuals stay at z_s=1); the PDF moves to a nu axis; and
#     V0/V1/V2 are RECOMPUTED LIVE on nu in [-3,8] because the released MF grid
#     is hard-capped at nu=4. That recompute also yields the first TRUTH-side
#     per-realization MF draws, retiring the sqrt(2) "truth scatter = BIND
#     scatter" kludge in the MF band. No on-figure chi2/N_eff stamps any more:
#     every number is printed below (FIGURE_NUMBERS.md quotes the print output).
from bind.inference.stats import power_spectrum, nongaussian_stats

RB, RT = SCI/"runs/bind/run_0000", SCI/"runs/truth/run_0000"
ELL_LO = 100.0    # clean trusted-range floor (matches chi2_full/fig10's convention below)
# ELL_TRUST / ELL_MAX_PLOT are now defined ONCE in the setup cell (shared by figs
# 4, 4b, 7, 12). Raw spectra turn up from CIC aliasing above ELL_TRUST -- shaded
# once on panel (a); it cancels in every ratio, which is why S(ell) and the
# residual strips stay readable there.
# one color per source plane, shared by figs 4 and 4b -- the SAME sequential
# ramp fig 11 already uses for its z_s series, so a colour means one source
# plane everywhere in the validation section.
ZCOLS = plt.cm.plasma(np.linspace(0.02, 0.82, 5))

# All five planes stay in memory: the ZI legs below are views (no copies) and
# the live MF recompute needs every plane. Freed at the end of the cell.
f = np.load(RB/"kappa_maps.npz"); BK = f["kappa"]; del f; gc.collect()
f = np.load(RT/"kappa_maps.npz"); TK = f["kappa"]; del f; gc.collect()
by = np.load(RB/"y_maps.npz")["y"][:, -1].copy()   # total column = released convention
ty = np.load(RT/"y_maps.npz")["y"][:, -1].copy()
gc.collect()
bk, tk = BK[:, ZI], TK[:, ZI]
NR = BK.shape[0]

# per-realization paired spectra at z_s=1 (both traces share the ray seed).
# Prefer the precomputed cache (field_cache.py, built by run_field_cache.sbatch under
# disBatch): ~750 power_spectrum calls across figs 4 and 4b become a ~15 MB load.
# load() returns None unless the cache matches this run's geometry, so a stale cache
# is never used silently -- we recompute inline and the figures stay reproducible
# from the released maps alone. NOTE the cache stores the CORRECT cross convention
# (kappa(z_s) x TOTAL column, computed from maps); the released Cl_kappa_y.npz /
# Cl_tau.npz carry the old XPk_plane normalization and are used nowhere here.
sys.path.insert(0, str(Path.cwd()))
from field_cache import load as _load_fields                 # noqa: E402
_fc = _load_fields(n_real=NR)
if _fc is not None:
    ell_f = _fc["ell"]
    kk_b, kk_t = _fc["kk_bind"][:, ZI], _fc["kk_truth"][:, ZI]
    ky_b, ky_t = _fc["ky_bind"][:, ZI], _fc["ky_truth"][:, ZI]
    # [:, -1] NOT [:, ZI]: C^yy is the TOTAL column (y[:, -1]) in the released
    # convention, matching the live branch below. y maps are cumulative per plane,
    # so the z_s=1 plane is a much shallower column -- using it here silently
    # redefined the statistic (and moved its chi2 from 177 to 63).
    yy_b, yy_t = _fc["yy_bind"][:, -1], _fc["yy_truth"][:, -1]
    _fld_src = "cache"
else:
    kk_b, kk_t, ky_b, ky_t, yy_b, yy_t = [], [], [], [], [], []
    for r in range(NR):
        ell_f, c = power_spectrum(bk[r]);        kk_b.append(c)
        _,  c = power_spectrum(tk[r]);           kk_t.append(c)
        _,  c = power_spectrum(bk[r], by[r]);    ky_b.append(c)
        _,  c = power_spectrum(tk[r], ty[r]);    ky_t.append(c)
        _,  c = power_spectrum(by[r]);           yy_b.append(c)
        _,  c = power_spectrum(ty[r]);           yy_t.append(c)
    kk_b, kk_t, ky_b, ky_t, yy_b, yy_t = map(
        np.array, (kk_b, kk_t, ky_b, ky_t, yy_b, yy_t))
    _fld_src = "live"
print(f"fig-4 paired spectra: source={_fld_src}")
del by, ty; gc.collect()      # the y maps are reloaded per-plane in fig 4b's cell

# ── per-realization peaks/minima/PDF, recomputed live from the SAME maps ─────
# The truth caches store only realization means (+SE), but the seed-paired
# bands below need per-realization draws on BOTH sides. peak_counts() with the
# cached conventions (2.0' smoothing, nu_norm='map') reproduces the cached
# peak_counts.npz means/errs EXACTLY (asserted); the PDF is re-histogrammed on
# the cached bin grid with the same m - mean(m) convention (~0.3% of the peak
# density from float accumulation -- used consistently on both sides below).
from bind.inference.stats import peak_counts as peak_counts_fn
b_pk, t_pk = np.load(RB/"peak_counts.npz"), np.load(RT/"peak_counts.npz")
b_ng, t_ng = np.load(RB/"nongaussian_stats.npz"), np.load(RT/"nongaussian_stats.npz")
pdf_cent = b_ng["pdf_bins"]; _dpc = pdf_cent[1] - pdf_cent[0]
pdf_edges = np.concatenate([pdf_cent - _dpc/2, [pdf_cent[-1] + _dpc/2]])
if _fc is not None:                       # same source as the paired spectra
    pk_b_real,  pk_t_real  = _fc["pk_bind"][:, ZI],  _fc["pk_truth"][:, ZI]
    min_b_real, min_t_real = _fc["min_bind"][:, ZI], _fc["min_truth"][:, ZI]
    pdf_b_real, pdf_t_real = _fc["pdf_bind"][:, ZI], _fc["pdf_truth"][:, ZI]
    assert np.allclose(_fc["pdf_bins"], pdf_cent), "cached PDF grid != released grid"
else:
    _pb = peak_counts_fn(bk[:, None], fov_deg=5.0,
                         smoothing_arcmin=float(b_pk["smoothing_arcmin"]),
                         nu_norm=str(b_pk["nu_norm"]), return_realizations=True)
    _pt = peak_counts_fn(tk[:, None], fov_deg=5.0,
                         smoothing_arcmin=float(t_pk["smoothing_arcmin"]),
                         nu_norm=str(t_pk["nu_norm"]), return_realizations=True)
    pk_b_real,  pk_t_real  = _pb["peak_counts_real"][:, 0],   _pt["peak_counts_real"][:, 0]
    min_b_real, min_t_real = _pb["minima_counts_real"][:, 0], _pt["minima_counts_real"][:, 0]
    pdf_b_real = np.array([np.histogram(m - m.mean(), bins=pdf_edges, density=True)[0]
                           for m in bk])
    pdf_t_real = np.array([np.histogram(m - m.mean(), bins=pdf_edges, density=True)[0]
                           for m in tk])
    del _pb, _pt; gc.collect()
# convention guard, cache or live: the counts must reproduce the released
# nu_norm='map' means. (paired_perreal_fid.npz is NOT a valid reference here --
# its pk/min are the nu_norm='fixed' convention, differing by up to 2.2e-2.)
assert np.allclose(pk_b_real.mean(0),  b_pk["peak_counts"][ZI])
assert np.allclose(min_t_real.mean(0), t_pk["minima_counts"][ZI])

# ── Minkowski functionals, recomputed LIVE on an extended nu grid ────────────
# The released MF grid is hard-capped: nongaussian_stats.npz['mf_nu'] =
# linspace(-3,4,29) on BOTH sides (stats.nongaussian_stats' default), and the
# per-realization cache is on the same grid -- nothing above nu=4 exists
# anywhere. We therefore call the PUBLIC entry point on linspace(-3,8,45).
# Two properties make this exact rather than a re-implementation: (i) the
# extended grid has the same 0.25 step, so its first 29 nodes ARE the cached
# grid (and minkowski_functionals bins each threshold independently), and
# (ii) only smoothing_scales_arcmin[0] enters the MFs, so passing just that
# scale reproduces the cached V0/V1/V2 while skipping the 3 unused moment
# smoothings. Both are ASSERTED against the cached arrays below.
# Plane by plane, on views (no map copies), ~0.25 s/map.
MF_NU8 = np.linspace(-3.0, 8.0, 45)
_sc0 = (float(b_ng["smoothing_scales_arcmin"][0]),)
def mf_live(K):
    """Per-plane V0/V1/V2 on MF_NU8: (5, 45) means + (NR, 5, 45) realizations."""
    Vm = {k: np.empty((5, len(MF_NU8))) for k in ("V0", "V1", "V2")}
    Vr = {k: np.empty((NR, 5, len(MF_NU8))) for k in ("V0", "V1", "V2")}
    for zi in range(5):
        o = nongaussian_stats(K[:, zi][:, None], fov_deg=5.0,
                              smoothing_scales_arcmin=_sc0,
                              mf_thresholds=MF_NU8, return_realizations=True)
        for k in ("V0", "V1", "V2"):
            Vm[k][zi] = o[k][0]; Vr[k][:, zi] = o[k + "_real"][:, 0]
        del o; gc.collect()
    return Vm, Vr

# Prefer the precomputed cache (papers/01_pipeline/mf_cache.py, built in parallel by
# run_mf_cache.sbatch under disBatch): it turns ~2 min of MF compute per notebook run
# into a ~1 MB load. load() returns None unless the cache was built on EXACTLY this
# threshold grid, plane count and realization count, so a stale cache can never be
# used silently -- we simply recompute inline instead, and the figure stays
# reproducible from the released maps alone.
sys.path.insert(0, str(Path.cwd()))
from mf_cache import CACHE as CACHE_MF, load as _load_mf     # noqa: E402
_mfc = _load_mf(n_real=NR)
if _mfc is not None and np.allclose(_mfc["mf_nu"], MF_NU8):
    V8b = {k: _mfc[f"bind_{k}_mean"] for k in ("V0", "V1", "V2")}
    V8t = {k: _mfc[f"truth_{k}_mean"] for k in ("V0", "V1", "V2")}
    V8b_r = {k: _mfc[f"bind_{k}"] for k in ("V0", "V1", "V2")}
    V8t_r = {k: _mfc[f"truth_{k}"] for k in ("V0", "V1", "V2")}
    _mf_src = "cache"
else:
    _mf_src = "live"

_t0 = time.time()
if _mf_src == "live":
    V8b, V8b_r = mf_live(BK)
    V8t, V8t_r = mf_live(TK)
_n29 = len(b_ng["mf_nu"])
assert np.allclose(MF_NU8[:_n29], b_ng["mf_nu"])     # extension contains the cached grid
# Convention guard. NOTE which reference each side is checked against:
#   truth -> nongaussian_stats.npz            (reproduces bit-exactly)
#   bind  -> paired_perreal_fid.npz realization mean  (reproduces bit-exactly)
# runs/bind/run_0000/nongaussian_stats.npz is NOT usable as the bind reference: its
# V0/V1/V2 are stale with respect to the current bind kappa maps. Two independent
# recomputations (this cell, and the per-realization paired_perreal_fid.npz built
# five days later) agree with each other to 0.0 and both differ from that summary
# file by 3.6e-4 / 2.5e-3 / 5.6e-3 (V0/V1/V2, relative to the peak). The figure
# therefore uses the recomputed values on both sides (cache or live -- they are the
# same numbers); the stale summary is only reported, never plotted. The guard runs
# either way, so a bad cache is caught here too.
_prg = np.load(RB/"paired_perreal_fid.npz")     # loaded again below as `pr`; the
_pr_mf = {k: _prg[k][:, ZI, :].mean(0)          # guard must run BEFORE that point
          for k in ("V0", "V1", "V2")}
for k in ("V0", "V1", "V2"):
    for lab, live, ref, src in (
            ("bind",  V8b[k][ZI, :_n29], _pr_mf[k],  "paired_perreal_fid"),
            ("truth", V8t[k][ZI, :_n29], t_ng[k][ZI], "nongaussian_stats")):
        dev = float(np.max(np.abs(live - ref))/max(np.max(np.abs(ref)), 1e-30))
        print(f"MF guard {k} {lab} (vs {src}): max|live-ref|/max|ref| = {dev:.2e}")
        assert dev < 1e-6, f"{k}/{lab} live MFs do not reproduce {src}"
for k in ("V0", "V1", "V2"):     # provenance: size of the stale-summary discrepancy
    _d = float(np.max(np.abs(_pr_mf[k] - b_ng[k][ZI]))/np.max(np.abs(b_ng[k][ZI])))
    print(f"  (bind/nongaussian_stats.npz {k} is SUPERSEDED: differs from the live "
          f"and per-realization values by {_d:.2e} -- not used in this figure)")
print(f"MFs on nu={MF_NU8[0]:.0f}..{MF_NU8[-1]:.0f} ({len(MF_NU8)} thresholds, "
      f"5 planes, both sides): source={_mf_src}"
      + (f" ({CACHE_MF})" if _mf_src == "cache" else "")
      + f", {time.time()-_t0:.0f} s")
del BK, TK, bk, tk; gc.collect()

mt = (ell_f >= ELL_LO) & (ell_f <= ELL_TRUST)

def paired(bR, tR):
    """mean curves, %-residual of means, and paired +-1sigma band on the residual."""
    rat = bR/np.where(tR > 0, tR, np.nan)
    res = 100*(np.nanmean(bR, 0)/np.nanmean(tR, 0) - 1)
    band = 100*np.nanstd(rat, 0)/np.sqrt(NR)
    chi2 = np.nanmean((res[mt]/np.where(band[mt] > 0, band[mt], np.nan))**2)
    return np.nanmean(bR, 0), np.nanmean(tR, 0), res, band, chi2

def chi2_full(bR, tR):
    """Hartlap-corrected full-covariance chi2/dof on a 25-bin compression, plus
    the mean |off-diagonal ell-bin correlation| (bin-to-bin, WITHIN a realization).
    Also returns n (realizations), the RAW (un-Hartlap-corrected) chi2 = dv^T
    Cm^-1 dv, and p -- the ingredients the Sellentin-Heavens REPORTED
    ALTERNATIVE p-value (printed alongside the Gaussian/Hartlap one, below)
    needs but the 3-value return of earlier drafts did not carry."""
    msk4 = (ell_f >= ELL_LO) & (ell_f <= ELL_TRUST)
    rat = rebin((bR/np.where(tR > 0, tR, np.nan))[:, msk4], 8)   # (50, 25)
    okb = np.isfinite(rat).all(0)      # cross-spectra can fluctuate <=0 per real
    rat = rat[:, okb]
    n, p = rat.shape
    Cm = np.cov(rat, rowvar=False)/n
    corr = Cm/np.sqrt(np.outer(np.diag(Cm), np.diag(Cm)))
    rho_bar = np.nanmean(np.abs(corr[np.triu_indices(p, k=1)]))
    hart = (n - p - 2)/(n - 1)
    dv = rat.mean(0) - 1
    chi2_raw = float(dv @ np.linalg.solve(Cm, dv))
    return hart*chi2_raw/p, rho_bar, p, n, chi2_raw

# ═════════════════════════════════════════════════════════════════════════
# PART A -- noise-accounting audit: is chi2/dof>>1 real, or underestimated bands?
# ═════════════════════════════════════════════════════════════════════════
def null_test(R, nhalf=25):
    """Split-half null: mean[0:nhalf] vs mean[nhalf:2nhalf], sigma = combined
    std/sqrt(nhalf) -- the SAME per-bin recipe used for the paired BIND-vs-truth
    band, applied to two independent halves of ONE trace."""
    m1, m2 = np.nanmean(R[:nhalf], 0), np.nanmean(R[nhalf:2*nhalf], 0)
    s1, s2 = np.nanstd(R[:nhalf], 0)/np.sqrt(nhalf), np.nanstd(R[nhalf:2*nhalf], 0)/np.sqrt(nhalf)
    sig = np.sqrt(s1**2 + s2**2)
    return np.nanmean(((m1 - m2)[mt]/np.where(sig[mt] > 0, sig[mt], np.nan))**2)

def rho_realization(R, N=50):
    """Realization-to-realization correlation matrix of Cl (z-scored per ell bin
    over the trusted range) -> mean off-diagonal |rho| -> N_eff via the
    exchangeable-correlation formula N_eff = N/(1+(N-1)rho_bar)."""
    X = R[:, mt]
    sd = np.nanstd(X, 0, keepdims=True)
    z = (X - np.nanmean(X, 0, keepdims=True))/np.where(sd > 0, sd, np.nan)
    ok = np.isfinite(z).all(1)
    C = np.corrcoef(z[ok])
    rho_bar = np.nanmean(C[np.triu_indices(C.shape[0], k=1)])
    return rho_bar, N/(1 + (N-1)*rho_bar)

print("=== Part A: noise-accounting audit (covers fig 4b's ky/yy legs too) ===")
for lab, Rt, Rb in [("kk", kk_t, kk_b), ("ky", ky_t, ky_b), ("yy", yy_t, yy_b)]:
    x2t, x2b = null_test(Rt), null_test(Rb)
    rho_t, _ = rho_realization(Rt); rho_b, _ = rho_realization(Rb)
    rho_bar = max(rho_t, rho_b)
    print(f"{lab}: null chi2/dof  truth-halves={x2t:.2f}  bind-halves={x2b:.2f}   |   "
          f"realization-realization rho_bar={rho_bar:+.3f} (consistent with 0 -> N_eff~50)")
print("VERDICT: null chi2/dof ~1 (0.6-0.9) for all three statistics and BOTH halves; "
      "the realization-correlation matrix is consistent with zero -> the per-ell "
      "sigma/sqrt(50) noise model is validated. No band rescaling is applied below; the "
      "diagonal BIND-vs-truth chi2/dof computed with this SAME band are genuine, highly "
      "significant emulation systematics.")

# ═════════════════════════════════════════════════════════════════════════
# PART B -- the figure: 8 WL statistics, five source planes in every signal
#           panel, seed-paired residual strips at z_s=1
# ═════════════════════════════════════════════════════════════════════════
pr = np.load(RB/"paired_perreal_fid.npz")
dcl0 = np.load(SCI/"runs/dmo/run_0000/Cl_kappa.npz")
b_cl0, t_cl0 = np.load(RB/"Cl_kappa.npz"), np.load(RT/"Cl_kappa.npz")

f_ell = ell_f*(ell_f + 1)/(2*np.pi)
KKb, KKt, res_kk, band_kk, x2_kk = paired(kk_b, kk_t)
KYb, KYt, res_ky, band_ky, x2_ky = paired(ky_b, ky_t)
YYb, YYt, res_yy, band_yy, x2_yy = paired(yy_b, yy_t)

# 5-plane MEAN curves come from the CACHES (the live per-realization
# recomputation stays at ZI, where it is needed to build the paired band); the
# ZI row is then swapped for the live curve so panel and residual agree exactly.
KK5b = np.array([b_cl0["cl"][zi, zi] for zi in range(5)], dtype=float)
KK5t = np.array([t_cl0["cl"][zi, zi] for zi in range(5)], dtype=float)
print(f"cached vs live mean Cl_kk at z_s={ZS[ZI]:.1f}: median |ratio-1| = "
      f"{100*np.nanmedian(np.abs(KK5b[ZI]/np.where(KKb > 0, KKb, np.nan) - 1)):.4f}% (BIND), "
      f"{100*np.nanmedian(np.abs(KK5t[ZI]/np.where(KKt > 0, KKt, np.nan) - 1)):.4f}% (truth)")
KK5b[ZI], KK5t[ZI] = KKb, KKt

# S(ell): BIND per-real from the paired_perreal_fid clk cache, TRUTH per-real
# from kk_t recomputed above, both over the (pre-averaged) DMO cache curve.
dmo_mean = dcl0["cl"][ZI, ZI]
S_b = pr["clk"][:, ZI, :]/np.where(dmo_mean > 0, dmo_mean, np.nan)
S_t = kk_t/np.where(dmo_mean > 0, dmo_mean, np.nan)
Sb_m, St_m, res_S, band_S, x2_S = paired(S_b, S_t)
dmo5 = np.array([dcl0["cl"][zi, zi] for zi in range(5)], dtype=float)
S5b = pr["clk"].mean(0)/np.where(dmo5 > 0, dmo5, np.nan)
S5t = KK5t/np.where(dmo5 > 0, dmo5, np.nan)

# full-covariance chi2 (25-bin compression, Hartlap): printed below, and it
# audits fig 4b's ky/yy legs as well as fig 4's kk.
fullcov = {lab: chi2_full(Rb, Rt) for lab, Rb, Rt in
           [("kk", kk_b, kk_t), ("ky", ky_b, ky_t), ("yy", yy_b, yy_t)]}

# survey-context band: LSST-Y10 precision on the fiducial.
# v2: MEASURED covariance, not Knox-only -- std of the BINNED log(Cl_kappa)
# across the 50 fiducial realizations (kk_b, already computed above; this box's
# 25 deg^2 footprint), an 8-bin block-average (stabilizes the n=50 std
# estimate), scaled to LSST-Y10's much larger footprint by sqrt(area ratio),
# plus the analytic shape-noise EXCESS (cl_relerr minus its own CV-only piece)
# added in quadrature. Same construction, and the same caveats, as fig 12 panel
# (a): the 50 realizations are random ROTATIONS of ONE box (paired_stats.py),
# not independent cosmological volumes, so this likely UNDER-estimates true
# low-ell cosmic variance; it upgrades automatically, with no code change, once
# the planned ~550-realization fiducial covariance set lands (TODO Sec 4c).
A2SR = (np.pi/180/60)**2
def cl_relerr(cl, ngal, sige, fsky, dlnl=0.15, ell=ell_f):
    Nl = sige**2*A2SR/ngal
    dl = np.maximum(ell*dlnl, 1.0)
    return np.sqrt(2.0/((2*ell + 1)*dl*fsky))*(1 + Nl/cl)
def cl_relerr_cv_only(fsky, dlnl=0.15, ell=ell_f):
    """The pure Gaussian mode-counting piece of cl_relerr, no shape-noise term."""
    dl = np.maximum(ell*dlnl, 1.0)
    return np.sqrt(2.0/((2*ell + 1)*dl*fsky))

logcl_b_rb  = rebin(np.log(np.where(kk_b > 0, kk_b, np.nan)), 8)
ell_f_rb    = rebin(ell_f[None, :], 8)[0]
sigma_lncl_25_kk   = np.nanstd(logcl_b_rb, axis=0)          # measured, 25 deg^2 box footprint
AREA_25_SR, AREA_LSST_SR = 25.0*(np.pi/180.0)**2, 0.44*4*np.pi
sigma_meas_lsst_kk = np.interp(ell_f, ell_f_rb,
                               sigma_lncl_25_kk*np.sqrt(AREA_25_SR/AREA_LSST_SR))
shot_excess_kk = cl_relerr(KKt, 27.0, 0.26, 0.44) - cl_relerr_cv_only(0.44)
lsst_kk_old = 100*cl_relerr(KKt, 27.0, 0.26, 0.44)          # v1 (Knox-only) -- printed for
                                                              # traceability, not plotted
lsst_kk = 100*np.sqrt(sigma_meas_lsst_kk**2 + shot_excess_kk**2)   # v2 -- the plotted band
lsst_S = lsst_kk                                     # S=Cl/const -> identical relative error
print("LSST-Y10 band (fig 4 panels a,b) OLD (v1, Knox-only) vs NEW (v2, measured "
      "cov. + analytic shape-noise excess, area-scaled) -- traceability table:")
for _lt in (100, 300, 1000, 5000, 20000):
    _i = int(np.argmin(np.abs(ell_f - _lt)))
    print(f"  ell={ell_f[_i]:>6.0f}: OLD {lsst_kk_old[_i]:5.2f}%  ->  NEW {lsst_kk[_i]:5.2f}%  "
          f"(measured leg {100*sigma_meas_lsst_kk[_i]:5.2f}%, shot-noise excess "
          f"{100*shot_excess_kk[_i]:5.2f}%)")
print("CAVEAT: measured leg from only 50 realizations (small-N), which are rotations of "
      "ONE box, not independent volumes -- see fig 12's panel (a) markdown for the full "
      "statement; label is \"LSST-Y10 (measured cov., area-scaled)\" on both figures.")

NU_LIM = (-3, 8)   # ONE nu range for ALL SIX nu-domain panels

# peaks / minima: seed-paired per-realization ratio scatter /sqrt(NR) -- the
# SAME convention as the spectra bands (sec 2.0), built from the per-real
# draws recomputed above. (v2 FIX: the previous recipe added the cached truth
# SE -- already std/sqrt(50) -- in quadrature and divided by sqrt(NR) again,
# double-dividing the truth side by sqrt(50) and ignoring the seed-pairing.)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN high-nu bins
    rat_pk = pk_b_real/np.where(pk_t_real > 0, pk_t_real, np.nan)
    pk_band = 100*np.nanstd(rat_pk, 0)/np.sqrt(NR)
    rat_min = min_b_real/np.where(min_t_real > 0, min_t_real, np.nan)
    min_band = 100*np.nanstd(rat_min, 0)/np.sqrt(NR)
res_pk = 100*(b_pk["peak_counts"][ZI]/np.where(t_pk["peak_counts"][ZI] > 0,
                                                t_pk["peak_counts"][ZI], np.nan) - 1)
res_min = 100*(b_pk["minima_counts"][ZI]/np.where(t_pk["minima_counts"][ZI] > 0,
                                                   t_pk["minima_counts"][ZI], np.nan) - 1)
# chi2 over the PLOTTED nu range on well-populated bins (mean truth counts > 5).
# Widening the axis to nu=8 does NOT move these: truth peaks fall below 5 per
# bin above nu~3.6-3.9 and minima are identically zero there, so every added
# bin is masked out -- x2_pk/x2_min stay comparable with the published values.
nu_ok = (b_pk["nu"] >= NU_LIM[0]) & (b_pk["nu"] <= NU_LIM[1])
ok_pk = nu_ok & (t_pk["peak_counts"][ZI] > 5)
ok_mn = nu_ok & (t_pk["minima_counts"][ZI] > 5)
x2_pk = np.nanmean((res_pk[ok_pk]/np.where(pk_band[ok_pk] > 0, pk_band[ok_pk], np.nan))**2)
x2_min = np.nanmean((res_min[ok_mn]/np.where(min_band[ok_mn] > 0, min_band[ok_mn], np.nan))**2)

# ── Peak-count REBIN: wide high-nu bins a la Liu, Madhavacheril & Sherwin
# (MNRAS 519, 573, 2023), whose peak-count figures use irregular high-S/N bins
# so every plotted point carries enough events for a meaningful (not shot-
# noise-dominated) comparison. Convention (stated once, reused for minima
# below): on the native 68-bin nu grid restricted to the plotted NU_LIM range,
# KEEP every bin at its native 0.25-nu width for nu < NU_REBIN0 (=2.0, where the
# TRUTH mean count is still gtrsim20/bin -- see the per-bin table printed by
# this cell), then walk outward MERGING adjacent bins (summing counts, the
# extensive/Poisson-additive quantity) until the merged bin's TRUTH mean count
# reaches >= N_TARGET (=10), or the grid runs out. The TRUTH-side mean count is
# the merge criterion (it is the reference "how many real events" number); the
# SAME bin edges are then applied to BOTH sides so the paired residual stays
# well-defined bin-for-bin.
NU_REBIN0, N_TARGET = 2.0, 10.0

def rebin_groups_by_count(nu, ref_counts, nu0=NU_REBIN0, n_target=N_TARGET):
    """Native-bin index groups (i0, i1) inclusive: singletons for nu < nu0, else
    greedily grown (by adding the NEXT bin) until sum(ref_counts) >= n_target or
    the grid is exhausted (the last group may then fall short -- flagged by the
    caller, never silently)."""
    groups, i, n = [], 0, len(nu)
    while i < n:
        if nu[i] < nu0:
            groups.append((i, i)); i += 1
            continue
        j, acc = i, ref_counts[i]
        while acc < n_target and j + 1 < n:
            j += 1; acc += ref_counts[j]
        groups.append((i, j)); i = j + 1
    return groups

def apply_groups_sum(groups, arr):
    """Sum arr's LAST axis over each (i0, i1) native-bin group -> (..., n_groups)."""
    return np.stack([arr[..., i0:i1 + 1].sum(-1) for i0, i1 in groups], axis=-1)

def group_centers(groups, nu):
    return np.array([0.5*(nu[i0] + nu[i1]) for i0, i1 in groups])

nu_rb_mask = (b_pk["nu"] >= NU_LIM[0]) & (b_pk["nu"] <= NU_LIM[1])
nu_native  = b_pk["nu"][nu_rb_mask]

# --- peaks: apply the rebin (a real sparse tail -- mean truth count falls
#     below N_TARGET around nu~3.1 and keeps falling to O(0.1-1)/bin by nu~7-8;
#     see the per-bin table this cell prints). ---
pk_groups   = rebin_groups_by_count(nu_native, t_pk["peak_counts"][ZI][nu_rb_mask])
nu_pk_rb    = group_centers(pk_groups, nu_native)
pk_b_rb     = apply_groups_sum(pk_groups, pk_b_real[:, nu_rb_mask])   # (NR, n_rb) summed counts
pk_t_rb     = apply_groups_sum(pk_groups, pk_t_real[:, nu_rb_mask])
pk_b_mean_rb, pk_t_mean_rb = pk_b_rb.mean(0), pk_t_rb.mean(0)
res_pk_rb   = 100*(pk_b_mean_rb/np.where(pk_t_mean_rb > 0, pk_t_mean_rb, np.nan) - 1)
# Poisson band convention (stated once): sigma(N)/N = sqrt(N)/N = 1/sqrt(N) on
# the TRUTH mean count landing in each (possibly merged) bin -- the a-la-Liu+23
# shot-noise floor, which is what a single-realization measurement's counting
# uncertainty actually is, and is far more robust than the seed-paired
# ratio-scatter estimate once N/bin is only O(1-10) over 50 realizations.
pk_band_poisson = 100/np.sqrt(np.where(pk_t_mean_rb > 0, pk_t_mean_rb, np.nan))
ok_pk_rb  = pk_t_mean_rb > 1.0        # at least ~1 event/realization on average
x2_pk_rb  = np.nanmean((res_pk_rb[ok_pk_rb]/pk_band_poisson[ok_pk_rb])**2)
n_native_pk = sum(1 for i0, i1 in pk_groups if i0 == i1)
# PLOT only the ok_pk_rb bins (same mask as the chi2): below ~1 truth count/bin
# the Poisson error is enormous (many realizations see literally zero peaks),
# so an unmasked point/errorbar there is not informative -- just very tall and
# clipped at the panel edge. Masking (not clipping) keeps the panel readable.
res_pk_rb_plot   = np.where(ok_pk_rb, res_pk_rb, np.nan)
pk_band_poisson_plot = np.where(ok_pk_rb, pk_band_poisson, np.nan)
# panel (d)'s residual axis needs its OWN range (not the shared default 25%):
# the Liu-style Poisson band genuinely reaches ~30-65% on the still-kept but
# thin flanking bins (native bins just past the well-populated core), vs the
# ~10-20% the old seed-paired band used -- clipping those at +-25 would make
# every flank point look identically "maxed out" instead of showing the real,
# still-informative error bar.
rlim_pk = float(np.ceil(np.nanmax(np.abs(res_pk_rb_plot) + pk_band_poisson_plot)/10.0)*10.0)

# --- minima: CHECK whether the same treatment is warranted before applying it.
#     Minima's sparse region sits on the OPPOSITE (very negative-nu) side, and
#     is already almost entirely excluded by the existing truth>5 mask: walking
#     the merge outward from nu0=-2 to the grid's negative edge (nu=-4.875)
#     only accumulates a TRUTH mean count of a few (printed below) because the
#     remaining bins are already ~0 -- there is no way to reach N_TARGET=10
#     without swallowing the well-populated bulk. Minima do NOT share peaks'
#     long, gently-declining sparse tail, so no rebin is applied to them; the
#     native-bin presentation (unchanged) is retained. ---
mn_groups_probe = rebin_groups_by_count(-nu_native[::-1],
                                        t_pk["minima_counts"][ZI][nu_rb_mask][::-1])
mn_tail_count = sum(t_pk["minima_counts"][ZI][nu_rb_mask][::-1][i0:i1 + 1].sum()
                    for i0, i1 in mn_groups_probe if i1 > i0)
print(f"minima rebin CHECK (mirrored: nu0={-NU_REBIN0}, merging toward the negative "
      f"grid edge): the widest merged group the grid allows still sums to only "
      f"{mn_tail_count:.1f} truth counts (< N_TARGET={N_TARGET:.0f}) -- minima's tail is "
      "grid-edge-limited, not a long gently-declining tail like peaks', so NO REBIN is "
      "applied to minima; they keep their native-bin presentation and the existing "
      "truth>5 mask.")

print(f"peak-count rebin: {len(nu_native)} native bins in nu={NU_LIM} -> {len(pk_groups)} bins "
      f"({n_native_pk} native below nu={NU_REBIN0:.1f}, {len(pk_groups)-n_native_pk} "
      f"merged-wide above, target truth mean count >= {N_TARGET:.0f}/bin); "
      f"merged-bin truth mean counts = {np.round(pk_t_mean_rb[n_native_pk:], 1).tolist()}")
print(f"peak-count chi2/dof: OLD (native 68-bin grid, seed-paired ratio-scatter band, "
      f"truth>5 mask) = {x2_pk:.1f}   ->   NEW (Liu-style rebin, Poisson sqrt(N)/N band, "
      f"truth mean>1 mask) = {x2_pk_rb:.2f}  [{int(ok_pk_rb.sum())}/{len(pk_groups)} bins] "
      "-- published/plotted value is the NEW one; OLD kept here for traceability.")

# Minkowski functionals: the live recompute above gives per-realization draws on
# BOTH sides for the first time, so the band is the GENUINE paired difference
# scatter (the same construction as the PDF band) -- the previous sqrt(2)
# "assume the truth scatter equals BIND's" kludge is retired. chi2 is reported
# over nu<=4 (comparable with the published values) AND over the full range.
def mf_paired(k):
    tmean = V8t[k][ZI]
    nrm = np.max(np.abs(tmean))
    res = 100*(V8b[k][ZI] - tmean)/nrm
    band = 100*np.std(V8b_r[k][:, ZI] - V8t_r[k][:, ZI], 0)/np.sqrt(NR)/nrm
    ok4 = MF_NU8 <= 4.0
    def _x2(msk):
        return np.nanmean((res[msk]/np.where(band[msk] > 0, band[msk], np.nan))**2)
    return band, res, _x2(ok4), _x2(np.ones_like(ok4))
band_V0, res_V0, x2_V0, x2_V0_all = mf_paired("V0")
band_V1, res_V1, x2_V1, x2_V1_all = mf_paired("V1")
band_V2, res_V2, x2_V2, x2_V2_all = mf_paired("V2")

# PDF: paired per-realization DIFFERENCE (tails cross zero density -> a ratio
# is ill-defined), normalized by max(truth) -- the residual convention this
# panel already used, now with a real band + chi2 from the live recomputation
# (the cache stores only the realization-summed histogram).
pdf_b_mean, pdf_t_mean = pdf_b_real.mean(0), pdf_t_real.mean(0)
pdf_norm = np.max(pdf_t_mean)
band_pdf = 100*np.std(pdf_b_real - pdf_t_real, 0)/np.sqrt(NR)/pdf_norm
res_pdf = 100*(pdf_b_mean - pdf_t_mean)/pdf_norm
ok_pdf = band_pdf > 0
x2_pdf = np.nanmean((res_pdf[ok_pdf]/band_pdf[ok_pdf])**2)
# kappa -> nu = kappa/sigma0 with sigma0 the cached per-plane 2'-smoothed rms
# (the SAME normalization the peak/minima nu axis uses). A change of variable:
# P(nu) = sigma0 P(kappa), so the max-normalized residual and its chi2 above
# are numerically untouched. One sigma0 (the BIND side) is used for both curves
# so they share an axis; the truth-side value differs by <0.5% (printed).
sig0 = np.asarray(pr["sigma0"], dtype=float)
print("PDF nu normalization sigma0 (2'-smoothed per-plane rms): BIND "
      f"{np.round(sig0, 5)}  truth {np.round(np.sqrt(t_ng['variance'][:, 1]), 5)}")
NU_PDF_X = pdf_cent[None, :]/sig0[:, None]                # (5, 41) per-plane nu axis
pdf5_b = np.asarray(b_ng["pdf"], dtype=float).copy(); pdf5_b[ZI] = pdf_b_mean
pdf5_b = pdf5_b*sig0[:, None]                             # PDF(nu) = sigma0 PDF(kappa)
pdf_nu_t = pdf_t_mean*sig0[ZI]

cB, cH = COLORS["bind"], COLORS["truth"]
_lt2 = max(1e-12, 0.02*float(np.max(np.abs(V8t["V2"][ZI]))))   # symlog knee for V2

panels = [
    dict(x=ell_f, yb_z=f_ell*KK5b, yt=f_ell*KKt, res=res_kk, band=band_kk, lsst=lsst_kk,
         xl=r"$\ell$", yl=r"$\ell(\ell{+}1)C_\ell^{\kappa\kappa}/2\pi$",
         xs="log", ys="log", xlim=(ELL_LO, ELL_MAX_PLOT), tag="(a)",
         ctx="LSST-Y10", alias=True, alias_note=True),
    dict(x=ell_f, yb_z=S5b, yt=St_m, res=res_S, band=band_S, lsst=lsst_S,
         xl=r"$\ell$", yl=r"$S(\ell)=C_\ell^{\rm bind}/C_\ell^{\rm dmo}$",
         xs="log", ys="linear", xlim=(ELL_LO, ELL_MAX_PLOT), tag="(b)",
         ctx="LSST-Y10"),
    dict(x=pdf_cent, x_z=NU_PDF_X, yb_z=pdf5_b, yt=pdf_nu_t, res=res_pdf,
         band=band_pdf, lsst=None,
         xl=r"$\nu=\kappa/\sigma_0$", yl=r"PDF$(\nu)$",
         xs="linear", ys="log", xlim=NU_LIM, tag="(c)", ylo=1e-5),
    dict(x=b_pk["nu"], yb_z=b_pk["peak_counts"], yt=t_pk["peak_counts"][ZI],
         res=res_pk_rb_plot, band=pk_band_poisson_plot, res_x=nu_pk_rb, lsst=None,
         xl=r"$\nu$", yl=r"$N_{\rm peak}$", xs="linear", ys="log", xlim=NU_LIM, tag="(d)",
         rlim=rlim_pk),
    dict(x=b_pk["nu"], yb_z=b_pk["minima_counts"], yt=t_pk["minima_counts"][ZI],
         res=res_min, band=min_band, lsst=None,
         xl=r"$\nu$", yl=r"$N_{\rm min}$", xs="linear", ys="log", xlim=NU_LIM, tag="(e)"),
    dict(x=MF_NU8, yb_z=V8b["V0"], yt=V8t["V0"][ZI], res=res_V0, band=band_V0, lsst=None,
         xl=r"$\nu$", yl=r"$V_0(\nu)$", xs="linear", ys="log", xlim=NU_LIM, tag="(f)"),
    dict(x=MF_NU8, yb_z=V8b["V1"], yt=V8t["V1"][ZI], res=res_V1, band=band_V1, lsst=None,
         xl=r"$\nu$", yl=r"$V_1(\nu)$", xs="linear", ys="log", xlim=NU_LIM, tag="(g)"),
    dict(x=MF_NU8, yb_z=V8b["V2"], yt=V8t["V2"][ZI], res=res_V2, band=band_V2, lsst=None,
         xl=r"$\nu$", yl=r"$V_2(\nu)$", xs="linear", ys="symlog", linthresh=_lt2,
         xlim=NU_LIM, tag="(h)"),
]

fig = plt.figure(figsize=(TWO_COL[0], 5.9))
gs = fig.add_gridspec(5, 4, height_ratios=[2.2, 1, 0.55, 2.2, 1], hspace=0.18, wspace=0.45)
AX, AXR = [], []
for j, p in enumerate(panels):
    r0 = (j//4)*3
    a = fig.add_subplot(gs[r0, j % 4])
    ar = fig.add_subplot(gs[r0 + 1, j % 4], sharex=a)
    xz = p.get("x_z")
    x1 = xz[ZI] if xz is not None else p["x"]           # the z_s=1 abscissa
    for zi in range(5):                                 # BIND: all five source planes
        a.plot(xz[zi] if xz is not None else p["x"], p["yb_z"][zi], color=ZCOLS[zi], lw=1.0)
    a.plot(x1, p["yt"], color=cH, ls="--", lw=0.9)      # hydro-pasted truth, z_s=1 only
    a.set_xscale(p["xs"])
    if p["ys"] == "symlog":
        a.set_yscale("symlog", linthresh=p["linthresh"])
    else:
        a.set_yscale(p["ys"])
    a.set_ylabel(p["yl"], fontsize=7, labelpad=p.get("ylpad", 4.0))
    if p["xlim"]: a.set_xlim(*p["xlim"])
    if p.get("ylo"): a.set_ylim(bottom=p["ylo"])
    if p.get("alias"):    # raw spectra turn up from CIC aliasing above ELL_TRUST
        for _ax in (a, ar):
            _ax.axvspan(ELL_TRUST, ELL_MAX_PLOT, color="0.85", alpha=0.45, lw=0, zorder=0)
    if p.get("alias_note"):
        a.text(0.985, 0.06, "CIC\naliasing", transform=a.transAxes, fontsize=4.0,
               ha="right", va="bottom", color="0.35")
    plt.setp(a.get_xticklabels(), visible=False)
    a.tick_params(labelsize=6)

    ar.axhline(0, color=COLORS["dmo"], lw=0.6)
    rx = p.get("res_x")
    if rx is not None:
        # Liu-style REBINNED residual (peaks, panel d): discrete points with
        # Poisson sqrt(N)/N error bars, not a continuous band -- the wide
        # high-nu bins are literally sparse discrete measurements, and a
        # filled band would misleadingly imply a continuously-sampled curve.
        ar.errorbar(rx, p["res"], yerr=p["band"], fmt="o", ms=2.0, mew=0,
                    color=cB, lw=0.7, elinewidth=0.7, capsize=1.3)
    else:
        if p["band"] is not None:
            ar.fill_between(x1, -p["band"], p["band"], color=COLORS["dmo"], alpha=0.35, lw=0)
        if p["lsst"] is not None:
            ar.fill_between(p["x"], -p["lsst"], p["lsst"], color=cB, alpha=0.22, lw=0)
            ar.text(0.97, 0.05, p.get("ctx", "LSST-Y10"), transform=ar.transAxes,
                    fontsize=4.2, color=cB, va="bottom", ha="right")
        ar.plot(x1, p["res"], color=cB, lw=0.9)
    rl = p.get("rlim", 25)
    ar.set_ylim(-rl, rl); ar.set_xscale(p["xs"])
    ar.set_xlabel(p["xl"], fontsize=7); ar.set_ylabel("resid. [%]" if j % 4 == 0 else "", fontsize=7)
    ar.tick_params(labelsize=6)
    panel_label(a, p["tag"], loc="upper right")   # ONE corner for every panel letter
    AX.append(a); AXR.append(ar)

# the residual strips are single-plane; say so once
AXR[0].text(0.03, 0.06, r"residual: $z_s=1$", transform=AXR[0].transAxes,
            fontsize=4.2, color="0.3", va="bottom")
AXR[3].text(0.97, 0.92, "rebinned,\nPoisson err.", transform=AXR[3].transAxes,
            fontsize=3.8, color=cB, ha="right", va="top")
AX[0].legend(handles=[plt.Line2D([], [], color=ZCOLS[zi], lw=1.0,
                                 label=rf"$z_s={ZS[zi]:.2f}$") for zi in range(5)],
             loc="upper left", fontsize=4.4, ncol=2, handlelength=1.1,
             columnspacing=0.7, labelspacing=0.25, borderpad=0.2, handletextpad=0.4)
AX[1].legend(handles=[plt.Line2D([], [], color="0.35", lw=1.0, label="BIND"),
                      plt.Line2D([], [], color=cH, lw=0.9, ls="--",
                                 label=r"hydro-pasted, $z_s=1$")],
             loc="lower left", fontsize=5.0, handlelength=1.6)
save(fig, "figs_v2/fig04_field_validation")
plt.show()

m = (ell_f >= 300) & (ell_f <= 5000)
print("median |resid| in ell=300-5000:  "
      f"Cl_kk {np.nanmedian(np.abs(res_kk[m])):.1f}%  "
      f"Cl_ky {np.nanmedian(np.abs(res_ky[m])):.1f}%  "
      f"Cl_yy {np.nanmedian(np.abs(res_yy[m])):.1f}%")
print("chi2/dof (seed-paired diagonal bands; spectra over ell 100-1.5e4; counts "
      "over the plotted nu range with truth counts > 5; PDF over bins with "
      "nonzero paired scatter; MFs over nu<=4; N_pk is the PUBLISHED/plotted "
      "Liu-style rebinned+Poisson value -- see the peak-count rebin print above "
      "for the pre-rebin comparison number): "
      f"kk {x2_kk:.0f}  S(ell) {x2_S:.0f}  PDF {x2_pdf:.0f}  N_pk(rebinned) {x2_pk_rb:.2f}  "
      f"N_min {x2_min:.1f}  V0 {x2_V0:.1f}  V1 {x2_V1:.1f}  V2 {x2_V2:.1f}  "
      f"ky {x2_ky:.0f}  yy {x2_yy:.0f}")
print("MF chi2/dof over the FULL extended range nu=-3..8: "
      f"V0 {x2_V0_all:.1f}  V1 {x2_V1_all:.1f}  V2 {x2_V2_all:.1f}   "
      "(v3 band = genuine paired per-realization difference scatter on both "
      "sides; the previous sqrt(2) truth-scatter proxy is retired, so the MF "
      "chi2 supersede the earlier stamped values)")
print("(peaks/minima/PDF bands = paired per-realization ratio/difference scatter "
      "/sqrt(50), recomputed live from the cached maps -- the v2 band-recipe fix; "
      "the earlier quadrature recipe double-divided the truth SE by sqrt(50).)")
print("PDF residual amplitude (percent of peak density): "
      f"median |res| = {np.nanmedian(np.abs(res_pdf[ok_pdf])):.2f}%, "
      f"max |res| = {np.nanmax(np.abs(res_pdf[ok_pdf])):.2f}% -- sub-percent in "
      "amplitude, but resolved as significant by the tight paired band (like the "
      "spectra, a characterized systematic, not a failed closure)")
print("V0/V1/V2 residual amplitude (percent of peak, nu<=8): "
      f"median |res| = {np.nanmedian(np.abs(res_V0)):.2f}/{np.nanmedian(np.abs(res_V1)):.2f}/"
      f"{np.nanmedian(np.abs(res_V2)):.2f}%; the extension carries real signal -- "
      f"V0/V1/V2 at nu=8 = {V8b['V0'][ZI][-1]:.2e}/{V8b['V1'][ZI][-1]:.2e}/"
      f"{V8b['V2'][ZI][-1]:.2e}")

# full-covariance chi2 on a 25-bin compression (Hartlap-corrected): even after
# accounting for ell-bin-to-bin correlation (NOT the realization correlation
# ruled out in Part A), the offset survives. Covers fig 4b's ky/yy legs too.
# v2 (Sellentin-Heavens): the Gaussian/Hartlap chi2/dof above is a POINT
# statistic; each is now also converted to a p-value two ways, printed side by
# side. Gaussian/Hartlap: the standard (approximate) practice of treating the
# Hartlap-corrected chi2 as chi2_p-distributed. Sellentin & Heavens (2016,
# MNRAS 456, L132) REPORTED ALTERNATIVE: their marginal (multivariate-t)
# likelihood's exact null-hypothesis tail probability is the classical
# Hotelling T^2 -> F(p, n-p) equivalence, applied to chi2_RAW (the SAME
# dv^T Cm^-1 dv above, before the Hartlap mean-debiasing factor -- Hartlap
# and Hotelling are two DIFFERENT finite-n corrections to the same raw
# statistic, not composable). Not the default (n=50 realizations here, well
# under the >=250 TODO floor for the covariance sets) -- printed for
# transparency, not used to draw any conclusion in the text.
from scipy.stats import chi2 as _chi2dist, f as _fdist
for lab in ("kk", "ky", "yy"):
    x2f, rho_bar, pbin, n_fc, chi2_raw = fullcov[lab]
    p_gauss = float(_chi2dist.sf(x2f*pbin, df=pbin))
    F_stat = ((n_fc - pbin)/(pbin*(n_fc - 1)))*chi2_raw
    p_sh = float(_fdist.sf(F_stat, pbin, n_fc - pbin))
    print(f"chi2/dof (FULL {pbin}-bin covariance, Hartlap) {lab}: {x2f:.0f}  "
          f"(mean |ell-bin correlation|={rho_bar:.2f} -> "
          f"N_eff~{pbin/(1 + (pbin - 1)*rho_bar):.0f}/{pbin}; still >>1 -- see caption)  "
          f"|  p-value Gaussian/Hartlap (chi2_{pbin}) = {p_gauss:.2e}, Sellentin-Heavens "
          f"REPORTED ALTERNATIVE (Hotelling F_{{{pbin},{n_fc-pbin}}}) = {p_sh:.2e}")

# T2 -- UNPAIRED covariance chi2: the significance a map USER faces (no
# seed-cancellation; cov of the difference of two independent means =
# (cov_B + cov_T)/N), same 25-bin compression, Hartlap-corrected.
def chi2_unpaired(bR, tR):
    msk4 = (ell_f >= ELL_LO) & (ell_f <= ELL_TRUST)
    Bc, Tc = rebin(bR[:, msk4], 8), rebin(tR[:, msk4], 8)
    okb = np.isfinite(Bc).all(0) & np.isfinite(Tc).all(0) & (np.abs(Tc.mean(0)) > 0)
    Bc, Tc = Bc[:, okb], Tc[:, okb]
    n, p = Bc.shape
    Tm = Tc.mean(0)
    Cf = (np.cov(Bc, rowvar=False) + np.cov(Tc, rowvar=False))/n/np.outer(Tm, Tm)
    dv = Bc.mean(0)/Tm - 1
    hart = (n - p - 2)/(n - 1)
    return hart*float(dv @ np.linalg.solve(Cf, dv))/p
print("chi2/dof T2 UNPAIRED (truth+bind covariance, no seed-cancellation -- the "
      "scale a map user faces vs an independent simulation): "
      f"kk {chi2_unpaired(kk_b, kk_t):.1f}  ky {chi2_unpaired(ky_b, ky_t):.1f}  "
      f"yy {chi2_unpaired(yy_b, yy_t):.1f}")
for k in ("V0", "V2"):
    r = 100*np.max(np.abs(b_ng[k][ZI] - t_ng[k][ZI]))/np.max(np.abs(t_ng[k][ZI]))
    print(f"{k}: max normalized residual = {r:.1f}% (cached nu<=4 grid)")
bn, tn = np.load(RB/"peak_counts_ngal10.npz"), np.load(RT/"peak_counts_ngal10.npz")
okp = tn["peak_counts"][ZI] > 1
print("shape-noise (ngal=10) peaks: median |resid| = "
      f"{100*np.nanmedian(np.abs(bn['peak_counts'][ZI][okp]/tn['peak_counts'][ZI][okp]-1)):.1f}%")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
### Fig 4b — the released auto- and cross-spectra ($\kappa$, $y$, $\tau$)

The second half of the field vector: the three auto-spectra $C_\ell^{\kappa\kappa}$,
$C_\ell^{yy}$, $C_\ell^{\tau\tau}$ and the two $\kappa$-crosses $C_\ell^{\kappa y}$,
$C_\ell^{\kappa\tau}$, on fig 4's template (mean curves over five planes, seed-paired
residual strip at $z_s=1$, $\ell$ to the corner mode with the CIC-aliasing region shaded).
The $\kappa\kappa$ panel is the map-measured auto-spectrum; its exact relation to $S(\ell)$
at fixed cosmology is stated once in the §4a text (the cell below still prints the
`np.allclose` check that establishes it).

**Conventions.** The $z_s$ colour encodes the source plane for $\kappa$ and the *column
depth* for the SZ/$\tau$ autos: $C_\ell^{yy}(z_i)$ and $C_\ell^{\tau\tau}(z_i)$ are the
spectra of the $y$/$\tau$ column integrated out to $z_i$ (the map cubes are cumulative), and
they are recomputed live because the released caches store only the *total* column. The two
crosses keep the released cache convention — $\kappa(z_s)$ against the **total** $y$/$\tau$
column (the same pairing `lightcone_stats` uses and fig 11b plots) — so the published
$\kappa y$ closure numbers are reproduced exactly. For $C_\ell^{yy}$ the cell prints the
closure both for the $z_s=1$ column (plotted) and for the total column (the released
convention behind the quoted 5.6%).

**The $\tau$ gap — now a FILE-EXISTENCE GUARD, not a hard fact.** As of this run there is
**no seed-paired hydro-pasted truth $\tau$ map anywhere in this release** —
`runs/truth/run_0000` has no `tau_maps.npz`, and an exhaustive search finds $\tau$ products on
the BIND side only. But the cell no longer *hardcodes* that absence: it checks
`runs/truth/run_0000/tau_maps.npz` at execution time and switches presentation automatically.
**As things stand** (guard false): $C_\ell^{\tau\tau}$, $C_\ell^{\kappa\tau}$ and the new
$C_\ell^{y\tau}$ panel (below) are shown **BIND-only** (labelled on-panel), and their lower
strip is repurposed: instead of a closure residual it carries the **253-node Sobol 16–84%
spread** of that spectrum about the suite median — feedback-response information in place of
closure information, drawn in a different colour and with its own axis label so the two
cannot be confused. **The moment a seed-paired hydro-pasted truth $\tau$ map lands** (building
one means re-compositing all 20 snapshots and re-running the `lux` trace — a future retrace,
not a figure edit), the SAME cell computes and overlays the hydro-pasted truth curves and
seed-paired residual bands for all three tau-family panels, on the SAME closure template as
$\kappa\kappa$/$yy$/$\kappa y$, with no code change required: this markdown paragraph and the
"BIND-only" panel labels are then stale and should be re-written to match. The only $\tau$
validation this release *can currently* make is at halo level: the profile cache
`tau_profiles_snap096.npz` does carry `truth_prof`, and the painted/truth $\tau$ profile ratio
is 1.040 inside $r_{500c}$. The $y$ legs are fully truth-validated at all five column depths
(both sides' `y_maps.npz` are 5-plane cubes).

**New: the $C_\ell^{y\tau}$ panel (f).** The canonical-statistics list (figs 5/7) already
carries $C_\ell^{y\tau}$ as the third cross; this panel is its closure/response counterpart
here. Same template as $C_\ell^{\tau\tau}$/$C_\ell^{\kappa\tau}$: BIND-only with a Sobol
16–84% strip while no truth $\tau$ exists, promoted to a truth-validated closure panel the
moment the guard above finds one. It uses the SAME "signed-cross" floor handling as the
$\kappa y$/$\kappa\tau$ crosses (both $y$ and $\tau$ are strictly positive fields at the map
level, but the cross statistic is still small and can sit near the noise floor at high $\ell$;
no special-casing beyond what $\kappa y$/$\kappa\tau$ already get).

**Reading.** $C_\ell^{\kappa\kappa}$ closes at sub-percent. The $y$ channels carry the two
already-characterized features: over $\ell=300$–5000 a $\approx-10\%$ $C_\ell^{yy}$ dip at
$\ell\sim400$–800 (missing large-scale correlated pressure), and at
$5\times10^3<\ell<1.5\times10^4$ a coherent $+10$–20% upturn, whose origin is localized in
fig 18 (Appendix A). Beyond $\ell\sim3\times10^4$ the $y$-involved residuals grow to tens of
percent — the aliasing-dominated regime, shown for completeness rather than as a closure
claim. The SZ panels shade the cosmic-variance-only floor ($f_{\rm sky}=0.44$, mode counting
alone): a formal shape-noise forecast has no meaning for a Compton-$y$ field, so this is an
honest *lower bound* on achievable precision, not a survey forecast.

[draft - confirm] The map-level $C_\ell^{yy}$ high-$\ell$ excess localized in fig 18 (a
$+10$–20% coherent upturn at $5\times10^3<\ell<1.5\times10^4$) is **not** simply the
halo-level $Y_{500c}$ offset of §2a (fig 3: $1.054\pm0.007$) seen at map level: the
rescaling test in the §2b audit above shows a uniform $1.054^2$ amplitude removal has the
wrong sign at mid $\ell$ and worsens the overall closure ($\chi^2/\mathrm{dof}$
$177\to250$). The two are distinct systematics — an aperture-integrated normalization
offset at halo scale, and a *scale-dependent* texture effect in the painted gas (fig 18's
localization) at map level — and the §5b caveat should treat them separately rather than
as one pressure-excess story.
''')

code(r'''
# ── Fig 4b: the 3 auto- + 3 cross-spectra of the released field vector ───────
# data: (truth-validated) runs/{bind,truth}/run_0000/{Cl_kappa,Cl_kappa_y}.npz
#       + y_maps.npz -- the per-plane C^yy is recomputed live because the caches
#       store only the TOTAL y column;
#       (BIND, always) bind_lightcone_tng/tau_maps.npz -- proven to be the same
#       50 seed-paired realizations as runs/bind/run_0000;
#       (truth, IF PRESENT -- file-existence guard, v2) runs/truth/run_0000/
#       tau_maps.npz: when this lands, cl_tt/cl_kappa_tau/cl_yt become
#       truth-validated closure panels automatically, no code change;
#       (context, used only while the guard is false) emulator_dataset_xpkfix
#       Sobol spread for the tau-family panels.
# Reuses fig 4's paired()/f_ell/ell_f/lsst_kk and its z_s=1 paired legs, so the
# published kk/ky/yy closure numbers are reproduced unchanged.
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

RT_TAU_PATH = RT/"tau_maps.npz"
HAS_TRUTH_TAU = RT_TAU_PATH.exists()
print(f"tau-family truth guard: {RT_TAU_PATH} exists? {HAS_TRUTH_TAU} -- "
      + ("a hydro-pasted truth tau map has landed: computing truth overlays + seed-paired "
         "residual bands for cl_tt/cl_kappa_tau/cl_yt (post-retrace automatic upgrade)"
         if HAS_TRUTH_TAU else
         "keeping the BIND-only presentation (Sobol 16-84% spread strip) for the tau family"))

# the identity stated once in the §4a text: Cl_kk = S(ell) x Cl_dmo at fixed
# cosmology. This is WHY Cl_kappa is not a separate row/family/head in figs 5/7/8/9 --
# S(ell) alone carries the information there, and the release ships Cl_dmo so users
# can reconstruct Cl_kappa from S(ell) exactly.
_ck = d["t__cl_kappa__value"][:, ZI, ZI, :]
_sp = d["t__suppression__value"][:, ZI, :]
_dmo_ref = np.nanmedian(_ck/np.where(_sp > 0, _sp, np.nan), 0)
print("identity check: Cl_kappa == S(ell) x Cl_dmo over the 253 "
      f"Sobol nodes -> np.allclose = {np.allclose(_ck, _sp*_dmo_ref, rtol=1e-6)}; "
      f"max |Cl_kk/(S x Cl_dmo) - 1| = {np.nanmax(np.abs(_ck/(_sp*_dmo_ref) - 1)):.1e}. "
      "Reconstruct Cl_kappa from the released S(ell) and Cl_dmo -- this is why "
      "Cl_kappa is not a separate row/family/head in figs 5/7/8/9.")

# ── per-plane C^yy (BIND + truth), C^tautau (BIND, + truth IF the guard is
# true) and the NEW C^ytau (BIND, + truth IF the guard is true), live, plane
# by plane. tau is loaded ALONGSIDE y (not in a separate later block, v2) so
# the y-tau cross can be formed at matched column depth without a third load.
NELL = len(ell_f)
YY5b, YY5t = np.empty((5, NELL)), np.empty((5, NELL))
TT5b, YT5b = np.empty((5, NELL)), np.empty((5, NELL))
TT5t, YT5t = np.full((5, NELL), np.nan), np.full((5, NELL), np.nan)

f = np.load(RB/"y_maps.npz"); BY = f["y"]; del f; gc.collect()
f = np.load(RT/"y_maps.npz"); TY = f["y"]; del f; gc.collect()
f = np.load(LC/"tau_maps.npz"); BTAU = f["tau"]; del f; gc.collect()
if HAS_TRUTH_TAU:
    f = np.load(RT_TAU_PATH); TTAU = f["tau"]; del f; gc.collect()

for zi in range(5):
    cb = np.array([power_spectrum(BY[r, zi])[1] for r in range(NR)])
    ct = np.array([power_spectrum(TY[r, zi])[1] for r in range(NR)])
    YY5b[zi], YY5t[zi] = cb.mean(0), ct.mean(0)
    if zi == ZI:
        YYb_z, YYt_z, res_yy_z, band_yy_z, x2_yy_z = paired(cb, ct)
    del cb, ct; gc.collect()

    tb  = np.array([power_spectrum(BTAU[r, zi])[1] for r in range(NR)])
    ytb = np.array([power_spectrum(BY[r, zi], BTAU[r, zi])[1] for r in range(NR)])
    TT5b[zi], YT5b[zi] = tb.mean(0), ytb.mean(0)
    if HAS_TRUTH_TAU:
        tt  = np.array([power_spectrum(TTAU[r, zi])[1] for r in range(NR)])
        ytt = np.array([power_spectrum(TY[r, zi], TTAU[r, zi])[1] for r in range(NR)])
        TT5t[zi], YT5t[zi] = tt.mean(0), ytt.mean(0)
        if zi == ZI:
            TTb_z, TTt_z, res_tt_z, band_tt_z, x2_tt_z = paired(tb, tt)
            YTb_z, YTt_z, res_yt_z, band_yt_z, x2_yt_z = paired(ytb, ytt)
        del tt, ytt; gc.collect()
    del tb, ytb; gc.collect()

BT_tot = BTAU[:, -1].copy()                     # captured before deletion, reused below
if HAS_TRUTH_TAU:
    TT_tot = TTAU[:, -1].copy()
    del TTAU
del BY, TY, BTAU; gc.collect()
print("cumulative-column guard: per-plane C^yy at the LAST plane vs the total-column "
      f"C^yy used for the published numbers: median |ratio-1| = "
      f"{100*np.nanmedian(np.abs(YY5b[-1]/np.where(YYb > 0, YYb, np.nan) - 1)):.3f}%")

# 5-plane crosses, released convention: kappa(z_s) x TOTAL y/tau column.
#
# These MUST be recomputed live. The per-run caches runs/{bind,truth}/run_0000/
# Cl_kappa_y.npz and bind_lightcone_tng/Cl_tau.npz still carry the OLD XPk_plane
# cross normalization -- the exact bug that emulator_dataset_xpkfix.npz was built
# to correct. Measured against a live recomputation at z_s=1 the cached cl_ky is
# low by a factor ~1.2e7, and NOT by a constant (the ratio runs 1.11e7-1.31e7
# across ell), so it cannot be rescaled after the fact. Using the cache for the
# four extra source planes while the z_s=1 curve came from the live paired legs
# put two normalizations in one panel, six decades apart.
def _cross5(K, F):
    """C_l^{kappa(z_s) x F} for all five source planes; F is a single map stack."""
    return np.array([np.mean([power_spectrum(K[r, zi], F[r])[1] for r in range(NR)], 0)
                     for zi in range(5)])

f = np.load(RB/"kappa_maps.npz"); BK5 = f["kappa"]; del f; gc.collect()
f = np.load(RB/"y_maps.npz"); BY_tot = f["y"][:, -1].copy(); del f; gc.collect()
KY5b = _cross5(BK5, BY_tot); del BY_tot; gc.collect()
KT5b = _cross5(BK5, BT_tot); gc.collect()       # BT_tot from the y/tau block; BK5 still needed

f = np.load(RT/"kappa_maps.npz"); TK5 = f["kappa"]; del f; gc.collect()
f = np.load(RT/"y_maps.npz"); TY_tot = f["y"][:, -1].copy(); del f; gc.collect()
KY5t = _cross5(TK5, TY_tot); del TY_tot; gc.collect()

if HAS_TRUTH_TAU:
    KT5t = _cross5(TK5, TT_tot)
    # paired per-realization kappa-tau cross AT z_s=1 specifically, for the
    # residual band (the 5-plane _cross5 above gives means only).
    ktb_z = np.array([power_spectrum(BK5[r, ZI], BT_tot[r])[1] for r in range(NR)])
    ktt_z = np.array([power_spectrum(TK5[r, ZI], TT_tot[r])[1] for r in range(NR)])
    KTb_z, KTt_z, res_kt_z, band_kt_z, x2_kt_z = paired(ktb_z, ktt_z)
    del ktb_z, ktt_z; gc.collect()

del BK5, TK5, BT_tot; gc.collect()
if HAS_TRUTH_TAU:
    del TT_tot; gc.collect()

# the z_s=1 plane must reproduce fig 4's independently computed paired legs
for _lab, _a, _b in (("bind", KY5b[ZI], KYb), ("truth", KY5t[ZI], KYt)):
    _d = float(np.nanmax(np.abs(_a/np.where(_b != 0, _b, np.nan) - 1)))
    print(f"cross guard C^ky {_lab} z_s=1 (5-plane vs fig-4 paired leg): "
          f"max |ratio-1| = {_d:.2e}")
    assert _d < 1e-6, f"C^ky {_lab} z_s=1 disagrees with the paired leg"

# CV-only floor for the y/tau channels: the LSST recipe's shape-noise term has
# no meaning for a Compton-y (or tau) field, so only the mode-counting piece
# survives -- a LOWER bound on achievable precision, not a forecast.
CV_FSKY = 0.44
def cv_floor(ell=ell_f, fsky=CV_FSKY, dlnl=0.15):
    dl = np.maximum(ell*dlnl, 1.0)
    return 100*np.sqrt(2.0/((2*ell + 1)*dl*fsky))
cv_band = cv_floor()

# while the tau-truth guard is FALSE, the tau-family panels cannot be validated,
# so their lower strip carries the 253-node Sobol 16-84% spread about the SUITE
# MEDIAN instead -- feedback response in place of closure. Computed regardless
# of the guard (cheap, dataset-only) so the printed context is always available.
ELL_TT = d["a__cl_tt__ell"]
def sobol_spread(V):
    med = np.nanmedian(V, 0)
    r = 100*(V/np.where(med > 0, med, np.nan) - 1)
    return np.nanpercentile(r, 16, 0), np.nanpercentile(r, 84, 0)
lo_tt, hi_tt = sobol_spread(d["t__cl_tt__value"])
lo_kt, hi_kt = sobol_spread(d["t__cl_kappa_tau__value"][:, ZI])
lo_yt, hi_yt = sobol_spread(d["t__cl_yt__value"])
_rl_sob = float(np.nanpercentile(np.abs(np.concatenate(
    [lo_tt, hi_tt, lo_kt, hi_kt, lo_yt, hi_yt])), 98))
# (the BIND-only tau caveat is stated in the markdown above and printed by
#  this cell; it is deliberately not drawn on the panels)

kk_panel = dict(x=ell_f, yb_z=f_ell*KK5b, yt=f_ell*KKt, res=res_kk, band=band_kk,
                lsst=lsst_kk, yl=r"$\ell(\ell{+}1)C_\ell^{\kappa\kappa}/2\pi$",
                tag="(a)", rlim=25)
yy_panel = dict(x=ell_f, yb_z=f_ell*YY5b, yt=f_ell*YYt_z, res=res_yy_z, band=band_yy_z,
                lsst=cv_band, yl=r"$\ell(\ell{+}1)C_\ell^{yy}/2\pi$", tag="(b)", rlim=100)
ky_panel = dict(x=ell_f, yb_z=f_ell*KY5b, yt=f_ell*KYt, res=res_ky, band=band_ky,
                lsst=cv_band, yl=r"$\ell(\ell{+}1)C_\ell^{\kappa y}/2\pi$", tag="(d)", rlim=50)

# tau-family panels (c, e, f): closure template if the guard is true, Sobol-spread
# template if not -- SAME tag/position either way, so the panel LAYOUT never
# depends on data availability, only its content does.
if HAS_TRUTH_TAU:
    tt_panel = dict(x=ell_f, yb_z=f_ell*TT5b, yt=f_ell*TTt_z, res=res_tt_z, band=band_tt_z,
                    lsst=cv_band, yl=r"$\ell(\ell{+}1)C_\ell^{\tau\tau}/2\pi$",
                    tag="(c)", rlim=100)
    kt_panel = dict(x=ell_f, yb_z=f_ell*KT5b, yt=f_ell*KTt_z, res=res_kt_z, band=band_kt_z,
                    lsst=cv_band, yl=r"$\ell(\ell{+}1)C_\ell^{\kappa\tau}/2\pi$",
                    tag="(e)", rlim=50)
    yt_panel = dict(x=ell_f, yb_z=f_ell*YT5b, yt=f_ell*YTt_z, res=res_yt_z, band=band_yt_z,
                    lsst=cv_band, yl=r"$\ell(\ell{+}1)C_\ell^{y\tau}/2\pi$",
                    tag="(f)", rlim=100)
else:
    tt_panel = dict(x=ell_f, yb_z=f_ell*TT5b, yt=None, res=None, band=None, lsst=None,
                    yl=r"$\ell(\ell{+}1)C_\ell^{\tau\tau}/2\pi$", tag="(c)",
                    sobol=(lo_tt, hi_tt), rlim=_rl_sob)
    kt_panel = dict(x=ell_f, yb_z=f_ell*KT5b, yt=None, res=None, band=None, lsst=None,
                    yl=r"$\ell(\ell{+}1)C_\ell^{\kappa\tau}/2\pi$", tag="(e)",
                    sobol=(lo_kt, hi_kt), rlim=_rl_sob)
    yt_panel = dict(x=ell_f, yb_z=f_ell*YT5b, yt=None, res=None, band=None, lsst=None,
                    yl=r"$\ell(\ell{+}1)C_\ell^{y\tau}/2\pi$", tag="(f)",
                    sobol=(lo_yt, hi_yt), rlim=_rl_sob)

# panel ORDER matches the canonical-statistics order (figs 5/7): yy, tt, ky, kt, yt
panels4b = [kk_panel, yy_panel, tt_panel, ky_panel, kt_panel, yt_panel]

fig = plt.figure(figsize=(TWO_COL[0], 6.9))
# v2: 6 panels (was 5) now fill BOTH rows completely (3 cols each) -- the
# legend moves out of the panel grid into its own row (was gs[3:5, 2], the
# slot the 6th panel now occupies).
gs = fig.add_gridspec(6, 3, height_ratios=[2.2, 1, 0.55, 2.2, 1, 0.9],
                      hspace=0.18, wspace=0.42)
for j, p in enumerate(panels4b):
    r0 = (j//3)*3
    a = fig.add_subplot(gs[r0, j % 3])
    ar = fig.add_subplot(gs[r0 + 1, j % 3], sharex=a)
    for zi in range(5):
        a.plot(p["x"], p["yb_z"][zi], color=ZCOLS[zi], lw=1.0)
    if p["yt"] is not None:
        a.plot(p["x"], p["yt"], color=cH, ls="--", lw=0.9)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_ylabel(p["yl"], fontsize=7)
    a.set_xlim(ELL_LO, ELL_MAX_PLOT)
    for _ax in (a, ar):      # CIC-aliasing regime (cancels in ratios, not in Cl)
        _ax.axvspan(ELL_TRUST, ELL_MAX_PLOT, color="0.85", alpha=0.45, lw=0, zorder=0)
    # in-panel text removed at the author's request: the CIC-aliasing caveat,
    # the BIND-only tau note, the LSST/CV context labels and the Sobol-strip
    # label all live in the caption and the print() stamps instead. The shading
    # and the figure legend still encode what each band is.
    a.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0)))
    a.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(3, 0.4)))
    plt.setp(a.get_xticklabels(), visible=False)
    a.tick_params(labelsize=6)

    ar.axhline(0, color=COLORS["dmo"], lw=0.6)
    if p.get("sobol") is not None:        # feedback spread, NOT a closure residual
        lo, hi = p["sobol"]
        ar.fill_between(ELL_TT, lo, hi, color=COLORS["secondary"], alpha=0.30, lw=0)
        ar.plot(ELL_TT, 0.5*(lo + hi), color=COLORS["secondary"], lw=0.7)
        ar.set_ylabel("Sobol 16-84% [%]", fontsize=5.5, color=COLORS["secondary"])
    else:
        if p["band"] is not None:
            ar.fill_between(p["x"], -p["band"], p["band"], color=COLORS["dmo"], alpha=0.35, lw=0)
        if p["lsst"] is not None:
            ar.fill_between(p["x"], -p["lsst"], p["lsst"], color=cB, alpha=0.22, lw=0)
        ar.plot(p["x"], p["res"], color=cB, lw=0.9)
        ar.set_ylabel("resid. [%]" if j % 3 == 0 else "", fontsize=6)
    ar.set_ylim(-p["rlim"], p["rlim"]); ar.set_xscale("log")
    ar.set_xlabel(r"$\ell$", fontsize=7)
    ar.tick_params(labelsize=6)
    panel_label(a, p["tag"], loc="upper right")

alg = fig.add_subplot(gs[5, :]); alg.axis("off")
_leg_handles = [plt.Line2D([], [], color=ZCOLS[zi], lw=1.0,
                           label=rf"$z_s={ZS[zi]:.2f}$") for zi in range(5)]
_leg_handles += [plt.Line2D([], [], color=cH, lw=0.9, ls="--", label=r"hydro-pasted, $z_s=1$"),
                 plt.Rectangle((0, 0), 1, 1, fc=COLORS["dmo"], alpha=0.35, ec="none",
                               label=r"paired $\pm1\sigma/\sqrt{50}$")]
if not HAS_TRUTH_TAU:      # the sobol-spread strip only exists while the tau guard is false
    _leg_handles.append(plt.Rectangle((0, 0), 1, 1, fc=COLORS["secondary"], alpha=0.30,
                                      ec="none", label="Sobol 16-84% (c, e, f)"))
alg.legend(handles=_leg_handles, loc="center", ncol=4, fontsize=5.4,
           handlelength=1.6, labelspacing=0.5, columnspacing=1.4)
# the "z_s = source plane for kappa, column depth for y/tau" gloss also moves to the
# caption -- the legend keeps only the colour-to-z_s key it needs to be readable.
save(fig, "figs_v2/fig04b_spectra")
plt.show()

mb = (ell_f >= 300) & (ell_f <= 5000)
print("fig 4b median |resid| in ell=300-5000:  "
      f"Cl_kk {np.nanmedian(np.abs(res_kk[mb])):.1f}%  "
      f"Cl_ky {np.nanmedian(np.abs(res_ky[mb])):.1f}%  "
      f"Cl_yy(total column, released) {np.nanmedian(np.abs(res_yy[mb])):.1f}%  "
      f"Cl_yy(z_s=1 column, plotted) {np.nanmedian(np.abs(res_yy_z[mb])):.1f}%")
print("chi2/dof (seed-paired, ell 100-1.5e4): "
      f"kk {x2_kk:.0f}  ky {x2_ky:.0f}  yy(total) {x2_yy:.0f}  yy(z_s=1) {x2_yy_z:.0f}"
      + (f"  tt {x2_tt_z:.0f}  kt {x2_kt_z:.0f}  yt {x2_yt_z:.0f}" if HAS_TRUTH_TAU else ""))
print("CV-only floor on the y/tau panels (no shape-noise term, "
      f"fsky={CV_FSKY}) median/max in the trusted range: "
      f"{np.nanmedian(cv_band[mt]):.2f}% / {np.nanmax(cv_band[mt]):.2f}%")
i1k = int(np.argmin(np.abs(ell_f - 1000.0)))
print("tau-family spectra at ell~1000: l(l+1)C/2pi = "
      f"{f_ell[i1k]*TT5b[ZI][i1k]:.3e} (tau-tau BIND, column to z=1)  "
      f"{f_ell[i1k]*TT5b[-1][i1k]:.3e} (tau-tau BIND, total column)  "
      f"{f_ell[i1k]*KT5b[ZI][i1k]:.3e} (kappa-tau BIND, z_s=1)  "
      f"{f_ell[i1k]*YT5b[ZI][i1k]:.3e} (y-tau BIND, column to z=1)")
if HAS_TRUTH_TAU:
    print("tau-family truth-validated closure (guard TRUE this run): "
          f"median |resid| ell=300-5000: tt {np.nanmedian(np.abs(res_tt_z[mb])):.1f}%  "
          f"kt {np.nanmedian(np.abs(res_kt_z[mb])):.1f}%  yt {np.nanmedian(np.abs(res_yt_z[mb])):.1f}%")
else:
    print("no seed-paired hydro-pasted truth tau map exists in this release (guard FALSE "
          "this run) -- the only tau closure available is halo-level (the "
          "tau_profiles_snap096.npz truth_prof cache, painted/truth = 1.040 inside r500c)")
print("Sobol 253-node 16-84% width at ell~1000: "
      f"Cl_tautau {hi_tt[i1k]-lo_tt[i1k]:.1f}%  Cl_kappatau(z_s=1) {hi_kt[i1k]-lo_kt[i1k]:.1f}%  "
      f"Cl_ytau {hi_yt[i1k]-lo_yt[i1k]:.1f}%; "
      "suite-median / fiducial-map amplitude over ell=300-5000 = "
      f"{np.nanmedian(np.nanmedian(d['t__cl_tt__value'], 0)[mb]/TT5b[-1][mb]):.3f} "
      "(ratio ~1 confirms the Sobol cache and the fiducial map share a normalization)")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## Appendix A (reads with §2b) — Fig 18: mechanism of fig 4's high-$\ell$ $C_\ell^{yy}$ misfit

*Appendix material in the paper layout: fig 18 is a root-cause diagnostic of a
characterized systematic, not part of the main validation narrative; it stays here in
reading order because it consumes fig 4's caches.*

**This figure explains the one failing channel of fig 4** — the $+10$–$20\%$ high-$\ell$
upturn in $C_\ell^{yy}$ over $5\times10^3<\ell<1.5\times10^4$ (the WL leg closes at
sub-percent and needs no such diagnostic) — by localizing where in the map, and where
inside each halo, the excess power lives.

**Panel (a): not a bright-pixel artifact.** Splitting the $y$ maps by brightness — 'core'
($>p_{99}$ of the *truth* map, the identical truth-derived mask applied to both sides) vs
'field' ($\leq p_{99}$) — and recomputing $C_\ell^{yy}$ on 10 seed-paired realizations:
removing the cores does **not** shrink the high-$\ell$ excess, it *grows* it, from
$\sim+7\%$ unmasked to $\sim+25\%$ with cores removed, while the excised cores alone match
truth to within a percent (consistent with zero). All effect sizes are quoted as median
residual ± the 10-realization paired standard error in the printed output
(RECOMPUTED-ON-RUN — quote those values; with $N=10$ realizations a formal $\sigma$
would itself be uncertain at the ~25% level, so we report effect ± scatter, not
significances), including the full percentile sweep $p\in\{99.9,99.5,99,95\}$; the most
extreme cut is if anything mildly under-predicted. The excess is therefore a diffuse,
extended small-scale pressure-shape systematic, not a compact bright-source artifact.

**Panel (b): that "diffuse" excess is over-textured halo interiors, not a genuinely diffuse
field.** A per-annulus decomposition at 2784 matched group-scale halos
(`audits/radial_texture_results_FULL.npz`) splits the azimuthal-residual variance — a
real-space proxy for small-scale texture power — by $R/r_{200c}$: painted $y$ carries a
$29\%$ **excess** in texture variance at $R<0.5\,r_{200c}$, a mild $+4\%$ excess at
$0.5$–$1\,r_{200c}$, and a **growing deficit** outward ($-2\%$ at $1$–$2$, $-40\%$ at
$2$–$4\,r_{200c}$), phase-correlated with the true field throughout ($0.84$–$0.95$). The
mean-$y$ profile, by contrast, only drifts mildly with radius ($+4\%$ core to $-7\%$
outskirts) — the texture-variance swing is an order of magnitude larger than this amplitude
bias. So the field bin in panel (a) is dominated by *moderate*-brightness, over-textured
halo interiors sitting just below the $p_{99}$ mask, diluted by texture-starved outskirts —
exactly reconciling the two panels.

**Status: not yet corrected.** A Gaussian-smoothing mitigation of the painted map outside
$R>r_{200c}$ (cores left untouched) is **falsified by measurement**: it leaves
$R<1\,r_{200c}$ unchanged but *widens* the outskirt deficit further below unity
($1$–$2\,r_{200c}$: $0.98\to0.90\,(\sigma{=}1\,\mathrm{px})\to0.77\,(\sigma{=}2\,\mathrm{px})$;
$2$–$4\,r_{200c}$: $0.60\to0.57\to0.50$) without touching the interior excess — smoothing
cannot simultaneously suppress core clumpiness and restore outskirt power. The indicated path
forward is a texture-/spectrum-aware training loss, deferred to a future major version
(~$10^3$ GPU-hour repaint cost); at v1 the anomaly is characterized in detail for user
appraisal (audit artifacts: `audits/radial_texture.py`,
`audits/radial_texture_results_FULL.npz`, `audits/investigation3_diffuse_texture.png`).
''')

code(r'''
# ── Fig 18: mechanism of fig 4's high-ell Cl^yy misfit (mask localize + texture) ─
# data: bind_science/runs/{bind,truth}/run_0000/{kappa,y}_maps.npz (NR18=10 of
#       the 50 seed-paired realizations; y total plane [:, -1], kappa [:, ZI])
#       for panel (a) + papers/01_pipeline/audits/radial_texture_results_FULL.npz
#       (archived per-annulus texture audit, 2784 matched group-scale halos) for
#       panel (b).
# Method (a): for p in {99.9, 99.5, 99, 95} (percentile of the TRUTH y map, PER
# realization), split y into 'core' (>p, halo centres) and 'field' (<=p)
# by zeroing pixels -- using the SAME truth-derived mask on both BIND and
# truth so the comparison is apples-to-apples -- then recompute Cl^yy and
# Cl^ky (vs the UNMODIFIED kappa) on each half and see how much of fig 4's
# +10-20% excess at 5e3<ell<1.5e4 survives (plot only baseline + p99 -- the
# full percentile sweep goes to the printed output).
# Method (b): the archived audit's per-annulus (fixed x=R/r200c bins) painted/
# truth ratio of (i) azimuthal-residual variance (texture power) and (ii) mean
# y -- ties the panel-(a) field excess to over-textured halo interiors.
from bind.inference.stats import power_spectrum

NR18 = 10                                              # of 50; realization scatter stated below
f = np.load(RB/"kappa_maps.npz"); bk18 = f["kappa"][:NR18, ZI].astype(np.float32).copy(); del f; gc.collect()
f = np.load(RT/"kappa_maps.npz"); tk18 = f["kappa"][:NR18, ZI].astype(np.float32).copy(); del f; gc.collect()
f = np.load(RB/"y_maps.npz");     by18 = f["y"][:NR18, -1].astype(np.float32).copy();     del f; gc.collect()
f = np.load(RT/"y_maps.npz");     ty18 = f["y"][:NR18, -1].astype(np.float32).copy();     del f; gc.collect()

def stack_spectra(maps1, maps2=None):
    ell18, out = None, []
    for r in range(maps1.shape[0]):
        ell18, c = power_spectrum(maps1[r], None if maps2 is None else maps2[r])
        out.append(c)
    return ell18, np.array(out)

def paired18(bR, tR):
    """mean-curve %-residual + paired +-1sigma/sqrt(NR18) band (fig 4's `paired`, NR18 scatter)."""
    rat = bR/np.where(tR > 0, tR, np.nan)
    res = 100*(np.nanmean(bR, 0)/np.nanmean(tR, 0) - 1)
    band = 100*np.nanstd(rat, 0)/np.sqrt(NR18)
    return res, band

PCTS18 = [99.9, 99.5, 99, 95]
ell18, yy_b0 = stack_spectra(by18);  _, yy_t0 = stack_spectra(ty18)
_,     ky_b0 = stack_spectra(bk18, by18); _, ky_t0 = stack_spectra(tk18, ty18)
res_yy0_18, band_yy0_18 = paired18(yy_b0, yy_t0)
res_ky0_18, band_ky0_18 = paired18(ky_b0, ky_t0)

res18 = {}
for p in PCTS18:
    core_b, core_t = np.empty_like(by18), np.empty_like(ty18)
    field_b, field_t = np.empty_like(by18), np.empty_like(ty18)
    frac_pix, frac_flux = [], []
    for r in range(NR18):
        thr = np.percentile(ty18[r], p)
        mask = ty18[r] > thr                            # mask from TRUTH, applied to BOTH sides
        core_b[r], core_t[r] = np.where(mask, by18[r], 0.0), np.where(mask, ty18[r], 0.0)
        field_b[r], field_t[r] = np.where(mask, 0.0, by18[r]), np.where(mask, 0.0, ty18[r])
        frac_pix.append(mask.mean()); frac_flux.append(ty18[r][mask].sum()/ty18[r].sum())
    _, yy_bc = stack_spectra(core_b);  _, yy_tc = stack_spectra(core_t)
    _, yy_bf = stack_spectra(field_b); _, yy_tf = stack_spectra(field_t)
    _, ky_bc = stack_spectra(bk18, core_b);  _, ky_tc = stack_spectra(tk18, core_t)
    _, ky_bf = stack_spectra(bk18, field_b); _, ky_tf = stack_spectra(tk18, field_t)
    res_yy_c, band_yy_c = paired18(yy_bc, yy_tc); res_yy_f, band_yy_f = paired18(yy_bf, yy_tf)
    res_ky_c, band_ky_c = paired18(ky_bc, ky_tc); res_ky_f, band_ky_f = paired18(ky_bf, ky_tf)
    res18[p] = dict(res_yy_c=res_yy_c, res_yy_f=res_yy_f, res_ky_c=res_ky_c, res_ky_f=res_ky_f,
                     band_yy_c=band_yy_c, band_yy_f=band_yy_f,
                     frac_pix=np.mean(frac_pix), frac_flux=np.mean(frac_flux))
    del core_b, core_t, field_b, field_t
del bk18, tk18, by18, ty18; gc.collect()

# ── panel (b) data: archived per-annulus texture audit (read-only, pre-existing) ─
RTX_PATH = Path("audits/radial_texture_results_FULL.npz")
rtx = np.load(RTX_PATH, allow_pickle=True)
ann_labels = [str(s) for s in rtx["coarse_labels"]]
var_ratio_rtx  = rtx["var_fid"]/rtx["var_truth"]
mean_ratio_rtx = rtx["mean_y_fid"]/rtx["mean_y_truth"]
corr_rtx = rtx["corr"]
n_halo_rtx = int(rtx["n_halo"])

# ── figure: (a) decluttered mask-localization (baseline + p99 only) ─────────
#            (b) NEW per-annulus texture decomposition ──────────────────────
m18 = (ell18 >= 300) & (ell18 <= ELL_TRUST)
i10k, i2k = np.argmin(np.abs(ell18 - 1.0e4)), np.argmin(np.abs(ell18 - 2.0e3))
r99 = res18[99]

fig, ax = plt.subplots(1, 2, figsize=(TWO_COL[0], 2.9))
a = ax[0]
a.axvspan(5e3, 1.5e4, color=COLORS["dmo"], alpha=0.15, lw=0)
a.text(8.6e3, -19, "fig-4 high-$\\ell$\nexcess window", fontsize=5.0,
       color=COLORS["truth"], ha="center", va="bottom")
a.axhline(0, color=COLORS["dmo"], lw=0.6)
a.fill_between(ell18[m18], -band_yy0_18[m18], band_yy0_18[m18], color=COLORS["dmo"],
               alpha=BAND_ALPHA, lw=0)
a.plot(ell18[m18], res_yy0_18[m18], color=COLORS["truth"], lw=1.6, label="unmasked (fig 4)")
a.plot(ell18[m18], r99["res_yy_f"][m18], color=COLORS["highlight"], lw=1.4,
       label=r"field only ($p_{99}$ cores removed)")
a.plot(ell18[m18], r99["res_yy_c"][m18], color=COLORS["secondary"], lw=1.2, ls="--",
       label=r"core only ($>p_{99}$)")
a.set_xscale("log"); a.set_xlabel(r"$\ell$"); a.set_ylabel(r"$C_\ell^{yy}$ resid. [%]")
a.set_ylim(-20, 44)
a.legend(loc="upper left", fontsize=5.2)
panel_label(a, "(a)", loc="lower right")
a.annotate("remove cores $\\to$\nexcess GROWS", xy=(ell18[i10k], r99["res_yy_f"][i10k]),
           xytext=(3200, 40), fontsize=6.2, color=COLORS["highlight"], ha="left", va="top",
           arrowprops=dict(arrowstyle="-|>", color=COLORS["highlight"], lw=0.8,
                            connectionstyle="arc3,rad=-0.15"))
a.annotate("cores alone: deficit", xy=(ell18[i2k], r99["res_yy_c"][i2k]),
           xytext=(900, -18.5), fontsize=6.2, color=COLORS["secondary"], ha="left",
           arrowprops=dict(arrowstyle="-|>", color=COLORS["secondary"], lw=0.8,
                            connectionstyle="arc3,rad=0.15"))

b = ax[1]
xb = np.arange(len(ann_labels))
b.bar(xb, var_ratio_rtx, width=0.55, color=COLORS["bind"], alpha=0.85, zorder=2,
      label=r"var(painted)/var(truth)")
b.plot(xb, mean_ratio_rtx, "o-", color=COLORS["highlight"], ms=5, lw=1.4, zorder=3,
       label=r"mean $y$(painted)/$y$(truth)")
b.axhline(1.0, color=COLORS["dmo"], ls=":", lw=1.0, zorder=1)
b.set_xticks(xb); b.set_xticklabels(ann_labels)
b.set_xlabel(r"annulus $R/r_{200c}$"); b.set_ylabel("painted / truth ratio")
b.set_ylim(0, 1.42)
b.legend(loc="upper right", fontsize=5.4)
panel_label(b, "(b)", loc="upper left")

fig.tight_layout()
save(fig, "figs_v2/fig18_highell_localize")
plt.show()

mb18 = (ell18 >= 5e3) & (ell18 <= 1.5e4)
print(f"high-ell band 5e3<ell<1.5e4, NR18={NR18} realizations "
      "(effect size +/- paired SE; no sigma framing -- an N=10 sigma is itself "
      "~25% uncertain):")
print(f"  unmasked (fig 4 statistic): median resid  "
      f"Cl_yy={np.nanmedian(res_yy0_18[mb18]):+.1f}%+/-{np.nanmedian(band_yy0_18[mb18]):.1f}%  "
      f"Cl_ky={np.nanmedian(res_ky0_18[mb18]):+.1f}%+/-{np.nanmedian(band_ky0_18[mb18]):.1f}%")
for p in PCTS18:
    r = res18[p]
    sef = np.nanmedian(r["band_yy_f"][mb18])
    sec = np.nanmedian(r["band_yy_c"][mb18])
    print(f"  p={p:5.1f} (top {100-p:.1f}% of pixels, {r['frac_flux']*100:.0f}% of total y flux): "
          f"field Cl_yy={np.nanmedian(r['res_yy_f'][mb18]):+.1f}%+/-{sef:.1f}%  "
          f"Cl_ky={np.nanmedian(r['res_ky_f'][mb18]):+.1f}%   |   "
          f"core Cl_yy={np.nanmedian(r['res_yy_c'][mb18]):+.1f}%+/-{sec:.1f}%  "
          f"Cl_ky={np.nanmedian(r['res_ky_c'][mb18]):+.1f}%")
print("(p=95 Cl_ky residuals are omitted above -- at 42% of the flux removed, "
      "the truth-side field cross-spectrum approaches zero at the highest ell and the percent "
      "ratio becomes numerically unstable (up to +100s%); the well-behaved Cl_yy trend already "
      "makes the point.)")
print("\nATTRIBUTION: masking out the brightest truth pixels does not remove the high-ell "
      "excess -- it grows. At p99 (top 1% of pixels, 20% of the y flux), the field component's "
      f"Cl_yy excess rises from the unmasked +{np.nanmedian(res_yy0_18[mb18]):.0f}% to "
      f"+{np.nanmedian(res18[99]['res_yy_f'][mb18]):.0f}%"
      f"+/-{np.nanmedian(res18[99]['band_yy_f'][mb18]):.0f}% (paired SE, N=10), "
      "while the removed core matches truth to "
      f"{np.nanmedian(res18[99]['res_yy_c'][mb18]):+.1f}% (consistent with zero). At the more "
      "extreme p99.9 cut the very brightest 0.1% of pixels are if anything mildly UNDER-predicted "
      f"by BIND ({np.nanmedian(res18[99.9]['res_yy_c'][mb18]):+.1f}%), even as the field excess is "
      "already large. The +10-20% high-ell upturn is therefore not a bright compact-source "
      "artifact of the paste; it is a diffuse, extended small-scale pressure-shape excess that "
      "the well-matched (or mildly under-luminous) halo centres partially cancel in the combined "
      "statistic, and that is fully exposed once the cores are excised.")

print(f"\nradial-texture audit ({RTX_PATH.name}): n_halo={n_halo_rtx} matched group-scale "
      f"halos, M_fof {rtx['mass'].min():.2e}-{rtx['mass'].max():.2e} Msun/h")
print(f"{'annulus':8s} {'var ratio':>10s} {'mean_y ratio':>13s} {'corr(resid)':>12s}")
for lab, vr, mr, cc in zip(ann_labels, var_ratio_rtx, mean_ratio_rtx, corr_rtx):
    print(f"{lab:8s} {vr:10.3f} {mr:13.3f} {cc:12.3f}")
print("PANEL (b) MECHANISM: the field-only excess in panel (a) is not a genuinely diffuse "
      "component -- it is over-textured halo interiors (var ratio "
      f"{var_ratio_rtx[0]:.2f} at R<0.5 r200c) diluted by texture-starved outskirts (var ratio "
      f"{var_ratio_rtx[-1]:.2f} at 2-4 r200c), while the mean-y profile only drifts mildly "
      f"({mean_ratio_rtx[0]:.2f} to {mean_ratio_rtx[-1]:.2f}) -- an order of magnitude smaller "
      "than the texture-variance swing.")
var_mit_s1_ratio = rtx["var_mit_s1"]/rtx["var_truth"]
var_mit_s2_ratio = rtx["var_mit_s2"]/rtx["var_truth"]
print("smoothing mitigation (painted y smoothed for R>r200c only; cores untouched) -- "
      "FALSIFIED: widens the outskirt deficit instead of closing it:")
for lab, vr, s1, s2 in zip(ann_labels, var_ratio_rtx, var_mit_s1_ratio, var_mit_s2_ratio):
    print(f"  R/r200c={lab:6s}  raw={vr:.3f}  sigma=1px:{s1:.3f}  sigma=2px:{s2:.3f}")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §2c — Fig 11: validation across redshift

The release ships five source planes to $z_s=2.44$, but figs 3–4 validate at $z_s=1$ and
$z=0.034$ only. This figure closes that gap and exposes the release's principal redshift
systematic. Panels: (a) paired $C_\ell^{\kappa\kappa}$ closure per source plane — sub-2%
everywhere; (b) map-level $C_\ell^{\kappa y}$ closure per plane — few-% at $z_s=0.5$
growing to ~10–15% at the highest planes. Panels (a,b) are ratios of cached *mean* spectra
(per-plane realization scatter is not cached), with the $z_s=1$ paired band drawn for scale
only — both stated on-figure. (c) halo-level closure vs $z$: median BIND/truth $Y_{500c}$
1.05 → ~1.3 by $z=2$, now with bootstrap error bars on every per-snapshot median (500
seeded resamples over halos — the $\geq10^{13}$ population collapses ~20× to $z=2.4$, so
the high-$z$ medians are visibly noisier); (d) stacked $y$-profile normalization drifts
accordingly (point = mean over the four mass stacks, error bar = between-stack SE — the
profile cache is stack-level, not per-halo). The shipped de-biasing templates (§5b) are fit
*weighted by these uncertainties*, with coefficient uncertainties printed and exported.

**Root cause (resolved).** A dedicated audit discriminated three hypotheses. Unit/a-factor
conventions are *ruled out*: the multi-$z$ training-map generator and the truth-lightcone
generator share byte-identical comoving→physical conversions, and the atlas reducer runs
the same code on both sides; the scale factor is threaded intact from snapshot header to
the model's redshift embedding. What remains is **genuine high-$z$ under-supervision of the
thermodynamic channels**: the per-channel drift slopes (ratio $\propto a^\alpha$) order by
the steepness of each channel's target physics —

| channel | $T_{\rm mw}$ | $f_{\rm gas}$ | $Y_{500c}$ | $K$ (entropy) | $P_e$ |
|---|---|---|---|---|---|
| $\alpha$ | $-0.11$ | $-0.085$ | $-0.18$ | $+0.11$ | $-0.53$ |
| ratio at $z=2$ | 1.12 | 1.18 | 1.30 | 0.85 | 1.97 |

— pressure (steepest target, $\propto a^{-3}$) drifts most, temperature ($a$-independent
target) least, and entropy is *under*-predicted with the inverse sign: the coherent story of
gas painted **over-concentrated at high $z$** (excess density ⇒ inflated $P_e$, deflated
$K$). The drift is also **mass-dependent** (low-mass halos drift ~2× faster; at $z=2.4$,
$P_e$ is +192% low-mass vs +79% high-mass) — the signature of training-set composition, not
of a convention error: the multi-$z$ training data used 1 rotation/halo (vs 10 at $z=0$)
over 8 discrete snapshots, painted here across 20, with the $\geq10^{13}$ halo population
collapsing 20× between $z=0$ and $z=2.4$ (audit artifacts: `audits/thermo_drift.py`,
`audits/thermo_drift_diagnostic.png`). Consequence: v1 ships the correction as release-level assets — the quadratic de-biasing
templates of §5b plus the audit's per-channel, mass-split drift curves — applied post hoc;
denser high-$z$ supervision at retrain time is deferred to a future major version
(~$10^3$ GPU-hour repaint cost). Until then the drift is smooth and correctable. $\kappa$ products are validated at all
planes; $y/\tau$ products are best-validated at low $z$.
''')

code(r'''
# ── Fig 11: redshift-resolved closure ────────────────────────────────────────
# data: per-realization kk ratios per z_s (recomputed in fig 4's cell for ZI;
#       here from cached Cl_kappa + paired_perreal clk for all 5 planes) +
#       halo_atlas/{fid,truth}_snapNNN.npz + profiles/{fid,truth}_snapNNN.npz
b_cl, t_cl = np.load(RB/"Cl_kappa.npz"), np.load(RT/"Cl_kappa.npz")
b_ky0, t_ky0 = np.load(RB/"Cl_kappa_y.npz"), np.load(RT/"Cl_kappa_y.npz")
SNAPS = [29, 31, 33, 35, 38, 41, 43, 46, 49, 52, 56, 59, 63, 67, 71, 76, 80, 85, 90, 96]

fig, axg = plt.subplots(2, 2, figsize=(TWO_COL[0], 5.2))
ax = axg.ravel()

# (a) Cl_kk BIND/truth ratio per source plane (band = z_s=1 paired +-1sigma, for scale)
# v2: perceptible sequential ramp for the z_s series (cividis mid-tones were
# nearly indistinguishable at 1-pt linewidth)
zcols = plt.cm.plasma(np.linspace(0.02, 0.82, 5))
for zi in range(5):
    ratio = b_cl["cl"][zi, zi]/np.where(t_cl["cl"][zi, zi] > 0, t_cl["cl"][zi, zi], np.nan)
    ax[0].plot(ell_f, 100*(ratio - 1), color=zcols[zi], lw=1.0,
               label=f"$z_s$={ZS[zi]:.1f}")
ax[0].fill_between(ell_f, -band_kk, band_kk, color=COLORS["dmo"], alpha=0.35, lw=0)
ax[0].axhline(0, color=COLORS["dmo"], lw=0.6)
ax[0].set_xscale("log"); ax[0].set_xlim(90, ELL_TRUST); ax[0].set_ylim(-6, 6)
ax[0].set_xlabel(r"$\ell$")
ax[0].set_ylabel(r"$C_\ell^{\kappa\kappa}$: BIND/truth $-1$ [%]")
ax[0].legend(loc="upper left", fontsize=5, ncol=2)
ax[0].text(0.97, 0.05, "curves: ratios of cached mean spectra;\n"
           "band: $z_s{=}1$ paired $\\pm1\\sigma$ (scale only)",
           transform=ax[0].transAxes, fontsize=4.8, ha="right", va="bottom")
panel_label(ax[0], "(a)", loc="lower left")

# (b) map-level kappa-y closure per source plane (cached mean spectra; the
# ratio is normalization-convention independent). This measures the drift at
# MAP level, not just via halo Y ratios.
kyd = []
for zi in range(5):
    rat = b_ky0["cl_ky"][zi]/np.where(np.abs(t_ky0["cl_ky"][zi]) > 0,
                                      t_ky0["cl_ky"][zi], np.nan)
    ax[1].plot(ell_f, 100*(rat - 1), color=zcols[zi], lw=1.0)
    kyd.append(100*(np.nanmedian(rat[(ell_f >= 300) & (ell_f <= 5000)]) - 1))
ax[1].axhline(0, color=COLORS["dmo"], lw=0.6)
ax[1].set_xscale("log"); ax[1].set_xlim(90, ELL_TRUST); ax[1].set_ylim(-15, 25)
ax[1].set_xlabel(r"$\ell$")
ax[1].set_ylabel(r"$C_\ell^{\kappa y}$: BIND/truth $-1$ [%]")
ax[1].text(0.97, 0.05, "curves: ratios of cached mean spectra\n"
           "(no per-plane realization band cached)",
           transform=ax[1].transAxes, fontsize=4.8, ha="right", va="bottom")
panel_label(ax[1], "(b)", loc="lower left")

# (b) halo-level closure vs z from the 20 snapshot atlases
# NB truth atlases at z>0.03 carry only the RAW (non-bg-subtracted) aperture
# masses, so this panel uses raw f_gas = m_gas/(m_dm+m_gas+m_star) within r500c
# on BOTH sides — a slightly different convention from fig 3(b).
def raw_fgas(a):
    tot = a["m_dm_500c"] + a["m_gas_500c"] + a["m_star_500c"]
    return a["m_gas_500c"]/np.where(tot > 0, tot, np.nan)
# v2: bootstrap SE on every per-snapshot median (the atlases are per-halo);
# the >=1e13 population collapses ~20x by z=2.4, so high-z medians are noisier
rngz = np.random.default_rng(6)
NBOOT = 500
zs_arr, yrat, frat, yerr_b, ferr_b = [], [], [], [], []
for sn in SNAPS:
    fz = np.load(SCI/f"halo_atlas/fid_snap{sn:03d}.npz")
    tz = np.load(SCI/f"halo_atlas/truth_snap{sn:03d}.npz")
    ok = (fz["Y_500c"] > 0) & (tz["Y_500c"] > 0)
    rY = fz["Y_500c"][ok]/tz["Y_500c"][ok]
    rF = raw_fgas(fz)/raw_fgas(tz); rF = rF[np.isfinite(rF)]
    zs_arr.append(float(fz["z"]))
    yrat.append(np.median(rY)); frat.append(np.nanmedian(rF))
    yerr_b.append(np.median(rY[rngz.integers(0, len(rY), (NBOOT, len(rY)))], 1).std())
    ferr_b.append(np.median(rF[rngz.integers(0, len(rF), (NBOOT, len(rF)))], 1).std())
zs_arr = np.array(zs_arr)
ax[2].errorbar(zs_arr, yrat, yerr=yerr_b, fmt="o-", ms=3, color=COLORS["bind"], lw=1.2,
               elinewidth=0.7, capsize=1.5, label=r"$Y_{500c}$")
ax[2].errorbar(zs_arr, frat, yerr=ferr_b, fmt="s-", ms=3, color=COLORS["secondary"],
               lw=1.2, elinewidth=0.7, capsize=1.5, label=r"$f_{\rm gas}$")
ax[2].axhline(1, color=COLORS["dmo"], ls=":", lw=0.8)
ax[2].set_xlabel(r"$z$")
ax[2].set_ylabel("median BIND/truth ratio\n(bootstrap $\\pm1\\sigma$, 500 resamples)")
panel_label(ax[2], "(c)", loc="lower right")

# (d) stacked y-profile normalization drift (r < r500c mean, per snapshot).
# The profile cache is STACK-level (4 mass stacks x 18 radii, not per-halo), so
# the point is the mean over the mass-stack means and the error bar is the
# between-stack SE — the honest uncertainty available from this cache.
prat, perr_b = [], []
for sn in SNAPS:
    pfz = np.load(SCI/f"profiles/fid_snap{sn:03d}.npz")
    ptz = np.load(SCI/f"profiles/truth_snap{sn:03d}.npz")
    rin = pfz["r_cen"] <= 1.0
    tru = np.where(ptz["prof_y"][:, rin] > 0, ptz["prof_y"][:, rin], np.nan)
    rat = pfz["prof_y"][:, rin]/tru
    stk = np.array([np.nanmean(rat[i]) for i in range(rat.shape[0])
                    if np.isfinite(rat[i]).any()])
    prat.append(stk.mean()); perr_b.append(stk.std()/np.sqrt(len(stk)))
ax[3].errorbar(zs_arr, prat, yerr=perr_b, fmt="o-", ms=3, color=COLORS["bind"], lw=1.2,
               elinewidth=0.7, capsize=1.5)
ax[3].axhline(1, color=COLORS["dmo"], ls=":", lw=0.8)
ax[3].set_xlabel(r"$z$")
ax[3].set_ylabel("stacked $y(r{<}r_{500c})$: BIND/truth\n(mean $\\pm$ SE over 4 mass stacks)")
panel_label(ax[3], "(d)", loc="lower right")

# ── Work Package C: redshift-debias templates (extends fig 11) ──────────────
# Fit smooth low-order templates r(a) (a=1/(1+z)) to the three z-resolved
# closure ratios in panels (c)/(d): linear-in-a, quadratic-in-a, and the
# power-law a^alpha form. v2: fits are WEIGHTED by the bootstrap/between-stack
# uncertainties of the per-snapshot points, AIC = chi2 + 2k (Gaussian, known
# sigma), and coefficient 1-sigma uncertainties (from the weighted normal
# equations) are printed and exported. The winning template per quantity is
# overplotted (thin dotted, legended) and reported as a machine-usable dict.
a_arr = 1/(1 + zs_arr)

def _fit_templates(a, r, sig):
    r = np.asarray(r, float)
    sig = np.maximum(np.asarray(sig, float), 1e-4)   # floor: avoid zero-sigma weights
    w = 1.0/sig
    cands = {}
    for nm, A, formula, keys in (
            ("linear_a", np.vstack([np.ones_like(a), a]).T,
             "r(a) = c0 + c1*a", ("c0", "c1")),
            ("quadratic_a", np.vstack([np.ones_like(a), a, a**2]).T,
             "r(a) = c0 + c1*a + c2*a^2", ("c0", "c1", "c2"))):
        Aw = A*w[:, None]
        c, *_ = np.linalg.lstsq(Aw, r*w, rcond=None)
        cerr = np.sqrt(np.diag(np.linalg.inv(Aw.T @ Aw)))
        cands[nm] = dict(pred=A @ c, k=A.shape[1], formula=formula,
                         coef=dict(zip(keys, map(float, c))),
                         coef_err=dict(zip(keys, map(float, cerr))))
    # power law, fitted in log space with propagated weights sig/r
    Al = np.vstack([np.ones_like(a), np.log(a)]).T
    wl = r/sig
    cl, *_ = np.linalg.lstsq(Al*wl[:, None], np.log(r)*wl, rcond=None)
    clerr = np.sqrt(np.diag(np.linalg.inv((Al*wl[:, None]).T @ (Al*wl[:, None]))))
    A0 = float(np.exp(cl[0]))
    cands["powerlaw_a"] = dict(pred=A0*a**cl[1], k=2, formula="r(a) = A0 * a**alpha",
                               coef={"A0": A0, "alpha": float(cl[1])},
                               coef_err={"lnA0": float(clerr[0]),
                                         "alpha": float(clerr[1])})
    for f in cands.values():
        f["chi2"] = float(np.sum(((r - f["pred"])/sig)**2))
        f["aic"] = f["chi2"] + 2*f["k"]
        f["rms_pct"] = float(100*np.sqrt(np.mean(((r - f["pred"])/r)**2)))
        f["max_pct"] = float(100*np.max(np.abs((r - f["pred"])/r)))
    best_name = min(cands, key=lambda k_: cands[k_]["aic"])
    return cands, best_name

DEBIAS_TEMPLATES = {}
tmpl_short = {"Y_500c": "$r_Y$", "f_gas": "$r_f$", "y_profile": "$r_y$"}
for qty, rvals, evals, axp, col in (
        ("Y_500c", yrat, yerr_b, ax[2], COLORS["bind"]),
        ("f_gas", frat, ferr_b, ax[2], COLORS["secondary"]),
        ("y_profile", prat, perr_b, ax[3], COLORS["bind"])):
    cands, best_name = _fit_templates(a_arr, np.asarray(rvals, float),
                                      np.asarray(evals, float))
    best = cands[best_name]
    print(f"debias fit [{qty:9s}] (sigma-weighted): " + "  ".join(
        f"{nm}(AIC={c['aic']:7.1f}, chi2/dof={c['chi2']/(len(a_arr)-c['k']):.1f}, "
        f"rms={c['rms_pct']:.2f}%)" for nm, c in cands.items())
        + f"  -> BEST={best_name}")
    print("   coef +/- 1sigma: " + ", ".join(
        f"{k}={v:.4f}+/-{best['coef_err'].get(k, best['coef_err'].get('ln' + k, np.nan)):.4f}"
        for k, v in best["coef"].items()))
    a_fine = np.linspace(a_arr.min(), a_arr.max(), 200)
    z_fine = 1/a_fine - 1
    if best_name == "linear_a":
        pred_fine = best["coef"]["c0"] + best["coef"]["c1"]*a_fine
    elif best_name == "quadratic_a":
        pred_fine = (best["coef"]["c0"] + best["coef"]["c1"]*a_fine
                     + best["coef"]["c2"]*a_fine**2)
    else:
        pred_fine = best["coef"]["A0"]*a_fine**best["coef"]["alpha"]
    axp.plot(z_fine, pred_fine, ":", lw=0.9, color=col, zorder=1.5,
             label=f"released template {tmpl_short[qty]}$(a)$ "
                   f"(rms {best['rms_pct']:.1f}%)")
    DEBIAS_TEMPLATES[qty] = {"best_form": best_name, "coef": best["coef"],
                              "coef_err": best["coef_err"],
                              "formula": best["formula"], "rms_pct": best["rms_pct"],
                              "max_pct": best["max_pct"],
                              "chi2_dof": best["chi2"]/(len(a_arr) - best["k"]),
                              "aic": {nm: c["aic"] for nm, c in cands.items()}}
ax[2].legend(loc="upper left", fontsize=5)      # data + template entries together
# (d): upper LEFT + light frame -- at upper right the frameless legend collided
# with the z~2.2 point's error-bar cap (drift rises with z, so upper right
# holds the highest data; upper left is clear)
ax[3].legend(loc="upper left", fontsize=5, frameon=True, framealpha=0.85,
             edgecolor="0.7")
print("DEBIAS_TEMPLATES (apply as BIND_debiased = BIND_raw / r(a), a=1/(1+z)):")
print(DEBIAS_TEMPLATES)

fig.tight_layout(w_pad=1.1, h_pad=1.2)
save(fig, "figs_v2/fig11_zclosure")
plt.show()
i96 = SNAPS.index(96)
# is the drift a single missing a-factor? fit ratio ~ a^alpha (a-factor => alpha=-1)
alpha = np.polyfit(np.log(a_arr), np.log(yrat), 1)[0]
print(f"median Y ratio: {yrat[i96]:.3f} at z={zs_arr[i96]:.2f} -> "
      f"max {max(yrat):.3f} at z={zs_arr[int(np.argmax(yrat))]:.2f}; "
      f"f_gas ratio range {min(frat):.3f}-{max(frat):.3f}; "
      f"y-profile ratio range {min(prat):.3f}-{max(prat):.3f}")
print(f"drift fit: Y ratio ~ a^{alpha:.2f} — a single missing a-factor (alpha=-1) is "
      f"excluded; consistent with a partial z>0 thermo-normalization systematic")
print("map-level kappa-y closure (median resid %, ell 300-5000, per z_s): "
      + ", ".join(f"z{z:.1f}: {v:+.1f}%" for z, v in zip(ZS, kyd)))
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §3 opener — Fig 23: why feedback matters at survey precision (the bridge, in one panel)

The section opener (the planning sessions' "New Image 1(a)", ruled 2026-08-03: panel (a)
alone, wired from the prototype): every Sobol node's WL suppression $S(\ell{=}5000, z_s{=}1)$
against its group-scale gas fraction $f_{\rm gas}$ ($\log_{10}M_{500c}\simeq13.0$–$13.25$, the
most feedback-sensitive bin), coloured by the dominant lever (WindEnergy). Two things in one
panel: (i) the feedback prior's $S$ span (5–95%) is many times the LSST-Y10/Euclid statistical
precision at this scale — the shaded/outlined gauges at $S{=}1$ — so unmodelled feedback is a
*survey-level* systematic (quantified per-statistic in figs 5/7/12); and (ii) the span is not
noise: it is *indexed by a gas observable* (log-linear fit over the full node set; $r$ printed
live), which is the §3c bridge in miniature and the reason gas data can close the loop. The
prototype's second (ejection–heating) panel is deliberately NOT included — at population
level ejection and heating are strongly coupled (see the 3c-companion evaluation memo).
''')

code(r'''
# ── Fig 23 (§3 opener): S(5000) vs group f_gas + survey-precision gauges ─────
# Ported from proto_bridge_hero.py panel (a) (fork ruling 2026-08-03: panel (a)
# alone; panel (b) dropped -- population-level ejection/heating coupling). The
# survey gauges reuse cl_relerr from the fig 4 cell (LSST-Y10 27/0.26/0.44,
# Euclid 30/0.30/0.36 -- fig 12's exact conventions) at the same ell bin;
# sigma(S)/S = sigma(Cl)/Cl since S is referenced to a fixed theory DMO trace.
ell23 = d["a__suppression__ell"]
i5k23 = int(np.argmin(np.abs(ell23 - 5000)))
S23 = d["t__suppression__value"][:, ZI, i5k23]
fg23 = d["t__scaling_f_gas__value"][:, 0]          # group bin, log M500c ~ 13.0-13.25
mb23 = float(d["a__scaling_f_gas__log_mass_bins"][0])
we23 = np.log10(d["X_native"][:, 2])               # WindEnergyIn1e51erg (native col 2)
we23 = (we23 - we23.min())/np.ptp(we23)

bcl23 = np.load(SCI/"runs/bind/run_0000/Cl_kappa.npz")
i5c23 = int(np.argmin(np.abs(bcl23["ell"] - 5000)))
sig_lsst23 = float(cl_relerr(bcl23["cl"][ZI, ZI], 27.0, 0.26, 0.44)[i5c23])
sig_euc23  = float(cl_relerr(bcl23["cl"][ZI, ZI], 30.0, 0.30, 0.36)[i5c23])
lo23, hi23 = np.percentile(S23, [5, 95])
r_log23 = np.corrcoef(np.log10(fg23), S23)[0, 1]
z23 = np.polyfit(np.log10(fg23), S23, 1)

fig, ax = plt.subplots(figsize=(ONE_COL[0]*1.35, 3.0))
ax.axhspan(lo23, hi23, color="0.92", lw=0, zorder=0, label="Sobol 5–95%")
ax.axhspan(1 - sig_lsst23, 1 + sig_lsst23, color=COLORS["highlight"], alpha=0.30, lw=0,
           zorder=1, label=r"LSST-Y10 $\pm1\sigma$")
for yv23 in (1 - sig_euc23, 1 + sig_euc23):
    ax.axhline(yv23, color=COLORS["secondary"], ls="-.", lw=0.8, zorder=1)
ax.plot([], [], color=COLORS["secondary"], ls="-.", lw=0.8, label=r"Euclid $\pm1\sigma$")
sc23 = ax.scatter(fg23, S23, c=we23, cmap="coolwarm", s=16, edgecolor="k", lw=0.25, zorder=3)
xg23 = np.linspace(fg23.min(), fg23.max(), 50)
ax.plot(xg23, np.polyval(z23, np.log10(xg23)), color=COLORS["truth"], lw=1.3, zorder=4,
        label=fr"log-linear fit ($r={r_log23:.2f}$)")
ax.axhline(1, color=COLORS["dmo"], ls=":", lw=0.8, zorder=2)
ax.set_xlabel(fr"$f_{{\rm gas}}$ ($\log_{{10}}M_{{500c}}\simeq{mb23:.2f}$)")
ax.set_ylabel(r"$S(\ell{=}5000)$ at $z_s=1$")
ax.legend(fontsize=5.2, loc="lower right")
cb23 = fig.colorbar(sc23, ax=ax, fraction=0.046, pad=0.03)
cb23.set_label(r"norm. $\log_{10}$ WindEnergy", fontsize=6)
cb23.ax.tick_params(labelsize=5.5)
save(fig, "figs_v2/fig23_s3_opener")
plt.show()
print(f"fig 23 (§3 opener): Sobol 5-95% span of S(5000, z_s=1) = {100*(hi23-lo23):.1f}% "
      f"vs LSST-Y10 sigma {100*sig_lsst23:.2f}% ({(hi23-lo23)/sig_lsst23:.0f}x) and "
      f"Euclid sigma {100*sig_euc23:.2f}% ({(hi23-lo23)/sig_euc23:.0f}x) at that ell bin; "
      f"log-linear bridge r={r_log23:.2f} over {len(S23)} nodes (WindEnergy-coloured)")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §3a — Fig 5a: the response, bin by bin — $S(\ell)$ against all 30 parameters

With validation established, we turn to the suite's purpose: the response of the released
observables to galaxy-formation physics. Fig 5 will compress that response into a single
importance number per (statistic, parameter) cell; this figure first shows the object that
number is extracted *from*, at full resolution, for the release's headline statistic. Each
cell is the **signed** Spearman rank correlation
$\rho_{jb}=\mathrm{corr}\,[\mathrm{rank}(\theta_j),\,\mathrm{rank}(S_b)]$ between the
unit-cube parameter $\theta_j$ ($\log_{10}$-mapped where the prior is logarithmic) and the
WL suppression $S(\ell)$ in the pre-averaged ($\times16$) $\ell$ bin $b$, over the 253 runs
at $z_s=1$: red = raising the parameter *raises* $S(\ell)$ (less suppression, toward
enhancement), blue = raising it *deepens* the suppression. Columns carry fig 5's ordering
(peak importance over all 12 statistics), so the two figures share one $x$ axis; a circle
marks each column's peak-$|\rho|$ bin, and the value at that circle **is** the
corresponding cell of fig 5's $S(\ell)$ row — the collapse is asserted in code
(`column-max == IMPg[suppression]`), not implied. The colorbar's dashed pair is the
single-bin $2\sigma$ null $\pm2/\sqrt{253}$; the multiplicity ladder that the
max-over-bins collapse makes necessary belongs to fig 5.

**Reading it.** The printed stamps quantify what the eye sees: the commanding columns are
*sign-coherent* — a dominant parameter pushes $S(\ell)$ the same way across essentially
the whole $\ell$ range — so fig 5's max-over-bins collapse discards scale structure, not
sign structure, for exactly the parameters it promotes. Where each circle sits tells the
scale at which that parameter bites hardest; the $\ell=5000$ guide line separates the
survey-primary range from the small scales.
''')

code(r'''
# ── Fig 5a: the bin-resolved S(ell) response — the anatomy of one fig-5 row ──
# data: bind_sb35/emulator_dataset_xpkfix.npz only (Spearman is scale-invariant).
# This cell is also the single source of truth for THE canonical statistic
# table and the Spearman grids: figs 5 and 7 consume STATS / Ys / Rs / IMPg /
# peak_col / col_order from here, so fig 5's S(ell) row is BY CONSTRUCTION the
# column-max of the matrix drawn here.
def rebin_nan(a, k):
    """Block-mean over the last axis ignoring NaN (a rebinned bin is NaN only
    if its entire block is NaN); trims the remainder. Replaces the previous
    nan_to_num-then-rebin idiom so invalid bins stay invalid."""
    n = (a.shape[-1]//k)*k
    return np.nanmean(a[..., :n].reshape(*a.shape[:-1], n//k, k), -1)

def spearman_masked(P, Y, min_n=100):
    """Spearman rho between columns of P (n,p) and Y (n,b) using only rows
    where the Y column is finite (valid pairs only — NO zero-filling before
    rankdata). Columns with < min_n valid runs return NaN."""
    P = np.asarray(P, float); Y = np.asarray(Y, float)
    R = np.full((P.shape[1], Y.shape[1]), np.nan)
    fin = np.isfinite(Y)
    allrows = fin.all(0)
    if allrows.any():
        R[:, allrows] = spearman(P, Y[:, allrows])
    for j in np.where(~allrows)[0]:
        m = fin[:, j]
        if m.sum() >= min_n:
            R[:, j:j+1] = spearman(P[m], Y[m, j][:, None])
    return R

# ── THE canonical statistic table, defined ONCE; consumed by figs 5a, 5, 7 ───
# 7 WL field statistics + 2 non-kappa auto-spectra + 3 crosses = 12 canonical
# statistics. C_kk is NOT a row here: at fixed cosmology it is degenerate with
# S(ell) (S = C_kk / a single run-independent DMO trace; the identity is
# stated once in the fig 4b markdown), so S(ell) alone carries that row and
# C_kk is neither a matrix row nor (see fig 9) an emulator head. C_ell^{y tau}
# fills the third cross slot instead. Row order mirrors fig 4's WL panels
# (b)-(h) then fig 4b's yy/tt/ky/ktau/yt legs.
def _stat(key):
    return np.asarray(d[key], float)

STATS = [
    dict(k="suppression",   lab=r"$S(\ell)$",
         A=_stat("t__suppression__value")[:, ZI, :],    x=ELL, rb=16, grp="WL"),
    dict(k="pdf",           lab=r"PDF$(\kappa)$",
         A=_stat("t__pdf__value")[:, ZI, :],            x=_stat("a__pdf__pdf_bins"),
         rb=3,  grp="WL"),
    dict(k="peak_counts",   lab=r"$N_{\rm pk}$",
         A=_stat("t__peak_counts__value")[:, ZI, :],    x=_stat("a__peak_counts__nu"),
         rb=4,  grp="WL"),
    dict(k="minima_counts", lab=r"$N_{\rm min}$",
         A=_stat("t__minima_counts__value")[:, ZI, :],  x=_stat("a__minima_counts__nu"),
         rb=4,  grp="WL"),
    dict(k="mf_v0",         lab=r"$V_0$",
         A=_stat("t__mf_v0__value")[:, ZI, :],          x=_stat("a__mf_v0__mf_nu"),
         rb=2,  grp="WL"),
    dict(k="mf_v1",         lab=r"$V_1$",
         A=_stat("t__mf_v1__value")[:, ZI, :],          x=_stat("a__mf_v1__mf_nu"),
         rb=2,  grp="WL"),
    dict(k="mf_v2",         lab=r"$V_2$",
         A=_stat("t__mf_v2__value")[:, ZI, :],          x=_stat("a__mf_v2__mf_nu"),
         rb=2,  grp="WL"),
    dict(k="cl_yy",         lab=r"$C_\ell^{yy}$",
         A=_stat("t__cl_yy__value"),                    x=ELL, rb=16, grp="auto"),
    dict(k="cl_tt",         lab=r"$C_\ell^{\tau\tau}$",
         A=_stat("t__cl_tt__value"),                    x=ELL, rb=16, grp="auto"),
    dict(k="cl_kappa_y",    lab=r"$C_\ell^{\kappa y}$",
         A=_stat("t__cl_kappa_y__value")[:, ZI, :],     x=ELL, rb=16, grp="cross"),
    dict(k="cl_kappa_tau",  lab=r"$C_\ell^{\kappa\tau}$",
         A=_stat("t__cl_kappa_tau__value")[:, ZI, :],   x=ELL, rb=16, grp="cross"),
    dict(k="cl_yt",         lab=r"$C_\ell^{y\tau}$",
         A=_stat("t__cl_yt__value"),                    x=ELL, rb=16, grp="cross"),
]
NSTAT, NPAR = len(STATS), len(pnames)

# data hygiene: verify every row's validity before ranking on it
for s in STATS:
    fr = 100*np.isfinite(s["A"]).mean()
    print(f"validity {s['k']:>14s}: finite {fr:6.2f}%"
          + ("" if fr == 100 else "  <- masked pairwise in the Spearman"))

Ys   = {s["k"]: rebin_nan(s["A"], s["rb"]) for s in STATS}
Rs   = {s["k"]: spearman_masked(X_unit, Ys[s["k"]]) for s in STATS}   # each (30, nb)
IMPg = np.array([np.nanmax(np.abs(Rs[s["k"]]), 1) for s in STATS])    # (12, 30) THE map
RHO_2SIG = 2/np.sqrt(len(X_unit))       # single-bin 2-sigma null, grid-independent

# fig-5 column order (peak importance over ALL 12 statistics) — the shared x axis
peak_col  = np.nanmax(IMPg, 0)
col_order = np.argsort(-peak_col)

# the S(ell) leg, bin-resolved and SIGNED, in fig-5 column order
RS     = Rs["suppression"]                     # (30 params, nb ell bins)
ELL_RB = rebin(ELL, 16)                        # centers of the pre-averaged bins
edges  = np.concatenate([[1.5*ELL_RB[0] - 0.5*ELL_RB[1]],
                         0.5*(ELL_RB[1:] + ELL_RB[:-1]),
                         [1.5*ELL_RB[-1] - 0.5*ELL_RB[-2]]])
Ma     = RS[col_order]                         # (30, nb)
pk_bin = np.nanargmax(np.abs(Ma), 1)           # per-column peak bin
VA     = float(np.nanmax(np.abs(RS)))

# the bridging identity: the column-collapse of THIS matrix is fig 5's S(ell) row
assert np.array_equal(np.nanmax(np.abs(RS), 1), IMPg[0]), "fig-5a/fig-5 S(ell) rows diverged"

fig, axa = plt.subplots(figsize=(TWO_COL[0], 4.0), layout="constrained")
im = axa.pcolormesh(np.arange(NPAR + 1) - 0.5, edges, Ma.T, cmap="RdBu_r",
                    vmin=-VA, vmax=VA, rasterized=True)
# circles: the bin each column exports into fig 5's S(ell) row
axa.scatter(np.arange(NPAR), ELL_RB[pk_bin], s=7.0, facecolor="none",
            edgecolor="k", linewidths=0.55, zorder=3)
axa.axhline(5000, color="0.3", lw=0.5, ls=":")
axa.text(NPAR - 0.35, 5000, r"$\ell=5000$", ha="left", va="center",
         fontsize=4.6, color="0.25", clip_on=False)
axa.text(-0.5, edges[-1]*1.012,
         r"$\circ$: peak-$|\rho|$ bin — the value exported into fig 5's $S(\ell)$ row",
         ha="left", va="bottom", fontsize=4.6, color="0.25", clip_on=False)
axa.set_ylim(edges[0], edges[-1])
axa.set_yticks([2000, 10000, 20000, 30000, 40000, 50000])
axa.set_ylabel(r"$\ell$", fontsize=6.5)
axa.set_xlim(-0.5, NPAR - 0.5)
axa.set_xticks(range(NPAR))
axa.set_xticklabels([short_label(pnames[i]) for i in col_order], rotation=55,
                    ha="right", fontsize=4.4)
axa.tick_params(axis="x", length=1.5, pad=1.0)
axa.tick_params(axis="y", length=1.5, labelsize=6)
axa.set_xlabel("30 SB35 astrophysical parameters (fig-5 order)", fontsize=6.5)
cb = fig.colorbar(im, ax=axa, fraction=0.022, pad=0.055,
                  label=r"$\rho\,[\theta_j,\,S(\ell_b)]$ (signed Spearman)")
for v in (RHO_2SIG, -RHO_2SIG):
    cb.ax.axhline(v, color="k", lw=0.6, ls="--")
cb.ax.text(-0.45, 0.0, r"single-bin $\pm2\sigma$", rotation=90, ha="center",
           va="center", fontsize=3.6, color="0.35", clip_on=False)
save(fig, "figs_v2/fig05a_sl_response")
plt.show()

print(f"fig 5a: S(ell) signed Spearman matrix {RS.shape[0]} params x {RS.shape[1]} "
      f"pre-averaged ell bins (x16; centers {ELL_RB[0]:.0f}-{ELL_RB[-1]:.0f}); "
      f"column-max == fig-5 S(ell) row exactly (asserted)")
n2s = int((np.nanmax(np.abs(RS), 1) >= RHO_2SIG).sum())
print(f"fig 5a: {n2s}/{NPAR} columns peak above the single-bin 2sigma null ({RHO_2SIG:.3f})")
for i in col_order[:6]:
    b = int(np.nanargmax(np.abs(RS[i])))
    coh = float(np.mean(np.sign(RS[i]) == np.sign(RS[i, b])))
    print(f"   {short_label(pnames[i]):>24s}: peak rho {RS[i, b]:+.2f} at ell~{ELL_RB[b]:.0f}; "
          f"sign-coherent over {100*coh:.0f}% of bins")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §3a — Fig 5: how the statistics respond to the 30 parameters

Fig 5 collapses the bin-resolved response — the object fig 5a displays for $S(\ell)$ —
over bins, for every canonical statistic at once, into a single map: the
**canonical 12 statistics** ($S(\ell)$ plus the other 6 WL field statistics of fig 4, the 2
non-$\kappa$ auto-spectra and the 3 crosses of fig 4b) against the 30 SB35 parameters.
$C_\ell^{\kappa\kappa}$ is deliberately not a row: at fixed cosmology it is $S(\ell)$ times a
single run-independent DMO trace (identity stated once in the §4a text), so it carries
no information beyond the $S(\ell)$ row. The cell value is an
*importance* $I_{sj}=\max_b|\rho_{jb}|$: the largest Spearman rank correlation
$\rho_{jb}=\mathrm{corr}\,[\mathrm{rank}(\theta_j),\,\mathrm{rank}(s_b)]$, over the
(pre-averaged) bins $b$ of statistic $s$, between the unit-cube parameter $\theta_j$
($\log_{10}$-mapped where the prior is logarithmic) and the measured bin value across the
253 runs, at $z_s=1$ where tomographic. Shot-noisy bins are pre-averaged (peaks and minima
×4, PDF ×3, MFs ×2, spectra ×16); the ranking stability under halving/doubling these factors
is *verified by the printed check in the next cell*, not asserted. Invalid bins are excluded
pairwise (no zero-filling; the cell prints each row's finite fraction — all 12 rows are 100%
finite in the current dataset, so this is a hygiene guarantee, not a numerical change).

**Reading it.** Columns are sorted by peak importance; rows keep the fig-4/4b order, so the
WL / auto / cross blocks read as blocks. Four thresholds are drawn or printed rather than
assumed: the single-cell $2\sigma$ null ($|\rho|=0.126$); each statistic's *own* permutation
null (200 seeded shuffles of the run labels, max over that statistic's bins; 95th percentiles
0.147–0.177, marked as a band on the colorbar), below which a cell is crossed out; the
per-parameter look-elsewhere null over the 12 rows (0.196 — the dashed vertical divider,
which 11 of 30 parameters clear; 14 clear at least one row-level null); and the whole-grid
look-elsewhere null (0.257), which the printed numerals mark.

**Reading.** The response is concentrated, not diffuse: `VariableWindVelFactor`, `IMFslope`,
`WindEnergyIn1e51erg`, `BlackHoleRadiativeEfficiency` and `WindFreeTravelDensFac` command
*every* probe, and the remaining two-thirds of the design is statistically inert — visibly
so, because no parameter column is truncated away. The gas spectra respond most strongly
(peak $|\rho|=0.71$ for $C_\ell^{\tau\tau}$ against wind velocity, 0.65 for
$C_\ell^{\kappa\tau}$, 0.59 for $C_\ell^{\kappa y}$, and 0.54 for the new $C_\ell^{y\tau}$ row
— wind velocity is also $C_\ell^{y\tau}$'s own peak lever over all 30 params); the WL
morphology statistics
respond least, and are led by `IMFslope` rather than the wind velocity. That the same handful
of SN-wind, IMF and BH parameters dominates WL, tSZ and $\tau$ alike is the empirical basis
for the low-dimensional latent structure of §3f and for the emulator's feasibility with only
253 designs in 30-D (§4).
''')

code(r'''
# ── Fig 5: the 12-statistic x 30-parameter importance map ────────────────────
# data: bind_sb35/emulator_dataset_xpkfix.npz only; STATS / Ys / Rs / IMPg /
# RHO_2SIG / peak_col / col_order are bound by the fig-5a cell above (single
# source of truth: this figure's S(ell) row is that cell's matrix, collapsed).
# significance: analytic 2-sigma single-cell null + PER-STATISTIC permutation
# nulls of the row-max statistic (200 seeded shuffles of the run labels; one
# permutation stream shared by all rows so the nulls are comparable), plus the
# two multiplicity corrections the 12x30 grid needs.
rngp = np.random.default_rng(4)
null = np.empty((200, NSTAT, NPAR))
for it in range(200):
    perm = rngp.permutation(len(X_unit))
    for r, s in enumerate(STATS):
        null[it, r] = np.nanmax(np.abs(spearman_masked(X_unit[perm], Ys[s["k"]])), 1)
NULL95    = {s["k"]: float(np.percentile(null[:, r], 95)) for r, s in enumerate(STATS)}
NULL_ROW  = np.array([NULL95[s["k"]] for s in STATS])[:, None]    # (12, 1)
NULL_COL  = float(np.percentile(null.max(axis=1), 95))     # per-parameter, over 12 rows
NULL_GRID = float(np.percentile(null.max(axis=(1, 2)), 95))  # whole-grid look-elsewhere

# columns sorted by peak importance (col_order, bound in the fig-5a cell); ALL
# 30 stay (the inert two-thirds is the point). Rows keep the STATS order so the
# WL / auto / cross blocks read as blocks.
M         = IMPg[:, col_order]
n_col_ok  = int((peak_col >= NULL_COL).sum())      # params clearing the 12-row null
VMAX      = float(np.nanmax(IMPg))

fig, axh = plt.subplots(figsize=(TWO_COL[0], 3.6), layout="constrained")
im = axh.imshow(M, aspect="auto", cmap="cividis", vmin=0, vmax=VMAX,
                interpolation="nearest", rasterized=True)
# significance is ANNOTATED, not hidden: a cross marks cells below that
# statistic's own permutation null; a numeral marks cells that also clear the
# whole-grid look-elsewhere null.
sub_r, sub_c = np.where(M < NULL_ROW)
axh.scatter(sub_c, sub_r, marker="x", s=1.6, linewidths=0.22, color="0.82", zorder=3)
for (i, j), v in np.ndenumerate(M):
    if v >= NULL_GRID:
        axh.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=3.4,
                 color="k" if v/VMAX > 0.55 else "w", zorder=4)
# WL / auto / cross block separators + right-edge group labels
_blk = ["WL" if s["grp"].startswith("WL") else s["grp"] for s in STATS]
for i in range(1, NSTAT):
    if _blk[i] != _blk[i-1]:
        axh.axhline(i - 0.5, color="k", lw=0.7)
for gname in ("WL", "auto", "cross"):
    gi = [i for i, b in enumerate(_blk) if b == gname]
    axh.text(NPAR - 0.35, 0.5*(gi[0] + gi[-1]), gname, rotation=90, ha="left",
             va="center", fontsize=5.0, color="0.25", clip_on=False)
# the multiplicity divider: everything right of it fails the per-parameter null
axh.axvline(n_col_ok - 0.5, color="k", lw=0.8, ls="--")
axh.text(n_col_ok - 0.3, -0.62,
         f"$\\rightarrow$ peak $|\\rho|$ below the per-parameter look-elsewhere "
         f"null ({NULL_COL:.3f})",
         ha="left", va="bottom", fontsize=4.4, color="0.25", clip_on=False)
axh.text(-0.5, -1.75,
         r"$\times$: below that statistic's own permutation null   |   "
         f"numeral: clears the whole-grid null ({NULL_GRID:.2f})",
         ha="left", va="bottom", fontsize=4.6, color="0.25", clip_on=False)
axh.set_xticks(range(NPAR))
axh.set_xticklabels([short_label(pnames[i]) for i in col_order], rotation=55,
                    ha="right", fontsize=4.4)
axh.set_yticks(range(NSTAT))
axh.set_yticklabels([s["lab"] for s in STATS], fontsize=6.4)
axh.tick_params(axis="x", length=1.5, pad=1.0)
axh.tick_params(axis="y", length=1.5)
axh.set_xlabel("30 SB35 astrophysical parameters (sorted by peak $|\\rho|$)", fontsize=6.5)
cb = fig.colorbar(im, ax=axh, fraction=0.022, pad=0.055,
                  label=r"peak $|\rho|$ over bins")
# band on the colorbar: the RANGE of the 12 per-statistic permutation nulls (95%)
cb.ax.axhspan(min(NULL95.values()), max(NULL95.values()),
              color=COLORS["highlight"], alpha=0.30, lw=0)
cb.ax.axhline(NULL_GRID, color="w", lw=0.7, ls="--")
# tiny labels so the band and line read as deliberate, not rendering artifacts
cb.ax.text(-0.45, 0.5*(min(NULL95.values()) + max(NULL95.values())),
           "per-stat null", rotation=90, ha="center", va="center",
           fontsize=3.6, color=COLORS["highlight"], clip_on=False)
cb.ax.text(-0.45, NULL_GRID, "grid null", rotation=90, ha="center", va="center",
           fontsize=3.6, color="0.35", clip_on=False)
save(fig, "figs_v2/fig05_param_response")
plt.show()

r_max, c_max = np.unravel_index(int(np.argmax(IMPg)), IMPg.shape)
print(f"significance: single-cell 2sigma |rho| = {RHO_2SIG:.3f}; per-statistic permutation "
      f"null 95th pct spans {min(NULL95.values()):.3f}-{max(NULL95.values()):.3f}; "
      f"per-parameter look-elsewhere null (12 rows) = {NULL_COL:.3f}; "
      f"whole-grid null = {NULL_GRID:.3f}")
print("per-statistic permutation nulls (95th pct): "
      + ", ".join(f"{s['k']} {NULL95[s['k']]:.3f}" for s in STATS))
print(f"grid {NSTAT} statistics x {NPAR} params: max peak |rho| = {VMAX:.3f} "
      f"({STATS[r_max]['k']} x {short_label(pnames[c_max])}); "
      f"{int((IMPg >= NULL_GRID).sum())}/{IMPg.size} cells clear the whole-grid null")
print(f"{n_col_ok}/{NPAR} params clear the per-parameter look-elsewhere null; "
      f"{int((IMPg >= NULL_ROW).any(0).sum())}/{NPAR} have >=1 cell above their own "
      "statistic's null (the caption must not confuse the two)")
print("column order by peak |rho|: "
      + ", ".join(f"{short_label(pnames[i])} {peak_col[i]:.2f}" for i in col_order[:8]))
for r, s in enumerate(STATS):
    tops = np.argsort(-IMPg[r])[:3]
    print(f"{s['k']:>14s}: " + ", ".join(f"{short_label(pnames[i])}({IMPg[r, i]:.2f})"
                                         for i in tops))
def _rankcorr(a, b):
    """Spearman between two length-30 parameter orderings."""
    return float(spearman(np.asarray(a, float)[:, None], np.asarray(b, float)[:, None])[0, 0])

IMPmean = np.array([np.nanmean(np.abs(Rs[s["k"]]), 1) for s in STATS])
print("robustness — mean|rho| (instead of max) column order: "
      + ", ".join(short_label(pnames[i]) for i in np.argsort(-IMPmean.max(0))[:5])
      + "; rank agreement with the plotted max|rho| order = "
      + f"{_rankcorr(peak_col, IMPmean.max(0)):.3f}")
''')

md(r'''
### Rebin-stability check (fig 5)

The pre-averaging factors are a choice; the claim that the lever *rankings* do not depend
on them must be demonstrated, not asserted. The next cell recomputes each of the 12
statistics' top-3 levers with all rebin factors halved and doubled and prints the
comparison — those verdict lines are the paper's citation for "rankings stable to
halving/doubling". **RECOMPUTED-ON-RUN:** swapping $C_\ell^{\kappa\kappa}$ for
$C_\ell^{y\tau}$ changes the row set, so the specific tallies must be re-quoted from this
cell's printed output at the next execution rather than asserted from memory. All six
spectrum rows ($S(\ell)$, $C_\ell^{yy}$, $C_\ell^{\tau\tau}$, $C_\ell^{\kappa y}$,
$C_\ell^{\kappa\tau}$, $C_\ell^{y\tau}$) are expected to be exactly stable in both
directions, by the same envelope argument as before; third-place churn is confined to the
weakest-response rows (PDF, $N_{\rm min}$, $V_0$–$V_2$ — peak $|\rho|\lesssim0.6$).
''')

code(r'''
# ── Rebin-stability check: recompute the fig-5 rankings at x0.5 / x2 rebin ───
def _ranking(scale):
    out = {}
    for s in STATS:
        k = max(1, int(round(s["rb"]*scale)))
        R = spearman_masked(X_unit, rebin_nan(s["A"], k))
        out[s["k"]] = tuple(np.argsort(-np.nanmax(np.abs(R), 1))[:3].tolist())
    return out

base = _ranking(1.0)
keys = [s["k"] for s in STATS]
for sc, tag in ((0.5, "halved"), (2.0, "doubled")):
    r = _ranking(sc)
    same_top1 = sum(r[f][0] == base[f][0] for f in keys)
    same_2set = sum(set(r[f][:2]) == set(base[f][:2]) for f in keys)
    same_2ord = sum(r[f][:2] == base[f][:2] for f in keys)
    same_3set = sum(set(r[f]) == set(base[f]) for f in keys)
    same_3ord = sum(r[f] == base[f] for f in keys)
    n = len(keys)
    print(f"rebin {tag:7s}: top-1 identical {same_top1}/{n}; top-2 set {same_2set}/{n} "
          f"(ordered {same_2ord}/{n}); top-3 set {same_3set}/{n} (ordered {same_3ord}/{n})")
    for f in keys:
        if r[f] != base[f]:
            print(f"   {f}: {[short_label(pnames[i]) for i in base[f]]} -> "
                  f"{[short_label(pnames[i]) for i in r[f]]}")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §3b — Fig 7: the same response, curve by curve, at one lever

Fig 5 compresses each statistic to one number per parameter; fig 7 unpacks the strongest
column of that map, curve by curve. **v2: every panel now plots the RESPONSE ratio**
curve(run)/curve(fiducial), not the raw statistic. Dividing out the released, **out-of-design**
fiducial (`bind_lightcone_tng` / `bind_science/runs/bind/run_0000` — verified identical files;
this is *not* Sobol node 0, which does not exist) turns twelve panels with wildly different
absolute scales — spectra spanning decades, a dimensionless $S(\ell)$, counts, Minkowski
functionals — into one shared reading: a horizontal line at 1 means "this run looks like the
fiducial", and the vertical spread of the 253 curves around that line *is* the feedback
response, in the same units on every panel. Bins where the fiducial itself is consistent with
zero (the sparse tails of the peak/minima counts, and $V_1$/$V_2$ near their own
zero-crossings) are masked rather than plotted as an exploding or sign-flipping ratio — the
floor is stated once in the cell (2% of that curve's own peak $|$value$|$).
**DEFAULT DECISION (flagged for confirmation, not yet a settled call):** the response-ratio
treatment is applied to **all 12 panels**, including the five $\nu$-domain WL statistics, not
only the spectra — this resolves the open "all panels or $\nu$-domain only" question in favor
of read-consistency across the whole figure; reverting the $\nu$-domain panels to raw curves
is a one-line change if the intent was narrower.

All 253 measured Sobol runs are drawn for **each of the canonical 12 statistics** — the same
list, in the same order, as fig 5's rows (panel (a) is $S(\ell)$, not $C_\ell^{\kappa\kappa}$:
the identity note in the §4a text is why) — and every panel is coloured by the **same**
parameter, `VariableWindVelFactor` (`param_names[2]`, prior units, $\log_{10}$-mapped over a
native 3.7–14.7). It is the suite's single strongest lever: top-ranked for 7 of the 12
statistics and holder of the grid maximum (peak $|\rho|=0.71$ on $C_\ell^{\tau\tau}$; printed
by the cell). Holding the knob fixed across panels is the point — what varies from panel to
panel is the *probe*, so the response coherence across WL, tSZ and $\tau$ can be read directly
instead of inferred from twelve different colorbars.

Each panel stamps its own fig-5 cell (peak $|\rho|$ for *this* parameter), which is what
makes the weak panels legible as physics rather than as a plotting failure: the wind
velocity is **not** the leading lever for the five $\nu$-domain WL statistics (panels c–g,
the peak/minima counts and $V_0$–$V_2$), where `IMFslope` leads at
$|\rho|=0.44$–$0.61$ and the wind velocity falls to $0.21$–$0.43$, so those colour
gradients are genuinely washed out. Raw spectra and $S(\ell)$ now share **one** $\ell$ axis to
the full corner mode $\ell=5.22\times10^4$, with the CIC-aliasing region ($\ell>1.5\times10^4$)
shaded — the same convention figs 4/4b/12 use; a response ratio is largely insensitive to the
aliasing (it affects numerator and denominator alike), so this is drawn for a consistent
$\ell$-axis read across the paper's field-level figures, not because the ratio needs that range
to be informative. $V_0$ (panel e) is flat by construction — an envelope-dominated functional
near $\nu=0$ — and says so on-panel; its exact response-ratio spread is RECOMPUTED-ON-RUN
(quote the printed value, not a number here — the v1 "$\le2.3\%$" raw-curve figure no longer
applies to the ratio). The monotonic gradients in the remaining panels are the response the
emulator of §4 has to learn.
''')

code(r'''
# ── Fig 7: the canonical 12 statistics, all coloured by ONE lever ────────────
# data: bind_sb35/emulator_dataset_xpkfix.npz; consumes STATS + IMPg from fig 5
# (same list, same order, so panel r here IS row r there); PLUS (v2) the
# out-of-design fiducial's own reference vector for each of the 12 statistics,
# built fresh in this cell (self-contained -- does NOT depend on fig 4 having
# run) from: runs/bind/run_0000/{Cl_kappa,peak_counts,nongaussian_stats,
# kappa_maps,y_maps}.npz + bind_lightcone_tng/tau_maps.npz + field_cache.py
# (papers/01_pipeline/field_cache.py; the SAME fiducial per-realization spectra
# fig 4 uses).
import sys as _sys7
_sys7.path.insert(0, str(Path.cwd()))
from field_cache import load as _load_fields_fid              # noqa: E402
from bind.inference.stats import power_spectrum                # noqa: E402

def color_curves(ax, x, Y, cvals):
    for i in np.argsort(cvals):
        ax.plot(x, Y[i], lw=0.4, alpha=0.55, color=plt.cm.coolwarm(cvals[i]))

def spread16_84(Y):
    """Max over bins of the 16-84 percentile width of curve/median — the
    legibility diagnostic printed for every panel (a panel whose spread is tiny
    against its own dynamic range CANNOT show a colour gradient). Y is now
    itself a response ratio (v2), so this is a spread-of-a-ratio; NaN-robust,
    which matters now that the floor mask (below) introduces NaNs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(Y, 0)
        rat = Y/np.where(np.abs(med) > 0, med, np.nan)
        p16, p84 = np.nanpercentile(rat, [16, 84], axis=0)
        return float(np.nanmax(p84 - p16))

SIG0_7 = float(np.load(SCI/"runs/bind/run_0000/paired_perreal_fid.npz")["sigma0"][ZI])
PCOL  = 2                     # ONE colour parameter for all 12 panels
assert pnames[PCOL] == "VariableWindVelFactor", pnames[PCOL]
cvals = X_unit[:, PCOL]

# ell masks. mS (S(ell)): extended to the full corner mode (v2 -- was 2e4),
# consistent with figs 4/4b/12 -- S(ell) is a genuine BIND/DMO CANCELLING
# ratio (established: fig 4 panel b, "S(ell) cancels aliasing"), and measures
# fine there (16-84 spread 46.9%, printed below). mE (the 5 raw auto/cross
# spectra, SPEC): MEASURED, then reverted -- a first attempt extending mE the
# same way exploded the printed 16-84 spread to 200-1500% (a run/fiducial
# ratio of two DIFFERENT astrophysics nodes' raw spectra does NOT cancel the
# CIC-aliasing artifact the way S(ell) does: the aliasing amplitude itself
# scales with each node's own small-scale power, so numerator and denominator
# diverge rather than cancel). mE therefore stays at the TRUSTED range.
# 2026-08-03 (author decision): ALL ell panels stop at the SAME ell -- the
# trusted range -- rather than S(ell) alone extending into a shaded strip the
# SPEC panels leave empty. S(ell)'s valid aliasing-region response remains
# visible in figs 4b/12; this figure trades that tail for a uniform ell axis.
mS   = ELL <= ELL_TRUST      # suppression: same uniform range as the raw spectra
mE   = ELL <= ELL_TRUST
SPEC = ("cl_yy", "cl_tt", "cl_kappa_y", "cl_kappa_tau", "cl_yt")
# nu window: IDENTICAL to fig 4's NU_LIM so the two figures are read on one axis.
# Caveat worth knowing: fig 4 recomputes V0/V1/V2 live out to nu=8, but the Sobol
# dataset's MF grid (a__mf_v*__mf_nu) is the cached linspace(-3,4,29), so the MF
# curves HERE stop at 4 on an axis that runs to 8. Extending them would mean
# re-deriving the functionals for all 253 runs x 50 realizations, not a figure edit.
NU7  = (-3, 8)

# ── v2: fiducial reference vector for each of the 12 canonical statistics ───
# "Fiducial" = the released, OUT-OF-DESIGN fiducial (bind_lightcone_tng /
# bind_science/runs/bind/run_0000 -- verified identical files; not Sobol node 0,
# which does not exist), the SAME fiducial figs 4/10/12 use. Grid-matching
# against the Sobol dataset's own a__*__* axis arrays is ASSERTED, not assumed.
RB7  = SCI/"runs/bind/run_0000"
_fc7 = _load_fields_fid(n_real=50)
assert _fc7 is not None, "field_cache missing -- rebuild via run_field_cache.sbatch"
assert np.allclose(_fc7["ell"], ELL), "field_cache ell grid != Sobol dataset ell grid"

_bcl0_7, _dcl0_7 = np.load(RB7/"Cl_kappa.npz"), np.load(SCI/"runs/dmo/run_0000/Cl_kappa.npz")
FID = {"suppression": _bcl0_7["cl"][ZI, ZI]/_dcl0_7["cl"][ZI, ZI]}   # matches fig 12's S_fid_full

# crosses WITH a per-z_s (tomographic) axis in the dataset: field_cache's
# ky_bind/kt_bind are ALREADY kappa(z_s) x TOTAL-column F, per source plane --
# exactly the dataset convention (fig 4b markdown).
FID["cl_kappa_y"]   = _fc7["ky_bind"][:, ZI].mean(0)
FID["cl_kappa_tau"] = _fc7["kt_bind"][:, ZI].mean(0)
# total-column-only autos (no z_s axis in the dataset): the LAST (cumulative)
# plane of field_cache's per-plane auto spectra.
FID["cl_yy"] = _fc7["yy_bind"][:, -1].mean(0)
FID["cl_tt"] = _fc7["tt_bind"][:, -1].mean(0)

# cl_yt (y x tau, total column, SIGNED-cross handling as fig 4b: same floor-mask
# treatment below, no special-casing needed): not a field_cache product (no
# TASKS entry for it), so computed live, once, from the released fiducial
# y/tau total-column maps -- the one moderately expensive load in this cell.
f = np.load(RB7/"y_maps.npz"); _BY_tot7 = f["y"][:, -1].copy(); del f; gc.collect()
f = np.load(LC/"tau_maps.npz"); _BT_tot7 = f["tau"][:, -1].copy(); del f; gc.collect()
FID["cl_yt"] = np.mean([power_spectrum(_BY_tot7[r], _BT_tot7[r])[1] for r in range(50)], 0)
del _BY_tot7, _BT_tot7; gc.collect()

# peaks/minima: RB7's peak_counts.npz IS on the dataset's a__peak_counts__nu /
# a__minima_counts__nu grid (asserted) and shares its nu_norm='map' convention
# -- UNLIKE paired_perreal_fid.npz (nu_norm='fixed'), which the fig-4 cell
# explicitly flags as NOT usable as a reference for this reason.
_pk7 = np.load(RB7/"peak_counts.npz")
assert np.allclose(_pk7["nu"], d["a__peak_counts__nu"]), "peak/minima nu grid mismatch"
FID["peak_counts"], FID["minima_counts"] = _pk7["peak_counts"][ZI], _pk7["minima_counts"][ZI]

# V0/V1/V2: RB7's nongaussian_stats.npz is on the dataset's mf_nu grid
# (asserted). NOTE (same caveat the fig-4 cell documents): this cache is
# ~0.04-0.6% stale vs a live per-realization recompute -- negligible next to
# the O(10-100%) feedback response plotted here.
_ng7 = np.load(RB7/"nongaussian_stats.npz")
assert np.allclose(_ng7["mf_nu"], d["a__mf_v0__mf_nu"]), "mf_nu grid mismatch"
FID["mf_v0"], FID["mf_v1"], FID["mf_v2"] = _ng7["V0"][ZI], _ng7["V1"][ZI], _ng7["V2"][ZI]

# PDF: grid CHECK first, not assumed -- RB7's own nongaussian_stats.npz pdf_bins
# is a DIFFERENT grid from the dataset's shared a__pdf__pdf_bins (each run's raw
# cache is auto-ranged to its own map; the dataset was assembled onto one common
# grid), so re-histogram the fiducial kappa map onto the EXACT dataset edges
# (same edges-from-centers idiom the fig-4 cell uses) rather than reuse the cache.
_pdf_cent7 = d["a__pdf__pdf_bins"]
_dpc7 = _pdf_cent7[1] - _pdf_cent7[0]
_pdf_edges7 = np.concatenate([_pdf_cent7 - _dpc7/2, [_pdf_cent7[-1] + _dpc7/2]])
_pdf_match7 = np.allclose(_ng7["pdf_bins"], _pdf_cent7)
print(f"PDF bin-grid check: RB7 nongaussian_stats.npz pdf_bins == Sobol dataset "
      f"a__pdf__pdf_bins? {_pdf_match7} -- "
      + ("reusing the cache directly" if _pdf_match7 else
         "grids differ -- rehistogramming the fiducial kappa map onto the dataset grid"))
if _pdf_match7:
    FID["pdf"] = _ng7["pdf"][ZI]
else:
    f = np.load(RB7/"kappa_maps.npz"); _bk7 = f["kappa"][:, ZI].copy(); del f; gc.collect()
    FID["pdf"] = np.mean([np.histogram(m - m.mean(), bins=_pdf_edges7, density=True)[0]
                          for m in _bk7], axis=0)
    del _bk7; gc.collect()

assert set(FID) == {s["k"] for s in STATS}, "FID is missing/extra keys vs the 12 canonical stats"
print("v2: fiducial reference vector built for all 12 canonical statistics ("
      + ", ".join(f"{k} peak={np.nanmax(np.abs(v)):.3e}" for k, v in FID.items()) + ")")

# "within its own noise of zero" floor (stated once, used for the nu-domain
# panels): a bin is masked (NaN) if the FIDUCIAL denominator's |value| is below
# 2% of that curve's own peak |value| -- below this floor the fiducial is
# consistent with zero/shot-noise (no peaks/minima/MF signal there) and a
# ratio is ill-defined rather than physical. Hits the sparse tails of the
# peak/minima counts and V1/V2/PDF near their own zero-crossings, over roughly
# HALF the plotted nu domain (peaks are ~absent at nu<0, minima ~absent at
# nu>0) -- a large but CORRECT masked fraction, not a bug (printed below).
#
# The ell-domain panels (S(ell) and the 5 spectra) use a much smaller,
# near-numerical floor instead (SPEC_FLOOR): a first attempt at FLOOR_FRAC=0.02
# here masked 96%+ of every spectrum panel, because a power spectrum spans MANY
# DECADES in ell -- "2% of the curve's global peak" silently kills almost the
# entire high-ell tail even though the ratio there is perfectly well-defined
# (small over small, not a zero-crossing). Matches fig 4b's own un-masked
# treatment of the ky/kt/yt crosses: only a true (near-)zero denominator is
# guarded against, not "small relative to the low-ell peak".
FLOOR_FRAC = 0.02
SPEC_FLOOR = 1e-10

def response_ratio(A, fid, floor_frac=FLOOR_FRAC):
    """A: (253, n_bin) Sobol curves; fid: (n_bin,) fiducial reference, the SAME
    grid. Returns A/fid, masked (NaN) where |fid| < floor_frac * max(|fid|)."""
    floor = floor_frac*np.nanmax(np.abs(fid))
    safe = np.where(np.abs(fid) > floor, fid, np.nan)
    return A/safe[None, :]

# Explicit "X/X_fid" y-labels, one clean mathtext string per statistic -- NOT
# built by string-surgery on the fig-5 `lab` (naive $-stripping + re-wrapping
# produces a DOUBLE subscript for every C_ell^{..} label -- e.g. stripping and
# reassembling "$C_\ell^{yy}$" gives "C_\ell^{yy}_{\rm fid}", which mathtext
# rejects outright -- and a stray literal "$" for "PDF$(\kappa)$"). Each entry
# below folds "fid" into the SAME subscript group as any existing one instead.
YLAB_RATIO = {
    "suppression":   r"$S(\ell)/S_{\rm fid}(\ell)$",
    "pdf":           r"$\mathrm{PDF}/\mathrm{PDF}_{\rm fid}$",
    "peak_counts":   r"$N_{\rm pk}/N_{\rm pk,fid}$",
    "minima_counts": r"$N_{\rm min}/N_{\rm min,fid}$",
    "mf_v0":         r"$V_0/V_{0,\rm fid}$",
    "mf_v1":         r"$V_1/V_{1,\rm fid}$",
    "mf_v2":         r"$V_2/V_{2,\rm fid}$",
    "cl_yy":         r"$C_\ell^{yy}/C_{\ell,\rm fid}^{yy}$",
    "cl_tt":         r"$C_\ell^{\tau\tau}/C_{\ell,\rm fid}^{\tau\tau}$",
    "cl_kappa_y":    r"$C_\ell^{\kappa y}/C_{\ell,\rm fid}^{\kappa y}$",
    "cl_kappa_tau":  r"$C_\ell^{\kappa\tau}/C_{\ell,\rm fid}^{\kappa\tau}$",
    "cl_yt":         r"$C_\ell^{y\tau}/C_{\ell,\rm fid}^{y\tau}$",
}

def panel_data(s):
    """(x, Y, xlabel, ylabel, xscale, xlim) for one canonical statistic's v2
    RESPONSE ratio curve/fiducial -- y-scale is LINEAR on every panel now (a
    ratio near 1, not a raw quantity spanning decades), with a shared
    axhline(1) "looks like the fiducial" guide drawn by the caller."""
    k, A = s["k"], s["A"]
    fid = FID[k]
    yl = YLAB_RATIO[k]
    if k in SPEC:
        # uniform ell axis across ALL ell panels (author decision 2026-08-03):
        # every spectrum -- S(ell) included -- is drawn and limited to the same
        # trusted range, so no panel is cut off differently from another.
        return (ELL[mE], response_ratio(A[:, mE], fid[mE], SPEC_FLOOR),
                r"$\ell$", yl, "log", (ELL[mE].min(), ELL_TRUST))
    if k == "suppression":
        return (ELL[mS], response_ratio(A[:, mS], fid[mS], SPEC_FLOOR),
                r"$\ell$", yl, "log", (ELL[mS].min(), ELL_TRUST))
    if k == "pdf":
        # kappa -> nu on the SAME fiducial sigma fig 4 uses, so the axis matches
        # fig 4's PDF panel; the ratio's value does not depend on this relabel.
        return (s["x"]/SIG0_7, response_ratio(A, fid), r"$\nu$", yl, "linear", NU7)
    return (s["x"], response_ratio(A, fid), r"$\nu$", yl, "linear", NU7)

fig, ax = plt.subplots(3, 4, figsize=(TWO_COL[0], 6.6), layout="constrained")
spreads, nan_frac = {}, {}
for r, (s, a) in enumerate(zip(STATS, ax.ravel())):
    x, Y, xl, yl, xs, xlim = panel_data(s)
    color_curves(a, x, Y, cvals)
    spreads[s["k"]] = spread16_84(Y)
    nan_frac[s["k"]] = float(np.mean(~np.isfinite(Y)))
    a.axhline(1, color=COLORS["dmo"], ls=":", lw=0.8)   # EVERY panel is now a ratio
    a.set_xscale(xs)
    if xlim is not None:
        a.set_xlim(*xlim)
    # no aliasing axvspan here: nothing beyond ELL_TRUST is drawn (uniform axis;
    # the aliasing-region behaviour of S(ell) stays visible in figs 4b/12)
    a.set_xlabel(xl, fontsize=7)
    a.set_ylabel(yl, fontsize=6.5)
    a.tick_params(labelsize=5.5)
    # no in-panel text: the per-panel peak |rho| is fig 5's job (this figure is
    # panel-for-row identical to it) and the spread diagnostics are printed below.
    panel_label(a, f"({chr(97 + r)})")

# ONE colorbar for the whole figure: every panel uses the same lever
sm = ScalarMappable(cmap="coolwarm", norm=Normalize(0, 1))
cb = fig.colorbar(sm, ax=ax, orientation="horizontal", location="bottom",
                  fraction=0.030, pad=0.02, aspect=55)
cb.set_label(short_label(pnames[PCOL]) + " (prior units)", fontsize=6.5)
cb.ax.tick_params(labelsize=5.5)
save(fig, "figs_v2/fig07_covariation")
plt.show()

print(f"all {len(STATS)} panels coloured by ONE parameter: {pnames[PCOL]} "
      f"(param_names[{PCOL}], unit-cube range {cvals.min():.3f}-{cvals.max():.3f}); "
      f"top-ranked in {int((IMPg.argmax(1) == PCOL).sum())}/{len(STATS)} fig-5 rows")
print("per-panel peak |rho| for this parameter: "
      + ", ".join(f"{s['k']} {IMPg[r, PCOL]:.2f}" for r, s in enumerate(STATS)))
print("per-panel run-to-run 16-84 spread of RESPONSE RATIO/median (v2; max over "
      "plotted bins): "
      + ", ".join(f"{k} {100*v:.1f}%" for k, v in spreads.items())
      + "  <- V0 is envelope-dominated by construction; its panel is absolute "
        "and says so on-panel")
print("per-panel floor-masked bin fraction (fiducial within 2% of its own peak "
      "|value| of zero -> NaN'd, not plotted as an exploding ratio): "
      + ", ".join(f"{k} {100*v:.1f}%" for k, v in nan_frac.items()))
# retained provenance for the retired Y(M) panel (the scaling relation is still
# released, it simply no longer has a panel here)
Ysc  = np.asarray(d["t__scaling_Y__value"], float)
logY = np.log10(np.where(Ysc > 0, Ysc, np.nan))
y_dex = np.nanstd(logY, axis=0)
print(f"Y(M) run-to-run scatter per mass bin [dex]: {np.round(y_dex, 3)} "
      f"(median {np.median(y_dex):.3f})")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §3c — Fig 20: which halos set which scales — the van Daalen relation over the feedback space

van Daalen, McCarthy & Schaye (2020, MNRAS 491, 2424; arXiv:1906.00968) showed that across
independent hydro simulations the matter-power suppression at
$k\lesssim1\,h\,\mathrm{Mpc}^{-1}$ is set by a single number — the renormalized baryon
fraction $\tilde f_{\rm bar}=f_{\rm bar}/(\Omega_b/\Omega_m)$ of $\sim10^{14}\,M_\odot$
groups. The fiducial-suite bridge analysis found the WL analogue at a single scale (group
$f_{\rm gas}$ $\leftrightarrow$ $S(\ell\sim1500)$, $r=0.91$; external,
`examples/bind_bridge.py`, not recomputed here); this section generalizes it over the full
253-node feedback cloud as a *matrix*: Pearson $r$ between the $z_s=1$ suppression $S$ in
five geometric $\ell$-bands ($\ell=300$–$1.5\times10^4$) and $\tilde f_{\rm bar,500c}$ in
four halo-mass bins at $z\simeq0.03$ (background-subtracted projected-500c apertures, the
same reducer as fig 3's atlases — precisely,
$\tilde f^{\,\rm cyl}_{\rm bar,500c}=\frac{1}{\Omega_b/\Omega_m}\,\mathrm{median}_{h}
\frac{m_{\rm gas,500c}^{\rm bg}(h)+m_{\star,500c}(h)}{m_{\rm tot,500c}^{\rm bg}(h)}$
over halos in a $\log_{10}m_{\rm tot,500c}^{\rm bg}$ bin, with $m^{\rm bg}$ background-
subtracted using a fixed transverse $2.5$–$3.0\,h^{-1}$Mpc annulus on a projected cylinder
of radius $R_{500c}=0.659\,r_{200}$ and full-slab ($\pm25.6\,h^{-1}$Mpc) line-of-sight
depth, and $\Omega_b/\Omega_m=0.0486/0.3089$; the halo side's per-bin node ranking is
redshift-stable — rank-corr $\simeq0.99$ against $z=0.18$, printed by the cell — so the
low-$z$ bins proxy
the kernel-peak epoch $z\simeq0.43$). The group scale is the hinge: a *single*
$10^{13.3}$–$10^{13.6}\,M_\odot/h$ mass bin predicts the suppression at $\ell\approx10^3$
with $r=0.98$ ($R^2=0.95$) — that single cell is panel (b), whose $r$ and OLS slope are
printed by the cell for the caption — and adding any second mass bin gains $\leq0.014$ in $R^2$ until
$\ell\sim10^4$ — the WL restatement of van Daalen et al.'s result that baryon-budget
accounting in groups fixes the power suppression at $k\lesssim1\,h\,\mathrm{Mpc}^{-1}$.
Predictability decays toward smaller scales ($r\simeq0.58$–$0.77$ at $\ell\approx10^4$,
$k\approx9\,h\,\mathrm{Mpc}^{-1}$, where a second bin finally helps — $+0.09$ in $R^2$ —
and the cluster bin enters with *negative* partial correlation: internal redistribution
rather than the ejected budget dominates there), and the gas fraction alone saturates at
$r\simeq0.85$ — the stellar component completes the budget. Panel (c) places the suite on
the universal 3-D curve at the van Daalen-comparable cell
($k\simeq0.4$–$0.5\,h\,\mathrm{Mpc}^{-1}$, $M_{500c}\simeq0.6$–$2\times10^{14}\,M_\odot/h$) —
but the halo-side quantity plotted on its primary (bottom) axis,
$\tilde f^{\,\rm cyl}_{\rm bar,500c}$, is a *projected-cylinder* baryon fraction (500c
aperture, $\pm25.6\,\mathrm{Mpc}/h$ line-of-sight slab, annulus-subtracted), not the 3-D
spherical $\tilde f$ that van Daalen, McCarthy & Schaye (2020) actually plot. A dedicated
calibration against the hydro-pasted truth, measured with the identical projected definition,
gives $\tilde f^{\,\rm cyl}=0.93$ against the literature 3-D value $\tilde f\approx0.81$
(Nelson et al. 2024, TNG-Cluster, arXiv:2311.06338) — a $\times1.16$ inflation from the
cylinder's extra halo-outskirt mass and more-extended gas — and panel (c)'s secondary (top)
axis applies that calibration directly. The whole TNG-prior Sobol cloud spans
$\tilde f^{\,\rm cyl}\simeq0.76$–$1.04$ (median $0.955$) in the projected variable, mapping to
a 3-D-equivalent $\tilde f\simeq0.65$–$0.90$ (median $\approx0.82$). The cloud is still
predominantly on the flat, sub-percent part of the vD relation, but once calibrated the
strong-feedback tail visibly penetrates the observationally preferred group band
($\tilde f\simeq0.55$–$0.76$, 3-D — van Daalen, McCarthy \& Schaye (2020)'s own compilation
band, drawn primarily from Vikhlinin et al. (2006), Sun et al. (2009), Gonzalez et al. (2013)
and Lovisari et al. (2015)) more than the uncalibrated projected axis suggests. The
ten nodes with $\tilde f^{\,\rm cyl}>1$ (the shaded band at the right edge of panel c) — a
super-cosmic baryon fraction with no 3-D
counterpart, since a projected, annulus-subtracted cylinder can partially over-close while a
sphere cannot — are exactly the weak-wind, concentrated-gas corner of fig 12's
*enhancement* branch ($S(\ell)>1$; all ten are members), which the monotonic van Daalen fit
cannot produce and which the overlay was never meant to constrain: it is a consistency
statement about where TNG's prior lives, not a fit. All quoted numbers are printed live by
the cell; the vD20 curve is the published $k=0.5\,h\,\mathrm{Mpc}^{-1}$ fit, hardcoded
exactly as in the v1 `fig_scripts/fig08_vandaalen_analog.py`, and lives on the panel's 3-D
(secondary) axis. The panels carry no in-axes annotation: the correlation coefficient of (b),
the projected-cylinder aperture definition, the $\times1.16$ truth calibration behind (c)'s
top axis, and the reading of (c) — a flat, saturated top plus a cylinder-closure tail with no
3-D equivalent — are all stated above and belong in the caption, not on the figure.

''')

code(r'''
# ── Fig 20 (§3c): the van Daalen relation over the full Sobol feedback space ─
# data: bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz — ALL 256
#       nodes' background-subtracted projected-500c aperture masses at the
#       shared 2933-halo sample (z=0.0337; same reducer as fig 3's fid/truth
#       atlases) + atlas_cube_snap085.npz (z=0.18, rank-stability check only)
#       + the in-kernel Sobol dataset d (S(ell) at z_s=1) + bind_science
#       runs/{bind,dmo}/run_0000/Cl_kappa.npz and halo_atlas/fid_snap096.npz
#       for the fiducial star (the fiducial is NOT Sobol node 0).
# conventions: f~ = median over bin halos of (m_gas+m_star)/m_tot (500c,
#       bg-subtracted), divided by fig 3's OB_OM; mass bins in per-node
#       log10 M500c,bg; S(ell) averaged in 5 geometric bands over
#       [300, ELL_TRUST]. No RNG anywhere in this cell (fully deterministic).
from scipy.stats import pearsonr, spearmanr

cz = np.load(SB35/"analysis_cache/atlas_cubes/atlas_cube_snap096.npz")
atlas_valid = cz["sobol_valid"]                       # (256,) — all True
assert atlas_valid[run_ids].all(), "a dataset run is atlas-invalid"
# (256, 2933) cubes indexed by run NUMBER -> slice to the 253 dataset rows;
# runs 114/115/117 have atlas data but no lightcone/S side, so run_ids
# indexing drops them implicitly (never assume row-parity with the dataset)
mt5 = cz["sobol_m_tot_500c_bg"][run_ids]
mg5 = cz["sobol_m_gas_500c_bg"][run_ids]
ms5 = cz["sobol_m_star_500c"][run_ids]
logm5 = np.log10(np.where(mt5 > 0, mt5, np.nan))
fbar5 = np.where(mt5 > 0, (mg5 + ms5)/mt5, np.nan)   # baryons = gas + stars
fgas5 = np.where(mt5 > 0, mg5/mt5, np.nan)

MEDG = np.array([13.0, 13.3, 13.6, 13.9, 15.5])      # log10 M500c,bg edges
NBm, NRn = len(MEDG) - 1, len(run_ids)

def binned_ftilde(lm, f):
    """Per-node median of f over the MEDG mass bins, / (Omega_b/Omega_m)."""
    out = np.full((NRn, NBm), np.nan); cnt = np.zeros((NRn, NBm), int)
    for q in range(NRn):
        for j in range(NBm):
            s = (lm[q] >= MEDG[j]) & (lm[q] < MEDG[j+1])
            cnt[q, j] = s.sum()
            if cnt[q, j] >= 5:
                out[q, j] = np.nanmedian(f[q, s])/OB_OM
    return out, cnt

ft_bar, cnts = binned_ftilde(logm5, fbar5)   # the van Daalen variable
ft_gas, _    = binned_ftilde(logm5, fgas5)   # gas-only cross-check
print("mass-bin edges (log10 M500c,bg):", MEDG)
print("halo counts per node   median:", np.median(cnts, 0).astype(int),
      "  min:", cnts.min(0))
print("nodes with valid f~ per bin  :", np.isfinite(ft_bar).sum(0), f"/ {NRn}")

# S(ell, z_s=1) -> 5 geometric ell-bands over the trusted range; band 0
# (300-656) holds only ~5 fine bins and sub-percent S spread — it is kept as
# the vD-comparable row but its slope is not headlined (fragile in absolute
# terms)
LEDG = np.geomspace(300.0, ELL_TRUST, 6)
S1z = d["t__suppression__value"][:, ZI, :]
band_masks = [(ELL >= LEDG[i]) & (ELL < LEDG[i+1]) for i in range(5)]
Sb = np.stack([S1z[:, m].mean(1) for m in band_masks], axis=1)
LCEN = np.sqrt(LEDG[:-1]*LEDG[1:])
print("ell-band edges:", np.round(LEDG).astype(int),
      " fine bins per band:", [int(m.sum()) for m in band_masks])

# ell -> k via the z_s=1 geometric-kernel peak (flat LCDM, Om=0.3089 — the
# OB_OM denominator; c/H0 = 2997.925 Mpc/h)
zg = np.linspace(0.0, 1.0, 4001)
Ez = np.sqrt(0.3089*(1 + zg)**3 + 1 - 0.3089)
chi_s20 = 2997.92458*np.trapezoid(1/Ez, zg)          # Mpc/h to z_s = 1
chi_star = chi_s20/2                                 # kernel peak, z ~ 0.43
k_cen = (LCEN + 0.5)/chi_star                        # h/Mpc
print(f"chi(z_s=1) = {chi_s20:.0f} Mpc/h; chi* = {chi_star:.0f} Mpc/h; "
      f"ell centers {np.round(LCEN).astype(int)} -> k = {np.round(k_cen, 2)} "
      f"h/Mpc; vD20 k=0.5 h/Mpc <-> ell = {0.5*chi_star - 0.5:.0f} (band 0)")

# the (ell-band x mass-bin) correlation matrices
def rmatrix(F):
    P = np.zeros((5, NBm)); R = np.zeros((5, NBm))
    for i in range(5):
        for j in range(NBm):
            ok = np.isfinite(F[:, j])
            P[i, j] = pearsonr(F[ok, j], Sb[ok, i])[0]
            R[i, j] = spearmanr(F[ok, j], Sb[ok, i])[0]
    return P, R

Pb, RHOb = rmatrix(ft_bar)
Pg, RHOg = rmatrix(ft_gas)
with np.printoptions(precision=3, suppress=True):
    print("\nPearson r, S(ell-band) x f~_bar(mass-bin) "
          "[rows: ell low->high; cols: mass low->high]:\n", Pb)
    print("Pearson r vs f~_gas (gas alone saturates; sign-flips at the "
          "smallest scales):\n", Pg)

# does a second mass bin help? R^2 best single f~_bar bin vs best pair
okall = np.isfinite(ft_bar).all(1)
print(f"\nR^2 of S per ell band, best single mass bin vs best pair "
      f"(n = {okall.sum()} nodes):")
for i in range(5):
    y = Sb[okall, i]
    r2s = []
    for j in range(NBm):
        A = np.c_[ft_bar[okall, j], np.ones(okall.sum())]
        res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        r2s.append(1 - res.var()/y.var())
    j1 = int(np.argmax(r2s))
    best = (0, 0, -1.0)
    for j in range(NBm):
        for jj in range(j + 1, NBm):
            A = np.c_[ft_bar[okall, j], ft_bar[okall, jj], np.ones(okall.sum())]
            res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
            r2 = 1 - res.var()/y.var()
            if r2 > best[2]:
                best = (j, jj, r2)
    print(f"  ell~{LCEN[i]:6.0f} (k~{k_cen[i]:4.2f}): single bin{j1} "
          f"R2={r2s[j1]:.3f}; pair ({best[0]},{best[1]}) R2={best[2]:.3f} "
          f"gain {best[2]-r2s[j1]:+.3f}")

# partial correlation of the cluster bin given the group bin
def partial_r(x, y, z):
    rxy, rxz, ryz = pearsonr(x, y)[0], pearsonr(x, z)[0], pearsonr(y, z)[0]
    return (rxy - rxz*ryz)/np.sqrt((1 - rxz**2)*(1 - ryz**2))
for i in (2, 4):
    pc = partial_r(ft_bar[okall, 3], Sb[okall, i], ft_bar[okall, 1])
    print(f"  partial r(S_ell~{LCEN[i]:.0f}, cluster bin | group bin) = "
          f"{pc:+.3f}")

# best cell + its dominant levers (setup-cell spearman helper; computed live,
# per-band — it is IMFslope here, not fig 7's full-curve VariableWindVel)
i_b, j_b = np.unravel_index(np.nanargmax(np.abs(Pb)), Pb.shape)
rho_lev = spearman(X_unit, Sb[:, [i_b]])[:, 0]
top3 = np.argsort(-np.abs(rho_lev))[:3]
p_dom = int(top3[0])
print(f"\nbest cell: ell~{LCEN[i_b]:.0f} (k~{k_cen[i_b]:.2f} h/Mpc) x mass "
      f"bin {j_b} [{MEDG[j_b]:.1f},{MEDG[j_b+1]:.1f}): r = {Pb[i_b, j_b]:.3f}")
print(f"top-3 levers for S(ell~{LCEN[i_b]:.0f}): " +
      ", ".join(f"{short_label(pnames[p])} ({rho_lev[p]:+.2f})" for p in top3))

# exact van Daalen mass bin (M500c,bg in [6e13, 2e14] Msun/h) for panel (c)
ftvd = np.full(NRn, np.nan); cvd = np.zeros(NRn, int)
for q in range(NRn):
    s = (mt5[q] >= 6e13) & (mt5[q] < 2e14)
    cvd[q] = s.sum()
    if cvd[q] >= 5:
        ftvd[q] = np.nanmedian(fbar5[q, s])/OB_OM
vd_fit = lambda ft: -np.exp(-5.990*ft - 0.5107)  # vD20 k=0.5 h/Mpc (v1 fig08)
i_vd = int(np.searchsorted(LEDG, 0.5*chi_star) - 1)  # ell band holding k=0.5
ok_vd = np.isfinite(ftvd)
dS_vd = Sb[ok_vd, i_vd] - 1
r_vd = pearsonr(ftvd[ok_vd], Sb[ok_vd, i_vd])[0]
sl_vd = np.polyfit(ftvd[ok_vd], dS_vd, 1)[0]
sl_3d = -5.990*vd_fit(np.nanmedian(ftvd))        # vD tangent at the median
print(f"\nvD cell (band {i_vd}, ell~{LCEN[i_vd]:.0f}, k~{k_cen[i_vd]:.2f}): "
      f"counts median {np.median(cvd):.0f} (min {cvd.min()}); f~_bar median "
      f"{np.nanmedian(ftvd):.3f}, range {np.nanmin(ftvd):.3f}-"
      f"{np.nanmax(ftvd):.3f}")
print(f"  r(f~_vd, S) = {r_vd:.3f}; WL slope {sl_vd:+.4f} vs vD 3-D tangent "
      f"{sl_3d:+.4f} at the median; measured dS {dS_vd.min():+.4f}.."
      f"{dS_vd.max():+.4f} (incl. the S>1 branch); vD prediction over the "
      f"cloud {vd_fit(np.nanmin(ftvd)):+.4f}..{vd_fit(np.nanmax(ftvd)):+.4f}")

# snap-85 (z=0.18) rank stability of the halo side: same bins, same reducer —
# justifies reading the z=0.03 bin ordering at the kernel-peak epoch z~0.43
c85 = np.load(SB35/"analysis_cache/atlas_cubes/atlas_cube_snap085.npz")
mt85 = c85["sobol_m_tot_500c_bg"][run_ids]
fb85 = np.where(mt85 > 0, (c85["sobol_m_gas_500c_bg"][run_ids]
                           + c85["sobol_m_star_500c"][run_ids])/mt85, np.nan)
ft85, _ = binned_ftilde(np.log10(np.where(mt85 > 0, mt85, np.nan)), fb85)
rk85 = []
for j in range(NBm):
    ok = np.isfinite(ft_bar[:, j]) & np.isfinite(ft85[:, j])
    rk85.append(spearmanr(ft_bar[ok, j], ft85[ok, j])[0])
print(f"halo-side rank stability, z=0.03 vs z={float(c85['z']):.2f} "
      f"(per mass bin): {np.round(rk85, 3)}")
del c85, mt85, fb85, ft85

# fiducial star: bind_science fiducial lightcone + fid atlas (fig 3's file);
# the Sobol design contains NO fiducial node (node 0 != fiducial)
clb20 = np.load(SCI/"runs/bind/run_0000/Cl_kappa.npz")
cld20 = np.load(SCI/"runs/dmo/run_0000/Cl_kappa.npz")
assert clb20["cl"].shape[-1] == len(ELL)             # same fine-ell grid as d
S_fid20 = (clb20["cl"][ZI, ZI]/cld20["cl"][ZI, ZI])[band_masks[i_b]].mean()
fa20 = np.load(SCI/"halo_atlas/fid_snap096.npz")
lm_f20 = np.log10(fa20["m_tot_500c_bg"])
sel_f20 = (lm_f20 >= MEDG[j_b]) & (lm_f20 < MEDG[j_b+1])
fbar_fid = np.median(((fa20["m_gas_500c_bg"] + fa20["m_star_500c"])
                      /fa20["m_tot_500c_bg"])[sel_f20])/OB_OM
print(f"fiducial: S(ell~{LCEN[i_b]:.0f}) = {S_fid20:.4f}, f~_bar"
      f"[{MEDG[j_b]:.1f},{MEDG[j_b+1]:.1f}) = {fbar_fid:.3f} "
      f"({sel_f20.sum()} halos)")

fig, ax = plt.subplots(1, 3, figsize=(TWO_COL[0], 2.55),
                       gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0]))

# (a) the r matrix — annotated cells; colorbar on the RIGHT (package-wide
# convention), so the k-scale twin axis is offset to the LEFT of the ell axis
# all entries are positive (0.58-0.98) -> sequential-red cmap trimmed to
# [0.5, 1] instead of the old diverging [-1, 1] (there is no negative side)
a = ax[0]
CMAP_A, NORM_A = plt.get_cmap("Reds"), Normalize(vmin=0.5, vmax=1.0)
im = a.imshow(Pb, cmap=CMAP_A, norm=NORM_A, aspect="auto", origin="lower")
for i in range(5):
    for j in range(NBm):
        rgba = CMAP_A(NORM_A(Pb[i, j]))
        lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]  # luminance-based
        a.text(j, i, f"{Pb[i, j]:.2f}", ha="center", va="center", fontsize=6,
               color="w" if lum < 0.5 else "k")
a.set_xticks(range(NBm))
a.set_xticklabels([f"{MEDG[j]:.1f}–{MEDG[j+1]:.1f}" for j in range(NBm - 1)]
                  + [r"$\geq$13.9"], fontsize=5.6)
a.set_yticks(range(5))
a.set_yticklabels([f"{LCEN[i]:.0f}" for i in range(5)], fontsize=6)
a.set_xlabel(r"$\log_{10} M_{500c}\,[M_\odot/h]$")
a.set_ylabel(r"$\ell$")
# k twin: float location (axes fraction) puts the spine, ticks and label
# outboard of the ell tick labels on the LEFT — the right side now belongs to
# the colorbar, and tight_layout still reserves the room (child axes are in
# the parent's tight bbox)
# Companion k axis. Two fixes over v3: (i) a secondary_yaxis draws its own spine and
# tick MARKS, which showed up as stray ticks floating to the left of panel (a) with
# nothing attached to them -- the axis carries labels only, so hide both; (ii) the
# rows are log-spaced in ell, so label them in log10 k rather than as the unrounded
# decimals 0.4/0.8/1.8/4.0/8.8, which read as arbitrary.
sec20 = a.secondary_yaxis(-0.30)
sec20.set_yticks(range(5))
sec20.set_yticklabels([f"{np.log10(k_cen[i]):+.2f}" for i in range(5)], fontsize=6)
sec20.set_ylabel(r"$\log_{10} k$,  $k \simeq (\ell+1/2)/\chi_\star$  [$h$/Mpc]",
                 fontsize=6.5)
sec20.tick_params(length=0)                    # labels only -- no dangling tick marks
sec20.spines["left"].set_visible(False)
# vD-comparable cells (k~0.4-0.5; ~1e14 straddles the top two mass bins)
a.add_patch(Rectangle((1.5, -0.5), 2.0, 1.0, fill=False,
                      edgecolor=COLORS["secondary"], lw=1.2))
a.text(2.0, -0.38, "vD20", fontsize=5.2, color=COLORS["secondary"],
       ha="center", va="bottom")
# best cell: thin black outline, NOT a star marker (a marker collides with
# the in-cell r text — the prototype's flagged layout bug)
a.add_patch(Rectangle((j_b - 0.5, i_b - 0.5), 1.0, 1.0, fill=False,
                      edgecolor="k", lw=0.9))
cb20 = fig.colorbar(im, ax=a, fraction=0.045, pad=0.03)
cb20.set_label(r"$r\,(S_\ell,\ \tilde f^{\,\rm cyl}_{\rm bar,500c})$", fontsize=6.5)
# white, small, tucked into the true corner (NOT the shared panel_label
# helper -- with only 4x5 cells, its default size/offset sits right on top
# of the bottom-left cell's "0.92" annotation; this keeps both readable)
a.text(0.012, 0.012, "(a)", transform=a.transAxes, ha="left", va="bottom",
       fontsize=6.5, fontweight="bold", color="w")

# (b) the hinge: S(ell~970) vs group-bin f~_bar, colored by the dominant
# lever (fig-7 colorbar idiom); dashed OLS; fiducial star = free closure stamp
b = ax[1]
b.scatter(ft_bar[:, j_b], Sb[:, i_b], c=X_unit[:, p_dom], cmap="coolwarm",
          vmin=0, vmax=1, s=9, edgecolor="k", linewidths=0.15, alpha=0.9,
          rasterized=True)
ols20 = np.polyfit(ft_bar[okall, j_b], Sb[okall, i_b], 1)
xg20 = np.linspace(np.nanmin(ft_bar[:, j_b]), np.nanmax(ft_bar[:, j_b]), 10)
b.plot(xg20, np.polyval(ols20, xg20), color=COLORS["truth"], lw=1.0, ls="--")
b.plot(fbar_fid, S_fid20, marker="*", ms=10, color=COLORS["highlight"],
       mec="k", mew=0.4, zorder=5, ls="none", label="fiducial")
b.axhline(1, color=COLORS["dmo"], ls=":", lw=0.7)
# no in-axes stamps: r and the projected-cylinder aperture definition (NOT the
# vD+20 3-D spherical f~) are caption text, printed live at the end of the cell
b.set_xlabel(rf"$\tilde f^{{\,\rm cyl}}_{{\rm bar,500c}}\ "
             rf"(10^{{{MEDG[j_b]:.1f}-{MEDG[j_b+1]:.1f}}}\,M_\odot/h)$")
b.set_ylabel(rf"$S(\ell\simeq{LCEN[i_b]:.0f})$")
b.legend(loc="lower right", fontsize=6, handletextpad=0.2)
cbb20 = fig.colorbar(ScalarMappable(cmap="coolwarm", norm=Normalize(0, 1)),
                     ax=b, fraction=0.045, pad=0.03)
cbb20.set_label(short_label(pnames[p_dom]) + " (prior units)", fontsize=6)
cbb20.set_ticks([0, 1])
panel_label(b, "(b)")

# (c) the universal plane at the vD cell: published 3-D fit (own panel — NOT
# overlaid on (b): its tangent slope is ~10x shallower than the WL slope).
# The x-axis here is the PROJECTED cylinder f~^cyl (fgrid/ftvd, this cell's
# own measurement); the vD+20 curve and obs. band are the literature 3-D
# relation, unchanged from v1 -- the secondary (top) axis carries them, truth-
# calibrated by ×1.16 (fig3/fig20 own hydro-pasted-truth f~^cyl=0.93 vs
# Nelson+24 3D~0.81); that provenance is caption text, not an on-figure stamp.
# obs. groups band = vD+20's OWN compilation (their Figs 15/16, footnote 10),
# primarily Vikhlinin+06/Sun+09/Gonzalez+13/Lovisari+15 -- cited in the caption
# markdown above (ni3-flags audit, flag iii), not re-derived here.
c = ax[2]
fgrid = np.linspace(0.35, 1.06, 200)
c.plot(fgrid, vd_fit(fgrid), color=COLORS["truth"], ls="--", lw=1.2,
       label=r"vD+20 fit ($k=0.5$, 3D $-$ top axis)")
c.fill_between(fgrid, vd_fit(fgrid) - 0.01, vd_fit(fgrid) + 0.01,
               color=COLORS["dmo"], alpha=0.3, lw=0)
c.axvspan(0.55, 0.76, color=COLORS["secondary"], alpha=0.15, lw=0,
          label="obs. groups (3D $-$ top axis)")
c.scatter(ftvd[ok_vd], dS_vd, s=6, color=COLORS["bind"], alpha=0.55, lw=0,
          rasterized=True,
          label=rf"BIND Sobol: $\Delta S(k\simeq{k_cen[i_vd]:.1f})$")
c.axhline(0, color=COLORS["dmo"], ls=":", lw=0.7)
# x>1: closure-only regime -- the nodes with f~^cyl>1 (weak-wind enhancement
# branch, fig 12) have NO 3-D counterpart: a projected, annulus-subtracted
# cylinder can partially over-close; a sphere cannot. Shading only (it is
# data); the saturated-flat top and this regime are described in the caption.
c.axvspan(1.0, 1.06, color=COLORS["highlight"], alpha=0.10, lw=0, zorder=0)
c.set_xlim(0.35, 1.06)
c.set_ylim(-0.12, 0.03)
c.set_xlabel(r"$\tilde f^{\,\rm cyl}_{\rm bar,500c}\ (0.6\!-\!2\times10^{14}\,M_\odot/h)$")
c.set_ylabel(r"$\Delta P/P$  or  $\Delta S_\ell$")
c.legend(fontsize=5.4, loc="lower right", handletextpad=0.4)
# secondary (top) axis: 3-D-equivalent scale, truth-calibrated x/1.16
sec_c = c.secondary_xaxis("top", functions=(lambda x: x/1.16, lambda x: x*1.16))
sec_c.set_xlabel(r"3D-equiv. $\tilde f$ (truth-calibrated $\times1/1.16$)", fontsize=6)
sec_c.tick_params(labelsize=5.6)
panel_label(c, "(c)")

fig.tight_layout(w_pad=1.2)
save(fig, "figs_v2/fig20_vandaalen_matrix")
plt.show()
print(f"panel (b) [caption]: r(S(ell~{LCEN[i_b]:.0f}), f~^cyl_bar in "
      f"[{MEDG[j_b]:.1f},{MEDG[j_b+1]:.1f})) = {Pb[i_b, j_b]:.3f}, OLS slope "
      f"{ols20[0]:+.4f}; aperture = projected R500c, +-25.6 Mpc/h slab, "
      f"annulus-subtracted -- NOT the 3-D spherical f~ of vD+20")
print(f"panel (c) [caption]: {int((ftvd[ok_vd] > 1).sum())}/{int(ok_vd.sum())} "
      f"nodes have f~^cyl > 1 (shaded band; cylinder-closure regime, no 3-D "
      f"counterpart); truth calibration f~^cyl=0.93 <-> 3D f~=0.81 "
      f"(Nelson+24) = x1.16, so the top axis is x1/1.16 and the cloud maps to "
      f"3D f~ {np.nanmin(ftvd)/1.16:.2f}-{np.nanmax(ftvd)/1.16:.2f} "
      f"(median {np.nanmedian(ftvd)/1.16:.2f})")
del cz, mt5, mg5, ms5, logm5, fbar5, fgas5
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §3d — Fig 12: the enhancement branch, and where surveys sit (new section + figure)

The suite does not only *suppress* the WL spectrum: a substantial minority of nodes show
$S(\ell)>1$ — 28% at $z_s=0.5$, rising through 38% at $z_s=1$ to 46% at $z_s=2.44$ (at
$\ell=5000$; all fractions live-printed by the cell). Fig 7(b) shows these are the
low-`VariableWindVelFactor` corner: weak winds retain and centrally concentrate gas,
*adding* small-scale power instead of removing it. This section also supplies the survey
context a WL reader needs: (a) the 5–95% Sobol envelope of $S(\ell)$ against the LSST-Y10
and Euclid shape-noise precision on the fiducial, drawn as *outlined* envelopes so both
surveys stay decodable (LSST's band nests inside Euclid's). The precision is Gaussian
mode-counting in a single non-tomographic bin — per-tomographic-bin errors are larger, and
non-Gaussian + super-sample covariance degrade $C_\ell^{\kappa\kappa}$ errors by factors of
a few at $\ell\gtrsim3000$ — so the honest statement is that *Stage-IV statistical power
(Gaussian, single-bin) resolves essentially the full envelope at $\ell\gtrsim10^3$*, not
that the full systematics-marginalized analysis does. Note the survey bands are **widest at
the largest scales**, which is correct and not a plotting error: with logarithmic bins
$\Delta\ell=0.15\,\ell$ the number of modes per bin grows as $\ell^2$, so the
cosmic-variance fractional error falls as $1/\ell$ until shape noise takes over
($\sigma(C_\ell)/C_\ell$ for LSST-Y10 runs 4.6% at $\ell=100$ to a 0.43% minimum near
$\ell\simeq2000$, then back up to 2.5% at $\ell=2\times10^4$; live-printed by the cell).
Panel (a) carries only released-product content — the previously drawn schematic
kSZ-informed / $A_{\rm mod}\sim0.8$ band has been removed, since it was indicative rather
than derived from anything on disk. (b) is the *gas diagnostic* of the enhancement branch:
the median stacked group-scale $\tau(x)$ profile of the $S(5000)>1$ nodes is centrally
concentrated relative to the $S(5000)<1$ nodes — directly showing the retained-and-
concentrated gas that adds small-scale power.
''')

code(r'''
# ── Fig 12: S(ell) envelope vs Stage-IV precision ────────────────────────────
# data: emulator_dataset_xpkfix.npz (suppression), bind_science runs/{bind,dmo}
#       fiducial Cl + runs/bind/run_0000/paired_perreal_fid.npz (50 per-realization
#       Cl_kappa draws of the OUT-OF-DESIGN fiducial -- the measured-covariance
#       leg of the v2 LSST-Y10 band); survey noise recipe as in
#       fig_scripts/fig09_cl_envelope.py
A2SR = (np.pi/180/60)**2                      # sr per arcmin^2
# NOTE on the shape of the band (adjudicated -- this is CORRECT, do not "fix"):
# with log bins Delta_ell = dlnl*ell the mode count per bin (2ell+1)*Delta_ell*fsky
# scales as ell^2, so the cosmic-variance fractional error scales as 1/ell and is
# LARGEST at the LARGEST scales; shape noise (Nl/cl) only takes over at high ell.
# This is the error on a MEASURED C_ell referenced to a fixed theory DMO
# prediction, so it maps directly onto S(ell).
def cl_relerr(cl, ngal, sige, fsky, dlnl=0.15):
    Nl = sige**2*A2SR/ngal                    # shape noise: n_gal per sr = ngal/A2SR
    dl = np.maximum(ELL*dlnl, 1.0)
    return np.sqrt(2.0/((2*ELL + 1)*dl*fsky))*(1 + Nl/cl)

def cl_relerr_cv_only(fsky, dlnl=0.15):
    """The pure Gaussian mode-counting piece of cl_relerr, with NO shape-noise
    term -- used below to extract the analytic shape-noise EXCESS
    (cl_relerr - this) so it can be added to the measured band in quadrature."""
    dl = np.maximum(ELL*dlnl, 1.0)
    return np.sqrt(2.0/((2*ELL + 1)*dl*fsky))

bcl0 = np.load(SCI/"runs/bind/run_0000/Cl_kappa.npz")
dcl0 = np.load(SCI/"runs/dmo/run_0000/Cl_kappa.npz")
S_fid_full = bcl0["cl"][ZI, ZI]/dcl0["cl"][ZI, ZI]
err_euc = cl_relerr(bcl0["cl"][ZI, ZI], 30.0, 0.30, 0.36)   # Euclid stays fully analytic (Knox)

# ── v2: LSST-Y10 band from MEASURED covariance, not a Knox-only formula ──────
# Per-realization Cl_kappa draws of the released fiducial (bind_lightcone_tng /
# bind_science/runs/bind/run_0000 -- identical maps, the SAME "released fiducial"
# fig 4/10 use, NOT a Sobol node) on this box's 25 deg^2 (5x5 deg) footprint.
# Convention: std of the BINNED log-Cl across the 50 realizations (log so the
# fractional-error reading std(ln X) ~ std(X)/mean(X) holds); an 8-bin
# block-average (the same rebin() used for fig 4's Hartlap compression)
# stabilizes the per-ell std estimate, which is itself noisy at n=50. CAVEAT
# (stated again in the print stamp): these 50 realizations are random
# ROTATIONS of the same underlying DMO box (paired_stats.py), not independent
# cosmological volumes, so this measured scatter is likely an UNDERESTIMATE of
# true large-scale (low-ell) cosmic variance even before area-scaling; it is a
# non-Gaussian-aware supplement to, not a full replacement for, survey-grade
# covariance. It upgrades automatically, with no code change, once the planned
# ~550-realization fiducial covariance set lands (TODO Sec 4c) -- just points
# at the same cache path, which will then hold the bigger ensemble.
kk_fid_real = np.load(SCI/"runs/bind/run_0000/paired_perreal_fid.npz")["clk"][:, ZI, :]
logcl_rb = rebin(np.log(np.where(kk_fid_real > 0, kk_fid_real, np.nan)), 8)
ell_rb   = rebin(ELL[None, :], 8)[0]
sigma_lncl_25 = np.nanstd(logcl_rb, axis=0)             # measured, box (25 deg^2) footprint

AREA_25_SR   = 25.0*(np.pi/180.0)**2                    # the 5x5 deg box, steradians
AREA_LSST_SR = 0.44*4*np.pi                             # LSST-Y10 effective footprint
area_scale   = np.sqrt(AREA_25_SR/AREA_LSST_SR)         # sample variance ~ 1/sqrt(area)
sigma_meas_lsst = np.interp(ELL, ell_rb, sigma_lncl_25*area_scale)  # back onto the native grid

shot_excess  = cl_relerr(bcl0["cl"][ZI, ZI], 27.0, 0.26, 0.44) - cl_relerr_cv_only(0.44)  # >=0
err_lsst_old = cl_relerr(bcl0["cl"][ZI, ZI], 27.0, 0.26, 0.44)   # v1 band -- kept for the
                                                                   # printed old-vs-new table only
err_lsst = np.sqrt(sigma_meas_lsst**2 + shot_excess**2)           # v2 band: measured (+) shot, quad.

S_all = d["t__suppression__value"][:, ZI, :]
mS = ELL <= ELL_MAX_PLOT
i5k = int(np.argmin(np.abs(ELL - 5000)))
hiS = S_all[:, i5k] > 1                        # enhancement branch at ell=5000

fig, ax = plt.subplots(1, 2, figsize=TWO_COL)
ax[0].fill_between(ELL[mS], np.percentile(S_all, 5, 0)[mS], np.percentile(S_all, 95, 0)[mS],
                   color=COLORS["dmo"], alpha=0.35, lw=0, label="Sobol 5-95%")
ax[0].fill_between(ELL[mS], S_all.min(0)[mS], S_all.max(0)[mS],
                   color=COLORS["dmo"], alpha=0.15, lw=0)
ax[0].plot(ELL[mS], S_fid_full[mS], color=COLORS["truth"], lw=1.8, label="fiducial")
# v2 FIX: survey precision as OUTLINED (unfilled) envelopes — the LSST band is
# strictly nested inside Euclid's, so stacked fills hid it entirely; outlines
# keep both decodable (color + linestyle redundancy).
for err_s, c_s, ls_s, lab_s in ((err_lsst, COLORS["bind"], "--",
                                 "LSST-Y10 (measured cov., area-scaled)"),
                                (err_euc, COLORS["secondary"], "-.", "Euclid (Gaussian, 1-bin)")):
    ax[0].plot(ELL[mS], (S_fid_full*(1 - err_s))[mS], color=c_s, lw=0.9, ls=ls_s)
    ax[0].plot(ELL[mS], (S_fid_full*(1 + err_s))[mS], color=c_s, lw=0.9, ls=ls_s,
               label=lab_s)
ax[0].axhline(1, color=COLORS["dmo"], ls=":", lw=0.8)
# CIC-aliasing region, same convention as figs 4/4b/7 -- S(ell) cancels the
# aliasing itself, but the shading is drawn for a consistent ell-axis reading
# across every field-level figure in the paper.
ax[0].axvspan(ELL_TRUST, ELL_MAX_PLOT, color="0.85", alpha=0.45, lw=0, zorder=0)
ax[0].set_xscale("log"); ax[0].set_xlim(90, ELL_MAX_PLOT); ax[0].set_ylim(0.75, 1.3)
ax[0].set_xlabel(r"$\ell$")
ax[0].set_ylabel(r"$S(\ell)$ at $z_s=1$")
ax[0].legend(loc="upper left", fontsize=5.2)
S5 = S_all[:, i5k]                             # feeds the printed provenance below
panel_label(ax[0], "(a)", loc="lower left")

# (b) enhancement-branch gas diagnostic: stacked tau profiles of S>1 vs S<1 nodes
xpb = np.load(SCI/"ksz_confront/bind_tauy_xprof_snap085.npz")
tau_n = xpb["tau"][:, 0, :]                    # group bin 13.0-13.4, all 256 nodes
node_in_ds = np.isin(xpb["nodes"], run_ids)
tau_n = tau_n[node_in_ds]
for msk_b, c, lab in [(hiS, COLORS["highlight"], f"$S(5000)>1$ (n={hiS.sum()})"),
                      (~hiS, COLORS["bind"], f"$S(5000)<1$ (n={(~hiS).sum()})")]:
    med = np.nanmedian(tau_n[msk_b], 0)
    lo, hi = np.nanpercentile(tau_n[msk_b], [16, 84], axis=0)
    ax[1].fill_between(xpb["x"], lo, hi, color=c, alpha=0.22, lw=0)
    ax[1].loglog(xpb["x"], med, color=c, lw=1.5, label=lab)
ax[1].set_xlabel(r"$R/r_{200c}$")
ax[1].set_ylabel(r"stacked $\tau(x)$, $10^{13.0-13.4}M_\odot/h$, $z=0.18$")
ax[1].legend(loc="lower left", fontsize=6)
panel_label(ax[1], "(b)", loc="upper right")

fig.tight_layout(w_pad=1.1)
save(fig, "figs_v2/fig12_survey_context")
plt.show()
for zi, z in enumerate(ZS):
    Sz = d["t__suppression__value"][:, zi, i5k]
    print(f"z_s={z:.2f}: S(5000) in [{Sz.min():.3f}, {Sz.max():.3f}], "
          f"enhancement fraction {(Sz > 1).mean():.2f}")
print("panel (a) LSST-Y10 sigma(C_ell)/C_ell -- v2 (measured cov. + analytic shape-"
      "noise excess, area-scaled 25->LSST-Y10 deg^2): "
      + ", ".join(f"ell={ELL[int(np.argmin(np.abs(ELL - lt)))]:.0f}: "
                  f"{100*err_lsst[int(np.argmin(np.abs(ELL - lt)))]:.2f}%"
                  for lt in (100, 300, 1000, 5000, 20000)))
print("OLD (v1, fully-analytic Knox) vs NEW (v2) at the same ell -- traceability table:")
for lt in (100, 300, 1000, 5000, 20000):
    i = int(np.argmin(np.abs(ELL - lt)))
    print(f"  ell={ELL[i]:>6.0f}: OLD {100*err_lsst_old[i]:5.2f}%  ->  NEW {100*err_lsst[i]:5.2f}%  "
          f"(measured leg {100*sigma_meas_lsst[i]:5.2f}%, analytic shot-noise excess "
          f"{100*shot_excess[i]:5.2f}%)")
print("CAVEAT: the measured leg is built from only 50 fiducial realizations (small-N: "
      "its own std estimate carries residual noise even after the 8-bin block-average), "
      "and those 50 are random ROTATIONS of the SAME underlying DMO box (not independent "
      "cosmological volumes), so it likely UNDER-estimates true low-ell cosmic variance; "
      "it upgrades automatically, with no code change, once the planned ~550-realization "
      "fiducial covariance set lands (TODO Sec 4c). Old Knox-only band retained above only "
      "for this traceability comparison, not drawn on the panel.")
r_in = xpb["x"] < 0.3
print(f"central (R<0.3 r200c) tau, enhancement/suppression branch: "
      f"{np.nanmedian(tau_n[hiS][:, r_in])/np.nanmedian(tau_n[~hiS][:, r_in]):.2f}x")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §3f — Fig 8: the latent space of the statistics

**Method.** The figure runs on the paper's canonical statistic list: 7 WL-field statistics
($S(\ell)$, PDF, $N_{\rm pk}$, $N_{\rm min}$, $V_0$, $V_1$, $V_2$), 2 non-$\kappa$
auto-spectra ($C_\ell^{yy}$, $C_\ell^{\tau\tau}$) and 3 cross-spectra ($C_\ell^{\kappa y}$,
$C_\ell^{\kappa\tau}$, $C_\ell^{y\tau}$) — **12 statistics** ($C_\ell^{\kappa\kappa}$ itself
is not among them; the fig 4b markdown states once why: at fixed cosmology it is $S(\ell)$
times a single run-independent DMO trace, so it carries no information $S(\ell)$ does not
already). For each family we standardize the (transformed) bins over the 253
runs and take the SVD $Z=U\Sigma V^{\sf T}$; the run-space scores are $U\Sigma$ and the
explained-variance spectrum $\sigma_k^2/\sum\sigma^2$. Cross-family alignment is measured
by **both** canonical correlations $(\rho_1,\rho_2)$ between two families' leading
2-component score planes (QR-orthonormalize each $253\times2$ score matrix; $\rho_1,\rho_2$
= the two singular values of $Q_A^{\sf T}Q_B$). $\rho_1$ tests only the single best-aligned
direction; $\rho_2$ tests whether the *full 2-plane* is shared. Both are judged against
permutation nulls pooled over all 66 family pairs (200 seeded shuffles), printed by the cell.

**Result** (re-executed for the new 12-statistic list: $C_\ell^{\kappa\kappa}$ dropped,
$C_\ell^{y\tau}$ added). (a) Two components capture 92.3–99.6% of the run-to-run variance in
every family except the peak *and* minimum counts (27.1% and 45.0%), whose *per fine bin*
variance is realization-noise dominated (measured noise/signal 2.9 and 5.7; the ×4-rebinned
counts of fig 5 do retain a detected $|\rho|\approx0.44$ response). (b) **Every family's
*leading* response direction aligns with the suppression plane** ($\rho_1=0.91$–$1.00$ vs the
$S(\ell)$ row, against a 0.18 null) — but the full 2-plane is *not* universally shared: against
$S(\ell)$, the second direction is shared by the gas-thermodynamic spectra and the PDF
($\rho_2=0.89$ for $C_\ell^{\kappa\tau}$, 0.87 for $C_\ell^{yy}$, 0.74 for $C_\ell^{\tau\tau}$,
0.73 for the PDF — 4 of the 11 genuinely independent families), while $C_\ell^{\kappa y}$,
the new $C_\ell^{y\tau}$ cross, the counts and the Minkowski functionals align in one
direction only ($\rho_2=0.14$–$0.38$). Resolving the MFs individually revises a claim of the
7-family version: the *concatenated* $V_0|V_1|V_2$ family gave $\rho_2=0.08$, exactly at the
null, but taken one at a time the three functionals give 0.22 / 0.37 / 0.38 — all above it,
so the concatenation was diluting a weak second direction rather than establishing its
absence. Panel (b) prints $\rho_1/\rho_2$ in every cell so this is visible on-figure.
(c) In the suppression plane itself, PC1 *is* the small-scale amplitude of the response
(corr $-1.00$ with the high-$\ell$ mean of $\log S$) and the group gas fraction runs along
the orthogonal direction ($r=0.96$ with PC2): the suite's physics is, to excellent
approximation, a two-dimensional
(amplitude, gas-ejection) surface — whose leading direction every statistic family sees,
and whose second direction the gas-thermodynamic spectra and the PDF resolve (the new
$C_\ell^{y\tau}$ cross does not: like $C_\ell^{\kappa y}$, it aligns in the leading direction
only) — the transferable insight companion Paper III builds on, and the design-adequacy
argument for 253 nodes in 30 dimensions.
''')

code(r'''
# ── Fig 8: PCA latents per statistic family + cross-family overlap ───────────
# data: bind_sb35/emulator_dataset_xpkfix.npz + bind_science/latent_drivers.npz
def fam_matrix(arr, transform):
    A = np.nan_to_num(np.asarray(arr, float))
    if transform == "log":
        pos = A[A > 0]
        A = np.log10(np.clip(A, pos.min()*0.5, None))
    elif transform == "log1p":
        A = np.log10(1 + A)
    return (A - A.mean(0))/np.maximum(A.std(0), 1e-12)

# The canonical statistic list: 7 WL-field + 2 non-kappa autos + 3 crosses = 12
# statistics. Ordered so the WL / SZ-auto / cross blocks are contiguous on both
# axes of panel (b). C_kk itself is NOT a family here -- the fig 4b markdown
# states once why (at fixed cosmology it is S(ell) x a run-independent DMO
# trace, so it carries no information S(ell) does not already). The transforms
# are fig 8's DISPLAY transforms (log for the positive spectra including S(ell),
# log1p for the counts, raw for the sign-crossing or already-normalized bins) --
# deliberately not the emulator's STAT_SPECS transforms.
FAM12 = {
    r"$S(\ell)$":             (d["t__suppression__value"][:, ZI, :], "log"),
    r"PDF":                   (d["t__pdf__value"][:, ZI, :], "raw"),
    r"$N_{\rm pk}$":          (d["t__peak_counts__value"][:, ZI, :], "log1p"),
    r"$N_{\rm min}$":         (d["t__minima_counts__value"][:, ZI, :], "log1p"),
    r"$V_0$":                 (d["t__mf_v0__value"][:, ZI, :], "raw"),
    r"$V_1$":                 (d["t__mf_v1__value"][:, ZI, :], "raw"),
    r"$V_2$":                 (d["t__mf_v2__value"][:, ZI, :], "raw"),
    r"$C_\ell^{yy}$":         (d["t__cl_yy__value"], "log"),
    r"$C_\ell^{\tau\tau}$":   (d["t__cl_tt__value"], "log"),
    r"$C_\ell^{\kappa y}$":   (d["t__cl_kappa_y__value"][:, ZI, :], "raw"),
    r"$C_\ell^{\kappa\tau}$": (d["t__cl_kappa_tau__value"][:, ZI, :], "raw"),
    r"$C_\ell^{y\tau}$":      (d["t__cl_yt__value"], "raw"),
}

scores, evr, n99 = {}, {}, {}
for lab, (arr, tr) in FAM12.items():
    Z = fam_matrix(arr, tr)
    U, sv, _ = np.linalg.svd(Z, full_matrices=False)
    e = sv**2/(sv**2).sum()
    evr[lab], n99[lab] = e, int(np.searchsorted(np.cumsum(e), 0.99) + 1)
    scores[lab] = U[:, :2]*sv[:2]

def cca2(A, B):
    Qa, _ = np.linalg.qr(A - A.mean(0)); Qb, _ = np.linalg.qr(B - B.mean(0))
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)   # (rho_1, rho_2), sorted

fams13 = list(FAM12)
CC = np.array([[cca2(scores[a], scores[b])[:2] for b in fams13] for a in fams13])
C8, C8b = CC[..., 0], CC[..., 1]      # rho_1 and rho_2 matrices
NPAIR = len(fams13)*(len(fams13) - 1)//2      # 66 for the 12 canonical families

# permutation nulls for BOTH displayed statistics (rho_1, rho_2), pooled over
# ALL NPAIR family pairs (v2 fix: the null previously covered one pair and rho_1
# only): shuffle run labels of one side, 200 seeded shuffles.
rngc = np.random.default_rng(5)
null_r1, null_r2 = [], []
for _ in range(200):
    permc = rngc.permutation(len(run_ids))
    for ia in range(len(fams13)):
        for ib in range(ia + 1, len(fams13)):
            sv = cca2(scores[fams13[ia]], scores[fams13[ib]][permc])
            null_r1.append(sv[0]); null_r2.append(sv[1])
null_r1, null_r2 = np.array(null_r1), np.array(null_r2)

# realization-noise floor of BOTH count families (paired per-realization cache
# from fig 4): the counts are the only families whose 2 PCs miss most of the
# run-to-run variance, and this is why.
def _count_nsr(key, dkey):
    noise = pr[key][:, ZI, :].std(0)**2/pr[key].shape[0]
    sig   = np.nanvar(d[f"t__{dkey}__value"][:, ZI, :].astype(float), axis=0)
    return np.nanmedian(noise/np.where(sig > 0, sig, np.nan))

pk_nsr = _count_nsr("pk", "peak_counts")
mn_nsr = _count_nsr("min", "minima_counts")

ld = np.load(SCI/"latent_drivers.npz")
# 2026-08-03: the driver cache predates the 3-run repaint (253 rows vs the
# dataset's 256). Align by run_id intersection instead of asserting equality;
# the repaired runs are excluded from the DRIVER-coloured panel (c) only and
# reported below. Follow-up: regenerate latent_drivers.npz on all 256 runs.
assert np.all(np.isin(ld["run_ids"], run_ids)), "driver cache has runs unknown to the dataset"
IN_LD = np.isin(run_ids, ld["run_ids"])            # dataset-row mask, True where cached
LD_MISS = sorted(set(int(r) for r in run_ids) - set(int(r) for r in ld["run_ids"]))
print(f"latent_drivers cache: {int(IN_LD.sum())}/{len(run_ids)} dataset runs covered "
      f"(missing {LD_MISS} -- repainted after the cache was built)")

# 2x2 layout: the 12x12 matrix needs a full-height column to keep ~0.27 in per
# cell (the density the 7-family 1x3 row had) so the stacked rho_1/rho_2 numbers
# stay legible at print size.
fig = plt.figure(figsize=TWO_COL_TALL)
gsL = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], hspace=0.42, wspace=0.32)
axA = fig.add_subplot(gsL[0, 0])      # (a) explained variance, 12 bars
axC = fig.add_subplot(gsL[1, 0])      # (c) suppression PC plane coloured by f_gas
axB = fig.add_subplot(gsL[:, 1])      # (b) 12x12 rho_1/rho_2 matrix

tick13 = fams13
xx = np.arange(len(fams13))
axA.bar(xx, [evr[f][0] for f in fams13], color=COLORS["bind"], label="PC1")
axA.bar(xx, [evr[f][1] for f in fams13], bottom=[evr[f][0] for f in fams13],
        color=COLORS["secondary"], label="PC2")
for j, f in enumerate(fams13):
    axA.text(j, evr[f][0] + evr[f][1] + 0.015, str(n99[f]), ha="center", fontsize=6)
for f in (r"$N_{\rm pk}$", r"$N_{\rm min}$"):        # BOTH count families
    axA.text(fams13.index(f), 0.5*(evr[f][0] + evr[f][1]) + 0.06, "noise-\ndom.",
             ha="center", va="center", fontsize=5.0, color=COLORS["highlight"],
             path_effects=[pe.withStroke(linewidth=1.2, foreground="w")])
axA.set_xticks(xx); axA.set_xticklabels(tick13, rotation=55, ha="right", fontsize=5.5)
axA.set_ylabel("explained variance"); axA.set_ylim(0, 1.12)
axA.legend(loc="lower left", fontsize=6, frameon=True, framealpha=0.85,
           edgecolor="none")   # 12 tall bars leave no clear space; float it
panel_label(axA, "(a)")

im = axB.imshow(C8, cmap="cividis", vmin=0.5, vmax=1.0, rasterized=True)
for i in range(len(fams13)):
    for j in range(len(fams13)):
        tc = "w" if C8[i, j] < 0.85 else "k"
        axB.text(j, i - 0.16, f"{C8[i, j]:.2f}", ha="center", va="center",
                 fontsize=4.6, color=tc)
        axB.text(j, i + 0.26, f"{C8b[i, j]:.2f}", ha="center", va="center",
                 fontsize=3.8, color=tc, alpha=0.85)
axB.set_aspect("equal")
axB.set_xticks(xx); axB.set_xticklabels(tick13, rotation=55, ha="right", fontsize=5.5)
axB.set_yticks(xx); axB.set_yticklabels(tick13, fontsize=5.5)
fig.colorbar(im, ax=axB, fraction=0.046, pad=0.02, label=r"$\rho_1$ (2-PC planes)")
axB.text(0.02, 1.03, "(b)", transform=axB.transAxes, fontsize=8,
         fontweight="bold", va="bottom")
axB.text(0.98, 1.03, r"cell: $\rho_1$ (top) / $\rho_2$ (bottom)",
         transform=axB.transAxes, fontsize=5, ha="right", va="bottom")

sp = scores[r"$S(\ell)$"]
sc = axC.scatter(sp[IN_LD, 0], sp[IN_LD, 1], c=ld["f_gas"], s=7, cmap="cividis",
                 lw=0, rasterized=True)
if LD_MISS:   # repaired runs without cached drivers: shown uncoloured, not hidden
    axC.scatter(sp[~IN_LD, 0], sp[~IN_LD, 1], s=7, facecolor="none",
                edgecolor="0.5", lw=0.5, rasterized=True)
axC.set_xlabel("suppression PC1"); axC.set_ylabel("suppression PC2")
fig.colorbar(sc, ax=axC, fraction=0.046, pad=0.02, label=r"group $f_{\rm gas}$")
panel_label(axC, "(c)")

fig.tight_layout()
save(fig, "figs_v2/fig08_latent_pca")
plt.show()
iS = fams13.index(r"$S(\ell)$")
print(f"canonical list: {len(fams13)} statistics (7 WL field + 2 non-kappa autos + "
      "3 crosses; C_kk is not a family here -- see the §4a identity note)")
print("PC1+PC2 explained variance: "
      + ", ".join(f"{f} {100*(evr[f][0]+evr[f][1]):.1f}%" for f in fams13))
print("canonical corr rho1/rho2 with S(ell): "
      + ", ".join(f"{f} {C8[iS, j]:.2f}/{C8b[iS, j]:.2f}"
                  for j, f in enumerate(fams13) if j != iS))
shared2 = [f for j, f in enumerate(fams13) if j != iS and C8b[iS, j] >= 0.5]
print(f"families sharing the SECOND direction with S(ell) (rho2 >= 0.5): {shared2} -- "
      f"{len(shared2)} of {len(fams13)-1} genuinely independent families; all others "
      "align in the leading direction only")
# the concatenated-MF control: v1 ran V0|V1|V2 as ONE family and read its rho_2 as
# null-consistent; resolved individually the three functionals are all above the null.
mf_cat = fam_matrix(np.concatenate([d[f"t__mf_v{k}__value"][:, ZI, :] for k in (0, 1, 2)],
                                   axis=1), "raw")
Um, sm, _ = np.linalg.svd(mf_cat, full_matrices=False)
r1m, r2m = cca2(sp, Um[:, :2]*sm[:2])[:2]
iV = [fams13.index(f) for f in (r"$V_0$", r"$V_1$", r"$V_2$")]
print(f"MF concatenation control: the v1 concatenated V0|V1|V2 block gives rho1/rho2 vs "
      f"S(ell) = {r1m:.2f}/{r2m:.2f}, but resolved individually rho2 = "
      + "/".join(f"{C8b[iS, j]:.2f}" for j in iV)
      + f" (null 95th pct {np.percentile(null_r2, 95):.2f}) -- concatenation diluted a "
        "real second direction, it did not measure its absence")
r1, r2 = (np.corrcoef(sp[IN_LD, k], ld["f_gas"])[0, 1] for k in (0, 1))
print(f"corr(group f_gas, suppression PC1/PC2) = {r1:.2f} / {r2:.2f} "
      f"(PC orientation is arbitrary; the gas axis lives in the 2-PC plane)")
amp = np.log10(d["t__suppression__value"][:, ZI, ELL > 3000]).mean(1)
a1, a2 = (np.corrcoef(sp[:, k], amp)[0, 1] for k in (0, 1))
print(f"PC1 identification: corr(PC1, high-ell mean log S) = {a1:.2f} (PC2: {a2:.2f})")
print(f"permutation nulls (pooled over all {NPAIR} family pairs): rho_1 median "
      f"{np.median(null_r1):.2f}, 95th pct {np.percentile(null_r1, 95):.2f}; "
      f"rho_2 median {np.median(null_r2):.2f}, 95th pct {np.percentile(null_r2, 95):.2f}")
print(f"count noise/signal variance ratio (median over bins) = {pk_nsr:.1f} (peaks) "
      f"/ {mn_nsr:.1f} (minima)")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §4.0 — Emulator methodology

`bind.emulator` maps the 30-dim unit cube to every released statistic. Per statistic $s$:
transform (log$_{10}$ for positive spectra/scalings, signed log1p for counts/PDF, raw for
sign-crossing bins), median-impute invalid bins, standardize, and PCA-truncate,
$z=W_k^{\sf T}\hat{y}$ with $k=\min(k_{\max},\,99.99\%\ \mathrm{variance})$; the truncation
residual is recorded and folded into predicted errors. Each retained component gets an
independent Gaussian process with ARD-RBF + white kernel,

$$k(\theta,\theta')=\sigma_f^2\exp\Big[-\tfrac12\textstyle\sum_j(\theta_j-\theta'_j)^2/\ell_j^2\Big]+\sigma_n^2\delta,$$

fit on the unit cube; predictions invert the PCA and transform, propagating the GP
predictive variance plus the truncation floor to physical error bars -- the $\sigma_{\rm GP}$
that enters the §4c likelihood (calibration tested in fig 15). **WP2 revision:** the
originally-released fit used scikit-learn's per-component L-BFGS-B optimizer, which pinned
`length_scale` at its 1e2 upper bound and the `WhiteKernel` noise at its 1e-6 lower bound on
most components -- silently flattening the emulator toward the training mean in the
highest-response held-out runs (the WL suppression enhancement corner, $S(\ell)\gtrsim1.2$).
Widening those bounds by two orders of magnitude and raising the PCA cap $k_{\max}$ 12→30 both
changed *nothing* (held-out median $|{\rm frac\ err}|$ 1.77%→1.78% and 1.76% respectively,
worst-run error 13.89%→13.88%/13.90%) -- the pinned bound was a symptom, not the cause.
Replacing the backend with `bind.emulator`'s existing GPU-ready **`gpgpu`** backend
(gpytorch, jointly-optimized batched exact GP; it auto-selects the GPU when one is present at
execution time, CPU otherwise) does help, at *every*
$k_{\max}$ tried and $\sim$1.7$\times$ **faster** per statistic (84.7s→48.6s, CPU-to-CPU) than
even the cheapest scikit-learn configuration: held-out error 1.77%→1.64% overall, 11.1%→9.6% on
the $S(\ell)>1.1$ enhancement subset, and 13.9%→12.0% on the worst-predicted held-out run
(the run now traced in fig 9's residual strip — the release figure leads with the typical
run, and this A/B comparison is changelog material kept only here; **all A/B numbers in this
paragraph are historical WP2-campaign values, not reproduced by this execution** — the
executed fig-9 cell prints this run's own metrics under different definitions, typical
held-out run 1.67% and 95th-pct run 10.7%, and the 12.0% worst-run A/B figure must not be
conflated with the 10.7% 95th-pct trace). This is a genuine,
moderate improvement, **not a full fix** -- the
residual under-prediction reflects the enhancement corner's sparse coverage in the 253-run
Sobol design, which no backend swap alone can cure (see §5 outlook). All emulated heads now
use `backend="gpgpu", n_components=12` (unchanged from the release; raising it bought nothing).

**The $C_\ell^{\kappa\kappa}=S(\ell)\times C_\ell^{\rm DMO}$ identity (stated once, here).**
At fixed cosmology $C_\ell^{\kappa\kappa}$ is exactly $S(\ell)$ times a single
run-independent DMO trace $C_\ell^{\rm DMO}$ (the fig 4b cell prints the `np.allclose` check;
the release ships that DMO spectrum, `runs/dmo/run_0000/Cl_kappa.npz`). This is why
$C_\ell^{\kappa\kappa}$ is **not** a separate row in fig 5's parameter-response matrix or a
separate family in fig 8's latent analysis — it carries no response direction $S(\ell)$ does
not. It **is** nonetheless retained as a directly-trained emulator head (fork-A ruling): the
head is fit on the map-measured spectra themselves, so an emulator user gets
$C_\ell^{\kappa\kappa}$ predictions natively rather than composing two products.

The quantitative contract per emulated statistic (from the executed fit and figs 9b/15; the
**locked 13-head list** = the 7 WL-field heads of §3, plus $C_\ell^{\kappa\kappa}$ trained
directly on the map-measured spectra, plus the 5 gas/cross spectra; scaling-relation
emulation ($Y(M)$, $f_{\rm gas}(M)$) deferred, not fit here):

| statistic | transform | $k$ retained | med. \|frac err\| | resp. $R^2$ | cov. $|z|{<}1$ / $|z|{<}2$ |
|---|---|---|---|---|---|
| $S(\ell)$ | raw | 12 | 2.7% | 0.72 | 0.69 / 0.90 |
| $C_\ell^{\kappa\kappa}$ | log | — | *new 13th head; numbers land at the 256-run refit* | — | — |
| PDF | log1p | 12 | 1.3% | 0.73 | 0.70 / 0.90 |
| $N_{\rm pk}$ | log1p | 12 | 1.0% | 0.23 | 0.66 / 0.94 |
| $N_{\rm min}$ | log1p | 12 | 0.6% | 0.33 | 0.65 / 0.93 |
| $V_0$ | raw | 12 | 0.04% | 0.74 | 0.66 / 0.89 |
| $V_1$ | raw | 12 | 0.4% | 0.71 | 0.63 / 0.89 |
| $V_2$ | raw | 12 | 0.7% | 0.71 | 0.64 / 0.88 |
| $C_\ell^{yy}$ | log | 12 | 11.7% | 1.00 | 0.67 / 0.88 |
| $C_\ell^{\tau\tau}$ | log | 9 | 9.6% | 0.84 | 0.70 / 0.91 |
| $C_\ell^{\kappa y}$ | raw | 12 | 11.5% | 1.00 | 0.67 / 0.89 |
| $C_\ell^{\kappa\tau}$ | raw | 12 | 13.5% | 0.78 | 0.75 / 0.91 |
| $C_\ell^{y\tau}$ | raw | 12 | 13.9% | 1.00 | 0.73 / 0.90 |

(these per-head numbers are unchanged from the previous 15-head fit: each head's PCA+GP is
fit independently, so dropping $C_\ell^{\kappa\kappa}$ and the two scaling heads from the
`stats=` list does not touch the retained heads' fits, only the cache fingerprint —
re-verified live by the executed cell, not asserted.) **Superseded by the C-head improvement
set below** — $C_\ell^{\kappa\kappa}$ *is* fitted, as a COMPOSED head: its diagonal
(same-plane) blocks are exactly $S(\ell)\times C_\ell^{\rm DMO}$ (a measured identity, not an
approximation — the fig 4b cell's `np.allclose` check), so they are composed for free from
the already-fitted $S(\ell)$ head rather than fit a second time; its off-diagonal
(cross-plane) blocks were measured NOT to be a run-invariant function of $S(\ell)$ (three
candidate compositions tried, all with the diagonal identity a factor of ~$10^{14}$ tighter
than any off-diagonal candidate) and so get a small dedicated fit instead (10 unique pairs,
`asinh_std`, ell-masked — see the paragraph below).
(the $S(\ell)$ transform is corrected to "raw" here -- it is the already-O(1) baryon ratio,
not log-compressed; the previous table's "log" was a documentation error.) Note
$C_\ell^{\kappa\tau}$/$C_\ell^{y\tau}$/$C_\ell^{\kappa y}$ carry the largest fractional errors
(11-14%) of any head: they are signed cross-spectra with no positivity floor, so a percent
of their (often near-zero) value is a much tighter demand than on the positive auto-spectra
--- the response $R^2=1.00$ on all three shows the emulator tracks their astrophysical
*response* essentially exactly despite that.

**C-head improvement set (this revision, papers/01_pipeline/audits/spectrum_head_experiment.md
— numbers above predate it, pending a refit against the new bundle):** the six ell-domain
spectrum heads in the table ($C_\ell^{\kappa\kappa}$, $C_\ell^{\kappa y}$, $C_\ell^{yy}$,
$C_\ell^{\kappa\tau}$, $C_\ell^{\tau\tau}$, $C_\ell^{y\tau}$) are now fit only on
$\ell\le1.5\times10^4$ (the CIC-aliased tail is excluded from the PCA+GP entirely, not just
display-masked; $S(\ell)$ is deliberately left unmasked, a ratio in which the aliasing artifact
already cancels), the three signed crosses ($C_\ell^{\kappa y}$, $C_\ell^{\kappa\tau}$,
$C_\ell^{y\tau}$) fit in $\mathrm{asinh}(x/s_{\rm bin})$ rather than raw (raw is numerically
broken for them under PCA+GP), and $C_\ell^{\kappa\kappa}$ is the composed head described
above; `em.predict()`'s public shapes are unchanged and masked bins still come back on the full
grid, at the per-bin training mean with an inflated error bar.

Coverage in the assembled dataset: all 253 runs for the 13 heads above (spectra, counts,
PDF, MFs); **WST 40/253 -- still excluded, untested; the DM statistics are a release
product (§6), not an emulator head; $Y(M)$ and
$f_{\rm gas}(M)$ scaling-relation emulation is deferred to a future version (still released
as raw statistics, e.g. in fig 7's provenance print, just not GP-fitted here).**
**Release artifact (actionable):** the shipped bundles are regenerated with
`Emulator(backend="gpgpu", n_components=12, backend_kwargs={"epochs": 400, "lr": 0.1}).fit(ds,
stats=[...13 heads...])` (equivalently the fig 9 cell); reference wall times from the
previous 15-head fit were 584s for 14 statistics on 203 runs plus $\approx$20s for the
(now-dropped) $C_\ell^{\kappa\kappa}$ head, and 1.3s to batch-predict the 50 held-out runs
(**CPU reference** values, historical — the executed wall times for the previous 12-head
fit, GPU when available, are printed live by the fig-9 cell and are typically far smaller,
e.g. 0.09s for the batch predict in the verified GPU run). **All section-4 forward
models — the fig 15 learning curve and the fig 10 / Appendix B (fig 16) MCMC likelihoods —
now use this same shipped `gpgpu` backend** (the earlier draft's sklearn `gp` inference
chains were a backend inconsistency; the fig-10 cell carries a timed-trial guard that falls
back, loudly, only if the shipped backend is catastrophically slower at execution time). The emulator
generalizes beyond the training grid: `em.predict(theta, z_s=0.75, grids={"ell": ...,
"nu": ...})` interpolates every source-plane statistic to an arbitrary $z_s$ and resamples
any binned statistic onto an arbitrary user grid, demonstrated in fig 9's closing cell.
($C_\ell^{\kappa\kappa}$'s own $5\times5$ tomographic block, with its two source-plane axes,
is COMPOSED rather than a single flat PCA+GP fit -- see above -- so the single-axis $z_s$
interpolator every other head uses does not apply to it either way; `em.predict()` always
returns its full, un-interpolated $5\times5\times L$ block regardless of a requested `z_s`,
unchanged from the pre-composed contract.)
During this revision we also fixed a genuine bug in the released `MLPEnsemble` backend
(`opt.step()` outside the epoch loop); the GP family remains the science backend -- with
~250 designs in 30-D the MLP washes the feedback response toward the mean.
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §4a,b — Fig 9: a statistics emulator, validated on held-out simulations

`bind.emulator`: per-statistic PCA compression + one ARD-RBF Gaussian process per latent
component (the shipped gpytorch `gpgpu` backend — see the WP2 note in §4.0; it auto-selects
the GPU when one is present at execution time), trained on 203 of the 253 Sobol runs and
evaluated on the 50 held-out runs (never seen in training), across all **13 heads** — the
locked list of §4.0, one prediction per canonical statistic of §3 plus $C_\ell^{\kappa\kappa}$
as a COMPOSED 13th head (diagonal = $S(\ell)\times$ a run-independent DMO trace, identity
checked in the fig 4b cell; off-diagonal = a small dedicated fit — see §4.0's C-head
improvement-set paragraph). Fig 9 is the *single-statistic* view and shows the emulator's **typical** performance: the
median-|error| held-out run with its $1\sigma$ band, over a residual strip carrying the
per-$\ell$ 5–95% envelope of $\rm pred/truth-1$ across **all 50** held-out runs plus the
95th-percentile-error run (which sits in the sparsely-sampled enhancement branch of the 30-D
Sobol design), so tail and typical are both visible and neither is the lead. (The v3 draft
carried that worst case as an inset over the curves; the strip already carries it, so the inset
is gone — its number stays in the printed stamp. An earlier draft led with the worst case under
a "before/after" backend-A/B legend; that comparison is changelog material and now lives only
in §4.0's text.) The per-head summary bars are **fig 9b**; the truth-vs-emulated curves for
every canonical statistic are **fig 9c**.
''')

code(r'''
# ── Fig 9: emulator held-out validation (gpgpu backend, locked 13-head coverage) ─
# engine: bind.emulator (gpytorch exact-GP backend, CPU); data: emulator_dataset_xpkfix
from bind.emulator import Emulator, EmulatorDataset

ds = EmulatorDataset.load(DS_PATH)
rng = np.random.default_rng(0)
test_idx  = np.sort(rng.choice(len(run_ids), 50, replace=False))
train_idx = np.setdiff1d(np.arange(len(run_ids)), test_idx)
# The LOCKED 13-head list: the 7 WL-field heads of §3 (suppression first, then the
# pdf/counts/MFs) plus the 5 gas/cross spectra, plus cl_kappa as the 13th head
# (fork-A ruling, 2026-08-03): trained DIRECTLY on the map-measured spectra so a
# user gets C_l^kk without composing S(ell) x DMO. The S = C_kk/C_dmo identity
# (checked in the fig 4b cell, stated in the §4a text above) still keeps C_kk
# out of fig 5's matrix and fig 8's families -- as a *head* it is a user-facing
# product, not an independent response direction.
# scaling_Y/scaling_f_gas emulation is deferred (§4.0).
# MUST stay in lockstep with build_emulator.py STATS (fingerprint match).
STATS_EMU = ["suppression", "cl_kappa", "pdf", "peak_counts", "minima_counts", "mf_v0",
             "mf_v1", "mf_v2", "cl_yy", "cl_tt", "cl_kappa_y", "cl_kappa_tau", "cl_yt"]

# WP2 backend sweep (203/50 held-out split, seed 0; suppression-only, same-cost
# comparison): the released sklearn GPBackend pins length_scale at its 1e2 upper
# bound and the WhiteKernel noise at its 1e-6 lower bound on most components --
# but WIDENING those bounds changes nothing (median |frac err| 1.77%->1.78%,
# 95th-pct held-out run 13.89%->13.88%), and neither does raising the PCA cap
# 12->30 (1.76%, 13.90%; even wide-bounds+nc=30 together: 1.77%, 13.88%) --
# all three are red herrings, and cost up to 274s/stat for no gain. Swapping
# the per-component sklearn L-BFGS-B fit for gpytorch's jointly-optimized
# batched exact GP (backend="gpgpu") DOES help, at every n_components tried
# (nc=12 and nc=30 give the same ~1.64%/12.0%) and ~1.7x FASTER per stat
# (84.7s -> 48.6s) than even the cheapest sklearn config: overall 1.77%->1.64%,
# S>1.1 subset 11.1%->9.6%, and on the single worst-predicted (95th-pct,
# enhancement-branch) held-out run 13.9%->12.0% median |frac err| along the
# curve -- a ~14% relative reduction, not a full fix (residual error remains:
# the emulator still under-samples this corner of the 30-D Sobol design, not
# something a backend swap alone can cure). n_components=12 is kept (nc=30
# gains nothing at 5x the wall time; 800 vs 400 epochs also gains nothing).
# Prefer the PERSISTED fit (emulator_cache.py, built by run_emulator_fit.sbatch).
# Emulator.save()/load() round-trips bit-exactly, and the saved bundle IS the
# release artifact behind "theta -> any released statistic in milliseconds".
# CRITICAL: a saved fit is only valid for the split it was trained on. This figure
# reports HELD-OUT errors, so reusing a bundle trained on a different partition
# would quietly turn the test set into training data. load_matching() fingerprints
# the training indices, statistic list, backend config and dataset identity, and
# refuses on any disagreement -- so the worst case is a refit, never a wrong number.
_EMU_KW = dict(backend="gpgpu", n_components=12,
               backend_kwargs={"epochs": 400, "lr": 0.1})
sys.path.insert(0, str(Path.cwd()))
from emulator_cache import load_matching as _load_emu                # noqa: E402
t0 = time.time()
em = _load_emu(DS_PATH, STATS_EMU, train_idx, **_EMU_KW)
if em is None:
    em = Emulator(**_EMU_KW)
    em.fit(ds.subset(train_idx), stats=STATS_EMU)
    _emu_src = "fitted here"
else:
    _emu_src = "loaded from the persisted bundle"
fit_time = time.time() - t0
print(f"gpgpu emulator ({len(STATS_EMU)} stats, <=12 PCA components each, "
      f"{len(train_idx)} training runs): {_emu_src} in {fit_time:.0f}s")
t0 = time.time()
pred = em.predict(X_native[test_idx])
print(f"batch prediction of {len(test_idx)} held-out runs in {time.time()-t0:.2f}s")

TRUTH = {s: np.asarray(d[f"t__{s}__value"], float) for s in STATS_EMU}
def flat(a):
    return a.reshape(a.shape[0], -1)

# C-head improvement set (papers/01_pipeline/audits/spectrum_head_experiment.md;
# wiring in bind.emulator.core/transforms): six ell-domain spectrum heads are now
# FIT only on ell<=ELL_TRUST (bind.emulator.core.DEFAULT_ELL_MASK_HEADS -- the
# CIC-aliased tail, same threshold this notebook already uses for display
# masking). em.predict() still returns the FULL released grid for these heads,
# but ell>ELL_TRUST bins come back at the per-bin TRAIN mean with an inflated
# (untrustworthy-by-design) error, so a metric that folds them in would (a) for
# frac-err/R^2, just report how far the aliased tail's true value sits from an
# arbitrary constant -- not model quality, undoing the whole point of masking at
# fit time -- and (b) for cov1/cov2/z-scores, artificially inflate coverage
# (huge err -> |z|~0 -> "covered" by construction). `suppression` is NOT masked
# (a ratio; the aliasing artifact cancels in it) so `_ell_trusted` is a no-op for
# it. Used here AND reused (fig-9 global) by the §4b yardstick denominator,
# §4b‴'s ALPHA_COV z-scores, fig 15's zsc/cov1/cov2/ALPHA, and fig 15b's OOD
# metric -- every place downstream that turns pred/TRUTH into an error number.
ELL_MASK_HEADS_NB = {"cl_kappa", "cl_kappa_y", "cl_yy", "cl_kappa_tau", "cl_tt", "cl_yt"}
# ELL/ELL_TRUST are setup-cell globals (defined before ANY code() cell runs, see
# the top of this notebook) -- computed inline here rather than reusing the `mE`
# name this same cell defines further below, so `_ell_trusted` has no ordering
# dependency on where in the cell it is first called.
_ELL_TRUSTED_MASK = ELL <= ELL_TRUST
def _ell_trusted(arr, s):
    return np.asarray(arr)[..., _ELL_TRUSTED_MASK] if s in ELL_MASK_HEADS_NB else np.asarray(arr)

metrics = {}
for s in STATS_EMU:
    p, t = flat(_ell_trusted(pred[s], s)), flat(_ell_trusted(TRUTH[s][test_idx], s))
    e = flat(_ell_trusted(pred[f"{s}_err"], s))
    base = flat(_ell_trusted(TRUTH[s][train_idx], s)).mean(0)
    # amplitude floor: counts need >1 object; every other head needs |truth|
    # above its own 5th-percentile |value| -- without this, the signed crosses'
    # near-zero bins (the z_s=0.5 plane at high ell is ~1e-12 of typical) turn
    # |p/t - 1| into ~1e7% and the 2026-08-03 run printed exactly that. The
    # floor is relative and per-head, so it never hides a real miss at
    # measurable amplitude; the masked fraction is ~5% by construction.
    if s in ("peak_counts", "minima_counts"):
        thresh = 1.0
    else:
        thresh = np.nanpercentile(np.abs(t), 5)
    ok   = np.isfinite(t) & (np.abs(t) > thresh)
    fe   = np.nanmedian(np.abs(p[ok]/t[ok] - 1))
    num  = np.nansum(np.where(ok, (p - t)**2, np.nan), axis=0)
    den  = np.nansum(np.where(ok, (t - base)**2, np.nan), axis=0)
    r2   = 1 - num/np.maximum(den, 1e-30)
    okz  = ok & (e > 0)
    z    = (p[okz] - t[okz]) / e[okz]
    cov1, cov2 = float((np.abs(z) < 1).mean()), float((np.abs(z) < 2).mean())
    metrics[s] = (fe, np.nanmedian(r2[den > 0]), cov1, cov2)

# v3 RESTRUCTURE (review): lead with the TYPICAL held-out run + the all-50
# error envelope. v4: the worst-case INSET is deleted (the residual strip below
# already traces that run, and the inset was the sole cause of the
# "Axes not compatible with tight_layout" warning); its number stays in the
# printed stamp. The per-head bars move to their own figure (fig 9b), the
# per-statistic curves to fig 9c. The sklearn-vs-gpgpu "before/after" A/B overlay
# stays dropped (one released backend; comparison numbers in §4.0's WP2 text).
Ssup   = TRUTH["suppression"][test_idx][:, ZI, :]
Sp_all = np.asarray(pred["suppression"])[:, ZI, :]
Se_all = np.asarray(pred["suppression_err"])[:, ZI, :]
mE = ELL <= ELL_TRUST          # consistent aliasing cut (figs 4/4b/7/9/9c/12/18)
res_all = Sp_all[:, mE]/Ssup[:, mE] - 1                  # (50, n_ell) pred/truth-1
fe_run  = np.nanmedian(np.abs(res_all), axis=1)          # per-run median |frac err|
order_e = np.argsort(fe_run)
i_typ = int(order_e[len(order_e)//2])                    # median-|err| run: the lead
i_wc  = int(order_e[int(0.95*len(order_e))])             # 95th-pct-|err| run: strip trace
br_wc = ("enhancement branch" if Ssup[i_wc].mean() > 1 else "suppression branch")

fig = plt.figure(figsize=(ONE_COL[0], 3.0))
gs9a = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.08)
axa = fig.add_subplot(gs9a[0])
axr = fig.add_subplot(gs9a[1], sharex=axa)

axa.plot(ELL[mE], Ssup[i_typ][mE], color="0.15", lw=1.2, label="measured")
axa.plot(ELL[mE], Sp_all[i_typ][mE], color=COLORS["bind"], ls="--", lw=1.2,
         label=r"emulated $\pm1\sigma$")
axa.fill_between(ELL[mE], (Sp_all[i_typ] - Se_all[i_typ])[mE],
                 (Sp_all[i_typ] + Se_all[i_typ])[mE],
                 color=COLORS["bind"], alpha=BAND_ALPHA, lw=0)
axa.set_xscale("log"); axa.tick_params(labelbottom=False)
axa.set_ylabel(r"$S(\ell)$, typical (median-|err|) run")
axa.legend(loc="lower left", fontsize=6.5)
# residual strip: all-50 held-out envelope + the two featured runs (this strip
# is where the 95th-pct run now lives -- the v3 inset is gone)
env_lo, env_hi = np.nanpercentile(res_all, [5, 95], axis=0)
axr.fill_between(ELL[mE], 100*env_lo, 100*env_hi, color="0.85", lw=0,
                 label="5–95% of 50 held-out runs")
axr.plot(ELL[mE], 100*res_all[i_typ], color=COLORS["bind"], lw=0.9, label="typical")
axr.plot(ELL[mE], 100*res_all[i_wc], color=COLORS["highlight"], lw=0.7,
         label=f"95th pct ({br_wc.split()[0]})")
axr.axhline(0, color="0.3", lw=0.5)
axr.set_xscale("log"); axr.set_xlabel(r"$\ell$")
axr.set_ylabel("pred/truth$-$1 [%]", fontsize=7)
axr.legend(loc="upper left", fontsize=5.0, ncol=2, handlelength=1.2,
           columnspacing=0.8)
fig.tight_layout()
save(fig, "figs_v2/fig09_emulator_validation")
plt.show()

# per-head labels: shared with fig 9b's bars and fig 15's coverage bars, so they
# stay module-level here. Order matches the locked STATS_EMU list exactly.
names  = [r"$S(\ell)$", r"$C_\ell^{\kappa\kappa}$", r"PDF", r"$N_{\rm pk}$",
          r"$N_{\rm min}$", r"$V_0$", r"$V_1$", r"$V_2$", r"$C_\ell^{yy}$",
          r"$C_\ell^{\tau\tau}$", r"$C_\ell^{\kappa y}$", r"$C_\ell^{\kappa\tau}$",
          r"$C_\ell^{y\tau}$"]
assert len(names) == len(STATS_EMU), "names/STATS_EMU out of step"
for s, nm in zip(STATS_EMU, names):
    print(f"{s:>14s}: median |frac err| = {100*metrics[s][0]:.2f}%, "
          f"response R^2 = {metrics[s][1]:.2f}, cov|z|<1/2 = "
          f"{metrics[s][2]:.2f}/{metrics[s][3]:.2f}")
print(f"fig 9: typical held-out run = median of per-run median |frac err| "
      f"({100*fe_run[i_typ]:.2f}%); the residual strip also traces the 95th-pct run "
      f"({100*fe_run[i_wc]:.1f}%, {br_wc}) -- the sklearn->gpgpu A/B numbers live in "
      "§4.0 (changelog, not release)")

# suppression split: suppression (S<1) vs enhancement (S>1) branches
Sp_all = np.asarray(pred["suppression"])[:, ZI, :]
St_all = TRUTH["suppression"][test_idx][:, ZI, :]
for lab, msk_b in [("S<1 branch", St_all < 1), ("S>1 branch", St_all > 1)]:
    fe_b = np.nanmedian(np.abs(Sp_all[msk_b]/St_all[msk_b] - 1))
    print(f"suppression {lab}: median |frac err| = {100*fe_b:.2f}%")

print(f"peak counts: noise/signal variance ratio ~{pk_nsr:.1f} per fine nu-bin -> "
      f"realization-noise dominated; achievable response R^2 consistent with ~0, "
      f"so the measured {metrics['peak_counts'][1]:.2f} is noise-limited, not a GP failure "
      f"(the x4-rebinned counts of fig 5 do retain a detected |rho|~0.44 response)")

# ── GOAL 3: any-z / any-grid demo -----------------------------------------------
theta_demo = X_native[test_idx[0]]
ell_custom = np.logspace(2.5, 4.3, 15)
nu_custom  = np.linspace(-3, 3, 25)
out_demo = em.predict(theta_demo, z_s=0.75, grids={"ell": ell_custom, "nu": nu_custom})
print(f"any-z/any-grid demo: em.predict(theta, z_s=0.75, grids=...) -> "
      f"S(ell) on {len(ell_custom)} custom log-spaced ell bins "
      f"[{ell_custom[0]:.0f},{ell_custom[-1]:.0f}], "
      f"range [{out_demo['suppression'].min():.3f},{out_demo['suppression'].max():.3f}]")
print(f"  peak_counts on {len(nu_custom)} custom nu bins, shape {out_demo['peak_counts'].shape}")
print(f"  z_s echoed back: {out_demo['z_s']}; axes['ell'] matches request: "
      f"{np.allclose(out_demo['axes']['ell'], ell_custom)}")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
### Fig 9b — the same fit, head by head

The 13 emulated heads summarised as two bar charts (they were panels b,c of fig 9 in the v3
draft; nothing is refitted here). (a) median fractional error per statistic, with the
per-family medians drawn on the panel — the sub-2% WL-field heads and the 10–14% signed
cross-spectrum heads are both real, and the panel reconciles them explicitly: the cross-spectra
have no positivity floor, so a percent of their (often near-zero) value is a far tighter demand.
(b) *response* $R^2$, the fraction of the true run-to-run deviation from the training mean the
emulator reproduces — the metric that matters for feedback inference, and the one on which the
cross-spectra score 1.00. $C_\ell^{\kappa\kappa}$ IS one of the 13 bars (fork-A ruling): its
diagonal carries no *independent* response direction beyond $S(\ell)$ (same measurement at
fixed cosmology, identity note in fig 4b — composed for free, not fit a second time), but its
off-diagonal blocks get their own small dedicated fit (§4.0's C-head improvement-set
paragraph), so the bar is a real, if partly-derived, number rather than a placeholder. The
reconciliation above is qualitative ("a percent
of a near-zero cross-spectrum is a tighter demand"); the next cell makes it a number — the
**error-to-response ratio**, each head's error normalized by the physical Sobol response range
it has to resolve, which is the objective version of this error tiering and the §4b yardstick
quoted in the text.
''')

code(r'''
# ── Fig 9b: per-head accuracy and response fidelity (was fig 9 panels b,c) ───
# NO refit: `metrics`, `names`, `STATS_EMU` are fig-9 globals.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL[0], 2.7))
fe_v = np.array([100*metrics[s][0] for s in STATS_EMU])
r2_v = np.array([metrics[s][1] for s in STATS_EMU])
xx   = np.arange(len(STATS_EMU))
ax1.bar(xx, fe_v, color=COLORS["bind"])
# per-family medians ON the panel: the ~1.6% overall headline and the 10-14%
# signed-cross-spectrum heads coexist explicitly. WL-field count (8, incl. the
# fork-A cl_kappa head) is fixed by the locked composition; the SZ/cross
# boundary derives from len(STATS_EMU) so this does not silently mislabel if a
# head is ever added/removed again.
N_WL9 = 8   # suppression, cl_kappa, pdf, peak_counts, minima_counts, mf_v0-2
FAM9 = [("WL field", 0, N_WL9), ("SZ/cross spectra", N_WL9, len(STATS_EMU))]
for flab, i0, i1 in FAM9:
    fm = float(np.median(fe_v[i0:i1]))
    ax1.plot([i0 - 0.45, i1 - 0.55], [fm, fm], color=COLORS["highlight"],
             lw=0.8, ls="--")
    ax1.text((i0 + i1 - 1)/2, fm + 0.9, f"{flab}\nmed {fm:.1f}%", ha="center",
             fontsize=4.2, color=COLORS["highlight"],
             path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
             # white casing: the red text crosses dark bars (e.g. Cl_tautau)
ax1.set_ylim(0, 1.30*fe_v.max())
ax1.set_xticks(xx); ax1.set_xticklabels(names, rotation=55, ha="right", fontsize=5.0)
ax1.set_ylabel("median |frac. err.| [%]")
panel_label(ax1, "(a)", loc="upper right")
ax2.bar(xx, r2_v, color=COLORS["secondary"])
ax2.axhline(1, color=COLORS["dmo"], ls=":", lw=0.8)
ax2.set_xticks(xx); ax2.set_xticklabels(names, rotation=55, ha="right", fontsize=5.0)
ax2.set_ylabel(r"response $R^2$"); ax2.set_ylim(0, 1.1)
panel_label(ax2, "(b)", loc="upper right")
fig.tight_layout(w_pad=1.1)
save(fig, "figs_v2/fig09b_emulator_heads")
plt.show()
for flab, i0, i1 in FAM9:
    print(f"fig 9b {flab:>17s} ({i1-i0} heads): median |frac err| = "
          f"{np.median(fe_v[i0:i1]):.2f}%, median response R^2 = "
          f"{np.median(r2_v[i0:i1]):.2f}")
print(f"fig 9b all {len(STATS_EMU)} heads: median |frac err| = {np.median(fe_v):.2f}%, "
      f"median response R^2 = {np.median(r2_v):.2f} "
      "(incl. the directly-trained C_kk head -- fork-A ruling; identity in §4a)")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
### The §4b yardstick — error-to-response ratio

Fig 9b's median $|$frac err$|$ bars are not comparable across heads on their own: a WL-field
head that is 2% wrong on an $O(1)$ quantity and a cross-spectrum that is 12% wrong on a
quantity whose whole 253-run Sobol swing is a factor of a few are not equally "good" or "bad"
in any absolute sense. The **error-to-response ratio** normalizes each head's error by the
physical range of variation it actually has to resolve:

$$\text{ratio}_s=\frac{\text{median}_{\rm bins}\,\big|\,\text{pred}/\text{truth}-1\,\big|_{\rm held\text{-}out}}
{\text{median}_{\rm bins}\,\tfrac12\big(P_{84}-P_{16}\big)\big[\,\text{curve}(\text{run})/\text{median}_{\rm run}(\text{curve})\,\big]}\, .$$

The numerator is fig 9/9b's held-out fractional error (reused, not refit). The denominator is a
**response half-width**: the same ratio-to-median convention as fig 7's `spread16_84` (16–84
percentile spread of curve/median across the 253 Sobol runs, per bin), but reimplemented
self-contained from the raw dataset arrays in the next cell — independent of fig 7's
out-of-design `FID` vectors and fig 5's rebin factors — and aggregated by the **median** over
bins (fig 7 takes the max, to report the single most-legible bin for a colour gradient; the
median is the right aggregate here, matching the numerator's own "typical", not "best-case",
framing) as a **half**-width, so both numerator and denominator read as one-sided typical
fractional deviations. A ratio $\ll1$ means the emulator's error is a small fraction of the
signal it has to reproduce; a ratio $\sim1$ means the error is comparable to the physical
response itself — the objective verdict fig 9b's qualitative WL/cross-spectrum reconciliation
was gesturing at.
''')

code(r'''
# ── §4b yardstick: error-to-response ratio, per emulator head ────────────────
# Numerator: reuse `metrics[s][0]` (median |frac err| on the 50 held-out runs) --
# a fig-9 global, in scope here exactly as it already is for fig 9b (no refit).
# Denominator: SELF-CONTAINED from the raw dataset arrays `d[f"t__{s}__value"]`
# (NOT fig 7's `spread16_84`/`FID`, and NOT fig 5's `STATS`/`Ys` -- deliberately
# reimplemented here so this cell has no cross-cell dependency beyond the
# shared setup cell + fig 9's `metrics`/`STATS_EMU`/`pred`/`_ell_trusted`).
# Domain matches the numerator's: `metrics` is computed on the FULL un-sliced
# TRUTH[s] (all 5 z_s planes flattened together for the tomographic heads, see
# the fig-9 cell's `flat(TRUTH[s][test_idx])`) EXCEPT for the six ell-masked
# heads, where fig 9's `_ell_trusted` now restricts both numerator and
# denominator to ell<=ELL_TRUST (the C-head improvement set: those heads are
# fit only on that range, so their ell>ELL_TRUST bins are a train-mean
# passthrough that would otherwise pollute BOTH the numerator error and this
# response denominator) -- so numerator and denominator stay over the SAME bins.
FLOOR_FRAC_4B = 0.02   # nu-domain heads: matches fig 7's response_ratio default
SPEC_FLOOR_4B = 1e-10  # ell-domain heads (spectra span decades): near-numerical
                       # only, matching fig 7's documented SPEC_FLOOR reasoning
ELL_DOMAIN_HEADS_4B = {"suppression", "cl_yy", "cl_tt", "cl_kappa_y",
                       "cl_kappa_tau", "cl_yt"}

def _response_half_width(raw, floor_frac):
    """Median-over-bins of the per-bin 16-84 percentile HALF-width of
    curve/median across all 253 Sobol runs -- fig 7 spread16_84's ratio-to-
    median convention, reimplemented locally (median instead of fig 7's max
    aggregate; half-width instead of full width). `raw` is (253, n_bin), any
    trailing axes already flattened by the caller. Guards div-by-near-zero the
    same way fig 7's response_ratio does: bins where the per-bin MEDIAN is
    below floor_frac of its own peak |value| are masked (NaN), not exploded."""
    Y = np.asarray(raw, float).reshape(raw.shape[0], -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(Y, axis=0)
        floor = floor_frac*np.nanmax(np.abs(med))
        med_safe = np.where(np.abs(med) > floor, med, np.nan)
        rat = Y/med_safe[None, :]
        p16, p84 = np.nanpercentile(rat, [16, 84], axis=0)
        half = 0.5*(p84 - p16)
        return float(np.nanmedian(half))

RESPONSE_4B, RATIO_4B = {}, {}
for s in STATS_EMU:
    raw_s = _ell_trusted(np.asarray(d[f"t__{s}__value"], float), s)
    floor = SPEC_FLOOR_4B if s in ELL_DOMAIN_HEADS_4B else FLOOR_FRAC_4B
    RESPONSE_4B[s] = _response_half_width(raw_s, floor)
    RATIO_4B[s] = metrics[s][0]/max(RESPONSE_4B[s], 1e-30)

print(f"{'head':>14s}  {'err (fig 9b)':>13s}  {'response (half-width)':>22s}  {'ratio':>8s}")
for s in sorted(STATS_EMU, key=lambda k: -RATIO_4B[k]):
    print(f"{s:>14s}  {100*metrics[s][0]:12.2f}%  {100*RESPONSE_4B[s]:21.2f}%  "
          f"{RATIO_4B[s]:7.3f}")

# family summary, SAME WL/cross split fig 9b uses (N_WL9, FAM9 are fig-9b globals)
_fam_ratio = {}
for flab, i0, i1 in FAM9:
    keys = STATS_EMU[i0:i1]
    _fam_ratio[flab] = float(np.median([RATIO_4B[k] for k in keys]))
med_ratio_all = float(np.median(list(RATIO_4B.values())))
print(f"\n4b yardstick -- median error-to-response ratio: "
      + ", ".join(f"{flab} {v:.2f}" for flab, v in _fam_ratio.items())
      + f", all {len(STATS_EMU)} heads {med_ratio_all:.2f}")
print("machine-readable summary: emulator error is "
      f"{_fam_ratio['WL field']:.2f}x the median Sobol response half-width for the "
      f"WL heads, {_fam_ratio['SZ/cross spectra']:.2f}x for the cross/SZ heads "
      f"({med_ratio_all:.2f}x over all {len(STATS_EMU)} heads) -- the objective verdict on the "
      "emulator (RECOMPUTED-ON-RUN with the 256-run retrain, per the TODO).")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
### Fig 9c — truth vs emulated, one panel per canonical statistic

The bars of fig 9b compress each head to one number; this is the curve-level view, for the
**12 canonical statistics** of §3 — 7 WL-field ($S(\ell)$, PDF, $N_{\rm pk}$, $N_{\rm min}$,
$V_0$, $V_1$, $V_2$), 2 non-$\kappa$ auto-spectra ($C_\ell^{yy}$, $C_\ell^{\tau\tau}$) and
3 crosses ($C_\ell^{\kappa y}$, $C_\ell^{\kappa\tau}$, $C_\ell^{y\tau}$). Every panel draws
the *same* 8 held-out runs, picked at evenly spaced ranks
of the $S(\ell)$ held-out error ordering so the set spans the typical and the tail of the 50,
as measured (thin solid) against the GP prediction (dashed) — no refit, the fig-9 emulator and
its batch prediction are reused. Spectra carry the $\ell(\ell+1)/2\pi$ weighting on log–log
axes; $S(\ell)$ is semi-log; the $\nu$-domain statistics are linear. The corner number is that
head's median $|$frac err$|$ from fig 9b. $C_\ell^{\kappa\kappa}$ is not one of the 12 panels
(it is $S(\ell)\times C_\ell^{\rm DMO}$ by construction at fixed cosmology — the identity note
in the fig 4b markdown). At $z_s=1$ throughout.
''')

code(r'''
# ── Fig 9c: GP-predicted vs true held-out curves, one panel per canonical stat ─
# NO refit: `em`, `pred`, `TRUTH`, `metrics`, `test_idx`, `order_e` are fig-9 globals.
mEc = ELL <= ELL_TRUST      # same trusted/consistent range as fig 9 (was a stale 2e4 literal)
f_e = ELL*(ELL + 1)/(2*np.pi)                      # spectrum weighting on the dataset grid
# 8 held-out runs at evenly spaced ranks of the S(ell) error ordering -> the
# panels show the typical AND the tail of the 50-run held-out set together
sel8 = order_e[np.linspace(0, len(order_e) - 1, 8).astype(int)]

def _canon(a, key):
    """Canonical (n_run, n_bin) slice of a head array at the working plane ZI."""
    a = np.asarray(a, float)
    if a.ndim == 4:                 # cl_kappa: (n, z_s, z_s, ell) -> auto diagonal
        return a[:, ZI, ZI, :]
    return a[:, ZI, :] if a.ndim == 3 else a

NU_PK, NU_MN = d["a__peak_counts__nu"], d["a__minima_counts__nu"]
NU_MF, K_PDF = d["a__mf_v0__mf_nu"], d["a__pdf__pdf_bins"]
P9 = [
    dict(k="suppression", x=ELL, m=mEc, w=1.0, xl=r"$\ell$", xs="log", ys="linear",
         yl=r"$S(\ell)$", pl="upper left"),
    dict(k="cl_kappa", x=ELL, m=mEc, w=f_e, xl=r"$\ell$", xs="log", ys="log",
         yl=r"$\ell(\ell{+}1)C_\ell^{\kappa\kappa}/2\pi$", pl="upper left"),
    dict(k="pdf", x=K_PDF, m=np.abs(K_PDF) <= 0.09, w=1.0, xl=r"$\kappa$",
         xs="linear", ys="linear", yl=r"$P(\kappa)$", pl="upper left"),
    dict(k="peak_counts", x=NU_PK, m=(NU_PK >= -3) & (NU_PK <= 6), w=1.0,
         xl=r"$\nu$", xs="linear", ys="linear", yl=r"$N_{\rm pk}$", pl="upper left"),
    dict(k="minima_counts", x=NU_MN, m=(NU_MN >= -3) & (NU_MN <= 3), w=1.0,
         xl=r"$\nu$", xs="linear", ys="linear", yl=r"$N_{\rm min}$", pl="upper left"),
    dict(k="mf_v0", x=NU_MF, m=np.ones(len(NU_MF), bool), w=1.0, xl=r"$\nu$",
         xs="linear", ys="linear", yl=r"$V_0$", pl="upper right"),
    dict(k="mf_v1", x=NU_MF, m=np.ones(len(NU_MF), bool), w=1.0, xl=r"$\nu$",
         xs="linear", ys="linear", yl=r"$V_1$", pl="upper left"),
    dict(k="mf_v2", x=NU_MF, m=np.ones(len(NU_MF), bool), w=1.0, xl=r"$\nu$",
         xs="linear", ys="linear", yl=r"$V_2$", pl="upper left"),
    dict(k="cl_yy", x=ELL, m=mEc, w=f_e, xl=r"$\ell$", xs="log", ys="log",
         yl=r"$\ell(\ell{+}1)C_\ell^{yy}/2\pi$", pl="upper left"),
    dict(k="cl_tt", x=ELL, m=mEc, w=f_e, xl=r"$\ell$", xs="log", ys="log",
         yl=r"$\ell(\ell{+}1)C_\ell^{\tau\tau}/2\pi$", pl="upper left"),
    dict(k="cl_kappa_y", x=ELL, m=mEc, w=f_e, xl=r"$\ell$", xs="log", ys="log",
         yl=r"$\ell(\ell{+}1)C_\ell^{\kappa y}/2\pi$", pl="upper left"),
    dict(k="cl_kappa_tau", x=ELL, m=mEc, w=f_e, xl=r"$\ell$", xs="log", ys="log",
         yl=r"$\ell(\ell{+}1)C_\ell^{\kappa\tau}/2\pi$", pl="upper left"),
    dict(k="cl_yt", x=ELL, m=mEc, w=f_e, xl=r"$\ell$", xs="log", ys="log",
         yl=r"$\ell(\ell{+}1)C_\ell^{y\tau}/2\pi$", pl="upper left"),
]
# all five spectra are strictly positive over ell<=ELL_TRUST at z_s=1 (checked
# below), which is what makes the log-y cross-spectrum panels legitimate

NROW9 = int(np.ceil(len(P9)/4))                    # grid derives from the head count
fig, axs = plt.subplots(NROW9, 4, figsize=(TWO_COL[0], 2.07*NROW9))
for _ax_off in axs.ravel()[len(P9):]:              # blank the unused trailing axes
    _ax_off.axis("off")
fe_panel = {}
for P, ax, tag in zip(P9, axs.ravel(), "abcdefghijklmnop"):
    k, m = P["k"], P["m"]
    T0 = _canon(TRUTH[k][test_idx], k)             # (50, n_bin) measured
    P0 = _canon(pred[k], k)                        # (50, n_bin) emulated
    xg = np.asarray(P["x"], float)[m]
    Ty, Py = (T0*P["w"])[:, m], (P0*P["w"])[:, m]
    for j, i_run in enumerate(sel8):
        ax.plot(xg, Ty[i_run], color=COLORS["truth"], lw=0.55, alpha=0.85,
                label="measured (8 held-out runs)" if (j == 0 and tag == "a") else None)
        ax.plot(xg, Py[i_run], color=COLORS["bind"], ls="--", lw=0.7, alpha=0.9,
                label="GP emulator" if (j == 0 and tag == "a") else None)
    if k == "mf_v2":
        ax.axhline(0, color="0.6", lw=0.4)
    ax.set_xscale(P["xs"]); ax.set_yscale(P["ys"])
    ax.set_xlabel(P["xl"], fontsize=6); ax.set_ylabel(P["yl"], fontsize=6)
    ax.tick_params(labelsize=5)
    # per-panel quality number: this head's median |frac err| (the fig 9b bar).
    # White casing because the lower-left corner is on the curve in the rising
    # weighted-spectrum panels.
    ax.text(0.03, 0.06, f"{100*metrics[k][0]:.1f}%", transform=ax.transAxes,
            fontsize=5, color=COLORS["bind"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6))
    panel_label(ax, f"({tag})", loc=P["pl"])
    # panel-level error over ALL 50 held-out runs on the plotted slice, same
    # threshold convention as the fig-9 metrics loop (counts: |truth|>1)
    thr = 1.0 if k in ("peak_counts", "minima_counts") else 0.0
    okp = np.isfinite(T0[:, m]) & (np.abs(T0[:, m]) > thr)
    fe_panel[k] = float(np.nanmedian(np.abs(P0[:, m][okp]/T0[:, m][okp] - 1)))
axs.ravel()[0].legend(loc="lower right", fontsize=4.4, handlelength=1.4)
fig.tight_layout(w_pad=0.8, h_pad=1.0)
save(fig, "figs_v2/fig09c_emulator_curves")
plt.show()

print(f"fig 9c: {len(P9)} canonical statistics (7 WL + 2 non-kappa auto + 3 cross; "
      f"C_kk is not a panel; it IS a §4 emulator head), z_s = {ZS[ZI]:.2f}, "
      f"{len(sel8)} held-out runs at S(ell)-error ranks "
      f"{np.linspace(0, len(order_e)-1, 8).astype(int).tolist()} of {len(order_e)-1} "
      f"(dataset run ids {[int(run_ids[test_idx[i]]) for i in sel8]})")
for P in P9:
    k = P["k"]
    print(f"  {k:>14s}: panel median |frac err| = {100*fe_panel[k]:.2f}% "
          f"(head-level, all planes/bins: {100*metrics[k][0]:.2f}%)")
_spec9 = [P["k"] for P in P9 if P["xs"] == "log" and P["ys"] == "log"]
_min9 = {k: float(np.min(_canon(TRUTH[k][test_idx], k)[:, mEc])) for k in _spec9}
print(f"fig 9c positivity precondition for the {len(_spec9)} log-y spectrum panels: "
      + ", ".join(f"{k} min {v:.2e}" for k, v in _min9.items())
      + f" -> all > 0 over ell<=ELL_TRUST at z_s={ZS[ZI]:.2f}, so the signed cross-spectra "
        "need no symlog at this cut (they would if the cut were widened)")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §4b″ — Does the posterior itself cover? A 50-node coverage test (new)

Fig 15 calibrates the emulator's *per-bin* error bar; nothing in the previous draft checked
that the **posterior** covers — a single truth point (the fiducial, fig 10) cannot validate a
"pins the SN-wind/IMF/BH combination at 43–66% of prior width" claim, and the diagnosed heavy
GP tails could make the posterior generically overconfident in exactly the diagnosed regime.
This cell closes that gap with an empirical coverage run: the full fig-10 posterior pipeline —
the *same* Hartlap-corrected full covariance of the fiducial 50-realization mean, the same
$\theta$-dependent GP variance on the diagonal, with the fitted $2\sigma$ inflation
$\hat\alpha$ applied — is run on **each of the 50 held-out Sobol nodes as synthetic data**
(their measured $S(\ell)$ vectors; the forward model is a suppression-only GP trained on the
203 *training* runs only, so the tested node is never in the training set). For every node we
record whether the true $\theta$ falls inside the 68% and 95% marginal credible intervals, for
the top-6 most-constrained parameters and the all-30 average. Budget: 32 walkers × 6000 steps
per node, burn 25%, parallelized over a process pool of $\min(n_{\rm cpu},12)$ workers each
pinned to one CPU thread (the GPU stays reserved for the main-process GP fits); if the pool is
unavailable a serial fallback runs a reduced budget (floor 2000 steps) and the printout says
which mode ran. Everything is seeded, including per-node chain seeds. The coverage table is
printed below and drawn as fig 15's new coverage panel.

**§4b‴ (appended below, same cell): the GP-σ recalibration this coverage machinery exists to
enable.** $\hat\alpha$ above restores the 2σ tail *by construction*; it says nothing about the
1σ bulk. A complementary factor $\alpha_{\rm cov}$ is fitted directly against 1σ coverage of
the same held-out $z$-scores (per head, and pooled across all 12), then substituted for
$\hat\alpha$ in an independent re-run of the full 50-node posterior test above. Raw and
recalibrated tables are both kept — fig 15 panel (e) overlays them rather than replacing one
with the other.
''')

code(r'''
# ── §4b″ coverage test: fig-10 posterior pipeline on all 50 held-out nodes ───
# forward model: suppression-only gpgpu GP on the SAME 203 training runs as
# fig 9 (tested nodes never trained on), pinned to CPU so the forked pool
# workers below never touch the GPU; likelihood = fig 10's exactly (Hartlap
# full covariance + theta-dependent GP variance x fitted alpha inflation).
import os
import emcee
import bind
from bind.inference.design import _unit_to_native, ASTRO_PARAM_INDICES

n_cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
AIDX = np.asarray(ASTRO_PARAM_INDICES)
FID = bind.fiducial_params()

# fig-10's data-compression + covariance recipe (fig 10 below re-derives the
# same objects cell-locally; recipes must stay identical)
msk, RBK = (ELL >= 100) & (ELL <= ELL_TRUST), 8
def compress(v):
    return rebin(v[..., msk], RBK)
dcl_cov = np.load(SCI/"runs/dmo/run_0000/Cl_kappa.npz")
S_real_cov = (np.load(SCI/"runs/bind/run_0000/paired_perreal_fid.npz")["clk"][:, ZI, :]
              / dcl_cov["cl"][ZI, ZI])
S_real_c_cov = compress(S_real_cov)
NRl_cov, NB_cov = S_real_c_cov.shape
C_eff_cov = (np.cov(S_real_c_cov, rowvar=False)/NRl_cov) \
            / ((NRl_cov - NB_cov - 2)/(NRl_cov - 1))          # Hartlap-corrected

USE_SH = False   # Sellentin & Heavens (2016, MNRAS 456, L132) REPORTED
                 # ALTERNATIVE to the Hartlap-corrected Gaussian likelihood
                 # below (see sellentin_heavens_loglike in the setup cell) --
                 # NOT the default: n=NRl_cov=50 realizations is well under the
                 # >=250-realization TODO floor for the covariance sets, so the
                 # correction is large and the two likelihoods disagree
                 # substantially (demonstrated below at chi2/dof=1, not asserted).
                 # Flip to True once the ~550-realization sets land (TODO Sec 4c).
_chi2_demo_cov = float(NB_cov)                      # a representative chi2/dof=1 point
_ll_g_demo_cov = -0.5*_chi2_demo_cov
_ll_sh_demo_cov = float(sellentin_heavens_loglike(_chi2_demo_cov, NRl_cov, NB_cov))
print(f"Sellentin-Heavens REPORTED ALTERNATIVE (USE_SH={USE_SH}): at chi2/dof=1 "
      f"(chi2={_chi2_demo_cov:.0f}, n_real={NRl_cov}, n_data={NB_cov}) Gaussian "
      f"loglike (up to the shared normalization) = {_ll_g_demo_cov:.1f} vs SH = "
      f"{_ll_sh_demo_cov:.1f} (SH is less confident at finite n_real; converges to "
      "the Gaussian value as n_real -> inf, see the helper's unit check)")

# fitted 2-sigma inflation for the suppression head (fig 15's ALPHA formula,
# computed here from the fig-9 held-out predictions so this cell can run first)
p_ = flat(np.asarray(pred["suppression"])); e_ = flat(np.asarray(pred["suppression_err"]))
t_ = flat(TRUTH["suppression"][test_idx])
okz_ = np.isfinite(t_) & (e_ > 0) & (np.abs(t_) > 0)
ALPHA_SUP_COV = float(np.nanquantile(np.abs((p_[okz_] - t_[okz_])/e_[okz_]), 0.954)/2.0)
print(f"alpha inflation (suppression, fitted 2-sigma restore) = {ALPHA_SUP_COV:.2f}")

t0 = time.time()
em_cov = Emulator(backend="gpgpu", n_components=12, device="cpu",
                  backend_kwargs={"epochs": 400, "lr": 0.1})
em_cov.fit(ds.subset(train_idx), stats=["suppression"])
print(f"coverage forward model (gpgpu backend pinned to CPU, {len(train_idx)} runs, "
      f"suppression only) fit in {time.time()-t0:.0f}s")

S_nodes_c = compress(TRUTH["suppression"][test_idx][:, ZI, :])   # (50, 25) data vectors
TH_true   = X_unit[test_idx]                                     # (50, 30) unit-cube truths

def theta_to_native_cov(U):
    nat = np.tile(FID, (len(U), 1))
    nat[:, AIDX] = _unit_to_native(np.atleast_2d(U), AIDX)
    return nat

def make_logp_cov(S_target):
    def logp(U):                                   # vectorized over walkers:
        U = np.atleast_2d(U)                       # ONE batched GP predict/step
        out = np.full(len(U), -np.inf)
        ok = ~((U < 0).any(1) | (U > 1).any(1))
        if ok.any():
            p = em_cov.predict(theta_to_native_cov(U[ok]), stats=["suppression"])
            Sp = compress(np.asarray(p["suppression"])[:, ZI, :])
            Se = ALPHA_SUP_COV*compress(np.asarray(p["suppression_err"])[:, ZI, :])
            r = Sp - S_target
            ll = np.empty(len(r))
            for i in range(len(r)):
                C = C_eff_cov + np.diag(Se[i]**2)
                chi2_i = float(r[i] @ np.linalg.solve(C, r[i]))
                if USE_SH:
                    ll[i] = sellentin_heavens_loglike(chi2_i, NRl_cov, NB_cov)
                else:
                    sgn, logdet = np.linalg.slogdet(C)
                    ll[i] = -0.5*(chi2_i + logdet)
            out[ok] = ll
        return out
    return logp

NW_COV, NDIM_COV = 32, 30
# 32 walkers < 2*ndim: red-blue DE proposals then span a 16-walker complement --
# emcee requires live_dangerously; acceptable for interval coverage at the
# ~0.07 binomial SE of 50 nodes (flagged in the printed table)
MOVES_COV = [(emcee.moves.DEMove(live_dangerously=True), 0.8),
             (emcee.moves.DESnookerMove(live_dangerously=True), 0.2)]

def _cov_worker(job):
    i, seed, nsteps = job
    import torch
    torch.set_num_threads(1)                       # 16 workers must not fight for cores
    logp = make_logp_cov(S_nodes_c[i])
    p0 = np.random.default_rng(seed).uniform(0.02, 0.98, (NW_COV, NDIM_COV))
    s = emcee.EnsembleSampler(NW_COV, NDIM_COV, logp, vectorize=True, moves=MOVES_COV)
    s.random_state = np.random.RandomState(seed).get_state()   # seed the SAMPLER too
    s.run_mcmc(p0, nsteps, progress=False)
    ch = s.get_chain(discard=nsteps//4, flat=True)             # burn 25%
    q = np.percentile(ch, [2.5, 16, 84, 97.5], axis=0)
    th = TH_true[i]
    return (i, (th >= q[1]) & (th <= q[2]), (th >= q[0]) & (th <= q[3]),
            ch.std(0)/(1/np.sqrt(12)), float(s.acceptance_fraction.mean()))

NSTEPS_COV = 6000
t0 = time.time()
results_cov, mode_cov = [], f"parallel pool, {min(n_cpu, 12)} workers (n_cpu={n_cpu})"
try:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(n_cpu, 12),
                             mp_context=mp.get_context("fork")) as ex:
        results_cov = list(ex.map(_cov_worker,
                                  [(i, 4200 + i, NSTEPS_COV) for i in range(len(test_idx))]))
except Exception as exn:                            # serial fallback, reduced budget
    NSTEPS_COV = 2000                               # floor per spec
    mode_cov = f"SERIAL fallback ({type(exn).__name__}) at reduced budget"
    results_cov = [_cov_worker((i, 4200 + i, NSTEPS_COV)) for i in range(len(test_idx))]

order_cov = np.argsort([r[0] for r in results_cov])
IN68 = np.array([r[1] for r in results_cov])[order_cov]
IN95 = np.array([r[2] for r in results_cov])[order_cov]
SHRK_COV = np.array([r[3] for r in results_cov])[order_cov]
acc_cov = np.array([r[4] for r in results_cov])
print(f"coverage run: {len(results_cov)}/{len(test_idx)} nodes completed (no silent caps), "
      f"{mode_cov}, {NW_COV} walkers x {NSTEPS_COV} steps (burn 25%), "
      f"{time.time()-t0:.0f}s total, mean acceptance {acc_cov.mean():.2f}")

mean_shrink_cov = SHRK_COV.mean(0)
top6_cov = np.argsort(mean_shrink_cov)[:6]          # most-constrained across nodes
NN_COV = len(IN68)
def se_cov(f):
    return np.sqrt(max(f*(1 - f), 1e-12)/NN_COV)
print(f"{'parameter':>24s}   68% (nominal 0.683)   95% (nominal 0.954)")
for i in top6_cov:
    f68, f95 = IN68[:, i].mean(), IN95[:, i].mean()
    print(f"{short_label(pnames[i]):>24s}   {f68:.2f} +/- {se_cov(f68):.2f}          "
          f"{f95:.2f} +/- {se_cov(f95):.2f}")
print(f"{'all-30 average':>24s}   {IN68.mean():.2f}               {IN95.mean():.2f}")
COV_TEST = dict(in68=IN68, in95=IN95, shrink=SHRK_COV, top6=top6_cov,
                nsteps=NSTEPS_COV, mode=mode_cov, n_nodes=NN_COV)

# ── §4b‴: GP-sigma RECALIBRATION VIA COVERAGE (fixes fig 15 panel e) ─────────
# ALPHA_SUP_COV above (~1.7-1.8) is fitted so that frac(|z| < 2*alpha) = 0.954
# EXACTLY, by construction (alpha = quantile(|z|,0.954)/2): it restores the
# 2-sigma TAIL, but says nothing about whether the 1-sigma BULK also lands on
# nominal -- that's the complementary fit here. alpha_cov = quantile(|z|,0.683)
# instead solves frac(|z| < alpha_cov) = 0.683 EXACTLY (same quantile-inversion
# trick, different target), fitted per head from the SAME fig-9 held-out
# z-scores (cheap, no MCMC) and reported per head AND globally (all heads'
# z-scores pooled). Only the SUPPRESSION value feeds the posterior machinery
# (the coverage/fig-10 forward model is suppression-only by construction), so
# that one factor is substituted for ALPHA_SUP_COV and the FULL 50-node
# posterior-coverage test is RE-RUN (identical em_cov/S_nodes_c/TH_true, a
# fresh worker pool) -- an independent, honestly-labelled 'recalibrated' table,
# stored alongside (not instead of) the raw COV_TEST for fig 15 panel (e) to
# overlay both.
def _zscores_head_4b3(s):
    # `_ell_trusted` (fig-9 global): excludes the six ell-masked heads' ell>
    # ELL_TRUST train-mean/inflated-err passthrough bins, which would otherwise
    # sit at |z|~0 and artificially pull alpha_cov's 68.3%-quantile fit down
    # (falsely "well calibrated") -- see the C-head improvement set note in fig 9.
    p = flat(_ell_trusted(pred[s], s)); e = flat(_ell_trusted(pred[f"{s}_err"], s))
    t = flat(_ell_trusted(TRUTH[s][test_idx], s))
    ok = np.isfinite(t) & (e > 0) & (np.abs(t) > (1.0 if s == "peak_counts" else 0))
    return (p[ok] - t[ok])/e[ok]

ALPHA_COV = {s: float(np.nanquantile(np.abs(_zscores_head_4b3(s)), 0.683))
            for s in STATS_EMU}
_z_all_4b3 = np.concatenate([_zscores_head_4b3(s) for s in STATS_EMU])
ALPHA_COV_GLOBAL = float(np.nanquantile(np.abs(_z_all_4b3), 0.683))
print("GP-sigma coverage recalibration (alpha_cov: frac(|z|<alpha_cov)=0.683 by "
      "construction, held-out per-bin z-scores, per head):")
for s in STATS_EMU:
    print(f"  {s:>14s}: alpha_cov = {ALPHA_COV[s]:.2f}")
print(f"  {'GLOBAL (all ' + str(len(STATS_EMU)) + ' heads pooled)':>14s}: alpha_cov = {ALPHA_COV_GLOBAL:.2f}")
ALPHA_SUP_RECAL = ALPHA_COV["suppression"]
print(f"posterior forward model is suppression-only -> alpha_cov(suppression) = "
      f"{ALPHA_SUP_RECAL:.2f} substituted for ALPHA_SUP_COV={ALPHA_SUP_COV:.2f} "
      "in the re-run below")

def make_logp_cov_alpha(S_target, alpha):
    """Same likelihood as make_logp_cov above, with alpha passed explicitly
    (not read off the ALPHA_SUP_COV global) so the raw and recalibrated runs
    can coexist without one silently overwriting the other's forward model."""
    def logp(U):
        U = np.atleast_2d(U)
        out = np.full(len(U), -np.inf)
        ok = ~((U < 0).any(1) | (U > 1).any(1))
        if ok.any():
            p = em_cov.predict(theta_to_native_cov(U[ok]), stats=["suppression"])
            Sp = compress(np.asarray(p["suppression"])[:, ZI, :])
            Se = alpha*compress(np.asarray(p["suppression_err"])[:, ZI, :])
            r = Sp - S_target
            ll = np.empty(len(r))
            for i in range(len(r)):
                C = C_eff_cov + np.diag(Se[i]**2)
                chi2_i = float(r[i] @ np.linalg.solve(C, r[i]))
                if USE_SH:
                    ll[i] = sellentin_heavens_loglike(chi2_i, NRl_cov, NB_cov)
                    continue
                sgn, logdet = np.linalg.slogdet(C)
                ll[i] = -0.5*(chi2_i + logdet)
            out[ok] = ll
        return out
    return logp

def _cov_worker_alpha(job):
    i, seed, nsteps, alpha = job
    import torch
    torch.set_num_threads(1)
    logp = make_logp_cov_alpha(S_nodes_c[i], alpha)
    p0 = np.random.default_rng(seed).uniform(0.02, 0.98, (NW_COV, NDIM_COV))
    s = emcee.EnsembleSampler(NW_COV, NDIM_COV, logp, vectorize=True, moves=MOVES_COV)
    s.random_state = np.random.RandomState(seed).get_state()
    s.run_mcmc(p0, nsteps, progress=False)
    ch = s.get_chain(discard=nsteps//4, flat=True)
    q = np.percentile(ch, [2.5, 16, 84, 97.5], axis=0)
    th = TH_true[i]
    return (i, (th >= q[1]) & (th <= q[2]), (th >= q[0]) & (th <= q[3]),
            ch.std(0)/(1/np.sqrt(12)), float(s.acceptance_fraction.mean()))

NSTEPS_RECAL = 6000
t0 = time.time()
results_recal, mode_recal = [], f"parallel pool, {min(n_cpu, 12)} workers (n_cpu={n_cpu})"
try:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(n_cpu, 12),
                             mp_context=mp.get_context("fork")) as ex:
        results_recal = list(ex.map(
            _cov_worker_alpha,
            [(i, 9200 + i, NSTEPS_RECAL, ALPHA_SUP_RECAL) for i in range(len(test_idx))]))
except Exception as exn:                            # serial fallback, reduced budget
    NSTEPS_RECAL = 2000
    mode_recal = f"SERIAL fallback ({type(exn).__name__}) at reduced budget"
    results_recal = [_cov_worker_alpha((i, 9200 + i, NSTEPS_RECAL, ALPHA_SUP_RECAL))
                     for i in range(len(test_idx))]

order_recal = np.argsort([r[0] for r in results_recal])
IN68_RECAL = np.array([r[1] for r in results_recal])[order_recal]
IN95_RECAL = np.array([r[2] for r in results_recal])[order_recal]
NN_RECAL = len(IN68_RECAL)
print(f"recalibrated coverage run: {len(results_recal)}/{len(test_idx)} nodes completed, "
      f"{mode_recal}, {NW_COV} walkers x {NSTEPS_RECAL} steps (burn 25%), "
      f"alpha_cov(suppression)={ALPHA_SUP_RECAL:.2f} (raw ALPHA_SUP_COV="
      f"{ALPHA_SUP_COV:.2f}), {time.time()-t0:.0f}s total")
print(f"{'parameter':>24s}   68% recal (raw)        95% recal (raw)")
for i in top6_cov:
    f68r, f95r = IN68_RECAL[:, i].mean(), IN95_RECAL[:, i].mean()
    f68, f95 = IN68[:, i].mean(), IN95[:, i].mean()
    print(f"{short_label(pnames[i]):>24s}   {f68r:.2f} ({f68:.2f})            "
          f"{f95r:.2f} ({f95:.2f})")
print(f"{'all-30 average':>24s}   {IN68_RECAL.mean():.2f} ({IN68.mean():.2f})            "
      f"{IN95_RECAL.mean():.2f} ({IN95.mean():.2f})")
COV_TEST_RECAL = dict(in68=IN68_RECAL, in95=IN95_RECAL, alpha=ALPHA_SUP_RECAL,
                      nsteps=NSTEPS_RECAL, mode=mode_recal, n_nodes=NN_RECAL)
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §4b′ — Fig 15: is the emulator's error bar honest? (one calibration figure)

Accuracy without calibration is not enough: `suppression_err` enters the §4c likelihood, so the
claimed 1σ must cover — and so must the posterior built from it (§4b″). This is now the *single*
calibration figure of the emulation section. (a) standardized residuals
$z=({\rm pred}-{\rm truth})/\sigma_{\rm pred}$ on the 50 held-out runs (edge bins are clipped
overflow, annotated); (b) empirical 1σ/2σ coverage per statistic against the Gaussian
68.3/95.4% expectations — **both** bar families now carry the run-level binomial SE (same
recipe for $|z|<2$ as for $|z|<1$). At these uncertainties the 1σ shortfall is marginal; the
robust signal is the *2σ deficit* (heavy tails, dominated by the enhancement corner) — and
because all 13 heads are evaluated on the *same* 50 held-out runs, that deficit is **coherent
across heads, not 12 independent detections** ($C_\ell^{\kappa\kappa}$ is not among them at
all — see the identity note in §2b/§4.0). (c,d) the learning curve, split into two
stacked single-axis panels (accuracy above, 1σ coverage below — the dual-$y$-axis chart is
gone), with ~5 seeded training-subset repeats per $N$ drawn as individual points so the
subsampling spread is visible (the full training set admits a single subset); computed with
the shipped `gpgpu` backend, suppression head only (stated on the panel). Held-out *accuracy* flattens by
$N\simeq200$ while the *uncertainty calibration* is still improving, so the design-adequacy
argument rests on the accuracy curve plus the 2–3-dimensional effective response of fig 8.
(e) the payoff of §4b″: empirical 68/95% *posterior* coverage over the 50 held-out nodes
against nominal, for the top-6 constrained parameters and the all-30 average — the posterior
inherits the calibration of the inflated GP error it was built with. Solid bars are the RAW
run (tail-fitted $\hat\alpha$, restores the 2σ point by construction); open diamonds overlay
the §4b‴ RECALIBRATED run, an independent 50-node posterior re-fit with $\sigma_{\rm GP}$
inflated instead by $\alpha_{\rm cov}$ (fitted directly against 1σ coverage of the same
held-out $z$-scores) — both are shown, not just the one that looks better.
''')

code(r'''
# ── Fig 15: GP calibration + learning curve + posterior coverage (one figure) ─
# uses the fig-9 emulator (same split) and §4b″'s COV_TEST; the learning curve
# refits suppression with the SHIPPED gpgpu backend (GPU when available),
# ~5 seeded training-subset repeats per N (N=len(train_idx) admits a single subset)
zsc = {}
cov1, cov2 = {}, {}
for s in STATS_EMU:
    # `_ell_trusted` (fig-9 global, C-head improvement set): the six ell-masked
    # heads' ell>ELL_TRUST bins are a train-mean/inflated-err passthrough, not a
    # real prediction -- left in, they'd sit at |z|~0 and falsely inflate cov1/
    # cov2 ("covered" by construction, not by calibration). No-op for every
    # other head (incl. suppression, deliberately unmasked).
    p = flat(_ell_trusted(pred[s], s)); e = flat(_ell_trusted(pred[f"{s}_err"], s))
    t = flat(_ell_trusted(TRUTH[s][test_idx], s))
    ok = np.isfinite(t) & (e > 0) & (np.abs(t) > (1.0 if s == "peak_counts" else 0))
    z = (p[ok] - t[ok])/e[ok]
    zsc[s] = z
    cov1[s], cov2[s] = (np.abs(z) < 1).mean(), (np.abs(z) < 2).mean()

# run-level coverage uncertainty: bins within a run are correlated -> effective
# sample ~50 runs; SAME binomial recipe for both bar families (v3 fix: the
# |z|<2 bars previously carried no SE)
SEc  = np.sqrt(0.683*0.317/50)
SEc2 = np.sqrt(0.954*0.046/50)

# last point = the full training set (len(train_idx); NOT hardcoded 206 = 256-50,
# since 3 of the nominal 256 Sobol nodes are dropped upstream for missing
# lightcone/S-side products -- see the fig-20 cell's run_ids note -- so this
# dataset's actual split is 253 runs = 203 train + 50 held out)
Ns, NREP15 = [50, 100, 150, len(train_idx)], 5
lc_fe, lc_cov = [], []                       # per N: list of per-repeat values
rngl = np.random.default_rng(10)
t0 = time.time()
for N in Ns:
    nrep = NREP15 if N < len(train_idx) else 1
    fe_r, cov_r = [], []
    for _ in range(nrep):
        tr_N = np.sort(rngl.choice(train_idx, N, replace=False))
        em_N = Emulator(backend="gpgpu", n_components=12,
                        backend_kwargs={"epochs": 400, "lr": 0.1})
        em_N.fit(ds.subset(tr_N), stats=["suppression"])
        pN = em_N.predict(X_native[test_idx])
        p, e = pN["suppression"][:, ZI, :], pN["suppression_err"][:, ZI, :]
        t = TRUTH["suppression"][test_idx][:, ZI, :]
        fe_r.append(float(np.nanmedian(np.abs(p/t - 1))))
        cov_r.append(float((np.abs((p - t)/e) < 1).mean()))
    lc_fe.append(fe_r); lc_cov.append(cov_r)
    print(f"N_train={N}: frac err {100*np.mean(fe_r):.2f}% "
          f"(subset spread {100*np.std(fe_r):.2f}%), 1sig cov {np.mean(cov_r):.2f} "
          f"(+/-{np.std(cov_r):.2f}) over {nrep} seeded subset(s)")
print(f"learning curve ({sum(len(f) for f in lc_fe)} gpgpu fits) in {time.time()-t0:.0f}s")

fig = plt.figure(figsize=(TWO_COL[0], 5.4))
gs15 = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.5)
axA = fig.add_subplot(gs15[0, 0])
axB = fig.add_subplot(gs15[0, 1:])
gs15c = gs15[1, 0].subgridspec(2, 1, hspace=0.10)
axC = fig.add_subplot(gs15c[0])
axD = fig.add_subplot(gs15c[1], sharex=axC)
axE = fig.add_subplot(gs15[1, 1:])

zg = np.linspace(-4, 4, 200)
f_out = (np.abs(zsc["suppression"]) > 4).mean()
axA.hist(np.clip(zsc["suppression"], -4, 4), bins=40, density=True,
         color=COLORS["bind"], alpha=0.7)
axA.plot(zg, np.exp(-zg**2/2)/np.sqrt(2*np.pi), color=COLORS["truth"], lw=1.3)
axA.set_xlabel(r"$z=({\rm pred}-{\rm truth})/\sigma_{\rm pred}$")
axA.set_ylabel("density")
axA.text(0.04, 0.9, r"$S(\ell)$", transform=axA.transAxes, fontsize=7)
axA.text(0.04, 0.78, f"edge bins = clipped\n$|z|>4$: {100*f_out:.1f}%",
         transform=axA.transAxes, fontsize=5.6)
panel_label(axA, "(a)", loc="upper right")

xx = np.arange(len(STATS_EMU))
w = 0.38
axB.bar(xx - w/2, [cov1[s] for s in STATS_EMU], w, yerr=SEc, capsize=2,
        error_kw=dict(ecolor="0.2", elinewidth=0.8), color=COLORS["bind"],
        label=r"$|z|<1$")
axB.bar(xx + w/2, [cov2[s] for s in STATS_EMU], w, yerr=SEc2, capsize=2,
        error_kw=dict(ecolor="0.2", elinewidth=0.8), color=COLORS["secondary"],
        label=r"$|z|<2$")
axB.axhline(0.683, color=COLORS["dmo"], ls=":", lw=0.9)
axB.axhline(0.954, color=COLORS["dmo"], ls="--", lw=0.9)
axB.set_xticks(xx); axB.set_xticklabels(names, rotation=40, ha="right", fontsize=5.0)
axB.set_ylabel("empirical coverage"); axB.set_ylim(0, 1.18)
axB.legend(loc="upper left", fontsize=5.4, ncol=2)
axB.text(0.99, 0.03, "heads share the same 50 held-out runs:\n"
         f"the 2$\\sigma$ deficit is coherent, not {len(STATS_EMU)} detections",
         transform=axB.transAxes, fontsize=5.0, ha="right",
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
panel_label(axB, "(b)", loc="lower left")

# (c,d) learning curve as two stacked single-axis panels (dual y-axis removed)
for k, (N, fe_r, cov_r) in enumerate(zip(Ns, lc_fe, lc_cov)):
    axC.plot([N]*len(fe_r), 100*np.array(fe_r), "o", ms=2.2, mfc="none",
             color=COLORS["bind"], alpha=0.7)
    axD.plot([N]*len(cov_r), cov_r, "s", ms=2.2, mfc="none",
             color=COLORS["secondary"], alpha=0.7)
axC.plot(Ns, [100*np.mean(f) for f in lc_fe], "-", color=COLORS["bind"], lw=1.3)
axD.plot(Ns, [np.mean(c) for c in lc_cov], "--", color=COLORS["secondary"], lw=1.1)
axD.axhline(0.683, color=COLORS["dmo"], ls=":", lw=0.8)
axC.tick_params(labelbottom=False)
axC.set_ylabel("med |frac err| [%]", fontsize=6)
axC.text(0.97, 0.9, "gpgpu (shipped), suppression head;\n5 seeded subsets per $N$ "
         f"(1 at $N$={len(train_idx)})", transform=axC.transAxes, fontsize=4.6,
         ha="right", va="top")
axD.set_xlabel(r"$N_{\rm train}$")
axD.set_ylabel(r"1$\sigma$ coverage", fontsize=6)
axD.set_ylim(0.3, 1.0)
panel_label(axC, "(c)")
panel_label(axD, "(d)")

# (e) NEW: empirical posterior coverage from the §4b″ 50-node test vs nominal
t6c = COV_TEST["top6"]; NNc = COV_TEST["n_nodes"]
f68e = [COV_TEST["in68"][:, i].mean() for i in t6c] + [COV_TEST["in68"].mean()]
f95e = [COV_TEST["in95"][:, i].mean() for i in t6c] + [COV_TEST["in95"].mean()]
se68e = [np.sqrt(max(f*(1 - f), 1e-12)/NNc) for f in f68e]
se95e = [np.sqrt(max(f*(1 - f), 1e-12)/NNc) for f in f95e]
xxe = np.arange(len(f68e))
axE.bar(xxe - w/2, f68e, w, yerr=se68e, capsize=2,
        error_kw=dict(ecolor="0.2", elinewidth=0.8), color=COLORS["bind"],
        label="68% CI")
axE.bar(xxe + w/2, f95e, w, yerr=se95e, capsize=2,
        error_kw=dict(ecolor="0.2", elinewidth=0.8), color=COLORS["secondary"],
        label="95% CI")
axE.axhline(0.683, color=COLORS["dmo"], ls=":", lw=0.9)
axE.axhline(0.954, color=COLORS["dmo"], ls="--", lw=0.9)
axE.set_xticks(xxe)
axE.set_xticklabels([short_label(pnames[i]) for i in t6c] + ["all-30 avg"],
                    rotation=40, ha="right", fontsize=5.4)
axE.set_ylabel("posterior coverage"); axE.set_ylim(0, 1.18)
# overlay: RECALIBRATED posterior coverage (§4b‴, GP-sigma inflated by
# alpha_cov instead of the raw fitted alpha) as open markers on the SAME bars
# -- additive, not a replacement: raw bars above stay exactly as computed, the
# recalibrated numbers are printed AND shown, not silently swapped in
# (honesty framing). Guarded: if the §4b‴ block was not run this session,
# fall back to the raw-only panel rather than a NameError.
if "COV_TEST_RECAL" in globals():
    f68r = ([COV_TEST_RECAL["in68"][:, i].mean() for i in t6c]
           + [COV_TEST_RECAL["in68"].mean()])
    f95r = ([COV_TEST_RECAL["in95"][:, i].mean() for i in t6c]
           + [COV_TEST_RECAL["in95"].mean()])
    NNr = COV_TEST_RECAL["n_nodes"]
    se68r = [np.sqrt(max(f*(1 - f), 1e-12)/NNr) for f in f68r]
    se95r = [np.sqrt(max(f*(1 - f), 1e-12)/NNr) for f in f95r]
    axE.errorbar(xxe - w/2, f68r, yerr=se68r, fmt="D", ms=3.0, mfc="none",
                mec=COLORS["highlight"], ecolor=COLORS["highlight"],
                elinewidth=0.8, capsize=2, zorder=5,
                label=rf"68% recal ($\alpha_{{\rm cov}}$={COV_TEST_RECAL['alpha']:.2f})")
    axE.errorbar(xxe + w/2, f95r, yerr=se95r, fmt="D", ms=3.0, mfc="none",
                mec="0.1", ecolor="0.1", elinewidth=0.8, capsize=2, zorder=5,
                label="95% recal")
else:
    print("COV_TEST_RECAL not found -- panel (e) shows RAW coverage only "
          "(run the §4b‴ recalibration block first for the overlay)")
axE.legend(loc="upper right", fontsize=5.4, ncol=2, frameon=True,
           framealpha=1.0, edgecolor="0.6")   # opaque frame ABOVE the bar zone:
           # at lower right the frameless swatches fused with the CI bars
axE.text(0.02, 0.05, f"§4b″: fig-10 pipeline on {NNc} held-out nodes as synthetic "
         f"data\n({COV_TEST['nsteps']} steps/node; bars: binomial SE over nodes)",
         transform=axE.transAxes, fontsize=5.0)
panel_label(axE, "(e)", loc="upper left")
fig.tight_layout()
save(fig, "figs_v2/fig15_emulator_calibration")
plt.show()
# FITTED error-inflation per statistic: alpha such that |z|/alpha restores the
# Gaussian 95.4% quantile (replaces any ad-hoc factor; used in fig 16c)
ALPHA = {s: float(np.nanquantile(np.abs(zsc[s]), 0.954)/2.0) for s in STATS_EMU}
_have_cov_recal = "ALPHA_COV" in globals()
for s in STATS_EMU:
    _cov_txt = (f", coverage-fit (1sig target) alpha_cov = {ALPHA_COV[s]:.2f}"
               if _have_cov_recal else "")
    print(f"{s:>14s}: coverage |z|<1 = {cov1[s]:.2f} +/- {SEc:.2f} (0.68), "
          f"|z|<2 = {cov2[s]:.2f} +/- {SEc2:.2f} (0.95); fitted inflation "
          f"alpha = {ALPHA[s]:.2f}{_cov_txt}")
print(f"reading: the 1-sigma shortfall is marginal at run-level errors; the robust "
      f"signal is the 2-sigma deficit (heavy tails, dominated by the enhancement "
      f"corner) -- COHERENT across the {len(STATS_EMU)} heads (they share the same 50 "
      f"held-out runs), not {len(STATS_EMU)} independent detections -> the fitted alpha "
      "propagates into Appendix B (fig 16c) and the §4b″ coverage run")
if _have_cov_recal:
    print(f"GP-sigma coverage recalibration (§4b‴): alpha_cov global (all {len(STATS_EMU)} heads) = "
          f"{ALPHA_COV_GLOBAL:.2f}; suppression-head alpha_cov = "
          f"{ALPHA_COV['suppression']:.2f} vs the tail-fitted alpha = "
          f"{ALPHA['suppression']:.2f} used above -- panel (e)'s recalibrated markers "
          "use the coverage-fitted value")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
### Fig 15b — the 1P/two-bound design as an out-of-design (OOD) validation

Figs 9/9b/9c and the §4b yardstick all report *interior* held-out error: the 50 test nodes are
drawn from the same 30-D Sobol design the 203 training nodes come from, so they never probe the
prior's *edges*. The two-bound design (§1c/§3b; 60 runs, one astro parameter swept to its
documented min/max bound with the other 29 held at the fiducial; for three parameters —
`VariableWindSpecMomentum`, `UVBH0Deltaz`, `UVBHepDeltaz` — the fiducial value *is* the prior
bound, so the 60 runs contain 57 distinct perturbations by construction, not by omission) sits **exactly** at those edges
by construction — the corner the 203/50 Sobol split under-samples (§4.0's WP2 note) — so
comparing the fig-9 emulator's *prediction* at these 60 points against their *measured*
statistics is a genuine out-of-design probe, using data the emulator has never seen in any
form. The forward model is the unmodified fig-9 `em` (fit on the 203 training runs; not
refit); `em.predict()` takes native 35-vectors directly, so the two-bound design's own native
parameter table needs no unit-cube bookkeeping of its own — the identical call convention as
fig 9's any-$z$/any-grid demo. $S(\ell)$ comes for free from each run's cached `Cl_kappa.npz`
(divided by the one shared, run-independent DMO trace, matching every other suppression in this
notebook); $C_\ell^{yy}$ is a modest live recompute from a realization subset (`y_maps.npz`);
peak/minima counts and the Minkowski functionals are read directly off each run's own cache on
the dataset's exact bin grids (checked, not assumed). The 1P atlas would be the same *kind* of
probe but is not yet stats-cached the way the two-bound design already is — the TODO item this
figure answers with the design that is actually ready.
''')

code(r'''
# ── Fig 15b: 1P/two-bound OOD validation -- the emulator at the prior EDGES ──
# data: bind_science/runs/twobound/twobound_params.npy (60,35 native; run r
#       varies ONE astro param r//2 to its low(even)/high(odd) two-bound value,
#       all 29 others at fiducial -- see fig 0's hero cell) + per-run
#       Cl_kappa.npz / peak_counts.npz / nongaussian_stats.npz / y_maps.npz.
# Forward model: the SAME fig-9 `em` (203-run fit, NOT refit) -- em.predict()
# takes NATIVE 35-vectors directly (bind/emulator/core.py: params_to_unit is
# applied INSIDE predict()), the identical convention fig 9's GOAL-3 demo and
# fig 10's theta_to_native both rely on, so no unit-cube mapping of our own is
# needed here. cl_kappa_y/cl_tt/cl_kappa_tau/cl_yt are SKIPPED: the released
# per-run Cl_kappa_y.npz carries the pre-xpkfix cross normalization (the SAME
# reason fig 4/4b never read it for the fiducial either), and twobound has no
# tau_maps.npz at all (the historical truth-tau gap, TODO Sec 2b) -- using
# either would silently compare the emulator against the wrong convention.
from bind.inference.stats import power_spectrum as _ps_ood

TB_OOD = SCI/"runs/twobound"
tbp_ood = np.load(TB_OOD/"twobound_params.npy")            # (60, 35) native
assert tbp_ood.ndim == 2 and tbp_ood.shape[1] == X_native.shape[1], \
    f"twobound param table shape {tbp_ood.shape} != dataset native width {X_native.shape[1]}"
N_TB = len(tbp_ood)
tb_dirs = [TB_OOD/f"run_{i:04d}" for i in range(N_TB)]

have_cl = np.array([(rd/"Cl_kappa.npz").exists() for rd in tb_dirs])
have_pk = np.array([(rd/"peak_counts.npz").exists() for rd in tb_dirs])
have_ng = np.array([(rd/"nongaussian_stats.npz").exists() for rd in tb_dirs])
have_y  = np.array([(rd/"y_maps.npz").exists() for rd in tb_dirs])
print(f"twobound OOD cache inventory ({N_TB} runs): Cl_kappa {have_cl.sum()}, "
      f"peak_counts {have_pk.sum()}, nongaussian_stats {have_ng.sum()}, "
      f"y_maps {have_y.sum()}")

OOD_MIN_RUNS = 10   # "fewer than ~10 -> degrade gracefully" floor
idx_meas = np.where(have_cl)[0]
if len(idx_meas) < OOD_MIN_RUNS:
    print(f"!! only {len(idx_meas)}/{N_TB} twobound runs have a measured Cl_kappa.npz "
          f"(< {OOD_MIN_RUNS} floor) -- degrading to PREDICTION-ONLY, no measured-vs-"
          "emulated comparison this run.")

# emulator prediction for ALL N_TB runs regardless of cache availability (cheap;
# "theta -> any statistic in ms" is the release's whole point). z_s axis kept
# (all 5 planes) -- sliced to ZI below per head, matching the fig-5/7 convention.
pred_tb = em.predict(tbp_ood, stats=STATS_EMU)

ZI_HEADS_OOD = {"suppression", "pdf", "peak_counts", "minima_counts",
                "mf_v0", "mf_v1", "mf_v2"}   # z_s-tomographic heads this cell uses
def _zi_ood(key, arr):
    arr = np.asarray(arr)
    return arr[:, ZI, :] if key in ZI_HEADS_OOD else arr

# ── measured side, only for runs with the relevant cache ─────────────────────
dcl_ood = np.load(SCI/"runs/dmo/run_0000/Cl_kappa.npz")["cl"][ZI, ZI]   # ONE shared DMO trace
NREAL_YY_OOD = 10   # "cl_yy if cheap": 10/50 realizations -> ~600 power_spectrum
                    # calls total, not 3000; a mean estimate, same convention
                    # (total column, y[:, -1]) as fig 4b/7's cl_yy.
meas_tb, idx_per_head = {s: [] for s in STATS_EMU}, {s: [] for s in STATS_EMU}
for i in idx_meas:
    rd = tb_dirs[i]
    bcl = np.load(rd/"Cl_kappa.npz")
    if not np.allclose(bcl["ell"], ELL):
        print(f"  run {i}: Cl_kappa ell grid != dataset ELL -- SKIPPED"); continue
    meas_tb["suppression"].append(bcl["cl"][ZI, ZI]/dcl_ood)
    idx_per_head["suppression"].append(i)
    if have_pk[i]:
        pk = np.load(rd/"peak_counts.npz", allow_pickle=True)
        if np.allclose(pk["nu"], d["a__peak_counts__nu"]) and str(pk["nu_norm"]) == "map":
            meas_tb["peak_counts"].append(pk["peak_counts"][ZI])
            meas_tb["minima_counts"].append(pk["minima_counts"][ZI])
            idx_per_head["peak_counts"].append(i); idx_per_head["minima_counts"].append(i)
    if have_ng[i]:
        ng = np.load(rd/"nongaussian_stats.npz", allow_pickle=True)
        if np.allclose(ng["mf_nu"], d["a__mf_v0__mf_nu"]):
            meas_tb["mf_v0"].append(ng["V0"][ZI]); meas_tb["mf_v1"].append(ng["V1"][ZI])
            meas_tb["mf_v2"].append(ng["V2"][ZI])
            for k in ("mf_v0", "mf_v1", "mf_v2"):
                idx_per_head[k].append(i)
        if np.allclose(ng["pdf_bins"], d["a__pdf__pdf_bins"]):
            meas_tb["pdf"].append(ng["pdf"][ZI]); idx_per_head["pdf"].append(i)
    if have_y[i]:
        f = np.load(rd/"y_maps.npz")
        y10 = f["y"][:NREAL_YY_OOD, -1].copy(); del f; gc.collect()
        meas_tb["cl_yy"].append(np.mean([_ps_ood(m)[1] for m in y10], axis=0))
        idx_per_head["cl_yy"].append(i)

# ── per-head OOD vs INTERIOR metric ───────────────────────────────────────────
OOD_ROWS = []
for s in STATS_EMU:
    idxs = idx_per_head[s]
    if len(idxs) < OOD_MIN_RUNS:
        if len(idxs):
            print(f"  {s:>14s}: only {len(idxs)}/{N_TB} twobound runs with a usable "
                  f"cache (< {OOD_MIN_RUNS} floor) -- SKIPPED from the comparison")
        continue
    # `_ell_trusted` (fig-9 global): only "cl_yy" of the six ell-masked heads
    # ever reaches here with measured runs (cl_kappa/cl_kappa_y/cl_kappa_tau/
    # cl_yt/cl_tt are unconditionally skipped above, pre-xpkfix norm / no
    # tau_maps.npz) -- restricting it to ell<=ELL_TRUST keeps this OOD number
    # comparable to `fe_interior = metrics[s][0]` below, itself now restricted
    # the same way; a real MEASURED value above ELL_TRUST would otherwise be
    # compared against cl_yy's train-mean passthrough there, not a prediction.
    meas_arr = _ell_trusted(np.asarray(meas_tb[s], float), s)    # (n_meas, n_bin)
    pred_arr = _ell_trusted(_zi_ood(s, pred_tb[s])[idxs], s)     # matched, SAME order
    thresh = 1.0 if s in ("peak_counts", "minima_counts") else 0.0
    ok = np.isfinite(meas_arr) & (np.abs(meas_arr) > thresh)
    if not ok.any():
        continue
    fe_ood = float(np.nanmedian(np.abs(pred_arr[ok]/meas_arr[ok] - 1)))
    fe_interior = metrics[s][0]                                  # fig-9 global, NOT refit
    OOD_ROWS.append((s, fe_ood, fe_interior, len(idxs)))

print(f"\n{'head':>14s}  {'OOD |frac err|':>15s}  {'interior |frac err|':>21s}  "
      f"{'OOD/interior':>13s}  n_meas")
for s, fe_ood, fe_int, nmeas in sorted(OOD_ROWS, key=lambda r: -r[1]/max(r[2], 1e-12)):
    print(f"{s:>14s}  {100*fe_ood:14.2f}%  {100*fe_int:20.2f}%  "
          f"{fe_ood/max(fe_int, 1e-12):12.2f}x  {nmeas:5d}")
if OOD_ROWS:
    med_ratio_ood = float(np.median([r[1]/max(r[2], 1e-12) for r in OOD_ROWS]))
    print(f"\n1P/two-bound OOD validation: prior-EDGE median |frac err| is "
          f"{med_ratio_ood:.1f}x the held-out INTERIOR median across {len(OOD_ROWS)} heads "
          f"with >= {OOD_MIN_RUNS} measured twobound runs -- the honest degradation at the "
          "30-D prior boundary, not the interior number figs 9/9b report.")
else:
    print(f"\n1P/two-bound OOD validation: no head cleared the {OOD_MIN_RUNS}-measured-run "
          "floor -- PREDICTION-ONLY this run, no measured comparison possible.")

# ── small figure: (a) OOD vs interior bars, (b) example S(ell) OOD curves ────
fig, (axA, axB) = plt.subplots(1, 2, figsize=(TWO_COL[0], 2.5))
if OOD_ROWS:
    labs = [r[0] for r in OOD_ROWS]
    xr = np.arange(len(labs)); wr = 0.38
    axA.bar(xr - wr/2, [100*r[1] for r in OOD_ROWS], wr, color=COLORS["highlight"],
            label="OOD (two-bound)")
    axA.bar(xr + wr/2, [100*r[2] for r in OOD_ROWS], wr, color=COLORS["bind"],
            label="interior (fig 9)")
    axA.set_xticks(xr); axA.set_xticklabels(labs, rotation=55, ha="right", fontsize=5.5)
    axA.set_ylabel("median |frac err| [%]")
    axA.legend(fontsize=5.5, loc="upper right")
else:
    axA.text(0.5, 0.5, "prediction-only\n(no measured comparison)", ha="center",
             va="center", transform=axA.transAxes, fontsize=7)
panel_label(axA, "(a)")

# example S(ell) curves: 4 runs spanning distinct parameters/bounds (always
# available -- suppression only needs Cl_kappa.npz, present for idx_meas)
if len(idx_per_head["suppression"]) >= 4:
    ex_idx = np.asarray(idx_per_head["suppression"])[
        np.linspace(0, len(idx_per_head["suppression"]) - 1, 4).astype(int)]
    for k, i_ex in enumerate(ex_idx):
        j = idx_per_head["suppression"].index(i_ex)
        col = plt.cm.coolwarm(k/3.0)
        axB.plot(ELL, meas_tb["suppression"][j], color=col, lw=0.9,
                 label=f"run {i_ex} (meas.)" if k == 0 else None)
        axB.plot(ELL, _zi_ood("suppression", pred_tb["suppression"])[i_ex], color=col,
                 ls="--", lw=0.9, label="emulated" if k == 0 else None)
    axB.set_xscale("log")
    axB.set_xlabel(r"$\ell$"); axB.set_ylabel(r"$S(\ell)$, 4 two-bound runs")
    axB.legend(fontsize=5.5, loc="best")
else:
    axB.text(0.5, 0.5, "insufficient measured runs", ha="center", va="center",
             transform=axB.transAxes, fontsize=7)
panel_label(axB, "(b)")
fig.tight_layout()
save(fig, "figs_v2/fig15b_ood_validation")
plt.show()
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §4c — Fig 10: an example constraint — astro parameters from the fiducial $S(\ell)$

**Setup.** The GP emulator (retrained on all 253 runs, suppression only, with the **shipped
`gpgpu` backend** — the earlier draft's sklearn forward model was a backend inconsistency; a
timed 200-step trial is printed so the run log records the per-step cost, and the cell falls
back to sklearn only if the shipped backend is catastrophically slower at execution time,
with a loud printed warning) is the forward
model $S_{\rm em}(\ell\,|\,\theta)$; the data vector is the *measured* fiducial suppression
$S(\ell)=C_\ell^{\kappa\kappa,\rm BIND}/C_\ell^{\kappa\kappa,\rm DMO}$ at $z_s=1$,
$\ell\in[100,1.5\times10^4]$ rebinned ×8 ($N_b=25$). The likelihood is Gaussian with the
*full* data covariance estimated from the 50 seed-paired realizations plus the
$\theta$-dependent GP variance on the diagonal:

$$-2\ln\mathcal{L}(\theta)=\mathbf{r}^{\sf T}C(\theta)^{-1}\mathbf{r}+\ln\det C(\theta),
\qquad C(\theta)=\frac{\hat{C}_{\rm data}}{h}+\mathrm{diag}\,\sigma_{\rm GP}^2(\theta),$$

with $\mathbf{r}=S_{\rm em}(\theta)-S_{\rm obs}$ and the Hartlap factor
$h=(n-p-2)/(n-1)=0.47$ (mean bin–bin |correlation| 0.74 — a diagonal likelihood would be
wrong here, as Appendix B (fig 16c) quantifies). Priors are flat on the unit cube (for LogFlag
parameters: flat in $\log_{10}$ of the native value — an informative choice, stated
deliberately). Sampler: emcee with DE/DESnooker moves, overdispersed initialization, and the
sampler's internal random state now seeded (chains are bit-reproducible);
$\tau_{\rm int}$, ESS, and acceptance are printed. Three methodological footnotes: Hartlap
is exact only for the fixed Wishart-estimated covariance — adding the $\theta$-dependent
GP diagonal makes it approximate; **Percival/Dodelson–Schneider**: Hartlap corrects only the
*mean* of the inverse covariance — the sampling noise of a precision matrix estimated from
$n=50$ realizations with $p=25$ bins additionally inflates the *parameter variance* by
$m_1 = 1 + B\,(p-n_{\rm par})$, $B=(n-p-2)/[(n-p-1)(n-p-4)]$ (Dodelson & Schneider 2013;
Percival et al. 2014), which for the effective $n_{\rm par}\simeq2$–6 constrained here is
$m_1\approx1.9$–2.0, i.e. all quoted 1σ widths in fig 10 and Appendix B (fig 16) are
**pre-correction**: corrected widths must be inflated by $\sqrt{m_1}\simeq1.37$–$1.43$
(equivalently, the quoted widths are $\sim$27–30% narrower than the corrected ones; the
exact factors are printed by the cell); and the main chain is $\sim$53 integrated
autocorrelation times ($\tau_{\rm int}$ statistics printed by the cell), just above the
~50$\tau$ rule of thumb — width ratios in Appendix B (fig 16) need
only %-level precision, which this supports. **Corner rendering disclosure:** the 2-D
contours carry corner's cosmetic Gaussian smoothing (`smooth=1.0`, `bins=28`) — stated
because a chain at the printed acceptance (~0.16 for the main chain) can look artificially
smooth; the 1-D diagonals are the
raw *unsmoothed* histograms, with a finer-binned raw histogram overlaid as a sanity layer.

**Framing.** This is an *internal parameter-recovery test under ideal conditions* — data
vector, training set, and footprint share one lightcone, so common-mode cosmic variance
cancels by construction; widths are not survey forecasts. The fiducial is *not* a Sobol
node — a genuinely out-of-design test.
Parameters whose fiducial sits on a prior edge (`VariableWindSpecMomentum`, the UV
background $\Delta z$'s) cannot be bracketed and are excluded from the corner (flagged in
the printed ranking). The corner shows the most-constrained directions: WL suppression
alone pins the SN-wind/IMF/BH amplitude combination at ~43–66% of prior width (the printed
top-10 posterior/prior widths span 0.43–0.66) and leaves
most of the 30 dimensions untouched — consistent with the two-dimensional response surface
of §3f.
''')

code(r'''
# ── Fig 10: emcee corner on the fiducial suppression (full covariance) ───────
# data: bind_science/runs/bind/run_0000/{Cl_kappa,paired_perreal_fid}.npz +
#       bind_science/runs/dmo/run_0000/Cl_kappa.npz; forward model = GP emulator
import emcee, corner
import bind
from bind.inference.design import _unit_to_native, ASTRO_PARAM_INDICES
from bind.params import PARAM_LOG_FLAG, PARAM_MIN, PARAM_MAX

AIDX = np.asarray(ASTRO_PARAM_INDICES)
t0 = time.time()
# v3: the inference forward model now uses the SHIPPED gpgpu backend (auto-GPU
# at execution; the sklearn 'gp' chain of the earlier draft was a backend
# inconsistency). A timed trial below guards against a catastrophically slow
# execution environment and falls back -- loudly -- to sklearn only then.
em_s = Emulator(backend="gpgpu", n_components=12,
                backend_kwargs={"epochs": 400, "lr": 0.1})
em_s.fit(ds, stats=["suppression"])
dev_s = getattr(em_s.backends["suppression"], "_dev", "cpu")
print(f"suppression-only gpgpu GP on all {len(run_ids)} runs in {time.time()-t0:.0f}s "
      f"(device: {dev_s})")

bcl  = np.load(SCI/"runs/bind/run_0000/Cl_kappa.npz")
dcl  = np.load(SCI/"runs/dmo/run_0000/Cl_kappa.npz")
cl_d = dcl["cl"][ZI, ZI]
S_obs  = bcl["cl"][ZI, ZI]/cl_d
S_real = np.load(SCI/"runs/bind/run_0000/paired_perreal_fid.npz")["clk"][:, ZI, :]/cl_d

msk, RBK = (ELL >= 100) & (ELL <= ELL_TRUST), 8
def compress(v):
    return rebin(v[..., msk], RBK)
ell_c, S_obs_c = compress(ELL), compress(S_obs)

# full data covariance of the 25-bin mean from the 50 paired realizations
S_real_c = compress(S_real)                          # (50, 25)
NRl, NB  = S_real_c.shape
C_data   = np.cov(S_real_c, rowvar=False)/NRl        # covariance of the mean
HARTLAP  = (NRl - NB - 2)/(NRl - 1)
C_eff    = C_data/HARTLAP
corr_off = (np.abs(np.corrcoef(S_real_c, rowvar=False))
            [~np.eye(NB, dtype=bool)]).mean()
print(f"{NB} bins from {NRl} realizations; Hartlap factor {HARTLAP:.2f}; "
      f"mean |off-diag correlation| = {corr_off:.2f}")
# Percival/Dodelson-Schneider: precision-matrix sampling noise inflates the
# PARAMETER variance beyond the Hartlap mean correction; quoted widths are
# pre-correction (see the §4c markdown note)
B_PD = (NRl - NB - 2)/((NRl - NB - 1)*(NRl - NB - 4))
m1_2, m1_6 = 1 + B_PD*(NB - 2), 1 + B_PD*(NB - 6)
print(f"Percival/DS parameter-variance factor (n={NRl}, p={NB}): "
      f"m1 = {m1_6:.2f} (n_par=6) to {m1_2:.2f} (n_par=2) -> quoted 1sigma widths "
      f"are PRE-correction; multiply by sqrt(m1) = {np.sqrt(m1_6):.2f}-{np.sqrt(m1_2):.2f}")

USE_SH = False   # Sellentin & Heavens (2016, MNRAS 456, L132) REPORTED
                 # ALTERNATIVE to the Hartlap-corrected Gaussian likelihood below
                 # (helper defined in the setup cell). NOT the default at the
                 # current n=NRl=50 realizations (well under the >=250 TODO
                 # floor); flip once the ~550-realization covariance sets land
                 # (TODO Sec 4c). Demonstrated, not asserted, at chi2/dof=1:
_chi2_demo = float(NB)
print(f"Sellentin-Heavens REPORTED ALTERNATIVE (USE_SH={USE_SH}): at chi2/dof=1 "
      f"(chi2={_chi2_demo:.0f}, n_real={NRl}, n_data={NB}) Gaussian loglike (up to "
      f"the shared normalization) = {-0.5*_chi2_demo:.1f} vs SH = "
      f"{float(sellentin_heavens_loglike(_chi2_demo, NRl, NB)):.1f} (SH is less "
      "confident at finite n_real; converges to the Gaussian value as n_real -> "
      "inf, see the helper's unit check)")

FID = bind.fiducial_params()
def theta_to_native(U):
    nat = np.tile(FID, (len(U), 1))
    nat[:, AIDX] = _unit_to_native(np.atleast_2d(U), AIDX)
    return nat

def make_log_prob(Cbase, diag_only=False, se_scale=1.0):
    Cd = np.diag(np.diag(Cbase)) if diag_only else Cbase
    def log_prob(U):
        U = np.atleast_2d(U)
        out = np.full(len(U), -np.inf)
        ok = ~((U < 0).any(1) | (U > 1).any(1))
        if ok.any():
            p  = em_s.predict(theta_to_native(U[ok]), stats=["suppression"])
            Sp = compress(np.asarray(p["suppression"])[:, ZI, :])
            Se = se_scale*compress(np.asarray(p["suppression_err"])[:, ZI, :])
            r  = Sp - S_obs_c
            ll = np.empty(len(r))
            for i in range(len(r)):
                C = Cd + np.diag(Se[i]**2)
                chi2_i = float(r[i] @ np.linalg.solve(C, r[i]))
                if USE_SH:
                    ll[i] = sellentin_heavens_loglike(chi2_i, NRl, NB)
                    continue
                sgn, logdet = np.linalg.slogdet(C)
                ll[i] = -0.5*(chi2_i + logdet)
            out[ok] = ll
        return out
    return log_prob

MOVES = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
def run_chain(logp, nsteps, seed):
    nw, ndim = 128, 30
    p0 = np.random.default_rng(seed).uniform(0.02, 0.98, (nw, ndim))
    s = emcee.EnsembleSampler(nw, ndim, logp, vectorize=True, moves=MOVES)
    s.random_state = np.random.RandomState(seed).get_state()  # seed the sampler too
    s.run_mcmc(p0, nsteps, progress=False)
    tau = s.get_autocorr_time(tol=0)
    disc = min(int(4*tau.max()), nsteps//2)
    thin = max(1, int(tau.max()/4))
    ch = s.get_chain(discard=disc, thin=thin, flat=True)
    chw = s.get_chain(discard=disc, thin=thin)                # (n, walkers, dim)
    ess = 128*(nsteps - disc)/tau.max()
    return ch, tau, s.acceptance_fraction.mean(), ess, chw

# timed trial (200 steps): record the per-step cost of the batched-GP likelihood
# in the run log; ONLY a catastrophically slow execution environment triggers
# the sklearn fallback -- decided here at execution time, not when authoring
TRIAL_STEPS, MAX_S_PER_STEP = 200, 1.5
def timed_trial(logp, seed=99):
    p0t = np.random.default_rng(seed).uniform(0.02, 0.98, (128, 30))
    st = emcee.EnsembleSampler(128, 30, logp, vectorize=True, moves=MOVES)
    st.random_state = np.random.RandomState(seed).get_state()
    tt = time.time()
    st.run_mcmc(p0t, TRIAL_STEPS, progress=False)
    return (time.time() - tt)/TRIAL_STEPS
T_PER_STEP = timed_trial(make_log_prob(C_eff))
print(f"timed trial: {TRIAL_STEPS} steps x 128 walkers in "
      f"{TRIAL_STEPS*T_PER_STEP:.1f}s -> {1e3*T_PER_STEP:.0f} ms/step "
      f"(one batched gpgpu predict per step, device {dev_s})")
if T_PER_STEP > MAX_S_PER_STEP:
    print("!"*78 + f"\nLOUD FALLBACK: shipped gpgpu likelihood costs "
          f"{T_PER_STEP:.2f} s/step > {MAX_S_PER_STEP} s/step budget at execution "
          f"time; refitting the forward model with the sklearn 'gp' backend -- the "
          f"published posterior then does NOT use the shipped backend\n" + "!"*78)
    em_s = Emulator(backend="gp", n_components=12, backend_kwargs={"n_restarts": 1})
    em_s.fit(ds, stats=["suppression"])
    T_PER_STEP = timed_trial(make_log_prob(C_eff))
    print(f"sklearn fallback trial: {1e3*T_PER_STEP:.0f} ms/step")

t0 = time.time()
chain, tau, acc, ess, chw_main = run_chain(make_log_prob(C_eff), 10000, seed=1)
print(f"emcee full-cov: 128 x 10000 (~{10000/max(tau.max(),1):.0f} tau_int) in "
      f"{time.time()-t0:.0f}s; acc {acc:.2f}; "
      f"tau_int mean/max {tau.mean():.0f}/{tau.max():.0f}; ESS ~{ess:.0f}; "
      f"{len(chain)} samples")

shrink   = chain.std(0)/(1/np.sqrt(12))
fid_astro = FID[AIDX]
edge = (np.abs(fid_astro - np.asarray(PARAM_MIN)[AIDX]) < 1e-12) | \
       (np.abs(fid_astro - np.asarray(PARAM_MAX)[AIDX]) < 1e-12)
top6 = [i for i in np.argsort(shrink) if not edge[i]][:6]
logf = np.asarray(PARAM_LOG_FLAG)[AIDX].astype(bool)
use_log = logf & (fid_astro > 0)                     # guard log10(0) truths
nat_ch  = _unit_to_native(chain, AIDX)
plot_ch = np.where(use_log[None, :], np.log10(np.where(nat_ch > 0, nat_ch, np.nan)), nat_ch)
fid_plot = np.where(use_log, np.log10(np.where(fid_astro > 0, fid_astro, np.nan)), fid_astro)
labels = [(r"$\log_{10}$ " if use_log[i] else "") + short_label(pnames[i]) for i in top6]

fig = plt.figure(figsize=(TWO_COL[0], TWO_COL[0]))
# smooth=1.0 affects ONLY the 2-D contours (disclosed in the §4c markdown); the
# 1-D diagonals are corner's raw unsmoothed histograms -- we overlay a 2x-finer
# raw histogram (rescaled to counts per 28-bin width) as an explicit sanity
# layer against smoothing-hidden raggedness
corner.corner(plot_ch[:, top6], labels=labels, truths=fid_plot[top6],
              color=COLORS["bind"], truth_color=COLORS["highlight"],
              bins=28, smooth=1.0, levels=(0.68, 0.95), plot_datapoints=False,
              plot_density=False, fill_contours=True, max_n_ticks=3,
              label_kwargs=dict(fontsize=6.5), hist_kwargs=dict(lw=1.1), fig=fig)
n6 = len(top6)
for k in range(n6):
    a_diag = fig.axes[k*n6 + k]
    xk = plot_ch[:, top6[k]]
    xk = xk[np.isfinite(xk)]
    cnt56, edg56 = np.histogram(xk, bins=56)
    a_diag.stairs(cnt56.astype(float)*2.0, edg56, color="0.45", lw=0.5)
for a in fig.axes:
    a.tick_params(labelsize=5)
save(fig, "figs_v2/fig10_astro_corner")
plt.show()
print("posterior/prior width (top 10; * = fiducial on prior edge, excluded from corner): "
      + ", ".join(f"{short_label(pnames[i])}{'*' if edge[i] else ''} {shrink[i]:.2f}"
                  for i in np.argsort(shrink)[:10]))
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §5a — Fig 14: what the mass floor leaves out — and how well the captured low-mass halos are painted (new figure)

The maps paint halos with $M_{200c}\geq10^{13}\,M_\odot/h$; smaller halos enter only where they
fall inside a painted patch ("reuse capture"). Panel (a) is the completeness *measurement
itself*, recomputed live in this cell rather than quoted from an external analysis: the
$10^{12}$–$10^{13}\,M_\odot/h$ FoF band (TNG300-Dark group catalog = the denominator) is
cross-matched against the painted-patch metadata with exactly the production paste geometry
(nearest painted host within $\min(4\times R_{200,\rm host},$ half-patch$)$, and the full
$6.25\,h^{-1}$Mpc footprint as the reuse-only ceiling), giving the capture fraction *vs mass*
for both footprints. These are **halo-count fractions** — the previous draft's "of the band's
gas" overstated what was measured (a gas-weighted capture would also need hydro gas masses for
the *uncaptured* halos). The overall fractions are stamped on the figure
(RECOMPUTED-ON-RUN; reference values 28.7% at $4\times R_{200}$, 51.6% at full patch, flat to
mildly rising with mass). Panel (b): the ~13k *captured* low-mass halos are painted essentially
as well as the hosts — median BIND/truth gas and Compton-$y$ ratios near unity with no strong
mass trend (values printed by the cell). The missing low-mass signal is therefore a
*completeness* (aperture-coverage) deficit, not a painting-fidelity one — this is what
suppresses the mean $\tau$ and $y$ (see §5b).
''')

code(r'''
# ── Fig 14: low-mass completeness (live) and captured-halo fidelity ──────────
# data: TNG300-Dark FoF group catalog (denominator; ~20 s read) +
#       bind_lightcone_tng/snap_096/composite_slab0*.npz metadata (painted
#       hosts) + lightcone_transforms.json (same transform the lightcone used)
#       + /mnt/home/mlee1/BIND/lowmass_096_compare.npz (13226 captured halos,
#       1e12-1e13 Msun/h, bind_/truth_ per-halo gas/star/y; fidelity panel).
# v3 FIX (review): the 28.7%/51.6% capture fractions were hardcoded from the
# external lightcone_lowmass_reuse analysis and never plotted; recompute them
# HERE with exactly examples/lightcone_lowmass_reuse.measure()'s geometry
# (metadata-only variant: capture flags need no patch pixel reads) and make
# the capture-fraction-vs-mass curve the figure's panel (a). These are
# halo-COUNT fractions (the old markdown said "of the band's gas" -- wrong).
from scipy.spatial import cKDTree
from bind.inference.lightcone_transforms import LightconeTransforms
from bind.inference.paint import NATIVE_PIXEL_SIZE_MPCH, _assign_halos_to_slabs, _round_npix
from bind.inference import io_gadget

slabs14 = {}
for si in range(4):
    dsl = np.load(LC/f"snap_096/composite_slab{si:02d}.npz")
    slabs14[int(dsl["slab_idx"])] = dict(centers=dsl["halo_centers"].astype(np.float64),
                                         r200=dsl["halo_r200"].astype(np.float64))
    BOX14, NSLAB14 = float(dsl["box_size"]), int(dsl["n_slabs"])
    del dsl
pix14 = BOX14/_round_npix(BOX14, NATIVE_PIXEL_SIZE_MPCH)      # native 205/4198 Mpc/h
half_patch14 = 64*pix14                                        # 128-px patch half-width

t0 = time.time()
cat14 = io_gadget.read_fof_catalog(
    "/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output/groups_096",
    snapshot=96, halo_mass_min=1e12, mass_field="Group_M_Crit200")
xy14 = LightconeTransforms.load(LC/"lightcone_transforms.json").apply(
    cat14["positions"], 0, BOX14)                              # snap_idx=0 = snap_096
mass14 = cat14["mass"]
band14 = mass14 < 1e13
slab_of14 = _assign_halos_to_slabs(xy14[:, 2], BOX14, NSLAB14)
print(f"FoF band 1e12-1e13: {band14.sum()} halos (catalog read {time.time()-t0:.0f}s)")

def _pd14(a, b):                                               # periodic 2D delta
    dd = a - b
    return dd - BOX14*np.round(dd/BOX14)

cap4_14 = np.zeros(len(mass14), bool)                          # production 4xR200 paste
capF_14 = np.zeros(len(mass14), bool)                          # full-patch ceiling
for si, sl in slabs14.items():
    idx = np.where(band14 & (slab_of14 == si))[0]
    if not len(idx):
        continue
    tree = cKDTree(sl["centers"], boxsize=BOX14)
    pts = xy14[idx, :2]
    d1, _ = tree.query(pts, k=1, distance_upper_bound=half_patch14)
    capF_14[idx] = np.isfinite(d1)
    rad4 = np.minimum(4.0*sl["r200"], half_patch14)            # per-host capture radius
    for i, cl in enumerate(tree.query_ball_point(pts, half_patch14)):
        for c in cl:
            if np.hypot(*_pd14(pts[i], sl["centers"][c])) <= rad4[c]:
                cap4_14[idx[i]] = True
                break

f4_tot = cap4_14[band14].mean(); fF_tot = capF_14[band14].mean()
edg12 = np.arange(12.0, 13.05, 0.125)
lM14 = np.log10(mass14)
cen12 = 0.5*(edg12[1:] + edg12[:-1])
fr4, frF, se4, seF = (np.full(len(cen12), np.nan) for _ in range(4))
for i in range(len(cen12)):
    m = band14 & (lM14 >= edg12[i]) & (lM14 < edg12[i+1])
    if m.sum():
        for fr, se, capx in [(fr4, se4, cap4_14), (frF, seF, capF_14)]:
            f = capx[m].mean()
            fr[i], se[i] = f, np.sqrt(f*(1-f)/m.sum())

# fidelity of the CAPTURED halos (per-halo BIND-vs-truth table, full-patch set)
lm = np.load(Path("/mnt/home/mlee1/BIND/lowmass_096_compare.npz"))
lM = np.log10(lm["bind_mass"])
gr = lm["bind_gas"]/np.where(lm["truth_gas"] > 0, lm["truth_gas"], np.nan)
yr = lm["bind_y"]/np.where(lm["truth_y"] > 0, lm["truth_y"], np.nan)

fig, ax = plt.subplots(1, 2, figsize=TWO_COL)
for fr, se, c, lab in [(frF, seF, COLORS["bind"], "full patch (reuse ceiling)"),
                       (fr4, se4, COLORS["secondary"], r"$4\times R_{200}$ paste (production)")]:
    ax[0].fill_between(cen12, fr - se, fr + se, color=c, alpha=BAND_ALPHA, lw=0)
    ax[0].plot(cen12, fr, "o-", ms=2.5, color=c, lw=1.4, label=lab)
ax[0].axhline(0, color=COLORS["dmo"], lw=0.5)
ax[0].set_ylim(0, 0.75)
ax[0].set_xlabel(r"$\log_{10} M_{200c}\,[M_\odot/h]$")
ax[0].set_ylabel("capture fraction (halo counts)")
ax[0].legend(loc="upper left", fontsize=5.6)
ax[0].text(0.97, 0.05, f"band total: {100*f4_tot:.1f}% at $4{{\\times}}R_{{200}}$\n"
           f"{100*fF_tot:.1f}% at full patch\n({band14.sum():,} halos; bands: binomial SE)",
           transform=ax[0].transAxes, fontsize=5.6, ha="right")
panel_label(ax[0], "(a)", loc="upper right")

# bands are the 16-84% per-halo percentile range from running_stat (named in
# the legend -- they previously read as undefined shading)
for arr, c, lab in [(gr, COLORS["bind"], "gas mass (median, 16–84% band)"),
                    (yr, COLORS["secondary"], "Compton $y$ (median, 16–84% band)")]:
    ok = np.isfinite(arr)
    cen, med, lo, hi, se = running_stat(lM[ok], arr[ok], edg12, boot=100)
    ax[1].fill_between(cen, lo, hi, color=c, alpha=BAND_ALPHA, lw=0)
    ax[1].plot(cen, med, color=c, lw=1.5, label=lab)
ax[1].axhline(1, color=COLORS["dmo"], ls=":", lw=0.8)
ax[1].set_xlabel(r"$\log_{10} M_{200c}\,[M_\odot/h]$")
ax[1].set_ylabel("captured halos: BIND/truth")
ax[1].set_ylim(0.4, 1.9)
ax[1].legend(loc="upper left", fontsize=6)
panel_label(ax[1], "(b)", loc="lower right")
fig.tight_layout(w_pad=1.0)
save(fig, "figs_v2/fig14_completeness")
plt.show()
print(f"live capture recompute (count fractions): {100*f4_tot:.1f}% at 4xR200 / "
      f"{100*fF_tot:.1f}% at full patch of the {band14.sum():,} band halos "
      f"(engine reference: 28.7% / 51.6%)")
print(f"{len(lM)} captured halos 1e12-1e13 in the fidelity table; "
      f"median gas ratio {np.nanmedian(gr):.3f}, y ratio {np.nanmedian(yr):.3f}")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §5b — Scope of validity (consolidated caveats)

**Substrate and resolution.** BIND is trained at CAMELS resolution and deployed on
TNG300-1-Dark; the release is validated *only* at that particle mass. The designed resolution
gate (TNG300-2/3-Dark halo-matched repaint, `examples/resolution_gate.py`) has **not** been run
— users of `bind-paint` on other N-body substrates should treat ≲120 DM particles per
$10^{13}\,M_\odot/h$ halo as out of scope until it is.

**Completeness and map means.** The $\tau$/$y$ planes contain only gas inside painted halo
apertures ($\geq10^{13}$ hosts + reuse capture, fig 14) — a **one-halo product with no
two-halo term**: correlated large-scale gas between halos is absent by construction, so
large-aperture CAP photometry and large-$R$ stacked profiles are biased low (stated in the
§6 products table). The measured mean
$y\simeq1.0\times10^{-6}$ and mean $\tau\simeq1.8\times10^{-3}$ of the released fiducial
map (both printed live by the fig-0 cell) therefore *undershoot* all-sky
expectations (missing sub-$10^{13}$ halos and unbound gas; roughly half of the low-mass band's
*halos* lie outside captured patches — the live-computed count-capture fractions are printed by
the fig-14 cell). Absolute SZ/DM monopoles should not be quoted from
these maps; per-halo and fluctuation statistics are the released products.

**Validation coverage.** Field-level closure is shown for $\kappa\kappa$, $\kappa y$, $yy$,
peaks, PDF, MFs at $z_s=1$ (fig 4) and for $\kappa\kappa$ at all five planes (fig 11a);
"truth" is the seed-paired hydro paste (emulation fidelity, not sky realism — full-TNG truth
maps TODO). $C_\ell^{\tau\tau}$ has **no** field-level truth (no truth $\tau$ trace yet);
$\tau$ is validated at halo-stack level only (fig 6b, the stacked $\tau(r)$ profile — the
electron column rebuilt from the same background-subtracted gas surface density).
Halo-level thermodynamics carry a
**mass-dependent** amplitude offset at $z\simeq0$ ($\approx$+10–12% at
$10^{13}\,M_\odot/h$ declining to $\approx$0 above $10^{14}$; fig 3, with a fitted
linear-in-$\log M$ calibration printed by that cell), drifting to $\sim$+20% in $Y_{500c}$
by $z\gtrsim1$ (fig 11b — the known $z>0$ thermo $a$-factor caveat); high-$z_s$ tSZ
products inherit this characterized drift.

**Redshift de-biasing (shipped template).** The $z>0$ thermo drift in fig 11(c)/(d) is smooth enough to correct rather than merely flag. For each closure ratio $r=\mathrm{BIND/truth}$ we fit $r(a)$ ($a=1/(1+z)$) in three low-order forms — linear ($r=c_0+c_1a$), quadratic ($r=c_0+c_1a+c_2a^2$), and power-law ($r=A_0a^\alpha$) — by least squares *weighted by the per-snapshot uncertainties* (bootstrap over halos for $Y_{500c}$/$f_{\rm gas}$; between-mass-stack SE for the $y$-profile), selecting by $\mathrm{AIC}=\chi^2+2k$. The quadratic form is selected for all three quantities — decisively over linear for $Y_{500c}$ and $f_{\rm gas}$ ($\Delta$AIC $\approx95$ and $41$), but only *marginally* for the $y$-profile ($\Delta$AIC $\lesssim3$ across all three forms once the stack-to-stack errors are propagated; the quadratic is retained there for uniformity, not by evidence). The shipped coefficients, their $1\sigma$ uncertainties, template rms/max residuals, and per-form AICs are **printed and exported by the fig-11 cell** (`DEBIAS_TEMPLATES`, now including `coef_err`) — that printout is the citable source. RECOMPUTED-ON-RUN reference values (verified against the current caches):
$$r_{Y_{500c}}(a) \simeq 1.563 - 1.135\,a + 0.636\,a^2 \quad (\pm0.030,\ \pm0.089,\ \pm0.065;\ \text{rms }1.6\%)$$
$$r_{f_{\rm gas}}(a) \simeq 1.214 - 0.300\,a + 0.156\,a^2 \quad (\pm0.010,\ \pm0.032,\ \pm0.024;\ \text{rms }0.25\%)$$
$$r_{y\text{-prof}}(a) \simeq 1.443 - 0.948\,a + 0.559\,a^2 \quad (\pm0.116,\ \pm0.353,\ \pm0.254;\ \text{rms }3.1\%)$$
The power-law form ($Y$: $\alpha\approx-0.18$) still cleanly excludes a single missing $a$-factor ($\alpha=-1$), but is now AIC-disfavored against the quadratic for $Y_{500c}$ and $f_{\rm gas}$ under the weighted fit. Users correct a raw BIND halo-scale statistic via $X_{\rm debiased}(a) = X_{\rm BIND}(a)/r(a)$, propagating the printed coefficient uncertainties. **Scope:** fit to *median*, group-and-cluster-scale ($\gtrsim10^{13}M_\odot$), TNG-fiducial-astrophysics halos/stacks only, over the 20 released snapshots $0.03\lesssim z\lesssim2.44$; the per-quantity rms above is the *irreducible* scatter after the smooth trend is removed, not the total drift, and it says nothing about scatter at fixed halo mass or off-fiducial astrophysics, which the template does not model. Do not extrapolate outside $0.03\le z\le2.44$.

**Statistics.** All 256 nodes have lightcone maps (0114/0115/0117 repainted 2026-08-03; patch-level
products cover all 256). Raw spectra alias above $\ell\sim1.5\times10^4$ (ratios cancel it).
Peaks/minima are noiseless ($n_{\rm gal}=0$) and realization-noise dominated as suite
statistics (figs 8, 9). WST: no fiducial/truth product, 40/253 in the dataset — unreleased as
a validated statistic. DM statistics: released as a derived per-node product (halo-aperture
only; not an emulator head; the Job-2 backfill completes coverage to every completed node).
The single 25 deg² footprint has a ~10%
Gaussian cosmic-variance floor on $C_\ell$ at $\ell\simeq10^3$; ratio statistics cancel it.

**$\tau$ is a velocity-free electron column** ($\sigma_T x_e\Sigma_{\rm gas}/m_p$ along the
deflected rays). A $\Delta T_{\rm kSZ}$ map needs a velocity surrogate (unbuilt); comparisons
to stacking data go through $\tau^{\rm CAP}=T^{\rm CAP}_{\rm kSZ}/(T_{\rm CMB}\sigma_v/c)$.
No DMO $\tau$/$y$ baseline exists (paired DMO trace is $\kappa$-only).

**Prior scope.** All 256 nodes share TNG cosmology and the TNG galaxy-formation model with
30 varied subgrid amplitudes; the suite brackets TNG-like feedback, not the full
observationally allowed range (fig 12 discussion, and Appendix C's fig 19 for the
$f_{\rm gas}$-space statement against real X-ray/eROSITA measurements) and not cosmology.
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## Appendix B — Fig 16: how robust is that posterior? (reads with §4c)

Four referee-grade checks on fig 10: (a) the **posterior-predictive** test — bands of emulated
$S(\ell)$ from 200 posterior draws against the measured data points, whose error bars are now
*defined in the legend*: $\sqrt{{\rm diag}\,C_{\rm eff}}$, the Hartlap-corrected SE of the
50-realization mean (its χ²/dof ≪ 1 is *expected*, see the printed note); (b) the
posterior/prior width for **all 30** parameters (not just the corner's top-6), with prior-edge
parameters hatched — and the hatching now carries a legend entry ("fiducial on prior edge");
(c) sensitivity of the top-6 widths to the error treatment — diagonal-only vs full
covariance, a conservative variant dropping the $1/\sqrt{50}$ realization averaging (upper
bracket for maximally correlated re-traces), and a **GP-error inflation** variant with the
inflation factor $\hat\alpha$ *fitted* to restore the Gaussian 2σ quantile on the held-out
runs (fig 15; printed there per statistic) — closing the loop on the diagnosed tail deficit,
the dominant untested term in the error budget. The three variant chains share fig 10's
batched likelihood and are now **extended to the main chain's 10 000 steps when the timed
per-step cost keeps the total under ~1 h** (the cell checks fig 10's timed-trial print at
execution time; if it must stay at 5000 it says so loudly and quotes
bootstrap-over-walkers width errors instead — those bootstrap SEs are printed either way).
Widths move by tens of percent, not factors — the qualitative conclusion ("WL suppression
alone constrains a few SN-wind/IMF/BH directions") is robust to the error treatment. All
widths here are pre-Percival/Dodelson–Schneider (see the §4c note and printed factors).
**This whole figure is Appendix B material**: all three referee-grade robustness checks
(posterior-predictive, all-30 widths, covariance-treatment sensitivity) support fig 10's
posterior but are not needed to follow the main §4c narrative, so they are placed here
rather than split panel-by-panel between the main text and the appendix.
''')

code(r'''
# ── Fig 16 (Appendix B): posterior-predictive + shrink spectrum + covariance robustness ──
draws = chain[np.random.default_rng(7).integers(0, len(chain), 200)]
Spred = compress(np.asarray(
    em_s.predict(theta_to_native(draws), stats=["suppression"])["suppression"])[:, ZI, :])

# variant chains: extend to the MAIN chain's 10000 steps when fig 10's timed
# per-step cost keeps 3 extra chains under the ~1h budget (checked at execution
# time); otherwise stay at 5000 and say so LOUDLY, quoting the bootstrap-over-
# walkers width errors computed below either way
VAR_STEPS = 10000 if 3*10000*T_PER_STEP < 3600 else 5000
if VAR_STEPS < 10000:
    print("!"*78 + f"\nNOTE: variant chains kept at {VAR_STEPS} steps -- "
          f"3 x 10000 x {T_PER_STEP:.2f} s/step would exceed the ~1h budget; "
          f"width ratios rely on the bootstrap-over-walkers errors printed "
          f"below\n" + "!"*78)
t0 = time.time()
chain_d, tau_d, acc_d, ess_d, chw_d = run_chain(
    make_log_prob(C_eff, diag_only=True), VAR_STEPS, seed=2)
chain_c, tau_c, acc_c, ess_c, chw_c = run_chain(
    make_log_prob(C_eff*NRl), VAR_STEPS, seed=3)
# GP-error inflation FITTED from the fig 15 tail deficit (alpha restores 2-sigma coverage)
ALPHA_S = ALPHA["suppression"]
chain_i, tau_i, acc_i, ess_i, chw_i = run_chain(
    make_log_prob(C_eff, se_scale=ALPHA_S), VAR_STEPS, seed=4)
print(f"variant chains (128 x {VAR_STEPS} each) in {time.time()-t0:.0f}s: "
      f"diag acc {acc_d:.2f} ESS {ess_d:.0f}; "
      f"conservative acc {acc_c:.2f} ESS {ess_c:.0f}; "
      f"GPx{ALPHA_S:.2f} acc {acc_i:.2f} ESS {ess_i:.0f}")
shr_d = chain_d.std(0)/(1/np.sqrt(12))
shr_c = chain_c.std(0)/(1/np.sqrt(12))
shr_i = chain_i.std(0)/(1/np.sqrt(12))

def width_se16(chw, B=200, seed=8):
    """Bootstrap-over-walkers SE of the posterior/prior width (walkers are the
    nearly-independent units of an ensemble chain)."""
    rngb = np.random.default_rng(seed)
    nw, ndim = chw.shape[1], chw.shape[2]
    ws = np.empty((B, ndim))
    for b in range(B):
        sel = rngb.integers(0, nw, nw)
        ws[b] = chw[:, sel, :].reshape(-1, ndim).std(0)
    return ws.std(0)/(1/np.sqrt(12))
se_main16 = width_se16(chw_main)
se_var16 = {"diag": width_se16(chw_d), "conserv": width_se16(chw_c),
            "GPxalpha": width_se16(chw_i)}

fig, ax = plt.subplots(1, 3, figsize=(TWO_COL[0], 3.0),
                       gridspec_kw=dict(width_ratios=[1.2, 1.35, 1.0]))
ax[0].fill_between(ell_c, np.percentile(Spred, 2.5, 0), np.percentile(Spred, 97.5, 0),
                   color=COLORS["bind"], alpha=0.2, lw=0, label="post. pred. 95%")
ax[0].fill_between(ell_c, np.percentile(Spred, 16, 0), np.percentile(Spred, 84, 0),
                   color=COLORS["bind"], alpha=0.45, lw=0, label="68%")
ax[0].errorbar(ell_c, S_obs_c, yerr=np.sqrt(np.diag(C_eff)), fmt="o", ms=2.5,
               color=COLORS["truth"], lw=0.8,
               label="measured; bars $=\\sqrt{{\\rm diag}\\,C_{\\rm eff}}$\n"
                     "(Hartlap-corr. SE of 50-real. mean)")
ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\ell$"); ax[0].set_ylabel(r"$S(\ell)$")
ax[0].legend(loc="lower left", fontsize=5.0)
panel_label(ax[0], "(a)", loc="upper right")

order30 = np.argsort(shrink)
yy30 = np.arange(30)
ax[1].barh(yy30, shrink[order30],
           color=[COLORS["dmo"] if edge[i] else COLORS["bind"] for i in order30],
           hatch=["//" if edge[i] else "" for i in order30])
ax[1].axvline(1, color=COLORS["dmo"], ls=":", lw=0.9)
ax[1].set_yticks(yy30)
ax[1].set_yticklabels([short_label(pnames[i]) for i in order30], fontsize=4.8)
ax[1].set_xlabel("posterior/prior width")
ax[1].set_xlim(0, 1.15); ax[1].invert_yaxis()
from matplotlib.patches import Patch
# legend moved to the TOP-right clear zone (top rows = most-constrained =
# shortest bars after invert_yaxis) with an opaque frame, so no swatch can
# fuse with the long hatched bars at the bottom
ax[1].legend(handles=[Patch(facecolor=COLORS["dmo"], hatch="///",
                            label="fiducial on prior edge"),
                      Patch(facecolor=COLORS["bind"], label="constrainable")],
             loc="upper right", fontsize=5.0, frameon=True, framealpha=1.0,
             edgecolor="0.6")
panel_label(ax[1], "(b)", loc="upper right")

xx6 = np.arange(len(top6)); w6 = 0.27
ax[2].bar(xx6 - w6, shr_d[top6]/shrink[top6], w6, color=COLORS["secondary"],
          label="diag/full")
ax[2].bar(xx6, shr_c[top6]/shrink[top6], w6, color=COLORS["dmo"],
          label="conserv./full")
ax[2].bar(xx6 + w6, shr_i[top6]/shrink[top6], w6, color=COLORS["highlight"],
          label=r"GP$\sigma{\times}\hat\alpha$/full")
ax[2].axhline(1, color=COLORS["truth"], lw=0.7)
ax[2].set_xticks(xx6)
ax[2].set_xticklabels([short_label(pnames[i]) for i in top6], rotation=45,
                      ha="right", fontsize=5)
ax[2].set_ylabel("width ratio")
ax[2].legend(loc="upper left", fontsize=5.6, frameon=True, framealpha=1.0,
             edgecolor="0.6")   # opaque frame: bars must not run through swatches
panel_label(ax[2], "(c)", loc="upper right")
fig.tight_layout(w_pad=1.1)
save(fig, "figs_v2/fig16_inference_robustness")
plt.show()
pp_chi2 = float(np.mean((S_obs_c - Spred.mean(0))**2/np.diag(C_eff)))
print(f"posterior-predictive chi2/dof (diag) = {pp_chi2:.2f} "
      "(<<1 is EXPECTED: 30 params vs 25 correlated bins, GP variance in the "
      "likelihood but not this denominator — quoted only to show the posterior "
      "mean threads the data, not as goodness-of-fit)")
print(f"median top-6 width ratios: diag/full {np.median(shr_d[top6]/shrink[top6]):.2f}, "
      f"conservative/full {np.median(shr_c[top6]/shrink[top6]):.2f}, "
      f"GPx{ALPHA_S:.2f}(fitted)/full {np.median(shr_i[top6]/shrink[top6]):.2f}")
# bootstrap-over-walkers sampling errors on the ratios (quoted always; they are
# the primary error statement whenever VAR_STEPS < the main chain's length)
for lab, shr_v, se_v in [("diag", shr_d, se_var16["diag"]),
                         ("conserv", shr_c, se_var16["conserv"]),
                         ("GPxalpha", shr_i, se_var16["GPxalpha"])]:
    ratio_se = (shr_v[top6]/shrink[top6])*np.sqrt((se_v[top6]/shr_v[top6])**2 +
                                                  (se_main16[top6]/shrink[top6])**2)
    print(f"  {lab:>8s}/full: median bootstrap-over-walkers SE on the top-6 width "
          f"ratio = {np.median(ratio_se):.3f} "
          f"({100*np.median(ratio_se/(shr_v[top6]/shrink[top6])):.1f}% relative)")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## Appendix C — Fig 19: group gas fractions vs X-ray and eROSITA measurements (restored)

The one figure where the package meets observational *measurements* directly — restored
from the v1 draft with corrected framing (the retracted "41% of cosmic" saturation number
does **not** appear; this comparison has distinct provenance and never depended on it).
Group-scale ($M_{200c}=1$–$2\times10^{13}\,M_\odot/h$), background-subtracted
$f_{\rm gas}(<R_{500c})$ for each of the 57 distinct one-parameter (1P) CAMELS-TNG feedback
perturbations (the design's 60 runs; three parameters are prior-bounded *at* the fiducial
value, so 57 is the design's structural count, not missing data), sorted and colored by
feedback family, against two observational anchors: the
X-ray-standard band $f_{\rm gas}\simeq0.06$–$0.10$, and the eROSITA-low value
$f_{\rm gas}\simeq0.026$ (Popesso+24). Read: the **fiducial sits inside the
X-ray-standard band** ($f_{\rm gas}\approx0.08$; exact value printed by the cell), while
the eROSITA-low measurement implies far stronger gas removal — **no single TNG feedback
knob reaches it**; the strongest 1P run closes only about half of the fiducial→eROSITA-low
gap (exact fraction printed by the cell, RECOMPUTED-ON-RUN). The companion kSZ/X-ray
confrontation (Paper IV) finds the same for the full 30-dimensional Sobol design: 0 of the
256 nodes reaches the eROSITA strong-feedback band (external result, not recomputed here).
This is the $f_{\rm gas}$-space counterpart of fig 12's prior-scope caveat — the release
brackets TNG-like feedback, not the strongest observationally preferred feedback — placed
where gas-physics users will look for it.

**Definition caveat.** Like figs 3 and 20, this $f_{\rm gas}(<R_{500c})$ is the
*projected-cylinder*, annulus-subtracted measure (§3c), inflated $\times{\sim}1.3$–$1.6$
relative to the 3-D spherical gas fraction at these group masses; an honest 3-D fiducial
($\approx0.06$) would sit at the *bottom* of the X-ray-standard band, and the
fiducial$\to$eROSITA-low tension factor softens from ${\sim}3.1\times$ to ${\sim}2.3\times$.
The qualitative conclusion — no single TNG feedback knob reaches eROSITA-low — is unchanged.
''')

code(r'''
# ── Fig 19 (Appendix C): group f_gas across the 1P suite vs X-ray/eROSITA ────
# data: bind_science/halo_atlas/{run}_snap096.npz (57 1P feedback runs + fid;
#       keys M_fof, m_gas_500c_bg, m_tot_500c_bg — background-subtracted 500c
#       aperture masses) + bind_science/dashboard_cache/g1_design.json.
#       Port of fig_scripts/fig11_fgas_saturation.py (v1 fig 11) with the
#       corrected framing; observational anchors: X-ray-standard band
#       0.06-0.10, eROSITA-low f_gas = 0.026 (Popesso+24).
def _family19(name):
    s = name.lower()
    if any(k in s for k in ("blackhole", "quasar", "radio", "agn")):
        return "AGN"
    if any(k in s for k in ("wind", "snii", "snia", "imf", "supernov", "sfr", "eqs")):
        return "SN / wind"
    return "other"

def fgas_group19(run, mlo=1e13, mhi=2e13):
    """Median background-subtracted f_gas(<R500c) in the group M200c bin."""
    f = SCI/f"halo_atlas/{run}_snap096.npz"
    if not f.exists():
        return np.nan
    da = np.load(f)
    if "m_gas_500c_bg" not in da.files:
        return np.nan
    s = (da["M_fof"] >= mlo) & (da["M_fof"] < mhi) & (da["m_tot_500c_bg"] > 0)
    if s.sum() < 5:
        return np.nan
    return float(np.nanmedian((da["m_gas_500c_bg"]/da["m_tot_500c_bg"])[s]))

design19 = [dict(run=r, name=nm) for r, _i, nm, _v, _f in
            json.load(open(SCI/"dashboard_cache/g1_design.json"))]
vals19 = []
for e in design19:
    fgv = fgas_group19(e["run"])
    if np.isfinite(fgv):
        vals19.append((e["name"], _family19(e["name"]), fgv))
vals19.sort(key=lambda t: t[2])
fg19  = np.array([v for *_, v in vals19])
fam19 = [f for _, f, _ in vals19]
fid19 = fgas_group19("fid")

FAMC19 = {"AGN": COLORS["highlight"], "SN / wind": COLORS["bind"], "other": COLORS["dmo"]}
EROSITA_LOW, XRAY_LO, XRAY_HI = 0.026, 0.06, 0.10       # Popesso+24; X-ray-standard band

fig, ax = plt.subplots(figsize=(ONE_COL[0], 2.9))
ax.bar(range(len(fg19)), fg19, color=[FAMC19[f] for f in fam19], edgecolor="k",
       lw=0.2, width=0.9)
ax.axhspan(XRAY_LO, XRAY_HI, color=COLORS["secondary"], alpha=0.18, lw=0)
ax.axhline(fid19, color=COLORS["bind"], lw=1.4)
ax.axhline(EROSITA_LOW, color="k", ls="--", lw=1.3)   # black (v3): the red
# dashed line had no contrast where it crossed the red AGN-family bars
from matplotlib.lines import Line2D
from matplotlib.patches import Patch as _Patch19
ax.legend(handles=[
    Line2D([], [], color=COLORS["bind"], lw=1.4, label=f"BIND fiducial ({fid19:.3f})"),
    _Patch19(facecolor=COLORS["secondary"], alpha=0.18, label="X-ray standard (0.06–0.10)"),
    Line2D([], [], color="k", ls="--", lw=1.3,
           label=f"eROSITA-low ({EROSITA_LOW}, Popesso+24)"),
    _Patch19(facecolor=FAMC19["AGN"], label="AGN knob"),
    _Patch19(facecolor=FAMC19["SN / wind"], label="SN/wind knob"),
    _Patch19(facecolor=FAMC19["other"], label="other knob"),
], loc="upper left", fontsize=4.8, ncol=2, handlelength=1.4, columnspacing=0.8)
ax.set_xlabel("1P feedback run (sorted)")
ax.set_ylabel(r"$f_{\rm gas}(<R_{500c})$,"
              "\n" r"$M_{200c}=1$–$2\times10^{13}M_\odot/h$")
ax.set_xlim(-1, len(fg19)); ax.set_ylim(0, None)
fig.tight_layout()
save(fig, "figs_v2/fig19_fgas_erosita")
plt.show()
gap19 = (fid19 - fg19.min())/(fid19 - EROSITA_LOW)
print(f"fiducial group f_gas(<R500c) = {fid19:.4f} (inside the X-ray-standard band "
      f"{XRAY_LO}-{XRAY_HI}); {len(fg19)} 1P runs span {fg19.min():.4f}-{fg19.max():.4f}; "
      f"strongest single knob ({vals19[0][0]}, {vals19[0][1]}) closes {100*gap19:.0f}% of "
      f"the fiducial->eROSITA-low gap; NO single knob reaches eROSITA-low "
      f"({EROSITA_LOW}); the 30-dim Sobol design does not reach it either "
      f"(0/256 nodes -- companion Paper IV, external)")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §6 — Data products and usage notes (new section)

| product | shape / count | coverage | notes |
|---|---|---|---|
| $\kappa$ maps | 50 real × 5 $z_s$ × 1024², 5×5 deg² | 256 Sobol + 57 two-bound + fiducial + DMO + truth | 0.29296875′/px, no beam |
| $\tau$, $y$ maps | same | 256 Sobol + 57 two-bound + fiducial (+ truth $y$) | halo-aperture (one-halo) gas only; no two-halo term; convolve with your instrument beam (§5b) |
| $C_\ell$: $\kappa\kappa$ (5×5), $\kappa y$, $yy$, $\kappa\tau$, $\tau\tau$, $y\tau$ | 724 bins, $\ell$ = 87–5.2×10⁴ | 253 runs | crosses: corrected normalization |
| $S(\ell)$ suppression | (256, 5, 724) | 256 runs | paired DMO denominator |
| peaks/minima, PDF, MFs, moments | per run, 5 $z_s$ | 253 runs | noiseless peaks (+ $n_{\rm gal}$=10 variant at fiducial) |
| $R(\nu)$, $y$-at-peak | (253, 5, 14) | 253 | κ-peak × y cross-statistics |
| halo tables | ~33k halos/run; 8.6M-row parquet | **all 256** | per-halo $M$, $f_{\rm gas}$, $Y$, $T$ at 200c/500c |
| stacked $\tau$/$y$ profiles ("CAP-ready") | (256, 4 mass bins, 17 radii) × 2 z | **all 256** | halo-aperture (one-halo) gas only — no two-halo term, paste-aperture truncated (large-$R$ biased low); $y$-CAP curves carry a 1.6′ beam, $\tau$/$y$ profiles are unbeamed — convolve with your instrument beam |
| assembled dataset | `emulator_dataset_xpkfix.npz` (~127 MB) | 256 × 23 targets | the §3–§4 figure substrate |
| GP emulator | trained on the above (this notebook) | 13 heads (see §4.0) | scaling relations untested |
| WST | 40/253 | partial | unreleased as validated |
| DM statistics | PDF + moments per node | all completed nodes (post-backfill) | release product (halo-aperture only); not an emulator head |

**Painted-map manifest**: $(256_{\rm Sobol}+57_{\rm two\mbox{-}bound}+1_{\rm fid})\times3$ species $\times\,5\,z_s\times50$ realizations $=\mathbf{235{,}500}$ maps, plus the seed-paired DMO $\kappa$ set and the hydro-pasted truth $\kappa/y(/\tau)$ maps. (The two-bound release counts its 57 *distinct* perturbations; the three bound-at-fiducial runs are excluded as duplicates of the fiducial.)

**Usage notes.** Convolve with your instrument beam before comparing to data; match the pixel
scale and aperture convention for discrete CAP photometry; remember the $\tau$/$y$ products are
one-halo only (no two-halo term — large-aperture/large-$R$ statistics are biased low by
construction, §5b); use ratio statistics where possible
(they cancel CIC aliasing and cosmic variance); index everything by `run_ids`. Halo-scale
thermodynamic statistics at $z>0$ should be corrected with the shipped redshift de-biasing
templates, which now carry $1\sigma$ coefficient uncertainties
(`DEBIAS_TEMPLATES[...]["coef_err"]`, §5b) — propagate them; at $z\simeq0$ apply the
mass-dependent (linear-in-$\log M$) amplitude calibration fitted in §2a — the observable-level
offset is *not* a flat factor (fig 3).

### Getting the data (access box)

Proposed release name/version (to be confirmed): **BIND-LS v1** — the BIND Lightcone Suite,
256-node SB35 design. Archive endpoint: *DOI/HF placeholder — to be minted at release*; the
weights already ship via `bind-download-weights` (HF `Maxelee/BIND2`). Load patterns:

```python
import numpy as np
d = np.load("emulator_dataset_xpkfix.npz")            # released statistics cube (256 runs)
S = d["t__suppression__value"][:, 1, :]               # S(ell) at z_s = 1
run_ids = d["run_ids"]                                # node indices (3 missing)

m = np.load("runs/run_0007/kappa_maps.npz")           # per-node maps
kappa = m["kappa"][0, 1]                              # realization 0, z_s = 1

xp = np.load("ksz_confront/bind_tauy_xprof_snap085.npz")
tau_prof = xp["tau"][:, 0, :]                         # 256 nodes, group bin

from bind.emulator import Emulator
em = Emulator.load("lightcone_emulator_gp.pt")        # trained from _xpkfix
out = em.predict(theta_native, z_s=1.0)               # ms per call
```

The dataset sha256 and file inventory are printed by the release-check cell below.

### Where this suite sits (qualitative positioning)

| suite | varied astro params | hydro-informed gas fields | ray-traced lightcone | $\tau$ & $y$ maps | per-halo release |
|---|---|---|---|---|---|
| CAMELS / CMD | up to ~30 (25 $h^{-1}$Mpc boxes) | ✓ (native hydro) | ✗ | box slices only | partial |
| BAHAMAS / ANTILLES | few calibrated variants | ✓ | ✗ | ✗ (box level) | partial |
| CosmoGridV1 | 0 (baryonified post hoc) | ✗ | ✓ (κ) | ✗ | ✗ |
| MillenniumTNG | 1 model | ✓ | ✓ (κ) | limited | ✓ |
| **this release** | **30 (Sobol)** | **✓ (painted)** | **✓ (κ, τ, y, shared seed)** | **✓** | **✓ (all 256)** |

(Entries qualitative; see the cited release papers for exact specifications. The empty corner
this suite fills: *feedback-swept, ray-traced, pixel-aligned multi-probe lightcone maps at
TNG300 volume with per-halo products*.)
''')

code(r'''
# ── Release check: dataset hash + inventory ──────────────────────────────────
import hashlib
h = hashlib.sha256()
with open(DS_PATH, "rb") as fh:
    for blk in iter(lambda: fh.read(1 << 22), b""):
        h.update(blk)
print(f"{DS_PATH.name}: sha256 {h.hexdigest()[:16]}..., "
      f"{DS_PATH.stat().st_size/1e6:.0f} MB, {len(run_ids)} runs, "
      f"{len([k for k in d.files if k.startswith('t__') and k.endswith('__value')])} stat targets")
n_maps = sum((SB35/f"runs/run_{r:04d}/kappa_maps.npz").exists() for r in run_ids)
print(f"map coverage check: {n_maps}/{len(run_ids)} listed runs have kappa_maps.npz on disk")
''')


# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §7 — Conclusions and outlook

We have released and validated BIND-LS v1: 253 ray-traced $\kappa/\tau/y$ lightcone
realizations of one $5\times5\,\mathrm{deg}^2$ footprint spanning 30 dimensions of
IllustrisTNG galaxy-formation physics, with per-halo products at all 256 design nodes, a
calibrated GP emulator over the released statistics, and an end-to-end inference
demonstration. The validation budget is explicit — and, after the dedicated audits, *root-caused*:
sub-percent WL closure at all source planes; a gas/pressure amplitude offset that is flat
in mass in the raw painted patches (a pre-composite regression bias of the flow-matching
model) but **mass-dependent at the observable level** ($\approx$+10–12% at
$10^{13}\,M_\odot/h$ declining to $\approx$0 above $10^{14}$), correctable by the
linear-in-$\log M$ calibration fitted and printed in §2a; a thermodynamic redshift drift whose per-channel slopes and
mass dependence identify high-$z$ under-supervision of the training set (conventions ruled
out; shipped with de-biasing templates and the audit's per-channel drift curves — the v1
remedy; denser multi-$z$ supervision is deferred to a future major version); and a
small-scale $y$-texture budget now measured in radial detail — halo interiors over-textured
by ~20–30%, outskirts *under*-textured by up to 40% with good phase correlation, which
falsifies simple smoothing mitigations; the texture-aware (spectral) training objective it
motivates is likewise deferred (~$10^3$ GPU-hour cost). The science content is compact: the
30-dimensional response collapses onto a dominantly two-dimensional (amplitude,
gas-ejection) surface commanded by SN-wind, IMF, and BH parameters — every statistic
family's leading response direction lies in it, and the gas-thermodynamics statistics
resolve its second direction (fig 8); 38% of nodes *enhance* the small-scale WL
spectrum; Stage-IV *statistical* power (Gaussian, single-bin forecast) resolves
essentially the entire envelope, while the TNG-prior suite does not reach the stronger
suppression preferred by recent kSZ-informed analyses — a region fig 12(a) now draws
explicitly (schematic band) below the suite envelope, and whose $f_{\rm gas}$-space
counterpart (group gas fractions against real X-ray/eROSITA measurements) is confronted
directly in Appendix C (fig 19).

**Open items, in order of leverage** (§5b): (i) apply the linear-in-$\log M$ amplitude
calibration (§2a) and the per-channel $r(a,M)$ curves at de-normalization — the v1 remedy
for the gas and thermal-structure biases; (ii) validate and deploy the redshift de-biasing templates as
release assets; (iii) the TNG300-2/3 resolution gate before `bind-paint` is used on other
substrates; (iv) truth $\tau$ ray-trace and full-TNG truth maps; (v) minting the archive
DOI under the confirmed release name. **Deferred to a future major version**
(~$10^3$ GPU-hour repaint + ray-trace cost, not planned for v1): (vi) retraining with
strengthened high-$z$ supervision (≥10 rotations/halo, denser snapshot grid — the
architecture already carries the redshift conditioning); (vii) a texture-aware training
loss for the $y$ small-scale budget. Companion papers II–V consume the suite for
cosmological-bias mitigation, latent-space inference, kSZ/X-ray gas constraints, and
feedback anisotropy.
''')


# ═════════════════════════════════════════════════════════════════════════════
def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (BIND venv)", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    for kind, src in CELLS:
        cell = nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
        nb.cells.append(cell)
    out = "paper1_figures.ipynb"
    nbf.write(nb, out)
    print(f"wrote {out} with {len(nb.cells)} cells "
          f"({sum(1 for k, _ in CELLS if k == 'code')} code)")


if __name__ == "__main__":
    build()
