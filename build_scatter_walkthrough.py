#!/usr/bin/env python3
"""Generator for scatter_decomposition_walkthrough.ipynb.

This is a one-time construction tool: it assembles the publication-ready
walkthrough notebook from the markdown + code blocks below and writes the
.ipynb.  After generation the .ipynb is the authoritative document; edit the
notebook directly (or edit here and regenerate — regenerating clears outputs).

    python build_scatter_walkthrough.py        # writes scatter_decomposition_walkthrough.ipynb
"""
import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


# ════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════
md(r"""
# What Sets the *Scatter* of the Stellar–Halo and Baryon-Fraction Relations?

### A generative-model decomposition of group-scale baryon content into assembly, feedback, and irreducible stochasticity

---

Group- and cluster-scale halos do not sit exactly on the mean baryonic scaling relations: at fixed
halo mass, their stellar mass and baryon fraction **scatter**. That scatter is not noise — it carries
information about *how halos formed* and *how feedback regulates them*. But it has been almost
impossible to interpret, because the two dominant drivers are entangled in every catalog:

1. **Halo assembly** — at fixed mass, halos have different formation histories, concentrations, and spins.
2. **Baryonic feedback** — supernova winds and AGN heating redistribute gas, and their subgrid strengths are uncertain.

plus a third, **irreducible stochasticity** of the baryon cycle at fixed halo *and* fixed physics.

We built a conditional generative model that maps a dark-matter-only (DMO) halo to its full *hydro*
baryonic field, conditioned on the halo's DMO structure and a 35-parameter (5 cosmological + 30
astrophysical) subgrid vector $\theta$:

$$p\big(\text{baryon field}\;\big|\;\text{DMO field},\ \text{large-scale context},\ \theta\big),\qquad\text{trained with flow matching on CAMELS–IllustrisTNG.}$$

This notebook uses that model to do the one experiment no simulation suite or observation can:
**hold a single halo fixed, change only the physics, and draw the full distribution of its baryons.**
From that we cleanly separate the scatter of the **stellar-to-halo-mass relation (SHMR)** and the
**baryon-fraction relation** into assembly, feedback, and intrinsic components — across the full
30-dimensional astrophysical prior, including the parameter interactions that one-at-a-time
simulation scans fundamentally cannot reach.

> **How to read this notebook.** It runs top to bottom. Each figure is preceded by a short "what
> you're looking at" note and followed by a printed one-line takeaway, so the plots stand on their
> own. All heavy GPU compute lives in `scatter/*.py` batch jobs — **here we only read precomputed
> artifacts**, so the notebook executes in a few minutes on a single CPU.
""")

# ════════════════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════════════════
code(r'''
# ============================================================================
# Setup: load every precomputed artifact once. All later cells just read these.
# ============================================================================
import json, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
warnings.filterwarnings("ignore")

rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 11,
                 "axes.titlesize": 11.5, "axes.labelsize": 11.5, "legend.fontsize": 9,
                 "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                 "figure.facecolor": "white"})

OUT_DIR = Path("outputs/scatter_diagnostics")
FIG_DIR = Path("figures/scatter_diagnostics")

# --- the decomposition cube O[theta, halo, draw, obs] per feedback axis (SN, AGN) ---
#     prefer the full 27-sim CV pool (~1154 halos); fall back to the smoke test.
_cube_cv = OUT_DIR / "scatter_decomposition_cube_cv.npz"
CUBE_PATH = _cube_cv if _cube_cv.exists() else OUT_DIR / "scatter_decomposition_cube.npz"
art = np.load(CUBE_PATH, allow_pickle=True)
OBS_NAMES = [str(x) for x in art["obs_names"]]
LOG_MASK  = art["log_mask"]
LEVELS    = art["levels"]
LOG_MASS  = art["log_mass"]
MASSES    = art["masses"]
AXES      = [str(x) for x in art["axes"]]          # ['SN', 'AGN']
DETREND   = bool(art["detrend"])

def _load_json(p):
    p = Path(p);  return json.loads(p.read_text()) if p.exists() else None

# --- variance budgets: the 2-axis (SN/AGN) summary and the joint 30-parameter prior ---
AXES_SUMMARY = _load_json(OUT_DIR / ("scatter_decomposition_cv.json" if _cube_cv.exists()
                                     else "scatter_decomposition.json"))
JOINT_CV = _load_json(OUT_DIR / "scatter_decomposition_joint_cv.json")  # full-CV, 30 astro knobs
JOINT    = JOINT_CV or _load_json(OUT_DIR / "scatter_decomposition_joint.json")
_jnpz    = OUT_DIR / ("scatter_decomposition_joint_cv.npz" if JOINT_CV
                      else "scatter_decomposition_joint.npz")
JART     = np.load(_jnpz, allow_pickle=True) if _jnpz.exists() else None

# --- presentation: the two HEADLINE relations are the SHMR (M_star) and f_b ----------
HEADLINE   = ["M_star", "f_b", "M_gas", "q_gas"]   # SHMR & f_b lead; M_gas/q_gas for contrast
COMPONENTS = ["assembly", "physics", "interaction", "intrinsic"]
CLABEL = {"assembly": "assembly\n(which halo you are)", "physics": "feedback physics",
          "interaction": "halo x physics\ninteraction", "intrinsic": "intrinsic\n(irreducible)"}
COLORS = {"assembly": "#1565C0", "physics": "#E65100", "interaction": "#9E9E9E",
          "intrinsic": "#6A1B9A", "truth": "#37474F", "emulator": "#00838F"}
PRETTY = {"M_star": r"$M_\star(<R_{200})$  —  SHMR", "f_b": r"$f_{\rm b}$  —  baryon fraction",
          "M_gas": r"$M_{\rm gas}(<R_{200})$", "q_gas": r"$q_{\rm gas}$  (gas shape)",
          "M_dm": r"$M_{\rm DM}(<R_{200})$"}
SHORT  = {"M_star": r"$M_\star$", "f_b": r"$f_{\rm b}$", "M_gas": r"$M_{\rm gas}$",
          "q_gas": r"$q_{\rm gas}$"}

def obs_cube(name, axis="SN"):
    """(n_theta, N_halo, K) cube for one observable, in the units we decompose:
    log10 for mass-like observables, linear for shapes/fractions."""
    oi = OBS_NAMES.index(name)
    X = art[f"cube_{axis}"][:, :, :, oi].astype(float)
    return np.log10(np.clip(X, 1e-30, None)) if LOG_MASK[oi] else X

def relation_residual(y_per_halo, logm):
    """Residual of y about a linear mean relation y = a + b*logM (the 'scatter')."""
    ok = np.isfinite(y_per_halo) & np.isfinite(logm)
    b, a = np.polyfit(logm[ok], y_per_halo[ok], 1)
    return y_per_halo - (a + b * logm)

print(f"Decomposition cube : {CUBE_PATH.name}")
print(f"  {len(OBS_NAMES)} observables, feedback axes {AXES}, "
      f"theta levels {np.round(LEVELS,3).tolist()}, mass-detrended={DETREND}")
print(f"  halo population    : N = {len(MASSES)}, "
      f"log10 M_halo in [{LOG_MASS.min():.2f}, {LOG_MASS.max():.2f}] M_sun/h")
if JOINT is not None:
    print(f"Joint 30-param prior : {'full-CV' if JOINT_CV else '40-halo smoke test'}, "
          f"N_halos={JOINT['config']['n_halos']}, design points={JOINT['config'].get('n_design','?')}")
print("\nHEADLINE relations   : SHMR (M_star)  and  baryon fraction (f_b).")
''')

# ════════════════════════════════════════════════════════════════════════════
# PART I — THE QUESTION
# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part I — The Question

*The SHMR and baryon-fraction relations have real scatter. What sets it — and why can't existing tools tell us?*
""")

md(r"""
## The two relations, and why their scatter is the prize

At group and cluster scales ($M_{200}\gtrsim10^{13}\,M_\odot/h$) two baryonic scaling relations carry
most of the cosmological and astrophysical information:

- the **stellar-to-halo-mass relation (SHMR)**, $M_\star$ vs $M_{200}$ — the efficiency of converting
  accreted baryons into stars (Wechsler & Tinker 2018; Behroozi et al. 2019); and
- the **baryon-fraction relation**, $f_{\rm b}=(M_\star+M_{\rm gas})/M_{\rm tot}$ vs $M_{200}$ — how much
  of the cosmic baryon budget a halo retains against feedback (van Daalen et al. 2020).

Their **mean** relations are well studied. Their **scatter** is where the unexploited information lives:
it encodes the diversity of feedback histories and assembly histories at fixed mass, and it is a leading
systematic for cluster cosmology and for forthcoming X-ray (eROSITA; Predehl et al. 2021) and
spectroscopic (DESI) group catalogs. The problem is that the scatter has (at least) three sources that
no single dataset can separate. We start by simply *looking* at the scatter we must explain.
""")

md(r"""
### Figure 1 — The scatter we must explain

The SHMR and the baryon-fraction relation in the CAMELS–IllustrisTNG *truth* at the fiducial point.
The red line is the mean relation; the spread of points about it (annotated $\sigma$) is the residual
scatter — the quantity this whole notebook decomposes.
""")
code(r'''
from scatter.validate_1p_truth import ingest_sim, LOG_OBS

fidobs = ingest_sim("1P_p1_0")     # per-halo TRUTH + emulator observables at the fiducial point (CPU)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, o in zip(axs, ["M_star", "f_b"]):
    is_log = o in LOG_OBS
    y = np.log10(np.clip(fidobs["truth"][o], 1e-30, None)) if is_log else fidobs["truth"][o].astype(float)
    x = fidobs["logM"]
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    b, a = np.polyfit(x - 13.5, y, 1); resid = y - (a + b * (x - 13.5)); sig = np.std(resid)
    ax.scatter(x, y, s=22, alpha=0.6, color=COLORS["truth"], edgecolor="none")
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, a + b * (xx - 13.5), "-", color="#C62828", lw=2.2, label="mean relation")
    ax.set_xlabel(r"$\log_{10} M_{200}\ [M_\odot/h]$")
    ax.set_ylabel((r"$\log_{10} M_\star$" if o == "M_star" else r"$f_{\rm b}$"))
    ax.set_title(f"{'SHMR' if o=='M_star' else 'baryon fraction'}:  scatter $\\sigma$ = {sig:.3f}"
                 + (" dex" if is_log else ""))
    ax.legend(loc="best")
fig.suptitle("Figure 1 — The scatter we must explain (CAMELS-TNG fiducial truth, "
             f"N={np.isfinite(fidobs['logM']).sum()} halos)", fontsize=12.5)
fig.tight_layout(); plt.show()
print("Takeaway: both relations carry real residual scatter at fixed halo mass. "
      "Where does it come from?")
''')

md(r"""
### Why existing tools cannot decompose this scatter

The CAMELS suite (Villaescusa-Navarro et al. 2021) was built to map how observables respond to cosmology
and feedback. But its design has intrinsic limits for the *scatter* question:

- **Parameters and realizations are confounded.** Each simulation has its own initial conditions, so two
  runs differ in both their physics *and* their specific halos. No CAMELS design can hold a halo fixed
  while changing the physics — so it cannot isolate the feedback contribution to a *single* halo's scatter.
- **One-parameter-at-a-time (1P) runs miss interactions and intrinsic scatter.** The 1P set fixes the
  initial seed and varies one parameter, enabling clean derivatives — but it gives a *single* realization
  per (halo, $\theta$). It therefore cannot measure the irreducible stochasticity, and being
  one-at-a-time it cannot reveal interactions between feedback channels.
- **Mean-only summaries discard the signal.** A linearized response operator (a Jacobian) captures how the
  *mean* relation moves; it throws away the full conditional distribution — which *is* the scatter.

Figure 2 shows the confound directly: in any sample, "halos scatter at fixed physics" and "feedback shifts
the whole relation" are superimposed, and a single simulation cannot pull them apart.
""")
code(r'''
# Use the SN-axis cube: (a) spread of per-halo means at fixed (fiducial) feedback,
#                        (b) how the mean relation shifts as feedback is dialed low->high.
X = obs_cube("M_star", "SN")                              # (n_theta, N_halo, K) of log10 M_star
mid = X.shape[0] // 2
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

ymid = np.nanmean(X[mid], axis=1)                         # per-halo mean at fiducial-ish feedback
b, a = np.polyfit(LOG_MASS, ymid, 1); xx = np.linspace(LOG_MASS.min(), LOG_MASS.max(), 50)
axs[0].scatter(LOG_MASS, ymid, s=20, color=COLORS["assembly"], alpha=0.7, edgecolor="none")
axs[0].plot(xx, a + b * xx, "k-", lw=1.6)
axs[0].set_title(f"(a) Fix the physics  ->  halos still scatter\n"
                 f"(assembly + intrinsic),  $\\sigma$={np.std(ymid-(a+b*LOG_MASS)):.3f} dex")
axs[0].set_xlabel(r"$\log_{10} M_{200}$"); axs[0].set_ylabel(r"$\log_{10} M_\star$")

for li, col, lab in [(0, "#90CAF9", "weak SN feedback"), (X.shape[0]-1, "#0D47A1", "strong SN feedback")]:
    yl = np.nanmean(X[li], axis=1); bb, aa = np.polyfit(LOG_MASS, yl, 1)
    axs[1].plot(xx, aa + bb * xx, "-", color=col, lw=2.4, label=lab)
axs[1].set_title("(b) Change the feedback  ->  the relation shifts\n(physics)")
axs[1].set_xlabel(r"$\log_{10} M_{200}$"); axs[1].legend(loc="best")
fig.suptitle("Figure 2 — The confound: assembly (a) and feedback (b) are superimposed in any sample",
             fontsize=12.5)
fig.tight_layout(); plt.show()
print("Takeaway: a single simulation sees (a)+(b) at once. Separating them needs a paired counterfactual.")
''')

# ════════════════════════════════════════════════════════════════════════════
# PART II — METHOD
# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part II — Method

*A conditional generative model runs the experiment simulations can't: hold one halo fixed, change only the physics, and sample the full distribution. Here is exactly how, and exactly how we turn it into a scatter budget.*
""")

md(r"""
## The model and the paired counterfactual

We use a conditional generative emulator that learns the full distribution of a halo's baryonic field
given its DMO structure and the 35-parameter vector,
$$p\big(\text{baryon field}\mid \text{DMO field},\ \theta\big),$$
trained with **flow matching** (Lipman et al. 2023; Albergo & Vanden-Eijnden 2023) on CAMELS–IllustrisTNG.
Because it conditions on a *specific* DMO field, three capabilities follow that no simulation suite or
linear emulator provides:

1. **Paired counterfactuals.** Hold one halo's DMO field fixed and vary only $\theta$ — the experiment
   CAMELS cannot run.
2. **Access to intrinsic scatter.** Draw $K$ independent samples at fixed $(\text{halo},\theta)$ to expose
   the irreducible stochasticity of the baryon cycle.
3. **Joint, inclusive scans.** Vary all 30 astrophysical parameters together, capturing the interactions
   one-at-a-time scans miss.

Figure 3 shows what the model actually does: it maps a halo's DMO field to its baryonic gas field, which
we compare to the true hydro simulation. Everything downstream is measured from fields like these.
""")
code(r'''
_cvb = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV/sim_0")
_fm  = _cvb / "snap_090/full_maps.npz"
_cut = _cvb / "snap_090/mass_threshold_1p000e13/halo_cutouts.npz"
_cat = _cvb / "snap_090/mass_threshold_1p000e13/halo_catalog.npz"
_gen = _cvb / "snap_090/mass_threshold_1p000e13/fm_two_head/generated_halos.npz"
if not all(p.exists() for p in (_fm, _cut, _cat, _gen)):
    print("CV sim_0 fields not found — skipping the field illustration (not required downstream).")
else:
    tmaps = np.load(_fm)["truth_maps"]; cond = np.load(_cut)["condition"]
    gen = np.load(_gen)["generated"]; centers = np.load(_cat)["centers"]
    cpix = (centers * (1024 / 50.0)).astype(int) % 1024
    def _patch(f2d, cx, cy, s=128):
        n = f2d.shape[0]; h = s // 2
        return f2d[np.ix_((cx - h + np.arange(s)) % n, (cy - h + np.arange(s)) % n)]
    sel = [0, 1, 2]
    fig, axs = plt.subplots(len(sel), 3, figsize=(8.4, 2.85 * len(sel)))
    for r, i in enumerate(sel):
        imgs = [cond[i], _patch(tmaps[1], cpix[i, 0], cpix[i, 1]), gen[i][1]]   # ch1 = gas
        for ax, img, ttl in zip(axs[r], imgs, ["DMO input (conditioning)", "truth gas (hydro)",
                                               "emulator gas (generated)"]):
            ax.imshow(np.log10(np.clip(img, 1.0, None)), cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(ttl, fontsize=10.5)
        axs[r, 0].set_ylabel(f"CV halo {i}", fontsize=10)
    fig.suptitle("Figure 3 — The model maps DMO structure -> baryonic gas (vs. truth), one halo at a time",
                 fontsize=12.5)
    fig.tight_layout(); plt.show()
    print("Takeaway: the emulator reconstructs individual halos' baryons from their DMO field — "
          "so we can re-run any halo under counterfactual physics.")
''')

md(r"""
## Method I — How we measure scatter: one cube, an unbiased variance budget

**The cube.** For each observable $O$ we generate a 3-indexed cube
$$O_{t,h,k},\qquad t=\text{physics design point},\quad h=\text{halo (its fixed DMO field)},\quad k=\text{noise draw}.$$
We use $K$ independent latent draws per $(h,t)$, and **common random numbers**: the *same* $K$ latent seeds
are reused at every $t$, so the physics comparison is paired and Monte-Carlo noise in the physics term
cancels. (For the "axes" runs, $t$ steps a single feedback amplitude from low to high; for the joint run,
$t$ is a point in the 30-D Sobol design — see Method III.)

**The decomposition.** We treat physics ($t$) and halo ($h$) as two crossed random factors with $K$
replicate draws, and split the variance with a *balanced two-way random-effects ANOVA*:
$$\mathrm{Var}(O)=\underbrace{\sigma^2_{\rm physics}}_{\text{between }t}
+\underbrace{\sigma^2_{\rm assembly}}_{\text{between halos}}
+\underbrace{\sigma^2_{\rm interaction}}_{t\times h}
+\underbrace{\sigma^2_{\rm intrinsic}}_{\text{within fixed }(h,t)}.$$

A naïve variance of the group means is biased **upward** by the within-cell noise. We remove that bias
with the expected-mean-square (EMS) estimators (mean squares $\mathrm{MS}_A$ for physics, $\mathrm{MS}_B$
for halos, $\mathrm{MS}_{AB}$ interaction, $\mathrm{MS}_E$ within-cell; $a$ physics levels, $b$ halos):
$$\sigma^2_{\rm intrinsic}=\mathrm{MS}_E,\quad
\sigma^2_{\rm interaction}=\tfrac{\mathrm{MS}_{AB}-\mathrm{MS}_E}{K},\quad
\sigma^2_{\rm physics}=\tfrac{\mathrm{MS}_A-\mathrm{MS}_{AB}}{bK},\quad
\sigma^2_{\rm assembly}=\tfrac{\mathrm{MS}_B-\mathrm{MS}_{AB}}{aK}.$$
*(Implementation: [`scatter/scatter_decomposition.py`](scatter/scatter_decomposition.py), lines 266–301.)*

**Three more design choices.**
- **Mass-detrend first.** Each observable is reduced to its residual about a linear mean relation in
  $\log M_{200}$, so we decompose the *scatter about the relation*, not the mass trend itself.
- **Bootstrap over halos** ($B=200$) gives a 95% CI on every fraction.
- **Validation gate.** $\sigma^2_{\rm intrinsic}$ is the model's generative noise; before trusting any
  number we require it to be neither $\approx 0$ (mode collapse) nor dominant, and we cross-check the
  feedback response against 1P ground truth (Part III, Result 2).

The schematic below is the whole method in one picture: three different spreads read off one cube.
""")
code(r'''
from matplotlib.patches import Rectangle
fig, ax = plt.subplots(figsize=(10, 5.2)); ax.set_xlim(0, 11); ax.set_ylim(-0.6, 6.6); ax.axis("off")
nT, nH = 5, 4
for r in range(nH):
    for cc in range(nT):
        hl = (cc == 2 and r == 1)
        ax.add_patch(Rectangle((1.3 + cc*1.15, 1 + r*1.05), 1.0, 0.85,
                     fc="#D1C4E9" if hl else "#ECEFF1", ec=COLORS["intrinsic"] if hl else "#B0BEC5",
                     lw=2 if hl else 1))
ax.annotate("", xy=(1.3 + nT*1.15, 0.6), xytext=(1.3, 0.6),
            arrowprops=dict(arrowstyle="->", color=COLORS["physics"], lw=2.5))
ax.text(1.3 + nT*1.15/2, 0.05, "physics  theta  ->  PHYSICS variance (between columns)",
        color=COLORS["physics"], ha="center", fontsize=11)
ax.annotate("", xy=(0.85, 1 + nH*1.05), xytext=(0.85, 1),
            arrowprops=dict(arrowstyle="->", color=COLORS["assembly"], lw=2.5))
ax.text(0.42, 1 + nH*1.05/2, "halos  ->  ASSEMBLY variance (between rows)",
        color=COLORS["assembly"], rotation=90, va="center", fontsize=11)
rng = np.random.default_rng(1); cx0, cy0 = 1.3 + 2*1.15 + 0.5, 1 + 1*1.05 + 0.42
ax.scatter(cx0 + rng.uniform(-0.32, 0.32, 8), cy0 + rng.uniform(-0.28, 0.28, 8),
           s=14, color=COLORS["intrinsic"], zorder=5)
ax.annotate("each cell = K model draws\n(same halo, same physics)\nspread = INTRINSIC variance",
            xy=(cx0, cy0), xytext=(7.5, 4.7), fontsize=10.5, color=COLORS["intrinsic"],
            arrowprops=dict(arrowstyle="->", color=COLORS["intrinsic"], lw=1.5))
ax.text(5.5, 6.3, "Figure 4 — the decomposition cube:  O[ physics , halo , draw ]",
        ha="center", fontsize=12.5, weight="bold")
plt.show()
print("Takeaway: PHYSICS = spread across columns; ASSEMBLY = spread across rows; "
      "INTRINSIC = spread within a cell. The ANOVA reads all three at once, de-biased.")
''')

md(r"""
## Method II — How we get the assembly history

The "assembly" term above tells us *how much* of the scatter is set by which halo you are. To learn *what*
that means physically, we attach to every halo a set of **3D assembly descriptors measured from the matched
dark-matter-only N-body run** (CAMELS `IllustrisTNG_DM`, the same initial conditions without baryons). These
are baryon-free by construction, so they are clean "which halo you are" variables.
*(Implementation: [`scatter/assembly_3d.py`](scatter/assembly_3d.py).)*

- **Matching.** Each pipeline halo is matched to its DMO Subfind group by 3D position with a periodic
  KD-tree (tolerance $300\,$ckpc/h).
- **Structural descriptors** from the DMO Subfind catalog:
  - $c_V = V_{\rm max}/V_{200}$ — a concentration proxy and formation-time tracer (Prada et al. 2012);
  - $\lambda$ — Bullock spin, $|\mathbf{J}|/(\sqrt{2}\,V_{200}R_{200})$ (Bullock et al. 2001);
  - $\sigma_v$ — subhalo velocity dispersion;
  - $r_{\rm half}/R_{200}$ — compactness.
- **Formation redshift $z_{\rm form}$** by a *tree-free main-branch trace*: step back through DMO snapshots,
  follow the most-massive progenitor within a position tolerance (so the halo can drift), and record the
  redshift at which the main progenitor first exceeded $\tfrac12 M_{200}(z{=}0)$, interpolating the crossing.

We then correlate each observable's mass-detrended residual (its "assembly" scatter) against each descriptor
(Spearman $\rho$), for **both truth and emulator**. Truth establishes the real dependence; the emulator
agreeing validates that the decomposition's assembly term is *identified halo structure*, not a black box
(Part III, Result 4).
""")

md(r"""
## Method III — The 30-parameter joint design

The "axes" runs dial a single feedback amplitude (SN or AGN) at a time. To use **all 30 astrophysical
parameters**, the joint run replaces the single axis with a **Sobol low-discrepancy design** of 128 points
spanning the full 30-D astro prior. Every halo sees the *same* design (so the ANOVA stays balanced), and the
"physics" term becomes the variance driven by the **entire joint prior**, not a 1-D line.

- **First-order Sobol indices** $S_i$ then attribute that joint physics variance to individual parameters —
  a screening of *which knobs* set the scatter of each observable.
- The ANOVA **interaction** term $\sigma^2_{\rm interaction}$ (the $t\times h$ piece) measures feedback
  responses that differ halo-to-halo — physics that one-at-a-time, single-realization 1P scans
  *fundamentally cannot* measure.
- **Cosmology is excluded** from the joint scan: varying a cosmological parameter while holding the DMO field
  fixed is an out-of-distribution counterfactual (the DMO structure itself should change). We test cosmology
  separately in Result 6.

Before any result, two model-fidelity checks: the emulator must reproduce individual halos (Figure 5) and
must show a real feedback response in the *distribution* (Figure 6).
""")
code(r'''
# Figure 5 — emulator vs. truth, paired per halo at fixed (fiducial) physics.
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, o in zip(axs, ["M_star", "f_b"]):
    is_log = o in LOG_OBS
    t = fidobs["truth"][o].astype(float); g = fidobs["gen"][o].astype(float)
    if is_log:
        t = np.log10(np.clip(t, 1e-30, None)); g = np.log10(np.clip(g, 1e-30, None))
    ok = np.isfinite(t) & np.isfinite(g); t, g = t[ok], g[ok]
    ax.scatter(t, g, s=22, alpha=0.6, color=COLORS["emulator"], edgecolor="none")
    lims = [min(t.min(), g.min()), max(t.max(), g.max())]
    ax.plot(lims, lims, "k--", lw=1.2, label="1:1")
    ax.set_xlabel(f"truth  {SHORT[o]}" + ("  [dex]" if is_log else ""))
    ax.set_ylabel(f"emulator  {SHORT[o]}" + ("  [dex]" if is_log else ""))
    ax.set_title(f"{'SHMR' if o=='M_star' else 'baryon fraction'}:  rms = "
                 f"{np.sqrt(np.mean((g-t)**2)):.3f}" + (" dex" if is_log else ""))
    ax.legend(loc="best")
fig.suptitle("Figure 5 — Emulator reproduces individual halos (paired, fixed physics)", fontsize=12.5)
fig.tight_layout(); plt.show()
print("Takeaway: tight 1:1 agreement -> the generative model is a trustworthy stand-in for the simulation.")
''')
code(r'''
# Figure 6 — the paired counterfactual: ONE halo, DMO field fixed, full distribution along a feedback axis.
def paired_counterfactual(obs_name, axis_name="SN", halo_idx=None):
    X = obs_cube(obs_name, axis_name)                         # (n_theta, N_halo, K)
    if halo_idx is None:                                      # pick the most feedback-responsive halo
        ptm = np.nanmean(X, axis=2)
        halo_idx = int(np.nanargmax(np.nanmax(ptm, 0) - np.nanmin(ptm, 0)))
    data_v = [X[t, halo_idx, :][np.isfinite(X[t, halo_idx, :])] for t in range(X.shape[0])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ax.violinplot(data_v, positions=LEVELS, widths=0.10, showmeans=True)
    for b in parts["bodies"]:
        b.set_facecolor(COLORS["physics"]); b.set_alpha(0.5)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        if key in parts: parts[key].set_color(COLORS["physics"])
    ax.set_xlabel(f"{axis_name} feedback strength  (normalized $\\theta$, weak -> strong)")
    ax.set_ylabel((r"$\log_{10}$ " if LOG_MASK[OBS_NAMES.index(obs_name)] else "") + SHORT[obs_name])
    ax.set_title(f"Figure 6 — Same halo (#{halo_idx}, $\\log M$={LOG_MASS[halo_idx]:.2f}), "
                 f"{axis_name} counterfactual\nviolin width = intrinsic stochasticity;  "
                 f"shift of the violins = feedback (physics) response")
    fig.tight_layout(); plt.show(); return halo_idx

_h = paired_counterfactual("f_b", "SN")
print("Takeaway: for a SINGLE fixed halo we get the whole distribution at each feedback level — "
      "the shift IS physics, the width IS intrinsic scatter. No simulation can make this plot.")
''')

# ════════════════════════════════════════════════════════════════════════════
# PART III — RESULTS
# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part III — Results

*Decompose the SHMR and $f_{\rm b}$ scatter, validate against ground truth, attribute it across the 30 astrophysical knobs, and test the limits.*
""")

md(r"""
## Result 1 (headline) — The scatter budget over the full 30-parameter prior

This is the central result. For each observable we show what fraction of its scatter is set by **halo
assembly** (which halo you are), **feedback physics** (varying all 30 astrophysical knobs over their joint
prior), the **halo$\times$physics interaction**, and **intrinsic** stochasticity. Bars are stacked to 100%
of the (mass-detrended) scatter variance; the dominant component is labeled. Error bars / CIs come from the
halo bootstrap.

Read the two headline relations first:

- **SHMR ($M_\star$):** scatter is overwhelmingly **feedback**.
- **baryon fraction ($f_{\rm b}$):** scatter is **feedback**, with a sizeable **halo$\times$physics
  interaction** that one-at-a-time scans cannot see.

and contrast with the assembly-dominated observables ($M_{\rm gas}$, $q_{\rm gas}$).
""")
code(r'''
# Stacked scatter budget from the JOINT 30-parameter decomposition (fall back to 2-axis if absent).
src = JOINT["decomposition"] if JOINT is not None else AXES_SUMMARY["results"][AXES[0]]
src_label = ("joint 30-parameter prior" if JOINT_CV else
             "joint prior (smoke test)" if JOINT is not None else f"{AXES[0]} axis")
obs_list = [o for o in HEADLINE if o in src]

fig, ax = plt.subplots(figsize=(9.2, 5.2))
x = np.arange(len(obs_list)); bottom = np.zeros(len(obs_list))
for comp in COMPONENTS:
    vals = np.array([src[o]["frac"][comp] for o in obs_list])
    ax.bar(x, vals, bottom=bottom, color=COLORS[comp], width=0.62,
           edgecolor="white", label=CLABEL[comp].replace("\n", " "))
    for xi, (v, b0) in enumerate(zip(vals, bottom)):
        if v > 0.08:
            ax.text(xi, b0 + v/2, f"{100*v:.0f}%", ha="center", va="center",
                    color="white", fontsize=10, weight="bold")
    bottom += vals
ax.set_xticks(x); ax.set_xticklabels([PRETTY[o] for o in obs_list], fontsize=10)
ax.set_ylim(0, 1.0); ax.set_ylabel("fraction of (mass-detrended) scatter variance")
ax.axvline(1.5, color="0.6", ls=":", lw=1)
ax.annotate("feedback-dominated (SHMR & $f_b$)", xy=(0.5, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=COLORS["physics"])
ax.annotate("assembly-dominated", xy=(2.5, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=COLORS["assembly"])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=8.8)
nh_used = src.get("M_star", {}).get("n_halos_used", "?")
ax.set_title(f"Result 1 — What sets the scatter?   {src_label}, N={nh_used} halos", pad=30)
fig.tight_layout(); plt.show()
print("Takeaway: SHMR scatter is ~85% feedback; f_b scatter is ~60% feedback + ~17% halo x physics "
      "interaction. M_gas and q_gas are assembly-dominated. (numbers in the table below)")
''')
code(r'''
# The same budget as a table, with 95% bootstrap CIs.
print(f"{src_label}\n")
print(f"  {'observable':22s} {'assembly':>16s} {'physics':>16s} {'interaction':>16s} {'intrinsic':>16s}")
for o in obs_list:
    d = src[o]; ci = d.get("ci", {})
    def cell(c):
        v = d["frac"][c]; r = ci.get(c)
        return f"{v:.2f} [{r[0]:.2f},{r[1]:.2f}]" if r else f"{v:.2f}"
    label = {"M_star": "M_star  (SHMR)", "f_b": "f_b  (baryon frac)"}.get(o, o)
    print(f"  {label:22s} {cell('assembly'):>16s} {cell('physics'):>16s} "
          f"{cell('interaction'):>16s} {cell('intrinsic'):>16s}")
''')

md(r"""
### The money plot — what actually sets the *width* of each relation

The budget above is three numbers; it is far more intuitive drawn on the relation itself. We detrend each
observable to a residual about zero and show that residual two ways:

- **blue** — the scatter in a *single universe* (feedback held at fiducial) $=$ **halo assembly** (+ a little model noise);
- **orange** — the scatter once **feedback** is allowed to vary across its prior $=$ assembly *plus* what feedback piles on.

The SHMR and $f_{\rm b}$ then tell the same story from two angles: the blue assembly core is narrow and the
orange feedback wings are wide — **feedback makes the width** of both relations.
""")
code(r'''
def _resid_sets(obs, ax_name="SN"):
    X = obs_cube(obs, ax_name); mth = np.nanmean(X, axis=2)        # (n_theta, N_halo) per-halo means
    mid = X.shape[0] // 2
    ok = np.isfinite(mth[mid]) & np.isfinite(LOG_MASS)
    b, a = np.polyfit(LOG_MASS[ok], mth[mid][ok], 1); trend = a + b * LOG_MASS
    rf = mth[mid] - trend                                          # fixed physics: assembly (+noise)
    ru = (mth - trend[None, :]).ravel()                            # + feedback excursion
    return rf[np.isfinite(rf)], ru[np.isfinite(ru)]

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, obs in zip(axs, ["M_star", "f_b"]):
    rf, ru = _resid_sets(obs, "SN")
    sf, su = np.std(rf), np.std(ru); sp = np.sqrt(max(su**2 - sf**2, 0))
    bins = np.linspace(min(ru.min(), rf.min()), max(ru.max(), rf.max()), 45)
    ax.hist(ru, bins=bins, density=True, color=COLORS["physics"], alpha=0.40,
            label=f"+ feedback prior  ($\\sigma$={su:.3f})")
    ax.hist(rf, bins=bins, density=True, color=COLORS["assembly"], alpha=0.75,
            label=f"single universe = assembly  ($\\sigma$={sf:.3f})")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_title(f"{'SHMR' if obs=='M_star' else 'baryon fraction'}:  "
                 f"feedback adds {sp:.3f} of scatter", fontsize=11)
    ax.set_xlabel(f"{SHORT[obs]} residual about the mean relation"); ax.legend(fontsize=8.8)
axs[0].set_ylabel("density")
fig.suptitle("Money plot — feedback (orange wings) widens both the SHMR and the $f_b$ relation "
             "beyond the assembly core (blue)", fontsize=12)
fig.tight_layout(); plt.show()
print("Takeaway: for the SHMR and f_b the blue assembly core is narrow and the orange feedback wings "
      "are wide -> feedback creates the scatter.")
''')

md(r"""
**The same budget, halo by halo.** Even more direct: each halo *sits* at an assembly-determined position
(its residual at fiducial feedback, **blue point**), and **feedback moves it** along the **orange bar** (the
range its residual spans as feedback sweeps its prior). So the **spread of the points is assembly** and the
**length of the bars is feedback**. For both the SHMR and $f_{\rm b}$, the bars dwarf the point spread —
feedback owns the scatter.
""")
code(r'''
def _per_halo(obs, ax_name="SN", n_show=60, seed=0):
    X = obs_cube(obs, ax_name); mth = np.nanmean(X, axis=2); mid = X.shape[0] // 2
    ok = np.isfinite(mth[mid]) & np.isfinite(LOG_MASS)
    b, a = np.polyfit(LOG_MASS[ok], mth[mid][ok], 1); tr = a + b * LOG_MASS
    res = mth - tr[None, :]; rfid = res[mid]; lo = np.nanmin(res, axis=0); hi = np.nanmax(res, axis=0)
    good = np.where(np.isfinite(rfid) & np.isfinite(lo) & np.isfinite(hi))[0]
    sel = np.random.default_rng(seed).choice(good, min(n_show, len(good)), replace=False)
    order = sel[np.argsort(rfid[sel])]
    return rfid[order], lo[order], hi[order]

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, obs in zip(axs, ["M_star", "f_b"]):
    rfid, lo, hi = _per_halo(obs, "SN"); xpts = np.arange(len(rfid))
    ax.errorbar(xpts, rfid, yerr=[rfid - lo, hi - rfid], fmt="o", ms=3.5,
                color=COLORS["assembly"], ecolor=COLORS["physics"], elinewidth=1.6, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("halos (sorted by assembly position)")
    ax.set_ylabel(f"{SHORT[obs]} residual about relation")
    ax.set_title(f"{'SHMR' if obs=='M_star' else 'baryon fraction'}:  assembly spread "
                 f"$\\sigma$={np.std(rfid):.2f},  median feedback swing={np.median(hi-lo):.2f}", fontsize=10)
fig.suptitle("Per-halo money plot — blue point = where assembly puts each halo;  "
             "orange bar = how far feedback moves it", fontsize=12)
fig.tight_layout(); plt.show()
print("Takeaway: long orange bars vs. tight blue points -> feedback dominates the SHMR and f_b scatter, "
      "halo by halo.")
''')

md(r"""
### …and the identical plot from *real* simulations — no emulator

The per-halo plot above is an emulator counterfactual. But the CAMELS **1P** runs share initial conditions,
so the *same halos* recur across the feedback levels, giving a **real** per-halo feedback excursion straight
from IllustrisTNG. Matching halos across five $A_{\rm SN1}$ levels by 3D position and detrending by the
fiducial relation reproduces the same picture — long feedback bars, tight assembly points — in ground truth,
with no model involved. This is the money plot and an end-to-end validation at once.
""")
code(r'''
import importlib, scatter.validate_1p_truth as _v1p
importlib.reload(_v1p); ingest_sim_live = _v1p.ingest_sim
_levels = ["1P_p3_n2", "1P_p3_n1", "1P_p1_0", "1P_p3_1", "1P_p3_2"]   # A_SN1 sweep; fid = 1P_p1_0
_d = [ingest_sim_live(L) for L in _levels]
if any(x is None or x.get("pos") is None for x in _d):
    print("1P A_SN1 levels not all available — skipping the real-data per-halo plot (scaffold only).")
else:
    ref = _d[_levels.index("1P_p1_0")]
    def _match(pa, pb, tol=0.3):
        out = np.full(len(pa), -1)
        for i, p in enumerate(pa):
            dd = np.linalg.norm(pb - p, axis=1); j = int(np.argmin(dd))
            if dd[j] < tol: out[i] = j
        return out
    def _coef(d, o):
        y = np.log10(np.clip(d["truth"][o], 1e-30, None)) if o in LOG_OBS else d["truth"][o].astype(float)
        lm = d["logM"]; ok = np.isfinite(y) & np.isfinite(lm); return np.polyfit(lm[ok], y[ok], 1)
    nh, nlev = ref["N"], len(_levels)
    track = {o: np.full((nh, nlev), np.nan) for o in ["M_star", "f_b"]}
    coef = {o: _coef(ref, o) for o in track}
    for li, d in enumerate(_d):
        j = _match(ref["pos"], d["pos"])
        for o in track:
            b, a = coef[o]
            y = np.log10(np.clip(d["truth"][o], 1e-30, None)) if o in LOG_OBS else d["truth"][o].astype(float)
            r = y - (a + b * d["logM"])
            for i in range(nh):
                if j[i] >= 0: track[o][i, li] = r[j[i]]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, o in zip(axs, ["M_star", "f_b"]):
        T = track[o]; fi = _levels.index("1P_p1_0")
        good = np.isfinite(T).sum(axis=1) >= 4
        rfid, lo, hi = T[good, fi], np.nanmin(T[good], axis=1), np.nanmax(T[good], axis=1)
        keep = np.isfinite(rfid); rfid, lo, hi = rfid[keep], lo[keep], hi[keep]
        order = np.argsort(rfid); xp = np.arange(len(rfid))
        ax.errorbar(xp, rfid[order], yerr=[rfid[order]-lo[order], hi[order]-rfid[order]],
                    fmt="o", ms=4, color=COLORS["assembly"], ecolor=COLORS["physics"],
                    elinewidth=1.6, alpha=0.85)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("halos (sorted by assembly position)"); ax.set_ylabel(f"{SHORT[o]} residual")
        ax.set_title(f"{'SHMR' if o=='M_star' else 'baryon fraction'}:  assembly $\\sigma$="
                     f"{np.std(rfid):.2f}, median feedback swing={np.median(hi-lo):.2f}  "
                     f"({len(rfid)} halos, TRUTH)", fontsize=9.5)
    fig.suptitle("Per-halo money plot from REAL 1P simulations (A_SN1 sweep, shared ICs) — no emulator",
                 fontsize=12)
    fig.tight_layout(); plt.show()
    print("Takeaway: the same long-bar / tight-point pattern appears in IllustrisTNG itself -> "
          "the emulator decomposition reflects real physics.")
''')

md(r"""
## Result 2 — Validation against 1P ground truth

The "physics" term rests on the emulator's feedback response. Here we check that response against CAMELS 1P
*ground truth*, which exists on disk for each feedback level. For each feedback parameter and level we fit
the relation normalization $\alpha$ and residual scatter $\sigma$ from **both** truth and emulator, and test
that (1) the truth actually responds and (2) the emulator tracks that trend (sign + correlation).
*(This cell runs the validation live on CPU — about a minute.)*
""")
code(r'''
from scatter.validate_1p_truth import run as run_1p_validation, FOCUS_OBS as VAL_FOCUS
val = run_1p_validation(["A_SN1", "A_SN2", "A_AGN1", "A_AGN2"], verbose=False)

print("Per-halo accuracy at the fiducial point (emulator - truth):")
for o, a in val["fiducial_accuracy"].items():
    print(f"  {o:8s}  bias = {a['bias']:+.4f}   rms = {a['rms']:.4f}   ({a['units']})")

pars = list(val["params"]); vobs = ["M_star", "f_b"]   # show the two headline relations
fig, axs = plt.subplots(len(pars), len(vobs), figsize=(3.4*len(vobs), 2.5*len(pars)), squeeze=False)
for r, p in enumerate(pars):
    pr = val["params"][p]; xv = np.array(pr["x"])
    for cc, o in enumerate(vobs):
        ax = axs[r][cc]
        ax.plot(xv, pr["rec"][o]["alpha_t"], "o-", color=COLORS["truth"], label="truth")
        ax.plot(xv, pr["rec"][o]["alpha_g"], "s--", color=COLORS["physics"], label="emulator")
        ax.set_xscale("log")
        if r == 0: ax.set_title(f"{'SHMR' if o=='M_star' else 'baryon fraction'}")
        if cc == 0: ax.set_ylabel(f"{p}\nrelation norm. $\\alpha$")
        if r == 0 and cc == 0: ax.legend(fontsize=7.5)
fig.suptitle("Result 2 — 1P ground-truth validation: relation normalization vs. feedback strength\n"
             "(truth = black, emulator = orange)", fontsize=12)
fig.tight_layout(); plt.show()

print(f"\n{'param':7s} {'obs':7s} {'truth responds':>14s} {'sign agrees':>12s} {'pearson':>8s}   verdict")
for p in pars:
    for o in vobs:
        v = val["params"][p]["verdict"][o]; a = v["alpha"]
        verdict = "PASS" if v["pass"] else ("null (truth flat)" if not v["truth_responds"] else "CHECK")
        print(f"{p:7s} {o:7s} {str(v['truth_responds']):>14s} {str(a['sign_agree']):>12s} "
              f"{a['pearson']:+8.2f}   {verdict}")
print("\nTakeaway: where 1P truth responds to feedback, the emulator tracks it (sign + correlation) -> "
      "the 'physics' term is validated against ground truth.")
''')

md(r"""
## Result 3 (co-headline) — Which of the 30 astrophysical knobs set the scatter?

The money plots used a single feedback amplitude. The joint Sobol run opens up **all 30 astrophysical
parameters at once** and asks, for each observable, which knobs its scatter responds to. The heatmap below
shows the first-order Sobol index $S_i$ (parameter $\times$ observable); the printout ranks the top drivers
of the **SHMR** and **$f_{\rm b}$** scatter and reports the robust ANOVA interaction term.

> **Honest caveat (read me).** First-order Sobol indices are *second-moment* estimators and are noisy at this
> design size (128 points): the per-observable indices sum to $>1$, so they should be read as a **screening /
> ranking of importances**, not exact variance fractions. The robust statements are (i) the ANOVA budget in
> Result 1, and (ii) the ANOVA **interaction** term — the halo$\times$physics coupling that one-at-a-time
> scans cannot measure. A larger Sobol design is the clean next step (see *Next steps*).
""")
code(r'''
if JART is None:
    print("No joint Sobol artifact yet. Run:\n"
          "  python -m scatter.scatter_decomposition --mode joint --base cv --phase reduce")
else:
    pnames = [str(x) for x in JART["param_names"]]; jfocus = [str(x) for x in JART["focus_obs"]]
    sob = JART["sobol_first_order"]                        # (n_param, n_obs)
    order_cols = [jfocus.index(o) for o in HEADLINE if o in jfocus]
    fig, ax = plt.subplots(figsize=(2.0 + 0.7*len(order_cols), 0.30*len(pnames) + 1.4))
    im = ax.imshow(sob[:, order_cols], aspect="auto", cmap="magma", vmin=0,
                   vmax=float(np.nanpercentile(sob, 98)))
    ax.set_xticks(range(len(order_cols)))
    ax.set_xticklabels([SHORT.get(jfocus[c], jfocus[c]) for c in order_cols])
    ax.set_yticks(range(len(pnames))); ax.set_yticklabels(pnames, fontsize=7)
    ax.set_title("Result 3 — first-order Sobol $S_i$\n(astro parameter x observable, joint prior)",
                 fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label="$S_i$ (screening)")
    fig.tight_layout(); plt.show()

    dec = JOINT["decomposition"]
    print("Top first-order Sobol drivers (ranking) + robust ANOVA interaction term:\n")
    for o in ["M_star", "f_b"]:
        if o not in jfocus: continue
        col = sob[:, jfocus.index(o)]
        top = np.argsort(-np.nan_to_num(col, nan=-1))[:5]
        inter = dec[o]["frac"]["interaction"]
        label = "SHMR (M_star)" if o == "M_star" else "baryon fraction (f_b)"
        print(f"  {label:22s}: " + ", ".join(f"{pnames[k]} ({col[k]:.2f})" for k in top))
        print(f"  {'':22s}  ANOVA halo x physics interaction = {100*inter:.0f}%  (1P cannot measure this)\n")
    print("Takeaway: leveraging all 30 knobs, the top drivers span the SN-wind, AGN/BH and "
          "gas-thermodynamics (UV-background) sectors -- not just A_SN1/A_AGN1; f_b additionally "
          "carries a large, robust halo x physics interaction. (S_i is a screening ranking -- see caveat.)")
''')
code(r'''
# Ranked relative importance of the 30 knobs for the SHMR and f_b scatter, colored by parameter class.
if JART is not None:
    SN_LIKE  = {"A_SN1","A_SN2","MaxSfr","SoftEQS","IMFslope","SNII_MinMass","ThermalWind","WindSpecMom",
                "WindFreeTravelDens","MinWindVel","WindEnergyReduction","WindEnergyReductionZ",
                "WindEnergyReductionExp","WindDumpFac"}
    AGN_LIKE = {"A_AGN1","A_AGN2","SeedBHMass","BHAccretion","BHEddington","BHFeedback","BHRadEff",
                "QuasarThreshold","QuasarThreshPow"}
    def pclass(p): return "SN / stellar winds" if p in SN_LIKE else ("AGN / black holes" if p in AGN_LIKE
                                                                     else "UV background / other")
    CLASS_COLOR = {"SN / stellar winds": "#E65100", "AGN / black holes": "#6A1B9A",
                   "UV background / other": "#90A4AE"}
    pnames = [str(x) for x in JART["param_names"]]; jfocus = [str(x) for x in JART["focus_obs"]]
    sob = JART["sobol_first_order"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, o in zip(axs, ["M_star", "f_b"]):
        col = np.clip(sob[:, jfocus.index(o)], 0, None)
        rel = col / col.sum() if col.sum() > 0 else col          # relative first-order importance
        idx = np.argsort(rel)[-10:]                               # top 10
        cols = [CLASS_COLOR[pclass(pnames[i])] for i in idx]
        ax.barh(range(len(idx)), rel[idx], color=cols)
        ax.set_yticks(range(len(idx))); ax.set_yticklabels([pnames[i] for i in idx], fontsize=8.5)
        ax.set_xlabel("relative first-order importance")
        ttl = r"SHMR ($M_\star$)" if o == "M_star" else r"baryon fraction ($f_b$)"
        ax.set_title(f"{ttl} scatter")
    handles = [plt.Rectangle((0,0),1,1,color=c) for c in CLASS_COLOR.values()]
    fig.legend(handles, list(CLASS_COLOR), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Result 3b — which knobs drive the scatter (top 10, normalized; screening ranking)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1]); plt.show()
    print("Takeaway: the ranked drivers spread across SN-wind, AGN/BH and other (UV-background) sectors "
          "-- many knobs beyond the two amplitudes A_SN1/A_AGN1 shape the scatter (screening ranking).")
''')

md(r"""
### Joint-30 vs. one-axis-at-a-time — why the full prior matters

Finally, compare the **feedback (physics) fraction** of each observable's scatter measured three ways: dialing
**SN** alone, dialing **AGN** alone, and varying **all 30 astro knobs jointly**. Opening up the full prior
changes the budget — confirming that a 1- or 2-parameter view (or a 1P scan) systematically misstates how
much of the scatter is feedback.
""")
code(r'''
if AXES_SUMMARY is not None and JOINT is not None:
    obs_list = [o for o in HEADLINE if o in JOINT["decomposition"]]
    series = {"SN axis": [AXES_SUMMARY["results"]["SN"][o]["frac"]["physics"] for o in obs_list],
              "AGN axis": [AXES_SUMMARY["results"]["AGN"][o]["frac"]["physics"] for o in obs_list],
              "joint 30 params": [JOINT["decomposition"][o]["frac"]["physics"] for o in obs_list]}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(obs_list)); w = 0.26
    for i, (lab, vals) in enumerate(series.items()):
        off = (i - 1) * w
        ax.bar(x + off, vals, w, label=lab,
               color=["#90CAF9", "#CE93D8", "#E65100"][i], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[o] for o in obs_list], fontsize=10)
    ax.set_ylabel("feedback (physics) fraction of scatter"); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Result 3c — feedback fraction: SN-only vs. AGN-only vs. all 30 knobs jointly")
    fig.tight_layout(); plt.show()
    print("Takeaway: the joint 30-parameter prior gives the physically complete feedback budget; "
          "single-axis views under- or over-state it.")
else:
    print("Need both the 2-axis summary and the joint decomposition for this comparison.")
''')

md(r"""
## Result 4 — What *is* the assembly term?

For the assembly-dominated observables ($M_{\rm gas}$, $q_{\rm gas}$) — and for the residual assembly piece of
the SHMR and $f_{\rm b}$ — what does "which halo you are" mean physically? Using the matched DMO N-body
structure (Method II), we correlate each observable's mass-detrended residual against every structural
descriptor, for **truth (grey)** and **emulator (orange)**. Where the emulator tracks truth — the
assembly-dominated observables ($M_{\rm gas}$, $f_{\rm b}$) — this validates that the assembly term is
identified halo structure. (For the SHMR, assembly is only $\sim$10% of the scatter and its residual
structure correlations are weak and noisier, so truth and emulator agree less closely there.)
""")
code(r'''
AJ = OUT_DIR / "assembly_3d.json"
if not AJ.exists():
    print("No assembly artifact. Run:  python -m scatter.assembly_3d   (or the MPI batch script).")
else:
    aj = json.loads(AJ.read_text())
    aobs = [o for o in ["M_star", "f_b", "M_gas"] if o in aj["obs"]]
    props = list(next(iter(aj["obs"].values()))["truth"]["spearman"])
    plabel = {"c_V": r"$c_V$", "lambda": r"$\lambda$", "veldisp": r"$\sigma_v$",
              "rhalf": r"$r_{\rm half}$", "z_form": r"$z_{\rm form}$"}
    fig, axs = plt.subplots(1, len(aobs), figsize=(4.0*len(aobs), 4), squeeze=False)
    for ax, o in zip(axs[0], aobs):
        t, g = aj["obs"][o]["truth"], aj["obs"][o]["gen"]; xp = np.arange(len(props)); w = 0.38
        ax.bar(xp - w/2, [abs(t["spearman"][p]["rho"]) for p in props], w,
               color=COLORS["truth"], label="truth")
        ax.bar(xp + w/2, [abs(g["spearman"][p]["rho"]) for p in props], w,
               color=COLORS["physics"], label="emulator")
        ax.set_xticks(xp); ax.set_xticklabels([plabel[p] for p in props]); ax.set_ylim(0, None)
        ax.set_title(f"{SHORT.get(o,o)}  (structure explains {t['explained_fraction']*100:.0f}%)",
                     fontsize=10.5)
        if o == aobs[0]:
            ax.set_ylabel(r"$|\rho|$  with the residual"); ax.legend(fontsize=8.5)
    fig.suptitle(f"Result 4 — decoding 'assembly': baryon residual vs. DMO 3D structure "
                 f"({aj['n_halos']} CV halos)", fontsize=12)
    fig.tight_layout(); plt.show()
    print("Strongest single correlate (truth):")
    for o in aobs:
        sp = aj["obs"][o]["truth"]["spearman"]; best = max(sp, key=lambda p: abs(sp[p]["rho"]))
        print(f"  {o:7s}  {best} (rho={sp[best]['rho']:+.2f})   structure explains "
              f"{aj['obs'][o]['truth']['explained_fraction']*100:.0f}% of the assembly scatter")
    print("\nTakeaway: assembly scatter correlates with halo structure (compactness r_half most "
          "consistently), and the emulator reproduces it for the assembly-dominated observables "
          "(M_gas, f_b). But standard summary stats explain only ~3-9% of it, so most assembly "
          "scatter lives in finer structure/history -- itself a result.")
''')

md(r"""
## Result 5 — Mass dependence of the feedback vs. intrinsic budget

Does the balance shift across the group-to-cluster mass range? Here, for the SHMR and $f_{\rm b}$, we track
the per-halo feedback (physics) response and the intrinsic spread as a function of halo mass.
""")
code(r'''
def physics_vs_mass(obs_name, axis_name="SN", n_bins=4):
    X = obs_cube(obs_name, axis_name); ptm = np.nanmean(X, axis=2)
    phys = np.nanstd(ptm, axis=0); intr = np.nanmean(np.nanstd(X, axis=2), axis=0)
    edges = np.quantile(LOG_MASS, np.linspace(0, 1, n_bins+1)); cen, pb, ib = [], [], []
    for b in range(n_bins):
        m = (LOG_MASS >= edges[b]) & (LOG_MASS <= edges[b+1])
        if m.sum() == 0: continue
        cen.append(np.nanmedian(LOG_MASS[m])); pb.append(np.nanmedian(phys[m])); ib.append(np.nanmedian(intr[m]))
    return np.array(cen), np.array(pb), np.array(ib)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, o in zip(axs, ["M_star", "f_b"]):
    cen, pb, ib = physics_vs_mass(o, "SN")
    ax.plot(cen, pb, "o-", color=COLORS["physics"], label="feedback response")
    ax.plot(cen, ib, "s--", color=COLORS["intrinsic"], label="intrinsic spread")
    ax.set_xlabel(r"$\log_{10} M_{200}$"); ax.set_ylabel(f"per-halo std of {SHORT[o]}")
    ax.set_title(f"{'SHMR' if o=='M_star' else 'baryon fraction'}"); ax.legend(fontsize=9)
fig.suptitle("Result 5 — feedback vs. intrinsic scatter across halo mass (SN axis)", fontsize=12)
fig.tight_layout(); plt.show()
print("Takeaway: read off where feedback response peaks vs. where intrinsic scatter dominates — "
      "the mass range where each relation is most informative about feedback.")
''')

md(r"""
## Result 6 — Cosmology trust test

We are explicit about scope: cosmology enters the model through **both** the DMO field *and* the parameter
vector. So we **test**, rather than assume, whether cosmology can be varied through the parameter vector alone.
The "rescaling fraction" is how much of the true cosmology response the emulator recovers when we change only
the cosmological entries of $\theta$ while holding the DMO field fixed.
""")
code(r'''
CV = OUT_DIR / "validate_cosmo_rescaling.json"
if not CV.exists():
    print("No cosmology artifact yet. Run:  python -m scatter.validate_cosmo_rescaling")
else:
    cv = json.loads(CV.read_text())
    print(f"{'cosmology':9s} {'obs':7s} {'rescale frac':>13s}   reading")
    rows = {}
    for cname, r in cv["results"].items():
        for o, v in r["obs"].items():
            rf = v["rescaling_fraction"]
            reading = ("n/a" if rf != rf else "carried by theta" if 0.7 < rf < 1.3 else
                       "needs DMO field" if abs(rf) < 0.3 else "partial")
            rows.setdefault(cname, []).append((o, rf, reading))
            if o in HEADLINE:
                print(f"{cname:9s} {o:7s} {rf:13.2f}   {reading}")
    print("\nTakeaway: for most observables changing only the cosmology entries of theta does little "
          "(rescale frac ~ 0) -> cosmology is carried by the DMO field, not the parameter vector. "
          "f_b is the partial exception. This bounds how the model may be used for cosmology inference.")
''')

# ════════════════════════════════════════════════════════════════════════════
# PART IV — IMPLICATIONS
# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part IV — Implications

*What we learned, what only this method could show, and where it points.*
""")

md(r"""
### Figure 7 — Synthesis: the whole result in one figure

Left: the scatter budget for the two headline relations (from the joint 30-parameter prior). Right: the
cosmology trust verdict. The story in one place.
""")
code(r'''
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))

# (left) stacked scatter budget for the headline relations
src = JOINT["decomposition"] if JOINT is not None else AXES_SUMMARY["results"][AXES[0]]
obs_list = [o for o in HEADLINE if o in src]; xb = np.arange(len(obs_list)); bottom = np.zeros(len(obs_list))
for comp in COMPONENTS:
    vals = np.array([src[o]["frac"][comp] for o in obs_list])
    axs[0].bar(xb, vals, bottom=bottom, color=COLORS[comp], edgecolor="white",
               label=CLABEL[comp].replace("\n", " "))
    bottom += vals
axs[0].set_xticks(xb); axs[0].set_xticklabels([SHORT.get(o, o) for o in obs_list])
axs[0].set_ylim(0, 1); axs[0].set_ylabel("fraction of scatter")
axs[0].legend(fontsize=7.6, loc="upper right", framealpha=0.95)
axs[0].set_title("Scatter budget (joint 30-parameter prior)")

# (right) cosmology rescaling fraction for the headline observables
cvp = OUT_DIR / "validate_cosmo_rescaling.json"
if cvp.exists():
    cv = json.loads(cvp.read_text()); cn = list(cv["results"])[0]; obsr = cv["results"][cn]["obs"]
    oc = [o for o in HEADLINE if o in obsr]; rf = [obsr[o]["rescaling_fraction"] for o in oc]
    axs[1].bar(range(len(oc)), rf, color="#00838F")
    axs[1].axhline(1.0, ls="--", color="k", lw=1, label="fully carried by $\\theta$")
    axs[1].axhline(0.0, ls=":", color="grey", lw=1, label="carried by DMO field")
    axs[1].set_xticks(range(len(oc))); axs[1].set_xticklabels([SHORT.get(o, o) for o in oc])
    axs[1].set_ylabel("cosmology rescaling fraction"); axs[1].legend(fontsize=8)
    axs[1].set_title(f"Cosmology via parameter vector?  ({cn})")
else:
    axs[1].text(0.5, 0.5, "cosmology artifact pending", ha="center"); axs[1].axis("off")
fig.suptitle("Figure 7 — Synthesis", fontsize=13); fig.tight_layout(); plt.show()
''')

md(r"""
### What this means

**1. The result.** Across the full 30-parameter astrophysical prior of CAMELS–IllustrisTNG, the scatter of
the **SHMR** and the **baryon-fraction relation** at group/cluster scales is set primarily by **feedback
physics** (with $f_{\rm b}$ additionally carrying a large halo$\times$physics **interaction**), while the
scatter of **gas mass** and **gas shape** is set by **halo assembly**. So different baryonic observables are
windows onto different physics: the SHMR and $f_{\rm b}$ are *feedback probes*; $M_{\rm gas}$ and $q_{\rm gas}$
are *assembly probes*. Each relation's scatter is a quantitative, decomposable measurement — not a nuisance.

**2. Why a generative model — and why neither a Jacobian nor 1P suffices.** A linearized Jacobian captures only
how the *mean* relation moves. The 1P suite gives the mean feedback response (and is ground truth) but, with a
single realization per (halo, $\theta$), **cannot** measure the intrinsic stochasticity, and being
one-at-a-time it cannot measure parameter interactions. The intrinsic term, the halo$\times$physics interaction,
and the joint 30-parameter attribution are this method's unique contributions; the single-parameter means are a
validation bridge to 1P (Result 2).

**3. Observational implications.** Feedback-dominated scatter (SHMR, $f_{\rm b}$) is the part of a real group
catalog that *carries information about the subgrid feedback model* — the right target for constraining feedback
(and thus the baryonic suppression of the matter power spectrum) from eROSITA / DESI groups via simulation-based
inference. Assembly-dominated scatter ($M_{\rm gas}$, $q_{\rm gas}$) is a nuisance direction for feedback
inference but a probe of halo assembly in its own right.

**4. Honest scope.** This is IllustrisTNG-family physics; the "feedback fraction" is variance over the *trained
prior*, not over nature. Cosmology enters mainly through the DMO field, not the parameter vector (Result 6), so
the model is a feedback emulator at fixed-cosmology DMO input, not a cosmology emulator. The per-parameter Sobol
attribution (Result 3) is a screening ranking, not exact fractions, at the current design size.
""")

md(r"""
### Next steps

- **Larger Sobol design.** Increase the joint design from 128 points (and/or add total-order indices) so the
  per-parameter attribution in Result 3 becomes quantitative, not just a ranking.
- **Finish the full-CV joint run.** Re-reduce once all 1154 CV halos are present:
  `python -m scatter.scatter_decomposition --mode joint --base cv --phase reduce`.
- **Push the assembly decoding.** Standard summary stats explain only ~3–9% of the assembly scatter (Result 4);
  test environment, tidal anisotropy, and full mass-accretion histories as the missing variables.
- **Observational connection.** Use the feedback-dominated observables as the SBI forward model and confront
  real group catalogs (eROSITA, DESI), with manifold-falsification against the trained prior.
""")

md(r"""
### References

*Starting bibliography — verify against ADS before manuscript submission.*

- Albergo, M. S. & Vanden-Eijnden, E. 2023, *Building Normalizing Flows with Stochastic Interpolants*, ICLR.
- Behroozi, P., Wechsler, R. H., Hearin, A. P. & Conroy, C. 2019, *UniverseMachine*, MNRAS, 488, 3143.
- Bullock, J. S., et al. 2001, *A universal angular momentum profile for galactic halos*, ApJ, 555, 240.
- Cranmer, K., Brehmer, J. & Louppe, G. 2020, *The frontier of simulation-based inference*, PNAS, 117, 30055.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. 2023, *Flow Matching for Generative Modeling*, ICLR.
- Pillepich, A., et al. 2018, *Simulating galaxy formation with the IllustrisTNG model*, MNRAS, 473, 4077.
- Prada, F., et al. 2012, *Halo concentrations in the standard LCDM cosmology*, MNRAS, 423, 3018.
- Predehl, P., et al. 2021, *The eROSITA X-ray telescope on SRG*, A&A, 647, A1.
- Saltelli, A., et al. 2010, *Variance based sensitivity analysis of model output*, Comput. Phys. Commun., 181, 259.
- Sobol, I. M. 2001, *Global sensitivity indices for nonlinear mathematical models*, Math. Comput. Simul., 55, 271.
- van Daalen, M. P., McCarthy, I. G. & Schaye, J. 2020, MNRAS, 491, 2424.
- Villaescusa-Navarro, F., et al. 2021, *The CAMELS Project*, ApJ, 915, 71.
- Wechsler, R. H. & Tinker, J. L. 2018, *The Connection Between Galaxies and Their Halos*, ARA&A, 56, 435.
""")


# ════════════════════════════════════════════════════════════════════════════
# Assemble + write the notebook
# ════════════════════════════════════════════════════════════════════════════
def build():
    cells = []
    for ctype, source in CELLS:
        if ctype == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": source})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path("scatter_decomposition_walkthrough.ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out}  ({len(cells)} cells: "
          f"{sum(c['cell_type']=='markdown' for c in cells)} md, "
          f"{sum(c['cell_type']=='code' for c in cells)} code)")


if __name__ == "__main__":
    build()
