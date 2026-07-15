"""Generate examples/wlemu_tutorial.ipynb (the WL-emulator tutorial + validation).

The notebook is fully self-contained given the repo: it uses the packaged
emulator artifact, the bundled k-fold validation predictions
(data/wlemu_validation.npz), and the bundled sample maps (data/kappa_sample.npz).
Regenerate with:  python examples/_build_wlemu_tutorial.py [--execute]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
OUT = HERE / "wlemu_tutorial.ipynb"

cells: list = []


def md(src: str):
    cells.append(nbf.v4.new_markdown_cell(src.strip()))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src.strip()))


# ---------------------------------------------------------------------------
md(r"""
# BIND weak-lensing statistics emulator — tutorial & validation

**`bind.wlemu`** gives you the weak-lensing convergence summary statistics of a
5×5 deg field — power spectrum, one-point PDF, peak counts, minima counts,
Minkowski functionals, wavelet-scattering coefficients, and amplitude moments —
as an **instant, self-consistent function of baryonic-feedback physics**: the 30
astrophysical parameters of the CAMELS SB35 prior (cosmology fixed at the
IllustrisTNG fiducial) and a source redshift $z_s \in \{0.5, 1.0, 1.5, 2.0, 2.44\}$.

**Where it comes from.** BIND painted baryons onto IllustrisTNG dark-matter-only
halos out to $z=2.5$ at **253 Sobol points** of the astro-parameter space (same
DMO halos everywhere → a controlled *same-halo* experiment). The painted volumes
were tiled into lightcones and **raytraced** to 1024² convergence maps at 5 source
planes, 50 map realizations per parameter point. All statistics were measured
from those maps, and each redshift's full 383-dim statistics vector is emulated
*jointly* (PCA + one exact ARD-Matérn GP per coefficient). Because every
statistic derives from the *same* underlying maps and one joint posterior, the
predictions are mutually **self-consistent** — the joint response across
statistics is the response of the maps, not of independently fitted curves.

**What you need.** `pip install bind` (prediction is numpy-only — no GPU, no
torch at inference). The emulator artifact (~3 MB) ships inside the package;
this notebook additionally uses the bundled validation file
(`data/wlemu_validation.npz`) and sample maps (`data/kappa_sample.npz`).

**Sections.** 1 quick start · 2 the full statistics vector · 3 feedback response ·
4 which parameters matter · 5 **validation on held-out raytraced truth** ·
6 self-consistency across statistics · 7 a toy inference · 8 your own maps ·
9 performance & CLI · 10 caveats.
""")

# ---------------------------------------------------------------------------
code(r"""
import json, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bind.wlemu import WLEmulator, measure_stats
from bind.wlemu.fit import frac_err_by_block, sigma_err_by_block

# -- consistent, colorblind-safe style (Okabe-Ito) --------------------------
C_TRUTH, C_EMU, C_ALT, C_GREEN, C_GRAY = "#111111", "#D55E00", "#0072B2", "#009E73", "#9A9A9A"
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.frameon": False,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})

def band(ax, x, mu, sd, color, label=None, alpha=0.25, lw=1.8):
    ax.plot(x, mu, color=color, lw=lw, label=label)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=alpha, lw=0)

emu = WLEmulator.load()
print(f"source redshifts : {emu.source_redshifts.tolist()}")
print(f"parameters       : {emu.n_params} (SB35 astro; cosmology fixed at TNG fiducial)")
print(f"statistics       : {emu.n_stats} dims in blocks "
      + str({b: emu.block_slices[b].stop - emu.block_slices[b].start for b in emu.block_names}))
print(f"training design  : {emu.provenance['n_train_runs']} Sobol points x "
      f"{emu.n_real} map realizations, fit {emu.provenance['created']}")
""")

md(r"""
## 1. Quick start

`predict` takes parameters either **on the unit cube** (arrays, the native Sobol
coordinates) or as a **dict of physical values** (anything not named stays at the
IllustrisTNG fiducial). Every block comes back with a calibrated GP uncertainty
(`<block>_std`). Bin coordinates live on the emulator: `emu.ell`, `emu.pdf_x`,
`emu.peak_x`, `emu.mink_thr`.
""")

code(r"""
theta_fid = emu.fiducial_params()               # (30,) unit cube
pred = emu.predict(theta_fid, z_source=1.0)     # dict of blocks (+ _std)

# stronger SN winds, physical units (all other parameters at fiducial):
pred_sn = emu.predict({"WindEnergyIn1e51erg": 14.0}, z_source=1.0)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
ax = axes[0]
shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(emu.source_redshifts)))
for i, z in enumerate(emu.source_redshifts):
    p = emu.predict(theta_fid, z_idx=i)
    ax.loglog(emu.ell, p["Cl"], color=shades[i], lw=1.8, label=f"$z_s={z:g}$")
ax.set(xlabel=r"$\ell$", ylabel=r"$C_\ell$", title="fiducial convergence power spectrum")
ax.legend(ncol=2)

ax = axes[1]
sf = np.sqrt(np.diag(emu.covariance(z_idx=1)))[emu.block_slices["Cl"]]
ax.fill_between(emu.ell, 1 - sf / pred["Cl"], 1 + sf / pred["Cl"],
                color=C_GRAY, alpha=0.3, lw=0, label="single-field scatter")
band(ax, emu.ell, np.ones_like(emu.ell), pred["Cl_std"] / pred["Cl"], C_TRUTH,
     "fiducial (GP $\\sigma$)")
ax.plot(emu.ell, pred_sn["Cl"] / pred["Cl"], color=C_EMU, lw=1.8, label="max SN-wind energy")
ax.set_xscale("log")
ax.set_ylim(0.9, 1.1)
ax.set(xlabel=r"$\ell$", ylabel="ratio to fiducial", title=r"feedback response at $z_s=1$")
ax.legend(loc="lower left")
fig.tight_layout()
""")

md(r"""
## 2. The full statistics vector

One call returns all nine blocks. Two uncertainty scales matter and the emulator
gives you both: the **GP uncertainty** on the *expectation* value (orange band —
usually invisibly small), and the **single-field scatter** of one 5×5 deg
realization (gray band, from `emu.covariance`, estimated from the 50 raytraced
realizations per parameter point). The PDF, peaks, minima and Minkowski
functionals follow standard WL practice: measured on the map smoothed at
2 arcmin, in per-map S/N units $\nu = (\kappa - \bar\kappa)/\sigma_\kappa$.
""")

code(r"""
zi = 1  # z_s = 1.0
p = emu.predict(theta_fid, z_idx=zi)
sig_field = np.sqrt(np.diag(emu.covariance(z_idx=zi)))

PANEL = dict(Cl=(r"$\ell$", r"$C_\ell$", "log", "log"),
             pdf=(r"$\nu$", "PDF", "linear", "log"),
             peak=(r"$\nu$", "peaks / map", "linear", "linear"),
             min=(r"$\nu$", "minima / map", "linear", "linear"),
             V0=(r"$\nu$", "$V_0$", "linear", "linear"),
             V1=(r"$\nu$", "$V_1$", "linear", "linear"),
             V2=(r"$\nu$", "$V_2$", "linear", "linear"),
             scat=("coefficient", "scattering", "linear", "log"),
             moments=("", "", "linear", "linear"))

fig, axes = plt.subplots(3, 3, figsize=(11, 8.4))
for ax, blk in zip(axes.ravel(), emu.block_names):
    s = emu.block_slices[blk]
    x = emu.coords_for(blk)
    if blk == "moments":
        lab = [r"$\langle\kappa\rangle$", r"$\sigma_\kappa$",
               r"$\langle\kappa\rangle_{2'}$", r"$\sigma_{\kappa,2'}$"]
        ax.errorbar(x, p[blk], yerr=sig_field[s], fmt="o", ms=5, color=C_EMU, capsize=3)
        ax.set_xticks(x, lab)
        ax.set_title("moments (raw & smoothed)")
        continue
    ax.fill_between(x, p[blk] - sig_field[s], p[blk] + sig_field[s],
                    color=C_GRAY, alpha=0.35, lw=0, label="single-field scatter")
    band(ax, x, p[blk], p[blk + "_std"], C_EMU, "emulator (GP $\\sigma$)")
    xl, yl, xs, ys = PANEL[blk]
    ax.set(xlabel=xl, ylabel=yl, xscale=xs, yscale=ys, title=blk)
    if blk == "pdf":
        ax.set_ylim(bottom=1e-6)
axes[0, 0].legend()
fig.suptitle(f"all statistic blocks at the fiducial, $z_s={emu.source_redshifts[zi]:g}$", y=1.005)
fig.tight_layout()
""")

md(r"""
## 3. Feedback response

The whole point of the tool: how does *every* WL statistic respond as you move
the feedback dials? Below, supernova-wind energy (ASN1, `WindEnergyIn1e51erg`)
and kinetic AGN-feedback energy (AAGN1, `RadioFeedbackFactor`) swept across
their SB35 prior ranges, everything else fiducial.
""")

code(r"""
def sweep(name, n=7, zi=1):
    i = emu.param_names.index(name)
    thetas = np.tile(theta_fid, (n, 1))
    thetas[:, i] = np.linspace(0.0, 1.0, n)
    phys = emu.unit_to_physical(thetas)[:, i]
    return thetas, phys, emu.predict(thetas, z_idx=zi, return_std=False)

fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.6), sharex="col")
p0 = emu.predict(theta_fid, z_idx=1, return_std=False)
for row, name in enumerate(["WindEnergyIn1e51erg", "RadioFeedbackFactor"]):
    thetas, phys, pr = sweep(name)
    shades = plt.cm.Oranges(np.linspace(0.35, 0.95, len(phys)))
    for j in range(len(phys)):
        axes[row, 0].plot(emu.ell, pr["Cl"][j] / p0["Cl"], color=shades[j], lw=1.6)
        axes[row, 1].plot(emu.peak_x, pr["peak"][j] - p0["peak"], color=shades[j], lw=1.6)
    axes[row, 0].set(xscale="log", ylabel=f"{name}\n$C_\\ell$ ratio to fiducial")
    axes[row, 1].set(ylabel="$\\Delta$ peaks / map")
    sm = plt.cm.ScalarMappable(cmap="Oranges",
                               norm=plt.Normalize(phys.min(), phys.max()))
    fig.colorbar(sm, ax=axes[row, :], pad=0.01, label=name, aspect=28)
axes[1, 0].set_xlabel(r"$\ell$")
axes[1, 1].set_xlabel(r"$\nu$")
""")

md(r"""
## 4. Which parameters matter?

For each parameter, sweep it across its full prior range (all others fiducial)
and measure how far each statistic block moves, expressed as the **integrated
detectability for a single 5×5 deg field**:
$S/N = \max_v \sqrt{\sum_{\rm bins} (\Delta_{\rm bin}(v)/\sigma_{\rm bin})^2}$
with $\sigma$ the single-field scatter. This global sensitivity map of WL
statistics to the TNG feedback model reads directly off the emulator in seconds.
(Sweeps are conditional — one axis at a time through the fiducial; for weakly
responding parameters the ranking is emulator-limited, so treat the bottom of
the table as "consistent with no detectable response".)
""")

code(r"""
zi, nv = 1, 9
sig_field = np.sqrt(np.diag(emu.covariance(z_idx=zi)))
ok = sig_field > 0                                    # mask empty histogram tails
p_fid = emu.predict_vector(theta_fid, z_idx=zi, return_std=False)[0]

score = np.zeros((emu.n_params, len(emu.block_names)))
for i in range(emu.n_params):
    thetas = np.tile(theta_fid, (nv, 1))
    thetas[:, i] = np.linspace(0.0, 1.0, nv)
    v = emu.predict_vector(thetas, z_idx=zi, return_std=False)
    zres = np.where(ok, (v - p_fid) / np.where(ok, sig_field, 1.0), 0.0)  # (nv, D)
    for j, b in enumerate(emu.block_names):
        s = emu.block_slices[b]
        score[i, j] = np.sqrt((zres[:, s] ** 2).sum(axis=1)).max()

order = np.argsort(score.max(axis=1))[::-1]
fig, ax = plt.subplots(figsize=(7.5, 9))
im = ax.imshow(score[order], cmap="Oranges", aspect="auto",
               vmin=0, vmax=np.percentile(score, 98))
ax.set_xticks(range(len(emu.block_names)), emu.block_names, rotation=45, ha="right")
ax.set_yticks(range(emu.n_params), [emu.param_names[i] for i in order], fontsize=7.5)
ax.grid(False)
fig.colorbar(im, ax=ax, label="integrated response S/N, one field", shrink=0.6)
ax.set_title(f"global sensitivity at $z_s={emu.source_redshifts[zi]:g}$")
fig.tight_layout()
print("most influential:", ", ".join(emu.param_names[i] for i in order[:5]))
""")

md(r"""
## 5. Validation: held-out raytraced truth

**Protocol.** 13-fold cross-validation over the 253 Sobol points: every
parameter point is predicted by an emulator fit with that point's fold *removed*,
using the identical recipe as the shipped artifact. So every number below is a
genuine out-of-sample prediction of a raytraced simulation the emulator never
saw. The held-out predictions ship with the repo
(`data/wlemu_validation.npz`), so this section is fully reproducible without any
access to the training maps.

Two scores per block (pooled over all 253 points × 5 redshifts):

- **frac** — mean |pred − truth| / |truth| over populated bins;
- **err/SEM** — median |pred − truth| in units of the *realization-noise floor*
  (the standard error of the 50-map truth mean). 1.0 would mean the emulator is
  exactly as uncertain as the truth measurement itself; the emulator sits at
  **0.09–0.7 (typically ≈0.2)** — a fraction of the noise floor of a
  50-realization raytracing campaign.

The one coherent systematic worth knowing about: the last few $C_\ell$ bins
($\ell \gtrsim 10^4$, approaching the map resolution) carry a median residual of
$\sim\!2$ SEM — still sub-percent in $C_\ell$, and covered by the GP uncertainty.
Include the GP $\sigma$ in your error budget (as §7 does) and it is handled.
""")

code(r"""
import pandas as pd

V = np.load(Path("data") / "wlemu_validation.npz")
R, Z, D = V["truth"].shape
sizes = V["block_sizes"]
offs = np.concatenate([[0], np.cumsum(sizes)])
slices = {str(b): slice(int(offs[i]), int(offs[i + 1])) for i, b in enumerate(V["block_names"])}
t, pr, se = (V[k].reshape(R * Z, D).astype(np.float64) for k in ("truth", "pred", "sem"))

frac = frac_err_by_block(t, pr, slices)
sig = sigma_err_by_block(t, pr, se, slices)
floor = {k: float(np.mean((se[:, s] / np.maximum(np.abs(t[:, s]), 1e-300))
                          [np.abs(t[:, s]) > 1e-6 * np.abs(t[:, s]).max()]))
         for k, s in slices.items()}
table = pd.DataFrame({"frac err [%]": {k: 100 * v for k, v in frac.items()},
                      "err / SEM (median)": sig,
                      "noise floor frac [%]": {k: 100 * v for k, v in floor.items()}})
print(json.loads(str(V["provenance"]))["note"], "\n")
table.round(2)
""")

md(r"""
Context: the same protocol was used to compare **nine emulator designs**
(nearest-neighbor, linear/quadratic ridge, RBF, gradient-boosted trees, MLP
ensembles, joint-z GP, per-z GP) and the **field-level generative emulator**
(a 1024² flow-matching diffusion model that generates κ maps directly, with the
statistics then measured on its samples). The per-z GP shipped here won on every
block — and beats the field-level route by 5–100× per statistic. Historical
fixed-split numbers (fit 213 / test 20 runs, fractional error in %):

| design | Cl | pdf | peak | min | V0 | V1 | V2 | scat | moments |
|---|---|---|---|---|---|---|---|---|---|
| nearest neighbor | 4.28 | 1.23 | 6.88 | 8.42 | 0.18 | 1.37 | 1.94 | 1.34 | 1.48 |
| quadratic ridge | 2.25 | 0.88 | 5.04 | 6.96 | 0.12 | 0.95 | 1.33 | 0.87 | 0.84 |
| **per-z GP (this tool)** | **2.06** | **0.67** | **5.06** | **7.08** | **0.09** | **0.66** | **0.92** | **0.56** | **0.83** |
| field-level flow matching | 9.77 | 12.55 | 19.67 | 49.91 | 3.83 | 9.05 | 11.58 | 3.64 | 260 |

(The apparent few-% errors on peaks/minima are dominated by near-empty histogram
tail bins — in noise-floor units they sit at 0.25–0.3, see the table above.)
""")

code(r"""
# Held-out truth vs prediction: examples + pooled per-bin residuals at z_s = 1
rng = np.random.default_rng(4)
show = rng.choice(R, 3, replace=False)
zi = 1
Vt, Vp, Vs = V["truth"][:, zi], V["pred"][:, zi], V["sem"][:, zi]
Vps = V["pred_std"][:, zi]

stats4 = ["Cl", "peak", "pdf", "V2"]
xs = {"Cl": V["ell"], "peak": V["peak_x"], "pdf": V["pdf_x"], "V2": V["mink_thr"]}
fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.6), height_ratios=[2, 1.2])
for j, blk in enumerate(stats4):
    s, x = slices[blk], xs[blk]
    ax = axes[0, j]
    for i, r in enumerate(show):
        ax.plot(x, Vt[r, s], color=C_TRUTH, lw=1.4, alpha=0.85,
                label="raytraced truth" if i == 0 else None)
        ax.plot(x, Vp[r, s], color=C_EMU, lw=1.4, ls="--", dashes=(4, 2),
                label="held-out prediction" if i == 0 else None)
    if blk == "Cl":
        ax.set(xscale="log", yscale="log")
    if blk == "pdf":
        ax.set_yscale("log"); ax.set_ylim(bottom=1e-5)
    ax.set_title(blk)

    ax = axes[1, j]
    zres = (Vp[:, s] - Vt[:, s]) / np.where(Vs[:, s] > 0, Vs[:, s], np.inf)
    lo, mid, hi = np.nanpercentile(zres, [16, 50, 84], axis=0)
    ax.axhspan(-1, 1, color=C_GRAY, alpha=0.3, lw=0,
               label="noise floor ($\\pm$SEM)" if j == 0 else None)
    ax.plot(x, mid, color=C_EMU, lw=1.8)
    ax.fill_between(x, lo, hi, color=C_EMU, alpha=0.25, lw=0)
    ax.set_ylim(-2.2, 2.2)
    if blk == "Cl":
        ax.set_xscale("log")
    ax.set_xlabel({"Cl": r"$\ell$"}.get(blk, r"$\nu$"))
    if j == 0:
        ax.set_ylabel("(pred $-$ truth)/SEM")
axes[0, 0].legend(loc="lower left")
axes[1, 0].legend(loc="lower left")
fig.suptitle("held-out validation at $z_s=1$: 3 example points (top); "
             "median & 16–84% of per-bin residuals over all 253 points (bottom)", y=1.01)
fig.tight_layout()
""")

md(r"""
**Are the quoted uncertainties honest?** If the GP's predictive $\sigma$ is
calibrated, the held-out z-scores $(\mathrm{pred}-\mathrm{truth})/\sigma_{GP}$
should be $\sim\mathcal{N}(0,1)$. They are, to a good approximation: 68%
intervals cover 54–69% and 95% intervals cover 80–92% depending on block
(mildly heavy tails — treat $\sigma_{GP}$ as accurate to a few tens of percent;
the spike at 0 is saturated/empty bins that are predicted exactly).
""")

code(r"""
zsc = (pr - t) / np.where(V["pred_std"].reshape(R * Z, D) > 0,
                          V["pred_std"].reshape(R * Z, D), np.inf)
cov1 = {k: float(np.mean(np.abs(zsc[:, s]) < 1)) for k, s in slices.items()}
cov2 = {k: float(np.mean(np.abs(zsc[:, s]) < 2)) for k, s in slices.items()}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
g = np.linspace(-4, 4, 200)
axes[0].hist(zsc.ravel()[np.isfinite(zsc.ravel())], bins=120, range=(-4, 4),
             density=True, color=C_EMU, alpha=0.75)
axes[0].plot(g, np.exp(-g**2 / 2) / np.sqrt(2 * np.pi), color=C_TRUTH, lw=1.8,
             label=r"$\mathcal{N}(0,1)$")
axes[0].set(xlabel="(pred $-$ truth) / GP $\\sigma$", ylabel="density",
            title="held-out z-scores, all blocks pooled")
axes[0].legend()
axes[1].bar(np.arange(len(cov1)) - 0.18, list(cov1.values()), 0.36, color=C_EMU, label="|z|<1")
axes[1].bar(np.arange(len(cov2)) + 0.18, list(cov2.values()), 0.36, color=C_ALT, label="|z|<2")
axes[1].axhline(0.683, color=C_TRUTH, lw=1, ls=":"); axes[1].axhline(0.954, color=C_TRUTH, lw=1, ls=":")
axes[1].set_xticks(range(len(cov1)), list(cov1), rotation=45, ha="right")
axes[1].set(ylabel="coverage", ylim=(0, 1.1), title="per-block coverage vs 68/95%")
axes[1].legend(loc="lower right")
fig.tight_layout()
""")

md(r"""
## 6. Self-consistency: joint responses across statistics

Because all statistics are emulated jointly from the same maps, the emulator
reproduces not just each statistic's response but the **correlations between
statistics' responses** across parameter space. Below, three integrated
features per held-out point — the high-$\ell$ power ($\ell>5000$), the
high-significance peak abundance ($\nu>2.5$), and the map rms $\sigma_\kappa$:
the emulator tracks each truth axis, and the truth's cross-statistic
correlation structure is reproduced. (Truth correlations are mildly *diluted*
by per-point measurement noise — each truth value is a 50-map mean that still
carries the stochasticity of BIND's independent generations — while the
emulator's regression is noise-free, so its correlations sit slightly higher.)
""")

code(r"""
zi = 1
D_ = t.shape[1]
W = np.zeros((3, D_))
W[0, np.arange(D_)[slices["Cl"]][V["ell"] > 5e3]] = 1
W[1, np.arange(D_)[slices["peak"]][V["peak_x"] > 2.5]] = 1
W[2, np.arange(D_)[slices["moments"]][1]] = 1
lab = [r"$\sum C_\ell\,(\ell>5000)$", r"peaks ($\nu>2.5$)", r"$\sigma_\kappa$"]

Ft = V["truth"][:, zi].astype(np.float64) @ W.T
Fp = V["pred"][:, zi].astype(np.float64) @ W.T
rt, rp = Ft / Ft.mean(0) - 1, Fp / Ft.mean(0) - 1

fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.1))
for j in range(3):
    ax = axes[j]
    ax.scatter(100 * rt[:, j], 100 * rp[:, j], s=8, color=C_EMU, alpha=0.6, lw=0)
    lim = max(np.abs(ax.get_xlim()).max(), np.abs(ax.get_ylim()).max())
    ax.plot([-lim, lim], [-lim, lim], color=C_TRUTH, lw=1)
    r = np.corrcoef(rt[:, j], rp[:, j])[0, 1]
    ax.set(xlabel="truth fluctuation [%]", title=f"{lab[j]}   r={r:.2f}")
    if j == 0:
        ax.set_ylabel("emulated fluctuation [%]")
Ct, Cp = np.corrcoef(rt.T), np.corrcoef(rp.T)
im = axes[3].imshow(np.tril(Ct) + np.triu(Cp, 1), cmap="RdBu_r", vmin=-1, vmax=1)
for a in range(3):
    for b in range(3):
        axes[3].text(b, a, f"{(np.tril(Ct) + np.triu(Cp, 1))[a, b]:.2f}",
                     ha="center", va="center", fontsize=8,
                     color="w" if abs((np.tril(Ct) + np.triu(Cp, 1))[a, b]) > 0.6 else "k")
axes[3].set_xticks(range(3), ["$C_\\ell$", "peaks", "$\\sigma_\\kappa$"])
axes[3].set_yticks(range(3), ["$C_\\ell$", "peaks", "$\\sigma_\\kappa$"])
axes[3].grid(False)
axes[3].set_title("response corr: truth (lower)\nvs emulated (upper)")
fig.colorbar(im, ax=axes[3], shrink=0.85)
fig.tight_layout()
""")

md(r"""
## 7. A toy inference: recovering feedback parameters, with coverage

The full closed loop: take a **held-out** simulation's statistics
($C_\ell$ + well-populated peak bins) as pseudo-data, use the emulator as the
theory model, and scan the likelihood over the two dominant feedback dials —
the galactic-wind speed (`VariableWindVelFactor`, ASN2) and the black-hole
radiative efficiency — holding the other 28 at their true values. To make this
a *statement* rather than an anecdote, we repeat the scan for the 12 most
central held-out points and check where the truth lands in each posterior.

*(Conventions: pseudo-data is the noise-free truth mean, so this is a
forecast-style consistency test — contours show the information content of one
5×5 deg field. The error budget is the recommended one: single-field covariance
**plus the emulator's predictive variance** on the diagonal, with the Hartlap
debiasing factor for the 50-realization covariance and a data vector kept well
below 50 dims.)*
""")

code(r"""
zi = 1
scan_params = ["VariableWindVelFactor", "BlackHoleRadiativeEfficiency"]
pa, pb = (emu.param_names.index(n) for n in scan_params)
run_true = V["params_unit"].astype(np.float64)

# data vector: all Cl bins + well-populated peak bins
mask_cl = np.zeros(D, bool); mask_cl[slices["Cl"]] = True
peak_mean = V["truth"][:, zi, slices["peak"]].mean(0)
mask_pk = np.zeros(D, bool)
mask_pk[np.arange(D)[slices["peak"]][peak_mean > 30]] = True
mask = mask_cl | mask_pk

cov0 = np.zeros((D, D))
idx_all = np.concatenate([np.arange(D)[slices["Cl"]], np.arange(D)[slices["peak"]]])
cov0[np.ix_(idx_all, idx_all)] = emu.covariance(z_idx=zi, blocks=("Cl", "peak"))
hartlap = (emu.n_real - mask.sum() - 2) / (emu.n_real - 1)

def chi2_grid(r_obs, n=41):
    theta_true = run_true[r_obs]
    obs = V["truth"][r_obs, zi].astype(np.float64)
    _, sd_gp = emu.predict_vector(theta_true, z_idx=zi)   # emulator error budget
    cv = cov0.copy()
    cv[np.diag_indices(D)] += np.atleast_2d(sd_gp)[0] ** 2
    Ci = hartlap * np.linalg.inv(cv[np.ix_(mask, mask)])
    g = np.linspace(0, 1, n)
    A, B = np.meshgrid(g, g, indexing="ij")
    thetas = np.tile(theta_true, (n * n, 1))
    thetas[:, pa], thetas[:, pb] = A.ravel(), B.ravel()
    Vg = emu.predict_vector(thetas, z_idx=zi, return_std=False)
    d = (Vg - obs)[:, mask]
    return g, np.einsum("nd,de,ne->n", d, Ci, d).reshape(n, n)

# coverage over the 12 most central held-out points
cand = np.argsort(((run_true[:, [pa, pb]] - 0.5) ** 2).sum(1))[:12]
dchi_truth = {}
for r in cand:
    g, c2 = chi2_grid(int(r))
    ia, ib = (int(round(run_true[r, j] * (len(g) - 1))) for j in (pa, pb))
    dchi_truth[int(r)] = float((c2 - c2.min())[ia, ib])
vals = np.array(list(dchi_truth.values()))
print(f"truth position in the 2-dof posterior, 12 held-out points: "
      f"median dchi2 = {np.median(vals):.2f}; inside 68% contour: "
      f"{(vals < 2.30).sum()}/12; inside 95%: {(vals < 6.17).sum()}/12")

# representative example (dchi2 closest to the median)
r_show = int(cand[np.argmin(np.abs(vals - np.median(vals)))])
g, c2 = chi2_grid(r_show)

from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.contour(g, g, (c2 - c2.min()).T, levels=[2.30, 6.17],
           colors=C_EMU, linewidths=[2, 1])
ax.plot(*run_true[r_show, [pa, pb]], marker="x", ms=11, mew=2.5, color=C_TRUTH,
        ls="none")
handles = [Line2D([], [], color=C_EMU, lw=2, label=r"68% / 95% ($C_\ell$+peaks)"),
           Line2D([], [], color=C_TRUTH, marker="x", ms=9, mew=2.5, ls="none",
                  label="truth (held-out sim)")]
ax.set(xlabel=f"{scan_params[0]} (unit cube)", ylabel=f"{scan_params[1]} (unit cube)",
       title=f"one 5x5 deg field, $z_s=1$ (held-out point {r_show})")
ax.legend(handles=handles, loc="upper left")
fig.tight_layout()
print(f"data dims: {mask.sum()} (Cl {mask_cl.sum()} + peaks {mask_pk.sum()}), "
      f"Hartlap factor {hartlap:.2f}")
""")

md(r"""
## 8. Using it against your own maps

`bind.wlemu.stats.measure_stats` is the *exact* estimator set the emulator was
trained on (numpy port of the training pipeline, verified to float32 precision) —
measure your own κ maps with it and the numbers are directly comparable to
emulator predictions. Three raytraced sample maps ship with the repo.
""")

code(r"""
S = np.load(Path("data") / "kappa_sample.npz")
maps = S["kappa"].astype(np.float64)      # (3, 1024, 1024), z_s = 1
theta_s = S["params_unit"].astype(np.float64)
t0 = time.time(); st = measure_stats(maps); dt = time.time() - t0
p = emu.predict(theta_s, z_source=float(S["source_redshift"]))
sig_field = np.sqrt(np.diag(emu.covariance(z_source=float(S["source_redshift"]))))

fig = plt.figure(figsize=(12, 3.6))
ax = fig.add_subplot(1, 3, 1)
im = ax.imshow(np.arcsinh(maps[0] / 0.02), cmap="magma", extent=[0, 5, 0, 5])
ax.set(title=r"sample $\kappa$ map ($z_s=1$, asinh stretch)", xlabel="deg", ylabel="deg")
ax.grid(False)
for k, (blk, xs, logy) in enumerate([("Cl", emu.ell, True), ("peak", emu.peak_x, False)]):
    ax = fig.add_subplot(1, 3, k + 2)
    s = emu.block_slices[blk]
    for i in range(len(maps)):
        ax.plot(xs, st[blk][i], color=C_GRAY, lw=1.1,
                label="measured fields (3)" if i == 0 else None)
    band(ax, xs, p[blk], sig_field[s], C_EMU, "emulator $\\pm$ field scatter")
    if blk == "Cl":
        ax.set(xscale="log", yscale="log")
    ax.set(title=blk, xlabel=r"$\ell$" if blk == "Cl" else r"$\nu$")
    ax.legend()
fig.tight_layout()
print(f"measured all 9 blocks for {len(maps)} maps in {dt:.1f}s (numpy, single CPU)")
""")

md(r"""
## 9. Performance, CLI, refitting

- **Throughput**: the first call builds Cholesky factors (~0.4 s once); after
  that, ~1–2 ms per parameter point batched, on a laptop CPU.
- **CLI**: `bind-wlemu --z 1.0 --set WindEnergyIn1e51erg=7.2 --out pred.npz`
  (and `bind-wlemu --list-params`).
- **Refitting** (e.g. on your own statistics cache):
  `python -m bind.wlemu.fit --stats stats_cache.npz --artifact-out my_emu.npz
  --kfold 13 --validation-out val.npz` (needs `pip install bind[wlemu-fit]`),
  then `WLEmulator.load("my_emu.npz")`.
""")

code(r"""
th = np.random.default_rng(0).uniform(0.1, 0.9, (2000, 30))
emu.predict_vector(th[:2], z_idx=1, return_std=False)      # warm-up
t0 = time.time(); emu.predict_vector(th, z_idx=1, return_std=False)
dt = time.time() - t0
print(f"{len(th)} predictions in {dt:.2f} s -> {1e3 * dt / len(th):.2f} ms/point "
      f"({len(th) / dt:,.0f} points/s)")
from bind.wlemu.emulator import DEFAULT_ARTIFACT
print(f"artifact: {DEFAULT_ARTIFACT.stat().st_size / 1e6:.1f} MB, numpy-only inference")
""")

md(r"""
## 10. Caveats & conventions

- **Cosmology is fixed** (TNG fiducial: $\Omega_m$=0.3089, $\sigma_8$=0.8159,
  $\Omega_b$=0.0486, $h$=0.6774, $n_s$=0.9667). This tool isolates the
  *baryonic-feedback response* of WL statistics; it is not a cosmology emulator.
- **Source redshifts are the 5 discrete raytraced planes** — no interpolation
  between them (yet).
- **Field geometry**: 5×5 deg flat-sky periodic patch at 1024²; statistics
  conventions as in §2 (2-arcmin Gaussian smoothing, per-map S/N units). Use
  `bind.wlemu.stats` to match them exactly.
- **Covariance**: estimated from 50 noise-paired realizations of one field —
  keep data vectors well below 50 dims before inverting and apply a
  Hartlap-style correction; it is averaged over parameter points (parameter
  dependence of the covariance is not emulated).
- **Prior box**: parameters outside the SB35 ranges extrapolate (a warning is
  raised). The 50 realizations reuse the same underlying lightcone fields at
  every parameter point, which is what makes parameter *responses* far more
  precise than single-field scatter.
- **Upstream model**: the maps inherit BIND's trained regime (halos
  $\geq 10^{13}\,M_\odot/h$ baryonified) and the lightcone construction; see the
  BIND paper for the painting fidelity and `docs/wl_emulator.md` for design
  details.
""")

# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--kernel", default="torch3")
    args = ap.parse_args()

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                              "name": "python3"}
    nb.metadata.language_info = {"name": "python"}
    if args.execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=1200, kernel_name=args.kernel,
                                resources={"metadata": {"path": str(HERE)}})
        client.execute()
        nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                                  "name": "python3"}
    nbf.write(nb, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
