"""r5_figs.py -- two diagnostic figures for referee comment 5.

  r5_fig1_noise_floor.png -- the latent noise budget: three independent
      estimates of sigma(lambda) as a fraction of the across-design spread.
  r5_fig2_absorption.png  -- the absorption accounting: measured noise
      absorption vs the attenuation penalty vs the 0.38 lambda-over-theta
      gap, plus the lens-plane geometry that caps the shared channel.

Writes into papers/01_pipeline/referee/figs/ (png only).
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline/referee/work")
from paper_style import COLORS, ONE_COL, TWO_COL, panel_label, setup  # noqa: E402
from r5_common import OUT, STATS                                     # noqa: E402

setup()
import matplotlib.pyplot as plt                                      # noqa: E402

FIG = "/mnt/home/mlee1/BIND/papers/01_pipeline/referee/figs"
boot = np.load(OUT / "r5_bootstrap.npz")
rep = np.load(OUT / "r5_replicas.npz")
inf = np.load(OUT / "r5_inflation.npz")
repro = np.load(OUT / "r5_repro.npz")
tru = np.load(OUT / "r5_truth.npz")

LN = [str(s) for s in boot["lat_names"]]
SHORT = [n.replace("[", "\n[") for n in LN]
SIG_DES = boot["SIG_DES"]

# ── fig 1: latent noise floor ─────────────────────────────────────────────
fig, (ax, bx) = plt.subplots(1, 2, figsize=(TWO_COL[0], 2.9),
                             gridspec_kw=dict(width_ratios=[1.35, 1.0]))
x = np.arange(8)
sA = np.median(boot["SIG_A"], 0) / SIG_DES
sB = np.median(boot["SIG_B"], 0) / SIG_DES
sR = rep["SIG_REP"] / SIG_DES
ax.bar(x - 0.27, 100 * sA, 0.26, color="0.72",
       label="bootstrap, independent draw")
ax.bar(x, 100 * sB, 0.26, color=COLORS["bind"],
       label="bootstrap, node-specific part")
ax.bar(x + 0.27, 100 * sR, 0.26, color=COLORS["highlight"],
       label="3 independent paints (measured)")
ax.set_xticks(x)
ax.set_xticklabels(SHORT, fontsize=4.6, rotation=0)
ax.set_ylabel(r"$\sigma(\lambda_i)\ /\ \sigma_{\rm design}(\lambda_i)$  [%]",
              fontsize=7)
ax.legend(fontsize=5.2, loc="upper left", frameon=False)
ax.set_title("noise floor of each measured latent", fontsize=7)
panel_label(ax, "a")

# noise-to-signal VARIANCE ratio q, log scale
bx.semilogy(x, boot["qA"], "o-", ms=4, color="0.55", lw=1.1,
            label=r"$q$ bootstrap (indep.)")
bx.semilogy(x, boot["qB"], "s-", ms=4, color=COLORS["bind"], lw=1.1,
            label=r"$q$ bootstrap (node-spec.)")
bx.semilogy(x, rep["q_rep"], "D-", ms=4, color=COLORS["highlight"], lw=1.3,
            label=r"$q$ 3 paints (measured)")
bx.axhline(0.01, color="0.8", lw=0.6, ls=":")
bx.text(0.05, 0.0108, "1%", fontsize=5.2, color="0.5", ha="left")
bx.set_xticks(x)
bx.set_xticklabels([f"$\\lambda_{i}$" for i in range(8)], fontsize=6)
bx.set_ylabel(r"$q_i={\rm Var(noise)}/{\rm Var(design)}$", fontsize=7)
bx.legend(fontsize=5.2, loc="lower right", frameon=False)
panel_label(bx, "b")
fig.tight_layout()
fig.savefig(f"{FIG}/r5_fig1_noise_floor.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ── fig 2: absorption accounting + geometry ───────────────────────────────
fig, (ax, bx) = plt.subplots(1, 2, figsize=(TWO_COL[0], 2.9))
gap = repro["r2_lambda"] - repro["r2_theta_linear"]
xs = np.arange(len(STATS))
absb = np.array([float(rep[f"{s}__inflate"]) for s in STATS])
cap = np.array([float(rep[f"{s}__cap"]) for s in STATS])
atten = np.array([float(rep[f"{s}__var_Dhn"]) / float(rep[f"{s}__var_des"])
                  for s in STATS])
fn = np.array([float(rep[f"{s}__f_noise"]) for s in STATS])
ax.bar(xs, gap, 0.62, color="0.85", label=r"$\lambda$ over $\theta$ gap in CV $R^2$")
pos = absb > 0
ax.plot(xs[pos], absb[pos], "o", ms=5, color=COLORS["highlight"],
        label="measured noise absorption $2{\\rm Cov}/{\\rm SS}$ (inflates)")
ax.plot(xs[~pos], -absb[~pos], "o", ms=5, mfc="none", mew=1.2,
        color=COLORS["highlight"],
        label="same, but NEGATIVE (deflates)")
ax.plot(xs, cap, "_", ms=11, mew=1.6, color=COLORS["highlight"],
        label=r"its ceiling at corr $=1$")
ax.plot(xs, atten, "v", ms=4.5, color=COLORS["bind"],
        label=r"attenuation penalty ${\rm Var}(\hat D_n)/{\rm SS}$")
ax.set_yscale("log")
ax.set_ylim(1e-5, 1.2)
ax.set_xticks(xs)
ax.set_xticklabels([r"$S(\ell)$", "PDF", "peaks", "minima",
                    r"$V_0$", r"$V_1$", r"$V_2$"], fontsize=6)
ax.set_ylabel(r"fraction of the statistic's design variance", fontsize=6.6)
ax.legend(fontsize=5.0, loc="lower right", frameon=False, ncol=1)
ax.set_title("noise absorption vs the gap it would have to explain",
             fontsize=7)
panel_label(ax, "a")

bx.semilogx(inf["ell_band"], 100 * inf["F096"], "o-", ms=3.5,
            color=COLORS["bind"], lw=1.3)
bx.fill_between(inf["ell_band"], 0, 100 * inf["F096"],
                color=COLORS["bind"], alpha=0.15, lw=0)
bx.set_xlabel(r"$\ell$", fontsize=7)
bx.set_ylabel(r"share of $C_\ell^{\kappa\kappa}$ from the $z<0.061$ planes"
              "\n" r"(the only ones $\lambda$ is measured from)  [%]",
              fontsize=6.2)
bx.set_ylim(0, 3.2)
bx.axhline(100 * inf["F096"].mean(), color="0.6", lw=0.7, ls="--")
bx.text(2.2e4, 100 * inf["F096"].mean() + 0.08,
        f"band mean {100*inf['F096'].mean():.2f}%", fontsize=5.4,
        color="0.4", ha="right")
bx.set_title(r"$z_s=1$ lensing kernel: the shared-draw channel", fontsize=7)
panel_label(bx, "b")
fig.tight_layout()
fig.savefig(f"{FIG}/r5_fig2_absorption.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ── fig 3: truth-provenance closure ───────────────────────────────────────
fig, ax = plt.subplots(figsize=ONE_COL)
xg = tru["clk_grid"]
sp = tru["clk__sig_pred"]
ax.plot(xg, tru["S_true"], color="#111111", lw=1.5,
        label="TNG300 full hydro (measured)")
ax.plot(xg, tru["S_bind"], color="0.55", lw=1.2,
        label="BIND fiducial paint (measured)")
ax.plot(xg, tru["clk__pred_bind"], color=COLORS["bind"], lw=1.6, ls="--",
        label=r"model $\leftarrow\lambda$ from BIND patches")
ax.plot(xg, tru["clk__pred_truth"], color=COLORS["highlight"], lw=1.6,
        ls=":", label=r"model $\leftarrow\lambda$ from HYDRO patches")
ax.fill_between(xg, tru["clk__pred_bind"] - sp, tru["clk__pred_bind"] + sp,
                color=COLORS["bind"], alpha=0.13, lw=0)
ax.set_xscale("log")
ax.axhline(1, color="0.85", lw=0.5)
ax.set_xlabel(r"$\ell$", fontsize=7)
ax.set_ylabel(r"$S(\ell)$", fontsize=7)
ax.legend(fontsize=5.2, loc="lower left", frameon=False)
ax.set_title(r"swapping $\lambda$'s provenance to zero-noise hydro patches",
             fontsize=7)
fig.tight_layout()
fig.savefig(f"{FIG}/r5_fig3_truth_provenance.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)
print(f"wrote 3 figures into {FIG}")
