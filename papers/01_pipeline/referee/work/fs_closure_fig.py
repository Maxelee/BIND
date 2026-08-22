"""Model-closure comparison: does the family-basis model reach the corrected fiducial better?

Reads the closure products rebuilt for the fiducial swap and draws, in the suite style:
  (a) S(l) for the OLD (mis-conditioned, CAMELS-cosmology) fiducial: measured vs model
  (b) S(l) for the NEW (twobound/run_0049, TNG300-cosmology) fiducial: measured vs model
  (c) residuals (model - measured) for both, against the model's own predictive scatter
  (d) per-statistic median |model/measured - 1|, old vs new

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python fs_closure_fig.py
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from paper_style import COLORS, TWO_COL_TALL, panel_label, setup  # noqa: E402

SRC = "/mnt/home/mlee1/ceph/referee_work/fidswap/closure_fidswap.npz"
OUT = "/mnt/home/mlee1/BIND/papers/01_pipeline/referee/figs/fs_model_closure"
BUNDLE = "/mnt/home/mlee1/BIND/papers/01_pipeline/figs_preview/family_model_bundle.npz"

d = np.load(SRC, allow_pickle=True)
ell = d["ell"]
om, op = d["old_meas_clk"], d["oldpred_clk"]
nm, npd = d["new_meas_clk"], d["new_pred_clk"]

# model's shipped per-bin predictive scatter (empirical CV residual sd), if present
b = np.load(BUNDLE, allow_pickle=True)
sig = b["clk__sig_pred"] if "clk__sig_pred" in b.files else None

OLD_C, NEW_C = COLORS["highlight"], COLORS["bind"]
MEAS_C, GREY = COLORS["truth"], COLORS["dmo"]

setup()
fig = plt.figure(figsize=TWO_COL_TALL)
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.26,
              left=0.085, right=0.985, top=0.94, bottom=0.085)

# ---- (a) and (b): measured vs model, one panel each ------------------------
for k, (ax_i, meas, pred, col, tag, sub) in enumerate([
    (gs[0, 0], om, op, OLD_C, "(a)", "old fiducial  (CAMELS cosmology)"),
    (gs[0, 1], nm, npd, NEW_C, "(b)", "new fiducial  (TNG300 cosmology)"),
]):
    ax = fig.add_subplot(ax_i)
    ax.fill_between(ell, meas, pred, color=col, alpha=0.20, lw=0, zorder=1)
    ax.plot(ell, meas, color=MEAS_C, lw=1.6, zorder=3, label="measured")
    ax.plot(ell, pred, color=col, lw=1.6, ls="--", dashes=(4, 2), zorder=4,
            label="family-basis model")
    ax.set_xscale("log")
    ax.set_xlim(ell.min(), ell.max())
    ax.set_ylim(0.865, 1.012)
    ax.set_xlabel(r"multipole $\ell$")
    if k == 0:
        ax.set_ylabel(r"$S(\ell)=C_\ell^{\kappa\kappa}/C_\ell^{\kappa\kappa,\rm DMO}$")
    else:
        ax.set_yticklabels([])
    ax.axhline(1.0, color=GREY, lw=0.6, ls=":", zorder=0)
    panel_label(ax, tag, "lower left")
    ax.text(0.035, 0.155, sub, transform=ax.transAxes, fontsize=6.6, color=col,
            va="bottom", ha="left")
    med = np.median(np.abs(pred - meas))
    ax.text(0.965, 0.06, f"median |resid| = {med:.4f}", transform=ax.transAxes,
            fontsize=6.6, ha="right", va="bottom", color=MEAS_C)
    ax.legend(loc="upper right", handlelength=1.9, borderaxespad=0.4)

# ---- (c) residuals, both, against the model's own scatter -------------------
axc = fig.add_subplot(gs[1, 0])
if sig is not None and len(sig) == len(ell):
    axc.fill_between(ell, -sig, sig, color=GREY, alpha=0.28, lw=0, zorder=0,
                     label=r"model $\sigma_{\rm pred}$")
axc.axhline(0.0, color=GREY, lw=0.6, zorder=1)
axc.plot(ell, op - om, color=OLD_C, lw=1.6, marker="o", ms=2.6, zorder=3, label="old fiducial")
axc.plot(ell, npd - nm, color=NEW_C, lw=1.6, marker="s", ms=2.6, zorder=4, label="new fiducial")
axc.set_xscale("log")
axc.set_xlim(ell.min(), ell.max())
axc.set_xlabel(r"multipole $\ell$")
axc.set_ylabel(r"model $-$ measured")
panel_label(axc, "(c)", "upper left")
axc.legend(loc="lower left", handlelength=1.9, borderaxespad=0.4)

# ---- (d) per-statistic closure ---------------------------------------------
axd = fig.add_subplot(gs[1, 1])
stats = [("clk", r"$S(\ell)$"), ("pk", "peaks"), ("mn", "minima"),
         ("v0", r"$V_0$"), ("v1", r"$V_1$"), ("v2", r"$V_2$"), ("pdf", r"$\kappa$ PDF")]
old_e, new_e, labs = [], [], []
for s, lab in stats:
    mo, po_ = d[f"old_meas_{s}"], d[f"oldpred_{s}"]
    mn_, pn_ = d[f"new_meas_{s}"], d[f"new_pred_{s}"]
    if s == "clk":                      # absolute, not fractional (S is already a ratio)
        a = np.median(np.abs(po_ - mo)) * 100
        c = np.median(np.abs(pn_ - mn_)) * 100
    else:
        k1 = mo > 0.02 * mo.max()
        k2 = mn_ > 0.02 * mn_.max()
        a = np.median(np.abs(po_[k1] / mo[k1] - 1)) * 100
        c = np.median(np.abs(pn_[k2] / mn_[k2] - 1)) * 100
    old_e.append(a)
    new_e.append(c)
    labs.append(lab)
y = np.arange(len(labs))
h = 0.36
axd.barh(y + h / 2, old_e, height=h, color=OLD_C, label="old fiducial",
         edgecolor="white", linewidth=0.6)
axd.barh(y - h / 2, new_e, height=h, color=NEW_C, label="new fiducial",
         edgecolor="white", linewidth=0.6)
axd.set_yticks(y)
axd.set_yticklabels(labs)
axd.invert_yaxis()
axd.set_xscale("log")
axd.set_xlabel("median closure error  [%]")
axd.set_xlim(8e-3, 30)
for i, (a, c) in enumerate(zip(old_e, new_e)):
    axd.text(c * 0.82, i - h / 2, f"{c:.2f}", fontsize=5.8, ha="right", va="center",
             color=MEAS_C)
panel_label(axd, "(d)", "upper right")
axd.legend(loc="center right", handlelength=1.4, borderaxespad=0.6)

fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
print("wrote", OUT + ".png / .pdf")
print("\nS(l) median |resid|:  old %.4f   new %.4f   (%.1fx better)"
      % (np.median(np.abs(op - om)), np.median(np.abs(npd - nm)),
         np.median(np.abs(op - om)) / np.median(np.abs(npd - nm))))
for lab, a, c in zip(labs, old_e, new_e):
    print(f"  {lab:12s} old {a:7.3f}%   new {c:7.3f}%")
