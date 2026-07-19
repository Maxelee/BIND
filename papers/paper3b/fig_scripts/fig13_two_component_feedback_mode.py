#!/usr/bin/env python
"""Figure: two-component radial -> feedback-mode decomposition (p7).
(a) normalized radial shape s=y(r)/y(8') per science bin vs the 313-unit grid
    shape envelope; (b) data/model per radius with amplitude-only A(b) line;
    (c) verdict Delta(b) bars (raw / grid-MC / sigma_pos-inflated) vs 2sigma line,
    with CIB-deprojection range whiskers.
House style: paper_style.py.  Saves PDF (figs/) + PNG preview (figs_preview/).
"""
import sys, json
import numpy as np
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import setup, save, panel_label, COLORS, TWO_COL_TALL
import matplotlib.pyplot as plt

B = "/mnt/home/mlee1/ceph/paper3/B"
STACK = f"{B}/wp2_measurement/stack_wiener_sm2am_fid.npz"
FID   = f"{B}/wp4_mocks/grid_tfwiener/bind_run_0000.npz"
TWO   = f"{B}/wp4_mocks/model_grid_tfwiener.npz"
SOBOL = f"{B}/wp4_mocks/sobol/model_grid_tfwiener_sb35.npz"
JSON  = f"{B}/wp5_inference/two_component_feedback_mode.json"
OUTSTEM = "/mnt/home/mlee1/BIND-paper3b/papers/paper3b/figs/fig13_two_component_feedback_mode"

ds = np.load(STACK, allow_pickle=True)
y_data = ds["y_mean"]; y_cov = ds["y_cov"]; radii = ds["cap_radii_arcmin"]
y_fid = np.load(FID, allow_pickle=True)["y_mean"]
y_two = np.load(TWO, allow_pickle=True)["y_mean"]
y_so  = np.load(SOBOL, allow_pickle=True)["y_mean"]
with open(JSON) as f:
    R = json.load(f)

R4 = 4
SCI = [1, 2, 3, 4]
labels = [r"$\nu\,[1,2)$", r"$\nu\,[2,3)$", r"$\nu\,[3,4)$", r"$\nu\,[4,12)$"]
colors = plt.cm.cividis(np.linspace(0.08, 0.85, 4))

# normalized shapes
def s_norm(y, b):
    return y[..., b, :] / y[..., b, R4:R4+1]

# per-radius jk sigma on s (linearized ratio through {r,4} sub-cov)
def sigma_s(b):
    out = np.zeros(5)
    y4 = y_data[b, R4]; v = y_cov[b]
    for r in range(5):
        yr = y_data[b, r]; s = yr / y4
        out[r] = abs(s) * np.sqrt(v[r, r]/yr**2 + v[R4, R4]/y4**2 - 2*v[r, R4]/(yr*y4))
    return out

setup()
fig = plt.figure(figsize=TWO_COL_TALL)
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.32,
                      left=0.08, right=0.985, top=0.965, bottom=0.095)

# ---- Panel (a): normalized shape small-multiples per science bin ------
grid_all = np.concatenate([y_two, y_so], axis=0)   # (313,5,5)
for j, b in enumerate(SCI):
    ax = fig.add_subplot(gs[0, j])
    sg = grid_all[:, b, :] / grid_all[:, b, R4:R4+1]       # (313,5)
    ax.fill_between(radii, sg.min(0), sg.max(0), color=COLORS["dmo"], alpha=0.35,
                    lw=0, label="grid" if j == 3 else None)
    ax.plot(radii, y_fid[b] / y_fid[b, R4], color=COLORS["dmo"], lw=1.1,
            label="TNG fid" if j == 3 else None)
    sd = y_data[b] / y_data[b, R4]
    ax.errorbar(radii, sd, yerr=sigma_s(b), fmt="o", ms=3.0, color=COLORS["truth"],
                lw=0.9, capsize=1.5, label="data" if j == 3 else None, zorder=5)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 4, 6, 8]); ax.set_xticklabels(["2", "3", "4", "6", "8"])
    ax.set_ylim(-0.03, 1.08)
    ax.text(0.62, 0.16, labels[j], transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color=colors[j], fontweight="bold")
    if j == 0:
        ax.set_ylabel(r"$s=y(\theta)/y(8')$")
    else:
        ax.set_yticklabels([])
    if j == 3:
        ax.legend(loc="upper left", fontsize=5.8, handlelength=1.1, borderpad=0.2)
    ax.set_xlabel(r"$\theta$ [arcmin]")
panel_label(fig.add_subplot(gs[0, 0], frame_on=False, xticks=[], yticks=[]), "(a)")

# ---- Panel (b): data/model per radius with amplitude-only A(b) line ---
axb = fig.add_subplot(gs[1, 0:2])
A_perbin = R["component1"]["per_bin_own_amplitude"]
for j, b in enumerate(SCI):
    ratio = y_data[b] / y_fid[b]
    axb.plot(radii, ratio, "o-", ms=3.0, lw=0.9, color=colors[j], label=labels[j])
    axb.axhline(A_perbin[f"bin{b}"], color=colors[j], ls=":", lw=0.8)
axb.set_xscale("log")
axb.set_xticks([2, 3, 4, 6, 8]); axb.set_xticklabels(["2", "3", "4", "6", "8"])
axb.set_xlabel(r"$\theta$ [arcmin]")
axb.set_ylabel(r"data / model  $\langle Y\rangle$")
axb.legend(loc="lower right", fontsize=6, ncol=1, handlelength=1.3, borderpad=0.3)
axb.text(0.5, 0.985, "dotted: amp-only $A(\\nu)$", transform=axb.transAxes,
         ha="center", va="top", fontsize=6, color="0.35")
panel_label(axb, "(b)", loc="upper left")

# ---- Panel (c): verdict Delta bars + 2sigma line + CIB range ----------
axc = fig.add_subplot(gs[1, 2:4])
Draw = np.array(R["verdict"]["Delta_raw"])
Dmc  = np.array(R["verdict"]["Delta_grid_mc_inflated"])
Dpos = np.array(R["verdict"]["Delta_sigma_pos_inflated"])
cib  = R["cib_deproj_robustness"]["Delta_range_per_bin_min_max"]
x = np.arange(4); w = 0.26
axc.bar(x - w, Draw, w, color=COLORS["highlight"], label="raw")
axc.bar(x,      Dmc,  w, color=COLORS["bind"],      label="+grid MC")
axc.bar(x + w,  Dpos, w, color=COLORS["dmo"],       label=r"+$\sigma_{\rm pos}$")
# CIB-deprojection range whiskers on the raw bars
for i, b in enumerate(SCI):
    lo, hi = cib[f"bin{b}"]
    axc.plot([x[i]-w, x[i]-w], [lo, hi], color=COLORS["truth"], lw=0.9, zorder=6)
    axc.plot([x[i]-w-0.06, x[i]-w+0.06], [lo, lo], color=COLORS["truth"], lw=0.9)
    axc.plot([x[i]-w-0.06, x[i]-w+0.06], [hi, hi], color=COLORS["truth"], lw=0.9)
axc.axhline(2.0, ls="--", lw=0.8, color="0.3")
axc.axhline(0.0, lw=0.6, color="0.6")
axc.set_xticks(x); axc.set_xticklabels([r"$[1,2)$", r"$[2,3)$", r"$[3,4)$", r"$[4,12)$"],
                                       fontsize=6.2)
axc.set_xlabel(r"peak-significance bin $\nu$")
axc.set_ylabel(r"$\Delta(\nu)=(\min_{\rm grid}C-C_{\rm data})/\sigma_C$")
axc.legend(loc="upper right", fontsize=6, handlelength=1.0, ncol=1, borderpad=0.25)
axc.text(1.5, 2.14, r"$2\sigma$", fontsize=6, color="0.3", ha="center", va="bottom")
axc.text(0.02, 0.02, "whisker: CIB-deproj range", transform=axc.transAxes,
         ha="left", va="bottom", fontsize=5.6, color="0.35")
panel_label(axc, "(c)")

save(fig, OUTSTEM)
print("panel A: data central shape below grid band in 4/4 bins")
print("Delta_raw", Draw, "Delta_pos", Dpos)
