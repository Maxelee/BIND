import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "/tmp/claude-2107/-mnt-home-mlee1-BIND/eaad8798-3159-4ecc-ad3b-687c176db4cc/scratchpad/inv3"

d200 = np.load(f"{OUTDIR}/radial_texture_results.npz")
dfull = np.load(f"{OUTDIR}/radial_texture_results_FULL.npz")

labels = [s.decode() if isinstance(s, bytes) else str(s) for s in dfull["coarse_labels"]]
x_annulus = np.arange(4)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# --- Panel A: fine radial mean profile ---
ax = axes[0, 0]
fc = dfull["fine_centers"]
good = np.isfinite(fc) & (fc <= 4.0)
ax.plot(fc[good], dfull["fine_profile_fid"][good], color="#d95f02", lw=2, label="BIND (painted)")
ax.plot(fc[good], dfull["fine_profile_truth"][good], color="#1b9e77", lw=2, label="TNG (truth)")
ax.set_yscale("log")
ax.set_xlabel(r"$x = R / r_{200c}$")
ax.set_ylabel(r"mean Compton-$y$")
ax.set_title(f"(a) azimuthal mean $y$ profile, N={int(dfull['n_halo'])} group halos\nsnap 096 (z=0.034), raw patches (pre-composite)")
ax.legend(frameon=False)
for e in [0.5, 1, 2]:
    ax.axvline(e, color="gray", lw=0.6, ls=":")
ax.set_xlim(0, 4)

# --- Panel B: fluctuation power (variance of azimuthal residual) by annulus ---
ax = axes[0, 1]
w = 0.18
ax.bar(x_annulus - 1.5*w, dfull["var_truth"], width=w, color="#1b9e77", label="truth")
ax.bar(x_annulus - 0.5*w, dfull["var_fid"], width=w, color="#d95f02", label="painted (raw)")
ax.bar(x_annulus + 0.5*w, dfull["var_mit_s1"], width=w, color="#e6ab02", label=r"painted, mit. $\sigma$=1px")
ax.bar(x_annulus + 1.5*w, dfull["var_mit_s2"], width=w, color="#a6761d", label=r"painted, mit. $\sigma$=2px")
ax.set_yscale("log")
ax.set_xticks(x_annulus, labels)
ax.set_xlabel(r"annulus $x=R/r_{200c}$")
ax.set_ylabel(r"Var[$y$ $-$ azimuthal mean] (small-scale texture power)")
ax.set_title("(b) fluctuation power: painted is UNDER-textured\nfor $x>1$, OVER-textured for $x<0.5$")
ax.legend(frameon=False, fontsize=8)

# --- Panel C: ratio (excess %) with mitigation, showing gap does not close ---
ax = axes[1, 0]
ratio_raw = 100 * (dfull["var_fid"] / dfull["var_truth"] - 1)
ratio_s1 = 100 * (dfull["var_mit_s1"] / dfull["var_truth"] - 1)
ratio_s2 = 100 * (dfull["var_mit_s2"] / dfull["var_truth"] - 1)
ax.plot(x_annulus, ratio_raw, "o-", color="#d95f02", lw=2, ms=8, label="painted (raw)")
ax.plot(x_annulus, ratio_s1, "s--", color="#e6ab02", lw=2, ms=7, label=r"mit. $\sigma$=1px")
ax.plot(x_annulus, ratio_s2, "^--", color="#a6761d", lw=2, ms=7, label=r"mit. $\sigma$=2px")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x_annulus, labels)
ax.set_xlabel(r"annulus $x=R/r_{200c}$")
ax.set_ylabel(r"(Var$_{painted}$/Var$_{truth}$ $-$ 1) [%]")
ax.set_title("(c) mitigation prototype: smoothing $x>1$\nWIDENS the deficit, does not close a gap")
ax.legend(frameon=False)

# --- Panel D: attribution + correlation ---
ax = axes[1, 1]
ax2 = ax.twinx()
excess = dfull["excess_ss"]
colors = ["#d95f02" if e > 0 else "#377eb8" for e in excess]
ax.bar(x_annulus - 0.2, excess, width=0.4, color=colors, label="excess sum-of-squares\n(painted $-$ truth)")
ax.set_yscale("symlog", linthresh=1e-9)
for a in range(4):
    ax.annotate(f"{excess[a]:.1e}", (x_annulus[a]-0.2, excess[a]),
                textcoords="offset points", xytext=(0, 6 if excess[a] > 0 else -14),
                ha="center", fontsize=7.5)
ax2.plot(x_annulus + 0.2, dfull["corr"], "D-", color="#7570b3", ms=8, lw=2, label="texture corr(painted,truth)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x_annulus, labels)
ax.set_xlabel(r"annulus $x=R/r_{200c}$")
ax.set_ylabel("excess residual sum-of-squares (symlog)", color="#d95f02")
ax2.set_ylabel("azimuthal-residual correlation coeff.", color="#7570b3")
ax2.set_ylim(0, 1)
ax.set_title("(d) attribution: net texture excess is core-driven\n(orange bar) even though only outskirts show a\nreal field-level Cl$_{yy}$ excess -> mechanism mismatch")

fig.suptitle("Investigation 3: radial decomposition of painted-vs-truth $y$ texture "
              "(snap_096, group-scale halos $10^{13}$-$10^{14}\\,M_\\odot/h$)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUTDIR}/investigation3_diffuse_texture.png", dpi=150)
print("saved figure")

# quick summary text file
with open(f"{OUTDIR}/summary_numbers.txt", "w") as fh:
    fh.write(f"N halos (full sample) = {int(dfull['n_halo'])}\n")
    fh.write(f"N halos (subsample, matches task's '~200') = {int(d200['n_halo'])}\n\n")
    for tag, d in [("FULL(N=2784)", dfull), ("SUB(N=200)", d200)]:
        fh.write(f"--- {tag} ---\n")
        for a, lab in enumerate(labels):
            fh.write(f"  x={lab:6s}  meanY ratio={d['mean_y_fid'][a]/d['mean_y_truth'][a]:.3f}  "
                     f"var ratio(raw)={d['var_fid'][a]/d['var_truth'][a]:.3f}  "
                     f"var ratio(mit s=1)={d['var_mit_s1'][a]/d['var_truth'][a]:.3f}  "
                     f"var ratio(mit s=2)={d['var_mit_s2'][a]/d['var_truth'][a]:.3f}  "
                     f"corr={d['corr'][a]:.3f}\n")
        fh.write("\n")
print("saved summary_numbers.txt")
