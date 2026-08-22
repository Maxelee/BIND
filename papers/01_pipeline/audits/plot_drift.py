import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/claude-2107/-mnt-home-mlee1-BIND/eaad8798-3159-4ecc-ad3b-687c176db4cc/scratchpad/investigation1/thermo_drift_rows.pkl", "rb") as f:
    rows = pickle.load(f)

a = np.array([r["a"] for r in rows])
z = np.array([r["z"] for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.axhline(1.0, color="0.6", lw=1, ls=":")
for key, lab, col in [("fgas_ratio", r"$f_{\rm gas}$ (mass, $a^0$ expected)", "k"),
                       ("T_ratio", r"$T_{\rm mw}$ (a-independent)", "tab:blue"),
                       ("Y_ratio", r"$Y_{500c}$ (mixed)", "tab:green"),
                       ("K_ratio", r"entropy $K$ ($\propto T/n_e^{2/3}$)", "tab:purple"),
                       ("Pe_ratio", r"$P_e$ (density-driven, $\propto a^{-3}$ target)", "tab:red")]:
    y = np.array([r[key] for r in rows])
    ax.plot(a, y, "o-", color=col, label=lab, ms=4)
ax.set_xlabel("scale factor a"); ax.set_ylabel("BIND / truth (median, halo-matched)")
ax.set_title("z-drift by channel: steepness tracks target's a-power")
ax.legend(fontsize=8); ax.grid(alpha=0.2); ax.invert_xaxis()

ax = axes[1]
ax.axhline(1.0, color="0.6", lw=1, ls=":")
for key, lab, col, ls in [("T_ratio_lo", "T, low mass", "tab:blue", "-"),
                           ("T_ratio_hi", "T, high mass", "tab:blue", "--"),
                           ("Pe_ratio_lo", "Pe, low mass", "tab:red", "-"),
                           ("Pe_ratio_hi", "Pe, high mass", "tab:red", "--")]:
    y = np.array([r[key] for r in rows])
    ax.plot(a, y, ls, color=col, label=lab, marker="o", ms=3)
ax.set_xlabel("scale factor a"); ax.set_ylabel("BIND / truth")
ax.set_title("mass-split: low-mass halos drift more (capacity signature)")
ax.legend(fontsize=8); ax.grid(alpha=0.2); ax.invert_xaxis()

fig.suptitle("Investigation 1: z>0 thermodynamic drift diagnostics (halo_atlas, fid vs truth, 20 snaps)")
fig.tight_layout()
fig.savefig("/tmp/claude-2107/-mnt-home-mlee1-BIND/eaad8798-3159-4ecc-ad3b-687c176db4cc/scratchpad/investigation1/thermo_drift_diagnostic.png", dpi=140)
print("wrote thermo_drift_diagnostic.png")
