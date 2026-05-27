"""Projection effects figure for the BIND paper (§5.5).

Compares the BIND "2D" model (50 Mpc/h LOS projection) with a BIND "cube"
model trained on 6.25 Mpc/h (patch-depth) projections. Both run on the
same 128x128 grid, same halos, same cosmology. The point of the figure
is to (i) show that LOS projection meaningfully changes the truth
distributions of certain observables and (ii) demonstrate that each
model is faithful to its own projection depth.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

CACHE_2D = "analysis_physics_cache/obs_fm_two_head.npz"
CACHE_CUBE = "analysis_physics_cache/obs_fm_cube_two_head.npz"
OUT_PDF = "paper_figures/fig_projection_2d_vs_cube.pdf"
OUT_PNG = "paper_figures/fig_projection_2d_vs_cube.png"


def load_cv(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    sel = d["suite"] == "CV"
    out = {k: d[k][sel] for k in d.files}
    return out


d2 = load_cv(CACHE_2D)
dc = load_cv(CACHE_CUBE)

# Match halos by (sim_id, logM) — both arrays are already in sim_id order
# but cube has 1073 while 2D has 1154, so we trim 2D to the cube indices
# using a (sim_id, logM) key. The first 1073 typically match in order.
key2 = np.array([f"{s}|{m:.6f}" for s, m in zip(d2["sim_id"], d2["logM"])])
keyc = np.array([f"{s}|{m:.6f}" for s, m in zip(dc["sim_id"], dc["logM"])])
idx2 = {k: i for i, k in enumerate(key2)}
matched_2d = np.array([idx2[k] for k in keyc if k in idx2])
matched_cube = np.array([i for i, k in enumerate(keyc) if k in idx2])
print(f"matched halos: {matched_2d.size} / {keyc.size} (cube) / {key2.size} (2D)")


def get(d, key, idx):
    v = d[key][idx]
    return v


# Observables to compare — chosen to span LOS-contamination sensitivity
PANELS = [
    ("f_b_norm", r"$f_{b}/f_{b,{\rm cosmic}}$", False, None),
    ("q_DM",     r"$q_{\rm DM}$",               False, None),
    ("dq_DM",    r"$\Delta q_{\rm DM}$",         False, None),
    ("Sigma_gas_c", r"$\log_{10}\,\Sigma_{\rm gas,c}$", True, None),
]

# Per-panel x-range overrides (set after first pass if needed)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
LW = 1.8

for ax, (key, latex, take_log, _xlim) in zip(axes, PANELS):
    series = []  # list of (vals, color, ls, label)
    for d, idx, color, kind in [
        (d2, matched_2d, "0.30", "truth_2D"),
        (d2, matched_2d, "tab:red", "BIND_2D"),
        (dc, matched_cube, "tab:blue", "truth_cube"),
        (dc, matched_cube, "tab:orange", "BIND_cube"),
    ]:
        prefix = "truth_" if "truth" in kind else "gen_"
        v = get(d, prefix + key, idx).astype(np.float64)
        ok = np.isfinite(v)
        if take_log:
            ok &= v > 0
            v = np.log10(v[ok])
        else:
            v = v[ok]
        series.append((v, color, kind))

    # Set a robust shared x-range from the union of all four (5..95 pct)
    all_v = np.concatenate([s[0] for s in series])
    lo, hi = np.nanpercentile(all_v, [1, 99])
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    xs = np.linspace(lo, hi, 256)

    # KDE plot
    for v, color, kind in series:
        if v.size < 5:
            continue
        try:
            kde = gaussian_kde(v, bw_method=0.4)
            ax.plot(xs, kde(xs), color=color, lw=LW,
                    ls="-" if "truth" in kind else "--",
                    label=kind.replace("_", " "))
        except Exception:
            ax.hist(v, bins=30, range=(lo, hi), density=True,
                    histtype="step", color=color, lw=LW,
                    ls="-" if "truth" in kind else "--",
                    label=kind.replace("_", " "))

    ax.set_xlim(lo, hi)
    ax.set_xlabel(latex, fontsize=11)
    ax.set_ylabel("density", fontsize=10)
    ax.grid(alpha=0.2, ls=":")
    if ax is axes[0]:
        ax.legend(fontsize=8, loc="best", frameon=False, ncol=1)

fig.suptitle(
    "Projection effects: BIND-2D (50 Mpc/h LOS) vs BIND-cube "
    f"(6.25 Mpc/h LOS), CV suite, $N_{{\\rm halos}}={matched_2d.size}$",
    fontsize=11, y=1.02,
)
fig.tight_layout()
os.makedirs("paper_figures", exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight", dpi=200)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
print(f"Saved {OUT_PDF}")
print(f"Saved {OUT_PNG}")


# ---------------------------------------------------------------------------
# Sidecar summary numbers used in the paper text

print("\nProjection-effect summary on matched CV halos:")
print(f"{'observable':<14s} {'truth_2D':>10s} {'truth_cube':>11s} "
      f"{'2D-cube':>9s} {'BIND_2D':>9s} {'BIND_cube':>10s}")
for key, _, take_log, _ in PANELS:
    v_t2 = d2[f"truth_{key}"][matched_2d].astype(float)
    v_g2 = d2[f"gen_{key}"][matched_2d].astype(float)
    v_tc = dc[f"truth_{key}"][matched_cube].astype(float)
    v_gc = dc[f"gen_{key}"][matched_cube].astype(float)
    if take_log:
        for a in (v_t2, v_g2, v_tc, v_gc):
            a[a <= 0] = np.nan
        v_t2 = np.log10(v_t2); v_g2 = np.log10(v_g2)
        v_tc = np.log10(v_tc); v_gc = np.log10(v_gc)
    m_t2 = np.nanmedian(v_t2); m_g2 = np.nanmedian(v_g2)
    m_tc = np.nanmedian(v_tc); m_gc = np.nanmedian(v_gc)
    print(f"{key:<14s} {m_t2:>10.3f} {m_tc:>11.3f} "
          f"{m_t2 - m_tc:>9.3f} {m_g2 - m_t2:>9.3f} {m_gc - m_tc:>10.3f}")
