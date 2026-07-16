#!/usr/bin/env python
"""fig05_spearman_response.png -- model-free (emulator-free) per-bin Spearman
response of seven WL field statistics to the 30 SB35 astro parameters.

For each statistic (C_ell suppression S(ell), peak counts, minima counts,
kappa PDF, Minkowski V0/V1/V2), computes the Spearman rank correlation of
every bin (multipole ell, or threshold nu/kappa) with every one of the 30
astro parameters, over the n=253 measured Sobol runs at source plane z_s=1.
Deliberately emulator-free (rank correlation on the measured data vectors
only). Columns (parameters) are ordered once, by peak |Spearman r| with the
C_ell suppression, and that order is reused in every panel.

Data (already cached, read-only):
  /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz
    t__suppression__value/valid, a__suppression__ell (or a__cl_kappa__ell,
      identical grid -- suppression uses its own axis key), X_unit,
      param_names, t__peak_counts__value/valid, a__peak_counts__nu,
      t__minima_counts__value/valid, a__minima_counts__nu,
      t__pdf__value/valid, a__pdf__pdf_bins,
      t__mf_v0/v1/v2__value/valid, a__mf_v0/v1/v2__mf_nu

Source: sobol-sb35/examples/wl_latent_sbi.ipynb cell 3 (SS1), engine
functions `bin_param_correlation`/`param_importance`/`stat_axis` in
sobol-sb35/examples/wl_latent_sbi.py (lines ~211-277). Ported verbatim below
(pure numpy/scipy rank correlation on a cached array -- no fitting, no
emulator).

Placeholder this replaces: figs/fig05_spearman_response.png (byte-identical
to examples/wl_latent_sbi_figs/f1_corr_heatmaps.png).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import TWO_COL, panel_label, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

DATASET = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")
ELL_MAX_DEFAULT = 2.0e4
Z_IDX = 1  # source plane index -> z_s = 1.0

STATS1 = ["suppression", "peak_counts", "minima_counts", "pdf", "mf_v0", "mf_v1", "mf_v2"]
TITLES = {"suppression": r"$C_\ell$ suppression $S(\ell)$", "peak_counts": "peak counts",
          "minima_counts": "minima counts", "pdf": r"$\kappa$ PDF",
          "mf_v0": "Minkowski $V_0$", "mf_v1": "Minkowski $V_1$", "mf_v2": "Minkowski $V_2$"}
REBIN = {"peak_counts": 4, "minima_counts": 4, "pdf": 3, "mf_v0": 2, "mf_v1": 2, "mf_v2": 2}
AXIS_LABEL = {"ell": r"$\ell$", "nu": r"$\nu=\kappa/\sigma$", "pdf_bins": r"$\kappa$",
              "mf_nu": r"$\nu$"}


def stat_axis(d, stat):
    keys = [k for k in d.files if k.startswith(f"a__{stat}__")]
    if not keys:
        raise KeyError(f"no a__{stat}__* axis in dataset")
    return keys[0].split("__")[-1], np.asarray(d[keys[0]])


def _rank_cols(a):
    return np.column_stack([rankdata(a[:, j]) for j in range(a.shape[1])])


def _coarsen(Y, axis_vals, factor):
    if not factor or factor <= 1:
        return Y, axis_vals
    n = Y.shape[1]
    edges = list(range(0, n, factor)) + [n]
    with np.errstate(invalid="ignore"):
        Yc = np.column_stack([np.nanmean(Y[:, a:b], axis=1) for a, b in zip(edges[:-1], edges[1:])])
        ax = np.array([np.nanmean(axis_vals[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    return Yc, ax


def bin_param_correlation(d, stat, z_idx=1, rebin=1):
    val = np.asarray(d[f"t__{stat}__value"])
    valid = np.asarray(d[f"t__{stat}__valid"], bool)
    axis_name, axis = stat_axis(d, stat)
    Y = val[valid]
    z_s = None
    if Y.ndim == 4:
        z_s = float(d["source_redshifts"][z_idx]); Y = Y[:, z_idx, z_idx, :]
    elif Y.ndim == 3:
        z_s = float(d["source_redshifts"][z_idx]); Y = Y[:, z_idx, :]
    X = np.asarray(d["X_unit"])[valid]
    Y, axis = _coarsen(Y.astype(float), axis, rebin)
    good = np.isfinite(Y).all(0) & (Y.std(0) > 0)
    X = _rank_cols(X)
    Yr = np.full_like(Y, np.nan)
    Yr[:, good] = _rank_cols(Y[:, good]); Y = Yr
    Xn = X - X.mean(0); Xn /= (np.linalg.norm(Xn, axis=0) + 1e-12)
    corr = np.full((Y.shape[1], X.shape[1]), np.nan)
    Yg = Y[:, good]; Yn = Yg - Yg.mean(0); Yn /= (np.linalg.norm(Yn, axis=0) + 1e-12)
    corr[good] = Yn.T @ Xn
    return dict(corr=corr, axis=axis, axis_name=axis_name, z_s=z_s, n_runs=int(valid.sum()))


def param_importance(d, stat, **kw):
    c = bin_param_correlation(d, stat, **kw)["corr"]
    return np.argsort(-np.nanmax(np.abs(c), axis=0))


def param_importance_ell_restricted(d, stat, ell_lo=100.0, ell_hi=ELL_MAX_DEFAULT, **kw):
    """Same ranking as param_importance, but peak |r| only over the ell band
    actually plotted/quoted elsewhere in this script (ell in [ell_lo, ell_hi]),
    not the full unrestricted axis. Used for the shared column order so the
    figure's own x-axis ordering matches the printed top-driver diagnostic
    (and main.tex's caption, which cites the ell-restricted ranking)."""
    res = bin_param_correlation(d, stat, **kw)
    c = res["corr"]
    if res["axis_name"] == "ell":
        keep = (res["axis"] >= ell_lo) & (res["axis"] <= ell_hi)
        c = c[keep]
    return np.argsort(-np.nanmax(np.abs(c), axis=0))


def main():
    setup()
    d = np.load(DATASET, allow_pickle=True)
    pn = [str(s) for s in d["param_names"]]
    # ell-restricted order (matches the printed top-driver diagnostic below
    # and the caption's explicit |r| ranking) -- NOT the unrestricted
    # param_importance, which disagrees on the #1/#2 ranking.
    order = param_importance_ell_restricted(d, "suppression", z_idx=Z_IDX)

    # one column of 7 stacked panels (shared x = the 30-param order) so the
    # parameter labels appear once, at full width, instead of 7x illegibly small
    results = []
    for s in STATS1:
        res = bin_param_correlation(d, s, z_idx=Z_IDX, rebin=REBIN.get(s, 1))
        name, yv, C = res["axis_name"], res["axis"], res["corr"][:, order]
        if name == "ell":
            keep = (yv >= 100) & (yv <= ELL_MAX_DEFAULT)
            C, yv = C[keep], yv[keep]
        results.append((s, res, name, yv, C))
    vmax = max(np.nanpercentile(np.abs(C), 99) for *_, C in results)

    fig, axes = plt.subplots(len(STATS1), 1, figsize=(TWO_COL[0], 8.6), sharex=True,
                              constrained_layout=True)
    letters = "abcdefg"
    im = None
    for i, (s, res, name, yv, C) in enumerate(results):
        ax = axes[i]
        im = ax.imshow(C, aspect="auto", origin="lower", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, rasterized=True)
        ti = np.linspace(0, len(yv) - 1, min(4, len(yv))).astype(int)
        fmt = ("{:.0f}".format if name == "ell" else "{:.1f}".format)
        ax.set_yticks(ti); ax.set_yticklabels([fmt(yv[j]) for j in ti], fontsize=6.5)
        ax.set_ylabel(AXIS_LABEL.get(name, name), fontsize=7.5, labelpad=1)
        panel_label(ax, f"({letters[i]}) {TITLES[s]}", loc="upper right")
    axes[-1].set_xticks(range(30))
    axes[-1].set_xticklabels([pn[j] for j in order], rotation=90, fontsize=6.5)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.022, pad=0.012, aspect=55)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Spearman $r$", fontsize=8)
    axes[0].text(1.0, 1.28, f"$n={results[0][1]['n_runs']}$, $z_s={results[0][1]['z_s']:g}$",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=7.5)

    save(fig, "figs/fig05_spearman_response")

    cs = bin_param_correlation(d, "suppression", z_idx=Z_IDX)
    ell = cs["axis"]; band = (ell >= 100) & (ell <= ELL_MAX_DEFAULT)
    Cband = cs["corr"][band]; peak = np.nanmax(np.abs(Cband), axis=0)
    print(f"top C_l-suppression drivers (|Spearman r|, 100<ell<{ELL_MAX_DEFAULT:.0f}, z_s={cs['z_s']:g}):")
    for i in np.argsort(-peak)[:8]:
        print(f"    {pn[i]:34s} |r|={peak[i]:.2f}")


if __name__ == "__main__":
    main()
