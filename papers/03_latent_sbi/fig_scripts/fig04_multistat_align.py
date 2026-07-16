#!/usr/bin/env python
"""fig04_multistat_align.png -- do the other WL summary statistics share the
kappa-C_ell 2-D feedback latent?

For each of eight WL/tSZ summary statistics (peak counts, minima counts,
kappa PDF, Minkowski V0/V1/V2, C_ell^{kappa y}, C_ell^{yy}), reduce it to its
own top-2 PCA latent, then measure the canonical correlation of that latent
with the top-2 PCA latent of the reference statistic (kappa_2pt, i.e. the
auto C_ell suppression stack). Bar height = |canonical correlation| for
latent-1 (blue) and latent-2 (orange) per statistic. All quantities are a
closed-form PCA + CCA on cached arrays -- no emulator, no training.

Data (already cached, read-only):
  /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz
    t__cl_kappa__value (253,5,5,724), t__cl_kappa__valid (253,)
    t__peak_counts__value/valid, t__minima_counts__value/valid,
    t__pdf__value/valid, t__mf_v0/v1/v2__value/valid,
    t__cl_kappa_y__value/valid, t__cl_yy__value/valid
    a__cl_kappa_y__ell, a__cl_yy__ell

Source: sobol-sb35/examples/wl_latent_sbi.ipynb cell 12 (SS2d), which calls
into sobol-sb35/examples/wl_stat_latents.py (load_stat/reduce_stat). The
small load_stat/reduce_stat helpers are ported verbatim below (pure
numpy/sklearn PCA reduction of a cached array -- not a re-run of any
science engine).

Placeholder this replaces: figs/fig04_multistat_align.png (byte-identical
to examples/wl_latent_sbi_figs/f2d_multistat_align.png).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

DATASET = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")
ELL_MAX = 36864.0  # map Nyquist, matches wl_stat_latents.load_stat default

# per-statistic transform (verbatim from wl_stat_latents.TRANSFORM)
TRANSFORM = {
    "peak_counts": "log1p", "minima_counts": "log1p", "pdf": "log1p",
    "mf_v0": "raw", "mf_v1": "raw", "mf_v2": "raw",
    "cl_kappa_y": "raw", "cl_yy": "log",
}

STATS2 = ["peak_counts", "minima_counts", "pdf", "mf_v0", "mf_v1", "mf_v2",
          "cl_kappa_y", "cl_yy"]
LABELS = {"peak_counts": "peaks", "minima_counts": "minima", "pdf": "PDF",
          "mf_v0": r"MF $V_0$", "mf_v1": r"MF $V_1$", "mf_v2": r"MF $V_2$",
          "cl_kappa_y": r"$C_\ell^{\kappa y}$", "cl_yy": r"$C_\ell^{yy}$"}


def _apply(x, how):
    with np.errstate(divide="ignore", invalid="ignore"):
        if how == "log":
            return np.log10(np.abs(x) + 1e-30)
        if how == "log1p":
            return np.log10(1.0 + np.clip(x, 0, None))
    return x


def load_stat(d, name):
    """Return (feature matrix (Nall, d), valid mask (Nall,)) for a statistic."""
    if name == "kappa_2pt":
        cl = d["t__cl_kappa__value"]
        ell = d["a__cl_kappa__ell"]
        sel = ell <= ELL_MAX
        autos = np.stack([cl[:, i, i, :] for i in range(cl.shape[-2])], axis=1)
        X = _apply(autos[:, :, sel], "log")
        valid = d["t__cl_kappa__valid"]
    else:
        X = d[f"t__{name}__value"]
        valid = d[f"t__{name}__valid"]
        if name in ("cl_kappa_y", "cl_yy"):
            ax = d.get(f"a__{name}__ell")
            if ax is not None:
                sel = ax <= ELL_MAX
                X = X[..., sel]
        X = _apply(X, TRANSFORM.get(name, "raw"))
    X = X.reshape(X.shape[0], -1)
    return X, np.asarray(valid, bool)


def reduce_stat(X, n=2):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xz = np.nan_to_num((X - mu) / sd)
    k = min(n, Xz.shape[1], Xz.shape[0] - 1)
    p = PCA(n_components=k).fit(Xz)
    return p.transform(Xz), p.explained_variance_ratio_


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    setup()
    d = np.load(DATASET, allow_pickle=True)

    Xk, vk = load_stat(d, "kappa_2pt")
    Zk, _ = reduce_stat(Xk[vk], n=2)

    a1, a2 = [], []
    for nm in STATS2:
        Xs, vs = load_stat(d, nm)
        common = vs & vk
        Za, _ = reduce_stat(Xs[common], n=2)
        Zb, _ = reduce_stat(Xk[common], n=2)
        U, V = CCA(n_components=2).fit(Za, Zb).transform(Za, Zb)
        a1.append(abs(corr(U[:, 0], V[:, 0])))
        a2.append(abs(corr(U[:, 1], V[:, 1])))

    fig, ax = plt.subplots(figsize=ONE_COL)
    x = np.arange(len(STATS2))
    ax.bar(x - 0.2, a1, 0.4, label="latent 1", color=COLORS["bind"])
    ax.bar(x + 0.2, a2, 0.4, label="latent 2", color=COLORS["highlight"])
    ax.axhline(1.0, color="k", ls=":", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in STATS2], rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel(r"alignment to $\kappa$-$C_\ell$ 2-D latent")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.1)

    save(fig, "figs/fig04_multistat_align")
    print("alignment (latent1, latent2):")
    for nm, r1, r2 in zip(STATS2, a1, a2):
        print(f"    {nm:16s} {r1:.2f}  {r2:.2f}")


if __name__ == "__main__":
    main()
