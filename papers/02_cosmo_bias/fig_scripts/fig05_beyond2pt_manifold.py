#!/usr/bin/env python
"""Fig 5 -- dimensionality & complementarity of the baryon-feedback manifold
across 10 weak-lensing/tSZ summary statistics (the "beyond-2pt" extension of
the power-spectrum-only capstone result).

Left: cumulative feedback-manifold variance vs number of baryon-template
nuisance modes, one curve per statistic (linearised active-subspace method,
Constantine 2015: regress standardised observable on the 30 astro params,
SVD the resulting Jacobian). Statistics whose per-realization feedback
signal is below the per-realization noise floor (kappa peaks, minima) are
drawn dashed.

Right: 10x10 principal-angle subspace-alignment matrix between each
statistic's 2-mode active feedback directions and those of the kappa power
spectrum (1 = identical directions, <1 = complementary).

Data (already-cached; no re-derivation, no simulation, no cosmology library):
  - /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz (237/253 valid runs
    after finite+valid filtering) -- X_unit, t__{stat}__value/valid for the
    10 statistics, a__cl_kappa__ell / a__cl_kappa_y__ell for the ell cap.

Source: examples/lightcone_beyond2pt.py (wl-cosmo-bias worktree),
extract()/manifold()/subspace_overlap() reproduced verbatim (pure numpy
linalg.lstsq + linalg.svd), plotting block in main() lines ~163-211. The
script's one pyccl call, kappa_cost_anchor(), only prints a console
cross-check and is not part of the figure -- omitted here.
Placeholder this replaces: figs/fig05_beyond2pt_manifold.png (md5-identical
to examples/figures_lightcone/beyond2pt_manifold.png).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

DATASET = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")
ELL_TRUST = 1.5e4  # above this CIC aliasing dominates run-to-run scatter

STATS = [
    ("cl_kappa", r"$\kappa\,C_\ell$ (2pt)"),
    ("peak_counts", r"$\kappa$ peaks"),
    ("minima_counts", r"$\kappa$ minima"),
    ("pdf", r"$\kappa$ PDF"),
    ("mf_v0", "Minkowski $V_0$"),
    ("mf_v1", "Minkowski $V_1$"),
    ("mf_v2", "Minkowski $V_2$"),
    ("moments", r"$\kappa$ moments"),
    ("peak_y", "tSZ $y$ peaks"),
    ("cl_kappa_y", r"$\kappa\times y$"),
]


def extract(d, stat: str):
    """(V, X): feedback-response matrix (n_valid, D) and unit params (n_valid, 30)."""
    val = d[f"t__{stat}__value"]
    ok = d[f"t__{stat}__valid"].copy()
    X = d["X_unit"]

    if stat == "cl_kappa":  # (n,5,5,nell) -> auto spectra, ell-capped
        ell = d["a__cl_kappa__ell"]
        m = ell <= ELL_TRUST
        V = np.stack([val[:, i, i, :][:, m] for i in range(5)], axis=1)
    elif stat == "cl_kappa_y":  # (n,5,nell) ell-capped
        ell = d["a__cl_kappa_y__ell"]
        m = ell <= ELL_TRUST
        V = val[:, :, m]
    else:
        V = val
    V = V.reshape(V.shape[0], -1)

    finite = np.isfinite(V).all(1)
    keep = ok & finite
    V, X = V[keep], X[keep]
    good = V.std(0) > 0
    return V[:, good], X


def manifold(V: np.ndarray, X: np.ndarray, m_sub: int = 2):
    """Linearised active-subspace (Constantine 2015): denoised feedback
    Jacobian Beta = d(standardised stat)/d(theta), its SVD gives the
    ordered active feedback directions and the manifold's singular
    spectrum (dimensionality)."""
    mu = V.mean(0)
    Vs = (V - mu) / V.std(0)
    Xc = X - X.mean(0)
    Beta, *_ = np.linalg.lstsq(Xc, Vs, rcond=None)
    Vhat = Xc @ Beta
    P, sig, _ = np.linalg.svd(Beta, full_matrices=False)
    evr = sig**2 / (sig**2).sum()
    cum = np.cumsum(evr)
    snr = np.linalg.norm(Vhat) / np.linalg.norm(Vs - Vhat)
    return dict(cum=cum, snr=snr, Q=P[:, :m_sub])


def subspace_overlap(Qa: np.ndarray, Qb: np.ndarray) -> tuple[float, float]:
    """Mean & min cos of principal angles between two param-space subspaces."""
    cos = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    cos = np.clip(cos, 0, 1)
    return float(cos.mean()), float(cos.min())


def main() -> None:
    setup()
    d = np.load(DATASET, allow_pickle=True)

    info = {}
    for key, label in STATS:
        V, X = extract(d, key)
        info[key] = manifold(V, X)
        info[key]["label"] = label

    n = len(STATS)
    O = np.ones((n, n))
    for i, (ki, _) in enumerate(STATS):
        for j, (kj, _) in enumerate(STATS):
            O[i, j], _ = subspace_overlap(info[ki]["Q"], info[kj]["Q"])

    # Stacked layout (not side-by-side): the 10x10 matrix needs close to the
    # full page width to keep its tick/cell text at the style's default size
    # (see FIGURE_STYLE.md rule 7 -- resize the figure rather than shrink
    # fonts). Panel (a) also benefits from the extra width for its 10-entry
    # legend.
    fig = plt.figure(figsize=(TWO_COL[0], 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.28)
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1])

    cols = plt.cm.turbo(np.linspace(0.05, 0.95, n))
    for c, (key, _) in zip(cols, STATS):
        I = info[key]
        x = np.arange(1, len(I["cum"]) + 1)
        nl = I["snr"] < 1.0  # feedback below per-realization noise
        a1.plot(x, I["cum"], "--" if nl else "-o", ms=2.5, color=c,
                alpha=0.5 if nl else 1.0, lw=1.1,
                label=I["label"] + (" (noise)" if nl else ""))
    a1.axhline(0.90, color="0.5", ls=":", lw=0.7)
    a1.axvline(2, color=COLORS["highlight"], lw=1.1)
    a1.set_xlim(0.6, 10)
    a1.set_ylim(0.4, 1.06)
    a1.set_xlabel("baryon-template nuisance modes $N$")
    a1.set_ylabel("cumulative feedback-manifold variance")
    a1.legend(ncol=2, loc="lower right", labelspacing=0.3, columnspacing=1.0)
    panel_label(a1, "(a)")

    im = a2.imshow(O, vmin=0, vmax=1, cmap="cividis")
    a2.set_xticks(range(n))
    a2.set_yticks(range(n))
    labs = [info[k]["label"] for k, _ in STATS]
    a2.set_xticklabels(labs, rotation=90)
    a2.set_yticklabels(labs)
    for i in range(n):
        for j in range(n):
            a2.text(j, i, f"{O[i, j]:.2f}", ha="center", va="center",
                    color="w" if O[i, j] < 0.6 else "k")
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.03)
    cb.set_label("subspace alignment")
    # Every corner cell of the matrix holds real data (diagonal = 1.00), so
    # the usual in-axes panel_label corner would sit on top of a value;
    # place the tag just above the axes instead.
    a2.text(0.0, 1.02, "(b)", transform=a2.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold")

    save(fig, "figs/fig05_beyond2pt_manifold")


if __name__ == "__main__":
    main()
