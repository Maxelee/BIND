"""N=1000 campaign vs the N=50 suite: precision gain and consistency.

Consumes the per-realization products written by ``n1000_stats_stream.py`` and
compares, for the same seed-paired realizations:

  * N = 50   — the first 50 realizations (identical seeds to the old suite)
  * N = 1000 — the full campaign run

Two questions per statistic: does the MEAN move (it should not — same maps), and
how much does the ERROR shrink (it should be sqrt(20) = 4.47x).  The stored
50-real products are overlaid as an independent cross-check of the streaming
reduction against the production pipeline.

    python examples/n1000_vs_n50_figure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANA = Path("/mnt/home/mlee1/ceph/bind_n1000/analysis")
SCI = Path("/mnt/home/mlee1/ceph/bind_science/runs")

# validated categorical palette (dataviz reference instance)
C_1000, C_50, C_ALT, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b66"
GRID = dict(color="#d9d8d2", lw=0.6, alpha=0.8)
ZI = 1          # z_s = 1.0
N50 = 50


def se(a, n):
    return a[:n].std(0) / np.sqrt(n)


def gain(a, mask=None):
    """Median per-bin SE reduction, N=50 -> N=all."""
    n = a.shape[0]
    s50, sN = se(a, N50), se(a, n)
    m = np.ones(a.shape[-1], bool) if mask is None else mask
    m = m & (sN > 0)
    return float(np.median(s50[m] / sN[m]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ana", type=Path, default=ANA)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or (args.ana / "n1000_vs_n50.png")

    S = {k: np.load(args.ana / f"stream_stats_{k}_run_0000.npz")
         for k in ("truth", "dmo", "bind", "twobound")}
    ell, nu = S["truth"]["ell"], S["truth"]["nu"]
    NT = S["truth"]["peaks"].shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------- A: peak counts, mean +- SE ----------
    pk = S["truth"]["peaks"][:, ZI]
    ax = axes[0, 0]
    m50, mN = pk[:N50].mean(0), pk.mean(0)
    s50, sN = se(pk, N50), se(pk, NT)
    k = mN > 0.05
    ax.fill_between(nu[k], m50[k] - s50[k], m50[k] + s50[k], color=C_50, alpha=0.45, lw=0)
    ax.fill_between(nu[k], mN[k] - sN[k], mN[k] + sN[k], color=C_1000, alpha=0.6, lw=0)
    ax.plot(nu[k], mN[k], color=C_1000, lw=1.5)
    old = np.load(SCI / "truth/run_0000/peak_counts.npz")["peak_counts"][ZI]
    ax.plot(nu[k], old[k], color="k", lw=0, marker="o", ms=2.5, alpha=0.55)
    ax.set_yscale("log")
    ax.set_xlim(-2, 10)
    ax.set_xlabel(r"peak S/N  $\nu$")
    ax.set_ylabel("peaks per map per bin")
    ax.set_title(r"truth $\kappa$ peaks, $z_s\!=\!1$", fontsize=11)
    ax.text(5.4, 12, "N = 50", color=C_50, fontsize=10)
    ax.text(5.4, 4.5, f"N = {NT}", color=C_1000, fontsize=10)
    ax.text(5.4, 1.7, "stored product", color="k", fontsize=9, alpha=0.7)

    # ---------- B: relative error, peaks & minima ----------
    ax = axes[0, 1]
    for arr, lab, c in ((S["truth"]["peaks"][:, ZI], "peaks", C_1000),
                        (S["truth"]["minima"][:, ZI], "minima", C_ALT)):
        mN2 = arr.mean(0)
        k2 = mN2 > 0.5
        ax.plot(nu[k2], 100 * se(arr, N50)[k2] / arr[:N50].mean(0)[k2],
                color=c, lw=1.6, ls="--")
        ax.plot(nu[k2], 100 * se(arr, NT)[k2] / mN2[k2], color=c, lw=1.9, label=lab)
    ax.set_yscale("log")
    ax.set_xlim(-3.5, 8)
    ax.set_xlabel(r"S/N  $\nu$")
    ax.set_ylabel("SE / mean  [%]")
    ax.set_title("relative error: dashed N=50, solid N=1000", fontsize=11)
    ax.legend(frameon=False, fontsize=10)

    # ---------- C: paired S(l) suppression ----------
    ax = axes[0, 2]
    s_re = S["bind"]["cl"][:, ZI] / S["dmo"]["cl"][:, ZI]     # per-realization ratio
    m50, mN = s_re[:N50].mean(0), s_re.mean(0)
    s50, sN = se(s_re, N50), se(s_re, NT)
    ax.fill_between(ell, m50 - s50, m50 + s50, color=C_50, alpha=0.45, lw=0, label="N = 50")
    ax.fill_between(ell, mN - sN, mN + sN, color=C_1000, alpha=0.65, lw=0, label=f"N = {NT}")
    ax.plot(ell, mN, color=C_1000, lw=1.4)
    ax.axhline(1.0, color=C_REF, lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(200, 2e4)
    ax.set_ylim(0.9, 1.05)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$S(\ell)=C_\ell^{\rm BIND}/C_\ell^{\rm DMO}$")
    ax.set_title(r"baryonic suppression, seed-paired ($z_s\!=\!1$)", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="lower left")

    # ---------- D: consistency of the means ----------
    ax = axes[1, 0]
    pulls = []
    for lab, arr in (("peaks", S["truth"]["peaks"][:, ZI]),
                     ("minima", S["truth"]["minima"][:, ZI]),
                     (r"$C_\ell$", S["truth"]["cl"][:, ZI]),
                     ("V0", S["truth"]["V0"][:, ZI]),
                     ("PDF", S["truth"]["pdf"][:, ZI])):
        d = arr.mean(0) - arr[:N50].mean(0)
        s = se(arr, N50)
        k3 = s > 0
        pulls.append((lab, (d[k3] / s[k3])))
    ax.boxplot([p[1] for p in pulls], tick_labels=[p[0] for p in pulls], showfliers=False,
               medianprops=dict(color=C_1000, lw=2),
               boxprops=dict(color=C_REF), whiskerprops=dict(color=C_REF),
               capprops=dict(color=C_REF))
    ax.axhline(0, color=C_REF, lw=0.9, ls=":")
    ax.set_ylabel(r"(mean$_{1000}$ - mean$_{50}$) / SE$_{50}$")
    ax.set_title("means agree: N=50 was unbiased, just noisy", fontsize=11)

    # ---------- E: error-reduction factor per statistic ----------
    ax = axes[1, 1]
    stats = [("peaks", S["truth"]["peaks"][:, ZI], S["truth"]["peaks"][:, ZI].mean(0) > 0.5),
             ("minima", S["truth"]["minima"][:, ZI], S["truth"]["minima"][:, ZI].mean(0) > 0.5),
             (r"$C_\ell$", S["truth"]["cl"][:, ZI], None),
             ("V0", S["truth"]["V0"][:, ZI], None),
             ("V1", S["truth"]["V1"][:, ZI], None),
             ("V2", S["truth"]["V2"][:, ZI], None),
             ("PDF", S["truth"]["pdf"][:, ZI], S["truth"]["pdf"][:, ZI].mean(0) > 1e-4),
             (r"$S(\ell)$", s_re, None)]
    g = [gain(a, m) for _, a, m in stats]
    ax.bar(range(len(g)), g, color=C_1000, width=0.62)
    ax.axhline(np.sqrt(NT / N50), color=C_50, lw=1.8, ls="--")
    ax.text(len(g) - 0.4, np.sqrt(NT / N50) + 0.12, r"$\sqrt{N/50}=4.47$",
            color=C_50, fontsize=10, ha="right")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([s[0] for s in stats], fontsize=9)
    ax.set_ylabel(r"error reduction  SE$_{50}$ / SE$_{1000}$")
    ax.set_ylim(0, 5.6)
    ax.set_title("measured precision gain, every statistic", fontsize=11)

    # ---------- F: tail-bin convergence vs N ----------
    ax = axes[1, 2]
    Ns = np.unique(np.geomspace(10, NT, 45).astype(int))

    def rare_bin(arr, target=1.0, from_high=True):
        """Index of the rarest populated bin in the tail (~`target` counts/map).

        Picked by count, never by a hardcoded nu: the minima tail dies out well
        before the peak tail (minima counts are identically 0 by nu ~ -2.9), so
        a fixed nu would silently select an empty bin.
        """
        m = arr.mean(0)
        ok = np.flatnonzero(m >= target)
        return ok[-1] if from_high else ok[0]

    pk_a, mn_a = S["truth"]["peaks"][:, ZI], S["truth"]["minima"][:, ZI]
    i_pk, i_mn = rare_bin(pk_a), rare_bin(mn_a, from_high=False)
    for arr, idx, lab, c, ls in (
            (pk_a, i_pk, rf"peaks $\nu\!\approx\!{nu[i_pk]:.1f}$", C_1000, "-"),
            (mn_a, i_mn, rf"minima $\nu\!\approx\!{nu[i_mn]:.1f}$", C_ALT, (0, (5, 3)))):
        ax.plot(Ns, [100 * se(arr, n)[idx] / max(arr[:n].mean(0)[idx], 1e-9) for n in Ns],
                color=c, lw=1.9, ls=ls, label=lab)
    ref = 100 * se(pk_a, N50)[i_pk] / pk_a[:N50].mean(0)[i_pk]
    ax.plot(Ns, ref * np.sqrt(N50 / Ns), color=C_REF, lw=1.5, ls=":", label=r"$\propto 1/\sqrt{N}$")
    ax.axvline(N50, color=C_50, lw=1.2)
    ax.text(53, 24, "old suite", color=C_50, fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of realizations  N")
    ax.set_ylabel("SE / mean  [%]")
    ax.set_title("rare-count bins converge as $1/\\sqrt{N}$", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    fig.suptitle("BIND N1000 campaign vs the 50-realization suite — same seeds, "
                 f"{NT} realizations (truth / bind / dmo / twobound complete)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=150, facecolor="white")
    print("saved", out)

    summary = {lab: round(gn, 3) for (lab, _, _), gn in zip(stats, g)}
    summary["expected_sqrt_N_over_50"] = round(float(np.sqrt(NT / N50)), 3)
    summary["max_|pull|_of_means"] = {p[0]: round(float(np.abs(p[1]).max()), 2) for p in pulls}
    (args.ana / "n1000_vs_n50_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
