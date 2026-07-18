"""WP-A5 task 5 deliverable: the probe-subset TENSION figure + consistency
quantification.

Overlays the three posteriors — f_gas-only (FINAL chains), kSZ-only, and
the f_gas+kSZ joint (subset chains, 31-dim) — in the reduced coordinates
the plan names:

- f_group: the model-frame group-bin spherical f_gas prediction
  `FgasBlock.predict[:, 0]` (comparable across subsets; the eRASS1 data
  value is the reference line);
- T_kSZ(1'): `KszBlock.predict[:, 0]` vs the Qu m3 value (for the
  30-column f_gas-only chain the f_sat nuisance sits at its prior
  midpoint — documented);
- QuasarThresholdPower (a pre-registered tripwire dimension): does the
  kSZ pull share the f_gas fit's edge direction?
- f_sat: the satellite-fraction nuisance posterior vs its uniform prior
  (prior-dominated or data-pulled?).

Subset consistency (a5_tension.json): pairwise Gaussian z-scores
|mu1 - mu2| / sqrt(s1^2 + s2^2) and histogram-overlap coefficients on
f_group and T(1'), plus each chain's 30-dim edge-mass tripwire table.

Run (after the subsets sbatch lands):
    python analysis/paper3a/scripts/fig_a5_tension.py
Out: wp5_chains/figures/a5_tension_pub.{pdf,png}, wp5_chains/a5_tension.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.ksz import KszBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402
from analysis.paper3a.style import COL, apply, refline  # noqa: E402

N_SUB = 20_000
SEED = 5

CHAIN_SETS = {
    "fgas": ("a5_final_seed{k}.npz", COL["model"], "f_gas only (FINAL)"),
    "kszonly": ("a5_subset_kszonly_seed{k}.npz", COL["aux"], "kSZ only (m3)"),
    "joint": ("a5_subset_joint_seed{k}.npz", COL["variant"], "f_gas + kSZ"),
}


def load_flat(pattern: str) -> np.ndarray | None:
    chains = []
    for k in range(4):
        p = CHAINS / pattern.format(k=k)
        if not p.exists():
            return None
        f = np.load(p, allow_pickle=False)
        c = f["chain"].astype(np.float64)
        chains.append(c.reshape(-1, c.shape[-1]))
    return np.concatenate(chains)


def overlap_coefficient(a: np.ndarray, b: np.ndarray, n_bins: int = 60) -> float:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    ha, _ = np.histogram(a, bins=n_bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=n_bins, range=(lo, hi), density=True)
    w = (hi - lo) / n_bins
    return float(np.minimum(ha, hb).sum() * w)


def gauss_z(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(a.mean() - b.mean())
                 / np.sqrt(a.var(ddof=1) + b.var(ddof=1)))


def main() -> None:
    rng = np.random.default_rng(SEED)
    flats = {}
    for name, (pattern, _, _) in CHAIN_SETS.items():
        flat = load_flat(pattern)
        if flat is None:
            if name == "fgas":
                raise SystemExit("FINAL f_gas chains missing — nothing to compare")
            raise SystemExit(f"{name} subset chains not on disk yet — run after "
                             "the a5_subsets sbatch lands")
        flats[name] = flat[rng.choice(len(flat), size=min(N_SUB, len(flat)),
                                      replace=False)]

    ksz = KszBlock()
    fgas = FgasBlock(emu=ksz.emu)

    summ = {}
    for name, flat in flats.items():
        summ[name] = {
            "f_group": fgas.predict(flat)[:, 0],
            "t1": ksz.predict(flat)[:, 0],
        }

    iq = pm.ASTRO_NAMES.index("QuasarThresholdPower")
    lo_f, hi_f = ksz.F_SAT_RANGE

    apply()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))

    ax = axes[0, 0]
    for name, (_, col, label) in CHAIN_SETS.items():
        ax.hist(summ[name]["f_group"], bins=50, density=True, histtype="step",
                lw=1.8, color=col, label=label)
    refline(ax, x=float(fgas.data.values[0]))
    ax.annotate("eRASS1", (float(fgas.data.values[0]), 0.97),
                xycoords=("data", "axes fraction"), fontsize=7,
                ha="left", va="top", color=COL["ref"])
    ax.set_xlabel(r"$f_{\rm gas}^{\rm group}(\theta)$ (model frame)")
    ax.set_yticks([])
    ax.legend(fontsize=6.5, loc="upper right")

    ax = axes[0, 1]
    for name, (_, col, label) in CHAIN_SETS.items():
        ax.hist(summ[name]["t1"], bins=50, density=True, histtype="step",
                lw=1.8, color=col, label=label)
    t1_data = float(ksz.values[0])
    t1_sig = float(np.sqrt(ksz.cov_data[0, 0]))
    ax.axvspan(t1_data - t1_sig, t1_data + t1_sig, color=COL["shade"],
               alpha=0.35, lw=0)
    refline(ax, x=t1_data)
    ax.annotate("Qu m3", (t1_data, 0.97), xycoords=("data", "axes fraction"),
                fontsize=7, ha="left", va="top", color=COL["ref"])
    ax.set_xlabel(r"$T_{\rm kSZ}(1')$ [$\mu$K arcmin$^2$]")
    ax.set_yticks([])

    ax = axes[1, 0]
    for name, (_, col, label) in CHAIN_SETS.items():
        ax.hist(flats[name][:, iq], bins=40, range=(0, 1), density=True,
                histtype="step", lw=1.8, color=col, label=label)
    ax.set_xlabel("QuasarThresholdPower (unit cube)")
    ax.set_yticks([])

    ax = axes[1, 1]
    for name in ("kszonly", "joint"):
        _, col, label = CHAIN_SETS[name]
        f_sat = lo_f + (hi_f - lo_f) * flats[name][:, 30]
        ax.hist(f_sat, bins=40, range=(lo_f, hi_f), density=True,
                histtype="step", lw=1.8, color=col, label=label)
    ax.axhline(1.0 / (hi_f - lo_f), color=COL["ref"], lw=0.9, ls=":")
    ax.annotate("uniform prior", (0.02, 0.95), xycoords="axes fraction",
                fontsize=7, va="top", color=COL["ref"])
    ax.set_xlabel(r"$f_{\rm sat}$ (Bigwood prior range)")
    ax.set_yticks([])
    ax.legend(fontsize=6.5, loc="lower right")

    fig.suptitle("WP-A5 probe-subset tension: reduced-coordinate posteriors",
                 fontsize=9)
    fig.tight_layout()
    figdir = CHAINS / "figures"
    figdir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"a5_tension_pub.{ext}", dpi=200)
    print(f"wrote {figdir}/a5_tension_pub.pdf/.png")

    pairs = [("fgas", "kszonly"), ("fgas", "joint"), ("kszonly", "joint")]
    out = {"n_sub_per_chain": N_SUB, "pairs": {}}
    for a, b in pairs:
        out["pairs"][f"{a}_vs_{b}"] = {
            coord: {"gauss_z": gauss_z(summ[a][coord], summ[b][coord]),
                    "overlap_coeff": overlap_coefficient(summ[a][coord],
                                                         summ[b][coord])}
            for coord in ("f_group", "t1")
        }
    out["edge_mass_gt15pct"] = {}
    for name, flat in flats.items():
        edge = {}
        for i in range(30):
            e = {"low": float(np.mean(flat[:, i] < 0.05)),
                 "high": float(np.mean(flat[:, i] > 0.95))}
            if max(e.values()) > 0.15:
                edge[pm.ASTRO_NAMES[i]] = e
        out["edge_mass_gt15pct"][name] = edge
    (CHAINS / "a5_tension.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["pairs"], indent=2))
    print(f"wrote {CHAINS}/a5_tension.json")


if __name__ == "__main__":
    main()
