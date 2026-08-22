"""Diffuse-gas validation: pasted-truth vs +diffuse-approx vs full-hydro traces.

Compares three seed-paired ray-traced variants of the SAME lightcone:
  truth      — pasted hydro halo patches on DMO background (existing production
               trace; gas/y/tau are zero outside paste apertures, ~80-85% of sky)
  diffuse    — truth composites + f_b * DMO * (1-alpha) gas at T=1e4 K outside
               the paste regions (run_tng_validation_trace.sh task 1)
  hydro_full — full-box hydro projection, gas everywhere (task 0)

All traces share base seed 1992, so realization r has identical plane geometry
across variants: every comparison below is PAIRED per realization (cosmic
variance cancels; error bars are the per-real scatter of the ratio itself).

Memory-conscious: fields are processed one at a time (~3.5 GB peak), so this
runs inside a ~10 GB session.

Outputs (under --out, default <val_root>/analysis):
  tng_full_validation.png          6-panel figure
  tng_full_validation_summary.md   the decision numbers

  python examples/tng_full_validation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bind.inference.stats import power_spectrum

VAL_ROOT = Path("/mnt/home/mlee1/ceph/tng_full_validation")
TRUTH_RUN = Path("/mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000")
ZS_LABELS = ("0.5", "1.0", "1.5", "2.0", "2.44")
VARIANTS = ("truth", "diffuse", "hydro_full")


def load_maps(run_dir: Path, field: str, n_real: int) -> np.ndarray:
    """(n_real, 5, N, N) float32 — the paired first n_real realizations.

    Streams only the needed prefix out of the npz zip member instead of
    ``np.load(...)[key][:n]``, which decompresses the WHOLE array first (the
    550-real truth cubes are ~11.5 GB and OOM a ~10 GB session).
    """
    import zipfile

    import numpy.lib.format as fmt

    with zipfile.ZipFile(run_dir / f"{field}_maps.npz") as z, z.open(f"{field}.npy") as f:
        version = fmt.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = fmt.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = fmt.read_array_header_2_0(f)
        if fortran:
            raise ValueError("unexpected Fortran-order npy")
        if shape[0] < n_real:
            raise ValueError(f"{run_dir}/{field}: only {shape[0]} reals (< {n_real})")
        per = int(np.prod(shape[1:])) * dtype.itemsize
        buf = f.read(n_real * per)
    return np.frombuffer(buf, dtype=dtype).reshape((n_real, *shape[1:]))


def cl_per_real(maps: np.ndarray, fov_deg: float = 5.0):
    """Per-realization auto-Cl for one z_s: (n_real, N, N) -> (ell, (n_real, n_ell))."""
    cls, ell = [], None
    for r in range(maps.shape[0]):
        ell, cl = power_spectrum(maps[r], fov_deg=fov_deg)
        cls.append(cl)
    return ell, np.asarray(cls)


def paired_ratio(num: np.ndarray, den: np.ndarray):
    """Mean and SE of the per-real ratio (paired seeds -> per-real division is fair)."""
    r = num / np.where(den > 0, den, np.nan)
    return np.nanmean(r, axis=0), np.nanstd(r, axis=0) / np.sqrt(r.shape[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val_root", type=Path, default=VAL_ROOT)
    p.add_argument("--truth_run", type=Path, default=TRUTH_RUN)
    p.add_argument("--n_real", type=int, default=50)
    p.add_argument("--zs_idx", type=int, default=1, help="z_s index for Cl panels (1 = z_s=1.0)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or (args.val_root / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    zi = args.zs_idx

    runs = {
        "truth": args.truth_run,
        "diffuse": args.val_root / "runs/diffuse/run_0000",
        "hydro_full": args.val_root / "runs/hydro_full/run_0000",
    }

    means: dict[str, dict[str, list[float]]] = {f: {} for f in ("tau", "y", "kappa")}
    cl: dict[str, dict[str, np.ndarray]] = {f: {} for f in ("tau", "y", "kappa")}
    pdf: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ell = None
    kappa_diff = kappa_scale = None

    # ---- one field at a time to bound memory ----
    for f in ("tau", "y", "kappa"):
        maps = {}
        for n, rd in runs.items():
            maps[n] = load_maps(rd, f, args.n_real)
            print(f"[load] {n}/{f} {maps[n].shape}")
        for n in VARIANTS:
            means[f][n] = [float(maps[n][:, z].mean()) for z in range(5)]
            e, c = cl_per_real(maps[n][:, zi])
            ell, cl[f][n] = e, c
        if f == "tau":
            for n in VARIANTS:
                t = maps[n][:, zi].ravel()
                hist, edges = np.histogram(np.log10(np.clip(t, 1e-12, None)), bins=80,
                                           range=(-8, -2), density=True)
                pdf[n] = (0.5 * (edges[1:] + edges[:-1]), hist)
        if f == "kappa":
            kappa_diff = float(np.abs(maps["diffuse"] - maps["truth"]).max())
            kappa_scale = float(np.abs(maps["truth"]).max())
        del maps   # free ~3 GB before the next field

    # ---- summary markdown ----
    lines = ["# TNG full-hydro / diffuse-gas validation", "",
             f"Paired first {args.n_real} realizations, base seed 1992.", "",
             "## Mean map values (all realizations, per z_s)", "",
             "| field | z_s | truth (pasted) | +diffuse | hydro_full | pasted/full | +diffuse/full |",
             "|---|---|---|---|---|---|---|"]
    for f in ("tau", "y"):
        for z in range(5):
            m = {n: means[f][n][z] for n in VARIANTS}
            lines.append(
                f"| {f} | {ZS_LABELS[z]} | {m['truth']:.4e} | {m['diffuse']:.4e} | "
                f"{m['hydro_full']:.4e} | {m['truth']/m['hydro_full']:.3f} | "
                f"{m['diffuse']/m['hydro_full']:.3f} |")
    lines += ["", f"## Paired Cl ratios vs hydro_full at z_s={ZS_LABELS[zi]} "
              "(band means over three ell ranges)", "",
              "| field | variant | ell<1000 | 1000-5000 | >5000 |", "|---|---|---|---|---|"]
    bands = [(ell < 1000), (ell >= 1000) & (ell < 5000), (ell >= 5000)]
    for f in ("tau", "y", "kappa"):
        for n in ("truth", "diffuse"):
            mean, _ = paired_ratio(cl[f][n], cl[f]["hydro_full"])
            vals = [np.nanmean(mean[b]) for b in bands]
            lines.append(f"| Cl_{f} | {n} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += ["", "## Sanity checks", "",
              f"- max |kappa_diffuse - kappa_truth| = {kappa_diff:.3e} "
              f"(max |kappa| = {kappa_scale:.3e}) — expect ~float32 rounding only.", ""]

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    colors = {"truth": "tab:red", "diffuse": "tab:orange", "hydro_full": "k"}
    for ax, f in zip(axes[0], ("tau", "y", "kappa")):
        for n in ("truth", "diffuse"):
            mean, se = paired_ratio(cl[f][n], cl[f]["hydro_full"])
            ax.plot(ell, mean, color=colors[n], label=f"{n} / hydro_full")
            ax.fill_between(ell, mean - se, mean + se, color=colors[n], alpha=0.3)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_title(rf"$C_\ell^{{{f}}}$ ratio, $z_s={ZS_LABELS[zi]}$")
        ax.legend(fontsize=8)
    for ax, f in zip(axes[1][:2], ("tau", "y")):
        for n in VARIANTS:
            ax.plot(range(5), means[f][n], "o-", color=colors[n], label=n)
        ax.set_xticks(range(5))
        ax.set_xticklabels(ZS_LABELS)
        ax.set_xlabel(r"$z_s$")
        ax.set_yscale("log")
        ax.set_title(rf"$\langle {f} \rangle$")
        ax.legend(fontsize=8)
    ax = axes[1][2]
    for n in VARIANTS:
        ax.plot(*pdf[n], color=colors[n], label=n)
    ax.set_xlabel(r"$\log_{10}\tau$")
    ax.set_title(rf"$\tau$ PDF, $z_s={ZS_LABELS[zi]}$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tng_full_validation.png", dpi=150)
    print(f"[fig] {out / 'tng_full_validation.png'}")

    lines += [
        "## Reading the result", "",
        "- `pasted/full` on mean tau quantifies the missing diffuse electron column",
        "  (the systematic under test); `+diffuse/full` shows how much the f_b",
        "  approximation recovers.",
        "- If `+diffuse/full` tau is ~1 at ell<1000 and the tau PDF floor matches,",
        "  the approximation is a good campaign candidate for tau.",
        "- y at T=1e4K adds ~nothing by construction; the y gap vs hydro_full",
        "  measures WHIM y that NEITHER construction captures.",
        "- kappa ratios isolate the mass-channel construction (pasted vs full",
        "  hydro); diffuse == truth by design.", "",
        "NOTE: y channels use the pipeline's PHYSICAL (proper-area) convention in",
        "all three variants (the truth-lightcone convention, truth_lightcone.py);",
        "stage-A composites were rescaled from legacy by 1/a^2 on 2026-08-11.",
    ]
    (out / "tng_full_validation_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[md] {out / 'tng_full_validation_summary.md'}")


if __name__ == "__main__":
    main()
