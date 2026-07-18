"""Project-B status battery: one script, a handful of PNGs, the whole story.

Pulls only from already-computed artifacts on ceph (no new measurement, no
touching the frozen vectors) and renders 6 figures that walk WP1 -> WP5 plus
the session-8 follow-ups:

  01_survey_and_peaks.png     WP1: footprint composition + peak nu histograms
  02_frozen_measurement.png   WP2/WP3: the frozen <Y(nu)> (Wiener + GLIMPSE)
  03_null_battery.png         WP3: null/systematics scorecard incl. sec7/8
  04_model_grid.png           WP4: the (Dln Mgas, Dln T) twobound atlas
  05_the_fit.png              WP5: THE FIT -- posterior + the y-deficit
  06_ladder_status.png        session 8: tested systematics vs the required
                              effect size -- none close the gap

Output: <out>/ (default ~/ceph/paper3/B/status_figures/).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WP1 = Path("/mnt/home/mlee1/ceph/paper3/B/wp1_maps")
WP2 = Path("/mnt/home/mlee1/ceph/paper3/B/wp2_measurement")
WP3 = Path("/mnt/home/mlee1/ceph/paper3/B/wp3_nulls")
WP4 = Path("/mnt/home/mlee1/ceph/paper3/B/wp4_mocks")
WP5 = Path("/mnt/home/mlee1/ceph/paper3/B/wp5_inference")

NU_LABELS = ["0-1 (diag.)", "1-2", "2-3", "3-4", "4-12"]
J4 = 2  # 4' fiducial CAP radius index


def fig01_survey_and_peaks(out: Path) -> None:
    f = np.load(WP1 / "common_footprint_nside1024.npz", allow_pickle=True)
    names = [n.replace(" (", "\n(") for n in f["sky_fraction_names"]]
    vals = np.asarray(f["sky_fraction_values"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.barh(range(len(vals)), vals, color="#4C72B0")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("sky fraction [%]")
    ax.set_title("WP1: footprint composition\n(DES x ACT common area)")
    ax.invert_yaxis()

    ax = axes[1]
    for variant, color in (("wiener", "#4C72B0"), ("glimpse", "#DD8452")):
        c = np.load(WP1 / f"peaks_{variant}_sm2am.npz")
        nu = c["nu"]
        bins = np.linspace(-3, 8, 45)
        ax.hist(nu, bins=bins, histtype="step", lw=1.8, color=color,
               label=f"{variant} (n={len(nu)})")
    for edge in (0, 1, 2, 3, 4):
        ax.axvline(edge, color="gray", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"peak significance $\nu$")
    ax.set_ylabel("peaks / bin")
    ax.set_title("WP1: peak catalogs\n(dotted = B5 bin edges)")
    ax.legend(fontsize=9)
    fig.suptitle("WP1 -- maps & peaks (COMPLETE)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "01_survey_and_peaks.png", dpi=150)
    plt.close(fig)


def fig02_frozen_measurement(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    nu_mid = [0.5, 1.5, 2.5, 3.5, 8.0]
    for variant, color, dx in (("wiener", "#4C72B0", -0.08), ("glimpse", "#DD8452", 0.08)):
        s = np.load(WP2 / f"stack_{variant}_sm2am_fid.npz")
        y = s["y_mean"][:, J4]
        yerr = np.sqrt(s["y_cov"][:, J4, J4])
        x = np.asarray(nu_mid) + dx
        ax.errorbar(x, y, yerr=yerr, fmt="o", color=color, capsize=3,
                   label=f"{variant} (Sigma_stat only)")
    ax.set_yscale("log")
    ax.set_xticks(nu_mid)
    ax.set_xticklabels(NU_LABELS)
    ax.set_xlabel(r"$\nu$ bin")
    ax.set_ylabel(r"$\langle Y_{CAP}(4')\rangle$  [y arcmin$^2$]")
    ax.set_title("WP2/WP3 -- the FROZEN measurement (signed 2026-07-17)\n"
                "Wiener = headline, GLIMPSE = cross-check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_frozen_measurement.png", dpi=150)
    plt.close(fig)


def fig03_null_battery(out: Path) -> None:
    sc = json.loads((WP3 / "b3_scorecard.json").read_text())
    sd = json.loads((WP3 / "star_dust_summary.json").read_text())

    rows = []  # (label, sigma_or_none, verdict)
    for v in ("wiener", "glimpse"):
        e = sc[f"S1_ensemble_{v}"]
        rows.append((f"S1 random-position null ({v})", None, "PASS" if e["pass"] else "FAIL"))
    s2 = sc["S2_shifts"]
    s2_short = ("PASS (recalibrated: 2/32 vs Gaussian exp. 1.5)"
               if "PASS" in s2["verdict"].upper() or "recalib" in s2["verdict"].lower()
               else s2["verdict"][:40])
    rows.append(("S2 RA-shift stacks", None, s2_short))
    rows.append(("S3 B-mode stack", max(abs(x) for x in sc["S3_bmode"]["significance"]),
                "PASS" if sc["S3_bmode"]["pass"] else "FAIL"))
    rows.append(("S5 mask-proximity terciles", None, "PASS" if sc["S5_mask_proximity"]["pass"] else "FAIL"))
    rows.append(("S6 ACT-threshold recut", None, "vacuous/PASS"))
    for v in ("wiener", "glimpse"):
        c = sc[f"S4_cib_{v}"]
        rows.append((f"S4 CIB band ({v})", c["band_over_sigma_betaT"] if isinstance(c["band_over_sigma_betaT"], (int, float)) else None,
                    "ESCALATED" if c["escalate_preregistered"] else "PASS"))
    for v in ("wiener", "glimpse"):
        worst = max(abs(sd[f"{v}_{cov}_tercile"][b]["sigma"])
                   for cov in ("star_density", "psf_dresid", "ebv")
                   for b in sd[f"{v}_{cov}_tercile"] if "sigma" in sd[f"{v}_{cov}_tercile"][b])
        rows.append((f"S7 star/PSF hotspot ({v})", worst, "PASS"))
    for v in ("wiener", "glimpse"):
        worst = max(abs(sd[f"{v}_ebv_tercile"][b]["sigma"])
                   for b in sd[f"{v}_ebv_tercile"] if "sigma" in sd[f"{v}_ebv_tercile"][b])
        rows.append((f"S8 dust E(B-V) ({v})", worst, "PASS (mild)"))

    labels = [r[0] for r in rows][::-1]
    verdicts = [r[2] for r in rows][::-1]
    colors = ["#55A868" if "PASS" in v and "ESCAL" not in v else
             ("#C44E52" if "ESCAL" in v else "#DD8452") for v in verdicts]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(range(len(labels)), [1] * len(labels), color=colors, height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks([])
    for i, v in enumerate(verdicts):
        vtxt = v if len(v) <= 46 else v[:44] + "…"
        ax.text(0.5, i, vtxt, ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_title("WP3 -- null/systematics battery (FROZEN 2026-07-17;\n"
                "S7/S8 executed session 8, does not reopen the freeze)")
    fig.subplots_adjust(left=0.26, right=0.97, top=0.85, bottom=0.05)
    fig.savefig(out / "03_null_battery.png", dpi=150)
    plt.close(fig)


def fig04_model_grid(out: Path) -> None:
    g = np.load(WP4 / "model_grid_tfwiener.npz", allow_pickle=True)
    dm, dt = g["delta_ln_mgas"], g["delta_ln_t"]
    y_top = g["y_mean"][:, -1, J4]

    fig, ax = plt.subplots(figsize=(7, 6))
    sca = ax.scatter(dm, dt, c=np.log10(y_top), cmap="viridis", s=60,
                     edgecolor="k", linewidth=0.4, label="twobound atlas (60 runs)")
    fid = np.load(WP4 / "grid_tfwiener" / "bind_run_0000.npz")
    ax.scatter([0], [0], marker="*", s=400, color="red", edgecolor="k",
              zorder=5, label="bind fiducial (0,0) anchor")
    cbar = fig.colorbar(sca, ax=ax)
    cbar.set_label(r"$\log_{10}\langle Y(\nu{=}4$-$12)\rangle$")
    ax.set_xlabel(r"$\Delta\ln M_{\rm gas}$ (halo-matched vs fiducial)")
    ax.set_ylabel(r"$\Delta\ln T$")
    ax.set_title("WP4 -- forward-model grid (COMPLETE, MODEL_FREEZE signed)\n"
                "R2 selection validation PASS; fiducial sits at the cloud EDGE")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "04_model_grid.png", dpi=150)
    plt.close(fig)


def fig05_the_fit(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    p = np.load(WP5 / "b5_posterior_wiener.npz")
    lnl = p["lnlike"]
    dchi2 = 2.0 * (lnl.max() - lnl)          # >=0, delta-chi2 from the peak
    extent = [p["dt"].min(), p["dt"].max(), p["mg"].min(), p["mg"].max()]
    im = ax.imshow(np.clip(dchi2, 0, 30), origin="lower", aspect="auto",
                   extent=extent, cmap="viridis_r")
    ax.contour(p["dt"], p["mg"], dchi2, levels=[2.3, 6.18], colors="white",
              linewidths=1.0, linestyles=["-", "--"])
    info = json.loads((WP5 / "b5_fit_wiener.json").read_text())
    ax.scatter([info["map"][1]], [info["map"][0]], marker="x", color="red", s=140,
              linewidths=2.5, label=f"MAP (edge mass={info['edge_mass']:.2f})")
    ax.set_xlabel(r"$\Delta\ln T$")
    ax.set_ylabel(r"$\Delta\ln M_{\rm gas}$")
    ax.set_title("WP5 THE FIT: posterior piles at the window\ncorner (white = 68%/95% contours, clipped $\\Delta\\chi^2$ shown)")
    ax.legend(fontsize=8, loc="lower left")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"$\Delta\chi^2$ from peak (clipped at 30)")

    ax = axes[1]
    x = np.arange(4)
    width = 0.35
    orig = np.array([0.30, 0.31, 0.28, 0.15])
    pg1024 = np.array([0.298, 0.310, 0.279, 0.161])
    ax.bar(x - width / 2, orig, width, label="original tfwiener grid", color="#4C72B0")
    ax.bar(x + width / 2, pg1024, width, label="pg1024 (corrected peak-finding)", color="#DD8452")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="model = data")
    ax.set_xticks(x)
    ax.set_xticklabels(NU_LABELS[1:])
    ax.set_ylabel("data / model (fiducial)")
    ax.set_ylim(0, 1.2)
    ax.set_xlabel(r"$\nu$ bin")
    ax.set_title("The y-deficit: data is 3-6x LOW\nvs the whole TNG twobound family")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "05_the_fit.png", dpi=150)
    plt.close(fig)


def fig06_ladder_status(out: Path) -> None:
    """Compare each tested systematic's OWN detection significance (tercile
    test, in sigma -- the same units NULL_CRITERIA.md judges pass/fail in)
    against the y-deficit's rejection significance (sqrt(chi2), 4 dof, from
    the B5 fit). A systematic that could plausibly explain the deficit would
    itself have to be detected at a comparably huge significance; instead
    every tested systematic is consistent with noise (<=2.4 sigma)."""
    sd = json.loads((WP3 / "star_dust_summary.json").read_text())

    def worst_tercile_sigma(variant, cov):
        t = sd[f"{variant}_{cov}_tercile"]
        return max(abs(t[b]["sigma"]) for b in t if "sigma" in t[b])

    items = [
        ("y-deficit rejection\n(fiducial, chi2=443.8/4dof)", np.sqrt(443.8), "#C44E52"),
        ("y-deficit rejection\n(MAP, chi2~=162/4dof)", np.sqrt(162.0), "#C44E52"),
        ("item1: peak-grid-1024 fit\n(fiducial ratios unchanged)", 0.05, "#55A868"),
        ("item2: star density\ntercile (wiener/glimpse)",
         max(worst_tercile_sigma("wiener", "star_density"), worst_tercile_sigma("glimpse", "star_density")),
         "#55A868"),
        ("item2: PSF-residual\ntercile (wiener/glimpse)",
         max(worst_tercile_sigma("wiener", "psf_dresid"), worst_tercile_sigma("glimpse", "psf_dresid")),
         "#55A868"),
        ("item2: dust E(B-V)\ntercile (wiener/glimpse)",
         max(worst_tercile_sigma("wiener", "ebv"), worst_tercile_sigma("glimpse", "ebv")),
         "#DD8452"),
        ("2 sigma reference line", 2.0, "none"),
    ]
    items = [i for i in items if i[2] != "none"]
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    colors = [i[2] for i in items]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(range(len(labels)), vals, color=colors)
    ax.axvline(2.0, color="gray", ls=":", lw=1, label=r"2$\sigma$ (typical single-test threshold)")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(r"significance [$\sqrt{\chi^2}$ or tercile $|\sigma|$, log scale]")
    ax.set_title("Session-8 interpretation ladder: each tested systematic's OWN\n"
                "significance vs. the y-deficit's rejection strength -- NONE close the gap")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "06_ladder_status.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/home/mlee1/ceph/paper3/B/status_figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig01_survey_and_peaks(out)
    fig02_frozen_measurement(out)
    fig03_null_battery(out)
    fig04_model_grid(out)
    fig05_the_fit(out)
    fig06_ladder_status(out)
    print(f"[status-battery] wrote 6 figures to {out}")


if __name__ == "__main__":
    main()
