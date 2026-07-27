#!/usr/bin/env python
"""M5 (docs/ksz_lightcone_map_plan.md §3, phase M5): the map-level
recreation of the per-halo headline figure -- f~gas vs logM200 at two
redshifts (snap 85, BGS, z=0.18; snap 46, ELG, z=1.16) -- mirroring
``examples/_build_ksz_paper_nb.py`` SS6b ``f6b_lowmass`` (lines ~854-896),
but built from the ray-traced lightcone maps (``lightcone_cap_stack.py
--sample massbin85/massbin46``) instead of individually painted halo
patches.

**The observable**, per mass bin ``b`` and aperture ``theta_d``:

    f_gas(b, theta_d) = CAP_tau_gas(b, theta_d) / CAP_mat(b, theta_d)
    f~gas = f_gas / F_B,   F_B = Omega_b/Omega_m = 0.049/0.3089

CAP_tau_gas comes from the fiducial/per-node numerator shards
(``{sample}_tau_run{run}_beamnone_src4.npz``); CAP_mat comes from the
SINGLE shared ``capmat_massbin_lightcone.npz`` (``lightcone_capmat_merge.py
--massbin``) -- mass-bin selection is a pure FoF M200 cut, so (unlike
bgs110/bgs1125's per-node CAP_mat) it never depends on the Sobol node and
one fiducial massplane trace, stacked at the SAME per-bin positions used by
every node's numerator, is the correct denominator everywhere. Numerator
and denominator are therefore automatically stacked on IDENTICAL per-bin
halo samples -- see ``lightcone_cap_stack.py``/``lightcone_capmat_merge.py``
module docstrings for why this needs no per-node fix (contrast the BGS
M*-cut case, which does).

Two panels, exactly mirroring ``f6b_lowmass``'s layout/styling:
  (left)  BGS, z=0.18 (snap 85): f~gas at 1.0*theta200 (the r200-matched
          aperture, matching the per-halo reference and the DESIxACT BGS
          data's own theta(r200) interpolation) vs bin-center logM200.
  (right) ELG, z=1.16 (snap 46): f~gas at the FIXED 1.0' aperture (the
          ELG data's own innermost measured bin -- ELG hosts never reach
          their own r200 in the data) vs bin-center logM200.

Both panels: thin blue SB35 node lines (from whichever per-node shards
exist on disk -- gracefully degrades to fiducial-only before the 253-node
sweep lands), thick blue fiducial, black data stars (``m5_data_points.npz``),
dotted "cosmic" line at f~gas=1, grey "patch reuse" shading below
logM200=13 (BIND painted only M200>=1e13 as centrals), a dashed fiducial
in_patch-only variant inside that shaded region (reuse-regime honesty --
the ALL-halo curve there is diluted by un-painted background/2-halo pixels,
see module note below), and (right panel only) a green "ELG hosts" band
mirroring the original figure.

**Map-level vs per-halo, an expected divergence at low mass (read before
being alarmed by the numbers).** The >=1e13 bins close to the per-halo
reference to ~5-8% (the M5 sanity gate). The <1e13 bins do NOT -- even the
in_patch-only substack sits well below the per-halo reuse curve. This is a
genuine map-level finding, not a bug: the per-halo reuse pipeline reads the
CAP directly off the local 6.25 Mpc/h patch (no line-of-sight contamination
beyond the patch's own depth), while this pipeline's tau map is the FULL
line-of-sight integral to z_s=2.44 -- for small-aperture, low-mass halos
the correlated 2-halo/LOS signal swamps a proportionally weaker 1-halo
term, pulling the CAP-compensated f~gas down at all apertures (verified
monotonically RISING with theta/theta200 toward the eventual "cosmic"
crossing, the same qualitative shape as the high-mass bins -- see
docs/ksz_lightcone_map_plan.md M5 verdict for the numbers). This is exactly
why the plan calls for the shaded low-mass region and the in_patch-only
dashed variant instead of a clean, single trusted curve there.

Usage
-----
    python examples/lightcone_m5_fgas_mass.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CEPH = Path("/mnt/home/mlee1/ceph")
KS = CEPH / "bind_science/ksz_confront"
LIGHTCONE = KS / "lightcone"
CAT_DIR = LIGHTCONE / "catalogs"
SHARD_DIR = LIGHTCONE / "shards"
FIG_DIR = LIGHTCONE / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PATCH_REUSE_GREY = "0.93"
ELG_GREEN = "tab:green"
IN_PATCH_MAX_LOGM = 13.0

PANELS = [
    # sample,     label,             snap_key,   theta_col,       skip_edge_bins_ge13_for_dashed
    ("massbin85", r"$z{=}0.18$ · BGS", "snap085", "r200mult1.0"),
    ("massbin46", r"$z{=}1.16$ · ELG", "snap046", "fixed_arcmin"),
]


def _theta_col_index(theta_kind: np.ndarray, theta_value: np.ndarray, which: str) -> int:
    if which == "fixed_arcmin":
        idx = np.nonzero(theta_kind == "fixed_arcmin")[0]
        return int(idx[0])
    if which == "r200mult1.0":
        idx = np.nonzero(theta_kind == "r200mult")[0]
        return int(idx[np.argmin(np.abs(theta_value[idx] - 1.0))])
    raise ValueError(which)


def _node_shards(sample: str) -> list[Path]:
    """Per-node numerator shards already on disk (excludes the fiducial and
    massplane shards) -- gracefully returns [] before the 253-node sweep."""
    out = []
    for p in sorted(SHARD_DIR.glob(f"{sample}_tau_run*_beamnone_src4.npz")):
        if "runfid" in p.name:
            continue
        out.append(p)
    return out


def main():
    capmat = np.load(LIGHTCONE / "capmat_massbin_lightcone.npz", allow_pickle=True)
    F_B = float(capmat["F_B"])
    theta_kind = capmat["theta_kind"]
    theta_value = capmat["theta_value"]

    data = np.load(CAT_DIR / "m5_data_points.npz", allow_pickle=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    metrics = {"F_B": F_B}

    for ax, (sample, zlab, snap_key, theta_which) in zip(axes, PANELS):
        edges = capmat[f"{snap_key}_mass_bin_edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        n_bins = len(centers)
        theta_idx = _theta_col_index(theta_kind, theta_value, theta_which)

        mat_mean = capmat[f"{snap_key}_mean"][:, theta_idx].astype(np.float64)          # (n_bins,)
        mat_mean_ip = capmat[f"{snap_key}_mean_inpatch"][:, theta_idx].astype(np.float64)

        # ---------------- fiducial ----------------
        fid = np.load(SHARD_DIR / f"{sample}_tau_runfid_beamnone_src4.npz", allow_pickle=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            fid_fgas = fid["mean"][:, theta_idx] / mat_mean / F_B
            fid_fgas_ip = fid["mean_inpatch"][:, theta_idx] / mat_mean_ip / F_B

        # ---------------- SB35 nodes (whatever shards exist so far) --------
        node_paths = _node_shards(sample)
        n_plotted = 0
        node_rows = []   # (n_nodes_loaded, n_bins), NaN where a bin has no landing galaxies
        for p in node_paths:
            d = np.load(p, allow_pickle=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                row = d["mean"][:, theta_idx] / mat_mean / F_B
            m = np.isfinite(row)
            if m.sum() > 1:
                ax.plot(centers[m], row[m], "-", color="tab:blue", lw=0.4, alpha=0.13, zorder=1)
                n_plotted += 1
            node_rows.append(row)
        node_matrix = np.array(node_rows) if node_rows else np.full((0, n_bins), np.nan)
        n_expected = 253
        sb35_label = (f"SB35 nodes ({n_plotted}/{n_expected})" if n_plotted >= n_expected
                       else f"SB35 nodes ({n_plotted}/{n_expected} shards so far)")
        ax.plot([], [], "-", color="tab:blue", lw=1.0, alpha=0.5, label=sb35_label)

        # per-bin SB35 envelope (min/max across all loaded nodes) -- for the
        # verdict's "envelope range per bin" metric, not separately plotted
        # here (the thin blue lines above already show the full spread).
        with np.errstate(invalid="ignore"):
            envelope_min = np.nanmin(node_matrix, axis=0) if len(node_matrix) else np.full(n_bins, np.nan)
            envelope_max = np.nanmax(node_matrix, axis=0) if len(node_matrix) else np.full(n_bins, np.nan)

        # ---------------- shading + cosmic line ----------------
        ax.axvspan(edges[0], IN_PATCH_MAX_LOGM, color=PATCH_REUSE_GREY, zorder=0)
        ax.text(0.5 * (edges[0] + IN_PATCH_MAX_LOGM), 0.06, "patch reuse",
                 fontsize=6, ha="center", color="0.45")
        ax.axvline(IN_PATCH_MAX_LOGM, color="0.6", ls=":", lw=0.7)
        ax.axhline(1, color="k", ls=":", lw=0.8)

        # ---------------- fiducial curve (all halos, full range) ----------
        mf = np.isfinite(fid_fgas)
        ax.plot(centers[mf], fid_fgas[mf], "-o", color="tab:blue", lw=2.6, ms=4.5,
                 zorder=4, label="BIND fiducial (all)")

        # ---------------- in_patch-only dashed variant (sub-1e13 only) ----
        ip_mask = np.isfinite(fid_fgas_ip) & (edges[1:] <= IN_PATCH_MAX_LOGM)
        if ip_mask.any():
            ax.plot(centers[ip_mask], fid_fgas_ip[ip_mask], "--^", color="tab:blue",
                     lw=1.8, ms=4.0, alpha=0.85, zorder=5,
                     label="BIND fiducial (in_patch only)")

        # ---------------- data points ----------------
        if sample == "massbin85":
            lm, fd, fe = data["bgs_logM200"], data["bgs_fgas"], data["bgs_err"]
            lab = r"DESI$\times$ACT BGS ($r_{200}$, 5 $M_\star$ cuts)"
        else:
            lm, fd, fe = data["elg_logM200"], data["elg_fgas"], data["elg_err"]
            lab = r"DESI$\times$ACT ELG (fixed 1.0$'$, 3 cuts)"
        ax.errorbar(lm, fd, yerr=fe, fmt="*", color="k", ms=9, capsize=2, alpha=.85, zorder=6)
        ax.plot([], [], "*", color="k", ms=9, label=lab)

        ax.set_xlabel(r"$\log_{10} M_{200}\,[M_\odot/h]$")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, 1.18)
        ax.text(.5, .965, zlab, transform=ax.transAxes, fontsize=8.5, ha="center", va="top")
        ax.legend(loc="lower right", fontsize=5.6)

        metrics[f"{sample}_bin_centers"] = centers.tolist()
        metrics[f"{sample}_fid_fgas_all"] = fid_fgas.tolist()
        metrics[f"{sample}_fid_fgas_inpatch"] = fid_fgas_ip.tolist()
        metrics[f"{sample}_n_gal_all"] = fid["n_gal"].tolist()
        metrics[f"{sample}_n_gal_inpatch"] = fid["n_gal_inpatch"].tolist()
        metrics[f"{sample}_n_sb35_shards"] = n_plotted
        metrics[f"{sample}_envelope_min"] = envelope_min.tolist()
        metrics[f"{sample}_envelope_max"] = envelope_max.tolist()

        print(f"[M5] {sample} ({zlab}): fid f~gas(all)={np.round(fid_fgas,4)} "
              f"n_sb35_shards={n_plotted}")
        print(f"[M5] {sample} envelope min={np.round(envelope_min,4)} max={np.round(envelope_max,4)}")

        # ---------------- BGS-data "touch" count (>=1e13 bins) -------------
        # For each of the 5 DESIxACT BGS points, find the mass bin containing
        # its host logM200, then count how many SB35 nodes' f~gas value in
        # THAT bin overlaps the data point's own [fgas-err, fgas+err] --
        # i.e. how many node curves visually "touch" that star on the figure.
        if sample == "massbin85" and node_matrix.shape[0] > 0:
            bgs_bin_idx = np.clip(np.searchsorted(edges, lm, side="right") - 1, 0, n_bins - 1)
            touch_counts = {}
            per_node_touch = np.zeros((node_matrix.shape[0], len(lm)), dtype=bool)
            for i, (b_idx, f_i, e_i, cut) in enumerate(zip(bgs_bin_idx, fd, fe, data["bgs_mstar_cuts"])):
                vals = node_matrix[:, b_idx]
                touching = np.isfinite(vals) & (vals >= f_i - e_i) & (vals <= f_i + e_i)
                touch_counts[str(cut)] = int(touching.sum())
                per_node_touch[:, i] = touching
            n_touch_all5 = int(np.all(per_node_touch, axis=1).sum())
            n_touch_any = int(np.any(per_node_touch, axis=1).sum())
            metrics["bgs_touch_counts_per_cut"] = touch_counts
            metrics["bgs_touch_n_nodes_all5"] = n_touch_all5
            metrics["bgs_touch_n_nodes_any"] = n_touch_any
            metrics["bgs_touch_n_nodes_total"] = int(node_matrix.shape[0])
            print(f"[M5] BGS-data touch counts per M*-cut (out of {node_matrix.shape[0]} nodes): "
                  f"{touch_counts}")
            print(f"[M5] nodes touching ALL 5 BGS points: {n_touch_all5}; "
                  f"touching AT LEAST ONE: {n_touch_any}")

    axes[0].set_ylabel(r"$\tilde f_{\rm gas}$")
    axes[0].text(14.3, 1.02, "cosmic", fontsize=6)
    axes[1].axvspan(12.0, 12.4, color=ELG_GREEN, alpha=.12, zorder=0)
    axes[1].text(12.2, 1.07, "ELG hosts", fontsize=6, ha="center", color=ELG_GREEN)

    fig.suptitle("M5 -- map-level $\\tilde f_{\\rm gas}$ vs $\\log M_{200}$: BGS ($z{=}0.18$) "
                 "and ELG ($z{=}1.16$)", fontsize=11.5, fontweight="bold", y=1.03)
    caption = (
        "Map-level recreation of the per-halo headline figure (_build_ksz_paper_nb.py SS6b "
        "f6b_lowmass): f~gas = CAP_tau_gas/CAP_mat/F_B, stacked in 7 fixed FoF-M200 bins on "
        "the ray-traced lightcone tau map (left: 1.0*theta200 aperture; right: fixed 1.0' "
        "aperture, the ELG data's own innermost bin). Grey = BIND's sub-1e13 patch-reuse "
        "regime (dashed = in_patch-only substack, reuse-regime honesty); the pooled 'all' "
        "curve there is diluted by un-painted background/2-halo pixels and diverges from the "
        "per-halo reference -- see module docstring / M5 verdict for why this is expected, "
        "not a bug. Thin lines = individual SB35 Sobol nodes (sparse until the 253-node sweep "
        "lands, see docs/ksz_lightcone_map_plan.md M5)."
    )
    import textwrap
    fig.text(0.5, -0.06, "\n".join(textwrap.wrap(caption, width=175)), ha="center", va="top", fontsize=7.2)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig_path = FIG_DIR / "M5_fgas_mass.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"[M5] wrote {fig_path}")

    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, list) or len(v) < 10},
                      indent=2, default=float))
    return metrics


if __name__ == "__main__":
    main()
