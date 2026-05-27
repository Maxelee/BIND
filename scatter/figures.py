"""scatter/figures.py
Phase 4: Headline figures for the BIND scatter paper.

Produces:
  paper_figures/scatter/fig2_scatter_vs_mean.pdf   -- headline scatter vs mean Jacobian
  paper_figures/scatter/fig3_scatter_contours.pdf  -- sigma contours in (A_SN1, A_AGN1) plane
  paper_figures/scatter/fig4_inter_vs_intra.pdf    -- inter vs intra scatter breakdown

Fig 1 (calibration) is produced by scatter/calibration_cv.py.

Usage
-----
  # All figures (fig2 requires J_mean_and_scatter.npz to exist)
  python scatter/figures.py --all

  # Individual figures
  python scatter/figures.py --fig2 --jac scatter/J_mean_and_scatter.npz
  python scatter/figures.py --fig3
  python scatter/figures.py --fig4

  # Fig3 is expensive (~2 hr); skip in dry run
  python scatter/figures.py --fig2 --fig4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scatter.scatter_jacobian import PARAM_NAMES
from scatter.measure_scatter import ALL_OBS_NAMES, LOG_MASK, HEADLINE_OBS_NAMES

RUN_DIR  = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
CV_ROOT  = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
FIG_DIR  = Path("/mnt/home/mlee1/vdm_bind2/paper_figures/scatter")
SCAT_DIR = Path("/mnt/home/mlee1/vdm_bind2/scatter")

# Parameter group colors for fig2
PARAM_GROUPS = {
    "cosmo":    {"color": "#2166ac", "marker": "o", "indices": [0, 1, 6, 7, 8]},
    "SN":       {"color": "#d73027", "marker": "s", "indices": [2, 4]},
    "AGN":      {"color": "#fc8d59", "marker": "^", "indices": [3, 5]},
    "SN_sub":   {"color": "#fdae61", "marker": "D", "indices": list(range(9, 22))},
    "AGN_sub":  {"color": "#d7191c", "marker": "v", "indices": list(range(22, 29))},
    "other":    {"color": "#969696", "marker": "x", "indices": list(range(29, 35))},
}

# Headline observables for fig2: 3 panels
FIG2_OBS = ["M_gas", "M_star", "dq_DM"]


def save_fig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Saved {pdf}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: scatter vs mean Jacobian (headline plot)

def make_fig2(jac_path: Path):
    d = np.load(jac_path, allow_pickle=True)
    J_mean      = d["J_mean"]        # (N_obs, N_params)
    J_log_sigma = d["J_log_sigma"]   # (N_obs, N_params)
    J_mean_se   = d.get("J_mean_se", np.zeros_like(J_mean))
    J_log_sigma_se = d.get("J_log_sigma_se", np.zeros_like(J_log_sigma))
    obs_names   = list(d["obs_names"])

    n_panels = len(FIG2_OBS)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5),
                             sharex=False, sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, obs_name in zip(axes, FIG2_OBS):
        if obs_name not in obs_names:
            ax.set_title(f"{obs_name} (not found)")
            continue
        o = obs_names.index(obs_name)

        xvals = J_mean[o]         # (35,)
        yvals = J_log_sigma[o]    # (35,)
        xerr  = J_mean_se[o]
        yerr  = J_log_sigma_se[o]

        for group_name, ginfo in PARAM_GROUPS.items():
            idxs = [j for j in ginfo["indices"] if j < len(PARAM_NAMES)]
            if not idxs:
                continue
            x = xvals[idxs]
            y = yvals[idxs]
            xe = xerr[idxs]
            ye = yerr[idxs]
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.any():
                continue
            ax.errorbar(
                x[finite], y[finite],
                xerr=xe[finite] if np.isfinite(xe[finite]).all() else None,
                yerr=ye[finite] if np.isfinite(ye[finite]).all() else None,
                fmt=ginfo["marker"], color=ginfo["color"], alpha=0.8,
                markersize=7, label=group_name, capsize=3, lw=1,
            )
            # Label high-impact points
            for ii, j in enumerate(idxs):
                if not finite[ii]:
                    continue
                impact = np.sqrt(xvals[j]**2 + yvals[j]**2)
                top_n = np.argsort(np.sqrt(xvals**2 + yvals**2))[::-1][:6]
                if j in top_n:
                    ax.annotate(
                        PARAM_NAMES[j], (xvals[j], yvals[j]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points",
                        color=ginfo["color"], alpha=0.9,
                    )

        # Reference lines
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.4)

        ax.set_xlabel(r"$\partial \langle \bar{Y} \rangle / \partial \theta_j$", fontsize=12)
        ax.set_ylabel(r"$\partial \log \sigma_{\rm inter} / \partial \theta_j$", fontsize=12)
        obs_label = obs_name.replace("_", r"\_") if obs_name != "dq_DM" else r"$\Delta q_{\rm DM}$"
        ax.set_title(f"{obs_name}", fontsize=13)
        if obs_name == FIG2_OBS[0]:
            ax.legend(fontsize=7, loc="best", ncol=1, framealpha=0.7)

    fig.suptitle("Parameter responses: mean vs scatter\n"
                 "(x: shifts the mean, y: shifts the scatter)",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig2_scatter_vs_mean")


# ---------------------------------------------------------------------------
# Fig 3: scatter contours in (A_SN1, A_AGN1) plane

def make_fig3(
    K: int = 10, n_steps: int = 20, batch_size: int = 4,
    max_halos: int = 100, n_grid: int = 5,
    force_recompute: bool = False,
):
    import torch
    from data import NormStats, log_transform
    from fd_jacobian_cv import load_cv_halos, normalize_inputs, normalize_params_fid
    from scatter.measure_scatter import measure_scatter

    cache_path = SCAT_DIR / "fig3_grid_data.npz"

    # Indices in normalized-param space
    A_SN1_idx  = 2   # p2
    A_AGN1_idx = 3   # p3

    if cache_path.exists() and not force_recompute:
        print(f"Loading fig3 grid data from {cache_path}")
        d = np.load(cache_path)
        grid_sigma_inter = d["grid_sigma_inter"]  # (n_grid, n_grid, N_obs)
        grid_sigma_total = d["grid_sigma_total"]
        asn1_vals  = d["asn1_vals"]
        aagn1_vals = d["aagn1_vals"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        norm_stats = NormStats.load(RUN_DIR / "norm_stats.npz")
        from train import FlowMatchingLit
        lit = FlowMatchingLit.load_from_checkpoint(
            str(RUN_DIR / "checkpoints" / "last.ckpt"), map_location=device
        )
        lit.eval()
        if hasattr(lit, "ema"):
            del lit.ema
        model_fm = lit.fm
        model_fm.model.eval()

        cv = load_cv_halos(CV_ROOT)
        cv["params"][:, 14] = 0.0
        cond_norm, ls_norm = normalize_inputs(cv, norm_stats)
        cond_4d = cond_norm[:, np.newaxis]
        p_norm_fid = normalize_params_fid(cv["params"][0], norm_stats)

        # Subset halos
        rng = np.random.default_rng(0)
        N_TOT = len(cv["masses"])
        idx = np.sort(rng.choice(N_TOT, size=min(max_halos, N_TOT), replace=False))

        # Grid: ±3 sigma around fiducial in normalized space (fiducial = 0.5)
        asn1_vals  = np.linspace(0.2, 0.8, n_grid)
        aagn1_vals = np.linspace(0.2, 0.8, n_grid)

        N_obs = len(ALL_OBS_NAMES)
        grid_sigma_inter = np.full((n_grid, n_grid, N_obs), np.nan)
        grid_sigma_total = np.full((n_grid, n_grid, N_obs), np.nan)

        for i, asn1 in enumerate(asn1_vals):
            for j2, aagn1 in enumerate(aagn1_vals):
                theta = p_norm_fid.copy()
                theta[A_SN1_idx]  = asn1
                theta[A_AGN1_idx] = aagn1
                r = measure_scatter(
                    model_fm = model_fm,
                    norm_stats = norm_stats,
                    theta_norm = theta,
                    dmo_conds  = cond_4d[idx],
                    ls_conds   = ls_norm[idx],
                    masses     = cv["masses"][idx],
                    r200_pix   = cv["radii_pix"][idx],
                    K = K, n_steps = n_steps,
                    device = str(device), batch_size = batch_size,
                    dmo_raw  = cv["cond_raw"][idx],
                    omega_m  = cv["params"][idx, 0].astype(np.float64),
                    seed = 42,
                )
                grid_sigma_inter[i, j2] = r["sigma_inter"]
                grid_sigma_total[i, j2] = r["sigma_total"]
                print(f"  grid ({i},{j2}) A_SN1={asn1:.2f} A_AGN1={aagn1:.2f}  "
                      f"sigma_inter(M_gas)={r['sigma_inter'][ALL_OBS_NAMES.index('M_gas')]:.4f}",
                      flush=True)

        np.savez_compressed(
            cache_path,
            grid_sigma_inter = grid_sigma_inter,
            grid_sigma_total = grid_sigma_total,
            asn1_vals  = asn1_vals,
            aagn1_vals = aagn1_vals,
        )
        print(f"Saved grid data to {cache_path}")

    # Plot: 2 panels (M_gas sigma_inter and sigma_total)
    plot_obs = ["M_gas", "M_star"]
    fig, axes = plt.subplots(1, len(plot_obs), figsize=(6 * len(plot_obs), 5))
    if len(plot_obs) == 1:
        axes = [axes]

    for ax, obs_name in zip(axes, plot_obs):
        o = ALL_OBS_NAMES.index(obs_name)
        Z = grid_sigma_inter[:, :, o].T  # (n_grid_aagn1, n_grid_asn1)

        im = ax.contourf(asn1_vals, aagn1_vals, Z, levels=10, cmap="RdYlBu_r")
        ax.contour(asn1_vals, aagn1_vals, Z, levels=10, colors="k", linewidths=0.5, alpha=0.4)
        plt.colorbar(im, ax=ax, label=r"$\sigma_{\rm inter}$ (dex or linear)")

        ax.axvline(0.5, color="white", ls="--", lw=1, alpha=0.7, label="fiducial")
        ax.axhline(0.5, color="white", ls="--", lw=1, alpha=0.7)

        ax.set_xlabel(r"$A_{\rm SN1}$ (normalized)", fontsize=12)
        ax.set_ylabel(r"$A_{\rm AGN1}$ (normalized)", fontsize=12)
        ax.set_title(f"Scatter: {obs_name}", fontsize=12)
        ax.legend(fontsize=8)

    fig.suptitle(r"$\sigma_{\rm inter}(\log M)$ in the $(A_{\rm SN1}, A_{\rm AGN1})$ plane",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig3_scatter_contours")


# ---------------------------------------------------------------------------
# Fig 4: inter vs intra scatter breakdown

def make_fig4(bind_K10_path: Path | None = None):
    """Bar chart comparing sigma_inter vs sigma_intra at fiducial."""
    if bind_K10_path is None:
        bind_K10_path = SCAT_DIR / "cv_bind_obs_K10.npz"

    if not bind_K10_path.exists():
        print(f"Fig4: {bind_K10_path} not found — run calibration_cv.py first")
        return

    d = np.load(bind_K10_path)
    obs_tensor = d["obs_tensor"]   # (N_h, K, N_obs)
    masses     = d["masses"]

    N_h, K, N_obs = obs_tensor.shape
    Y = np.full_like(obs_tensor, np.nan, dtype=np.float64)
    for o in range(N_obs):
        x = obs_tensor[:, :, o].astype(np.float64)
        if LOG_MASK[o]:
            with np.errstate(divide="ignore", invalid="ignore"):
                Y[:, :, o] = np.where(x > 0, np.log10(x), np.nan)
        else:
            Y[:, :, o] = x

    Y_bar = np.nanmean(Y, axis=1)
    sigma_inter = np.nanstd(Y_bar, axis=0, ddof=1)
    sigma_intra = np.nanmean(np.nanstd(Y, axis=1, ddof=1), axis=0)

    # Plot headline observables only
    obs_to_plot = HEADLINE_OBS_NAMES
    plot_idxs   = [ALL_OBS_NAMES.index(n) for n in obs_to_plot if n in ALL_OBS_NAMES]
    plot_names  = [obs_to_plot[i] for i in range(len(obs_to_plot))
                   if obs_to_plot[i] in ALL_OBS_NAMES]

    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(plot_idxs)), 5))
    x = np.arange(len(plot_idxs))
    w = 0.35
    ax.bar(x - w/2, sigma_inter[plot_idxs], w, label=r"$\sigma_{\rm inter}$ (halo-to-halo)",
           color="steelblue", alpha=0.85)
    ax.bar(x + w/2, sigma_intra[plot_idxs], w, label=r"$\sigma_{\rm intra}$ (model noise)",
           color="firebrick", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"$\sigma$ (log$_{10}$ or linear)")
    ax.set_title("BIND: inter-halo scatter vs model stochasticity (fiducial θ)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig4_inter_vs_intra")


# ---------------------------------------------------------------------------
# CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fig2", action="store_true")
    ap.add_argument("--fig3", action="store_true")
    ap.add_argument("--fig4", action="store_true")
    ap.add_argument("--jac", type=str, default="scatter/J_mean_and_scatter.npz",
                    help="Path to scatter Jacobian npz for fig2")
    ap.add_argument("--fig3_halos", type=int, default=100)
    ap.add_argument("--fig3_grid", type=int, default=5)
    ap.add_argument("--fig3_K", type=int, default=10)
    ap.add_argument("--fig3_steps", type=int, default=20)
    ap.add_argument("--fig3_batch", type=int, default=4)
    ap.add_argument("--fig3_force", action="store_true")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.fig2:
        jac_path = Path(args.jac)
        if jac_path.exists():
            print("Making fig2 ...")
            make_fig2(jac_path)
        else:
            print(f"fig2: Jacobian not found at {jac_path}; skipping")

    if args.all or args.fig3:
        print("Making fig3 (may take ~2 hr) ...")
        make_fig3(
            K=args.fig3_K, n_steps=args.fig3_steps,
            batch_size=args.fig3_batch,
            max_halos=args.fig3_halos,
            n_grid=args.fig3_grid,
            force_recompute=args.fig3_force,
        )

    if args.all or args.fig4:
        print("Making fig4 ...")
        make_fig4()


if __name__ == "__main__":
    main()
