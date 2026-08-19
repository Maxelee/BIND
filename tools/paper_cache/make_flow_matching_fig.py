#!/usr/bin/env python3
"""Fig. `fig:flow_matching` — flow-matching trajectory + field-level comparison.

Runs the sampler live on one CV halo (default: the halo closest to
M200c = 1e14 M_sun/h in CV/sim_0) and snapshots the Euler trajectory at
t=0 (Gaussian noise), t=0.5 and t=1 (generated), then puts them beside the
paired IllustrisTNG truth patch and the signed residual.

Rows are the three mass channels [DM (hydro), Gas, Stars]; columns are
[t=0, t=0.5, t=1, Truth, (BIND-Truth)/Truth, smoothed residual].

The smoothed column (referee/advisor request) shows the residual at the scale
of a Gaussian kernel (--smooth_sigma, default 1 px). By default the two
*fields* are smoothed before taking the ratio (G⊗BIND / G⊗Truth − 1), i.e. a
mass-weighted local residual — smoothing the raw fractional residual instead
(--smooth_mode resid) is dominated by near-empty-truth pixels where the ratio
diverges.

    python tools/paper_cache/make_flow_matching_fig.py            # defaults
    python tools/paper_cache/make_flow_matching_fig.py --target_mass 1e14.5

Writes examples/paper_figures/flow_matching_diagram.{pdf,png} and copies the
PDF to imgs/ (where main.tex expects it).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_config as C  # noqa: E402

REPO = C.REPO_ROOT
FIG_DIR = REPO / "examples" / "paper_figures"
IMGS_DIR = REPO / "imgs"

# The paper spine model: fm_redshift evaluated at z=0 (a=1), 20 Euler steps —
# identical settings to the suite eval that produced every other figure
# (see <suite>/CV/sim_0/snap_090/.../fm_redshift/summary.json).
SUITE = Path("/mnt/home/mlee1/ceph/fm_redshift_suite/CV/sim_0/snap_090")
MASS_DIR = SUITE / "mass_threshold_1p000e13"
RUN_DIR = Path("/mnt/home/mlee1/ceph/fm_runs/fm_redshift")
CKPT = RUN_DIR / "checkpoints" / "last.ckpt"
NORM = RUN_DIR / "norm_stats.npz"

ROW_LABELS = ["DM (hydro)", "Gas", "Stars"]
ROW_CMAPS = ["jet", "jet", "jet"]
RESID_VMAX = [1.0, 1.0, 1.0]  # shared across rows so the panels are comparable
COL_TITLES = [r"$t=0$  (noise)", r"$t=0.5$", r"$t=1$  (BIND)", "Truth",
              r"BIND / Truth $-\,1$"]


def sample_trajectory(model, cond, large_scale, params, *, n_steps=20,
                      snap_ts=(0.0, 0.5, 1.0), seed=0, scale_factor=1.0):
    """Euler-integrate the flow from noise to data, keeping intermediate states.

    Mirrors ``FlowMatching.sample`` exactly (same normalization, bf16 autocast
    and step count as the suite eval) but records x_t at the requested times.
    Returns {t: (3+N_thermo, 128, 128) physical-space field}.
    """
    from bind.inference.pipeline import normalize_cutout, _denormalize_to_physical

    ns = model.norm_stats
    dev = model.device
    c_n, ls_n, p_n = normalize_cutout(
        {"condition": cond, "large_scale": large_scale}, ns, params)
    if model.param_indices is not None:
        p_n = p_n[model.param_indices]

    cond_t = torch.from_numpy(np.asarray(c_n, np.float32)[None]).to(dev)
    ls_t = None if model.no_large_scale else \
        torch.from_numpy(np.asarray(ls_n, np.float32)[None]).to(dev)
    par_t = torch.from_numpy(np.asarray(p_n, np.float32)[None]).to(dev)
    sf_t = (torch.full((1,), float(scale_factor), device=dev)
            if model.condition_redshift else None)
    sf_kw = {"scale_factor": sf_t} if sf_t is not None else {}

    fm = model.fm
    fm.model.eval()
    torch.manual_seed(seed)
    x = torch.randn(1, fm.out_channels, cond_t.shape[2], cond_t.shape[3], device=dev)

    want = sorted(float(t) for t in snap_ts)
    snaps: dict[float, torch.Tensor] = {}

    def maybe_keep(t_now):
        for t_w in want:
            if abs(t_now - t_w) < 1e-9 and t_w not in snaps:
                snaps[t_w] = x.detach().clone()

    dt = 1.0 / n_steps
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if dev.type == "cuda" else torch.autocast("cpu", enabled=False))
    with torch.no_grad(), ctx:
        maybe_keep(0.0)
        for i in range(n_steps):
            t = torch.full((1,), i * dt, device=dev)
            inp = torch.cat([x, cond_t, ls_t], dim=1) if ls_t is not None \
                else torch.cat([x, cond_t], dim=1)
            x = x + fm.model(inp, t, par_t, **sf_kw) * dt
            maybe_keep((i + 1) * dt)

    missing = [t for t in want if t not in snaps]
    if missing:
        raise ValueError(f"t={missing} is not on the n_steps={n_steps} Euler grid")

    return {t: _denormalize_to_physical(
                v.float().cpu().numpy().astype(np.float32), ns)[0]
            for t, v in snaps.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target_mass", type=float, default=1e14,
                    help="pick the CV halo with M200c closest to this (M_sun/h)")
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth_sigma", type=float, default=1.0,
                    help="Gaussian kernel width (pixels) for the smoothed-residual column")
    ap.add_argument("--smooth_mode", choices=["fields", "resid"], default="fields",
                    help="'fields': smooth BIND and Truth, then ratio (default); "
                         "'resid': normalized convolution of the masked residual map")
    ap.add_argument("--outname", default="flow_matching_diagram")
    args = ap.parse_args()

    from bind.inference.paint import Model

    cat = np.load(MASS_DIR / "halo_catalog.npz")
    cuts = np.load(MASS_DIR / "halo_cutouts.npz")
    full = np.load(SUITE / "full_maps.npz")

    masses = cat["masses"]
    h = int(np.argmin(np.abs(np.log10(masses) - np.log10(args.target_mass))))
    params = np.asarray(cat["params"][h], np.float64)   # CV: p14 already 0
    cx, cy = C.centers_to_pixels(cat["centers"])[h]
    print(f"[fig] CV/sim_0 halo {h}: log10 M200c = {np.log10(masses[h]):.3f}, "
          f"R200c = {cat['r200s'][h]:.3f} Mpc/h, centre pix = ({cx}, {cy})")

    truth = np.stack([C.extract_patch(full["truth_maps"][c], cx, cy)
                      for c in range(C.N_MASS_CH)])

    model = Model.from_files(CKPT, NORM, device="auto")
    print(f"[fig] {model}")
    traj = sample_trajectory(model, cuts["condition"][h], cuts["large_scale"][h],
                             params, n_steps=args.n_steps, seed=args.seed)
    gen = traj[1.0]

    # Consistency check against the cached suite-eval realization (different
    # noise draw, so masses agree only statistically).
    cached = np.load(MASS_DIR / "fm_redshift" / "generated_halos.npz")["generated"][h]
    for c, name in enumerate(ROW_LABELS):
        print(f"[fig] {name:11s} M_patch: truth {truth[c].sum():.4e}  "
              f"this draw {gen[c].sum():.4e}  cached {cached[c].sum():.4e}")

    # ── plot ────────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 11, "font.family": "serif",
                         "mathtext.fontset": "cm"})
    col_titles = COL_TITLES + [rf"smoothed ($\sigma={args.smooth_sigma:g}$ px)"]
    fig, axes = plt.subplots(3, 6, figsize=(17.8, 8.6),
                             gridspec_kw={"hspace": 0.04, "wspace": 0.04})

    for r in range(3):
        panels = [traj[0.0][r], traj[0.5][r], gen[r], truth[r]]
        # Common stretch per row, set by the truth panel so all four field
        # panels are directly comparable.
        pos = truth[r][truth[r] > 0]
        lo = np.quantile(pos, 0.05) if pos.size else 1e-30
        hi = np.quantile(pos, 0.999) if pos.size else 1.0
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.log10(np.clip(img, lo, None)), origin="lower",
                      cmap=ROW_CMAPS[r], interpolation="nearest",
                      vmin=np.log10(lo), vmax=np.log10(hi))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(COL_TITLES[c], fontsize=13, pad=8)
        axes[r, 0].set_ylabel(ROW_LABELS[r], fontsize=13)

        # residual — masked (grey) where the truth field is empty
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = (gen[r] - truth[r]) / truth[r]
        resid = np.ma.masked_where(~np.isfinite(resid) | (truth[r] <= 0), resid)

        # smoothed residual at the kernel scale. For sparse fields (Stars) the
        # denominator is floored at `lo` (5th pct of positive truth pixels,
        # ~a single star particle) so the ratio of two smoothed *tails* does
        # not saturate far from any real star; grey where neither smoothed
        # field reaches the floor. Dense fields keep the exact ratio.
        if args.smooth_mode == "fields":
            gs = gaussian_filter(gen[r], args.smooth_sigma)
            ts = gaussian_filter(truth[r], args.smooth_sigma)
            if (truth[r] <= 0).mean() > 0.05:   # sparse field
                resid_s = (gs - ts) / np.maximum(ts, lo)
                resid_s = np.ma.masked_where((ts < lo) & (gs < lo), resid_s)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    resid_s = gs / ts - 1.0
                resid_s = np.ma.masked_where(~np.isfinite(resid_s) | (ts <= 0),
                                             resid_s)
        else:
            num = gaussian_filter(resid.filled(0.0), args.smooth_sigma)
            den = gaussian_filter((~resid.mask).astype(float), args.smooth_sigma)
            with np.errstate(divide="ignore", invalid="ignore"):
                resid_s = num / den
            resid_s = np.ma.masked_where(den < 0.05, resid_s)

        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("0.85")
        for c, img in ((4, resid), (5, resid_s)):
            ax = axes[r, c]
            im = ax.imshow(img, origin="lower", cmap=cmap, interpolation="nearest",
                           vmin=-RESID_VMAX[r], vmax=RESID_VMAX[r])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=13, pad=8)
        cb = fig.colorbar(im, ax=axes[r, 5], fraction=0.046, pad=0.02)
        cb.set_label(r"$\Delta M / M_{\rm True}$", fontsize=11)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    IMGS_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{args.outname}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{args.outname}.png", dpi=200, bbox_inches="tight")
    shutil.copy(pdf, IMGS_DIR / pdf.name)
    print(f"[fig] wrote {pdf}\n[fig] wrote {IMGS_DIR / pdf.name}")


if __name__ == "__main__":
    main()
