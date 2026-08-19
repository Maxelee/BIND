#!/usr/bin/env python
"""Figures for the paper's *redshift dependence* subsection (Figs. R1-R4).

Load-only: consumes ``zcache.npz`` / ``zsweep.npz`` written by
``build_zcache.py`` and writes into ``paper_figures/``:

    fig_z_mass_error.pdf       R1  accuracy vs z          (same estimator as
                                   fig_mass_error, regrouped on redshift)
    fig_z_radial_pct_diff.pdf  R2  profile error vs z     (same estimator as
                                   fig_radial_pct_diff, coloured by redshift)
    fig_z_evolution.pdf        R3  f_b and SHMR evolution, truth vs BIND
    fig_z_response.pdf         R4  response to the redshift label at fixed DMO

Redshift is an ordered magnitude, so it is encoded with a perceptually uniform
sequential ramp; truth and BIND are separated by line style and marker fill, never
by colour alone.

Usage:  python tools/paper_cache/make_z_figures.py [--tag ""] [--figures R1 R3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# The figure set the paper actually draws on lives beside the notebooks that
# build it (the repo-root paper_figures/ holds an older May snapshot).
FIG_DIR = REPO / "examples" / "paper_figures"
CACHE_DIR = Path("/mnt/home/mlee1/ceph/paper_cache/fm_redshift")

# Snapshots carrying enough held-out halos for a percentile estimate. The
# companion lightcone paper spans z = 0-2.5, so z <= 2.002 is the range that
# actually has to hold up; snap 32 (z=3.01, 95 halos) is shown in R1 only, flagged
# sparse, and snap 24 (z=4.01, 13 halos) is excluded from every figure.
MAIN_SNAPS = [90, 82, 74, 60, 52, 44]
SPARSE_SNAPS = [32]
Z_THREE = [0.0000, 1.0452, 2.0020]        # the three redshifts of Fig. R3

MASS_NAMES = ["DM_hydro", "Gas", "Stars"]
CH_DISPLAY = {"DM_hydro": "DM (hydro)", "Gas": "Gas", "Stars": "Stars",
              "Total": "Total"}
CH_COLORS = {"DM_hydro": "#4C72B0", "Gas": "#DD8452",
             "Stars": "#55A868", "Total": "#4D4D4D"}
# Mathtext-safe subscripts (the display names carry spaces and parentheses,
# which do not survive \rm{...}).
CH_SUB = {"DM_hydro": r"\rm DM", "Gas": r"\rm gas", "Stars": r"\star"}
TRUTH_C, BIND_C = "k", "#C44E52"
ZCMAP = plt.get_cmap("viridis")


def _style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "notebook"])
    except Exception:
        pass
    plt.rcParams.update({
        "axes.grid": False, "savefig.dpi": 300, "figure.dpi": 110,
        "legend.frameon": False,
    })


def save_fig(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  saved {name}.pdf / .png")


def load(tag=""):
    sfx = f"_{tag}" if tag else ""
    zc = dict(np.load(CACHE_DIR / f"zcache{sfx}.npz", allow_pickle=True))
    sweep_path = CACHE_DIR / f"zsweep{sfx}.npz"
    sw = dict(np.load(sweep_path, allow_pickle=True)) if sweep_path.exists() else None
    return zc, sw


def z_color(z, zmax=2.05):
    """Sequential ramp over redshift, kept off the extreme light end."""
    return ZCMAP(0.08 + 0.84 * np.clip(z / zmax, 0, 1))


def _boot_median_prof(ph, n_boot=200, seed=0):
    """Bootstrap 68% CI on the median profile, resampling halos."""
    n = len(ph)
    if n < 10:
        return np.full(ph.shape[1], np.nan), np.full(ph.shape[1], np.nan)
    rng = np.random.RandomState(seed)
    with np.errstate(invalid="ignore"):
        meds = np.array([np.nanmedian(ph[rng.randint(0, n, n)], axis=0)
                         for _ in range(n_boot)])
    return np.percentile(meds, 16, axis=0), np.percentile(meds, 84, axis=0)


def _boot_median_ci(x, n_boot=400, seed=0):
    """68% CI on the median (the band that matters for 'is the bias zero?')."""
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    meds = np.median(rng.choice(x, (n_boot, len(x)), replace=True), axis=1)
    return np.percentile(meds, 16), np.percentile(meds, 84)


# ── R1: accuracy vs redshift ────────────────────────────────────────────────
def fig_r1(zc):
    """Integrated-mass error re-cut with redshift on the x-axis.

    Identical estimator to the paper's Fig. fig_mass_error --- (gen - truth)/truth
    per channel --- with the halo population grouped by snapshot instead of by
    evaluation suite.
    """
    snaps_all = MAIN_SNAPS + SPARSE_SNAPS
    snap, z = zc["snap"], zc["z"]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.2), sharex=True)
    rows = [("m_ap_t", "m_ap_g",
             r"$\Delta M / M_{\rm truth}$" + "\n" + r"(within $R_{200c}$)"),
            ("m_full_t", "m_full_g",
             r"$\Delta M / M_{\rm truth}$" + "\n" + "(full patch)")]

    for ax, (kt, kg, ylab) in zip(axes, rows):
        T, G = zc[kt], zc[kg]
        for ci, ch in enumerate(MASS_NAMES + ["Total"]):
            if ch == "Total":
                t, g = T.sum(axis=1), G.sum(axis=1)
            else:
                t, g = T[:, ci], G[:, ci]
            with np.errstate(divide="ignore", invalid="ignore"):
                res = (g - t) / t
            zs, med, lo, hi, sparse = [], [], [], [], []
            for s in snaps_all:
                m = (snap == s) & np.isfinite(res)
                if m.sum() < 10:
                    continue
                zs.append(z[m][0])
                med.append(np.median(res[m]))
                lo.append(np.percentile(res[m], 16))
                hi.append(np.percentile(res[m], 84))
                sparse.append(s in SPARSE_SNAPS)
            zs, med = np.array(zs), np.array(med)
            lo, hi, sparse = np.array(lo), np.array(hi), np.array(sparse)
            c = CH_COLORS[ch]
            ax.fill_between(zs, lo, hi, color=c, alpha=0.13, lw=0)
            ax.plot(zs, med, "-", color=c, lw=1.9, label=CH_DISPLAY[ch], zorder=3)
            ax.plot(zs[~sparse], med[~sparse], "o", color=c, ms=6,
                    mec="w", mew=0.8, zorder=4)
            # Sparse snapshots drawn hollow so they are not read as equally solid.
            ax.plot(zs[sparse], med[sparse], "o", color="w", ms=6, mec=c,
                    mew=1.6, zorder=4)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.35, 0.35)
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_axisbelow(True)

    zmax = max(zc["z"][zc["snap"] == s][0] for s in snaps_all)
    axes[0].axvspan(2.05, zmax + 0.15, color="0.85", alpha=0.5, lw=0, zorder=0)
    axes[1].axvspan(2.05, zmax + 0.15, color="0.85", alpha=0.5, lw=0, zorder=0)
    axes[0].text(0.5 * (2.05 + zmax + 0.15), 0.29, "sparse", ha="center",
                 fontsize=9, color="0.35")
    axes[0].legend(ncol=4, loc="lower left", fontsize=11)
    axes[1].set_xlabel(r"redshift $z$")
    axes[1].set_xlim(-0.1, zmax + 0.15)
    fig.tight_layout()
    save_fig(fig, "fig_z_mass_error")
    return fig


# ── R2: radial profile error vs redshift ────────────────────────────────────
def fig_r2(zc):
    """Fractional profile error, curves coloured by redshift rather than suite.

    Left column keeps the comoving x-axis of the paper's z=0 profile figure (the
    scale on which pixelization and over-smoothing act); the right column uses
    r/R200c, which removes the growth of R200c itself from the comparison. If the
    inner-radius degradation is a fixed *resolution* effect it stays put in the
    left column and drifts in the right.
    """
    snap, z = zc["snap"], zc["z"]
    grids = [("prof_com_t", "prof_com_g", zc["r_cen_com"],
              r"$r\ [{\rm Mpc}\,h^{-1}]$", "comoving radius"),
             ("prof_rr_t", "prof_rr_g", zc["rr_cen"],
              r"$r / R_{200c}$", "scaled radius")]

    # Per-row limits: the three channels sit at very different error levels
    # (DM ~1%, gas ~2%, stars ~10%), so one shared y-scale would compress two
    # of the three rows into a flat line and hide the actual z-dependence.
    ylims = [0.09, 0.14, 0.30]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.5), sharex="col")
    for col, (kt, kg, rcen, xlab, coltitle) in enumerate(grids):
        for ci, ch in enumerate(MASS_NAMES):
            ax = axes[ci, col]
            for s in MAIN_SNAPS:
                m = snap == s
                if m.sum() < 10:
                    continue
                t, g = zc[kt][m][:, ci], zc[kg][m][:, ci]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ph = np.where(t > 0, (g - t) / t, np.nan)
                zl = z[m][0]
                c = z_color(zl)
                med = np.nanmedian(ph, axis=0)
                # Bootstrap CI on the median, not the halo-to-halo spread: the
                # question is whether the curves differ *between* redshifts, and
                # the population scatter (16-84 ~ +/-30% for stars) is an order
                # of magnitude larger than that difference either way.
                lo, hi = _boot_median_prof(ph)
                ax.fill_between(rcen, lo, hi, color=c, alpha=0.30, lw=0, zorder=2)
                ax.plot(rcen, med, "-", color=c, lw=1.9, zorder=3,
                        label=f"$z={zl:.2f}$" if (ci == 0 and col == 0) else None)
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
            ax.set_xscale("log")
            ax.set_ylim(-ylims[ci], ylims[ci])
            ax.grid(which="both", alpha=0.18, lw=0.5)
            ax.set_axisbelow(True)
            if col == 0:
                sub = CH_SUB[ch]
                ax.set_ylabel(rf"$\Delta \Sigma_{{{sub}}} / \Sigma_{{{sub}}}$")
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.sharey(axes[ci, 0])
            if ci == 0:
                ax.set_title(coltitle)
            if ci == 2:
                ax.set_xlabel(xlab)
    axes[0, 0].legend(ncol=2, fontsize=10, loc="lower left")
    axes[0, 1].text(0.97, 0.06, "shaded: bootstrap 68% CI on the median",
                    transform=axes[0, 1].transAxes, ha="right", fontsize=9,
                    color="0.35")
    fig.tight_layout()
    save_fig(fig, "fig_z_radial_pct_diff")
    return fig


# ── R3: does the model reproduce evolution, not just stability? ─────────────
def _binned(x, y, edges, min_n=8):
    """Median and 16-84 of y in bins of x."""
    cen, med, lo, hi, nn = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b) & np.isfinite(y)
        if m.sum() < min_n:
            continue
        cen.append(0.5 * (a + b))
        med.append(np.median(y[m]))
        lo.append(np.percentile(y[m], 16))
        hi.append(np.percentile(y[m], 84))
        nn.append(int(m.sum()))
    return (np.array(cen), np.array(med), np.array(lo), np.array(hi),
            np.array(nn))


def fig_r3(zc):
    """Two relations that genuinely evolve, at three redshifts, truth vs BIND.

    Figs. R1/R2 only establish that the error does not grow with z --- they would
    look identical if the model ignored the redshift label entirely. These are
    relations whose *normalization moves with z*, so reproducing them requires the
    conditioning to carry physics.
    """
    z, hm = zc["z"], zc["halo_mass"]
    logM = np.log10(hm)
    fb0 = zc["omega_b"] / zc["omega_m"]
    T, G = zc["m_ap_t"], zc["m_ap_g"]

    with np.errstate(divide="ignore", invalid="ignore"):
        fb_t = (T[:, 1] + T[:, 2]) / T.sum(axis=1) / fb0
        fb_g = (G[:, 1] + G[:, 2]) / G.sum(axis=1) / fb0
        shmr_t = T[:, 2] / hm
        shmr_g = G[:, 2] / hm

    edges = np.arange(13.0, 14.76, 0.25)
    fig = plt.figure(figsize=(15.0, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.15], hspace=0.06, wspace=0.33)

    panels = [
        (0, fb_t, fb_g, r"$f_b(<\!R_{200c})\,/\,f_{b,0}$", (0.45, 1.15), False),
        (1, shmr_t, shmr_g, r"$M_\star(<\!R_{200c})\,/\,M_{200c}$",
         (2e-3, 6e-2), True),
    ]
    for col, yt, yg, ylab, ylim, logy in panels:
        ax = fig.add_subplot(gs[0, col])
        axr = fig.add_subplot(gs[1, col], sharex=ax)
        for zl in Z_THREE:
            m = np.isclose(z, zl, atol=1e-3)
            if m.sum() < 20:
                continue
            c = z_color(zl)
            ct, mt, lt, ht, _ = _binned(logM[m], yt[m], edges)
            cg, mg, _, _, _ = _binned(logM[m], yg[m], edges)
            ax.fill_between(ct, lt, ht, color=c, alpha=0.13, lw=0)
            ax.plot(ct, mt, "-", color=c, lw=2.0, zorder=3)
            ax.plot(cg, mg, "--", color=c, lw=1.8, zorder=4)
            ax.plot(cg, mg, "o", color="w", mec=c, mew=1.6, ms=6, zorder=5)
            n = min(len(mt), len(mg))
            axr.plot(ct[:n], mg[:n] / mt[:n], "--o", color=c, lw=1.5, ms=5,
                     mec=c, mfc="w", mew=1.4)
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_axisbelow(True)
        plt.setp(ax.get_xticklabels(), visible=False)
        if col == 0:
            # Colour carries redshift; line style carries truth vs BIND. Both
            # need a key, or the panel is unreadable.
            zkeys = [Line2D([], [], color=z_color(zl), lw=2.2,
                            label=f"$z={zl:.2f}$") for zl in Z_THREE]
            style = [Line2D([], [], color="0.35", lw=2.0, ls="-", label="True"),
                     Line2D([], [], color="0.35", lw=1.8, ls="--", marker="o",
                            mfc="w", mec="0.35", ms=6, label="BIND")]
            ax.legend(handles=zkeys + style, fontsize=9, ncol=2,
                      loc="lower right")
        axr.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
        axr.set_ylim(0.80, 1.20)
        axr.set_ylabel("BIND / True", fontsize=11)
        axr.set_xlabel(r"$\log_{10}\, M_{200c}\ [M_\odot\,h^{-1}]$")
        axr.grid(alpha=0.2, lw=0.5)
        axr.set_axisbelow(True)

    # Third panel: the evolution itself, at fixed halo mass.
    ax = fig.add_subplot(gs[:, 2])
    mbins = [(13.0, 13.5, "o", r"$13.0\!-\!13.5$"),
             (13.5, 14.5, "s", r"$13.5\!-\!14.5$")]
    for lo_m, hi_m, mk, lab in mbins:
        zs, tt, gg, tlo, thi = [], [], [], [], []
        for s in MAIN_SNAPS:
            m = (zc["snap"] == s) & (logM >= lo_m) & (logM < hi_m)
            # 40 keeps the sparse high-z end of the massive bin (N=39 at z=2)
            # from contributing a median the sample cannot support.
            if m.sum() < 40:
                continue
            zs.append(z[m][0])
            tt.append(np.median(fb_t[m]))
            gg.append(np.median(fb_g[m]))
            c16, c84 = _boot_median_ci(fb_t[m])
            tlo.append(c16)
            thi.append(c84)
        zs = np.array(zs)
        ax.fill_between(zs, tlo, thi, color=TRUTH_C, alpha=0.15, lw=0)
        ax.plot(zs, tt, "-", color=TRUTH_C, lw=2.0, marker=mk, ms=6, zorder=3,
                label=f"True, $\\log M$ {lab}")
        ax.plot(zs, gg, "--", color=BIND_C, lw=1.8, marker=mk, ms=6, mfc="w",
                mec=BIND_C, mew=1.5, zorder=4, label=f"BIND, $\\log M$ {lab}")
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"$f_b(<\!R_{200c})\,/\,f_{b,0}$")
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.set_title("evolution at fixed halo mass")

    fig.suptitle("", y=1.0)
    save_fig(fig, "fig_z_evolution")
    return fig


# ── R4: is the redshift conditioning actually used? ─────────────────────────
def fig_r4(zc, sw):
    """Hold the DMO patch and parameters fixed; sweep only the redshift label.

    Same idea as the 1P butterfly figure, with z in place of an astrophysical
    parameter. The initial ODE noise is identical at every z, so the residual maps
    isolate the effect of the conditioning. Off-grid redshifts --- values that
    match no training snapshot --- are drawn hollow: if the model had memorized the
    discrete snapshots rather than learned a function of a, they would not fall on
    the same smooth curve.
    """
    from bind.data import PIX_MPC_H

    zg, is_tr = sw["z_grid"], sw["is_train_z"].astype(bool)
    fields, hm = sw["fields"], sw["halo_mass"]
    r200_0 = sw["r200"][0]                       # z=0 aperture, held fixed
    nz, nh = fields.shape[:2]

    n = fields.shape[-1]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rpix = np.hypot(xx - c, yy - c) * PIX_MPC_H

    # A fixed aperture isolates the response of the field from the (trivial)
    # change in R200c(z) at fixed M200c.
    m_ap = np.zeros((nz, nh, 3))
    y_ap = np.zeros((nz, nh))
    for hi in range(nh):
        mask = rpix < r200_0[hi]
        for zi in range(nz):
            m_ap[zi, hi] = (fields[zi, hi, :3] * mask).sum(axis=(-2, -1))
            y_ap[zi, hi] = fields[zi, hi, 3][mask].sum() * PIX_MPC_H ** 2

    fig = plt.figure(figsize=(15.0, 9.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.30], hspace=0.22,
                          wspace=0.42)

    hero = int(np.argmax(hm))
    show_z = [0.0, 0.4679, 1.0452, 2.0020]
    zi_show = [int(np.argmin(np.abs(zg - t))) for t in show_z]
    ref = np.log10(fields[zi_show[0], hero, 1] + 1e-6)

    for k, zi in enumerate(zi_show):
        ax = fig.add_subplot(gs[0, k])
        img = np.log10(fields[zi, hero, 1] + 1e-6)
        ax.imshow(img, cmap="magma", vmin=np.percentile(ref, 5),
                  vmax=np.percentile(ref, 99.9))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$z_{{\rm label}} = {zg[zi]:.2f}$")
        if k == 0:
            ax.set_ylabel(r"$\log_{10}\,\Sigma_{\rm gas}$")

        axd = fig.add_subplot(gs[1, k])
        d = img - ref
        im = axd.imshow(d, cmap="RdBu_r", norm=TwoSlopeNorm(0, -0.6, 0.6))
        axd.set_xticks([])
        axd.set_yticks([])
        if k == 0:
            axd.set_ylabel(r"$\Delta \log_{10}\,\Sigma_{\rm gas}$")
            # The first residual panel is identically zero by construction; say
            # so rather than leaving a blank square the reader has to decode.
            axd.text(0.5, 0.5, "reference", transform=axd.transAxes,
                     ha="center", va="center", fontsize=11, color="0.45")
        if k == len(zi_show) - 1:
            cb = fig.colorbar(im, ax=axd, fraction=0.046, pad=0.04)
            cb.set_label(r"vs $z_{\rm label}=0$", fontsize=9)

    # Response curves, normalized to the z=0 label. No truth curve is drawn here
    # on purpose: the aperture is pinned to R200c(z=0) so that every change is
    # attributable to the label, whereas the truth population is measured inside
    # the growing R200c(z). The two conventions move f_b in opposite directions,
    # and overlaying them would read as a contradiction rather than a comparison.
    # The truth comparison is made properly, at matched aperture, in Fig. R3.
    curves = [
        (r"$M_{\rm gas}$", m_ap[:, :, 1], "#DD8452", False),
        (r"$M_\star$", m_ap[:, :, 2], "#55A868", False),
        (r"$f_b$", (m_ap[:, :, 1] + m_ap[:, :, 2]) / m_ap.sum(axis=2),
         "#4C72B0", False),
        (r"$Y_{200}$", y_ap, "#8172B3", True),
    ]

    for k, (lab, arr, col, logy) in enumerate(curves):
        ax = fig.add_subplot(gs[2, k])
        rel = arr / arr[0][None, :]
        med = np.median(rel, axis=1)
        ax.fill_between(zg, np.percentile(rel, 16, axis=1),
                        np.percentile(rel, 84, axis=1), color=col, alpha=0.15, lw=0)
        ax.plot(zg, med, "-", color=col, lw=2.0, zorder=3)
        ax.plot(zg[is_tr], med[is_tr], "o", color=col, ms=7, mec="w", mew=0.8,
                zorder=5, label="training $z$")
        ax.plot(zg[~is_tr], med[~is_tr], "D", color="w", mec=col, mew=1.8, ms=7,
                zorder=5, label="off-grid $z$")
        ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.4)
        # Only Y_200 spans a decade; forcing the near-unity mass ratios onto a log
        # axis turns their tick labels into unreadable "9x10^-1" clutter.
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"$z_{\rm label}$")
        ax.set_ylabel(rf"{lab}$(z)\,/\,${lab}$(0)$")
        ax.grid(alpha=0.2, lw=0.5, which="both")
        ax.set_axisbelow(True)
        if k == 0:
            ax.legend(fontsize=9, loc="lower left")
        if lab == r"$Y_{200}$":
            # The mass channels carry no scale-factor conversion at all, but the
            # thermo channels do, and those a-factors have never been checked
            # against an independent z>0 reference.
            ax.set_title("thermo: $a$-factors not\nindependently validated",
                         fontsize=8.5, color="0.35")

    fig.text(0.5, 0.985, f"fixed DMO patch and parameters "
             f"(hero halo $\\log M_{{200c}}={np.log10(hm[hero]):.2f}$; "
             f"curves: median over {nh} halos, identical initial noise)",
             ha="center", fontsize=11)
    save_fig(fig, "fig_z_response")
    return fig


# ── Snapshot table ──────────────────────────────────────────────────────────
def snapshot_table(zc, data_root="/mnt/home/mlee1/ceph/train_data_multiz_128_cpu"):
    """Emit the deluxetable of snapshots, redshifts and halo counts.

    Counts are read off the on-disk file lists rather than hard-coded, so the
    table cannot drift from the dataset it describes.
    """
    from bind.data import SNAPSHOT_REDSHIFTS

    train_list = Path(data_root) / "train" / "file_list_cache_multiz.txt"
    train_counts = {}
    if train_list.exists():
        txt = train_list.read_text()
        for s in SNAPSHOT_REDSHIFTS:
            train_counts[s] = txt.count(f"snap_{s:03d}/")

    idx = np.load(CACHE_DIR / "zindex_test.npz", allow_pickle=True)
    test_snap = idx["snap"]

    rows = []
    for s in sorted(SNAPSHOT_REDSHIFTS, reverse=True):
        z = SNAPSHOT_REDSHIFTS[s]
        n_tr = train_counts.get(s, 0)
        n_te = int((test_snap == s).sum())
        if s in MAIN_SNAPS:
            role = "train + evaluate"
        elif s in SPARSE_SNAPS:
            role = "train; sparse evaluation"
        else:
            role = "train only (too sparse to evaluate)"
        rows.append(rf"{s} & {z:.3f} & {n_tr:,} & {n_te:,} & {role} \\")

    tbl = "\n".join([
        r"\begin{deluxetable*}{lcccl}",
        r"\tabletypesize{\footnotesize}",
        r"\tablecaption{Snapshots of the multi-redshift training set. Halo counts "
        r"are the number of $M_{200c}\geq10^{13}\,M_\odot\,h^{-1}$ cutouts (one "
        r"projection per halo) in each split. The two highest-redshift snapshots "
        r"are retained in training but are too sparse to support a percentile "
        r"estimate, and are excluded from the evaluation figures."
        r"\label{tab:snapshots}}",
        r"\tablewidth{0pt}",
        r"\tablehead{\colhead{Snapshot} & \colhead{$z$} & "
        r"\colhead{$N_{\rm halo}$ (train)} & \colhead{$N_{\rm halo}$ (test)} & "
        r"\colhead{Role}}",
        r"\startdata",
        *rows,
        r"\enddata",
        r"\end{deluxetable*}",
    ])
    out = FIG_DIR / "tab_z_snapshots.tex"
    FIG_DIR.mkdir(exist_ok=True)
    out.write_text(tbl + "\n")
    print(tbl)
    print(f"\n  saved {out}")
    return tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--figures", nargs="*",
                    default=["R1", "R2", "R3", "R4"])
    ap.add_argument("--table", action="store_true",
                    help="also emit the snapshot deluxetable")
    ap.add_argument("--mass_table", default=None,
                    help="multi-z suite mass table from build_zmass.py; when given, "
                         "R1 is rebuilt per-suite on the CV/1P/SB35 population "
                         "instead of the SB35-only patch cache")
    args = ap.parse_args()

    _style()
    zc, sw = load(args.tag)
    print(f"loaded {len(zc['z'])} patches; "
          f"z levels = {sorted(set(np.round(zc['z'], 4).tolist()))}")

    if "R1" in args.figures:
        if args.mass_table:
            import pickle
            with open(args.mass_table, "rb") as f:
                fig_r1_matched(pickle.load(f))
        else:
            fig_r1(zc)
    if "R2" in args.figures:
        fig_r2(zc)
    if "R3" in args.figures:
        fig_r3(zc)
    if "R4" in args.figures:
        if sw is None:
            print("  zsweep cache missing — R4 skipped")
        else:
            fig_r4(zc, sw)
    if args.table:
        snapshot_table(zc)



# ── R1 (matched population): accuracy vs z, per suite ───────────────────────
SUITE_ORDER = ["CV", "1P", "Test"]
SUITE_LABEL = {"CV": "CV", "1P": "1P", "Test": "SB35"}
SUITE_COLOR = {"CV": "tab:green", "1P": "tab:blue", "Test": "tab:red"}


def fig_r1_matched(df, min_n=30):
    """Fig. R1 on the same CV/1P/SB35 decomposition as the paper's Fig. 3.

    Deliberately NOT a combined median over the three suites. The suite mix
    changes with redshift -- 1P falls from 56% of the population at z=0 to 20%
    at z=2, and is absent at z=0.209 because it has no hydro snapdir_082 -- and
    because the stellar bias differs in sign between suites (positive at
    near-fiducial CV/1P, negative across the wide SB35 prior), a combined curve
    swings non-monotonically from composition alone (+4.3, -4.2, +6.4, +7.6,
    +9.2, +0.4 percent). That is the very artifact this figure exists to remove.
    Fig. 3 also reports the three suites separately, so per-suite curves are both
    the honest and the concordant choice.
    """
    fig, axes = plt.subplots(2, 4, figsize=(17.0, 7.6), sharex=True, sharey="row")
    rows = [("_rvir", r"$\Delta M / M_{\rm truth}$" + "\n" + r"(within $R_{200c}$)"),
            ("", r"$\Delta M / M_{\rm truth}$" + "\n" + "(full patch)")]
    channels = MASS_NAMES + ["Total"]

    def med_by_z(sub, ch, suf):
        zs, md, lo, hi = [], [], [], []
        for z in sorted(sub["z"].unique()):
            s = sub[sub["z"] == z]
            if len(s) < min_n:
                continue
            if ch == "Total":
                t = sum(s[f"truth_{c}{suf}"] for c in MASS_NAMES).to_numpy()
                g = sum(s[f"gen_{c}{suf}"] for c in MASS_NAMES).to_numpy()
            else:
                t = s[f"truth_{ch}{suf}"].to_numpy()
                g = s[f"gen_{ch}{suf}"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (g - t) / t
            r = r[np.isfinite(r)]
            if len(r) < min_n:
                continue
            zs.append(z)
            md.append(np.median(r))
            lo.append(np.percentile(r, 16))
            hi.append(np.percentile(r, 84))
        return map(np.array, (zs, md, lo, hi))

    for row, (suf, ylab) in enumerate(rows):
        for col, ch in enumerate(channels):
            ax = axes[row, col]
            for suite in SUITE_ORDER:
                sub = df[df["suite"] == suite]
                if not len(sub):
                    continue
                zs, md, lo, hi = med_by_z(sub, ch, suf)
                if not len(zs):
                    continue
                c = SUITE_COLOR[suite]
                ax.fill_between(zs, lo, hi, color=c, alpha=0.12, lw=0)
                ax.plot(zs, md, "-o", color=c, lw=1.9, ms=5.5, mec="w", mew=0.8,
                        label=SUITE_LABEL[suite] if (row == 0 and col == 0) else None)
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
            ax.grid(alpha=0.2, lw=0.5)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(CH_DISPLAY[ch])
            if col == 0:
                ax.set_ylabel(ylab)
            if row == 1:
                ax.set_xlabel(r"redshift $z$")
        axes[row, 0].set_ylim(-0.42, 0.42)
    axes[0, 0].legend(ncol=3, loc="lower left", fontsize=10)
    # 1P has no hydro snapdir_082, so its curve is genuinely absent at z=0.209
    # rather than dropped for quality; say so instead of leaving a mystery gap.
    axes[0, 2].text(0.98, 0.04, "1P absent at $z=0.21$\n(no snapdir_082)",
                    transform=axes[0, 2].transAxes, ha="right", fontsize=8.5,
                    color="0.35")
    fig.tight_layout()
    save_fig(fig, "fig_z_mass_error")
    return fig

if __name__ == "__main__":
    main()
