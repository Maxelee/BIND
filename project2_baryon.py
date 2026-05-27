"""Project 2: Baryon fraction profiles & closure radius vs CAMELS parameters.

Reuses the per-halo profiles already in
``analysis_physics_cache/halo_features_fm_two_head.npz`` (mean Σ per radial
annulus for DMO, hydro DM, gas, stars) and the per-halo R200c written by
``project1_morphology.shapes_for_sim`` into ``proj1_shapes.npz``.

Per-halo derived quantities:
    Σ_c(R/R200c)       — surface density at fixed R/R200c grid (interp)
    M_enc_c(<R/R200c)  — enclosed projected mass per channel
    f_b(<R/R200c)      — (Σ_gas + Σ_⋆) / (Σ_DM + Σ_gas + Σ_⋆) cumulative
    R_c                — closure radius: smallest R with f_b/f_b_cosmic ≥ 0.9
                         (NaN if never reached)

Figures:
    proj2_fig1_sigma_profiles.{pdf,png}    — Σ(R/R200c) per channel × mass × A_AGN1
    proj2_fig2_fb_cumulative.{pdf,png}     — f_b(<R)/f_b_cosmic vs R/R200c
    proj2_fig3_closure_relation.{pdf,png}  — R_c/R200c vs f_b(<R200c)/f_b_cosmic
    proj2_fig4_alpha_gamma_vs_feedback.{pdf,png} — fitted relation params
    proj2_fig5_power_suppression.{pdf,png} — P(k) suppression vs A_AGN1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from project1_morphology import (
    BASE_CACHE, CACHE_DIR, FIG_DIR, MPC_PER_PIX, N_PARAMS, PARAM_LABELS,
    PATCH_PIX, SHAPES_CACHE, _onep_param_index, _param_use_log,
    build_shape_cache, join_with_base,
)
from metrics import power_spectrum_2d  # noqa: E402

# ----------------------------------------------------------------------
# constants

P2_CACHE = CACHE_DIR / 'proj2_baryon.npz'

# R/R200c grid where Σ, f_b, M_enc are sampled per halo
R_OVER_R200C = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0,
                          1.25, 1.5, 1.75, 2.0, 2.5, 3.0])

OMEGA_B_FIXED = 0.049         # CAMELS holds Ω_b fixed
CLOSURE_THRESHOLD = 0.90      # f_b/f_b_cosmic at which R_c is defined

plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif', 'mathtext.fontset': 'cm',
    'figure.dpi': 110, 'savefig.bbox': 'tight',
})


# ----------------------------------------------------------------------
# annulus geometry — match metrics.radial_profile binning exactly

def _annulus_pixel_counts(n_pix=PATCH_PIX, n_bins=32):
    """Number of pixel centres in each radial annulus, computed against the
    same grid used by metrics.radial_profile (centred on patch centre)."""
    H, W = n_pix, n_pix
    yy, xx = np.mgrid[:H, :W] - np.array([H / 2, W / 2])[:, None, None]
    r = np.sqrt(xx ** 2 + yy ** 2)
    bins = np.linspace(0, min(H, W) / 2, n_bins + 1)
    counts = np.array([
        ((r >= bins[k]) & (r < bins[k + 1])).sum() for k in range(n_bins)
    ])
    centres = 0.5 * (bins[:-1] + bins[1:])
    return centres, counts.astype(np.float64), bins


_R_CENTRES_PIX, _N_PIX_PER_BIN, _BIN_EDGES_PIX = _annulus_pixel_counts()


def _enclosed_mass_per_bin(profile_2d):
    """Cumulative enclosed mass per radial bin from a (N, n_r) profile.

    profile[i, k] is the mean field value in annulus k, so the mass in
    that annulus is profile * N_pix_in_annulus (in field-units pixels;
    valid for surface-mass-density fields where each pixel value is the
    column-summed mass). The output has same shape as input."""
    annulus_mass = profile_2d * _N_PIX_PER_BIN[None, :]
    return np.cumsum(annulus_mass, axis=1)


def _interp_at_r_over_r200c(profile_2d, r200c_pix, r_over):
    """Interpolate per-halo profile at R/R200c values.

    Args:
        profile_2d: (N, n_r) per-halo radial profile
        r200c_pix:  (N,) per-halo R200c in pixels
        r_over:     (M,) target R/R200c grid
    Returns:
        (N, M) interpolated values; NaN where R_target > patch radius
    """
    N = profile_2d.shape[0]
    M = len(r_over)
    out = np.full((N, M), np.nan)
    rmax_pix = _BIN_EDGES_PIX[-1]
    for i in range(N):
        if not np.isfinite(r200c_pix[i]) or r200c_pix[i] <= 0:
            continue
        r_targets = r_over * r200c_pix[i]
        valid = r_targets < rmax_pix
        if valid.any():
            out[i, valid] = np.interp(
                r_targets[valid], _R_CENTRES_PIX, profile_2d[i],
                left=np.nan, right=np.nan,
            )
    return out


def _enclosed_at_r_over_r200c(profile_2d, r200c_pix, r_over):
    """Enclosed mass per channel at R/R200c using cumulative bin-mass.

    Returns (N, M) cumulative mass; NaN where R_target > patch radius."""
    cum = _enclosed_mass_per_bin(profile_2d)             # (N, n_r)
    N = profile_2d.shape[0]
    out = np.full((N, len(r_over)), np.nan)
    rmax_pix = _BIN_EDGES_PIX[-1]
    for i in range(N):
        if not np.isfinite(r200c_pix[i]) or r200c_pix[i] <= 0:
            continue
        r_targets = r_over * r200c_pix[i]
        valid = r_targets < rmax_pix
        if valid.any():
            out[i, valid] = np.interp(
                r_targets[valid], _R_CENTRES_PIX, cum[i],
                left=0.0, right=cum[i, -1],
            )
    return out


def _closure_radius_pix(profile_dm, profile_gas, profile_star, r200c_pix,
                        f_b_cosmic, threshold=CLOSURE_THRESHOLD):
    """Smallest R [pix] where M_b(<R)/(M_b(<R)+M_dm(<R)) >= threshold * f_b_cosmic.
    NaN if never reached within patch."""
    cum_dm   = _enclosed_mass_per_bin(profile_dm)
    cum_gas  = _enclosed_mass_per_bin(profile_gas)
    cum_star = _enclosed_mass_per_bin(profile_star)
    cum_tot  = cum_dm + cum_gas + cum_star
    cum_b    = cum_gas + cum_star
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cum_tot > 0, cum_b / cum_tot, np.nan) / f_b_cosmic[:, None]
    N = profile_dm.shape[0]
    out = np.full(N, np.nan)
    for i in range(N):
        # find first bin index where ratio >= threshold
        ok = np.isfinite(ratio[i]) & (ratio[i] >= threshold)
        if ok.any():
            k = np.argmax(ok)
            # linear interp inside bin pair (k-1, k) for sub-bin precision
            if k > 0 and np.isfinite(ratio[i, k - 1]):
                lo, hi = ratio[i, k - 1], ratio[i, k]
                if hi > lo:
                    frac = (threshold - lo) / (hi - lo)
                    out[i] = (1 - frac) * _R_CENTRES_PIX[k - 1] \
                        + frac * _R_CENTRES_PIX[k]
                    continue
            out[i] = _R_CENTRES_PIX[k]
    return out


# ----------------------------------------------------------------------
# build per-halo derived cache

def build_p2_cache(rebuild=False):
    if P2_CACHE.exists() and not rebuild:
        print(f'[p2 cache] loading {P2_CACHE}')
        z = np.load(P2_CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}

    base, shapes = join_with_base()
    r200c_pix = shapes['r200c_mpch'] / MPC_PER_PIX
    omega_m = base['params'][:, 0]
    f_b_cosmic = OMEGA_B_FIXED / np.where(omega_m > 0, omega_m, np.nan)

    # per-channel profiles
    p_dm_t    = base['p_t'][:, 0]   # truth DM hydro
    p_gas_t   = base['p_t'][:, 1]
    p_star_t  = base['p_t'][:, 2]
    p_dm_g    = base['p_g'][:, 0]
    p_gas_g   = base['p_g'][:, 1]
    p_star_g  = base['p_g'][:, 2]
    p_dmo     = base['p_dmo']

    # Σ at R/R200c grid
    Sigma_t_dm   = _interp_at_r_over_r200c(p_dm_t,   r200c_pix, R_OVER_R200C)
    Sigma_t_gas  = _interp_at_r_over_r200c(p_gas_t,  r200c_pix, R_OVER_R200C)
    Sigma_t_star = _interp_at_r_over_r200c(p_star_t, r200c_pix, R_OVER_R200C)
    Sigma_g_dm   = _interp_at_r_over_r200c(p_dm_g,   r200c_pix, R_OVER_R200C)
    Sigma_g_gas  = _interp_at_r_over_r200c(p_gas_g,  r200c_pix, R_OVER_R200C)
    Sigma_g_star = _interp_at_r_over_r200c(p_star_g, r200c_pix, R_OVER_R200C)

    # Enclosed mass at the same R/R200c grid (for cumulative f_b)
    M_t_dm   = _enclosed_at_r_over_r200c(p_dm_t,   r200c_pix, R_OVER_R200C)
    M_t_gas  = _enclosed_at_r_over_r200c(p_gas_t,  r200c_pix, R_OVER_R200C)
    M_t_star = _enclosed_at_r_over_r200c(p_star_t, r200c_pix, R_OVER_R200C)
    M_g_dm   = _enclosed_at_r_over_r200c(p_dm_g,   r200c_pix, R_OVER_R200C)
    M_g_gas  = _enclosed_at_r_over_r200c(p_gas_g,  r200c_pix, R_OVER_R200C)
    M_g_star = _enclosed_at_r_over_r200c(p_star_g, r200c_pix, R_OVER_R200C)

    with np.errstate(divide='ignore', invalid='ignore'):
        fb_t = (M_t_gas + M_t_star) / (M_t_dm + M_t_gas + M_t_star)
        fb_g = (M_g_gas + M_g_star) / (M_g_dm + M_g_gas + M_g_star)
        fb_t_norm = fb_t / f_b_cosmic[:, None]
        fb_g_norm = fb_g / f_b_cosmic[:, None]

    # closure radius (pixels) → convert to R200c units
    Rc_pix_t = _closure_radius_pix(p_dm_t, p_gas_t, p_star_t,
                                   r200c_pix, f_b_cosmic)
    Rc_pix_g = _closure_radius_pix(p_dm_g, p_gas_g, p_star_g,
                                   r200c_pix, f_b_cosmic)
    Rc_over_R200_t = Rc_pix_t / r200c_pix
    Rc_over_R200_g = Rc_pix_g / r200c_pix

    out = dict(
        suite=base['suite'], sim_id=base['sim_id'],
        logM=base['logM'], params=base['params'],
        r200c_mpch=shapes['r200c_mpch'], f_b_cosmic=f_b_cosmic,
        r_over_r200c=R_OVER_R200C,
        Sigma_t_dm=Sigma_t_dm, Sigma_t_gas=Sigma_t_gas, Sigma_t_star=Sigma_t_star,
        Sigma_g_dm=Sigma_g_dm, Sigma_g_gas=Sigma_g_gas, Sigma_g_star=Sigma_g_star,
        fb_t=fb_t, fb_g=fb_g, fb_t_norm=fb_t_norm, fb_g_norm=fb_g_norm,
        Rc_over_R200_t=Rc_over_R200_t, Rc_over_R200_g=Rc_over_R200_g,
    )
    np.savez_compressed(P2_CACHE, **out)
    print(f'Wrote {P2_CACHE} ({P2_CACHE.stat().st_size/1e6:.1f} MB)')
    return out


# ----------------------------------------------------------------------
# helpers

def _agg_curve(arr, sel, n_min=10):
    """Median ± 16/84 percentile over selected halos, columnwise."""
    sub = arr[sel]
    n = np.isfinite(sub).sum(axis=0)
    med = np.where(n >= n_min, np.nanmedian(sub, axis=0), np.nan)
    lo  = np.where(n >= n_min, np.nanquantile(sub, 0.16, axis=0), np.nan)
    hi  = np.where(n >= n_min, np.nanquantile(sub, 0.84, axis=0), np.nan)
    return med, lo, hi, n


def _onep_sel(sim_id, j):
    return np.array([(_onep_param_index(s) == j) or (s == '1P_p1_0')
                     for s in sim_id])


def _per_sim_spearman_columnwise(value_2d, suite, sim_id, params, logM,
                                  mass_lo, mass_hi, suite_name='Test',
                                  n_min_per_sim=3, min_sims=10):
    """For a per-halo 2D array `value_2d` of shape (N, M), aggregate to
    per-sim medians within the mass slice and return Spearman ρ between
    each parameter (35) and the per-sim median, for each of the M columns.
    Returns (35, M) array of ρ."""
    from scipy.stats import spearmanr
    sel = (suite == suite_name) & (logM >= mass_lo) & (logM < mass_hi)
    out = np.full((N_PARAMS, value_2d.shape[1]), np.nan)
    if sel.sum() < 50:
        return out

    sub_sim = sim_id[sel]
    sub_val = value_2d[sel]                      # (n_sel, M)

    # build per-sim parameter table
    uniq_sims = np.unique(sub_sim)
    sim_to_params = {}
    for sid in uniq_sims:
        idx = np.argmax((suite == suite_name) & (sim_id == sid))
        sim_to_params[sid] = params[idx]

    # per-sim median for each column
    per_sim_med = []
    per_sim_param = []
    for sid in uniq_sims:
        rows = sub_sim == sid
        if rows.sum() < n_min_per_sim:
            continue
        med = np.nanmedian(sub_val[rows], axis=0)
        per_sim_med.append(med)
        per_sim_param.append(sim_to_params[sid])
    if len(per_sim_med) < min_sims:
        return out

    Y = np.vstack(per_sim_med)                   # (S, M)
    P = np.vstack(per_sim_param)                 # (S, 35)
    for j in range(N_PARAMS):
        x = np.log10(P[:, j]) if _param_use_log(j + 1) else P[:, j]
        if not np.isfinite(x).all() or x.std() < 1e-9:
            continue
        for c in range(Y.shape[1]):
            y = Y[:, c]
            if np.isfinite(y).sum() < 6:
                continue
            ok = np.isfinite(y)
            out[j, c] = spearmanr(x[ok], y[ok]).statistic
    return out


def _per_sim_spearman_scalar(value_1d, suite, sim_id, params, logM,
                              mass_lo, mass_hi, suite_name='Test',
                              n_min_per_sim=3, min_sims=10):
    """Same as columnwise but for a (N,) per-halo value. Returns shape (35,)."""
    return _per_sim_spearman_columnwise(
        value_1d[:, None], suite, sim_id, params, logM, mass_lo, mass_hi,
        suite_name, n_min_per_sim, min_sims,
    )[:, 0]


# ----------------------------------------------------------------------
# Fig 1 — Σ profiles per channel × mass × A_AGN1

def plot_sigma_profiles(p2):
    """Full 35-parameter × R/R200c map of how each parameter modulates the
    *baryon* surface density Σ_gas + Σ_⋆. Truth and BIND2 panels share the
    color scale; difference panel highlights any emulator-specific bias.
    Mass slice 13 ≤ log M < 14 gives the largest sim sample at group scale."""
    suite = p2['suite']; sim_id = p2['sim_id']; logM = p2['logM']
    params = p2['params']; r = p2['r_over_r200c']

    Sigma_t = p2['Sigma_t_gas'] + p2['Sigma_t_star']    # baryon surface density
    Sigma_g = p2['Sigma_g_gas'] + p2['Sigma_g_star']

    rho_t = _per_sim_spearman_columnwise(
        Sigma_t, suite, sim_id, params, logM, 13.0, 14.0)
    rho_g = _per_sim_spearman_columnwise(
        Sigma_g, suite, sim_id, params, logM, 13.0, 14.0)
    rho_d = rho_g - rho_t

    param_labels = [PARAM_LABELS.get(j + 1, f'p{j+1}') for j in range(N_PARAMS)]
    rcol_labels = [f'{rv:.2g}' for rv in r]

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True,
                             constrained_layout=True)
    panels = [
        (axes[0], rho_t, 'Truth',         'coolwarm', (-0.6, 0.6)),
        (axes[1], rho_g, 'BIND2',         'coolwarm', (-0.6, 0.6)),
        (axes[2], rho_d, 'BIND2 − Truth', 'PiYG',     (-0.3, 0.3)),
    ]
    for ax, M, title, cmap, (vmin, vmax) in panels:
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        ax.set_xticks(np.arange(len(r)))
        ax.set_xticklabels(rcol_labels, rotation=60, ha='right', fontsize=7)
        ax.set_xlabel(r'$R/R_{200c}$')
        ax.set_title(title)
        if ax is axes[0]:
            ax.set_yticks(np.arange(N_PARAMS))
            ax.set_yticklabels(param_labels, fontsize=7)
        thresh = 0.25 if 'BIND2 −' not in title else 0.20
        for ri in range(M.shape[0]):
            for ci in range(M.shape[1]):
                if np.isfinite(M[ri, ci]) and abs(M[ri, ci]) > thresh:
                    ax.text(ci, ri, f'{M[ri, ci]:+.2f}',
                            ha='center', va='center', fontsize=5,
                            color='black')
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.set_label('Spearman ρ' if 'BIND2 −' not in title else r'$\Delta\rho$')

    fig.suptitle(r'Per-parameter response of total baryon surface density '
                 r'$\Sigma_{\rm gas}+\Sigma_\star$  '
                 r'(SB35 per-sim aggregation, $\log M\in[13,14)$)',
                 y=1.01)
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj2_fig1_sigma_profiles.{ext}')
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 2 — cumulative f_b(<R) / f_b_cosmic

def plot_fb_cumulative(p2):
    """Per-parameter Spearman map for f_b(<R)/f_b_cosmic over the full 14-point
    R/R200c grid. Below the heatmap, three example median curves (truth + BIND2)
    are stacked at the three top-impact parameters' extreme values to anchor
    the abstract correlation map to a familiar profile picture."""
    from scipy.stats import spearmanr

    suite = p2['suite']; sim_id = p2['sim_id']; logM = p2['logM']
    params = p2['params']; r = p2['r_over_r200c']

    rho_t = _per_sim_spearman_columnwise(
        p2['fb_t_norm'], suite, sim_id, params, logM, 13.0, 14.0)
    rho_g = _per_sim_spearman_columnwise(
        p2['fb_g_norm'], suite, sim_id, params, logM, 13.0, 14.0)

    # rank parameters by max |ρ| over R for the truth response
    impact = np.nanmax(np.abs(rho_t), axis=1)
    top_j = np.argsort(-impact)[:3] + 1            # 1-based

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1.0])

    param_labels = [PARAM_LABELS.get(j + 1, f'p{j+1}') for j in range(N_PARAMS)]
    rcol_labels = [f'{rv:.2g}' for rv in r]

    panels = [
        (fig.add_subplot(gs[0, 0]), rho_t, 'Truth',         'coolwarm', (-0.6, 0.6)),
        (fig.add_subplot(gs[0, 1]), rho_g, 'BIND2',         'coolwarm', (-0.6, 0.6)),
        (fig.add_subplot(gs[0, 2]), rho_g - rho_t, 'BIND2 − Truth',
         'PiYG', (-0.3, 0.3)),
    ]
    for ax, M, title, cmap, (vmin, vmax) in panels:
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        ax.set_xticks(np.arange(len(r)))
        ax.set_xticklabels(rcol_labels, rotation=60, ha='right', fontsize=7)
        ax.set_xlabel(r'$R/R_{200c}$')
        ax.set_title(title)
        if ax is panels[0][0]:
            ax.set_yticks(np.arange(N_PARAMS))
            ax.set_yticklabels(param_labels, fontsize=7)
        else:
            ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.set_label('Spearman ρ' if 'BIND2 −' not in title else r'$\Delta\rho$')

    # bottom row: example median curves at the extreme values of the three
    # most-impactful parameters, to anchor the heatmap to actual profiles
    sb = suite == 'Test'
    mass_sel_base = sb & (logM >= 13.0) & (logM < 14.0)
    for ci, j in enumerate(top_j):
        ax = fig.add_subplot(gs[1, ci])
        pvals = params[mass_sel_base, j - 1]
        if pvals.size == 0:
            ax.axis('off'); continue
        edges = np.quantile(pvals, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cmap = plt.get_cmap('plasma')
        colors = cmap(np.linspace(0.05, 0.92, 5))
        for k in range(5):
            sel_k = mass_sel_base & (params[:, j - 1] >= edges[k]) \
                & (params[:, j - 1] <= edges[k + 1])
            if sel_k.sum() < 10:
                continue
            med_t, _, _, _ = _agg_curve(p2['fb_t_norm'], sel_k)
            med_g, _, _, _ = _agg_curve(p2['fb_g_norm'], sel_k)
            ax.plot(r, med_t, color=colors[k], lw=1.4)
            ax.plot(r, med_g, color=colors[k], lw=0.9, ls='--')
        ax.axhline(1.0, color='gray', lw=0.6, ls=':')
        ax.set_xscale('log')
        ax.set_xlabel(r'$R/R_{200c}$')
        ax.set_ylim(0, 1.5)
        ax.set_title(rf'{PARAM_LABELS[j]}  (truth solid, BIND2 dashed; quintile colour)',
                     fontsize=9)
        ax.grid(alpha=0.3)
        if ci == 0:
            ax.set_ylabel(r'$f_b(<R)/f_{b,\rm cosmic}$')

    fig.suptitle(r'Cumulative baryon-fraction response: 35-parameter $\rho$ map '
                 r'plus median profiles at quintile values of the top-3 drivers  '
                 r'(SB35, $\log M\!\in\![13,14)$)',
                 y=1.02)
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj2_fig2_fb_cumulative.{ext}')
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 3 — closure radius vs f_b(<R200c) (Ayromlou universal relation)

def _fb_at_R200c(p2, kind='t'):
    """Per-halo f_b(<R200c)/f_b_cosmic — pick column with R/R200c ≈ 1.0."""
    r = p2['r_over_r200c']
    idx = int(np.argmin(np.abs(r - 1.0)))
    return p2[f'fb_{kind}_norm'][:, idx]


def plot_closure_relation(p2):
    """R_c/R200c vs f_b(<R200c)/f_b_cosmic for all SB35+1P halos.
    Left panel: greyscale density (hexbin) of the full ensemble + Ayromlou
    universal-relation fit, so the underlying scaling is unambiguous.
    Other panels: same hexbin background overlaid with quintile-median
    tracks for the top-3 parameters that most strongly modulate R_c
    (per-sim Spearman ranking). Each parameter gets a distinct colour
    family, so the parameter modulation is readable rather than a blur."""
    suite = p2['suite']; sim_id = p2['sim_id']; logM = p2['logM']
    params = p2['params']
    sb_or_1p = (suite == 'Test') | (suite == '1P')

    fb_t = _fb_at_R200c(p2, 't')
    Rc_t = p2['Rc_over_R200_t']
    sel = sb_or_1p & (logM >= 13.0) & np.isfinite(fb_t) & np.isfinite(Rc_t)
    x = fb_t[sel]; y = Rc_t[sel]
    print(f'[fig3] n halos = {sel.sum()}')

    # rank parameters by per-sim Spearman with R_c/R200c
    rho_Rc = _per_sim_spearman_scalar(
        p2['Rc_over_R200_t'], suite, sim_id, params, logM, 13.0, 15.0)
    top_j = np.argsort(-np.abs(np.nan_to_num(rho_Rc)))[:3] + 1
    print(f'[fig3] top-3 R_c drivers: {list(top_j)} '
          f'(rho={rho_Rc[top_j-1].round(2).tolist()})')

    # universal-relation fit
    ok = (x > 0) & (x < 1.5) & (y > 0)
    A = np.vstack([1 - x[ok], np.ones_like(x[ok])]).T
    coef, *_ = np.linalg.lstsq(A, y[ok] - 1.0, rcond=None)
    beta_fit = float(coef[0])
    gamma_fit = float(coef[1])

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharey=True, sharex=True)
    axes = axes.ravel()

    # panel 0: density only + universal-relation line
    ax0 = axes[0]
    hb = ax0.hexbin(x, y, gridsize=50, cmap='Greys', mincnt=1, bins='log',
                    extent=[0, 1.4, 0.5, 3.5])
    xx = np.linspace(0, 1.0, 100)
    ax0.plot(1 - xx, 1 + gamma_fit + beta_fit * xx, 'r-', lw=1.8,
             label=rf'universal fit: $R_c/R_{{200c}}-1={gamma_fit:+.2f}+{beta_fit:+.2f}\,(1-f_b/f_{{b,c}})$')
    ax0.axhline(1.0, color='gray', lw=0.6, ls=':')
    ax0.axvline(1.0, color='gray', lw=0.6, ls=':')
    ax0.set_xlim(0, 1.4); ax0.set_ylim(0.5, 3.5)
    ax0.set_xlabel(r'$f_b(<R_{200c})/f_{b,\rm cosmic}$')
    ax0.set_ylabel(r'$R_c/R_{200c}$  (closure radius)')
    ax0.set_title(f'all SB35+1P halos  ($n={int(sel.sum())}$)', fontsize=10)
    ax0.legend(loc='upper right', fontsize=7)
    ax0.grid(alpha=0.3)
    cb0 = fig.colorbar(hb, ax=ax0, shrink=0.85, pad=0.01)
    cb0.set_label(r'$\log_{10}$ count')

    # panels 1-3: hexbin background (light) + median tracks per quintile
    cmap_families = ['Blues', 'Oranges', 'Greens']
    for k, j in enumerate(top_j):
        ax = axes[k + 1]
        ax.hexbin(x, y, gridsize=50, cmap='Greys', mincnt=1, bins='log',
                  alpha=0.35, extent=[0, 1.4, 0.5, 3.5])
        # quintile-median tracks
        pvals = params[sel, j - 1]
        edges = np.quantile(pvals, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cmap = plt.get_cmap(cmap_families[k])
        colors = cmap(np.linspace(0.3, 0.95, 5))
        # bin in f_b axis to compute median R_c per (param-quintile, f_b-bin)
        fb_bins = np.linspace(0.2, 1.3, 12)
        fb_centres = 0.5 * (fb_bins[:-1] + fb_bins[1:])
        for q in range(5):
            sel_q = (pvals >= edges[q]) & (pvals <= edges[q + 1])
            if sel_q.sum() < 30:
                continue
            xs = x[sel_q]; ys = y[sel_q]
            med = np.full(len(fb_centres), np.nan)
            for b in range(len(fb_centres)):
                inb = (xs >= fb_bins[b]) & (xs < fb_bins[b + 1])
                if inb.sum() >= 5:
                    med[b] = np.median(ys[inb])
            ax.plot(fb_centres, med, color=colors[q], lw=1.6,
                    label=rf'$Q_{q+1}\!\in\![{edges[q]:.2g},{edges[q+1]:.2g}]$')
        ax.plot(1 - xx, 1 + gamma_fit + beta_fit * xx, 'r-', lw=1.0,
                ls=':', alpha=0.8, label='universal fit')
        ax.axhline(1.0, color='gray', lw=0.5, ls=':')
        ax.axvline(1.0, color='gray', lw=0.5, ls=':')
        ax.set_xlim(0, 1.4); ax.set_ylim(0.5, 3.5)
        ax.set_xlabel(r'$f_b(<R_{200c})/f_{b,\rm cosmic}$')
        ax.set_title(rf'{PARAM_LABELS[j]}  ($\rho={rho_Rc[j-1]:+.2f}$)',
                     fontsize=10)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(alpha=0.3)

    fig.suptitle(r'Closure radius vs baryon deficit  — '
                 r'top-3 drivers shown as quintile-median tracks over the '
                 r'shared hexbin density',
                 y=1.02)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj2_fig3_closure_relation.{ext}')
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 4 — fitted relation parameters (β slope, γ intercept) vs feedback

def _fit_beta_gamma(fb_norm, rc_over_r200, sel):
    """Fit linear model R_c/R200c - 1 = γ + β (1 - f_b/f_b_cosmic) over `sel`."""
    x = 1.0 - fb_norm[sel]
    y = rc_over_r200[sel] - 1.0
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1.0) & (np.abs(y) < 3.0)
    if ok.sum() < 30:
        return np.nan, np.nan, np.nan, np.nan
    A = np.vstack([x[ok], np.ones_like(x[ok])]).T
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    res = y[ok] - A @ coef
    sigma = float(np.std(res, ddof=2))
    sx = float(np.sqrt(((x[ok] - x[ok].mean()) ** 2).sum()))
    return float(coef[0]), float(coef[1]), sigma / max(sx, 1e-12), int(ok.sum())


def _fit_beta_gamma_at_node(fb, rc, sel):
    """Fit β,γ for the universal relation R_c/R200c - 1 = γ + β (1 - f_b/f_b_c)
    on `sel` halos."""
    return _fit_beta_gamma(fb, rc, sel)


def plot_alpha_gamma_vs_feedback(p2):
    """Per-1P-node fit of (β, γ) for ALL 35 CAMELS parameters. Two row sets
    show truth (solid) and BIND2 (dashed) values; the y-range is the spread
    of each fit parameter across the 1P scan, so a near-flat line means the
    universal relation truly is independent of that parameter while a wide
    spread flags a parameter-dependent break of universality.

    Plotted is the *log-range* of each parameter's effect on β and γ, so
    the figure stays compact and the modulation strength can be read off
    directly."""
    suite = p2['suite']; sim_id = p2['sim_id']
    params = p2['params']; logM = p2['logM']
    fb_t = _fb_at_R200c(p2, 't'); fb_g = _fb_at_R200c(p2, 'g')
    Rc_t = p2['Rc_over_R200_t']; Rc_g = p2['Rc_over_R200_g']
    is_1p = suite == '1P'
    mass_mask = (logM >= 13.0)

    beta_range_t = np.full(N_PARAMS, np.nan)
    gamma_range_t = np.full(N_PARAMS, np.nan)
    beta_range_g = np.full(N_PARAMS, np.nan)
    gamma_range_g = np.full(N_PARAMS, np.nan)
    beta_med_t = np.full(N_PARAMS, np.nan)
    gamma_med_t = np.full(N_PARAMS, np.nan)
    n_nodes = np.zeros(N_PARAMS, dtype=int)

    for j in range(1, N_PARAMS + 1):
        sel_node = _onep_sel(sim_id, j) & is_1p & mass_mask
        pvals = params[sel_node, j - 1]
        uniq = np.sort(np.unique(pvals))
        if len(uniq) < 3:
            continue
        bt, gt, bg, gg = [], [], [], []
        for v in uniq:
            sel_v_t = sel_node & np.isclose(params[:, j - 1], v) \
                & np.isfinite(fb_t) & np.isfinite(Rc_t)
            sel_v_g = sel_node & np.isclose(params[:, j - 1], v) \
                & np.isfinite(fb_g) & np.isfinite(Rc_g)
            b_t, g_t, _, _ = _fit_beta_gamma_at_node(fb_t, Rc_t, sel_v_t)
            b_g, g_g, _, _ = _fit_beta_gamma_at_node(fb_g, Rc_g, sel_v_g)
            if np.isfinite(b_t):
                bt.append(b_t); gt.append(g_t); bg.append(b_g); gg.append(g_g)
        if len(bt) < 3:
            continue
        bt, gt, bg, gg = map(np.asarray, (bt, gt, bg, gg))
        beta_range_t[j - 1]  = float(np.nanmax(bt) - np.nanmin(bt))
        gamma_range_t[j - 1] = float(np.nanmax(gt) - np.nanmin(gt))
        beta_range_g[j - 1]  = float(np.nanmax(bg) - np.nanmin(bg))
        gamma_range_g[j - 1] = float(np.nanmax(gg) - np.nanmin(gg))
        beta_med_t[j - 1]    = float(np.nanmedian(bt))
        gamma_med_t[j - 1]   = float(np.nanmedian(gt))
        n_nodes[j - 1] = len(bt)

    # global truth-only fit for reference
    sel_all = is_1p & mass_mask & np.isfinite(fb_t) & np.isfinite(Rc_t)
    b_glob, g_glob, _, _ = _fit_beta_gamma_at_node(fb_t, Rc_t, sel_all)

    fig, axes = plt.subplots(2, 1, figsize=(15, 7.5), sharex=True)
    x = np.arange(1, N_PARAMS + 1)
    width = 0.4

    for ax, label, t_arr, g_arr, ref in [
        (axes[0], r'$\beta$  spread across 1P scan',
         beta_range_t, beta_range_g, b_glob),
        (axes[1], r'$\gamma$  spread across 1P scan',
         gamma_range_t, gamma_range_g, g_glob),
    ]:
        ax.bar(x - width/2, t_arr, width, color='k', label='Truth')
        ax.bar(x + width/2, g_arr, width, color='tab:orange', label='BIND2')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)
        # annotate where there is no scan
        for j in range(N_PARAMS):
            if n_nodes[j] < 3:
                ax.text(j + 1, 0, '·', ha='center', va='bottom',
                        color='gray', fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [PARAM_LABELS.get(j, f'p{j}') for j in x],
        rotation=45, ha='right', fontsize=8,
    )
    axes[0].text(0.01, 0.96, rf'global truth fit: '
                 rf'$\beta={b_glob:+.2f}$, $\gamma={g_glob:+.2f}$',
                 transform=axes[0].transAxes, va='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white',
                           ec='gray', alpha=0.85), fontsize=9)
    fig.suptitle('Universality test of the closure relation: per-parameter '
                 'spread of fitted β and γ across each 1P scan',
                 y=1.0)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj2_fig4_alpha_gamma_vs_feedback.{ext}')
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 5 — power-spectrum suppression P_hydro(k)/P_DMO(k) vs A_AGN1

def _stack_pk(profile_2d_per_halo_idx_unused, fields_dict, mass_norm):
    raise NotImplementedError  # placeholder, see plot_power_suppression


P5_AXES = [
    (1, r'$\Omega_m$',    False),
    (2, r'$\sigma_8$',    False),
    (3, r'$A_{\rm SN1}$',  True),
    (4, r'$A_{\rm AGN1}$', True),
    (5, r'$A_{\rm SN2}$',  True),
    (6, r'$A_{\rm AGN2}$', True),
]


def _pk_for_sim(sim_dir, snap, mass_tag, model_name, box_size, n_pix_full,
                centers_to_pixels_fn, extract_patch_fn):
    """Average 2D power spectrum of total baryons (truth & BIND2) and DMO,
    averaged over halo patches in this sim."""
    rec_full = sim_dir / snap / 'full_maps.npz'
    rec_cat  = sim_dir / snap / mass_tag / 'halo_catalog.npz'
    rec_gen  = sim_dir / snap / mass_tag / model_name / 'generated_halos.npz'
    if not (rec_full.exists() and rec_cat.exists() and rec_gen.exists()):
        return None
    fm = np.load(rec_full)
    cat = np.load(rec_cat)
    gen = np.load(rec_gen)['generated']
    centers_pix = centers_to_pixels_fn(cat['centers'])
    truth_maps = fm['truth_maps']
    dmo_field = fm['dmo_fullbox']
    pk_t_acc = pk_g_acc = pk_dmo_acc = None
    n = 0
    box_patch = box_size * 128 / n_pix_full
    for i, (cx, cy) in enumerate(centers_pix):
        tot_t = sum(extract_patch_fn(truth_maps[c], cx, cy) for c in range(3))
        tot_g = gen[i].sum(axis=0)
        patch_dmo = extract_patch_fn(dmo_field, cx, cy)
        for arr, key in [(tot_t, 't'), (tot_g, 'g'), (patch_dmo, 'dmo')]:
            k_c, pk = power_spectrum_2d(arr - arr.mean(), box_size=box_patch)
            if key == 't':
                pk_t_acc = pk if pk_t_acc is None else pk_t_acc + pk
            elif key == 'g':
                pk_g_acc = pk if pk_g_acc is None else pk_g_acc + pk
            else:
                pk_dmo_acc = pk if pk_dmo_acc is None else pk_dmo_acc + pk
        n += 1
    if n == 0:
        return None
    return k_c, pk_t_acc / n, pk_g_acc / n, pk_dmo_acc / n


def plot_power_suppression(p2):
    """Patch-scale P_hydro(k)/P_DMO(k) along *each* of the six primary 1P
    axes (Ω_m, σ_8, A_SN1, A_SN2, A_AGN1, A_AGN2). Two columns: Truth and
    BIND2. Each row is one parameter. Curves coloured by parameter value."""
    from project1_morphology import (
        SUITE_ROOT, SNAP, MASS_TAG, MODEL_NAME, BOX_SIZE, N_PIX_FULL,
        centers_to_pixels, extract_patch,
    )
    sim_id = p2['sim_id']
    params = p2['params']

    # cache (sid → (k, pk_t, pk_g, pk_dmo))
    pk_cache = {}

    fig, axes = plt.subplots(len(P5_AXES), 2, figsize=(11, 2.4 * len(P5_AXES)),
                             sharey=True, sharex=True)

    for ri, (j, plbl, use_log) in enumerate(P5_AXES):
        sids = sorted(set(s for s in sim_id
                          if s.startswith(f'1P_p{j}_') or s == '1P_p1_0'))
        if not sids:
            for col in range(2):
                axes[ri, col].axis('off')
            continue
        sim_to_pval = {}
        for sid in sids:
            mask = sim_id == sid
            sim_to_pval[sid] = float(np.median(params[mask, j - 1]))
        # colour mapping
        vals = np.array(list(sim_to_pval.values()))
        if use_log:
            norm = plt.Normalize(np.log10(vals.min()), np.log10(vals.max()))
            transform = lambda v: np.log10(v)
        else:
            norm = plt.Normalize(vals.min(), vals.max())
            transform = lambda v: v
        cmap = plt.get_cmap('plasma')

        for sid in sids:
            if sid not in pk_cache:
                pk = _pk_for_sim(SUITE_ROOT / '1P' / sid, SNAP, MASS_TAG,
                                 MODEL_NAME, BOX_SIZE, N_PIX_FULL,
                                 centers_to_pixels, extract_patch)
                if pk is None:
                    continue
                pk_cache[sid] = pk
            k, pk_t, pk_g, pk_dmo = pk_cache[sid]
            with np.errstate(divide='ignore', invalid='ignore'):
                r_t = pk_t / pk_dmo
                r_g = pk_g / pk_dmo
            ok = (k > 0) & np.isfinite(r_t)
            col = cmap(norm(transform(sim_to_pval[sid])))
            axes[ri, 0].plot(k[ok], r_t[ok], color=col, lw=1.2)
            axes[ri, 1].plot(k[ok], r_g[ok], color=col, lw=1.2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes[ri, :], shrink=0.92, pad=0.01)
        cb.set_label(rf'{plbl}' + (r' (log)' if use_log else ''),
                     fontsize=9)
        for ax in axes[ri]:
            ax.axhline(1.0, color='gray', lw=0.5, ls=':')
            ax.set_xscale('log')
            ax.set_ylim(0.4, 1.4)
            ax.grid(alpha=0.3)
        axes[ri, 0].set_ylabel(rf'{plbl}' + '\n' + r'$P_{\rm hydro}/P_{\rm DMO}$',
                               fontsize=9)

    axes[0, 0].set_title('Truth')
    axes[0, 1].set_title('BIND2')
    for ax in axes[-1]:
        ax.set_xlabel(r'$k$  [$h$/Mpc]')

    fig.suptitle(r'Patch-scale power suppression along all six primary 1P axes',
                 y=1.0)
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj2_fig5_power_suppression.{ext}')
    plt.close(fig)


# ----------------------------------------------------------------------
# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--skip-fig5', action='store_true',
                        help='skip the slow per-patch P(k) figure')
    args = parser.parse_args()

    p2 = build_p2_cache(rebuild=args.rebuild)
    print(f'Total halos: {len(p2["logM"])}')

    print('plot 1: surface-density profiles')
    plot_sigma_profiles(p2)
    print('plot 2: cumulative baryon fraction')
    plot_fb_cumulative(p2)
    print('plot 3: closure-radius universal relation')
    plot_closure_relation(p2)
    print('plot 4: fitted relation parameters vs feedback')
    plot_alpha_gamma_vs_feedback(p2)
    if not args.skip_fig5:
        print('plot 5: power-spectrum suppression (slow — re-reads patches)')
        plot_power_suppression(p2)
    print(f'Figures written under {FIG_DIR}/proj2_fig*.pdf')


if __name__ == '__main__':
    main()
