"""Project 3: Chua+21 baryonic spherization, extended over CAMELS SB35.

Replicates the central findings of Chua et al. 2021 (MNRAS, arXiv:2109.00012)
on dark-matter halo spherization by galaxy formation, then extends them from
their nine hand-tuned feedback variants to the continuous 35-dimensional
CAMELS SB35 parameter grid using the BIND2 generative emulator.

Outputs (paper_figures/proj3_*):
    fig1_dq_vs_mass    — Δq, Δs vs M200c, truth (CAMELS hydro) and BIND2,
                          replicating Chua Fig 4 in 2D
    fig2_cubic_fit     — s_FP/s_DMO vs M200c with the Chua-Eq.5 cubic fit;
                          our (α,β,γ,δ) printed alongside Chua Table 4 values
    fig3_shape_vs_fstar — q, s vs m★/M200; reproduces the positive correlation
                          (Chua Fig 8 + Table 5)
    fig4_param_drivers  — 35-parameter Spearman heatmap of ⟨Δq⟩, ⟨Δs⟩, and
                          ⟨m★/M200⟩ across SB35 — the new SB35 result
    fig5_universality   — Δq vs m★/M200 from every SB35 sim, coloured by
                          parameter group; tests Chua's claim that the m★/M200
                          relation is parameter-invariant in 35 dimensions

Run:
    python project3_chua_extension.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from project1_morphology import (  # noqa: E402
    BASE_CACHE, CACHE_DIR, FIG_DIR, GROUP_COLORS, MPC_PER_PIX, N_PARAMS,
    PARAM_GROUP, PARAM_LABELS, build_shape_cache, _param_use_log,
)

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'figure.dpi': 110,
    'savefig.bbox': 'tight',
})

# Chua+21 Table 4: best-fit cubic for s_FP / s_DMO at r = 0.12 R200
CHUA_FIT = dict(alpha=1.50, beta=13.09, gamma=-0.127, delta=-0.026)


# ----------------------------------------------------------------------
# data assembly

def load():
    base = np.load(BASE_CACHE, allow_pickle=False)
    shapes = build_shape_cache(rebuild=False)

    if (len(base['logM']) != len(shapes['r200c_mpch'])
            or not np.array_equal(base['suite'], shapes['suite'])
            or not np.array_equal(base['sim_id'], shapes['sim_id'])):
        raise RuntimeError(
            'Row order mismatch between halo_features and proj1_shapes. '
            'Re-run analysis_physics.ipynb section 2 to refresh the base cache.'
        )

    out = {k: base[k] for k in ('suite', 'sim_id', 'logM', 'params')}

    # Δshape against the DMO input projection. We compute both at the inner
    # aperture (a0 = 0.5 R200c) — Chua's region of largest baryonic effect —
    # and the outer aperture (a1 = R200c) where the effect should vanish.
    for ai in (0, 1):
        for ax in ('q',):
            for kind in ('truth', 'gen'):
                key = f'd{ax}_{kind}_a{ai}'
                out[key] = shapes[f'{kind}_DM_{ax}_a{ai}'] - shapes[f'dmo_DM_{ax}_a{ai}']
        for ax in ('q',):
            out[f'{ax}_dmo_a{ai}']   = shapes[f'dmo_DM_{ax}_a{ai}']
            out[f'{ax}_truth_a{ai}'] = shapes[f'truth_DM_{ax}_a{ai}']
            out[f'{ax}_gen_a{ai}']   = shapes[f'gen_DM_{ax}_a{ai}']

    # We only have a 2D proxy for s = c/a. Treating the projected q as our
    # s-equivalent retains the relative-spherization signal (Δq is what
    # Chua reports anyway in Fig 7). We omit the s vs T decomposition.

    # Stellar-mass-fraction proxy. truth_star_sum_a1 is the integrated
    # stellar surface density within R200c; multiply by pixel area to get
    # mass [M_sun/h] (the maps are M_sun/h per (Mpc/h)^2). M200 from cat is
    # already M_sun/h. The result is dimensionless and ratio-comparable to
    # Chua's m_star/M200, modulo a 2D-projection geometric factor we absorb
    # into the figure axis label.
    pix_area = MPC_PER_PIX ** 2
    m200 = 10.0 ** out['logM']
    out['fstar_truth'] = (shapes['truth_star_sum_a1'] * pix_area) / m200
    out['fstar_gen']   = (shapes['gen_star_sum_a1']   * pix_area) / m200

    print(f'loaded {len(out["logM"])} halos across {len(np.unique(out["sim_id"]))} sims')
    return out


# ----------------------------------------------------------------------
# helpers

def _per_sim_median(d, value, suite='Test', mass_lo=13.0, mass_hi=14.5,
                    n_min=3):
    """Return DataFrame with per-sim median value and the sim's parameter row,
    filtered to the requested suite + mass range."""
    sel = (d['suite'] == suite) & (d['logM'] >= mass_lo) & (d['logM'] < mass_hi) \
        & np.isfinite(value)
    if sel.sum() < 30:
        return None
    df = pd.DataFrame({
        'sim_id': d['sim_id'][sel],
        'val':    value[sel],
    })
    agg = df.groupby('sim_id', as_index=False).agg(
        val=('val', 'median'),
        n=('val', 'count'),
    )
    agg = agg[agg['n'] >= n_min]
    if agg.empty:
        return None
    sim_to_params = {}
    suite_arr = d['suite']; sim_arr = d['sim_id']; params = d['params']
    for sid in agg['sim_id'].values:
        idx = np.argmax((suite_arr == suite) & (sim_arr == sid))
        sim_to_params[sid] = params[idx]
    P = np.vstack([sim_to_params[s] for s in agg['sim_id']])
    for j in range(N_PARAMS):
        agg[f'p{j+1}'] = P[:, j]
    return agg


def _spearman_array(value, d, mass_lo, mass_hi, suite='Test'):
    agg = _per_sim_median(d, value, suite=suite, mass_lo=mass_lo, mass_hi=mass_hi)
    if agg is None or len(agg) < 10:
        return np.full(N_PARAMS, np.nan)
    rho = np.full(N_PARAMS, np.nan)
    for j in range(N_PARAMS):
        x = agg[f'p{j+1}'].to_numpy().astype(float)
        if _param_use_log(j + 1) and np.all(x > 0):
            x = np.log10(x)
        if not np.isfinite(x).all() or x.std() < 1e-9:
            continue
        rho[j] = spearmanr(x, agg['val']).statistic
    return rho


# ----------------------------------------------------------------------
# Fig 1 — Δq, Δs vs M200c (Chua Fig 4 replication, 2D version)

def fig1_dq_vs_mass(d):
    """Two panels: Δq at inner aperture (0.5 R200c) and at outer aperture
    (R200c). Truth (solid black) and BIND2 (dashed orange). Chua's claim
    is that ⟨Δq⟩ peaks at M ~ 2e12, then declines at the virial radius.
    Our mass cut starts at 1e13 (the L50n512 catalog floor) so we test the
    high-mass side of the Chua peak."""
    bins = np.linspace(13.0, 14.5, 8)
    cen = 0.5 * (bins[:-1] + bins[1:])
    sb = d['suite'] == 'Test'

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, ai, title in zip(axes, (0, 1),
                             (r'$r=0.5\,R_{200c}$ (inner)',
                              r'$r=R_{200c}$ (outer)')):
        for kind, color, ls, lbl in [
            ('truth', 'k', '-', 'CAMELS hydro'),
            ('gen', 'tab:orange', '--', 'BIND2'),
        ]:
            v = d[f'dq_{kind}_a{ai}']
            med = np.full_like(cen, np.nan)
            lo = np.full_like(cen, np.nan)
            hi = np.full_like(cen, np.nan)
            for k in range(len(cen)):
                m = sb & (d['logM'] >= bins[k]) & (d['logM'] < bins[k + 1]) \
                    & np.isfinite(v)
                if m.sum() < 15:
                    continue
                vv = v[m]
                med[k] = np.median(vv)
                lo[k]  = np.quantile(vv, 0.16)
                hi[k]  = np.quantile(vv, 0.84)
            if kind == 'truth':
                ax.fill_between(cen, lo, hi, color=color, alpha=0.12)
            ax.plot(cen, med, ls, color=color, lw=2.0, label=lbl)
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_xlabel(r'$\log_{10}\,M_{200c}$  [$M_\odot/h$]')
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', fontsize=9)
    axes[0].set_ylabel(r'$\Delta q \equiv q_{\rm FP}-q_{\rm DMO}$')

    fig.suptitle('Baryonic spherization of dark matter — '
                 'Chua+21 Fig 4 replication on CAMELS SB35 (2D)',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj3_fig1_dq_vs_mass.{ext}')
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 2 — cubic fit s_FP / s_DMO vs M200 (Chua Eq. 5 replication)

def _chua_cubic(logM, alpha, beta, gamma, delta):
    return alpha + gamma * (logM - beta) ** 2 + delta * (logM - beta) ** 3


def fig2_cubic_fit(d):
    """Replicates Chua Eq. 5 / Fig 5: scatter of (M200, q_FP/q_DMO) and
    the cubic fit. We use q in place of s (we don't have 3D c/a). The
    Chua fit is overlaid as a reference; their (α,β,γ,δ) was tuned to TNG50
    + TNG100 in 3D and is not expected to match in absolute amplitude in
    2D, but the location and shape of the peak should agree."""
    sb = d['suite'] == 'Test'
    ai = 0
    valid = sb & np.isfinite(d['q_dmo_a0']) & (d['q_dmo_a0'] > 0.05)

    ratio_t = d['q_truth_a0'][valid] / d['q_dmo_a0'][valid]
    ratio_g = d['q_gen_a0'][valid]   / d['q_dmo_a0'][valid]
    logM = d['logM'][valid]

    # cubic fit (use truth, weighted uniformly per halo)
    p0 = (1.0, 12.0, -0.05, 0.0)
    popt_t, _ = curve_fit(_chua_cubic, logM, ratio_t, p0=p0, maxfev=20000)
    popt_g, _ = curve_fit(_chua_cubic, logM, ratio_g, p0=p0, maxfev=20000)

    bins = np.linspace(13.0, 14.5, 12)
    cen = 0.5 * (bins[:-1] + bins[1:])
    med_t = np.array([np.median(ratio_t[(logM >= bins[k]) & (logM < bins[k + 1])])
                      if ((logM >= bins[k]) & (logM < bins[k + 1])).sum() >= 10
                      else np.nan for k in range(len(cen))])
    med_g = np.array([np.median(ratio_g[(logM >= bins[k]) & (logM < bins[k + 1])])
                      if ((logM >= bins[k]) & (logM < bins[k + 1])).sum() >= 10
                      else np.nan for k in range(len(cen))])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(logM, ratio_t, s=2, alpha=0.06, color='black')
    ax.plot(cen, med_t, 'ko-', lw=2, ms=6, label='CAMELS hydro median')
    ax.plot(cen, med_g, 'o--', color='tab:orange', lw=2, ms=6,
            label='BIND2 median')

    grid = np.linspace(13.0, 14.5, 80)
    ax.plot(grid, _chua_cubic(grid, *popt_t), 'k-', lw=1.5, alpha=0.6,
            label=f'cubic fit (truth):\n  α={popt_t[0]:.2f}, β={popt_t[1]:.2f}, '
                  f'γ={popt_t[2]:.3f}, δ={popt_t[3]:.3f}')
    ax.plot(grid, _chua_cubic(grid, *popt_g), '--', color='tab:orange',
            lw=1.5, alpha=0.8,
            label=f'cubic fit (BIND2):\n  α={popt_g[0]:.2f}, β={popt_g[1]:.2f}, '
                  f'γ={popt_g[2]:.3f}, δ={popt_g[3]:.3f}')
    ax.plot(grid, _chua_cubic(grid, **CHUA_FIT), ':', color='tab:purple',
            lw=2.0,
            label=f'Chua+21 Table 4 (3D): α={CHUA_FIT["alpha"]}, '
                  f'β={CHUA_FIT["beta"]}, γ={CHUA_FIT["gamma"]}, '
                  f'δ={CHUA_FIT["delta"]}')

    ax.axhline(1.0, color='gray', lw=0.6)
    ax.set_xlim(13.0, 14.5)
    ax.set_ylim(0.7, 1.6)
    ax.set_xlabel(r'$\log_{10}\,M_{200c}$  [$M_\odot/h$]')
    ax.set_ylabel(r'$q_{\rm FP}\,/\,q_{\rm DMO}$')
    ax.set_title('Cubic fit to the spherization ratio — Chua Eq. 5 form')
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj3_fig2_cubic_fit.{ext}')
    plt.close(fig)

    return popt_t, popt_g


# ----------------------------------------------------------------------
# Fig 3 — shape vs m★/M200 with Pearson r and Spearman ρ (Chua Fig 8 + Table 5)

def fig3_shape_vs_fstar(d):
    """Per-sim median q vs per-sim median m★/M200, both inner aperture.
    Chua finds Pearson r ≈ 0.3, Spearman ρ ≈ 0.35 across their model
    variants. We reproduce the test on the SB35 grid and report the
    coefficients for both the truth and the BIND2 prediction."""
    ai = 0
    sb = d['suite'] == 'Test'

    agg_t = _per_sim_median(d, d['q_truth_a0'])
    agg_g = _per_sim_median(d, d['q_gen_a0'])
    fagg_t = _per_sim_median(d, d['fstar_truth'])
    fagg_g = _per_sim_median(d, d['fstar_gen'])

    j_t = agg_t.merge(fagg_t.rename(columns={'val': 'fstar'})[['sim_id', 'fstar']],
                      on='sim_id')
    j_g = agg_g.merge(fagg_g.rename(columns={'val': 'fstar'})[['sim_id', 'fstar']],
                      on='sim_id')

    # also raw per-halo correlation (closer to Chua's per-halo Pearson r)
    raw_mask = sb & np.isfinite(d['q_truth_a0']) & np.isfinite(d['fstar_truth']) \
        & (d['fstar_truth'] > 0)
    raw_q = d['q_truth_a0'][raw_mask]
    raw_f = d['fstar_truth'][raw_mask]

    raw_pearson = pearsonr(raw_q, np.log10(raw_f)).statistic
    raw_spearman = spearmanr(raw_q, raw_f).statistic
    sim_pearson_t = pearsonr(j_t['val'], np.log10(j_t['fstar'])).statistic
    sim_spearman_t = spearmanr(j_t['val'], j_t['fstar']).statistic
    sim_pearson_g = pearsonr(j_g['val'], np.log10(j_g['fstar'])).statistic
    sim_spearman_g = spearmanr(j_g['val'], j_g['fstar']).statistic

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
    ax = axes[0]
    ax.scatter(np.log10(raw_f), raw_q, s=2, alpha=0.05, color='black',
               label='per-halo (truth)')
    bins = np.linspace(np.log10(raw_f).min(), np.log10(raw_f).max(), 8)
    cen = 0.5 * (bins[:-1] + bins[1:])
    med = np.array([np.median(raw_q[(np.log10(raw_f) >= bins[k])
                                    & (np.log10(raw_f) < bins[k + 1])])
                    for k in range(len(cen))])
    ax.plot(cen, med, 'ko-', lw=2, ms=6, label='median (truth)')
    ax.set_xlabel(r'$\log_{10}\,(m_\star/M_{200c})$  [proj.]')
    ax.set_ylabel(r'$q$  ($r=0.5\,R_{200c}$, truth)')
    ax.set_title('Per-halo: shape vs stellar fraction')
    txt = (f'per-halo: Pearson r = {raw_pearson:+.3f}\n'
           f'per-halo: Spearman ρ = {raw_spearman:+.3f}\n'
           f'(Chua Table 5 fiducial: r=0.26, ρ=0.34)')
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray'))
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right', fontsize=9)

    ax = axes[1]
    ax.scatter(np.log10(j_t['fstar']), j_t['val'], color='black', s=20,
               label='per-sim median (truth)')
    ax.scatter(np.log10(j_g['fstar']), j_g['val'], color='tab:orange', s=20,
               marker='x', label='per-sim median (BIND2)')
    ax.set_xlabel(r'$\log_{10}\,(m_\star/M_{200c})$  [per-sim median]')
    ax.set_ylabel(r'$q$  per-sim median')
    ax.set_title('Per-sim: SB35 grid — does the relation persist?')
    txt = (f'truth: Pearson r = {sim_pearson_t:+.3f},  '
           f'Spearman ρ = {sim_spearman_t:+.3f}\n'
           f'BIND2: Pearson r = {sim_pearson_g:+.3f},  '
           f'Spearman ρ = {sim_spearman_g:+.3f}')
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray'))
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle(r'Halo sphericity correlates with $m_\star/M_{200c}$ — '
                 r'Chua Fig 8 replication on SB35',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj3_fig3_shape_vs_fstar.{ext}')
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 4 — 35-parameter Spearman heatmap of ⟨Δq⟩, ⟨Δs⟩, ⟨m★/M200⟩

def fig4_param_drivers(d):
    """The new SB35 result. For each of the 35 parameters compute the
    per-sim Spearman ρ vs three observables: Δq (inner), Δq (outer),
    m★/M200. Show truth (filled bars) and BIND2 (open bars). The
    parameter ordering is by |ρ_truth| of Δq_inner — the Chua-headline
    observable."""

    rho_dqi_t = _spearman_array(d['dq_truth_a0'], d, 13.0, 14.5)
    rho_dqi_g = _spearman_array(d['dq_gen_a0'],   d, 13.0, 14.5)
    rho_dqo_t = _spearman_array(d['dq_truth_a1'], d, 13.0, 14.5)
    rho_dqo_g = _spearman_array(d['dq_gen_a1'],   d, 13.0, 14.5)
    rho_fs_t  = _spearman_array(d['fstar_truth'], d, 13.0, 14.5)
    rho_fs_g  = _spearman_array(d['fstar_gen'],   d, 13.0, 14.5)

    mag = np.where(np.isfinite(rho_dqi_t), np.abs(rho_dqi_t), -1.0)
    order = np.argsort(-mag)

    panels = [
        (r'$\Delta q$  (inner, $0.5\,R_{200c}$)', rho_dqi_t, rho_dqi_g),
        (r'$\Delta q$  (outer, $R_{200c}$)',      rho_dqo_t, rho_dqo_g),
        (r'$m_\star/M_{200c}$',                    rho_fs_t,  rho_fs_g),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 12), sharey=True)
    y = np.arange(N_PARAMS)
    h = 0.40
    for ax, (title, rho_t, rho_g) in zip(axes, panels):
        rho_t_s = rho_t[order]
        rho_g_s = rho_g[order]
        labels = [PARAM_LABELS[j + 1] for j in order]
        groups = [PARAM_GROUP[j + 1] for j in order]
        colors = [GROUP_COLORS[g] for g in groups]
        ax.barh(y - h/2, rho_t_s, h, color=colors, edgecolor='black', lw=0.5)
        ax.barh(y + h/2, rho_g_s, h, color='none', edgecolor=colors, lw=1.5)
        ax.axvline(0, color='gray', lw=0.6)
        ax.axvline(+0.1, color='black', lw=1.0, ls='--', alpha=0.6)
        ax.axvline(-0.1, color='black', lw=1.0, ls='--', alpha=0.6)
        ax.axvspan(-0.1, 0.1, color='gray', alpha=0.06, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        for tick, grp in zip(ax.get_yticklabels(), groups):
            tick.set_color(GROUP_COLORS[grp])
            tick.set_fontweight('bold')
        ax.invert_yaxis()
        ax.set_xlim(-0.6, 0.6)
        ax.set_xlabel(rf'Spearman $\rho$(p, {title})', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.25, axis='x')

    from matplotlib.patches import Patch
    grp_handles = [Patch(facecolor=GROUP_COLORS[g], edgecolor='black',
                         label=g)
                   for g in ['cosmo', 'SN', 'AGN', 'other']]
    style_handles = [
        Patch(facecolor='lightgray', edgecolor='black', label='Truth (filled)'),
        Patch(facecolor='none', edgecolor='black', lw=1.6,
              label='BIND2 (outline)'),
    ]
    axes[0].legend(handles=grp_handles, loc='lower right',
                   bbox_to_anchor=(1.0, 0.02), title='parameter group',
                   fontsize=9, title_fontsize=9, frameon=True)
    axes[2].legend(handles=style_handles, loc='lower right',
                   bbox_to_anchor=(1.0, 0.02), fontsize=9, frameon=True)

    fig.suptitle(r'Drivers of dark-matter halo spherization across the SB35 '
                 r'parameter space (per-sim Spearman, $\log M\!\in\![13,14.5)$)',
                 fontsize=13, y=1.00)
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj3_fig4_param_drivers.{ext}')
    plt.close(fig)

    return order, rho_dqi_t, rho_dqi_g


# ----------------------------------------------------------------------
# Fig 5 — universality: Δq vs m★/M200 across all SB35 sims, group-coloured

def fig5_universality(d):
    """Test of Chua's universality claim in 35 dimensions. If the m★/M200
    relation is parameter-invariant, sims from any region of parameter space
    should fall on a single curve. We plot per-sim median Δq vs per-sim
    median m★/M200 and colour each sim by its dominant parameter axis
    (the parameter with the largest deviation from the SB35 centre)."""
    agg_dq = _per_sim_median(d, d['dq_truth_a0'])
    agg_fs = _per_sim_median(d, d['fstar_truth'])
    # Both aggregates carry duplicate p1..p35 columns; drop them on the
    # right-hand side before merging so we don't get _x/_y suffixes.
    j = agg_dq.merge(
        agg_fs.rename(columns={'val': 'fstar'})[['sim_id', 'fstar']],
        on='sim_id',
    )

    P = j[[f'p{k+1}' for k in range(N_PARAMS)]].to_numpy().astype(float)
    Pn = P.copy()
    # Take log10 only when the parameter is strictly positive everywhere
    # (CAMELS' α_IMF is negative; logging it would inject NaN into the
    # z-score and make argmax pick that column for every sim).
    for k in range(N_PARAMS):
        if _param_use_log(k + 1) and np.all(Pn[:, k] > 0):
            Pn[:, k] = np.log10(Pn[:, k])
    std = Pn.std(axis=0)
    valid = std > 1e-9
    z = np.zeros_like(Pn)
    z[:, valid] = (Pn[:, valid] - Pn[:, valid].mean(axis=0)) / std[valid]
    dom_param = np.argmax(np.abs(z), axis=1)
    dom_group = np.array([PARAM_GROUP[k + 1] for k in dom_param])

    # also fit a single global trend
    x = np.log10(j['fstar'].to_numpy())
    y = j['val'].to_numpy()
    fit = np.polyfit(x, y, deg=1)
    rho_glob = spearmanr(x, y).statistic

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for grp in ('cosmo', 'SN', 'AGN', 'other'):
        m = dom_group == grp
        ax.scatter(x[m], y[m], color=GROUP_COLORS[grp], s=28,
                   alpha=0.85, edgecolor='black', lw=0.4,
                   label=f'{grp}-dominated  (n={m.sum()})')
    grid = np.linspace(x.min(), x.max(), 50)
    ax.plot(grid, np.polyval(fit, grid), 'k--', lw=1.6,
            label=f'global linear fit  (slope = {fit[0]:+.2f})')

    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel(r'$\log_{10}\,(m_\star/M_{200c})$  per-sim median  [proj.]')
    ax.set_ylabel(r'$\langle\Delta q\rangle$  per-sim median  '
                  r'($r=0.5\,R_{200c}$, truth)')
    ax.set_title(r'$\Delta q$–$m_\star/M_{200c}$ universality across the SB35 grid'
                 + '\n' + rf'global Spearman $\rho$ = {rho_glob:+.3f}',
                 fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc='best', fontsize=9, frameon=True)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj3_fig5_universality.{ext}')
    plt.close(fig)


# ----------------------------------------------------------------------
# main

def main():
    d = load()

    print('fig 1 — Δq vs M200c (Chua Fig 4)')
    fig1_dq_vs_mass(d)
    print('fig 2 — cubic fit s_FP/s_DMO (Chua Eq. 5)')
    popt_t, popt_g = fig2_cubic_fit(d)
    print(f'  truth fit: α={popt_t[0]:.3f} β={popt_t[1]:.3f} '
          f'γ={popt_t[2]:.3f} δ={popt_t[3]:.3f}')
    print(f'  BIND2 fit: α={popt_g[0]:.3f} β={popt_g[1]:.3f} '
          f'γ={popt_g[2]:.3f} δ={popt_g[3]:.3f}')
    print(f'  Chua T4 :  α={CHUA_FIT["alpha"]} β={CHUA_FIT["beta"]} '
          f'γ={CHUA_FIT["gamma"]} δ={CHUA_FIT["delta"]}')
    print('fig 3 — shape vs m★/M200 (Chua Fig 8 + Table 5)')
    fig3_shape_vs_fstar(d)
    print('fig 4 — 35-parameter Spearman drivers')
    order, rho_t, rho_g = fig4_param_drivers(d)
    print('  top-5 drivers of ⟨Δq⟩ (truth):')
    for j in order[:5]:
        nm = PARAM_LABELS[j + 1].replace('$', '').replace('\\', '')
        print(f'    p{j+1} {nm:<12} ρ_truth={rho_t[j]:+.3f}  '
              f'ρ_BIND2={rho_g[j]:+.3f}')
    print('fig 5 — Δq vs m★/M200 universality')
    fig5_universality(d)
    print(f'figures written under {FIG_DIR}/proj3_fig*.pdf')


if __name__ == '__main__':
    main()
