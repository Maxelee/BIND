#!/usr/bin/env python3
"""F4-shapes referee figures (paper cache for the fiducial model, M200c >= 1e13).

1. shape_1_axisratio.pdf  (REPLACES the paper figure)
   rows = DM/Gas/Stars, columns = CV/1P/SB35. Truth filled, BIND heavy step
   outline, and the DMO grey dashed histogram in ALL NINE panels: the DMO map
   is one field, and its q distribution is the identity baseline for every
   channel -- the Gas and Stars rows show those shapes are NOT inherited.
   Median lines (truth + BIND) as in the current figure.

2. fig_shape_perhalo.pdf  (NEW)
   2 rows x 3 channel columns, CV suite (>= 1e13).
   Top: per-halo q_gen vs q_true 2D density, 1:1 line, Pearson r annotated
        (DM panel also shows r for q_dmo vs q_true).
   Bottom: major-axis misalignment |dphi| histogram per channel (phi from
        e1,e2 = eps cos2phi, eps sin2phi), random-orientation null (flat
        1/90 per deg), median |dphi| and mean cos 2dphi annotated.

Everything is read from the single cache artifact shapes.pkl (per-halo arrays
truth/gen/dmo q,e1,e2 built row-aligned in one pass -- no cross-builder join).

Model selection is entirely via the paper_config env vars, e.g.:
    export PAPER_SUITE_ROOT=/mnt/home/mlee1/ceph/fm_testsuite
    export PAPER_MODEL_SUBDIR=fm_two_head
    export PAPER_MASS_DIR=mass_threshold_1p000e13
    export PAPER_MODEL_TAG=fm_two_head
    python f4_shapes.py
"""
import sys
sys.path.insert(0, '/mnt/home/mlee1/vdm_bind2/tools/paper_cache')
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as sps

import paper_config as C

# ── style: identical to the paper notebooks (examples/_build_paper_nbs.py) ──
try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'notebook'])
except Exception:
    pass

FIG_DIR = Path('/mnt/home/mlee1/vdm_bind2/examples/paper_figures')
FIG_DIR.mkdir(exist_ok=True)


def save_fig(fig, name, ext=('pdf', 'png')):
    for e in ext:
        fig.savefig(FIG_DIR / f'{name}.{e}', dpi=300, bbox_inches='tight')
    print('  saved', name)


SUITE_COLORS = C.SUITE_COLORS
SUITE_DISPLAY = C.SUITE_DISPLAY
DMO_COLOR = '0.35'

# ── load the spine shapes artifact ──────────────────────────────────────────
sh = pickle.load(open(C.CACHE_DIR / 'shapes.pkl', 'rb'))
A = sh['arrays']
suite = np.asarray(A['suite'])
masses = np.asarray(A['masses'])
log_m = np.log10(masses)
assert log_m.min() >= 13.0 - 1e-9, 'spine shapes.pkl should already be >=1e13'
print(f'shapes.pkl: {len(masses)} halos, '
      f'{dict(zip(*np.unique(suite, return_counts=True)))}, '
      f'logM in [{log_m.min():.2f}, {log_m.max():.2f}]')

# eps convention check: eps = hypot(e1,e2) = (1-q)/(1+q)  (e1,e2 = eps cos2phi, eps sin2phi)
for key in ('dm', 'gas', 'star'):
    for src in ('truth', 'gen'):
        q = A[f'{src}_{key}_q']
        eps = np.hypot(A[f'{src}_{key}_e1'], A[f'{src}_{key}_e2'])
        assert np.allclose(eps, (1 - q) / (1 + q), atol=1e-8)
assert np.allclose(np.hypot(A['dmo_e1'], A['dmo_e2']),
                   (1 - A['dmo_q']) / (1 + A['dmo_q']), atol=1e-8)

CHANNELS = [('dm', 'DM'), ('gas', 'Gas'), ('star', 'Stars')]
CH_TITLE = {'dm': 'DM (hydro)', 'gas': 'Gas', 'star': 'Stars'}
SUITES = ('CV', '1P', 'Test')


def eps_of(prefix):
    return np.hypot(A[f'{prefix}_e1'], A[f'{prefix}_e2'])


def phi_of(prefix):
    """Major-axis position angle in radians, in (-pi/2, pi/2]."""
    return 0.5 * np.arctan2(A[f'{prefix}_e2'], A[f'{prefix}_e1'])


def wrap_dphi(dphi):
    """Wrap an angle difference into (-pi/2, pi/2] (position angles mod pi)."""
    return (dphi + np.pi / 2) % np.pi - np.pi / 2


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — shape_1_axisratio: 3 channel rows x 3 suite columns,
#            DMO grey dashed baseline in ALL nine panels.
# ════════════════════════════════════════════════════════════════════════════
bins = np.linspace(0, 1, 31)
fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.6), sharex=True, sharey='row',
                         gridspec_kw={'hspace': 0.10, 'wspace': 0.10})

row_ymax = np.zeros(3)
for row, (key, name) in enumerate(CHANNELS):
    for col, s in enumerate(SUITES):
        ax = axes[row, col]
        m = suite == s
        color = SUITE_COLORS[s]
        qt = A[f'truth_{key}_q'][m]
        qg = A[f'gen_{key}_q'][m]
        qd = A['dmo_q'][m]
        h1, _, _ = ax.hist(qt, bins=bins, density=True, histtype='stepfilled',
                           alpha=0.40, color=color)
        h2, _, _ = ax.hist(qg, bins=bins, density=True, histtype='step',
                           lw=2.2, color=color)
        h3, _, _ = ax.hist(qd, bins=bins, density=True, histtype='step',
                           lw=1.4, ls='--', color=DMO_COLOR)
        row_ymax[row] = max(row_ymax[row], h1.max(), h2.max(), h3.max())
        # median lines (truth + BIND), as in the current figure
        ax.axvline(np.median(qt), color=color, ls='-', lw=1.0, alpha=0.75)
        ax.axvline(np.median(qg), color=color, ls=':', lw=1.5)
        if row == 0:
            ax.set_title(SUITE_DISPLAY[s], fontsize=13)
        if col == 0:
            ax.set_ylabel(rf'$\rho_{{\rm {name}}}(q)$')
        if row == 2:
            ax.set_xlabel(r'$q$')
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))

for row in range(3):
    axes[row, 0].set_ylim(0, 1.09 * row_ymax[row])

# one legend, generic colors (per-panel colors follow the suite columns)
leg = [
    Patch(facecolor='0.6', alpha=0.45, label='Truth'),
    Line2D([0], [0], color='0.25', lw=2.2, label='BIND'),
    Line2D([0], [0], color=DMO_COLOR, lw=1.4, ls='--', label='DMO'),
    Line2D([0], [0], color='0.25', ls='-', lw=1.0, alpha=0.75, label='Truth median'),
    Line2D([0], [0], color='0.25', ls=':', lw=1.5, label='BIND median'),
]
axes[0, 0].legend(handles=leg, loc='upper left', fontsize=9.5, framealpha=0.9)
save_fig(fig, 'shape_1_axisratio')
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — fig_shape_perhalo: per-halo fidelity (CV suite, >= 1e13).
#   Top: q_gen vs q_true 2D density + 1:1 + Pearson r (DM adds r for DMO).
#   Bottom: |dphi| misalignment histogram + random-orientation null.
# ════════════════════════════════════════════════════════════════════════════
cv = suite == 'CV'
CV_COLOR = SUITE_COLORS['CV']

fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0),
                         gridspec_kw={'hspace': 0.28, 'wspace': 0.24})

perhalo_stats = {}
for col, (key, name) in enumerate(CHANNELS):
    qt = A[f'truth_{key}_q'][cv]
    qg = A[f'gen_{key}_q'][cv]
    qd = A['dmo_q'][cv]
    r_gen = sps.pearsonr(qt, qg)[0]
    r_dmo = sps.pearsonr(qt, qd)[0]

    # ── top: 2D density ────────────────────────────────────────────────────
    ax = axes[0, col]
    lo = min(np.quantile(qt, 0.002), np.quantile(qg, 0.002))
    lo = max(0.0, lo - 0.03)
    hb = ax.hexbin(qt, qg, gridsize=34, cmap='Greens', mincnt=1,
                   extent=(lo, 1, lo, 1), linewidths=0.15)
    ax.plot([lo, 1], [lo, 1], color='k', ls='--', lw=1.1, zorder=3)
    ax.set_xlim(lo, 1); ax.set_ylim(lo, 1)
    ax.set_aspect('equal')
    ax.set_title(CH_TITLE[key], fontsize=13)
    ax.set_xlabel(r'$q_{\rm truth}$')
    if col == 0:
        ax.set_ylabel(r'$q_{\rm BIND}$')
    txt = rf'$r = {r_gen:.2f}$'
    if key == 'dm':
        txt += '\n' + rf'$r_{{\rm DMO}} = {r_dmo:.2f}$'
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=11, bbox=dict(facecolor='white', alpha=0.8,
                                   edgecolor='none', pad=2.5))

    # ── bottom: major-axis misalignment ────────────────────────────────────
    dphi = wrap_dphi(phi_of(f'gen_{key}')[cv] - phi_of(f'truth_{key}')[cv])
    abs_dphi_deg = np.degrees(np.abs(dphi))
    med_dphi = np.median(abs_dphi_deg)
    mean_cos = np.mean(np.cos(2 * dphi))

    ax = axes[1, col]
    dbins = np.linspace(0, 90, 31)
    ax.hist(abs_dphi_deg, bins=dbins, density=True, histtype='stepfilled',
            alpha=0.40, color=CV_COLOR)
    ax.hist(abs_dphi_deg, bins=dbins, density=True, histtype='step',
            lw=2.0, color=CV_COLOR)
    ax.axhline(1.0 / 90.0, color='0.35', ls='--', lw=1.4)
    ax.axvline(med_dphi, color=CV_COLOR, ls=':', lw=1.5)
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel(r'$|\Delta\phi|$ [deg]')
    if col == 0:
        ax.set_ylabel(r'$p(|\Delta\phi|)$ [deg$^{-1}$]')
    ax.text(0.96, 0.95,
            rf'median $|\Delta\phi| = {med_dphi:.1f}^\circ$' + '\n'
            + rf'$\langle\cos 2\Delta\phi\rangle = {mean_cos:.2f}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=10.5,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2.5))
    if col == 2:
        ax.text(88, 1.0 / 90.0 * 1.25, 'random orientation',
                ha='right', va='bottom', fontsize=9, color='0.35')

    perhalo_stats[key] = dict(r_gen=r_gen, r_dmo=r_dmo,
                              med_dphi=med_dphi, mean_cos=mean_cos)

fig.suptitle(r'Per-halo shape fidelity — CV suite, $M_{200c}\geq 10^{13}\,'
             r'M_\odot/h$', fontsize=13, y=0.98)
save_fig(fig, 'fig_shape_perhalo')
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# Headline stats
# ════════════════════════════════════════════════════════════════════════════
print('\n================ STATS ================')
print('\n-- per channel x suite: median/std of q (truth, BIND, DMO) + KS p(truth vs BIND) --')
print(f"{'ch':6s} {'suite':5s} {'med_T':>6s} {'std_T':>6s} {'med_B':>6s} "
      f"{'std_B':>6s} {'med_DMO':>7s} {'std_DMO':>7s} {'KS_p':>9s} {'KS_D':>6s}")
for key, name in CHANNELS:
    for s in SUITES:
        m = suite == s
        qt = A[f'truth_{key}_q'][m]
        qg = A[f'gen_{key}_q'][m]
        qd = A['dmo_q'][m]
        ks = sps.ks_2samp(qt, qg)
        print(f'{key:6s} {SUITE_DISPLAY[s]:5s} {np.median(qt):6.3f} {qt.std():6.3f} '
              f'{np.median(qg):6.3f} {qg.std():6.3f} {np.median(qd):7.3f} '
              f'{qd.std():7.3f} {ks.pvalue:9.2e} {ks.statistic:6.3f}')

print('\n-- per-halo Pearson r + misalignment (CV and pooled all suites) --')
print(f"{'ch':6s} {'sel':7s} {'r(qT,qB)':>8s} {'r(qT,qDMO)':>10s} "
      f"{'med|dphi|':>9s} {'<cos2dphi>':>10s} {'N':>6s}")
for key, name in CHANNELS:
    for sel_name, m in (('CV', cv), ('pooled', np.ones(len(suite), bool))):
        qt = A[f'truth_{key}_q'][m]
        qg = A[f'gen_{key}_q'][m]
        qd = A['dmo_q'][m]
        r_gen = sps.pearsonr(qt, qg)[0]
        r_dmo = sps.pearsonr(qt, qd)[0]
        dphi = wrap_dphi(phi_of(f'gen_{key}')[m] - phi_of(f'truth_{key}')[m])
        print(f'{key:6s} {sel_name:7s} {r_gen:8.3f} {r_dmo:10.3f} '
              f'{np.median(np.degrees(np.abs(dphi))):9.1f} '
              f'{np.mean(np.cos(2*dphi)):10.3f} {m.sum():6d}')

print('\n-- DMO ellipticity baseline: median eps ratio DMO/truth per channel (pooled + per suite) --')
print(f"{'ch':6s} {'sel':7s} {'med_eps_T':>9s} {'med_eps_DMO':>11s} {'ratio D/T':>9s}")
for key, name in CHANNELS:
    for sel_name in ('pooled',) + SUITES:
        m = np.ones(len(suite), bool) if sel_name == 'pooled' else suite == sel_name
        et = eps_of(f'truth_{key}')[m]
        ed = eps_of('dmo')[m]
        lab = SUITE_DISPLAY.get(sel_name, sel_name)
        print(f'{key:6s} {lab:7s} {np.median(et):9.4f} {np.median(ed):11.4f} '
              f'{np.median(ed)/np.median(et):9.2f}')

print('\n-- BIND ellipticity ratio BIND/truth per channel (pooled) --')
for key, name in CHANNELS:
    et = eps_of(f'truth_{key}')
    eg = eps_of(f'gen_{key}')
    print(f'{key:6s} med_eps_T={np.median(et):.4f} med_eps_B={np.median(eg):.4f} '
          f'ratio B/T={np.median(eg)/np.median(et):.3f}')

print('\ndone.')
