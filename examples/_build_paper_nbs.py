#!/usr/bin/env python3
"""Generate the four load-only BIND2 paper-figure notebooks from the parallel
cache (tools/paper_cache/). Every figure loads a precomputed artifact and plots;
no heavy compute, no model loads in the notebooks themselves.

    python _build_paper_nbs.py     # writes the 4 .ipynb into examples/

All analysis is restricted to the trained regime, M200c >= 1e13 (the training
cut): the shared SETUP defines `BINS` (mass bins at/above 13.0) and every
per-bin loop iterates over it. The 1e12-1e13 extrapolation regime and the
low-mass covering paint were dropped from the paper.

Notebooks:
  paper_figures2.ipynb     main spine: mass / profiles / field / P(k) / shapes
                           by mass bin (trained regime only)
  paper_fig_thermo.ipynb   thermodynamic validation (same model)
  paper_fig_redshift.ipynb redshift evolution + conditioning response
  paper_fig_models.ipynb   Appendix A VDM-vs-FM, Appendix B observable-conditioned
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = "/mnt/home/mlee1/vdm_bind2/examples"


def md(s):
    return new_markdown_cell(s)


def code(s):
    return new_code_cell(s)


# ════════════════════════════════════════════════════════════════════════════
# Shared setup (imported cache + style; identical across notebooks)
# ════════════════════════════════════════════════════════════════════════════
SETUP = r'''
import sys
sys.path.insert(0, '/mnt/home/mlee1/vdm_bind2/tools/paper_cache')
import os, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import paper_config as C

CACHE = C.CACHE_DIR
def L(name):
    p = CACHE / name
    if name.endswith('.pkl'):
        return pickle.load(open(p, 'rb'))
    return dict(np.load(p, allow_pickle=False))

FIG_DIR = Path('paper_figures'); FIG_DIR.mkdir(exist_ok=True)
def save_fig(fig, name, ext=('pdf', 'png')):
    for e in ext:
        fig.savefig(FIG_DIR / f'{name}.{e}', dpi=300, bbox_inches='tight')
    print('  saved', name)

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'notebook'])
except Exception:
    pass
# plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'mathtext.fontset': 'cm',
#                      'figure.dpi': 110, 'savefig.dpi': 300, 'axes.grid': False})

SUITE_COLORS = C.SUITE_COLORS; SUITE_DISPLAY = C.SUITE_DISPLAY
MASS_CH = C.MASS_CHANNELS; CH_DISPLAY = C.CH_DISPLAY
BIN_LABELS = C.MASS_BIN_LABELS; N_BINS = C.N_MASS_BINS
PARAM_LABELS = C.PARAM_LABELS
# Trained-regime restriction: the paper only uses halos with M200c >= 1e13 (the
# training cut). Lower bins exist in the cache but are never plotted.
MIN_LOG_M200 = 13.0
BINS = [b for b in range(N_BINS) if C.MASS_EDGES[b] >= MIN_LOG_M200]
BIN_CMAP = np.zeros((N_BINS, 4))
BIN_CMAP[BINS] = plt.cm.viridis(np.linspace(0.1, 0.9, len(BINS)))
print('Cache :', CACHE)
print('Model :', C.MODEL_TAG, '| suites', {s: 0 for s in C.SUITES})
print('Files :', sorted(p.name for p in CACHE.glob("*.pkl")) + sorted(p.name for p in CACHE.glob("*.npz")))
'''


# ════════════════════════════════════════════════════════════════════════════
# MAIN notebook — paper_figures2.ipynb
# ════════════════════════════════════════════════════════════════════════════
def main_notebook():
    cells = [
        md("# BIND2 Methods Paper — Figures (main)\n\n"
           "**One model**, mass + thermodynamics as a function of redshift and "
           "parameters (`fm_redshift` @ z=0), validated by **halo-mass bin** in the "
           "**trained regime** ($M_{200c}\\ge 10^{13}$, the training cut). Every "
           "figure loads a precomputed cache built by `tools/paper_cache/` — no "
           "compute here. Build the cache with `bash run_paper_cache.sh` (see the "
           "repo README of that dir).\n\n"
           "Sections: §1 showcase · §2 integrated mass + parameter response · §3 "
           "profiles + parameter response · §4 field-level (1P) · §5 power spectrum · "
           "§6 halo shapes + parameter response."),
        code(SETUP),

        md("## §1 · Full-box showcase (Fig 1)\n\nDMO → BIND2 mass fields vs hydro truth "
           "for one CV box. The BIND row shows the taper-free pasted patches (`hydro_canvas`)."),
        code(r'''
rec = C.discover_sims(('CV',))[0]
fm = C.load_full_maps(rec); comp = C.load_composite(rec)
dmo, truth = fm['dmo_fullbox'], fm['truth_maps']
canvas = comp['hydro_canvas']     # patches-on-black; drives the clean residual
composite = comp['composite']     # blended full-box map

def showcase(bind_field, tag):
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(16, 8), sharex=True, sharey=True, gridspec_kw={'wspace': 0.005, 'hspace': 0.05})
    def _im(ax, img, title=None, vmax=None):
        pos = img[img > 0]; vmax = vmax or np.quantile(pos, 0.999); vmin = max(np.quantile(pos, 0.05), 1e-8)
        ax.imshow(img, origin='lower', cmap='magma', norm=SymLogNorm(linthresh=max(vmin,1e-6), vmin=0, vmax=vmax, base=10), extent=[0, 50, 0, 50])
        if title: ax.set_title(title)

    vmax = []
    for c in range(3):
        pos = np.concatenate([truth[c].ravel(), bind_field[c].ravel()]); vmax.append(np.quantile(pos[pos>0], 0.999))
    _im(axs[0,0], dmo, 'DMO (Nbody)')
    for c in range(3): _im(axs[0,c+1], truth[c], CH_DISPLAY[MASS_CH[c]], vmax[c])
    _im(axs[1,0], dmo)
    for c in range(3): _im(axs[1,c+1], bind_field[c], vmax=vmax[c])

    # X label only on bottom row
    for ax in axs[1]: ax.set_xlabel('X [Mpc/h]')
    # Y label only on leftmost column
    axs[0,0].set_ylabel('Y [Mpc/h]'); axs[1,0].set_ylabel('Y [Mpc/h]')

    # Row super-labels
    fig.text(0.01, 0.70, 'Truth', va='center', ha='center', fontsize=20, fontweight='bold', rotation='vertical')
    fig.text(0.01, 0.30, 'BIND',  va='center', ha='center', fontsize=20, fontweight='bold', rotation='vertical')

    fig.subplots_adjust(left=0.06)
    plt.tight_layout()
    save_fig(fig, tag); plt.show()

showcase(canvas, 'fig1_showcase')                # v1: BIND row = pasted patches (matches original)
'''),

        md("## §2 · Integrated mass + parameter response\n\n"
           "Fig 2 consolidated gen-vs-truth scatter + mass distribution **per mass "
           "bin** (R200 aperture); KS/median-bias table; Fig 3a parameter → mass "
           "Spearman (`mass_param.pkl`); Fig 3a-bis dominant correlations vs bin."),
        code(r'''
tbl = L('mass_table.pkl')
tbl = tbl[tbl.mass_bin.isin(BINS)]   # trained regime only (>=1e13)
# Fig 2 — consolidated: gen vs truth scatter (left) + mass distribution (right), per mass bin, R200 aperture

# Pre-compute global x range for scatter (left col) across all bins and channels
_all_sc = []
for b in BINS:
    sub = tbl[tbl.mass_bin == b]
    for ch in MASS_CH:
        t = sub[f'truth_{ch}_rvir']; g = sub[f'gen_{ch}_rvir']; m = (t > 0) & (g > 0)
        if m.sum() > 0: _all_sc.extend([np.log10(t[m]).values, np.log10(g[m]).values])
sc_lo = np.concatenate(_all_sc).min() - 0.15
sc_hi = np.concatenate(_all_sc).max() + 0.15

fig, axes = plt.subplots(len(BINS), 2, figsize=(10, 3*len(BINS)), constrained_layout=True,
                         sharex='col')

for row, b in enumerate(BINS):
    sub = tbl[tbl.mass_bin == b]
    ax_sc, ax_hi = axes[row, 0], axes[row, 1]

    for c, ch in enumerate(MASS_CH):
        t = sub[f'truth_{ch}_rvir']; g = sub[f'gen_{ch}_rvir']
        m = (t > 0) & (g > 0); xt = np.log10(t[m]); xg = np.log10(g[m])
        ax_sc.scatter(xt, xg, s=6, alpha=0.15, color=f'C{c}', rasterized=True,
                      label=CH_DISPLAY[ch] if row == 0 else None)
        # bin medians with errorbars
        bins_s = np.linspace(xt.min(), xt.max(), 12)
        bc = 0.5 * (bins_s[:-1] + bins_s[1:]); idx = np.digitize(xt, bins_s) - 1
        med = [np.median(xg[idx==i]) if (idx==i).sum() > 3 else np.nan for i in range(len(bc))]
        sd  = [np.std(xg[idx==i])    if (idx==i).sum() > 3 else np.nan for i in range(len(bc))]
        ax_sc.errorbar(bc, med, yerr=sd, fmt='o', ms=4, color=f'C{c}',
                       mec='k', ecolor='k', capsize=2, zorder=5, lw=0.8)
        bins_h = np.linspace(min(xt.min(), xg.min()), max(xt.max(), xg.max()), 25)
        ax_hi.hist(xt, bins=bins_h, density=True, histtype='step', lw=1.5, color=f'C{c}')
        ax_hi.hist(xg, bins=bins_h, density=True, histtype='stepfilled', alpha=0.35, color=f'C{c}')

    ax_sc.plot([sc_lo, sc_hi], [sc_lo, sc_hi], 'k--', lw=0.8)
    ax_sc.set_ylim(sc_lo, sc_hi)   # y per-panel; x shared via sharex='col'

    ax_sc.set_ylabel(BIN_LABELS[b] + '\n' + r'$\log_{10} M_{\rm BIND}$')
    ax_hi.set_ylabel(r'$\rho$')
    if row < len(BINS) - 1:
        ax_sc.tick_params(labelbottom=False); ax_hi.tick_params(labelbottom=False)
    else:
        ax_sc.set_xlabel(r'$\log_{10} M_{\rm truth}\ [r \leq R_{200}]$')
        ax_hi.set_xlabel(r'$\log_{10} M\ [r \leq R_{200}]$')

axes[0, 0].set_xlim(sc_lo, sc_hi)   # set once; propagates to all left panels via sharex
axes[0, 0].set_title(r'$M_{\rm BIND}$ vs $M_{\rm truth}$')
axes[0, 1].set_title('Mass distribution')
axes[0, 0].legend(fontsize=12)
hist_handles = [Line2D([0],[0], color='gray', lw=1.5, label='Truth'),
                Patch(facecolor='gray', alpha=0.35, label='BIND')]
axes[0, 1].legend(handles=hist_handles, fontsize=12)

save_fig(fig, 'fig2_mass_by_bin'); plt.show()
'''),
        code(r'''
# Quantitative distribution comparison: KS test + median log-bias, per channel × mass bin
from scipy.stats import ks_2samp

ks_stat  = np.full((len(BINS), len(MASS_CH)), np.nan)
ks_pval  = np.full((len(BINS), len(MASS_CH)), np.nan)
log_bias = np.full((len(BINS), len(MASS_CH)), np.nan)   # median(log M_BIND - log M_truth)

for row, b in enumerate(BINS):
    sub = tbl[tbl.mass_bin == b]
    for c, ch in enumerate(MASS_CH):
        t = sub[f'truth_{ch}_rvir']; g = sub[f'gen_{ch}_rvir']
        m = (t > 0) & (g > 0)
        if m.sum() < 10: continue
        xt, xg = np.log10(t[m].values), np.log10(g[m].values)
        st, pv = ks_2samp(xt, xg)
        ks_stat[row, c]  = st
        ks_pval[row, c]  = pv
        log_bias[row, c] = np.median(xg - xt)

ch_labels = [CH_DISPLAY[ch] for ch in MASS_CH]
bin_labels = [BIN_LABELS[b] for b in BINS]
fig, axes = plt.subplots(1, 2, figsize=(10, 0.6*len(BINS) + 2.5), constrained_layout=True)

for ax, data, title, fmt, cmap, vmin, vmax in [
    (axes[0], log_bias, r'Median $\log_{10}(M_{\rm BIND}/M_{\rm truth})$',
     '{:+.3f}', 'RdBu_r', -0.3, 0.3),
    (axes[1], np.log10(np.clip(ks_pval, 1e-10, 1)),
     r'KS $\log_{10}(p)$  [dashed = $p=0.05$]',
     '{:.1f}', 'RdYlGn', -6, 0),
]:
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    for row in range(len(BINS)):
        for c in range(len(MASS_CH)):
            v = data[row, c]
            if np.isfinite(v):
                ax.text(c, row, fmt.format(v), ha='center', va='center', fontsize=9,
                        color='white' if abs(v) > 0.6*(vmax-vmin)/2 + vmin else 'k')
    ax.set_xticks(range(len(MASS_CH))); ax.set_xticklabels(ch_labels)
    ax.set_yticks(range(len(BINS))); ax.set_yticklabels(bin_labels)
    ax.set_ylabel(r'$\log_{10} M_{200c}$ bin'); ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)

# mark KS significance threshold (p=0.05 → log p ≈ -1.3)
axes[1].axvline(-1, color='k', ls='--', lw=0.5)   # visual guide only; significance is per-cell

save_fig(fig, 'fig2_mass_ks_bias'); plt.show()

print("\nMedian log-bias  (rows=bins, cols=channels):")
print(pd.DataFrame(log_bias.round(3), index=bin_labels, columns=ch_labels).to_string())
print("\nKS p-value:")
print(pd.DataFrame(ks_pval.round(4), index=bin_labels, columns=ch_labels).to_string())
'''),
        code(r'''
# Fig 3a — parameter -> mass Spearman (True / BIND / residual)
mp = L('mass_param.pkl'); rho = mp['rho_mass']['trained']; chans = mp['channels']
astro = [j for j in range(C.N_PARAMS) if (j+1) not in C.COSMO_PARAM_IDX]
order = sorted(astro)   # all astro params, in index order (matches the original layout)
xl = [PARAM_LABELS[j+1] for j in order]
fig, axes = plt.subplots(len(chans), 1, figsize=(13, 1.7*len(chans)), sharex=True)
for ci,(ch,ax) in enumerate(zip(chans, axes)):
    d = np.vstack([rho['True'][ci,order], rho['BIND'][ci,order], rho['True'][ci,order]-rho['BIND'][ci,order]])
    im = ax.imshow(d, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    for r in range(3):
        for k in range(len(order)):
            if np.isfinite(d[r,k]): ax.text(k,r,f'{d[r,k]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(d[r,k])>0.4 else 'k')
    ax.axhline(1.5, color='k', lw=1, ls='--'); ax.set_yticks([0,1,2]); ax.set_yticklabels(['True','BIND',r'$\Delta$']); ax.set_ylabel(CH_DISPLAY.get(ch,ch))
axes[-1].set_xticks(range(len(order))); axes[-1].set_xticklabels(xl, rotation=45, ha='right')
fig.colorbar(im, ax=axes, fraction=0.12).set_label(r'$\rho_S$')
save_fig(fig, 'fig3a_spearman_param_mass'); plt.show()
'''),
        code(r'''
# Fig 3a-bis — how dominant parameter correlations shift with mass bin
from scipy.stats import spearmanr

# Merge per-sim params from mass_param.pkl onto tbl (keyed by suite + sim_id)
# sim_table has columns p1..p35; rename to param_0..param_34
if 'param_0' not in tbl.columns:
    _mp = L('mass_param.pkl')
    _st = _mp['sim_table'].copy()
    _st = _st.rename(columns={f'p{j+1}': f'param_{j}' for j in range(C.N_PARAMS)})
    _key = ['suite', 'sim_id']
    tbl = tbl.merge(_st[_key + [f'param_{j}' for j in range(C.N_PARAMS)]], on=_key, how='left')

param_cols = sorted([c for c in tbl.columns if c.startswith('param_')],
                    key=lambda c: int(c.split('_')[1]))
if not param_cols:
    print("tbl has no param_* columns — check sim_table keys in mass_param.pkl")
else:
    astro = [j for j in range(C.N_PARAMS) if (j+1) not in C.COSMO_PARAM_IDX]

    # per-bin Spearman: truth mass & gen mass vs astro params
    rho_true = {ch: np.full((len(BINS), len(astro)), np.nan) for ch in MASS_CH}
    rho_bind = {ch: np.full((len(BINS), len(astro)), np.nan) for ch in MASS_CH}
    for row, b in enumerate(BINS):
        sub = tbl[tbl.mass_bin == b]
        for ch in MASS_CH:
            t = sub[f'truth_{ch}_rvir']; g = sub[f'gen_{ch}_rvir']
            m = (t > 0) & (g > 0); tv = t[m].values; gv = g[m].values
            for pi, j in enumerate(astro):
                pv = sub[param_cols[j]][m].values
                if pv.std() > 1e-10 and len(tv) > 10:
                    rho_true[ch][row, pi] = spearmanr(tv, pv)[0]
                    rho_bind[ch][row, pi] = spearmanr(gv, pv)[0]

    N_TOP = 6
    cmap_p = plt.cm.tab10
    fig, axes = plt.subplots(len(MASS_CH), 1, figsize=(8, 3.2*len(MASS_CH)), sharex=True)

    for ci, (ch, ax) in enumerate(zip(MASS_CH, axes)):
        # rank params by peak |rho| across bins in truth
        importance = np.nanmax(np.abs(rho_true[ch]), axis=0)
        top_idx = np.argsort(importance)[::-1][:N_TOP]
        for ki, pi in enumerate(top_idx):
            j = astro[pi]; lbl = PARAM_LABELS[j+1]; col = cmap_p(ki / N_TOP)
            ax.plot(range(len(BINS)), rho_true[ch][:, pi], 'o-',  color=col, lw=1.8, ms=5, label=lbl)
            ax.plot(range(len(BINS)), rho_bind[ch][:, pi], 'o--', color=col, lw=1.2, ms=4, alpha=0.6)
        ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
        ax.set_ylim(-1, 1); ax.set_ylabel(f'{CH_DISPLAY[ch]}\n' + r'$\rho_S$')
        ax.legend(fontsize=7, ncol=2, loc='best'); ax.grid(alpha=0.2)

    axes[-1].set_xticks(range(len(BINS))); axes[-1].set_xticklabels([BIN_LABELS[b] for b in BINS], rotation=20)
    axes[-1].set_xlabel(r'$\log_{10} M_{200c}$ bin')
    axes[0].set_title('Top parameter–mass Spearman by bin  (solid = Truth, dashed = BIND)')

    # shared legend for truth vs BIND style
    style_handles = [Line2D([0],[0], color='gray', lw=1.8, ls='-',  label='Truth'),
                     Line2D([0],[0], color='gray', lw=1.2, ls='--', label='BIND', alpha=0.6)]
    axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + style_handles,
                   labels=axes[0].get_legend_handles_labels()[1] + ['Truth','BIND'],
                   fontsize=7, ncol=3, loc='best')

    save_fig(fig, 'fig3a_param_mass_by_bin'); plt.show()
'''),

        md("## §3 · Radial profiles + parameter response\n\n"
           "Fig 4 profile fractional residual **by mass bin** (`profiles.pkl`); Fig 3b "
           "parameter → profile Spearman (`profiles_r200.pkl`, previously broken — now fixed)."),
        code(r'''
pr = L('profiles.pkl'); r = pr['r']
fig, axes = plt.subplots(len(MASS_CH), 1, figsize=(7.2, 3.1*len(MASS_CH)), sharex=True)
for c, ax in enumerate(axes):
    for b in BINS:
        if b not in pr['by_bin']: continue
        band = pr['by_bin'][b]
        ax.plot(r, band['pctdiff_med'][c], color=BIN_CMAP[b], lw=1.8, label=BIN_LABELS[b] if c==0 else None)
        ax.fill_between(r, band['pctdiff_p16'][c], band['pctdiff_p84'][c], color=BIN_CMAP[b], alpha=0.10)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.6); ax.set_xscale('log'); ax.set_ylim(-1,1)
    ax.set_ylabel(rf'$\Delta\Sigma_{{\rm {CH_DISPLAY[MASS_CH[c]]}}}/\Sigma$'); ax.grid(which='both', alpha=0.2)
    if c==0: ax.legend(title=r'$\log_{10}M_{200}$', fontsize=8, ncols=2)
axes[-1].set_xlabel(r'$r$ [Mpc$/h$]')
save_fig(fig, 'fig4_radial_pctdiff_by_bin'); plt.show()
'''),
        code(r'''
# Fig 3b — parameter -> radial profile Spearman (BIND & residual), FIXED
p2 = L('profiles_r200.pkl'); rr = p2['r_over_r200']; rho = p2['rho_prof']['trained']
astro = [j for j in range(C.N_PARAMS) if (j+1) not in C.COSMO_PARAM_IDX]
order = sorted(astro); xl=[PARAM_LABELS[j+1] for j in order]   # all astro params
r1 = np.argmin(np.abs(rr-1.0))
fig, axes = plt.subplots(len(MASS_CH), 2, figsize=(14, 3.0*len(MASS_CH)), sharex=True, sharey=True, gridspec_kw={'hspace':0.1, 'wspace':0.1})
for c in range(len(MASS_CH)):
    for si,(lab,data) in enumerate([('BIND', rho['BIND'][c][:,order]), ('True − BIND', (rho['True'][c]-rho['BIND'][c])[:,order])]):
        ax = axes[c,si]; im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5, origin='upper')
        ax.axhline(r1, color='k', ls='--', lw=1, alpha=0.7)
        if si==0: ax.set_ylabel(f'{CH_DISPLAY[MASS_CH[c]]}\n$r/R_{{200}}$')
        if c==0: ax.set_title(lab)
    axes[c,0].set_yticks([np.argmin(np.abs(rr-v)) for v in (0.1,0.5,1,2)]); axes[c,0].set_yticklabels(['0.1','0.5','1','2'])
for ax in axes[-1]: ax.set_xticks(range(len(order))); ax.set_xticklabels(xl, rotation=45, ha='right', fontsize=8)
fig.colorbar(im, ax=axes, fraction=0.12).set_label(r'$\rho_S$')
save_fig(fig, 'fig3b_spearman_param_profile'); plt.show()
'''),

        md("## §4 · Field-level response (1P set, Fig 6)\n\n"
           "For each feedback parameter, the log-ratio field between its high and low "
           "1P variant (most-massive halo) — truth vs BIND2 (`field1p.npz`)."),
        code(r'''
f1 = L('field1p.npz'); params = f1['params']
from scipy.stats import pearsonr
nch = 3; nrow = len(params)
fig, axes = plt.subplots(nrow, 2*nch, figsize=(2.0*2*nch, 2.0*nrow))
for ri, j in enumerate(params):
    t_hi,t_lo = f1[f'p{j}_t_hi'], f1[f'p{j}_t_lo']; g_hi,g_lo = f1[f'p{j}_g_hi'], f1[f'p{j}_g_lo']
    for c in range(nch):
        eps = 1e-3*np.percentile(np.r_[t_hi[c][t_hi[c]>0], t_lo[c][t_lo[c]>0]] if (t_hi[c]>0).any() else [1e-6], 99)
        tlr = np.log10((t_hi[c]+eps)/(t_lo[c]+eps)); glr = np.log10((g_hi[c]+eps)/(g_lo[c]+eps))
        v = np.quantile(np.abs(np.r_[tlr.ravel(), glr.ravel()]), 0.995) or 0.1
        for ki,(fld,kind) in enumerate([(tlr,'Truth'),(glr,'BIND')]):
            ax = axes[ri, c*2+ki]; ax.imshow(fld, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v); ax.set_xticks([]); ax.set_yticks([])
            if ri==0: ax.set_title(f'{CH_DISPLAY[MASS_CH[c]]}\n{kind}', fontsize=8)
            if c==0 and ki==0: ax.set_ylabel(PARAM_LABELS[int(j)], fontsize=9, rotation=0, ha='right', va='center', labelpad=22)
            if kind=='BIND':
                s_t,s_g = tlr.std(), glr.std()
                if s_t>1e-9 and s_g>1e-9:
                    ax.text(0.04,0.96,f'r={pearsonr(tlr.ravel(),glr.ravel())[0]:+.2f}\nA={s_g/s_t:.2f}', transform=ax.transAxes, va='top', fontsize=6, bbox=dict(fc='white', ec='none', alpha=0.7))
save_fig(fig, 'fig6_field_response_1p'); plt.show()
'''),

        md("## §5 · Full-box power spectrum (Fig 5)\n\n"
           "Total-matter $P(k)$ with the corrected **shared-content paste**, trained "
           "regime only: BIND ($M_{200c}\\ge 10^{13}$) vs truth vs DMO, plus the "
           "hydro-replaced control (truth patches of the **same** $\\ge 10^{13}$ halos "
           "pasted with the same aperture) that isolates model vs aperture error "
           "(`pk_fixed.npz`)."),
        code(r'''
# Diagnostic: inspect pk.npz to understand array shapes and k coverage
_pk = L('pk.npz')
k_diag = _pk['k']
print(f"k range: {k_diag.min():.3f} – {k_diag.max():.3f} h/Mpc  ({len(k_diag)} bins)")
print(f"knyq (1024px, 50 Mpc/h) = {np.pi*1024/50:.2f} h/Mpc")
print()
for s in ('CV', 'Test', '1P', 'SB35'):
    key = f'{s}_bind'
    if key not in _pk: continue
    arr = _pk[key]
    print(f"{s:6s}  bind shape={arr.shape}  finite={np.isfinite(arr).mean():.2f}  "
          f"range=[{np.nanmin(arr):.2e}, {np.nanmax(arr):.2e}]")
    # check for anomalies at high k
    hk = k_diag > 30
    if hk.any():
        t = _pk[f'{s}_truth']
        ratio = arr[:, hk] / t[:, hk]
        print(f"  k>30:  BIND/truth  median={np.nanmedian(ratio):.3f}  "
              f"p16={np.nanquantile(ratio,0.16):.3f}  p84={np.nanquantile(ratio,0.84):.3f}")
'''),
        code(r'''
# §5 · Full-box power spectrum — shared-content paste, trained regime (>=1e13) only.
# Reads pk_fixed.npz, built by tools/paper_cache/build_pk_fixed.py:
#     python build_pk_fixed.py --metric fixed --pool 6   # CPU: >=1e13 shared paste + hydro-replace
#     python build_pk_fixed.py --reduce                  # -> pk_fixed.npz

pkf = L('pk_fixed.npz'); k = pkf['k']
knyq = np.pi*1024/50.0
km = k <= knyq; k = k[km]                      # drop super-Nyquist bins
suites = [s for s in ('CV','Test','1P') if f'{s}_fixed_truth' in pkf]

def med(a):  return np.median(a[:, km], 0)
def band(num, den):
    r = num[:, km] / den[:, km]
    return np.median(r, 0), np.quantile(r, 0.16, 0), np.quantile(r, 0.84, 0)

fig, axes = plt.subplots(2, len(suites), figsize=(5.2*len(suites), 8), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1], 'hspace': 0}, squeeze=False)
for col, s in enumerate(suites):
    at, ab = axes[0, col], axes[1, col]
    tr, dm = pkf[f'{s}_fixed_truth'], pkf[f'{s}_fixed_dmo']
    bind = pkf[f'{s}_fixed_ge13']
    fin = np.isfinite(bind).all(1)             # sims lacking >=1e13 halos are NaN rows

    at.plot(k, med(tr)/med(dm), 'k', lw=1.4, label='Truth/DMO')
    at.plot(k, med(bind[fin])/med(dm[fin]), color='tab:orange', lw=1.4,
            label=r'BIND ($M_{200c}\geq 10^{13}$)/DMO')
    rm, rlo, rhi = band(bind[fin], tr[fin])
    ab.plot(k, rm, color='tab:orange', lw=1.3, label='BIND/Truth')
    ab.fill_between(k, rlo, rhi, color='tab:orange', alpha=0.16)

    # hydro-replaced control: truth patches of the SAME >=1e13 halos, same shared
    # paste (isolates model error from paste/aperture error)
    hr, tr_hr = pkf.get(f'{s}_fixed_hr_ge13'), tr
    if hr is None or not np.isfinite(hr).any():
        # cache predates hr_ge13 -> fall back to the legacy all-catalog-halo control
        hr, tr_hr = pkf.get(f'{s}_pkfile_hydro_replace'), pkf.get(f'{s}_pkfile_truth', tr)
        if hr is not None and col == 0:
            print('WARNING: pk_fixed.npz lacks hr_ge13 — showing the legacy (>=1e12) '
                  'hydro-replace; rerun build_pk_fixed.py --metric fixed then --reduce')
    if hr is not None and np.isfinite(hr).any():
        fh = np.isfinite(hr).all(1)
        at.plot(k, med(hr[fh])/med(dm[fh]), color='tab:blue', lw=1.2, label='Hydro-repl/DMO')
        ab.plot(k, np.median(hr[fh][:, km]/tr_hr[fh][:, km], 0), color='tab:blue', ls='-.',
                lw=1, label='Hydro-repl/Truth')
    ab.plot(k, np.median(dm[:, km]/tr[:, km], 0), 'gray', ls='--', lw=1, label='DMO/Truth')

    at.axhline(1, color='gray', lw=0.6, ls='--'); at.set_xscale('log')
    at.grid(which='both', alpha=0.25); at.set_title(SUITE_DISPLAY[s]); at.tick_params(labelbottom=False)
    ab.axhspan(0.8, 1.2, color='tab:green', alpha=0.08); ab.axhline(1, color='k', lw=0.6, ls='--')
    ab.set_ylim(0.5, 1.5); ab.set_xscale('log'); ab.grid(which='both', alpha=0.25)
    ab.set_xlabel(r'$k$ [$h$/Mpc]')
    if col == 0:
        at.set_ylabel(r'$P/P_{\rm DMO}$'); at.legend(fontsize=8)
        ab.set_ylabel(r'$P/P_{\rm Truth}$'); ab.legend(fontsize=7, loc='lower left')

save_fig(fig, 'fig5_total_field_pk_fixed'); plt.show()
'''),

        md("## §6 · Halo shapes + parameter response (Fig 7)\n\n"
           "Mass-weighted 2D axis-ratio $q$ distributions (truth vs BIND vs DMO) and the "
           "**shape parameter response** (`shapes.pkl`)."),
        code(r'''
# §6 intro — example halo patches + mass-weighted quadrupole ellipse illustration
# Shows one halo near each of 10^14.5, 10^14, 10^13.5, 10^13 M_sun.

_rec0   = C.discover_sims(('CV',))[0]
_fm0    = C.load_full_maps(_rec0)
_cat0   = C.load_catalog(_rec0)
_gen0   = C.load_generated(_rec0)
_cpx0   = C.centers_to_pixels(_cat0['centers'])
_truth0 = C.extract_truth_mass_patches(_fm0, _cpx0)     # (N, 3, 128, 128)
_gen0m  = _gen0[:, :C.N_MASS_CH]                         # (N, 3, 128, 128)
_dmo0   = np.stack([C.extract_patch(_fm0['dmo_fullbox'], cx, cy) for cx, cy in _cpx0])
_r200_0 = C.r200_pix_patch(_cat0)
_masses0 = np.log10(np.asarray(_cat0['masses']))

# pick one halo closest to each target mass (require r200 > 4 px)
_valid0 = np.where(np.isfinite(_r200_0) & (_r200_0 > 4))[0]
_targets = [14.5, 14.0, 13.5, 13.0]
_picks = np.array([
    _valid0[np.argmin(np.abs(_masses0[_valid0] - t))]
    for t in _targets
])
print("Selected halo log-masses:", _masses0[_picks])

def _quad_shape(img, r200, thr=0.0):
    """Mass-weighted quadrupole axis ratio q and position angle pa [rad]."""
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    ap = (xx - (W - 1) / 2) ** 2 + (yy - (H - 1) / 2) ** 2 <= r200 ** 2
    w = np.maximum(img.astype(float) - thr, 0.0) * ap
    tot = w.sum()
    if tot < 1e-30 or int((w > 0).sum()) < 5:
        return np.nan, np.nan
    xc = (xx * w).sum() / tot;  yc = (yy * w).sum() / tot
    dx, dy = xx - xc, yy - yc
    Qxx = (dx ** 2 * w).sum() / tot
    Qyy = (dy ** 2 * w).sum() / tot
    Qxy = (dx * dy * w).sum() / tot
    evals, evecs = np.linalg.eigh([[Qxx, Qxy], [Qxy, Qyy]])
    lmin, lmax = evals[0], evals[1]
    if lmax < 1e-30 or lmin < 0:
        return np.nan, np.nan
    return np.sqrt(lmin / lmax), np.arctan2(evecs[1, 1], evecs[0, 1])

from matplotlib.patches import Ellipse as _Ell, Circle as _Circ

_ROWS       = 4
_row_labels = ['DMO', 'DM (hydro)', 'Gas', 'Stars']
_row_cmaps  = ['gray_r', 'Blues', 'Greens', 'Purples']
_row_thrs   = [None, 0.0, 0.0, C.STAR_THRESH]   # None → skip ellipse (DMO)

N_SHOW = len(_picks)
fig, axes = plt.subplots(_ROWS, N_SHOW, figsize=(2.5 * N_SHOW, 2.5 * _ROWS),
                         gridspec_kw={'hspace': 0.04, 'wspace': 0.04})

for col, hi in enumerate(_picks):
    r200 = float(_r200_0[hi])
    _imgs = [_dmo0[hi], _truth0[hi, 0], _truth0[hi, 1], _truth0[hi, 2]]
    for row in range(_ROWS):
        ax = axes[row, col]
        img = _imgs[row]
        pos = img[img > 0]
        if len(pos):
            lo_v = np.quantile(pos, 0.05)
            hi_v = np.quantile(pos, 0.999)
            ax.imshow(np.log10(np.clip(img, lo_v, None)), origin='lower',
                      cmap=_row_cmaps[row], interpolation='nearest',
                      vmin=np.log10(lo_v), vmax=np.log10(hi_v))
        else:
            ax.imshow(np.zeros_like(img), origin='lower', cmap=_row_cmaps[row])
        ax.set_xticks([]); ax.set_yticks([])

        # R200c aperture circle
        ax.add_patch(_Circ((63.5, 63.5), r200, color='lime',
                           fill=False, lw=1.2, ls='--', zorder=3))

        # quadrupole shape ellipse (hydro channels only)
        if _row_thrs[row] is not None:
            q_v, pa_v = _quad_shape(img, r200, thr=_row_thrs[row])
            if np.isfinite(q_v):
                ax.add_patch(_Ell((63.5, 63.5),
                                  width=2 * r200, height=2 * r200 * q_v,
                                  angle=np.degrees(pa_v),
                                  edgecolor='white', facecolor='none',
                                  lw=1.8, zorder=4))
                ax.text(0.97, 0.03, f'$q={q_v:.2f}$',
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=7.5, color='white',
                        bbox=dict(facecolor='k', alpha=0.35, pad=1.5, boxstyle='round,pad=0.2'))

        if col == 0:
            ax.set_ylabel(_row_labels[row], fontsize=9)
        if row == 0:
            ax.set_title(rf'$\log M={_masses0[hi]:.1f}$', fontsize=8.5)

# legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
_leg = [
    Line2D([0], [0], color='lime', ls='--', lw=1.4,
           label=r'$R_{200c}$ aperture'),
    Line2D([0], [0], color='gray', lw=1.8,
           label=r'Quadrupole ellipse  ($q = \sqrt{\lambda_\mathrm{min}/\lambda_\mathrm{max}}$)'),
]
fig.legend(handles=_leg, loc='lower center', ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.03), framealpha=0.85)
fig.suptitle('Halo patches and mass-weighted quadrupole shape measurement', fontsize=11, y=1.01)
save_fig(fig, 'fig6_shape_illustration'); plt.show()
'''),
        code(r'''
sh = L('shapes.pkl'); A = sh['arrays']; suite = A['suite']
fig, axes = plt.subplots(3, 1, figsize=(6.5, 9), sharex=True)
bins = np.linspace(0,1,31)
for row,(key,name) in enumerate([('dm','DM'),('gas','Gas'),('star','Stars')]):
    ax = axes[row]
    for src,ls,lab in [('truth','-','Truth'),('gen','--','BIND')]:
        q = A[f'{src}_{key}_q']; q=q[np.isfinite(q)]
        ax.hist(q, bins=bins, density=True, histtype='stepfilled' if src=='truth' else 'step', alpha=0.4 if src=='truth' else 1, lw=2, color='tab:gray' if src=='truth' else 'tab:orange', label=lab)
    if key=='dm':
        qd=A['dmo_q']; qd=qd[np.isfinite(qd)]; ax.hist(qd, bins=bins, density=True, histtype='step', ls=':', color='k', label='DMO')
    ax.set_ylabel(rf'$\rho_{{\rm {name}}}(q)$');
    if row==0: ax.legend()
axes[-1].set_xlabel('$q$ (minor/major axis ratio)')
save_fig(fig, 'fig7_shape_axisratio'); plt.show()
'''),
        code(r'''
# shape parameter response — BIND (gen) axis-ratio q vs params
rho = sh['rho_shape']['trained']; mets = list(sh['response_metrics'])
gen_rows = [i for i,m in enumerate(mets) if m.startswith('gen_') and 'eps' not in m]
astro = [j for j in range(C.N_PARAMS) if (j+1) not in C.COSMO_PARAM_IDX]
order = sorted(astro); xl=[PARAM_LABELS[j+1] for j in order]
fig, ax = plt.subplots(figsize=(12, 3.4))
d = rho[np.ix_(gen_rows, order)]; im = ax.imshow(d, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax.set_yticks(range(len(gen_rows))); ax.set_yticklabels([mets[i].replace('gen_','') for i in gen_rows])
ax.set_xticks(range(len(order))); ax.set_xticklabels(xl, rotation=45, ha='right')
for i in range(len(gen_rows)):
    for kk in range(len(order)):
        if np.isfinite(d[i,kk]): ax.text(kk,i,f'{d[i,kk]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(d[i,kk])>0.4 else 'k')
fig.colorbar(im, ax=ax, fraction=0.02).set_label(r'$\rho_S$'); ax.set_title('BIND shape response to parameters')
save_fig(fig, 'fig7b_shape_param_response'); plt.show()
'''),
    ]
    return cells


# ════════════════════════════════════════════════════════════════════════════
# THERMO notebook — paper_fig_thermo.ipynb
# ════════════════════════════════════════════════════════════════════════════
def thermo_notebook():
    cells = [
        md("# BIND2 Thermodynamics — Paper Figures\n\nSame model, gas-thermodynamic "
           "outputs ($y, T, K, P_e$). Load-only from `tools/paper_cache/` "
           "(`thermo_*` artifacts + `field1p.npz`). CV + 1P (SB35 has no thermo truth)."),
        code(SETUP),
        code(r'''
THK = C.THERMO_CHANNELS; THD = C.THERMO_DISPLAY
'''),
        md("## T1 · Thermo field showcase (truth vs BIND)\n\nHero 1P halo, from `field1p.npz` (channels 3–6)."),
        code(r'''
f1 = L('field1p.npz'); j = int(f1['params'][0])
t = f1[f'p{j}_t_hi']; g = f1[f'p{j}_g_hi']    # (7,128,128)
fig, axes = plt.subplots(2, len(THK), figsize=(3*len(THK), 6))
def lg(im): p=im[im>0]; return np.log10(np.clip(im, p.min() if len(p) else 1e-30, None))
for c in range(len(THK)):
    for r,(src,arr) in enumerate([('Truth',t),('BIND',g)]):
        im = arr[3+c]; axes[r,c].imshow(lg(im), cmap='magma'); axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
        if r==0: axes[r,c].set_title(THD[THK[c]])
        if c==0: axes[r,c].set_ylabel(src)
save_fig(fig, 'figT1_thermo_showcase'); plt.show()
'''),
        md("## T2/T3 · Per-pixel accuracy + PDF (`thermo_pixels.npz`)"),
        code(r'''
tp = L('thermo_pixels.npz')
fig, axes = plt.subplots(2, len(THK), figsize=(3.2*len(THK), 6.2))
for c,name in enumerate(THK):
    t = tp[f'{name}_truth']; g = tp[f'{name}_gen']
    ax = axes[0,c]; ax.hexbin(t, g, gridsize=45, bins='log', cmap='viridis'); lo,hi=np.percentile(t,[1,99]); ax.plot([lo,hi],[lo,hi],'r--',lw=1)
    ax.set_title(f'{THD[name]}: bias {np.median(g-t):+.3f} dex'); ax.set_xlabel(r'$\log_{10}$ truth');
    if c==0: ax.set_ylabel(r'$\log_{10}$ BIND')
    ax2 = axes[1,c]; b=np.linspace(min(t.min(),g.min()), max(t.max(),g.max()), 60)
    ax2.hist(t, bins=b, density=True, alpha=0.4, color='tab:gray', label='Truth'); ax2.hist(g, bins=b, density=True, histtype='step', lw=2, color='tab:orange', label='BIND')
    ax2.set_xlabel(r'$\log_{10}$ value');
    if c==0: ax2.set_ylabel('PDF'); ax2.legend()
save_fig(fig, 'figT2_thermo_pixels'); plt.show()
'''),
        md("## T4 · Thermo radial profiles by mass bin (`thermo_profiles.pkl`)"),
        code(r'''
pr = L('thermo_profiles.pkl'); r = pr['r']
fig, axes = plt.subplots(len(THK), 1, figsize=(7, 2.8*len(THK)), sharex=True)
for c, ax in enumerate(axes):
    for b in BINS:
        if b not in pr['by_bin']: continue
        band = pr['by_bin'][b]
        ax.plot(r, band['pctdiff_med'][c], color=BIN_CMAP[b], lw=1.7, label=BIN_LABELS[b] if c==0 else None)
        ax.fill_between(r, band['pctdiff_p16'][c], band['pctdiff_p84'][c], color=BIN_CMAP[b], alpha=0.10)
    ax.axhline(0, color='k', ls='--', lw=0.8); ax.set_xscale('log'); ax.set_ylim(-1,1); ax.set_ylabel(rf'$\Delta$ {THD[THK[c]]}/{THD[THK[c]]}'); ax.grid(which='both', alpha=0.2)
    if c==0: ax.legend(title=r'$\log_{10}M_{200c}$', fontsize=8)
axes[-1].set_xlabel(r'$r$ [Mpc/$h$]')
save_fig(fig, 'figT4_thermo_profiles_by_bin'); plt.show()
'''),
        md("## T5 · Scaling relations Y–M, T–M, K–M, P–M (`thermo_scaling.pkl`)"),
        code(r'''
sc = L('thermo_scaling.pkl')
keys = [('Y','$Y_{200}$'),('Tx','$T_{200}$'),('K','$K_{200}$'),('P','$P_{200}$')]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for ax,(key,lab) in zip(axes, keys):
    x = sc['logM'];
    for src,cc in [('truth','k'),('gen','tab:orange')]:
        y = sc[f'{src}_{key}']; m=(y>0)&np.isfinite(x);
        ax.scatter(x[m], np.log10(y[m]), s=3, alpha=0.05, color=cc, rasterized=True)
        b=np.linspace(x[m].min(), x[m].max(), 14); bc=0.5*(b[:-1]+b[1:]); idx=np.digitize(x[m],b)-1; yl=np.log10(y[m])
        med=[np.median(yl[idx==i]) if (idx==i).sum()>5 else np.nan for i in range(len(bc))]
        ax.plot(bc, med, 'o-', color=cc, label={'truth':'Truth','gen':'BIND'}[src], mec='k')
    ax.set_xlabel(r'$\log_{10}M_{200c}$'); ax.set_ylabel(rf'$\log_{{10}}$ {lab}'); ax.set_title(lab);
axes[0].legend()
save_fig(fig, 'figT5_scaling_relations'); plt.show()
'''),
        md("## T6 · Joint mass + SZ scaling-relation scatter & residual corner plot "
           "(`mass_table.pkl` + `thermo_scaling.pkl`)\n\n"
           "Per-halo scaling relations on **CV** (fixed fiducial parameters, so residuals "
           "are halo-to-halo scatter, not parameter response), trained regime "
           "$M_{200c}\\ge 10^{13}$, **clean halos only**: the eval maps are full-box-depth "
           "(50 Mpc/$h$) projections, so an R200c aperture also sums every *projected* "
           "neighbor along the LOS — an additive, strictly-positive boost that lifts "
           "$M_\\star$, $M_{\\rm gas}$ and $Y$ coherently and manufactures a spurious "
           "$+1\\sigma$ cross-relation ridge (diagnosed in figT6c below). Halos with "
           "external catalog mass $>0.2\\,M_{200}$ inside the aperture (~6%) are dropped "
           "before fitting. Residuals are w.r.t. each population's **own** OLS power-law "
           "fit; the corner plot classifies halos into the $\\pm 1\\sigma$ tails of the "
           "*column* relation and tests whether BIND (dotted) reproduces the truth's "
           "(solid) cross-relation outlier structure. figT6d isolates the physics that "
           "remains after cleaning."),
        code(r'''
from scipy import stats

# Per-halo join: mass_table (R200-aperture masses) + thermo_scaling (Y, Tx) come
# from the same catalogs in the same row order -> align by (suite, sim, row) and
# verify with the shared log-mass column (guards against silent scrambling).
mt = L('mass_table.pkl'); th = L('thermo_scaling.pkl')
mt = mt.assign(row=mt.groupby(['suite', 'sim_id']).cumcount())
th = th.assign(row=th.groupby(['suite', 'sim_id']).cumcount())
tab = mt.merge(th, on=['suite', 'sim_id', 'row'], suffixes=('', '_th'))
assert np.allclose(tab['log_m200c'], tab['logM']), 'mass/thermo halo rows misaligned'

MASS_MIN_LOG = np.log10(C.PARAM_RESPONSE_MASS_MIN)
tab = tab[(tab.suite == 'CV') & (tab.log_m200c >= MASS_MIN_LOG)].reset_index(drop=True)
M200 = 10.0 ** tab['log_m200c'].to_numpy()

# R200-aperture contamination by projected neighbors: the maps are full-box-depth
# projections, so the aperture sums every halo along the LOS. External mass adds
# coherently (and strictly positively) to Mstar, Mgas and Y -> flag before fitting.
recs = {r['sim_id']: r for r in C.discover_sims(('CV',))}
contam = np.full(len(tab), np.nan)          # external catalog mass / own M200
for sim_id, sub in tab.groupby('sim_id'):
    cat = C.load_catalog(recs[sim_id])
    cen2 = np.asarray(cat['centers']); m_all = np.asarray(cat['masses'], float)
    r200 = np.asarray(cat['r200s'], float)
    d = np.abs(cen2[:, None, :] - cen2[None, :, :])
    d = np.minimum(d, C.BOX_SIZE - d)                       # periodic
    dproj = np.hypot(d[..., 0], d[..., 1])
    rows = sub['row'].to_numpy()
    assert np.allclose(np.log10(m_all[rows]), sub['log_m200c']), sim_id
    for i, ri in zip(sub.index, rows):
        ins = dproj[ri] < r200[ri]; ins[ri] = False
        contam[i] = m_all[ins].sum() / m_all[ri]
CONTAM_MAX = 0.2
clean = contam < CONTAM_MAX
print(f'{len(tab)} CV halos with log10 M200c >= {MASS_MIN_LOG:.0f}; removing '
      f'{(~clean).sum()} aperture-contaminated (M_ext > {CONTAM_MAX} M200) '
      f'-> {clean.sum()} clean')

def fit_mean_relation(x, y, fit_on=None):
    """OLS log10(y) = a*log10(x) + b fitted on the `fit_on` subset; residuals are
    returned for ALL valid halos (so contaminated halos can still be plotted
    against the clean relation in figT6c). sigma = clean-sample scatter."""
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    fmask = valid if fit_on is None else (valid & fit_on)
    a, b, *_ = stats.linregress(np.log10(x[fmask]), np.log10(y[fmask]))
    res = np.full(len(x), np.nan)
    res[valid] = np.log10(y[valid]) - (a * np.log10(x[valid]) + b)
    return a, b, np.nanstd(res[fmask]), valid, res

g = lambda k: tab[k].to_numpy()
# key -> (x_truth, y_truth, x_bind, y_bind, xlabel, ylabel, title)
RELATIONS = {
    'Mgas-Mstar': (g('truth_Stars_rvir'), g('truth_Gas_rvir'), g('gen_Stars_rvir'), g('gen_Gas_rvir'),
                   r'$M_\star$', r'$M_{\rm gas}$', r'$M_{\rm gas}-M_\star$'),
    'Mdm-Mstar':  (g('truth_Stars_rvir'), g('truth_DM_hydro_rvir'), g('gen_Stars_rvir'), g('gen_DM_hydro_rvir'),
                   r'$M_\star$', r'$M_{\rm dm}$', r'$M_{\rm dm}-M_\star$'),
    'SHMR':       (M200, g('truth_Stars_rvir'), M200, g('gen_Stars_rvir'),
                   r'$M_{200c}$', r'$M_\star$', 'SHMR'),
    'BaryonFrac': (M200, g('truth_Gas_rvir') + g('truth_Stars_rvir'), M200, g('gen_Gas_rvir') + g('gen_Stars_rvir'),
                   r'$M_{200c}$', r'$M_{\rm baryon}$', r'$(M_{\rm gas}+M_\star)-M_{200}$'),
    'Y-M':        (M200, g('truth_Y'), M200, g('gen_Y'),
                   r'$M_{200c}$', r'$Y_{200}$', r'$Y-M_{200}$'),
    'Y-T':        (g('truth_Tx'), g('truth_Y'), g('gen_Tx'), g('gen_Y'),
                   r'$T_{200}$', r'$Y_{200}$', r'$Y-T$'),
}
# auxiliary relations for the physics figure (figT6d); not in the grid/corner
AUX = {
    'Mgas-M': (M200, g('truth_Gas_rvir'), M200, g('gen_Gas_rvir'),
               r'$M_{200c}$', r'$M_{\rm gas}$', r'$M_{\rm gas}-M_{200}$'),
    'T-M':    (M200, g('truth_Tx'), M200, g('gen_Tx'),
               r'$M_{200c}$', r'$T_{200}$', r'$T-M_{200}$'),
}
fits = {'Truth': {}, 'BIND': {}}
print(f"{'relation':>12s}   truth a/sigma      BIND a/sigma   (clean-sample fits)")
for key, (xt, yt, xg, yg, xl, yl, ti) in {**RELATIONS, **AUX}.items():
    for src, x, y in [('Truth', xt, yt), ('BIND', xg, yg)]:
        a, b, s, v, r = fit_mean_relation(x, y, fit_on=clean)
        fits[src][key] = dict(alpha=a, beta=b, sigma=s, valid=v, mask=v & clean,
                              resid=r, x=x, y=y, xlabel=xl, ylabel=yl, title=ti)
    ft, fg = fits['Truth'][key], fits['BIND'][key]
    print(f"{key:>12s}   {ft['alpha']:+.2f} / {ft['sigma']:.2f}     {fg['alpha']:+.2f} / {fg['sigma']:.2f}")
'''),
        code(r'''
keys = list(RELATIONS)
VLIM = 0.25
fig, axes = plt.subplots(2, len(keys), figsize=(3.3 * len(keys), 6.4), constrained_layout=True)
for row, src in enumerate(['Truth', 'BIND']):
    for col, key in enumerate(keys):
        ax = axes[row, col]; fd = fits[src][key]
        lx = np.log10(fd['x'][fd['mask']]); ly = np.log10(fd['y'][fd['mask']])
        sc = ax.scatter(lx, ly, c=fd['resid'][fd['mask']], cmap='RdBu_r',
                        vmin=-VLIM, vmax=VLIM, s=14, rasterized=True, zorder=3)
        xl_ = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(xl_, fd['alpha'] * xl_ + fd['beta'], 'k--', lw=1.2)
        ax.fill_between(xl_, fd['alpha'] * xl_ + fd['beta'] - fd['sigma'],
                        fd['alpha'] * xl_ + fd['beta'] + fd['sigma'], color='k', alpha=0.10)
        ax.text(0.04, 0.96, f"{src}\n" + rf"$\alpha={fd['alpha']:.2f},\ \sigma={fd['sigma']:.2f}$ dex",
                transform=ax.transAxes, va='top', fontsize=8, color='0.35')
        if row == 0:
            ax.set_title(fd['title'])
        if row == 1:
            ax.set_xlabel(r'$\log_{10}($' + fd['xlabel'] + r'$)$')
        ax.set_ylabel(r'$\log_{10}($' + fd['ylabel'] + r'$)$')
cb = fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01)
cb.set_label('residual [dex]')
save_fig(fig, 'figT6a_scaling_scatter'); plt.show()
'''),
        code(r'''
from scipy.stats import gaussian_kde

CORNER_LABELS = {'Mgas-Mstar': r'$r_{M_{\rm gas}-M_\star}$', 'Mdm-Mstar': r'$r_{M_{\rm dm}-M_\star}$',
                 'SHMR': r'$r_{\rm SHMR}$', 'BaryonFrac': r'$r_{M_{\rm bar}-M_{200}}$',
                 'Y-M': r'$r_{Y-M}$', 'Y-T': r'$r_{Y-T}$'}
ckeys = list(RELATIONS)
both = np.ones(len(tab), bool)
for k in ckeys:
    both &= fits['Truth'][k]['mask'] & fits['BIND'][k]['mask']
R = {src: np.stack([fits[src][k]['resid'][both] for k in ckeys], 1) for src in ('Truth', 'BIND')}
sig = {src: np.array([fits[src][k]['sigma'] for k in ckeys]) for src in ('Truth', 'BIND')}
nrel = len(ckeys)
rng_lim = []
for j in range(nrel):
    v = np.r_[R['Truth'][:, j], R['BIND'][:, j]]
    lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
    pad = 0.15 * (hi - lo); rng_lim.append((lo - pad, hi + pad))

def kde_contour(ax, x, y, color, ls):
    """1/2/3-sigma-enclosure KDE contours of the (x, y) residual sub-population."""
    if len(x) < 10 or np.ptp(x) < 1e-10 or np.ptp(y) < 1e-10:
        return
    kde = gaussian_kde(np.vstack([x, y]), bw_method='scott')
    xg, yg = np.mgrid[x.min()-0.2*np.ptp(x):x.max()+0.2*np.ptp(x):80j,
                      y.min()-0.2*np.ptp(y):y.max()+0.2*np.ptp(y):80j]
    z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
    zs = np.sort(z.ravel())[::-1]; cf = np.cumsum(zs) / zs.sum()
    lv = sorted({float(zs[min(np.searchsorted(cf, f), len(zs)-1)]) for f in (0.6827, 0.9545, 0.9973)})
    ax.contour(xg, yg, z, levels=lv, colors=[color], linewidths=1.2, linestyles=ls, alpha=0.85)

fig, axes = plt.subplots(nrel, nrel, figsize=(2.4 * nrel, 2.4 * nrel), constrained_layout=True)
for i in range(nrel):
    for j in range(nrel):
        ax = axes[i, j]
        if j > i:
            ax.set_visible(False); continue
        tails = {src: (R[src][:, j] > sig[src][j], R[src][:, j] < -sig[src][j]) for src in R}
        if i == j:
            bins = np.linspace(*rng_lim[i], 35)
            ax.hist(R['Truth'][tails['Truth'][0], i], bins=bins, color='tomato', density=True, alpha=0.7)
            ax.hist(R['Truth'][tails['Truth'][1], i], bins=bins, color='steelblue', density=True, alpha=0.7)
            ax.hist(R['BIND'][tails['BIND'][0], i], bins=bins, color='tomato', density=True, histtype='step', ls=':', lw=1.8)
            ax.hist(R['BIND'][tails['BIND'][1], i], bins=bins, color='steelblue', density=True, histtype='step', ls=':', lw=1.8)
            ax.axvline(sig['Truth'][i], c='tomato', lw=0.8, ls=':', alpha=0.6)
            ax.axvline(-sig['Truth'][i], c='steelblue', lw=0.8, ls=':', alpha=0.6)
            ax.axvline(0, c='k', lw=0.7, ls='--', alpha=0.4)
            ax.set_title(CORNER_LABELS[ckeys[i]], fontsize=10, pad=2)
        else:
            kde_contour(ax, R['Truth'][tails['Truth'][0], j], R['Truth'][tails['Truth'][0], i], 'tomato', '-')
            kde_contour(ax, R['Truth'][tails['Truth'][1], j], R['Truth'][tails['Truth'][1], i], 'steelblue', '-')
            kde_contour(ax, R['BIND'][tails['BIND'][0], j], R['BIND'][tails['BIND'][0], i], 'tomato', ':')
            kde_contour(ax, R['BIND'][tails['BIND'][1], j], R['BIND'][tails['BIND'][1], i], 'steelblue', ':')
            ax.axhline(0, c='k', lw=0.4, ls='--', alpha=0.3); ax.axvline(0, c='k', lw=0.4, ls='--', alpha=0.3)
            ax.set_ylim(rng_lim[i])
        ax.set_xlim(rng_lim[j])
        if j == 0 and i > 0:
            ax.set_ylabel(CORNER_LABELS[ckeys[i]] + '\n[dex]')
        elif j > 0:
            ax.set_yticklabels([])
        if i == nrel - 1:
            ax.set_xlabel(CORNER_LABELS[ckeys[j]] + '\n[dex]')
        else:
            ax.set_xticklabels([])
fig.legend(handles=[Line2D([0], [0], color=c, ls=ls, lw=1.6, label=lb) for c, ls, lb in
                    [('tomato', '-', r'Truth $>+1\sigma$'), ('steelblue', '-', r'Truth $<-1\sigma$'),
                     ('tomato', ':', r'BIND $>+1\sigma$'), ('steelblue', ':', r'BIND $<-1\sigma$')]],
           loc='upper right', frameon=True, fontsize=11)
save_fig(fig, 'figT6b_residual_corner'); plt.show()
'''),
        md("### T6c · Why cleaning is necessary — the $+1\\sigma$ ridge is aperture "
           "contamination\n\n"
           "$r_{\\rm SHMR}$ vs $r_{Y-M}$ for **all** halos (residuals w.r.t. the "
           "clean-sample fits), colored by the projected external catalog mass inside "
           "the R200c aperture. Joint positive outliers have median "
           "$M_{\\rm ext}\\approx M_{200}$ (half have a *more massive* projected "
           "companion), and only ~16% of the contaminating mass is 3D-associated "
           "($|\\Delta{\\rm LOS}|<5$ Mpc) — mostly chance superposition through the "
           "full-depth projection. There is no 'negative neighbor', so only the "
           "positive tail is affected; BIND inherits the ridge because the neighbor "
           "is visible in its DMO conditioning patch."),
        code(r'''
from scipy.stats import spearmanr

rS = fits['Truth']['SHMR']; rY = fits['Truth']['Y-M']
mall = rS['valid'] & rY['valid']
def _rho(sel):
    return spearmanr(rS['resid'][sel], rY['resid'][sel]).statistic
hi_all = mall & (rS['resid'] > rS['sigma'])
lo_all = mall & (rS['resid'] < -rS['sigma'])
print(f"rho(r_SHMR, r_Y-M) +1s tail: all={_rho(hi_all):+.2f}  clean={_rho(hi_all & clean):+.2f}  "
      f"contaminated={_rho(hi_all & ~clean):+.2f};  -1s tail: {_rho(lo_all):+.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), constrained_layout=True, sharex=True, sharey=True)
sc = axes[0].scatter(rS['resid'][mall], rY['resid'][mall], c=np.log10(1 + contam[mall]),
                     cmap='inferno_r', vmin=0, vmax=1.0, s=18)
fig.colorbar(sc, ax=axes[0], fraction=0.05).set_label(r'$\log_{10}(1+M_{\rm ext}/M_{200})$ in aperture')
axes[0].set_title('all halos, colored by projected\nneighbor mass in the R200c aperture')
axes[1].scatter(rS['resid'][mall & clean], rY['resid'][mall & clean], s=12, c='0.55',
                alpha=0.5, label='clean')
axes[1].scatter(rS['resid'][mall & ~clean], rY['resid'][mall & ~clean], s=18, c='crimson',
                alpha=0.8, label=rf'contaminated ($M_{{\rm ext}}>{CONTAM_MAX}\,M_{{200}}$)')
axes[1].set_title(rf'$\rho_{{+1\sigma}}$: all $={_rho(hi_all):+.2f}$, '
                  rf'clean $={_rho(hi_all & clean):+.2f}$'
                  '\n' rf'$\rho_{{-1\sigma}} = {_rho(lo_all):+.2f}$')
axes[1].legend(fontsize=9, loc='lower right')
for ax in axes:
    ax.axvline(rS['sigma'], c='tomato', lw=0.9, ls=':')
    ax.axvline(-rS['sigma'], c='steelblue', lw=0.9, ls=':')
    ax.axhline(0, c='k', lw=0.5, ls='--', alpha=0.4); ax.axvline(0, c='k', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlabel(r'$r_{\rm SHMR}$ [dex]')
axes[0].set_ylabel(r'$r_{Y-M}$ [dex]')
save_fig(fig, 'figT6c_aperture_contamination'); plt.show()
'''),
        md("### T6d · The physics that survives cleaning: baryon-rich halos, mediated "
           "by gas mass\n\n"
           "In the clean sample the $+1\\sigma$ SHMR halos are still SZ-bright — because "
           "they are **gas-rich** at fixed $M_{200}$ (left; the partial correlation of "
           "$r_{\\rm SHMR}$ with $r_Y$ at fixed $r_{M_{\\rm gas}}$ nearly vanishes, and "
           "$r_Y$ tracks $r_{M_{\\rm gas}}$ tightly, middle). Star-**poor** halos are "
           "*not* gas-poor: downward scatter comes from channel-specific suppression "
           "(star-formation inefficiency vs feedback gas ejection), so the $-1\\sigma$ "
           "tails decorrelate (right). BIND (hatched) reproduces the full "
           "tail-conditional structure."),
        code(r'''
from scipy.stats import spearmanr

def tail_rho(src, xkey, ykey, side):
    fx, fy = fits[src][xkey], fits[src][ykey]
    m = fx['mask'] & fy['mask']
    m &= (fx['resid'] > fx['sigma']) if side == '+' else (fx['resid'] < -fx['sigma'])
    return spearmanr(fx['resid'][m], fy['resid'][m]).statistic, int(m.sum())

rS, rG, rY = (fits['Truth'][k] for k in ('SHMR', 'Mgas-M', 'Y-M'))
mm = rS['mask'] & rG['mask'] & rY['mask']
hi = mm & (rS['resid'] > rS['sigma']); lo = mm & (rS['resid'] < -rS['sigma'])

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)

# (a) star-rich halos are gas-rich; star-poor halos are NOT gas-poor
ax = axes[0]
ax.scatter(rS['resid'][mm & ~hi & ~lo], rG['resid'][mm & ~hi & ~lo], s=8, c='0.7', alpha=0.4)
ax.scatter(rS['resid'][hi], rG['resid'][hi], s=16, c='tomato',
           label=rf'$+1\sigma$: $\rho={tail_rho("Truth", "SHMR", "Mgas-M", "+")[0]:+.2f}$')
ax.scatter(rS['resid'][lo], rG['resid'][lo], s=16, c='steelblue',
           label=rf'$-1\sigma$: $\rho={tail_rho("Truth", "SHMR", "Mgas-M", "-")[0]:+.2f}$')
ax.axhline(0, c='k', lw=0.5, ls='--', alpha=0.4); ax.axvline(0, c='k', lw=0.5, ls='--', alpha=0.4)
ax.set_xlabel(r'$r_{\rm SHMR}$ [dex]'); ax.set_ylabel(r'$r_{M_{\rm gas}-M}$ [dex]')
ax.legend(fontsize=9); ax.set_title('clean sample (Truth)')

# (b) the Y excess is carried by gas mass
ax = axes[1]
sc = ax.scatter(rG['resid'][mm], rY['resid'][mm], c=rS['resid'][mm], cmap='RdBu_r',
                vmin=-0.2, vmax=0.2, s=12)
fig.colorbar(sc, ax=ax, fraction=0.05).set_label(r'$r_{\rm SHMR}$ [dex]')
rho_gy = spearmanr(rG['resid'][mm], rY['resid'][mm]).statistic
ax.axhline(0, c='k', lw=0.5, ls='--', alpha=0.4); ax.axvline(0, c='k', lw=0.5, ls='--', alpha=0.4)
ax.set_xlabel(r'$r_{M_{\rm gas}-M}$ [dex]'); ax.set_ylabel(r'$r_{Y-M}$ [dex]')
ax.set_title(rf'$\rho(r_{{M_{{\rm gas}}}}, r_Y) = {rho_gy:+.2f}$ — gas-mass mediated')

# (c) tail-conditional correlations, Truth vs BIND
ax = axes[2]
XKEYS = ['Y-M', 'Mgas-M', 'T-M']
xpos = np.arange(len(XKEYS)); w = 0.19
for k, (src, side, color, hatch) in enumerate([('Truth', '+', 'tomato', None), ('BIND', '+', 'tomato', '//'),
                                               ('Truth', '-', 'steelblue', None), ('BIND', '-', 'steelblue', '//')]):
    vals = [tail_rho(src, 'SHMR', xk, side)[0] for xk in XKEYS]
    ax.bar(xpos + (k - 1.5) * w, vals, w, color=color, hatch=hatch,
           edgecolor='k', lw=0.5, alpha=0.9 if hatch is None else 0.45,
           label=rf'{src} ${side}1\sigma$')
ax.axhline(0, c='k', lw=0.8)
ax.set_xticks(xpos); ax.set_xticklabels([r'$r_{Y-M}$', r'$r_{M_{\rm gas}-M}$', r'$r_{T-M}$'])
ax.set_ylabel(r'$\rho_S$ with $r_{\rm SHMR}$ in tail')
ax.legend(fontsize=8, ncol=2); ax.set_title('tail-conditional correlations')
save_fig(fig, 'figT6d_outlier_physics'); plt.show()
'''),
        md("## T7 · Thermo parameter response (`thermo_param.pkl`)"),
        code(r'''
tpar = L('thermo_param.pkl'); rho = tpar['rho_thermo']['trained']; keys = list(tpar['keys'])
astro = [j for j in range(C.N_PARAMS) if (j+1) not in C.COSMO_PARAM_IDX]
order = sorted(astro); xl=[PARAM_LABELS[j+1] for j in order]
fig, axes = plt.subplots(1, 2, figsize=(15, 3.6), sharey=True)
for ax,(src) in zip(axes, ['True','BIND']):
    d = rho[src][:,order]; im = ax.imshow(d, aspect='auto', cmap='RdBu_r', vmin=-0.6, vmax=0.6)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys); ax.set_xticks(range(len(order))); ax.set_xticklabels(xl, rotation=45, ha='right'); ax.set_title(src)
    for i in range(len(keys)):
        for kk in range(len(order)):
            if np.isfinite(d[i,kk]): ax.text(kk,i,f'{d[i,kk]:.2f}',ha='center',va='center',fontsize=6,color='white' if abs(d[i,kk])>0.45 else 'k')
fig.colorbar(im, ax=axes, fraction=0.012).set_label(r'$\rho_S$')
save_fig(fig, 'figT7_thermo_param_response'); plt.show()
'''),
        md("## T8 · Thermo field response (1P butterfly, `field1p.npz` channels 3–6)"),
        code(r'''
f1 = L('field1p.npz'); params=f1['params']
from scipy.stats import pearsonr
fig, axes = plt.subplots(len(params), 2*len(THK), figsize=(1.9*2*len(THK), 1.9*len(params)))
for ri,j in enumerate(params):
    t_hi,t_lo,g_hi,g_lo = (f1[f'p{j}_{x}'] for x in ('t_hi','t_lo','g_hi','g_lo'))
    if t_hi.shape[0] < 7:  # this 1P sim lacked thermo truth
        for a in axes[ri]: a.axis('off');
        continue
    for c in range(len(THK)):
        eps = 1e-3*np.percentile(np.r_[t_hi[3+c][t_hi[3+c]>0], t_lo[3+c][t_lo[3+c]>0]] if (t_hi[3+c]>0).any() else [1e-6], 99)
        tlr=np.log10((t_hi[3+c]+eps)/(t_lo[3+c]+eps)); glr=np.log10((g_hi[3+c]+eps)/(g_lo[3+c]+eps))
        v=np.quantile(np.abs(np.r_[tlr.ravel(),glr.ravel()]),0.995) or 0.1
        for ki,(fld,kind) in enumerate([(tlr,'Truth'),(glr,'BIND')]):
            ax=axes[ri,c*2+ki]; ax.imshow(fld, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v); ax.set_xticks([]); ax.set_yticks([])
            if ri==0: ax.set_title(f'{THD[THK[c]]}\n{kind}', fontsize=7)
            if c==0 and ki==0: ax.set_ylabel(PARAM_LABELS[int(j)], fontsize=8, rotation=0, ha='right', va='center', labelpad=20)
save_fig(fig, 'figT8_thermo_field_response'); plt.show()
'''),
    ]
    return cells


# ════════════════════════════════════════════════════════════════════════════
# REDSHIFT notebook — paper_fig_redshift.ipynb
# ════════════════════════════════════════════════════════════════════════════
def redshift_notebook():
    cells = [
        md("# BIND2 Redshift — Paper Figures\n\nThe **same** conditioned model as a "
           "function of scale factor $a=1/(1+z)$ (`redshift_evo.npz`). Truth-validated "
           "patch-level multi-z comparison lives in `analysis_redshift.ipynb`; z>0 thermo "
           "absolute amplitudes are not yet truth-validated (see WORKLOG)."),
        code(SETUP),
        md("## Redshift response of one group ($z=0,0.5,1,2$)"),
        code(r'''
rz = L('redshift_evo.npz'); z = rz['z_levels']; gz = rz['gz']   # (nz,7,128,128)
rows = [('Gas',1,False),('compton_y',3,True),('T',4,True)]
def lg(im): p=im[im>0]; return np.log10(np.clip(im, p.min() if len(p) else 1e-30, None))
fig, axes = plt.subplots(len(rows), len(z), figsize=(3*len(z), 3*len(rows)))
for r,(lab,ch,_) in enumerate(rows):
    for c in range(len(z)):
        axes[r,c].imshow(lg(gz[c,ch]), cmap='magma'); axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
        if r==0: axes[r,c].set_title(f'z = {z[c]:.1f}')
        if c==0: axes[r,c].set_ylabel(lab)
fig.suptitle(f"fm_redshift · group logM={float(rz['hero_logM']):.2f} · redshift response", y=1.0)
save_fig(fig, 'figR1_redshift_response'); plt.show()
'''),
        md("## Amplitude evolution with redshift (BIND, stacked groups)"),
        code(r'''
rz = L('redshift_evo.npz'); z = rz['z_levels']; amp = rz['amp']  # (nz,3): Gas mass, mean y, mean T
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax,(k,lab) in zip(axes, [(0,'Gas patch mass'),(1,r'mean Compton-$y$'),(2,'mean $T$')]):
    ax.plot(z, amp[:,k]/amp[0,k], 'o-', color='tab:orange'); ax.set_xlabel('$z$'); ax.set_ylabel(f'{lab} (norm. to z=0)'); ax.set_title(lab); ax.grid(alpha=0.3)
fig.suptitle(f"BIND redshift evolution (mean over {int(rz['n_stack'])} groups)")
save_fig(fig, 'figR2_amplitude_evolution'); plt.show()
'''),
    ]
    return cells


# ════════════════════════════════════════════════════════════════════════════
# MODELS notebook — paper_fig_models.ipynb (Appendix A + B)
# ════════════════════════════════════════════════════════════════════════════
def models_notebook():
    cells = [
        md("# BIND2 Appendices — Model variants\n\n**A** VDM vs Flow-Matching "
           "(`vdm_fm.npz`); **B** observable-conditioned FM (`observables.npz`)."),
        code(SETUP),
        md("## Appendix A · VDM vs Flow-Matching\n\nSame UNet, two formulations. VDM is "
           "sharper at small scales (mass↔structure tension); FM is smoother/mass-correct."),
        code(r'''
vf = L('vdm_fm.npz')
def lg(im): p=im[im>0]; return np.log10(np.clip(im, p.min() if len(p) else 1e-30, None))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
ax[0].imshow(lg(vf['fm_gas']), cmap='magma'); ax[0].set_title('FM (fm_redshift) Gas'); ax[0].set_xticks([]); ax[0].set_yticks([])
ax[1].imshow(lg(vf['vdm_gas']), cmap='magma'); ax[1].set_title('VDM Gas'); ax[1].set_xticks([]); ax[1].set_yticks([])
ax[2].loglog(vf['kf'], vf['pf'], label='FM'); ax[2].loglog(vf['kv'], vf['pv'], label='VDM')
ax[2].set(xlabel='$k$ [$h$/Mpc]', ylabel='Gas $P(k)$ [patch]', title=f"Stacked patch P(k) — VDM/FM (k>20) = {float(vf['highk_ratio']):.2f}"); ax[2].legend()
save_fig(fig, 'figA_vdm_vs_fm'); plt.show()
'''),
        md("## Appendix B · Observable-conditioned emulation (M+Y+Tx)\n\nCondition on "
           "aperture-integrated R200 observables instead of the 35 parameters."),
        code(r'''
ob = L('observables.npz')
def lg(im): p=im[im>0]; return np.log10(np.clip(im, p.min() if len(p) else 1e-30, None))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
ax[0].imshow(lg(ob['truth_gas'][0]), cmap='magma'); ax[0].set_title('truth Gas'); ax[0].set_xticks([]); ax[0].set_yticks([])
ax[1].imshow(lg(ob['gen_gas'][0]), cmap='magma'); ax[1].set_title('BIND2 | M+Y+Tx Gas'); ax[1].set_xticks([]); ax[1].set_yticks([])
mt, mg = ob['truth_gas_mass'], ob['gen_gas_mass']
ax[2].loglog(mt, mg, '.', ms=8); lim=[mt.min(), mt.max()]; ax[2].plot(lim, lim, 'k--')
ax[2].set(xlabel='truth Gas mass [patch]', ylabel='obs-conditioned Gas mass', title='Gas recovery (M+Y+Tx)')
save_fig(fig, 'figB_observable_conditioned'); plt.show()
'''),
    ]
    return cells


def write(cells, path):
    nb = new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "torch3", "language": "python", "name": "torch3"}
    nb.metadata.language_info = {"name": "python"}
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path, f"({len(cells)} cells)")


if __name__ == "__main__":
    write(main_notebook(), f"{HERE}/paper_figures2.ipynb")
    write(thermo_notebook(), f"{HERE}/paper_fig_thermo.ipynb")
    write(redshift_notebook(), f"{HERE}/paper_fig_redshift.ipynb")
    write(models_notebook(), f"{HERE}/paper_fig_models.ipynb")
