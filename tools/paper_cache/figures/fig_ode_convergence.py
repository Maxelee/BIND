#!/usr/bin/env python3
"""ODE (Euler) convergence figure for Sec 7.1 (referee I-16).

Composite S(k)_N / S(k)_400 per step count, median over sims with the 16-84
band for the production N=50, CV and SB35 panels, from the paired-seed ladder
npz files produced by ode_convergence.py.

Run: source ~/venvs/torch3/bin/activate && python fig_ode_convergence.py
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O = '/mnt/home/mlee1/ceph/paper_cache/fm_two_head'
OUT = '/mnt/home/mlee1/vdm_bind2/examples/paper_figures/fig_ode_convergence'
STEPS = [('n20', 20, '0.65', '--'), ('n50', 50, 'tab:green', '-'),
         ('n100', 100, 'tab:blue', '-.'), ('n200', 200, 'tab:purple', ':')]

def rebin(k, y, nbin=40):
    edges = np.geomspace(k[1], k[-1], nbin + 1)
    idx = np.digitize(k, edges) - 1
    out_k, out_y = [], []
    for i in range(nbin):
        m = idx == i
        if m.sum():
            out_k.append(np.mean(k[m])); out_y.append(np.mean(y[m]))
    return np.array(out_k), np.array(out_y)

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=True)
for ax, (suite, path, nsim_label) in zip(
        axes, [('CV', f'{O}/ode_conv_cv.npz', None),
               ('SB35', f'{O}/ode_conv_sb35.npz', None)]):
    d = np.load(path, allow_pickle=True)
    sims = sorted({'/'.join(f.split('/')[:2]) for f in d.files})
    k = d[f'{sims[0]}/k']
    for name, n, color, ls in STEPS:
        ratios = np.array([d[f'{s}/{name}/pk'] / d[f'{s}/n400/pk'] for s in sims])
        med = np.median(ratios, 0)
        kk, mm = rebin(k, med)
        lw = 2.4 if name == 'n50' else 1.5
        ax.plot(kk, mm, color=color, ls=ls, lw=lw,
                label=rf'$N_{{\rm steps}} = {n}$' + (' (production)' if n == 50 else ''))
        if name == 'n50':
            p16 = np.percentile(ratios, 16, 0); p84 = np.percentile(ratios, 84, 0)
            k1, b1 = rebin(k, p16); k2, b2 = rebin(k, p84)
            ax.fill_between(k1, b1, b2, color=color, alpha=0.18, lw=0)
    ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlim(k[1], 64.3)
    ax.set_ylim(0.94, 1.02)
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    ax.text(0.05, 0.08, f'{suite}  ($n = {len(sims)}$ sims)',
            transform=ax.transAxes, fontsize=11)
axes[0].set_ylabel(r'$S_N(k)\, /\, S_{400}(k)$')
axes[0].legend(fontsize=9, loc='lower left', bbox_to_anchor=(0.03, 0.16), framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=300)
print('wrote', OUT)
