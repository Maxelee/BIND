"""Variance attribution of the matter-power suppression across the Sobol cube,
decomposed by halo subset (mass decade x gas-fraction-at-fixed-mass tercile).

Reads the partial-S(k) cache from tools/partial_supp_sobol.py and asks: which
halo population subset drives Var_theta[S(k)] at the weak-lensing scales?

Attribution uses the covariance decomposition of the variance:
  share_s = Cov_theta(dS_s, dS_full) / sum_t Cov_theta(dS_t, dS_full),
i.e. each subset's covariance with the total suppression, normalised across
subsets.  This is the natural partition of Var[dS_full] = sum_s Cov(dS_s,dS_full)
and -- unlike an LMG/regression split -- it is NOT washed out by the strong
collinearity between subsets (they all respond to the same feedback knobs), since
it weights each subset by the MAGNITUDE of its contribution's swing, not just its
correlation.  The subset-only paste inflates the raw sum (~1.8x) via the
mass-renorm + patch-overlap cross-term, so we report the *relative* (normalised)
share and show the cross-term explicitly in Panel C (honest caveat); absolute
"X% of variance" awaits the leave-one-out / field-additive partial (see notes).

Usage:
  python tools/fig_partial_supp.py --cache partial_supp_proto.npz   # prototype
  python tools/fig_partial_supp.py --cache partial_supp_sobol.npz   # full run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

CACHE = Path('analysis_physics_cache')
FIG = Path('paper_figures'); FIG.mkdir(exist_ok=True)
SOBOL = Path('/mnt/home/mlee1/ceph/sobol_ss_cv')


def cov_shares(contrib, dS_full):
    """Covariance variance-partition: share_s = Cov(c_s, dS_full)/Var(dS_full).

    With the 2-bracket Shapley contribution c_s the raw shares sum to ~1
    (additive), so raw IS the absolute fraction of Var[S]; also return the
    normalised version for the pure relative read.
    """
    cov = np.array([np.cov(contrib[:, s], dS_full)[0, 1]
                    for s in range(contrib.shape[1])])
    raw = cov / dS_full.var()
    return cov / cov.sum(), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='partial_supp_proto.npz')
    ap.add_argument('--ktarget', type=float, default=10.0)
    args = ap.parse_args()

    z = np.load(CACHE / args.cache, allow_pickle=True)
    dids = z['design_ids']
    k = z['k_box']
    S_full = z['S_full']                       # (D, nk)
    S_sub = z['S_sub']                         # (D, 9, nk)  subset-only paste
    S_loo = z['S_loo']                         # (D, 9, nk)  leave-one-subset-out
    mass_labels = [str(s) for s in z['mass_labels']]
    gas_labels = [str(s) for s in z['gas_labels']]
    ik = int(np.argmin(np.abs(k - args.ktarget)))
    D = len(dids)

    cube = np.load(SOBOL / 'cube.npz', allow_pickle=True)
    astro = [str(s) for s in cube['astro_names']]
    Dn = cube['design_norm'][dids]             # (D, 30) aligned to these designs

    dS_full = S_full[:, ik] - 1.0
    only = S_sub[:, :, ik] - 1.0               # (D, 9) subset painted alone ("first in")
    loo = S_full[:, ik][:, None] - S_loo[:, :, ik]   # (D, 9) removal effect ("last in")
    contrib = 0.5 * (only + loo)               # 2-bracket Shapley contribution

    _, raw = cov_shares(contrib, dS_full)      # raw sums ~1 -> absolute fraction of Var[S]
    grid = raw.reshape(3, 3)                   # rows=mass, cols=gas
    mean_c = contrib.mean(0).reshape(3, 3)     # mean absolute suppression contribution

    print(f'n_designs={D}  k={k[ik]:.2f}  Shapley share sum={raw.sum():.2f} '
          f'(close to 1 = additive)')
    print('absolute variance shares (% of Var[S]):')
    for m in range(3):
        print(f'  {mass_labels[m]:>10}: ' +
              '  '.join(f'{gas_labels[g]} {100*grid[m,g]:4.1f}%' for g in range(3)) +
              f'   | mass total {100*grid[m].sum():4.1f}%')
    print('  gas totals: ' +
          '  '.join(f'{gas_labels[g]} {100*grid[:,g].sum():4.1f}%' for g in range(3)))

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # A: absolute variance-share heatmap (mass x gas)
    im = axA.imshow(100 * grid, cmap='magma_r', aspect='auto')
    axA.set_xticks(range(3)); axA.set_xticklabels(gas_labels)
    axA.set_yticks(range(3)); axA.set_yticklabels(mass_labels)
    axA.set_title(f'A. share of Var[S(k={k[ik]:.0f})]  (Shapley; Σ={100*raw.sum():.0f}%)')
    for m in range(3):
        for g in range(3):
            axA.text(g, m, f'{100*grid[m,g]:.1f}%', ha='center', va='center',
                     color='white' if grid[m, g] > grid.max() * 0.5 else 'k',
                     fontsize=10)
    fig.colorbar(im, ax=axA, label='% of Var[S]')
    axA.set_ylabel('halo mass decade'); axA.set_xlabel('gas content at fixed mass')

    # B: marginal aggregation -- is it mass or gas?
    x = np.arange(3)
    axB.bar(x - 0.2, 100 * grid.sum(1), width=0.38, color='tab:blue',
            label='by mass decade')
    axB.bar(x + 0.2, 100 * grid.sum(0), width=0.38, color='tab:red',
            label='by gas content')
    axB.set_xticks(x)
    axB.set_xticklabels([f'{mass_labels[i]}\n/ {gas_labels[i]}' for i in range(3)],
                        fontsize=8)
    axB.set_ylabel('% of Var[S]'); axB.set_title('B. mass axis vs gas axis')
    axB.legend(fontsize=8)

    # C: contribution brackets -- subset-only (over) / LOO (under) / Shapley (additive)
    axC.scatter(dS_full, only.sum(1), s=12, c='tab:orange', label='Σ subset-only (over)')
    axC.scatter(dS_full, loo.sum(1), s=12, c='tab:blue', label='Σ leave-one-out (under)')
    axC.scatter(dS_full, contrib.sum(1), s=22, c='k', label='Σ Shapley (mean)')
    lim = [float(min(dS_full.min(), loo.sum(1).min())),
           float(max(dS_full.max(), only.sum(1).max()))]
    axC.plot(lim, lim, 'k--', lw=1)
    slope = float(np.polyfit(dS_full, contrib.sum(1), 1)[0])
    axC.set_xlabel(r'$\Delta S_{\rm full}$ (all halos)')
    axC.set_ylabel(r'$\sum_s$ contribution')
    axC.set_title(f'C. the Shapley midpoint is additive\n'
                  f'(Σ Shapley vs ΔS_full: slope {slope:.2f})')
    axC.legend(fontsize=7)

    # D: direction-1 -- does mass-decade dominance shift with feedback strength?
    knob = 'WindEnergyIn1e51erg'
    ki = astro.index(knob)
    lo = Dn[:, ki] <= np.median(Dn[:, ki])
    c_lo = contrib[lo].reshape(int(lo.sum()), 3, 3).mean(0).sum(1)   # mean contrib per mass
    c_hi = contrib[~lo].reshape(int((~lo).sum()), 3, 3).mean(0).sum(1)
    axD.bar(x - 0.2, c_lo, width=0.38, color='tab:green', label=f'low {knob}')
    axD.bar(x + 0.2, c_hi, width=0.38, color='tab:purple', label=f'high {knob}')
    axD.axhline(0, color='k', lw=0.6)
    axD.set_xticks(x); axD.set_xticklabels(mass_labels)
    axD.set_ylabel(r'mean Shapley contribution to $\Delta S$')
    axD.set_title('D. does group-scale dominance shift with wind energy?')
    axD.legend(fontsize=8)

    name = 'partial_supp_decomposition' + ('_proto' if 'proto' in args.cache else '')
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=140, bbox_inches='tight')
    print(f'wrote {FIG / name}.png/.pdf')


if __name__ == '__main__':
    main()
