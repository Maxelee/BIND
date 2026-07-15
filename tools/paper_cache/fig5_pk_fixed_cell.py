# §5 · Full-box power spectrum — CORRECTED (shared-content paste), Fig 5 replacement.
# Paste this whole cell into paper_figures2.ipynb (after the setup cell that defines
# L/save_fig/SUITE_DISPLAY). Reads pk_fixed.npz, built by tools/paper_cache/build_pk_fixed.py:
#     python build_pk_fixed.py --metric fixed --pool 6      # CPU: >=1e13 / >=1e12 shared paste
#     python build_pk_fixed.py --metric cover               # GPU: covering paint (Rc, >=1e10)
#     python build_pk_fixed.py --reduce                     # -> pk_fixed.npz
# Layout/styling identical to the original Fig 5 cell; the three BIND curves are the
# halo-population ladder: trained regime (>=1e13), extrapolation floor (>=1e12), and
# the full covering paint (all FoF halos >=1e10, generated-closure Rc apertures).

pkf = L('pk_fixed.npz'); k = pkf['k']
knyq = np.pi*1024/50.0
km = k <= knyq; k = k[km]                      # drop super-Nyquist bins
suites = [s for s in ('CV','Test','1P') if f'{s}_fixed_truth' in pkf]

def med(a):  return np.median(a[:, km], 0)
def band(num, den):
    r = num[:, km] / den[:, km]
    return np.median(r, 0), np.quantile(r, 0.16, 0), np.quantile(r, 0.84, 0)

# (stack key, truth/dmo namespace, color, label, shaded band in bottom panel)
VARIANTS = [
    ('fixed_ge13',  'fixed', 'tab:purple', r'BIND $M_{200c}\geq 10^{13}$',            False),
    ('fixed_ge12',  'fixed', 'tab:orange', r'BIND $M_{200c}\geq 10^{12}$',            True),
    ('cover_cover', 'cover', 'tab:red',    r'BIND covering ($R_c$, $\geq 10^{10}$)',  True),
]

fig, axes = plt.subplots(2, len(suites), figsize=(5.2*len(suites), 8), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1], 'hspace': 0}, squeeze=False)
for col, s in enumerate(suites):
    at, ab = axes[0, col], axes[1, col]
    tr_f, dm_f = pkf[f'{s}_fixed_truth'], pkf[f'{s}_fixed_dmo']
    at.plot(k, med(tr_f)/med(dm_f), 'k', lw=1.4, label='Truth/DMO')

    for key, ns, c, lab, shade in VARIANTS:
        if f'{s}_{key}' not in pkf: continue
        num, tr, dm = pkf[f'{s}_{key}'], pkf[f'{s}_{ns}_truth'], pkf[f'{s}_{ns}_dmo']
        fin = np.isfinite(num).all(1)          # sims lacking >=1e13 halos are NaN rows
        num, tr, dm = num[fin], tr[fin], dm[fin]
        if len(num) == 0: continue
        at.plot(k, med(num)/med(dm), color=c, lw=1.4, label=lab)
        rm, rlo, rhi = band(num, tr)
        ab.plot(k, rm, color=c, lw=1.3, label=lab)
        if shade: ab.fill_between(k, rlo, rhi, color=c, alpha=0.16)

    # hydro-replaced control (from the legacy pk.npz stacks; paste-immune for truth content)
    hr, tr_pk = pkf.get(f'{s}_pkfile_hydro_replace'), pkf.get(f'{s}_pkfile_truth')
    if hr is not None and np.isfinite(hr).any():
        at.plot(k, med(hr)/med(dm_f), color='tab:blue', lw=1.2, label='Hydro-repl/DMO')
        ab.plot(k, np.median(hr[:, km]/tr_pk[:, km], 0), color='tab:blue', ls='-.', lw=1,
                label='Hydro-repl/Truth')
    ab.plot(k, np.median(dm_f[:, km]/tr_f[:, km], 0), 'gray', ls='--', lw=1, label='DMO/Truth')

    at.axhline(1, color='gray', lw=0.6, ls='--'); at.set_xscale('log')
    at.grid(which='both', alpha=0.25); at.set_title(SUITE_DISPLAY[s]); at.tick_params(labelbottom=False)
    ab.axhspan(0.8, 1.2, color='tab:green', alpha=0.08); ab.axhline(1, color='k', lw=0.6, ls='--')
    ab.set_ylim(0.5, 1.5); ab.set_xscale('log'); ab.grid(which='both', alpha=0.25)
    ab.set_xlabel(r'$k$ [$h$/Mpc]')
    if col == 0:
        at.set_ylabel(r'$P/P_{\rm DMO}$'); at.legend(fontsize=8)
        ab.set_ylabel(r'$P/P_{\rm Truth}$'); ab.legend(fontsize=7, loc='lower left')

save_fig(fig, 'fig5_total_field_pk_fixed'); plt.show()
