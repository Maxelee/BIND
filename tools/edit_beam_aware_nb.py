"""One-shot edit of profile_fb_beam_aware.ipynb to add the three follow-ups
recommended after the v1↔v2 review:

  1. Multi-seed averaging in the §4 truth-eval and §6 naive↔aware cells
     (5 noise seeds, average the per-seed mean prediction before RMS).
  2. Radial Y vs Y+kSZ reduction overlay at the ACT (1.6′) beam.
  3. σ_mc sweep at fixed 1.6′ beam (the actually-actionable info-loss curve).

Inserts new cells in place; clears outputs of touched cells so the user re-runs.
"""
from __future__ import annotations
import json, pathlib

NB = pathlib.Path('profile_fb_beam_aware.ipynb')
nb = json.loads(NB.read_text())
cells = nb['cells']


def find(needle):
    for i, c in enumerate(cells):
        s = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        if needle in s:
            return i
    raise KeyError(needle)


def src_lines(s):
    """ipynb cell sources are list-of-lines with trailing newlines except last."""
    parts = s.splitlines(keepends=True)
    return parts


def code(s, cid=None):
    cell = dict(cell_type='code', metadata={}, source=src_lines(s),
                outputs=[], execution_count=None)
    if cid:
        cell['id'] = cid
    return cell


def md(s, cid=None):
    cell = dict(cell_type='markdown', metadata={}, source=src_lines(s))
    if cid:
        cell['id'] = cid
    return cell


# ---- Edit 1: §4 truth-eval (5-seed averaging) -----------------------------
i = find('def train_ridge(Bp, b, cols')
cells[i]['source'] = src_lines("""\
def train_ridge(Bp, b, cols, alpha=10.0):
    Xtr = _design_features(Bp, cols, b); Ytr = Bp['fb'][:, b, :]
    xs = StandardScaler().fit(Xtr); ys = StandardScaler().fit(Ytr)
    return xs, ys, Ridge(alpha=alpha).fit(xs.transform(Xtr), ys.transform(Ytr))

def predict_truth(Bp, tr, b, cols):
    xs, ys, reg = train_ridge(Bp, b, cols)
    feats = []
    for c in cols:
        v = tr[c][b]; v = np.log10(np.clip(v, 1e-30, None)) if c in LOGFEAT else v
        feats.append(v[None, :])
    X = np.nan_to_num(xs.transform(np.hstack(feats)), nan=0.0)
    return ys.inverse_transform(reg.predict(X))[0]

def eval_truth(Bp_train, tr_test, cols, b, n_seeds=5, n_mc=200,
               f_in=0.05, f_out=0.40, base_seed=42):
    '''Train Ridge once on Bp_train; evaluate against noised tr_test, repeated
    over n_seeds independent noise pools of n_mc draws each. Returns the RMS of
    the seed-averaged mean prediction (tightens the small-RMS noise floor) plus
    the seed-averaged mean curve and 16/84 band across all draws.'''
    seed_means = []
    all_draws = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + 1000*s)
        draws = []
        for _ in range(n_mc):
            tr_n = {k: tr_test[k].copy() for k in tr_test}
            for c in cols:
                tr_n[c] = np.array([lognormal_noise(tr_n[c][bb], r, f_in, f_out, rng)
                                    for bb in range(3)])
            draws.append(predict_truth(Bp_train, tr_n, b, cols))
        draws = np.asarray(draws)
        seed_means.append(draws.mean(0))
        all_draws.append(draws)
    seed_means = np.asarray(seed_means)
    mean = seed_means.mean(0)
    pooled = np.concatenate(all_draws, 0)
    lo, hi = np.percentile(pooled, [16, 84], 0)
    return mean, lo, hi, seed_means

F_IN, F_OUT, NMC, N_SEEDS = 0.05, 0.40, 200, 5
rms = {}; rms_seed_std = {}; pred_curves = {}; pred_band = {}
for B_ in BEAMS:
    Bp_b = Bp_beam[B_['fwhm']]; tr_b = truth_beam[B_['fwhm']]
    for L, cols in LADDERS.items():
        for b, lbl in enumerate(MLBL):
            mean, lo, hi, sm_ = eval_truth(Bp_b, tr_b, cols, b,
                                            n_seeds=N_SEEDS, n_mc=NMC,
                                            f_in=F_IN, f_out=F_OUT)
            tru = truth['fb'][b]
            rms[(B_['fwhm'], L, lbl)] = float(np.sqrt(np.nanmean((mean - tru)**2)))
            # seed-to-seed scatter on the RMS itself, for the ± below
            rms_per_seed = np.sqrt(np.nanmean((sm_ - tru[None, :])**2, 1))
            rms_seed_std[(B_['fwhm'], L, lbl)] = float(rms_per_seed.std(ddof=1))
            pred_curves[(B_['fwhm'], L, lbl)] = mean
            pred_band[(B_['fwhm'], L, lbl)] = (lo, hi)

print(f'RMS|pred-truth| in f_b (×1e3) ± seed-scatter (N_SEEDS={N_SEEDS}, NMC={NMC})')
print(f'{"beam":24s} {"ladder":8s}'+''.join(f'{l:>16s}' for l in MLBL))
for B_ in BEAMS:
    for L in LADDERS:
        row = f"{B_['name']:24s} {L:8s}"
        for lbl in MLBL:
            v = rms[(B_['fwhm'], L, lbl)]*1e3
            e = rms_seed_std[(B_['fwhm'], L, lbl)]*1e3
            row += f'  {v:6.2f}±{e:4.2f}'
        print(row)
""")
cells[i]['outputs'] = []
cells[i]['execution_count'] = None


# ---- Edit 2: §6 naive↔aware (same multi-seed logic) -----------------------
i = find('rms_naive = {}')
cells[i]['source'] = src_lines("""\
rms_naive = {}; rms_naive_seed_std = {}
for B_ in BEAMS:
    tr_b = truth_beam[B_['fwhm']]
    for L, cols in LADDERS.items():
        for b, lbl in enumerate(MLBL):
            mean, _, _, sm_ = eval_truth(Bp_clean, tr_b, cols, b,
                                          n_seeds=N_SEEDS, n_mc=NMC,
                                          f_in=F_IN, f_out=F_OUT)
            tru = truth['fb'][b]
            rms_naive[(B_['fwhm'], L, lbl)] = float(np.sqrt(np.nanmean((mean - tru)**2)))
            rms_naive_seed_std[(B_['fwhm'], L, lbl)] = float(
                np.sqrt(np.nanmean((sm_ - tru[None, :])**2, 1)).std(ddof=1))

print('beam-naive (train on clean) vs beam-aware: RMS|pred-truth| in f_b (×1e3), Y+kSZ')
print(f'{"beam":24s} {"bin":14s} {"naive":>14s} {"aware":>14s} {"Δ":>8s}')
for B_ in BEAMS:
    for lbl in MLBL:
        n = rms_naive[(B_['fwhm'], 'Y+kSZ', lbl)] * 1e3
        en = rms_naive_seed_std[(B_['fwhm'], 'Y+kSZ', lbl)] * 1e3
        a = rms[(B_['fwhm'], 'Y+kSZ', lbl)] * 1e3
        ea = rms_seed_std[(B_['fwhm'], 'Y+kSZ', lbl)] * 1e3
        print(f"{B_['name']:24s} {lbl:14s}  {n:6.2f}±{en:4.2f}  {a:6.2f}±{ea:4.2f}  {a-n:+8.2f}")
""")
cells[i]['outputs'] = []
cells[i]['execution_count'] = None


# ---- Edit 3: insert radial Y vs Y+kSZ uplift overlay after #VSC-08d68240 ---
new_md1 = md("""\
### 4b. Radial structure of the kSZ uplift (ACT 1.6′ beam)
Where in the profile does adding kSZ help? Overlay the held-out reduction
curves for `Y` vs `Y+kSZ` at the 1.6′ beam, per mass bin.
""", cid='#VSC-uplift-md')
new_code1 = code("""\
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
ACT_FWHM = 1.6
for b, (ax, lbl) in enumerate(zip(axes, MLBL)):
    for L, c, ls in [('Y', 'C0', '--'), ('Y+kSZ', 'C3', '-')]:
        red = red_per_beam[(ACT_FWHM, L, lbl)]['red']
        ax.plot(r, red*100, ls, marker='o', color=c, ms=4, lw=1.6, label=L)
    ax.axhline(0, c='0.5', lw=0.6); ax.set_xscale('log')
    ax.set_xlabel('r [kpc/h]'); ax.set_title(lbl); ax.set_ylim(-20, 100)
axes[0].set_ylabel('held-out scatter reduction [%]'); axes[0].legend(fontsize=8)
fig.suptitle(f'kSZ radial uplift at FWHM={ACT_FWHM}′ beam-aware training\\n'
             '(curves = held-out reduction in f_b(r); kSZ helps where Y dies)')
fig.tight_layout(); fig.savefig(FIG/'ksz_radial_uplift.png', dpi=130)
print('saved ksz_radial_uplift.png')
""", cid='#VSC-uplift-code')
i = find('reduction_vs_beam.png')
cells[i+1:i+1] = [new_md1, new_code1]


# ---- Edit 4: σ_mc sweep at fixed 1.6′ after #VSC-87145651 (info-loss plot) -
new_md2 = md("""\
### 5b. σ_mc sweep at fixed 1.6′ beam
At realistic FWHM=1.6′ the kernel is dominated by miscentering (σ_mc=150 kpc/h
adds 150 kpc/h on top of the 53 kpc/h beam). Vary σ_mc with the beam fixed at
1.6′ to expose the centering-accuracy budget — the experimentally actionable
"how well must I centre my stack?" curve.
""", cid='#VSC-mc-md')
new_code2 = code("""\
SIGMA_MC_GRID = [0.0, 50.0, 100.0, 150.0, 250.0, 400.0]
ACT_FWHM = 1.6
sigma_beam = ACT_FWHM * 78.0 * FWHM2SIG
def _sigma_total_for(sigma_mc):
    return float(np.sqrt(sigma_beam**2 + sigma_mc**2))

# Pre-build smoothed designs and smoothed truth at each σ_total
Bp_mc, tr_mc, sig_arr = {}, {}, []
for sm_kpc in SIGMA_MC_GRID:
    st = _sigma_total_for(sm_kpc); sig_arr.append(st)
    Bp_mc[sm_kpc] = smooth_designs(Bp_clean, st)
    tr_mc[sm_kpc] = smooth_truth(truth, st)

rms_mc = {}; rms_mc_err = {}
for sm_kpc in SIGMA_MC_GRID:
    for L, cols in LADDERS.items():
        for b, lbl in enumerate(MLBL):
            mean, _, _, sm_ = eval_truth(Bp_mc[sm_kpc], tr_mc[sm_kpc], cols, b,
                                          n_seeds=N_SEEDS, n_mc=NMC,
                                          f_in=F_IN, f_out=F_OUT)
            tru = truth['fb'][b]
            rms_mc[(sm_kpc, L, lbl)] = float(np.sqrt(np.nanmean((mean - tru)**2)))
            rms_mc_err[(sm_kpc, L, lbl)] = float(
                np.sqrt(np.nanmean((sm_ - tru[None, :])**2, 1)).std(ddof=1))

print(f'σ_mc sweep at FWHM={ACT_FWHM}′, Y+kSZ; RMS×1e3 ± seed scatter')
print(f'{"σ_mc [kpc/h]":>14s} {"σ_total":>10s}'+''.join(f'{l:>16s}' for l in MLBL))
for sm_kpc, st in zip(SIGMA_MC_GRID, sig_arr):
    row = f'{sm_kpc:14.0f} {st:10.1f}'
    for lbl in MLBL:
        v = rms_mc[(sm_kpc, 'Y+kSZ', lbl)]*1e3
        e = rms_mc_err[(sm_kpc, 'Y+kSZ', lbl)]*1e3
        row += f'  {v:6.2f}±{e:4.2f}'
    print(row)

fig, ax = plt.subplots(figsize=(7, 4.5))
mc_arr = np.array(SIGMA_MC_GRID)
for b, (lbl, c) in enumerate(zip(MLBL, BIN_COLORS)):
    rms_curve = np.array([rms_mc[(sm, 'Y+kSZ', lbl)] for sm in SIGMA_MC_GRID]) * 1e3
    err_curve = np.array([rms_mc_err[(sm, 'Y+kSZ', lbl)] for sm in SIGMA_MC_GRID]) * 1e3
    ax.errorbar(mc_arr, rms_curve, yerr=err_curve, fmt='o-', color=c,
                label=f'{lbl}  Y+kSZ', capsize=3)
    sm_floor = np.nanmean(np.nanstd(Bp_clean['fb'][:, b, :], 0)) * 1e3
    ax.axhline(sm_floor, ls='--', c=c, lw=0.8, alpha=0.7,
               label=f'{lbl}  $\\sigma_\\mathrm{{marg}}$ floor')
ax.set_xlabel(f'miscentering $\\sigma_{{mc}}$ [kpc/h]   (FWHM={ACT_FWHM}′ fixed)')
ax.set_ylabel('RMS|pred-truth| in $f_b$  [×10$^{-3}$]')
ax.set_title('Information loss vs miscentering at fixed ACT beam (Y+kSZ, beam-aware)')
ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(FIG/'info_loss_vs_sigma_mc.png', dpi=130)
print('saved info_loss_vs_sigma_mc.png')
""", cid='#VSC-mc-code')
i = find('info_loss_vs_beam.png')
cells[i+1:i+1] = [new_md2, new_code2]


# ---- Update §7 JSON to include σ_mc sweep + seed errors ------------------
i = find("json.dump(out, open(FIG/'beam_aware_results.json'")
cells[i]['source'] = src_lines("""\
import json
out = {
    'r_kpc': r.tolist(), 'mass_bins': MLBL,
    'pixel_scale_kpc_h': float(PIX_KPC_H), 'sigma_mc_kpc_h': 150.0,
    'beams': [{'name': B_['name'], 'fwhm_arcmin': B_['fwhm'], 'sigma_total_kpc_h': B_['sigma']}
              for B_ in BEAMS],
    'noise': dict(f_in=F_IN, f_out=F_OUT, n_mc=NMC, n_seeds=N_SEEDS),
    'design_reduction_vs_r': {
        f"{B_['fwhm']}|{L}|{lbl}": red_per_beam[(B_['fwhm'], L, lbl)]['red'].tolist()
        for B_ in BEAMS for L in LADDERS for lbl in MLBL},
    'rms_truth_pred_aware': {
        f"{B_['fwhm']}|{L}|{lbl}": rms[(B_['fwhm'], L, lbl)]
        for B_ in BEAMS for L in LADDERS for lbl in MLBL},
    'rms_truth_pred_aware_seed_std': {
        f"{B_['fwhm']}|{L}|{lbl}": rms_seed_std[(B_['fwhm'], L, lbl)]
        for B_ in BEAMS for L in LADDERS for lbl in MLBL},
    'rms_truth_pred_naive': {
        f"{B_['fwhm']}|{L}|{lbl}": rms_naive[(B_['fwhm'], L, lbl)]
        for B_ in BEAMS for L in LADDERS for lbl in MLBL},
    'rms_truth_pred_naive_seed_std': {
        f"{B_['fwhm']}|{L}|{lbl}": rms_naive_seed_std[(B_['fwhm'], L, lbl)]
        for B_ in BEAMS for L in LADDERS for lbl in MLBL},
    'sigma_mc_sweep': {
        'fwhm_arcmin': ACT_FWHM, 'sigma_mc_grid_kpc_h': list(SIGMA_MC_GRID),
        'sigma_total_kpc_h': sig_arr,
        'rms': {f"{sm}|{L}|{lbl}": rms_mc[(sm, L, lbl)]
                for sm in SIGMA_MC_GRID for L in LADDERS for lbl in MLBL},
        'rms_seed_std': {f"{sm}|{L}|{lbl}": rms_mc_err[(sm, L, lbl)]
                         for sm in SIGMA_MC_GRID for L in LADDERS for lbl in MLBL},
    },
    'truth_fb': {l: truth['fb'][b].tolist() for b, l in enumerate(MLBL)},
}
json.dump(out, open(FIG/'beam_aware_results.json', 'w'), indent=2)
print('wrote', FIG/'beam_aware_results.json')
""")
cells[i]['outputs'] = []
cells[i]['execution_count'] = None


NB.write_text(json.dumps(nb, indent=1))
print(f'updated {NB} -> {len(cells)} cells')
