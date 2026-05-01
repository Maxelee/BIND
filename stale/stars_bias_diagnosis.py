"""Stars-bias sign-flip diagnostic — runs three tests to localize the cause.

Hypotheses from paper_figures Fig 2 observation (CV median ≈ −0.09,
SB35 median ≈ +0.11):

  1. CV-calibrated Stars correction is cosmology/astro-dependent.
  2. Halo-population shift via cosmology (SB35 picks lower-mass halos).
  3. BIND2 under-tracks feedback response amplitude.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SUITE_ROOT = Path('/mnt/home/mlee1/ceph/fm_testsuite')
SNAP = 'snap_090'
MASS_TAG = 'mass_threshold_1p000e13'
MODEL_NAME = 'fm_base'
BOX_SIZE = 50.0
N_PIX_FULL = 1024
PATCH_PIX = 128
N_PARAMS = 35

_param_meta = pd.read_csv(
    '/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv'
)
PARAM_NAMES = {i + 1: name for i, name in enumerate(_param_meta['ParamName'])}
STARS = 2  # channel index


def sim_record(sd: Path, suite: str) -> dict:
    snap = sd / SNAP
    mass = snap / MASS_TAG
    model = mass / MODEL_NAME
    r = {
        'suite': suite, 'sim_id': sd.name,
        'full_maps': snap / 'full_maps.npz',
        'halo_catalog': mass / 'halo_catalog.npz',
        'generated': model / 'generated_halos.npz',
    }
    r['available'] = all(r[k].exists() for k in ('full_maps', 'halo_catalog', 'generated'))
    return r


def discover(suites=('CV', '1P', 'Test')):
    recs = []
    for s in suites:
        root = SUITE_ROOT / s
        if not root.exists():
            continue
        for sd in sorted(root.iterdir()):
            if sd.is_dir():
                recs.append(sim_record(sd, s))
    return pd.DataFrame(recs)


def extract_patch(field_2d, cx_pix, cy_pix, size=PATCH_PIX):
    n = field_2d.shape[0]
    h = size // 2
    ix = (cx_pix - h + np.arange(size)) % n
    iy = (cy_pix - h + np.arange(size)) % n
    return field_2d[np.ix_(ix, iy)]


def per_sim_payload(rec):
    """Return dict of per-halo arrays for Stars: truth_mass, gen_mass,
    halo_mass, and params (35,)."""
    cat = np.load(rec['halo_catalog'])
    gen = np.load(rec['generated'])['generated']           # (N, 3, P, P)
    truth_full = np.load(rec['full_maps'])['truth_maps']   # (3, N_PIX, N_PIX)
    centers = cat['centers']
    masses = cat['masses']
    params = (
        cat['params'][0].astype(np.float64) if 'params' in cat.files else None
    )
    ppm = N_PIX_FULL / BOX_SIZE
    truth_s = np.zeros(len(centers), dtype=np.float64)
    for i, (cx, cy) in enumerate(centers):
        cxp = int(cx * ppm) % N_PIX_FULL
        cyp = int(cy * ppm) % N_PIX_FULL
        truth_s[i] = extract_patch(truth_full[STARS], cxp, cyp).sum()
    gen_s = gen[:, STARS, :, :].sum(axis=(1, 2)).astype(np.float64)
    return {
        'truth_s': truth_s, 'gen_s': gen_s,
        'halo_mass': np.asarray(masses, np.float64), 'params': params,
    }


# ---------- build halo-level & sim-level tables ----------
sims = discover()
sims = sims[sims['available']].reset_index(drop=True)
print(f'[discover] CV={sum(sims.suite=="CV")}, 1P={sum(sims.suite=="1P")}, '
      f'Test={sum(sims.suite=="Test")}')

halo_rows = []
sim_rows = []
for rec in sims.to_dict('records'):
    try:
        p = per_sim_payload(rec)
    except Exception as exc:
        print(f'[skip] {rec["suite"]}/{rec["sim_id"]}: {exc}')
        continue
    n = len(p['halo_mass'])
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = (p['gen_s'] - p['truth_s']) / p['truth_s']
        ratio = p['truth_s'] / p['gen_s']
    for i in range(n):
        halo_rows.append({
            'suite': rec['suite'], 'sim_id': rec['sim_id'],
            'halo_mass': p['halo_mass'][i],
            'log_halo': np.log10(p['halo_mass'][i]),
            'truth_s': p['truth_s'][i], 'gen_s': p['gen_s'][i],
            'rel_stars': rel[i], 'ratio_stars': ratio[i],
        })
    row = {
        'suite': rec['suite'], 'sim_id': rec['sim_id'],
        'n_halos': n,
        'median_rel_stars': np.nanmedian(rel),
        'median_ratio_stars': np.nanmedian(ratio),
        'median_log_halo': np.nanmedian(np.log10(p['halo_mass'])),
        'n_halos_above_threshold': n,
    }
    if p['params'] is not None:
        for j in range(N_PARAMS):
            row[f'p{j+1}'] = p['params'][j]
    sim_rows.append(row)

halo_tbl = pd.DataFrame(halo_rows)
sim_tbl = pd.DataFrame(sim_rows)
print(f'[tables] halos={len(halo_tbl)}  sims={len(sim_tbl)}')


# =============================================================
# TEST 1 — Is the Stars correction factor parameter-dependent?
# =============================================================
print('\n' + '=' * 60)
print('TEST 1 — Stars correction factor vs 35 parameters (SB35)')
print('=' * 60)
sb35 = sim_tbl[sim_tbl['suite'] == 'Test'].copy()
print(f'SB35 sims: {len(sb35)}')
cv_med = sim_tbl[sim_tbl['suite'] == 'CV']['median_ratio_stars'].median()
p1_med = sim_tbl[sim_tbl['suite'] == '1P']['median_ratio_stars'].median()
sb_med = sb35['median_ratio_stars'].median()
print(f'CV   median ratio(truth/gen):  {cv_med:.4f}')
print(f'1P   median ratio(truth/gen):  {p1_med:.4f}')
print(f'SB35 median ratio(truth/gen):  {sb_med:.4f}')
print(f'SB35 ratio std across sims:    {sb35["median_ratio_stars"].std():.4f}')

y = sb35['median_ratio_stars'].to_numpy()
corrs = []
for j in range(1, N_PARAMS + 1):
    pname = f'p{j}'
    if pname not in sb35:
        continue
    x = sb35[pname].to_numpy()
    # use Spearman (robust, doesn't assume linearity)
    rho, pval = spearmanr(x, y, nan_policy='omit')
    corrs.append({'param': pname, 'name': PARAM_NAMES.get(j, pname),
                  'rho': rho, 'pval': pval})
corr_df = pd.DataFrame(corrs).sort_values('pval').reset_index(drop=True)
top = corr_df.head(10)
print('\nTop 10 params by |Spearman ρ| significance (SB35 sim-level):')
print(top.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
n_sig = int((corr_df['pval'] < 0.05).sum())
print(f'\n→ {n_sig} / {len(corr_df)} params show p<0.05 correlation with Stars ratio')


# =============================================================
# TEST 2 — Halo-mass-dependent bias + distribution shift
# =============================================================
print('\n' + '=' * 60)
print('TEST 2 — Halo-mass dependence of Stars bias, and halo-mass shift')
print('=' * 60)
for s in ['CV', '1P', 'Test']:
    sub = halo_tbl[halo_tbl['suite'] == s]
    logm = sub['log_halo']
    rel = sub['rel_stars']
    rho, pval = spearmanr(logm, rel, nan_policy='omit')
    print(f'  {s:>5}  n={len(sub):5d}  '
          f'median log10M = {logm.median():.3f}  '
          f'mean log10M = {logm.mean():.3f}  '
          f'Spearman(logM, rel_stars) = {rho:+.3f} (p={pval:.2e})')

print('\nStars rel-error median per log10M bin, per suite:')
bins = [13.0, 13.2, 13.5, 13.8, 14.2, 16.0]
labels = [f'[{bins[i]:.1f},{bins[i+1]:.1f})' for i in range(len(bins) - 1)]
halo_tbl['log_bin'] = pd.cut(halo_tbl['log_halo'], bins=bins, labels=labels,
                              include_lowest=True)
pivot = halo_tbl.pivot_table(
    index='log_bin', columns='suite', values='rel_stars',
    aggfunc='median', observed=True,
)
print(pivot.to_string(float_format=lambda v: f'{v:+.3f}'))
count_pivot = halo_tbl.pivot_table(
    index='log_bin', columns='suite', values='rel_stars',
    aggfunc='count', observed=True,
)
print('\nHalo counts per bin (shift of halo-mass dist is a big-deal signal):')
print(count_pivot.to_string())


# =============================================================
# TEST 3 — Does BIND2 under-track feedback response amplitude?
# =============================================================
print('\n' + '=' * 60)
print('TEST 3 — Feedback response amplitude (1P amp_ratio on Stars)')
print('=' * 60)

oneP_tbl = sim_tbl[sim_tbl['suite'] == '1P']
feedback_params = [3, 4, 5, 6]  # A_SN1, A_SN2, A_AGN1, A_AGN2

def most_massive_diff_mass(rec_hi, rec_lo, channel=STARS):
    """Return (truth_diff_mass, gen_diff_mass) at the most massive halo in
    each sim (variant-2 halo minus variant-n2 halo patch integrals)."""
    p_hi = per_sim_payload(rec_hi)
    p_lo = per_sim_payload(rec_lo)
    i_hi = int(np.argmax(p_hi['halo_mass']))
    i_lo = int(np.argmax(p_lo['halo_mass']))
    return (p_hi['truth_s'][i_hi] - p_lo['truth_s'][i_lo],
            p_hi['gen_s'][i_hi] - p_lo['gen_s'][i_lo])


def find_1p_pair(j):
    hi = sims.query(f"sim_id == '1P_p{j}_2'")
    lo = sims.query(f"sim_id == '1P_p{j}_n2'")
    if len(hi) == 0 or len(lo) == 0:
        return None, None
    return hi.iloc[0].to_dict(), lo.iloc[0].to_dict()


# (a) patch-level diff-amplitude ratio on most-massive halo
print('\n(a) Patch-level integrated Stars response on most-massive halo:')
print(f'{"p":>3} {"param":>22}  {"ΔTruth":>12}  {"ΔBIND2":>12}  {"amp":>7}  {"sign":>5}')
amp_rows = []
for j in range(1, N_PARAMS + 1):
    hi, lo = find_1p_pair(j)
    if hi is None:
        continue
    try:
        dt, dg = most_massive_diff_mass(hi, lo)
    except Exception:
        continue
    amp = dg / dt if dt != 0 else np.nan
    same_sign = np.sign(dt) == np.sign(dg) if dt != 0 and dg != 0 else False
    amp_rows.append({
        'p': j, 'name': PARAM_NAMES.get(j, f'p{j}'),
        'truth_diff': dt, 'gen_diff': dg,
        'amp_ratio': amp, 'same_sign': bool(same_sign),
    })
    if j in feedback_params:
        tag = '  ← feedback'
    else:
        tag = ''
    print(f'{"p"+str(j):>3} {PARAM_NAMES.get(j,"")[:22]:>22}  '
          f'{dt:+12.3e}  {dg:+12.3e}  '
          f'{(str(round(amp,3)) if np.isfinite(amp) else "nan"):>7}  '
          f'{str(same_sign):>5}{tag}')

amp_df = pd.DataFrame(amp_rows)
print('\nFeedback-param summary:')
if len(amp_df):
    fb = amp_df[amp_df['p'].isin(feedback_params)]
    print(fb[['p', 'name', 'truth_diff', 'gen_diff', 'amp_ratio']]
          .to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    print(f'\nMean amp_ratio for feedback params: {fb["amp_ratio"].mean():+.3f}')
    print(f'Mean amp_ratio for non-feedback:    '
          f'{amp_df[~amp_df["p"].isin(feedback_params)]["amp_ratio"].mean():+.3f}')

# (b) BONUS: does Stars ratio shift with 1P parameter distance from fiducial?
print('\n(b) Per-sim Stars ratio vs 1P parameter variant ("2" vs "n2"):')
for j in feedback_params:
    for variant in ['2', 'n2']:
        sub = oneP_tbl[oneP_tbl['sim_id'] == f'1P_p{j}_{variant}']
        if len(sub):
            r = sub['median_ratio_stars'].iloc[0]
            logm = sub['median_log_halo'].iloc[0]
            print(f'  1P_p{j}_{variant:>3}  {PARAM_NAMES.get(j,"")[:20]:>20}  '
                  f'ratio={r:.4f}   median_log_halo={logm:.3f}')


# =============================================================
# SUMMARY
# =============================================================
print('\n' + '=' * 60)
print('SUMMARY — which hypothesis survives?')
print('=' * 60)

cv_ratio = sim_tbl[sim_tbl['suite'] == 'CV']['median_ratio_stars'].median()
sb_ratio = sb35['median_ratio_stars'].median()
sb_spread = sb35['median_ratio_stars'].std()
dmed_cv_sb = cv_ratio - sb_ratio

print(f'CV  median per-sim ratio(truth/gen) : {cv_ratio:.4f}')
print(f'SB35 median                         : {sb_ratio:.4f}')
print(f'Δ(CV − SB35)                        : {dmed_cv_sb:+.4f}')
print(f'SB35 per-sim ratio sim-to-sim std   : {sb_spread:.4f}')
print()
print(f'H1 (param-dependent correction): {n_sig} params with p<0.05 Spearman '
      f'vs Stars ratio on SB35 (/{len(corr_df)})')
print(f'H2 (halo-mass shift): See Test 2 table — if CV vs SB35 medians per '
      f'halo-mass bin are similar, halo-mass shift alone is not the cause.')
if len(amp_df):
    fb_amp_mean = amp_df[amp_df['p'].isin(feedback_params)]['amp_ratio'].mean()
    print(f'H3 (feedback amp under-tracking): feedback-param mean amp_ratio = '
          f'{fb_amp_mean:+.3f} (≈1 = good tracking; <1 = BIND2 under-responds)')

# Persist full tables for further inspection
out = Path('/mnt/home/mlee1/vdm_bind2/stars_bias_diagnosis_tables')
out.mkdir(exist_ok=True)
sim_tbl.to_csv(out / 'sim_tbl.csv', index=False)
halo_tbl.to_csv(out / 'halo_tbl.csv', index=False)
corr_df.to_csv(out / 'test1_sb35_param_correlations.csv', index=False)
if len(amp_df):
    amp_df.to_csv(out / 'test3_1p_amp_ratios.csv', index=False)
print(f'\nTables written to {out}/')
