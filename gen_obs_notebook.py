"""
Generate analysis_observables.ipynb — Physical Observables: Truth vs. BIND2

Run with:  python gen_obs_notebook.py
"""
from pathlib import Path
import nbformat as nbf

c = nbf.v4


def md(src):
    return c.new_markdown_cell(src)


def code(src):
    return c.new_code_cell(src)


# ─────────────────────────────────────────────────────────────────────────────
SETUP = """\
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import binned_statistic
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except ImportError:
    pass

sys.path.insert(0, '/mnt/home/mlee1/vdm_bind2')

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,
})

# ── paths ──────────────────────────────────────────────────────────────────
SUITE_ROOT  = Path('/mnt/home/mlee1/ceph/fm_testsuite')
SNAP        = 'snap_090'
MASS_TAG    = 'mass_threshold_1p000e13'
MODEL_NAME  = 'fm_two_head'
BOX_SIZE    = 50.0
N_PIX_FULL  = 1024
PATCH_PIX   = 128
PATCH_BOX   = BOX_SIZE * PATCH_PIX / N_PIX_FULL   # 6.25 Mpc/h
MPC_PER_PIX = PATCH_BOX / PATCH_PIX
N_PARAMS    = 35

FIG_DIR   = Path('paper_figures')
FIG_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path('analysis_physics_cache')
CACHE_DIR.mkdir(exist_ok=True)

# ── suite style ────────────────────────────────────────────────────────────
SUITES        = ('CV', '1P', 'Test')
SUITE_COLORS  = {'CV': 'tab:green', '1P': 'tab:blue', 'Test': 'tab:red'}
SUITE_DISPLAY = {'CV': 'CV', '1P': '1P', 'Test': 'SB35'}

# ── observable metadata ────────────────────────────────────────────────────
OBS_KEYS = [
    'M_dm', 'M_gas', 'M_star',
    'f_b', 'f_b_norm',
    'Rc_over_R200',
    'q_DM', 'q_gas', 'q_star',
    'dq_DM',
    'Sigma_gas_c',
]
OBS_LATEX = {
    'M_dm':         r'$M_{\\rm DM}(<R_{200})$',
    'M_gas':        r'$M_{\\rm gas}(<R_{200})$',
    'M_star':       r'$M_\\star(<R_{200})$',
    'f_b':          r'$f_b(<R_{200})$',
    'f_b_norm':     r'$f_b / f_{b,{\\rm cos}}$',
    'Rc_over_R200': r'$R_{\\rm closure}/R_{200}$',
    'q_DM':         r'$q_{\\rm DM}$',
    'q_gas':        r'$q_{\\rm gas}$',
    'q_star':       r'$q_\\star$',
    'dq_DM':        r'$\\Delta q_{\\rm DM}$',
    'Sigma_gas_c':  r'$\\Sigma_{\\rm gas,c}$',
}

# use relative error for positive-definite quantities; absolute for ratios/fractions
OBS_RELATIVE_ERR = {
    'M_dm': True, 'M_gas': True, 'M_star': True,
    'f_b': False, 'f_b_norm': False,
    'Rc_over_R200': False,
    'q_DM': False, 'q_gas': False, 'q_star': False,
    'dq_DM': False,
    'Sigma_gas_c': True,
}

def save_fig(fig, name, ext=('pdf', 'png')):
    for e in ext:
        out = FIG_DIR / f'{name}.{e}'
        fig.savefig(out)
        print(f'  wrote {out}')

print(f'MODEL: {MODEL_NAME}')
print(f'Patch box: {PATCH_BOX:.3f} Mpc/h  (1 pix = {MPC_PER_PIX*1000:.1f} kpc/h)')
"""

SIM_DISCOVERY = """\
def sim_record(sim_dir: Path, suite_name: str) -> dict:
    snap  = sim_dir / SNAP
    mass  = snap / MASS_TAG
    model = mass / MODEL_NAME
    rec   = {
        'suite':        suite_name,
        'sim_id':       sim_dir.name,
        'sim_dir':      sim_dir,
        'full_maps':    snap / 'full_maps.npz',
        'halo_catalog': mass / 'halo_catalog.npz',
        'cutouts':      mass / 'halo_cutouts.npz',
        'generated':    model / 'generated_halos.npz',
    }
    rec['available'] = all(
        rec[k].exists()
        for k in ('full_maps', 'halo_catalog', 'cutouts', 'generated')
    )
    return rec


def discover_sims(suites=SUITES) -> pd.DataFrame:
    recs = []
    for suite in suites:
        root = SUITE_ROOT / suite
        if not root.exists():
            print(f'[skip] {root} missing')
            continue
        for sd in sorted(root.iterdir()):
            if sd.is_dir():
                recs.append(sim_record(sd, suite))
    return pd.DataFrame(recs)


sims_all = discover_sims()
sims     = sims_all[sims_all['available']].reset_index(drop=True)
cv_sims   = sims[sims['suite'] == 'CV'].reset_index(drop=True)
oneP_sims = sims[sims['suite'] == '1P'].reset_index(drop=True)
test_sims = sims[sims['suite'] == 'Test'].reset_index(drop=True)
print(f'Available sims: CV={len(cv_sims)}  1P={len(oneP_sims)}  SB35={len(test_sims)}')
"""

PHYSICS_HELPERS = """\
# ── geometry constants — MUST match fd_jacobian_cv.py exactly ──────────────
RHO_CRIT          = 2.775e11     # M_sun/h per (Mpc/h)^3
OMEGA_B_FIXED     = 0.049
CLOSURE_THRESHOLD = 0.90
_NB               = 32

_yy, _xx       = (np.mgrid[:PATCH_PIX, :PATCH_PIX]
                  - np.array([PATCH_PIX / 2, PATCH_PIX / 2])[:, None, None])
_RR_PIX        = np.sqrt(_xx ** 2 + _yy ** 2)
_BIN_EDGES_PIX = np.linspace(0, PATCH_PIX / 2, _NB + 1)
_R_CENTRES_PIX = 0.5 * (_BIN_EDGES_PIX[:-1] + _BIN_EDGES_PIX[1:])
_BIN_MASKS     = [
    (_RR_PIX >= _BIN_EDGES_PIX[k]) & (_RR_PIX < _BIN_EDGES_PIX[k + 1])
    for k in range(_NB)
]
_N_PIX_PER_BIN = np.array([m.sum() for m in _BIN_MASKS], dtype=np.float64)
_R_MPC         = _R_CENTRES_PIX * MPC_PER_PIX     # radial bin centres in Mpc/h


def r200c_pix(m200c_msunh):
    \"\"\"R200c in patch pixels from halo mass [M_sun/h].\"\"\"
    r_mpc = (3.0 * m200c_msunh / (4.0 * np.pi * 200.0 * RHO_CRIT)) ** (1.0 / 3.0)
    return r_mpc / MPC_PER_PIX


def aperture_sum(field_2d, r_pix):
    \"\"\"Sum of positive pixel values within a circular aperture of radius r_pix.\"\"\"
    return float(np.maximum(field_2d, 0.0)[_RR_PIX < r_pix].sum())


def _radial_profile_2d(field_2d):
    \"\"\"Azimuthal mean in _NB annular bins from centre to PATCH_PIX/2.\"\"\"
    return np.array([
        field_2d[m].mean() if c > 0 else 0.0
        for m, c in zip(_BIN_MASKS, _N_PIX_PER_BIN)
    ])


def axis_ratio_q(field_2d, r_aper_pix, max_iter=5, tol=1e-3, min_pixels=8):
    \"\"\"
    Iterative ellipsoidal moment axis ratio q = b/a in [0, 1].

    Uses mass-weighted 2D quadrupole moments of the positive flux within an
    adaptive elliptical aperture (semi-major axis = r_aper_pix, semi-minor
    axis = q * r_aper_pix).  Returns NaN for empty or insufficiently resolved
    fields.
    \"\"\"
    H, W   = field_2d.shape
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    f      = field_2d.astype(np.float64)
    q, pa  = 1.0, 0.0
    for _ in range(max_iter):
        c_r, s_r = np.cos(pa), np.sin(pa)
        x_rot = c_r * dx + s_r * dy
        y_rot = -s_r * dx + c_r * dy
        a, b  = r_aper_pix, max(r_aper_pix * q, 1.0)
        mask  = (x_rot / a) ** 2 + (y_rot / b) ** 2 < 1.0
        if mask.sum() < min_pixels:
            return np.nan
        w   = np.where(mask, np.maximum(f, 0.0), 0.0)
        tot = w.sum()
        if tot <= 0:
            return np.nan
        Qxx = (dx * dx * w).sum() / tot
        Qyy = (dy * dy * w).sum() / tot
        Qxy = (dx * dy * w).sum() / tot
        evals, evecs = np.linalg.eigh(np.array([[Qxx, Qxy], [Qxy, Qyy]]))
        lam_min, lam_max = float(evals[0]), float(evals[1])
        if lam_max <= 0 or lam_min < 0:
            return np.nan
        q_new  = float(np.sqrt(lam_min / lam_max))
        pa_new = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
        if abs(q_new - q) < tol:
            return q_new
        q, pa  = q_new, pa_new
    return q


def closure_radius_pix(p_dm, p_gas, p_star, f_b_cosmic,
                       threshold=CLOSURE_THRESHOLD):
    \"\"\"
    Smallest radius (in pixels) at which the enclosed baryon fraction
    f_b(< r) first reaches `threshold` * f_b_cosmic.

    Computed from cumulative sums of the azimuthal-mean profiles weighted by
    annular pixel area.  Returns NaN when the ratio never exceeds the threshold
    within the patch.
    \"\"\"
    cum_dm   = np.cumsum(p_dm   * _N_PIX_PER_BIN)
    cum_gas  = np.cumsum(p_gas  * _N_PIX_PER_BIN)
    cum_star = np.cumsum(p_star * _N_PIX_PER_BIN)
    cum_tot  = cum_dm + cum_gas + cum_star
    cum_b    = cum_gas + cum_star
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cum_tot > 0, cum_b / cum_tot, np.nan) / f_b_cosmic
    ok = np.isfinite(ratio) & (ratio >= threshold)
    if not ok.any():
        return np.nan
    k = int(np.argmax(ok))
    if k > 0 and np.isfinite(ratio[k - 1]):
        lo, hi = ratio[k - 1], ratio[k]
        if hi > lo:
            frac = (threshold - lo) / (hi - lo)
            return (1 - frac) * _R_CENTRES_PIX[k - 1] + frac * _R_CENTRES_PIX[k]
    return _R_CENTRES_PIX[k]


def observables_from_phys(phys_3HW, r200_pix, f_b_cosmic, q_DMO_const):
    \"\"\"
    Compute all 11 scalar observables from a (3, 128, 128) physical-unit map.

    Parameters
    ----------
    phys_3HW   : (3, H, W) array  [DM | Gas | Stars] in M_sun/h per pixel
    r200_pix   : float   R200c in patch pixels
    f_b_cosmic : float   Omega_b / Omega_m for this simulation
    q_DMO_const: float   axis ratio of the DMO condition map (for dq_DM)
    \"\"\"
    dm, gas, star = phys_3HW[0], phys_3HW[1], phys_3HW[2]
    r_aper = max(min(r200_pix, PATCH_PIX / 2 - 2), 4.0)

    # 1–3. Enclosed masses within R200c
    M_dm   = aperture_sum(dm,   r_aper)
    M_gas  = aperture_sum(gas,  r_aper)
    M_star = aperture_sum(star, r_aper)
    M_tot  = M_dm + M_gas + M_star
    M_b    = M_gas + M_star

    # 4–5. Baryon fractions
    f_b      = M_b / M_tot if M_tot > 0 else np.nan
    f_b_norm = (f_b / f_b_cosmic
                if (M_tot > 0 and np.isfinite(f_b_cosmic)) else np.nan)

    # 6. Closure radius
    p_dm   = _radial_profile_2d(np.maximum(dm,   0.0))
    p_gas  = _radial_profile_2d(np.maximum(gas,  0.0))
    p_star = _radial_profile_2d(np.maximum(star, 0.0))
    Rc_pix = closure_radius_pix(p_dm, p_gas, p_star, f_b_cosmic)
    Rc_over_R200 = Rc_pix / r200_pix if np.isfinite(Rc_pix) else np.nan

    # 7–9. Projected axis ratios
    q_dm   = axis_ratio_q(dm,   r_aper)
    q_gas  = axis_ratio_q(gas,  r_aper)
    q_star = axis_ratio_q(star, r_aper)

    # 10. DM back-reaction in shape relative to DMO
    dq_DM = q_dm - q_DMO_const

    # 11. Central gas surface density within 0.1 R200c
    r_c = max(0.1 * r200_pix, 2.0)
    Sigma_gas_c = float(np.maximum(gas, 0.0)[_RR_PIX < r_c].mean())

    return {
        'M_dm': M_dm, 'M_gas': M_gas, 'M_star': M_star,
        'f_b': f_b, 'f_b_norm': f_b_norm,
        'Rc_over_R200': Rc_over_R200,
        'q_DM': q_dm, 'q_gas': q_gas, 'q_star': q_star,
        'dq_DM': dq_DM, 'Sigma_gas_c': Sigma_gas_c,
    }

print('Physics helpers ready.')
"""

CACHE_BUILD = """\
CACHE_FILE = CACHE_DIR / f'obs_{MODEL_NAME}.npz'


def _extract_patch(field_2d, cx, cy, size=PATCH_PIX):
    \"\"\"Periodic patch extraction from a 2D full-box map.\"\"\"
    n    = field_2d.shape[0]
    half = size // 2
    ix   = (cx - half + np.arange(size)) % n
    iy   = (cy - half + np.arange(size)) % n
    return field_2d[np.ix_(ix, iy)]


def _centers_to_pixels(centers_mpc):
    ppm = N_PIX_FULL / BOX_SIZE
    return (np.asarray(centers_mpc) * ppm).astype(np.int64) % N_PIX_FULL


def _ingest_one(rec):
    \"\"\"Compute 11 observables (truth and gen) for every halo in one simulation.\"\"\"
    fm   = np.load(rec['full_maps'])
    cat  = np.load(rec['halo_catalog'])
    cuts = np.load(rec['cutouts'])
    gen  = np.load(rec['generated'])['generated']    # (N, 3, 128, 128)

    centers_pix = _centers_to_pixels(cat['centers'])
    masses      = cat['masses'].astype(np.float64)
    params_arr  = (cat['params'].astype(np.float32)
                   if 'params' in cat.files
                   else np.full((len(masses), N_PARAMS), np.nan, np.float32))
    n = len(masses)

    truth_obs = {k: np.full(n, np.nan, np.float64) for k in OBS_KEYS}
    gen_obs   = {k: np.full(n, np.nan, np.float64) for k in OBS_KEYS}

    for i, (cx, cy) in enumerate(centers_pix):
        mass    = float(masses[i])
        omega_m = float(params_arr[i, 0]) if params_arr[i, 0] > 0 else 0.3
        r200p   = r200c_pix(mass)
        f_b_cos = OMEGA_B_FIXED / max(omega_m, 1e-10)
        r_aper  = max(min(r200p, PATCH_PIX / 2 - 2), 4.0)

        # q_DMO from the pre-extracted DMO condition map
        q_dmo = axis_ratio_q(np.maximum(cuts['condition'][i], 0.0), r_aper)

        # Truth: extract patch from full-box maps
        truth_patch = np.stack([
            _extract_patch(fm['truth_maps'][c], cx, cy) for c in range(3)
        ])
        t_obs = observables_from_phys(truth_patch, r200p, f_b_cos, q_dmo)
        g_obs = observables_from_phys(gen[i],       r200p, f_b_cos, q_dmo)

        for k in OBS_KEYS:
            truth_obs[k][i] = t_obs[k]
            gen_obs[k][i]   = g_obs[k]

    return {
        'suite':  np.array([rec['suite']]  * n),
        'sim_id': np.array([rec['sim_id']] * n),
        'logM':   np.log10(masses),
        'params': params_arr,
        **{f'truth_{k}': truth_obs[k] for k in OBS_KEYS},
        **{f'gen_{k}':   gen_obs[k]   for k in OBS_KEYS},
    }


def build_obs_cache(sims_df, force=False):
    if CACHE_FILE.exists() and not force:
        print(f'Loading cache: {CACHE_FILE}')
        z = np.load(CACHE_FILE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    print(f'Building observable cache for {len(sims_df)} sims...')
    parts = []
    for ki, rec in enumerate(sims_df.to_dict('records')):
        try:
            parts.append(_ingest_one(rec))
        except Exception as exc:
            print(f'  [skip] {rec["suite"]}/{rec["sim_id"]}: {exc}')
        if (ki + 1) % 10 == 0:
            print(f'  {ki + 1}/{len(sims_df)} sims done')

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    np.savez_compressed(CACHE_FILE, **merged)
    print(f'Wrote {CACHE_FILE}  ({CACHE_FILE.stat().st_size / 1e6:.1f} MB)  '
          f'{len(merged["logM"])} halos')
    return merged


cache     = build_obs_cache(sims, force=False)
suite_arr = cache['suite'].astype(str)
print(f'\\nTotal halos: {len(cache["logM"])}')
for s in SUITES:
    print(f'  {SUITE_DISPLAY[s]:>5s}: {(suite_arr == s).sum()}')
"""

DATAFRAME_BUILD = """\
def build_obs_df(cache):
    suite_arr = cache['suite'].astype(str)
    sim_arr   = cache['sim_id'].astype(str)
    df = pd.DataFrame({
        'suite':  suite_arr,
        'sim_id': sim_arr,
        'logM':   cache['logM'].astype(np.float64),
    })
    for j in range(N_PARAMS):
        df[f'p{j+1}'] = cache['params'][:, j]
    for k in OBS_KEYS:
        df[f'truth_{k}'] = cache[f'truth_{k}'].astype(np.float64)
        df[f'gen_{k}']   = cache[f'gen_{k}'].astype(np.float64)
    return df


obs_df = build_obs_df(cache)
print(f'obs_df shape: {obs_df.shape}')
print(obs_df.groupby('suite').size())
"""

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers

PLOT_HELPERS = """\
def _error_vals(df, obs_key, suite):
    \"\"\"Per-halo error for a given observable and suite.\"\"\"
    sub = df[df['suite'] == suite]
    t   = sub[f'truth_{obs_key}'].to_numpy()
    g   = sub[f'gen_{obs_key}'].to_numpy()
    if OBS_RELATIVE_ERR[obs_key]:
        with np.errstate(divide='ignore', invalid='ignore'):
            err = np.where(np.isfinite(t) & (t > 0), (g - t) / t, np.nan)
    else:
        err = np.where(np.isfinite(t) & np.isfinite(g), g - t, np.nan)
    return err[np.isfinite(err)]


def _bp_style(ax, data_list, positions, colors, clip=None):
    \"\"\"Draw boxplots with suite colours at the given positions.\"\"\"
    if clip is not None:
        data_list = [np.clip(d, -clip, clip) for d in data_list]
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color='k', linewidth=1.5),
        whiskerprops=dict(color='k', linewidth=0.8),
        capprops=dict(color='k', linewidth=0.8),
        flierprops=dict(marker='.', markersize=1.5, alpha=0.25, linestyle='none'),
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for flier, color in zip(bp['fliers'], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
    return bp


def error_boxplot(ax, obs_key, df=obs_df, clip=1.0):
    \"\"\"Boxplot of per-halo errors per suite.\"\"\"
    data = [_error_vals(df, obs_key, s) for s in SUITES]
    cols = [SUITE_COLORS[s] for s in SUITES]
    _bp_style(ax, data, list(range(len(SUITES))), cols, clip=clip)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_xticks(range(len(SUITES)))
    ax.set_xticklabels([SUITE_DISPLAY[s] for s in SUITES])
    err_type = r'$(F_{\\rm gen} - F_{\\rm truth})/F_{\\rm truth}$' if OBS_RELATIVE_ERR[obs_key] else r'$F_{\\rm gen} - F_{\\rm truth}$'
    ax.set_ylabel(err_type, fontsize=9)
    ax.set_ylim(-clip * 1.1, clip * 1.1)
    ax.grid(axis='y', alpha=0.25)
    # annotate medians
    for i, d in enumerate(data):
        if len(d):
            med = np.median(d)
            ax.text(i, clip * 0.92, f'{med:+.3f}',
                    ha='center', va='top', fontsize=7.5, alpha=0.9)


def scatter_two(ax, obs_key, df=obs_df, log_x=True, log_y=True):
    \"\"\"Truth vs. BIND2 scatter coloured by suite.\"\"\"
    for suite in SUITES:
        sub = df[df['suite'] == suite]
        t   = sub[f'truth_{obs_key}'].to_numpy()
        g   = sub[f'gen_{obs_key}'].to_numpy()
        mask = np.isfinite(t) & np.isfinite(g)
        if log_x:
            mask &= t > 0
        if log_y:
            mask &= g > 0
        tx = np.log10(t[mask]) if log_x else t[mask]
        gx = np.log10(g[mask]) if log_y else g[mask]
        ax.scatter(tx, gx, s=2, alpha=0.25, color=SUITE_COLORS[suite],
                   rasterized=True, label=SUITE_DISPLAY[suite])
    # identity line
    lo = ax.get_xlim()[0]; hi = ax.get_xlim()[1]
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)
    pre = r'$\\log_{10}\\,' if log_x else ''
    post = '$' if log_x else ''
    ax.set_xlabel(f'{pre}Truth{post}', fontsize=9)
    ax.set_ylabel(f'{pre}BIND2{post}', fontsize=9)


def running_median_overlay(ax, obs_key, df=obs_df, log_x=True, log_y=True, n_bins=20):
    \"\"\"Running median of gen vs. truth, pooled across all suites.\"\"\"
    t_all, g_all = [], []
    for suite in SUITES:
        sub  = df[df['suite'] == suite]
        t    = sub[f'truth_{obs_key}'].to_numpy()
        g    = sub[f'gen_{obs_key}'].to_numpy()
        mask = np.isfinite(t) & np.isfinite(g)
        if log_x: mask &= t > 0
        if log_y: mask &= g > 0
        t_all.append(np.log10(t[mask]) if log_x else t[mask])
        g_all.append(np.log10(g[mask]) if log_y else g[mask])
    t_all = np.concatenate(t_all)
    g_all = np.concatenate(g_all)
    if len(t_all) < 10:
        return
    bs = binned_statistic(t_all, g_all, statistic='median', bins=n_bins)
    bc = 0.5 * (bs.bin_edges[:-1] + bs.bin_edges[1:])
    ok = np.isfinite(bs.statistic)
    ax.plot(bc[ok], bs.statistic[ok], 'k-', lw=2.0, zorder=5, label='median')

print('Plot helpers ready.')
"""

# ─────────────────────────────────────────────────────────────────────────────
# Plotting code per section

MASSES_FIG = """\
# Scatter: truth vs. BIND2 for M_dm, M_gas, M_star  (log–log scale)
mass_keys  = ['M_dm', 'M_gas', 'M_star']
mass_titles = [r'DM mass $M_{\\rm DM}(<R_{200})$',
               r'Gas mass $M_{\\rm gas}(<R_{200})$',
               r'Stellar mass $M_\\star(<R_{200})$']

fig, axes = plt.subplots(2, 3, figsize=(13, 9))

# Row 1: scatter truth vs. gen
for col, (k, title) in enumerate(zip(mass_keys, mass_titles)):
    ax = axes[0, col]
    for suite in SUITES:
        sub  = obs_df[obs_df['suite'] == suite]
        t    = sub[f'truth_{k}'].to_numpy()
        g    = sub[f'gen_{k}'].to_numpy()
        mask = np.isfinite(t) & np.isfinite(g) & (t > 0) & (g > 0)
        ax.scatter(np.log10(t[mask]), np.log10(g[mask]),
                   s=2, alpha=0.2, color=SUITE_COLORS[suite],
                   rasterized=True, label=SUITE_DISPLAY[suite])
    # identity line
    lo = ax.get_xlim()[0]; hi = ax.get_xlim()[1]
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)
    running_median_overlay(ax, k, log_x=True, log_y=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'$\\log_{10}$ Truth  [$M_\\odot/h$]', fontsize=9)
    ax.set_ylabel(r'$\\log_{10}$ BIND2  [$M_\\odot/h$]', fontsize=9)
    if col == 2:
        handles = [mpatches.Patch(color=SUITE_COLORS[s], label=SUITE_DISPLAY[s])
                   for s in SUITES]
        ax.legend(handles=handles, loc='upper left', fontsize=8)

# Row 2: relative error boxplots
for col, k in enumerate(mass_keys):
    ax = axes[1, col]
    error_boxplot(ax, k, clip=0.8)

axes[1, 1].set_xlabel('Suite', fontsize=10)
fig.suptitle('Enclosed Masses within $R_{200c}$ — Truth vs. BIND2', y=1.002, fontsize=13)
plt.tight_layout()
save_fig(fig, 'obs_masses')
plt.show()
"""

FB_FIG = """\
# Baryon fraction: f_b and f_b_norm — violin distributions + error boxplot
fb_keys   = ['f_b', 'f_b_norm']
fb_titles = [r'Baryon fraction $f_b = (M_{\\rm gas} + M_\\star) / M_{\\rm tot}$',
             r'Normalised baryon fraction $f_b / f_{b,{\\rm cosmic}}$']

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

for col, (k, title) in enumerate(zip(fb_keys, fb_titles)):
    # Row 0: violin comparison truth vs gen per suite
    ax0 = axes[0, col]
    pos_t = np.arange(len(SUITES)) * 3.0
    pos_g = pos_t + 1.1
    for i, suite in enumerate(SUITES):
        sub = obs_df[obs_df['suite'] == suite]
        tv  = sub[f'truth_{k}'].dropna().to_numpy()
        gv  = sub[f'gen_{k}'].dropna().to_numpy()
        vt  = ax0.violinplot([tv], positions=[pos_t[i]], widths=0.85,
                             showmedians=True, showextrema=False)
        vg  = ax0.violinplot([gv], positions=[pos_g[i]], widths=0.85,
                             showmedians=True, showextrema=False)
        for part in vt['bodies']:
            part.set_facecolor('k'); part.set_alpha(0.3)
        vt['cmedians'].set_color('k'); vt['cmedians'].set_linewidth(1.5)
        for part in vg['bodies']:
            part.set_facecolor(SUITE_COLORS[suite]); part.set_alpha(0.55)
        vg['cmedians'].set_color(SUITE_COLORS[suite]); vg['cmedians'].set_linewidth(1.5)
    tick_pos = (pos_t + pos_g) / 2
    ax0.set_xticks(tick_pos)
    ax0.set_xticklabels([SUITE_DISPLAY[s] for s in SUITES])
    ax0.set_ylabel(OBS_LATEX[k], fontsize=10)
    ax0.set_title(title, fontsize=10)
    handles = [mpatches.Patch(facecolor='k', alpha=0.4, label='Truth'),
               mpatches.Patch(facecolor='gray', alpha=0.55, label='BIND2')]
    ax0.legend(handles=handles, loc='best', fontsize=8)

    # Row 1: error boxplot
    ax1 = axes[1, col]
    error_boxplot(ax1, k, clip=0.5)
    ax1.set_xlabel('Suite', fontsize=10)

fig.suptitle('Baryon Fraction within $R_{200c}$', y=1.002, fontsize=13)
plt.tight_layout()
save_fig(fig, 'obs_baryon_fraction')
plt.show()
"""

RC_FIG = """\
# Closure radius: Rc_over_R200 — scatter + error boxplot
k     = 'Rc_over_R200'
title = r'Closure radius $R_{\\rm closure}/R_{200}$'

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: scatter
ax = axes[0]
for suite in SUITES:
    sub  = obs_df[obs_df['suite'] == suite]
    t    = sub[f'truth_{k}'].to_numpy()
    g    = sub[f'gen_{k}'].to_numpy()
    mask = np.isfinite(t) & np.isfinite(g)
    ax.scatter(t[mask], g[mask], s=3, alpha=0.25,
               color=SUITE_COLORS[suite], rasterized=True,
               label=SUITE_DISPLAY[suite])
lo = ax.get_xlim()[0]; hi = ax.get_xlim()[1]
ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)
# running median
t_all = np.concatenate([obs_df[obs_df['suite'] == s][f'truth_{k}'].to_numpy() for s in SUITES])
g_all = np.concatenate([obs_df[obs_df['suite'] == s][f'gen_{k}'].to_numpy() for s in SUITES])
mask_all = np.isfinite(t_all) & np.isfinite(g_all)
if mask_all.sum() > 20:
    bs = binned_statistic(t_all[mask_all], g_all[mask_all], statistic='median', bins=20)
    bc = 0.5 * (bs.bin_edges[:-1] + bs.bin_edges[1:])
    ok = np.isfinite(bs.statistic)
    ax.plot(bc[ok], bs.statistic[ok], 'k-', lw=2, zorder=5)
ax.set_xlabel(r'Truth $R_{\\rm closure}/R_{200}$', fontsize=10)
ax.set_ylabel(r'BIND2 $R_{\\rm closure}/R_{200}$', fontsize=10)
ax.set_title(title, fontsize=10)
handles = [mpatches.Patch(color=SUITE_COLORS[s], label=SUITE_DISPLAY[s]) for s in SUITES]
ax.legend(handles=handles, fontsize=8)

# Right: error boxplot
error_boxplot(axes[1], k, clip=0.5)
axes[1].set_xlabel('Suite', fontsize=10)
axes[1].set_title('Absolute error', fontsize=10)

fig.suptitle(title, y=1.002, fontsize=13)
plt.tight_layout()
save_fig(fig, 'obs_closure_radius')
plt.show()
"""

Q_FIG = """\
# Projected axis ratios: q_DM, q_gas, q_star — scatter + error boxplot
q_keys   = ['q_DM', 'q_gas', 'q_star']
q_titles = [r'DM axis ratio $q_{\\rm DM}$',
            r'Gas axis ratio $q_{\\rm gas}$',
            r'Stellar axis ratio $q_\\star$']

fig, axes = plt.subplots(2, 3, figsize=(13, 9))

# Row 0: scatter truth vs gen
for col, (k, title) in enumerate(zip(q_keys, q_titles)):
    ax = axes[0, col]
    for suite in SUITES:
        sub  = obs_df[obs_df['suite'] == suite]
        t    = sub[f'truth_{k}'].to_numpy()
        g    = sub[f'gen_{k}'].to_numpy()
        mask = np.isfinite(t) & np.isfinite(g) & (t > 0) & (g > 0)
        ax.scatter(t[mask], g[mask], s=2, alpha=0.2, color=SUITE_COLORS[suite],
                   rasterized=True, label=SUITE_DISPLAY[suite])
    lo = ax.get_xlim()[0]; hi = ax.get_xlim()[1]
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)
    t_all = np.concatenate([obs_df[obs_df['suite'] == s][f'truth_{k}'].to_numpy() for s in SUITES])
    g_all = np.concatenate([obs_df[obs_df['suite'] == s][f'gen_{k}'].to_numpy() for s in SUITES])
    mask_all = np.isfinite(t_all) & np.isfinite(g_all) & (t_all > 0) & (g_all > 0)
    if mask_all.sum() > 20:
        bs = binned_statistic(t_all[mask_all], g_all[mask_all],
                              statistic='median', bins=20)
        bc = 0.5 * (bs.bin_edges[:-1] + bs.bin_edges[1:])
        ok = np.isfinite(bs.statistic)
        ax.plot(bc[ok], bs.statistic[ok], 'k-', lw=2, zorder=5)
    ax.set_xlabel('Truth  q', fontsize=9)
    ax.set_ylabel('BIND2  q', fontsize=9)
    ax.set_title(title, fontsize=10)
    if col == 2:
        handles = [mpatches.Patch(color=SUITE_COLORS[s], label=SUITE_DISPLAY[s])
                   for s in SUITES]
        ax.legend(handles=handles, fontsize=8)

# Row 1: error boxplots
for col, k in enumerate(q_keys):
    error_boxplot(axes[1, col], k, clip=0.4)

axes[1, 1].set_xlabel('Suite', fontsize=10)
fig.suptitle('Projected Axis Ratios within $R_{200c}$', y=1.002, fontsize=13)
plt.tight_layout()
save_fig(fig, 'obs_axis_ratios')
plt.show()
"""

DQ_FIG = """\
# dq_DM = q_DM(gen) − q_DM(DMO) : DM shape back-reaction
k     = 'dq_DM'
title = r'DM shape back-reaction $\\Delta q_{\\rm DM} = q_{\\rm DM} - q_{\\rm DM,DMO}$'

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: violin distributions per suite (truth and gen)
ax = axes[0]
pos_t = np.arange(len(SUITES)) * 3.0
pos_g = pos_t + 1.1
for i, suite in enumerate(SUITES):
    sub = obs_df[obs_df['suite'] == suite]
    tv  = sub[f'truth_{k}'].dropna().to_numpy()
    gv  = sub[f'gen_{k}'].dropna().to_numpy()
    if len(tv) < 5 or len(gv) < 5:
        continue
    vt = ax.violinplot([tv], positions=[pos_t[i]], widths=0.85,
                       showmedians=True, showextrema=False)
    vg = ax.violinplot([gv], positions=[pos_g[i]], widths=0.85,
                       showmedians=True, showextrema=False)
    for part in vt['bodies']:
        part.set_facecolor('k'); part.set_alpha(0.3)
    vt['cmedians'].set_color('k')
    for part in vg['bodies']:
        part.set_facecolor(SUITE_COLORS[suite]); part.set_alpha(0.55)
    vg['cmedians'].set_color(SUITE_COLORS[suite])
tick_pos = (pos_t + pos_g) / 2
ax.set_xticks(tick_pos)
ax.set_xticklabels([SUITE_DISPLAY[s] for s in SUITES])
ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_ylabel(OBS_LATEX[k], fontsize=10)
ax.set_title(title, fontsize=10)
handles = [mpatches.Patch(facecolor='k', alpha=0.4, label='Truth'),
           mpatches.Patch(facecolor='gray', alpha=0.55, label='BIND2')]
ax.legend(handles=handles, fontsize=8)

# Right: absolute error
error_boxplot(axes[1], k, clip=0.3)
axes[1].set_xlabel('Suite', fontsize=10)
axes[1].set_title('Absolute error', fontsize=10)

fig.suptitle(title, y=1.002, fontsize=13)
plt.tight_layout()
save_fig(fig, 'obs_dq_dm')
plt.show()
"""

SIGMA_FIG = """\
# Sigma_gas_c: central gas surface density within 0.1 * R200c
k     = 'Sigma_gas_c'
title = r'Central gas surface density $\\Sigma_{\\rm gas,c}$ (within $0.1\\,R_{200}$)'

fig, axes = plt.subplots(2, 1, figsize=(7, 9))

# Top: scatter log–log
ax = axes[0]
for suite in SUITES:
    sub  = obs_df[obs_df['suite'] == suite]
    t    = sub[f'truth_{k}'].to_numpy()
    g    = sub[f'gen_{k}'].to_numpy()
    mask = np.isfinite(t) & np.isfinite(g) & (t > 0) & (g > 0)
    ax.scatter(np.log10(t[mask]), np.log10(g[mask]),
               s=2, alpha=0.25, color=SUITE_COLORS[suite],
               rasterized=True, label=SUITE_DISPLAY[suite])
lo = ax.get_xlim()[0]; hi = ax.get_xlim()[1]
ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)
running_median_overlay(ax, k, log_x=True, log_y=True)
ax.set_xlabel(r'$\\log_{10}$ Truth  [$M_\\odot/h$/pix²]', fontsize=10)
ax.set_ylabel(r'$\\log_{10}$ BIND2  [$M_\\odot/h$/pix²]', fontsize=10)
ax.set_title(title, fontsize=10)
handles = [mpatches.Patch(color=SUITE_COLORS[s], label=SUITE_DISPLAY[s]) for s in SUITES]
ax.legend(handles=handles, fontsize=8)

# Bottom: relative error boxplot
error_boxplot(axes[1], k, clip=1.0)
axes[1].set_xlabel('Suite', fontsize=10)

fig.suptitle(title, y=1.002, fontsize=13)
plt.tight_layout()
save_fig(fig, 'obs_sigma_gas_c')
plt.show()
"""

SUMMARY_FIG = """\
# Summary: relative / absolute error for all 11 observables, per suite
n_obs  = len(OBS_KEYS)
n_cols = 3
n_rows = (n_obs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
axes = axes.flatten()

CLIP_PER_OBS = {
    'M_dm': 0.8, 'M_gas': 0.8, 'M_star': 0.8,
    'f_b': 0.3, 'f_b_norm': 0.5,
    'Rc_over_R200': 0.5,
    'q_DM': 0.4, 'q_gas': 0.4, 'q_star': 0.4,
    'dq_DM': 0.3, 'Sigma_gas_c': 1.0,
}

for i, k in enumerate(OBS_KEYS):
    ax   = axes[i]
    clip = CLIP_PER_OBS[k]
    error_boxplot(ax, k, clip=clip)
    ax.set_title(OBS_LATEX[k], fontsize=11)
    if OBS_RELATIVE_ERR[k]:
        ax.set_ylabel(r'$(F_{\\rm gen} - F_{\\rm truth}) / F_{\\rm truth}$', fontsize=8)
    else:
        ax.set_ylabel(r'$F_{\\rm gen} - F_{\\rm truth}$', fontsize=8)

for j in range(n_obs, len(axes)):
    axes[j].axis('off')

fig.suptitle('BIND2 Observable Accuracy — All 11 Observables', y=1.002, fontsize=14)
plt.tight_layout()
save_fig(fig, 'obs_summary_all')
plt.show()
"""

MASS_HALO_TREND = """\
# Per-observable: how does the error depend on halo mass?
# For each observable, scatter the relative/absolute error vs. log10 M200c,
# coloured by suite, with a running-median line.

n_obs  = len(OBS_KEYS)
n_cols = 3
n_rows = (n_obs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes      = axes.flatten()

CLIP_PER_OBS = {
    'M_dm': 0.8, 'M_gas': 0.8, 'M_star': 0.8,
    'f_b': 0.3, 'f_b_norm': 0.5,
    'Rc_over_R200': 0.5,
    'q_DM': 0.4, 'q_gas': 0.4, 'q_star': 0.4,
    'dq_DM': 0.3, 'Sigma_gas_c': 1.0,
}

for oi, k in enumerate(OBS_KEYS):
    ax   = axes[oi]
    clip = CLIP_PER_OBS[k]
    for suite in SUITES:
        sub  = obs_df[obs_df['suite'] == suite]
        logm = sub['logM'].to_numpy()
        err  = _error_vals(sub, k, suite) if False else None   # placeholder
        # recompute inline so we keep logM alignment
        t = sub[f'truth_{k}'].to_numpy()
        g = sub[f'gen_{k}'].to_numpy()
        if OBS_RELATIVE_ERR[k]:
            with np.errstate(divide='ignore', invalid='ignore'):
                err = np.where(np.isfinite(t) & (t > 0), (g - t) / t, np.nan)
        else:
            err = np.where(np.isfinite(t) & np.isfinite(g), g - t, np.nan)
        mask = np.isfinite(err)
        ax.scatter(logm[mask], np.clip(err[mask], -clip, clip),
                   s=1.5, alpha=0.15, color=SUITE_COLORS[suite], rasterized=True)
        # running median per suite
        if mask.sum() > 10:
            bs = binned_statistic(logm[mask], err[mask],
                                  statistic='median', bins=12)
            bc = 0.5 * (bs.bin_edges[:-1] + bs.bin_edges[1:])
            ok = np.isfinite(bs.statistic)
            ax.plot(bc[ok], np.clip(bs.statistic[ok], -clip, clip),
                    '-', color=SUITE_COLORS[suite], lw=1.8,
                    label=SUITE_DISPLAY[suite])
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel(r'$\\log_{10} M_{200c}$  [$M_\\odot/h$]', fontsize=9)
    err_lbl = r'$\\Delta F/F$' if OBS_RELATIVE_ERR[k] else r'$\\Delta F$'
    ax.set_ylabel(err_lbl, fontsize=9)
    ax.set_ylim(-clip * 1.1, clip * 1.1)
    ax.set_title(OBS_LATEX[k], fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    if oi == 0:
        ax.legend(fontsize=7.5, loc='best')

for j in range(n_obs, len(axes)):
    axes[j].axis('off')

fig.suptitle('Observable Error vs. Halo Mass', y=1.002, fontsize=13)
plt.tight_layout()
save_fig(fig, 'obs_error_vs_mass')
plt.show()
"""

# ─────────────────────────────────────────────────────────────────────────────
# Build the notebook

nb = nbf.v4.new_notebook()

nb.cells = [

    # ── Title ──────────────────────────────────────────────────────────────
    md("""\
# Physical Observables — Truth vs. BIND2

Before examining Jacobians (in `analysis_cv_derivatives.ipynb`), this notebook
walks through the **11 scalar physical observables** computed by `fd_jacobian_cv.py`.
For each observable we show (i) its mathematical definition and code, and (ii) the
comparison between truth and BIND2 predictions across the **CV**, **1P**, and
**SB35 (Test)** simulation suites.

| # | Key | Description |
|---|-----|-------------|
| 1 | `M_dm`         | DM mass enclosed within $R_{200c}$ |
| 2 | `M_gas`        | Gas mass enclosed within $R_{200c}$ |
| 3 | `M_star`       | Stellar mass enclosed within $R_{200c}$ |
| 4 | `f_b`          | Baryon fraction $f_b = (M_{\\rm gas}+M_\\star)/M_{\\rm tot}$ within $R_{200c}$ |
| 5 | `f_b_norm`     | $f_b$ normalised by the cosmic baryon fraction $\\Omega_b/\\Omega_m$ |
| 6 | `Rc_over_R200` | Closure radius $R_{\\rm cl}/R_{200c}$: smallest $r$ where $f_b(< r)\\geq 0.9\\,f_{b,{\\rm cos}}$ |
| 7 | `q_DM`         | Projected DM axis ratio $q=b/a$ within $R_{200c}$ |
| 8 | `q_gas`        | Projected gas axis ratio within $R_{200c}$ |
| 9 | `q_star`       | Projected stellar axis ratio within $R_{200c}$ |
|10 | `dq_DM`        | DM back-reaction in shape: $q_{\\rm DM} - q_{\\rm DM,DMO}$ |
|11 | `Sigma_{\\rm gas,c}` | Mean gas surface density within $0.1\\,R_{200c}$ |

**Data flow:** loads pre-generated `generated_halos.npz` + `full_maps.npz` from the
test-suite tree; builds a per-halo observable cache once, then plots.
No model inference required.
"""),

    # ── Section 0 ──────────────────────────────────────────────────────────
    md("## 0. Setup"),
    code(SETUP),

    # ── Section 1 ──────────────────────────────────────────────────────────
    md("## 1. Simulation Discovery"),
    code(SIM_DISCOVERY),

    # ── Section 2 ──────────────────────────────────────────────────────────
    md("""\
## 2. Physics Observable Functions

All functions are **identical to those in `fd_jacobian_cv.py`** so that the
observable values computed here are exactly the quantities whose Jacobians appear
in the derivative notebook.

### 2.1 Aperture sums — $M_{\\rm DM}$, $M_{\\rm gas}$, $M_\\star$

$$M_X(<R_{200c}) = \\sum_{r_{ij} < R_{200c}} \\max(\\Sigma_{X,ij}, 0) \\cdot A_{\\rm pix}$$

where $r_{ij}$ is the distance from the patch centre and $A_{\\rm pix}$ is the
pixel area (already absorbed into the denormalised pixel values in units of $M_\\odot/h$
per pixel).

### 2.2 Baryon fractions

$$f_b = \\frac{M_{\\rm gas} + M_\\star}{M_{\\rm DM} + M_{\\rm gas} + M_\\star}, \\qquad
f_{b,{\\rm norm}} = \\frac{f_b}{\\Omega_b / \\Omega_m}$$

### 2.3 Closure radius $R_{\\rm cl}/R_{200c}$

Defined as the smallest radius at which the enclosed baryon fraction first reaches
$0.9\\,f_{b,{\\rm cosmic}}$, obtained by linear interpolation on the cumulative
azimuthal-mean profiles.

### 2.4 Projected axis ratio $q = b/a$

Iterative moment-of-inertia algorithm: starts with a circular aperture, computes
the mass-weighted 2D quadrupole tensor $\\mathbf{Q}$, sets $q = \\sqrt{\\lambda_{\\rm min}/\\lambda_{\\rm max}}$,
rotates to the eigenvector frame, and repeats until $|\\Delta q|<10^{-3}$ (max 5 iter).

### 2.5 DM shape back-reaction $\\Delta q_{\\rm DM}$

$$\\Delta q_{\\rm DM} = q_{\\rm DM}^{\\rm hydro} - q_{\\rm DM}^{\\rm DMO}$$

Positive values indicate baryons make DM more spherical (contraction).

### 2.6 Central gas surface density $\\Sigma_{\\rm gas,c}$

Mean gas surface density within $r < \\max(0.1\\,R_{200c},\\,2\\,{\\rm pix})$.
"""),
    code(PHYSICS_HELPERS),

    # ── Section 3 ──────────────────────────────────────────────────────────
    md("""\
## 3. Build / Load Observable Cache

One pass over all test-suite sims computes the 11 observables for both the truth
hydro maps and the BIND2 generated maps.  Results are cached to
`analysis_physics_cache/obs_fm_two_head.npz`.  Re-run with `force=True` to rebuild.
"""),
    code(CACHE_BUILD),
    code(DATAFRAME_BUILD),
    code(PLOT_HELPERS),

    # ── Section 4 ──────────────────────────────────────────────────────────
    md("""\
## 4. Enclosed Masses: $M_{\\rm DM}$, $M_{\\rm gas}$, $M_\\star$

**Top row:** truth vs. BIND2 scatter (log–log) for each mass component,
coloured by suite.  The black dashed line is $y=x$ and the solid black curve
is the running median pooled across all suites.

**Bottom row:** relative error $(F_{\\rm gen} - F_{\\rm truth})/F_{\\rm truth}$
as boxplots per suite.  Annotated numbers are per-suite medians.
"""),
    code(MASSES_FIG),

    # ── Section 5 ──────────────────────────────────────────────────────────
    md("""\
## 5. Baryon Fractions: $f_b$ and $f_b / f_{b,{\\rm cosmic}}$

**Top row:** violin distributions of truth (black) and BIND2 (coloured) per suite.

**Bottom row:** absolute error $F_{\\rm gen} - F_{\\rm truth}$.

$f_{b,{\\rm norm}} = 1$ means the simulation has recovered the cosmic baryon
fraction inside $R_{200c}$.  Deviations from 1 signal AGN/SN feedback expelling
baryons beyond the halo boundary.
"""),
    code(FB_FIG),

    # ── Section 6 ──────────────────────────────────────────────────────────
    md("""\
## 6. Closure Radius $R_{\\rm closure}/R_{200c}$

Measures how far out the baryons are enclosed.  $R_{\\rm cl}/R_{200}\\approx 1$
means the baryon fraction profile reaches the cosmic value only at the virial radius;
smaller values indicate more compact baryon distributions.

**Left:** truth vs. BIND2 scatter.  **Right:** absolute error per suite.
"""),
    code(RC_FIG),

    # ── Section 7 ──────────────────────────────────────────────────────────
    md("""\
## 7. Projected Axis Ratios: $q_{\\rm DM}$, $q_{\\rm gas}$, $q_\\star$

The projected axis ratio $q = b/a \\in (0, 1]$ ($q=1$ is circular).
The iterative moment algorithm converges to the shape of the mass distribution
within the $R_{200c}$ elliptical aperture.

**Top row:** truth vs. BIND2 scatter.  **Bottom row:** absolute error per suite.
"""),
    code(Q_FIG),

    # ── Section 8 ──────────────────────────────────────────────────────────
    md("""\
## 8. DM Shape Back-Reaction $\\Delta q_{\\rm DM}$

$\\Delta q_{\\rm DM} = q_{\\rm DM}^{\\rm hydro} - q_{\\rm DM}^{\\rm DMO}$ quantifies
baryonic back-reaction on the DM shape:
- $\\Delta q > 0$: baryons make DM more spherical (cooling + adiabatic contraction)
- $\\Delta q < 0$: baryons elongate DM (rare; AGN feedback can pull DM asymmetrically)

**Left:** violin distributions of truth (black) vs. BIND2 (coloured).
**Right:** absolute error distribution per suite.
"""),
    code(DQ_FIG),

    # ── Section 9 ──────────────────────────────────────────────────────────
    md("""\
## 9. Central Gas Surface Density $\\Sigma_{\\rm gas,c}$

Mean gas surface density within $r < 0.1\\,R_{200c}$ (or $\\geq 2$ pixels if
$0.1\\,R_{200c}$ is smaller).  A direct probe of the cool-core / AGN-feedback
equilibrium.

**Top:** truth vs. BIND2 scatter (log–log).  **Bottom:** relative error per suite.
"""),
    code(SIGMA_FIG),

    # ── Section 10 ──────────────────────────────────────────────────────────
    md("## 10. Summary — All 11 Observables"),
    code(SUMMARY_FIG),

    # ── Section 11 ──────────────────────────────────────────────────────────
    md("""\
## 11. Error vs. Halo Mass

Does the accuracy of each observable depend on halo mass?  Each point is one halo;
the solid curves are running medians per suite.
"""),
    code(MASS_HALO_TREND),
]

out_path = Path('/mnt/home/mlee1/vdm_bind2/analysis_observables.ipynb')
with open(out_path, 'w') as fh:
    nbf.write(nb, fh)
print(f'Wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} kB)')
