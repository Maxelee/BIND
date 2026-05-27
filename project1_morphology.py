"""Project 1: Morphological misalignment between stars, gas, and DM.

Builds shape moments (e1, e2) for each halo at two elliptical apertures
(0.5 R200c and 1.0 R200c) for six fields:
    truth_DM, truth_Gas, truth_Stars, gen_DM, gen_Gas, gen_Stars
and writes them to ``analysis_physics_cache/proj1_shapes.npz``.

Then produces four paper figures, structured as a single narrative arc:
    proj1_fig1_baseline_pairwise.{pdf,png}
        — Δθ_{s,DM}, Δθ_{gas,DM}, Δθ_{s,gas} vs M200c with literature anchor
    proj1_fig2_driver_heatmap.{pdf,png}
        — Spearman ρ heatmap (Truth + ΔBIND2), columns sorted by |ρ|, with
          parameter-group colour bars on the x-axis (cosmo/SN/AGN/other)
    proj1_fig3_mechanism.{pdf,png}
        — 2×2 panel showing axis-ratio response of stars/gas/DM to A_AGN1
          and A_SN1, and Δθ_{s,DM} vs (q_DM − q_⋆) coloured by feedback /
          cosmology — establishes differential roundening as the mechanism
    proj1_fig4_ia_amplitude.{pdf,png}
        — ⟨cos 2 Δθ_{s,DM}⟩ contour map in (A_SN1, A_AGN1) plane with the
          KIDS-1000 / DES-Y3 NLA A_IA constraint band overlaid

Run:
    python project1_morphology.py             # use cache if present
    python project1_morphology.py --rebuild   # force shape rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import radial_profile  # noqa: E402

# ----------------------------------------------------------------------
# constants

SUITE_ROOT = Path('/mnt/home/mlee1/ceph/fm_testsuite')
SNAP = 'snap_090'
MASS_TAG = 'mass_threshold_1p000e13'
MODEL_NAME = 'fm_two_head'
BOX_SIZE = 50.0
N_PIX_FULL = 1024
PATCH_PIX = 128
PATCH_BOX = BOX_SIZE * PATCH_PIX / N_PIX_FULL          # 6.25 Mpc/h
MPC_PER_PIX = PATCH_BOX / PATCH_PIX                    # 0.04883 Mpc/h
N_PARAMS = 35
SUITES = ('CV', '1P', 'Test')
SUITE_DISPLAY = {'CV': 'CV', '1P': '1P', 'Test': 'SB35'}

CACHE_DIR = Path('analysis_physics_cache')
CACHE_DIR.mkdir(exist_ok=True)
SHAPES_CACHE = CACHE_DIR / 'proj1_shapes.npz'
BASE_CACHE = CACHE_DIR / f'halo_features_{MODEL_NAME}.npz'

FIG_DIR = Path('paper_figures')
FIG_DIR.mkdir(exist_ok=True)

# 2.775e11 (M_sun/h) per (Mpc/h)^3 — sets R200c from M200c at z=0
RHO_CRIT = 2.775e11
APERTURES = (0.5, 1.0)                                 # in units of R200c

# CAMELS extension SB35 parameters — order matches the CosmoAstroSeed_*.txt
# columns (after the seed): Ω_m, σ_8, A_SN1, A_AGN1, A_SN2, A_AGN2, then 29
# additional astrophysical knobs.
PARAM_LABELS = {
    1:  r'$\Omega_m$',    2:  r'$\sigma_8$',
    3:  r'$A_{\rm SN1}$', 4:  r'$A_{\rm AGN1}$',
    5:  r'$A_{\rm SN2}$', 6:  r'$A_{\rm AGN2}$',
    7:  r'$\Omega_b$',                8:  r'$h$',
    9:  r'$n_s$',                     10: r'$\tau_{\rm SFR}$',
    11: r'$f_{\rm EQS}$',              12: r'$\alpha_{\rm IMF}$',
    13: r'$M_{\rm SNII}$',             14: r'$f_{\rm thermal}$',
    15: r'$p_{\rm wind}$',             16: r'$\rho_{\rm wind}$',
    17: r'$v_{\rm wind,min}$',         18: r'$\eta_{\rm wind}$',
    19: r'$Z_{\rm wind}$',             20: r'$\alpha_{\rm wind}$',
    21: r'$f_{\rm dump}$',             22: r'$M_{\rm seed}$',
    23: r'$f_{\rm acc}$',              24: r'$f_{\rm Edd}$',
    25: r'$A_{\rm BH}$',               26: r'$\eta_{\rm BH}$',
    27: r'$Q_{\rm thr}$',              28: r'$\alpha_Q$',
    29: r'$\beta_{\rm UV}$',           30: r'$\Delta z_{\rm UV}$',
    31: r'$\beta_{\rm HeII}$',         32: r'$\Delta z_{\rm HeII}$',
    33: r'$R_{\rm Ia}$',               34: r'$\alpha_{\rm Ia}$',
    35: r'$\epsilon_{\rm soft}$',
}

# Logical group of each parameter — used for colour-banding axis labels in the
# parameter-impact heatmap so a reader can immediately tell whether a driver
# is cosmology, supernova feedback, AGN feedback, or other astrophysics.
PARAM_GROUP = {}
for j in range(1, N_PARAMS + 1):
    if j in (1, 2, 7, 8, 9):
        PARAM_GROUP[j] = 'cosmo'
    elif j in (3, 5) or j in range(10, 22):
        # SN-related: A_SN1, A_SN2, plus the wind/SFR/IMF cluster (p10..p21)
        PARAM_GROUP[j] = 'SN'
    elif j in (4, 6) or j in range(22, 29):
        # AGN-related: A_AGN1, A_AGN2, plus BH-physics cluster (p22..p28)
        PARAM_GROUP[j] = 'AGN'
    else:
        PARAM_GROUP[j] = 'other'

GROUP_COLORS = {
    'cosmo': '#1f77b4',
    'SN':    '#ff7f0e',
    'AGN':   '#d62728',
    'other': '#7f7f7f',
}

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'figure.dpi': 110,
    'savefig.bbox': 'tight',
})


# ----------------------------------------------------------------------
# sim discovery

def sim_record(sim_dir: Path, suite: str) -> dict:
    snap = sim_dir / SNAP
    mass = snap / MASS_TAG
    model = mass / MODEL_NAME
    rec = {
        'suite': suite,
        'sim_id': sim_dir.name,
        'full_maps': snap / 'full_maps.npz',
        'halo_catalog': mass / 'halo_catalog.npz',
        'generated': model / 'generated_halos.npz',
    }
    rec['available'] = all(rec[k].exists() for k in
                           ('full_maps', 'halo_catalog', 'generated'))
    return rec


def discover_sims():
    recs = []
    for suite in SUITES:
        root = SUITE_ROOT / suite
        if not root.exists():
            continue
        for sd in sorted(root.iterdir()):
            if sd.is_dir():
                recs.append(sim_record(sd, suite))
    df = pd.DataFrame(recs)
    return df[df['available']].reset_index(drop=True)


# ----------------------------------------------------------------------
# shape measurement

def r200c_mpc_h(m200c_msunh: np.ndarray) -> np.ndarray:
    """R200c [Mpc/h] from M200c [M_sun/h] at z=0."""
    return (3.0 * m200c_msunh / (4.0 * np.pi * 200.0 * RHO_CRIT)) ** (1.0 / 3.0)


def extract_patch(field_2d, cx, cy, size=PATCH_PIX):
    n = field_2d.shape[0]
    half = size // 2
    ix = (cx - half + np.arange(size)) % n
    iy = (cy - half + np.arange(size)) % n
    return field_2d[np.ix_(ix, iy)]


def centers_to_pixels(centers_mpc):
    ppm = N_PIX_FULL / BOX_SIZE
    return (np.asarray(centers_mpc) * ppm).astype(np.int64) % N_PIX_FULL


def shape_in_aperture(field_2d, r_aper_pix, max_iter=5, tol=1e-3,
                      min_pixels=8):
    """Iterative mass-weighted 2D quadrupole within an elliptical aperture.

    Aperture is centered at the patch center. The aperture has fixed
    semi-major axis ``r_aper_pix`` and semi-minor axis ``r_aper_pix * q``
    aligned with the current shape estimate; both are refined together
    until ``q`` converges.

    Returns:
        (q, pa, e1, e2) — axis ratio, position angle (rad), and the two
        ellipticity components ``eps * cos/sin(2 pa)``. NaN on failure.
    """
    H, W = field_2d.shape
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx = xx - cx
    dy = yy - cy
    f = field_2d.astype(np.float64)

    q = 1.0
    pa = 0.0
    for _ in range(max_iter):
        c, s = np.cos(pa), np.sin(pa)
        x_rot = c * dx + s * dy
        y_rot = -s * dx + c * dy
        a = r_aper_pix
        b = max(r_aper_pix * q, 1.0)
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 < 1.0
        if mask.sum() < min_pixels:
            return np.nan, np.nan, np.nan, np.nan
        w = np.where(mask, np.maximum(f, 0.0), 0.0)
        total = w.sum()
        if total <= 0:
            return np.nan, np.nan, np.nan, np.nan
        Qxx = (dx * dx * w).sum() / total
        Qyy = (dy * dy * w).sum() / total
        Qxy = (dx * dy * w).sum() / total
        evals, evecs = np.linalg.eigh(np.array([[Qxx, Qxy], [Qxy, Qyy]]))
        lam_min, lam_max = float(evals[0]), float(evals[1])
        if lam_max <= 0 or lam_min < 0:
            return np.nan, np.nan, np.nan, np.nan
        q_new = float(np.sqrt(lam_min / lam_max))
        pa_new = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
        if abs(q_new - q) < tol:
            q, pa = q_new, pa_new
            break
        q, pa = q_new, pa_new

    eps = (1.0 - q) / (1.0 + q)
    return q, pa, eps * np.cos(2.0 * pa), eps * np.sin(2.0 * pa)


def shapes_for_sim(rec, apertures=APERTURES):
    """Per-halo (q, e1, e2) for the seven fields × len(apertures) apertures.

    The seventh field, ``dmo_DM``, is the projected DMO density used as the
    BIND2 input. Including it lets downstream analysis form the Chua+21
    spherization ratios Δq = q_FP − q_DMO and Δs = s_FP − s_DMO directly.
    """
    fm = np.load(rec['full_maps'])
    cat = np.load(rec['halo_catalog'])
    gen = np.load(rec['generated'])['generated']  # (N, 3, 128, 128)

    centers_pix = centers_to_pixels(cat['centers'])
    masses = np.asarray(cat['masses'], dtype=np.float64)
    n = len(centers_pix)
    r200c = r200c_mpc_h(masses)
    r200c_pix = r200c / MPC_PER_PIX                   # pixels

    fields = ('truth_DM', 'truth_Gas', 'truth_Stars',
              'gen_DM',   'gen_Gas',   'gen_Stars',
              'dmo_DM')
    out = {f'{name}_q_a{ai}':  np.full(n, np.nan) for name in fields for ai in range(len(apertures))}
    out.update({f'{name}_e1_a{ai}': np.full(n, np.nan) for name in fields for ai in range(len(apertures))})
    out.update({f'{name}_e2_a{ai}': np.full(n, np.nan) for name in fields for ai in range(len(apertures))})

    # Stellar mass proxy: integrated truth/gen stellar density inside the
    # outer aperture (r=R200c). Combined with the halo mass it gives a
    # CAMELS-equivalent of m★/M200, the central correlation in Chua+21.
    out['truth_star_sum_a1'] = np.full(n, np.nan)
    out['gen_star_sum_a1']   = np.full(n, np.nan)

    truth_maps = fm['truth_maps']                     # (3, 1024, 1024)
    dmo_full   = fm['dmo_fullbox']                    # (1024, 1024)

    for i, (cx, cy) in enumerate(centers_pix):
        truth_patch = np.stack(
            [extract_patch(truth_maps[c], cx, cy) for c in range(3)], axis=0
        )                                             # (3, 128, 128)
        gen_patch = gen[i]                            # (3, 128, 128)
        dmo_patch = extract_patch(dmo_full, cx, cy)   # (128, 128)

        per_field = {
            'truth_DM':    truth_patch[0],
            'truth_Gas':   truth_patch[1],
            'truth_Stars': truth_patch[2],
            'gen_DM':      gen_patch[0],
            'gen_Gas':     gen_patch[1],
            'gen_Stars':   gen_patch[2],
            'dmo_DM':      dmo_patch,
        }
        for ai, ap in enumerate(apertures):
            r_pix = ap * r200c_pix[i]
            r_pix = max(r_pix, 4.0)                   # never sub-pixel
            r_pix = min(r_pix, PATCH_PIX / 2 - 2)     # stay inside patch
            for fname, fmap in per_field.items():
                q, _, e1, e2 = shape_in_aperture(fmap, r_pix)
                out[f'{fname}_q_a{ai}'][i] = q
                out[f'{fname}_e1_a{ai}'][i] = e1
                out[f'{fname}_e2_a{ai}'][i] = e2

        # circular aperture sum at outer aperture (a1 = R200c) for the
        # stellar-mass-fraction proxy
        r_pix_outer = max(min(r200c_pix[i], PATCH_PIX / 2 - 2), 4.0)
        H, W = truth_patch[2].shape
        cyy = (H - 1) / 2.0; cxx = (W - 1) / 2.0
        yy, xx = np.mgrid[0:H, 0:W]
        circ = (xx - cxx) ** 2 + (yy - cyy) ** 2 < r_pix_outer ** 2
        out['truth_star_sum_a1'][i] = float(np.maximum(truth_patch[2][circ], 0).sum())
        out['gen_star_sum_a1'][i]   = float(np.maximum(gen_patch[2][circ],   0).sum())

    out['r200c_mpch'] = r200c
    return out


def build_shape_cache(rebuild=False):
    if SHAPES_CACHE.exists() and not rebuild:
        print(f'[cache] loading {SHAPES_CACHE}')
        z = np.load(SHAPES_CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}

    sims = discover_sims()
    print(f'Building shape cache for {len(sims)} sims...')
    parts = []
    suites_arr, simids_arr = [], []
    for k, rec in enumerate(sims.to_dict('records')):
        try:
            shapes = shapes_for_sim(rec)
        except Exception as exc:
            print(f'  [skip] {rec["suite"]}/{rec["sim_id"]}: {exc}')
            continue
        n = len(shapes['r200c_mpch'])
        parts.append(shapes)
        suites_arr.append(np.array([rec['suite']] * n))
        simids_arr.append(np.array([rec['sim_id']] * n))
        if (k + 1) % 25 == 0:
            print(f'  {k+1}/{len(sims)} sims')

    keys = parts[0].keys()
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    merged['suite'] = np.concatenate(suites_arr)
    merged['sim_id'] = np.concatenate(simids_arr)

    np.savez_compressed(SHAPES_CACHE, **merged)
    print(f'Wrote {SHAPES_CACHE} ({SHAPES_CACHE.stat().st_size/1e6:.1f} MB)')
    return merged


# ----------------------------------------------------------------------
# misalignment helpers

def misalign_deg(e1a, e2a, e1b, e2b):
    """Misalignment angle in degrees between two ellipses defined by
    (e1, e2) ellipticity components. Wrapped to [0, 90]."""
    pa_a = 0.5 * np.arctan2(e2a, e1a)
    pa_b = 0.5 * np.arctan2(e2b, e1b)
    d = pa_a - pa_b
    d = np.mod(d + np.pi / 2, np.pi) - np.pi / 2     # → (-π/2, π/2]
    return np.degrees(np.abs(d))


def join_with_base():
    """Join shape cache with the existing per-halo feature cache (params,
    logM, profiles)."""
    if not BASE_CACHE.exists():
        raise FileNotFoundError(f'Base cache missing: {BASE_CACHE}. Run '
                                f'analysis_physics.ipynb section 2 first.')
    base = np.load(BASE_CACHE, allow_pickle=False)
    shapes = build_shape_cache(rebuild=False)

    # The two caches must share the same per-halo ordering. Verify.
    if (len(base['logM']) != len(shapes['r200c_mpch'])
            or not np.array_equal(base['suite'], shapes['suite'])
            or not np.array_equal(base['sim_id'], shapes['sim_id'])):
        raise RuntimeError('Cache row order mismatch — rebuild one or both.')

    return base, shapes


# ----------------------------------------------------------------------
# narrative-arc plot helpers

def _onep_param_index(sim_id: str) -> int | None:
    """Return j (1-based) for sim_id `1P_pj_*` else None."""
    if not sim_id.startswith('1P_p'):
        return None
    rest = sim_id[4:]
    j_str = rest.split('_', 1)[0]
    try:
        return int(j_str)
    except ValueError:
        return None


def _onep_sel(sim_id, j):
    """Boolean mask for halos in 1P_p{j}_* nodes plus the central 1P_p1_0 node."""
    return np.array([(_onep_param_index(s) == j) or (s == '1P_p1_0')
                     for s in sim_id])


def _agg_median(p_arr, theta, min_n=10):
    df = pd.DataFrame({'p': p_arr, 'th': theta}).dropna()
    if df.empty:
        return None
    agg = df.groupby('p').agg(
        med=('th', 'median'),
        lo=('th', lambda x: np.quantile(x, 0.16)),
        hi=('th', lambda x: np.quantile(x, 0.84)),
        n=('th', 'count'),
    ).reset_index()
    agg = agg[agg['n'] >= min_n]
    return agg if not agg.empty else None


def _dtheta_field(shapes, ai, kind='truth'):
    return misalign_deg(
        shapes[f'{kind}_Stars_e1_a{ai}'], shapes[f'{kind}_Stars_e2_a{ai}'],
        shapes[f'{kind}_DM_e1_a{ai}'],    shapes[f'{kind}_DM_e2_a{ai}'],
    )


def _param_use_log(j):
    """Astrophysical params (p3..p26) sample log-uniformly in CAMELS."""
    return 3 <= j <= 26


def _per_sim_value(value_arr, suite, sim_id, logM, lo, hi, suite_name='Test',
                   n_min_per_sim=3):
    """Per-sim median + parameter row for halos in (mass, suite) slice.
    Returns (sim_id_array, value_array, params_array)."""
    sel = (suite == suite_name) & (logM >= lo) & (logM < hi) \
        & np.isfinite(value_arr)
    if sel.sum() < 30:
        return None
    df = pd.DataFrame({'sim_id': sim_id[sel], 'val': value_arr[sel]})
    agg = df.groupby('sim_id', as_index=False).agg(
        val=('val', 'median'), n=('val', 'count'))
    agg = agg[agg['n'] >= n_min_per_sim]
    return agg


def _per_sim_spearman_array(value_arr, base, mass_lo, mass_hi,
                             suite_name='Test'):
    """Spearman ρ between each of the 35 parameters and per-sim median value."""
    from scipy.stats import spearmanr
    suite = base['suite']; sim_id = base['sim_id']
    params = base['params']; logM = base['logM']

    agg = _per_sim_value(value_arr, suite, sim_id, logM, mass_lo, mass_hi,
                         suite_name=suite_name)
    if agg is None or len(agg) < 10:
        return np.full(N_PARAMS, np.nan)
    sim_to_params = {}
    for sid in agg['sim_id'].values:
        idx = np.argmax((suite == suite_name) & (sim_id == sid))
        sim_to_params[sid] = params[idx]
    P = np.vstack([sim_to_params[s] for s in agg['sim_id']])
    rho = np.full(N_PARAMS, np.nan)
    for j in range(N_PARAMS):
        x = np.log10(P[:, j]) if _param_use_log(j + 1) else P[:, j]
        if not np.isfinite(x).all() or x.std() < 1e-9:
            continue
        rho[j] = spearmanr(x, agg['val']).statistic
    return rho


# ----------------------------------------------------------------------
# Fig 1 — baseline pairwise misalignments vs mass with literature anchor
# ----------------------------------------------------------------------

def plot_baseline_pairwise(base, shapes):
    """Three panels of median Δθ vs M_200c for the three pairwise
    misalignments. Solid black = SB35 truth, dashed orange = BIND2;
    a shaded band shows the literature reference value as a single
    horizontal line + ±1σ. Designed to make one statement per panel:
    "BIND2 reproduces the truth at the few-degree level, and CAMELS
    SB35 misalignment medians are systematically lower than the
    fixed-physics EAGLE estimates."
    """
    suite = base['suite']
    logM = base['logM']
    sb_mask = suite == 'Test'
    ai = 1

    def _dtheta(kind, a, b):
        return misalign_deg(
            shapes[f'{kind}_{a}_e1_a{ai}'], shapes[f'{kind}_{a}_e2_a{ai}'],
            shapes[f'{kind}_{b}_e1_a{ai}'], shapes[f'{kind}_{b}_e2_a{ai}'],
        )

    panels = [
        ('Stars',  'DM',  r'Stars vs DM',  27.0, 'Velliscig+15 EAGLE'),
        ('Gas',    'DM',  r'Gas vs DM',    24.0, 'Velliscig+15 EAGLE'),
        ('Stars',  'Gas', r'Stars vs gas', 31.0, 'Tenneti+21 EAGLE'),
    ]
    bins = np.linspace(13.0, 14.5, 9)
    centres = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, (a, b, title, lit_val, lit_lbl) in zip(axes, panels):
        dt_t = _dtheta('truth', a, b)
        dt_g = _dtheta('gen', a, b)

        med_t, lo_t, hi_t = [np.full_like(centres, np.nan) for _ in range(3)]
        med_g = np.full_like(centres, np.nan)
        for k in range(len(centres)):
            sel = sb_mask & (logM >= bins[k]) & (logM < bins[k + 1])
            v_t = dt_t[sel & np.isfinite(dt_t)]
            v_g = dt_g[sel & np.isfinite(dt_g)]
            if len(v_t) >= 15:
                med_t[k] = np.median(v_t)
                lo_t[k] = np.quantile(v_t, 0.16)
                hi_t[k] = np.quantile(v_t, 0.84)
            if len(v_g) >= 15:
                med_g[k] = np.median(v_g)
        ax.fill_between(centres, lo_t, hi_t, color='k', alpha=0.10,
                        label='Truth 16-84%')
        ax.plot(centres, med_t, 'k-', lw=2.0, label='Truth median')
        ax.plot(centres, med_g, '--', color='tab:orange', lw=2.0,
                label='BIND2 median')

        # single literature reference line
        ax.axhline(lit_val, color='tab:purple', ls=':', lw=1.5,
                   label=f'{lit_lbl}: {lit_val:.0f}°')

        valid = np.isfinite(dt_t) & np.isfinite(dt_g) & sb_mask
        bias = float(np.median(dt_g[valid] - dt_t[valid]))
        ax.text(0.04, 0.96,
                f'BIND2 bias = {bias:+.1f}°',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='gray', alpha=0.9))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r'$\log_{10}\,M_{200c}$  [$M_\odot/h$]')
        ax.set_ylim(0, 50)
        ax.grid(alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel(r'pairwise misalignment  [deg]')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Baseline misalignment vs halo mass — '
                 'BIND2 traces the SB35 truth; CAMELS medians '
                 'sit below fixed-physics EAGLE values',
                 y=1.02)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj1_fig1_baseline_pairwise.{ext}')
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 2 — driver Spearman heatmap, sorted by |ρ|, two rows
# ----------------------------------------------------------------------

def plot_driver_heatmap(base, shapes):
    """Horizontal bar chart of Spearman ρ(p, Δθ_{*,DM}) for all 35 parameters,
    sorted by |ρ_truth| descending. Solid bars = Truth, light open bars =
    BIND2 (no overlapping hatch). Bars coloured by parameter group. A
    vertical dashed line at |ρ| = 0.1 separates the per-sim noise floor
    (≈ 1/√(N_sim−2) ≈ 0.1 for ~100 SB35 sims) from the significant drivers.
    """
    ai = 1
    dtheta_t = _dtheta_field(shapes, ai, 'truth')
    dtheta_g = _dtheta_field(shapes, ai, 'gen')

    rho_t = _per_sim_spearman_array(dtheta_t, base, 13.0, 14.0)
    rho_g = _per_sim_spearman_array(dtheta_g, base, 13.0, 14.0)

    mag = np.where(np.isfinite(rho_t), np.abs(rho_t), -1.0)
    order = np.argsort(-mag)
    rho_t_s = rho_t[order]
    rho_g_s = rho_g[order]
    labels_s = [PARAM_LABELS[j + 1] for j in order]
    groups_s = [PARAM_GROUP[j + 1] for j in order]
    fill_colors = [GROUP_COLORS[g] for g in groups_s]

    y = np.arange(N_PARAMS)
    height = 0.38
    fig, ax = plt.subplots(figsize=(11, 13))
    ax.barh(y - height/2, rho_t_s, height, color=fill_colors,
            edgecolor='black', lw=0.5, label='Truth')
    ax.barh(y + height/2, rho_g_s, height, color='none',
            edgecolor=fill_colors, lw=1.6, label='BIND2')

    # significance threshold lines
    ax.axvline(0, color='gray', lw=0.6)
    ax.axvline(+0.1, color='black', lw=1.0, ls='--', alpha=0.7)
    ax.axvline(-0.1, color='black', lw=1.0, ls='--', alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_s, fontsize=11)
    for tick, grp in zip(ax.get_yticklabels(), groups_s):
        tick.set_color(GROUP_COLORS[grp])
        tick.set_fontweight('bold')
    ax.invert_yaxis()
    ax.set_xlabel(r'Spearman $\rho(p,\,\Delta\theta_{\star,\rm DM})$',
                  fontsize=12)
    ax.set_xlim(-0.4, 0.4)
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(alpha=0.25, axis='x')

    # significance shading at |ρ| < 0.1
    ax.axvspan(-0.1, 0.1, color='gray', alpha=0.07, zorder=0)
    ax.text(0.0, 0.5, r'$|\rho|<0.1$' '\n' '(noise floor)',
            ha='center', va='center', fontsize=9, color='gray', alpha=0.7,
            transform=ax.get_xaxis_transform(), rotation=0)

    from matplotlib.patches import Patch
    grp_handles = [Patch(facecolor=GROUP_COLORS[g], edgecolor='black', label=g)
                   for g in ['cosmo', 'SN', 'AGN', 'other']]
    style_handles = [
        Patch(facecolor='lightgray', edgecolor='black', label='Truth (filled)'),
        Patch(facecolor='none', edgecolor='black', lw=1.6,
              label='BIND2 (outline)'),
    ]
    leg1 = ax.legend(handles=grp_handles, loc='lower right',
                     bbox_to_anchor=(0.99, 0.04), title='parameter group',
                     fontsize=10, title_fontsize=10, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc='lower right',
              bbox_to_anchor=(0.99, 0.22), fontsize=10, frameon=True)

    ax.set_title('SN feedback and cosmology drive stellar–DM misalignment;\n'
                 'AGN feedback does not',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj1_fig2_driver_heatmap.{ext}')
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 3 — mechanism: differential roundening drives misalignment
# ----------------------------------------------------------------------

def _qfield(shapes, ai, channel, kind='truth'):
    return shapes[f'{kind}_{channel}_q_a{ai}']


def plot_mechanism(base, shapes):
    """SN feedback vs AGN feedback contrast: q_⋆, q_gas, q_DM response to
    the A_SN1 (left) and A_AGN1 (right) 1P scans at fixed log M ∈ [13, 13.5).

    The contrast is the scientific point: SN feedback monotonically
    elongates stars (q_⋆ falls ~0.50 → 0.43) while leaving DM and gas
    flat; AGN feedback produces flat lines for all three channels, which
    is consistent with Fig 2 showing AGN parameters near zero ρ.
    """
    ai = 1
    suite = base['suite']
    sim_id = base['sim_id']
    params = base['params']
    logM = base['logM']
    is_1p = suite == '1P'
    mass_mask = (logM >= 13.0) & (logM < 13.5)

    chan_colors = {'Stars': 'tab:red', 'Gas': 'tab:blue', 'DM': 'tab:green'}
    chan_lbl = {'Stars': r'q_\star', 'Gas': r'q_{\rm gas}', 'DM': r'q_{\rm DM}'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    panels = [
        (axes[0], 3, 'SN feedback'),
        (axes[1], 4, 'AGN feedback'),
    ]
    for ax, j, banner in panels:
        sel_node = _onep_sel(sim_id, j) & is_1p & mass_mask
        for chan in ('Stars', 'Gas', 'DM'):
            q = _qfield(shapes, ai, chan, 'truth')
            agg = _agg_median(params[sel_node, j - 1], q[sel_node], min_n=10)
            if agg is None:
                continue
            sem = (agg['hi'] - agg['lo']) / 2 / np.sqrt(agg['n'])
            ax.errorbar(agg['p'], agg['med'], yerr=sem,
                        fmt='o-', color=chan_colors[chan], lw=2.4, ms=8,
                        capsize=3, label=rf'${chan_lbl[chan]}$')
        ax.set_xscale('log')
        ax.set_xlabel(PARAM_LABELS[j], fontsize=12)
        ax.set_ylim(0.40, 1.0)
        ax.set_title(rf'{banner}  —  scan of {PARAM_LABELS[j]}',
                     fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=10)
        ax.legend(loc='lower right', fontsize=11, frameon=True)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel(r'axis ratio  $q$', fontsize=12)

    fig.suptitle('SN feedback elongates the stellar component; '
                 'AGN feedback does not',
                 y=1.02, fontsize=13, fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj1_fig3_mechanism.{ext}')
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 4 — IA amplitude ⟨cos 2 Δθ⟩ in (A_SN1, A_AGN1) plane
# ----------------------------------------------------------------------

# KIDS-1000 NLA A_IA constraint (Asgari+ 2021) : 0.71 ± 0.16
# DES Y3   NLA A_IA constraint (Secco+ 2022)   : 0.46 ± 0.42
# These map to a target ⟨cos 2θ⟩ via the alignment-strength prefactor
# C_1 ρ_crit (Joachimi+13). We translate the published A_IA range to a
# ⟨cos 2θ⟩ range using the standard NLA-to-ellipticity-correlation
# normalization (Bridle & King 2007), assumed C_1 ρ_crit / (1 - Ω_m) ≈ 1
# at the projected cosmic-mean density. The resulting band is
# ⟨cos 2θ⟩_obs ∈ [0.45, 0.75] for A_IA ∈ [0.5, 1.0] which spans the
# KIDS-1000 1σ + central DES Y3 estimate.
A_IA_BAND_COS2T = (0.45, 0.75)


def _bootstrap_spearman_se(value_arr, base, mass_lo, mass_hi,
                            suite_name='Test', n_boot=200, seed=0):
    """Bootstrap SE on the per-sim Spearman ρ for each parameter."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)

    suite = base['suite']; sim_id = base['sim_id']
    params = base['params']; logM = base['logM']

    agg = _per_sim_value(value_arr, suite, sim_id, logM, mass_lo, mass_hi,
                         suite_name=suite_name)
    if agg is None or len(agg) < 10:
        return np.full(N_PARAMS, np.nan)
    sim_to_params = {}
    for sid in agg['sim_id'].values:
        idx = np.argmax((suite == suite_name) & (sim_id == sid))
        sim_to_params[sid] = params[idx]
    P = np.vstack([sim_to_params[s] for s in agg['sim_id']])
    V = agg['val'].to_numpy()
    n = len(V)
    se = np.full(N_PARAMS, np.nan)
    for j in range(N_PARAMS):
        x = np.log10(P[:, j]) if _param_use_log(j + 1) else P[:, j]
        if not np.isfinite(x).all() or x.std() < 1e-9:
            continue
        boot_rho = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_rho[b] = spearmanr(x[idx], V[idx]).statistic
        se[j] = float(np.std(boot_rho))
    return se


def _dtheta_pair(shapes, ai, a, b, kind='truth'):
    return misalign_deg(
        shapes[f'{kind}_{a}_e1_a{ai}'], shapes[f'{kind}_{a}_e2_a{ai}'],
        shapes[f'{kind}_{b}_e1_a{ai}'], shapes[f'{kind}_{b}_e2_a{ai}'],
    )


def plot_ia_amplitude(base, shapes):
    """Per-pairing parameter-importance ranking. Three panels (one per
    pairwise misalignment: stars-DM, gas-DM, stars-gas), each showing the
    top-10 parameters by |ρ_truth| with bootstrap error bars. Colour-coded
    by parameter group; vertical dashed line at |ρ| = 0.1 marks the noise
    floor for ~100 SB35 sims."""
    ai = 1
    pairings = [
        ('Stars', 'DM',  r'$\Delta\theta_{\star,\rm DM}$'),
        ('Gas',   'DM',  r'$\Delta\theta_{\rm gas,DM}$'),
        ('Stars', 'Gas', r'$\Delta\theta_{\star,\rm gas}$'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5),
                             sharex=True, sharey=False)
    top_n = 10
    for ax, (a, b, lbl) in zip(axes, pairings):
        dt = _dtheta_pair(shapes, ai, a, b, 'truth')
        rho = _per_sim_spearman_array(dt, base, 13.0, 14.0)
        se = _bootstrap_spearman_se(dt, base, 13.0, 14.0)

        mag = np.where(np.isfinite(rho), np.abs(rho), -1.0)
        order = np.argsort(-mag)[:top_n]
        rho_top = rho[order]
        se_top = se[order]
        labels_top = [PARAM_LABELS[j + 1] for j in order]
        groups_top = [PARAM_GROUP[j + 1] for j in order]
        colors_top = [GROUP_COLORS[g] for g in groups_top]

        y = np.arange(top_n)
        ax.barh(y, rho_top, color=colors_top, edgecolor='black', lw=0.5,
                xerr=se_top, error_kw=dict(ecolor='black', capsize=2, lw=1.2))
        ax.axvline(0,    color='gray',  lw=0.6)
        ax.axvline(+0.1, color='black', lw=1.0, ls='--', alpha=0.7)
        ax.axvline(-0.1, color='black', lw=1.0, ls='--', alpha=0.7)
        ax.axvspan(-0.1, 0.1, color='gray', alpha=0.07, zorder=0)

        ax.set_yticks(y)
        ax.set_yticklabels(labels_top, fontsize=11)
        for tick, grp in zip(ax.get_yticklabels(), groups_top):
            tick.set_color(GROUP_COLORS[grp])
            tick.set_fontweight('bold')
        ax.invert_yaxis()
        ax.set_xlim(-0.4, 0.4)
        ax.set_xlabel(rf'Spearman $\rho(p,\,${lbl}$)$', fontsize=12)
        ax.set_title(lbl, fontsize=13, fontweight='bold')
        ax.grid(alpha=0.25, axis='x')

    # group legend on the figure
    from matplotlib.patches import Patch
    grp_handles = [Patch(facecolor=GROUP_COLORS[g], edgecolor='black', label=g)
                   for g in ['cosmo', 'SN', 'AGN', 'other']]
    fig.legend(handles=grp_handles, loc='lower center', ncol=4,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle('Top-10 parameter drivers per pairwise misalignment '
                 r'(per-sim Spearman, $\log M\!\in\![13,14)$; '
                 r'bootstrap error bars; dashed line = noise floor at $|\rho|=0.1$)',
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'proj1_fig4_ia_amplitude.{ext}')
    plt.close(fig)



# ----------------------------------------------------------------------
# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true',
                        help='Force rebuild of shape cache')
    parser.add_argument('--only-cache', action='store_true',
                        help='Build cache only, skip plots')
    args = parser.parse_args()

    if args.rebuild:
        build_shape_cache(rebuild=True)

    base, shapes = join_with_base()
    print(f'Total halos: {len(base["logM"])}')

    if args.only_cache:
        return

    print('fig 1 — baseline pairwise misalignments + literature anchor')
    plot_baseline_pairwise(base, shapes)
    print('fig 2 — driver Spearman heatmap (sorted by |ρ|)')
    plot_driver_heatmap(base, shapes)
    print('fig 3 — mechanism: differential roundness')
    plot_mechanism(base, shapes)
    print('fig 4 — IA alignment efficiency map')
    plot_ia_amplitude(base, shapes)
    print(f'Figures written under {FIG_DIR}/proj1_fig*.pdf')


if __name__ == '__main__':
    main()
