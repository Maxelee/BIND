"""Project 6: Synthesis — BIND as a learned response operator on halo observables.

Implements the cohesive narrative that re-frames the 35-parameter sensitivity
study as a small, low-rank physical response operator.  Builds five products,
all from the cached BIND samples in ``analysis_physics_cache/``:

    1. Gradient heatmap   — 35 × N_obs response matrix g_ik = d ln O_k / d ln θ_i
       (rows clustered, columns annotated; one figure summarises every parameter)
    2. SVD modes          — singular spectrum + parameter loadings + observable
       loadings of the dominant response modes
    3. Coupling matrix    — pair interaction strength C_ij from project-4 pairs,
       embedded in a 35 × 35 frame with focused inset on the SN-AGN block
    4. Fisher information — F = G^T diag(1/σ_k²) G; eigenvalue spectrum +
       corner heatmap on 12 most-informative parameters
    5. Covariance check   — observable covariance from BIND vs truth at three
       parameter nodes (fiducial, A_SN1 high, A_AGN1 high) using halo-to-halo
       scatter

Run:
    python project6_synthesis.py             # use cached observables if present
    python project6_synthesis.py --rebuild   # force observable recomputation

Outputs (paper_figures/):
    proj6_fig1_gradient_heatmap.{pdf,png}
    proj6_fig2_svd_modes.{pdf,png}
    proj6_fig3_coupling_matrix.{pdf,png}
    proj6_fig4_fisher.{pdf,png}
    proj6_fig5_covariance.{pdf,png}
    proj6_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from metrics import radial_profile  # noqa: E402

# ----------------------------------------------------------------------
# constants

CACHE_DIR = ROOT / 'analysis_physics_cache'
FIG_DIR = ROOT / 'paper_figures'
FIG_DIR.mkdir(exist_ok=True)

MODEL_NAME = 'fm_two_head'
PROJ5_CACHE = CACHE_DIR / f'proj5_marginals_{MODEL_NAME}.npz'
PROJ4_CACHE = CACHE_DIR / f'proj4_pairs_v2_{MODEL_NAME}.npz'
BASE_CACHE = CACHE_DIR / f'halo_features_{MODEL_NAME}.npz'
PROJ6_CACHE = CACHE_DIR / f'proj6_synthesis_{MODEL_NAME}.npz'

PARAM_META_CSV = '/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv'
SUITE_ROOT = Path('/mnt/home/mlee1/ceph/fm_testsuite')
SNAP, MASS_TAG = 'snap_090', 'mass_threshold_1p000e13'

N_PARAMS = 35
N_GRID = 7
PATCH_PIX = 128
N_RBINS = 32
FID_IDX = 3  # central index of the 7-point grids

PRETTY = {
    0: r'$\Omega_m$', 1: r'$\sigma_8$',
    2: r'$A_{\rm SN1}$', 3: r'$A_{\rm AGN1}$',
    4: r'$A_{\rm SN2}$', 5: r'$A_{\rm AGN2}$',
    6: r'$\Omega_b$', 7: r'$h$', 8: r'$n_s$',
    9: r'$\tau_{\rm SFR}$', 10: r'$f_{\rm EQS}$', 11: r'$\alpha_{\rm IMF}$',
    12: r'$M_{\rm SNII}$', 13: r'$f_{\rm therm}$',
    14: r'$p_{\rm wind}$', 15: r'$\rho_{\rm wind}$', 16: r'$v_{w,\min}$',
    17: r'$\eta_{\rm w,Z}$', 18: r'$Z_{w,0}$', 19: r'$\alpha_{w,Z}$',
    20: r'$f_{\rm dump}$', 21: r'$M_{\rm seed}$', 22: r'$f_{\rm Bondi}$',
    23: r'$f_{\rm Edd}$', 24: r'$\epsilon_{\rm therm}$', 25: r'$\epsilon_r$',
    26: r'$\chi_{\rm crit}$', 27: r'$\alpha_Q$',
    28: r'$\beta_{\rm UV}$', 29: r'$\Delta z_{\rm UV}$',
    30: r'$\beta_{\rm HeII}$', 31: r'$\Delta z_{\rm HeII}$',
    32: r'$R_{\rm Ia}$', 33: r'$\alpha_{\rm Ia}$', 34: r'$\epsilon_{\rm soft}$',
}

PARAM_GROUP = {}
for j in range(N_PARAMS):
    if j in (0, 1, 6, 7, 8):
        PARAM_GROUP[j] = 'cosmo'
    elif j in (2, 4) or j in range(9, 21):
        PARAM_GROUP[j] = 'SN'
    elif j in (3, 5) or j in range(21, 28):
        PARAM_GROUP[j] = 'AGN'
    else:
        PARAM_GROUP[j] = 'other'

GROUP_COLORS = {'cosmo': '#1f77b4', 'SN': '#ff7f0e',
                'AGN': '#d62728', 'other': '#7f7f7f'}

plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif', 'mathtext.fontset': 'cm',
    'figure.dpi': 110, 'savefig.bbox': 'tight',
})


# ----------------------------------------------------------------------
# observable extractor

_yy, _xx = np.mgrid[0:PATCH_PIX, 0:PATCH_PIX] - PATCH_PIX / 2
_RR = np.sqrt(_xx ** 2 + _yy ** 2)
_BINS_PIX = np.linspace(0, PATCH_PIX / 2, N_RBINS + 1)
_BIN_COUNTS = np.array([((_RR >= _BINS_PIX[k]) & (_RR < _BINS_PIX[k + 1])).sum()
                         for k in range(N_RBINS)], dtype=np.float64)
_INNER_BINS = 8
_CENTRAL_PIX = 16


def _radial(maps2d):
    out = np.empty((len(maps2d), N_RBINS))
    for i, m in enumerate(maps2d):
        _, p = radial_profile(m, n_bins=N_RBINS)
        out[i] = p
    return out


def _axis_ratio(field_2d):
    """Single-pass mass-weighted quadrupole axis ratio of full patch.
    Returns q in (0, 1]; nan on degenerate input."""
    f = np.maximum(field_2d.astype(np.float64), 0.0)
    tot = f.sum()
    if tot <= 0:
        return np.nan
    cy = (f.sum(axis=1) * np.arange(PATCH_PIX)).sum() / tot
    cx = (f.sum(axis=0) * np.arange(PATCH_PIX)).sum() / tot
    dy = np.arange(PATCH_PIX) - cy
    dx = np.arange(PATCH_PIX) - cx
    Qxx = (f * dx[None, :] ** 2).sum() / tot
    Qyy = (f * dy[:, None] ** 2).sum() / tot
    Qxy = (f * dy[:, None] * dx[None, :]).sum() / tot
    evals = np.linalg.eigvalsh(np.array([[Qxx, Qxy], [Qxy, Qyy]]))
    if evals[1] <= 0 or evals[0] < 0:
        return np.nan
    return float(np.sqrt(evals[0] / evals[1]))


# small list of observables — chosen to span the physics dimensions described
# in parameter_dependencies.md (baryon budget, central concentration, CGM
# extent, shape, DM back-reaction). Eight observables, each per halo.
OBS_NAMES = [
    'M_star', 'M_gas', 'f_gas',
    'sigma_gas_central', 'compact_star',
    'compact_DM', 'q_DM', 'q_star',
]
OBS_LABELS = {
    'M_star':            r'$M_\star$',
    'M_gas':             r'$M_{\rm gas}$',
    'f_gas':             r'$f_{\rm gas}$',
    'sigma_gas_central': r'$\Sigma_{\rm gas,c}$',
    'compact_star':      r'$c_\star$',
    'compact_DM':        r'$c_{\rm DM}$',
    'q_DM':              r'$q_{\rm DM}$',
    'q_star':            r'$q_\star$',
}


def compute_observables(patches_NCHW: np.ndarray) -> dict:
    """Return dict[name] = (N,) array for the OBS_NAMES set.

    patches_NCHW: (N, 3, H, W) with channel order DM, gas, star (linear units).
    """
    dm, gas, star = patches_NCHW[:, 0], patches_NCHW[:, 1], patches_NCHW[:, 2]
    M_DM = dm.sum(axis=(1, 2))
    M_gas = gas.sum(axis=(1, 2))
    M_star = star.sum(axis=(1, 2))
    M_bar = M_gas + M_star + 1e-30
    f_gas = M_bar / (M_DM + M_bar + 1e-30)

    half = PATCH_PIX // 2
    s, e = half - _CENTRAL_PIX // 2, half + _CENTRAL_PIX // 2
    sigma_gas_central = gas[:, s:e, s:e].mean(axis=(1, 2))

    p_star = _radial(star)
    p_dm = _radial(dm)
    enc_star = (p_star * _BIN_COUNTS).cumsum(axis=1)
    enc_dm = (p_dm * _BIN_COUNTS).cumsum(axis=1)
    compact_star = enc_star[:, _INNER_BINS - 1] / (enc_star[:, -1] + 1e-30)
    compact_DM = enc_dm[:, _INNER_BINS - 1] / (enc_dm[:, -1] + 1e-30)

    q_dm = np.array([_axis_ratio(dm[i]) for i in range(len(patches_NCHW))])
    q_st = np.array([_axis_ratio(star[i]) for i in range(len(patches_NCHW))])

    return dict(
        M_star=M_star, M_gas=M_gas, f_gas=f_gas,
        sigma_gas_central=sigma_gas_central,
        compact_star=compact_star, compact_DM=compact_DM,
        q_DM=q_dm, q_star=q_st,
    )


# ----------------------------------------------------------------------
# parameter metadata

def load_param_meta():
    df = pd.read_csv(PARAM_META_CSV)
    return dict(
        names=df['ParamName'].tolist(),
        log=df['LogFlag'].astype(bool).tolist(),
        fid=df['FiducialVal'].astype(float).tolist(),
    )


def label(i): return PRETTY.get(i, load_param_meta()['names'][i])


# ----------------------------------------------------------------------
# Step A — extract O_NHV cube from proj5 marginals

def build_obs_cube(rebuild=False):
    """Returns:
        OBS_HPV: (N_obs, N_halos, N_params, N_grid)
        param_grids: (35, 7)
        base_params: (N_halos, 35)
    """
    if PROJ6_CACHE.exists() and not rebuild:
        z = np.load(PROJ6_CACHE, allow_pickle=False)
        OBS_HPV = z['OBS_HPV']
        return OBS_HPV, z['param_grids'], z['base_params']

    print('Extracting observables from proj5 marginals patches...')
    d = np.load(PROJ5_CACHE)
    patches = d['patches']            # (20, 35, 7, 3, 128, 128)
    param_grids = d['param_grids']    # (35, 7)
    base_params = d['base_params']    # (20, 35)
    n_h = patches.shape[0]

    OBS_HPV = np.zeros((len(OBS_NAMES), n_h, N_PARAMS, N_GRID), dtype=np.float64)
    for hi in range(n_h):
        flat = patches[hi].reshape(N_PARAMS * N_GRID, 3, PATCH_PIX, PATCH_PIX)
        feat = compute_observables(flat)
        for k, name in enumerate(OBS_NAMES):
            OBS_HPV[k, hi] = feat[name].reshape(N_PARAMS, N_GRID)
        print(f'  halo {hi + 1}/{n_h}')
    np.savez_compressed(PROJ6_CACHE,
                        OBS_HPV=OBS_HPV,
                        obs_names=np.asarray(OBS_NAMES, dtype='U30'),
                        param_grids=param_grids,
                        base_params=base_params)
    print(f'Wrote {PROJ6_CACHE} ({PROJ6_CACHE.stat().st_size / 1e6:.0f} MB)')
    return OBS_HPV, param_grids, base_params


# ----------------------------------------------------------------------
# Step B — gradient matrix g_ik = d ln O_k / d ln theta_i

def fractional_gradient(OBS_HPV, param_grids, meta):
    """Local fractional response at the fiducial grid index.

    For each (param i, observable k), average across halos and compute
    g_ik = d ln O_k / d ln theta_i  (log params)
         = (theta_fid / O_fid) * dO/dtheta  (linear params)
    using central differences across grid indices [FID_IDX-1, FID_IDX+1].

    Returns G (N_params, N_obs).
    """
    O_mean = OBS_HPV.mean(axis=1)                       # (N_obs, N_params, N_grid)
    G = np.zeros((N_PARAMS, len(OBS_NAMES)))
    for i in range(N_PARAMS):
        t = param_grids[i]
        is_log = meta['log'][i]
        x_lo, x_hi = t[FID_IDX - 1], t[FID_IDX + 1]
        if is_log:
            dx = np.log(x_hi) - np.log(x_lo)
        else:
            dx = (x_hi - x_lo) / t[FID_IDX]              # frac change in θ
        for k in range(len(OBS_NAMES)):
            o = O_mean[k, i]
            o_lo, o_hi, o_fid = o[FID_IDX - 1], o[FID_IDX + 1], o[FID_IDX]
            if not np.isfinite(o_fid) or abs(o_fid) < 1e-30:
                G[i, k] = 0.0
                continue
            dy = (o_hi - o_lo) / o_fid                   # frac change in O
            if dx == 0:
                G[i, k] = 0.0
            else:
                G[i, k] = dy / dx
    return G


# ----------------------------------------------------------------------
# Step C — clustering of parameter rows by their effect vector

def cluster_rows(G, n_clusters=6):
    """Cluster parameters by the cosine-similarity of their gradient vectors.
    Returns row-permutation that orders rows by cluster label."""
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-30)
    d = pdist(Gn, metric='cosine')
    Z = linkage(d, method='average')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    perm = np.argsort(labels * 100 + np.arange(N_PARAMS))
    return perm, labels[perm]


def fig1_gradient_heatmap(G, perm, save=True):
    Gp = G[perm]
    vmax = max(np.abs(Gp).max(), 1e-30)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7.5, 12.5))
    fig.subplots_adjust(left=0.30, right=0.88, top=0.95, bottom=0.06)

    im = ax.imshow(Gp, aspect='auto', cmap='RdBu_r', norm=norm)
    for ii in range(N_PARAMS):
        for k in range(len(OBS_NAMES)):
            v = Gp[ii, k]
            color = 'white' if abs(v) > 0.5 * vmax else 'black'
            ax.text(k, ii, f'{v:+.2f}', ha='center', va='center',
                    fontsize=6, color=color)
    ax.set_xticks(range(len(OBS_NAMES)))
    ax.set_xticklabels([OBS_LABELS[n] for n in OBS_NAMES],
                       rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(N_PARAMS))
    ax.set_yticklabels([f'{ip:>2d}  {label(ip)}' for ip in perm], fontsize=9)
    for tlab, ip in zip(ax.get_yticklabels(), perm):
        tlab.set_color(GROUP_COLORS[PARAM_GROUP[ip]])

    # narrow group strip just to the LEFT of the y-tick labels, in figure coords
    bbox = ax.get_position()
    strip_w = 0.02
    ax_grp = fig.add_axes([bbox.x0 - strip_w - 0.005, bbox.y0,
                           strip_w, bbox.height])
    grp_rgb = np.array([plt.matplotlib.colors.to_rgb(GROUP_COLORS[PARAM_GROUP[ip]])
                        for ip in perm]).reshape(N_PARAMS, 1, 3)
    ax_grp.imshow(grp_rgb, aspect='auto', extent=[0, 1, N_PARAMS - 0.5, -0.5])
    ax_grp.set_xticks([]); ax_grp.set_yticks([])
    for spine in ax_grp.spines.values():
        spine.set_visible(False)

    legend_handles = [plt.matplotlib.patches.Patch(color=c, label=g)
                      for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_handles, loc='lower left',
              bbox_to_anchor=(0.0, -0.10), ncol=4, fontsize=8, frameon=False)

    ax.set_title(r'Gradient operator  $g_{ik} = \partial \ln \mathcal{O}_k / \partial \ln \theta_i$',
                 fontsize=12, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('fractional response per fractional Δθ')
    if save:
        for ext in ('pdf', 'png'):
            fig.savefig(FIG_DIR / f'proj6_fig1_gradient_heatmap.{ext}', dpi=150)
    return fig


# ----------------------------------------------------------------------
# Step D — SVD of response operator

def svd_decompose(G):
    """G ≈ U S V^T.  Rows of G are parameters (35), cols are observables (N_obs).
    U: (35, r), S: (r,), V: (N_obs, r).  r = min(35, N_obs)."""
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    return U, S, Vt.T


def fig2_svd_modes(G, U, S, V, n_modes=5, save=True):
    fig, axes = plt.subplots(2, max(3, n_modes), figsize=(12.5, 7.0),
                             gridspec_kw={'height_ratios': [1.0, 2.0]})
    # singular spectrum
    ax = axes[0, 0]
    cum = (S ** 2).cumsum() / (S ** 2).sum()
    ax.bar(range(1, len(S) + 1), S / S.max(), color='0.4')
    ax.set_xlabel('mode'); ax.set_ylabel(r'$\sigma_i / \sigma_1$')
    ax.set_title('Singular spectrum')
    axb = ax.twinx()
    axb.plot(range(1, len(S) + 1), cum, 'o-', color='C3', ms=4)
    axb.set_ylabel('cum. variance', color='C3')
    axb.set_ylim(0, 1.05); axb.tick_params(axis='y', colors='C3')

    # observable loadings (top-2 modes shown together for compactness)
    ax = axes[0, 1]
    width = 0.35
    x = np.arange(len(OBS_NAMES))
    for m, c in zip(range(min(n_modes, 2)), ['C0', 'C1']):
        ax.bar(x + (m - 0.5) * width, V[:, m], width, color=c, label=f'mode {m+1}')
    ax.set_xticks(x); ax.set_xticklabels([OBS_LABELS[n] for n in OBS_NAMES],
                                          rotation=25, ha='right', fontsize=8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('Observable loadings (modes 1–2)')
    ax.legend(fontsize=8, frameon=False)

    # parameter contribution to total spectrum (effective rank weighted)
    ax = axes[0, 2]
    eff = (U * S[None, :]) ** 2  # (35, r)
    weight = eff.sum(axis=1)
    order = np.argsort(weight)[::-1]
    bar_colors = [GROUP_COLORS[PARAM_GROUP[i]] for i in order[:15]]
    ax.barh(range(15), weight[order[:15]] / weight.max(), color=bar_colors)
    ax.set_yticks(range(15))
    ax.set_yticklabels([label(i) for i in order[:15]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('total response weight (norm.)')
    ax.set_title('Top-15 active parameters')

    if n_modes > 3:
        for k in range(3, n_modes):
            axes[0, k].axis('off')

    # bottom row: parameter loadings of each of the first n_modes
    for m in range(n_modes):
        ax = axes[1, m]
        load = U[:, m] * S[m]
        order = np.argsort(np.abs(load))[::-1][:12]
        colors = [GROUP_COLORS[PARAM_GROUP[i]] for i in order]
        ax.barh(range(len(order)), load[order], color=colors)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([label(i) for i in order], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel(f'loading × σ_{m+1}')
        var_frac = S[m] ** 2 / (S ** 2).sum()
        ax.set_title(f'Mode {m+1}  ({100 * var_frac:.0f}% var)')

    fig.suptitle(r'SVD of the response operator $G = U \Sigma V^\top$',
                 fontsize=12)
    fig.tight_layout()
    if save:
        for ext in ('pdf', 'png'):
            fig.savefig(FIG_DIR / f'proj6_fig2_svd_modes.{ext}', dpi=150)
    return fig


# ----------------------------------------------------------------------
# Step E — pair coupling matrix

def _pair_coupling(surface_HOXY, obs_idx):
    """Average residual amplitude across selected observables on a pair surface.

    surface_HOXY: (N_halos, N_obs_in_panel, N_grid_x, N_grid_y) per-halo cube.
    obs_idx: list of observable indices to average over.

    Returns scalar C in units of fractional residual std / |mean|.
    """
    ratios = []
    for k in obs_idx:
        # mean over halos → (n_y, n_x)
        M = surface_HOXY[:, k].mean(axis=0)
        grand = M.mean()
        f_eff = M.mean(axis=0) - grand           # along x
        g_eff = M.mean(axis=1) - grand           # along y
        h = M - grand - g_eff[:, None] - f_eff[None, :]
        ratios.append(np.sqrt((h ** 2).mean()) / (abs(grand) + 1e-30))
    return float(np.mean(ratios))


def build_coupling_matrix(meta):
    """Construct 35x35 coupling matrix.  Off-diagonal entries from project-4
    pair scans (6 measured pairs, both upper and lower triangle filled).  All
    other entries left as NaN."""
    if not PROJ4_CACHE.exists():
        return None, None
    d = np.load(PROJ4_CACHE)
    patches = d['patches']     # (6, 20, 49, 3, 128, 128)
    pair_x = d['pair_x_idx']
    pair_y = d['pair_y_idx']
    pair_names = d['pair_names']

    n_pairs = patches.shape[0]
    C = np.full((N_PARAMS, N_PARAMS), np.nan)

    # Choose the four observables that span the 4 dominant physical responses
    couple_obs = ['sigma_gas_central', 'f_gas', 'M_star', 'compact_DM']
    obs_idx = [OBS_NAMES.index(n) for n in couple_obs]

    pair_records = []
    for pi in range(n_pairs):
        n_h = patches.shape[1]
        # patches[pi]: (20, 49, 3, H, W) — 49 = 7×7 cells, x sweep across, y across
        cube = np.zeros((n_h, len(OBS_NAMES), 7, 7))
        for hi in range(n_h):
            flat = patches[pi, hi].reshape(49, 3, PATCH_PIX, PATCH_PIX)
            feat = compute_observables(flat)
            for k, name in enumerate(OBS_NAMES):
                cube[hi, k] = feat[name].reshape(7, 7)  # (y, x)
        c_val = _pair_coupling(cube, obs_idx)
        i, j = int(pair_x[pi]), int(pair_y[pi])
        C[i, j] = c_val
        C[j, i] = c_val
        pair_records.append(dict(name=str(pair_names[pi]), x=i, y=j,
                                  C=c_val))
        print(f'  pair {pi+1}/{n_pairs} {pair_names[pi]}: C={c_val:.3f}')
    return C, pd.DataFrame(pair_records)


def fig3_coupling_matrix(C, pair_df, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5),
                             gridspec_kw={'width_ratios': [1.6, 1.0]})

    # Full 35x35 with NaN mask
    ax = axes[0]
    masked = np.ma.masked_invalid(C)
    cmap = plt.cm.magma_r.copy()
    cmap.set_bad('lightgrey')
    vmax = max(np.nanmax(C), 1e-30)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
    ax.set_xticks(range(N_PARAMS))
    ax.set_yticks(range(N_PARAMS))
    ax.set_xticklabels([label(i) for i in range(N_PARAMS)],
                       rotation=90, fontsize=6)
    ax.set_yticklabels([label(i) for i in range(N_PARAMS)], fontsize=6)
    ax.set_title('Pair coupling matrix $C_{ij}$ — measured pairs only\n'
                 '(grey = not yet sampled)', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02,
                 label='RMS interaction / mean')
    # group bands on tick labels
    for tlabel, idx in zip(ax.get_xticklabels(), range(N_PARAMS)):
        tlabel.set_color(GROUP_COLORS[PARAM_GROUP[idx]])
    for tlabel, idx in zip(ax.get_yticklabels(), range(N_PARAMS)):
        tlabel.set_color(GROUP_COLORS[PARAM_GROUP[idx]])

    # Right: bar chart of measured C_ij values, colored by interpretation
    ax = axes[1]
    pair_df_sorted = pair_df.sort_values('C', ascending=True)
    colors = []
    for _, r in pair_df_sorted.iterrows():
        gx, gy = PARAM_GROUP[r['x']], PARAM_GROUP[r['y']]
        if {gx, gy} == {'SN', 'AGN'}:
            colors.append('crimson')
        elif gx == 'cosmo' and gy == 'cosmo':
            colors.append('steelblue')
        else:
            colors.append('goldenrod')
    ax.barh(range(len(pair_df_sorted)), pair_df_sorted['C'], color=colors)
    ax.set_yticks(range(len(pair_df_sorted)))
    ax.set_yticklabels([f'{label(r["x"])} × {label(r["y"])}'
                        for _, r in pair_df_sorted.iterrows()], fontsize=9)
    ax.set_xlabel('coupling $C$ (fractional)')
    ax.set_title('Measured pair couplings\n(red = SN×AGN, blue = cosmo×cosmo)',
                 fontsize=10)
    ax.axvline(0.02, color='k', lw=0.7, ls='--')
    ax.text(0.02, len(pair_df_sorted) - 0.3, '  2% threshold',
            fontsize=8, color='k')

    fig.tight_layout()
    if save:
        for ext in ('pdf', 'png'):
            fig.savefig(FIG_DIR / f'proj6_fig3_coupling_matrix.{ext}', dpi=150)
    return fig


# ----------------------------------------------------------------------
# Step F — Fisher information

def fisher_matrix(G, sigma):
    """F_ij = sum_k (1/σ_k²) g_ik g_jk on the FRACTIONAL gradient G.
    sigma: (N_obs,) fractional observational uncertainty per observable."""
    W = 1.0 / np.maximum(sigma, 1e-12) ** 2
    return G @ np.diag(W) @ G.T


def estimate_obs_scatter(OBS_HPV, floor=0.10):
    """Per-observable fractional std for the Fisher weighting.

    Combines two pieces:
      (i) BIND's own intrinsic halo-to-halo scatter at the fiducial
          parameter slot, *averaged across the 35 single-parameter columns*
          so we marginalise over which axis was nominally fiducial;
      (ii) a measurement-uncertainty floor ``floor`` (default 10 %) so that
          observables with vanishingly small intrinsic scatter (e.g. f_gas
          when total mass is tightly constrained by patch-mass matching) do
          not artificially dominate the Fisher matrix.
    """
    O_at_fid = OBS_HPV[:, :, :, FID_IDX]                  # (N_obs, N_halos, N_params)
    mean_per_param = np.abs(O_at_fid.mean(axis=1))
    std_per_param = O_at_fid.std(axis=1)
    sigma_intrinsic = (std_per_param / (mean_per_param + 1e-30)).mean(axis=1)
    return np.maximum(sigma_intrinsic, floor)


def fig4_fisher(G, F, save=True, top_n=12):
    eigvals = np.sort(np.linalg.eigvalsh(F))[::-1]
    eigvals = np.maximum(eigvals, 0)
    cum = eigvals.cumsum() / max(eigvals.sum(), 1e-30)

    diag_F = np.diag(F)
    order = np.argsort(diag_F)[::-1][:top_n]
    F_sub = F[np.ix_(order, order)]
    # normalise to correlation form for visibility
    d = np.sqrt(np.maximum(np.diag(F_sub), 1e-30))
    F_corr = F_sub / np.outer(d, d)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5),
                             gridspec_kw={'width_ratios': [0.9, 1.1]})
    ax = axes[0]
    ax.semilogy(np.arange(1, len(eigvals) + 1), eigvals, 'o-', color='C0')
    ax.set_xlabel('eigenvalue rank')
    ax.set_ylabel(r'$\lambda_i(F)$  (information per direction)')
    ax.grid(alpha=0.3, which='both')
    axb = ax.twinx()
    axb.plot(np.arange(1, len(eigvals) + 1), cum, 's-', color='C3', ms=4)
    axb.set_ylabel('cumulative information', color='C3')
    axb.tick_params(axis='y', colors='C3'); axb.set_ylim(0, 1.05)
    n_eff = float((eigvals.sum()) ** 2 / max((eigvals ** 2).sum(), 1e-30))
    rank99 = int(np.searchsorted(cum, 0.99) + 1)
    ax.set_title(f'Fisher eigenvalue spectrum  '
                 f'(modes for 99% info = {rank99},  PR = {n_eff:.1f})',
                 fontsize=10)

    ax = axes[1]
    vmax = np.abs(F_corr).max()
    im = ax.imshow(F_corr, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(top_n)); ax.set_yticks(range(top_n))
    labs = [label(i) for i in order]
    ax.set_xticklabels(labs, rotation=90, fontsize=8)
    ax.set_yticklabels(labs, fontsize=8)
    for tl, idx in zip(ax.get_xticklabels(), order):
        tl.set_color(GROUP_COLORS[PARAM_GROUP[idx]])
    for tl, idx in zip(ax.get_yticklabels(), order):
        tl.set_color(GROUP_COLORS[PARAM_GROUP[idx]])
    fig.colorbar(im, ax=ax, fraction=0.045, label='normalised $F_{ij}$')
    ax.set_title(f'Fisher correlation among top-{top_n} parameters', fontsize=10)

    fig.tight_layout()
    if save:
        for ext in ('pdf', 'png'):
            fig.savefig(FIG_DIR / f'proj6_fig4_fisher.{ext}', dpi=150)
    return fig, n_eff


# ----------------------------------------------------------------------
# Step G — covariance check at multiple parameter nodes

def _load_node_observables(suite, sim_id):
    """For a 1P/CV sim, compute per-halo observables for both truth and BIND."""
    base = SUITE_ROOT / suite / sim_id / SNAP / MASS_TAG
    cuts_path = base / 'halo_cutouts.npz'
    gen_path = base / MODEL_NAME / 'generated_halos.npz'
    if not (cuts_path.exists() and gen_path.exists()):
        return None
    truth = np.load(cuts_path)['large_scale']                     # (N, 3, H, W)
    gen = np.load(gen_path)['generated']                          # (M, 3, H, W)
    n_match = min(len(truth), len(gen))
    truth, gen = truth[:n_match], gen[:n_match]
    truth_obs = compute_observables(truth)
    gen_obs = compute_observables(gen)
    return truth_obs, gen_obs


def covariance_at_nodes(node_specs):
    """node_specs: list[(label, suite, sim_id)] → returns dict[label] = dict
    with `truth_cov`, `gen_cov`, `truth_mean`, `gen_mean`, `n` keys."""
    results = {}
    for nodelabel, suite, sim_id in node_specs:
        out = _load_node_observables(suite, sim_id)
        if out is None:
            print(f'  skip {nodelabel}: artifacts missing for {suite}/{sim_id}')
            continue
        t_obs, g_obs = out
        # build per-halo observable matrix (n_halos, n_obs) in log space for
        # mass/concentration variables, linear for ratios
        log_obs = {'M_star', 'M_gas', 'sigma_gas_central'}
        T = np.column_stack([np.log10(np.maximum(t_obs[n], 1e-30)) if n in log_obs
                              else t_obs[n] for n in OBS_NAMES])
        G_ = np.column_stack([np.log10(np.maximum(g_obs[n], 1e-30)) if n in log_obs
                              else g_obs[n] for n in OBS_NAMES])
        mask = np.all(np.isfinite(T) & np.isfinite(G_), axis=1)
        T, G_ = T[mask], G_[mask]
        if len(T) < 3:
            continue
        # standardise by truth std so the two cov matrices are comparable
        scale = T.std(axis=0)
        scale[scale == 0] = 1.0
        Tn, Gn = T / scale, G_ / scale
        results[nodelabel] = dict(
            truth_cov=np.cov(Tn.T),
            gen_cov=np.cov(Gn.T),
            truth_mean=T.mean(axis=0),
            gen_mean=G_.mean(axis=0),
            n=int(len(T)),
        )
        print(f'  {nodelabel}: n_halos={len(T)}')
    return results


def _cov_distance(C1, C2):
    """Symmetric scale-free distance between two PSD matrices.
    Frobenius norm of (corr1 - corr2)."""
    def _to_corr(M):
        d = np.sqrt(np.maximum(np.diag(M), 1e-30))
        return M / np.outer(d, d)
    return float(np.linalg.norm(_to_corr(C1) - _to_corr(C2), 'fro'))


def fig5_covariance(cov_results, save=True):
    nodes = list(cov_results.keys())
    n_nodes = len(nodes)
    fig, axes = plt.subplots(2, n_nodes, figsize=(3.6 * n_nodes, 6.8))
    if n_nodes == 1:
        axes = axes.reshape(2, 1)
    abbrev = [OBS_LABELS[n] for n in OBS_NAMES]
    distances = {}
    for ci, name in enumerate(nodes):
        r = cov_results[name]
        T, G_ = r['truth_cov'], r['gen_cov']
        # render correlation matrices for visibility
        def _corr(M):
            d = np.sqrt(np.maximum(np.diag(M), 1e-30))
            return M / np.outer(d, d)
        for ri, (mat, ttl) in enumerate(zip([_corr(T), _corr(G_)],
                                             [f'truth — {name}',
                                              f'BIND — {name}'])):
            ax = axes[ri, ci]
            im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(OBS_NAMES)))
            ax.set_yticks(range(len(OBS_NAMES)))
            ax.set_xticklabels(abbrev, rotation=90, fontsize=7)
            if ci == 0:
                ax.set_yticklabels(abbrev, fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.set_title(f'{ttl}  (n={r["n"]})', fontsize=9)
        distances[name] = _cov_distance(T, G_)
        axes[1, ci].set_xlabel(f'∥corr_T − corr_G∥_F = {distances[name]:.2f}',
                                fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025, label='correlation')
    fig.suptitle('Observable covariance: truth vs BIND across parameter nodes',
                 fontsize=12)
    if save:
        for ext in ('pdf', 'png'):
            fig.savefig(FIG_DIR / f'proj6_fig5_covariance.{ext}',
                        dpi=150, bbox_inches='tight')
    return fig, distances


# ----------------------------------------------------------------------
# main

def main(rebuild=False):
    meta = load_param_meta()
    print('--- step A: extract observables on 35×7 marginal grid ---')
    OBS_HPV, param_grids, base_params = build_obs_cube(rebuild=rebuild)
    print(f'  OBS_HPV {OBS_HPV.shape}  (n_obs, n_halo, n_param, n_grid)')

    print('--- step B/C: gradient matrix + clustering + heatmap (Fig. 1) ---')
    G = fractional_gradient(OBS_HPV, param_grids, meta)
    perm, _ = cluster_rows(G, n_clusters=6)
    fig1_gradient_heatmap(G, perm)

    print('--- step D: SVD modes (Fig. 2) ---')
    U, S, V = svd_decompose(G)
    fig2_svd_modes(G, U, S, V, n_modes=5)

    print('--- step E: pair coupling matrix (Fig. 3) ---')
    C, pair_df = build_coupling_matrix(meta)
    if C is not None:
        fig3_coupling_matrix(C, pair_df)
    else:
        pair_df = pd.DataFrame()

    print('--- step F: Fisher information (Fig. 4) ---')
    sigma = estimate_obs_scatter(OBS_HPV)
    F = fisher_matrix(G, sigma)
    _, n_eff = fig4_fisher(G, F)

    print('--- step G: covariance comparison (Fig. 5) ---')
    nodes = [
        ('fiducial',     '1P', '1P_p1_0'),
        ('A_SN1 high',   '1P', '1P_p3_2'),
        ('A_AGN1 high',  '1P', '1P_p4_2'),
    ]
    covs = covariance_at_nodes(nodes)
    cov_dists = {}
    if covs:
        _, cov_dists = fig5_covariance(covs)

    summary = dict(
        n_observables=len(OBS_NAMES),
        n_parameters=N_PARAMS,
        observables=OBS_NAMES,
        singular_values=S.tolist(),
        cumvar_modes=(((S ** 2).cumsum() / (S ** 2).sum()).tolist()),
        fisher_eff_rank=n_eff,
        sigma_per_obs=sigma.tolist(),
        coupling_pairs=pair_df.to_dict(orient='records') if len(pair_df) else [],
        cov_distances=cov_dists,
    )
    out_path = FIG_DIR / 'proj6_summary.json'
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print('Wrote', out_path)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rebuild', action='store_true',
                   help='Recompute observable cube (ignores cache).')
    args = p.parse_args()
    main(rebuild=args.rebuild)
