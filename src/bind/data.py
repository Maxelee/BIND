"""Dataset and normalization for cosmological baryonic field painting."""

import os
import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass, field


_ASSETS_DIR = Path(__file__).parent / "assets"
# Bundled SB35 metadata (param min/max + log flags) and the per-sim parameter
# table. Override either via environment variable for a custom suite.
SB35_CSV = os.environ.get("BIND_SB35_CSV", str(_ASSETS_DIR / "SB35_param_minmax.csv"))
SB35_PARAMS_TXT = os.environ.get(
    "BIND_SB35_PARAMS_TXT",
    str(_ASSETS_DIR / "CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt"),
)
_sb_meta = pd.read_csv(SB35_CSV)
PARAM_LOG_FLAG = _sb_meta['LogFlag'].to_numpy().astype(np.int32)        # (35,)
PARAM_MIN_RAW  = _sb_meta['MinVal'].to_numpy().astype(np.float64)        # (35,)
PARAM_MAX_RAW  = _sb_meta['MaxVal'].to_numpy().astype(np.float64)        # (35,)
# Log-space bounds for log-flagged params; linear bounds otherwise.
# Use the linear value as a placeholder for non-log-flagged params before
# np.where selects, to avoid log10(negative) warnings on irrelevant entries.
_log_safe_min = np.where(PARAM_LOG_FLAG == 1, PARAM_MIN_RAW, 1.0)
_log_safe_max = np.where(PARAM_LOG_FLAG == 1, PARAM_MAX_RAW, 1.0)
PARAM_MIN_NORM = np.where(PARAM_LOG_FLAG == 1,
                          np.log10(_log_safe_min),
                          PARAM_MIN_RAW).astype(np.float32)
PARAM_MAX_NORM = np.where(PARAM_LOG_FLAG == 1,
                          np.log10(_log_safe_max),
                          PARAM_MAX_RAW).astype(np.float32)


def _default_param_log_flag():
    return PARAM_LOG_FLAG.astype(np.int32)


# Thermodynamic gas-derived fields stored alongside the mass targets in the
# rotated2_128 training files. Canonical channel order — used everywhere thermo
# channels are emitted (data, model, paint, docs).
#   compton_y   : dimensionless line-of-sight Compton-y (extensive sum)
#   temperature : mass-weighted gas temperature [K]      (intensive)
#   entropy     : mass-weighted entropy [keV cm^2]        (intensive)
#   pressure    : mass-weighted thermal pressure [Pa]      (intensive)
# All four are strictly positive with a huge dynamic range, so they use a
# per-channel log10 transform (NOT log10(1+x), which collapses sub-unity fields
# like compton_y/pressure to ~0). A per-channel floor makes the transform
# zero-safe for the rare empty (no hot gas) pixel.
THERMO_KEYS = ('compton_y', 'temperature', 'entropy', 'pressure')
N_THERMO = len(THERMO_KEYS)


def thermo_forward(x, mean, std, floor):
    """Standardized zero-safe log10 transform for thermo fields.

    t = (log10(max(x, floor)) - mean) / std.  Args may be scalars (one channel)
    or broadcastable arrays (mean/std/floor shaped (C,1,1) against x shaped
    (C,H,W)).  Shared by AstroDataset and inference so the forward/inverse math
    has a single source of truth.
    """
    return (np.log10(np.maximum(x, floor)) - mean) / (std + 1e-8)


def thermo_inverse(t, mean, std):
    """Invert :func:`thermo_forward` back to physical units (always > 0)."""
    return np.power(10.0, t * std + mean)


# ---------------------------------------------------------------------------
# Observable conditioning
# ---------------------------------------------------------------------------
# Alternative to the 35-param conditioning: condition the emulator on
# aperture-integrated *observable* scalars measured (in projection) within R200
# of each halo's own maps — the quantities an SZ/X-ray/optical survey reports —
# plus the lensing-like mass M200. The DMO map stays the image conditioning and
# the outputs (mass + thermo fields) are unchanged; only the conditioning vector
# changes. The observable vector flows through the same model slot as ``params``
# (the 'params' batch key), so the UNet/encoder are unchanged apart from
# n_params = N_OBS. See :func:`compute_observables`.

# Critical density today, h-units: rho_crit,0 = 2.7754e11 h^2 Msun/Mpc^3. With
# M in Msun/h and R in Mpc/h the explicit h's cancel, so R200c is a pure function
# of M200c at z=0 (independent of the per-sim cosmology):
#     M200c = (4/3) pi R200c^3 * 200 * rho_crit,0
RHO_CRIT_0 = 2.7754e11        # h^2 Msun / Mpc^3

# Map cutout geometry (shared with data_generation): condition/target/thermo maps
# are halo-centered projections of a 6.25 Mpc/h box on a 128^2 grid, carved from
# the central 128 px of a 1024^2 projection of the full 50 Mpc/h box. So the pixel
# side is 50/1024 Mpc/h and the halo sits at the geometric center.
BOX_MPC_H = 50.0
FULL_NPIX = 1024
PIX_MPC_H = BOX_MPC_H / FULL_NPIX     # 0.048828 Mpc/h per pixel


def m200c_to_r200c(m200c):
    """R200c [Mpc/h] from M200c [Msun/h] at z=0 (h cancels in h-units)."""
    return np.cbrt(np.asarray(m200c, np.float64)
                   / (4.0 / 3.0 * np.pi * 200.0 * RHO_CRIT_0))


# Aperture-integrated scalar observables, in canonical order. All are strictly
# positive with a huge dynamic range, so (like the thermo channels) they use a
# per-feature log10 transform with a floor, then standardization.
#   Y_200     integrated Compton-y, sum(compton_y)*pix_area  [(Mpc/h)^2]  (SZ)
#   Mgas_200  projected gas mass,  sum(target[Gas])  in r<R200 [code mass](X-ray/SZ)
#   Mstar_200 projected stellar mass, sum(target[Stars])      [code mass] (optical)
#   Tx_200    gas-mass-weighted mean temperature [K]                      (X-ray)
#   K_200     gas-mass-weighted mean entropy [keV cm^2]
#   P_200     gas-mass-weighted mean pressure [Pa]
#   M_200     halo mass M200c [Msun/h]                                    (lensing)
OBSERVABLE_KEYS = ('Y_200', 'Mgas_200', 'Mstar_200', 'Tx_200', 'K_200',
                   'P_200', 'M_200')
N_OBS = len(OBSERVABLE_KEYS)
# Map keys (beyond 'target'/'halo_mass') needed to compute the observables.
_OBS_REQUIRED_KEYS = THERMO_KEYS


def _aperture_mask(n, r200_mpc_h, pix=PIX_MPC_H):
    """Boolean (n,n) projected circular aperture r < R200 about the map center."""
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    return (np.hypot(xx - c, yy - c) * pix) < r200_mpc_h


def r200_from_sample(d):
    """R200c [Mpc/h] for a sample: prefer a saved 'r200' key, else derive it
    deterministically from the saved M200c ('halo_mass')."""
    files = getattr(d, 'files', d.keys() if hasattr(d, 'keys') else ())
    if 'r200' in files:
        return float(d['r200'])
    return float(m200c_to_r200c(float(d['halo_mass'])))


def compute_observables(d):
    """Aperture-integrated scalar observables within R200 (physical units).

    Returns a (N_OBS,) float64 vector in OBSERVABLE_KEYS order, computed from the
    halo's own maps inside the projected r<R200 circular aperture. T/K/P are
    gas-mass-weighted (weights = Gas surface density on pixels where the field is
    > 0). These are the raw physical observables; :func:`thermo_forward` then
    log/standardizes them with the obs_* stats from
    ``compute_norm_stats(..., condition_observables=True)``.
    """
    target = d['target']
    gas, star = target[1], target[2]
    n = gas.shape[-1]
    ap = _aperture_mask(n, r200_from_sample(d))
    pix_area = PIX_MPC_H ** 2

    def gas_weighted(field):
        w = gas * (field > 0)
        num = float((field.astype(np.float64) * w)[ap].sum())
        den = float(w[ap].sum())
        return num / den if den > 0 else 0.0

    return np.array([
        float(d['compton_y'][ap].sum()) * pix_area,   # Y_200
        float(gas[ap].sum()),                          # Mgas_200
        float(star[ap].sum()),                         # Mstar_200
        gas_weighted(d['temperature']),                # Tx_200
        gas_weighted(d['entropy']),                    # K_200
        gas_weighted(d['pressure']),                   # P_200
        float(d['halo_mass']),                         # M_200
    ], dtype=np.float64)


# --- Observable input-dropout (mask_observables=True) ---------------------- #
# To let the model condition on whatever *subset* of observables a survey (or
# another sim suite) actually provides, we train with random per-observable
# masking. The conditioning vector becomes 2*N_OBS: the normalized observables
# with dropped entries zeroed, concatenated with a 0/1 presence mask. The mask
# is what lets the model distinguish a genuinely mean-valued observable (which
# normalizes to ~0) from a missing one. At inference, supply only the observables
# you have and set their mask bits to 1 (see pack_observable_conditioning).
def sample_observable_keep_mask(p_full=0.25, p_keep=0.5, rng=None):
    """Random presence mask (N_OBS,) bool for input-dropout training.

    With probability ``p_full`` keep every observable (so the model stays sharp
    on the fully-observed case used for validation / full-info science); else
    keep each observable independently with probability ``p_keep`` — covering
    realistic survey subsets (e.g. SZ+X-ray+lensing = Y,Tx,M) down to a single
    observable. An all-False draw is the unconditional case, consistent with the
    classifier-free-guidance zero vector.
    """
    r = rng if rng is not None else np.random
    if r.random() < p_full:
        return np.ones(N_OBS, dtype=bool)
    return r.random(N_OBS) < p_keep


def pack_observable_conditioning(obs_norm, keep_mask):
    """Pack normalized observables + presence mask into a (2*N_OBS,) vector
    ``[obs_norm * keep_mask, keep_mask]`` (float32). Dropped entries are zeroed
    and flagged by the mask. Inverse convention is implicit: the model reads the
    second half to know which of the first half are real."""
    keep = np.asarray(keep_mask, dtype=np.float32)
    return np.concatenate([obs_norm.astype(np.float32) * keep, keep]).astype(np.float32)


@dataclass
class NormStats:
    """Per-channel normalization statistics for log10(1+x) transformed fields.

    The Stars channel can optionally be split into two heads
    (occupancy + conditional density) when ``stars_two_head=True``. In that
    mode the AstroDataset emits a 4-channel target [DM_hydro, Gas,
    occupancy_norm, density_norm] and the model has out_ch=4.
    Inference (test_suite/pipeline.py) recombines them via a soft multiplier
    before writing the standard 3-channel artifact to disk.

    When ``predict_thermo=True`` the dataset additionally appends N_THERMO gas
    thermo channels (THERMO_KEYS order) after the mass target, each normalized
    via :func:`thermo_forward`.  Defaults are safe passthroughs so old
    ``norm_stats.npz`` files load with ``predict_thermo=False`` and behave
    identically to the original mass-only pipeline.
    """
    target_mean: np.ndarray   # (3,)
    target_std: np.ndarray    # (3,)
    cond_mean: float = 0.0
    cond_std: float = 1.0
    ls_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,)
    ls_std: np.ndarray = field(default_factory=lambda: np.ones(3))    # (3,)
    # param bounds: in log10 space for params with PARAM_LOG_FLAG[j]==1.
    param_min: np.ndarray = field(default_factory=lambda: PARAM_MIN_NORM.copy())
    param_max: np.ndarray = field(default_factory=lambda: PARAM_MAX_NORM.copy())
    param_log_flag: np.ndarray = field(default_factory=_default_param_log_flag)
    # Stars two-head fields (used iff stars_two_head=True). Defaults are safe
    # passthroughs; old norm_stats.npz files load with stars_two_head=False
    # and the rest of the pipeline behaves identically to single-head.
    stars_two_head: bool = False
    stars_occ_mean: float = 0.0     # E[occ]; standardize occ → (occ-μ)/σ
    stars_occ_std: float = 1.0      # sqrt(p(1-p)) when computed from data
    stars_cond_mean: float = 0.0    # mean of log10(1+stars) on occupied pixels
    stars_cond_std: float = 1.0     # std  of log10(1+stars) on occupied pixels
    # Thermo fields (used iff predict_thermo=True).
    predict_thermo: bool = False
    thermo_mean: np.ndarray = field(default_factory=lambda: np.zeros(N_THERMO, np.float32))
    thermo_std: np.ndarray = field(default_factory=lambda: np.ones(N_THERMO, np.float32))
    thermo_floor: np.ndarray = field(default_factory=lambda: np.ones(N_THERMO, np.float32))
    # Observable-conditioning fields (used iff condition_observables=True). The
    # conditioning vector becomes N_OBS aperture-integrated observables
    # (OBSERVABLE_KEYS) instead of the 35 params; obs_* are the per-feature
    # log10-transform floor/mean/std (mirroring thermo_*). Safe defaults so old
    # files load with condition_observables=False.
    condition_observables: bool = False
    obs_mean: np.ndarray = field(default_factory=lambda: np.zeros(N_OBS, np.float32))
    obs_std: np.ndarray = field(default_factory=lambda: np.ones(N_OBS, np.float32))
    obs_floor: np.ndarray = field(default_factory=lambda: np.ones(N_OBS, np.float32))
    # Observable input-dropout (used iff condition_observables=True). When set,
    # the conditioning vector is packed to 2*N_OBS = [obs*mask, mask] and the
    # model is trained over random observable subsets, so it tolerates a missing
    # subset at inference. Default False keeps old runs at n_params=N_OBS.
    mask_observables: bool = False

    def save(self, path):
        # np.savez stores bools as 0-d arrays — cast on load.
        np.savez(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        # Backward compatibility: older norm_stats.npz lacks fields added later.
        # We default any missing key to the dataclass default.
        kwargs = {}
        for k, fld in cls.__dataclass_fields__.items():
            if k in d.files:
                v = d[k]
                if fld.type is bool or fld.type == 'bool':
                    v = bool(v)
                elif fld.type is float or fld.type == 'float':
                    v = float(v)
                kwargs[k] = v
            elif k == 'param_log_flag':
                kwargs[k] = PARAM_LOG_FLAG.astype(np.int32)
            # other missing keys fall back to the dataclass default
        return cls(**kwargs)


def log_transform(x):
    """log10(1 + x) safe for zero values."""
    return np.log10(1.0 + x)


def _compute_thermo_stats(thermo_chunks):
    """Per-channel floor/mean/std for the thermo log10 transform.

    floor = 10**(0.1 percentile of log10 over positive pixels), giving a
    zero-safe lower bound that does not distort the bulk distribution.
    mean/std are taken over ALL pixels after clipping to the floor, matching
    :func:`thermo_forward` applied at training time.
    """
    thermo_mean = np.zeros(N_THERMO, np.float32)
    thermo_std = np.ones(N_THERMO, np.float32)
    thermo_floor = np.ones(N_THERMO, np.float32)
    for j, k in enumerate(THERMO_KEYS):
        a = np.concatenate([c.ravel() for c in thermo_chunks[k]]).astype(np.float64)
        pos = a[a > 0]
        if pos.size == 0:
            continue
        floor = float(10.0 ** np.percentile(np.log10(pos), 0.1))
        t = np.log10(np.maximum(a, floor))
        thermo_floor[j] = floor
        thermo_mean[j] = float(t.mean())
        thermo_std[j] = float(max(t.std(), 1e-6))
    return {
        'predict_thermo': True,
        'thermo_mean': thermo_mean,
        'thermo_std': thermo_std,
        'thermo_floor': thermo_floor,
    }


def _compute_obs_stats(obs_rows):
    """Per-feature floor/mean/std for the observable log10 transform.

    Mirrors :func:`_compute_thermo_stats` but over the (N, N_OBS) matrix of raw
    observable vectors (one row per sampled halo, from
    :func:`compute_observables`).
    """
    obs = np.stack(obs_rows).astype(np.float64)   # (N, N_OBS)
    obs_floor = np.ones(N_OBS, np.float32)
    obs_mean = np.zeros(N_OBS, np.float32)
    obs_std = np.ones(N_OBS, np.float32)
    for j in range(N_OBS):
        col = obs[:, j]
        pos = col[col > 0]
        if pos.size == 0:
            continue
        floor = float(10.0 ** np.percentile(np.log10(pos), 0.1))
        t = np.log10(np.maximum(col, floor))
        obs_floor[j] = floor
        obs_mean[j] = float(t.mean())
        obs_std[j] = float(max(t.std(), 1e-6))
    return {
        'condition_observables': True,
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'obs_floor': obs_floor,
    }


def compute_norm_stats(file_list, n_samples=5000, seed=42, stars_two_head=False,
                       predict_thermo=False, condition_observables=False):
    """Compute normalization statistics from a random subset of files.

    When ``stars_two_head=True`` we additionally compute statistics for the
    Stars two-head training:
      - stars_occ_mean / stars_occ_std: standardization for the binary
        occupancy mask (mean = occupied fraction, std = sqrt(p*(1-p)))
      - stars_cond_mean / stars_cond_std: mean & std of log10(1+stars)
        computed only over occupied pixels (avoids the zero-pixel
        domination that biases the standard target_mean[2] / target_std[2]).

    When ``predict_thermo=True`` we also compute per-channel floor/mean/std
    for the THERMO_KEYS fields (see :func:`_compute_thermo_stats`).

    When ``condition_observables=True`` we also compute per-feature
    floor/mean/std for the aperture-integrated observables (OBSERVABLE_KEYS,
    see :func:`compute_observables` / :func:`_compute_obs_stats`). This needs the
    thermo maps, so files lacking them are skipped.
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(file_list), min(n_samples, len(file_list)), replace=False)

    targets, conds, large_scales, params = [], [], [], []
    raw_stars_chunks = []  # only populated when stars_two_head
    thermo_chunks = {k: [] for k in THERMO_KEYS}  # only populated when predict_thermo
    obs_rows = []  # only populated when condition_observables
    for idx in indices:
        d = np.load(file_list[idx])
        targets.append(log_transform(d['target']))        # (3,128,128)
        conds.append(log_transform(d['condition']))        # (128,128)
        large_scales.append(log_transform(d['large_scale']))  # (3,128,128)
        params.append(d['params'])
        if stars_two_head:
            raw_stars_chunks.append(d['target'][2])        # raw mass density
        # Not every no-lowmass sim has thermo maps (the thermo job processed
        # most sims, not all), so skip files lacking the keys here.
        if predict_thermo and all(k in d.files for k in THERMO_KEYS):
            for k in THERMO_KEYS:
                thermo_chunks[k].append(d[k])
        if condition_observables and all(k in d.files for k in _OBS_REQUIRED_KEYS):
            obs_rows.append(compute_observables(d))

    targets = np.stack(targets)       # (N,3,128,128)
    conds = np.stack(conds)           # (N,128,128)
    large_scales = np.stack(large_scales)  # (N,3,128,128)
    params = np.stack(params)         # (N,35)

    target_mean = targets.mean(axis=(0, 2, 3)).astype(np.float32)
    target_std = targets.std(axis=(0, 2, 3)).astype(np.float32)

    extra = {}
    if stars_two_head:
        raw_stars = np.stack(raw_stars_chunks)              # (N,128,128)
        occ = (raw_stars > 0).astype(np.float32)
        p = float(occ.mean())                                # occupied fraction
        # std of a Bernoulli; clip to avoid std=0 when the sample is degenerate
        occ_std = float(np.sqrt(max(p * (1 - p), 1e-8)))
        log_density = np.log10(1.0 + raw_stars)
        occ_mask = occ > 0
        if occ_mask.any():
            cond_mean = float(log_density[occ_mask].mean())
            cond_std  = float(log_density[occ_mask].std())
        else:
            cond_mean, cond_std = 0.0, 1.0
        extra = {
            'stars_two_head': True,
            'stars_occ_mean': p,
            'stars_occ_std':  occ_std,
            'stars_cond_mean': cond_mean,
            'stars_cond_std':  max(cond_std, 1e-6),
        }

    if predict_thermo:
        extra.update(_compute_thermo_stats(thermo_chunks))

    if condition_observables:
        if not obs_rows:
            raise RuntimeError(
                'condition_observables=True but none of the sampled files had the '
                'thermo maps required to compute observables. Point at the '
                'rotated2_128 (thermo) data path, or raise n_samples.')
        extra.update(_compute_obs_stats(obs_rows))

    return NormStats(
        target_mean=target_mean,
        target_std=target_std,
        cond_mean=float(conds.mean()),
        cond_std=float(conds.std()),
        ls_mean=large_scales.mean(axis=(0, 2, 3)).astype(np.float32),
        ls_std=large_scales.std(axis=(0, 2, 3)).astype(np.float32),
        # Bounds come from the SB35 csv (log10 where flagged) so the
        # normalization range is well-defined for any sim, not just the
        # training subset. See note in NormStats docstring above.
        param_min=PARAM_MIN_NORM.copy(),
        param_max=PARAM_MAX_NORM.copy(),
        param_log_flag=PARAM_LOG_FLAG.astype(np.int32),
        **extra,
    )


def load_file_list(data_root, split='train'):
    """Load file paths from the precomputed no-lowmass cache."""
    cache = Path(data_root) / split / 'file_list_cache_no_lowmass.txt'
    with open(cache) as f:
        return [line.strip() for line in f if line.strip()]


class AstroDataset(Dataset):
    """Dataset for cosmological baryonic field painting.
    
    Returns dict with keys: target (3,128,128), condition (1,128,128),
    large_scale (3,128,128), params (N,) — all normalized.

    If ``param_indices`` is provided (a list/array of integer indices), only
    those columns of the 35-param vector are returned.  Normalization is still
    computed over all 35 params first; the selection happens afterwards, so
    ``norm_stats`` does not need to change.
    """

    def __init__(self, file_list, norm_stats, param_indices=None):
        self.file_list = file_list
        self.ns = norm_stats
        self.param_indices = (
            np.asarray(param_indices, dtype=np.int64)
            if param_indices is not None else None
        )

    def __len__(self):
        return len(self.file_list)

    def _normalize(self, x, mean, std):
        return (x - mean) / (std + 1e-8)

    def _normalize_params(self, p):
        # log10 first for log-flagged params, then min/max scale to [0, 1].
        # Bounds in self.ns.param_min/max are already in log space for the
        # log-flagged entries.
        p = np.where(self.ns.param_log_flag == 1,
                     np.log10(np.maximum(p, 1e-30)),
                     p)
        rang = self.ns.param_max - self.ns.param_min
        return (p - self.ns.param_min) / (rang + 1e-8)

    def _build_target(self, raw_target):
        """Return either (3, H, W) standard or (4, H, W) two-head target.

        Two-head layout: [DM_hydro, Gas, occupancy_norm, density_norm].
        For unoccupied pixels the conditional density_norm is set to 0
        (the model is free to predict anything there because it gets
        masked at inference; using 0 keeps gradients well-behaved).
        """
        if not self.ns.stars_two_head:
            target = log_transform(raw_target)              # (3,H,W)
            return self._normalize(
                target, self.ns.target_mean[:, None, None],
                self.ns.target_std[:, None, None],
            )

        # Standard standardization for DM_hydro and Gas
        log_dm = log_transform(raw_target[0])
        log_gas = log_transform(raw_target[1])
        dm = (log_dm - self.ns.target_mean[0]) / (self.ns.target_std[0] + 1e-8)
        gas = (log_gas - self.ns.target_mean[1]) / (self.ns.target_std[1] + 1e-8)

        # Stars split into (occupancy, conditional log-density)
        raw_stars = raw_target[2]
        occ = (raw_stars > 0).astype(np.float32)
        log_density = log_transform(raw_stars)
        occ_norm = (occ - self.ns.stars_occ_mean) / (self.ns.stars_occ_std + 1e-8)
        density_norm = (log_density - self.ns.stars_cond_mean) / (
            self.ns.stars_cond_std + 1e-8
        )
        # Mask conditional density at unoccupied pixels; the soft multiplier
        # at inference will multiply through by ~0 there anyway.
        density_norm = np.where(occ > 0, density_norm, 0.0)

        return np.stack([dm, gas, occ_norm, density_norm]).astype(np.float32)

    def _build_thermo(self, d):
        """Return (N_THERMO, H, W) normalized thermo channels in THERMO_KEYS order."""
        chans = [
            thermo_forward(d[k].astype(np.float64), self.ns.thermo_mean[j],
                           self.ns.thermo_std[j], self.ns.thermo_floor[j])
            for j, k in enumerate(THERMO_KEYS)
        ]
        return np.stack(chans).astype(np.float32)

    def _build_item(self, d):
        target = self._build_target(d['target'])
        if self.ns.predict_thermo:
            target = np.concatenate([target, self._build_thermo(d)], axis=0)

        cond = log_transform(d['condition'])[None]  # (1,128,128)
        cond = self._normalize(cond, self.ns.cond_mean, self.ns.cond_std)

        ls = log_transform(d['large_scale'])  # (3,128,128)
        ls = self._normalize(ls, self.ns.ls_mean[:, None, None],
                             self.ns.ls_std[:, None, None])

        # The 'params' slot carries either the normalized 35-param vector or, in
        # observable-conditioning mode, the normalized N_OBS observable vector.
        if self.ns.condition_observables:
            params = thermo_forward(
                compute_observables(d), self.ns.obs_mean,
                self.ns.obs_std, self.ns.obs_floor,
            ).astype(np.float32)
            if self.ns.mask_observables:
                # Random observable subset -> 2*N_OBS [obs*mask, mask] vector.
                params = pack_observable_conditioning(
                    params, sample_observable_keep_mask())
        else:
            params = self._normalize_params(d['params'].astype(np.float32))
            if self.param_indices is not None:
                params = params[self.param_indices]

        return {
            'target': torch.from_numpy(target.astype(np.float32)),
            'condition': torch.from_numpy(cond.astype(np.float32)),
            'large_scale': torch.from_numpy(ls.astype(np.float32)),
            'params': torch.from_numpy(params),
        }

    def __getitem__(self, idx):
        # The thermo maps are required either as prediction targets
        # (predict_thermo) or to build the observable conditioning
        # (condition_observables); without them a file is unusable.
        needs_maps = self.ns.predict_thermo or self.ns.condition_observables
        if not needs_maps:
            return self._build_item(np.load(self.file_list[idx]))
        # Most — but not all — no-lowmass sims have thermo maps appended. A file
        # lacking them is unusable, so resample a random index. Missing files
        # cluster by sim (consecutive in the list), so a random jump escapes
        # immediately where linear probing could get stuck.
        for _ in range(100):
            d = np.load(self.file_list[idx])
            if all(k in d.files for k in THERMO_KEYS):
                return self._build_item(d)
            idx = np.random.randint(len(self.file_list))
        raise RuntimeError('No file with all THERMO_KEYS found in 100 tries')


def get_loaders(data_root, norm_stats, batch_size=64, num_workers=8,
                max_train=None, max_test=None):
    """Create train and test dataloaders."""
    train_files = load_file_list(data_root, 'train')
    test_files = load_file_list(data_root, 'test')
    if max_train:
        train_files = train_files[:max_train]
    if max_test:
        test_files = test_files[:max_test]

    train_ds = AstroDataset(train_files, norm_stats)
    test_ds = AstroDataset(test_files, norm_stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Cube dataset (6.25 Mpc/h^3 box projections — no large-scale conditioning)
# ---------------------------------------------------------------------------

def load_file_list_cube(data_root, split='train'):
    """Recursively enumerate all .npz files under data_root/split/.

    On first call the result is written to a cache file
    (file_list_cache.txt inside the split directory) so that subsequent
    runs on slow distributed filesystems skip the expensive rglob scan.
    """
    root = Path(data_root) / split
    cache = root / 'file_list_cache.txt'
    if cache.exists():
        with open(cache) as f:
            files = [line.strip() for line in f if line.strip()]
        if files:
            return files
    # Cache missing or empty — scan and write it.
    files = sorted(str(p) for p in root.rglob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No .npz files found under {root}')
    with open(cache, 'w') as f:
        f.write('\n'.join(files) + '\n')
    print(f'[load_file_list_cube:{split}] cached {len(files)} paths → {cache}')
    return files


def compute_norm_stats_cube(file_list, n_samples=5000, seed=42,
                            stars_two_head=False):
    """Compute normalization statistics for the cube (no-large-scale) dataset.

    Files are expected to have keys: dm, dm_hydro, gas, star, conditional_params.
    Returns a NormStats instance; ls_mean/ls_std are kept as zeros/ones because
    this dataset has no large-scale conditioning channel.
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(file_list), min(n_samples, len(file_list)), replace=False)

    targets, conds, params_list = [], [], []
    raw_stars_chunks = []
    for idx in indices:
        d = np.load(file_list[idx])
        raw_target = np.stack([d['dm_hydro'], d['gas'], d['star']])  # (3,128,128)
        targets.append(log_transform(raw_target))
        conds.append(log_transform(d['dm']))
        params_list.append(d['conditional_params'])
        if stars_two_head:
            raw_stars_chunks.append(d['star'])

    targets = np.stack(targets)        # (N,3,128,128)
    conds = np.stack(conds)            # (N,128,128)
    params_arr = np.stack(params_list) # (N,35)

    target_mean = targets.mean(axis=(0, 2, 3)).astype(np.float32)
    target_std = targets.std(axis=(0, 2, 3)).astype(np.float32)

    extra = {}
    if stars_two_head:
        raw_stars = np.stack(raw_stars_chunks)
        occ = (raw_stars > 0).astype(np.float32)
        p = float(occ.mean())
        occ_std = float(np.sqrt(max(p * (1 - p), 1e-8)))
        log_density = np.log10(1.0 + raw_stars)
        occ_mask = occ > 0
        if occ_mask.any():
            cond_mean = float(log_density[occ_mask].mean())
            cond_std  = float(log_density[occ_mask].std())
        else:
            cond_mean, cond_std = 0.0, 1.0
        extra = {
            'stars_two_head': True,
            'stars_occ_mean': p,
            'stars_occ_std':  occ_std,
            'stars_cond_mean': cond_mean,
            'stars_cond_std':  max(cond_std, 1e-6),
        }

    return NormStats(
        target_mean=target_mean,
        target_std=target_std,
        cond_mean=float(conds.mean()),
        cond_std=float(conds.std()),
        # No large-scale conditioning; keep neutral defaults for compatibility.
        ls_mean=np.zeros(3, dtype=np.float32),
        ls_std=np.ones(3, dtype=np.float32),
        param_min=PARAM_MIN_NORM.copy(),
        param_max=PARAM_MAX_NORM.copy(),
        param_log_flag=PARAM_LOG_FLAG.astype(np.int32),
        **extra,
    )


def load_sb35_param_table(path=SB35_PARAMS_TXT):
    """Load the SB35 parameter table.

    Returns a numpy array of shape (N, 35) where index i corresponds to
    simulation SB35_i.  Columns are the 35 raw parameter values (no log
    transform applied); the seed column is dropped.
    """
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f):
            if lineno == 0:  # header
                continue
            cols = line.split()
            # cols[0] = 'SB35_N', cols[1:-1] = 35 params, cols[-1] = seed
            rows.append([float(v) for v in cols[1:-1]])
    return np.array(rows, dtype=np.float64)  # (N, 35)


class CubeAstroDataset(Dataset):
    """Dataset for cube-projected fields (6.25 Mpc/h^3 box, no large-scale).

    File keys: dm (condition), dm_hydro / gas / star (targets),
    conditional_params (35 SB35 parameters).  When conditional_params is
    absent (e.g. test splits generated without it), the parameters are looked
    up from the SB35 params table using the sim number encoded in the parent
    directory name (e.g. "sim_1001" → row 1001).

    Returns dict with keys: target (C,128,128), condition (1,128,128),
    params (N,) — all normalized.  No 'large_scale' key is emitted.
    """

    def __init__(self, file_list, norm_stats, param_indices=None,
                 sb35_table=None):
        self.file_list = file_list
        self.ns = norm_stats
        self.param_indices = (
            np.asarray(param_indices, dtype=np.int64)
            if param_indices is not None else None
        )
        # Loaded lazily on first miss; can be pre-supplied to avoid I/O.
        self._sb35_table = sb35_table

    def __len__(self):
        return len(self.file_list)

    def _lookup_sb35_params(self, file_path):
        """Return raw (35,) float32 params for file_path via the SB35 table."""
        if self._sb35_table is None:
            self._sb35_table = load_sb35_param_table()
        m = re.search(r'sim_(\d+)', str(file_path))
        if m is None:
            raise KeyError(f'Cannot extract sim index from path: {file_path}')
        idx = int(m.group(1))
        return self._sb35_table[idx].astype(np.float32)

    def _normalize(self, x, mean, std):
        return (x - mean) / (std + 1e-8)

    def _normalize_params(self, p):
        p = np.where(self.ns.param_log_flag == 1,
                     np.log10(np.maximum(p, 1e-30)),
                     p)
        rang = self.ns.param_max - self.ns.param_min
        return (p - self.ns.param_min) / (rang + 1e-8)

    def _build_target(self, raw_target):
        """Return (3,H,W) or (4,H,W) two-head target — same logic as AstroDataset."""
        if not self.ns.stars_two_head:
            target = log_transform(raw_target)
            return self._normalize(
                target, self.ns.target_mean[:, None, None],
                self.ns.target_std[:, None, None],
            )

        log_dm = log_transform(raw_target[0])
        log_gas = log_transform(raw_target[1])
        dm = (log_dm - self.ns.target_mean[0]) / (self.ns.target_std[0] + 1e-8)
        gas = (log_gas - self.ns.target_mean[1]) / (self.ns.target_std[1] + 1e-8)

        raw_stars = raw_target[2]
        occ = (raw_stars > 0).astype(np.float32)
        log_density = log_transform(raw_stars)
        occ_norm = (occ - self.ns.stars_occ_mean) / (self.ns.stars_occ_std + 1e-8)
        density_norm = (log_density - self.ns.stars_cond_mean) / (
            self.ns.stars_cond_std + 1e-8
        )
        density_norm = np.where(occ > 0, density_norm, 0.0)

        return np.stack([dm, gas, occ_norm, density_norm]).astype(np.float32)

    def __getitem__(self, idx):
        # A small number of files may be unreadable; skip them by trying the
        # next few indices.  Cap at 100 to avoid a runaway loop on ceph.
        for attempt in range(min(100, len(self.file_list))):
            try:
                return self._load_item((idx + attempt) % len(self.file_list))
            except (KeyError, Exception):
                continue
        raise RuntimeError(f'No valid item found in 100 attempts starting from index {idx}')

    def _load_item(self, idx):
        d = np.load(self.file_list[idx])

        raw_target = np.stack([
            d['dm_hydro'].astype(np.float64),
            d['gas'].astype(np.float64),
            d['star'].astype(np.float64),
        ])  # (3,128,128)
        target = self._build_target(raw_target)

        cond = log_transform(d['dm'])[None]  # (1,128,128)
        cond = self._normalize(cond, self.ns.cond_mean, self.ns.cond_std)

        # conditional_params may be absent in some splits; fall back to SB35
        # table lookup using the sim number in the parent directory name.
        if 'conditional_params' in d.files:
            raw_params = d['conditional_params'].astype(np.float32)
        else:
            raw_params = self._lookup_sb35_params(self.file_list[idx])
        params = self._normalize_params(raw_params)
        if self.param_indices is not None:
            params = params[self.param_indices]

        return {
            'target': torch.from_numpy(target.astype(np.float32)),
            'condition': torch.from_numpy(cond.astype(np.float32)),
            'params': torch.from_numpy(params),
        }
