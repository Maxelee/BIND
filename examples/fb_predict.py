"""Predict baryon fraction f_b from observables with a masked observable-conditioned
BIND model (feature/observable-conditioning, --mask_observables).

Shared primitives for the sim-validation suite (fb_prediction_validation.ipynb) and
the real-data application (eROSITA groups). The model conditions on a DMO image + a
SUBSET of the R200 observables (with the baryon-mass observables withheld) and predicts
the full baryon field; f_b is then re-measured from the generated maps. For real halos,
which have no DMO image, we marginalize over simulation DMO templates at matched mass
(:func:`predict_fb_marginal`) -- the same procedure is tested on sims via the DMO-swap.

All observable definitions reuse bind.data.compute_observables, so they match training
exactly. Masks are packed as [obs*keep, keep] (the 2*N_OBS convention the masked model
was trained on).
"""
import numpy as np
import torch

from bind.data import (OBSERVABLE_KEYS, N_OBS, THERMO_KEYS, PIX_MPC_H,
                       compute_observables, _aperture_mask)
from bind.inference.pipeline import _denormalize_to_physical

__all__ = ['subset_keep', 'pack_cond', 'generate', 'aperture_fb', 'aperture_fb_profile',
           'measure_obs_from_maps', 'predict_fb_marginal']


def subset_keep(keys):
    """0/1 keep vector (N_OBS,) selecting the OBSERVABLE_KEYS named in ``keys``."""
    return np.array([k in set(keys) for k in OBSERVABLE_KEYS], dtype=np.float32)


def pack_cond(obs_norm, keep):
    """(B, N_OBS) normalized observables + keep -> (B, 2*N_OBS) [obs*keep, keep].

    ``keep`` may be (N_OBS,) or (B, N_OBS); dropped observables are zeroed and flagged
    by the mask, exactly as the masked model was trained.
    """
    obs_norm = np.asarray(obs_norm, dtype=np.float32)
    keep = np.broadcast_to(np.asarray(keep, dtype=np.float32), obs_norm.shape)
    return np.concatenate([obs_norm * keep, keep], axis=1).astype(np.float32)


@torch.no_grad()
def generate(fm, ns, cond, ls, obs_norm, keep, device, n_steps=50, seed=0):
    """Run the masked model on a batch. ``cond`` (B,1,H,W) and ``ls`` (B,3,H,W) are
    tensors; ``obs_norm`` (B, N_OBS) normalized observables; ``keep`` the subset mask.
    Fixed seed -> reproducible noise (so subsets are compared at matched noise).
    Returns physical maps (B, 3+N_THERMO, H, W) = [DM_hydro, Gas, Stars, *thermo]."""
    torch.manual_seed(seed)
    p = torch.from_numpy(pack_cond(obs_norm, keep)).to(device)
    g = fm.sample(cond.to(device), ls.to(device), p, n_steps=n_steps)
    return _denormalize_to_physical(g.float().cpu().numpy(), ns)


def aperture_fb(maps, M200, r200):
    """Projected baryon fraction f_b = (Mgas + Mstar)/M200 within R200, from physical
    maps (3+N_THERMO, H, W). M200 in the same (code) mass units as the maps."""
    ap = _aperture_mask(maps.shape[-1], r200)
    return float((maps[1][ap].sum() + maps[2][ap].sum()) / M200)


def aperture_fb_profile(maps, M200, radii_mpc):
    """Cumulative f_b(<r) at a set of circular aperture radii (Mpc/h). Denominator is
    the total matter (DM_hydro+Gas+Stars) within r, so this is the *local* baryon
    fraction profile rather than baryon/M200."""
    n = maps.shape[-1]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - c, yy - c) * PIX_MPC_H
    out = np.empty(len(radii_mpc))
    for i, r in enumerate(radii_mpc):
        m = rr < r
        bary = maps[1][m].sum() + maps[2][m].sum()
        tot = maps[0][m].sum() + bary
        out[i] = bary / tot if tot > 0 else 0.0
    return out


def measure_obs_from_maps(maps, M200, r200):
    """Re-measure the N_OBS observables from generated maps (same definition as training)."""
    d = {'target': maps[:3], 'halo_mass': float(M200), 'r200': float(r200)}
    for j, k in enumerate(THERMO_KEYS):
        d[k] = maps[3 + j]
    return compute_observables(d)


def predict_fb_marginal(fm, ns, obs_norm_row, keep, dmo_bank, ls_bank, M200, r200,
                        device, n_templates=8, n_steps=50, seed0=0):
    """Predict f_b for ONE halo's observables by marginalizing over DMO templates.

    ``obs_norm_row`` (N_OBS,) normalized observables; ``dmo_bank`` (K,1,H,W) and
    ``ls_bank`` (K,3,H,W) candidate DMO contexts (e.g. matched-mass sim halos). Returns
    the array of f_b over the sampled templates -- its spread is the DMO-marginal
    predictive uncertainty. This is the real-data path (no true DMO available) and the
    sim DMO-swap test in one function.
    """
    K = len(dmo_bank)
    sel = np.random.default_rng(seed0).choice(K, size=min(n_templates, K), replace=False)
    obs_b = np.repeat(np.asarray(obs_norm_row, np.float32)[None], len(sel), axis=0)
    maps = generate(fm, ns, dmo_bank[sel], ls_bank[sel], obs_b, keep, device, n_steps, seed0)
    return np.array([aperture_fb(maps[i], M200, r200) for i in range(len(maps))])
