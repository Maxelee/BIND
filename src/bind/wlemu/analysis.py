"""Correlation-aware sensitivity analysis for :mod:`bind.wlemu` (numpy-only).

The naive "global sensitivity" score used in early drafts of the tutorial and
paper figures was
``S/N = sqrt(sum_bins (Delta_bin/sigma_bin)^2)``,
i.e. it treated every bin of a statistic block as an independent detection.
The 18 :math:`C_\\ell` bins (and the 113 scattering-transform coefficients) of
a single field are strongly correlated with each other, so a coherent
amplitude shift is counted once per bin instead of once per block -- an
overcount of order :math:`\\sqrt{n_{\\rm bins}}`. Combined with a
``max``-over-sweep, the GP's own (uncorrelated-looking) interpolation noise
also adds a positive floor, so *every* parameter -- including ones with no
real physical response -- picks up a nonzero, block-dependent score, and
weakly-identified nulls can rank above genuinely responsive parameters.

This module fixes both problems with a whitened, correlation-aware score:
:func:`block_whitener` builds a low-rank whitening transform from each
block's own single-field covariance (masking zero-variance bins and capping
the rank at ``kmax`` and at :data:`DOF_CAP`, the noise-paired-realization
effective-dof guard -- see ``emulator.py``/the wlemu-v2 plan for why the
n_real=50 realizations do not give the naive ``253*49`` dof one might assume),
and :func:`sensitivity` uses it to score parameter sweeps, additionally
subtracting the GP-noise floor (estimated from the GP's own predictive sigma
at the fiducial, propagated through the same whitening transform) in
quadrature.
"""

from __future__ import annotations

import numpy as np

# Cap on the number of whitening eigenmodes kept per block, independent of
# the caller's `kmax`: the noise-paired-realization dof guard (effective dof
# is ~49 across the n_real=50 map realizations, not 253*49 -- see
# emulator.py / the wlemu-v2 plan for the derivation).
DOF_CAP = 40


def block_whitener(emu, z_idx, block: str, kmax: int = 30):
    """Whitening transform for one statistic block's single-field covariance.

    Parameters
    ----------
    emu : WLEmulator
    z_idx : int
        Source-plane index, forwarded to :meth:`WLEmulator.covariance`.
    block : str
        One of ``emu.block_names``.
    kmax : int
        Requested number of whitening eigenmodes; the actual number kept is
        ``K = min(kmax, DOF_CAP, n_unmasked_bins)``.

    Returns
    -------
    W : (K, n_unmasked) ndarray
        The whitening matrix ``Lambda_K^{-1/2} @ E_K.T``: applying it to a
        delta-vector restricted to the unmasked bins (``delta[mask]``)
        yields ``K`` approximately independent, unit-variance combinations.
    mask : (n_bins,) bool ndarray
        True for bins with nonzero single-field variance. Bins with exactly
        zero variance (e.g. the sparse tails of the peak/min histograms,
        where no realization ever populates the bin) are excluded before
        the eigendecomposition, since a zero row/column of a covariance
        matrix carries no information and would make it singular.
    """
    cov = emu.covariance(z_idx=z_idx, blocks=(block,))
    var = np.diag(cov)
    mask = var > 0
    csub = cov[np.ix_(mask, mask)]
    if csub.size == 0:
        return np.zeros((0, 0)), mask
    evals, evecs = np.linalg.eigh(csub)          # ascending
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    K = min(kmax, DOF_CAP, evals.size)
    evals = np.clip(evals[:K], 1e-300, None)
    evecs = evecs[:, :K]
    W = (evecs / np.sqrt(evals)).T                # (K, n_unmasked)
    return W, mask


def sensitivity(emu, z_idx, nv: int = 9, kmax: int = 30) -> dict:
    """Correlation-aware, noise-floor-debiased global sensitivity map.

    For each of the ``emu.n_params`` parameters, sweep it across the full
    unit-cube prior ``[0, 1]`` (all others held at
    ``emu.fiducial_params()``) and score every statistic block by the norm
    of the *whitened* response (see :func:`block_whitener`), rather than a
    per-bin sqrt-sum-of-squares.

    Returns
    -------
    dict with:
      - ``"param_names"``: list, length ``n_params``.
      - ``"blocks"``: list, length ``n_blocks`` (``emu.block_names``).
      - ``"raw"``: ``(n_params, n_blocks)`` max-over-sweep whitened score
        (no noise-floor subtraction).
      - ``"floor"``: ``(n_blocks,)`` GP-noise floor per block (independent
        of the parameter being swept): ``sqrt(tr(W diag(sigma_GP^2) W.T))``
        using the GP predictive sigma at the fiducial.
      - ``"debiased"``: ``(n_params, n_blocks)``
        ``sqrt(max(raw**2 - floor**2, 0))`` -- the recommended score for
        ranking/plotting.
    """
    blocks = list(emu.block_names)
    theta_fid = emu.fiducial_params()
    fid = emu.predict(theta_fid, z_idx=z_idx, return_std=True)

    whiteners = {b: block_whitener(emu, z_idx, b, kmax=kmax) for b in blocks}

    floor = np.zeros(len(blocks))
    for j, b in enumerate(blocks):
        W, mask = whiteners[b]
        if W.size == 0:
            continue
        sig_gp2 = fid[b + "_std"][mask] ** 2
        floor[j] = np.sqrt(np.sum((W ** 2) @ sig_gp2))

    n_params = emu.n_params
    raw = np.zeros((n_params, len(blocks)))
    for i in range(n_params):
        thetas = np.tile(theta_fid, (nv, 1))
        thetas[:, i] = np.linspace(0.0, 1.0, nv)
        v = emu.predict(thetas, z_idx=z_idx, return_std=False)
        for j, b in enumerate(blocks):
            W, mask = whiteners[b]
            if W.size == 0:
                continue
            delta = v[b][:, mask] - fid[b][mask][None, :]     # (nv, n_unmasked)
            raw[i, j] = np.linalg.norm(delta @ W.T, axis=1).max()

    debiased = np.sqrt(np.maximum(raw ** 2 - floor[None, :] ** 2, 0.0))
    return {
        "param_names": list(emu.param_names),
        "blocks": blocks,
        "raw": raw,
        "floor": floor,
        "debiased": debiased,
    }
