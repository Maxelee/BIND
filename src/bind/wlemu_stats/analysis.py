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


def _eigen_whiten(cov: np.ndarray, kmax: int = 30):
    """Zero-variance-masked, rank-capped whitening transform for a covariance
    matrix -- the recipe shared by :func:`block_whitener` (one statistic
    block) and :func:`active_subspace` (the full concatenated 383-dim
    vector). Mask out zero-variance bins, eigendecompose the remainder, keep
    the top ``K = min(kmax, DOF_CAP, n_unmasked)`` modes.

    Returns
    -------
    W : (K, n_unmasked) ndarray
        The whitening matrix ``Lambda_K^{-1/2} @ E_K.T``: applying it to a
        delta-vector restricted to the unmasked bins (``delta[mask]``)
        yields ``K`` approximately independent, unit-variance combinations.
    mask : (n,) bool ndarray
        True for bins with nonzero variance. Bins with exactly zero variance
        (e.g. the sparse tails of the peak/min histograms, where no
        realization ever populates the bin) are excluded before the
        eigendecomposition, since a zero row/column of a covariance matrix
        carries no information and would make it singular.
    """
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
    mask : (n_bins,) bool ndarray
        See :func:`_eigen_whiten`.
    """
    cov = emu.covariance(z_idx=z_idx, blocks=(block,))
    return _eigen_whiten(cov, kmax=kmax)


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


# ---------------------------------------------------------------------------
# Reduced feedback space: an active-subspace decomposition of the joint
# statistics response, rotated onto physically-labeled axes (a gas-fraction
# direction from an independent halo-integrated dataset), plus a fiducial-
# point posterior forecast along those two axes. See docs/wl_emulator.md
# ("Reduced feedback space") and docs/plans/wlemu_v2.md T4 for the recipe.
# ---------------------------------------------------------------------------


def _sobol_anchors(d: int, n: int, seed: int = 0) -> np.ndarray:
    """``n`` space-filling points in the ``d``-dim unit cube: Sobol sequence
    if scipy provides it, else a deterministic-seeded uniform fallback (same
    "Sobol-or-seeded-fallback" convention as the rest of the wlemu-v2 work,
    e.g. the T2 continuous-z LOO validation)."""
    try:
        from scipy.stats import qmc
        return qmc.Sobol(d=d, seed=seed).random(n)
    except Exception:
        return np.random.default_rng(seed).random((n, d))


def active_subspace(emu, z_idx, anchors: np.ndarray | None = None,
                     kmax: int = 40, h: float = 0.02, n_anchors: int = 16,
                     seed: int = 0) -> dict:
    """Active-subspace decomposition of the joint (all-block) statistics
    response to the 30 astro parameters.

    Builds one global whitener ``W`` over the full concatenated 383-dim
    statistics vector (same recipe as :func:`block_whitener`, see
    :func:`_eigen_whiten`), then averages the whitened-response Gram matrix
    ``J^T J`` (``J`` the whitened-output Jacobian, central finite differences
    at ``h`` in unit-cube coordinates, clipped to ``[0, 1]``) over the
    fiducial parameter point plus ``anchors`` (default: ``n_anchors``
    Sobol-sampled prior points, see :func:`_sobol_anchors`). The leading
    eigenvectors of the resulting 30x30 matrix are the directions in
    parameter space the joint statistics respond to most strongly.

    Parameters
    ----------
    emu : WLEmulator
    z_idx : int
    anchors : (n, 30) unit-cube array, optional
        Extra points (beyond the fiducial) to average the Jacobian over. If
        omitted, ``n_anchors`` Sobol points are generated internally.
    kmax : int
        Cap on the number of global whitening modes (passed to
        :func:`_eigen_whiten`; also capped at :data:`DOF_CAP`).
    h : float
        Central-difference step in unit-cube coordinates.
    n_anchors, seed : only used when ``anchors`` is not given.

    Returns
    -------
    dict with ``eigenvalues`` (descending, (30,)), ``eigenvectors`` ((30,
    30), columns matching), ``M`` (the averaged 30x30 Gram matrix), ``W``/
    ``mask`` (the global whitener, for reuse), ``anchors`` (the (n+1, 30)
    points actually used, fiducial first), and ``param_names``.
    """
    n_params = emu.n_params
    cov = emu.covariance(z_idx=z_idx)
    W, mask = _eigen_whiten(cov, kmax=kmax)

    theta_fid = emu.fiducial_params()
    if anchors is None:
        anchors = _sobol_anchors(n_params, n_anchors, seed=seed)
    anchors = np.atleast_2d(np.asarray(anchors, float))
    pts = np.vstack([theta_fid[None, :], anchors])

    idx = np.arange(n_params)
    M = np.zeros((n_params, n_params))
    for pt in pts:
        up = np.tile(pt, (n_params, 1))
        dn = np.tile(pt, (n_params, 1))
        up[idx, idx] = np.clip(pt[idx] + h, 0.0, 1.0)
        dn[idx, idx] = np.clip(pt[idx] - h, 0.0, 1.0)
        steps = up[idx, idx] - dn[idx, idx]
        steps = np.where(steps > 0, steps, np.nan)   # guard degenerate (h~0) steps
        Vp = emu.predict_vector(up, z_idx=z_idx, return_std=False)
        Vn = emu.predict_vector(dn, z_idx=z_idx, return_std=False)
        Jraw = ((Vp - Vn)[:, mask] / steps[:, None]).T    # (n_unmasked, 30)
        Jraw = np.nan_to_num(Jraw)
        Jw = W @ Jraw                                       # (K, 30)
        M += Jw.T @ Jw
    M /= len(pts)

    evals, evecs = np.linalg.eigh(M)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    return {
        "eigenvalues": evals, "eigenvectors": evecs, "M": M,
        "W": W, "mask": mask, "anchors": pts,
        "param_names": list(emu.param_names),
    }


def standardized_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Standardized-linear-regression direction: z-score every column of
    ``X`` and ``y``, fit OLS, return the coefficient vector (length
    ``X.shape[1]``). A simple, scale-free "which way does y increase
    fastest" direction in parameter space; rows with a non-finite ``y`` (or
    any non-finite ``X`` entry) are dropped before fitting.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xs = (X[m] - X[m].mean(0)) / X[m].std(0)
    ys = (y[m] - y[m].mean()) / y[m].std()
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    return beta


def rotate_to_physical_axes(eigenvectors: np.ndarray, g: np.ndarray, n_top: int = 2) -> dict:
    """Rotate the top ``n_top`` active-subspace eigenvectors onto a
    physically-labeled pair of axes: ``a1`` aligned with the projection of a
    given direction ``g`` (e.g. the gas-fraction direction from
    :func:`standardized_direction`), ``a2`` its orthogonal complement within
    ``span(eigenvectors[:, :n_top])`` (picked, in eigenvalue order, as the
    first basis vector with a nonzero residual after removing ``a1``).

    Returns
    -------
    dict with ``a1``, ``a2`` ((30,) unit vectors), ``capture`` (``||P g|| /
    ||g||``, the fraction of ``g`` captured by the ``n_top``-dim plane —
    gate this at >= 0.6, see docs/wl_emulator.md), and ``Vk`` (the ``(30,
    n_top)`` basis used).
    """
    Vk = eigenvectors[:, :n_top]
    Pg = Vk @ (Vk.T @ g)
    norm_g = np.linalg.norm(g)
    capture = float(np.linalg.norm(Pg) / norm_g) if norm_g > 0 else 0.0
    a1 = Pg / np.linalg.norm(Pg)
    a2 = None
    for j in range(n_top):
        v = Vk[:, j] - (Vk[:, j] @ a1) * a1
        if np.linalg.norm(v) > 1e-8:
            a2 = v / np.linalg.norm(v)
            break
    if a2 is None:
        raise ValueError("span(eigenvectors[:, :n_top]) is degenerate (rank < 2)")
    return {"a1": a1, "a2": a2, "capture": capture, "Vk": Vk}


def reduced_grid_theta(u_center: np.ndarray, a1: np.ndarray, a2: np.ndarray,
                        alpha1: np.ndarray, alpha2: np.ndarray, tol: float = 1e-8):
    """Unit-cube parameter grid ``theta(alpha) = u_center + alpha1*a1 +
    alpha2*a2`` over a meshgrid of ``alpha1 x alpha2``.

    Parameters
    ----------
    tol : float
        Numerical slack on the ``[0, 1]`` containment test, so that
        ``alpha1=alpha2=0`` (``theta = u_center`` exactly) is not spuriously
        invalidated by floating-point round-off when ``u_center`` itself sits
        exactly on the prior boundary in some dimension (e.g. a fiducial
        parameter pinned to its min/max).

    Returns
    -------
    A1, A2 : (n1, n2) ndarray
        The alpha meshgrid (``indexing="ij"``).
    thetas : (n1*n2, 30) ndarray
        The corresponding parameter points (not clipped).
    valid : (n1*n2,) bool ndarray
        True where ``thetas`` falls inside ``[0, 1]^30`` (within ``tol``) --
        points outside the prior box should be masked out of any
        likelihood/contour built from this grid.
    """
    A1, A2 = np.meshgrid(np.asarray(alpha1, float), np.asarray(alpha2, float), indexing="ij")
    thetas = (u_center[None, :] + A1.ravel()[:, None] * a1[None, :]
              + A2.ravel()[:, None] * a2[None, :])
    valid = np.all((thetas >= -tol) & (thetas <= 1.0 + tol), axis=1)
    return A1, A2, thetas, valid


def gaussian_chi2(data: np.ndarray, pred: np.ndarray, cov: np.ndarray,
                   hartlap: float = 1.0) -> np.ndarray:
    """Gaussian chi^2 of a batch of predictions against fixed ``data``,
    ``hartlap * (pred - data) @ inv(cov) @ (pred - data)`` per row.

    Parameters
    ----------
    data : (D,) ndarray
    pred : (N, D) ndarray
    cov : (D, D) ndarray
    hartlap : float
        Precision-matrix debiasing factor (e.g. ``(n_real - D - 2) / (n_real - 1)``).

    Returns
    -------
    (N,) ndarray of chi^2 values.
    """
    d = np.asarray(pred, float) - np.asarray(data, float)[None, :]
    Ci = np.linalg.inv(cov)
    return hartlap * np.einsum("nd,de,ne->n", d, Ci, d)


# ---------------------------------------------------------------------------
# Follow-ups to the reduced-feedback-space section: (1) a systematic
# eigenvector <-> physical-observable match (replacing the single hand-picked
# gas-fraction axis used by :func:`rotate_to_physical_axes`), and (2) a
# Laplace/Fisher approximation to the fiducial posterior directly in the
# top-K active-subspace coordinates, so the 2D (a1, a2) grid in
# :func:`reduced_grid_theta` can be extended to a K-dim corner plot without a
# combinatorial dense grid. See docs/wl_emulator.md ("Reduced feedback
# space") for the writeup and the tutorial's cells following the (a1, a2)
# posterior for worked examples of both.
# ---------------------------------------------------------------------------


def identify_axes(eigenvectors: np.ndarray, candidates: dict, k: int,
                   weak_threshold: float = 0.3) -> dict:
    """Full cosine-similarity match between the top ``k`` active-subspace
    eigenvectors and a library of candidate physical directions (e.g. the
    four ``g_*`` vectors in ``examples/data/wlemu_phys_dirs.npz``).

    Unlike :func:`rotate_to_physical_axes` (which hand-picks *one* candidate
    to define ``a1`` and only checks ``a2`` post hoc against the others),
    this scores every (eigenvector, candidate) pair symmetrically: each
    eigenvector gets its single best-matching candidate (and a "weak match"
    flag when no candidate explains it well), and each candidate gets its
    full loading vector across all ``k`` eigenvectors (a direction can spread
    across several eigenvectors rather than aligning with just one).

    Parameters
    ----------
    eigenvectors : (n_params, n_params) ndarray
        ``active_subspace(...)["eigenvectors"]``, columns are unit vectors
        in descending-eigenvalue order.
    candidates : dict[str, (n_params,) ndarray]
        Named candidate direction vectors (not required to be unit norm).
    k : int
        Number of leading eigenvectors to match.
    weak_threshold : float
        A best-match cosine with ``abs(cosine) < weak_threshold`` is flagged
        ``weak=True`` -- no candidate in the library explains that
        eigenvector well.

    Returns
    -------
    dict with:
      - ``"candidate_names"``: list, length ``n_cand``.
      - ``"cosine_matrix"``: ``(k, n_cand)`` ndarray, signed cosine
        similarity of each eigenvector (rows) against each candidate
        (columns).
      - ``"best_match"``: list of length ``k``, one dict per eigenvector:
        ``{"eigenvector": i, "candidate": name, "cosine": float, "weak": bool}``.
      - ``"candidate_loadings"``: dict, ``name -> (k,) ndarray`` -- each
        candidate's cosine similarity against every one of the ``k``
        eigenvectors (a column of ``cosine_matrix``).
    """
    names = list(candidates.keys())
    G = np.stack([np.asarray(candidates[n], float) for n in names], axis=1)  # (n_params, n_cand)
    G = G / np.linalg.norm(G, axis=0, keepdims=True)
    Vk = eigenvectors[:, :k]                                    # already unit-norm columns
    cos = Vk.T @ G                                               # (k, n_cand)

    best_idx = np.argmax(np.abs(cos), axis=1)
    best_cos = cos[np.arange(k), best_idx]
    best_match = [
        {"eigenvector": i, "candidate": names[int(best_idx[i])],
         "cosine": float(best_cos[i]), "weak": bool(abs(best_cos[i]) < weak_threshold)}
        for i in range(k)
    ]
    candidate_loadings = {names[j]: cos[:, j].copy() for j in range(len(names))}
    return {
        "candidate_names": names,
        "cosine_matrix": cos,
        "best_match": best_match,
        "candidate_loadings": candidate_loadings,
    }


def alpha_jacobian(emu, z_idx, u_center: np.ndarray, eigenvectors: np.ndarray,
                    data_mask: np.ndarray, k: int, h: float = 0.02) -> np.ndarray:
    """Jacobian of a masked data-vector's predicted mean with respect to the
    top ``k`` active-subspace coordinates ``alpha``, i.e. the columns of
    ``d(pred[data_mask])/d(alpha_1..alpha_k)`` at ``theta(alpha) = u_center +
    sum_j alpha_j * eigenvectors[:, j]``, by central finite differences at
    ``u_center`` (same ``h`` convention as :func:`active_subspace`: a step of
    ``h`` in unit-cube coordinates, clipped to ``[0, 1]``).

    Each eigenvector direction touches multiple raw parameters at once, so
    (unlike :func:`active_subspace`'s per-parameter sweep) clipping to the
    cube is not guaranteed to leave the step exactly antiparallel/parallel to
    ``eigenvectors[:, j]``; the finite-difference denominator therefore uses
    the *actual* clipped step projected back onto ``eigenvectors[:, j]``
    (``(up - dn) @ eigenvectors[:, j]``, which reduces to ``2*h`` when no
    clipping occurs) rather than the nominal ``2*h``. Directions whose
    projected step is degenerate (<= 0, i.e. ``u_center`` already pinned at
    the cube boundary along that direction) get a zero column.

    Parameters
    ----------
    emu : WLEmulator
    z_idx : int
    u_center : (n_params,) ndarray
        Expansion point (typically ``emu.fiducial_params()``).
    eigenvectors : (n_params, n_params) ndarray
    data_mask : (n_stats,) bool ndarray
        Which entries of ``emu.predict_vector`` to keep.
    k : int
    h : float

    Returns
    -------
    (n_data, k) ndarray, ``n_data = data_mask.sum()``.
    """
    Vk = eigenvectors[:, :k]
    n_data = int(np.sum(data_mask))
    J = np.zeros((n_data, k))
    for j in range(k):
        vj = Vk[:, j]
        up = np.clip(u_center + h * vj, 0.0, 1.0)
        dn = np.clip(u_center - h * vj, 0.0, 1.0)
        step = float((up - dn) @ vj)
        if not (step > 0):
            continue
        Vp = emu.predict_vector(up[None, :], z_idx=z_idx, return_std=False)[0]
        Vn = emu.predict_vector(dn[None, :], z_idx=z_idx, return_std=False)[0]
        J[:, j] = (Vp - Vn)[data_mask] / step
    return J


def laplace_alpha_covariance(J: np.ndarray, cov: np.ndarray, hartlap: float = 1.0) -> dict:
    """Laplace/Fisher approximation to the posterior covariance in
    active-subspace ``alpha`` coordinates, for a noise-free mock at the
    fiducial (MAP = fiducial exactly, by construction, so only the curvature
    at that point is needed -- no optimization).

    ``Fisher = J.T @ (hartlap * inv(cov)) @ J``; ``Sigma = inv(Fisher)``.

    Parameters
    ----------
    J : (n_data, k) ndarray
        E.g. from :func:`alpha_jacobian`.
    cov : (n_data, n_data) ndarray
        The same effective data covariance used in :func:`gaussian_chi2`
        (single-field covariance + GP predictive variance on the diagonal).
    hartlap : float
        Precision-matrix debiasing factor, as in :func:`gaussian_chi2`.

    Returns
    -------
    dict with ``"fisher"`` ((k, k)), ``"sigma"`` ((k, k), ``inv(fisher)``),
    and ``"cond"`` (condition number of ``fisher`` -- a large value flags
    near-degenerate ``alpha`` directions, i.e. the data subset used cannot
    separate them individually even though they are orthogonal in the
    whitened full-statistics sense that defined them).
    """
    Ci = hartlap * np.linalg.inv(cov)
    fisher = J.T @ Ci @ J
    sigma = np.linalg.inv(fisher)
    return {"fisher": fisher, "sigma": sigma, "cond": float(np.linalg.cond(fisher))}


def confidence_ellipse(cov2: np.ndarray, level: float = 2.30, n_pts: int = 200,
                        center=(0.0, 0.0)) -> np.ndarray:
    """Boundary points of the Gaussian confidence ellipse ``{x : (x -
    center).T @ inv(cov2) @ (x - center) = level}`` for a 2x2 covariance
    block.

    ``level`` is a 2-dof delta-chi^2 threshold (2.30 -> 68%, 6.17 -> 95%,
    matching the convention used for the dense-grid contours elsewhere in
    this module, e.g. the ``levels=[2.30, 6.17]`` calls against
    :func:`gaussian_chi2` output), *not* a "number of sigma".

    Returns
    -------
    (n_pts, 2) ndarray of (x, y) boundary points.
    """
    evals, evecs = np.linalg.eigh(np.asarray(cov2, float))
    evals = np.clip(evals, 0.0, None)
    t = np.linspace(0.0, 2.0 * np.pi, n_pts)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)            # (n_pts, 2)
    pts = circle * np.sqrt(level * evals)[None, :]
    pts = pts @ evecs.T
    return pts + np.asarray(center, float)[None, :]
