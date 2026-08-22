"""Spherical baryon-correction control for the WL anisotropy study.

The science question (``examples/wl_anisotropy_paper.ipynb``): *how important is the
**anisotropic** part of the baryonic back-reaction for weak-lensing statistics?*

A spherical baryon-correction model (BCM, Schneider & Teyssier) displaces each
halo's particles **radially**, ``r -> r + d(r)``, to match the spherically-averaged
hydro profile.  Crucially it *keeps the halo's triaxiality and substructure* — a
radial displacement of a triaxial DMO halo still produces an **anisotropic**
correction.  The faithful projected counterfactual (:func:`bcm_warp_patch`,
``mode="bcm_warp"``) therefore warps each DMO patch radially to the hydro monopole
so that its triaxial structure **co-moves with the mass flow**; only the feedback's
own angular structure (bipolar outflows, anisotropic ejection) separates it from
BIND.  A coarser control (:func:`azimuthal_average_patch`, ``mode="monopole"``)
keeps only the *monopole of the correction* ``DeltaSigma = Sigma_hydro - Sigma_dmo``,
which forces the correction itself to be circular; the gap between the two measures
how much apparent "anisotropy" is merely triaxiality-tracing that a real BCM would
reproduce.  Either way the counterfactual is built *without touching 3D particles
or re-running N-body* — from the per-halo patches already in each
``composite_slab*.npz``, re-composited onto the same DMO background with the same
circular-taper weights.

Pushing each total through the *identical* ray-tracing
(:func:`bind.inference.lightcone_maps.assemble_lightcone`) and statistics pipeline,
with **matched realization seeds**, gives the paired decomposition for any statistic
``X`` (``Cl(ell)``, peaks/minima ``N(nu)``, PDF, moments, Minkowski functionals):

    dX_baryon = X_bind - X_dmo            # total baryonic effect
    dX_sph    = X_sph  - X_dmo            # what a (faithful, radial) BCM captures
    dX_aniso  = X_bind - X_sph            # feedback anisotropy a BCM misses  <- answer
    f_aniso   = dX_aniso / dX_baryon
    # secondary: dX_mono = X_mono - X_dmo; (X_sph - X_mono) isolates triaxiality-tracing

Because the totals come from the same halos with the same per-snapshot transverse
shifts, cosmic variance and shape noise cancel in ``dX_aniso``.

This module only rebuilds the **total-matter** slab (channel-summed), which is all
:func:`assemble_lightcone` needs for ``kappa``.  Writing a parallel ``snap_<NNN>/
composite_slab*.npz`` tree whose single-channel ``composite`` / ``composite_mono`` /
``composite_bind`` keys (each ``.sum(0)`` a total-matter map) lets the unmodified
``bind-lightcone-maps`` / ``bind-lightcone-stats`` CLIs run on it via ``--mass_key``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pipeline import circular_taper_weight, paste_halos_2d, square_taper_weight


def azimuthal_average_patch(patch: np.ndarray) -> np.ndarray:
    """Render the azimuthal (monopole) average of a square patch about its centre.

    Each pixel is replaced by the mean of its 1-pixel-wide radial annulus, so the
    patch total is **conserved exactly** (pure radial redistribution) while all
    ``m >= 1`` angular structure — the quadrupole a spherical BCM discards — is
    removed.  ``patch`` is ``(P, P)``; the centre is the geometric centre, which
    is where halo cutouts place the halo.
    """
    pp = patch.shape[0]
    c = pp // 2                  # halo sits at index size//2 (extract_periodic_cutout)
    yy, xx = np.mgrid[0:pp, 0:pp]
    r = np.hypot(xx - c, yy - c)
    rint = np.rint(r).astype(np.int64)
    nbin = rint.max() + 1
    flat = patch.ravel().astype(np.float64)
    idx = rint.ravel()
    tot = np.bincount(idx, weights=flat, minlength=nbin)
    cnt = np.bincount(idx, minlength=nbin)
    prof = tot / np.where(cnt > 0, cnt, 1)
    return prof[rint].astype(patch.dtype, copy=False)


def _radial_cumulative(patch: np.ndarray, c: int) -> np.ndarray:
    """Enclosed (cumulative) monopole mass of ``patch`` on integer-radius bins.

    ``cum[k]`` is the total mass within ``round(r) <= k`` about centre ``c`` — the
    projected enclosed-mass profile ``M(<r)`` that defines the BCM displacement.
    """
    pp = patch.shape[0]
    yy, xx = np.mgrid[0:pp, 0:pp]
    rint = np.rint(np.hypot(xx - c, yy - c)).astype(np.int64)
    nbin = int(rint.max()) + 1
    mass = np.bincount(rint.ravel(), weights=patch.ravel().astype(np.float64),
                       minlength=nbin)
    return np.cumsum(mass)


def _cic_deposit(tx: np.ndarray, ty: np.ndarray, m: np.ndarray, pp: int) -> np.ndarray:
    """Cloud-in-cell scatter of point masses ``m`` at float coords ``(tx, ty)``.

    ``tx`` is the column (x) and ``ty`` the row (y); output is indexed ``[y, x]``.
    Mass is conserved up to the small fraction pushed off the ``pp x pp`` grid.
    """
    out = np.zeros((pp, pp), dtype=np.float64)
    tx = np.clip(tx, 0.0, pp - 1.0 - 1e-6)
    ty = np.clip(ty, 0.0, pp - 1.0 - 1e-6)
    x0 = np.floor(tx).astype(np.int64)
    y0 = np.floor(ty).astype(np.int64)
    fx = tx - x0
    fy = ty - y0
    for dxg, dyg, wx, wy in ((0, 0, 1 - fx, 1 - fy), (1, 0, fx, 1 - fy),
                             (0, 1, 1 - fx, fy), (1, 1, fx, fy)):
        np.add.at(out, (np.clip(y0 + dyg, 0, pp - 1),
                        np.clip(x0 + dxg, 0, pp - 1)), wx * wy * m)
    return out


def bcm_warp_patch(dcut: np.ndarray, hydro: np.ndarray) -> np.ndarray:
    """2D radial-displacement BCM: warp the DMO patch to the hydro monopole.

    The literal Schneider–Teyssier picture in projection: build the purely radial
    displacement ``r -> r'(r)`` that maps the DMO enclosed-mass profile onto the
    (mass-matched) ``hydro`` one — ``M_dmo(<r) = M_hydro(<r')`` — then push every
    DMO pixel **radially** by it, keeping its azimuthal angle fixed.  The output's
    *monopole* equals the hydro profile by construction, while all ``m >= 1``
    angular structure (triaxiality, substructure, filaments) is inherited from the
    DMO halo and **co-moves with the radial mass flow** — exactly what a radial BCM
    does, and unlike :func:`azimuthal_average_patch`, which leaves the DMO structure
    at its original radius and adds an isotropic ring.  Total mass is conserved
    (CIC deposit); ``r'(r) -> r`` where the profiles agree, so the outskirts are
    untouched.  ``dcut`` and ``hydro`` are ``(P, P)`` with the halo at the centre.
    """
    pp = dcut.shape[0]
    c = pp // 2
    yy, xx = np.mgrid[0:pp, 0:pp]
    dx = (xx - c).astype(np.float64)
    dy = (yy - c).astype(np.float64)
    r = np.hypot(dx, dy)

    cum_d = _radial_cumulative(dcut, c)
    cum_h = _radial_cumulative(hydro, c)
    rad = np.arange(cum_d.size, dtype=np.float64)
    # invert M_hydro(<r') = M_dmo(<r): xp must be strictly increasing, so add a
    # tiny ramp to break flat (zero-mass) annuli — negligible vs the enclosed mass.
    cum_h_mono = np.maximum.accumulate(cum_h) + 1e-9 * rad
    g_at_bin = np.interp(cum_d, cum_h_mono, rad)        # r -> r' on the bin grid
    rprime = np.interp(r.ravel(), rad, g_at_bin).reshape(pp, pp)
    scale = np.divide(rprime, r, out=np.zeros_like(r), where=r > 0)

    out = _cic_deposit((c + dx * scale).ravel(), (c + dy * scale).ravel(),
                       dcut.ravel().astype(np.float64), pp)
    return out.astype(dcut.dtype, copy=False)


def spherical_total_slab(
    d,
    *,
    r200_factor: float = 4.0,
    taper_frac: float = 0.15,
    mode: str = "bcm_warp",
    symmetrize: bool | None = None,
    dmo: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct a composite slab's **total-matter** map under one BCM counterfactual.

    Reproduces :func:`bind.inference.pipeline.build_bind_composite`'s total-matter
    field from the per-halo arrays saved in a ``composite_slab*.npz`` (``d``):
    mass-matched total patches pasted onto the ``dmo`` background with the same
    circular Hann taper of radius ``r200_factor * R200c``, then a mass-conserving
    global rescale.  ``mode`` selects how each halo's patch is built:

    - ``"bcm_warp"`` (default) — the **faithful** projected Schneider–Teyssier BCM
      (:func:`bcm_warp_patch`): displace the DMO patch *radially* to the hydro
      monopole, so triaxiality/substructure co-move with the mass flow and only the
      feedback's angular structure separates it from BIND.
    - ``"monopole"`` — add the *azimuthally averaged* baryonic correction
      (``DeltaSigma = Sigma_hydro - Sigma_dmo``) back onto the unchanged DMO.  This
      is "monopole-only correction in projection": it keeps the DMO field's own
      triaxiality but forces the *correction* to be circular, leaving DMO structure
      at its original radius.  A coarser control than ``bcm_warp`` — the gap between
      the two measures how much apparent anisotropy is just triaxiality-tracing.
    - ``"bind"`` — the full BIND total (a re-paste matching the stored
      ``composite.sum(0)`` to round-off), so all modes share one code path.

    ``symmetrize`` is the legacy toggle: ``True`` maps to ``"monopole"``, ``False``
    to ``"bind"``; when given it overrides ``mode``.

    ``dmo`` may be passed explicitly for runs whose slabs do not store it (the
    twobound/truth composites share the DMO background, saved only in the fid tree).

    Returns the ``(npix, npix)`` total-matter map (float32).
    """
    if symmetrize is not None:                 # back-compat with the boolean API
        mode = "monopole" if symmetrize else "bind"
    if mode not in ("bcm_warp", "monopole", "bind"):
        raise ValueError(f"unknown mode {mode!r}")
    dmo = (d["dmo"] if dmo is None else dmo).astype(np.float64)
    npix = dmo.shape[0]
    box = float(d["box_size"])
    n_halos = int(d["n_halos"])

    if n_halos == 0:
        return dmo.astype(np.float32)

    from .pipeline import extract_periodic_cutout
    gen = d["generated_patches"].astype(np.float64)        # (n, 3, P, P)
    centers = np.asarray(d["halo_centers"], dtype=np.float64)
    r200 = np.asarray(d["halo_r200"], dtype=np.float64)
    patch_pix = gen.shape[-1]
    ppm = npix / box

    # Build each mass-matched total patch from generated_patches + the DMO cutout,
    # so this is self-contained across runs (twobound/truth slabs store neither the
    # assembled ``composite`` nor ``patch_scales``/``condition_sums``). The mass match
    # s = Sigma_dmo_cutout / Sigma_hydro_patch reproduces build_bind_composite exactly.
    total = np.empty((n_halos, patch_pix, patch_pix), dtype=np.float64)
    for h in range(n_halos):
        cx = int(centers[h, 0] * ppm) % npix
        cy = int(centers[h, 1] * ppm) % npix
        dcut = extract_periodic_cutout(dmo, cx, cy, patch_pix)
        raw = gen[h].sum(0)                                 # hydro total (3 channels)
        H = raw * (dcut.sum() / (raw.sum() + 1e-30))        # mass-matched
        if mode == "bind":
            total[h] = H
        elif mode == "monopole":                            # isotropic correction ring
            total[h] = dcut + azimuthal_average_patch(H - dcut)
        else:                                               # bcm_warp: radial displacement
            total[h] = bcm_warp_patch(dcut, H)

    halos = [{"halo_center": centers[h], "r200": float(r200[h])}
             for h in range(n_halos)]
    weights_list = [
        circular_taper_weight(
            patch_pix, r_pix=float(r200[h]) * ppm * r200_factor,
            taper_frac=taper_frac,
        )
        for h in range(n_halos)
    ]
    square_taper = square_taper_weight(patch_pix, taper_frac=taper_frac)
    canvas, w_accum = paste_halos_2d(
        npix, box, halos, total[:, None], square_taper, weights_list=weights_list
    )
    alpha = np.clip(w_accum, 0.0, 1.0)
    field = (1.0 - alpha) * dmo + alpha * canvas[0]
    field *= dmo.sum() / (field.sum() + 1e-30)             # mass conservation
    return field.astype(np.float32)


def write_spherical_slab(
    src_slab: Path | str,
    out_slab: Path | str,
    *,
    dmo_slab: Path | str | None = None,
    r200_factor: float = 4.0,
    taper_frac: float = 0.15,
) -> dict:
    """Write a triple-total slab: BCM-warp, monopole, and BIND totals in one file.

    All single-channel total-matter maps are reconstructed from the per-halo
    ``generated_patches`` (the twobound/truth composites are lightweight — no
    assembled ``composite``/``dmo``), so the kappa decomposition needs only this
    one file via ``bind-lightcone-maps --mass_key <KEY>``:

    - ``composite``      — faithful BCM (radial displacement)  -> ``kappa_sph``
    - ``composite_mono`` — monopole-only correction (circular) -> ``kappa_mono``
    - ``composite_bind`` — full BIND total                     -> ``kappa_bind``

    ``kappa_dmo`` is shared across all runs (same DMO + seeds), taken once from the
    fid tree.  ``dmo_slab`` supplies the shared DMO background for runs whose
    composites do not store ``dmo``; when ``None`` the source slab's own ``dmo``.
    """
    src_slab = Path(src_slab)
    out_slab = Path(out_slab)
    out_slab.parent.mkdir(parents=True, exist_ok=True)
    d = np.load(src_slab)
    if "dmo" in d.files:
        dmo = None                                         # spherical_total_slab reads d["dmo"]
    elif dmo_slab is not None:
        dmo = np.load(dmo_slab)["dmo"]
    else:
        raise KeyError(f"{src_slab.name} has no 'dmo'; pass dmo_slab (shared fid tree)")
    kw = dict(r200_factor=r200_factor, taper_frac=taper_frac, dmo=dmo)
    sph = spherical_total_slab(d, mode="bcm_warp", **kw)
    mono = spherical_total_slab(d, mode="monopole", **kw)
    bind = spherical_total_slab(d, mode="bind", **kw)
    dmo_sum = float(d["dmo"].sum()) if "dmo" in d.files else float(dmo.sum())
    np.savez_compressed(
        out_slab,
        composite=sph[None],                   # BCM-warp total   (sum(0) -> kappa_sph)
        composite_mono=mono[None],             # monopole total   (sum(0) -> kappa_mono)
        composite_bind=bind[None],             # BIND total       (sum(0) -> kappa_bind)
        box_size=float(d["box_size"]),
        n_halos=int(d["n_halos"]),
        slab_idx=int(d["slab_idx"]) if "slab_idx" in d.files else -1,
        n_slabs=int(d["n_slabs"]) if "n_slabs" in d.files else -1,
    )
    # cheap sanity: each total's mass relative to DMO (should be ~0)
    return {"slab": out_slab.name, "n_halos": int(d["n_halos"]),
            "sph_rel_dmo": float(sph.sum() / (dmo_sum + 1e-30) - 1.0),
            "mono_rel_dmo": float(mono.sum() / (dmo_sum + 1e-30) - 1.0),
            "bind_rel_dmo": float(bind.sum() / (dmo_sum + 1e-30) - 1.0)}


def write_spherical_tree(
    src_snap_root: Path | str,
    out_root: Path | str,
    snapshot: int,
    *,
    dmo_root: Path | str | None = None,
    r200_factor: float = 4.0,
    taper_frac: float = 0.15,
    verbose: bool = True,
) -> list[dict]:
    """Write dual-total (spherical + BIND) slabs for one snapshot into ``out_root``.

    Mirrors the source ``snap_<NNN>/composite_slab*.npz`` layout so the standard
    ``bind-lightcone-maps`` runs on ``out_root`` with ``--manifest_root`` pointed
    at the shared stage-1 manifests.  ``dmo_root`` is the shared DMO tree (the fid
    ``bind_lightcone_tng`` lightcone) used when the source composites do not carry
    their own ``dmo`` (twobound/truth); the same-name slab in ``dmo_root`` is read.
    """
    src_snap_root = Path(src_snap_root)
    out_root = Path(out_root)
    dmo_root = Path(dmo_root) if dmo_root is not None else None
    snap_dir = src_snap_root / f"snap_{snapshot:03d}"
    out_dir = out_root / f"snap_{snapshot:03d}"
    slabs = sorted(snap_dir.glob("composite_slab*.npz"))
    if not slabs:
        raise FileNotFoundError(f"no composite slabs in {snap_dir}")
    diags = []
    for sl in slabs:
        dmo_slab = (dmo_root / f"snap_{snapshot:03d}" / sl.name) if dmo_root else None
        info = write_spherical_slab(
            sl, out_dir / sl.name, dmo_slab=dmo_slab, r200_factor=r200_factor,
            taper_frac=taper_frac,
        )
        diags.append(info)
        if verbose:
            print(f"[spherical] snap {snapshot:03d} {info['slab']}: "
                  f"n_halos={info['n_halos']} sph/dmo={info['sph_rel_dmo']:+.1e} "
                  f"mono/dmo={info['mono_rel_dmo']:+.1e} "
                  f"bind/dmo={info['bind_rel_dmo']:+.1e}")
    return diags
