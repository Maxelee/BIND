"""Validation tooling for the BIND lightcone projection / convergence pipeline.

Two checks, in increasing order of what they need:

1. :func:`compare_projection` — *projected mass-map* level (no raytracing).
   Sums the per-slab BIND ``composite_slab*.npz`` into a full-box total-matter
   map and compares its 2-D power spectrum against a truth hydro projection and
   the lightcone's own DMO.  It isolates three things:

     * ``BIND / DMO_lc``   — the baryonic suppression BIND produces (the science
       signal; computed against the *same-pipeline* DMO so the projection MAS
       artifact cancels).
     * ``hydro / DMO_truth`` — the true suppression for reference.
     * ``DMO_lc / DMO_truth`` — the pure projection/aliasing mismatch between
       pipelines.  Where this departs from 1 sets the trustworthy small-scale
       limit (``k_safe`` / ``ell_safe``).

   Key finding this was written to track (snap096, TNG300): BIND reproduces the
   true suppression to a few %, and the high-k "upturn" seen in convergence
   power is a raw-CIC aliasing artifact common to DMO+BIND — *not* the
   generative model.  Re-running stage 1 with ``--mas_correct`` (interlacing +
   CIC deconvolution, stored as ``dmo_aa``) removes it.

2. :func:`compare_kappa_to_kappatng` — *convergence-map* level.  Given a BIND
   kappa map and a kappaTNG(-dark) reference map (same field size / pixel scale),
   computes the angular power ratio and the ell above which they diverge.  This
   is the "first thing to check" when validating fiducial BIND against kappaTNG.

Everything here is read-only analysis; nothing writes into the pipeline outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Power spectra
# ---------------------------------------------------------------------------

def power_spectrum_2d(
    field: np.ndarray,
    box_size: float,
    *,
    n_bins: int = 50,
    k_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic 2-D power spectrum of ``delta = field/mean - 1``.

    Parameters
    ----------
    field : (N, N) array — a projected map (mass or kappa); mean-normalised
        internally so absolute units don't matter.
    box_size : float — transverse side length (Mpc/h for mass maps, or the field
        angular size for kappa maps; units just set the k axis).
    n_bins, k_max : binning controls.  ``k_max`` defaults to the grid Nyquist.

    Returns
    -------
    (k, P) with k in 2*pi/box_size units.
    """
    N = field.shape[0]
    d = field.astype(np.float64)
    d = d / d.mean() - 1.0
    fk = np.fft.fft2(d)
    pk = (fk * np.conj(fk)).real / N**4
    kf = 2.0 * np.pi / box_size
    fr = np.fft.fftfreq(N, d=1.0 / N)
    kk = np.hypot(*np.meshgrid(kf * fr, kf * fr, indexing="ij"))
    k_max = k_max or np.pi * N / box_size
    bins = np.logspace(np.log10(kf), np.log10(k_max), n_bins)
    idx = np.digitize(kk.ravel(), bins)
    pkr, kkr = pk.ravel(), kk.ravel()
    kc = np.zeros(n_bins - 1)
    pc = np.zeros(n_bins - 1)
    for i in range(1, n_bins):
        m = idx == i
        if m.any():
            kc[i - 1] = kkr[m].mean()
            pc[i - 1] = pkr[m].mean()
    ok = kc > 0
    return kc[ok], pc[ok]


def _center_crop(field: np.ndarray, n: int) -> np.ndarray:
    off = (field.shape[0] - n) // 2
    return field[off:off + n, off:off + n]


# ---------------------------------------------------------------------------
# Stage-2 composite -> full-box maps
# ---------------------------------------------------------------------------

def load_composite_fullbox(
    composite_dir: str | Path,
    *,
    n_slabs: int | None = None,
) -> dict:
    """Sum per-slab ``composite_slab*.npz`` into full-box BIND + DMO maps.

    Returns dict with ``bind`` (DM_hydro+Gas+Stars summed over slabs), ``dmo``
    (the composite background — anti-aliased if stage 1 used ``--mas_correct``),
    ``box_size`` and ``npix``.
    """
    composite_dir = Path(composite_dir)
    files = sorted(composite_dir.glob("composite_slab*.npz"))
    if n_slabs is not None:
        files = files[:n_slabs]
    if not files:
        raise FileNotFoundError(f"no composite_slab*.npz in {composite_dir}")
    bind = dmo = None
    box_size = None
    for f in files:
        d = np.load(f)
        comp = d["composite"].sum(axis=0).astype(np.float64)   # (N, N) total mass
        dmoi = d["dmo"].astype(np.float64)
        bind = comp if bind is None else bind + comp
        dmo = dmoi if dmo is None else dmo + dmoi
        box_size = float(d["box_size"])
    return {"bind": bind, "dmo": dmo, "box_size": box_size, "npix": bind.shape[0]}


def load_truth_projection(npz_path: str | Path) -> tuple[np.ndarray, float]:
    """Load a ``{dmo,hydro,...}.npz`` projected field (``field`` + ``box_size``)."""
    d = np.load(npz_path)
    return d["field"].astype(np.float64), float(d["box_size"])


# ---------------------------------------------------------------------------
# Check 1: projected mass-map comparison
# ---------------------------------------------------------------------------

def compare_projection(
    composite_dir: str | Path,
    truth_hydro_npz: str | Path,
    truth_dmo_npz: str | Path,
    *,
    artifact_tol: float = 0.10,
    k_floor: float = 0.5,
    ax=None,
):
    """Common-grid 2-D power comparison of BIND vs truth (see module docstring).

    Crops all maps to a common grid (the smallest of the four) so they share a
    Nyquist frequency — essential, since comparing maps of different npix in
    physical k is what makes the raw ratio look like a runaway high-k upturn.

    Returns a dict of ``(k, ratio)`` curves plus ``k_safe`` (the k below which
    the DMO_lc/DMO_truth pipeline mismatch stays within ``artifact_tol``).
    """
    fb = load_composite_fullbox(composite_dir)
    hydro, Lh = load_truth_projection(truth_hydro_npz)
    dmop, Ld = load_truth_projection(truth_dmo_npz)

    n = min(fb["npix"], hydro.shape[0], dmop.shape[0])
    bind = _center_crop(fb["bind"], n)
    dmolc = _center_crop(fb["dmo"], n)
    hydro = _center_crop(hydro, n)
    dmop = _center_crop(dmop, n)
    Lb = fb["box_size"] * n / fb["npix"]
    Lh = Lh * n / hydro.shape[0] if hydro.shape[0] != n else Lh

    kB, pB = power_spectrum_2d(bind, Lb)
    kL, pL = power_spectrum_2d(dmolc, Lb)
    kh, ph = power_spectrum_2d(hydro, Ld)
    kd, pd = power_spectrum_2d(dmop, Ld)

    # Interpolate onto a common k grid for clean ratios.
    kt = np.logspace(np.log10(max(kB[0], kd[0])), np.log10(min(kB[-1], kd[-1])), 60)
    def I(k, p):
        return np.interp(kt, k, p)
    bind_dmo = I(kB, pB) / I(kL, pL)        # BIND suppression (science)
    hyd_dmo = I(kh, ph) / I(kd, pd)         # truth suppression
    artifact = I(kL, pL) / I(kd, pd)        # pipeline projection mismatch

    # k_safe: below this the DMO_lc/DMO_truth mismatch stays within tol.  DMO_lc
    # and DMO_truth are *different LOS realizations*, so a few-% scatter at all k
    # is sample variance, not the aliasing artifact.  Median-smooth to suppress
    # that scatter, ignore k < k_floor (cosmic-variance-dominated), and require a
    # *sustained* departure (the artifact rise is monotonic once it dominates).
    sm = np.array([np.median(artifact[max(0, i - 1):i + 2]) for i in range(len(artifact))])
    bad = (np.abs(sm - 1.0) > artifact_tol) & (kt >= k_floor)
    # first k beyond which it is bad and stays bad
    k_safe = float(kt[-1])
    for i in range(len(kt)):
        if bad[i] and bad[i:].mean() > 0.5:
            k_safe = float(kt[i])
            break

    if ax is not None:
        ax.semilogx(kt, bind_dmo, c="firebrick", lw=2, label="BIND/DMO (same pipeline)")
        ax.semilogx(kt, hyd_dmo, c="steelblue", lw=2, label="hydro/DMO (truth)")
        ax.semilogx(kt, artifact, c="k", lw=2, ls="--", label="DMO_lc/DMO_truth (artifact)")
        ax.axvline(k_safe, c="gray", ls=":", label=f"k_safe={k_safe:.1f}")
        ax.axhline(1, c="gray", lw=0.6)
        ax.set_ylim(0.6, 2.3)
        ax.set_xlabel("k [h/Mpc]")
        ax.set_ylabel("ratio")
        ax.legend(fontsize=8)

    return {
        "k": kt,
        "bind_over_dmo": bind_dmo,
        "hydro_over_dmo": hyd_dmo,
        "artifact": artifact,
        "k_safe": k_safe,
    }


# ---------------------------------------------------------------------------
# Check 2: convergence-map comparison (the kappaTNG validation)
# ---------------------------------------------------------------------------

def angular_power_spectrum(
    kappa: np.ndarray,
    field_size_rad: float,
    *,
    n_bins: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Angular power C_ell of a flat-sky convergence map.

    ``field_size_rad`` is the angular side length of the map in radians.
    """
    k, P = power_spectrum_2d(kappa, field_size_rad, n_bins=n_bins)
    return k, P   # k here already equals multipole ell since box is in radians


def compare_kappa_to_kappatng(
    bind_kappa: np.ndarray,
    ref_kappa: np.ndarray,
    field_size_rad: float,
    *,
    tol: float = 0.10,
    ax=None,
):
    """Compare a BIND convergence map to a kappaTNG(-dark) reference.

    Both maps must share field size and pixel scale.  Returns ``(ell, ratio)``
    and ``ell_safe`` (where the ratio first leaves ``1 +/- tol``) — the first
    thing to verify when validating fiducial BIND against kappaTNG.
    """
    lb, cb = angular_power_spectrum(bind_kappa, field_size_rad)
    lr, cr = angular_power_spectrum(ref_kappa, field_size_rad)
    ell = np.logspace(np.log10(max(lb[0], lr[0])), np.log10(min(lb[-1], lr[-1])), 50)
    ratio = np.interp(ell, lb, cb) / np.interp(ell, lr, cr)
    bad = np.abs(ratio - 1.0) > tol
    ell_safe = float(ell[bad][0]) if bad.any() else float(ell[-1])
    if ax is not None:
        ax.semilogx(ell, ratio, c="firebrick", lw=2, label="BIND / kappaTNG")
        ax.axhline(1, c="gray", lw=0.6)
        ax.axvline(ell_safe, c="gray", ls=":", label=f"ell_safe={ell_safe:.0f}")
        ax.set_xlabel(r"multipole $\ell$")
        ax.set_ylabel(r"$C_\ell^{\rm BIND}/C_\ell^{\rm kappaTNG}$")
        ax.legend(fontsize=8)
    return {"ell": ell, "ratio": ratio, "ell_safe": ell_safe}


if __name__ == "__main__":
    # Smoke run against the snap096 outputs used during development.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LC = "/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096"
    PR = "/mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG/snap096/projected"
    fig, ax = plt.subplots(figsize=(7, 5))
    res = compare_projection(LC, f"{PR}/hydro.npz", f"{PR}/dmo.npz", ax=ax)
    fig.tight_layout()
    fig.savefig("/tmp/lightcone_projection_check.png", dpi=110)
    print(f"k_safe (DMO_lc vs DMO_truth within 10%) = {res['k_safe']:.2f} h/Mpc")
    print("saved /tmp/lightcone_projection_check.png")
