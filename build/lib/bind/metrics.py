"""Evaluation metrics: power spectrum, radial profiles, mass comparison, distributions."""

import numpy as np
import Pk_library as PKL


CHANNEL_NAMES = ['DM_hydro', 'Gas', 'Stars']


def power_spectrum_2d(field, box_size=6.25, MAS='None', threads=1):
    """Compute azimuthally averaged 2D power spectrum via Pylians Pk_library.

    Args:
        field: (H, W) 2D array (in original or normalized units)
        box_size: physical box size in Mpc/h
        MAS: mass-assignment scheme used to paint the field ('None', 'NGP',
             'CIC', 'TSC', 'PCS')
        threads: number of OpenMP threads for Pylians
    Returns:
        k: wavenumber bin centres in h/Mpc (1D)
        pk: power spectrum in (Mpc/h)^2 (1D)
    """
    delta = np.asarray(field, dtype=np.float32) / np.mean(field)
    Pk2D = PKL.Pk_plane(delta, box_size, 'CIC', threads)
    return Pk2D.k, Pk2D.Pk


def radial_profile(field, n_bins=15, logspace=True):
    """Compute azimuthally averaged radial profile centered on the image.

    Args:
        field: (H, W) 2D array
        n_bins: number of radial bins
        logspace: if True, use logarithmically spaced bins (r_min=1 px = 50 kpc)
    Returns:
        r_centers: radial bin centers in pixels
        profile: mean value per bin
    """
    H, W = field.shape
    y, x = np.mgrid[:H, :W] - np.array([H / 2, W / 2])[:, None, None]
    r = np.sqrt(x**2 + y**2)

    r_max = min(H, W) / 2
    if logspace:
        bins = np.logspace(np.log10(1.0), np.log10(r_max), n_bins + 1)
    else:
        bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.sum() > 0:
            profile[i] = field[mask].mean()
    return r_centers, profile


def compute_mass(fields_norm, mean, std):
    """Compute total mass from normalized log-space fields.

    Args:
        fields_norm: (N, H, W) normalized fields
        mean, std: normalization stats for this channel
    Returns:
        masses: (N,) total mass per sample (in original units)
    """
    log_fields = fields_norm * std + mean  # undo standardization
    original = 10.0 ** log_fields - 1.0    # undo log10(1+x)
    return original.sum(axis=(1, 2))


def batch_power_spectra(fields, box_size=6.25):
    """Compute power spectra for a batch of fields.

    Args:
        fields: (N, H, W)
    Returns:
        k: wavenumber centers
        pk_mean: mean power spectrum
        pk_std: std of power spectrum
    """
    pks = []
    for i in range(len(fields)):
        k, pk = power_spectrum_2d(fields[i], box_size)
        pks.append(pk)
    pks = np.stack(pks)
    return k, pks.mean(0), pks.std(0)


def batch_profiles(fields, n_bins=15, logspace=True):
    """Compute radial profiles for a batch of fields.

    Args:
        fields: (N, H, W)
        n_bins: number of radial bins
        logspace: if True, use logarithmically spaced bins
    Returns:
        r: radial bin centers
        prof_mean: mean profile
        prof_std: std of profile
    """
    profs = []
    for i in range(len(fields)):
        r, prof = radial_profile(fields[i], n_bins, logspace=logspace)
        profs.append(prof)
    profs = np.stack(profs)
    return r, profs.mean(0), profs.std(0)


def power_spectrum_pylians_2d(field_2d, box_size=50.0, MAS='None', threads=4,
                              as_overdensity=True):
    """Azimuthally averaged 2D P(k) via Pylians Pk_plane.

    Args:
        field_2d: (H, W) 2D map. If as_overdensity=True the field is
            converted to delta = field / mean - 1 before being passed in.
        box_size: physical box size in Mpc/h
        MAS: mass-assignment-scheme deconvolution ('None', 'NGP', 'CIC',
            'TSC', 'PCS'). Use 'None' for maps that are already gridded
            via summing particles (no kernel).
        threads: OpenMP threads
        as_overdensity: convert to overdensity before spectrum (standard).
    Returns:
        k: wavenumber bins (Mpc/h)^-1
        pk: power per bin
        nmodes: number of modes per bin
    """
    import Pk_library as PKL
    f = np.asarray(field_2d, dtype=np.float32)
    if as_overdensity:
        mu = f.mean()
        if mu <= 0:
            raise ValueError('field mean must be positive to compute overdensity')
        f = f / mu - 1.0
    p = PKL.Pk_plane(f, box_size, MAS=MAS, threads=threads, verbose=False)
    return np.asarray(p.k), np.asarray(p.Pk), np.asarray(p.Nmodes)


def batch_power_spectra_pylians(fields, box_size=50.0, MAS='None', threads=4,
                                as_overdensity=True):
    """Apply power_spectrum_pylians_2d across a batch; returns k, mean, std."""
    pks = []
    k_ref = None
    for f in fields:
        k, pk, _ = power_spectrum_pylians_2d(
            f, box_size=box_size, MAS=MAS, threads=threads,
            as_overdensity=as_overdensity,
        )
        if k_ref is None:
            k_ref = k
        pks.append(pk)
    pks = np.stack(pks)
    return k_ref, pks.mean(0), pks.std(0)


def pixel_ks_test(truth, generated, n_samples=1_000_000, positive_only=True,
                  seed=0):
    """KS test on pooled pixel values from truth and generated fields.

    Args:
        truth, generated: arrays of arbitrary shape (flattened internally).
        n_samples: cap on the number of pixels per side sent to ks_2samp
            (scipy is O(n log n); > 1e6 is slow for little statistical gain).
        positive_only: if True, keep only strictly positive pixels (density
            fields have a physical floor at 0).
        seed: RNG seed for the sub-sampling.
    Returns:
        ks_stat, p_value
    """
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    t = truth.ravel()
    g = generated.ravel()
    if positive_only:
        t = t[t > 0]
        g = g[g > 0]
    if len(t) > n_samples:
        t = rng.choice(t, size=n_samples, replace=False)
    if len(g) > n_samples:
        g = rng.choice(g, size=n_samples, replace=False)
    res = ks_2samp(t, g)
    return float(res.statistic), float(res.pvalue)
