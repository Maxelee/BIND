"""Evaluation metrics: power spectrum, radial profiles, mass comparison, distributions."""

import numpy as np


CHANNEL_NAMES = ['DM_hydro', 'Gas', 'Stars']


def power_spectrum_2d(field, box_size=6.25):
    """Compute azimuthally averaged 2D power spectrum via FFT.

    Args:
        field: (H, W) 2D array (in original or normalized units)
        box_size: physical box size in Mpc/h
    Returns:
        k: wavenumber bins (1D)
        pk: power per bin (1D)
    """
    N = field.shape[0]
    fft = np.fft.fft2(field)
    pk2d = np.abs(fft) ** 2 * (box_size / N) ** 2

    kfreq = np.fft.fftfreq(N, d=box_size / N) * 2 * np.pi
    kx, ky = np.meshgrid(kfreq, kfreq, indexing='ij')
    kmag = np.sqrt(kx**2 + ky**2)

    # Azimuthal average
    k_bins = np.linspace(0, kfreq.max() * np.sqrt(2), N // 2)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    pk = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask = (kmag >= k_bins[i]) & (kmag < k_bins[i + 1])
        if mask.sum() > 0:
            pk[i] = pk2d[mask].mean()
    return k_centers, pk


def radial_profile(field, n_bins=32):
    """Compute azimuthally averaged radial profile centered on the image.

    Args:
        field: (H, W) 2D array
        n_bins: number of radial bins
    Returns:
        r_centers: radial bin centers in pixels
        profile: mean value per bin
    """
    H, W = field.shape
    y, x = np.mgrid[:H, :W] - np.array([H / 2, W / 2])[:, None, None]
    r = np.sqrt(x**2 + y**2)

    r_max = min(H, W) / 2
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


def batch_profiles(fields, n_bins=32):
    """Compute radial profiles for a batch of fields.

    Args:
        fields: (N, H, W)
    Returns:
        r: radial bin centers
        prof_mean: mean profile
        prof_std: std of profile
    """
    profs = []
    for i in range(len(fields)):
        r, prof = radial_profile(fields[i], n_bins)
        profs.append(prof)
    profs = np.stack(profs)
    return r, profs.mean(0), profs.std(0)
