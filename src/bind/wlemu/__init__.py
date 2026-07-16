"""``bind.wlemu`` — field-level weak-lensing emulator.

A conditional **flow-matching** generator that maps the 30 SB35 astro/feedback
parameters (+ a source redshift ``z_s``) **directly to a convergence map**
``θ → κ`` — the field-level analogue of the summary-statistic ``bind.emulator``.
Trained on the 256-point SB35 Sobol suite of ray-traced lightcones
(``bind_sb35/runs/run_NNNN/kappa_maps.npz``: 50 realisations × 5 source planes ×
1024² each), it draws fresh κ maps for any parameter point in milliseconds and
reproduces the full statistic distribution (power spectrum, PDF, peaks) *with*
cosmic variance, not just the mean.

Why flow matching (vs. latent-CFM / score / wavelet-flow): BIND itself is a
param-conditioned flow-matching UNet, so the conditioning machinery
(``AdaGroupNorm`` on a param embedding + sinusoidal time + redshift embedding)
is already validated here; sampling is fast (~20–50 NFE) which matters because
the emulator's value is drawing *many* maps; and we avoid the compression loss
of a latent bottleneck — which would wash out exactly the small-scale baryon
response we care about.

Workflow::

    bind-wlemu-cache  --resolution 256 --out <cache_dir>      # one-time, CPU/big-mem
    bind-wlemu-train  --cache <cache_dir> --run_name fm_kappa  # GPU (SLURM)
    >>> from bind.wlemu import WLEmulator
    >>> em = WLEmulator.load("runs/fm_kappa/best.pt")
    >>> kappa = em.generate(bind.fiducial_params(), z_s=1.0, n=64)  # (64,256,256)

Engine: ``data.py`` (cache + arcsinh-per-z normalisation), ``model.py``
(``FieldUNet`` + ``FieldCFM``), ``train.py`` (DDP loop), ``sample.py``
(``WLEmulator``).  Demo/validation notebook: ``examples/wl_field_emulator.ipynb``.
"""

from __future__ import annotations

from bind.wlemu.data import (  # noqa: F401
    ASTRO_PARAM_NAMES,
    SOURCE_REDSHIFTS,
    KappaCache,
    KappaCacheDataset,
    KappaNorm,
    build_cache,
)
from bind.wlemu.model import FieldCFM, FieldUNet  # noqa: F401
from bind.wlemu.sample import WLEmulator  # noqa: F401

__all__ = [
    "ASTRO_PARAM_NAMES",
    "SOURCE_REDSHIFTS",
    "KappaCache",
    "KappaCacheDataset",
    "KappaNorm",
    "build_cache",
    "FieldUNet",
    "FieldCFM",
    "WLEmulator",
]
