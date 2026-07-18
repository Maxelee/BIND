"""WP-A5 inference: posterior over the 30 SB35 astro parameters.

First (and deciding) data axis: eRASS1 f_gas(M500) at z ~ 0 with the
mandatory-component treatment from the gate call — per-cluster mass-PDF
convolution (:mod:`fgas`) and the per-theta CylToSph factors — over the v4
emulator with the Sigma_theory floor. kSZ joins the data vector only after
the mock-sample selection model (miscentering/satellites/mass calibration);
the likelihood is block-structured so probe subsets (plan task 5) are the
same code path with blocks toggled.

Method: affine-invariant MCMC (emcee; available in torch3) — the emulator
is ~ms per batched evaluation and the likelihood Gaussian, so MCMC is the
plan's preferred first route. Recovery tests gate any data fit.
"""

from .fgas import FgasBlock  # noqa: F401
from .sampler import run_mcmc  # noqa: F401
