"""WP-A4 halo-observable emulator: theta -> painted gas observables.

Layout mirrors ``bind.wlemu`` (the WL-statistics emulator on the
``feature/wl-emu`` branch), whose per-z PCA + exact-ARD-Matern-5/2-GP recipe
this package ports to the A3 gate observables. Self-contained on the
``analysis/paper3a-gas-calibration`` branch: no import of ``bind`` (whose
pandas dependency is broken in the torch3 venv) — SB35 parameter metadata is
read directly from the CSV asset by :mod:`params_meta`.

Modules
-------
- ``params_meta``: SB35 parameter table + unit-cube transforms (numpy-only).
- ``dataset``: 1,902 v2 gate tables -> one training npz (X, Y, bootstrap SEM).
- ``gasemu``: numpy-only ``GasEmulator`` inference class (the A5 dependency).
- ``fit``: gpytorch fitting, k-fold validation, artifact export.
- ``velocity``: linear-theory sigma_v(z) and LOS velocity correlation r_v(chi)
  for the kSZ decorrelation forward model.
- ``forward``: emulator outputs -> data-space predictions (CylToSph, kSZ
  normalization + decorrelation); the WP-A5 likelihood calls this.
"""

from .gasemu import GasEmulator  # noqa: F401
