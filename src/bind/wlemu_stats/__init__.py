"""BIND weak-lensing statistics emulator (``bind.wlemu_stats``).

Renamed from ``bind.wlemu`` in v0.3.0: that name now belongs to the
field-level flow-matching kappa-map generator; this package is the v0.2.0
GP summary-statistics emulator, unchanged apart from the new import path.

Instant, self-consistent weak-lensing convergence summary statistics
(power spectrum, PDF, peaks, minima, Minkowski functionals, scattering
coefficients, moments) as a function of the 30 CAMELS SB35 astrophysical
parameters, trained on statistics measured from convergence maps raytraced
through BIND-baryonified IllustrisTNG-DMO lightcones.

Quick start::

    from bind.wlemu import WLEmulator
    emu = WLEmulator.load()
    pred = emu.predict({"WindEnergyIn1e51erg": 7.2}, z_source=1.0)

See ``examples/wlemu_tutorial.ipynb`` for the full guide and the validation
against held-out raytraced truth, and ``docs/wl_emulator.md`` for the design.
"""

from .emulator import BLOCKS, LOG_BLOCKS, WLEmulator
from .stats import measure_stats, stats_vector

__all__ = ["WLEmulator", "measure_stats", "stats_vector", "BLOCKS", "LOG_BLOCKS"]
