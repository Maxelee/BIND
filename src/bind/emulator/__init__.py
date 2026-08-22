"""``bind.emulator`` — fast surrogate for ray-traced lightcone statistics.

Maps the 30 SB35 astro/feedback parameters (+ a source/lens redshift) to every
field-level statistic of the BIND→TNG lightcone — ``C_ell`` (κκ tomographic, κy,
yy, κτ, ττ, yτ), peak/minima counts, the convergence PDF + Minkowski functionals,
the wavelet scattering transform, dispersion-measure stats, the tSZ ``R(ν)`` at
peaks, and the Y–M / f_gas–M / T–M relations — instantly, with calibrated
uncertainty.

Workflow:
    bind-emulator-stats   # backfill WST + DM onto the run suite (GPU)
    bind-emulator-assemble --out emulator_dataset.npz
    >>> from bind.emulator import Emulator, EmulatorDataset
    >>> ds = EmulatorDataset.load("emulator_dataset.npz")
    >>> em = Emulator(backend="mlp").fit(ds); em.save("lightcone_emulator.pt")
    >>> em = Emulator.load("lightcone_emulator.pt")
    >>> stats = em.predict(bind.fiducial_params(), z_s=1.0)   # dict of arrays + 1sigma
"""

from __future__ import annotations

from bind.emulator.dataset import (  # noqa: F401
    ASTRO_PARAM_NAMES,
    SOURCE_REDSHIFTS,
    EmulatorDataset,
    Target,
    assemble,
    params_to_unit,
)

__all__ = [
    "EmulatorDataset",
    "Target",
    "assemble",
    "params_to_unit",
    "ASTRO_PARAM_NAMES",
    "SOURCE_REDSHIFTS",
    "Emulator",
]


def __getattr__(name: str):
    # Lazy so the (torch-heavy) engine is only imported when actually used.
    if name == "Emulator":
        from bind.emulator.core import Emulator
        return Emulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
