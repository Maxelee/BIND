"""Per-theta CylToSph: the gate call's mandatory component #2, measured.

The CAMELS L50n512/1P suite (same 30-astro-parameter +-bound design as the
painted twobound bundle) gives d ln CylToSph / d u per parameter and gate
mass bin at z=0 (``camels1p_cyltosph_analysis.py`` ->
``cyltosph_theta_model.npz``). The trend is applied *multiplicatively* on
the wp2 absolute calibration (run 2451362): box-depth and selection
differences between the L50 measurement and the painted slab cancel at
first order in the fiducial-normalized ratio.

    CylToSph(theta, bin) = wp2_factor(snap) * exp( s(bin) . (u - u_fid) )

Headline numbers (group bin 13.0-13.4, z=0): WindEnergyIn1e51erg +0.43,
IMFslope -0.31, QuasarThresholdPower -0.28 — the wind sector dominates with
the SAME positive sign as f_gas itself, so weak-wind (data-preferred)
models have a lower spherical conversion and the model envelope extends
further toward the eRASS1 group points than the fixed-factor gate showed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.paper3a.gate.aggregate import cyltosph_for_snap

from . import params_meta as pm

MODEL_NPZ = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator/cyltosph_theta_model.npz")


class CylToSphTheta:
    """theta-dependent CylToSph factors per gate mass bin."""

    def __init__(self, path: Path = MODEL_NPZ):
        with np.load(path, allow_pickle=False) as f:
            names = [str(n) for n in f["param_names"]]
            if names != pm.ASTRO_NAMES:
                raise ValueError("cyltosph model parameter order != SB35 astro order")
            self.slopes = np.asarray(f["slopes_dlnC2S_du"], float)   # (30, n_bins)
            self.bins = [str(b) for b in f["bins"]]
        self.n_bins = self.slopes.shape[1]
        self.u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)

    def factors(self, params_unit, snap: str) -> np.ndarray:
        """CylToSph per measured gate bin at theta: ``(..., n_bins)``.

        ``params_unit``: unit-cube astro vector(s) ``(30,)`` or ``(N, 30)``.
        The wp2 snap factor anchors the absolute scale; bins beyond the
        measured three inherit the last measured bin's slope (cluster-scale
        dependence is weaker, see the summary json).
        """
        u = np.atleast_2d(np.asarray(params_unit, float))
        base, _ = cyltosph_for_snap(snap)
        trend = np.exp((u - self.u_fid) @ self.slopes)               # (N, n_bins)
        out = base * trend
        return out[0] if np.ndim(params_unit) == 1 else out

    def factors_full_bins(self, params_unit, snap: str, n_total: int = 5) -> np.ndarray:
        """Like :meth:`factors` but padded to the gate's ``n_total`` mass
        bins by repeating the highest measured bin's factor."""
        f = np.atleast_2d(self.factors(params_unit, snap))
        pad = np.repeat(f[:, -1:], n_total - f.shape[1], axis=1)
        out = np.concatenate([f, pad], axis=1)
        return out[0] if np.ndim(params_unit) == 1 else out
