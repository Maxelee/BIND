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

MASS DEPENDENCE (``mass_dependent=True``) — VARIANT ONLY, NOT THE DEFAULT
------------------------------------------------------------------------
systematics-hunt FINDINGS_fgas.md candidate 1: the absolute factor above
(``cyltosph_for_snap`` -> 0.5274944874262141 at snap 096) is a scalar
applied to all five gate mass bins, while the sphere/cylinder ratio is a
strong function of mass — and the measurement is sitting unused in the very
npz this module loads (``cyltosph_theta_model.npz::fiducial_l50_median =
[0.48852243 0.61397801 0.71489969]``). Only the *theta* response is per-bin;
the *mass* response was measured, persisted, then dropped.

Pooling all 175 CAMELS-L50 1P sims (5,280 halos) gives a saturating form,

    C2S(M) = 1 / (1 + R0 * 10 ** (-0.541 * (logM500 - 13.2))),
    R0 = 1/0.4828 - 1,

i.e. [0.483, 0.606, 0.717, 0.806, 0.886] at the gate-bin centres
[13.2, 13.6, 14.0, 14.4, 14.9] — against a flat 0.5275 that is
[x0.92, x1.15, x1.36, x1.53, x1.68].

*** WHY THIS IS A VARIANT AND MUST NOT BECOME THE DEFAULT (yet). ***
1. The shape is TRANSPORTED. It was fitted on CAMELS-L50n512 at snap 090
   and is being carried onto TNG300 at snap 096 — a different box, a
   different resolution, a different halo selection. The decisive
   like-for-like measurement (per-bin C2S on TNG300-hydro truth;
   ``_calibrate_cyltosph`` already computes the per-halo sph/cyl ratios and
   needs them retained and binned) is a POPEYE job that HAS NOT RUN.
   Pre-registered there: CONFIRMED if per-bin C2S is monotonic in mass and
   the 13.0-13.4 -> 13.8-14.2 ratio > 1.25 (L50 gives 1.49); REFUTED if
   < 1.10.
2. Bins 4 and 5 (14.2-14.6, 14.6-15.2) are PURE EXTRAPOLATION. The L50 box
   is 50 h^-1 Mpc and contains no such halos at all (n_by_bin =
   [20, 6, 3, 0, 0]); the x1.53 and x1.68 are a fitted model, not a
   measurement.
Changing a likelihood DEFAULT on a transported, partly-extrapolated shape
would invert this project's measure-then-model discipline. It is wired as
a WP-A8 variant (``c2s_massdep``) so it can be measured before it is
believed.

NORMALIZATION CHOICE. Only the SHAPE is imported; the wp2 absolute anchor
is preserved. The per-bin factors are multiplied by C2S(M_b) divided by the
gate-bin-count-weighted mean of C2S over the snapshot's own frozen halo
catalogue, so a flat re-derivation returns exactly ``cyltosph_for_snap``.
This is deliberate: 0.5275 is a DIRECT TNG300 measurement and is the one
number here that was not transported. The audit's own cross-check says the
two agree anyway — count-weighting C2S over the TNG300 snap-096 gate counts
[1377, 507, 144, 30, 5] gives 0.5350 vs the measured 0.5275, 1.4% — so
running unanchored instead would move every bin by that same 1.4%, far
inside the pre-registered Test C thresholds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.paper3a.gate.aggregate import cyltosph_for_snap

from . import params_meta as pm

MODEL_NPZ = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator/cyltosph_theta_model.npz")
GATE_TABLES = Path("/mnt/ceph/users/mlee1/paper3/A/wp3_gate/operator_tables_v3")

# The pooled CAMELS-L50 1P saturating fit (5,280 halos). See the module
# docstring: VARIANT ONLY, transported, bins 4-5 extrapolated.
C2S_MASSDEP_PIVOT_LOGM500 = 13.2      # first gate-bin centre, log10 M500 Msun/h
C2S_MASSDEP_PIVOT_VALUE = 0.4828      # C2S at the pivot
C2S_MASSDEP_SLOPE = 0.541             # -dlog10(contamination)/dlogM500
# gate bin edges of the fiducial operator tables (log10 M500 Msun/h)
GATE_LOGM500_EDGES = (13.0, 13.4, 13.8, 14.2, 14.6, 15.2)


def c2s_massdep(logm500_msunh) -> np.ndarray:
    """Mass-dependent sphere/cylinder gas-mass ratio, pooled CAMELS-L50 1P.

    ``C2S(M) = 1 / (1 + R0 * 10 ** (-slope * (logM500 - pivot)))`` with
    ``R0`` fixed so ``C2S(pivot) = C2S_MASSDEP_PIVOT_VALUE``. Saturating by
    construction (-> 1 as M -> inf), which is the right asymptote: the LOS
    contamination of the cylinder scales as M^0.844 while the sphere scales
    as M^1.385, so the contaminated fraction falls as M^-0.54.
    """
    r0 = 1.0 / C2S_MASSDEP_PIVOT_VALUE - 1.0
    x = np.asarray(logm500_msunh, float) - C2S_MASSDEP_PIVOT_LOGM500
    return 1.0 / (1.0 + r0 * 10.0 ** (-C2S_MASSDEP_SLOPE * x))


class CylToSphTheta:
    """theta-dependent CylToSph factors per gate mass bin.

    ``mass_dependent=False`` (DEFAULT) reproduces the frozen behaviour
    exactly: one wp2 scalar per snapshot times the per-bin theta trend.
    ``mass_dependent=True`` is the WP-A8 ``c2s_massdep`` variant — read the
    module docstring before using it, and do not make it the default until
    the Popeye per-bin measurement has run.
    """

    def __init__(self, path: Path = MODEL_NPZ, mass_dependent: bool = False):
        with np.load(path, allow_pickle=False) as f:
            names = [str(n) for n in f["param_names"]]
            if names != pm.ASTRO_NAMES:
                raise ValueError("cyltosph model parameter order != SB35 astro order")
            self.slopes = np.asarray(f["slopes_dlnC2S_du"], float)   # (30, n_bins)
            self.bins = [str(b) for b in f["bins"]]
        self.n_bins = self.slopes.shape[1]
        self.u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
        self.mass_dependent = bool(mass_dependent)
        self._mass_shape_cache: dict[tuple[str, int], np.ndarray] = {}

    # -------------------------------------------------------- mass shape --

    def _mass_shape(self, snap: str, n_total: int) -> np.ndarray:
        """``(n_total,)`` anchor-preserving mass shape, all ones when
        ``mass_dependent`` is False.

        Normalized so that the gate-bin-count-weighted mean over the
        snapshot's own frozen halo catalogue is exactly 1 — i.e. the wp2
        absolute factor is untouched and only the L50 *shape* is imported.
        """
        if not self.mass_dependent:
            return np.ones(n_total)
        key = (snap, n_total)
        if key not in self._mass_shape_cache:
            edges = np.asarray(GATE_LOGM500_EDGES, float)
            if n_total != len(edges) - 1:
                raise ValueError(f"gate has {len(edges) - 1} mass bins, "
                                 f"asked for {n_total}")
            centers = 0.5 * (edges[1:] + edges[:-1])
            with np.load(GATE_TABLES / f"fiducial_run_0000_snap{snap}.npz",
                         allow_pickle=False) as f:
                logm500 = np.log10(np.asarray(f["m500_msunh"], float))
            counts = np.array([((logm500 >= edges[b]) & (logm500 < edges[b + 1])).sum()
                               for b in range(n_total)], float)
            if counts.sum() <= 0:
                raise ValueError(f"snap {snap}: no halos in the gate bins")
            c2s = c2s_massdep(centers)
            self._mass_shape_cache[key] = c2s / np.average(c2s, weights=counts)
        return self._mass_shape_cache[key]

    # ----------------------------------------------------------- factors --

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
        if self.mass_dependent:
            n_total = len(GATE_LOGM500_EDGES) - 1
            out = out * self._mass_shape(snap, n_total)[None, :self.n_bins]
        return out[0] if np.ndim(params_unit) == 1 else out

    def factors_full_bins(self, params_unit, snap: str, n_total: int = 5) -> np.ndarray:
        """Like :meth:`factors` but padded to the gate's ``n_total`` mass
        bins by repeating the highest measured bin's factor.

        With ``mass_dependent=True`` only the *theta trend* is padded that
        way; the mass shape is evaluated at every bin centre, including the
        two the L50 box could not measure (see the module docstring —
        those two are extrapolation).
        """
        u = np.atleast_2d(np.asarray(params_unit, float))
        base, _ = cyltosph_for_snap(snap)
        trend = np.exp((u - self.u_fid) @ self.slopes)               # (N, n_bins)
        pad = np.repeat(trend[:, -1:], n_total - trend.shape[1], axis=1)
        out = base * np.concatenate([trend, pad], axis=1)
        if self.mass_dependent:
            out = out * self._mass_shape(snap, n_total)[None, :]
        return out[0] if np.ndim(params_unit) == 1 else out
