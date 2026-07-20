"""The Qu et al. 2026 LRG kSZ likelihood block (WP-A5 task 5).

Data side (fixed at construction)
---------------------------------
- `load_ksz_qu2026_lrg_by_mass()["m3"]`: the third stellar-mass-quartile
  CAP T_kSZ profile (9 radii, 1-6 arcmin — the same grid as the gate
  tables) with its released full covariance (fig15). The estimator
  normalization E[T_hat] = T_CMB (sigma_true/c) tau_CAP *already includes*
  the r ~ 0.65 velocity-reconstruction correction (A1 units audit) —
  nothing is divided in here.

WHY m3 AND NOT THE HEADLINE FULL-LRG VECTOR (documented choice,
2026-07-18): the model observable is the emulated group-bin stack
(halos with log10 M500 in [13.2, 13.7]). The full-LRG sample's GGL *mean*
(13.30) sits inside that bin, but its host-mass *distribution* is
dominated by sub-floor halos (the two lower quartiles are at log M500
12.48 / 12.92 — this docstring previously quoted 11.86 / 12.49, which
were axis-tick labels, not masses; see the retraction below. Both sets
sit far below the group bin's lower edge, so the conclusion is
unchanged) — since T_kSZ ∝ tau ∝ M, a bin-stack model structurally
over-predicts a broad-sample stack, and modeling the full composition
would need the unreleased per-galaxy mass distribution. The m3 quartile
is the sample that is dead-center in the emulator bin; m4 is excluded.
The mass arguments that were recorded here for BOTH of those choices have
since been corrected — read the two blocks immediately below before
quoting either. The wp5 REPORT carries the fig26 / Table III context:
Qu's own *unrescaled* TNG comparison over-predicts their data by ~2.7x
(their free-amplitude "rescaling factor" 0.367, which they label
physically implausible) — i.e. the amplitude-excess direction this block
tests is present in the paper's own sim comparison.

*** RETRACTED 2026-07-20 — the m4 "R4-class mass conflict". ***
WHAT WAS CLAIMED (here, and in WP-A1 / WP-A5): that m3 was "the one
sample whose mass is BOTH methodologically uncontested (GGL 13.41 vs
abundance tick 13.43 — agreement, unlike m4: 13.81 vs 14.57, an R4-class
conflict) AND dead-center in the emulator bin", m4 being excluded partly
on that 0.75 dex conflict.
WHY IT IS WRONG: the 14.57 came from Qu's `m200c_ticks`, which is a
SECONDARY-AXIS LABEL ARRAY, not per-quartile halo masses. Their Fig. 2
caption says so outright — the top axis shows "approximate halo masses
M200c inferred from the [60] stellar-to-halo mass relation". See
`data_vectors.load_ksz_qu2026_lrg_by_mass` and the plans repo's
systematics-hunt/FINDINGS_ksz.md section 2.
WHAT IS ACTUALLY TRUE: recomputing each quartile's mean log M* from the
released histograms — which reproduces Qu's own `fig15b_mean_mstar` to
four decimals, so it is validated against the authors' own product — and
evaluating the same SHMR gives

    quartile   correct log M500   Siegel GGL
    m1         12.48              12.91
    m2         12.92              13.15
    m3         13.26              13.41
    m4         13.75              13.82

m4 agrees with Siegel to 0.07 dex, not 0.75. THE m4 EXCLUSION STANDS,
but it now rests on the Siegel/Bigwood *amplitude* grounds alone: they
omit M500 >~ 10^13.3 from their primary analyses on a spec-vs-photo
discrepancy. This rescues nothing on the deficit — adding m4 would
DEEPEN it, because the data's own m3->m4 scaling at 1' is 2.39x over
0.41 dex (the Siegel GGL m3->m4 separation), i.e. dlogT/dlogM = 0.92,
steeper than the model's 0.722.

*** OPEN — NEEDS ADJUDICATION (raised 2026-07-20, not settled): the m3
mass agreement, i.e. this block's stated reason for using m3. ***
The justification quoted above compares GGL 13.41 against Qu's
abundance tick 13.43 and calls that agreement. Those are DIFFERENT MASS
DEFINITIONS: the GGL number is an M500, the tick an M200c. Put in a
common M500 frame, m3 recomputes to 13.26 against GGL 13.41 — a 0.15 dex
gap, not agreement. The apparent agreement came from the ~0.14 dex frame
mismatch roughly cancelling the offset. (m3 is also the one quartile the
tick error barely moved, its mean M* happening to land on the 11.5 tick,
which is why none of this was caught earlier.)
NOTHING IN THE LIKELIHOOD CHANGES, and no selection is rewritten here:
the model is the stack over the fixed (13.2, 13.7) M200c Msun/h gate bin
regardless of the target mass, so this moves no prediction and no
posterior. It is recorded because it is load-bearing for the PAPER's
stated reason for choosing m3, and must be adjudicated — with the mass
frames made explicit — before that reason is written down again. Held
identically in the plans repo; do not close it here.

Model side (per theta; the whole chain is LINEAR in sigma_r)
------------------------------------------------------------
emulated ksz0_sr (90 radial bins) -> velocity-decorrelation weighting +
floor -> synthetic map -> beam -> CAP -> T_kSZ  ==  A_snap @ sigma_r,
with A_snap (9 x 90) built ONCE by probing `ForwardModel
.ksz_tksz_from_profile` with unit basis vectors (exactness unit-tested).
Per-eval cost is then a matmul, keeping the emcee budget.

- z model: the LRG stack spans z = 0.4-1.1; the model interpolates the two
  bracketing snaps (063 z=0.599, 056 z=0.791) linearly at the
  count-weighted effective redshift z_eff = 0.742 (fig03 dN/dz; the
  per-quartile dN/dz is not released, so the full-sample distribution
  stands in for m3 — a documented approximation consistent with Qu's own
  Med(z) = 0.76). The alternative 4-z-bin count-weighted snap mixture is
  evaluated once at the fiducial and its fractional difference carried as
  the z-modeling systematic (`sys_zmix`).
- Sample model (the 5e prerequisite, now live): satellite dilution via
  `ForwardModel._dilution_ratio` — per-aperture MC mean-offset CAP ratio at
  f_sat = 1 against the fiducial reference profile (the cached 5e design:
  offset kinematics are theta-independent). The diluted prediction
  T -> [(1 - f_sat) + f_sat * ratio_sat] * T is EXACT in f_sat (the MC
  expectation is linear in the satellite fraction), so f_sat rides as an
  explicit nuisance chain dimension (index 30) with the Bigwood prior
  U(0.10, 0.30) (`BIGWOOD_SATELLITE_FRACTION_RANGE`). Mis-centering
  f_mis = 0 per Bigwood's own finding (sub-dominant pixel quantization).
  Satellite Rayleigh scale r_sat = comoving R500c of the GGL target mass
  at z_eff (~0.38 Mpc/h); the x0.5 / x2 bracket is carried as `sys_rsat`.

Covariance
----------
full Qu 9x9 data covariance
 (+) diagonal model terms, fractional of the prediction:
     emulation floor (v4 k-fold ksz0_sr residuals propagated through the
       SAME linear operator — the floor in the observable's own frame)
     velocity normalization 6% (linear-theory sigma_v vs Qu's
       AbacusSummit sigma_true; session-4 measurement)
     beyond-slab correlated gas 3% (A4 budget: gas outside +-25.6 Mpc/h
       adds signal the painted maps cannot contain)
     sys_zmix, sys_rsat (computed at init, see above).

$BIND_PAPER3A_LOGM500_SHIFT — the WP-A8 mass-calibration variation
------------------------------------------------------------------
BUG FIX 2026-07-20 (systematics-hunt FINDINGS_ksz.md defect 3). The env
var existed and was read, but it reached the prediction through TWO dead
ends only: `r_sat` (the satellite Rayleigh scale) and
`MockSampleConfig.logm200c_mean` (which no code path in this block ever
reads — `draw_center_offsets_hmpc` consumes f_sat / f_mis / r_sat_hmpc /
sigma_mis_hmpc and nothing else). The MODEL's mass was the fixed emulator
stack over the (13.2, 13.7) M200c Msun/h gate bin regardless of the
target, so a +0.10 dex shift moved the prediction by -1.0% at 1' with a
radius-dependent profile, instead of raising it. The WP-A8 variation-1
verdict recorded before this fix ("worst coordinate movement 0.028 sigma")
therefore measured a ~-1% perturbation, not the intended +18%.

WHICH IMPLEMENTATION, AND WHY. Two were available:

  (a) re-select the emulator mass bin. Impossible as stated: the emulator
      carries exactly TWO kSZ stacks, the gate bins (13.2, 13.7) and
      (13.7, 15.5) M200c Msun/h, whose stack-effective masses are
      0.56 dex apart. There is no bin to re-select at 0.10 dex.

  (b) apply a measured dlogT/dlogM500 response. A hard-coded scalar
      (the 0.722 quoted by the audit) would freeze the FIDUCIAL
      astrophysics into a systematic that is evaluated across the whole
      30-dim chain, and would force one number onto nine radii.

What is implemented is the continuous generalization of (a), which is
also the per-theta form of (b): the two emulated stacks are treated as
two samples of T_kSZ(logM500) and the prediction is moved LOG-LINEARLY
between them,

    gamma(theta, radius) = log10( T_bin1 / T_bin0 ) / (logM_1 - logM_0)
    T -> T * 10 ** ( gamma * shift )

with logM_0, logM_1 the stack-effective masses log10<M500> [Msun] of the
two gate bins, read from the frozen fiducial operator table for the
snapshot in use. This is exact bin re-selection at shift = +-(logM_1 -
logM_0), it is recomputed at every theta (so the systematic tracks the
astrophysics rather than the fiducial), it is per-radius (a heavier host
is angularly larger, so its CAP profile is genuinely shallower — the
response should NOT be a scalar), and it is free: `emu.predict` already
returns every block, so ksz1_sr costs no extra GP evaluation.

*** CIRCULARITY WARNING — READ BEFORE QUOTING ANY VERIFICATION NUMBER. ***
The audit's expected +18.1% was itself derived from the ratio of these
same two gate-bin stacks (dlogT/dlogM500 = 0.722 across bins 0 and 1 at
snap 063). Recovering ~1.18 from this code therefore CONFIRMS THE
PLUMBING AND NOTHING ABOUT THE PHYSICS: the same measurement is on both
sides of the comparison. It is not evidence that the model's mass
response is right. The physics content of this propagation is exactly
the content of the two-bin ratio, no more.

The verification that carries real information is the RADIAL SHAPE. The
bug's ratio ran 0.9898 -> 0.9995 across the nine radii, MONOTONICALLY
approaching unity, because it was the r_sat dilution term leaking — a
pure geometry artifact with no mass content. Measured after the fix, at
the fiducial theta, pred(+0.10)/pred(0) =

    [1.1735 1.1812 1.1894 1.1934 1.1923 1.1902 1.1893 1.1887 1.1870]

— mean 1.1872, total radial spread 1.69%, and NOT monotone (it peaks at
2.875' and comes back down), so it is not the leaked-geometry shape.
pred(-0.10)/pred(0) = 0.8420 mean, and the two are log-symmetric to
0.06% at every radius (their product is 0.9993-0.9999), which is what a
log-linear response must give and the dilution artifact did not.
Sign and magnitude flip is the other real check: -1.0% -> +18.7%.

Default path: with the env var unset the shift is 0.0 and the response
branch is NOT ENTERED AT ALL — `predict` executes the identical
statements it did before this change. Verified bit-identical (not merely
close) between an unset environment and an explicit shift of 0.0.

The r_sat propagation that was already there is CORRECT and is kept: a
heavier target really does have a larger R500c, hence a broader satellite
offset distribution. The fix adds the amplitude response it was missing;
it does not replace it.

`sys_zmix` deliberately carries no mass response: it is a RATIO of two
forward evaluations, so a common multiplicative factor cancels exactly.

All of the above is asserted in
`inference/tests/test_a8_variant_fixes.py`, whose
`test_full_bin_span_shift_reproduces_the_other_gate_bin` is the one
NON-circular structural check available: shifting by exactly the two
bins' separation must return the bin-1 stack to machine precision,
because a log-linear interpolation is pinned at both endpoints by
construction rather than fitted to them.

The wp2 tau_CAP validation-bias variants (`ksz_wp2bias`, `ksz_wp2bias_all`)
---------------------------------------------------------------------------
OFF by default; see `wp2_tau_bias_divisor` below. Not a default change.
NOT driven by an environment variable (an earlier draft of this docstring
named a `$BIND_PAPER3A_KSZ_WP2BIAS`; no such var is read anywhere).
`run_joint_ab_fit.py::_blocks` sets `KszBlock.wp2_bias_divisor` directly
from `wp2_tau_bias_divisor(self.radii, mode)`. With it left at None the
division is not performed at all, so the default prediction is unchanged
bit-for-bit.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from analysis.paper3a.data_vectors.data_vectors import load_ksz_qu2026_lrg_by_mass
from analysis.paper3a.emulator.forward import ForwardModel
from analysis.paper3a.emulator.gasemu import GasEmulator
from analysis.paper3a.observables.constants import TNG_H, TNG_OMEGA_M
from analysis.paper3a.observables.mock_sample import (
    BIGWOOD_SATELLITE_FRACTION_RANGE,
    SIEGEL_GGL_LOGM500_SYS_DEX,
    SIEGEL_GGL_LOGM500_TARGETS,
    MockSampleConfig,
)

WP4 = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator")
QU_ROOT = Path("/mnt/ceph/users/mlee1/paper3/A/wp1_data/ksz_qu_2604.19744")

BLOCK = "ksz0_sr"                 # group kSZ bin [13.2, 13.7] ∋ m3 GGL 13.41
BLOCK_HI = "ksz1_sr"              # the OTHER emulated gate bin [13.7, 15.5];
                                  # the second point of the mass response
DATA_BIN = "m3"                   # see module docstring: the mass-uncontested quartile
GATE_TABLES = Path("/mnt/ceph/users/mlee1/paper3/A/wp3_gate/operator_tables_v3")
# $BIND_PAPER3A_LOGM500_SHIFT shifts the GGL target mass by that many
# dex, for the WP-A8 grid's mass-calibration variation. The 1-sigma
# scale to use is SIEGEL_GGL_LOGM500_SYS_DEX = 0.10 -- the stellar-mass-
# estimator systematic from Siegel Sec. 4.2.1, which dominates the
# 0.008-0.02 dex GGL statistical error by an order of magnitude.
# It moves BOTH r_sat (already, since 2026-07-18) and -- since the
# 2026-07-20 fix -- the prediction amplitude, through the model's own
# two-gate-bin mass response. See the module docstring.
LOGM500_SHIFT_DEX = float(os.environ.get("BIND_PAPER3A_LOGM500_SHIFT", 0.0))
LOGM500_TARGET = (SIEGEL_GGL_LOGM500_TARGETS["lrg_m3_spec"]
                  + LOGM500_SHIFT_DEX)
# numerical guard on the two-bin stack ratio; the measured value is
# 2.50-2.72 at both likelihood snaps, so this never binds in practice and
# exists only so a pathological emulator draw cannot produce a NaN loglike.
MASS_RESPONSE_RATIO_CLIP = (1e-3, 1e3)
VEL_NORM_SYS = 0.06               # linear sigma_v vs Qu nominal (session 4)
SLAB_SYS = 0.03                   # beyond-slab correlated gas (A4 budget)
N_MC_DILUTION = 128     # mean-ratio MC error ~1%, well under the sys terms;
                        # each draw costs an 8x-supersampled CAP weight build
RHO_CRIT0 = 2.775e11              # h^2 Msun / Mpc^3


def r500c_comoving_hmpc(logm500_msun: float, z: float,
                        omega_m: float = TNG_OMEGA_M, h: float = TNG_H) -> float:
    """Comoving R500c [Mpc/h] of a halo of M500c = 10^logm500 Msun at z."""
    m_msunh = 10.0 ** logm500_msun * h
    e2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
    r_proper = (3.0 * m_msunh / (4.0 * np.pi * 500.0 * RHO_CRIT0 * e2)) ** (1.0 / 3.0)
    return r_proper * (1.0 + z)


# --------------------------------------------------------------------------
# wp2 painted-vs-truth tau_CAP validation bias — VARIANT ONLY, off by default
# --------------------------------------------------------------------------
# FINDINGS_bind_layer.md section 1: `inference/fgas.py` divides its f_gas
# prediction by PAINT_BIAS = 1.074 (the wp2 painted-vs-TNG300-hydro-truth
# bias) and carries a 2% residual; this block carries NO equivalent term
# for the wp2-measured painted-vs-truth tau_CAP bias, although wp2 told A5
# to either correct the mean or carry Sigma_model, and the
# sigma_model_ksz_tauCAP_snap*.npz covariances are loaded nowhere in the
# likelihood. The two probes that receive opposite treatment are exactly
# the two that disagree at 6 sigma, so the asymmetry has to be tested.
#
# frac_bias = stack(painted - truth) / |stack(truth)| at snap 063 (z=0.599),
# the highest-z snapshot wp2 measured and the closest one below this
# block's z_eff = 0.742. NO z-extrapolation is applied: the measured trend
# degrades monotonically with z (at 6' it runs .133 -> .691 -> 1.000 ->
# 1.095 over z = 0.034 -> 0.599), so using snap 063 UNDERSTATES the bias at
# z_eff. That is the conservative direction for a variant whose purpose is
# to ask whether the omission matters.
WP2_VALIDATION = Path("/mnt/ceph/users/mlee1/paper3/A/wp2_validation")
WP2_TAU_CAP_BIAS_SNAP = "063"
WP2_TAU_CAP_BIAS_RADII_ARCMIN = (1.0, 2.25, 3.5, 4.75, 6.0)
# The variant READS the frozen wp2 artifact rather than trusting a
# transcription; the tuple below is the value on record and is asserted
# against the file every time, so the two cannot silently diverge.
WP2_TAU_CAP_FRAC_BIAS = (0.126, 0.088, 0.197, 0.847, 1.095)
# *** THE OUTER TWO RADII WERE VALIDATED 2026-07-20. *** The open worry
# was that CAP is a COMPENSATED filter, so stack(truth) at large theta is
# a difference of large numbers and could be near zero, making frac_bias
# 0.847 / 1.095 an absolutely insignificant small-denominator artifact.
# That hypothesis is now REFUTED, from rusty -- the wp2 artifacts turned
# out to be mirrored at /mnt/ceph/users/mlee1/paper3/A/wp2_validation/
# (frozen; read-only), so no Popeye job was needed. Deriving
# stack(truth) = bias / frac_bias from the npz, at snap 063, n_halos=601:
#
#   theta   stack(truth)   bias (painted-truth)  sqrt(diag Sigma_model)  off/sig
#   1.00'     4.152e-04          5.219e-05             2.575e-06           20.3
#   2.25'     2.359e-03          2.071e-04             8.273e-06           25.0
#   3.50'     3.467e-03          6.831e-04             1.553e-05           44.0
#   4.75'     4.210e-03          3.565e-03             5.316e-05           67.1
#   6.00'     5.101e-03          5.585e-03             7.781e-05           71.8
#
# stack(truth) is MONOTONICALLY INCREASING in theta: the outer-radius truth
# stack is the LARGEST of the five, 12.3x the innermost, not the smallest.
# The small-denominator escape hatch fails on its own terms. Independently,
# the pre-registered criterion in FINDINGS_bind_layer.md candidate 1 step 1
# ("<1 sigma => artifact, >2 sigma => the block carries an uncorrected
# factor-2 error on 2 of 9 radii") resolves >2 sigma at 67.1 and 71.8. Both
# readings agree. (Sigma_model is a bootstrap over 601 halos, i.e. an error
# on the MEAN, so the sigma test is weak rather than wrong -- it will read
# ">2 sigma" almost unconditionally; see RUNBOOK.md 4.1, criterion flagged
# for amendment, NOT changed. The absolute-magnitude reading is the one
# carrying the weight here.) A4 separately argues the opposite about the
# same radii ("the composite's two-halo excess profile matches linear
# theory ... the painted field is healthy"); that contradiction is NOT
# resolved by this read -- A4 and wp2 still cannot both be fully right,
# which is why both modes are kept and run rather than one being adopted.
#
# Hence the mode switch, and hence its default:
#   "inner" (DEFAULT) -- correct only theta <= 3.5', where the bias is
#                        7-13%, well measured, and consistent across all
#                        four wp2 snapshots. This is what the variant
#                        `ksz_wp2bias` runs. Retained as the CONSERVATIVE
#                        BRACKET, not as the believed answer.
#   "all"             -- correct all nine radii, including the factor
#                        1.85-2.1 implied at 4.75'/6'. VALIDATED (above);
#                        run as `ksz_wp2bias_all`. Still deliberately NOT
#                        the default: validated means the correction is
#                        evidence-backed, not that the likelihood should
#                        silently adopt it. The inner-vs-all pair is the
#                        measurement -- their difference is how much of the
#                        effect lives in the outer two radii.
WP2_TAU_CAP_BIAS_INNER_MAX_ARCMIN = 3.5


def wp2_tau_cap_frac_bias(snap: str = WP2_TAU_CAP_BIAS_SNAP) -> np.ndarray:
    """(5,) frac_bias = stack(painted - truth) / |stack(truth)| at the wp2
    radii, from the frozen `sigma_model_ksz_tauCAP_snap{snap}.npz`."""
    with np.load(WP2_VALIDATION / f"sigma_model_ksz_tauCAP_snap{snap}.npz",
                 allow_pickle=False) as f:
        frac = np.asarray(f["frac_bias"], float)
    if snap == WP2_TAU_CAP_BIAS_SNAP and not np.allclose(
            frac, WP2_TAU_CAP_FRAC_BIAS, atol=5e-4):
        raise ValueError(
            f"wp2 frac_bias at snap {snap} reads {frac} but this module has "
            f"{WP2_TAU_CAP_FRAC_BIAS} on record — the frozen artifact moved, "
            "or this module is stale. Adjudicate before running any variant "
            "that consumes it.")
    return frac


def wp2_tau_bias_divisor(radii_arcmin, mode: str = "inner") -> np.ndarray:
    """(n,) divisor (1 + frac_bias) for the kSZ prediction — the f_gas
    PAINT_BIAS treatment applied to tau_CAP. 1.0 where no correction is
    applied, so `pred / divisor` is an exact no-op there.

    `mode`: "inner" (theta <= 3.5', the well-measured regime — the
    conservative bracket) or "all" (every radius, including the factor
    1.85-2.1 at 4.75'/6' — VALIDATED 2026-07-20 against the absolute
    stack(truth), see the module-level table).
    """
    if mode not in ("inner", "all"):
        raise ValueError(f"mode must be 'inner' or 'all', got {mode!r}")
    r = np.asarray(radii_arcmin, float)
    frac = np.interp(r, WP2_TAU_CAP_BIAS_RADII_ARCMIN, wp2_tau_cap_frac_bias())
    if mode == "inner":
        frac = np.where(r <= WP2_TAU_CAP_BIAS_INNER_MAX_ARCMIN, frac, 0.0)
    return 1.0 + frac


def lrg_zeff_and_binweights(fig03_path: Path | None = None):
    """(z_eff, [(z_center_b, weight_b)]) from the released fig03 dN/dz."""
    d = np.load(fig03_path or (QU_ROOT / "fig03_redshift_distribution.npz"))
    edges = d["bin_edges"]
    c = 0.5 * (edges[1:] + edges[:-1])
    counts = [d[f"z{i}_counts"] for i in range(1, 5)]
    tot = np.sum(counts, axis=0)
    zeff = float(np.sum(c * tot) / tot.sum())
    zw = []
    for cnt in counts:
        n = cnt.sum()
        zw.append((float(np.sum(c * cnt) / n), float(n)))
    wsum = sum(w for _, w in zw)
    return zeff, [(z, w / wsum) for z, w in zw]


class KszBlock:
    """Callable likelihood block: (theta, f_sat) -> T_kSZ(9 radii) vs the
    frozen Qu LRG vector under the full data covariance + model terms.

    Chain convention: columns [0:30] are the SB35 unit cube; column 30 (if
    present) is the f_sat nuisance, uniform on the unit interval mapped to
    `BIGWOOD_SATELLITE_FRACTION_RANGE`. Without a column 30 the midpoint
    f_sat = 0.20 is used (fixed-nuisance mode, for diagnostics only).
    """

    N_THETA = 30
    F_SAT_RANGE = BIGWOOD_SATELLITE_FRACTION_RANGE

    def __init__(self, emu: GasEmulator | None = None,
                 fm: ForwardModel | None = None,
                 kfold_path: Path | None = None):
        self.emu = emu or GasEmulator.load()
        self.fm = fm or ForwardModel(emulator=self.emu)
        self.data = load_ksz_qu2026_lrg_by_mass()[DATA_BIN]
        self.radii = np.asarray(self.data.bins, float)
        self.values = np.asarray(self.data.values, float)
        self.cov_data = np.asarray(self.data.covariance, float)

        ds = np.load(WP4 / "gasemu_dataset_v4.npz", allow_pickle=False)
        self.r_centers = np.asarray(ds["sigma_r_centers_mpch"], float)
        n_r = len(self.r_centers)

        self.zeff, self.zbin_weights = lrg_zeff_and_binweights()
        snap_z = self.emu.snap_z
        self.snap_lo, self.snap_hi = "063", "056"          # bracket z_eff
        z_lo, z_hi = snap_z[self.snap_lo], snap_z[self.snap_hi]
        self.w_hi = (self.zeff - z_lo) / (z_hi - z_lo)

        # ---- linear operators by basis probing (exact; unit-tested) -------
        # only the two likelihood snaps need the full operator; the z-mix
        # systematic below uses direct fiducial forward calls instead
        self._ops = {s: self._probe_operator(s, n_r)
                     for s in (self.snap_lo, self.snap_hi)}

        # ---- GGL mass-calibration shift (WP-A8 variation 1) ----------------
        # Off unless $BIND_PAPER3A_LOGM500_SHIFT is set. When off, NOTHING
        # here runs and `predict` takes the pre-2026-07-20 code path
        # verbatim, so the default prediction is bit-identical.
        self.logm500_shift = LOGM500_SHIFT_DEX
        self._dlogm_gate_bins = None
        if self.logm500_shift != 0.0:
            self._dlogm_gate_bins = {
                s: self._gate_bin_logm500_span(s)
                for s in (self.snap_lo, self.snap_hi)}

        # ---- wp2 tau_CAP validation bias (VARIANT; off by default) ---------
        # `run_joint_ab_fit.py::_blocks` sets this to
        # wp2_tau_bias_divisor(self.radii, mode) for the ksz_wp2bias
        # variants. None -> no division at all (exact no-op).
        self.wp2_bias_divisor = None
        self.wp2_bias_mode = None

        # ---- satellite dilution (5e cached-ratio design) -------------------
        self.r_sat = r500c_comoving_hmpc(LOGM500_TARGET, self.zeff)
        ratios = {}               # (snap, r_sat scale) -> (9,) all-sat ratio
        for s in (self.snap_lo, self.snap_hi):
            for scale in (1.0, 0.5, 2.0):
                cfg = MockSampleConfig(
                    logm200c_mean=LOGM500_TARGET, logm200c_sigma=0.2,
                    f_sat=1.0, r_sat_hmpc=scale * self.r_sat)
                ratios[(s, scale)] = self.fm._dilution_ratio(
                    self._fid_sigma(s), self.r_centers, snap_z[s], self.radii,
                    cfg, n_mc=N_MC_DILUTION, seed=0, n_floor_bins=10,
                    cosmology=None)
        self._ratio_sat = {s: ratios[(s, 1.0)] for s in (self.snap_lo, self.snap_hi)}
        # r_sat bracket -> fractional systematic at the nominal f_sat = 0.2
        f_nom = 0.5 * (self.F_SAT_RANGE[0] + self.F_SAT_RANGE[1])
        sys_rsat = np.zeros(len(self.radii))
        for s in (self.snap_lo, self.snap_hi):
            d_nom = 1.0 - f_nom + f_nom * ratios[(s, 1.0)]
            for scale in (0.5, 2.0):
                d_alt = 1.0 - f_nom + f_nom * ratios[(s, scale)]
                sys_rsat = np.maximum(sys_rsat, np.abs(d_alt / d_nom - 1.0))
        self.sys_rsat = sys_rsat

        # ---- z-mixture systematic (4-bin count-weighted vs 2-snap z_eff) ---
        t_fid_2snap = self._t_central_fid((self.snap_lo, self.snap_hi),
                                          {self.snap_lo: 1.0 - self.w_hi,
                                           self.snap_hi: self.w_hi})
        zmix_snaps = ("067", "063", "056", "049")
        zs = np.array([snap_z[s] for s in zmix_snaps])
        t_per_snap = np.array([
            self.fm.ksz_tksz_from_profile(self._fid_sigma(s), self.r_centers,
                                          snap_z[s], self.radii).tksz
            for s in zmix_snaps])
        t_mix = np.zeros(len(self.radii))
        for z_b, w_b in self.zbin_weights:
            zc = np.clip(z_b, zs.min(), zs.max())
            t_b = np.array([np.interp(zc, zs, t_per_snap[:, j])
                            for j in range(len(self.radii))])
            t_mix += w_b * t_b
        self.sys_zmix = np.abs(t_mix / t_fid_2snap - 1.0)

        # ---- emulation floor through the same operator ---------------------
        kf = np.load(kfold_path or (WP4 / "gasemu_kfold_v4.npz"), allow_pickle=False)
        snaps_kf = [str(s) for s in kf["snaps"]]
        sl = self.emu.block_slices[BLOCK]
        fr = np.zeros(len(self.radii))
        for s, w in ((self.snap_lo, 1.0 - self.w_hi), (self.snap_hi, self.w_hi)):
            zi = snaps_kf.index(s)
            t_true = kf["truth"][:, zi, sl].astype(float) @ self._ops[s].T
            t_pred = kf["pred"][:, zi, sl].astype(float) @ self._ops[s].T
            fr += w * np.sqrt(np.nanmean(((t_pred - t_true) / t_true) ** 2, axis=0))
        self.emul_frac = fr

        self.sys_frac = np.sqrt(self.emul_frac**2 + VEL_NORM_SYS**2 + SLAB_SYS**2
                                + self.sys_zmix**2 + self.sys_rsat**2)

    # ----------------------------------------------------------- internals --

    def _probe_operator(self, snap: str, n_r: int) -> np.ndarray:
        """(9, n_r) matrix st. T = A @ sigma_r, by unit-basis probing of the
        exact `ksz_tksz_from_profile` code path (linear end to end)."""
        z = self.emu.snap_z[snap]
        cols = np.empty((n_r, len(self.radii)))
        for j in range(n_r):
            e = np.zeros(n_r)
            e[j] = 1.0
            cols[j] = self.fm.ksz_tksz_from_profile(e, self.r_centers, z,
                                                    self.radii).tksz
        return cols.T

    def _gate_bin_logm500_span(self, snap: str) -> float:
        """logM_1 - logM_0 [dex]: the separation of the two emulated kSZ
        gate bins in stack-effective mass, at `snap`.

        The stack is a plain mean of per-halo CAP profiles and tau_CAP is
        very nearly linear in halo mass, so the mass that characterizes a
        stack is log10<M500> (log of the LINEAR mean), not <log10 M500>.
        Masses come from the frozen fiducial operator table, converted
        M500 Msun/h -> Msun to match the Siegel GGL target's frame.

        Approximation, stated: the stacks cap at `max_halos_per_bin = 200`
        per slab, which this uses every in-bin halo instead. Measured cost
        at snap 063 bin 0: log10<M500> = 13.4533 (all 1034) vs 13.4485
        (first 200) = 0.005 dex on a 0.56 dex span, i.e. <0.2% on the
        response factor.
        """
        from analysis.paper3a.gate.gate_operators import DEFAULT_KSZ_MASS_BINS

        with np.load(GATE_TABLES / f"fiducial_run_0000_snap{snap}.npz",
                     allow_pickle=False) as f:
            logm200 = np.asarray(f["logm200"], float)
            m500_msun = np.asarray(f["m500_msunh"], float) / TNG_H
        eff = []
        for lo, hi in DEFAULT_KSZ_MASS_BINS:
            sel = (logm200 >= lo) & (logm200 < hi)
            if not sel.any():
                raise ValueError(f"snap {snap}: gate bin [{lo}, {hi}) is empty "
                                 "— cannot define the kSZ mass response")
            eff.append(float(np.log10(np.mean(m500_msun[sel]))))
        span = eff[1] - eff[0]
        if not span > 0.1:
            raise ValueError(f"snap {snap}: gate-bin mass span {span:.4f} dex "
                             "is degenerate — refusing to divide by it")
        return span

    def _mass_response(self, snap: str, t_lo: np.ndarray,
                       t_hi: np.ndarray) -> np.ndarray:
        """(N, 9) factor 10**(gamma * shift) re-pointing the prediction from
        the gate-bin-0 stack to the shifted GGL target mass, where
        gamma = dlog10 T / dlog10 M500 is read off the block's own two
        emulated mass stacks at this theta and this radius.

        See the module docstring for why this and not a hard-coded slope —
        and for why recovering ~1.18 at +0.10 dex is a CIRCULAR check.
        """
        ratio = np.clip(t_hi / t_lo, *MASS_RESPONSE_RATIO_CLIP)
        gamma = np.log10(ratio) / self._dlogm_gate_bins[snap]
        return 10.0 ** (gamma * self.logm500_shift)

    def _fid_sigma(self, snap: str) -> np.ndarray:
        from analysis.paper3a.emulator import params_meta as pm

        cache = getattr(self, "_fid_sigma_cache", None)
        if cache is None:
            cache = self._fid_sigma_cache = {}
        if snap not in cache:
            u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
            out = self.emu.predict(u_fid, snap, return_std=False)[BLOCK]
            cache[snap] = np.atleast_2d(out)[0]
        return cache[snap]

    def _t_central_fid(self, snaps, weights) -> np.ndarray:
        return np.sum([weights[s] * (self._ops[s] @ self._fid_sigma(s))
                       for s in snaps], axis=0)

    # ------------------------------------------------------------- forward --

    def _split(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.atleast_2d(np.asarray(u, float))
        theta = u[:, : self.N_THETA]
        lo, hi = self.F_SAT_RANGE
        if u.shape[1] > self.N_THETA:
            f_sat = lo + (hi - lo) * u[:, self.N_THETA]
        else:
            f_sat = np.full(len(u), 0.5 * (lo + hi))
        return theta, f_sat

    def predict(self, u: np.ndarray) -> np.ndarray:
        """(N, 9) diluted T_kSZ predictions [muK arcmin^2] at the Qu radii."""
        theta, f_sat = self._split(u)
        out = np.zeros((len(theta), len(self.radii)))
        for s, w in ((self.snap_lo, 1.0 - self.w_hi), (self.snap_hi, self.w_hi)):
            pred = self.emu.predict(theta, s, return_std=False)
            sig = np.atleast_2d(pred[BLOCK])
            t_central = sig @ self._ops[s].T
            if self.logm500_shift != 0.0:
                # BLOCK_HI rides along in the same GP call — no extra cost
                t_hi = np.atleast_2d(pred[BLOCK_HI]) @ self._ops[s].T
                t_central = t_central * self._mass_response(s, t_central, t_hi)
            dil = 1.0 - f_sat[:, None] + f_sat[:, None] * self._ratio_sat[s][None, :]
            out += w * t_central * dil
        if self.wp2_bias_divisor is not None:
            out = out / np.asarray(self.wp2_bias_divisor, float)[None, :]
        return out

    def cov(self, pred: np.ndarray) -> np.ndarray:
        """(N, 9, 9) total covariance: data + diagonal model terms."""
        pred = np.atleast_2d(pred)
        c = np.broadcast_to(self.cov_data, (len(pred),) + self.cov_data.shape).copy()
        d = (self.sys_frac[None, :] * pred) ** 2
        c[:, np.arange(len(self.radii)), np.arange(len(self.radii))] += d
        return c

    # ----------------------------------------------------------- likelihood --

    def loglike(self, u: np.ndarray) -> np.ndarray:
        pred = self.predict(u)
        c = self.cov(pred)
        r = pred - self.values[None, :]
        sol = np.linalg.solve(c, r[:, :, None])[:, :, 0]
        chi2 = np.sum(r * sol, axis=1)
        _, logdet = np.linalg.slogdet(c)
        n = len(self.radii)
        return -0.5 * (chi2 + logdet + n * np.log(2 * np.pi))

    def chi2(self, u: np.ndarray) -> np.ndarray:
        pred = self.predict(u)
        c = self.cov(pred)
        r = pred - self.values[None, :]
        sol = np.linalg.solve(c, r[:, :, None])[:, :, 0]
        return np.sum(r * sol, axis=1)
