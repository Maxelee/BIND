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
dominated by sub-floor halos (Qu's own abundance-matching ticks put the
lower quartiles at 11.86 / 12.49) — since T_kSZ ∝ tau ∝ M, a bin-stack
model structurally over-predicts a broad-sample stack, and modeling the
full composition would need the unreleased per-galaxy mass distribution.
The m3 quartile is the one sample whose mass is BOTH methodologically
uncontested (GGL 13.41 vs abundance tick 13.43 — agreement, unlike m4:
13.81 vs 14.57, an R4-class conflict) AND dead-center in the emulator
bin. m4 is therefore excluded, and the wp5 REPORT carries the fig26 /
Table III context: Qu's own *unrescaled* TNG comparison over-predicts
their data by ~2.7x (their free-amplitude "rescaling factor" 0.367,
which they label physically implausible) — i.e. the amplitude-excess
direction this block tests is present in the paper's own sim comparison.

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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.paper3a.data_vectors.data_vectors import load_ksz_qu2026_lrg_by_mass
from analysis.paper3a.emulator.forward import ForwardModel
from analysis.paper3a.emulator.gasemu import GasEmulator
from analysis.paper3a.observables.constants import TNG_H, TNG_OMEGA_M
from analysis.paper3a.observables.mock_sample import (
    BIGWOOD_SATELLITE_FRACTION_RANGE,
    SIEGEL_GGL_LOGM500_TARGETS,
    MockSampleConfig,
)

WP4 = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator")
QU_ROOT = Path("/mnt/ceph/users/mlee1/paper3/A/wp1_data/ksz_qu_2604.19744")

BLOCK = "ksz0_sr"                 # group kSZ bin [13.2, 13.7] ∋ m3 GGL 13.41
DATA_BIN = "m3"                   # see module docstring: the mass-uncontested quartile
LOGM500_TARGET = SIEGEL_GGL_LOGM500_TARGETS["lrg_m3_spec"]
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
            sig = np.atleast_2d(self.emu.predict(theta, s, return_std=False)[BLOCK])
            t_central = sig @ self._ops[s].T
            dil = 1.0 - f_sat[:, None] + f_sat[:, None] * self._ratio_sat[s][None, :]
            out += w * t_central * dil
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
