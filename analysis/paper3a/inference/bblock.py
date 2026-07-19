"""JOINT_AB_PLAN step 4: the B-side likelihood L_B(coords) on rusty —
a faithful reimplementation of B5's frozen model layer (wp5 REPORT
task 1), validated against the recorded numbers.

B5 design being mirrored:
- data: frozen Wiener <Y(nu)>, 4 science nu bins at the 4' headline
  radius (stack_wiener_sm2am_fid.npz; recorded [4.71,12.58,37.34,87.23]
  e-6, sigma_total [0.97,2.31,7.19,28.63]e-6);
- model: one exact GP per nu bin on log10<Y> over (dln M_gas, dln T),
  anisotropic squared-exponential + per-point noise from the grid MC
  errors, constant mean, hyperparameters by marginal likelihood;
- C = Sigma_stat (diagonal at the 4-bin/one-radius vector: the frozen
  covariance carries cross-RADIUS terms per bin, no cross-bin) +
  diag sigma_sys(decomposed)^2 + diag[((r_b-1) yhat_b)^2] (selection
  residual, FINAL tf R^2 ratios) + diag[GP var].

Departures from B5, all forced and disclosed:
- training set: the rusty grid mirror has the 60 twobound units only;
  B5 trained on 61 (+ the bind (0,0) anchor). The anchor is included
  automatically when the 62-unit table is re-rsync'd (run name 'bind');
  until then the chi2-anchor validation quantifies the difference.
- R^2 ratio orientation ([bin][radius] vs [radius][bin]) is not
  self-describing in b4_summary; validation tries both and the winner
  (closest to the three recorded chi2 anchors) is recorded in the
  emitted json.

Joint-fit frame mapping (pre-registered in JOINT_AB_PLAN step-4 spec):
the GP lives in the grid's bind-reference frame; mirror coordinates map
in as c_B = c_mirror + TWOBOUND_REF_OFFSET (measured at run_0018,
confirmed by Popeye to 4 decimals; cause: the bind fiducial was painted
at CAMELS-CV cosmology). Coordinate uncertainty is propagated through
GP finite-difference gradients with sigma_c^2 = err_mirror^2 +
(SLOPE_SYS*c)^2 + OFFSET_SYS^2 and added as a diagonal tier.

sigma_pos nuisance (ladder item 3): per-bin dilution of the 4' CAP
value from a nonneg Gaussian-mixture fit to the near-origin unit's
CAP(2-8') curve; blurring the mean profile == scattering positions;
prior U[0, 3.6'] applied by the sampler.

Run: python -m analysis.paper3a.inference.bblock   (validation mode)
Out: wp6_propagation/bblock_validation.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, nnls

B = Path("/mnt/ceph/users/mlee1/paper3/B")
GRID = B / "wp4_mocks/model_grid_tfwiener.npz"
STACK = B / "wp2_measurement/stack_wiener_sm2am_fid.npz"
SYS = B / "wp3_nulls/sigma_sys_wiener_decomposed.npz"
SUMMARY = B / "wp4_mocks/b4_summary_tfwiener.json"
WP6 = Path("/mnt/ceph/users/mlee1/paper3/A/wp6_propagation")

RADIUS_IDX = 2                     # 4' headline (cap_radii [2,3,4,6,8])
SCI = slice(1, 5)                  # science nu bins [1,2),[2,3),[3,4),[4,12)

# recorded B5 anchors (wp5 REPORT task 3) — validation targets
REC_DATA = np.array([4.71, 12.58, 37.34, 87.23]) * 1e-6
REC_SIGMA_TOT = np.array([0.97, 2.31, 7.19, 28.63]) * 1e-6
REC_CHI2 = {"fiducial": 443.8, "run_0004": 313.1, "corner": 161.8}
CORNER = np.array([-0.35, -0.03])

# frame mapping (JOINT_AB_PLAN step-4 spec)
TWOBOUND_REF_OFFSET = np.array([-0.0387, +0.0143])
SLOPE_SYS = 0.10
OFFSET_SYS = np.array([0.016, 0.014])
SIGMA_POS_MAX_ARCMIN = 3.6


class _GP2D:
    """Exact GP, constant mean, anisotropic SE kernel, fixed per-point
    noise; amplitude/lengths by marginal likelihood (multistart)."""

    def __init__(self, X, y, yerr):
        self.X, self.y, self.yerr = X, y, yerr
        self.mean = float(np.mean(y))
        r = self.y - self.mean
        span = X.max(0) - X.min(0)

        def nll(p):
            amp, lx, ly = np.exp(p)
            K = self._kernel(X, X, amp, (lx, ly))
            K[np.diag_indices_from(K)] += yerr**2 + 1e-12
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            a = np.linalg.solve(L, r)
            return float(0.5 * a @ a + np.log(np.diag(L)).sum())

        best = None
        for f in (0.5, 1.0, 2.0, 4.0):
            p0 = np.log([np.std(r) + 1e-3, f * span[0], f * span[1]])
            o = minimize(nll, p0, method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-6,
                                  "maxiter": 2000})
            if best is None or o.fun < best.fun:
                best = o
        self.amp, self.lx, self.ly = np.exp(best.x)
        K = self._kernel(X, X, self.amp, (self.lx, self.ly))
        K[np.diag_indices_from(K)] += yerr**2 + 1e-12
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(
            self._L.T, np.linalg.solve(self._L, r))

    @staticmethod
    def _kernel(A, Bm, amp, ls):
        d0 = (A[:, None, 0] - Bm[None, :, 0]) / ls[0]
        d1 = (A[:, None, 1] - Bm[None, :, 1]) / ls[1]
        return amp**2 * np.exp(-0.5 * (d0**2 + d1**2))

    def predict(self, Xs):
        Xs = np.atleast_2d(Xs)
        Ks = self._kernel(Xs, self.X, self.amp, (self.lx, self.ly))
        mu = self.mean + Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        var = np.maximum(self.amp**2 - np.einsum("ij,ij->j", v, v), 1e-12)
        return mu, var


class _CapDilution:
    """D_b(sigma_pos): CAP(4') dilution from a nonneg Gaussian-mixture
    fit to the near-origin unit's CAP(2-8') curve. For a 2D Gaussian of
    width s, CAP(R) = (2 a s^2 / R^2) (1 - exp(-R^2/2s^2))^2; blurring
    by sigma maps s^2 -> s^2 + sigma^2 (linearity of the stack)."""

    WIDTHS = np.geomspace(0.5, 12.0, 24)          # arcmin basis

    def __init__(self, radii, cap_curves):
        self.radii = radii
        self.amps = []
        for cy in cap_curves:                      # one curve per bin
            M = self._cap_matrix(radii, self.WIDTHS)
            a, _ = nnls(M, cy)
            self.amps.append(a)

    @staticmethod
    def _cap_matrix(R, s):
        R = np.asarray(R, float)[:, None]
        s2 = np.asarray(s, float)[None, :] ** 2
        return (2 * s2 / R**2) * (1 - np.exp(-R**2 / (2 * s2))) ** 2

    def dilution(self, sigma_pos, radius_idx=RADIUS_IDX):
        R = self.radii[radius_idx:radius_idx + 1]
        out = []
        for a in self.amps:
            base = float(self._cap_matrix(R, self.WIDTHS) @ a)
            sb = np.sqrt(self.WIDTHS**2 + sigma_pos**2)
            blur = float(self._cap_matrix(R, sb) @ a)
            out.append(blur / base)
        return np.array(out)

    def dilution_batch(self, sigma_pos, radius_idx=RADIUS_IDX):
        """(N,) sigma_pos -> (N, 4) dilution factors."""
        sig = np.asarray(sigma_pos, float)
        R = float(self.radii[radius_idx])
        s2 = self.WIDTHS[None, :] ** 2 + sig[:, None] ** 2   # (N, W)
        M = (2 * s2 / R**2) * (1 - np.exp(-R**2 / (2 * s2))) ** 2
        base = self._cap_matrix(np.array([R]), self.WIDTHS)[0]
        out = np.empty((len(sig), len(self.amps)))
        for b, a in enumerate(self.amps):
            out[:, b] = (M @ a) / float(base @ a)
        return out


class BBlock:
    def __init__(self, r2_orientation="bin_radius"):
        g = np.load(GRID, allow_pickle=True)
        names = [str(n) for n in g["run_names"]]
        keep = [i for i, n in enumerate(names) if n != "truth"
                and "truth" not in n]
        self.train_names = [names[i] for i in keep]
        self.has_anchor = any(n == "bind" or "bind" in n
                              for n in self.train_names)
        coords = np.stack([g["delta_ln_mgas"], g["delta_ln_t"]],
                          axis=1)[keep]
        y = np.asarray(g["y_mean"], float)[keep][:, SCI, RADIUS_IDX]
        mc = np.asarray(g["y_mc_err"], float)[keep][:, SCI, RADIUS_IDX]
        self.gps = [_GP2D(coords, np.log10(y[:, b]),
                          mc[:, b] / (y[:, b] * np.log(10)))
                    for b in range(4)]

        s = np.load(STACK, allow_pickle=True)
        self.data = np.asarray(s["y_mean"], float)[SCI, RADIUS_IDX]
        cov = np.asarray(s["y_cov"], float)          # (bin, rad, rad)
        self.var_stat = cov[SCI, RADIUS_IDX, RADIUS_IDX]
        # sigma_sys file: rank-1 CIB covariance ACROSS nu bins at the 4'
        # radius — sigma_sys == outer(half_band, half_band); keep the
        # full 4x4 block (off-diagonals are part of the frozen design)
        sysn = np.load(SYS, allow_pickle=True)
        self.cov_sys = np.asarray(sysn["sigma_sys"], float)[SCI, SCI]
        self.sig_sys = np.sqrt(np.diag(self.cov_sys))
        r2 = np.asarray(json.loads(SUMMARY.read_text())
                        ["r2_selection_validation"]["ratio_bind_over_truth"])
        self.r2 = (r2[SCI, RADIUS_IDX] if r2_orientation == "bin_radius"
                   else r2[RADIUS_IDX, SCI])
        self.r2_orientation = r2_orientation

        # near-origin unit for the dilution profile (anchor if present)
        d = np.linalg.norm(coords, axis=1)
        i0 = int(np.argmin(d))
        radii = np.asarray(s["cap_radii_arcmin"], float)
        curves = np.asarray(g["y_mean"], float)[keep][i0, SCI, :]
        self.dil = _CapDilution(radii, curves)
        self.profile_unit = self.train_names[i0]

    # ---- core -----------------------------------------------------------
    def model(self, coords_B, sigma_pos=0.0):
        """Model vector + diagonal theta-dependent variance at grid-frame
        coords (the full C adds cov_sys and Sigma_stat)."""
        mu = np.empty(4)
        gvar = np.empty(4)
        for b, gp in enumerate(self.gps):
            m, v = gp.predict(coords_B)
            mu[b], gvar[b] = m[0], v[0]
        y = 10.0 ** mu
        if sigma_pos > 0:
            y = y * self.dil.dilution(sigma_pos)
        var_diag = (((self.r2 - 1.0) * y) ** 2
                    + (y * np.log(10) * np.sqrt(gvar)) ** 2)
        return y, var_diag

    def cov(self, y, var_diag, coord_tier=None):
        C = np.diag(self.var_stat + var_diag) + self.cov_sys
        if coord_tier is not None:
            C = C + np.diag(coord_tier)
        return C

    def chi2(self, coords_B, sigma_pos=0.0, coord_var=None):
        y, var_diag = self.model(coords_B, sigma_pos)
        tier = (self._coord_var_tier(coords_B, y, coord_var, sigma_pos)
                if coord_var is not None else None)
        C = self.cov(y, var_diag, tier)
        r = y - self.data
        return float(r @ np.linalg.solve(C, r)), y, C

    def _coord_var_tier(self, coords_B, y, coord_var, sigma_pos=0.0):
        eps = 1e-3
        tier = np.zeros(4)
        for i in range(2):
            cp = np.array(coords_B, float)
            cp[i] += eps
            yp, _ = self.model(cp, sigma_pos)
            dy = (yp - y) / eps
            tier = tier + dy**2 * coord_var[i]
        return tier

    def loglike(self, coords_mirror, coords_err, sigma_pos):
        """L_B for the joint fit: mirror-frame coords in, grid-frame
        chi2 out, with the pre-registered coordinate systematics."""
        if not (0.0 <= sigma_pos <= SIGMA_POS_MAX_ARCMIN):
            return -np.inf
        c = np.asarray(coords_mirror, float) + TWOBOUND_REF_OFFSET
        cvar = (np.asarray(coords_err, float) ** 2
                + (SLOPE_SYS * c) ** 2 + OFFSET_SYS**2)
        chi2, y, C = self.chi2(c, sigma_pos, coord_var=cvar)
        sign, logdet = np.linalg.slogdet(C)
        return -0.5 * (chi2 + logdet)

    # ---- batched path for the emcee walker vectorization ---------------
    def model_batch(self, coords_B, sigma_pos):
        """(N,2) grid-frame coords + (N,) sigma_pos -> y (N,4),
        theta-dependent diagonal variance (N,4)."""
        coords_B = np.atleast_2d(coords_B)
        mu = np.empty((len(coords_B), 4))
        gvar = np.empty((len(coords_B), 4))
        for b, gp in enumerate(self.gps):
            m, v = gp.predict(coords_B)
            mu[:, b], gvar[:, b] = m, v
        y = 10.0 ** mu * self.dil.dilution_batch(np.asarray(sigma_pos, float))
        var_diag = (((self.r2[None, :] - 1.0) * y) ** 2
                    + (y * np.log(10) * np.sqrt(gvar)) ** 2)
        return y, var_diag

    def loglike_batch(self, coords_mirror, coords_err, sigma_pos):
        """(N,2) mirror coords, (N,2) errs, (N,) sigma_pos -> (N,) logL.
        Same math as loglike(); finite-difference coordinate tier."""
        c = np.atleast_2d(coords_mirror) + TWOBOUND_REF_OFFSET[None, :]
        sig = np.asarray(sigma_pos, float)
        cvar = (np.atleast_2d(coords_err) ** 2
                + (SLOPE_SYS * c) ** 2 + OFFSET_SYS[None, :] ** 2)
        y, vd = self.model_batch(c, sig)
        eps = 1e-3
        tier = np.zeros_like(y)
        for i in range(2):
            cp = c.copy()
            cp[:, i] += eps
            yp, _ = self.model_batch(cp, sig)
            tier += ((yp - y) / eps) ** 2 * cvar[:, i:i + 1]
        C = np.broadcast_to(self.cov_sys, (len(y), 4, 4)).copy()
        d = self.var_stat[None, :] + vd + tier
        C[:, np.arange(4), np.arange(4)] += d
        r = y - self.data[None, :]
        sol = np.linalg.solve(C, r[:, :, None])[:, :, 0]
        chi2 = np.einsum("nb,nb->n", r, sol)
        sign, logdet = np.linalg.slogdet(C)
        out = -0.5 * (chi2 + logdet)
        out[(sig < 0) | (sig > SIGMA_POS_MAX_ARCMIN)] = -np.inf
        return out


def main() -> None:
    out = {"recorded": {"data_1e6": (REC_DATA * 1e6).tolist(),
                        "sigma_tot_1e6": (REC_SIGMA_TOT * 1e6).tolist(),
                        "chi2": REC_CHI2}}
    g = np.load(GRID, allow_pickle=True)
    names = [str(n) for n in g["run_names"]]
    i4 = names.index([n for n in names if "run_0004" in n][0])
    c_run4 = np.array([float(g["delta_ln_mgas"][i4]),
                       float(g["delta_ln_t"][i4])])

    for orient in ("bin_radius", "radius_bin"):
        blk = BBlock(r2_orientation=orient)
        sig_tot = np.sqrt(blk.var_stat + blk.sig_sys**2)
        res = {"data_match": bool(np.allclose(blk.data, REC_DATA,
                                              rtol=0, atol=5e-9)),
               "sigma_tot_1e6": np.round(sig_tot * 1e6, 3).tolist(),
               "has_anchor": blk.has_anchor,
               "n_train": len(blk.train_names),
               "profile_unit": blk.profile_unit,
               "r2_at_4am": blk.r2.tolist(),
               "chi2": {}}
        for tag, c in (("fiducial", np.zeros(2)), ("run_0004", c_run4),
                       ("corner", CORNER)):
            chi2, y, _ = blk.chi2(c)
            res["chi2"][tag] = {"value": round(chi2, 1),
                                "recorded": REC_CHI2[tag],
                                "ratio_vs_recorded": round(
                                    chi2 / REC_CHI2[tag], 3),
                                "data_over_model":
                                    np.round(blk.data / y, 3).tolist()}
        res["dilution_at_2am"] = np.round(
            blk.dil.dilution(2.0), 3).tolist()
        res["dilution_at_3p6am"] = np.round(
            blk.dil.dilution(3.6), 3).tolist()
        out[orient] = res

    WP6.mkdir(exist_ok=True)
    (WP6 / "bblock_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
