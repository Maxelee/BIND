"""WP-B5 model layer: GP interpolation of the frozen B4 grid over the
(Delta ln M_gas, Delta ln T) plane.

Produced by Project B WP-B5 (see bind-paper3-plans, private repo). Inputs are
the MODEL_FREEZE.md-hashed artifacts only: `model_grid_tfwiener.npz` (60
twobound units) plus the bind-fiducial unit as the (0, 0) anchor — the
session-6 fig-6 finding (the fiducial sits at the EDGE of the twobound
cloud) makes the anchor load-bearing for extrapolation toward
stronger-than-TNG feedback. The TNG truth unit is NEVER trained on (it is
the validation object).

Model: one exact GP per nu bin on log10<Y>, anisotropic squared-exponential
kernel + per-point Gaussian noise from the grid MC errors (delta method:
sigma_log10 = sigma / (y ln 10)), hyperparameters by marginal-likelihood
maximization. The GP posterior variance is the model-error tier carried into
the B5 likelihood (it contains both the grid-MC noise and the interpolation
error, and grows off the sampled cloud — honest extrapolation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

WP4 = Path("/mnt/home/mlee1/ceph/paper3/B/wp4_mocks")
RAD_IDX = 2                     # 4' fiducial radius
B5_BINS = slice(1, 5)
LN10 = np.log(10.0)


def _unit_vec(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    return (np.asarray(d["y_mean"], float)[B5_BINS, RAD_IDX],
            np.asarray(d["y_mc_err"], float)[B5_BINS, RAD_IDX])


@dataclass
class GridTable:
    """Training table: N points x 4 bins (+ per-point MC errors)."""

    coords: np.ndarray          # (N, 2) = (dln_mgas, dln_t)
    y: np.ndarray               # (N, 4) <Y> at 4' [y arcmin^2]
    y_err: np.ndarray           # (N, 4) grid MC error
    names: list

    @classmethod
    def load(cls, grid_npz: Path = WP4 / "model_grid_tfwiener.npz",
             include_fiducial_anchor: bool = True,
             fiducial_unit: Path = WP4 / "grid_tfwiener" / "bind_run_0000.npz",
             ) -> "GridTable":
        g = np.load(grid_npz, allow_pickle=True)
        coords = np.stack([np.asarray(g["delta_ln_mgas"], float),
                           np.asarray(g["delta_ln_t"], float)], axis=1)
        y = np.asarray(g["y_mean"], float)[:, B5_BINS, RAD_IDX]
        e = np.asarray(g["y_mc_err"], float)[:, B5_BINS, RAD_IDX]
        names = [str(n) for n in g["run_names"]]
        if include_fiducial_anchor:
            yf, ef = _unit_vec(fiducial_unit)
            coords = np.vstack([coords, [0.0, 0.0]])
            y = np.vstack([y, yf])
            e = np.vstack([e, ef])
            names.append("bind/run_0000 (fiducial anchor)")
        if not (y > 0).all():
            raise ValueError("non-positive <Y> in the grid — log model invalid")
        return cls(coords, y, e, names)


class GP2D:
    """Exact GP, anisotropic squared-exponential + fixed per-point noise.

    k(x, x') = a^2 exp(-1/2 sum_d (x_d - x'_d)^2 / l_d^2);  y ~ N(0, K + D)
    with D = diag(sigma_i^2). Hyperparameters (log a, log l1, log l2) by
    L-BFGS on the negative log marginal likelihood. Targets are centered on
    their mean internally.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, yerr: np.ndarray):
        self.X = np.asarray(X, float)
        self.mu0 = float(np.mean(y))
        self.y = np.asarray(y, float) - self.mu0
        self.var_n = np.asarray(yerr, float) ** 2
        # scale-aware inits: amplitude ~ target std, lengths ~ coord span/3
        span = np.ptp(self.X, axis=0)
        p0 = np.log([max(self.y.std(), 1e-3), *(np.maximum(span / 3.0, 1e-3))])
        res = minimize(self._nll, p0, method="L-BFGS-B",
                       bounds=[(-12, 6)] + [(np.log(s / 50.0), np.log(s * 10.0))
                                            for s in np.maximum(span, 1e-3)])
        self.params = res.x
        self._factorize(self.params)

    def _kernel(self, A: np.ndarray, B: np.ndarray, p) -> np.ndarray:
        a, l1, l2 = np.exp(p)
        d = (A[:, None, :] - B[None, :, :]) / np.array([l1, l2])
        return a ** 2 * np.exp(-0.5 * np.sum(d ** 2, axis=-1))

    def _factorize(self, p) -> None:
        K = self._kernel(self.X, self.X, p) + np.diag(self.var_n)
        K[np.diag_indices_from(K)] += 1e-12
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def _nll(self, p) -> float:
        try:
            K = self._kernel(self.X, self.X, p) + np.diag(self.var_n)
            K[np.diag_indices_from(K)] += 1e-12
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        return float(0.5 * self.y @ a + np.log(np.diag(L)).sum())

    def predict(self, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mean, variance) at query points (M, 2)."""
        Ks = self._kernel(np.asarray(Xq, float), self.X, self.params)
        mean = Ks @ self.alpha + self.mu0
        v = np.linalg.solve(self.L, Ks.T)
        kss = np.exp(self.params[0]) ** 2
        var = np.maximum(kss - np.sum(v ** 2, axis=0), 1e-12)
        return mean, var


@dataclass
class GridEmulator:
    """4 independent per-bin GPs on log10<Y> over the coordinate plane."""

    table: GridTable
    gps: list

    @classmethod
    def fit(cls, table: GridTable) -> "GridEmulator":
        gps = []
        for b in range(table.y.shape[1]):
            logy = np.log10(table.y[:, b])
            logerr = table.y_err[:, b] / (table.y[:, b] * LN10)
            gps.append(GP2D(table.coords, logy, logerr))
        return cls(table, gps)

    def predict(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(y (M,4), var_y (M,4)) in LINEAR units (delta-method back-transform)."""
        coords = np.atleast_2d(np.asarray(coords, float))
        y = np.empty((coords.shape[0], len(self.gps)))
        vy = np.empty_like(y)
        for b, gp in enumerate(self.gps):
            m, v = gp.predict(coords)
            y[:, b] = 10.0 ** m
            vy[:, b] = (y[:, b] * LN10) ** 2 * v
        return y, vy

    def loo_table(self) -> dict:
        """Leave-one-out: refit without each point, predict it, tabulate.

        Returns per-bin median/max |rel err| and the per-point pulls
        (err / sqrt(GP var + MC var)) whose std should be ~1 if the GP error
        model is calibrated.
        """
        t = self.table
        n, nbin = t.y.shape
        rel = np.zeros((n, nbin))
        pull = np.zeros((n, nbin))
        for i in range(n):
            keep = np.arange(n) != i
            sub = GridTable(t.coords[keep], t.y[keep], t.y_err[keep],
                            [t.names[j] for j in np.where(keep)[0]])
            em = GridEmulator.fit(sub)
            yp, vp = em.predict(t.coords[i])
            rel[i] = (yp[0] - t.y[i]) / t.y[i]
            pull[i] = (yp[0] - t.y[i]) / np.sqrt(vp[0] + t.y_err[i] ** 2)
        return {"rel": rel, "pull": pull,
                "median_abs_rel": np.median(np.abs(rel), axis=0),
                "max_abs_rel": np.abs(rel).max(axis=0),
                "pull_std": pull.std(axis=0)}
