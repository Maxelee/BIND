"""WP-B5 likelihood: frozen data vs the GP-interpolated model grid.

Covariance tiers (plan section 'Method', all explicit):
  C(theta) = Sigma_stat  (frozen jackknife, diagonal by construction)
           + Sigma_sys   (frozen decomposed CIB rank-1 term)
           + D_sel       (B4 selection residual: diag[((r_b - 1) yhat_b)^2]
                          with r = the FINAL tf-grid R2 bind/truth ratios)
           + D_model     (GP predictive variance at theta — grid-MC +
                          interpolation, grows off the sampled cloud)

Posterior: brute-force evaluation on a 2D coordinate grid (the plane is
2-dim; no sampler needed), uniform prior over the evaluation window
(documented; window wide enough to expose edge-piling, which is itself a
pre-registered finding — the exclusion-style tripwire).

Data access: `stack.frozen.load_frozen` is imported ONLY inside
`fit_frozen_data()` — recovery tests (mock-only) cannot touch the data path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gridmodel import B5_BINS, RAD_IDX, GridEmulator

WP4 = Path("/mnt/home/mlee1/ceph/paper3/B/wp4_mocks")

# evaluation window: generous around the twobound cloud
# (mgas in [-0.17, +0.10], t in [-0.014, +0.037]) so edge-piling is visible
DEFAULT_WINDOW = ((-0.35, 0.15), (-0.03, 0.06))


def selection_residual_ratios(summary_json: Path = WP4 / "b4_summary_tfwiener.json"
                              ) -> np.ndarray:
    """(4,) FINAL tf-grid R2 bind/truth ratios, B5 bins at 4' (frozen record)."""
    s = json.loads(Path(summary_json).read_text())
    r = np.asarray(s["r2_selection_validation"]["ratio_bind_over_truth"], float)
    return r[B5_BINS, RAD_IDX]


@dataclass
class B5Posterior:
    """Posterior over (dln_mgas, dln_t) on a rectangular grid."""

    mg: np.ndarray              # (nx,) grid axis
    dt: np.ndarray              # (ny,)
    lnlike: np.ndarray          # (nx, ny)

    @property
    def posterior(self) -> np.ndarray:
        p = np.exp(self.lnlike - self.lnlike.max())
        return p / p.sum()

    def mean_and_cov(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.posterior
        MG, DT = np.meshgrid(self.mg, self.dt, indexing="ij")
        m = np.array([np.sum(p * MG), np.sum(p * DT)])
        c = np.empty((2, 2))
        c[0, 0] = np.sum(p * (MG - m[0]) ** 2)
        c[1, 1] = np.sum(p * (DT - m[1]) ** 2)
        c[0, 1] = c[1, 0] = np.sum(p * (MG - m[0]) * (DT - m[1]))
        return m, c

    def map_point(self) -> np.ndarray:
        i, j = np.unravel_index(np.argmax(self.lnlike), self.lnlike.shape)
        return np.array([self.mg[i], self.dt[j]])

    def hpd_level(self, frac: float) -> float:
        """Posterior-density threshold enclosing `frac` of the mass."""
        p = np.sort(self.posterior.ravel())[::-1]
        c = np.cumsum(p)
        return float(p[np.searchsorted(c, frac)])

    def contains(self, point: np.ndarray, frac: float) -> bool:
        """Is `point` inside the `frac` highest-posterior-density region?"""
        p = self.posterior
        i = int(np.argmin(np.abs(self.mg - point[0])))
        j = int(np.argmin(np.abs(self.dt - point[1])))
        return bool(p[i, j] >= self.hpd_level(frac))

    def edge_mass(self, k_border: int = 2) -> float:
        """Posterior mass within k grid cells of the window border (tripwire)."""
        p = self.posterior
        inner = p[k_border:-k_border, k_border:-k_border].sum()
        return float(1.0 - inner)


def grid_loglike(y_data: np.ndarray, cov_base: np.ndarray,
                 emulator: GridEmulator,
                 sel_ratios: np.ndarray | None = None,
                 window=DEFAULT_WINDOW, n=161) -> B5Posterior:
    """ln L(theta) = -1/2 [r^T C^-1 r + ln det C] on the coordinate grid."""
    mg = np.linspace(*window[0], n)
    dt = np.linspace(*window[1], n)
    MG, DT = np.meshgrid(mg, dt, indexing="ij")
    pts = np.stack([MG.ravel(), DT.ravel()], axis=1)
    yhat, vhat = emulator.predict(pts)                        # (M,4), (M,4)

    lnl = np.empty(pts.shape[0])
    for i in range(pts.shape[0]):
        C = cov_base + np.diag(vhat[i])
        if sel_ratios is not None:
            C = C + np.diag(((sel_ratios - 1.0) * yhat[i]) ** 2)
        r = y_data - yhat[i]
        sign, logdet = np.linalg.slogdet(C)
        lnl[i] = -0.5 * (r @ np.linalg.solve(C, r) + logdet)
    return B5Posterior(mg, dt, lnl.reshape(n, n))


def fit_frozen_data(emulator: GridEmulator, variant: str = "wiener",
                    window=DEFAULT_WINDOW, n=161,
                    with_selection: bool = True,
                    sel_summary_json: Path = WP4 / "b4_summary_tfwiener.json",
                    ) -> tuple[B5Posterior, dict]:
    """THE data fit. Only call after recovery tests pass (plan rule)."""
    from ..stack.frozen import load_frozen                    # data path: here only

    f = load_frozen(variant)
    sel = selection_residual_ratios(sel_summary_json) if with_selection else None
    post = grid_loglike(f.y, f.cov_total, emulator, sel_ratios=sel,
                        window=window, n=n)
    m, c = post.mean_and_cov()
    info = {"variant": variant, "y_data": f.y.tolist(),
            "sigma_total": np.sqrt(np.diag(f.cov_total)).tolist(),
            "posterior_mean": m.tolist(),
            "posterior_cov": c.tolist(),
            "map": post.map_point().tolist(),
            "edge_mass": post.edge_mass()}
    return post, info
