"""Sigma_theory(theta): the theory-error budget the WP-A5 likelihood uses.

Plan task 3: combine, per predicted data-vector element,

1. **Emulation error** — the GP posterior std at theta (theta-dependent,
   grows off the design), floored by the k-fold held-out residual per bin
   (the GP's own error bars are calibrated but optimistic in the corners —
   the twobound out-of-design test quantifies this; see REPORT).
2. **Operator/BIND validation error** — the wp2 truth-validation
   systematics, applied in the painted frame: fractional f_gas bias band,
   CylToSph per-halo scatter / sqrt(N_stack), painted-y error, x_e.
3. **Finite-sample stochastic error** — the dataset bootstrap SEM per bin
   (shared DMO halos across theta: correlated between design points, so it
   enters the *absolute* prediction error once, not per-theta).

All three are variances added in quadrature into a diagonal
Sigma_theory(theta); off-diagonal emulation covariance is available from
the PCA structure but deferred until A5 decides its data-vector binning
(diagonal is conservative for well-separated bins).

kSZ terms additionally carry the forward-model systematics (velocity-
decorrelation kernel choice, beyond-slab completion <= 3%) — but NOT the
sample-selection modeling (miscentering, satellites, LRG mass calibration),
which is a *forward-model component* A5 fits with the Bigwood-style priors,
not an error inflation. See REPORT section "kSZ residual".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gasemu import GasEmulator

WP4 = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator")

# wp2 truth-validation systematics (run 2451362, final), painted frame.
WP2_FRAC_SYS = {
    "fgas_med": 0.074,      # painted f_gas high-bias band at z~0 (7.4-10.8%)
    "fgas_scat": 0.074,     # scales with the median bias
    "ym": 0.02,             # painted y validated <= 2% to z ~ 0.6
    "ksz0": 0.014,          # x_e / tau-map conversion error
    "ksz1": 0.014,
}


class SigmaTheory:
    """Callable per-snapshot diagonal theory covariance."""

    def __init__(self, emulator: GasEmulator | None = None,
                 kfold_path: Path = WP4 / "gasemu_kfold_v3.npz",
                 dataset_path: Path = WP4 / "gasemu_dataset_v3.npz"):
        self.emu = emulator or GasEmulator.load()
        kf = np.load(kfold_path, allow_pickle=False)
        snaps = [str(s) for s in kf["snaps"]]
        # per-(snap, dim) k-fold rms residual: the emulation-error floor
        resid = kf["pred"].astype(float) - kf["truth"].astype(float)
        self._kfold_rms = {s: np.sqrt(np.nanmean(resid[:, i] ** 2, axis=0))
                           for i, s in enumerate(snaps)}
        ds = np.load(dataset_path, allow_pickle=False)
        # bootstrap SEM per dim, median over design points (shared halos)
        self._sem = {s: np.nanmedian(ds[f"snap{s}_sem_sb35"], axis=0) for s in snaps}

    def __call__(self, params, snap: str) -> dict:
        """Diagonal sigma per observable dim at theta (painted frame),
        decomposed and combined. NaN at dims not emulated at this snap."""
        mean, gp_std = self.emu.predict_vector(params, snap, return_std=True)
        floor = self._kfold_rms[snap]
        emu_var = np.maximum(gp_std, floor) ** 2
        sem = self._sem[snap]
        stoch_var = np.where(np.isfinite(sem), sem**2, 0.0)
        sys_var = np.zeros_like(mean)
        for b, sl in self.emu.block_slices.items():
            # v3 profile/own blocks are kSZ-derived: same conversion systematic
            frac = WP2_FRAC_SYS.get(b, WP2_FRAC_SYS["ksz0"])
            sys_var[..., sl] = (frac * np.abs(mean[..., sl])) ** 2
        total = np.sqrt(emu_var + stoch_var + sys_var)
        return {
            "mean": mean,
            "sigma_total": total,
            "sigma_emulation": np.sqrt(emu_var),
            "sigma_stochastic": np.sqrt(stoch_var),
            "sigma_wp2_systematic": np.sqrt(sys_var),
        }

    def summary(self, snap: str) -> str:
        parts = {b: float(np.nanmedian(self._kfold_rms[snap][sl]))
                 for b, sl in self.emu.block_slices.items()}
        return json.dumps({"kfold_rms_floor_median": parts}, indent=2)
