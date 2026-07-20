"""JOINT_AB_PLAN step 4: the joint-fit B block — theta in, L_B out.

Chains are 32-dim: cols 0-29 the SB35 astro unit cube, col 30 the f_sat
nuisance (consumed by KszBlock, ignored here), col 31 the sigma_pos
nuisance in unit coordinates (sigma_pos = u31 * 3.6 arcmin; ladder-item-3
prior U[0, 3.6']).

Per theta: the gate-validated statsemu mirror gives (dln M_gas, dln T) +
propagated errors (GP std through the weighted sum + the CV floors);
BBlock maps them into the grid's bind-reference frame and evaluates the
frozen-B5 likelihood with the pre-registered coordinate-systematic tier
(slope band + suite-offset disagreement). See `bblock.py` and the
JOINT_AB_PLAN step-4 spec.
"""

from __future__ import annotations

import json

import numpy as np

from analysis.paper3a.emulator import params_meta as pm
from analysis.paper3a.emulator.statsemu import WP6, StatsEmulator
from analysis.paper3a.inference.bblock import BBlock
from analysis.paper3a.scripts.run_ab_gate import bin_weights, delta_coords

SIGMA_POS_MAX = 3.6            # arcmin; U[0, 3.6] prior via col 31


class JointBBlock:
    # Multiplier on the propagated coordinate errors. 1.0 is the
    # fiducial analysis; WP-A8's `emul2x` systematics variant sets 2.0
    # so this block's emulator-error tier is widened alongside the kSZ
    # and fgas ones. Applied inside `loglike`, so setting it on an
    # existing instance takes effect on the next call.
    err_scale: float = 1.0

    def __init__(self, bblock: BBlock | None = None,
                 emu: StatsEmulator | None = None):
        self.b = bblock or BBlock()
        self.emu = emu or StatsEmulator.load()
        raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz",
                      allow_pickle=False)
        edges = np.asarray(json.loads(str(raw["manifest"]))
                           ["meta"]["mass_bins"], float)
        self.w = bin_weights(edges)
        self.u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)

    def coords(self, U30: np.ndarray):
        """(N,30) unit cube -> mirror coords (N,2) + errors (N,2)."""
        cc = delta_coords(self.emu, np.atleast_2d(U30), self.u_fid, self.w)
        c = np.stack([np.atleast_1d(cc["dln_mgas"]),
                      np.atleast_1d(cc["dln_t"])], axis=1)
        e = np.stack([np.atleast_1d(cc["dln_mgas_err"]),
                      np.atleast_1d(cc["dln_t_err"])], axis=1)
        return c, e

    def loglike(self, U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(U)
        c, e = self.coords(U[:, :30])
        sigma_pos = U[:, 31] * SIGMA_POS_MAX
        return self.b.loglike_batch(c, self.err_scale * e, sigma_pos)

    # data-space hooks for the battery / assembly
    def predict(self, U: np.ndarray) -> np.ndarray:
        """(N,32) -> model <Y> vectors (N,4) in the grid frame."""
        U = np.atleast_2d(U)
        c, _ = self.coords(U[:, :30])
        from analysis.paper3a.inference.bblock import TWOBOUND_REF_OFFSET
        y, _ = self.b.model_batch(c + TWOBOUND_REF_OFFSET[None, :],
                                  U[:, 31] * SIGMA_POS_MAX)
        return y

    def chi2(self, u: np.ndarray):
        """MAP-style single-theta chi2 against the frozen B data."""
        u = np.asarray(u, float)
        c, e = self.coords(u[None, :30])
        from analysis.paper3a.inference.bblock import (
            SLOPE_SYS, OFFSET_SYS, TWOBOUND_REF_OFFSET)
        cB = c[0] + TWOBOUND_REF_OFFSET
        cvar = e[0] ** 2 + (SLOPE_SYS * cB) ** 2 + OFFSET_SYS**2
        return self.b.chi2(cB, u[31] * SIGMA_POS_MAX, coord_var=cvar)
