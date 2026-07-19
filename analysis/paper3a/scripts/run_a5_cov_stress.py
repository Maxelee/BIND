"""A5 robustness: the kSZ-covariance stress test recommended at the
decision-1 ruling — does the fgas-vs-kSZ exclusion survive if the kSZ
model systematics were UNDERESTIMATED by a factor 2?

Method (job-free): importance reweighting of the frozen kszonly chains.
For each posterior sample u, w = L_stress(u) / L_fid(u) where L_stress
doubles the systematic tiers in the kSZ covariance:
- variant "phys2x": VEL_NORM_SYS, SLAB_SYS, sys_zmix, sys_rsat doubled
  (emul_frac unchanged — emulator precision is measured, not assumed);
- variant "all2x": the whole sys_frac doubled (harsher than any
  plausible accounting error).
The reweighted f_group posterior is compared against the untouched
fgas-only chains: gauss z = |Δmedian| / sqrt(sigma_k^2 + sigma_f^2)
with sigma = (p84-p16)/2. Effective sample size (ESS) is reported —
weights widen the target so IS is conservative in the tails; a healthy
ESS validates the medians/quantiles used here.

PRE-REGISTERED CRITERION (recorded before running): the exclusion
stands if z >= 3 under BOTH variants.

Run: python analysis/paper3a/scripts/run_a5_cov_stress.py
Out: wp5_chains/a5_cov_stress.json
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.ksz import (  # noqa: E402
    SLAB_SYS, VEL_NORM_SYS, KszBlock)
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402
from analysis.paper3a.scripts.fig_a5_tension import load_flat  # noqa: E402

N_SUB = 40_000


def stressed(ksz: KszBlock, variant: str) -> KszBlock:
    s = copy.copy(ksz)
    if variant == "phys2x":
        s.sys_frac = np.sqrt(ksz.emul_frac**2
                             + (2 * VEL_NORM_SYS) ** 2 + (2 * SLAB_SYS) ** 2
                             + (2 * ksz.sys_zmix) ** 2
                             + (2 * ksz.sys_rsat) ** 2)
    elif variant == "all2x":
        s.sys_frac = 2.0 * ksz.sys_frac
    else:
        raise ValueError(variant)
    return s


def wq(x, w, q):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return float(np.interp(q * c[-1], c, x[i]))


def main() -> None:
    ksz = KszBlock()
    fgas = FgasBlock(emu=ksz.emu)

    flat_k = load_flat("a5_subset_kszonly_seed{k}.npz")
    flat_f = load_flat("a5_final_seed{k}.npz")
    rng = np.random.default_rng(11)
    sub_k = flat_k[rng.choice(len(flat_k), N_SUB, replace=False)]
    sub_f = flat_f[rng.choice(len(flat_f), N_SUB, replace=False)]

    fg_f = fgas.predict(sub_f[:, :30])[:, 0]
    med_f = float(np.percentile(fg_f, 50))
    sig_f = float(np.percentile(fg_f, 84) - np.percentile(fg_f, 16)) / 2

    fg_k = fgas.predict(sub_k[:, :30])[:, 0]
    ll_o = ksz.loglike(sub_k)

    out = {"criterion": "PASS if gauss z >= 3 under both variants "
                        "(pre-registered in this script's docstring)",
           "n_sub": N_SUB,
           "fgas_only": {"f_group_median": med_f, "sigma": sig_f},
           "fiducial": {}}

    med0 = float(np.percentile(fg_k, 50))
    sig0 = float(np.percentile(fg_k, 84) - np.percentile(fg_k, 16)) / 2
    z0 = abs(med0 - med_f) / np.hypot(sig0, sig_f)
    out["fiducial"] = {"f_group_median": med0, "sigma": sig0,
                       "z_vs_fgas": float(z0)}

    for variant in ("phys2x", "all2x"):
        s = stressed(ksz, variant)
        ll_s = s.loglike(sub_k)
        lw = ll_s - ll_o
        w = np.exp(lw - lw.max())
        ess = float(w.sum() ** 2 / (w**2).sum())
        med = wq(fg_k, w, 0.50)
        sig = (wq(fg_k, w, 0.84) - wq(fg_k, w, 0.16)) / 2
        z = abs(med - med_f) / np.hypot(sig, sig_f)
        imax = int(np.argmax(ll_s))
        chi2_map = float(s.chi2(sub_k[imax])[0])
        out[variant] = {
            "sys_frac_range": [float(s.sys_frac.min()),
                               float(s.sys_frac.max())],
            "ess": ess, "ess_frac": ess / N_SUB,
            "f_group_median": med, "sigma": float(sig),
            "z_vs_fgas": float(z),
            "chi2_at_reweighted_map": chi2_map,
            "PASS_z_ge_3": bool(z >= 3.0),
        }
    out["PASS"] = bool(all(out[v]["PASS_z_ge_3"]
                           for v in ("phys2x", "all2x")))
    path = CHAINS / "a5_cov_stress.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
