"""WP-A8 grid variation 1: the GGL mass-calibration systematic.

This was the one grid row deliberately left unrun because no published
sigma had been located and inventing one would have been the
remembered-number failure mode. The number has now been read from the
PRIMARY source -- Siegel et al. 2025 (2509.10455) Table 1 and Sec 4.2.1:

  * GGL statistical error on log10<M500>: 0.008-0.02 dex per bin
    (0.009 for LRG M3, the bin the A5 kSZ block uses);
  * stellar-mass-estimator systematic: >~ 0.10 dex -- "the different
    stellar mass estimates correspond to >~0.1 dex variation in the
    mean halo mass". This DOMINATES by an order of magnitude and is
    what a 1-sigma mass test should use.

Method mirrors run_a8_sysgrid: importance reweighting of the frozen
joint chain under a kSZ block rebuilt at the shifted target mass. The
shift is a re-pointing rather than a covariance widening, so IS is in
its well-behaved direction (contrast the three widening variants, which
collapsed to ESS 24-141 and needed fresh chains).

Rebuilding KszBlock is the expensive part (~70 s each, MC dilution), so
this runs the two shifts in one process.

Run: python analysis/paper3a/scripts/run_a8_massshift.py
Out: wp8_robustness/a8_massshift.json
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.inference.jointab import JointBBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402
from analysis.paper3a.observables.mock_sample import (  # noqa: E402
    SIEGEL_GGL_LOGM500_STAT_DEX, SIEGEL_GGL_LOGM500_SYS_DEX)

WP8 = Path("/mnt/ceph/users/mlee1/paper3/A/wp8_robustness")
N_SUB = 8_000
FLAG_AT = 0.5
ESS_MIN = 500.0


def wq(x, w, q):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return float(np.interp(q * c[-1], c, x[i]))


def ksz_loglike(U, shift):
    """Rebuild KszBlock at the shifted GGL target mass and score U."""
    os.environ["BIND_PAPER3A_LOGM500_SHIFT"] = f"{shift}"
    import analysis.paper3a.inference.ksz as kmod
    importlib.reload(kmod)
    blk = kmod.KszBlock()
    return blk.loglike(U), float(kmod.LOGM500_TARGET)


def main() -> None:
    WP8.mkdir(exist_ok=True)
    flat = np.concatenate(
        [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, 32) for k in range(4)])
    rng = np.random.default_rng(19)
    U = flat[rng.choice(len(flat), N_SUB, replace=False)]

    jb = JointBBlock()
    coords, _ = jb.coords(U[:, :30])

    sig = SIEGEL_GGL_LOGM500_SYS_DEX
    print(f"1-sigma mass shift = {sig} dex "
          f"(Siegel Sec 4.2.1 stellar-mass-estimator systematic; "
          f"GGL statistical is {SIEGEL_GGL_LOGM500_STAT_DEX['lrg_m3_spec']} "
          "dex, an order of magnitude smaller)\n")

    print("building the fiducial kSZ block ...", flush=True)
    lk_fid, m_fid = ksz_loglike(U, 0.0)

    def headline(w=None):
        w = np.ones(len(U)) if w is None else w
        return {"dln_mgas": wq(coords[:, 0], w, 0.50),
                "dln_t": wq(coords[:, 1], w, 0.50),
                "f_sat": wq(0.10 + 0.20 * U[:, 30], w, 0.50),
                "sigma_pos_arcmin": wq(U[:, 31] * 3.6, w, 0.50)}

    fid = headline()
    spread = {k: 0.5 * (wq(x, np.ones(len(U)), 0.84)
                        - wq(x, np.ones(len(U)), 0.16))
              for k, x in (("dln_mgas", coords[:, 0]),
                           ("dln_t", coords[:, 1]),
                           ("f_sat", 0.10 + 0.20 * U[:, 30]),
                           ("sigma_pos_arcmin", U[:, 31] * 3.6))}

    rows = {}
    for tag, shift in (("mass_plus_1sig", +sig), ("mass_minus_1sig", -sig)):
        print(f"building kSZ block at shift {shift:+.2f} dex ...", flush=True)
        lk, m_t = ksz_loglike(U, shift)
        dl = lk - lk_fid
        w = np.exp(dl - dl.max())
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        h = headline(w)
        mv = {k: round(float((h[k] - fid[k]) / spread[k]), 3) for k in fid}
        mv["flagged"] = bool(any(abs(v) > FLAG_AT for v in mv.values()
                                 if isinstance(v, float)))
        rows[tag] = {**mv, "ess": round(ess, 1),
                     "reliable": bool(ess >= ESS_MIN),
                     "logM500_target": m_t, "shift_dex": shift}
        print(f"  {tag:16s} ESS={ess:8.0f} logM500={m_t:.2f}  "
              + " ".join(f"{k}={mv[k]:+.2f}" for k in
                         ("dln_mgas", "dln_t", "f_sat", "sigma_pos_arcmin"))
              + ("  ** FLAGGED" if mv["flagged"] else ""))

    os.environ.pop("BIND_PAPER3A_LOGM500_SHIFT", None)
    out = {"variation": "WP-A8 grid v1 — GGL mass calibration +/- 1 sigma",
           "sigma_dex_used": sig,
           "sigma_source": "Siegel et al. 2025 (2509.10455) Sec 4.2.1: "
                           "stellar-mass-estimator systematic >~0.1 dex; "
                           "Table 1 GGL statistical is 0.009 dex for LRG M3",
           "logM500_fiducial": m_fid,
           "flag_threshold_sigma": FLAG_AT, "ess_min": ESS_MIN,
           "rows": rows}
    (WP8 / "a8_massshift.json").write_text(json.dumps(out, indent=2))
    worst = max(abs(rows[t][k]) for t in rows
                for k in ("dln_mgas", "dln_t"))
    print(f"\nworst coordinate movement: {worst:.3f} sigma "
          f"-> {'CLEAN' if worst < FLAG_AT else 'FLAGGED'}")
    print(f"wrote {WP8}/a8_massshift.json")


if __name__ == "__main__":
    main()
