"""WP-A8 stream 1: the systematics grid, importance-reweighted.

Reruns the joint A+B inference under every defensible analysis variation
that is expressible as a modified likelihood, WITHOUT new chains: for a
subsample of the frozen joint posterior, w = exp(L_variant - L_fiducial)
(only the changed block contributes — the others cancel), and the
headline coordinates are re-quoted under w. The cov-stress script
established the pattern; this generalizes it to the plan's grid.

Movement metric (pre-registered, from the WP-A8 plan): a variation is
FLAGGED if it moves a headline posterior median by more than 0.5x that
coordinate's statistical sigma ( = (p84-p16)/2 of the fiducial joint
posterior). Flagged variations get a dedicated paragraph in the paper.
ESS is reported per variant; a variant whose ESS collapses (< 500) is
marked UNRELIABLE rather than quoted.

Variant map (plan item -> implementation):
  3  central_only     f_sat forced to 0 (vs the fitted U[0.10,0.30])
  2  rsat_wide        the satellite-geometry/miscentering bracket
                      (sys_rsat) doubled in the kSZ covariance
  8  emul2x           Sigma_theory doubled EVERYWHERE: kSZ emul_frac x2,
                      fgas emul_frac x2, B-block propagated GP errs x2
  5  fgas_model2x     the fgas 2D-projection/transfer model tier
                      (PAINT_BIAS_RESID, C2S_TRANSFER_SYS) doubled
  5b b_coordsys2x     B's pre-registered coordinate-systematic tier
                      (SLOPE_SYS, OFFSET_SYS) doubled
  7  drop-one rows    quoted from the FROZEN subset chains/summaries
                      (kszonly, A-joint = drop-B) — no reweighting
  1  mass_shift       EXECUTED — see run_a8_massshift.py and
                      wp8_robustness/a8_massshift.json. The sigma is
                      Siegel 2509.10455 Sec 4.2.1 (>~0.1 dex
                      stellar-mass-estimator systematic, which dominates
                      the 0.009 dex Table-1 statistical error)
  4  cap_radii        DEFERRED to a re-fit: changes the data vector
  6  stochasticity    N/A here: wp4 diagonal Sigma_theory was confirmed
                      final and the A2 optional variants closed (see
                      wp2/wp4 REPORT notes)
  9  simba_spot       optional; not on rusty (SIMBA 1P data on Popeye)

Run: python analysis/paper3a/scripts/run_a8_sysgrid.py
Out: wp8_robustness/a8_sysgrid.json
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

from analysis.paper3a.inference import bblock as bblock_mod  # noqa: E402
from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.jointab import JointBBlock  # noqa: E402
from analysis.paper3a.inference.ksz import KszBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402

WP8 = Path("/mnt/ceph/users/mlee1/paper3/A/wp8_robustness")

# Subsample size. The deliverable is Delta(median)/sigma_stat against a
# 0.5-sigma flag threshold, so the Monte-Carlo error on a median,
# ~1.25/sqrt(N) sigma, only has to be small compared to 0.5 sigma:
# N = 8000 gives ~0.014 sigma, i.e. 35x margin. A first attempt at
# N = 30000 did not finish inside 30 minutes and bought nothing for this
# threshold -- the cost is dominated by GP predictive-sigma evaluations,
# which scale linearly in N.
N_SUB = 8_000
FLAG_AT = 0.5          # x sigma_stat, the plan's threshold
ESS_MIN = 500.0


def log(msg):
    """Progress with an explicit flush -- stdout is block-buffered when
    the script is run through a pipe, which hid all progress on the
    first attempt."""
    print(msg, flush=True)


def wq(x, w, q):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return float(np.interp(q * c[-1], c, x[i]))


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def load_joint(n, seed=17):
    flat = np.concatenate(
        [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, 32) for k in range(4)])
    rng = np.random.default_rng(seed)
    return flat[rng.choice(len(flat), n, replace=False)]


def headline(U, coords, w=None):
    """Weighted medians+sigmas of the four headline quantities."""
    if w is None:
        w = np.ones(len(U))
    out = {}
    for name, x in (("dln_mgas", coords[:, 0]), ("dln_t", coords[:, 1]),
                    ("f_sat", 0.10 + 0.20 * U[:, 30]),
                    ("sigma_pos_arcmin", U[:, 31] * 3.6)):
        out[name] = {"p16": wq(x, w, 0.16), "p50": wq(x, w, 0.50),
                     "p84": wq(x, w, 0.84)}
    return out


def movement(fid, var):
    """Delta(median)/sigma_stat per headline quantity + the flag."""
    out = {}
    for k in fid:
        sig = 0.5 * (fid[k]["p84"] - fid[k]["p16"])
        d = (var[k]["p50"] - fid[k]["p50"]) / sig if sig > 0 else np.nan
        out[k] = round(float(d), 3)
    out["flagged"] = bool(any(abs(v) > FLAG_AT for v in out.values()
                              if isinstance(v, float) and np.isfinite(v)))
    return out


def main() -> None:
    WP8.mkdir(exist_ok=True)
    U = load_joint(N_SUB)

    import time
    t0 = time.time()

    def el():
        return f"[{time.time()-t0:6.1f}s]"

    log(f"{el()} building KszBlock ...")
    ksz = KszBlock()
    log(f"{el()} building FgasBlock ...")
    fgas = FgasBlock(emu=ksz.emu)
    log(f"{el()} building JointBBlock ...")
    jb = JointBBlock()

    log(f"{el()} coords on {len(U)} samples (GP predictive sigma) ...")
    coords, cerr = jb.coords(U[:, :30])
    log(f"{el()} fiducial kSZ loglike ...")
    lk_fid = ksz.loglike(U)
    log(f"{el()} fiducial fgas loglike ...")
    lf_fid = fgas.loglike(U)
    sigma_pos = U[:, 31] * 3.6
    log(f"{el()} fiducial B loglike ...")
    lb_fid = jb.b.loglike_batch(coords, cerr, sigma_pos)
    log(f"{el()} fiducial done")

    fid = headline(U, coords)
    results = {"fiducial": fid}
    table = {}

    def add(name, dln_l, note):
        w = np.exp(dln_l - dln_l.max())
        e = ess(w)
        h = headline(U, coords, w)
        mv = movement(fid, h)
        table[name] = {**mv, "ess": round(e, 1),
                       "reliable": bool(e >= ESS_MIN), "note": note}
        results[name] = h
        log(f"  {name:16s} ESS={e:8.0f} "
              + " ".join(f"{k}={mv[k]:+.2f}" for k in
                         ("dln_mgas", "dln_t", "f_sat", "sigma_pos_arcmin"))
              + ("  ** FLAGGED" if mv["flagged"] else ""))

    # -- v3 central-only: f_sat = 0 exactly ------------------------------
    log(f"{el()} variants ...")
    k0 = copy.copy(ksz)
    k0.F_SAT_RANGE = (0.0, 0.0)
    add("central_only", k0.loglike(U) - lk_fid,
        "plan v3: f_sat=0 vs fitted U[0.10,0.30] mixture")

    # -- v2 satellite-geometry / miscentering bracket doubled ------------
    k2 = copy.copy(ksz)
    k2.sys_frac = np.sqrt(ksz.sys_frac**2 + 3.0 * ksz.sys_rsat**2)  # 2x tier
    add("rsat_wide", k2.loglike(U) - lk_fid,
        "plan v2: sys_rsat doubled in quadrature (widened miscentering/"
        "satellite-geometry prior)")

    # -- v8 Sigma_theory doubled everywhere ------------------------------
    k8 = copy.copy(ksz)
    k8.sys_frac = np.sqrt(ksz.sys_frac**2 + 3.0 * ksz.emul_frac**2)
    f8 = copy.copy(fgas)
    f8.emul_frac = 2.0 * fgas.emul_frac
    lb8 = jb.b.loglike_batch(coords, 2.0 * cerr, sigma_pos)
    add("emul2x",
        (k8.loglike(U) - lk_fid) + (f8.loglike(U) - lf_fid) + (lb8 - lb_fid),
        "plan v8: emulator-error tier doubled in kSZ + fgas + B blocks")

    # -- v5 fgas model tier doubled --------------------------------------
    from analysis.paper3a.inference import fgas as fgas_mod

    class Fgas2x(FgasBlock):
        def sigma(self, pred):
            emul = np.interp(self.data.bin_centers, self.model_centers,
                             self.emul_frac)
            frac = np.sqrt(emul**2 + (2 * fgas_mod.PAINT_BIAS_RESID) ** 2
                           + (2 * fgas_mod.C2S_TRANSFER_SYS) ** 2)
            return np.sqrt(self.data.stat_err[None, :] ** 2
                           + (frac[None, :] * pred) ** 2)

    f5 = Fgas2x.__new__(Fgas2x)
    f5.__dict__.update(fgas.__dict__)
    add("fgas_model2x", f5.loglike(U) - lf_fid,
        "plan v5: fgas 2D-projection/transfer tiers (PAINT_BIAS_RESID, "
        "C2S_TRANSFER_SYS) doubled")

    # -- v5b B coordinate-systematic tier doubled (sequential patch) -----
    sl, of = bblock_mod.SLOPE_SYS, bblock_mod.OFFSET_SYS
    try:
        bblock_mod.SLOPE_SYS, bblock_mod.OFFSET_SYS = 2 * sl, 2 * of
        lb5 = jb.b.loglike_batch(coords, cerr, sigma_pos)
    finally:
        bblock_mod.SLOPE_SYS, bblock_mod.OFFSET_SYS = sl, of
    add("b_coordsys2x", lb5 - lb_fid,
        "B's pre-registered coordinate-systematic tier (SLOPE_SYS, "
        "OFFSET_SYS) doubled")

    # -- fresh-chain variants, if run_a8_variants_disbatch.sh has landed --
    # These replace the reweighted rows above, which collapse (ESS << 500)
    # for every error-WIDENING variant: the widened posterior is broader
    # than the fiducial proposal, so IS has no support where it needs it.
    # The drop-one pair (plan v7) is here rather than reweighted for the
    # SAME structural reason: dropping a likelihood term broadens the
    # posterior, the direction in which IS has no support.
    FRESH = {
        "emul2x": "FRESH CHAINS (4 seeds) — supersedes the reweighted row, "
                  "which failed the ESS gate",
        "fgas_model2x": "FRESH CHAINS (4 seeds) — supersedes the reweighted "
                        "row, which failed the ESS gate",
        "b_coordsys2x": "FRESH CHAINS (4 seeds) — supersedes the reweighted "
                        "row, which failed the ESS gate",
        "no_ksz": "FRESH CHAINS (4 seeds), plan v7 drop-one: kSZ block "
                  "dropped, so f_gas + B alone. Not reweightable — dropping "
                  "a term broadens the posterior",
        "no_fgas": "FRESH CHAINS (4 seeds), plan v7 drop-one: f_gas block "
                   "dropped, so kSZ + B alone. Not reweightable — dropping "
                   "a term broadens the posterior",
        "c2s_massdep": "FRESH CHAINS (4 seeds), systematics-hunt: CylToSph "
                       "made mass-dependent instead of one scalar over all "
                       "five gate bins. Re-points the f_gas mass slope rather "
                       "than widening an error, so the fiducial chain is not "
                       "a valid importance proposal",
        "ksz_wp2bias": "FRESH CHAINS (4 seeds), systematics-hunt: kSZ divided "
                       "by the wp2 painted-vs-truth tau_CAP bias, theta <= "
                       "3.5' only — the conservative inner-radii bracket",
        "ksz_wp2bias_all": "FRESH CHAINS (4 seeds), systematics-hunt: the same "
                           "at all nine radii. The outer-radius bias is "
                           "validated (stack(truth) rises monotonically to "
                           "12.3x the innermost; offsets 67/72 sigma), so this "
                           "is the measurement and the inner-only row is the "
                           "bracket",
    }

    # A variant registered for fitting but missing here would land its chains
    # and then be silently dropped from the movement table -- it is iterated
    # by name, so absence is invisible rather than an error. That happened
    # once (the three systematics-hunt variants above). Fail loudly instead.
    try:
        from analysis.paper3a.scripts.run_joint_ab_fit import VARIANTS
    except ImportError:          # keep the grid runnable in isolation
        VARIANTS = ()
    missing = [v for v in VARIANTS if v not in FRESH]
    if missing:
        raise SystemExit(
            f"run_a8_sysgrid: {missing} are fittable variants with no FRESH "
            "entry. Add a description above, or the rows vanish silently.")

    for v, method in FRESH.items():
        paths = [CHAINS / f"joint_ab_{v}_seed{k}.npz" for k in range(4)]
        if not all(p.exists() for p in paths):
            log(f"  {v:16s} (chains absent — row left unrun)")
            continue
        flat = np.concatenate([np.load(p)["chain"].astype(float)
                               .reshape(-1, 32) for p in paths])
        rng = np.random.default_rng(23)
        Uv = flat[rng.choice(len(flat), min(N_SUB, len(flat)),
                             replace=False)]
        cv, _ = jb.coords(Uv[:, :30])
        h = headline(Uv, cv)
        mv = movement(fid, h)
        table[v] = {**mv, "ess": None, "reliable": True,
                    "method": method,
                    "note": table.get(v, {}).get("note", "")}
        results[v] = h
        log(f"  {v:16s} FRESH  "
            + " ".join(f"{k}={mv[k]:+.2f}" for k in
                       ("dln_mgas", "dln_t", "f_sat", "sigma_pos_arcmin"))
            + ("  ** FLAGGED" if mv["flagged"] else ""))

    # -- v7 drop-one rows from the frozen artifacts ----------------------
    syn = json.loads((Path("/mnt/ceph/users/mlee1/paper3/A/wp6_propagation")
                      / "ab_synthesis.json").read_text())
    _done = [v for v in ("no_ksz", "no_fgas") if v in table]
    table["drop_probes_note"] = {
        "note": "plan v7 (drop-one): the drop-B and kSZ-alone rows are "
                "quoted from the FROZEN subset chains — joint_ab (all "
                "blocks) vs a5 joint (drop B) vs kszonly; medians recorded "
                "in ab_synthesis.json. The two remaining combos are the "
                "fresh-chain rows `no_fgas` (kSZ+B) and `no_ksz` (fgas+B) "
                "in this same table, run by "
                "run_a8_dropone_disbatch.sh; "
                + (f"landed: {', '.join(_done)}." if _done
                   else "NOT YET RUN — those rows are absent above."),
        "rows_landed": _done,
        "ab_synthesis_medians": syn.get("medians", syn),
    }

    # -- deferred / N.A. rows (documented, per the acceptance criterion) --
    table["mass_shift"] = {
        "note": "plan v1 EXECUTED — see wp8_robustness/a8_massshift.json "
                "(run_a8_massshift.py). The sigma was located in the "
                "primary source: Siegel 2509.10455 Table 1 gives the GGL "
                "statistical error (0.009 dex at LRG M3) and Sec 4.2.1 the "
                "stellar-mass-estimator systematic (>~0.1 dex) that "
                "dominates it and is the one used. Result: clean, worst "
                "coordinate movement 0.028 sigma at ESS ~99%."}
    table["cap_radii"] = {
        "note": "plan v4 DEFERRED to a re-fit: alternative CAP radii sets "
                "change the data vector (different dof), outside "
                "importance-reweighting validity."}
    table["stochasticity"] = {
        "note": "plan v6 N/A: wp4 diagonal Sigma_theory confirmed final; "
                "A2 optional single-vs-multi-sample variants closed "
                "(wp2/wp4 REPORT notes)."}
    table["simba_spot"] = {
        "note": "plan v9 optional: SIMBA 1P inputs live on Popeye ceph; "
                "not runnable from rusty."}

    out = {"method": "importance reweighting of the frozen joint_ab chain "
                     f"({N_SUB} subsample of 750k); w = exp(dL) with only "
                     "the modified block contributing",
           "flag_threshold_sigma": FLAG_AT,
           "ess_min": ESS_MIN,
           "movement_table": table,
           "posteriors": results}
    (WP8 / "a8_sysgrid.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {WP8}/a8_sysgrid.json")


if __name__ == "__main__":
    main()
