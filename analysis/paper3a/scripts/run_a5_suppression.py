"""WP-A5 task 6: weak-lensing suppression prediction implied by the A5
posterior.

The A5 posterior (`inference/fgas.py`, `a5_final_seed{0-3}.npz`) is a
distribution over the 30 CAMELS-SB35 astro parameters, conditioned ONLY on
the eRASS1 f_gas(M500) data vector. This script asks a different question of
the SAME posterior samples: if these are the feedback prescriptions favored
by the group-scale gas data, what weak-lensing power-spectrum suppression do
they imply relative to the TNG fiducial baryon prescription?

The WL convergence power spectrum Cl(theta) is predicted by
`bind.wlemu.WLEmulator` (branch feature/wl-emu, READ-ONLY import from
/mnt/home/mlee1/vdm_bind2/src — a different checkout of the `bind` package
than the one this worktree's `analysis.paper3a` code lives beside). Both the
posterior chains and the emulator's parameter metadata share the same SB35
astro parameter ordering (verified below at runtime, not just assumed).

R(ell) = Cl(theta_posterior) / Cl(theta_fiducial) is the baryonic
suppression factor relative to the TNG fiducial *baryonified* prediction
(NOT gravity-only — the emulator only predicts baryonified statistics, see
CONTEXT in the task spec).

Run:  python analysis/paper3a/scripts/run_a5_suppression.py
Out:  wp5_chains/a5_suppression.json
      wp5_chains/figures/a5_suppression_pub.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]           # vdm_bind2-paper3a
sys.path.insert(0, str(_repo))
sys.path.insert(0, "/mnt/home/mlee1/vdm_bind2/src")    # READ-ONLY: feature/wl-emu

import matplotlib

matplotlib.use("Agg")

from bind.wlemu import WLEmulator  # noqa: E402

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402
from analysis.paper3a.style import COL, apply, condition_tag, fig_single, refline  # noqa: E402

Z_SOURCE = 1.0
N_THIN = 2000
ELL_TARGETS = [500, 1000, 1330, 2000, 5000]  # ell ~= k*chi(z=0.5), chi(0.5) ~= 1330 Mpc/h at k=1 h/Mpc
RNG_SEED = 0

OUT_JSON = CHAINS / "a5_suppression.json"
OUT_FIG_DIR = CHAINS / "figures"

CAVEAT = (
    "The A5 posterior conditions on eRASS1 f_gas(M500) ONLY (no WL data in "
    "the likelihood) and is edge-piled at the strong-AGN/IMF prior boundary "
    "(R1 boundary diagnostic fires: RadioFeedbackReiorientationFactor 21%, "
    "IMFslope 21%, QuasarThresholdPower 16% of posterior mass within 5% of "
    "their high edges; see REPORT.md). This R(ell) is therefore a "
    "CONDITIONAL prediction -- 'if the group-gas-favored feedback family is "
    "correct, this is the WL suppression it implies' -- NOT the paper's "
    "final joint-probe WL suppression number."
)


def load_posterior_samples(n_thin: int, rng_seed: int) -> np.ndarray:
    """Concatenate the 4 final A5 chains, flatten, and subsample to ~n_thin
    unit-cube astro parameter vectors (30,)."""
    chains = [np.load(CHAINS / f"a5_final_seed{k}.npz")["chain"].astype(float)
              for k in range(4)]
    flat = np.concatenate([c.reshape(-1, 30) for c in chains])
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(flat), size=min(n_thin, len(flat)), replace=False)
    return flat[idx], len(flat)


def align_params(emu: WLEmulator) -> None:
    """Verify (or fix) that the emulator's parameter ordering matches
    params_meta.ASTRO_NAMES -- the chains are stored in the params_meta
    order, and predict() assumes columns line up 1:1 with emu.param_names."""
    if emu.param_names == pm.ASTRO_NAMES:
        return
    raise AssertionError(
        "wlemu param_names != params_meta.ASTRO_NAMES -- chain columns need "
        f"remapping before prediction.\nemu:  {emu.param_names}\npm:   {pm.ASTRO_NAMES}"
    )


def main() -> None:
    apply()

    emu = WLEmulator.load()
    align_params(emu)
    assert Z_SOURCE in emu.source_redshifts or np.min(np.abs(emu.source_redshifts - Z_SOURCE)) < 1e-3, \
        f"z_source={Z_SOURCE} not among emulated planes {emu.source_redshifts.tolist()}"

    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    u_post, n_total = load_posterior_samples(N_THIN, RNG_SEED)
    n_used = len(u_post)

    ell = emu.ell
    Cl_fid = emu.predict(u_fid, z_source=Z_SOURCE, return_std=False)["Cl"]           # (n_ell,)
    Cl_post = emu.predict(u_post, z_source=Z_SOURCE, return_std=False)["Cl"]         # (N, n_ell)
    R = Cl_post / Cl_fid[None, :]

    # -------------------------------------------------------- sanity checks --
    Cl_fid_self = emu.predict(u_fid[None, :], z_source=Z_SOURCE, return_std=False)["Cl"][0]
    R_fid_self = Cl_fid_self / Cl_fid
    check_a_pass = bool(np.allclose(R_fid_self, 1.0, atol=1e-12))

    med = np.median(R, axis=0)
    hi_ell_mask = ell > 1000
    check_b_pass = bool(np.all(med[hi_ell_mask] < 1.0))

    # ------------------------------------------------------------- summary --
    p16, p84 = np.percentile(R, [16, 84], axis=0)
    p025, p975 = np.percentile(R, [2.5, 97.5], axis=0)

    def at_target(ell_target: float) -> dict:
        i = int(np.argmin(np.abs(ell - ell_target)))
        return {
            "ell_target": ell_target,
            "ell_used": float(ell[i]),
            "R_median": float(med[i]),
            "R_16": float(p16[i]),
            "R_84": float(p84[i]),
            "R_2p5": float(p025[i]),
            "R_97p5": float(p975[i]),
        }

    targets = {str(t): at_target(t) for t in ELL_TARGETS}

    result = {
        "task": "WP-A5 task 6 -- WL suppression prediction implied by the A5 posterior",
        "z_source": Z_SOURCE,
        "n_posterior_total": int(n_total),
        "n_posterior_used": int(n_used),
        "rng_seed": RNG_SEED,
        "ell": ell.tolist(),
        "R_median": med.tolist(),
        "R_16": p16.tolist(),
        "R_84": p84.tolist(),
        "R_2p5": p025.tolist(),
        "R_97p5": p975.tolist(),
        "targets": targets,
        "sanity_checks": {
            "a_fiducial_self_ratio_is_one": {
                "pass": check_a_pass,
                "max_abs_dev_from_1": float(np.max(np.abs(R_fid_self - 1.0))),
            },
            "b_posterior_median_suppressed_above_ell1000": {
                "pass": check_b_pass,
                "median_R_at_ell_gt_1000": med[hi_ell_mask].tolist(),
                "ell_gt_1000": ell[hi_ell_mask].tolist(),
            },
        },
        "caveat": CAVEAT,
        "provenance": {
            "chains": [str(CHAINS / f"a5_final_seed{k}.npz") for k in range(4)],
            "wlemu_artifact_provenance": emu.provenance,
            "wlemu_repo": "/mnt/home/mlee1/vdm_bind2 (branch feature/wl-emu, read-only)",
            "paper3a_repo": "/mnt/home/mlee1/vdm_bind2-paper3a (branch analysis/paper3a-gas-calibration)",
        },
    }

    if not check_a_pass:
        raise RuntimeError("Sanity check (a) FAILED: R at fiducial sample != 1. "
                            "Do not trust downstream numbers -- fix before proceeding.")
    if not check_b_pass:
        print("WARNING: sanity check (b) FAILED -- posterior-median R is NOT < 1 "
              "at all ell > 1000. Investigate the parameter mapping (unit-cube "
              "convention, log-flag alignment, column order) before drawing "
              "physics conclusions. Proceeding to write outputs for inspection.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT_JSON}")

    for t in ELL_TARGETS:
        r = targets[str(t)]
        print(f"ell~{t:5d} (used {r['ell_used']:8.1f}): "
              f"R = {r['R_median']:.4f}  [{r['R_16']:.4f}, {r['R_84']:.4f}] 68%  "
              f"[{r['R_2p5']:.4f}, {r['R_97p5']:.4f}] 95%")
    print(f"sanity (a) fiducial self-ratio == 1: {'PASS' if check_a_pass else 'FAIL'}")
    print(f"sanity (b) median R < 1 for ell>1000: {'PASS' if check_b_pass else 'FAIL'}")

    # ------------------------------------------------------------- figure --
    fig, ax = fig_single()
    ax.fill_between(ell, p025, p975, color=COL["model"], alpha=0.12, lw=0, label="posterior 95%")
    ax.fill_between(ell, p16, p84, color=COL["model"], alpha=0.30, lw=0, label="posterior 68%")
    ax.plot(ell, med, color=COL["model"], lw=1.8, label="posterior median")

    ymin = min(float(np.min(p025)), 1.0)
    ymax = max(float(np.max(p975)), 1.0)
    pad = 0.10 * (ymax - ymin)
    ax.set_ylim(ymin - 0.35 * pad, ymax + pad)

    refline(ax, y=1.0, label="TNG fiducial")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell\,/\,C_\ell^{\rm TNG-fid}$")
    condition_tag(ax, r"$z_s = 1.0$")
    ax.legend(loc="lower left", fontsize=6.5, frameon=False)
    fig.text(0.5, -0.06, "A5 posterior: f_gas-only, edge-piled (conditional prediction)",
              ha="center", va="top", fontsize=6.5, color="#666666", style="italic")

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG_DIR / f"a5_suppression_pub.{ext}")
    print(f"wrote {OUT_FIG_DIR}/a5_suppression_pub.pdf/.png")


if __name__ == "__main__":
    main()
