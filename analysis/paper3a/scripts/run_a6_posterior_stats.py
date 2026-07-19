"""WP-A6 execution: push the A5 FINAL posterior through the statistics
emulator (the A6 decision's no-new-lightcones path).

For every emulated target: predict at a thinned A5 posterior sample (house
convention: 2000 draws, same as `run_a5_suppression.py`) and at the TNG
fiducial, and store the posterior-predictive envelope (median / 68 / 95 per
statistic dimension) plus the fiducial reference. The suppression target is
additionally cross-checked against the INDEPENDENT wlemu-based prediction
(`wp5_chains/a5_suppression.json`, WP-A5 task 6): two separately trained
emulators (wlemu: kappa-map C_ell ratios; statsemu: the Popeye sb35_stats
measurement pass) pushed through the same posterior — their z_s = 1.0
R(ell) envelopes agreeing is the A6 validation the plan's "wlemu-vs-direct
spot-check" was scoped to provide, obtained without any new generation.

--with-gp-sigma (deviation follow-up, 2026-07-18): fold the statsemu GP
predictive std into the envelope — each posterior sample's prediction is
drawn as mu + sigma*eps (one standard-normal eps per sample, fixed seed)
before the quantiles, so the envelope carries emulator uncertainty in
quadrature with the posterior spread instead of being mean-only.

Run:  python analysis/paper3a/scripts/run_a6_posterior_stats.py
Out:  wp6_propagation/a6_posterior_stats.npz  (envelopes per target)
      wp6_propagation/a6_posterior_stats_summary.json
      wp6_propagation/figures/a6_suppression_envelope.png
      wp6_propagation/figures/a6_cl_kappa_y_envelope.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.emulator.statsemu import WP6, StatsEmulator  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402

N_POST = 2000
SEED = 11


def load_posterior_thin(n: int = N_POST) -> np.ndarray:
    chains = []
    for k in range(4):
        f = np.load(CHAINS / f"a5_final_seed{k}.npz", allow_pickle=False)
        chains.append(f["chain"].astype(np.float64).reshape(-1, 30))
    flat = np.concatenate(chains)
    rng = np.random.default_rng(SEED)
    return flat[rng.choice(len(flat), size=n, replace=False)]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--with-gp-sigma", action="store_true",
                    help="fold GP predictive std into the envelope "
                         "(mu + sigma*eps per posterior sample)")
    args = ap.parse_args()

    emu = StatsEmulator.load()
    post = load_posterior_thin()
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)

    arrays = {"source_redshifts": emu.source_redshifts,
              "n_posterior": np.array(N_POST)}
    summary = {"n_posterior": N_POST, "targets": {},
               "gp_sigma": bool(args.with_gp_sigma),
               "gp_sigma_note": ("envelopes drawn as mu + sigma*eps per "
                                 "posterior sample (seed 17)"
                                 if args.with_gp_sigma else
                                 "mean-only per sample")}
    rng = np.random.default_rng(17)
    for t in emu.targets:
        if args.with_gp_sigma:
            pred, sd = emu.predict(post, t, return_std=True)
            sd = np.nan_to_num(np.asarray(sd, float), nan=0.0)
            pred = pred + sd * rng.standard_normal((len(post),)
                                                   + (1,) * (pred.ndim - 1))
        else:
            pred = emu.predict(post, t)                 # (N, *shape)
        fid = emu.predict(u_fid, t)
        qs = np.nanpercentile(pred, [2.5, 16, 50, 84, 97.5], axis=0)
        arrays[f"{t}__q"] = qs.astype(np.float32)       # (5, *shape)
        arrays[f"{t}__fid"] = fid.astype(np.float32)
        summary["targets"][t] = {"shape": list(pred.shape[1:])}
        print(f"[a6] {t}: envelope {qs.shape}", flush=True)
        del pred

    # ---- wlemu cross-check on suppression at z_s = 1.0 ---------------------
    supp_json = CHAINS / "a5_suppression.json"
    if supp_json.exists() and "suppression" in emu.targets:
        wl = json.loads(supp_json.read_text())
        zs = list(emu.source_redshifts)
        iz = zs.index(1.0) if 1.0 in zs else int(np.argmin(np.abs(np.array(zs) - 1.0)))
        raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz",
                      allow_pickle=False)
        ell = np.asarray(raw["a__suppression__ell"], float)
        q = arrays["suppression__q"].astype(float)      # (5q, 5z, 724)
        # DEFINITION MATCH: wlemu's R is Cl(theta)/Cl(theta_fid); the
        # statsemu `suppression` target is Cl^painted/Cl^dmo. Divide the
        # statsemu envelope by its own fiducial prediction (ratio of
        # ratios — the DMO reference cancels) before comparing.
        fid = arrays["suppression__fid"].astype(float)  # (5z, 724)
        rows = []
        for le, rm, r16, r84 in zip(wl["ell"], wl["R_median"],
                                    wl["R_16"], wl["R_84"]):
            j = int(np.argmin(np.abs(ell - float(le))))
            rows.append({
                "ell_wlemu": float(le),
                "ell_statsemu": float(ell[j]),
                "wlemu_median": float(rm),
                "wlemu_68": [float(r16), float(r84)],
                "statsemu_median": float(q[2, iz, j] / fid[iz, j]),
                "statsemu_68": [float(q[1, iz, j] / fid[iz, j]),
                                float(q[3, iz, j] / fid[iz, j])],
            })
        summary["wlemu_crosscheck_zs1"] = rows
        summary["wlemu_crosscheck_note"] = (
            "statsemu values are re-normalized to its own TNG-fiducial "
            "prediction (ratio of ratios) to match wlemu's R definition; "
            "the raw statsemu suppression target is painted/DMO.")

    out = WP6 / "a6_posterior_stats.npz"
    np.savez_compressed(out, **arrays)
    (WP6 / "a6_posterior_stats_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")

    # ---- figures -----------------------------------------------------------
    figdir = WP6 / "figures"
    figdir.mkdir(exist_ok=True)
    raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz", allow_pickle=False)

    ell = np.asarray(raw["a__suppression__ell"], float)
    q = arrays["suppression__q"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for iz, z in enumerate(emu.source_redshifts):
        ax.plot(ell, q[2, iz], lw=1.4, label=f"$z_s$={z:g}")
        ax.fill_between(ell, q[1, iz], q[3, iz], alpha=0.25)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$R(\ell) = C_\ell^{\kappa\kappa}/C_\ell^{\rm dmo}$")
    ax.set_title("A5-posterior suppression envelope (median + 68%)")
    ax.legend(fontsize=7)

    ky = arrays["cl_kappa_y__q"].astype(float)
    ky_ell = np.asarray(raw["a__cl_kappa_y__ell"], float)
    ax = axes[1]
    for iz, z in enumerate(emu.source_redshifts):
        ax.plot(ky_ell, ky[2, iz] * ky_ell * (ky_ell + 1) / (2 * np.pi),
                lw=1.4, label=f"$z_s$={z:g}")
        ax.fill_between(ky_ell, ky[1, iz] * ky_ell * (ky_ell + 1) / (2 * np.pi),
                        ky[3, iz] * ky_ell * (ky_ell + 1) / (2 * np.pi), alpha=0.25)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1) C_\ell^{\kappa y} / 2\pi$")
    ax.set_title(r"A5-posterior $\kappa \times y$ envelope")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "a6_suppression_envelope.png", dpi=150)
    print(f"wrote {figdir}/a6_suppression_envelope.png")


if __name__ == "__main__":
    main()
