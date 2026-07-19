"""WP-A7 tasks 1-2: safe-scale tables for C_ell^kk + the HOS v0 pass.

Question answered per (survey, band): down to which ell is the
statistic usable, when the residual feedback uncertainty is
(a) UNINFORMED   — the full SB35 prior envelope of R(ell),
(b) GAS-CALIBRATED (A5)   — the f_gas-posterior envelope (GP-sigma
    folded, a6_posterior_stats.npz),
(c) GAS-CALIBRATED (JOINT A+B) — the joint_ab-posterior envelope
    (computed here, same mu+sigma*eps convention, seed 31)?

Criterion (plan task 1, DES convention): feedback-induced bias on the
amplitude parameter < 0.3 sigma of its statistical error. For a single
amplitude parameter with a scale-independent response alpha =
d lnC/d ln S8, alpha cancels EXACTLY in bias/sigma:
    b/sigma(<=lmax) = sum_b w_b dlnC_b / sqrt(sum_b w_b),
    w_b = (C_b/(C_b+N_b))^2 (2 l_b + 1) dl_b f_sky / 2,
so the table needs no absolute S8 derivative (documented v0 choice;
only the SHAPE of alpha(ell) enters at second order). dlnC_b is the
band's 68% half-width of R/R_fid — suppression bias is one-signed, so
the coherent sum is the realistic accounting, not a worst case.

The C_ell signal is the suite's measured DMO C_ell x the fiducial
suppression, z_s-interpolated to the survey's z_eff. ell restricted to
[100, 8000] (the validated envelope range).

HOS v0 (peaks / minima / wst): statistic-level criterion per dim —
band half-width < 0.3 x sigma_stat(survey), with sigma_stat scaled
from the suite's per-realization scatter (SEM x sqrt(50) per 25 deg^2
patch) to the survey area. COSMIC-VARIANCE ONLY (no shape noise in the
suite maps): sigma is underestimated, the criterion is therefore
CONSERVATIVE for declaring dims safe. Reported as safe-dim fractions,
not scales (v0; scale mapping deferred).

Run: python -m analysis.paper3a.cosmology.safescale
Out: wp7_cosmology/safescale_tables.json + figures/safescale_2pt.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.paper3a.cosmology.surveys import SURVEYS
from analysis.paper3a.emulator import params_meta as pm
from analysis.paper3a.emulator.statsemu import WP6, StatsEmulator
from analysis.paper3a.inference.sampler import CHAINS

WP7 = Path("/mnt/ceph/users/mlee1/paper3/A/wp7_cosmology")
ELL_MIN, ELL_MAX = 100.0, 8000.0
N_DRAW = 1500
CRIT = 0.3
HOS_TARGETS = ("peak_counts", "minima_counts", "pdf", "mf_v0", "mf_v1",
               "mf_v2", "wst")


def _band_draws(emu, U, target, rng):
    """mu + sigma*eps draws (the a6 GP-sigma convention)."""
    mu, sd = emu.predict(U, target, return_std=True)
    sd = np.nan_to_num(np.asarray(sd, float), nan=0.0)
    return mu + sd * rng.standard_normal((len(U),) + (1,) * (mu.ndim - 1))


def load_bands(emu):
    """Vs-fiducial suppression band per variant: (q16, q50, q84)."""
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    fid = emu.predict(u_fid, "suppression")            # (5 zs, L)
    out = {}

    rng = np.random.default_rng(29)
    U_prior = rng.uniform(0.0, 1.0, size=(N_DRAW, 30))
    pr = _band_draws(emu, U_prior, "suppression", rng) / fid[None]
    out["prior"] = np.nanpercentile(pr, [16, 50, 84], axis=0)

    q = np.load(WP6 / "a6_posterior_stats.npz")["suppression__q"].astype(float)
    out["fgas_post"] = q[1:4] / fid[None]              # (3, 5 zs, L)

    rng2 = np.random.default_rng(31)
    flat = np.concatenate(
        [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, 32) for k in range(4)])[:, :30]
    U_j = flat[rng2.choice(len(flat), N_DRAW, replace=False)]
    jj = _band_draws(emu, U_j, "suppression", rng2) / fid[None]
    out["joint_post"] = np.nanpercentile(jj, [16, 50, 84], axis=0)
    return out, fid


def zs_interp(arr, zs_grid, z_eff):
    """Linear z_s interpolation along the zs axis (axis -2)."""
    zs = np.asarray(zs_grid, float)
    j = int(np.clip(np.searchsorted(zs, z_eff) - 1, 0, len(zs) - 2))
    t = (z_eff - zs[j]) / (zs[j + 1] - zs[j])
    return (1 - t) * np.take(arr, j, axis=-2) + t * np.take(arr, j + 1,
                                                            axis=-2)


def two_point_table(bands, fid, ell, cl_dmo, zs_grid):
    sel = (ell >= ELL_MIN) & (ell <= ELL_MAX)
    l = ell[sel]
    edges = np.geomspace(ELL_MIN, ELL_MAX, 31)
    ib = np.clip(np.digitize(l, edges) - 1, 0, 29)
    dl = np.diff(edges)

    res = {}
    curves = {}
    for sv in SURVEYS:
        cl = zs_interp(cl_dmo, zs_grid, sv.z_eff)[sel] \
            * zs_interp(fid, zs_grid, sv.z_eff)[sel]
        res[sv.name] = {}
        curves[sv.name] = {}
        for name, q in bands.items():
            width = 0.5 * np.abs(zs_interp(q[2] - q[0], zs_grid,
                                           sv.z_eff))[sel]
            # band-average
            lb, cb, nb, wb, db = [], [], [], [], []
            for b in range(30):
                m = ib == b
                if not m.any():
                    continue
                lb.append(l[m].mean())
                cb.append(cl[m].mean())
                db.append(width[m].mean())
                wb.append(((cl[m].mean()
                            / (cl[m].mean() + sv.noise_cl)) ** 2
                           * (2 * l[m].mean() + 1) * dl[b] * sv.f_sky
                           / 2.0))
            lb, cb, db, wb = map(np.asarray, (lb, cb, db, wb))
            bias = np.cumsum(wb * db) / np.sqrt(np.cumsum(wb))
            safe = bias < CRIT
            lmax_safe = float(lb[safe][-1]) if safe.any() else float("nan")
            # first crossing rules (monotone growth is typical, not
            # guaranteed): take the last lmax before the FIRST breach
            if (~safe).any():
                k = int(np.argmax(~safe))
                lmax_safe = float(lb[k - 1]) if k > 0 else float("nan")
            res[sv.name][name] = {
                "lmax_safe": lmax_safe,
                "bias_over_sigma_at_8000": float(bias[-1]),
                "sigma_lnA_stat_at_8000_alpha2":
                    float(1.0 / (2.0 * np.sqrt(np.cumsum(wb)[-1]))),
            }
            curves[sv.name][name] = (lb, bias)
    return res, curves


def hos_table(emu, bands_src, raw):
    """Safe-dim fractions per HOS target x survey x band (v0)."""
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    out = {}
    for t in HOS_TARGETS:
        if t not in emu.targets:
            continue
        fid = emu.predict(u_fid, t)
        errk = f"t__{t}__err"
        if errk not in raw.files:
            continue
        sem = np.nanmedian(np.asarray(raw[errk], float), axis=0)
        sig1 = sem * np.sqrt(50.0)                     # per 25 deg^2 real
        widths = {}
        for name, U in bands_src.items():
            rng = np.random.default_rng(37)
            pred = _band_draws(emu, U, t, rng)
            qq = np.nanpercentile(pred, [16, 84], axis=0)
            widths[name] = 0.5 * np.abs(qq[1] - qq[0])
        out[t] = {}
        for sv in SURVEYS:
            scale = np.sqrt(25.0 / sv.area_deg2)
            sig = sig1 * scale
            out[t][sv.name] = {
                name: {"frac_dims_safe":
                       float(np.mean((w < CRIT * sig)[np.isfinite(sig)
                                                      & (sig > 0)]))}
                for name, w in widths.items()}
    return out


def main() -> None:
    emu = StatsEmulator.load()
    raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz",
                  allow_pickle=False)
    ell = np.asarray(raw["a__suppression__ell"], float)
    cl_dmo = np.asarray(raw["cl_dmo"], float)
    zs_grid = np.asarray(raw["source_redshifts"], float)

    bands, fid = load_bands(emu)
    tbl2, curves = two_point_table(bands, fid, ell, cl_dmo, zs_grid)

    rng = np.random.default_rng(29)
    U_prior = rng.uniform(0.0, 1.0, size=(N_DRAW, 30))
    rng2 = np.random.default_rng(31)
    flat = np.concatenate(
        [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, 32) for k in range(4)])[:, :30]
    post = np.load(CHAINS / "a5_final_seed0.npz")["chain"].astype(
        float).reshape(-1, 30)
    bands_src = {
        "prior": U_prior,
        "fgas_post": post[rng2.choice(len(post), N_DRAW, replace=False)],
        "joint_post": flat[rng2.choice(len(flat), N_DRAW, replace=False)],
    }
    tblh = hos_table(emu, bands_src, raw)

    out = {"criterion": f"|bias(S8-like amplitude)| < {CRIT} sigma_stat "
                        "(alpha cancels; see module docstring)",
           "ell_range": [ELL_MIN, ELL_MAX],
           "surveys": {s.name: {"n_eff": s.n_eff_arcmin2,
                                "sigma_e": s.sigma_e,
                                "area_deg2": s.area_deg2,
                                "z_eff": s.z_eff,
                                "provenance": s.provenance}
                       for s in SURVEYS},
           "two_point_clkk": tbl2,
           "hos_v0_frac_dims_safe": tblh,
           "hos_v0_note": "cosmic-variance-only sigma (no shape noise "
                          "in the suite maps) -> conservative; "
                          "fgas_post band here uses seed0 chain only "
                          "(v0)"}
    WP7.mkdir(exist_ok=True)
    (WP7 / "safescale_tables.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["two_point_clkk"], indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    cols = {"prior": "#999999", "fgas_post": "#0072B2",
            "joint_post": "#D55E00"}
    for ax, sv in zip(axes, SURVEYS):
        for name, (lb, bias) in curves[sv.name].items():
            ax.plot(lb, bias, color=cols[name], label=name)
        ax.axhline(CRIT, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\ell_{\rm max}$")
        ax.set_title(sv.name, fontsize=10)
    axes[0].set_ylabel(r"$|b(A)|/\sigma(A)$ cumulative")
    axes[0].legend(fontsize=8)
    fig.suptitle("WP-A7 safe scales: amplitude bias vs statistical error "
                 "(dashed: the 0.3σ criterion)", fontsize=11)
    fig.tight_layout()
    (WP7 / "figures").mkdir(exist_ok=True)
    fig.savefig(WP7 / "figures" / "safescale_2pt.png", dpi=150)
    print(f"wrote {WP7}/safescale_tables.json + figures/safescale_2pt.png")


if __name__ == "__main__":
    main()
