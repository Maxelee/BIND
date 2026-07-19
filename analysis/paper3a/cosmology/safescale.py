"""WP-A7 tasks 1-2 (FULL PASS, v1): safe-scale tables for C_ell^kk + HOS.

Question answered per (survey, band): down to which scale is the
statistic usable, when the residual feedback uncertainty is
(a) UNINFORMED   — the full SB35 prior envelope of R(ell),
(b) GAS-CALIBRATED (A5)   — the f_gas-posterior envelope,
(c) GAS-CALIBRATED (JOINT A+B) — the joint_ab-posterior envelope
    (all three GP-sigma folded, mu + sigma*eps convention)?

Criterion (plan task 1, DES convention): feedback-induced bias on the
amplitude parameter < 0.3 sigma of its statistical error. For a single
amplitude parameter with a scale-independent response alpha =
d lnC/d ln S8, alpha cancels EXACTLY in bias/sigma:
    b/sigma(<=lmax) = sum_l w_l dlnC_l / sqrt(sum_l w_l),
    w_l = (C_l/(C_l+N_l))^2 (2 l + 1) dl f_sky / 2,
so the table needs no absolute S8 derivative (documented choice; only
the SHAPE of alpha(ell) enters, at second order). dlnC_l is the band's
68% half-width of R/R_fid — suppression bias is one-signed, so the
coherent sum is the realistic accounting, not a worst case.

CHANGES FROM v0 (2026-07-19 first pass), each a v0 defect found on
re-reading; every one is recorded in the wp7 REPORT:

 1. DATASET CONSISTENCY. v0 predicted from the xpkfix dataset (via
    StatsEmulator.load()) but read err/cl_dmo/ell from the PRE-fix
    `emulator_dataset.npz`. Now both come from `statsemu.DATASET`.
    Verified bit-identical for every array this module consumes
    (cl_dmo, a__suppression__ell, t__{peak,minima}_counts__err,
    t__wst__err, t__suppression__value) — so v0's numbers were not
    contaminated, but the provenance was wrong.
 2. CONTINUOUS ell. v0 quantized ell into 30 log bands, which collapsed
    the f_gas and joint-A+B columns into the same lmax. Now the
    cumulative runs per-ell mode (dl from the grid) and lmax_safe is
    the interpolated 0.3-sigma crossing.
 3. ALL-SEED f_gas BAND. v0 drew the f_gas HOS band from seed0 only;
    now all four a5_final seeds, matching the joint band's treatment.
 4. HOS z_eff. v0's HOS table pooled all five source redshifts and
    varied only the survey AREA — so its per-survey rows differed by
    sqrt(area) alone. HOS is now z_s-interpolated to the survey z_eff,
    exactly as the two-point table already was.
 5. HOS sigma: COUNTING NOISE ADDED. sigma^2 = sigma_CV^2 + sigma_count^2
    with sigma_count = sqrt(N) on the survey-area-scaled counts (peaks
    and minima are counts; at survey area the CV term alone understates
    the error in the sparse tails). Shape-noise-INDUCED peaks are still
    NOT captured — that needs the maps, not the summary vectors; it is
    declared as the standing gap below and its sign is stated.
 6. EMULATOR-TRUST GATE — and the retraction of v0's WST headline.
    v0 reported "WST ~0.01 safe everywhere" and read it as WST being
    the most feedback-sensitive statistic in the set. It is not: wst is
    trained on n_scored=40 runs (every other target: 253) and its
    out-of-sample residual is err_rel_med = 8.9 in units of the
    measurement error, versus 0.32 (peaks) and 0.20 (minima). The
    emulator's own prediction error therefore EXCEEDS the survey
    statistical error it is being compared against, by ~5x at HSC-Y3
    — no safe-scale statement can be derived from it in either
    direction. Every HOS row now carries
    `emu_err_over_sigma_survey` = err_rel_med / sqrt(N_REAL *
    PATCH_DEG2 / area), and `emulator_trustworthy` = that ratio < 1.
    Rows failing the gate are reported but MUST NOT be quoted as
    feedback-sensitivity results.
    (A with/without-GP-sigma band comparison is also reported, but it
    is NOT the gate: sparse training corrupts the GP MEAN as well as
    its sigma, so wst passes that test while still being untrustworthy.
    That is why the out-of-sample residual is the criterion.)
 7. HONEST DENOMINATORS. frac_dims_safe is reported alongside
    n_dims_used, because dims with sigma == 0 (empty count bins) are
    excluded and that denominator differs per statistic (peaks 77%,
    minima 35% of dims) — v0's fractions were not comparable across
    statistics.
 8. SCALE MAPPING FOR PEAKS/MINIMA (v0 deferred this). The safe |nu|
    range is now reported: the nu threshold beyond which the cumulative
    amplitude bias breaches 0.3 sigma.
 9. PARAMETER-LEVEL COMPRESSION (plan task 2). Each HOS target is
    compressed onto the PC1 of its own feedback response, giving a
    single b/sigma for that amplitude direction — the alpha-free
    analogue of the two-point number, using the diagonal sigma we have.

STANDING GAP (stated, not papered over): the suite maps carry no shape
noise, so shape-noise-induced peaks/minima — which populate low |nu|
and DILUTE feedback sensitivity — are absent. Our HOS safe fractions
are therefore OPTIMISTIC at low |nu| and the counting term only
partially compensates. Closing it requires re-measuring the HOS on
noise-added maps (a Popeye/GPU item, not a rusty one).

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
from analysis.paper3a.emulator.statsemu import DATASET, WP6, StatsEmulator
from analysis.paper3a.inference.sampler import CHAINS

WP7 = Path("/mnt/ceph/users/mlee1/paper3/A/wp7_cosmology")
ELL_MIN, ELL_MAX = 100.0, 8000.0
N_DRAW = 1500
CRIT = 0.3
PATCH_DEG2 = 25.0          # suite realization footprint
N_REAL = 50                # realizations per run (err columns are SEM)
HOS_TARGETS = ("peak_counts", "minima_counts", "pdf", "mf_v0", "mf_v1",
               "mf_v2", "wst")
COUNT_TARGETS = ("peak_counts", "minima_counts")   # Poisson applies


# --------------------------------------------------------------- bands

def _band_draws(emu, U, target, rng, with_gp_sigma=True):
    """mu + sigma*eps draws (the a6 GP-sigma convention).

    with_gp_sigma=False returns the mu-only spread — used by the
    GP-ignorance diagnostic (change 6).
    """
    if not with_gp_sigma:
        return emu.predict(U, target)
    mu, sd = emu.predict(U, target, return_std=True)
    sd = np.nan_to_num(np.asarray(sd, float), nan=0.0)
    return mu + sd * rng.standard_normal((len(U),) + (1,) * (mu.ndim - 1))


def _joint_unit_draws(n, rng):
    flat = np.concatenate(
        [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, 32) for k in range(4)])[:, :30]
    return flat[rng.choice(len(flat), n, replace=False)]


def _fgas_unit_draws(n, rng):
    """All four a5_final seeds (change 3; v0 used seed0 only)."""
    flat = np.concatenate(
        [np.load(CHAINS / f"a5_final_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, 30) for k in range(4)])
    return flat[rng.choice(len(flat), n, replace=False)]


def band_sources(rng_seed=29):
    """The three parameter ensembles the bands are built from."""
    rng = np.random.default_rng(rng_seed)
    return {
        "prior": rng.uniform(0.0, 1.0, size=(N_DRAW, 30)),
        "fgas_post": _fgas_unit_draws(N_DRAW, np.random.default_rng(31)),
        "joint_post": _joint_unit_draws(N_DRAW, np.random.default_rng(33)),
    }


def load_bands(emu, srcs):
    """Vs-fiducial suppression band per variant: (q16, q50, q84)."""
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    fid = emu.predict(u_fid, "suppression")            # (5 zs, L)
    out = {}
    for i, (name, U) in enumerate(srcs.items()):
        rng = np.random.default_rng(101 + i)
        pred = _band_draws(emu, U, "suppression", rng) / fid[None]
        out[name] = np.nanpercentile(pred, [16, 50, 84], axis=0)
    return out, fid


def zs_interp(arr, zs_grid, z_eff):
    """Linear z_s interpolation along the zs axis (axis -2)."""
    zs = np.asarray(zs_grid, float)
    j = int(np.clip(np.searchsorted(zs, z_eff) - 1, 0, len(zs) - 2))
    t = (z_eff - zs[j]) / (zs[j + 1] - zs[j])
    return (1 - t) * np.take(arr, j, axis=-2) + t * np.take(arr, j + 1,
                                                            axis=-2)


def _crossing(x, y, level):
    """First x where y crosses level, linearly interpolated."""
    over = y >= level
    if not over.any():
        return float(x[-1]), False          # never breaches in range
    k = int(np.argmax(over))
    if k == 0:
        return float(x[0]), True            # breached immediately
    x0, x1, y0, y1 = x[k - 1], x[k], y[k - 1], y[k]
    if y1 == y0:
        return float(x0), True
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0)), True


# ---------------------------------------------------------- two-point

def two_point_table(bands, fid, ell, cl_dmo, zs_grid):
    """Continuous-ell cumulative amplitude bias (change 2)."""
    sel = (ell >= ELL_MIN) & (ell <= ELL_MAX)
    l = ell[sel]
    dl = np.gradient(l)

    res, curves = {}, {}
    for sv in SURVEYS:
        cl = zs_interp(cl_dmo, zs_grid, sv.z_eff)[sel] \
            * zs_interp(fid, zs_grid, sv.z_eff)[sel]
        w = ((cl / (cl + sv.noise_cl)) ** 2 * (2 * l + 1) * dl
             * sv.f_sky / 2.0)
        res[sv.name], curves[sv.name] = {}, {}
        for name, q in bands.items():
            width = 0.5 * np.abs(zs_interp(q[2] - q[0], zs_grid,
                                           sv.z_eff))[sel]
            cw = np.cumsum(w)
            bias = np.cumsum(w * width) / np.sqrt(cw)
            lmax, breached = _crossing(l, bias, CRIT)
            res[sv.name][name] = {
                "lmax_safe": lmax,
                "breaches_in_range": bool(breached),
                "bias_over_sigma_at_8000": float(bias[-1]),
                "sigma_lnA_stat_at_8000_alpha2":
                    float(1.0 / (2.0 * np.sqrt(cw[-1]))),
            }
            curves[sv.name][name] = (l, bias)
    return res, curves


# ---------------------------------------------------------------- HOS

def _hos_sigma(raw, target, sv, zs_grid, value_zs):
    """sigma_stat per dim at the survey's area and z_eff (change 5).

    sigma_CV: suite SEM x sqrt(N_REAL) = per-patch scatter, scaled by
    sqrt(PATCH_DEG2 / area). sigma_count: sqrt(N) Poisson on the
    area-scaled counts, for count-valued targets only.
    """
    sem = np.nanmedian(np.asarray(raw[f"t__{target}__err"], float), axis=0)
    sem_z = zs_interp(sem[None], zs_grid, sv.z_eff)[0]
    area_fac = np.sqrt(PATCH_DEG2 / sv.area_deg2)
    sig_cv = sem_z * np.sqrt(N_REAL) * area_fac

    if target in COUNT_TARGETS:
        n_survey = np.clip(value_zs, 0.0, None) * (sv.area_deg2 / PATCH_DEG2)
        sig_cnt = np.sqrt(n_survey) / (sv.area_deg2 / PATCH_DEG2)
        return np.sqrt(sig_cv ** 2 + sig_cnt ** 2), sig_cv
    return sig_cv, sig_cv


def _pc1_amplitude(dev, sigma, mask):
    """Compress the feedback response onto its PC1 (change 9).

    dev: (n_draw, D) deviations from fiducial. Returns b/sigma for the
    amplitude along the dominant response direction, with the diagonal
    sigma we have.
    """
    d = dev[:, mask]
    s = sigma[mask]
    if d.shape[1] < 2 or not np.isfinite(d).all():
        return float("nan")
    # whiten so PC1 is the best-measured response direction
    dw = d / s[None]
    u, sv_, vt = np.linalg.svd(dw - dw.mean(0, keepdims=True),
                               full_matrices=False)
    v = vt[0]
    amp = dw @ v                                   # already in sigma units
    return float(0.5 * (np.nanpercentile(amp, 84) - np.nanpercentile(amp, 16)))


def hos_table(emu, raw, srcs, zs_grid, val):
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    out = {}
    skipped = {}
    for t in HOS_TARGETS:
        if t not in emu.targets:
            skipped[t] = "not emulated"
            continue
        if f"t__{t}__err" not in raw.files:
            skipped[t] = ("no err column in the dataset — no sigma_stat, "
                          "so no safe-scale statement is possible")
            continue
        fid_all = emu.predict(u_fid, t)                     # (5, D)

        # band widths + the GP-ignorance diagnostic (change 6)
        preds, preds_nogp = {}, {}
        for i, (name, U) in enumerate(srcs.items()):
            preds[name] = _band_draws(emu, U, t, np.random.default_rng(201 + i))
            preds_nogp[name] = _band_draws(emu, U, t, None,
                                           with_gp_sigma=False)

        def _w(p, sv):
            pz = zs_interp(p, zs_grid, sv.z_eff)
            q = np.nanpercentile(pz, [16, 84], axis=0)
            return 0.5 * np.abs(q[1] - q[0]), pz

        v = val.get(t, {})
        err_rel = float(v.get("err_rel_med", np.nan))
        entry = {"n_train_scored": int(v.get("n_scored", -1)),
                 "err_rel_med": err_rel}
        for sv in SURVEYS:
            fid_z = zs_interp(fid_all[None], zs_grid, sv.z_eff)[0]
            sig, sig_cv = _hos_sigma(raw, t, sv, zs_grid, fid_z)
            mask = np.isfinite(sig) & (sig > 0)
            # emulator error vs the survey statistical error (change 6):
            # err_rel_med is in units of the suite SEM; sigma_survey is
            # SEM*sqrt(N_REAL*PATCH/area), so the ratio is this quotient.
            emu_over_sig = err_rel / np.sqrt(N_REAL * PATCH_DEG2
                                             / sv.area_deg2)
            row = {"n_dims_total": int(sig.size),
                   "n_dims_used": int(mask.sum()),
                   "emu_err_over_sigma_survey": float(emu_over_sig),
                   "emulator_trustworthy": bool(emu_over_sig < 1.0)}
            for name in srcs:
                w, pz = _w(preds[name], sv)
                w0, _ = _w(preds_nogp[name], sv)
                with np.errstate(invalid="ignore"):
                    frac = float(np.mean((w < CRIT * sig)[mask])) \
                        if mask.any() else float("nan")
                dev = pz - fid_z[None]
                row[name] = {
                    "frac_dims_safe": frac,
                    "band_width_med": float(np.nanmedian(w[mask])),
                    "band_width_med_no_gp_sigma": float(np.nanmedian(w0[mask])),
                    "gp_dominated": bool(
                        np.nanmedian(w0[mask]) < 0.5 * np.nanmedian(w[mask])),
                    "pc1_bias_over_sigma": _pc1_amplitude(dev, sig, mask),
                }
                # nu-resolved safe range for the count statistics (change 8)
                if t in COUNT_TARGETS:
                    nu = np.asarray(raw[f"a__{t}__nu"], float)
                    r = np.full(sig.size, np.nan)
                    r[mask] = w[mask] / (CRIT * sig[mask])
                    fin = np.isfinite(r)
                    bad = nu[fin][r[fin] > 1.0]
                    row[name]["nu_safe_max"] = (
                        float(np.min(np.abs(bad))) if bad.size else None)
            entry[sv.name] = row
        out[t] = entry
    return out, skipped


# --------------------------------------------------------------- main

def main() -> None:
    emu = StatsEmulator.load()
    raw = np.load(DATASET, allow_pickle=False)          # change 1
    ell = np.asarray(raw["a__suppression__ell"], float)
    cl_dmo = np.asarray(raw["cl_dmo"], float)
    zs_grid = np.asarray(raw["source_redshifts"], float)

    val = json.loads((WP6 / "statsemu_gp_validation.json").read_text())

    srcs = band_sources()
    bands, fid = load_bands(emu, srcs)
    tbl2, curves = two_point_table(bands, fid, ell, cl_dmo, zs_grid)
    tblh, skipped = hos_table(emu, raw, srcs, zs_grid, val)
    out = {
        "version": "v1 full pass (2026-07-19); see module docstring for the "
                   "nine changes from v0",
        "criterion": f"|bias(S8-like amplitude)| < {CRIT} sigma_stat "
                     "(alpha cancels; see module docstring)",
        "ell_range": [ELL_MIN, ELL_MAX],
        "dataset": str(DATASET),
        "n_draw": N_DRAW,
        "surveys": {s.name: {"n_eff": s.n_eff_arcmin2, "sigma_e": s.sigma_e,
                             "area_deg2": s.area_deg2, "z_eff": s.z_eff,
                             "provenance": s.provenance} for s in SURVEYS},
        "two_point_clkk": tbl2,
        "hos": tblh,
        "hos_skipped": skipped,
        "hos_n_train_scored": {t: val[t]["n_scored"] for t in HOS_TARGETS
                               if t in val},
        "standing_gap": "suite maps carry no shape noise; shape-noise-induced "
                        "peaks/minima populate low |nu| and dilute feedback "
                        "sensitivity, so HOS safe fractions are OPTIMISTIC "
                        "there. Counting noise is included; noise-added-map "
                        "re-measurement is a Popeye/GPU item.",
    }
    WP7.mkdir(exist_ok=True)
    (WP7 / "safescale_tables.json").write_text(json.dumps(out, indent=2))

    print("=== C_ell^kk safe scales (continuous ell) ===")
    for sv in SURVEYS:
        r = tbl2[sv.name]
        print(f"{sv.name:9s} prior {r['prior']['lmax_safe']:7.1f} | "
              f"fgas {r['fgas_post']['lmax_safe']:7.1f} | "
              f"joint {r['joint_post']['lmax_safe']:7.1f} | "
              f"b/sig@8000 {r['prior']['bias_over_sigma_at_8000']:6.2f}"
              f" -> {r['joint_post']['bias_over_sigma_at_8000']:6.2f}")
    print("\n=== HOS @ HSC-Y3 (TRUST gate first: emu err vs survey sigma) ===")
    for t, e in tblh.items():
        hs = e["HSC-Y3"]
        ok = hs["emulator_trustworthy"]
        tag = "OK " if ok else "UNTRUSTWORTHY"
        print(f"{t:15s} n_train={e['n_train_scored']:>4} "
              f"err_rel={e['err_rel_med']:6.2f} "
              f"emu/sig={hs['emu_err_over_sigma_survey']:6.2f} [{tag}] "
              f"dims={hs['n_dims_used']}/{hs['n_dims_total']}")
        print(f"{'':15s}   frac_safe {hs['prior']['frac_dims_safe']:.2f} -> "
              f"{hs['joint_post']['frac_dims_safe']:.2f}   "
              f"PC1 b/sig {hs['prior']['pc1_bias_over_sigma']:.2f} -> "
              f"{hs['joint_post']['pc1_bias_over_sigma']:.2f}"
              + ("" if ok else "   <-- DO NOT QUOTE"))
    for t, why in skipped.items():
        print(f"{t:15s} SKIPPED: {why}")

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
    print(f"\nwrote {WP7}/safescale_tables.json + figures/safescale_2pt.png")


if __name__ == "__main__":
    main()
