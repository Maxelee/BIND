"""WP-A7 task 3: the S8 exercise.

How much of a DES-Y6-style S8 offset does gas-calibrated feedback
absorb, and what does calibration buy relative to cutting scales?

SETUP. Mock C_ell^kk data at the fiducial cosmology with feedback
INJECTED at the strong side of the gas-calibrated posterior, then
analysed three ways:

  (1) UNCORRECTED   — no feedback model (R_model = 1), all scales.
      This is the "ignore feedback" analysis; it absorbs the missing
      suppression into the amplitude and biases S8 LOW.
  (2) CALIBRATED    — R_model = the posterior-median suppression, with
      the residual posterior band width added to the covariance as a
      correlated systematic. All scales kept.
  (3) SCALE-CUT     — R_model = 1 but restricted to ell <= lmax_safe
      (the uninformed/prior-band safe scale from task 1). This is the
      DES-style mitigation that calibration is competing against.

PARAMETERISATION. A single amplitude A with C_ell propto A^2, i.e.
A propto S8 at fixed shape — the same alpha = 2 convention already
adopted in safescale.py (`sigma_lnA_stat_at_8000_alpha2`). The model is
linear in p = A^2, so each analysis is an exact GLS solve, no sampler:

    p_hat = (m^T C^-1 d) / (m^T C^-1 m),  sigma_p = (m^T C^-1 m)^-1/2
    m = C_fid * R_model,  d = C_fid * R_true * A_true^2 (+ noise)

with a Gaussian band-power covariance
    C_ll = 2 (C_l + N_l)^2 / ((2l+1) dl f_sky)   (diagonal)
plus, for analysis (2), the calibration systematic
    C_sys = (C_fid * A^2)_l (C_fid * A^2)_l' rho_ll' w_l w_l'
with w = the posterior band half-width and rho = 1 (fully correlated:
suppression is one-signed and coherent in ell — the conservative and
physically correct choice, matching the coherent-sum accounting in
task 1).

We report S8 shifts as Delta_S8 / S8 = A_hat - 1 (exact at alpha = 2).
The absolute S8 scale enters only when converting to Delta_S8, done
with S8_fid = 0.83 and stated as such.

PEAKS. The plan also asks how much of the peak S/N a calibrated
analysis wins back relative to a scale cut. We deliberately do NOT
manufacture a d(peaks)/dS8: the suite is fixed-cosmology, so no
internal peak S8 response exists, and inventing one would be the kind
of remembered-number error the house rules exist to catch. Instead we
report the precision on the peak vector's OWN feedback-response
amplitude (the PC1 direction of safescale.py change 9), comparing the
nu-cut analysis with the band-inflated full-vector analysis. That is a
well-posed information-retention ratio, and it is labelled as such
rather than as a cosmological S8 S/N.

ACCEPTANCE (plan): a zero-feedback mock must recover zero shift in all
three analyses. Enforced in `null_test()`; main() refuses to write
results if it fails.

TWO CAVEATS THAT MUST TRAVEL WITH THESE NUMBERS.

(i) The injected truth is the joint-A+B posterior's 16th percentile —
the STRONG EDGE of the calibrated band, not its centre. Injecting at
the median would drive the calibrated bias to ~0 by construction and
would not be a test. The residual calibrated bias reported here is
therefore the ~1-sigma-unlucky case, and the scale-cut comparison is
correspondingly the honest one: an uninformed analyst sets lmax from
the width of their PRIOR band, which under-protects them precisely
when the truth sits at that band's edge. That is why the cut retains
MORE residual bias than the calibration despite discarding scales.

(ii) The joint A+B posterior is the best-fitting region of a model
this paper REJECTS (p_pp = 0.000, every block rejected at the MAP).
The band used here is "the range TNG feedback can be calibrated to",
not "the true feedback of the Universe". These numbers quantify what
gas calibration buys a WL analyst working within the TNG family; they
are not a claim that the injected suppression is correct. Any quote of
the Delta_S8 absorption must carry this.

Run: python -m analysis.paper3a.cosmology.s8exercise
Out: wp7_cosmology/s8_exercise.json + figures/s8_exercise.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.paper3a.cosmology.safescale import (
    CRIT, ELL_MAX, ELL_MIN, N_DRAW, N_REAL, PATCH_DEG2, WP7,
    _hos_sigma, _pc1_amplitude, band_sources, load_bands, zs_interp,
)
from analysis.paper3a.cosmology.surveys import SURVEYS
from analysis.paper3a.emulator import params_meta as pm
from analysis.paper3a.emulator.statsemu import DATASET, StatsEmulator

S8_FID = 0.83          # only converts fractional shifts to Delta_S8
RHO_SYS = 1.0          # calibration band correlated across ell


def gls(d, m, cov):
    """Exact GLS for p in d = m p; returns (p_hat, sigma_p)."""
    ci = np.linalg.inv(cov)
    denom = float(m @ ci @ m)
    return float(m @ ci @ d) / denom, float(denom ** -0.5)


def analyse(cl_fid, r_true, r_model, band_w, sv, ell, dl, use=None,
            with_sys=False):
    """One analysis. Returns A_hat, sigma_A, and the S/N of A."""
    sel = np.ones(len(ell), bool) if use is None else use
    c, rt, rm = cl_fid[sel], r_true[sel], r_model[sel]
    l, d_l = ell[sel], dl[sel]

    signal = c * rt                       # the true sky (A_true = 1)
    model = c * rm                        # the template, per unit p
    var = 2.0 * (signal + sv.noise_cl) ** 2 / ((2 * l + 1) * d_l * sv.f_sky)
    cov = np.diag(var)
    if with_sys:
        w = band_w[sel] * (c * rm)        # band half-width in C_ell units
        cov = cov + RHO_SYS * np.outer(w, w)

    p_hat, sig_p = gls(signal, model, cov)
    a_hat = float(np.sqrt(max(p_hat, 1e-12)))
    sig_a = float(sig_p / (2.0 * a_hat))
    return {"A_hat": a_hat, "sigma_A": sig_a,
            "frac_S8_shift": a_hat - 1.0,
            "delta_S8": (a_hat - 1.0) * S8_FID,
            "sigma_S8": sig_a * S8_FID,
            "snr_A": float(a_hat / sig_a),
            "n_ell_grid_points": int(sel.sum())}


def peak_information(emu, raw, srcs, zs_grid, sv, tables):
    """Information retention on the peak feedback amplitude (see docstring)."""
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    fid_all = emu.predict(u_fid, "peak_counts")
    fid_z = zs_interp(fid_all[None], zs_grid, sv.z_eff)[0]
    sig, _ = _hos_sigma(raw, "peak_counts", sv, zs_grid, fid_z)
    mask = np.isfinite(sig) & (sig > 0)
    nu = np.asarray(raw["a__peak_counts__nu"], float)

    rng = np.random.default_rng(401)
    mu, sd = emu.predict(srcs["joint_post"], "peak_counts", return_std=True)
    sd = np.nan_to_num(np.asarray(sd, float), nan=0.0)
    pred = mu + sd * rng.standard_normal((len(srcs["joint_post"]),) + (1,) * 2)
    pz = zs_interp(pred, zs_grid, sv.z_eff)
    dev = pz - fid_z[None]

    nu_cut = tables["hos"]["peak_counts"][sv.name]["prior"].get("nu_safe_max")
    cut_mask = mask & (np.abs(nu) < (nu_cut if nu_cut else np.inf))
    # nu_safe_max is the SMALLEST |nu| that breaches, so when a low-|nu|
    # bin already breaches there is NO safe nu range at all and the cut
    # empties the vector. Report that as the finding it is, not as nan.
    no_safe_range = bool(cut_mask.sum() < 2)
    return {
        "nu_cut_used": nu_cut,
        "n_dims_full": int(mask.sum()),
        "n_dims_after_cut": int(cut_mask.sum()),
        "no_safe_nu_range": no_safe_range,
        "pc1_bias_over_sigma_full": _pc1_amplitude(dev, sig, mask),
        "pc1_bias_over_sigma_cut": (None if no_safe_range
                                    else _pc1_amplitude(dev, sig, cut_mask)),
        "note": "information-retention on the peak feedback-response "
                "amplitude, NOT a cosmological S8 S/N (fixed-cosmology "
                "suite has no d(peaks)/dS8) — see module docstring."
                + (" A nu-cut is NOT an available mitigation here: the "
                   "lowest-|nu| bins already breach, so no safe nu range "
                   "exists at this survey's statistical precision."
                   if no_safe_range else ""),
    }


def null_test(cl_fid, band_w, sv, ell, dl):
    """ACCEPTANCE: zero feedback must give zero shift in all analyses."""
    one = np.ones_like(cl_fid)
    outs = {
        "uncorrected": analyse(cl_fid, one, one, band_w, sv, ell, dl),
        "calibrated": analyse(cl_fid, one, one, band_w, sv, ell, dl,
                              with_sys=True),
        "scale_cut": analyse(cl_fid, one, one, band_w, sv, ell, dl,
                             use=ell <= 1000.0),
    }
    worst = max(abs(o["frac_S8_shift"]) for o in outs.values())
    return {"PASS": bool(worst < 1e-8), "worst_abs_frac_shift": float(worst),
            "per_analysis": {k: v["frac_S8_shift"] for k, v in outs.items()}}


def main() -> None:
    emu = StatsEmulator.load()
    raw = np.load(DATASET, allow_pickle=False)
    ell_all = np.asarray(raw["a__suppression__ell"], float)
    cl_dmo = np.asarray(raw["cl_dmo"], float)
    zs_grid = np.asarray(raw["source_redshifts"], float)
    tables = json.loads((WP7 / "safescale_tables.json").read_text())

    srcs = band_sources()
    bands, fid = load_bands(emu, srcs)

    sel = (ell_all >= ELL_MIN) & (ell_all <= ELL_MAX)
    ell = ell_all[sel]
    dl = np.gradient(ell)

    results, null = {}, {}
    for sv in SURVEYS:
        cl_fid = (zs_interp(cl_dmo, zs_grid, sv.z_eff)[sel]
                  * zs_interp(fid, zs_grid, sv.z_eff)[sel])

        q = bands["joint_post"]
        r_lo = zs_interp(q[0], zs_grid, sv.z_eff)[sel]   # strong side
        r_med = zs_interp(q[1], zs_grid, sv.z_eff)[sel]
        band_w = 0.5 * np.abs(zs_interp(q[2] - q[0], zs_grid, sv.z_eff))[sel]
        one = np.ones_like(cl_fid)

        lmax_prior = tables["two_point_clkk"][sv.name]["prior"]["lmax_safe"]
        cut = ell <= lmax_prior

        results[sv.name] = {
            "lmax_cut_used": lmax_prior,
            "injected": "joint-A+B posterior 16th pct (strong side)",
            "uncorrected": analyse(cl_fid, r_lo, one, band_w, sv, ell, dl),
            "calibrated": analyse(cl_fid, r_lo, r_med, band_w, sv, ell, dl,
                                  with_sys=True),
            "scale_cut": analyse(cl_fid, r_lo, one, band_w, sv, ell, dl,
                                 use=cut),
            "peaks": peak_information(emu, raw, srcs, zs_grid, sv, tables),
        }
        null[sv.name] = null_test(cl_fid, band_w, sv, ell, dl)

    if not all(n["PASS"] for n in null.values()):
        raise SystemExit(f"ACCEPTANCE FAILED (zero-feedback null): {null}")

    out = {"parameterisation": "C_ell propto A^2, A propto S8 (alpha=2, "
                              "the safescale.py convention)",
           "S8_fid_for_conversion": S8_FID,
           "rho_sys": RHO_SYS,
           "null_test_zero_feedback": null,
           "surveys": results}
    (WP7 / "s8_exercise.json").write_text(json.dumps(out, indent=2))

    print("=== zero-feedback null (acceptance) ===")
    for k, v in null.items():
        print(f"  {k:9s} PASS={v['PASS']} worst |shift| = "
              f"{v['worst_abs_frac_shift']:.2e}")
    print("\n=== S8 exercise: strong-side feedback injected ===")
    for name, r in results.items():
        print(f"\n{name}  (scale cut at ell <= {r['lmax_cut_used']:.0f})")
        for k in ("uncorrected", "calibrated", "scale_cut"):
            a = r[k]
            print(f"  {k:12s} dS8 = {a['delta_S8']:+.4f} "
                  f"+/- {a['sigma_S8']:.4f}   "
                  f"bias/sigma = {abs(a['frac_S8_shift'])/a['sigma_A']:5.2f}   "
                  f"S/N(A) = {a['snr_A']:7.1f}  "
                  f"({a['n_ell_grid_points']} ell pts)")
        p = r["peaks"]
        if p["no_safe_nu_range"]:
            print(f"  peaks: NO safe nu range exists (lowest-|nu| bins "
                  f"already breach); PC1 b/sig full "
                  f"{p['pc1_bias_over_sigma_full']:.2f}")
        else:
            print(f"  peaks: dims {p['n_dims_after_cut']}/{p['n_dims_full']} "
                  f"after nu-cut; PC1 b/sig full "
                  f"{p['pc1_bias_over_sigma_full']:.2f} vs cut "
                  f"{p['pc1_bias_over_sigma_cut']:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, r) in zip(axes, results.items()):
        ks = ["uncorrected", "calibrated", "scale_cut"]
        y = [r[k]["delta_S8"] for k in ks]
        e = [r[k]["sigma_S8"] for k in ks]
        ax.errorbar(range(3), y, yerr=e, fmt="o", capsize=4, color="#0072B2")
        ax.axhline(0.0, color="k", lw=1, ls="--")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["uncorr.", "calib.", "cut"], fontsize=9)
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel(r"$\Delta S_8$ (injected strong-side feedback)")
    fig.suptitle("WP-A7 task 3: S8 bias and precision under three "
                 "feedback treatments", fontsize=11)
    fig.tight_layout()
    fig.savefig(WP7 / "figures" / "s8_exercise.png", dpi=150)
    print(f"\nwrote {WP7}/s8_exercise.json + figures/s8_exercise.png")


if __name__ == "__main__":
    main()
