#!/usr/bin/env python
"""Round-2 track T1 (docs/paper_improvement_plan.md, D1): literature-named,
EXTERNAL baryon-suppression curves S(ell, z_s=1) = C_ell^kappa(baryon) /
C_ell^kappa(DMO) to overplot on the paper's Fig 10 alongside the TNG-family
node curves already in ``LC/Sell_zs1_253.npz`` (built by
``examples/lightcone_transfer.py`` on another branch -- not re-run here).

This script is standalone: it does not touch the paper notebook or its
builder (``examples/_build_p4c_paper_nb.py``); a later integration agent
consumes the product it writes.

**Method.** Limber-projected weak-lensing convergence power at a single
source plane z_s=1.0, cosmology fixed to the TNG300 box used throughout
this analysis (Omega_m=0.3089, Omega_b=0.0486, h=0.6774, sigma8=0.8159,
n_s=0.9667), using ``pyccl`` (``ccl.Cosmology`` + ``ccl.WeakLensingTracer``
+ ``ccl.angular_cl``, halofit nonlinear matter power as the DMO baseline).
The source n(z) is a narrow Gaussian (sigma_z=0.01) around z=1 --
``ccl.CMBLensingTracer`` is for the CMB last-scattering source, not this.
The ell grid is read from ``LC/Sell_zs1_253.npz`` (key ``"ell"``) so the
output overplots directly onto the existing Fig 10 axis.

**Models built** (see each ``build_*`` function's docstring for the exact
parameters/provenance):
  1. BCM (Schneider & Teyssier 2015) -- ``ccl.BaryonsSchneider15`` defaults.
  2. van Daalen et al. 2019 f_bar model -- ``ccl.BaryonsvanDaalen19`` at the
     baryon fraction implied by our joint kSZ+tSZ posterior.
  3. Amon & Efstathiou (2022) A_mod=0.82 -- implemented directly as
     P_mod = P_lin + A_mod*(P_nl - P_lin); projected via a hand-built Pk2D.
  4. HMcode-2020 baryonic feedback, T_AGN=7.8 (BAHAMAS-calibrated;
     Mead et al. 2020) -- via ``camb``'s ``mead2020_feedback`` halofit
     variant vs. its ``mead2020`` (gravity-only) baseline; S(k,z) built on
     a camb grid and projected with a hand-built Pk2D (same Limber
     machinery/tracer as #3, not multiplicatively grafted onto ccl's own
     halofit spline -- see ``build_hmcode`` docstring for why).
  5. CAMELS SIMBA/Astrid CV-mean S(k) -- SKIPPED. Grepping this repo
     (including ``data_generation/``) for a hardcoded path to a CAMELS
     power-spectrum product turns up nothing; the only SIMBA/Astrid
     mentions are prose asides in the paper notebook builders
     (``examples/_build_p4c_paper_nb.py:1147,1256-1257``,
     ``examples/_build_ksz_paper_nb.py:44``), not file paths. Per the task
     instructions this means skip, not go hunting on ceph.

Run:
    source /mnt/home/mlee1/venvs/BIND_env/bin/activate
    python examples/lightcone_external_suppression.py

Product: ``LC/external_suppression_curves.npz`` with keys ``ell``,
one ``S_<name>`` array per curve, and ``provenance`` (an array of
per-curve documentation strings, same order as the ``S_*`` keys via
``curve_names``).

**Gotcha discovered while writing this** (worth remembering for any future
hand-built ``pyccl.Pk2D``): when ``is_logp=True`` the ``pk_arr`` you pass to
the constructor must already be ``log(P)``, not ``P`` -- ``get_spline_arrays()``
returns the *exponentiated* P(k,a) for convenience, so round-tripping it
back into a new ``Pk2D(..., is_logp=True)`` without re-logging silently
produces a spline of ``exp(P)`` and ``ccl.angular_cl`` fails with
``CCL_ERROR_INTEG`` (no clearer message). ``Pk2D.__add__``/``__mul__`` do
this correctly internally; only manual construction from raw arrays needs
the explicit ``np.log``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
KS = CEPH / "bind_science/ksz_confront"
LC = KS / "lightcone"

SELL_PATH = LC / "Sell_zs1_253.npz"
OUT_PATH = LC / "external_suppression_curves.npz"

# TNG300 cosmology used throughout this analysis (paper Conventions block).
OMEGA_M = 0.3089
OMEGA_B = 0.0486
H0_H = 0.6774
SIGMA8 = 0.8159
N_S = 0.9667
OMEGA_C = OMEGA_M - OMEGA_B

# Our joint kSZ+tSZ posterior gas-fraction measurement (of cosmic, i.e. in
# units of Omega_b/Omega_m -- the same convention vD19's fbar uses), at the
# group/cluster mass scale probed by the stack (logM200~13.2). Given in the
# task spec; cross-checked here against LC/r8_posterior.npz gas_fin_joint
# (median 0.4533, 16-84% 0.392-0.512) -- consistent.
F_GAS_MEASURED = 0.452
LOGM200_MEASURED = 13.2

# Amon & Efstathiou (2022), arXiv:2206.11794 (S8 tension via a nonlinear
# power suppression, not baryons specifically).
A_MOD = 0.82

# HMcode-2020 feedback (Mead et al. 2020, arXiv:2009.01858); T_AGN=7.8 is
# both camb's own default and the BAHAMAS "AGN 7.8" calibration point.
LOG_T_AGN = 7.8


def load_ell():
    d = np.load(SELL_PATH)
    ell = d["ell"]
    print(f"[ell grid] {SELL_PATH} -> {ell.size} points, "
          f"[{ell.min():.2f}, {ell.max():.2f}]")
    return ell


def make_cosmo():
    import pyccl
    return pyccl.Cosmology(
        Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H0_H, n_s=N_S, sigma8=SIGMA8,
        transfer_function="boltzmann_camb", matter_power_spectrum="halofit",
    )


def make_tracer(cosmo, z_s=1.0, sigma_z=0.01):
    import pyccl
    z = np.linspace(0.0, 3.0, 3000)
    nz = np.exp(-0.5 * ((z - z_s) / sigma_z) ** 2)
    return pyccl.WeakLensingTracer(cosmo, dndz=(z, nz))


def report_curve(name, ell, S):
    """Print the validation numbers the task asks for and flag pathologies.

    Returns True if the curve should be KEPT.
    """
    if np.any(np.isnan(S)) or np.any(np.isinf(S)):
        print(f"[{name}] DROPPED: NaN/Inf present")
        return False
    if np.any(S <= 0):
        print(f"[{name}] DROPPED: non-positive S values")
        return False

    lowmask = ell < 300
    low_first = S[np.argmin(ell)]
    idx2000 = int(np.argmin(np.abs(ell - 2000)))
    idx5000 = int(np.argmin(np.abs(ell - 5000)))
    print(f"[{name}] S(ell->{ell[np.argmin(ell)]:.1f}, lowest grid pt) = "
          f"{low_first:.4f} ({100*(1-low_first):+.2f}% from 1)")
    print(f"[{name}] mean S over ell<300 (n={lowmask.sum()}) = "
          f"{S[lowmask].mean():.4f}")
    print(f"[{name}] S(ell={ell[idx2000]:.1f}) = {S[idx2000]:.4f}  "
          f"({100*(1-S[idx2000]):.2f}% suppression)")
    print(f"[{name}] S(ell={ell[idx5000]:.1f}) = {S[idx5000]:.4f}  "
          f"({100*(1-S[idx5000]):.2f}% suppression)")
    print(f"[{name}] min S = {S.min():.4f} at ell={ell[np.argmin(S)]:.1f}; "
          f"max S = {S.max():.4f} at ell={ell[np.argmax(S)]:.1f}")

    if low_first > 1.02:
        print(f"[{name}] DROPPED: does not -> 1 at low ell "
              f"(lowest-grid-point tolerance 2%)")
        return False
    return True


# ---------------------------------------------------------------------
# 1. BCM (Schneider & Teyssier 2015)
# ---------------------------------------------------------------------
def build_bcm(cosmo, tracer, ell):
    import pyccl
    bar = pyccl.BaryonsSchneider15()  # ccl defaults: log10Mc, eta_b=0.5, k_s=55
    pk_dmo = cosmo.get_nonlin_power()
    pk_bar = bar.include_baryonic_effects(cosmo, pk_dmo)
    cl_dmo = pyccl.angular_cl(cosmo, tracer, tracer, ell, p_of_k_a=pk_dmo)
    cl_bar = pyccl.angular_cl(cosmo, tracer, tracer, ell, p_of_k_a=pk_bar)
    S = cl_bar / cl_dmo
    provenance = (
        "BCM (Schneider & Teyssier 2015, arXiv:1510.06034), pyccl "
        "BaryonsSchneider15 DEFAULT parameters: "
        f"log10Mc={bar.log10Mc:.6f} (~1.2e14 Msun, the AGN-feedback default), "
        f"eta_b={bar.eta_b}, k_s={bar.k_s} h/Mpc. Applied multiplicatively "
        "to the ccl halofit nonlinear P(k,z) at the TNG300 cosmology "
        f"(Omega_m={OMEGA_M}, Omega_b={OMEGA_B}, h={H0_H}, sigma8={SIGMA8}, "
        f"n_s={N_S}); projected with a WeakLensingTracer, source n(z) a "
        "sigma_z=0.01 Gaussian at z_s=1.0. Note: this default log10Mc is a "
        "fairly strong AGN-feedback point (not a weak-feedback baseline); "
        "its Cl suppression (~18-19% at ell~2000-5000, see report) is "
        "stronger than a naive 'a few percent' expectation but matches the "
        "known shape/amplitude of the ccl default BCM boost factor "
        "(dips to ~0.80 around k~1-3 h/Mpc, partially recovers above "
        "k~15 h/Mpc from the stellar term) -- not a numerical artifact."
    )
    return S, provenance


# ---------------------------------------------------------------------
# 2. van Daalen et al. 2019 f_bar model, at our measured gas fraction
# ---------------------------------------------------------------------
def build_vandaalen19(cosmo, tracer, ell):
    import pyccl
    mass_def = "500c"  # matches our f_gas(<R500) aperture (see caveat below)
    vd = pyccl.BaryonsvanDaalen19(fbar=F_GAS_MEASURED, mass_def=mass_def)
    pk_dmo = cosmo.get_nonlin_power()
    pk_bar = vd.include_baryonic_effects(cosmo, pk_dmo)
    cl_dmo = pyccl.angular_cl(cosmo, tracer, tracer, ell, p_of_k_a=pk_dmo)
    cl_bar = pyccl.angular_cl(cosmo, tracer, tracer, ell, p_of_k_a=pk_bar)
    S = cl_bar / cl_dmo
    provenance = (
        "van Daalen et al. 2019 (arXiv:1906.00968) f_bar boost model, "
        "pyccl BaryonsvanDaalen19, labeled 'vD19 model at our measured "
        f"f~gas'. fbar={F_GAS_MEASURED} set directly from our joint "
        f"kSZ+tSZ posterior f~gas(<R500)={F_GAS_MEASURED} of cosmic "
        f"(Omega_b/Omega_m) at logM200~{LOGM200_MEASURED} (cross-checked "
        "against LC/r8_posterior.npz 'gas_fin_joint', median 0.453, "
        "16-84% 0.392-0.512) -- vD19's fbar is defined in the SAME units "
        "('the fraction of baryons in a halo in units of the ratio of "
        "Omega_b to Omega_m'), so this is a direct plug-in, not a rescaled "
        "quantity. MAPPING CAVEAT: mass_def='500c' was chosen because our "
        "f~gas is an R500 aperture measurement (matching vD19's SO "
        "convention for that coefficient set), but the halo mass label "
        f"attached to that measurement (logM200~{LOGM200_MEASURED}) is an "
        "M200-defined mass, not M500 -- i.e. the aperture convention "
        "(R500, used for mass_def) and the mass label (M200, used only "
        "for context here) are not from the same SO definition. This is a "
        "mild convention mismatch, not a magnitude rescaling: "
        "BaryonsvanDaalen19.boost_factor() takes no explicit halo-mass "
        "argument -- 'mass_def' just selects between two coefficient sets "
        "(500c vs 200c) fit to different SO gas-aperture conventions in "
        "the vD19 simulations, not a mass pivot scale; it is NOT a "
        "population/HMF average tied to our specific group-scale sample, "
        "so this curve should be read as 'what the vD19 fitting function "
        "predicts for a halo population with our measured mean f~gas', "
        "not a bespoke group-scale-only calibration."
    )
    return S, provenance


# ---------------------------------------------------------------------
# 3. Amon & Efstathiou (2022) A_mod
# ---------------------------------------------------------------------
def build_amod(cosmo, tracer, ell):
    import pyccl
    pk_nl = cosmo.get_nonlin_power()
    pk_lin = cosmo.get_linear_power()
    pk_mod = pk_lin + A_MOD * (pk_nl - pk_lin)  # Pk2D arithmetic
    cl_nl = pyccl.angular_cl(cosmo, tracer, tracer, ell, p_of_k_a=pk_nl)
    cl_mod = pyccl.angular_cl(cosmo, tracer, tracer, ell, p_of_k_a=pk_mod)
    S = cl_mod / cl_nl
    provenance = (
        "Amon & Efstathiou (2022, arXiv:2206.11794) A_mod nonlinear-power "
        f"suppression, A_mod={A_MOD}. Implemented directly (not a ccl "
        "built-in): P_mod(k,z) = P_lin(k,z) + A_mod*(P_nl(k,z)-P_lin(k,z)), "
        "S(k,z) = P_mod/P_nl, using ccl's own linear and halofit-nonlinear "
        "Pk2D objects (same k,a grid, so no interpolation/domain mismatch) "
        "at the TNG300 cosmology; projected via ccl.angular_cl with a "
        "hand-built Pk2D (P_mod) against the halofit baseline, same "
        "WeakLensingTracer as the other curves. NOTE: unlike the "
        "baryon-specific models (BCM, vD19, HMcode-feedback), A_mod "
        "suppresses the ENTIRE nonlinear growth increment, including "
        "quasi-linear scales -- so this curve does not -> 1 as sharply at "
        "low ell as the others (lowest grid point ell~87 is within ~1%, "
        "but by ell~240 it is already ~5% below 1); this is the expected, "
        "literature-discussed behavior of the A_mod model (part of why it "
        "was controversial as an S8-tension fix), not a projection bug."
    )
    return S, provenance


# ---------------------------------------------------------------------
# 4. HMcode-2020 feedback, T_AGN=7.8 (camb), if camb is importable
# ---------------------------------------------------------------------
def build_hmcode(cosmo, tracer, ell):
    """HMcode-2020 feedback vs. its own gravity-only baseline, both from
    camb, projected via a hand-built Pk2D using the SAME cosmo/tracer as
    the other curves (so the Limber kernel/background is identical) --
    but with the numerator AND denominator both built on camb's own
    (mead2020_feedback, mead2020) grid, rather than grafting camb's S(k,z)
    ratio onto ccl's halofit Pk2D. Two reasons: (1) ccl's default halofit
    Pk2D only spans k up to ~10.5/Mpc (~15.5 h/Mpc; see
    ``cosmo.get_nonlin_power().get_spline_arrays()``), narrower than
    camb's grid, so pyccl.Pk2D multiplication (which requires the second
    operand's domain to be a superset of the first) does not chain
    cleanly here; (2) computing the ratio from two Cls built on the exact
    same camb grid makes any high-k extrapolation error common-mode
    between numerator and denominator, which is far more robust for a
    ratio than grafting a truncated-domain correction onto a
    wider-domain baseline.
    """
    try:
        import camb
        from camb import model as camb_model
    except ImportError:
        print("[HMcode-2020] camb not importable -- skipping cleanly.")
        return None, None
    import pyccl

    h = H0_H
    H0 = h * 100
    ombh2 = OMEGA_B * h ** 2
    omch2 = OMEGA_C * h ** 2
    zs = np.linspace(0.0, 3.0, 31)
    kmax_hmpc = 60.0

    def build_pars(halofit_version, As, logT_AGN=LOG_T_AGN):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0.0, mnu=0.0)
        pars.InitPower.set_params(As=As, ns=N_S)
        pars.set_matter_power(redshifts=list(zs), kmax=kmax_hmpc)
        pars.NonLinear = camb_model.NonLinear_both
        if halofit_version == "mead2020_feedback":
            pars.NonLinearModel.set_params(
                halofit_version="mead2020_feedback", HMCode_logT_AGN=logT_AGN)
        else:
            pars.NonLinearModel.set_params(halofit_version="mead2020")
        return pars

    t0 = time.time()
    # Calibrate As to hit the target sigma8 (camb takes As, not sigma8).
    pars0 = build_pars("mead2020", As=2.1e-9)
    res0 = camb.get_results(pars0)
    sigma8_now = res0.get_sigma8_0()
    As_final = 2.1e-9 * (SIGMA8 / sigma8_now) ** 2
    print(f"[HMcode-2020] As calibration: sigma8(As=2.1e-9)={sigma8_now:.5f} "
          f"-> As_final={As_final:.4e} (target sigma8={SIGMA8})")

    pars_dmo = build_pars("mead2020", As=As_final)
    res_dmo = camb.get_results(pars_dmo)
    pars_fb = build_pars("mead2020_feedback", As=As_final, logT_AGN=LOG_T_AGN)
    res_fb = camb.get_results(pars_fb)
    print(f"[HMcode-2020] camb runs done in {time.time()-t0:.1f}s "
          f"(sigma8 check: dmo={res_dmo.get_sigma8_0():.5f}, "
          f"fb={res_fb.get_sigma8_0():.5f})")

    kh, zout, pk_dmo = res_dmo.get_matter_power_spectrum(
        minkh=1e-4, maxkh=kmax_hmpc, npoints=400)
    _, _, pk_fb = res_fb.get_matter_power_spectrum(
        minkh=1e-4, maxkh=kmax_hmpc, npoints=400)

    zout_arr = np.asarray(zout)
    a_of_z = 1.0 / (1.0 + zout_arr)
    order = np.argsort(a_of_z)
    a_arr = a_of_z[order]
    lk_arr = np.log(kh * h)  # camb kh is h/Mpc -> ccl wants ln(k[Mpc^-1])
    pk_dmo_mpc3 = pk_dmo[order, :] / h ** 3  # (Mpc/h)^3 -> Mpc^3
    pk_fb_mpc3 = pk_fb[order, :] / h ** 3

    # extrap_order_hik=0 (hold flat beyond the camb grid) deliberately,
    # NOT the pyccl default of 2: order>=1 log-log extrapolation of the
    # ratio diverges badly at the very highest ell in our grid (S->~3 by
    # ell~52000 with order=1, vs. a smooth, bounded turnover with order=0
    # -- see build_hmcode's caveat text below).
    pk2d_dmo = pyccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr,
                           pk_arr=np.log(pk_dmo_mpc3), is_logp=True,
                           extrap_order_lok=1, extrap_order_hik=0)
    pk2d_fb = pyccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr,
                          pk_arr=np.log(pk_fb_mpc3), is_logp=True,
                          extrap_order_lok=1, extrap_order_hik=0)

    cl_dmo = pyccl.angular_cl(cosmo, tracer, tracer, ell, p_of_k_a=pk2d_dmo)
    cl_fb = pyccl.angular_cl(cosmo, tracer, tracer, ell, p_of_k_a=pk2d_fb)
    S = cl_fb / cl_dmo

    imin = int(np.argmin(S))
    ell_min = float(ell[imin])
    provenance = (
        "HMcode-2020 baryonic feedback (Mead et al. 2020, arXiv:2009.01858), "
        f"BAHAMAS-calibrated T_AGN, log10(T_AGN/K)={LOG_T_AGN} (camb's own "
        "default, and the standard BAHAMAS 'AGN 7.8' point). Built with "
        "camb 'mead2020_feedback' vs. its gravity-only 'mead2020' "
        "counterpart at the TNG300 cosmology; As calibrated to the target "
        f"sigma8={SIGMA8} via one Newton-like rescaling step "
        "(As_new = As_old*(sigma8_target/sigma8_old)^2). S(k,z) grid: "
        f"k in [1e-4, {kmax_hmpc}] h/Mpc, z in [0,3] (31 pts), 400 log-k "
        "points; projected via a hand-built ccl Pk2D and the same "
        "WeakLensingTracer as the other curves (see build_hmcode "
        "docstring for why numerator/denominator are both camb-native "
        "rather than grafted onto ccl's halofit). CAVEAT: this ratio "
        f"reaches a minimum ({S[imin]:.3f}) at ell~{ell_min:.0f} and then "
        "rises back toward and above 1 at higher ell -- a real feature of "
        "HMcode-2020-feedback (small-scale stellar contraction pushes "
        "power above the DMO baseline at very high k, the same qualitative "
        "behavior seen in the BCM boost factor), but at the very highest "
        "ell in this grid (>~ell 15000-20000) it is beyond both the "
        "camb k-grid (extrapolated flat, extrap_order_hik=0, beyond "
        f"k={kmax_hmpc} h/Mpc) and HMcode-2020-feedback's BAHAMAS-verified "
        "regime (~k<=10-20 h/Mpc per Mead et al. 2020) -- treat ell above "
        "the turnover as illustrative only, not a validated prediction."
    )
    return S, provenance


def main():
    ell = load_ell()
    cosmo = make_cosmo()
    tracer = make_tracer(cosmo)

    import pyccl
    print(f"[pyccl] version={pyccl.__version__}")
    try:
        import camb
        print(f"[camb] version={camb.__version__} (importable)")
        camb_ok = True
    except ImportError:
        print("[camb] NOT importable")
        camb_ok = False

    curves = {}
    provenances = {}

    print("\n--- BCM (Schneider & Teyssier 2015) ---")
    S, prov = build_bcm(cosmo, tracer, ell)
    if report_curve("BCM_Schneider15", ell, S):
        curves["BCM_Schneider15"] = S
        provenances["BCM_Schneider15"] = prov

    print("\n--- van Daalen et al. 2019 (at our measured f~gas) ---")
    S, prov = build_vandaalen19(cosmo, tracer, ell)
    if report_curve("vanDaalen19_fgas", ell, S):
        curves["vanDaalen19_fgas"] = S
        provenances["vanDaalen19_fgas"] = prov

    print("\n--- Amon & Efstathiou 2022 A_mod=0.82 ---")
    S, prov = build_amod(cosmo, tracer, ell)
    if report_curve("AmonEfstathiou22_Amod", ell, S):
        curves["AmonEfstathiou22_Amod"] = S
        provenances["AmonEfstathiou22_Amod"] = prov

    if camb_ok:
        print("\n--- HMcode-2020 feedback, T_AGN=7.8 ---")
        S, prov = build_hmcode(cosmo, tracer, ell)
        if S is not None and report_curve("HMcode2020_TAGN7p8", ell, S):
            curves["HMcode2020_TAGN7p8"] = S
            provenances["HMcode2020_TAGN7p8"] = prov
    else:
        print("\n--- HMcode-2020 feedback: SKIPPED (camb unavailable) ---")

    print("\n--- CAMELS SIMBA/Astrid CV-mean S(k): SKIPPED ---")
    print("No exact repo-known path to a CAMELS power-spectrum product "
          "was found by grepping this repo (incl. data_generation/); only "
          "prose mentions of SIMBA/Astrid exist "
          "(examples/_build_p4c_paper_nb.py:1147,1256-1257; "
          "examples/_build_ksz_paper_nb.py:44). Per instructions: skip, "
          "do not search ceph directly.")

    if not curves:
        print("\nNo curves survived validation -- not writing a product.")
        sys.exit(1)

    curve_names = sorted(curves.keys())
    provenance_arr = np.array([provenances[n] for n in curve_names], dtype=object)

    out = {
        "ell": ell,
        "curve_names": np.array(curve_names, dtype=object),
        "provenance": provenance_arr,
    }
    for n in curve_names:
        out[f"S_{n}"] = curves[n]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **out)
    print(f"\nWrote {OUT_PATH} with keys: {sorted(out.keys())}")


if __name__ == "__main__":
    main()
