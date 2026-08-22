"""R4-cosmo: propagate the Elbers et al. (2025, arXiv:2403.12967) feedback x
cosmology coupling through a Stage-IV-width cosmology posterior, onto this
paper's own S(ell) trough, and compare against this paper's sigma_pred and
the LSST-Y10 statistical floor.

Coupling (Elbers+2025 Eqs. 16, 21-24):
    xi^2 = f_b / c_v^2,  c_v = 1.24 (Om*s8)^(1/8)
    Delta F_b / (1-F_b) = -alpha' * Delta[ Ob/(Ob+Oc) * (Om*s8)^(-1/4) ],
    alpha' = 8.98  (= 0.65 * alpha, alpha = 13.8 +/- 0.6)
F_b is their notation for the suppression ratio itself (= this paper's S).

Run with: /mnt/home/mlee1/venvs/BIND_env/bin/python3 r4_elbers_bound.py
See papers/01_pipeline/referee/R4_cosmology_caveat_memo.md Sec. 3 for the
narrative writeup of this output.
"""
import numpy as np

# ---- fiducial cosmology: TNG300 / TNG300-Dark, Planck-2015 (main.tex l.395) ----
Om0, s80, Ob0 = 0.3089, 0.8158, 0.0486
X0 = (Ob0 / Om0) * (Om0 * s80) ** (-0.25)

alpha_p = 8.98  # Elbers+2025 Eq. 24

# this paper's own numbers (main.tex l.842-846, l.895-896)
S_fid = 0.881  # measured S(ell) trough
one_minus_S = 1 - S_fid
sigma_pred_trough, sigma_pred_vD = 0.019, 0.003
lsst_S_precision = 0.005  # ~0.5% on S (task-supplied LSST-Y10 floor assumption)

# log-derivatives of X = Ob * Om^(-5/4) * s8^(-1/4)  (NOT elasticities -- raw d/dOm, d/ds8)
a_Om, a_s8 = -1.25 / Om0, -0.25 / s80


def dS_from(dOm, ds8, label, sig_S8=None, sig_Om=None):
    dlnX = a_Om * dOm + a_s8 * ds8
    dX = X0 * dlnX
    frac = alpha_p * dX  # Delta(1-Fb)/(1-Fb)
    dS = -one_minus_S * frac  # Delta F_b = -(1-F_b) * alpha' * Delta X
    print(f"{label}")
    if sig_S8 is not None:
        print(f"    input: sigma(S8)={sig_S8:.4f} sigma(Om)={sig_Om:.4f}")
    print(
        f"    dOm={dOm:+.4f}  ds8={ds8:+.4f}  dlnX={dlnX:+.4f}  =>  "
        f"Delta(1-Fb)/(1-Fb)={frac * 100:+.2f}%  =>  Delta S = {dS:+.4f}"
    )
    print(
        f"    |Delta S| / sigma_pred(trough,0.019) = {abs(dS) / sigma_pred_trough:.2f}x   "
        f"|Delta S| / sigma_pred(vD,0.003) = {abs(dS) / sigma_pred_vD:.2f}x   "
        f"|Delta S| / LSST-Y10(0.005) = {abs(dS) / lsst_S_precision:.2f}x\n"
    )
    return dS


print(f"X0={X0:.5f}  a_Om={a_Om:.4f}  a_s8={a_s8:.4f}\n")

# Stage-IV widths: Wayland, Alonso & Zennaro 2025 (arXiv:2506.11943) Table 2
scenarios = {
    "tight (WL + long-term X-ray + long-term kSZ)": dict(
        sS8=(0.0088 + 0.0096) / 2, sOm=(0.011 + 0.012) / 2
    ),
    "loose (WL-only, baryons marginalized)": dict(
        sS8=(0.0170 + 0.0140) / 2, sOm=(0.022 + 0.021) / 2
    ),
    "task-example (optimistic textbook)": dict(sS8=0.005, sOm=0.003),
}

print("=" * 78)
print("CASE A -- worst-case coherent shift (Om, s8 move same sign; upper bound,")
print("no covariance assumed)")
print("=" * 78)
for name, sc in scenarios.items():
    dS8, dOm = sc["sS8"], sc["sOm"]
    ds8 = dS8  # approx delta(sigma8) ~ delta(S8) near fiducial (Om/0.3~1)
    dS_from(dOm, ds8, f"[{name}]", dS8, dOm)

print("=" * 78)
print("CASE B -- WL degeneracy direction: S8 = sigma8*sqrt(Om/0.3) held FIXED,")
print("only Om free (realistic single-probe cosmic-shear posterior shape;")
print("uses sigma(Om) only). d(sigma8) = -sigma8/(2 Om) * d(Om).")
print("=" * 78)
for name, sc in scenarios.items():
    dOm = sc["sOm"]
    ds8 = -s80 / (2 * Om0) * dOm
    dS_from(dOm, ds8, f"[{name}] (sigma(Om)={dOm:.4f})")

print("=" * 78)
print("CASE C -- Om fixed, only sigma8 free (for completeness / mechanism check)")
print("=" * 78)
for name, sc in scenarios.items():
    dS8 = sc["sS8"]
    dS_from(0.0, dS8, f"[{name}] (sigma(s8)~sigma(S8)={dS8:.4f})")

print("=" * 78)
print("Cross-check vs Elbers+2025's own quoted example ('a few percent -> 4-5%')")
print("=" * 78)
for f in [0.010, 0.015, 0.020]:
    dOm, ds8 = f * Om0, f * s80
    dS_from(dOm, ds8, f"  coherent {f * 100:.1f}% Om & s8 shift")
