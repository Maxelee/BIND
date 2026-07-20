"""WP-A7 companion: the OFF-MANIFOLD S8 injection.

The `s8exercise.py` result ("gas calibration removes ~93-99% of the S8
offset") injects the truth at the joint A+B posterior's 16th percentile --
the strong EDGE of the TNG-family band, Delta ln M_gas ~ -0.38. But this
paper's headline is that the data require Delta ln M_gas ~ -1.22, a factor
~3.8 OUTSIDE that band. So the 93-99% is a WITHIN-FAMILY self-calibration
number; it does not tell an analyst what happens when the real suppression
is off the TNG manifold. This script answers that.

METHOD, and why it is not the full C_ell machinery. The emulator cannot
predict a suppression at Delta ln M_gas = -1.22 -- that point is off its
training box, so `emu.predict(U, "suppression")` (which load_bands uses)
has nothing to extrapolate from. The off-manifold suppression instead
comes from van Daalen+2020 (1906.00968), the same relation the Wave-3 p5
work used, evaluated at the DATA-required baryon fraction. vd20 is a
single-scale relation at k = 0.5 h/Mpc, valid k < 1; it does NOT give the
full ell-shape the C_ell exercise needs, and extrapolating it to the
ell~8000 the safe-scale table reaches would be fabrication. So this is an
honest AMPLITUDE-LEVEL calculation at the vd20 pivot, reported as dP/P and
as an absorbed FRACTION (both dimensionless and robust), not as a per-ell
Delta S8.

THE ONE NUMBER. A TNG-calibrated correction can represent suppressions
only as deep as its own band edge. The off-manifold truth is far deeper.
So the absorbed fraction is bounded by

    absorbed_offmanifold ~ (dP/P at the TNG band edge)
                           ---------------------------
                           (dP/P at the data-required f_bar)

and the within-family 93% is recovered only when numerator = denominator.

INPUTS, all frozen and sourced:
  - p5_pk_feedback_implication.json:
      * vd20 coefficients d=-5.99, e=-0.5107 at k=0.5 (verified vs source)
      * ftilde_bar_TNG and ftilde_bar_data per nu (both halo catalogs)
      * dP_P_k0p5_data per nu; the conservative-after-nuisance branch
  - joint_ab_summary.json: the posterior Delta ln M_gas band edge (p16),
    which sets how deep the TNG band reaches.

CAVEATS carried in the artifact (see CAVEATS): vd20 k=0.5 pivot only;
extrapolation beyond vd20's calibrated f_bar window; the linear
Delta S8 ~ 0.5 |dP/P| S8 proxy BREAKS at off-manifold depth (which is
itself a finding -- the safe-scale formalism assumes small suppression);
and the whole result is conditional on the off-manifold suppression being
real (it inherits every p5 caveat, incl. the prior-conditional absolute
anchor).

Run: python analysis/paper3a/cosmology/s8_offmanifold.py
Out: wp7_cosmology/s8_offmanifold.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WP7 = Path("/mnt/ceph/users/mlee1/paper3/A/wp7_cosmology")
P5 = Path("/mnt/ceph/users/mlee1/paper3/B/wp5_inference/"
          "p5_pk_feedback_implication.json")
JSUM = Path("/mnt/ceph/users/mlee1/paper3/A/wp5_chains/joint_ab_summary.json")

S8_FID = 0.83                 # matches s8exercise.py
# The within-family self-calibration result this is the off-manifold twin
# of (s8_exercise.json, DES-Y6): absorbed 93.4% = (0.0361-0.0024)/0.0361.
WITHIN_FAMILY_ABSORBED = (0.0361 - 0.0024) / 0.0361


def vd20_dP_P(ftilde_bar, d, e):
    """van Daalen+2020 Fig.16/eq.footnote-11 at k=0.5 h/Mpc."""
    return -np.exp(d * np.asarray(ftilde_bar, float) + e)


def main() -> None:
    p5 = json.loads(P5.read_text())
    vd = p5["van_daalen_2020"]
    d, e = vd["coefficients_k0p5"]["d"], vd["coefficients_k0p5"]["e"]

    jsum = json.loads(JSUM.read_text())
    # strong edge of the posterior = most gas removed = p16 of dln_mgas
    dln_edge = float(jsum["coords_posterior"]["dln_mgas"]["p16"])
    dln_med = float(jsum["coords_posterior"]["dln_mgas"]["p50"])

    out = {
        "what": "off-manifold S8 injection: how much of the S8 bias a "
                "TNG-calibrated correction absorbs when the true suppression "
                "is at the DATA-required baryon fraction, not the TNG band "
                "edge",
        "within_family_reference": {
            "absorbed_fraction": round(WITHIN_FAMILY_ABSORBED, 4),
            "source": "s8_exercise.json DES-Y6 (injection at joint posterior "
                      "16th pct, INSIDE the TNG band)",
        },
        "vd20": {"d": d, "e": e, "pivot_k_h_per_Mpc": 0.5,
                 "validity_k_max": vd["full_model_validity_k_max"],
                 "verified_against_source": vd["verified_against_source"]},
        "posterior_band_edge": {
            "dln_mgas_p16_strong_edge": dln_edge,
            "dln_mgas_p50": dln_med,
            "note": "the TNG band reaches only this far; data require "
                    "~-1.22 (p5 acceptance), a factor ~3.8 beyond"},
        "per_bin": {},
        "CAVEATS": [
            "vd20 is a SINGLE-SCALE relation at k=0.5 h/Mpc (valid k<1); this "
            "is an amplitude-level result, NOT the per-ell C_ell exercise. "
            "Extrapolating vd20 to the ell~8000 the safe-scale table reaches "
            "would be fabrication, so it is deliberately not done.",
            "the band-edge f_bar uses a GAS-ONLY reduction "
            "(f_bar_edge = f_bar_TNG * exp(dln_mgas_edge)), which OVER-reduces "
            "(stars are held fixed in reality), making the band reach DEEPER "
            "and the absorbed fraction LARGER -- i.e. conservative against "
            "this script's own conclusion.",
            "Delta S8 ~ 0.5|dP/P|S8 is a small-signal proxy; at off-manifold "
            "depth |dP/P|~0.1-0.24 it BREAKS (implied Delta S8 exceeds what a "
            "linearization can carry). That the safe-scale formalism breaks "
            "off-manifold is itself the finding, not a bug in this script.",
            "conditional on the off-manifold suppression being real: inherits "
            "every p5 caveat, including the prior-conditional absolute anchor "
            "and the two candidate halo catalogs.",
            "absorbed = within_family_93.4pct / depth_ratio is a FIRST-ORDER "
            "model: the correction removes the same absolute suppression it "
            "removed within-family, since its template is unchanged. A full "
            "GLS amplitude fit against the deeper signal would differ by an "
            "O(1) factor, but the ORDER (single-digit to ~20% absorbed) is "
            "robust -- even 2x more removal leaves absorbed well under 50%.",
        ],
    }

    # both halo catalogs, per-bin mass-matched nu3 and nu4 (the bins with a
    # defensible absolute anchor per p5 acceptance)
    catalogs = {
        "remeasured_d1ce1133": p5["absolute_branch_BOTH_CATALOGS"][
            "remeasured_d1ce1133"]["per_bin_mass_matched"],
        "run0000_15481cba": p5["absolute_branch_BOTH_CATALOGS"][
            "run0000_15481cba"]["per_bin_mass_matched"],
    }

    for cat, block in catalogs.items():
        out["per_bin"][cat] = {}
        for nu in ("nu3", "nu4"):
            fbar_tng = float(block[nu]["ftilde_bar_TNG"])
            fbar_data = float(block[nu]["ftilde_bar_data"])
            # the deepest the TNG band represents: fiducial f_bar reduced by
            # the posterior's strong-edge gas evacuation (gas-only, see caveat)
            fbar_edge = fbar_tng * np.exp(dln_edge)

            dP_tng = float(vd20_dP_P(fbar_tng, d, e))
            dP_edge = float(vd20_dP_P(fbar_edge, d, e))
            dP_data = float(vd20_dP_P(fbar_data, d, e))

            # absorbed fraction: the band removes at most down to its edge; the
            # rest survives. Recover the within-family 93% when data==edge.
            depth_ratio = dP_data / dP_edge                  # >1 off-manifold
            absorbed = WITHIN_FAMILY_ABSORBED / depth_ratio
            residual_frac = 1.0 - absorbed

            out["per_bin"][cat][nu] = {
                "ftilde_bar_TNG": round(fbar_tng, 4),
                "ftilde_bar_band_edge": round(float(fbar_edge), 4),
                "ftilde_bar_data": round(fbar_data, 4),
                "dP_P_k0p5_TNG_fiducial": round(dP_tng, 5),
                "dP_P_k0p5_band_edge": round(dP_edge, 5),
                "dP_P_k0p5_data_offmanifold": round(dP_data, 5),
                "depth_ratio_data_over_bandedge": round(float(depth_ratio), 2),
                "absorbed_fraction_offmanifold": round(float(absorbed), 3),
                "residual_fraction_offmanifold": round(float(residual_frac), 3),
            }

    # the conservative-after-nuisance branch (deficit partly absorbed by
    # sigma8 / sigma_pos nuisances) -- the LEAST off-manifold case p5 carries
    cons = {}
    for cat in ("remeasured_d1ce1133", "run0000_15481cba"):
        branch = p5["conservative_after_nuisance_absorbers"][cat]
        cons[cat] = {}
        for scen in ("sigma8_DESY3_0p776", "sigma8_x_sigmapos"):
            per_bin = branch[scen]["per_bin"]
            row = per_bin[2]                     # nu3 (index 2), the anchor bin
            if not isinstance(row, dict):
                continue
            fbar_data = float(row["ftilde_bar_data"])
            fbar_tng = float(catalogs[cat]["nu3"]["ftilde_bar_TNG"])
            fbar_edge = fbar_tng * np.exp(dln_edge)
            dP_edge = float(vd20_dP_P(fbar_edge, d, e))
            dP_data = float(vd20_dP_P(fbar_data, d, e))
            depth_ratio = dP_data / dP_edge
            absorbed = WITHIN_FAMILY_ABSORBED / max(depth_ratio, 1.0)
            cons[cat][scen] = {
                "ftilde_bar_data": round(fbar_data, 4),
                "dP_P_k0p5_data": round(dP_data, 5),
                "depth_ratio_data_over_bandedge": round(float(depth_ratio), 2),
                "absorbed_fraction": round(float(min(absorbed, 1.0)), 3),
            }
    out["conservative_after_nuisance_nu3"] = cons

    WP7.mkdir(exist_ok=True)
    (WP7 / "s8_offmanifold.json").write_text(json.dumps(out, indent=2))

    # console summary
    print(f"within-family (INSIDE band) absorbed: "
          f"{WITHIN_FAMILY_ABSORBED:.1%}")
    print(f"posterior strong edge dln_mgas = {dln_edge:+.3f}  "
          f"(data require ~-1.22)")
    print("\noff-manifold absorbed fraction (data at the required f_bar):")
    for cat, bins in out["per_bin"].items():
        for nu, r in bins.items():
            print(f"  {cat:20s} {nu}: data dP/P={r['dP_P_k0p5_data_offmanifold']:+.3f}"
                  f"  band-edge dP/P={r['dP_P_k0p5_band_edge']:+.4f}"
                  f"  -> absorbed {r['absorbed_fraction_offmanifold']:.1%}"
                  f"  (residual {r['residual_fraction_offmanifold']:.0%})")
    print("\nconservative (after sigma8/sigma_pos nuisance absorption), nu3:")
    for cat, scen in out["conservative_after_nuisance_nu3"].items():
        for s, r in scen.items():
            print(f"  {cat:20s} {s:18s}: dP/P={r['dP_P_k0p5_data']:+.3f}"
                  f"  -> absorbed {r['absorbed_fraction']:.1%}")
    print(f"\nwrote {WP7}/s8_offmanifold.json")


if __name__ == "__main__":
    main()
