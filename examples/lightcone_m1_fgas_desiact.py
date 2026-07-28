#!/usr/bin/env python
"""P6b Fig M1 (docs/ksz_lightcone_map_plan.md §3-P6 stage B) -- THE headline:
map-level f~gas(theta) vs the real Ried Guachalla+25 DESI(BGS) x ACT kSZ
data, replacing/complementing the per-halo money plot
(``examples/_build_ksz_paper_nb.py`` §6, ``f6_ksz_confront``) with the
ray-traced-lightcone observable.

**The observable.** f_gas(theta_d) = CAP_tau(gas; theta_d) / CAP_mat(theta_d)
(constants cancel, see ``lightcone_capmat_merge.py``); f~gas = f_gas / F_B,
F_B = Omega_b/Omega_m. CAP_tau(gas) comes from the per-node, per-realization
fiducial+SB35 shards merged into ``taucap_lightcone.npz`` (P5/P6a). CAP_mat
is now a **per-node** realization-mean matter stack (``capmat_lightcone.npz``,
CORRECTED 2026-07-24 -- see that module's docstring): the massplane FIELD is
fiducial-shared (only one co-trace exists, total mass at a fixed halo is
feedback-insensitive) but it is stacked at EACH node's OWN M*-selected
galaxy positions, exactly mirroring the numerator's own per-node selection.

**Defect-and-fix note (read before trusting any number from an earlier run
of this script).** The first version of this script divided every node's
own CAP_tau by the FIDUCIAL's CAP_mat (the fiducial's own, much larger and
less mass-skewed BGS sample). Because each Sobol node's M*-cut selects a
DIFFERENT subset of halos (feedback changes which halos cross the stellar-
mass threshold), extreme-feedback nodes can select a tiny, very-massive-
skewed sample (e.g. node 64: 11 halos, mean logM200=14.38, vs the fiducial's
1753 halos at 13.44) -- dividing that node's own (correctly large) CAP_tau by
the fiducial's (much smaller, wrong-mass) CAP_mat inflated f~gas to
unphysical values (4-16) for such nodes. Fixed by computing CAP_mat per-node
(``lightcone_capmat_merge.py``); verified below via ``cosmic_crossing_check``
and the corrected envelope range.

**theta grid.** The shards' dimensionless ``xb`` grid (theta_d/theta200,
[0.3,3.0]) is mapped to arcmin via the FIDUCIAL sample's MEAN theta200
(``r200_comoving/chi[plane]``, in arcmin -- computed once per sample,
reusing ``lightcone_m2_ycap_liu.mean_theta200_arcmin``, the SAME convention
the plan specifies for M2), applied uniformly to the fiducial and all 253
SB35 node curves. This mirrors the aperture-mapping convention that P6a
already established and documented for M2.

**kSZ-consistency classification -- TWO variants, both reported.** For every
node k, interpolate its f~gas(theta) onto the DATA's own theta grid
(``KS/desact_zenodo/Fig8_BGS_BRIGHT-20.2_logm{11.00,11.25}.npz``, keys
``th,ratio,yerr,cov_ksz``), restrict to the 1-halo regime
theta <= 1.4*theta(r200), and compute
``chi2[k] = resid @ inv(cov_total[k]) @ resid`` with two choices of
``cov_total[k] = cov_ksz (+) cov_map_variant[k]``:

  (a) **survey-variance** -- ``cov_map_variant = cov_map`` (this node's own
      50-realization f~gas covariance, i.e. the scatter of a SINGLE 25deg^2
      realization's stack). Answers "would BIND be excluded if DESI's data
      were only as precise as one simulated lightcone realization" -- a
      much weaker test than DESI's actual precision, because a single
      realization only has ~40-90 galaxies landing in it (vs DESI's real,
      far larger sky footprint).
  (b) **DESI-precision** (**the headline, apples-to-apples vs the per-halo
      49/256, 65/256**) -- ``cov_map_variant = cov_map / n_real_used[k]``,
      the standard-error-of-the-mean of the realization covariance (Var of
      an R-draw average = Var(single draw)/R). This approximates the
      precision of a measurement that averages over many effectively-
      independent realizations/sky patches -- much closer in spirit to
      DESI's actual large-footprint precision, and to what the per-halo
      analysis's own bootstrap-based covariance represents.

Both ``cov_map`` variants get a small diagonal shrinkage (10% toward the
diagonal) purely as a numerical safeguard against near-singular estimates
from ~50 draws over ~6-7 theta bins. A node is *consistent* under the SAME
rule as the per-halo analysis: chi2 < N_dof + 2*sqrt(2*N_dof).

Two panels (bgs110, bgs1125); node coloring uses the DESI-precision (b)
classification (the headline); both (a) and (b) counts are annotated in the
title/caption alongside the per-halo 49/256, 65/256 for direct comparison.

Usage
-----
    python examples/lightcone_m1_fgas_desiact.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightcone_m2_ycap_liu import mean_theta200_arcmin, N_XB

CEPH = Path("/mnt/home/mlee1/ceph")
# Products root (Round-2 T3, docs/paper_improvement_plan.md): env-var
# override only (this script's own CLI is a bare sys.argv "--skip-fig"
# check, not argparse); behavior with $BIND_KSZ_PRODUCTS unset is
# byte-identical to before.
KS = Path(os.environ.get("BIND_KSZ_PRODUCTS", str(CEPH / "bind_science/ksz_confront")))
LIGHTCONE = KS / "lightcone"
FIG_DIR = LIGHTCONE / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ZEN = KS / "desact_zenodo"

SAMPLES = ["bgs110", "bgs1125"]
DCUT = {"bgs110": "11.00", "bgs1125": "11.25"}
SAMPLE_LABEL = {"bgs110": r"BGS $M_\star{>}10^{11.0}$", "bgs1125": r"BGS $M_\star{>}10^{11.25}$"}
PERHALO_CONSISTENT = {"bgs110": (49, 256), "bgs1125": (65, 256)}

SHRINK_ALPHA = 0.10  # diagonal-shrinkage fraction applied to each node's cov_map
MIN_REAL = 10


def shrink(cov: np.ndarray, alpha: float = SHRINK_ALPHA) -> np.ndarray:
    diag = np.diag(np.diag(cov))
    return (1 - alpha) * cov + alpha * diag


def main(skip_fig: bool = False):
    """Build Fig M1 and the per-node metrics.

    ``skip_fig=True`` (or CLI ``--skip-fig``) still runs every chi2/consistency
    computation (needed to populate ``ksz_consistent_nodes.npz``) but skips the
    final ``fig.savefig`` -- use it to refresh the node-list product without
    touching ``figs/M1_fgas_vs_desiact.png``.
    """
    taucap = np.load(LIGHTCONE / "taucap_lightcone.npz", allow_pickle=True)
    capmat = np.load(LIGHTCONE / "capmat_lightcone.npz", allow_pickle=True)
    F_B = float(capmat["F_B"])
    node_ids = taucap["node_ids"]
    assert np.array_equal(node_ids, capmat["node_ids"]), "taucap/capmat node_ids must align"

    theta200 = {s: mean_theta200_arcmin(s) for s in SAMPLES}
    print(f"[M1] mean theta200 (fiducial, arcmin): {theta200}")

    metrics = {"theta200_fid_arcmin": theta200, "F_B": F_B, "shrink_alpha": SHRINK_ALPHA}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))

    consistent_counts = {}
    fgas_at_theta200 = {}
    chi2_desi_by_sample = {}          # sample -> (253,) chi2, NaN for empty-sample/invalid nodes
    node_ids_consistent_by_sample = {}  # sample -> node_ids of the DESI-precision-consistent subset

    for ax, sample in zip(axes, SAMPLES):
        dz = np.load(ZEN / f"Fig8_BGS_BRIGHT-20.2_logm{DCUT[sample]}.npz")
        th_data, ratio_data, yerr_data, cov_ksz = dz["th"], dz["ratio"], dz["yerr"], dz["cov_ksz"]
        thr = theta200[sample]
        m1 = th_data <= 1.4 * thr
        ndof = int(m1.sum())

        xb = taucap["theta_value"][:N_XB]
        node_theta = xb * thr  # arcmin, shared by fiducial + all SB35 nodes

        # --- per-node denominator (CORRECTED: was fiducial-shared) ---
        mat_mean_node = capmat[f"sb35_mean_{sample}"][:, :N_XB].astype(np.float64)   # (253, 18)
        mat_mean_fid = capmat[f"fid_mean_{sample}"][:N_XB].astype(np.float64)        # (18,)

        fid_tau_mean = taucap[f"fid_mean_{sample}"][:N_XB]
        fid_tau_real = taucap[f"fid_real_{sample}"][:, :N_XB]  # (50, 18)
        fid_fgas_mean = fid_tau_mean / mat_mean_fid / F_B
        fid_fgas_real = fid_tau_real / mat_mean_fid[None, :] / F_B

        sb35_tau_mean = taucap[f"sb35_mean_{sample}"][:, :N_XB]     # (253, 18)
        sb35_tau_real = taucap[f"sb35_real_{sample}"][:, :, :N_XB]  # (253, 50, 18)
        with np.errstate(invalid="ignore", divide="ignore"):
            sb35_fgas_mean = sb35_tau_mean / mat_mean_node / F_B
            sb35_fgas_real = sb35_tau_real / mat_mean_node[:, None, :] / F_B

        # Fallback covariance (fiducial's own realization scatter, restricted to
        # the same m1 theta bins) for the rare node whose realizations are too
        # NaN-riddled (extreme-feedback nodes with very few landing galaxies in
        # some of the 50 realizations -- an entire-realization NaN, not a
        # per-theta pixel-count issue) to form its own covariance estimate.
        fid_real_m1 = np.array([np.interp(th_data[m1], node_theta, fid_fgas_real[r])
                                 for r in range(fid_fgas_real.shape[0])])
        cov_fid_fallback = np.cov(fid_real_m1, rowvar=False)

        n_nodes = sb35_fgas_mean.shape[0]
        chi2_survey = np.full(n_nodes, np.nan)
        chi2_desi = np.full(n_nodes, np.nan)
        n_real_used = np.full(n_nodes, -1, dtype=np.int64)
        n_fallback = 0
        for k in range(n_nodes):
            if not np.all(np.isfinite(sb35_fgas_mean[k])):
                continue  # empty-sample node (bgs1125 runs 64/87) or capmat n_gal=0
            pn = np.interp(th_data, node_theta, sb35_fgas_mean[k])
            real_at_data_m1 = np.array([
                np.interp(th_data[m1], node_theta, sb35_fgas_real[k, r]) for r in range(50)
            ])  # (50, ndof) -- only the m1 (1-halo-regime) columns are ever used
            finite = np.all(np.isfinite(real_at_data_m1), axis=1)
            n_real_used[k] = int(finite.sum())
            if finite.sum() >= MIN_REAL:
                cov_map = np.cov(real_at_data_m1[finite], rowvar=False)
                n_r = int(finite.sum())
            else:
                cov_map = cov_fid_fallback  # extreme-feedback node, too few valid realizations
                n_r = fid_real_m1.shape[0]
                n_fallback += 1
            resid = pn[m1] - ratio_data[m1]

            # (a) survey-variance: single-realization covariance
            cov_a = cov_ksz[np.ix_(m1, m1)] + shrink(np.atleast_2d(cov_map))
            chi2_survey[k] = float(resid @ np.linalg.inv(cov_a) @ resid)

            # (b) DESI-precision (headline): covariance of the realization MEAN
            cov_b = cov_ksz[np.ix_(m1, m1)] + shrink(np.atleast_2d(cov_map / n_r))
            chi2_desi[k] = float(resid @ np.linalg.inv(cov_b) @ resid)

        consistent_survey = np.isfinite(chi2_survey) & (chi2_survey < ndof + 2 * np.sqrt(2 * ndof))
        consistent_desi = np.isfinite(chi2_desi) & (chi2_desi < ndof + 2 * np.sqrt(2 * ndof))
        n_map_survey = int(consistent_survey.sum())
        n_map_desi = int(consistent_desi.sum())
        n_valid = int(np.isfinite(chi2_desi).sum())

        # stash the DESI-precision (headline) chi2 + consistent node IDs for
        # ksz_consistent_nodes.npz (positionally aligned with node_ids, i.e.
        # node_ids_all, so downstream consumers never need a separate index map)
        chi2_desi_by_sample[sample] = chi2_desi.copy()
        node_ids_consistent_by_sample[sample] = node_ids[consistent_desi].copy()
        consistent_counts[sample] = {
            "survey_variance": {"n_consistent": n_map_survey, "n_valid_nodes": n_valid},
            "desi_precision": {"n_consistent": n_map_desi, "n_valid_nodes": n_valid},
            "n_dof": ndof, "n_dof_theta_arcmin_max": float(1.4 * thr),
            "n_nodes_using_fiducial_cov_fallback": int(n_fallback),
            "median_n_real_used_for_cov": float(np.median(n_real_used[n_real_used >= 0])),
        }

        fid_at_r200 = float(np.interp(thr, node_theta, fid_fgas_mean))
        data_at_r200 = float(np.interp(thr, th_data, ratio_data))
        fgas_at_theta200[sample] = {"bind_fiducial": fid_at_r200, "data": data_at_r200,
                                     "ratio": fid_at_r200 / data_at_r200}
        consistent_counts[sample]["fiducial_n_gal_landing_per_25deg2_realization"] = float(
            np.mean(np.load(LIGHTCONE / "shards" / f"{sample}_tau_runfid_beamnone_src4.npz")["n_gal_real"]))
        finite_env = sb35_fgas_mean[np.all(np.isfinite(sb35_fgas_mean), axis=1)]
        consistent_counts[sample]["sb35_envelope_min_max"] = [float(np.nanmin(finite_env)),
                                                                 float(np.nanmax(finite_env))]

        # ---------------- plotting (colored by the DESI-precision/headline variant) ----------------
        for k in range(n_nodes):
            if not np.all(np.isfinite(sb35_fgas_mean[k])):
                continue
            color = "tab:orange" if consistent_desi[k] else "tab:blue"
            zorder = 3 if consistent_desi[k] else 1
            alpha = 0.35 if consistent_desi[k] else 0.10
            lw = 0.6 if consistent_desi[k] else 0.4
            ax.plot(node_theta, sb35_fgas_mean[k], "-", color=color, lw=lw, alpha=alpha, zorder=zorder)
        ax.plot([], [], "-", color="tab:blue", lw=1.0, alpha=0.5, label=f"SB35 ({n_valid})")
        ax.plot([], [], "-", color="tab:orange", lw=1.2, alpha=0.8,
                label=f"kSZ-consistent, DESI-precision ({n_map_desi})")

        lo16, hi84 = np.nanpercentile(fid_fgas_real, [16, 84], axis=0)
        ax.fill_between(node_theta, lo16, hi84, color="0.4", alpha=0.25, zorder=4,
                         label="fid 16-84% (50 real.)")
        ax.plot(node_theta, fid_fgas_mean, "-", color="k", lw=2.4, zorder=5, label="BIND fiducial")

        if n_valid:
            bi = int(np.nanargmin(chi2_desi))
            ax.plot(node_theta, sb35_fgas_mean[bi], "-", color="darkred", lw=1.1, alpha=0.9,
                    zorder=6, label=f"best-fit node (chi2_desi={chi2_desi[bi]:.1f})")

        ax.errorbar(th_data, ratio_data, yerr=yerr_data, fmt="o", color="k", ms=5, capsize=2,
                    zorder=7, label="Ried Guachalla+25 (real)")
        ax.axvline(thr, color="0.3", ls=":", lw=0.8)
        ax.axhline(1, color="k", ls=":", lw=0.7, alpha=0.6)
        ax.text(0.985, 1.01, "cosmic", transform=ax.get_yaxis_transform(), fontsize=6.5,
                ha="right", va="bottom", color="0.3")
        ax.set_xlabel(r"$\theta$ [arcmin]  (dotted = $\theta(r_{200})$)")
        ax.set_ylabel(r"$\tilde f_{\rm gas}$")
        ax.set_xlim(0, 11)
        ax.set_ylim(bottom=min(0, np.nanmin(lo16) * 1.1))
        perhalo_n, perhalo_N = PERHALO_CONSISTENT[sample]
        ax.set_title(f"{SAMPLE_LABEL[sample]}\nDESI-precision {n_map_desi}/{n_valid} "
                     f"(cf. per-halo {perhalo_n}/{perhalo_N})  |  survey-var. {n_map_survey}/{n_valid}",
                     fontsize=9.5)
        ax.legend(fontsize=6.3, loc="upper left", framealpha=0.9)

    metrics["consistent_counts"] = consistent_counts
    metrics["fgas_at_theta200"] = fgas_at_theta200
    metrics["perhalo_consistent"] = PERHALO_CONSISTENT

    # ---------------------------------------------------------------
    # ksz_consistent_nodes.npz -- the DESI-precision (headline) kSZ-consistent
    # node sets, factored out of the figure so other consumers (the M6 param-
    # constraint figure, the upcoming DES stream) don't need to re-derive the
    # chi2/consistency logic above. Positionally aligned: node_ids_all[i] is
    # the run ID for chi2_desi_bgs110[i] / chi2_desi_bgs1125[i] (both length
    # 253; bgs1125 has NaN at its 2 empty-M*-sample nodes -- runs 64/87 --
    # leaving 251 finite entries, matching the P6b headline "48/251").
    # ---------------------------------------------------------------
    nodes_readme = (
        "DESI-precision kSZ-consistent SB35 node sets from the map-level M1 "
        "f~gas(theta) confrontation (docs/ksz_lightcone_map_plan.md P6b; "
        "verdict KS/lightcone/verdicts/P6b.json). Built by "
        "examples/lightcone_m1_fgas_desiact.py::main(). "
        "chi2_total = cov_ksz + cov_map/n_real_used (realization-mean covariance, "
        "the apples-to-apples precision vs the per-halo 49/256, 65/256 test), "
        "restricted to the 1-halo regime theta <= 1.4*theta(r200); consistent "
        "iff chi2 < n_dof + 2*sqrt(2*n_dof). "
        "node_ids_all (253,): traced SB35 run IDs (0..255, excluding 114/115/117 "
        "-- never ray-traced, see P5 verdict), positionally aligned with "
        "chi2_desi_bgs110/chi2_desi_bgs1125. "
        "node_ids_bgs110 (31,) / node_ids_bgs1125 (48,): run IDs consistent under "
        "the BGS M*>10^11.00 / M*>10^11.25 cut respectively (DESI-precision "
        "variant only -- the weaker survey-variance variant, 234/253 & 246/251, "
        "is not stored here, see P6b.json for those counts). "
        "chi2_desi_bgs110 (253,): finite for all 253 nodes. "
        "chi2_desi_bgs1125 (253,) with 2 NaN entries (runs 64/87, zero galaxies "
        "at the M*>10^11.25 cut in this node) -- 251 finite values."
    )
    nodes_path = LIGHTCONE / "ksz_consistent_nodes.npz"
    np.savez(
        nodes_path,
        node_ids_all=node_ids,
        node_ids_bgs110=node_ids_consistent_by_sample["bgs110"],
        node_ids_bgs1125=node_ids_consistent_by_sample["bgs1125"],
        chi2_desi_bgs110=chi2_desi_by_sample["bgs110"],
        chi2_desi_bgs1125=chi2_desi_by_sample["bgs1125"],
        readme=nodes_readme,
    )
    print(f"[M1] wrote {nodes_path} "
          f"(bgs110 {node_ids_consistent_by_sample['bgs110'].size}/{node_ids.size}, "
          f"bgs1125 {node_ids_consistent_by_sample['bgs1125'].size}/"
          f"{int(np.isfinite(chi2_desi_by_sample['bgs1125']).sum())})")

    # Transparency check (mirrors the trusted per-halo plot's "cosmic" f~gas=1
    # asymptote annotation): the CAP-ratio observable is NOT bounded by 1 --
    # at large aperture (theta >> theta(r200)) it approaches/exceeds the
    # cosmic baryon fraction as the aperture picks up uncorrelated LOS + 2-halo
    # gas that is no longer a clean 1-halo measurement (this is the documented
    # P4 "smooth 2-halo excess" feature, not a units bug). Record where the
    # FIDUCIAL mean profile (not the noisier individual nodes) first crosses 1.
    cosmic_crossing = {}
    for sample in SAMPLES:
        mat_mean_fid = capmat[f"fid_mean_{sample}"][:N_XB].astype(np.float64)
        xb = taucap["theta_value"][:N_XB]
        fid_fgas_mean = taucap[f"fid_mean_{sample}"][:N_XB] / mat_mean_fid / F_B
        above = xb[fid_fgas_mean > 1.0]
        cosmic_crossing[sample] = {
            "fiducial_first_xb_above_cosmic": float(above.min()) if len(above) else None,
            "one_halo_regime_xb_cutoff": 1.4,
            "fiducial_max_within_1halo_regime": float(np.max(fid_fgas_mean[xb <= 1.4])),
        }
    metrics["cosmic_crossing_check"] = cosmic_crossing

    fig.suptitle(r"M1 -- map-level $\tilde f_{\rm gas}(\theta)$ vs DESI(BGS)$\times$ACT kSZ "
                 "(Ried Guachalla+25)", fontsize=12.5, fontweight="bold", y=1.01)
    caption = (
        "Per-node CAP_mat denominator (bug-fixed 2026-07-24, see module docstring): each node's "
        "f~gas uses ITS OWN M*-selected sample for both CAP_tau and CAP_mat, not the fiducial's. "
        "TWO chi2 variants, both shown: (a) survey-variance = cov_ksz + this node's own single-"
        "realization f~gas covariance (a weak test -- one 25deg^2 realization has only ~40-90 "
        "landing galaxies, far below DESI's real footprint); (b) DESI-precision (HEADLINE, colors "
        "the lines here) = cov_ksz + covariance of the realization MEAN (cov_map/n_real) -- the "
        "apples-to-apples number vs the per-halo 49/256, 65/256. Denominators differ (253 map-level "
        "nodes traced vs 256 per-halo nodes painted) -- fractions, not raw counts, are comparable."
    )
    import textwrap
    fig.text(0.5, -0.03, "\n".join(textwrap.wrap(caption, width=175)), ha="center", va="top", fontsize=7.5)
    fig.tight_layout(rect=[0, 0.01, 1, 0.90])
    if not skip_fig:
        fig_path = FIG_DIR / "M1_fgas_vs_desiact.png"
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        print(f"[M1] wrote {fig_path}")
    plt.close(fig)

    print(json.dumps(metrics, indent=2, default=float))
    return metrics


if __name__ == "__main__":
    main(skip_fig="--skip-fig" in sys.argv)
