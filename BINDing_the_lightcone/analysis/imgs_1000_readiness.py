"""Which paper figures can be rebuilt on the N=1000 campaign, right now.

SUPERSEDED (2026-08-21) by ``make_imgs_1000.py`` in this directory -- the
driver carries per-figure gates that match each builder's REAL dependencies
(this script's TWOBOUND gate over-constrains figA3, its SOBOL gate does not
check dataset freshness, and it knows nothing about the yt/texture caches),
and it also BUILDS + promotes the figures.  Kept for the quick tree summary.

Every figure in ``BINDing_the_lightcone/imgs/`` is checked against the data the
N1000 tree actually holds today.  Figures fall into three classes:

  READY    every input has an n1000 twin that exists -> rebuild now
  WAITING  inputs exist in principle but the campaign has not produced them yet
           (runs still tracing, or their stats not computed) -> rebuild later
  BLOCKED  no n1000 counterpart exists at all (per-halo atlases/profiles, the
           assembled Sobol emulator_dataset, one-off full-hydro lux traces)
           -> needs new pipeline work, not a path swap

    python BINDing_the_lightcone/analysis/imgs_1000_readiness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")

N1K = Path("/mnt/home/mlee1/ceph/bind_n1000")
STATS = ("Cl_kappa.npz", "peak_counts.npz", "nongaussian_stats.npz")


def n_with_stats(cat: str) -> tuple[int, int]:
    """(runs with all stats products, runs traced) for a category."""
    traced = sorted(N1K.glob(f"{cat}/run_*/.trace_complete"))
    ok = sum(1 for p in traced if all((p.parent / s).exists() for s in STATS))
    return ok, len(traced)


# figure -> (class, requirement, note)
FIGS = {
    "fig01_pipeline_diagram":  ("BLOCKED", "per-halo composite patches at 3 Sobol nodes",
                                "single-draw showcase; realization count does not change it"),
    "fig02_hero":              ("MAP",     "dmo + fiducial kappa_maps", ""),
    "fig03_halo_validation":   ("BLOCKED", "halo_atlas per-halo cache", "no n1000 per-halo pipeline"),
    "fig04_radial_profiles":   ("BLOCKED", "profiles per-halo cache", "no n1000 per-halo pipeline"),
    "fig05_field_validation":  ("MAP",     "fiducial + truth field stats", "needs an n1000 field_cache"),
    "fig05n_field_validation_noisy": ("MAP", "fiducial maps + noisy grid", "not in current main.tex"),
    "fig06_spectra_validation": ("MAP",    "fiducial/truth tau_maps + Cl", ""),
    "fig07_detectability":     ("SOBOL",   "emulator_dataset (256 nodes)", ""),
    "fig08_sl_response":       ("SOBOL",   "emulator_dataset (256 nodes)", ""),
    "fig09_param_response":    ("SOBOL",   "emulator_dataset (256 nodes)", ""),
    "fig10_covariation":       ("SOBOL",   "fiducial Cl/tau + emulator_dataset", ""),
    "figA1_fullhydro_wl":      ("BLOCKED", "one-off full-TNG300-hydro lux trace",
                                "validation product, not a campaign statistic"),
    "figA2_fullhydro_gas":     ("BLOCKED", "same full-hydro trace", ""),
    "figA3_texture_debias":    ("TWOBOUND", "twobound replicas 18/49/53", ""),
    "figA4_gas_families":      ("TWOBOUND", "all 60 twobound paired_stats", ""),
    "pfig_fm_basis":           ("SOBOL",   "emulator_dataset + latent coeffs", ""),
    "pfig_fm_freeamp":         ("SOBOL",   "emulator_dataset", ""),
    "pfig_fm_model_curves":    ("SOBOL",   "emulator_dataset + design mean", ""),
    "pfig_fm_model_curves_gas": ("SOBOL",  "emulator_dataset + gas families", ""),
    "pfig_fm_search_path":     ("SOBOL",   "emulator_dataset", ""),
    "pfig_s3b_cl_clusters":    ("TWOBOUND", "all 60 twobound paired_stats", ""),
    "pfig_s3b_pdf_clusters":   ("TWOBOUND", "all 60 twobound paired_stats", ""),
}


def main() -> None:
    sb_ok, sb_tr = n_with_stats("sb35")
    tb_ok, tb_tr = n_with_stats("twobound")
    fid_ok = all((N1K / "twobound/run_0049" / s).exists() for s in STATS)
    core_ok = all((N1K / c / "run_0000" / s).exists() for c in ("bind", "dmo", "truth") for s in STATS)
    print("N1000 tree status:")
    print(f"  core (bind/dmo/truth) stats complete : {core_ok}")
    print(f"  paper fiducial twobound/run_0049     : {'stats ready' if fid_ok else 'NOT ready'}")
    print(f"  twobound runs with stats             : {tb_ok}/60   (traced {tb_tr})")
    print(f"  sb35 Sobol runs with stats           : {sb_ok}/256  (traced {sb_tr})")
    print(f"  assembled n1000 emulator_dataset     : "
          f"{'yes' if list(N1K.glob('emulator_dataset*.npz')) else 'NO — assembly step not yet run'}")
    print()
    verdict = {"READY": [], "WAITING": [], "BLOCKED": []}
    for fig, (cls, req, note) in FIGS.items():
        if cls == "BLOCKED":
            v = "BLOCKED"
        elif cls == "MAP":
            v = "READY" if core_ok and fid_ok else "WAITING"
        elif cls == "TWOBOUND":
            v = "READY" if tb_ok >= 60 else "WAITING"
        else:                                     # SOBOL
            v = "READY" if (sb_ok >= 256 and list(N1K.glob("emulator_dataset*.npz"))) else "WAITING"
        verdict[v].append((fig, req, note))
    for v in ("READY", "WAITING", "BLOCKED"):
        print(f"=== {v}  ({len(verdict[v])} figures)")
        for fig, req, note in verdict[v]:
            print(f"   {fig:32s} needs {req}" + (f"  [{note}]" if note else ""))
        print()
    n_ready = len(verdict["READY"])
    print(f"-> {n_ready}/{len(FIGS)} figures rebuildable on n1000 today; "
          f"{len(verdict['WAITING'])} waiting on the campaign; "
          f"{len(verdict['BLOCKED'])} need new pipeline work.")


if __name__ == "__main__":
    main()
