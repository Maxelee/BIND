"""WP-B6: build the paper-3B data-product release bundle.

Curates every load-bearing (and small) frozen artifact of the B chain into a
single directory that ships alongside the paper — the walkthrough notebook
(`analysis/paper3b/notebooks/paperB_walkthrough.ipynb`) runs against it and
regenerates every figure. Bulk inputs (survey maps, the twobound atlas, the
per-unit grids) are NOT bundled: the maps are public downloads (DES Y3 mass
maps, ACT DR6 y, ACT DR5 catalog) and the atlas statistics enter only through
the bundled model-grid tables.

Writes: <out>/paper3b_products_v1/{wp1_maps,...,wp5_inference}/...,
MANIFEST.json (per-file sha256 + one-line description), SHA256SUMS, README.md.
Verifies the four signed-freeze hashes match `stack.frozen.FROZEN_HASHES`
before sealing the bundle.

    python -m analysis.paper3b.scripts.make_release_bundle
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

_repo = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(_repo))

from analysis.paper3b.stack.frozen import FROZEN_HASHES  # noqa: E402

B_ROOT = Path("/mnt/home/mlee1/ceph/paper3/B")
OUT = B_ROOT / "release_bundle" / "paper3b_products_v1"

# relpath -> one-line description (order = manifest order)
FILES = {
    # -- WP-B1: maps, footprint, peak catalogs --------------------------------
    "wp1_maps/common_footprint_nside1024.npz":
        "Harmonized DES Y3 x ACT DR6 footprint: apodized weight + conservative binary mask (Nside 1024)",
    "wp1_maps/peaks_wiener_sm2am.npz":
        "Frozen Wiener peak catalog at the fiducial 2' smoothing (positions, kappa_sm, nu; PEAK_DEFINITION chain)",
    "wp1_maps/peaks_glimpse_sm2am.npz":
        "GLIMPSE cross-check peak catalog at 2' smoothing",
    "wp1_maps/peaks_wiener_sm1am.npz": "Wiener peaks, 1' robustness smoothing",
    "wp1_maps/peaks_wiener_sm5am.npz": "Wiener peaks, 5' robustness smoothing",
    "wp1_maps/peaks_wiener_sm8am.npz": "Wiener peaks, 8' robustness smoothing",
    "wp1_maps/peaks_glimpse_sm1am.npz": "GLIMPSE peaks, 1' robustness smoothing",
    "wp1_maps/peaks_glimpse_sm5am.npz": "GLIMPSE peaks, 5' robustness smoothing",
    "wp1_maps/peaks_glimpse_sm8am.npz": "GLIMPSE peaks, 8' robustness smoothing",
    "wp1_maps/peak_catalog_summary.json":
        "Per-catalog counts / exclusions / nu-stats table",
    "wp1_maps/dr5_stack_profiles.npz":
        "ACT DR5 cluster-stack sanity anchor (units/beam check of the y-map chain)",
    "wp1_maps/dr5_stack_summary.json": "DR5 anchor summary numbers",
    # -- WP-B2: the frozen measurement ---------------------------------------
    "wp2_measurement/stack_wiener_sm2am_fid.npz":
        "FROZEN headline measurement: <Y_CAP> at Wiener peaks, 5 nu bins x 5 CAP radii + jackknife covs + stacked thumbnails (freeze-signed)",
    "wp2_measurement/stack_glimpse_sm2am_fid.npz":
        "FROZEN GLIMPSE cross-check stack (freeze-signed)",
    "wp2_measurement/stack_wiener_sm2am_deprojcib.npz":
        "CIB-deprojected y-map variant stack (Sigma_sys input)",
    "wp2_measurement/stack_glimpse_sm2am_deprojcib.npz":
        "CIB-deprojected variant, GLIMPSE",
    "wp2_measurement/stack_wiener_sm2am_nullpos.npz":
        "Random-position null stack (S1 battery input)",
    "wp2_measurement/stack_glimpse_sm2am_nullpos.npz":
        "Random-position null, GLIMPSE",
    "wp2_measurement/measurement_summary.json": "Per-run S/N summary table",
    # -- WP-B3: null battery + systematics budget ----------------------------
    "wp3_nulls/b3_scorecard.json":
        "Null-battery scorecard S1-S6 vs pre-registered criteria (NULL_CRITERIA.md)",
    "wp3_nulls/battery_summary.json": "Battery + CIB-band summary",
    "wp3_nulls/inline_tests.json": "Inline null-test numbers",
    "wp3_nulls/sigma_sys_wiener_decomposed.npz":
        "FROZEN Sigma_sys: decomposed rank-1 CIB term, Wiener (freeze-signed)",
    "wp3_nulls/sigma_sys_glimpse_decomposed.npz":
        "FROZEN Sigma_sys, GLIMPSE (freeze-signed)",
    "wp3_nulls/sigma_sys_wiener_strict.npz":
        "Strict (envelope) Sigma_sys variant, Wiener — the rejected broader convention, kept for comparison",
    "wp3_nulls/sigma_sys_glimpse_strict.npz": "Strict Sigma_sys, GLIMPSE",
    "wp3_nulls/ensemble_wiener.npz": "S1 random-position ensemble stats, Wiener",
    "wp3_nulls/ensemble_glimpse.npz": "S1 ensemble, GLIMPSE",
    "wp3_nulls/peaks_bmode_sm2am.npz": "S3 B-mode-map peak catalog",
    "wp3_nulls/sigma_stat_crossbin_wiener.npz":
        "Cross-bin jackknife supplement (off-diagonal robustness of the rejection)",
    "wp3_nulls/star_dust_summary.json":
        "Ladder item 2: DES star-density/PSF-residual + SFD dust covariate tests (section 7/8)",
    # -- WP-B4: forward model ------------------------------------------------
    "wp4_mocks/weights_desy3.npz":
        "DES Y3 n(z) -> source-plane weights (default/variant8/plain schemes) + fit residuals",
    "wp4_mocks/transfer_desy3.npz":
        "Empirical reconstruction transfer T(ell) for Wiener + GLIMPSE (kappa 2-pt only; B5 design decision)",
    "wp4_mocks/b5_transfer_summary.json":
        "Transfer validation: nu>=4 abundance closure numbers",
    "wp4_mocks/model_grid.npz":
        "Intrinsic-convention model grid table (pre-transfer-decision, kept for comparison)",
    "wp4_mocks/model_grid_tfwiener.npz":
        "FROZEN comparison grid: 60 twobound units, transfer-matched, n_seeds=32 (MODEL_FREEZE-signed)",
    "wp4_mocks/model_grid_tfwiener_pg1024.npz":
        "Ladder item 1 variant: coarse-grid peak-FINDING convention grid",
    "wp4_mocks/grid_tfwiener/bind_run_0000.npz":
        "The bind-fiducial anchor unit incl. per-peak (nu, Y) tables + mock conventions (neff/sigma_e/beam/smoothing)",
    "wp4_mocks/grid_tfwiener_pg1024/bind_run_0000.npz":
        "Fiducial anchor under the coarse-grid peak-FINDING convention (ladder item 1)",
    "wp4_mocks/b4_summary.json": "Intrinsic grid R2 selection validation",
    "wp4_mocks/b4_summary_tfwiener.json": "FINAL tf grid R2 validation + MC budget",
    "wp4_mocks/b4_summary_tfwiener_pg1024.json": "pg1024 variant validation",
    "wp4_mocks/b4_summary_tfglimpse.json": "GLIMPSE-transfer qualitative cross-check",
    "wp4_mocks/b4_summary_tfwiener_plain.json": "plane-weight variant (movement table)",
    "wp4_mocks/b4_summary_tfwiener_variant8.json": "8-shell weight variant (movement table)",
    # -- WP-B5: inference + interpretation ladder ----------------------------
    "wp5_inference/b5_loo.json": "GridEmulator leave-one-out validation (61 refits)",
    "wp5_inference/b5_recovery.json":
        "Pre-registered injection-recovery gate (run BEFORE any data look)",
    "wp5_inference/b5_fit_wiener.json": "The B5 fit: data vector, posterior moments, edge mass",
    "wp5_inference/b5_posterior_wiener.npz":
        "Posterior grid over (Delta ln M_gas, Delta ln T)",
    "wp5_inference/b5_loo_pg1024.json": "LOO on the pg1024 variant grid",
    "wp5_inference/b5_recovery_pg1024.json": "Recovery gate, pg1024 variant",
    "wp5_inference/b5_fit_wiener_pg1024.json": "Ladder item 1 re-fit (deficit unchanged)",
    "wp5_inference/b5_posterior_wiener_pg1024.npz": "pg1024 posterior grid",
    "wp5_inference/b5_hunt_record.json":
        "Post-unblinding systematic-hunt record (ordering disclosed)",
    "wp5_inference/posscatter/posscatter_summary.json":
        "Ladder item 3: positional-scatter dilution analysis (required sigma, bounds, residuals)",
    "wp5_inference/mbias/mbias_summary.json":
        "Ladder item 7: shear m-bias exact-cancellation demonstration",
    "wp5_inference/transfer_movement.json":
        "Ladder item 4: transfer-shape movement numbers (NULL: <=8% lever, jobs 2451614/15)",
    "wp5_inference/photo_anchor/photo_anchor_summary.json":
        "Ladder item 5: absolute photometric anchor vs Liu et al. 2025 (DESI LRG x same DR6 NILC y)",
    "wp5_inference/b5_chi2_supplement.json":
        "Every chi2 quoted in the WP5 record with its explicit covariance recipe (each reproduces exactly)",
    "wp4_mocks/sobol/b5_sobol_verdict.json":
        "Ladder item 8 pre-registered verdict: full-box rejection (chi2_min=339.6/4 over 253 Sobol units)",
    "wp4_mocks/sobol/model_grid_tfwiener_sb35.npz":
        "253-run SB35 Sobol model table through the identical transfer-matched chain",
    "wp4_mocks/sobol/b4_summary_tfwiener.json":
        "Sobol-tree assembly summary (R2 selection validation re-ran inside: PASS)",
    "wp4_mocks/sobol/sb35_coords.npz":
        "Per-run (delta_ln_Mgas, delta_ln_T) coordinates + raw means + reference provenance",
    "wp5_inference/b5_item6_cosmology_bracket.json":
        "Ladder item 6: analytic sigma8/cosmology bracket (absorbs <=1.5x plausible; cannot null)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, missing = {}, []
    for rel, desc in FILES.items():
        src = B_ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[rel] = {"sha256": sha256(dst), "bytes": dst.stat().st_size,
                         "description": desc}
    # the signed-freeze gate travels with the bundle
    for rel, want in FROZEN_HASHES.items():
        got = manifest.get(rel, {}).get("sha256")
        assert got == want, f"FREEZE HASH MISMATCH in bundle: {rel}"
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump({"version": "paper3b_products_v1",
                   "frozen_hashes": FROZEN_HASHES, "files": manifest}, f, indent=1)
    with open(OUT / "SHA256SUMS", "w") as f:
        for rel, m in manifest.items():
            f.write(f"{m['sha256']}  {rel}\n")
    (OUT / "README.md").write_text(
        "# Paper 3B data products (v1)\n\n"
        "Frozen data products of the DES Y3 kappa-peak x ACT DR6 Compton-y\n"
        "stacking analysis. `paperB_walkthrough.ipynb` (in the BIND repo,\n"
        "`analysis/paper3b/notebooks/`) documents the full chain and\n"
        "regenerates every figure from these files: point the environment\n"
        "variable `PAPER3B_PRODUCTS` at this directory and run all cells.\n\n"
        "Integrity: `sha256sum -c SHA256SUMS`; the measurement-freeze hashes\n"
        "are additionally pinned in MANIFEST.json (`frozen_hashes`) and\n"
        "re-verified inside the notebook.\n\n"
        "Bulk inputs are not included: DES Y3 mass maps (Jeffrey et al.),\n"
        "the ACT DR6 y map + DR5 cluster catalog, and the TNG-based mock\n"
        "atlas; download locations and provenance are in the notebook.\n")
    total = sum(m["bytes"] for m in manifest.values())
    print(f"bundle: {len(manifest)} files, {total/1e6:.1f} MB -> {OUT}")
    if missing:
        print("MISSING (not bundled):", missing)


if __name__ == "__main__":
    main()
