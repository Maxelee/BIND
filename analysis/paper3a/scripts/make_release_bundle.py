"""WP-A8: build the paper-3A data-product release bundle.

The A-side twin of `analysis/paper3b/scripts/make_release_bundle.py`,
deliberately mirroring its layout so a reader who has seen the B bundle
knows where to look in this one. Curates every load-bearing frozen
artifact of the A chain — the gas emulator, the statistics emulator, the
posterior chains, the propagated envelopes and the WP-A7 cosmology
tables — into a single directory that ships alongside the paper, with a
manifest, checksums, and a freeze gate.

FREEZE GATE (the A-side analogue of B's `FROZEN_HASHES`). Project A has
no single signed-hash constant; its freeze lives in the per-source
`wp1_data/<source>/provenance.json` records written at data assembly.
This script RE-VERIFIES every one of those sha256 records against the
raw files on disk before sealing, and copies the verified records into
the manifest. A mismatch aborts the build. So the bundle certifies that
it was constructed against the frozen inputs, even though it does not
redistribute them.

WHAT IS DELIBERATELY NOT BUNDLED. The raw WP1 observational inputs
(eROSITA eRASS1 catalogs, the Qu/Hadzhiyska kSZ profile sets, Pandey's
kappa-y vectors) are third-party data releases. We do not redistribute
them: we pin their hashes, cite them, and give the download URLs already
recorded in the provenance files. The SB35 run atlas and the lightcone
maps are bulk inputs that enter only through the bundled emulators and
statistics tables.

    python -m analysis.paper3a.scripts.make_release_bundle
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))

A_ROOT = Path("/mnt/ceph/users/mlee1/paper3/A")
OUT = A_ROOT / "release_bundle" / "paper3a_products_v1"

# relpath -> one-line description (order = manifest order)
FILES = {
    # -- WP-A1: the frozen data-vector index -----------------------------
    "wp1_data/provenance.json":
        "Directory-level index of the frozen WP1 observational sources "
        "(per-source provenance, hashes and licenses live alongside the raw "
        "files, which are third-party releases and are NOT redistributed here)",

    # -- WP-A2: observable-matching validation ---------------------------
    "wp2_validation/summary_snap096.json":
        "Cylinder-to-sphere / aperture-matching validation summary, z=0 snapshot",
    "wp2_validation/summary_snap071.json": "Same, snap 071",
    "wp2_validation/summary_snap067.json": "Same, snap 067",
    "wp2_validation/summary_snap063.json": "Same, snap 063",
    "wp4_emulator/cyltosph_theta_model.npz":
        "Fitted cylinder->sphere correction model used by the forward operator",

    # -- WP-A4: the gas-observable emulator ------------------------------
    "wp4_emulator/gasemu_gp_v4.npz":
        "FROZEN gas emulator (v4): per-snapshot PCA + ARD Matern-5/2 GPs over "
        "the 30-parameter CAMELS-TNG design",
    "wp4_emulator/gasemu_kfold_v4.npz":
        "v4 k-fold cross-validation predictions (the acceptance evidence)",
    "wp4_emulator/gasemu_outdesign_v4.npz":
        "v4 out-of-design validation set",
    "wp4_emulator/validation_summary_v4.json":
        "v4 validation scorecard vs the pre-registered acceptance criteria",
    "wp4_emulator/cyltosph_1p_summary.json":
        "1P-suite cylinder/sphere consistency table",

    # -- WP-A5: inference ------------------------------------------------
    "wp5_chains/a5_final_seed0.npz":
        "FROZEN f_gas-calibrated posterior chain, seed 0 (4 seeds; R-hat <= 1.001)",
    "wp5_chains/a5_final_seed1.npz": "f_gas posterior chain, seed 1",
    "wp5_chains/a5_final_seed2.npz": "f_gas posterior chain, seed 2",
    "wp5_chains/a5_final_seed3.npz": "f_gas posterior chain, seed 3",
    "wp5_chains/a5_final_summary.json":
        "f_gas posterior summary: marginals, MAP, per-block chi2",
    "wp5_chains/a5_final_ess.json": "Per-parameter ESS and convergence record",
    "wp5_chains/a5_fit_kszonly_summary.json":
        "kSZ-only fit summary (the individually-marginal probe: 10.8/9, edge-piled)",
    "wp5_chains/a5_fit_joint_summary.json":
        "A-side joint (f_gas + kSZ) fit summary: 55.3/14, p_pp = 0.000",
    "wp5_chains/a5_recovery.json":
        "Simulation-based recovery battery (the pre-registered gate that had to "
        "PASS before any data fit)",
    "wp5_chains/a5_recovery_kszonly.json": "Recovery battery, kSZ-only likelihood",
    "wp5_chains/a5_recovery_joint.json": "Recovery battery, A-side joint likelihood",
    "wp5_chains/a5_recovery_jointab.json":
        "Recovery battery for the joint A+B likelihood (4/4 summaries PASS)",
    "wp5_chains/a5_cov_stress.json":
        "kSZ covariance stress test (doubled systematics, importance reweighting)",
    "wp5_chains/a5_tension.json": "Probe-tension decomposition",
    "wp5_chains/a5_suppression.json": "Posterior suppression summary",
    "wp5_chains/a5_ksz_firstcontact.json": "kSZ first-contact diagnostic record",
    "wp5_chains/joint_ab_seed0.npz":
        "FROZEN joint A+B posterior chain, seed 0 — the four-probe result "
        "(750k samples across 4 seeds, R-hat <= 1.0011)",
    "wp5_chains/joint_ab_seed1.npz": "Joint A+B chain, seed 1",
    "wp5_chains/joint_ab_seed2.npz": "Joint A+B chain, seed 2",
    "wp5_chains/joint_ab_seed3.npz": "Joint A+B chain, seed 3",
    "wp5_chains/joint_ab_summary.json":
        "THE HEADLINE ARTIFACT: joint A+B posterior summary — per-block chi2 at "
        "the MAP, p_pp = 0.000, boundary tripwires, coordinate marginals",

    # -- WP-A6: propagation to WL statistics -----------------------------
    "wp6_propagation/statsemu_gp.npz":
        "FROZEN statistics emulator: 23 WL/tSZ targets over the SB35 Sobol "
        "design (log y-family transforms, 8-fold CV)",
    "wp6_propagation/statsemu_gp_validation.json":
        "Per-target emulator validation (n_scored, out-of-sample residuals) — "
        "READ THIS BEFORE USING A TARGET: not all 23 are trustworthy (see the "
        "WP-A7 emulator-trust gate; `wst` is trained on 40 runs, not 253)",
    "wp6_propagation/a6_posterior_stats.npz":
        "Posterior-propagated statistic envelopes with GP predictive sigma folded in",
    "wp6_propagation/a6_posterior_stats_summary.json": "Envelope summary table",
    "wp6_propagation/a6_correctness_battery.json":
        "5/5 propagation correctness battery",
    "wp6_propagation/a6_kappa_y_weighted.npz": "Source-weighted kappa-y product",
    "wp6_propagation/a6_pandey_compare.npz":
        "Comparison against the Pandey et al. kappa-y measurement (post-XPk-fix)",
    "wp6_propagation/a6_pandey_compare.json": "Pandey comparison summary",
    "wp6_propagation/ab_gate.json":
        "A-vs-B coordinate mirror gate v2, 60 twobound units (100%/100%)",
    "wp6_propagation/ab_gate_sobol.json":
        "Mirror gate on 253 Sobol units (98.4%/100%)",
    "wp6_propagation/ab_synthesis.json":
        "A+B synthesis in the fiducial-theta frame",
    "wp6_propagation/bblock_validation.json":
        "B-likelihood block validated to 0.3% of B5's recorded chi2",
    "wp6_propagation/joint_ab_plane.json": "Joint posterior in the decomposition plane",
    "wp6_propagation/probe_ladder.json": "Per-probe evacuation ladder numbers",

    # -- WP-A7: the cosmology statement ----------------------------------
    "wp7_cosmology/safescale_tables.json":
        "THE SURVEY-FACING DELIVERABLE: safe-scale tables (statistic x survey x "
        "calibration band), with the emulator-trust gate and honest denominators",
    "wp7_cosmology/s8_exercise.json":
        "The S8 exercise: bias/precision under ignore-feedback vs gas-calibrated "
        "vs scale-cut, plus the zero-feedback acceptance null",
    "wp7_cosmology/elbers_band.json":
        "Fixed-cosmology (Elbers et al. 2024) systematic band and its propagation "
        "into the safe scales",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_wp1_freeze() -> dict:
    """Re-verify every frozen WP1 source hash. Aborts on mismatch."""
    verified, checked, bad = {}, 0, []
    for prov in sorted((A_ROOT / "wp1_data").glob("*/provenance.json")):
        d = json.loads(prov.read_text())
        src = prov.parent.name
        entry = {}
        for fname, rec in d.get("files", {}).items():
            want = rec.get("sha256")
            f = prov.parent / fname
            if not want or not f.exists():
                continue
            got = sha256(f)
            checked += 1
            if got != want:
                bad.append(f"{src}/{fname}: on-disk {got[:12]} != frozen "
                           f"{want[:12]}")
            entry[fname] = want
        if entry:
            verified[src] = entry
    if bad:
        raise SystemExit("WP1 FREEZE GATE FAILED:\n  " + "\n  ".join(bad))
    print(f"WP1 freeze gate: {checked} raw files re-verified across "
          f"{len(verified)} sources — all match")
    return verified


def main() -> None:
    frozen = verify_wp1_freeze()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest, missing = {}, []
    for rel, desc in FILES.items():
        src = A_ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[rel] = {"sha256": sha256(dst), "bytes": dst.stat().st_size,
                         "description": desc}

    (OUT / "MANIFEST.json").write_text(json.dumps(
        {"version": "paper3a_products_v1",
         "wp1_frozen_source_hashes": frozen,
         "files": manifest}, indent=1))
    (OUT / "SHA256SUMS").write_text(
        "".join(f"{m['sha256']}  {rel}\n" for rel, m in manifest.items()))
    (OUT / "README.md").write_text(README)

    total = sum(m["bytes"] for m in manifest.values())
    print(f"bundle: {len(manifest)} files, {total/1e6:.1f} MB -> {OUT}")
    if missing:
        print("MISSING (not bundled):", missing)


README = """# Paper 3A data products (v1)

Frozen data products of the gas-calibrated BIND analysis: kSZ + X-ray gas
fractions -> a TNG feedback posterior -> propagated weak-lensing
statistics -> survey-facing safe-scale tables.

`paperA_walkthrough.ipynb` (BIND repo, `analysis/paper3a/notebooks/`)
documents the full chain and regenerates every figure from these files:
point `PAPER3A_PRODUCTS` at this directory and run all cells.

## Integrity

    sha256sum -c SHA256SUMS

`MANIFEST.json` additionally pins `wp1_frozen_source_hashes` — the sha256
of every raw observational input this analysis was frozen against. Those
raw files are third-party data releases and are **not redistributed
here**; the hashes let you verify you have the same bytes we did.
Download URLs, licenses and retrieval dates are in the per-source
`provenance.json` files described by `wp1_data/provenance.json`.

## Read this before using an emulator target

`statsemu_gp.npz` carries 23 targets and they are **not equally
trustworthy**. `statsemu_gp_validation.json` records, per target, the
number of training runs scored and the out-of-sample residual in units of
the measurement error. At least one target (`wst`, 40 training runs
against 253 for the rest) has a prediction error larger than the survey
statistical error it would be compared against, and no safe-scale
statement can be derived from it. The WP-A7 tables apply an explicit
`emulator_trustworthy` gate; honour it.

Note also that the emulator carries **no y-field morphology or
environment statistics** — the y-carrying targets are `cl_kappa_y`,
`cl_yy`, `cl_yt`, `peak_y` and `scaling_Y` only.

## What the posterior is, and is not

The joint A+B posterior is the best-fitting region of a model the paper
**rejects**: every block is rejected at the MAP and the posterior
predictive p-value is 0.000. Treat these chains as "the range TNG
feedback can be calibrated to", not as a measurement of the feedback of
the Universe. The WP-A7 S8 numbers inherit that framing.

## Scope (read before applying anything here to a survey)

1. **Model class: TNG.** Every band is a statement about the
   IllustrisTNG feedback family (the 30-parameter CAMELS-TNG design) —
   and the paper's own result is that this family does not fit the four
   probes jointly. Nothing here constrains non-TNG feedback physics.
2. **Halo-mass floor: M ≥ 1e13 h⁻¹M☉.** The emulators are trained above
   this floor; predictions below it are extrapolation and must not be
   used.
3. **Halo-pasting ceiling on 2-pt statistics.** The pasting pipeline
   captures ~90% of the true suppression on C_ℓ; quoted 2-pt corrections
   are incomplete at that level (peaks are exempt). Envelopes are
   validated for ℓ ≲ 8000 only.
4. **Fixed cosmology.** The calibration is at the TNG300 cosmology; the
   induced error elsewhere in (Ωm, σ8) is quantified as the Elbers band
   (`wp7_cosmology/elbers_band.json`, subdominant but nonzero) and must
   be carried, not dropped.

## Loading (the 5-line snippet)

```python
import json, numpy as np
from pathlib import Path
ROOT = Path("paper3a_products_v1")
tables = json.loads((ROOT / "wp7_cosmology/safescale_tables.json").read_text())
chain = np.load(ROOT / "wp5_chains/joint_ab_seed0.npz")["chain"]
```

## External code

The Pandey et al. comparison uses the GODMAX model implementation from
its public repository. We thank Shivam Pandey; please cite the GODMAX
papers when using that comparison product.

## Not included

Raw WP1 observational inputs (third-party, hashes pinned above), the SB35
run atlas, and the lightcone map stacks — bulk inputs that enter only
through the bundled emulators and statistics tables.
"""


if __name__ == "__main__":
    main()
