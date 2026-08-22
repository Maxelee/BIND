# New Image 2 decision memo + New Image 1 notes (2026-08-03)

Recovered from the prototypes agent's returned report (a harness guard blocked subagents
from writing report files directly; content below is its verbatim findings, lightly
reformatted). Companion prototype: `proto_bridge_hero.py` → `figs_preview/proto_bridge_hero.png`.

## New Image 2 (fig13 panel d, "177/253 nodes") — what the gap actually is

- Panel (d) reads `t__dm_pdf__*` from `emulator_dataset_xpkfix.npz`, which stacks per-run
  `bind_sb35/runs/run_NNNN/dm_stats.npz` shards. Direct ceph census: `tau_maps.npz` 253,
  `Cl_tau.npz` 253, **`dm_stats.npz` 177**. So all 253 completed runs HAVE the input τ maps;
  only the cheap post-process shard is missing on 76 runs.
- The 76 missing runs form contiguous periodic blocks (run_0005–0015, 0021–0031, … 0101–0110;
  everything ≥111 complete): timestamps show `dm_stats.npz` was a **separate serial backfill
  pass (Jun 23–24)** that was interrupted/partially dispatched — an operations artifact, not a
  physics/code/data problem. Verified by re-running `bind.inference.stats.dm_stats()` on
  run_0005's real τ maps in-process: clean, ~50 s single-core, well-formed output.
- **Completing 177→253(→256) is nearly free**: ~50 s/run CPU × 79 runs, then one
  `bind-emulator-assemble` pass. **Now folded into `jobs/job2_repair_caches.sbatch` step [1b]**
  — it runs automatically with the Job 2 submission, before the dataset reassembly.
- The 253→256 part is the separately-tracked corrupted-run repaint (Job 1, completed
  2026-08-03 14:00–14:05) — orthogonal to the DM-panel gap.

## Recommendation: demote to §6, do not cut to Paper IV

**(a) fold panel (d) into the §6 "what the release supports" material as a compact entry**,
out of the 4-panel fig13 headline figure. Reasons:
- §6's data-products table already carries the "DM statistics: 177/253" row — panel (d) is a
  picture of that row; after the backfill it reads 253/253 (256/256) either way.
- The fig13 markdown itself frames panel (d) as a capability demo for the FRB community
  ("partial coverage and halo-aperture-only, both stated on-figure"), not a physics result —
  the register of a §6 release note.
- Paper IV (`papers/04_ksz_gas/brief.md`) is scoped to per-halo CAP-filtered kSZ/eROSITA
  confrontation and has no FRB/DM section; a whole-lightcone DM PDF is a different
  methodological register — grafting it there removes from Paper I (the chartered release
  anchor) the one place a reader would look for "can I compute a DM PDF from these τ maps."
- Cutting entirely discards documentation of an already-computed derived product for zero
  compute savings once the backfill (sunk, trivial) is done.

## New Image 1(a) prototype — status

`proto_bridge_hero.py` on the canonical 253-run Sobol dataset: panel (a) S(ℓ=5000, z_s=1) vs
group-scale f_gas, colored by WindEnergy — **r = 0.71** (population-level analogue of the
retired 57-run bridge hero; weaker than fig20's tuned multi-bin r≈0.98 construction by
design). Panel (b) ejection–heating plane: **population-level ejection and heating are
strongly coupled (r = 0.86)**, unlike the near-decorrelated 1P version — the clean 2-D
decomposition of the retired figure may be partly an artifact of the 1-parameter-at-a-time
design. Flag this before deciding whether panel (b) survives. NOT wired into the builder;
awaits confirmation that this is the right "New Image 1(a)" candidate.

## New Image 1(b) recommendation

Use **fig13 panels (a,b)** (feedback-colored τ/y profile stacks) as the §3c profile-space
companion — via a forward cross-reference from §3c prose, not by moving/duplicating panels.
fig06 is validation-only (no feedback variation; §2a′) and would be a category mismatch; a
new figure would duplicate fig13(a,b). fig13(a,b) shows exactly the radial gas-redistribution
mechanism behind fig20's aperture-integrated f̃_bar ↔ S(ℓ) relation, same mass regime, same
color lever.
