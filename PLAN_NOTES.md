# PLAN_NOTES.md — Execution log for scatter paper

## Phase 0 — Reconnaissance (2026-05-12)

### Environment
- Python env: `/mnt/home/mlee1/venvs/torch3` — loads fine
- `from train import FlowMatchingLit` ✓
- Model checkpoint: `/mnt/home/mlee1/ceph/fm_runs/fm_two_head/checkpoints/last.ckpt`
- Norm stats: `/mnt/home/mlee1/ceph/fm_runs/fm_two_head/norm_stats.npz`

### CV data structure
- 27 CV simulation directories under `/mnt/home/mlee1/ceph/fm_testsuite/CV`
- Each sim: `snap_090/mass_threshold_1p000e13/`
  - `halo_catalog.npz` keys: `centers` (N,2), `params` (N,35), `masses` (N,), `halo_masses`, `halo_positions`, `radii` (N,) in kpc/h
  - `halo_cutouts.npz` keys: `condition` (N,128,128), `large_scale` (N,3,128,128)
  - `fm_two_head/generated_halos.npz`: `generated` (N,3,128,128) — stored as physical-space (already denormalized? need to verify)
- Hydro truth maps: `snap_090/full_maps.npz` keys: `dmo_fullbox` (1024,1024), `truth_maps` (3,1024,1024)
  - centers field stores positions in Mpc/h
- sim_0 has 45 halos above 10^13 M_sun/h threshold
- CAMELS bug: params[:,14] must be set to 0.0 (same as fd_jacobian_cv.py)

### Gating check
- ✓ Model loads
- ✓ CV path resolves (27 sims)
- ✓ condition shape (N, 128, 128), large_scale (N, 3, 128, 128) — C dimension added by normalize_inputs

---

## Phase 1 — Scatter measurement engine (2026-05-12)

- Implemented `scatter/measure_scatter.py`
- Implemented `scatter/test_measure_scatter.py` for gating check

---

## Deviations from plan

None yet.

## Open issues / human-handoff triggers

None yet.
