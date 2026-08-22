# Quarantined 2026-08-21

SOBOL-class renders built from the PARTIAL emulator_dataset_n1000.npz
(107/256 nodes, non-contiguous run_ids) — some additionally carried a
row-misalignment bug (positional indexing of 256-row sci50 model arrays
with partial-dataset positions). Kept for comparison only.

They are rebuilt into imgs_1000/ by `analysis/make_imgs_1000.py --build`
once the campaign completes and `--assemble` has produced the full
256-node dataset.
