# Training vs Test set overlap (fm_base, 2026-04-24)

Training data: `/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/` —
921 unique sim directories named `sim_N` where N is the SB35 index (sim_N maps
to `SB35_N` in `CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt`).

Test set: `/mnt/home/mlee1/ceph/fm_testsuite/Test/` — 102 sim directories
named `sim_SB35_N`. Note the different naming convention.

**Overlap (matched by suffix N): 0 / 102.** The Test set is fully held out
from the training set. No `generated_halos.npz` exists on disk for any of
the 921 training sims — running test_suite inference on them would take
hours.

Implication for Option 2 of the Stars-bias correction plan: we cannot
fit the regression on training-set residuals without first running
test-suite inference on the 921 training sims. Defaulting to a clean
held-out-test split (80/20 of the 102 Test sims) for now; user can
escalate to the proper fit later if needed.
