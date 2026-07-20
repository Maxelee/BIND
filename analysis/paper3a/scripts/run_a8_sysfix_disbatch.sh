#!/bin/bash
#SBATCH --job-name=a8_sysfix
#SBATCH --output=/mnt/home/mlee1/ceph/logs/a8_sysfix_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/a8_sysfix_%j.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A8: the two systematics-hunt A-side defects, as fresh-chain variants ──
#
# Written 2026-07-20 after the six-agent systematics hunt
# (bind-paper3-plans/systematics-hunt/SYNTHESIS.md). It covers A-side
# defects 1 and 2 of the three that hunt found. Defect 3 (the
# BIND_PAPER3A_LOGM500_SHIFT propagation bug) was a genuine BUG and was
# FIXED IN THE DEFAULT PATH instead — it needed no chains, only a re-run of
# run_a8_massshift.py, which importance-reweights the frozen joint chain.
#
# NEITHER VARIANT HERE CHANGES A DEFAULT. Both are measure-then-model
# probes: the corrections are real and are correctly signed, but each rests
# on a transport or a caveat that a variant can expose and a silent default
# change would bury.
#
#   c2s_massdep       CylToSph made mass-dependent across the five gate
#                     bins instead of one scalar 0.5275 (FINDINGS_fgas.md
#                     candidate 1). Anchor-preserving: only the pooled-L50
#                     SHAPE is imported, the wp2 TNG300 absolute factor is
#                     held fixed by construction.
#   ksz_wp2bias       kSZ prediction divided by the wp2 painted-vs-truth
#                     tau_CAP bias (1 + frac_bias) at theta <= 3.5' --
#                     the f_gas PAINT_BIAS = 1.074 treatment applied to the
#                     probe that never received one
#                     (FINDINGS_bind_layer.md section 1).
#   ksz_wp2bias_all   the same at all nine radii, including the factor
#                     1.85-2.1 the wp2 measurement implies at 4.75'/6'.
#
# ── DEVIATION FROM THE TASK BRIEF, STATED UP FRONT ──────────────────────────
# The brief specified 8 tasks = 2 variants x 4 seeds. This file has 12 =
# 3 x 4. The third variant (ksz_wp2bias_all) was to be written but NOT
# scheduled, because the outer two radii were believed to be possibly a
# small-denominator artifact of the compensated CAP, checkable only on
# Popeye. That check LANDED DURING THIS WORK and was verified independently
# here: the wp2 artifacts are mirrored on rusty at
# /mnt/ceph/users/mlee1/paper3/A/wp2_validation/, and deriving
# stack(truth) = bias/frac_bias from sigma_model_ksz_tauCAP_snap063.npz
# (n_halos=601) gives 4.152e-4 / 2.359e-3 / 3.467e-3 / 4.210e-3 / 5.101e-3
# across the five wp2 radii -- MONOTONICALLY INCREASING, with the outermost
# stack the LARGEST of the five at 12.3x the innermost. The denominator at
# 4.75'/6' is the biggest one there is, so the artifact hypothesis fails on
# its own terms and the outer-radius half of defect 2 is real.
# The inner/all pair is therefore the measurement: their DIFFERENCE is how
# much of the effect lives in the outer two radii, which is exactly the
# quantity A4 and wp2 disagree about (A4: "the painted field is healthy"
# out there; wp2: model-high by a factor ~2). That contradiction is NOT
# resolved -- which is why both are run and neither is adopted.
# To fall back to the brief's 8 tasks, delete the last four lines of
# a8_sysfixbatch.disbatch.
#
# ── WHY FRESH CHAINS AND NOT IMPORTANCE REWEIGHTING ─────────────────────────
# run_a8_sysgrid.py's reweighting of the frozen fiducial chain is valid only
# when the variant WIDENS an error tier, and it collapsed even then (ESS
# 24/130/141 of 8000). Both variants here RE-POINT the mean of a likelihood
# block -- c2s_massdep tilts the f_gas mass slope by ~1.8x across the data's
# 1.7 dex, ksz_wp2bias lowers the kSZ prediction by 7-13% (inner) or up to
# ~2x (all). A re-pointed target puts posterior mass where the fiducial
# proposal has essentially none, which is the same failure mode in a worse
# direction. Reweighting cannot reach either row.
#
# ── PRE-REGISTERED PASS/FAIL. FIXED BEFORE THE RUN. DO NOT EDIT AFTER. ──────
#
# c2s_massdep — FINDINGS_fgas.md Test C, verbatim:
#   MATTERS      if the joint f_gas block chi2 falls below 8/5 WHILE
#                dln M_gas moves below -0.45
#   CAVEAT ONLY  if the kappa-peak (B) block stays above 83/4, as every
#                configuration ever run has
#   Supporting context from the same finding, for interpretation only, not
#   a criterion: the group bin -- which sets f_group -- moves only x0.92,
#   so the mechanism being REAL and the mechanism CLOSING THE 6 SIGMA are
#   separate questions and only this chain answers the second. The auditor
#   put ~0.3 on it moving f_group far enough to matter. Do not read a
#   MATTERS verdict as the 6 sigma being explained.
#
# ksz_wp2bias / ksz_wp2bias_all — FINDINGS_bind_layer.md candidate 1
# step 2, verbatim:
#   MATERIAL     if the kSZ-only f_group posterior moves toward f_gas by
#                >= 1.5 sigma of the current separation -> a material
#                contributor to the 6 sigma; must be fixed and reported
#   DEAD         if < 0.5 sigma
#   PARTIAL      in between: report as a systematic, claim nothing
#   Independently, on the headline coordinate: joint dln M_gas median
#   moving < 0.5 sigma (< 0.012) -> the headline coordinate is robust
#   regardless of how the above reads.
#   NOTE the criterion is written for a kSZ-ONLY fit; these are JOINT
#   chains, so f_group is read off the joint posterior and the separation
#   is the frozen fgas-only (0.0248 +- 0.0011) vs kszonly (0.0143 +-
#   0.00145) gap. Stated here so the comparison is not re-chosen later.
#   The criterion above is applied SEPARATELY to each of the two forms.
#
# INNER-ONLY vs ALL-RADII — what the PAIR decides. Added 2026-07-20 when
# ksz_wp2bias_all was promoted from unvalidated to scheduled; fixed before
# the run, like everything else in this block. The two rows are read
# against each other on the same f_group movement used above, and the
# comparison is scored as:
#   INNER-CARRIED   the two movements agree to within 0.5 sigma of the
#                   separation -> the effect is carried by the 7-13% inner
#                   radii, which are well measured and consistent across
#                   all four wp2 snapshots. The conclusion then does NOT
#                   depend on the contested outer bins and either row may
#                   be quoted. Strongest outcome.
#   OUTER-CARRIED   `_all` moves >= 1.0 sigma further than inner-only ->
#                   the correction is outer-radius dominated. Still
#                   evidence-backed (the absolute stack(truth) read above),
#                   but it lands precisely where A4 and wp2 contradict each
#                   other, so the paper MUST carry both numbers as a
#                   bracket and may quote NEITHER alone.
#   INCOHERENT      the two move in opposite directions, or inner-only
#                   moves while `_all` does not -> the radial interpolation
#                   is suspect. Both rows are UNQUOTABLE; debug the
#                   divisor before scoring anything.
# Inner-only is retained as the CONSERVATIVE BRACKET on every outcome; it
# is not superseded by `_all` being validated.
#
# STANDING CAVEAT that no chain here can lift: BOTH corrections are
# transported from TNG300-hydro validations, and both exonerating
# validations in the hunt were also against TNG300. If TNG itself
# over-predicts, no variant in this file can see it. This measures internal
# consistency of our own chain, which is what defects 1 and 2 are.
#
# ── MECHANICS ───────────────────────────────────────────────────────────────
# 12 tasks fanned by disBatch inside ONE allocation, each single-core
# (emcee here is vectorized single-process -- NEVER MPI). Same recipe as the
# fiducial fit: 128 walkers x 30k, DE move mixture, and every task
# re-checks the a5_recovery_jointab gate itself, so a failed battery stops
# the fits rather than the job. Reference wall time: the earlier 12-task
# variant run took 1h51m.
#
# AFTER IT LANDS the driver re-runs run_a8_sysgrid.py and
# run_a8_variant_chi2.py below, exactly as run_a8_dropone_disbatch.sh does,
# so the new rows are scored alongside the existing variants.
#
# ⚠ ONE-LINE PREREQUISITE, NOT DONE HERE — run_a8_sysgrid.py WILL SILENTLY
#   SKIP THESE THREE ROWS AS IT STANDS. Its fresh-chain loop iterates a
#   HARDCODED `FRESH = {...}` dict (currently emul2x, fgas_model2x,
#   b_coordsys2x, no_ksz, no_fgas) rather than run_joint_ab_fit.VARIANTS,
#   and a variant missing from it logs "chains absent — row left unrun"
#   and moves on. That file was outside this session's write scope, so it
#   was NOT edited. Before submitting, add to that dict:
#
#     "c2s_massdep":     "FRESH CHAINS (4 seeds), systematics-hunt defect 1
#                         — mass-dependent CylToSph. Not reweightable: it
#                         re-points the f_gas mass slope",
#     "ksz_wp2bias":     "FRESH CHAINS (4 seeds), systematics-hunt defect 2
#                         — wp2 tau_CAP bias, inner radii (theta <= 3.5')",
#     "ksz_wp2bias_all": "FRESH CHAINS (4 seeds), systematics-hunt defect 2
#                         — wp2 tau_CAP bias, all nine radii",
#
#   run_a8_variant_chi2.py needs NO change: it already iterates
#   `("",) + VARIANTS`, so it picks the three up automatically.
#
# ⛔ Submit:  sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_a8_sysfix_disbatch.sh

set -euo pipefail

module load gcc disBatch
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2-paper3a
mkdir -p /mnt/home/mlee1/ceph/logs/joint_ab

echo "=== WP-A8 systematics-fix variants: 12 chains ==="
echo "    c2s_massdep, ksz_wp2bias, ksz_wp2bias_all x 4 seeds"
disBatch -p /mnt/home/mlee1/ceph/logs/joint_ab/db_a8_sysfix_ \
    analysis/paper3a/scripts/a8_sysfixbatch.disbatch

echo "=== chains written; re-scoring the systematics grid ==="
python -u analysis/paper3a/scripts/run_a8_sysgrid.py

echo "=== per-variant chi2 at each variant's own MAP ==="
python -u analysis/paper3a/scripts/run_a8_variant_chi2.py

echo "DONE"
