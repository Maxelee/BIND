#!/bin/bash
# Release acceptance gate: prove the engine still reproduces published numbers.
#
# WHY: cutting a release means editing the engine that produced the paper's
# figures (lint fixes, dependency changes, CLI defaults). A reordered import or
# a "harmless" cleanup that silently perturbs a number is exactly the failure a
# paper release must not ship. This regenerates Table 2 straight from the
# committed fm_two_head cache and diffs it, row by row, against the committed
# .tex -- mass_error_table.py fails loudly if any median / 16-84 / f50 / N
# disagrees, and the file itself must come back byte-identical.
#
# Requires the paper cache on ceph (Flatiron-only). Run before tagging.
#
#   bash tools/paper_cache/acceptance_gate.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

VENV=${VENV:-/mnt/home/mlee1/venvs/torch3}
# shellcheck disable=SC1091
source "$VENV/bin/activate"

TABLE="$REPO/tools/paper_cache/referee_figs/mass_error_table.tex"

export PAPER_SUITE_ROOT=${PAPER_SUITE_ROOT:-/mnt/home/mlee1/ceph/fm_testsuite}
export PAPER_MODEL_SUBDIR=${PAPER_MODEL_SUBDIR:-fm_two_head}
export PAPER_MASS_DIR=${PAPER_MASS_DIR:-mass_threshold_1p000e13}
export PAPER_MODEL_TAG=${PAPER_MODEL_TAG:-fm_two_head}
export PAPER_PRIOR_TEX="$TABLE"

echo "=== regenerating Table 2 from $PAPER_MODEL_TAG and diffing vs committed ==="
python tools/paper_cache/referee_figs/mass_error_table.py

if git diff --quiet -- "$TABLE"; then
    echo
    echo "ACCEPTANCE GATE PASSED: table reproduces byte-for-byte."
else
    echo
    echo "ACCEPTANCE GATE FAILED: regenerated table differs from the committed one." >&2
    git --no-pager diff -- "$TABLE" >&2
    exit 1
fi
