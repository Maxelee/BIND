"""scatter/robustness/run_all_robustness.py
Master script to run all 5 robustness checks sequentially.
Call after J_mean_and_scatter.npz exists.

Usage:
    python scatter/robustness/run_all_robustness.py [--skip_expensive]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent


def run(script: Path, label: str, skip: bool = False):
    if skip:
        print(f"\n=== {label}: SKIPPED ===\n")
        return None
    print(f"\n{'='*60}")
    print(f"=== {label} ===")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        capture_output=False,
    )
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_expensive", action="store_true",
                    help="Skip K=20 and full mass-bin checks (each ~4 hr)")
    args = ap.parse_args()

    checks = [
        ("Check 1: K budget",          OUT_DIR / "check_k_budget.py",           False),
        ("Check 2: Mass-bin stability", OUT_DIR / "check_mass_bins.py",          args.skip_expensive),
        ("Check 3: LOS contamination",  OUT_DIR / "check_los_contamination.py",  False),
        ("Check 4: Step-size",          OUT_DIR / "check_stepsize.py",           False),
        ("Check 5: Seed robustness",    OUT_DIR / "check_seed.py",              False),
    ]

    results = {}
    for label, script, skip in checks:
        rc = run(script, label, skip)
        results[label] = "SKIPPED" if skip else ("PASS" if rc == 0 else "FAIL")

    # Write SUMMARY.md
    summary_path = OUT_DIR / "SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("# Robustness Check Summary\n\n")
        for label, status in results.items():
            icon = "✓" if status == "PASS" else ("⚠" if status == "SKIPPED" else "✗")
            f.write(f"- {icon} {label}: **{status}**\n")
        f.write("\nSee individual files for details:\n")
        f.write("- `k_budget.pdf` — K convergence\n")
        f.write("- `mass_bins.pdf` — mass-bin stability\n")
        f.write("- `los_contamination.pdf` — LOS contamination test\n")
        f.write("- `stepsize.pdf` — step-size linearity\n")
        f.write("- `seed_robustness.pdf` — seed reproducibility\n")

    print(f"\nSaved {summary_path}")
    print("\n=== SUMMARY ===")
    for label, status in results.items():
        print(f"  {label}: {status}")


if __name__ == "__main__":
    main()
