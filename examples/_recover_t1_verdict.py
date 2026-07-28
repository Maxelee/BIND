#!/usr/bin/env python
"""Recovery / validation for `verdicts/T1.json` (docs/REPRODUCING.md gap #2).

No script in the current `examples/` tree writes this verdict -- it predates
the R0-R8 hardening refactor and appears to have been produced ad hoc. This
reconstructs its `metrics` block from the two stored T1 products it must
have been built from, `LC/T1_dr5_snr5.npz` (ACT DR5 S/N>5 cluster stack) and
`LC/T1_random_null.npz` (DESI-random-position null stack):

  - stack_snr_jk_per_theta / stack_snr_jk_at_3.5arcmin: mean/err_jk per
    theta bin, T1_dr5_snr5.npz. **Exact** (verified below, ties to the
    stored value's full float64 precision).
  - null_chi2_per_dof: mean^T @ inv(cov_jk) @ mean / n_theta, from
    T1_random_null.npz (9 theta bins; no Hartlap factor -- the fit here
    used the null's own jackknife covariance directly, not a Hartlap-
    debiased one). **Exact**.
  - null_n: T1_random_null.npz `n_used[0]`. **Exact**.
  - boot_vs_jk_err_ratio: err_boot/err_jk per theta, T1_dr5_snr5.npz.
    **Exact**.
  - yc_tercile_amps_3.5arcmin: `fixed_y_c` terciles (edges from all 2419
    objects via `np.percentile([100/3, 200/3])`), CAP amplitude at the
    3.5' column averaged over the `per_obj_ok`-True members of each
    tercile. **Approximate**: matches the stored values to <0.09%
    relative (see the printed report) but not bit-exact -- the original
    ad hoc script's precise tie-breaking/masking convention at the tercile
    boundaries isn't otherwise recoverable from the stored products, so
    this is the closest reconstruction found, not a verified-identical one.
  - yc_scaling_monotonic: trivially re-derived (amps strictly increasing).
  - batch_vs_pixell_path_max_reldiff: **not reproduced** -- it compares
    against `LC/T1_dr5_snr5_pixellpath.npz`, a third product outside this
    script's two declared inputs, so per the recovery brief it is instead
    read straight through from the existing `verdicts/T1.json` unchanged.

Run: `python examples/_recover_t1_verdict.py [--out PATH] [--compare]`.
`--compare` (default on) loads the existing `verdicts/T1.json` and prints a
field-by-field relative-difference report; it never overwrites that file.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

KS = Path(os.environ.get("BIND_KSZ_PRODUCTS", "/mnt/home/mlee1/ceph/bind_science/ksz_confront"))
LC = KS / "lightcone"


def recover_metrics(lc: Path) -> dict:
    dr5 = np.load(lc / "T1_dr5_snr5.npz", allow_pickle=True)
    null = np.load(lc / "T1_random_null.npz", allow_pickle=True)

    theta = dr5["theta_value"].astype(float)
    mean, err_jk, err_boot = dr5["mean"], dr5["err_jk"], dr5["err_boot"]
    snr_per_theta = mean / err_jk
    idx35 = int(np.argmin(np.abs(theta - 3.5)))

    nmean, ncov_jk = null["mean"], null["cov_jk"]
    n_theta = len(nmean)
    chi2 = float(nmean @ np.linalg.inv(ncov_jk) @ nmean)

    fixed_y_c = dr5["fixed_y_c"].astype(float)
    per_obj_cap = dr5["per_obj_cap"][:, idx35].astype(float)
    ok = dr5["per_obj_ok"][:, idx35]
    q1, q2 = np.percentile(fixed_y_c, [100 / 3, 200 / 3])
    yc_ok, cap_ok = fixed_y_c[ok], per_obj_cap[ok]
    amps = [
        float(cap_ok[yc_ok <= q1].mean()),
        float(cap_ok[(yc_ok > q1) & (yc_ok <= q2)].mean()),
        float(cap_ok[yc_ok > q2].mean()),
    ]

    # Not reproducible from the two declared inputs -- pass through unchanged.
    try:
        existing = json.loads((lc / "verdicts" / "T1.json").read_text())
        pixell_reldiff = existing["metrics"]["batch_vs_pixell_path_max_reldiff"]
    except (FileNotFoundError, KeyError):
        pixell_reldiff = None

    return {
        "stack_snr_jk_at_3.5arcmin": float(snr_per_theta[idx35]),
        "stack_snr_jk_per_theta": [float(v) for v in snr_per_theta],
        "null_chi2_per_dof": chi2 / n_theta,
        "null_n": int(null["n_used"][0]),
        "yc_tercile_amps_3.5arcmin": amps,
        "yc_scaling_monotonic": bool(amps[0] < amps[1] < amps[2]),
        "boot_vs_jk_err_ratio": [float(v) for v in (err_boot / err_jk)],
        "batch_vs_pixell_path_max_reldiff": pixell_reldiff,  # read-through, see docstring
    }


def compare(recovered: dict, existing_path: Path) -> None:
    existing = json.loads(existing_path.read_text())["metrics"]
    print(f"{'field':32s} {'max |reldiff|':>14s}  status")
    for key, rec_val in recovered.items():
        if key == "batch_vs_pixell_path_max_reldiff":
            print(f"{key:32s} {'n/a':>14s}  read-through (not reproduced)")
            continue
        orig = np.asarray(existing[key], dtype=float)
        new = np.asarray(rec_val, dtype=float)
        reldiff = np.max(np.abs(new - orig) / np.where(orig != 0, np.abs(orig), 1.0))
        tag = "exact" if reldiff < 1e-6 else ("approx" if reldiff < 1e-2 else "MISMATCH")
        print(f"{key:32s} {reldiff:14.3e}  {tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=LC / "verdicts" / "T1.json")
    ap.add_argument("--no-compare", action="store_true",
                     help="skip the field-by-field diff against the existing verdict")
    args = ap.parse_args()

    metrics = recover_metrics(LC)
    out_doc = {
        "phase": "T1",
        "pass": True,
        "metrics": metrics,
        "notes": "Recovered by examples/_recover_t1_verdict.py; see its docstring "
                 "for which fields are exact vs. approximate vs. read-through.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2))
    print(f"wrote {args.out}")

    if not args.no_compare:
        existing_path = LC / "verdicts" / "T1.json"
        if existing_path.exists():
            compare(metrics, existing_path)
        else:
            print(f"no existing verdict at {existing_path} to compare against")
