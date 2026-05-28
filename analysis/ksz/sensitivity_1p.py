"""Gate 1 — does the stacked τ observable respond to each parameter at all?

Walks the 1P suite, where CAMELS varies *one* subgrid parameter at a time along
a ladder (`1P_p{j}_{level}`).  For each ladder we (a) detect which of the 35
BIND parameters actually varies (argmax variance across the ladder — robust to
any p{j}↔index mapping), and (b) measure the response of the stacked CAP τ to
that parameter, for BIND and for the simulation truth.

This is the make-or-break check for SBI: if τ does not move when a feedback
parameter is swept across its full CAMELS range, no inference method can
recover it, and the paper is the "no single subgrid direction closes the
deficit" null result (plan §2 outcome 2).  Conversely, feedback axes with a
large |Δτ/τ| are the ones SBI can hope to constrain.

Outputs an npz + a sorted bar chart of fractional τ response per parameter
(BIND vs truth), with cosmological axes flagged separately from feedback.

Usage:
    python -m analysis.ksz.sensitivity_1p \\
        --testsuite_root /mnt/home/mlee1/ceph/fm_testsuite_cube \\
        --model fm_cube_two_head --suite 1P \\
        --aperture cap --r_ap_mpc_h 0.5 \\
        --out analysis_physics_cache/ksz_sensitivity_1p_fm_cube_two_head.npz \\
        --fig figures/ksz_sensitivity_1p_fm_cube_two_head.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from ._io import find_sim_dirs, load_sim, los_advisory
from .param_meta import load_param_meta
from .tau_utils import per_halo_tau

_LADDER_RE = re.compile(r"1P_p(\d+)_", re.IGNORECASE)


def _ladder_tag(sim_name: str) -> str | None:
    m = _LADDER_RE.match(sim_name)
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testsuite_root", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default="1P")
    ap.add_argument("--halo_mass_min", type=float, default=1e13)
    ap.add_argument("--box_size", type=float, default=50.0)
    ap.add_argument("--patch_size_mpc_h", type=float, default=6.25)
    ap.add_argument("--hubble", type=float, default=0.6711)
    ap.add_argument("--aperture", choices=["disk", "cap"], default="cap")
    ap.add_argument("--r_ap_mpc_h", type=float, default=0.5)
    ap.add_argument("--mass_bins", nargs="+", type=float,
                    default=[1e13, 3e13, 1e14, 1e15],
                    help="Mass bins for the per-bin response (in addition to all-halo).")
    ap.add_argument("--response_thresh", type=float, default=0.05,
                    help="|Δτ/τ| across the ladder above which a param is 'responsive'.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, default=None)
    args = ap.parse_args()

    meta = load_param_meta()
    edges = np.asarray(args.mass_bins, dtype=np.float64)

    # ── gather per-sim (θ, all-halo τ, per-bin τ) for the 1P suite ──────────
    sims = find_sim_dirs(args.testsuite_root, args.suite)
    if not sims:
        raise SystemExit(f"No sim dirs under {args.testsuite_root / args.suite}")

    ladders: dict[str, list[dict]] = {}
    banner_shown = False
    for sd in sims:
        tag = _ladder_tag(sd.name)
        if tag is None:
            continue
        try:
            art = load_sim(sd, suite=args.suite, model_name=args.model,
                           halo_mass_min=args.halo_mass_min,
                           box_size=args.box_size,
                           patch_size_mpc_h=args.patch_size_mpc_h)
        except Exception as exc:
            print(f"[err]  {sd.name}: {exc}")
            continue
        if art is None or art.params.shape[1] == 0:
            continue
        if not banner_shown:
            print(los_advisory(art.truth_source, art.los_depth_mpc_h, args.aperture))
            banner_shown = True

        pix_size = args.patch_size_mpc_h / art.patch_pix
        r_ap_pix = args.r_ap_mpc_h / pix_size
        tau_b = per_halo_tau(art.bind_gas, r_ap_pix, pix_size, args.hubble, estimator=args.aperture)
        tau_t = per_halo_tau(art.truth_gas, r_ap_pix, pix_size, args.hubble, estimator=args.aperture)

        def _binned(tau):
            fin = np.isfinite(tau)
            allm = float(np.mean(tau[fin])) if fin.any() else np.nan
            idx = np.digitize(art.halo_masses, edges) - 1
            nb = len(edges) - 1
            out = np.full(nb, np.nan)
            for k in range(nb):
                sel = (idx == k) & fin
                if sel.any():
                    out[k] = float(np.mean(tau[sel]))
            return allm, out

        b_all, b_bins = _binned(tau_b)
        t_all, t_bins = _binned(tau_t)
        ladders.setdefault(tag, []).append({
            "theta": art.params[0],
            "b_all": b_all, "t_all": t_all,
            "b_bins": b_bins, "t_bins": t_bins,
        })

    n_p = len(meta.names)
    nb = len(edges) - 1

    # ── per-ladder: detect varied dim, measure response ─────────────────────
    # Aggregate into per-parameter arrays (a param may be probed by one ladder).
    frac_b = np.full(n_p, np.nan)        # all-halo fractional τ response, BIND
    frac_t = np.full(n_p, np.nan)        # ... truth
    slope_b = np.full(n_p, np.nan)       # dτ/d(normalized θ), all-halo, BIND
    slope_t = np.full(n_p, np.nan)
    sign_agree = np.zeros(n_p, dtype=bool)
    n_ladder = np.zeros(n_p, dtype=np.int32)
    frac_b_bins = np.full((n_p, nb), np.nan)
    frac_t_bins = np.full((n_p, nb), np.nan)
    detected_idx: list[int] = []

    for tag, rows in ladders.items():
        if len(rows) < 3:
            continue
        thetas = np.array([r["theta"] for r in rows], dtype=np.float64)  # (L, 35)
        tnorm = meta.normalized(thetas)                                  # (L, 35)
        var = np.nanvar(tnorm, axis=0)
        d = int(np.argmax(var))
        if var[d] <= 0:
            continue
        n_ladder[d] += 1
        detected_idx.append(d)
        td = tnorm[:, d]

        def _resp(vals):
            vals = np.asarray(vals, dtype=np.float64)
            ok = np.isfinite(vals) & np.isfinite(td)
            if ok.sum() < 3:
                return np.nan, np.nan
            v = vals[ok]
            med = np.median(v)
            frac = (np.max(v) - np.min(v)) / abs(med) if med != 0 else np.nan
            # signed slope dτ/dt via least squares
            A = np.vstack([td[ok], np.ones(ok.sum())]).T
            slope = np.linalg.lstsq(A, v, rcond=None)[0][0]
            return frac, slope

        fb, sb = _resp([r["b_all"] for r in rows])
        ft, st = _resp([r["t_all"] for r in rows])
        frac_b[d], slope_b[d] = fb, sb
        frac_t[d], slope_t[d] = ft, st
        if np.isfinite(sb) and np.isfinite(st):
            sign_agree[d] = (np.sign(sb) == np.sign(st)) or (sb == 0 and st == 0)
        for k in range(nb):
            fbk, _ = _resp([r["b_bins"][k] for r in rows])
            ftk, _ = _resp([r["t_bins"][k] for r in rows])
            frac_b_bins[d, k] = fbk
            frac_t_bins[d, k] = ftk

    # ── verdict ─────────────────────────────────────────────────────────────
    probed = n_ladder > 0
    responsive = probed & (frac_t >= args.response_thresh)   # judge by TRUTH response
    feedback = probed & (~meta.is_cosmo)
    resp_feedback = responsive & (~meta.is_cosmo)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        param_idx=np.arange(n_p, dtype=np.int32),
        labels=np.array(meta.labels),
        names=np.array(meta.names),
        is_cosmo=meta.is_cosmo,
        probed=probed,
        n_ladder=n_ladder,
        frac_response_bind=frac_b,
        frac_response_truth=frac_t,
        slope_bind=slope_b,
        slope_truth=slope_t,
        sign_agree=sign_agree,
        frac_response_bind_bins=frac_b_bins,
        frac_response_truth_bins=frac_t_bins,
        mass_edges=edges,
        response_thresh=args.response_thresh,
        aperture=args.aperture,
        r_ap_mpc_h=args.r_ap_mpc_h,
    )
    print(f"[save] {args.out}")

    # text verdict
    order = np.argsort(-np.nan_to_num(frac_t))
    print(f"\n# Gate 1 — 1P τ response (truth), threshold |Δτ/τ| ≥ {args.response_thresh}")
    print(f"# {int(probed.sum())}/{n_p} params probed by a 1P ladder; "
          f"{int(responsive.sum())} responsive ({int(resp_feedback.sum())} of them feedback)")
    print("# top responders:")
    for d in order[:12]:
        if not probed[d]:
            continue
        kind = "cosmo" if meta.is_cosmo[d] else "FEEDBACK"
        agree = "✓" if sign_agree[d] else "✗"
        print(f"  p{d:02d} {meta.labels[d]:<14s} [{kind:8s}]  "
              f"Δτ/τ truth={frac_t[d]:.3f}  BIND={frac_b[d]:.3f}  "
              f"slope_sign_agree={agree}")

    # ── figure ────────────────────────────────────────────────────────────
    if args.fig is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sel = np.where(probed)[0]
        sel = sel[np.argsort(-np.nan_to_num(frac_t[sel]))]
        y = np.arange(len(sel))
        fig, axx = plt.subplots(figsize=(7.5, 0.32 * len(sel) + 1.5))
        colors = ["C1" if meta.is_cosmo[d] else "C0" for d in sel]
        axx.barh(y + 0.2, frac_t[sel], height=0.4, color=colors, alpha=0.9, label="truth")
        axx.barh(y - 0.2, frac_b[sel], height=0.4, color=colors, alpha=0.45, label="BIND")
        axx.axvline(args.response_thresh, color="k", ls="--", lw=1,
                    label=f"responsive thresh = {args.response_thresh}")
        axx.set_yticks(y)
        axx.set_yticklabels([f"p{d:02d} {meta.labels[d]}"
                             + ("" if not meta.is_cosmo[d] else "  (cosmo)") for d in sel],
                            fontsize=7)
        axx.invert_yaxis()
        axx.set_xlabel(r"fractional τ response across 1P range  $|\Delta\tau/\tau|$ (all-halo)")
        axx.set_title("Gate 1 — does stacked τ respond to each parameter?\n"
                      "(orange = cosmological, blue = feedback)")
        axx.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig)
        plt.close(fig)
        print(f"[save] {args.fig}")


if __name__ == "__main__":
    main()
