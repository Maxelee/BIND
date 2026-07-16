"""WP-B1 task 5 driver: DR5 cluster-stack sanity anchor on the ACT DR6 y map.

Stacks the native-CAR y map on the 4195 ACT DR5 SZ clusters (Hilton et al.
2009.11043) and on SNR-binned subsets, plus a random-position null of equal
size inside the fully-interior mask. This validates beams/units/mask handling
end-to-end BEFORE any novel stack (B1 acceptance: "reproduces a published
stacked-y profile within its errors").

Anchor logic: DR5 clusters were detected in ACT maps, so the stacked central
y must (a) be a >>10 sigma detection, (b) sit at the same order as the
catalog's own mean central Comptonization <y_c> (the map is ILC 1.6' FWHM
while y_c is a 2.4'-filter matched amplitude, so agreement is expected at the
tens-of-percent level, not exact), (c) decay to ~0 by 15', and (d) the random
null must be consistent with 0. Numbers + figure go to the REPORT.

Outputs (default /mnt/home/mlee1/ceph/paper3/B/wp1_maps/):
  dr5_stack_profiles.npz   per-selection r, profile, err, stacked thumbnails
  dr5_stack_summary.json   the (a)-(d) anchor numbers
  dr5_stack_sanity.png     profile figure (all / SNR bins / null)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(_repo))

from analysis.paper3b.maps import (  # noqa: E402
    load_act_mask,
    load_act_ymap,
    load_dr5_catalog,
    random_positions_in_mask,
    stack_catalog,
)

SNR_BINS = ((4.0, 6.0), (6.0, 10.0), (10.0, np.inf))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/home/mlee1/ceph/paper3/B/wp1_maps"))
    ap.add_argument("--r-max-arcmin", type=float, default=15.0)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[dr5] loading y map + mask + catalog ...", flush=True)
    ymap = load_act_ymap()
    mask = load_act_mask()
    cat = load_dr5_catalog()
    # keep clusters where the apodized mask is fully interior (clean beams/units)
    inm = mask.at(np.deg2rad(np.stack([cat.dec_deg, cat.ra_deg])), order=1) >= 0.99
    cat = cat.select(inm)
    print(f"[dr5] {len(cat)} / {inm.size} clusters inside the fully-interior mask", flush=True)

    out_npz, summary = {}, {"n_clusters_used": int(len(cat))}

    def _run(tag, ra, dec):
        st = stack_catalog(ymap, ra, dec, r_max_arcmin=args.r_max_arcmin, n_boot=args.n_boot)
        out_npz[f"{tag}_r"] = st.r_arcmin
        out_npz[f"{tag}_profile"] = st.profile
        out_npz[f"{tag}_err"] = st.profile_err
        out_npz[f"{tag}_stacked"] = st.stacked.astype(np.float32)
        summary[tag] = {
            "n": st.n_obj,
            "central_y": st.central_y,
            "central_snr": float(st.profile[0] / st.profile_err[0]),
            "outer_mean_10_15am": float(np.nanmean(st.profile[st.r_arcmin >= 10.0])),
        }
        print(f"[dr5] {tag:12s} n={st.n_obj:5d} y0={st.central_y:.3e} "
              f"SNR={summary[tag]['central_snr']:.1f}", flush=True)
        return st

    _run("all", cat.ra_deg, cat.dec_deg)
    for lo, hi in SNR_BINS:
        sel = (cat.snr >= lo) & (cat.snr < hi)
        _run(f"snr_{lo:g}_{hi:g}", cat.ra_deg[sel], cat.dec_deg[sel])
    ra_r, dec_r = random_positions_in_mask(mask, len(cat), seed=0)
    _run("random_null", ra_r, dec_r)

    summary["catalog_mean_y_c"] = float(cat.y_c.mean())
    summary["catalog_mean_fixed_y_c"] = float(cat.fixed_y_c.mean())
    summary["central_over_mean_y_c"] = summary["all"]["central_y"] / summary["catalog_mean_y_c"]
    np.savez_compressed(args.out_dir / "dr5_stack_profiles.npz", **out_npz)
    (args.out_dir / "dr5_stack_summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for tag, style in [("all", "k-o"), ("snr_4_6", "C0-"), ("snr_6_10", "C1-"),
                       ("snr_10_inf", "C3-"), ("random_null", "0.6")]:
        ax.errorbar(out_npz[f"{tag}_r"], out_npz[f"{tag}_profile"],
                    out_npz[f"{tag}_err"], fmt=style, ms=3, lw=1.2, capsize=2,
                    label=f"{tag} (n={summary[tag]['n']})")
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axhline(summary["catalog_mean_y_c"], color="g", ls="--", lw=1,
               label=r"catalog $\langle y_c \rangle$ (2.4$'$ filter)")
    ax.set_xlabel("r [arcmin]"); ax.set_ylabel(r"stacked Compton $y$")
    ax.set_title("WP-B1 task 5: ACT DR6 y-map stacked on DR5 SZ clusters")
    ax.legend(fontsize=8); ax.set_yscale("symlog", linthresh=1e-7)
    fig.tight_layout()
    fig.savefig(args.out_dir / "dr5_stack_sanity.png", dpi=150)
    print(f"[dr5] wrote {args.out_dir}/dr5_stack_sanity.png", flush=True)


if __name__ == "__main__":
    main()
