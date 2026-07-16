"""WP-B1 tasks 3+4 driver: harmonized footprint + peak catalogs (both DES variants).

Runs the frozen PEAK_DEFINITION chain on the DES Y3 Wiener AND GLIMPSE maps
(the "build both, let the DR5 anchor + nulls decide" recommendation — the
variant freeze is Max's call) over the {1, 2, 5, 8} arcmin smoothing set,
against the DES x ACT harmonized footprint. Deterministic: re-running
regenerates identical catalogs from the raw downloads (B1 acceptance).

Outputs (default /mnt/home/mlee1/ceph/paper3/B/wp1_maps/):
  common_footprint_nside1024.npz   weight + binary masks + sky fractions
  peaks_<variant>_sm<S>am.npz      PeakCatalog fields + frozen-bin histogram
  peak_catalog_summary.json        counts/exclusions/nu-stats table for REPORT

CPU-light (SHTs at Nside=1024); no Slurm needed. ~minutes.
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
    DEFAULT_ACT_THRESHOLD,
    FULL_SKY_DEG2,
    SMOOTHING_SET_ARCMIN,
    act_mask_to_healpix,
    build_peak_catalog,
    harmonize,
    load_act_mask,
    load_des_map,
    load_des_mask,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["wiener", "glimpse"])
    ap.add_argument("--scales-arcmin", nargs="+", type=float, default=list(SMOOTHING_SET_ARCMIN))
    ap.add_argument("--act-threshold", type=float, default=DEFAULT_ACT_THRESHOLD)
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/home/mlee1/ceph/paper3/B/wp1_maps"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[b1] loading masks + harmonizing footprint ...", flush=True)
    des_mask = load_des_mask()
    act_hp = act_mask_to_healpix(load_act_mask(), nside=1024)
    foot = harmonize(des_mask, act_hp, act_threshold=args.act_threshold)
    for line in foot.summary_lines():
        print("[b1]   " + line, flush=True)
    np.savez_compressed(
        args.out_dir / "common_footprint_nside1024.npz",
        weight=foot.weight.astype(np.float32), binary=foot.binary,
        act_threshold=foot.act_threshold, nside=foot.nside,
        sky_fraction_names=np.array(list(foot.sky_fractions), dtype=object),
        sky_fraction_values=np.array(list(foot.sky_fractions.values())),
    )

    summary = {
        "sky_fractions": {k: [v, v * FULL_SKY_DEG2] for k, v in foot.sky_fractions.items()},
        "act_threshold": args.act_threshold,
        "catalogs": {},
    }
    for variant in args.variants:
        m = load_des_map(variant, mask=des_mask)
        for s in args.scales_arcmin:
            cat = build_peak_catalog(m, foot.weight, foot.binary, s, variant)
            tag = f"{variant}_sm{s:g}am"
            np.savez_compressed(args.out_dir / f"peaks_{tag}.npz", **cat.to_npz_dict())
            summary["catalogs"][tag] = {
                "n_peaks": int(len(cat.ipix)), "n_raw": cat.n_raw,
                "n_excluded_mask_zone": cat.n_excluded,
                "map_mean": cat.map_mean, "map_sigma": cat.map_sigma,
                "nu_max": float(cat.nu.max()), "n_nu_ge_3": int((cat.nu >= 3).sum()),
            }
            print(f"[b1] {tag}: {len(cat.ipix)} peaks "
                  f"({cat.n_excluded} excluded near mask), sigma_map={cat.map_sigma:.3e}",
                  flush=True)

    (args.out_dir / "peak_catalog_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[b1] wrote {args.out_dir}/peak_catalog_summary.json", flush=True)


if __name__ == "__main__":
    main()
