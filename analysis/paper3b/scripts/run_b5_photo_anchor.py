"""WP-B5 interpretation-ladder item 5: absolute photometric anchor vs an external measurement.

Anchor: Liu et al. 2025 (arXiv:2502.08850) measured stacked <y_CAP(R)> of
the DESI DR9 *extended* photometric LRG sample (Zhou et al. 2023, 4 photo-z
bins) on the SAME ACT DR6+Planck NILC Compton-y product our chain uses,
with the SAME disk/sqrt(2)-ring CAP filter and units (y arcmin^2), radii
1-6'. We repeat the measurement with OUR pipeline (our map copy, our
thumbnail + CAP code, our masks) on the same public sample and compare
amplitudes — an end-to-end absolute check of the ILC-map handling, units,
beam and CAP implementation, fully independent of the DES peak chain and of
the DR5 cluster anchor.

Zenodo caveat (discovered 2026-07-18): the published fig3/fig4 CSVs
(zenodo 14706729) carry an upload bug — every (pz bin x map-variant) column
contains the SAME series. The comparison therefore doubles as an
identification test: our four per-bin curves are compared against the one
genuine published series (+ its errors); a match of exactly one bin at the
expected level identifies the column and anchors the chain; the other bins
are anchored relatively (their ratios are internal to our pipeline).
[Recommend notifying the authors.]

Known convention differences vs Liu et al. (documented, expected few-%):
their cluster mask is ACT DR6 SNR>6 (not public here) — we veto DR5 SNR>6
within 10'; their point-source discs come from the DR6 source list — we
rely on the released wide mask (wExtended) + aperture-clean cut; they stack
the full masked sample — we subsample per bin (seeded; errors scale up,
means unbiased).

Job-free, CPU: chunked pixell thumbnails; ~30-40 min at 120k/bin.
Outputs -> /mnt/home/mlee1/ceph/paper3/B/wp5_inference/photo_anchor/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(_repo))

from analysis.paper3b.maps import load_act_mask, load_act_ymap, load_dr5_catalog  # noqa: E402
from analysis.paper3b.maps.stack import extract_thumbnails  # noqa: E402
from analysis.paper3b.stack.cap import cap_filter_multi  # noqa: E402

B_ROOT = Path("/mnt/home/mlee1/ceph/paper3/B")
DL = B_ROOT / "downloads" / "liu2025_tsz_desi"
OUT = B_ROOT / "wp5_inference" / "photo_anchor"
CAT = DL / "dr9_lrg_pzbins.fits"   # the MAIN sample: White+22-cut per-bin
                                   # counts reproduce Liu Table I (the
                                   # extended sample is 3x denser — wrong one)
STARDENS = DL / "pixweight-dr7.1-0.22.0_stardens_64_ring.fits"
LIU_RADII = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.125, 4.75, 5.375, 6.0])
RES_ARCMIN = 0.5
R_MAX = 15.0                      # house thumbnail convention (61x61 @ 0.5')
APER_CLEAN = np.sqrt(2) * 6.0     # largest-aperture outer edge (Liu convention)
CLUSTER_VETO_ARCMIN = 10.0


def white22_cuts(cat):
    """The official quality cuts (quality_cuts.py, White et al. 2022)."""
    import healpy as hp
    keep = ~((cat["DEC"] < -10.5) & (cat["RA"] > 120) & (cat["RA"] < 260))
    for b in "GRZ":
        keep &= cat[f"PIXEL_NOBS_{b}"] >= 2
    keep &= cat["lrg_mask"] == 0
    keep &= cat["EBV"] < 0.15
    from astropy.io import fits as afits
    sd = afits.getdata(STARDENS)
    bad = set(sd["HPXPIXEL"][sd["STARDENS"] >= 2500])
    hp_idx = hp.ang2pix(64, np.asarray(cat["RA"]), np.asarray(cat["DEC"]),
                        lonlat=True, nest=False)
    keep &= ~np.isin(hp_idx, np.fromiter(bad, dtype=np.int64))
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-bin", type=int, default=120_000)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--n-boot", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=20260718)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from astropy.io import fits as afits
    print("[anchor] loading catalog ...", flush=True)
    cat = afits.getdata(CAT)
    nraw = len(cat)
    cat = cat[cat["pz_bin"] > 0]          # tomographic sample only (Liu Table I)
    keep = white22_cuts(cat)
    cat = cat[keep]
    print(f"[anchor] {keep.sum():,} LRGs after pz-bin + White+22 cuts "
          f"({nraw:,} raw)", flush=True)

    ymap = load_act_ymap()
    mask = load_act_mask()

    # footprint + aperture-clean: apodized mask ~1 at the centre and on the
    # largest-aperture circle (8 azimuthal probes)
    ra, dec = np.asarray(cat["RA"], float), np.asarray(cat["DEC"], float)
    mval = mask.at(np.deg2rad(np.stack([dec, ra])), order=1)
    ok = mval >= 0.99
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for a in ang:
        dra = APER_CLEAN / 60.0 * np.cos(a) / np.clip(np.cos(np.deg2rad(dec)), 1e-3, None)
        ddec = APER_CLEAN / 60.0 * np.sin(a)
        mv = mask.at(np.deg2rad(np.stack([dec + ddec, ra + dra])), order=1)
        ok &= mv >= 0.99
    print(f"[anchor] {ok.sum():,} inside the aperture-clean ACT footprint", flush=True)

    # DR5 SNR>6 cluster veto (proxy for Liu's DR6 SNR>6 mask)
    dr5 = load_dr5_catalog(snr_min=6.0)
    from scipy.spatial import cKDTree

    def xyz(r, d):
        r, d = np.radians(r), np.radians(d)
        return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r),
                         np.sin(d)], axis=1)

    tree = cKDTree(xyz(dr5.ra_deg, dr5.dec_deg))
    chord = 2 * np.sin(np.radians(CLUSTER_VETO_ARCMIN / 60.0) / 2)
    near = tree.query_ball_point(xyz(ra, dec), chord, return_length=True) > 0
    ok &= ~near
    print(f"[anchor] {ok.sum():,} after DR5 SNR>6 cluster veto "
          f"({int(near.sum()):,} vetoed)", flush=True)

    cat, ra, dec = cat[ok], ra[ok], dec[ok]
    rng = np.random.default_rng(args.seed)

    liu = None
    try:
        import pandas as pd
        df = pd.read_csv(DL / "fig3.csv", index_col=0).dropna()
        liu = {"radii": df["RApArcmin"].values,
               "y": df["pz1_act_dr6_fiducial"].values,
               "err": df["pz1_act_dr6_fiducial_err"].values}
    except Exception as e:                          # pragma: no cover
        print(f"[anchor] zenodo vector unavailable: {e}")

    summary = {"n_per_bin_target": args.n_per_bin, "radii_arcmin": LIU_RADII.tolist(),
               "cluster_veto_arcmin": CLUSTER_VETO_ARCMIN,
               "zenodo_bug_note": "all fig3 columns identical; comparison = "
                                  "identification test vs the one genuine series",
               "bins": {}}
    curves = {}
    for b in (1, 2, 3, 4):
        sel = np.where(cat["pz_bin"] == b)[0]
        n_avail = sel.size
        if n_avail > args.n_per_bin:
            sel = rng.choice(sel, args.n_per_bin, replace=False)
        per_obj = np.empty((sel.size, len(LIU_RADII)))
        for i0 in range(0, sel.size, args.chunk):
            ii = sel[i0:i0 + args.chunk]
            th = np.asarray(extract_thumbnails(ymap, ra[ii], dec[ii],
                                               r_max_arcmin=R_MAX,
                                               res_arcmin=RES_ARCMIN))
            per_obj[i0:i0 + len(ii)] = cap_filter_multi(th, LIU_RADII, RES_ARCMIN)
            print(f"[anchor] pz{b}: {min(i0+args.chunk, sel.size):,}/{sel.size:,}",
                  flush=True)
        mean = per_obj.mean(axis=0)
        boot = np.empty((args.n_boot, len(LIU_RADII)))
        for k in range(args.n_boot):
            boot[k] = per_obj[rng.integers(0, sel.size, sel.size)].mean(axis=0)
        err = boot.std(axis=0)
        curves[b] = (mean, err)
        row = {"n_available": int(n_avail), "n_stacked": int(sel.size),
               "y_cap": mean.tolist(), "y_cap_err": err.tolist()}
        if liu is not None:
            ratio = mean / liu["y"]
            chi2 = float((((mean - liu["y"]) ** 2)
                          / (err ** 2 + liu["err"] ** 2)).sum())
            row["ratio_to_liu_series"] = ratio.tolist()
            row["chi2_vs_liu_series"] = chi2
            print(f"[anchor] pz{b}: <ratio> = {np.mean(ratio):.3f}, "
                  f"chi2 vs series = {chi2:.1f}/9")
        summary["bins"][f"pz{b}"] = row

    with open(OUT / "photo_anchor_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[anchor] wrote {OUT/'photo_anchor_summary.json'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    cols = ["#0072B2", "#009E73", "#CC79A7", "#D55E00"]
    for b, c in zip((1, 2, 3, 4), cols):
        m, e = curves[b]
        ax.errorbar(LIU_RADII + 0.02 * b, m, e, fmt="o-", ms=3, color=c,
                    label=f"this chain, pz{b}")
    if liu is not None:
        ax.errorbar(liu["radii"], liu["y"], liu["err"], fmt="ks", ms=5,
                    mfc="none", label="Liu+25 zenodo series (bug: single column)")
    ax.set(xlabel="CAP radius R [arcmin]",
           ylabel=r"$\langle y_{\rm CAP}\rangle$ [$y\,$arcmin$^2$]",
           title="Ladder item 5: absolute photometric anchor vs Liu et al. 2025")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "photo_anchor.png", dpi=150)
    print(f"[anchor] wrote {OUT/'photo_anchor.png'}")


if __name__ == "__main__":
    main()
