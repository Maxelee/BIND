#!/usr/bin/env python
"""P6a/P6b (docs/ksz_lightcone_map_plan.md §3-P6): merge the P5 CAP shards.

Assembles the per-(sample, map, beam, run) shards written by
``examples/lightcone_cap_stack.py`` (E3) into consumer-facing products, one
per (sample-group, map type, beam) combination. Two shard populations feed
this:

  P6a (BGS, 2016 shards): both BGS M* cuts (``bgs110`` M*>11.0, ``bgs1125``
  M*>11.25) x {tau, y, y+beam1.6', kappa} at snap085 (z~0.26) --

    KS/lightcone/taucap_lightcone.npz      map=tau,   beam=none,   src_idx=4 (z_s=2.44)
    KS/lightcone/ycap_lightcone.npz        map=y,     beam=none,   src_idx=4 (z_s=2.44)
    KS/lightcone/ycap_beam_lightcone.npz   map=y,     beam=1.6am,  src_idx=4 (z_s=2.44)
    KS/lightcone/kcap_lightcone.npz        map=kappa, beam=none,   src_idx=1 (z_s=1.0)

  P6b (ELG + LRG, added 2026-07-24, ~2530 more shards): single-sample
  products, one array per npz instead of per-sample-suffixed arrays (see
  ``_sample_key`` below -- kept as ``<key>_bgs110``-style names for schema
  uniformity even though there's only one sample) --

    KS/lightcone/elgtau_lightcone.npz        ELG map=tau,   beam=none,    src_idx=4 (z_s=2.44)
    KS/lightcone/elgtau_smear_lightcone.npz  ELG map=tau,   beam=0.52am,  src_idx=4 (deflection-smearing beam, P2)
    KS/lightcone/elgy_lightcone.npz          ELG map=y,     beam=none,    src_idx=4 (z_s=2.44)
    KS/lightcone/elgy_beam_lightcone.npz     ELG map=y,     beam=1.6am,   src_idx=4 (z_s=2.44)
    KS/lightcone/elgkappa_lightcone.npz      ELG map=kappa, beam=none,    src_idx=3 (z_s=2.0, behind ELG z=1.16 shell)
    KS/lightcone/lrgtau_lightcone.npz        LRG map=tau,   beam=none,    src_idx=4 (z_s=2.44)
    KS/lightcone/lrgy_lightcone.npz          LRG map=y,     beam=none,    src_idx=4 (z_s=2.44)
    KS/lightcone/lrgy_beam_lightcone.npz     LRG map=y,     beam=1.6am,   src_idx=4 (z_s=2.44)
    KS/lightcone/lrgkappa_lightcone.npz      LRG map=kappa, beam=none,    src_idx=2 (z_s=1.5, behind LRG z=0.503 shell)

253 Sobol nodes were ever ray-traced (runs 0114/0115/0117 excluded, see
docs/ksz_lightcone_map_plan.md P5), for BGS *and* ELG *and* LRG alike
(verified: identical node-id sets across all three sample families). Runs
0064/0087 have zero galaxies passing the M*>11.25 cut (a physical empty
sample, verified against the per-halo reference counts in P3 -- NOT missing
data) -- those two rows are NaN in the bgs1125 arrays of every BGS product;
ELG/LRG use mass-proxy selection (no per-run M* dependence) so they have no
empty-sample rows.

The fiducial kappa shards (``*_kappa_runfid_beamnone_src{1,2,3}.npz``) did
not exist in the P5 sweep output (P5 only produced tau/y fiducial shards for
BGS) and were generated as one-off local calls to
``lightcone_cap_stack.run_one`` (no Slurm; ~30s each) before merging.

Per product npz keys (see the embedded ``readme`` array for the same text):
    node_ids (253,) int64                 -- Sobol node index, ascending
    theta_kind, theta_value (23,)          -- shared theta grid (copied verbatim
                                               from a shard: XB dimensionless
                                               theta/theta200 [18] + R200_MULT [5])
    map_type, beam_fwhm_arcmin, src_idx    -- scalar metadata
    fid_mean_<sample>      (n_theta,)
    fid_real_<sample>      (50, n_theta) f4
    fid_n_gal_<sample>     scalar int
    sb35_mean_<sample>     (253, n_theta) f8  [NaN row = empty/missing]
    sb35_real_<sample>     (253, 50, n_theta) f4  [NaN rows = empty/missing]
    n_gal_<sample>         (253,) int64  [-1 = shard missing]
    readme                 array of strings, this docstring's key summary

No realization covariance is precomputed here -- consumers build their own
per-node covariance from the raw ``sb35_real_<sample>[k]`` (50, n_theta)
array (e.g. ``np.cov(sb35_real[k].T)``), per the plan's instruction to keep
the raw reals rather than a baked-in covariance choice.

Usage
-----
    python examples/lightcone_cap_merge.py                # all products
    python examples/lightcone_cap_merge.py --only kappa    # kcap_lightcone.npz only
    python examples/lightcone_cap_merge.py --only elgtau   # elgtau_lightcone.npz only
    python examples/lightcone_cap_merge.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
LIGHTCONE = KS / "lightcone"
SHARD_DIR = LIGHTCONE / "shards"

BGS_SAMPLES = ["bgs110", "bgs1125"]
SKIP_RUNS = {114, 115, 117}          # never ray-traced (P5)
ALL_NODES = sorted(set(range(256)) - SKIP_RUNS)   # 253, ascending
N_REAL = 50

PRODUCTS = [
    dict(out="taucap_lightcone.npz", samples=BGS_SAMPLES, map_type="tau", beam=None, src_idx=4,
         z_s="2.44 (full LOS)"),
    dict(out="ycap_lightcone.npz", samples=BGS_SAMPLES, map_type="y", beam=None, src_idx=4,
         z_s="2.44 (full LOS)"),
    dict(out="ycap_beam_lightcone.npz", samples=BGS_SAMPLES, map_type="y", beam=1.6, src_idx=4,
         z_s="2.44 (full LOS)"),
    dict(out="kcap_lightcone.npz", samples=BGS_SAMPLES, map_type="kappa", beam=None, src_idx=1,
         z_s="1.0 (mass anchor, just behind BGS)"),
    # --- P6b additions: ELG (z=1.16, snap046) + LRG (z=0.503, snap067) ---
    dict(out="elgtau_lightcone.npz", samples=["elg"], map_type="tau", beam=None, src_idx=4,
         z_s="2.44 (full LOS)"),
    dict(out="elgtau_smear_lightcone.npz", samples=["elg"], map_type="tau", beam=0.52, src_idx=4,
         z_s="2.44 (full LOS); beam=0.52am is the P2-derived deflection-smearing kernel at the ELG-z shell, NOT an instrument beam"),
    dict(out="elgy_lightcone.npz", samples=["elg"], map_type="y", beam=None, src_idx=4,
         z_s="2.44 (full LOS)"),
    dict(out="elgy_beam_lightcone.npz", samples=["elg"], map_type="y", beam=1.6, src_idx=4,
         z_s="2.44 (full LOS)"),
    dict(out="elgkappa_lightcone.npz", samples=["elg"], map_type="kappa", beam=None, src_idx=3,
         z_s="2.0 (mass anchor, just behind ELG z=1.16)"),
    dict(out="lrgtau_lightcone.npz", samples=["lrg"], map_type="tau", beam=None, src_idx=4,
         z_s="2.44 (full LOS)"),
    dict(out="lrgy_lightcone.npz", samples=["lrg"], map_type="y", beam=None, src_idx=4,
         z_s="2.44 (full LOS)"),
    dict(out="lrgy_beam_lightcone.npz", samples=["lrg"], map_type="y", beam=1.6, src_idx=4,
         z_s="2.44 (full LOS)"),
    dict(out="lrgkappa_lightcone.npz", samples=["lrg"], map_type="kappa", beam=None, src_idx=2,
         z_s="1.5 (mass anchor, just behind LRG z=0.503)"),
]


def shard_path(sample: str, map_type: str, run, beam, src_idx: int) -> Path:
    run_tag = "fid" if run == "fid" else f"{int(run):04d}"
    beam_tag = "none" if beam is None else f"{beam:g}am"
    return SHARD_DIR / f"{sample}_{map_type}_run{run_tag}_beam{beam_tag}_src{src_idx}.npz"


def _load(path: Path):
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def merge_one(map_type: str, beam, src_idx: int, z_s: str, samples=BGS_SAMPLES, verbose: bool = True):
    out = {}
    theta_kind = theta_value = None
    n_theta = None
    missing_report = {}

    for sample in samples:
        fid_path = shard_path(sample, map_type, "fid", beam, src_idx)
        fid = _load(fid_path)
        if fid is None:
            raise SystemExit(f"missing fiducial shard (required): {fid_path}")
        if theta_kind is None:
            theta_kind = fid["theta_kind"]
            theta_value = fid["theta_value"]
            n_theta = len(theta_value)

        sb35_mean = np.full((len(ALL_NODES), n_theta), np.nan, dtype=np.float64)
        sb35_real = np.full((len(ALL_NODES), N_REAL, n_theta), np.nan, dtype=np.float32)
        n_gal = np.full(len(ALL_NODES), -1, dtype=np.int64)
        missing = []
        for k, node in enumerate(ALL_NODES):
            d = _load(shard_path(sample, map_type, node, beam, src_idx))
            if d is None:
                missing.append(node)
                continue
            sb35_mean[k] = d["mean"].astype(np.float64)
            sr = d["stack_real"].astype(np.float32)
            sb35_real[k, : sr.shape[0]] = sr
            n_gal[k] = int(d["n_gal"])
        missing_report[sample] = missing

        out[f"fid_mean_{sample}"] = fid["mean"].astype(np.float64)
        out[f"fid_real_{sample}"] = fid["stack_real"].astype(np.float32)
        out[f"fid_n_gal_{sample}"] = int(fid["n_gal"])
        out[f"sb35_mean_{sample}"] = sb35_mean
        out[f"sb35_real_{sample}"] = sb35_real
        out[f"n_gal_{sample}"] = n_gal

        if verbose:
            n_ok = len(ALL_NODES) - len(missing)
            print(f"    sample={sample}: {n_ok}/{len(ALL_NODES)} node shards found"
                  + (f", missing={missing}" if missing else ""))

    out["node_ids"] = np.array(ALL_NODES, dtype=np.int64)
    out["theta_kind"] = theta_kind
    out["theta_value"] = theta_value
    out["map_type"] = map_type
    out["beam_fwhm_arcmin"] = -1.0 if beam is None else float(beam)
    out["src_idx"] = int(src_idx)

    readme = [
        f"P6a merge (docs/ksz_lightcone_map_plan.md) -- map={map_type}, "
        f"beam_fwhm_arcmin={'none' if beam is None else beam}, src_idx={src_idx} "
        f"(source plane z_s={z_s}).",
        "node_ids (253,): Sobol node index 0..255 excluding {114,115,117} "
        "(never ray-traced, P5). Ascending, shared across all products/samples.",
        "theta_kind/theta_value (23,): shared theta grid, copied verbatim from a "
        "shard -- 'xb' = XB=linspace(0.3,3.0,18) dimensionless theta_d/theta200 "
        "multiples (examples/_reduce_fgas_cap.py convention), 'r200mult' = "
        "[0.25,0.5,0.75,1.0,1.4] theta_d/theta200 multiples. theta200 is "
        "per-halo (r200_comoving/chi[plane]); this grid is dimensionless.",
        f"Per sample <s> in {{{', '.join(samples)}}} "
        "(bgs110=M*>11.0/bgs1125=M*>11.25 @snap085 z~0.26; elg=mass-proxy @snap046 "
        "z=1.16; lrg=mass-proxy @snap067 z=0.503):",
        "  fid_mean_<s> (n_theta,): fiducial realization-mean CAP(theta) (50 real. "
        "for tau/y/kappa; capped to 47 for map_type='massplane', see "
        "lightcone_cap_stack.py docstring).",
        "  fid_real_<s> (n_real,n_theta) f4: fiducial per-realization CAP(theta).",
        "  fid_n_gal_<s>: fiducial selected+in_crop galaxy count.",
        "  sb35_mean_<s> (253,n_theta) f8: per-node realization-mean CAP(theta); "
        "NaN row = empty sample (0 galaxies pass the M* cut at that node -- "
        "runs 64/87 for bgs1125, a physical empty sample verified against the "
        "per-halo reference counts in P3, NOT missing data; ELG/LRG use a fixed "
        "mass-proxy selection with no per-run dependence, so no NaN rows occur).",
        "  sb35_real_<s> (253,n_real,n_theta) f4: per-node per-realization CAP(theta), "
        "raw (no covariance baked in) -- consumers build their own realization "
        "covariance per node, e.g. np.cov(sb35_real_<s>[k].T, ...).",
        "  n_gal_<s> (253,) int64: selected+in_crop galaxy count per node "
        "(-1 = shard file missing entirely, should not occur; "
        "empty-sample nodes report 0).",
        "Source: examples/lightcone_cap_merge.py, from "
        "KS/lightcone/shards/*.npz (P5 sweep output [BGS] / the P6b ELG+LRG sweep "
        "+ a handful of fiducial kappa/massplane shards generated locally, since "
        "the sweeps only covered the tau/y fiducials).",
    ]
    out["readme"] = np.array(readme)
    return out, missing_report


PRODUCT_TAGS = {p["out"].removesuffix("_lightcone.npz"): p["out"] for p in PRODUCTS}
# legacy P6a aliases (original 4-product --only vocabulary)
LEGACY_ALIASES = {"tau": "taucap_lightcone.npz", "y": "ycap_lightcone.npz",
                   "y_beam": "ycap_beam_lightcone.npz", "kappa": "kcap_lightcone.npz"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(set(PRODUCT_TAGS) | set(LEGACY_ALIASES)), default=None,
                     help="merge a single product instead of all of them "
                          "(e.g. 'kcap', 'elgtau', 'lrgy_beam'; 'tau'/'y'/'y_beam'/'kappa' "
                          "are the legacy P6a aliases for the 4 BGS products)")
    ap.add_argument("--out_dir", type=Path, default=LIGHTCONE)
    args = ap.parse_args()

    sel = dict(PRODUCT_TAGS)
    sel.update(LEGACY_ALIASES)
    products = PRODUCTS if args.only is None else [p for p in PRODUCTS if p["out"] == sel[args.only]]

    summary = {}
    for p in products:
        print(f"[merge] {p['out']}  (map={p['map_type']}, beam={p['beam']}, "
              f"src_idx={p['src_idx']}, z_s={p['z_s']}, samples={p['samples']})")
        out, missing = merge_one(p["map_type"], p["beam"], p["src_idx"], p["z_s"],
                                  samples=p["samples"])
        out_path = args.out_dir / p["out"]
        np.savez(out_path, **out)
        n_merged = {s: len(ALL_NODES) - len(missing[s]) for s in p["samples"]}
        summary[p["out"]] = {"n_nodes_merged": n_merged, "missing": missing,
                              "n_theta": len(out["theta_value"])}
        print(f"  -> wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB); "
              f"n_nodes_merged={n_merged}")

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
