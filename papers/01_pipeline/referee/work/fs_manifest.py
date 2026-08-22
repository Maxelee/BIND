#!/usr/bin/env python3
"""fidswap -- emit a machine-readable manifest of every rebuilt product.

Walks referee_work/fidswap/ and records path, size, mtime and the shipped file
each product replaces, so the figure re-render step can be driven off one file.

    python referee/work/fs_manifest.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

FS = Path("/mnt/home/mlee1/ceph/referee_work/fidswap")
SCI = Path("/mnt/home/mlee1/ceph/bind_science")
SB35 = Path("/mnt/home/mlee1/ceph/bind_sb35")

REPLACES = {
    "halo_atlas/fidtb49_snap{NNN}.npz": str(SCI / "halo_atlas/fid_snap{NNN}.npz"),
    "field_cache/field_stats_fidtb49.npz": str(SCI / "field_cache/field_stats_fid.npz"),
    "mf_cache/mf_nu8_snap096_fidtb49.npz": str(SCI / "mf_cache/mf_nu8_snap096.npz"),
    "profiles/perhalo_fidtb49_snap096.npz": str(SCI / "profiles/perhalo_fid_snap096.npz"),
    "composites/tb49_snap096_slabNN.npz":
        "/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096/composite_slabNN.npz",
}

NOT_REBUILT = {
    "bind_sb35/nu05_shards/sci_bind.npz":
        "per-realization nu05 draws for the new fiducial (build_nu_cache.py needs a "
        "--run_dir passthrough; the replica's own nu05_stats.npz has means only)",
    "bind_sb35/nu05n_shards/fid.npz":
        "noisy LSST-Y10 target; ~10 min for a 50-real shard, gates the R2 noisy twins",
    "bind_science/runs/twobound/run_*/paired_stats.npz":
        "60 twobound responses against the new fiducial -- the one job that needs a "
        "cluster allocation; see the measured per-task cost in the session report",
    "bind_science/runs/bind/run_0000/paired_perheal_fid.npz":
        "the fiducial per-realization cube the paired_stats CLI caches",
    "peak_counts_ngal10 / peak_counts_multi / peak_cross_*":
        "absent from every twobound replica; needed by the fig-4 shape-noise band",
}


def main():
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "canonical_fiducial": "bind_science/runs/twobound/run_0049",
           "replicas_built": [18, 49, 53], "products": [], "not_rebuilt": NOT_REBUILT}
    for p in sorted(FS.rglob("*.npz")):
        rel = str(p.relative_to(FS))
        key = None
        for k in REPLACES:
            if rel.split("_snap")[0] == k.split("_snap")[0].replace("{NNN}", ""):
                key = REPLACES[k]
                break
        out["products"].append({"path": str(p), "rel": rel,
                                "bytes": p.stat().st_size,
                                "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                       time.localtime(p.stat().st_mtime)),
                                "replaces": key})
    f = FS / "fidswap_manifest.json"
    f.write_text(json.dumps(out, indent=1))
    n = len(out["products"])
    tot = sum(d["bytes"] for d in out["products"]) / 1e9
    print(f"wrote {f}: {n} products, {tot:.2f} GB")


if __name__ == "__main__":
    main()
