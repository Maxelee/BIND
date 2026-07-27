#!/usr/bin/env python
"""P6b (docs/ksz_lightcone_map_plan.md §3-P6 stage B): the f~gas CAP_mat
denominator, from the fiducial co-traced massplane.

``FID/haloplane_trace/massplane_maps.npz`` (key ``'tau'``, shape
``(47,5,1024,1024)``) is the total-matter column painted into the tau slot
and traced through the **identical** fiducial geometry as every other map
(P2: kappa-identity bit-exact over all 47 realizations -- the co-trace shares
the exact seed/rot/disp of realizations 1..47 of the regular maps). Because
the constants that turn a painted density into a plane value
(``_tau_per_gas_pixel``-style) are the same for the matter and gas planes,

    f_gas(theta_d) = CAP_tau(gas; theta_d) / CAP_mat(theta_d)

with the physical constants cancelling exactly; the usual baryon fraction
normalization gives f~gas = f_gas / F_B, F_B = 0.049/0.3089
(``examples/_build_ksz_paper2_nb.py``'s constant, reused verbatim).

**CORRECTED 2026-07-24 (post-M1 defect fix).** The FIELD (the massplane map
itself) is fiducial-shared -- only one co-trace exists, and total matter at a
FIXED halo is feedback-insensitive at CAP apertures, so one trace legitimately
serves every node. But the PLAN's "CAP_mat is fiducial-shared" language was
mis-read in the first pass of this phase as "use the fiducial's OWN selected
galaxy POSITIONS for every node too" -- which is wrong for BGS: each Sobol
node's M*>cut selects a DIFFERENT subset of halos from THAT node's own
painted stellar mass (feedback changes which halos cross the threshold), the
exact same per-node selection already used for the tau-CAP numerator via
``mstar_matrix_snap085.npz``. Using the fiducial's fixed ~1750-halo,
mean logM200=13.44 sample as the denominator for e.g. node 64 (11 halos
selected, mean logM200=14.38 -- a ~1 dex mass mismatch, an extreme-feedback
node where almost nothing crosses M*>11 except the most massive halos)
inflates f~gas by roughly (M_node/M_fid)^~1 wherever CAP_mat scales with mass
-- exactly the unphysical f~gas>4-16 tails seen in the first version of M1.

**The fix:** the massplane FIELD stays fiducial-shared (only one trace
exists), but it is now stacked at **each node's own BGS selection**
(``lightcone_cap_stack.select_sample(sample, node)``, same function/positions
already used for the tau-CAP numerator) for bgs110/bgs1125 -- a genuine
per-node ``sb35_mean_<sample>``/``sb35_real_<sample>`` pair, mirroring
``taucap_lightcone.npz``'s schema. ELG and LRG select by a FIXED mass-proxy
cut with **no per-node dependence** (independently verified: ``select_sample
('elg'|'lrg', node)`` returns byte-identical positions for node in
{'fid', 5, 100, ...}) -- so their single fiducial-computed CAP_mat remains
exactly correct and is simply broadcast across the ``node_ids`` axis for a
uniform consumer API.

``--massbin`` (M5 addition, docs/ksz_lightcone_map_plan.md phase M5): builds
the SEPARATE ``capmat_massbin_lightcone.npz`` product -- the CAP_mat
denominator for the mass-binned f~gas-vs-logM200 figure (M5), from the same
fiducial co-traced massplane, stacked at the mass-bin catalogs' positions
(``desi_mock_massbin_snap085.npz`` / ``desi_mock_snap046.npz`` +
``_massbin.npz``) instead of the M*-selected BGS/ELG/LRG samples.

KEY DIFFERENCE FROM ``main()`` ABOVE: mass-bin selection is a pure FoF M200
cut on the lightcone catalog -- it does not depend on any Sobol node's
painted stellar mass, so (unlike bgs110/bgs1125, which needed the per-node
fix documented in ``main()``'s defect-and-fix history) there is no per-node
mismatch to guard against and no per-node sweep is needed here: the SAME
field (fiducial massplane) stacked at the SAME (node-independent) per-bin
positions is the correct denominator for every SB35 node's per-bin
tau-CAP numerator. This also means the numerator
(``lightcone_cap_stack.py --sample massbin85/46 --map tau``) and this
denominator (``--map massplane``) are AUTOMATICALLY stacked on IDENTICAL
halo samples per bin -- both read the exact same ``mass_bin`` column from
the exact same catalog file, with zero node-dependent re-selection anywhere
in the chain. This is the mass-binned-selection instance of the general
lesson from the BGS fix above (numerator and denominator must share halo
samples); it is automatic here rather than requiring a fix because the
selection itself never varies with the node -- there is no "per-node"
axis to get wrong in the first place.

Requires the four fiducial shards already built by
``lightcone_cap_stack.py --run fid --map {tau,massplane} --sample
massbin{85,46}`` (this function only merges the two ``--map massplane``
shards into one small product; it does not build them).

Usage
-----
    python examples/lightcone_capmat_merge.py
    python examples/lightcone_capmat_merge.py --massbin
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")

from lightcone_cap_stack import (
    GEOM_PATH, select_sample, theta_pix_grid, compute_stack, MASSPLANE_PATH, MASSPLANE_KEY,
)
from bind.inference.lux_geometry import load_geometry

CEPH = Path("/mnt/home/mlee1/ceph")
KS = CEPH / "bind_science/ksz_confront"
LIGHTCONE = KS / "lightcone"
SHARD_DIR = LIGHTCONE / "shards"

BGS_SAMPLES = ["bgs110", "bgs1125"]
SHARED_SAMPLES = ["elg", "lrg"]   # node-independent selection -- single trace suffices
SKIP_RUNS = {114, 115, 117}
ALL_NODES = sorted(set(range(256)) - SKIP_RUNS)
SRC_IDX = 4   # matches the gas-tau numerator's src_idx=4 (z_s=2.44, full LOS)
F_B = 0.0490 / 0.3089  # Omega_b/Omega_m (Planck/TNG) -- _build_ksz_paper2_nb.py convention


def fid_shard_path(sample: str) -> Path:
    return SHARD_DIR / f"{sample}_massplane_runfid_beamnone_src4.npz"


def main():
    out = {}
    theta_kind = theta_value = None
    scatter_pct = {}

    # ------------------------------------------------------------------
    # fiducial leg (all 4 samples): reuse the single-shard fiducial stack
    # already built by 'lightcone_cap_stack.py --run fid --map massplane'.
    # ------------------------------------------------------------------
    for sample in BGS_SAMPLES + SHARED_SAMPLES:
        d = np.load(fid_shard_path(sample), allow_pickle=True)
        if theta_kind is None:
            theta_kind = d["theta_kind"]
            theta_value = d["theta_value"]
        mat_real = d["stack_real"].astype(np.float32)
        mat_mean = d["mean"].astype(np.float64)
        out[f"fid_mean_{sample}"] = mat_mean
        out[f"fid_real_{sample}"] = mat_real
        out[f"fid_n_gal_{sample}"] = int(d["n_gal"])
        out[f"fid_n_real_{sample}"] = int(mat_real.shape[0])

        with np.errstate(invalid="ignore"):
            frac = np.nanstd(mat_real, axis=0) / np.abs(mat_mean)
        r200_idx = np.nonzero(theta_kind == "r200mult")[0]
        theta1_idx = r200_idx[np.argmin(np.abs(theta_value[r200_idx] - 1.0))]
        scatter_pct[sample] = float(frac[theta1_idx] * 100.0)
        print(f"[capmat] fiducial {sample}: n_real={mat_real.shape[0]} n_gal={int(d['n_gal'])} "
              f"CAP_mat realization-scatter @theta200=1: {scatter_pct[sample]:.2f}%")

    n_theta = len(theta_value)
    out["node_ids"] = np.array(ALL_NODES, dtype=np.int64)
    out["theta_kind"] = theta_kind
    out["theta_value"] = theta_value
    out["F_B"] = F_B
    out["src_idx"] = SRC_IDX
    out["z_s"] = "2.44 (full LOS, matches the gas-tau numerator's src_idx=4)"

    # ------------------------------------------------------------------
    # SHARED samples (elg, lrg): selection is node-independent -- broadcast
    # the fiducial stack across the node_ids axis (verified byte-identical
    # positions across nodes, see module docstring).
    # ------------------------------------------------------------------
    for sample in SHARED_SAMPLES:
        out[f"sb35_mean_{sample}"] = np.tile(out[f"fid_mean_{sample}"], (len(ALL_NODES), 1))
        out[f"sb35_real_{sample}"] = np.tile(out[f"fid_real_{sample}"][None, :, :], (len(ALL_NODES), 1, 1))
        out[f"n_gal_{sample}"] = np.full(len(ALL_NODES), out[f"fid_n_gal_{sample}"], dtype=np.int64)
        print(f"[capmat] {sample}: node-independent selection verified -- "
              f"broadcasting the single fiducial CAP_mat across all {len(ALL_NODES)} nodes.")

    # ------------------------------------------------------------------
    # BGS samples (bgs110, bgs1125): PER-NODE CAP_mat -- load the massplane
    # cube once, stack at EACH node's own M*-selected positions.
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"[capmat] loading massplane cube ({MASSPLANE_PATH}, key={MASSPLANE_KEY!r}, "
          f"src_idx={SRC_IDX}) once for the per-node BGS sweep ...", flush=True)
    d = np.load(MASSPLANE_PATH)
    cube = np.array(d[MASSPLANE_KEY][:, SRC_IDX])   # (n_real<=47, 1024, 1024)
    n_real = cube.shape[0]
    del d
    print(f"[capmat] loaded {cube.shape} in {time.time()-t0:.1f}s", flush=True)

    geom = load_geometry(GEOM_PATH)

    for sample in BGS_SAMPLES:
        sb35_mean = np.full((len(ALL_NODES), n_theta), np.nan, dtype=np.float64)
        sb35_real = np.full((len(ALL_NODES), n_real, n_theta), np.nan, dtype=np.float32)
        n_gal = np.full(len(ALL_NODES), -1, dtype=np.int64)
        t_sample = time.time()
        for k, node in enumerate(ALL_NODES):
            sel = select_sample(sample, node)
            n_gal[k] = sel["n_sel"]
            if sel["n_sel"] == 0:
                continue   # physically-empty node (bgs1125 runs 64/87) -- stays NaN
            tgrid = theta_pix_grid(sel["r200"], sel["plane_p"], geom)
            stack_real, n_gal_real = compute_stack(
                cube, geom, sel["pixel_i"], sel["pixel_j"], sel["plane_p"], tgrid,
                n_realizations=n_real)
            sb35_real[k] = stack_real.astype(np.float32)
            with np.errstate(invalid="ignore"):
                sb35_mean[k] = np.nanmean(stack_real, axis=0)
            if k % 40 == 0:
                print(f"[capmat]   {sample} node {node} ({k+1}/{len(ALL_NODES)}): "
                      f"n_sel={sel['n_sel']}  [{time.time()-t_sample:.0f}s elapsed]", flush=True)
        out[f"sb35_mean_{sample}"] = sb35_mean
        out[f"sb35_real_{sample}"] = sb35_real
        out[f"n_gal_{sample}"] = n_gal
        print(f"[capmat] {sample}: per-node CAP_mat done in {time.time()-t_sample:.0f}s "
              f"({int((n_gal>0).sum())}/{len(ALL_NODES)} nodes with >=1 galaxy)", flush=True)

    out["capmat_realization_scatter_pct_at_theta200eq1"] = np.array(
        [scatter_pct[s] for s in BGS_SAMPLES + SHARED_SAMPLES])
    out["capmat_realization_scatter_pct_samples"] = np.array(BGS_SAMPLES + SHARED_SAMPLES)

    readme = [
        "P6b capmat merge, CORRECTED (docs/ksz_lightcone_map_plan.md) -- the f~gas "
        "CAP_mat denominator from the fiducial co-traced massplane (FID/haloplane_trace/"
        "massplane_maps.npz, key 'tau', <=47 realizations, kappa-identity bit-exact "
        "vs the regular maps -- P2). The FIELD is fiducial-shared (one co-trace exists); "
        "for BGS the SELECTION POSITIONS are now per-node (bug-fixed, see module "
        "docstring) -- earlier used the fiducial's fixed selection for every node, "
        "inflating f~gas for mass-mismatched nodes (e.g. node 64: 11 halos "
        "mean logM200=14.38 vs the fiducial's 1753 halos at 13.44).",
        "f_gas(theta) = CAP_tau_gas(node,theta) / CAP_mat(node,theta) [constants cancel]; "
        "f~gas = f_gas / F_B. F_B = Omega_b/Omega_m = 0.049/0.3089.",
        "theta_kind/theta_value (n_theta,): identical grid to the tau/y/kappa CAP products.",
        "node_ids (253,): Sobol node index, ascending (same 253-node set as the other "
        "merged products).",
        "Per sample <s> in {bgs110, bgs1125}: PER-NODE denominator --",
        "  fid_mean_<s>/fid_real_<s>: the fiducial's OWN CAP_mat (fiducial's own BGS "
        "selection), for the fiducial f~gas only.",
        "  sb35_mean_<s> (253,n_theta): per-node realization-mean CAP_mat, using THAT "
        "node's own M*-selected positions (mstar_matrix_snap085.npz) -- NOT the "
        "fiducial's positions. NaN/n_gal=0 rows are the physically-empty bgs1125 "
        "nodes 64/87 (same convention as taucap_lightcone.npz).",
        "  sb35_real_<s> (253,n_real,n_theta) f4: per-node per-realization CAP_mat.",
        "  n_gal_<s> (253,): that node's own selected+in_crop galaxy count (matches "
        "taucap_lightcone.npz's n_gal_<s> exactly -- same selection function).",
        "Per sample <s> in {elg, lrg}: node-INDEPENDENT selection (fixed mass-proxy cut, "
        "verified byte-identical across nodes) -- sb35_mean_<s>/sb35_real_<s> are the "
        "single fiducial CAP_mat broadcast across all 253 nodes (not a per-node "
        "computation, just a uniform-API convenience).",
        "ERROR CONVENTION (unchanged): f~gas's realization scatter is carried by the "
        "NUMERATOR (that node's own tau-CAP realization spread) at a FIXED "
        "(realization-mean) denominator: fgas_tilde_real[k,r,theta] = "
        "tau_real[k,r,theta] / sb35_mean_<s>[k,theta] / F_B -- now using node k's OWN "
        "mean CAP_mat, not the fiducial's.",
        "capmat_realization_scatter_pct_at_theta200eq1 / _samples: the FIDUCIAL CAP_mat's "
        "47-realization scatter (std/mean*100) at theta/theta200=1, one value per sample "
        "-- informational, unchanged by this fix.",
        "Source: examples/lightcone_capmat_merge.py, from "
        "KS/lightcone/shards/{sample}_massplane_runfid_beamnone_src4.npz (fiducial leg) "
        "+ a direct per-node sweep over the massplane cube (BGS legs, this script, "
        "no shards written per node -- the massplane loads once, ~253x2 in-memory "
        "CAP stacks).",
    ]
    out["readme"] = np.array(readme)

    out_path = LIGHTCONE / "capmat_lightcone.npz"
    np.savez(out_path, **out)
    print(f"[capmat] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps({"capmat_realization_scatter_pct": scatter_pct}, indent=2))
    return scatter_pct


# ---------------------------------------------------------------------------
# --massbin: the M5 mass-binned CAP_mat denominator (see module docstring)
# ---------------------------------------------------------------------------

MASSBIN_SAMPLES_ORDER = ["massbin85", "massbin46"]
MASSBIN_LABELS = {"massbin85": "snap085", "massbin46": "snap046"}


def massbin_shard_path(sample: str) -> Path:
    return SHARD_DIR / f"{sample}_massplane_runfid_beamnone_src4.npz"


def main_massbin():
    """M5 plan Task 2: merge the two mass-binned fiducial massplane shards
    (``massbin85``/``massbin46``, ``--map massplane``) into
    ``KS/lightcone/capmat_massbin_lightcone.npz`` -- see module docstring
    for why (unlike the BGS legs of ``main()`` above) this needs no
    per-node sweep: mass-bin selection is node-independent by construction,
    so a SINGLE fiducial trace at the SAME per-bin positions used by the
    tau-CAP numerator legitimately serves every SB35 node."""
    out = {}
    for sample in MASSBIN_SAMPLES_ORDER:
        label = MASSBIN_LABELS[sample]
        shard = massbin_shard_path(sample)
        if not shard.exists():
            raise SystemExit(
                f"missing {shard} -- run: python examples/lightcone_cap_stack.py "
                f"--run fid --map massplane --sample {sample}")
        d = np.load(shard, allow_pickle=True)
        out[f"{label}_mean"] = d["mean"].astype(np.float64)
        out[f"{label}_real"] = d["stack_real"].astype(np.float32)
        out[f"{label}_mean_inpatch"] = d["mean_inpatch"].astype(np.float64)
        out[f"{label}_real_inpatch"] = d["stack_real_inpatch"].astype(np.float32)
        out[f"{label}_n_gal"] = d["n_gal"].astype(np.int64)
        out[f"{label}_n_gal_inpatch"] = d["n_gal_inpatch"].astype(np.int64)
        out[f"{label}_logM200_mean"] = d["logM200_mean"].astype(np.float64)
        out[f"{label}_mass_bin_edges"] = d["mass_bin_edges"].astype(np.float64)
        out[f"{label}_n_real"] = int(d["stack_real"].shape[1])
        if "theta_kind" not in out:
            out["theta_kind"] = d["theta_kind"]
            out["theta_value"] = d["theta_value"]
        print(f"[capmat_massbin] {label}: n_bins={len(d['n_gal'])} "
              f"n_real={d['stack_real'].shape[1]} n_gal_per_bin={d['n_gal'].tolist()}")

    out["F_B"] = F_B
    out["src_idx"] = SRC_IDX
    out["readme"] = np.array([
        "M5 plan Task 2 mass-binned CAP_mat denominator (docs/ksz_lightcone_map_plan.md).",
        "f_gas(bin,theta) = CAP_tau_gas(bin,theta) / CAP_mat(bin,theta) [constants cancel]; "
        "f~gas = f_gas / F_B, F_B = Omega_b/Omega_m = 0.049/0.3089.",
        "Mass-bin selection is a pure FoF M200 cut -- NODE-INDEPENDENT (unlike bgs110/"
        "bgs1125's M*-cut), so ONE fiducial massplane trace correctly serves every SB35 "
        "node's per-bin numerator; no per-node sweep needed (contrast main() above, which "
        "DOES need one for BGS -- see that function's docstring for the defect this avoids "
        "by construction: numerator and denominator here are stacked on IDENTICAL per-bin "
        "halo samples because both read the same mass_bin column from the same catalog, "
        "with no node-dependent re-selection anywhere in the chain).",
        "snap085_*/snap046_*: per-bin (7 bins, edges in *_mass_bin_edges) realization-mean "
        "('_mean') and per-realization ('_real', <=47 realizations) CAP_mat; '_inpatch' "
        "variants restrict to in_patch halos only (sub-1e13 reuse-regime honesty, "
        "populated for the 3 bins below the 1e13 painting floor, NaN elsewhere).",
        "theta_kind/theta_value (n_theta=24,): 18 xb + 5 r200mult + 1 fixed_arcmin "
        "(FIXED_APERTURE_ARCMIN=1.0', the ELG data's own innermost measured aperture, "
        "shared verbatim with the tau-CAP numerator shards' theta grid).",
        "logM200_mean (7,): the OBSERVED mean logM200 of halos landing in each bin (for "
        "diagnostics); the M5 figure plots against the geometric bin CENTER "
        "0.5*(edges[:-1]+edges[1:]) instead, matching examples/_reduce_fgas_lightcone.py's "
        "own convention (line ~343) and the per-halo headline figure's x-axis exactly.",
    ])

    out_path = LIGHTCONE / "capmat_massbin_lightcone.npz"
    np.savez(out_path, **out)
    print(f"[capmat_massbin] wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return out_path


if __name__ == "__main__":
    if "--massbin" in sys.argv:
        main_massbin()
    else:
        main()
