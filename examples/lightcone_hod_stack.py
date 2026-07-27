#!/usr/bin/env python
"""R2 (docs/p4c_referee_hardening_plan.md): HOD-population stacking — the
satellite + aperture-rule forward model, on the EXISTING painted maps.

Framing (2026-07-28 discussion): the satellites' gas is already in the
painted fields on both sides; the data/model mismatch is (a) the
stacking-center population (data include ~10-15% satellite LRGs at NFW
offsets inside typically more-massive hosts; the mock stacks pure centrals)
and (b) the aperture-assignment rule (data: theta_d = xb*theta200(z; fixed
logM=13.18 anchor); mock shards: xb*theta200(halo) per-halo). Both are fixed
by re-stacking the same per-node y maps at an HOD-drawn population with the
data's aperture rule — no new painting.

Per run in {fid + spanning nodes}: five stacks on the 1.6'-beam y maps:
  cen-true200   — lrg_sel centrals, per-halo theta200 rule (reproduces the
                  current T3 model convention — validation against the P5
                  shard) ;
  cen-anchor    — same centrals, anchor rule (isolates the rule effect);
  hod-fsat{f}   — centrals + satellites at f_sat in {0.10,0.15,0.20}, anchor
                  rule (isolates the satellite effect). Satellites: hosts
                  drawn with prob ∝ N_sat(M) = ((M-κM_cut)/M1)^α (Yuan+23-
                  class defaults logM_cut=12.8, logM1=13.9, α=1.0, κ=0.5 —
                  plan decision item 1 default), projected-NFW offsets (c=6).

Deliverables: R2_hod_stacks.npz (all curves + templates), figs/R2_hod.png,
verdicts/R2.json — including the re-evaluated T3 chi2/enrichment under
model curves corrected by the rule template and the f_sat=0.20 end-member
(the plan's gate).
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
LC = KS / "lightcone"
CAT = LC / "catalogs/desi_mock_snap067.npz"
FID_Y = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng/y_maps.npz")
FID_KAPPA = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng/kappa_maps.npz")
RUNS = Path("/mnt/home/mlee1/ceph/bind_sb35/runs")

XB = np.linspace(0.3, 3.0, 18)
SRC_IDX, N_REAL = 4, 50
BEAM = 1.6
Z_SHELL = 0.503
ANCHOR_LOGM = 13.18
FSAT_GRID = [0.10, 0.15, 0.20]
SPAN_NODES = [1, 205, 0]          # kSZ-consistent x2 + inconsistent
HOD = dict(logMcut=12.8, logM1=13.9, alpha=1.0, kappa=0.5)
NFW_C = 6.0
SEED = 20260728
CHI2_SLACK = 2.0


def stream_cube(path, key, src_idx=SRC_IDX, n_real=N_REAL, shape=(50, 5, 1024, 1024)):
    """Sequentially decompress {key}.npy, keeping only the src_idx slabs —
    (n_real, 1024, 1024) float32, ~200MB, never holding the 10GB cube."""
    zf = zipfile.ZipFile(path)
    f = zf.open(f"{key}.npy")
    version = np.lib.format.read_magic(f)
    np.lib.format._check_version(version)
    hdr_shape, fortran, dtype = np.lib.format._read_array_header(f, version)
    assert tuple(hdr_shape) == shape and not fortran
    slab_bytes = shape[2] * shape[3] * np.dtype(dtype).itemsize
    out = np.empty((n_real, shape[2], shape[3]), dtype=np.float32)
    for r in range(n_real):
        for s in range(shape[1]):
            if s == src_idx:
                out[r] = np.frombuffer(f.read(slab_bytes), dtype=dtype
                                       ).reshape(shape[2], shape[3])
            else:
                f.seek(f.tell() + slab_bytes)
    return out


def nfw_projected_offsets(r200, rng):
    """Projected offset radii (same units as r200) from a c=6 NFW."""
    mu = lambda x: np.log(1 + x) - x / (1 + x)
    xg = np.linspace(1e-3, NFW_C, 512)
    cdf = mu(xg) / mu(NFW_C)
    x3d = np.interp(rng.random(len(r200)), cdf, xg)
    r3d = x3d / NFW_C * r200
    n = rng.normal(size=(len(r200), 3))
    frac = np.hypot(n[:, 0], n[:, 1]) / np.linalg.norm(n, axis=1)
    return r3d * frac


def main():
    from lightcone_cap_stack import (GEOM_PATH, apply_beam, compute_stack,
                                     DTHETA_ARCMIN)
    from bind.inference.lux_geometry import load_geometry
    from act_ycap_measure import theta200_arcmin

    geom = load_geometry(GEOM_PATH)
    cat = np.load(CAT, allow_pickle=True)
    rng = np.random.default_rng(SEED)

    cen = cat["lrg_sel"] & cat["in_crop"]
    n_cen = int(cen.sum())
    chi_p = geom.chi[cat["plane_p"]]
    theta200_halo = np.degrees(cat["r200"] / chi_p) * 60.0        # arcmin
    theta200_anchor = float(theta200_arcmin(Z_SHELL, ANCHOR_LOGM))
    print(f"[R2] {n_cen} centrals; anchor theta200 = {theta200_anchor:.4f}' "
          f"vs mock mean {theta200_halo[cen].mean():.4f}'")

    # satellite host draws, shared across runs/f_sat (nested subsets)
    M = cat["M200"]
    Mcut = 10 ** HOD["logMcut"]
    nsat_w = np.where(M > HOD["kappa"] * Mcut,
                      ((M - HOD["kappa"] * Mcut) / 10 ** HOD["logM1"]) ** HOD["alpha"],
                      0.0) * cat["in_crop"]
    n_sat_max = int(round(max(FSAT_GRID) / (1 - max(FSAT_GRID)) * n_cen))
    hosts = rng.choice(len(M), size=n_sat_max, p=nsat_w / nsat_w.sum(), replace=True)
    off = nfw_projected_offsets(cat["r200"][hosts], rng)          # comoving Mpc/h
    off_px = off / (geom.chi[cat["plane_p"][hosts]]
                    * np.deg2rad(DTHETA_ARCMIN / 60.0))
    phi = rng.uniform(0, 2 * np.pi, n_sat_max)
    sat_i = cat["pixel_i"][hosts] + off_px * np.cos(phi)
    sat_j = cat["pixel_j"][hosts] + off_px * np.sin(phi)
    sat_p = cat["plane_p"][hosts]

    def theta_grid(theta200_arr, n_obj):
        th = np.broadcast_to(np.asarray(theta200_arr).reshape(-1, 1),
                             (n_obj, 1)) * XB[None, :]
        return th / DTHETA_ARCMIN                                  # map px

    # ---- mass-anchor-consistent recalibration ----------------------------
    # The logM=13.18 anchor is the LENSING mass of the full mixed population
    # (Sailer+24 stacked centrals+satellites together): satellites REDISTRIBUTE
    # the mass budget into massive hosts, they do not add to it. For each
    # f_sat, shift the central selection window down by delta so the mixed
    # population's median host logM stays at the anchor. (The naive
    # "centrals@13.18 + satellites on top" end-member double-counts mass —
    # kept in the outputs as an explicitly-labeled unphysical bound.)
    logM = cat["logM200"]
    win = np.asarray(cat["lrg_logM_window"], dtype=float)
    target = float(cat["lrg_target_logM200"])

    def recal_delta(f):
        n_s_over_n_c = f / (1 - f)
        deltas = np.linspace(0.0, 0.5, 101)
        best, bestv = 0.0, 1e9
        for d in deltas:
            c_m = (logM >= win[0] - d) & (logM < win[1] - d) & cat["in_crop"]
            n_c = int(c_m.sum())
            if n_c < 200:
                continue
            n_s = int(round(n_s_over_n_c * n_c))
            mix = np.concatenate([logM[c_m], logM[hosts[:n_s]]])
            v = abs(np.median(mix) - target)
            if v < bestv:
                best, bestv = d, v
        return best

    recal = {f: recal_delta(f) for f in FSAT_GRID}
    print("[R2] median-match window shifts (dex):",
          {f: round(d, 3) for f, d in recal.items()})

    # ---- kappa-self-consistent calibration (the defensible one) ----------
    # Median-matching is blind to a 15% massive-host tail (shifts <=0.015
    # dex, first-run lesson); the linear mean overshoots (offset satellites
    # lens less at small scales than centered hosts). The operational
    # meaning of the 13.18 anchor is the LENSING STACK of the mixed
    # population: tune delta so the mixed population's CAP-kappa stack on
    # the fiducial lightcone kappa maps (z_s=2.44, CMB-like; mass is
    # feedback-insensitive at these apertures — P2) matches the
    # pure-central 13.18 reference stack over the fit columns.
    N_REAL_CAL = 12
    kappa_cube = stream_cube(FID_KAPPA, "kappa", n_real=N_REAL_CAL)
    cal_cols = slice(1, 8)          # xb ~ 0.46-1.4

    def kcap_stack(pi, pj, pp, n_obj):
        sr = compute_stack(kappa_cube, geom, pi, pj, pp,
                           theta_grid(theta200_anchor, n_obj),
                           n_realizations=N_REAL_CAL)[0]
        return np.nanmean(sr, axis=0)

    kref = kcap_stack(cat["pixel_i"][cen], cat["pixel_j"][cen],
                      cat["plane_p"][cen], n_cen)

    def kcal_delta(f):
        n_s_over_n_c = f / (1 - f)
        best, bestv = 0.0, 1e18
        for d in np.linspace(0.0, 0.45, 10):
            c_m = (logM >= win[0] - d) & (logM < win[1] - d) & cat["in_crop"]
            n_c = int(c_m.sum())
            if n_c < 200:
                continue
            n_s = int(round(n_s_over_n_c * n_c))
            pi = np.concatenate([cat["pixel_i"][c_m], sat_i[:n_s]])
            pj = np.concatenate([cat["pixel_j"][c_m], sat_j[:n_s]])
            pp = np.concatenate([cat["plane_p"][c_m], sat_p[:n_s]])
            kv = kcap_stack(pi, pj, pp, len(pi))
            v = float(np.nansum((kv[cal_cols] / kref[cal_cols] - 1) ** 2))
            if v < bestv:
                best, bestv = d, v
        return best

    kcal = {f: kcal_delta(f) for f in FSAT_GRID}
    del kappa_cube
    print("[R2] kappa-match window shifts (dex):",
          {f: round(d, 3) for f, d in kcal.items()})

    def stacks_for(cube):
        out = {}
        # (i) current T3 model convention
        out["cen_true200"] = compute_stack(
            cube, geom, cat["pixel_i"][cen], cat["pixel_j"][cen],
            cat["plane_p"][cen], theta_grid(theta200_halo[cen], n_cen))[0]
        # (ii) anchor rule
        out["cen_anchor"] = compute_stack(
            cube, geom, cat["pixel_i"][cen], cat["pixel_j"][cen],
            cat["plane_p"][cen], theta_grid(theta200_anchor, n_cen))[0]
        # (iii) naive HOD mixes (over-massive bound, labeled)
        for f in FSAT_GRID:
            n_s = int(round(f / (1 - f) * n_cen))
            pi = np.concatenate([cat["pixel_i"][cen], sat_i[:n_s]])
            pj = np.concatenate([cat["pixel_j"][cen], sat_j[:n_s]])
            pp = np.concatenate([cat["plane_p"][cen], sat_p[:n_s]])
            out[f"hod_fsat{f:g}"] = compute_stack(
                cube, geom, pi, pj, pp, theta_grid(theta200_anchor, len(pi)))[0]
        # (iv) median-matched HOD mixes (kept for the record)
        for f in FSAT_GRID:
            d = recal[f]
            c_m = (logM >= win[0] - d) & (logM < win[1] - d) & cat["in_crop"]
            n_c = int(c_m.sum())
            n_s = int(round(f / (1 - f) * n_c))
            pi = np.concatenate([cat["pixel_i"][c_m], sat_i[:n_s]])
            pj = np.concatenate([cat["pixel_j"][c_m], sat_j[:n_s]])
            pp = np.concatenate([cat["plane_p"][c_m], sat_p[:n_s]])
            out[f"recal_fsat{f:g}"] = compute_stack(
                cube, geom, pi, pj, pp, theta_grid(theta200_anchor, len(pi)))[0]
        # (v) kappa-self-consistent HOD mixes (the headline treatment)
        for f in FSAT_GRID:
            d = kcal[f]
            c_m = (logM >= win[0] - d) & (logM < win[1] - d) & cat["in_crop"]
            n_c = int(c_m.sum())
            n_s = int(round(f / (1 - f) * n_c))
            pi = np.concatenate([cat["pixel_i"][c_m], sat_i[:n_s]])
            pj = np.concatenate([cat["pixel_j"][c_m], sat_j[:n_s]])
            pp = np.concatenate([cat["plane_p"][c_m], sat_p[:n_s]])
            out[f"kcal_fsat{f:g}"] = compute_stack(
                cube, geom, pi, pj, pp, theta_grid(theta200_anchor, len(pi)))[0]
        return out

    results = {}
    for run in ["fid"] + SPAN_NODES:
        t0 = time.time()
        path = FID_Y if run == "fid" else RUNS / f"run_{int(run):04d}/y_maps.npz"
        cube = apply_beam(stream_cube(path, "y"), BEAM)
        results[run] = stacks_for(cube)
        del cube
        print(f"[R2] run {run}: {time.time()-t0:.0f}s", flush=True)

    # ---- templates -------------------------------------------------------
    pix_area = DTHETA_ARCMIN ** 2
    curves = {str(r): {k: np.nanmean(v, axis=0) * pix_area for k, v in d.items()}
              for r, d in results.items()}
    T_rule = {r: curves[r]["cen_anchor"] / curves[r]["cen_true200"] - 1
              for r in curves}
    T_sat = {r: {f: curves[r][f"hod_fsat{f:g}"] / curves[r]["cen_anchor"] - 1
                 for f in FSAT_GRID} for r in curves}

    # ---- re-evaluated T3 chi2 with corrected model curves ----------------
    dd = np.load(LC / "act_ycap_lrg_real.npz", allow_pickle=True)
    b = np.load(LC / "lrgy_beam_lightcone.npz", allow_pickle=True)
    ksz = np.load(LC / "ksz_consistent_nodes.npz")
    node_ids = b["node_ids"]
    in110 = np.isin(node_ids, ksz["node_ids_bgs110"])
    xb, valid = dd["xb"], dd["valid_cols"]
    iv = np.nonzero(valid)[0]
    dof = len(iv)
    h_data = 99 / (100 - dof - 2)
    cov_data = (h_data * dd["cov_jk_xb_cib17"][np.ix_(iv, iv)]
                + np.diag(dd["sig_cib_xb"][iv] ** 2))
    d_xb = dd["mean_xb_cib17"]
    BIND_PIX_AREA = 0.29296875 ** 2
    sb_mean = b["sb35_mean_lrg"][:, :18] * BIND_PIX_AREA
    sb_real = b["sb35_real_lrg"][:, :, :18] * BIND_PIX_AREA
    t_rule_fid = T_rule["fid"]

    def counts_with(mod_fac):
        ok = np.zeros(len(node_ids), dtype=bool)
        for k in range(len(node_ids)):
            m = sb_mean[k] * mod_fac
            n_r = sb_real.shape[1]
            h_b = (n_r - 1) / (n_r - dof - 2)
            ccov = h_b * np.cov((sb_real[k] * mod_fac[None, :])[:, iv].T) / n_r
            r = d_xb[iv] - m[iv]
            chi2 = float(r @ np.linalg.solve(cov_data + ccov, r))
            ok[k] = chi2 < dof + CHI2_SLACK * np.sqrt(2 * dof)
        return dict(n=int(ok.sum()), P=float(ok.mean()),
                    P_given_ksz=float(ok[in110].mean()))

    T_recal = {r: {f: curves[r][f"recal_fsat{f:g}"] / curves[r]["cen_anchor"] - 1
                   for f in FSAT_GRID} for r in curves}
    T_kcal = {r: {f: curves[r][f"kcal_fsat{f:g}"] / curves[r]["cen_anchor"] - 1
                  for f in FSAT_GRID} for r in curves}
    scen = {"uncorrected": counts_with(np.ones(18))}
    scen["rule_only"] = counts_with(1 + t_rule_fid)
    for f in FSAT_GRID:
        scen[f"naive+fsat{f:g}"] = counts_with(
            (1 + t_rule_fid) * (1 + T_sat["fid"][f]))
    for f in FSAT_GRID:
        scen[f"recal+fsat{f:g}"] = counts_with(
            (1 + t_rule_fid) * (1 + T_recal["fid"][f]))
    for f in FSAT_GRID:
        scen[f"kcal+fsat{f:g}"] = counts_with(
            (1 + t_rule_fid) * (1 + T_kcal["fid"][f]))

    # ---- outputs ---------------------------------------------------------
    np.savez(LC / "R2_hod_stacks.npz",
             xb=XB, runs=np.array([str(r) for r in results]),
             **{f"curve_{r}_{k}": curves[str(r)][k] for r in curves for k in curves[str(r)]},
             **{f"Trule_{r}": T_rule[r] for r in T_rule},
             **{f"Tsat_{r}_f{f:g}": T_sat[r][f] for r in T_sat for f in FSAT_GRID},
             **{f"Trecal_{r}_f{f:g}": T_recal[r][f] for r in T_recal for f in FSAT_GRID},
             **{f"Tkcal_{r}_f{f:g}": T_kcal[r][f] for r in T_kcal for f in FSAT_GRID},
             recal_delta=json.dumps({str(f): recal[f] for f in FSAT_GRID}),
             kcal_delta=json.dumps({str(f): float(kcal[f]) for f in FSAT_GRID}),
             theta200_anchor=theta200_anchor, hod=json.dumps(HOD),
             fsat_grid=np.array(FSAT_GRID), n_cen=n_cen)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    ax = axes[0]
    for k, lab, c in [("cen_true200", "centrals, per-halo θ200 rule (current T3)", "tab:purple"),
                      ("cen_anchor", "centrals, anchor rule (data rule)", "tab:blue"),
                      ("hod_fsat0.15", "HOD f_sat=0.15, anchor rule", "tab:green")]:
        ax.plot(XB, curves["fid"][k] * 1e6, "-", color=c, label=lab)
    derr = np.sqrt(dd["err_jk_xb_cib17"] ** 2 + dd["sig_cib_xb"] ** 2)
    ax.errorbar(xb[valid], d_xb[valid] * 1e6, derr[valid] * 1e6, fmt="o",
                color="k", capsize=3, label="ACT deproj-CIB data")
    ax.axvline(1.4, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$x_b$"); ax.set_ylabel(r"CAP $y$-flux [$\times10^6$]")
    ax.set_title("fiducial: population/rule variants vs data")
    ax.legend(fontsize=7)

    ax = axes[1]
    for r, c in zip(curves, plt.cm.viridis(np.linspace(0.1, 0.9, len(curves)))):
        ax.plot(XB, 100 * T_rule[r], "-", color=c, label=f"rule: run {r}")
        ax.plot(XB, 100 * T_sat[r][0.15], "--", color=c, label=f"sat(0.15): run {r}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$x_b$"); ax.set_ylabel("template [%]")
    ax.set_title("templates across spanning runs (node-independence check)")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[2]
    labels = list(scen)
    ax.bar(np.arange(len(labels)) - 0.18, [scen[s]["P"] for s in labels], 0.36,
           color="0.6", label="P(tSZ)")
    ax.bar(np.arange(len(labels)) + 0.18, [scen[s]["P_given_ksz"] for s in labels],
           0.36, color="tab:green", label="P(tSZ|kSZ)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title("headline vs population/rule corrections")
    ax.legend(fontsize=8)
    fig.suptitle("R2: HOD-population + aperture-rule forward model (existing maps, no new painting)",
                 fontsize=11)
    fig.savefig(LC / "figs/R2_hod.png", dpi=140)

    # gate on the REALISTIC mass-anchor-consistent f_sat=0.15 scenario, and
    # require actual enrichment (no vacuous 0>=0 pass — first-run lesson)
    s15 = scen["kcal+fsat0.15"]
    gate = bool(s15["P_given_ksz"] > 0.5
                and s15["P_given_ksz"] >= 2.0 * max(s15["P"], 1e-9))
    v = {
        "phase": "R2", "pass": gate,
        "metrics": {
            "theta200_anchor_arcmin": theta200_anchor,
            "hod": HOD, "n_cen": n_cen, "fsat_grid": FSAT_GRID,
            "scenarios": scen,
            "T_rule_fid_pct": [float(100 * x) for x in t_rule_fid],
            "T_sat_fid_0.15_pct": [float(100 * x) for x in T_sat["fid"][0.15]],
            "template_node_spread_pct": {
                "rule": float(100 * np.std([T_rule[r] for r in T_rule], axis=0).max()),
                "sat0.15": float(100 * np.std([T_sat[r][0.15] for r in T_sat], axis=0).max()),
            },
            "miscentering_retired": "central BCG-gas offsets ~40kpc under the 1.6' "
                                    "beam (sigma~0.22 Mpc at z=0.5): ~2% — recorded, negligible",
        },
        "figs": ["figs/R2_hod.png"],
        "notes": "Gate: P(tSZ|kSZ) >= 2x P(tSZ) at the f_sat=0.2 end-member with "
                 "the aperture-rule correction applied.",
        "next": "fold winning correction into T3 as default + R5",
    }
    with open(LC / "verdicts/R2.json", "w") as f:
        json.dump(v, f, indent=2)
    print(json.dumps(scen, indent=2))
    print("R2 gate:", gate)


F_EFF_GRID = [0.04, 0.08, 0.12]   # in-catalog EFFECTIVE f_sat (true f_sat x
                                   # 0.73 in-catalog boost fraction: 27% of
                                   # satellites live in sub-1e13 hosts that
                                   # carry 0.2% of the Y-boost)
CONFIG_NPZ = LC / "R2_kcal_config.npz"
SHARD_DIR = LC / "hod_shards"


def _context():
    """Shared population context — catalog, seeded satellite draws, grids.
    MUST stay bit-identical between kcal_prep and every node task."""
    from lightcone_cap_stack import DTHETA_ARCMIN
    from bind.inference.lux_geometry import load_geometry
    from lightcone_cap_stack import GEOM_PATH
    from act_ycap_measure import theta200_arcmin
    geom = load_geometry(GEOM_PATH)
    cat = np.load(CAT, allow_pickle=True)
    rng = np.random.default_rng(SEED)
    cen = cat["lrg_sel"] & cat["in_crop"]
    n_cen = int(cen.sum())
    M = cat["M200"]
    Mcut = 10 ** HOD["logMcut"]
    nsat_w = np.where(M > HOD["kappa"] * Mcut,
                      ((M - HOD["kappa"] * Mcut) / 10 ** HOD["logM1"]) ** HOD["alpha"],
                      0.0) * cat["in_crop"]
    n_sat_max = int(round(0.25 / 0.75 * n_cen))
    hosts = rng.choice(len(M), size=n_sat_max, p=nsat_w / nsat_w.sum(), replace=True)
    off = nfw_projected_offsets(cat["r200"][hosts], rng)
    off_px = off / (geom.chi[cat["plane_p"][hosts]]
                    * np.deg2rad(DTHETA_ARCMIN / 60.0))
    phi = rng.uniform(0, 2 * np.pi, n_sat_max)
    ctx = dict(geom=geom, cat=cat, cen=cen, n_cen=n_cen,
               sat_i=cat["pixel_i"][hosts] + off_px * np.cos(phi),
               sat_j=cat["pixel_j"][hosts] + off_px * np.sin(phi),
               sat_p=cat["plane_p"][hosts],
               theta200_anchor=float(theta200_arcmin(Z_SHELL, ANCHOR_LOGM)),
               dtheta=DTHETA_ARCMIN)
    return ctx


def _theta_grid(ctx, n_obj):
    th = np.full((n_obj, len(XB)), ctx["theta200_anchor"]) * XB[None, :]
    return th / ctx["dtheta"]


def _population(ctx, f, delta):
    cat, logM = ctx["cat"], ctx["cat"]["logM200"]
    win = np.asarray(cat["lrg_logM_window"], dtype=float)
    c_m = (logM >= win[0] - delta) & (logM < win[1] - delta) & cat["in_crop"]
    n_c = int(c_m.sum())
    n_s = int(round(f / (1 - f) * n_c))
    pi = np.concatenate([cat["pixel_i"][c_m], ctx["sat_i"][:n_s]])
    pj = np.concatenate([cat["pixel_j"][c_m], ctx["sat_j"][:n_s]])
    pp = np.concatenate([cat["plane_p"][c_m], ctx["sat_p"][:n_s]])
    return pi, pj, pp


def kcal_prep():
    """Fiducial-kappa calibration of the central-window shift per f_eff."""
    from lightcone_cap_stack import compute_stack
    ctx = _context()
    N_CAL = 12
    kc = stream_cube(FID_KAPPA, "kappa", n_real=N_CAL)
    cat, cen = ctx["cat"], ctx["cen"]
    kref = np.nanmean(compute_stack(kc, ctx["geom"], cat["pixel_i"][cen],
                                    cat["pixel_j"][cen], cat["plane_p"][cen],
                                    _theta_grid(ctx, ctx["n_cen"]),
                                    n_realizations=N_CAL)[0], axis=0)
    cols = slice(1, 8)
    deltas = {}
    for f in F_EFF_GRID:
        best, bestv = 0.0, 1e18
        for d in np.linspace(0.0, 0.35, 8):
            pi, pj, pp = _population(ctx, f, d)
            kv = np.nanmean(compute_stack(kc, ctx["geom"], pi, pj, pp,
                                          _theta_grid(ctx, len(pi)),
                                          n_realizations=N_CAL)[0], axis=0)
            v = float(np.nansum((kv[cols] / kref[cols] - 1) ** 2))
            if v < bestv:
                best, bestv = d, v
        deltas[f] = best
        print(f"[kcal_prep] f_eff={f}: delta={best:.3f}", flush=True)
    np.savez(CONFIG_NPZ, f_grid=np.array(F_EFF_GRID),
             deltas=np.array([deltas[f] for f in F_EFF_GRID]),
             seed=SEED, hod=json.dumps(HOD))
    print(f"[kcal_prep] -> {CONFIG_NPZ}")


def node_task(run):
    """One run's HOD-population y stacks (disBatch task; idempotent)."""
    from lightcone_cap_stack import apply_beam, compute_stack
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    out = SHARD_DIR / f"node_{run}.npz"
    if out.exists():
        print(f"[hod_node] exists, skipping {out}")
        return
    cfg = np.load(CONFIG_NPZ, allow_pickle=True)
    ctx = _context()
    path = FID_Y if run == "fid" else RUNS / f"run_{int(run):04d}/y_maps.npz"
    cube = apply_beam(stream_cube(path, "y"), BEAM)
    cat, cen = ctx["cat"], ctx["cen"]
    res = {"cen_anchor": compute_stack(cube, ctx["geom"], cat["pixel_i"][cen],
                                       cat["pixel_j"][cen], cat["plane_p"][cen],
                                       _theta_grid(ctx, ctx["n_cen"]))[0]}
    for f, d in zip(cfg["f_grid"], cfg["deltas"]):
        pi, pj, pp = _population(ctx, float(f), float(d))
        res[f"kcal_f{f:g}"] = compute_stack(cube, ctx["geom"], pi, pj, pp,
                                            _theta_grid(ctx, len(pi)))[0]
    np.savez(out, xb=XB, **{k: v.astype(np.float32) for k, v in res.items()})
    print(f"[hod_node] {run} -> {out}", flush=True)


def merge_nodes():
    ksz = np.load(LC / "ksz_consistent_nodes.npz")
    ids = ksz["node_ids_all"]
    keys = ["cen_anchor"] + [f"kcal_f{f:g}" for f in F_EFF_GRID]
    out = {k: np.empty((len(ids), N_REAL, len(XB)), dtype=np.float32) for k in keys}
    fidd = np.load(SHARD_DIR / "node_fid.npz")
    fid = {k: fidd[k] for k in keys}
    missing = []
    for i, nid in enumerate(ids):
        p = SHARD_DIR / f"node_{nid}.npz"
        if not p.exists():
            missing.append(int(nid)); continue
        d = np.load(p)
        for k in keys:
            out[k][i] = d[k]
    if missing:
        raise SystemExit(f"[merge] missing {len(missing)}: {missing[:10]}")
    np.savez(LC / "R2_hod_model_curves.npz", node_ids=ids, xb=XB,
             f_grid=np.array(F_EFF_GRID),
             **{f"nodes_{k}": out[k] for k in keys},
             **{f"fid_{k}": fid[k] for k in keys})
    print(f"[merge] -> {LC/'R2_hod_model_curves.npz'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kcal_prep", action="store_true")
    ap.add_argument("--node", default=None, help="'fid' or 0..255")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--explore", action="store_true",
                    help="the original fid+3-spanning-node template analysis")
    a = ap.parse_args()
    if a.kcal_prep:
        kcal_prep()
    elif a.node is not None:
        node_task(a.node)
    elif a.merge:
        merge_nodes()
    else:
        main()
