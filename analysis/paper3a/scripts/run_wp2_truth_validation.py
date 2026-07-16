"""WP-A2 Popeye half — TNG300-hydro truth validation driver (plan task 5-6).

Runs the observable operators twice per matched halo — once on TNG300-hydro
*truth* projections (transformed stage-1 frame), once on BIND-painted fiducial
maps of the same DMO halos — and persists, per snapshot:

  * Sigma_model (bootstrap covariance of the stacked painted-minus-truth
    residual) for kSZ tau_CAP, cylindrical f_gas, and Y, via
    ``observables.truth_validation.save_sigma_model``;
  * the calibrated CylToSph correction (M_gas 3D-sphere<R500c from particles /
    M_gas projected cylinder from the 50 h^-1 Mpc truth slab), mean + scatter;
  * the x_e residual (tau from actual ElectronAbundance vs the x_e=1.158
    assumption on the same truth gas map) — the ionization-assumption error;
  * the Y-M relation (slope/norm/scatter) truth vs painted;
  * (snap 063 only) the multi-sample convergence of each operator over >=8
    generative draws -> the per-halo sample count.

One array task per snapshot (096/071/067/063 = z ~ 0.03/0.42/0.50/0.60, which
doubles as the z>0 thermo verdict, SHARED_CONTEXT caveat 6). Heavy: the truth
projection streams the full hydro snapshot (~600 files) and needs a big-memory
node — ⛔ Max submits (see run_wp2_truth_validation.sh). Pure orchestration over
unit-tested building blocks; **do not tune to agree** (plan step 7).

Requires the fiducial paint outputs (run_wp2_fiducial_paint.sh) under
--painted-root, one ``snap_<NNN>/composite_slab*.npz`` per snapshot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from bind.inference.pipeline import (  # noqa: E402
    circular_taper_weight,
    extract_periodic_cutout,
    paste_halos_2d,
    square_taper_weight,
)
from analysis.paper3a.observables import (  # noqa: E402
    CylToSphCorrection,
    FlatLCDM,
    KSZOperatorConfig,
    PatchGeometry,
    aperture_gas_mass_msunh,
    fgas_cylindrical,
    fit_ym_relation,
    ksz_cap_profile,
    multi_sample_convergence,
    stacked_residual_bootstrap,
    save_sigma_model,
    y_aperture_mpc2,
    X_E_FULLY_IONIZED,
)
from analysis.paper3a.observables import frame_transforms as ft  # noqa: E402
from analysis.paper3a.observables import truth_projection as tp  # noqa: E402
from analysis.paper3a.data_vectors.data_vectors import load_ksz_qu2026_lrg_fiducial  # noqa: E402

# snap -> a-priori z (manifest is authoritative; this is only for labels/logs)
SNAPS = {96: 0.0337, 71: 0.4200, 67: 0.5030, 63: 0.5985}
CONDITIONS = "/mnt/home/mlee1/ceph/bind_portable_twobound/conditions"
PATCH_PIX = 128
CUTOUT_PIX = 361  # 17.6 h^-1 Mpc, above the Qu-vector minimum extent at z=0.6
KSZ_MASS_BIN = (10 ** 13.3, 10 ** 13.6)  # M200c band for the kSZ stack (smoke convention)


def _mass_matched_composite(d, box, npix, channel_source):
    """Paste one channel into an alpha-normalised composite (smoke convention).

    channel_source(patches) -> (N, 128, 128). Mass channels are rescaled to
    their DMO condition sums; thermo channels are pasted as-is (already physical
    per-pixel units).
    """
    centers = d["halo_centers"]
    r200 = d["halo_r200"]
    n = len(r200)
    patches = channel_source(d).astype(np.float32).copy()
    ppm = npix / box
    halos = [{"halo_center": centers[i], "r200": float(r200[i])} for i in range(n)]
    weights_list = [circular_taper_weight(PATCH_PIX, r_pix=float(r200[i]) * ppm * 4.0, taper_frac=0.15)
                    for i in range(n)]
    sq = square_taper_weight(PATCH_PIX, taper_frac=0.15)
    # paste_halos_2d expects (N, C, H, W); wrap single channel as C=1
    canvas, w_accum = paste_halos_2d(npix, box, halos, patches[:, None], sq, weights_list=weights_list)
    alpha = np.clip(w_accum, 0.0, 1.0)
    return alpha * canvas[0]


def _gas_channel(d):
    return d["generated_patches"][:, 1]  # DM/gas/stars -> gas


def _y_channel(d):
    return d["thermo_patches"][:, 0]  # compton_y/T/K/P_e -> compton_y


def process_snapshot(snap: int, painted_root: Path, out_dir: Path, halo_subset: int) -> dict:
    manifest = ft.load_stage1_manifest(Path(CONDITIONS) / f"snap_{snap:03d}")
    box, npix, ppm = manifest.box_size, manifest.npix, manifest.npix / manifest.box_size
    z = manifest.redshift
    geom = PatchGeometry(pixel_mpch=box / npix, z=z, cosmology=FlatLCDM(omega_m=manifest.raw["Omega_m"]))
    depth = manifest.slab_depth_hmpc

    print(f"[snap {snap:03d}] z={z:.4f} depth={depth:.1f} h^-1Mpc — projecting truth ...")
    truth = tp.project_truth_maps(manifest)                       # per-slab maps
    hydro = tp.load_hydro_halo_catalog(manifest)                  # M500c/R500c source

    ksz_cfg = KSZOperatorConfig(
        radii_arcmin=load_ksz_qu2026_lrg_fiducial().bins,
        beam_fwhm_arcmin=1.6, z_eff=z, v_rms_over_c=None,          # tau_CAP only (A5 sets v_rms)
    )

    # accumulate per-halo observables (truth, painted) across slabs
    rec = {k: [] for k in ("ksz_t", "ksz_p", "fgas_t", "fgas_p", "y_t", "y_p",
                           "m500", "xe_true", "xe_assumed", "cyl_gas", "sph_gas",
                           "sph_center", "sph_r500")}
    for si in range(manifest.n_slabs):
        cf = np.load(Path(CONDITIONS) / f"snap_{snap:03d}" / f"stage1_slab{si:02d}.npz")
        pf = np.load(painted_root / f"snap_{snap:03d}" / f"composite_slab{si:02d}.npz")
        cond_centers = cf["halo_centers"]; cond_m200 = cf["halo_masses"]
        hidx, matched = tp.match_condition_halos(cond_centers, cond_m200, hydro, box)

        gas_comp_p = _mass_matched_composite(pf, box, npix, _gas_channel)
        y_comp_p = _mass_matched_composite(pf, box, npix, _y_channel)
        gas_slab_t, y_slab_t = truth["gas_sigma"][si], truth["compton_y"][si]
        gxe_slab_t = truth["gas_sigma_xe"][si]

        for k in range(len(cond_centers)):
            if not matched[k]:
                continue
            j = hidx[k]
            cx = int(cond_centers[k][0] * ppm) % npix
            cy = int(cond_centers[k][1] * ppm) % npix
            r500, m500 = float(hydro.r500c[j]), float(hydro.m500c[j])
            if m500 <= 0 or r500 <= 0:
                continue

            # --- kSZ CAP tau (composite cutouts; only the mass band is stacked) ---
            if KSZ_MASS_BIN[0] <= cond_m200[k] < KSZ_MASS_BIN[1]:
                gc_t = extract_periodic_cutout(gas_slab_t, cx, cy, CUTOUT_PIX)
                gc_p = extract_periodic_cutout(gas_comp_p, cx, cy, CUTOUT_PIX)
                rec["ksz_t"].append(ksz_cap_profile(gc_t, geom, ksz_cfg))
                rec["ksz_p"].append(ksz_cap_profile(gc_p, geom, ksz_cfg))

            # --- f_gas cylindrical (<R500c projected) truth vs painted ---
            gt = extract_periodic_cutout(gas_slab_t, cx, cy, CUTOUT_PIX)
            gp = extract_periodic_cutout(gas_comp_p, cx, cy, CUTOUT_PIX)
            rec["fgas_t"].append(fgas_cylindrical(gt, geom, r500, m500))
            rec["fgas_p"].append(fgas_cylindrical(gp, geom, r500, m500))
            rec["cyl_gas"].append(aperture_gas_mass_msunh(gt, geom, r500))

            # --- Y(<R500c) cylindrical truth vs painted ---
            yt = extract_periodic_cutout(y_slab_t, cx, cy, CUTOUT_PIX)
            yp = extract_periodic_cutout(y_comp_p, cx, cy, CUTOUT_PIX)
            rec["y_t"].append(y_aperture_mpc2(yt, geom, r500))
            rec["y_p"].append(y_aperture_mpc2(yp, geom, r500))
            rec["m500"].append(m500)

            # --- x_e residual (truth-only): tau_true vs tau_assumed on truth gas ---
            gxe_t = extract_periodic_cutout(gxe_slab_t, cx, cy, CUTOUT_PIX)
            rec["xe_true"].append(aperture_gas_mass_msunh(gxe_t, geom, r500))       # ~ x_e-weighted mass
            rec["xe_assumed"].append(X_E_FULLY_IONIZED * aperture_gas_mass_msunh(gt, geom, r500))

            # --- CylToSph: collect this halo's 3D center + R500 for the particle pass ---
            rec["sph_center"].append([hydro.centers_xy[j][0], hydro.centers_xy[j][1], hydro.los[j]])
            rec["sph_r500"].append(r500)

    n_matched = len(rec["m500"])
    print(f"[snap {snap:03d}] matched {n_matched} halos across slabs")

    # ---- CylToSph correction on a halo subset (particle sphere / map cylinder) ----
    cyltosph = _calibrate_cyltosph(manifest, rec, depth, halo_subset)

    # ---- assemble outputs ----
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_base = {
        "truth_inputs": manifest.hydro_snapdir(),
        "model_or_checkpoint": str(painted_root),
        "script": "analysis/paper3a/scripts/run_wp2_truth_validation.py",
        "date": "2026-07-16",
    }
    summary = {"snap": snap, "z": z, "n_matched": n_matched, "depth_hmpc": depth,
               "cyltosph": cyltosph}

    def _persist(name, t_list, p_list, cfg_label):
        t = np.atleast_2d(np.array(t_list, dtype=float))
        p = np.atleast_2d(np.array(p_list, dtype=float))
        if t.shape[0] and t.ndim == 2 and t.shape[0] > 2:
            res = stacked_residual_bootstrap(p if p.ndim == 2 else p[:, None],
                                             t if t.ndim == 2 else t[:, None])
            save_sigma_model(out_dir / f"sigma_model_{name}_snap{snap:03d}",
                             res, {**prov_base, "operator_config": cfg_label})
            summary[f"{name}_frac_bias"] = np.asarray(res["frac_bias"]).tolist()
            summary[f"{name}_n"] = int(res["n_halos"])

    # kSZ tau_CAP is (N, n_radii); f_gas/Y are scalars per halo -> (N,1)
    _persist("ksz_tauCAP", rec["ksz_t"], rec["ksz_p"], f"Qu2026 radii, beam 1.6', tau_CAP, cutout {CUTOUT_PIX}px")
    _persist("fgas_cyl", [[v] for v in rec["fgas_t"]], [[v] for v in rec["fgas_p"]], "f_gas(<R500c cyl)")
    _persist("Ycyl", [[v] for v in rec["y_t"]], [[v] for v in rec["y_p"]], "Y(<R500c cyl) proper Mpc^2")

    # Y-M relations
    m500 = np.array(rec["m500"])
    if len(m500) >= 3:
        summary["ym_truth"] = _ym_dict(fit_ym_relation(m500, np.array(rec["y_t"])))
        summary["ym_painted"] = _ym_dict(fit_ym_relation(m500, np.array(rec["y_p"])))

    # x_e residual: stacked truth tau_true/tau_assumed ratio in the R500 aperture
    if rec["xe_true"]:
        xt, xa = np.array(rec["xe_true"]), np.array(rec["xe_assumed"])
        summary["xe_residual_mean"] = float(np.mean(xt / np.where(xa > 0, xa, np.nan)))
        summary["xe_residual_std"] = float(np.nanstd(xt / np.where(xa > 0, xa, np.nan)))

    (out_dir / f"summary_snap{snap:03d}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[snap {snap:03d}] wrote {out_dir}/summary_snap{snap:03d}.json")
    return summary


def _calibrate_cyltosph(manifest, rec, depth, halo_subset) -> dict:
    """Mean+scatter of M_gas(3D sphere<R500c) / M_gas(cylinder<R500c) on a subset."""
    if not rec["sph_center"]:
        return {"factor": None, "note": "no matched halos"}
    centers = np.array(rec["sph_center"], dtype=float)
    r500 = np.array(rec["sph_r500"], dtype=float)
    cyl = np.array(rec["cyl_gas"], dtype=float)
    n = min(halo_subset, len(centers))
    sel = np.linspace(0, len(centers) - 1, n).astype(int)  # spread across the mass range
    sph = tp.spherical_gas_mass_from_particles(
        manifest.hydro_snapdir(), manifest.snapshot_index,
        centers[sel], r500[sel], manifest.box_size,
    )
    good = (cyl[sel] > 0) & (sph > 0)
    ratio = sph[good] / cyl[sel][good]
    corr = CylToSphCorrection(
        factor=float(np.mean(ratio)), depth_hmpc=float(depth), scatter=float(np.std(ratio)),
        provenance="run_wp2_truth_validation.py spherical_gas_mass_from_particles",
    )
    # round-trip through the depth-lock guard so the persisted factor is one the
    # A3/A5 f_gas operator will actually accept at this projection depth.
    _ = corr.apply(np.array([1.0]), map_depth_hmpc=float(depth))
    return {"factor": corr.factor, "scatter": corr.scatter, "depth_hmpc": corr.depth_hmpc,
            "n_halos": int(good.sum()), "provenance": corr.provenance}


def process_multisample(snap: int, multi_root: Path, out_dir: Path) -> dict:
    """Task 6: >=8 fiducial draws at one snapshot -> per-operator sample count.

    Expects ``multi_root/snap_<NNN>_s{k}/composite_slab*.npz`` for k=0..K-1.
    Builds the gas composite per draw, stacks the kSZ tau_CAP profile over a
    fixed matched-halo subset per draw, and reports
    ``multi_sample_convergence`` — the evidence for how many generative samples
    per halo each operator needs (SHARED_CONTEXT caveat 4).
    """
    manifest = ft.load_stage1_manifest(Path(CONDITIONS) / f"snap_{snap:03d}")
    box, npix, ppm = manifest.box_size, manifest.npix, manifest.npix / manifest.box_size
    geom = PatchGeometry(pixel_mpch=box / npix, z=manifest.redshift,
                         cosmology=FlatLCDM(omega_m=manifest.raw["Omega_m"]))
    ksz_cfg = KSZOperatorConfig(radii_arcmin=load_ksz_qu2026_lrg_fiducial().bins,
                                beam_fwhm_arcmin=1.6, z_eff=manifest.redshift, v_rms_over_c=None)
    draws = sorted(Path(multi_root).glob(f"snap_{snap:03d}_s*"))
    if len(draws) < 8:
        print(f"[multisample] only {len(draws)} draws found; task 6 needs >=8 — skipping")
        return {"n_draws": len(draws), "status": "insufficient draws"}

    per_sample = []  # (n_draws, n_halos, n_radii)
    for dpath in draws:
        prof = []
        for si in range(manifest.n_slabs):
            f = dpath / f"composite_slab{si:02d}.npz"
            if not f.exists():
                continue
            pf = np.load(f)
            gas_comp = _mass_matched_composite(pf, box, npix, _gas_channel)
            m200 = pf["halo_masses"]; centers = pf["halo_centers"]
            for k in range(len(centers)):
                if not (KSZ_MASS_BIN[0] <= m200[k] < KSZ_MASS_BIN[1]):
                    continue
                cx = int(centers[k][0] * ppm) % npix
                cy = int(centers[k][1] * ppm) % npix
                prof.append(ksz_cap_profile(extract_periodic_cutout(gas_comp, cx, cy, CUTOUT_PIX), geom, ksz_cfg))
        per_sample.append(prof)
    n_common = min(len(p) for p in per_sample)
    arr = np.array([p[:n_common] for p in per_sample])  # (n_draws, n_common, n_radii)
    conv = multi_sample_convergence(arr)
    out = {"n_draws": len(draws), "n_halos": n_common,
           "frac_dev_vs_k": np.asarray(conv["frac_dev_vs_k"]).tolist()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"multisample_snap{snap:03d}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[multisample] snap {snap:03d}: frac_dev vs k = {out['frac_dev_vs_k']}")
    return out


def _ym_dict(r):
    return {"alpha": r.alpha, "beta": r.beta, "pivot_log10m": r.pivot_log10m,
            "scatter_dex": r.scatter_dex, "n_halos": r.n_halos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", type=int, default=None, help="single snapshot; else use --array-index")
    ap.add_argument("--array-index", type=int, default=None, help="0-3 -> [96,71,67,63]")
    ap.add_argument("--painted-root", type=Path,
                    default=Path("/mnt/home/mlee1/ceph/paper3/A/wp2_fiducial/run_0000"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/mnt/home/mlee1/ceph/paper3/A/wp2_validation"))
    ap.add_argument("--halo-subset", type=int, default=200, help="halos for the CylToSph particle pass")
    ap.add_argument("--multisample-root", type=Path, default=None,
                    help="root with snap_<NNN>_s{k} draws; triggers task 6 at snap 063")
    args = ap.parse_args()

    order = [96, 71, 67, 63]
    snap = args.snap if args.snap is not None else order[args.array_index]
    process_snapshot(snap, args.painted_root, args.out_dir, args.halo_subset)
    if snap == 63 and args.multisample_root is not None:
        process_multisample(snap, args.multisample_root, args.out_dir)


if __name__ == "__main__":
    main()
