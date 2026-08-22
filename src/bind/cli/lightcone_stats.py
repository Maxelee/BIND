"""``bind-lightcone-stats`` (stage 3): summary statistics from a run's maps.

Reads ``kappa_maps.npz`` (+ optional ``y_maps.npz``) written by
``bind-lightcone-maps`` and writes the emulator-input statistics into the same
run directory (see :mod:`bind.inference.stats`):

* ``Cl_kappa.npz``          — WL auto/cross C_ell (+ suppression if kappa_dmo present)
* ``Cl_kappa_y.npz``        — WL x tSZ cross + tSZ auto (needs y_maps.npz)
* ``Cl_tau.npz``            — WL x tau cross + tau auto (+ y x tau) (needs tau_maps.npz)
* ``dm_stats.npz``          — dispersion-measure PDF + sigma_DM/F/moments (tau_maps.npz)
* ``peak_counts.npz``       — peak counts vs nu
* ``nongaussian_stats.npz`` — PDF, moments per scale, Minkowski functionals
* ``wst.npz``               — wavelet scattering coefficients (with --wst; kymatio)
* ``halo_scaling.npz``      — per-halo Y_500c/f_gas/f_star/T_mw (needs --snap_root
                              composites with saved patches)
* ``scaling_relations.npz`` — binned Y-M/f_gas-M/f_star-M/T-M (from halo_scaling)

    bind-lightcone-stats --run_dir /ceph/bind_science/runs/fiducial/run_0000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bind.inference import stats as S


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Dir holding kappa_maps.npz (and y_maps.npz); outputs go here")
    p.add_argument("--snap_root", type=Path, default=None,
                   help="Dir with snap_<NNN>/composite_slab*.npz for halo_scaling "
                        "(needs saved patches). Defaults to --run_dir.")
    p.add_argument("--snapshots", type=int, nargs="+", default=None)
    p.add_argument("--smoothing_arcmin", type=float, nargs="+", default=[2.0],
                   help="Smoothing scale(s) [arcmin] for peak_counts; the first "
                        "is used for peak_cross (R(nu) / profiles).")
    p.add_argument("--shape_noise_ngal", type=float, default=None,
                   help="Source density [arcmin^-2] for kappa shape noise "
                        "(default off). E.g. 10 (LSST-Y1), 27 (LSST-Y10).")
    p.add_argument("--sigma_e", type=float, default=0.26,
                   help="Per-component ellipticity dispersion.")
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--nu_norm", choices=["map", "noise", "fixed"], default=None,
                   help="nu normalisation for peaks; default 'noise' when shape "
                        "noise is on, 'fixed' when --nu_sigma0_from is given, "
                        "else 'map'.")
    p.add_argument("--nu_sigma0_from", type=Path, default=None,
                   help="Run dir whose kappa_maps.npz defines ONE fixed sigma "
                        "per source bin (nu = kappa_sm/sigma_fid for every run; "
                        "per-map sigma would absorb the sigma_kappa response). "
                        "Outputs get suffix '_nufid'.")
    p.add_argument("--nu_grid", choices=["canon", "legacy"], default="canon",
                   help="'canon' (default): PDF, peaks, minima and V0/V1/V2 all "
                        "report length-22 arrays on ONE axis, nu = -2.5..8 step "
                        "0.5 (stats.NU_CANON) — histograms binned on edges "
                        "centred on those values, MFs evaluated at them. "
                        "'legacy' keeps the old per-statistic grids (peaks "
                        "-5..12/68 bins, MF -3..4/29, PDF in kappa units).")
    p.add_argument("--out_suffix", default=None,
                   help="Suffix for peak_counts/peak_cross output names; "
                        "defaults to '' (noiseless) or '_ngal<N>' with noise.")
    p.add_argument("--no_halo_scaling", action="store_true")
    p.add_argument("--peaks_only", action="store_true",
                   help="Only recompute peak_counts/peak_cross (skip Cl, "
                        "nongaussian, halo_scaling) — for iterating on peak "
                        "settings over existing maps.")
    p.add_argument("--wst", action="store_true",
                   help="Also compute wavelet scattering coefficients (kymatio, "
                        "GPU if available) -> wst.npz.")
    p.add_argument("--wst_J", type=int, default=4)
    p.add_argument("--wst_L", type=int, default=4)
    p.add_argument("--wst_n_real", type=int, default=10,
                   help="Realizations used for WST (heaviest stat; default 10).")
    p.add_argument("--no_dm", action="store_true",
                   help="Skip dm_stats.npz even when tau_maps.npz is present.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rd = args.run_dir
    nu_norm = args.nu_norm or ("fixed" if args.nu_sigma0_from
                               else ("noise" if args.shape_noise_ngal else "map"))
    suffix = args.out_suffix
    if suffix is None:
        suffix = f"_ngal{args.shape_noise_ngal:g}" if args.shape_noise_ngal else ""
        if nu_norm == "fixed":
            suffix += "_nufid"
    nu_centers = S.NU_CANON if args.nu_grid == "canon" else None
    nu_bins = S.NU_EDGES_CANON if nu_centers is not None else None
    if nu_centers is not None and nu_norm != "fixed":
        print("[stats] WARNING: --nu_grid canon without --nu_sigma0_from — nu is "
              "normalised per map, so the canonical grid is NOT a common axis "
              "across runs (the sigma_kappa response gets absorbed).")
    sig0 = sig0_unsmoothed = sig0_ng = None
    nu_fiducial = "per-map"
    if nu_norm == "fixed":
        ref_path = Path(args.nu_sigma0_from)
        if ref_path.is_file():
            # precomputed table from bind.cli.nu_sigma0 (preferred: identical
            # sigma for every run, and no 21 GB re-read per job)
            t = np.load(ref_path)
            tab_scales = list(np.asarray(t["scales_arcmin"], float))
            idx = []
            for sc in args.smoothing_arcmin:
                near = [i for i, s in enumerate(tab_scales) if abs(s - sc) < 1e-9]
                if not near:
                    raise SystemExit(f"[stats] {ref_path} has no sigma0 for scale {sc}' "
                                     f"(has {tab_scales}); rebuild it with that scale.")
                idx.append(near[0])
            sig0 = np.asarray(t["sigma_smoothed"])[idx]
            sig0_unsmoothed = np.asarray(t["sigma_unsmoothed"])
            # the Minkowski functionals smooth at nongaussian's FIRST scale,
            # which is not the peak scale — look their sigma up separately or
            # V0/V1/V2 would be normalised by the wrong-scale sigma
            ng_near = [i for i, sc in enumerate(tab_scales)
                       if abs(sc - float(S.NG_SCALES_DEFAULT[0])) < 1e-9]
            if not ng_near:
                raise SystemExit(f"[stats] {ref_path} lacks sigma0 at the Minkowski "
                                 f"scale {S.NG_SCALES_DEFAULT[0]}'; rebuild the table.")
            sig0_ng = np.asarray(t["sigma_smoothed"])[ng_near]
            nu_fiducial = str(t["source_run"])
            print(f"[stats] fixed nu: sigma0 table {ref_path} "
                  f"(fiducial {t['source_run']}, {int(t['n_real'])} reals)")
        else:
            # legacy: derive from a run dir, streamed one realization at a time
            import zipfile

            import numpy.lib.format as fmt
            src = ref_path / "kappa_maps.npz"
            with np.load(src) as f:
                rfov = float(f["fov_deg"])
            with zipfile.ZipFile(src) as z, z.open("kappa.npy") as f:
                v = fmt.read_magic(f)
                shp, _, dt = (fmt.read_array_header_1_0(f) if v == (1, 0)
                              else fmt.read_array_header_2_0(f))
                nref = min(shp[0], 20)          # sigma is stable; cap for speed
                per = int(np.prod(shp[1:])) * dt.itemsize
                acc = np.zeros((len(args.smoothing_arcmin), shp[1]))
                accu = np.zeros(shp[1])
                for r in range(nref):
                    cube = np.frombuffer(f.read(per), dtype=dt).reshape(shp[1:])
                    for i in range(shp[1]):
                        m = cube[i].astype(np.float64)
                        accu[i] += m.std()
                        for k, sc in enumerate(args.smoothing_arcmin):
                            acc[k, i] += S._gaussian_smooth(m, sc, rfov).std()
            sig0, sig0_unsmoothed = acc / nref, accu / nref
            sig0_ng = None          # legacy path: MFs fall back to per-map sigma
            nu_fiducial = str(ref_path)
            print(f"[stats] fixed nu: sigma0 from {ref_path} ({nref} reals, "
                  f"scales {args.smoothing_arcmin}) = {np.array2string(sig0, precision=4)}")
    # provenance stamped into every nu-binned product, so a later sweep can tell
    # which convention a file was written under (and recompute if it changed)
    nu_prov = dict(nu_grid=str(args.nu_grid), nu_norm_used=str(nu_norm),
                   nu_fiducial=str(nu_fiducial),
                   nu_sigma0_table=(np.asarray(sig0) if sig0 is not None
                                    else np.zeros(0)),
                   nu_sigma0_unsmoothed_table=(np.asarray(sig0_unsmoothed)
                                               if sig0_unsmoothed is not None
                                               else np.zeros(0)))
    noise_kw = dict(shape_noise_ngal=args.shape_noise_ngal, sigma_e=args.sigma_e,
                    noise_seed=args.noise_seed, nu_norm=nu_norm)
    km = np.load(rd / "kappa_maps.npz")
    kappa = km["kappa"]                                  # (n_real, n_src, npix, npix)
    fov = float(km["fov_deg"])
    kdmo = km["kappa_dmo"] if "kappa_dmo" in km.files else None

    if not args.peaks_only:
        ck = S.cl_kappa(kappa, fov_deg=fov, kappa_dmo=kdmo)
        np.savez(rd / "Cl_kappa.npz", **ck)
        print(f"[stats] Cl_kappa.npz  cl{ck['cl'].shape}"
              + (" +suppression" if "suppression" in ck else ""))

    if args.wst:
        w = S.wst(kappa, J=args.wst_J, L=args.wst_L, n_real=args.wst_n_real)
        np.savez(rd / "wst.npz", **w)
        print(f"[stats] wst.npz  wst{w['wst'].shape} (J={args.wst_J},L={args.wst_L},"
              f" n_real={w['n_real_used']})")

    ypath = rd / "y_maps.npz"
    if ypath.exists():
        y = np.load(ypath)["y"]
        if not args.peaks_only:
            # tomographic y -> use the total (last source plane) for the cross power
            y_total = y[:, -1] if y.ndim == kappa.ndim else y
            cky = S.cl_kappa_y(kappa, y_total, fov_deg=fov)
            np.savez(rd / "Cl_kappa_y.npz", **cky)
            print(f"[stats] Cl_kappa_y.npz  cl_ky{cky['cl_ky'].shape}")
        # tSZ stacked at WL peaks -> R(nu) emulator target (needs per-source-bin y)
        if y.ndim == kappa.ndim:           # (n_real, n_src, npix, npix)
            pc = S.peak_cross_stats(kappa, y, fov_deg=fov,
                                    smoothing_arcmin=args.smoothing_arcmin[0],
                                    nu_sigma0=sig0[0] if sig0 is not None else None,
                                    **noise_kw)
            np.savez(rd / f"peak_cross{suffix}.npz", **pc)
            print(f"[stats] peak_cross{suffix}.npz  R{pc['R'].shape} "
                  f"+ profile{pc['profile'].shape}")
        else:
            print("[stats] y_maps not per-source-bin — skipping peak_cross (R(nu))")
    else:
        print("[stats] no y_maps.npz — skipping Cl_kappa_y / peak_cross")

    taupath = rd / "tau_maps.npz"
    if taupath.exists() and not args.peaks_only:
        tau = np.load(taupath)["tau"]
        tau_total = tau[:, -1] if tau.ndim == kappa.ndim else tau
        y_total_for_tau = None
        if ypath.exists():
            yt = np.load(ypath)["y"]
            y_total_for_tau = yt[:, -1] if yt.ndim == kappa.ndim else yt
        ckt = S.cl_kappa_tau(kappa, tau_total, y_maps=y_total_for_tau, fov_deg=fov)
        np.savez(rd / "Cl_tau.npz", **ckt)
        print(f"[stats] Cl_tau.npz  cl_kt{ckt['cl_kt'].shape}"
              + (" +cl_yt" if "cl_yt" in ckt else "") + " +cl_tt")
        if not args.no_dm and tau.ndim == kappa.ndim:
            dm = S.dm_stats(tau, fov_deg=fov)
            np.savez(rd / "dm_stats.npz", **dm)
            print(f"[stats] dm_stats.npz  pdf{dm['dm_pdf'].shape} F{dm['F'].shape}")
    elif not taupath.exists():
        print("[stats] no tau_maps.npz — skipping Cl_tau / dm_stats")

    sm = (args.smoothing_arcmin[0] if len(args.smoothing_arcmin) == 1
          else args.smoothing_arcmin)               # scalar -> legacy shapes
    pk = S.peak_counts(kappa, fov_deg=fov, smoothing_arcmin=sm,
                       nu_bins=nu_bins,
                       nu_sigma0=(sig0 if len(args.smoothing_arcmin) > 1
                                  else (sig0[0] if sig0 is not None else None)),
                       **noise_kw)
    np.savez(rd / f"peak_counts{suffix}.npz", **pk, **nu_prov)
    print(f"[stats] peak_counts{suffix}.npz  peaks{pk['peak_counts'].shape} + minima")

    if not args.peaks_only:
        ng = S.nongaussian_stats(kappa, fov_deg=fov, nu_centers=nu_centers,
                                 nu_sigma0=sig0_ng, nu_sigma0_unsmoothed=sig0_unsmoothed)
        np.savez(rd / "nongaussian_stats.npz", **ng, **nu_prov)
        print(f"[stats] nongaussian_stats.npz  pdf{ng['pdf'].shape}"
              + ("  (nu units, canonical axis)" if nu_centers is not None else ""))

    if not (args.no_halo_scaling or args.peaks_only):
        snap_root = args.snap_root or rd
        snaps = args.snapshots
        if snaps is None:
            snaps = sorted(int(p.name.split("_")[1]) for p in snap_root.glob("snap_*")
                           if (p / "composite_slab00.npz").exists())
        paths = [snap_root / f"snap_{s:03d}" / f"composite_slab{sl:02d}.npz"
                 for s in snaps for sl in range(8)
                 if (snap_root / f"snap_{s:03d}" / f"composite_slab{sl:02d}.npz").exists()]
        sample = np.load(paths[0]) if paths else None
        if paths and "generated_patches" in sample.files:
            hs = S.halo_scaling(paths)
            np.savez(rd / "halo_scaling.npz", **hs)
            print(f"[stats] halo_scaling.npz  {len(hs['halo_mass'])} halos")
            sr = S.scaling_relations(hs["halo_mass"], Y=hs.get("Y_500c"),
                                     f_gas=hs.get("f_gas_500c"),
                                     f_star=hs.get("f_star_500c"), T=hs.get("T_mw_500c"))
            np.savez(rd / "scaling_relations.npz", **sr)
            print(f"[stats] scaling_relations.npz  {len(sr['log_mass_bins'])} mass bins")
        else:
            print("[stats] no composites with saved patches — skipping halo_scaling")


if __name__ == "__main__":
    main()
