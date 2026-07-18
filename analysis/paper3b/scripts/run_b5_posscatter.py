"""WP-B5 interpretation-ladder item 3: reconstruction positional scatter beyond quantization.

Question: could shape-noise / reconstruction-driven scatter of the DES Y3
Wiener peak POSITIONS (beyond the nside-1024 pixel quantization already
exhausted by the ``--peak-grid-1024`` rung) dilute the stacked
<Y_CAP> enough to explain the B5 ~3-6x y-deficit?

Job-free CPU re-analysis of frozen products only — no reopening of the
signed freeze, no Sigma_sys change:

1. **Dilution model D(sigma, r)** — an isotropic Gaussian positional-scatter
   kernel applied to (a) the frozen data stacked thumbnails (nonparametric)
   and (b) a nonneg-Gaussian-mixture profile fitted to the FROZEN tf model
   grid's fiducial-anchor CAP curve (parametric, rendered on a padded grid so
   CAP(8') is edge-safe). Linearity of the stack makes blurring the mean
   thumbnail exactly equivalent to scattering the per-peak positions.
2. **Required scatter sigma***: solve D(sigma*, r) = observed data/model
   ratio per fit nu bin and radius; a single sigma must reproduce the whole
   D(r) curve to be an explanation.
3. **Bounds on the actual scatter**: (a) GLIMPSE->Wiener matched-peak offset
   distributions (same shear data, independent reconstruction priors);
   (b) stack-compactness — the implied broadening sqrt(w_obs^2 - w_model^2)
   of the data thumbnails vs the model profile width (the beam is common to
   data and mocks and cancels); (c) the ~1' pixel-quantization anchor whose
   nullity the pg1024 rung already established.

Outputs -> /mnt/home/mlee1/ceph/paper3/B/wp5_inference/posscatter/
  posscatter_summary.json + figures. Data loads are hash-guarded via
  ``stack.frozen`` (blinding discipline).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls
from scipy.spatial import cKDTree

from paper3b.stack import frozen
from paper3b.stack.cap import cap_filter

B_ROOT = Path("/mnt/home/mlee1/ceph/paper3/B")
OUT = B_ROOT / "wp5_inference" / "posscatter"
FID_UNIT = B_ROOT / "wp4_mocks" / "grid_tfwiener" / "bind_run_0000.npz"
PEAKS_W = B_ROOT / "wp1_maps" / "peaks_wiener_sm2am.npz"
PEAKS_G = B_ROOT / "wp1_maps" / "peaks_glimpse_sm2am.npz"

RES_ARCMIN = 0.5          # frozen thumbnail resolution (B1 anchor convention)
FIT_BINS = [1, 2, 3, 4]   # nu bins [1,2),[2,3),[3,4),[4,12] (frozen.B5_BINS)
RAD_IDX_4AM = 2           # 4' fiducial radius index in cap_radii [2,3,4,6,8]
BASIS_S_ARCMIN = np.array([0.75, 1.5, 2.5, 4.0, 6.0, 9.0])  # mixture widths
PAD_N = 121               # padded render grid (+-30') for edge-safe CAP(8')
SIGMA_GRID = np.arange(0.0, 12.01, 0.05)  # arcmin
PIX_QUANT_SIGMA = 3.44 / np.sqrt(6.0)     # ~1.4': rms of a uniform offset in
                                          # a 3.44' pixel, per-axis-equivalent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_gaussian_mixture(amps, widths_arcmin, n, res_arcmin):
    y, x = np.indices((n, n))
    r = np.hypot(y - (n - 1) / 2.0, x - (n - 1) / 2.0) * res_arcmin
    img = np.zeros((n, n))
    for a, s in zip(amps, widths_arcmin):
        img += a * np.exp(-0.5 * (r / s) ** 2)
    return img


def cap_multi(img, radii, res_arcmin):
    return np.array([cap_filter(img, r, res_arcmin) for r in radii])


def dilution_curve(img, radii, res_arcmin, sigma_grid):
    """D[s, r] = CAP(blur(img, s), r) / CAP(img, r)."""
    base = cap_multi(img, radii, res_arcmin)
    out = np.empty((len(sigma_grid), len(radii)))
    for i, s in enumerate(sigma_grid):
        b = gaussian_filter(img, s / res_arcmin, mode="nearest") if s > 0 else img
        out[i] = cap_multi(b, radii, res_arcmin) / base
    return out


def required_sigma(dcurve, sigma_grid, target):
    """Smallest sigma with D(sigma) <= target (D is monotone-decreasing)."""
    below = np.where(dcurve <= target)[0]
    if len(below) == 0:
        return np.inf
    i = below[0]
    if i == 0:
        return 0.0
    s0, s1 = sigma_grid[i - 1], sigma_grid[i]
    d0, d1 = dcurve[i - 1], dcurve[i]
    return float(s0 + (d0 - target) / (d0 - d1) * (s1 - s0))


def gaussian_core_width(img, res_arcmin, r_fit_arcmin=4.0):
    """Effective Gaussian width of the central peak, from the profile's
    curvature: fit log y(r) = c - r^2/(2 w^2) over r <= r_fit."""
    n = img.shape[0]
    y, x = np.indices(img.shape)
    r = np.hypot(y - (n - 1) / 2.0, x - (n - 1) / 2.0) * res_arcmin
    m = (r <= r_fit_arcmin) & (img > 0)
    if m.sum() < 8:
        return np.nan
    A = np.stack([np.ones(m.sum()), -0.5 * r[m] ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.log(img[m]), rcond=None)
    return float(1.0 / np.sqrt(coef[1])) if coef[1] > 0 else np.inf


def radec_to_xyz(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                     np.sin(dec)], axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # -- frozen data (hash-guarded) + frozen-vintage model anchor ------------
    meas = frozen.load_frozen("wiener")            # vector sanity anchor
    stack_path = frozen._verify("wp2_measurement/stack_wiener_sm2am_fid.npz")
    st = np.load(stack_path)
    radii = st["cap_radii_arcmin"]                 # [2,3,4,6,8]
    thumbs = st["stacked_thumbs"]                  # (5, 61, 61) mean y images
    y_data = st["y_mean"]                          # (5 nu, 5 rad) CAP table
    fid = np.load(FID_UNIT)
    y_model = fid["y_mean"]                        # (5 nu, 5 rad)
    assert np.allclose(y_data[frozen.B5_BINS, RAD_IDX_4AM], meas.y)
    ratios = y_data / y_model                      # observed D(r) per nu bin

    summary = {
        "inputs": {
            "stack": {"path": str(stack_path), "sha256": sha256(stack_path)},
            "fiducial_unit": {"path": str(FID_UNIT), "sha256": sha256(FID_UNIT)},
            "peaks_wiener": {"path": str(PEAKS_W), "sha256": sha256(PEAKS_W)},
            "peaks_glimpse": {"path": str(PEAKS_G), "sha256": sha256(PEAKS_G)},
        },
        "conventions": {
            "fit_bins_nu_edges": st["nu_edges"][1:].tolist(),
            "cap_radii_arcmin": radii.tolist(),
            "sigma_grid_max_arcmin": float(SIGMA_GRID[-1]),
            "pixel_quantization_sigma_arcmin": PIX_QUANT_SIGMA,
        },
        "bins": {},
    }

    # -- per-bin dilution model + required sigma -----------------------------
    print(f"{'bin':>8} {'route':>10} " + " ".join(f"sig*({r:g}')" for r in radii)
          + "   D(1.4')@4'  w_obs  w_mod  sig_impl")
    for b in FIT_BINS:
        obs = ratios[b]
        # (a) nonparametric: frozen data thumbnail
        d_data = dilution_curve(thumbs[b].astype(np.float64), radii,
                                RES_ARCMIN, SIGMA_GRID)
        # (b) parametric: nonneg Gaussian mixture matched to the MODEL CAP
        #     curve, rendered on a padded grid (edge-safe out to 8')
        design = np.stack([
            cap_multi(render_gaussian_mixture([1.0], [s], PAD_N, RES_ARCMIN),
                      radii, RES_ARCMIN) for s in BASIS_S_ARCMIN], axis=1)
        amps, resid = nnls(design, y_model[b])
        model_img = render_gaussian_mixture(amps, BASIS_S_ARCMIN, PAD_N,
                                            RES_ARCMIN)
        model_fit_frac = float(resid / np.linalg.norm(y_model[b]))
        d_model = dilution_curve(model_img, radii, RES_ARCMIN, SIGMA_GRID)

        row = {}
        for route, dcur in (("data_thumb", d_data), ("model_capfit", d_model)):
            sig_req = [required_sigma(dcur[:, j], SIGMA_GRID, obs[j])
                       for j in range(len(radii))]
            # single-sigma joint solution at 4' + its predicted full D(r)
            s4 = sig_req[RAD_IDX_4AM]
            pred = (dcur[np.argmin(np.abs(SIGMA_GRID - s4))]
                    if np.isfinite(s4) else np.full(len(radii), np.nan))
            row[route] = {
                "required_sigma_arcmin_per_radius": sig_req,
                "sigma_at_4am": s4,
                "D_of_sigma4_per_radius": pred.tolist(),
                "observed_ratio_per_radius": obs.tolist(),
                "D_at_pixel_quantization_4am":
                    float(dcur[np.argmin(np.abs(SIGMA_GRID - PIX_QUANT_SIGMA)),
                               RAD_IDX_4AM]),
            }
            print(f"{b:>8} {route:>10} "
                  + " ".join(f"{s:8.2f}" for s in sig_req)
                  + f"   {row[route]['D_at_pixel_quantization_4am']:.3f}")

        # stack-compactness: implied broadening of data vs model profile.
        # NOTE the mocks already include DES shape noise + 2' smoothing +
        # beam (run_b4_grid), so this excess bounds the EXTRA,
        # reconstruction-specific scatter only — and it is an over-credit,
        # since profile physics (ejection) also broadens the data stack.
        w_obs = gaussian_core_width(thumbs[b].astype(np.float64), RES_ARCMIN)
        w_mod = gaussian_core_width(model_img, RES_ARCMIN)
        sig_impl = (np.sqrt(max(w_obs ** 2 - w_mod ** 2, 0.0))
                    if np.isfinite(w_obs) and np.isfinite(w_mod) else np.nan)
        row["core_width_obs_arcmin"] = w_obs
        row["core_width_model_arcmin"] = w_mod
        row["implied_broadening_arcmin"] = sig_impl
        row["model_capfit_resid_frac"] = model_fit_frac
        if np.isfinite(sig_impl):
            d_credit = dilution_curve(model_img, radii, RES_ARCMIN,
                                      np.array([sig_impl]))[0]
            row["D_at_implied_broadening_per_radius"] = d_credit.tolist()
            row["residual_ratio_after_full_credit_per_radius"] = \
                (obs / d_credit).tolist()
        print(f"{'':>19} " + " " * 9 * len(radii)
              + f"          {w_obs:6.2f} {w_mod:6.2f} {sig_impl:8.2f}")
        summary["bins"][f"nu_{st['nu_edges'][b]:g}_{st['nu_edges'][b+1]:g}"] = row

    # -- empirical: GLIMPSE -> Wiener matched-peak offsets -------------------
    pw, pg = np.load(PEAKS_W), np.load(PEAKS_G)
    xyz_w = radec_to_xyz(pw["ra_deg"], pw["dec_deg"])
    xyz_g = radec_to_xyz(pg["ra_deg"], pg["dec_deg"])
    tree = cKDTree(xyz_w)
    dist, _ = tree.query(xyz_g, k=1)
    off_arcmin = np.degrees(2 * np.arcsin(dist / 2)) * 60.0
    edges = st["nu_edges"]
    offs = {}
    for b in FIT_BINS:
        m = (pg["nu"] >= edges[b]) & (pg["nu"] < edges[b + 1])
        o = off_arcmin[m & (off_arcmin < 15.0)]
        offs[f"nu_{edges[b]:g}_{edges[b+1]:g}"] = {
            "n_matched": int(o.size),
            "median_arcmin": float(np.median(o)) if o.size else np.nan,
            "p68_arcmin": float(np.percentile(o, 68)) if o.size else np.nan,
            "p95_arcmin": float(np.percentile(o, 95)) if o.size else np.nan,
        }
        print(f"G->W offsets nu[{edges[b]:g},{edges[b+1]:g}): "
              f"n={o.size}, median={offs[list(offs)[-1]]['median_arcmin']:.2f}', "
              f"p68={offs[list(offs)[-1]]['p68_arcmin']:.2f}'")
    summary["glimpse_to_wiener_offsets"] = offs

    with open(OUT / "posscatter_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"\nwrote {OUT/'posscatter_summary.json'}")

    # -- figure --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, b in zip(axes, FIT_BINS):
        key = f"nu_{st['nu_edges'][b]:g}_{st['nu_edges'][b+1]:g}"
        design_row = summary["bins"][key]
        d_model = dilution_curve(
            render_gaussian_mixture(
                nnls(np.stack([cap_multi(render_gaussian_mixture([1.0], [s],
                     PAD_N, RES_ARCMIN), radii, RES_ARCMIN)
                     for s in BASIS_S_ARCMIN], axis=1), y_model[b])[0],
                BASIS_S_ARCMIN, PAD_N, RES_ARCMIN),
            radii, RES_ARCMIN, SIGMA_GRID)
        for j, r in enumerate(radii):
            ax.plot(SIGMA_GRID, d_model[:, j], label=f"CAP({r:g}')")
            ax.axhline(ratios[b][j], ls=":", lw=0.8, color=f"C{j}")
        ax.axvline(PIX_QUANT_SIGMA, color="k", ls="--", lw=0.8,
                   label="pixel quant.")
        med = offs[key]["median_arcmin"]
        if np.isfinite(med):
            ax.axvline(med, color="gray", ls="-.", lw=0.8, label="G-W median")
        ax.set_title(f"$\\nu\\in[{st['nu_edges'][b]:g},{st['nu_edges'][b+1]:g})$")
        ax.set_xlabel("positional scatter $\\sigma$ [arcmin]")
        ax.set_xlim(0, SIGMA_GRID[-1])
    axes[0].set_ylabel("dilution $D(\\sigma, r)$  (model profile)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("B5 ladder item 3: required vs available positional scatter "
                 "(dotted = observed data/model ratio per radius)")
    fig.tight_layout()
    fig.savefig(OUT / "posscatter_dilution.png", dpi=150)
    print(f"wrote {OUT/'posscatter_dilution.png'}")


if __name__ == "__main__":
    main()
