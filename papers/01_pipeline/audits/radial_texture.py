"""Investigation 3: mechanism of the diffuse-field y excess (+20-30% Cl_yy at
5e3<ell<1.5e4, outside bright halo cores; cores slightly DEFICIENT).

Uses the SAME row-index matching as Investigation 2
(bind_lightcone_tng/snap_096 composite_slabNN.npz  vs
 bind_science/runs/truth/run_0000/snap_096 composite_slabNN.npz,
 halo_masses/halo_r200/halo_centers verified identical -> same halo, same row).

thermo_patches[:,0] = compton_y  (bind.data.THERMO_KEYS = compton_y,T,entropy,P_e)
Patch pixel scale: MULTISCALE_MPC[0]/patch_pix = 6.25/128 Mpc/h per pixel
(src/bind/inference/pipeline.py: extract_multiscale, MULTISCALE_MPC = (6.25,...)).

Method
------
(a) radial decomposition in x=R/r200c annuli {<0.5,0.5-1,1-2,2-4}: mean-y profile
    AND small-scale fluctuation power = variance of (y - azimuthal_mean_profile(x))
    computed with a FINE x-binning (each map gets its own azimuthal mean removed,
    so this isolates clumpy small-scale texture from any radial-profile-shape bias,
    which is reported separately as the mean-profile ratio).
    Real-space residual variance in an annulus is a valid proxy for the small-scale
    "power" living at scales <~ annulus width (Parseval: total variance = integral
    of the power spectrum; removing the smooth radial trend removes the large-scale
    part, so what's left over is small-scale-texture variance). We avoid masked 2D
    FFTs on thin rings, which suffer strong leakage at these sample sizes.
(b) attribution: each annulus's share of the total EXCESS residual sum-of-squares
    (painted - truth), i.e. how much of the total texture-power excess comes from
    that annulus.
(c) mitigation prototype: Gaussian-smooth (sigma=1,2 px) the PAINTED y map outside
    x>1 only (cores x<=1 untouched), recompute (a)/(b) for the smoothed map, and
    check whether it closes the gap to truth without moving the mean profile.
"""
import numpy as np
import gc
from scipy.ndimage import gaussian_filter

FID_DIR = "/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096"
TRUTH_DIR = "/mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000/snap_096"
OUTDIR = "/tmp/claude-2107/-mnt-home-mlee1-BIND/eaad8798-3159-4ecc-ad3b-687c176db4cc/scratchpad/inv3"

PIX_MPC = 6.25 / 128.0          # Mpc/h per pixel (extract_multiscale, MULTISCALE_MPC[0]=6.25)
PATCH = 128
MASS_LO, MASS_HI = 1e13, 1e14   # group-scale M_fof [Msun/h]
N_TARGET = 200
SEED = 42

COARSE_EDGES = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
N_COARSE = len(COARSE_EDGES) - 1
COARSE_LABELS = ["<0.5", "0.5-1", "1-2", "2-4"]

FINE_EDGES = np.concatenate([np.linspace(0.0, 4.0, 33), [np.inf]])  # 0.125-wide fine bins to x=4, +overflow
N_FINE = len(FINE_EDGES) - 1

# fixed pixel-offset grid (identical for every halo, in Mpc/h)
_idx = np.arange(PATCH) - PATCH // 2
DX, DY = np.meshgrid(_idx * PIX_MPC, _idx * PIX_MPC, indexing="ij")
RR_MPC = np.sqrt(DX**2 + DY**2)  # (128,128) Mpc/h, fixed


def azimuthal_residual(img, fine_bin_idx):
    """Subtract the per-fine-bin azimuthal mean from img; return residual."""
    profile = np.zeros(N_FINE)
    counts = np.bincount(fine_bin_idx.ravel(), minlength=N_FINE)
    sums = np.bincount(fine_bin_idx.ravel(), weights=img.ravel(), minlength=N_FINE)
    nz = counts > 0
    profile[nz] = sums[nz] / counts[nz]
    return img - profile[fine_bin_idx]


def gather_group_halos():
    """Collect (y_fid, y_truth, r200) for N_TARGET random group-scale halos,
    matched by row index across the 4 slabs of snap_096."""
    y_fid_all, y_truth_all, r200_all, mass_all = [], [], [], []
    for si in range(4):
        f = np.load(f"{FID_DIR}/composite_slab{si:02d}.npz", allow_pickle=True)
        t = np.load(f"{TRUTH_DIR}/composite_slab{si:02d}.npz", allow_pickle=True)
        masses = f["halo_masses"]
        assert np.allclose(masses, t["halo_masses"]), f"halo order mismatch slab {si}"
        r200 = f["halo_r200"]
        mask = (masses >= MASS_LO) & (masses < MASS_HI)
        idx = np.where(mask)[0]
        tp_f = f["thermo_patches"][idx, 0]   # (n_sel,128,128) compton_y, painted
        tp_t = t["thermo_patches"][idx, 0]   # (n_sel,128,128) compton_y, truth
        y_fid_all.append(tp_f.copy())
        y_truth_all.append(tp_t.copy())
        r200_all.append(r200[idx].copy())
        mass_all.append(masses[idx].copy())
        del f, t, tp_f, tp_t
        gc.collect()
        print(f"slab {si}: {mask.sum()} group-scale halos")

    y_fid_all = np.concatenate(y_fid_all, axis=0)
    y_truth_all = np.concatenate(y_truth_all, axis=0)
    r200_all = np.concatenate(r200_all)
    mass_all = np.concatenate(mass_all)
    n_avail = len(r200_all)
    print(f"total group-scale candidates: {n_avail}")

    rng = np.random.default_rng(SEED)
    sel = rng.choice(n_avail, size=min(N_TARGET, n_avail), replace=False)
    sel.sort()
    return y_fid_all[sel], y_truth_all[sel], r200_all[sel], mass_all[sel]


def main():
    y_fid, y_truth, r200, mass = gather_group_halos()
    n_halo = len(r200)
    print(f"stacking N={n_halo} group-scale halos, "
          f"M_fof in [{mass.min():.2e},{mass.max():.2e}], "
          f"r200c in [{r200.min():.3f},{r200.max():.3f}] Mpc/h")

    # accumulators: [coarse annulus]
    sum_y_fid = np.zeros(N_COARSE); sum_y_truth = np.zeros(N_COARSE)
    sum_n = np.zeros(N_COARSE)
    sum_r2_fid = np.zeros(N_COARSE); sum_r2_truth = np.zeros(N_COARSE)
    # cross term (pooled) for the texture correlation coefficient, and raw
    # pixel-level (painted-truth) difference variance (phase-mismatch diagnostic)
    sum_rfrt = np.zeros(N_COARSE)
    sum_diff2 = np.zeros(N_COARSE)
    # mitigation variants
    sum_y_mit = {1: np.zeros(N_COARSE), 2: np.zeros(N_COARSE)}
    sum_r2_mit = {1: np.zeros(N_COARSE), 2: np.zeros(N_COARSE)}

    # fine-binned azimuthal profile accumulation (for the profile plot), pooled
    fine_sum_fid = np.zeros(N_FINE); fine_sum_truth = np.zeros(N_FINE); fine_n = np.zeros(N_FINE)

    for i in range(n_halo):
        r2 = r200[i]
        X = RR_MPC / r2  # (128,128) dimensionless radius in units of r200c

        fine_idx = np.clip(np.digitize(X, FINE_EDGES) - 1, 0, N_FINE - 1)
        coarse_idx = np.digitize(X, COARSE_EDGES) - 1  # -1 .. N_COARSE ; values >=N_COARSE or <0 excluded below
        in_range = (coarse_idx >= 0) & (coarse_idx < N_COARSE)

        img_f = y_fid[i]
        img_t = y_truth[i]

        resid_f = azimuthal_residual(img_f, fine_idx)
        resid_t = azimuthal_residual(img_t, fine_idx)

        # mitigated painted map: gaussian-smooth outside x>1, keep core (x<=1) untouched
        for sigma in (1, 2):
            sm = gaussian_filter(img_f, sigma=sigma, mode="nearest")
            img_mit = np.where(X > 1.0, sm, img_f)
            resid_mit = azimuthal_residual(img_mit, fine_idx)
            ci = coarse_idx[in_range]
            np.add.at(sum_y_mit[sigma], ci, img_mit[in_range])
            np.add.at(sum_r2_mit[sigma], ci, resid_mit[in_range] ** 2)

        ci = coarse_idx[in_range]
        np.add.at(sum_n, ci, 1)
        np.add.at(sum_y_fid, ci, img_f[in_range])
        np.add.at(sum_y_truth, ci, img_t[in_range])
        np.add.at(sum_r2_fid, ci, resid_f[in_range] ** 2)
        np.add.at(sum_r2_truth, ci, resid_t[in_range] ** 2)
        np.add.at(sum_rfrt, ci, (resid_f[in_range] * resid_t[in_range]))
        np.add.at(sum_diff2, ci, (img_f[in_range] - img_t[in_range]) ** 2)

        np.add.at(fine_n, fine_idx.ravel(), 1)
        np.add.at(fine_sum_fid, fine_idx.ravel(), img_f.ravel())
        np.add.at(fine_sum_truth, fine_idx.ravel(), img_t.ravel())

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{n_halo}")

    mean_y_fid = sum_y_fid / sum_n
    mean_y_truth = sum_y_truth / sum_n
    var_fid = sum_r2_fid / sum_n
    var_truth = sum_r2_truth / sum_n
    var_mit = {s: sum_r2_mit[s] / sum_n for s in (1, 2)}
    mean_y_mit = {s: sum_y_mit[s] / sum_n for s in (1, 2)}

    fine_profile_fid = fine_sum_fid / np.where(fine_n > 0, fine_n, 1)
    fine_profile_truth = fine_sum_truth / np.where(fine_n > 0, fine_n, 1)
    fine_centers = 0.5 * (FINE_EDGES[:-1] + FINE_EDGES[1:])
    fine_centers[-1] = np.nan  # overflow bin has no meaningful center

    print("\n=== (a) mean-y profile and fluctuation power by annulus (painted vs truth) ===")
    print(f"{'annulus':8s} {'Npix':>10s} {'meanY_fid':>12s} {'meanY_truth':>12s} {'meanY ratio':>12s} "
          f"{'var_fid':>12s} {'var_truth':>12s} {'var ratio':>10s} {'excess %':>10s}")
    excess_ss = np.zeros(N_COARSE)  # excess sum-of-squares = (var_fid-var_truth)*N
    for a in range(N_COARSE):
        ratio_mean = mean_y_fid[a] / mean_y_truth[a]
        ratio_var = var_fid[a] / var_truth[a]
        excess_ss[a] = (sum_r2_fid[a] - sum_r2_truth[a])
        print(f"{COARSE_LABELS[a]:8s} {sum_n[a]:10.0f} {mean_y_fid[a]:12.4e} {mean_y_truth[a]:12.4e} "
              f"{ratio_mean:12.4f} {var_fid[a]:12.4e} {var_truth[a]:12.4e} {ratio_var:10.4f} "
              f"{100*(ratio_var-1):10.2f}")

    print("\n=== texture cross-correlation (painted vs truth, azimuthal-mean removed) "
          "and raw pixel-difference variance ===")
    print(f"{'annulus':8s} {'corr(resid_f,resid_t)':>22s} {'var(diff)':>12s} {'var(diff)/var_truth':>20s}")
    corr = np.zeros(N_COARSE); var_diff = np.zeros(N_COARSE)
    for a in range(N_COARSE):
        corr[a] = sum_rfrt[a] / np.sqrt(sum_r2_fid[a] * sum_r2_truth[a])
        var_diff[a] = sum_diff2[a] / sum_n[a]
        print(f"{COARSE_LABELS[a]:8s} {corr[a]:22.4f} {var_diff[a]:12.4e} {var_diff[a]/var_truth[a]:20.4f}")

    print("\n=== (b) attribution: share of total excess residual sum-of-squares by annulus ===")
    total_excess = excess_ss.sum()
    total_pos_excess = excess_ss[excess_ss > 0].sum()
    for a in range(N_COARSE):
        frac_all = excess_ss[a] / total_excess if total_excess != 0 else np.nan
        frac_pos = excess_ss[a] / total_pos_excess if (total_pos_excess != 0 and excess_ss[a] > 0) else 0.0
        print(f"{COARSE_LABELS[a]:8s} excess_SS={excess_ss[a]:12.4e}  "
              f"frac_of_net_excess={100*frac_all:7.2f}%  frac_of_positive_excess={100*frac_pos:7.2f}%")
    print(f"total net excess SS (fid-truth summed over annuli): {total_excess:.4e}")
    print(f"total POSITIVE excess SS (outskirt annuli only): {total_pos_excess:.4e}")

    print("\n=== (c) mitigation: Gaussian-smooth painted y for x>1, recompute ===")
    print(f"{'annulus':8s} {'var_truth':>12s} {'var_fid(raw)':>14s} {'var_mit(s=1)':>14s} {'var_mit(s=2)':>14s} "
          f"{'ratio_raw':>10s} {'ratio_s1':>10s} {'ratio_s2':>10s}")
    for a in range(N_COARSE):
        r_raw = var_fid[a] / var_truth[a]
        r_s1 = var_mit[1][a] / var_truth[a]
        r_s2 = var_mit[2][a] / var_truth[a]
        print(f"{COARSE_LABELS[a]:8s} {var_truth[a]:12.4e} {var_fid[a]:14.4e} {var_mit[1][a]:14.4e} "
              f"{var_mit[2][a]:14.4e} {r_raw:10.4f} {r_s1:10.4f} {r_s2:10.4f}")

    print("\nmean-profile check (mitigation should NOT move the mean profile):")
    print(f"{'annulus':8s} {'meanY_fid(raw)':>16s} {'meanY_mit(s=1)':>16s} {'meanY_mit(s=2)':>16s} "
          f"{'%change s=1':>12s} {'%change s=2':>12s}")
    for a in range(N_COARSE):
        pc1 = 100 * (mean_y_mit[1][a] / mean_y_fid[a] - 1)
        pc2 = 100 * (mean_y_mit[2][a] / mean_y_fid[a] - 1)
        print(f"{COARSE_LABELS[a]:8s} {mean_y_fid[a]:16.4e} {mean_y_mit[1][a]:16.4e} {mean_y_mit[2][a]:16.4e} "
              f"{pc1:12.3f} {pc2:12.3f}")

    np.savez(
        f"{OUTDIR}/radial_texture_results.npz",
        coarse_edges=COARSE_EDGES, coarse_labels=np.array(COARSE_LABELS),
        sum_n=sum_n, mean_y_fid=mean_y_fid, mean_y_truth=mean_y_truth,
        var_fid=var_fid, var_truth=var_truth,
        var_mit_s1=var_mit[1], var_mit_s2=var_mit[2],
        mean_y_mit_s1=mean_y_mit[1], mean_y_mit_s2=mean_y_mit[2],
        excess_ss=excess_ss, corr=corr, var_diff=var_diff,
        fine_centers=fine_centers, fine_profile_fid=fine_profile_fid, fine_profile_truth=fine_profile_truth,
        n_halo=n_halo, mass=mass, r200=r200,
    )
    print(f"\nsaved {OUTDIR}/radial_texture_results.npz")


if __name__ == "__main__":
    main()
