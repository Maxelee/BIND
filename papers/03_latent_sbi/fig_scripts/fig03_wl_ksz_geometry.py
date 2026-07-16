"""fig03_wl_ksz_geometry.py — canonical-correlation geometry between the WL
suppression latent and the kSZ tau/y latent.

Shows: two side-by-side scatter panels — the WL suppression latent (left)
and the kSZ tau/y latent (right), both coloured by the same inner-halo gas
fraction, with the canonical correlations and principal angles between the
two 2-D planes annotated.

Data (cached, read-only)
-------------------------
/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz
    t__cl_kappa__value, cl_dmo, a__cl_kappa__ell, t__cl_kappa__valid, X_unit,
    param_names, run_ids  (the WL side; same file/keys as fig01)
/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet
    run, snap, M200, M_gas_500, M_gas_200, M_tot_500, M_tot_200 (gas-zone labels)
/mnt/home/mlee1/ceph/bind_science/ksz_confront/bind_tauy_xprof_snap085.npz
    x (17,), tau (256,4,17), y (256,4,17), nodes (256,) — radial tau/y profiles
    feeding the kSZ latent (mass-bin index 1 = [13.4,13.8], matching the WL
    gas-zone mass bin)

Source: sobol-sb35/examples/wl_latent_sbi.ipynb cell 10 (§2c), engine calls
E.ksz_latent(), E.gas_zone_labels(), E.orient_to_gradient(), E.align_planes()
(sklearn CCA + QR-based principal angles — lightweight, not trained).
Reimplemented verbatim below (no worktree dependency).

Placeholder replaced: figs/fig03_wl_ksz_geometry.png (was byte-identical to
examples/wl_latent_sbi_figs/f2c_wl_vs_ksz_latent.png).
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import matplotlib.pyplot as plt  # noqa: E402
from paper_style import TWO_COL, panel_label, save, setup  # noqa: E402

CEPH = "/mnt/home/mlee1/ceph"
DATASET = f"{CEPH}/bind_sb35/emulator_dataset.npz"
PARQUET = f"{CEPH}/bind_sb35/analysis_cache/integrated.parquet"
KSZ_XPROF = f"{CEPH}/bind_science/ksz_confront/bind_tauy_xprof_snap085.npz"

F_B = 0.0486 / 0.3089            # TNG cosmic baryon fraction Omega_b/Omega_m
ELL_MAX = 2.0e4                  # WL Cl upper cut (below the CIC-aliased modes)


# ───────────────────────── engine helpers (verbatim from wl_latent_sbi.py) ──────────────────
def wl_suppression_stack(d, ell_lo=100.0, ell_hi=ELL_MAX):
    cl = d["t__cl_kappa__value"]
    cl_dmo = d["cl_dmo"]
    ell = d["a__cl_kappa__ell"]
    autos = np.stack([cl[:, i, i, :] for i in range(cl.shape[-2])], axis=1)  # (N,5,L)
    sel = (ell >= ell_lo) & (ell <= ell_hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        logS = np.log10(autos[:, :, sel] / cl_dmo[:, sel])
    R = logS.reshape(logS.shape[0], -1)
    return R, np.asarray(d["t__cl_kappa__valid"], bool)


def latent_svd(R):
    mu, sd = R.mean(0), R.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Rs = (R - mu) / sd
    Rs = Rs - Rs.mean(0)
    U, S, Vt = np.linalg.svd(Rs, full_matrices=False)
    lam = S**2 / np.sum(S**2)
    return U * S, lam, Vt


def gas_zone_labels(run_ids, snap=85, mlo=13.4, mhi=13.8):
    cols = ["run", "snap", "M200", "M_gas_500", "M_gas_200", "M_tot_500", "M_tot_200"]
    fo = pd.read_parquet(PARQUET, columns=cols)
    lo = np.log10(fo.M200.values)
    fo = fo[(fo.snap.values == snap) & (lo > mlo) & (lo < mhi)].copy()
    fo["f_in"] = fo.M_gas_500.values / fo.M_tot_500.values / F_B
    dgas = fo.M_gas_200.values - fo.M_gas_500.values
    dtot = fo.M_tot_200.values - fo.M_tot_500.values
    fo["f_out"] = dgas / np.where(dtot > 0, dtot, np.nan) / F_B
    g = fo.groupby("run")[["f_in", "f_out"]].median().reindex(run_ids)
    return g.f_in.values, g.f_out.values


def orient_to_gradient(Z, f_in, f_out):
    Z2 = Z[:, :2]
    m = np.isfinite(f_in)
    grad = np.array([np.cov(Z2[m, k], f_in[m])[0, 1] for k in range(2)])
    e1 = grad / (np.linalg.norm(grad) + 1e-12)
    e2 = np.array([-e1[1], e1[0]])
    Ze = Z2 @ np.c_[e1, e2]
    mo = np.isfinite(f_out) & np.isfinite(Ze[:, 1])
    if mo.sum() > 3 and np.corrcoef(f_out[mo], Ze[mo, 1])[0, 1] < 0:
        e2 = -e2
        Ze = Z2 @ np.c_[e1, e2]
    return Ze, e1, e2


def ksz_latent(mb=1, band=(0.3, 1.5), xprof=KSZ_XPROF):
    c = np.load(xprof, allow_pickle=True)
    x, tau, y, nodes = c["x"], c["tau"], c["y"], c["nodes"]
    m = (x >= band[0]) & (x <= band[1])
    R = np.hstack([np.log10(np.clip(tau[:, mb, m], 1e-30, None)),
                   np.log10(np.clip(y[:, mb, m], 1e-30, None))])
    good = np.isfinite(R).all(1)
    Z, lam, _ = latent_svd(R[good])
    return Z, lam, nodes[good]


def align_planes(Za, runs_a, Zb, runs_b):
    from sklearn.cross_decomposition import CCA
    ia = {int(r): i for i, r in enumerate(runs_a)}
    common = [int(r) for r in runs_b if int(r) in ia]
    A = np.array([Za[ia[r], :2] for r in common])
    B = np.array([Zb[list(runs_b).index(r), :2] for r in common])
    A = (A - A.mean(0)) / (A.std(0) + 1e-12)
    B = (B - B.mean(0)) / (B.std(0) + 1e-12)
    cca = CCA(n_components=2).fit(A, B)
    U, V = cca.transform(A, B)
    cc = [abs(np.corrcoef(U[:, k], V[:, k])[0, 1]) for k in range(2)]
    Qa = np.linalg.qr(A)[0]
    Qb = np.linalg.qr(B)[0]
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(sv, -1, 1)))
    return dict(canon_corr=cc, principal_angle_deg=angles.tolist(), n_common=len(common))


def main():
    setup()
    d = np.load(DATASET, allow_pickle=True)
    rid = d["run_ids"]

    R, vR = wl_suppression_stack(d)
    Z, lam, Vt = latent_svd(R[vR])
    f_in, f_out = gas_zone_labels(rid[vR])
    Ze, e1, e2 = orient_to_gradient(Z, f_in, f_out)
    mi = np.isfinite(f_in) & np.isfinite(Ze[:, 0])

    Zk, lamk, runsk = ksz_latent()
    fk_in, fk_out = gas_zone_labels(runsk)
    Zke, ek1, ek2 = orient_to_gradient(Zk, fk_in, fk_out)
    al = align_planes(Ze, rid[vR], Zke, runsk)
    cc = np.array(al["canon_corr"])
    ang = np.array(al["principal_angle_deg"])

    fig, ax = plt.subplots(1, 2, figsize=TWO_COL, constrained_layout=True)
    vmin = min(np.nanmin(f_in[mi]), np.nanmin(fk_in[np.isfinite(fk_in)]))
    vmax = max(np.nanmax(f_in[mi]), np.nanmax(fk_in[np.isfinite(fk_in)]))
    ax[0].scatter(Ze[mi, 0], Ze[mi, 1], c=f_in[mi], cmap="cividis", s=13,
                  edgecolor="0.3", lw=0.2, vmin=vmin, vmax=vmax)
    ax[0].set_xlabel(r"$\hat e_1$")
    ax[0].set_ylabel(r"$\hat e_2$")
    ax[0].text(0.03, 0.03, "WL latent", transform=ax[0].transAxes, fontsize=7.5)
    panel_label(ax[0], "(a)")

    mki = np.isfinite(fk_in)
    s1 = ax[1].scatter(Zke[mki, 0], Zke[mki, 1], c=fk_in[mki], cmap="cividis", s=13,
                       edgecolor="0.3", lw=0.2, vmin=vmin, vmax=vmax)
    ax[1].set_xlabel(r"$\hat e_1$")
    ax[1].set_ylabel(r"$\hat e_2$")
    ax[1].text(0.03, 0.03, r"kSZ $\tau/y$ latent", transform=ax[1].transAxes, fontsize=7.5)
    panel_label(ax[1], "(b)")

    cb = fig.colorbar(s1, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(r"inner $f_{\rm gas}(<R_{500c})/(\Omega_b/\Omega_m)$", fontsize=7.5)

    # canonical corr / principal-angle numbers go in the caption (style: no in-figure
    # titles); printed below for the caption pass to cite verbatim.
    save(fig, "figs/fig03_wl_ksz_geometry")
    print(f"shared runs n={al['n_common']}")
    print(f"axis-1: canon corr {cc[0]:.2f}, angle {ang[0]:.0f} deg")
    print(f"axis-2: canon corr {cc[1]:.2f}, angle {ang[1]:.0f} deg")


if __name__ == "__main__":
    main()
