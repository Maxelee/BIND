"""fig01_latent_axes_zones.py — the 2-D WL suppression latent: scree, param
loadings, and the two gas-zone colourings.

Shows
-----
(a) Linear PCA scree of the WL power-spectrum suppression $\\log_{10}S(\\ell,z_s)$
    stacked over the 253 SB35 Sobol runs: 2 components capture ~99% of the
    variance.
(b) The 8 astro parameters with the largest |loading| on either latent axis
    ($\\hat e_1$, $\\hat e_2$).
(c)/(d) The oriented latent plane, coloured by the per-run group-scale halo
    gas fraction: inner ($<R_{500c}$) in (c), outer ($R_{500c}\\to R_{200c}$)
    in (d); Pearson $r$ annotated.

Data (cached, read-only)
-------------------------
/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz
    t__cl_kappa__value (253,5,5,724), cl_dmo (5,724), a__cl_kappa__ell (724,),
    t__cl_kappa__valid (253,), X_unit (253,30), param_names (30,), run_ids (253,)
/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet
    columns run, snap, M200, M_gas_500, M_gas_200, M_tot_500, M_tot_200

Source: sobol-sb35/examples/wl_latent_sbi.ipynb cell 8 (§2b), which calls
engine helpers in wl_latent_sbi.py (E.wl_suppression_stack, E.latent_svd,
E.gas_zone_labels, E.orient_to_gradient, E.latent_loadings). All are pure
numpy/pandas closed-form reductions (z-scored SVD + a pandas groupby-median +
Pearson correlations) — no model training or emulator evaluation. Reimplemented
verbatim below so this script has no dependency on the (ephemeral) topic-branch
worktree.

Placeholder replaced: figs/fig01_latent_axes_zones.png (was byte-identical to
examples/wl_latent_sbi_figs/f2b_wl_latent_gaszones.png).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from paper_style import COLORS, TWO_COL_TALL, panel_label, save, setup  # noqa: E402
from param_labels import short_label  # noqa: E402

CEPH = "/mnt/home/mlee1/ceph"
DATASET = f"{CEPH}/bind_sb35/emulator_dataset.npz"
PARQUET = f"{CEPH}/bind_sb35/analysis_cache/integrated.parquet"

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


def latent_loadings(Ze, theta):
    out = np.zeros((Ze.shape[1], theta.shape[1]))
    for k in range(Ze.shape[1]):
        zk = (Ze[:, k] - Ze[:, k].mean()) / (Ze[:, k].std() + 1e-12)
        tz = (theta - theta.mean(0)) / (theta.std(0) + 1e-12)
        out[k] = tz.T @ zk / len(zk)
    return out


def main():
    setup()
    d = np.load(DATASET, allow_pickle=True)
    pn = [str(s) for s in d["param_names"]]
    rid = d["run_ids"]

    R, vR = wl_suppression_stack(d)
    Z, lam, Vt = latent_svd(R[vR])
    theta = d["X_unit"][vR]

    f_in, f_out = gas_zone_labels(rid[vR])
    Ze, e1, e2 = orient_to_gradient(Z, f_in, f_out)
    LOAD = latent_loadings(Ze, theta)
    mi = np.isfinite(f_in) & np.isfinite(Ze[:, 0])
    mo = np.isfinite(f_out) & np.isfinite(Ze[:, 1])
    r_in = np.corrcoef(f_in[mi], Ze[mi, 0])[0, 1]
    r_out = np.corrcoef(f_out[mo], Ze[mo, 1])[0, 1]

    fig, ax = plt.subplots(2, 2, figsize=TWO_COL_TALL, constrained_layout=True)

    # (a) scree
    ax[0, 0].bar(np.arange(1, 7), lam[:6], color=COLORS["bind"], alpha=0.85)
    ax[0, 0].plot(np.arange(1, 7), np.cumsum(lam[:6]), "o-", ms=3, lw=1, color="k")
    ax[0, 0].set_xlabel("component")
    ax[0, 0].set_ylabel("variance fraction")
    ax[0, 0].text(0.96, 0.5, f"2 comp:\n{lam[:2].sum()*100:.0f}%", transform=ax[0, 0].transAxes,
                  ha="right", va="center", fontsize=7)
    panel_label(ax[0, 0], "(a)")

    # (b) param loadings on each axis
    top = np.argsort(np.maximum(np.abs(LOAD[0]), np.abs(LOAD[1])))[::-1][:8][::-1]
    yp = np.arange(len(top))
    ax[0, 1].barh(yp - 0.2, LOAD[0, top], 0.38, color=COLORS["bind"], label=r"$\hat e_1$")
    ax[0, 1].barh(yp + 0.2, LOAD[1, top], 0.38, color=COLORS["secondary"], label=r"$\hat e_2$")
    ax[0, 1].axvline(0, color="k", lw=0.6)
    ax[0, 1].set_yticks(yp)
    ax[0, 1].set_yticklabels([short_label(pn[i]) for i in top], fontsize=6.5)
    ax[0, 1].set_xlabel("loading")
    ax[0, 1].legend(loc="lower right", ncol=1)
    panel_label(ax[0, 1], "(b)")

    # (c) inner gas fraction
    sc1 = ax[1, 0].scatter(Ze[mi, 0], Ze[mi, 1], c=f_in[mi], cmap="cividis", s=14,
                           edgecolor="0.3", lw=0.2)
    ax[1, 0].set_xlabel(r"$\hat e_1$")
    ax[1, 0].set_ylabel(r"$\hat e_2$")
    cb1 = fig.colorbar(sc1, ax=ax[1, 0], fraction=0.046)
    cb1.set_label(r"inner $f_{\rm gas}(<R_{500c})/(\Omega_b/\Omega_m)$", fontsize=6.5)
    ax[1, 0].text(0.04, 0.04, f"$r={r_in:+.2f}$", transform=ax[1, 0].transAxes, fontsize=7)
    panel_label(ax[1, 0], "(c)")

    # (d) outer gas fraction
    sc2 = ax[1, 1].scatter(Ze[mo, 0], Ze[mo, 1], c=f_out[mo], cmap="cividis", s=14,
                           edgecolor="0.3", lw=0.2)
    ax[1, 1].set_xlabel(r"$\hat e_1$")
    ax[1, 1].set_ylabel(r"$\hat e_2$")
    cb2 = fig.colorbar(sc2, ax=ax[1, 1], fraction=0.046)
    cb2.set_label(r"outer $f_{\rm gas}(R_{500c}\to R_{200c})/(\Omega_b/\Omega_m)$", fontsize=6.5)
    ax[1, 1].text(0.04, 0.04, f"$r={r_out:+.2f}$", transform=ax[1, 1].transAxes, fontsize=7)
    panel_label(ax[1, 1], "(d)")

    save(fig, "figs/fig01_latent_axes_zones")
    print(f"2-comp var={lam[:2].sum()*100:.1f}%  r_in={r_in:+.2f}  r_out={r_out:+.2f}")
    print("top-8 loading params (bottom->top of panel b):", [pn[i] for i in top])


if __name__ == "__main__":
    main()
