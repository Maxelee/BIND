"""fig02_latent_physical.py — the WL suppression latent plane, coloured by
every measured halo & field quantity.

Shows: an 11-panel grid — the same (latent-1, latent-2) suppression-SVD
plane as fig02b, each panel coloured by a different physical quantity
($f_{\\rm gas}$, $f_\\star$, $T$, $Y$, gas/DM/star concentration, $P_e$,
entropy $K$, the tSZ $y$-map amplitude, the kSZ $\\tau$-map amplitude), with
the Pearson correlations $r_1$ (latent 1) / $r_2$ (latent 2) annotated per
panel.

Data (cached, read-only)
-------------------------
/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz
    t__suppression__value (253,5,724), t__suppression__valid, a__suppression__ell,
    run_ids, t__cl_yy__value / t__cl_tt__value (+ a__cl_yy__ell / a__cl_tt__ell)
/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/halo_atlas/run_{rid:04d}_snap096.npz
    one file per of the 253 valid Sobol run IDs (5120 files present); keys
    M_fof, m_gas_500c_bg, m_star_500c, m_tot_500c_bg, T_mw_500c, Y_500c,
    m_gas_500c, m_gas_200c, m_dm_500c, m_dm_200c, m_star_200c, Pe_mw_500c, K_mw_500c

Source: sobol-sb35/examples/paper_lightcone_figs2.ipynb cell 26 (§6.1,
`fig_latent_colored`). SVD of log10(S(ell)) gives the 2-D latent (Zl); per-run
group-cluster (10^13.4-14.3 Msun) medians of each halo-atlas quantity give the
colour. All closed-form (np.linalg.svd + np.corrcoef); no engine re-run.
Reimplemented verbatim below (no worktree dependency).

Placeholder replaced: figs/fig02_latent_physical.png (was byte-identical to
figs_raw/paper_lightcone_figs2/cell026_out1.png).
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import matplotlib.pyplot as plt  # noqa: E402
from paper_style import TWO_COL_TALL, save, setup  # noqa: E402

CEPH = "/mnt/home/mlee1/ceph"
DATASET = f"{CEPH}/bind_sb35/emulator_dataset.npz"
SOBOL_ATLAS = f"{CEPH}/bind_sb35/analysis_cache/halo_atlas"

ZIDX = 1                    # z_s = 1 reference source plane
ELL_MAX_DEFAULT = 2.0e4     # WL Cl upper cut (below the CIC-aliased modes)


def halo_feats(d, rid):
    """Group-cluster (10^13.4-14.3 Msun) median of each halo quantity for one Sobol run."""
    p = f"{SOBOL_ATLAS}/run_{int(rid):04d}_snap096.npz"
    import os
    if not os.path.exists(p):
        return None
    dd = np.load(p)
    M = dd["M_fof"]
    sel = (M >= 10**13.4) & (M < 10**14.3) & (dd["m_tot_500c_bg"] > 0)
    if sel.sum() < 5:
        return None
    g = lambda x: np.nanmedian(x[sel])  # noqa: E731
    return {
        r"$f_{\rm gas}$": g(dd["m_gas_500c_bg"] / dd["m_tot_500c_bg"]),
        r"$f_\star$": g(dd["m_star_500c"] / dd["m_tot_500c_bg"]),
        r"$T$": g(np.log10(np.clip(dd["T_mw_500c"], 1e-30, None))),
        r"$Y$": g(np.log10(np.clip(dd["Y_500c"], 1e-30, None))),
        r"gas conc.": g(dd["m_gas_500c"] / np.clip(dd["m_gas_200c"], 1e-9, None)),
        r"DM conc.": g(dd["m_dm_500c"] / np.clip(dd["m_dm_200c"], 1e-9, None)),
        r"$\star$ conc.": g(dd["m_star_500c"] / np.clip(dd["m_star_200c"], 1e-9, None)),
        r"$P_e$": g(np.log10(np.clip(dd["Pe_mw_500c"], 1e-30, None))),
        r"$K$": g(np.log10(np.clip(dd["K_mw_500c"], 1e-9, None))),
    }


def specfeat(d, _sv, key, a=1000, b=5000):
    v = np.asarray(d[f"t__{key}__value"])[_sv]
    v = v[:, ZIDX, :] if v.ndim == 3 else v
    ell = np.asarray(d[f"a__{key}__ell"])
    return np.log10(np.clip(v, 1e-40, None))[:, (ell >= a) & (ell < b)].mean(1)


def main():
    setup()
    d = np.load(DATASET, allow_pickle=True)
    _sv = np.asarray(d["t__suppression__valid"], bool)
    S_SOBOL = np.asarray(d["t__suppression__value"])[_sv][:, ZIDX, :]
    SOB_RID = np.asarray(d["run_ids"])[_sv]

    ellS = np.asarray(d["a__suppression__ell"])
    lcut = (ellS >= 100) & (ellS <= ELL_MAX_DEFAULT)
    Rs = np.log10(np.clip(S_SOBOL[:, lcut], 1e-6, None))
    mu, sd = Rs.mean(0), Rs.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Rc = (Rs - mu) / sd
    Rc = Rc - Rc.mean(0)
    U, Sv, Vt = np.linalg.svd(Rc, full_matrices=False)
    Zl = (U * Sv)[:, :2]

    D = [halo_feats(d, r) for r in SOB_RID]
    ref = next(dd for dd in D if dd is not None)
    FEAT = {k: np.array([dd[k] if dd is not None else np.nan for dd in D]) for k in ref}
    FEAT[r"$y$-map"] = specfeat(d, _sv, "cl_yy")
    FEAT[r"$\tau$-map"] = specfeat(d, _sv, "cl_tt")
    keys = list(FEAT.keys())

    # sign convention (matches fig02b / fig01): latent1 ~ f_star, latent2 ~ f_gas
    if np.corrcoef(Zl[:, 0], np.nan_to_num(FEAT[r"$f_\star$"]))[0, 1] < 0:
        Zl[:, 0] *= -1
    if np.corrcoef(Zl[:, 1], np.nan_to_num(FEAT[r"$f_{\rm gas}$"]))[0, 1] < 0:
        Zl[:, 1] *= -1

    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(TWO_COL_TALL[0], TWO_COL_TALL[0] / ncol * nrow),
                             sharex=True, sharey=True, constrained_layout=True)
    axf = axes.ravel()
    for a, k in zip(axf, keys):
        v = FEAT[k]
        ok = np.isfinite(v)
        vz = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
        sc = a.scatter(Zl[ok, 0], Zl[ok, 1], c=vz[ok], s=8, cmap="RdBu_r", vmin=-2.2, vmax=2.2,
                       lw=0.1, edgecolor="0.4")
        r1 = np.corrcoef(Zl[ok, 0], v[ok])[0, 1]
        r2 = np.corrcoef(Zl[ok, 1], v[ok])[0, 1]
        a.text(0.5, 0.99, k, transform=a.transAxes, ha="center", va="top", fontsize=7.5)
        a.text(0.97, 0.03, f"$r_1$ {r1:+.2f}\n$r_2$ {r2:+.2f}", transform=a.transAxes,
              ha="right", va="bottom", fontsize=5.4, color="0.2")
    for a in axf[len(keys):]:
        a.set_visible(False)
    fig.supxlabel(r"latent 1 (feedback redistribution)", fontsize=9)
    fig.supylabel(r"latent 2 (gas amplitude)", fontsize=9)
    cb = fig.colorbar(sc, ax=axes, fraction=0.022, pad=0.015)
    cb.set_label("standardized value (per panel)", fontsize=8)

    save(fig, "figs/fig02_latent_physical")
    print("panel (feature: r1 with latent1, r2 with latent2):")
    for k in keys:
        v = FEAT[k]
        ok = np.isfinite(v)
        print(f"  {k:14s} r1={np.corrcoef(Zl[ok,0],v[ok])[0,1]:+.2f}  r2={np.corrcoef(Zl[ok,1],v[ok])[0,1]:+.2f}")


if __name__ == "__main__":
    main()
