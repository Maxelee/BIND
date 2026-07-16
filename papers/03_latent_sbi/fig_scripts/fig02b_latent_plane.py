"""fig02b_latent_plane.py — the WL suppression latent plane, with the 1P
feedback spokes projected through the same SVD basis.

Shows
-----
(a) Linear scree of the $z_s=1$ suppression latent $\\log_{10}S(\\ell)$ over the
    253 SB35 Sobol runs (PC1 ~89%, PC2 ~11%, cumulative ~99%).
(b) The 253 Sobol nodes on the (latent-1, latent-2) plane, coloured by
    group-cluster $f_{\\rm gas}$, with the measured fiducial (black star) and
    the one-parameter-at-a-time ("1P"/twobound) AGN and SN/wind spokes
    projected through the identical basis.

Data (cached, read-only)
-------------------------
/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz
    t__suppression__value (253,5,724), t__suppression__valid, a__suppression__ell,
    a__scaling_f_gas__log_mass_bins (7,), t__scaling_f_star__value / t__scaling_f_gas__value (253,7)
/mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json
    the 1P run -> (param, value, fiducial) table (57 runs)
/mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_stats.npz
    per-1P-run response arrays "<run>_clk" (ell,) = C_l(run)/C_l(fid) - 1, z_s=1
/mnt/home/mlee1/ceph/bind_science/runs/{bind,dmo}/run_0000/Cl_kappa.npz
    the measured fiducial (BIND) and DMO convergence auto-spectra -> S_fid = C_bind/C_dmo
/mnt/home/mlee1/BIND/src/bind/assets/SB35_param_minmax.csv
    packaged repo asset (not ceph): per-param LogFlag/Description, used only to classify
    each 1P run's varied parameter into an AGN / SN-wind / other family by keyword.

Source: sobol-sb35/examples/paper_lightcone_figs2.ipynb cell 22 (§5.1, `fig_latent_plane`),
preamble in cell 1 (DESIGN/GS/S_FID construction). Computation is SVD + a closed-form linear
projector (`Vt[:2].T`) — no retraining. Reimplemented verbatim below (no worktree dependency).

Placeholder replaced: figs/fig02b_latent_plane.png (was byte-identical to
figs_raw/paper_lightcone_figs2/cell022_out1.png).
"""
import csv
import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import matplotlib.pyplot as plt  # noqa: E402
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

CEPH = "/mnt/home/mlee1/ceph"
DATASET = f"{CEPH}/bind_sb35/emulator_dataset.npz"
CACHE = f"{CEPH}/bind_science/dashboard_cache"
RUNS = f"{CEPH}/bind_science/runs"
ACSV = "/mnt/home/mlee1/BIND/src/bind/assets/SB35_param_minmax.csv"

ZIDX = 1                    # z_s = 1 reference source plane
ELL_MAX_DEFAULT = 2.0e4     # WL Cl upper cut (below the CIC-aliased modes)

# family colours mapped onto the suite's fixed semantic palette:
# AGN -> highlight (red), SN/wind -> bind (blue), other -> dmo (grey) — matches the
# original figure's red/blue convention exactly while using the shared COLORS table.
FAMC = {"AGN": COLORS["highlight"], "SN / wind": COLORS["bind"], "other": COLORS["dmo"]}


def _auto_family(name):
    s = name.lower()
    if any(k in s for k in ("blackhole", "quasar", "radio", "agn")):
        return "AGN"
    if any(k in s for k in ("wind", "snii", "snia", "imf", "supernov", "sfr", "eqs")):
        return "SN / wind"
    return "other"


def main():
    setup()
    d = np.load(DATASET, allow_pickle=True)
    _sv = np.asarray(d["t__suppression__valid"], bool)
    S_SOBOL = np.asarray(d["t__suppression__value"])[_sv][:, ZIDX, :]

    ellS = np.asarray(d["a__suppression__ell"])
    lcut = (ellS >= 100) & (ellS <= ELL_MAX_DEFAULT)

    Rs = np.log10(np.clip(S_SOBOL[:, lcut], 1e-6, None))
    mu_l, sd_l = Rs.mean(0), Rs.std(0)
    sd_l = np.where(sd_l > 0, sd_l, 1.0)
    Rc = (Rs - mu_l) / sd_l
    rowm_l = Rc.mean(0)
    Rc = Rc - rowm_l
    U, Sv, Vt = np.linalg.svd(Rc, full_matrices=False)
    lam = Sv**2 / np.sum(Sv**2)

    def project(Smat):
        r = (np.log10(np.clip(np.atleast_2d(Smat)[:, lcut], 1e-6, None)) - mu_l) / sd_l - rowm_l
        return r @ Vt[:2].T

    Zlat = (U * Sv)[:, :2]

    # sign convention: latent 1 increases with f_star, latent 2 with group-cluster f_gas
    mb_s = np.asarray(d["a__scaling_f_gas__log_mass_bins"])
    g_grp = int(np.argmin(np.abs(mb_s - 13.38)))
    g_gas = int(np.argmin(np.abs(mb_s - 13.88)))
    fstar = np.asarray(d["t__scaling_f_star__value"])[_sv][:, g_grp]
    fgas_c = np.asarray(d["t__scaling_f_gas__value"])[_sv][:, g_gas]
    if np.corrcoef(Zlat[:, 0], fstar)[0, 1] < 0:
        Vt[0] *= -1
        Zlat[:, 0] *= -1
    if np.corrcoef(Zlat[:, 1], fgas_c)[0, 1] < 0:
        Vt[1] *= -1
        Zlat[:, 1] *= -1

    # ---- the 1P (twobound) design + response, and the measured fiducial ----
    _meta = {r["ParamName"]: r for r in csv.DictReader(open(ACSV))}
    design = []
    for run, _i, name, val, fid in json.load(open(f"{CACHE}/g1_design.json")):
        design.append(dict(run=run, name=name, val=float(val), fid=float(fid),
                           fam=_auto_family(name)))
    GS = dict(np.load(f"{CACHE}/g1_stats.npz", allow_pickle=True))
    runs_ok = [r for r in design if f"{r['run']}_clk" in GS]

    CL_BIND = np.load(f"{RUNS}/bind/run_0000/Cl_kappa.npz")["cl"]
    CL_DMO = np.load(f"{RUNS}/dmo/run_0000/Cl_kappa.npz")["cl"]
    S_FID = CL_BIND[ZIDX, ZIDX] / CL_DMO[ZIDX, ZIDX]

    fig, ax = plt.subplots(1, 2, figsize=TWO_COL, gridspec_kw=dict(width_ratios=[1, 1.55]),
                           constrained_layout=True)

    # (a) scree
    ax[0].bar(np.arange(1, 7), lam[:6] * 100, color=COLORS["dmo"], width=0.7)
    ax[0].plot(np.arange(1, 7), np.cumsum(lam[:6]) * 100, "-o", ms=3, color=COLORS["highlight"])
    ax[0].axhline(99, color="0.7", lw=0.7, ls=":")
    ax[0].text(2.7, 90, f"2 comp:\n{lam[:2].sum()*100:.1f}%", fontsize=7, color=COLORS["highlight"])
    ax[0].set_xlabel("latent component")
    ax[0].set_ylabel(r"variance of $\log_{10}S(\ell)$ [%]")
    ax[0].set_ylim(0, 105)
    panel_label(ax[0], "(a)")

    # (b) the shared plane, coloured by group-cluster gas content
    sc = ax[1].scatter(Zlat[:, 0], Zlat[:, 1], c=fgas_c, s=16, cmap="cividis",
                       lw=0.2, edgecolor="0.3", zorder=2, label="Sobol runs")
    cb = fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.02)
    cb.set_label(r"group-cluster $f_{\rm gas}$", fontsize=8)
    zf = project(S_FID)[0]
    ax[1].plot(*zf, "*", color="k", ms=13, mec="w", mew=0.5, zorder=5, label="fiducial")
    for r in runs_ok:
        p = project((1 + GS[f"{r['run']}_clk"]) * S_FID)[0]
        ax[1].plot([zf[0], p[0]], [zf[1], p[1]], color=FAMC[r["fam"]], lw=0.5, alpha=0.55, zorder=3)
        ax[1].plot(*p, "o", color=FAMC[r["fam"]], ms=2.6, zorder=4)
    for fam in ("AGN", "SN / wind"):
        ax[1].plot([], [], color=FAMC[fam], lw=1.4, label=f"1P: {fam}")
    ax[1].set_xlabel(r"latent 1 (feedback redistribution)")
    ax[1].set_ylabel(r"latent 2 (gas amplitude)")
    ax[1].legend(loc="upper left", ncol=2)
    panel_label(ax[1], "(b)", loc="lower right")

    save(fig, "figs/fig02b_latent_plane")
    print(f"suppression latent: 2 comps = {lam[:2].sum()*100:.1f}% of variance; "
          f"r[latent1,f_star]={np.corrcoef(Zlat[:,0], fstar)[0,1]:+.2f} "
          f"r[latent2,f_gas@13.9]={np.corrcoef(Zlat[:,1], fgas_c)[0,1]:+.2f}")


if __name__ == "__main__":
    main()
