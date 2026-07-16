"""fig13_bridge_residual — where the f_gas -> S(ell) bridge breaks, and why.

Shows (3-panel row):
  (a) partial correlation r[S(ell), X | f_gas] vs ell, for structural
      (c_dm, c_gas, f_star) and thermal (log Y, log K) group-scale halo
      summaries — what fills f_gas's blind spot as the residual grows at
      small scales.
  (b) |r[S, f_gas]| vs the 2-variable multi-R of {f_gas, c_dm} vs ell — a
      second structural variable (DM concentration) recovers most of the
      high-ell correlation that a f_gas-only bridge loses, i.e. the "lost"
      correlation is a deterministic 2nd degree of freedom, not noise.
  (c) mean off-bridge residual S - bridge(f_gas), split by feedback family
      (AGN / SN-wind / other), vs ell — the residual is feedback-mode
      specific (AGN above the bridge at high ell, SN/wind below).

Reproduces examples/_build_fgas_bridge_explore_nb.py section 4
("SS4 -- Where the bridge breaks, and what that tells us about feedback"),
lines 498-627 of the worktree copy (wl-tsz-bridge), which builds
examples/fgas_bridge_explore.ipynb. Placeholder this replaces was byte-
identical to examples/figures_lightcone/fig_bridge_residual.pdf.

Data (cached only, no recompute):
  - /mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz
    keys used: M_fof, m_gas_500c_bg, m_tot_500c_bg, m_dm_500c, m_dm_200c,
    m_gas_500c, m_gas_200c, m_star_500c, Y_500c, K_mw_500c
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_stats.npz
    (ell grid + per-run "<run>_clk" WL kappa-Cl response vectors)
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json
    (57 1P feedback runs: run id, varied param name/value/fiducial)
  - /mnt/home/mlee1/ceph/bind_science/runs/{bind,dmo}/run_0000/Cl_kappa.npz
    (fiducial kappa Cl, to build the absolute suppression curve S_fid)
  - /mnt/home/mlee1/BIND/src/bind/assets/SB35_param_minmax.csv
    (not actually needed for the family classifier here -- the run's own
    varied-parameter *name* is enough; kept import-compatible with source)

Feedback-family classifier is this notebook's OWN keyword-only _family(name)
(AGN: blackhole/quasar/radio/agn; SN/wind: wind/snii/snia/imf/supernov/
sfr/eqs; else "other") -- NOT the same function used for fig06/fig07 (which
also inspects the parameter description). This reproduces the n=18/31/8
AGN/SN/other split quoted in the placeholder caption, which the paper's own
\\todo{} already flags as differing from fig06/fig07's 20/25/12 split.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SCI = "/mnt/home/mlee1/ceph/bind_science"
CACHE = f"{SCI}/dashboard_cache"
ATLAS = f"{SCI}/halo_atlas"
RUNS = f"{SCI}/runs"
ZIDX = 1  # z_s = 1 reference plane
SNAP = 96  # z ~ 0


def _family(name):
    s = name.lower()
    if any(k in s for k in ("blackhole", "quasar", "radio", "agn")):
        return "AGN"
    if any(k in s for k in ("wind", "snii", "snia", "imf", "supernov", "sfr", "eqs")):
        return "SN / wind"
    return "other"


def summaries(run, mlo=1e13, mhi=2e13, snap=SNAP, nmin=5):
    """Group-binned halo summaries: scalar (f_gas), thermal, structural (profile shape)."""
    f = f"{ATLAS}/{run}_snap{snap:03d}.npz"
    try:
        d = np.load(f)
    except FileNotFoundError:
        return None
    s = (d["M_fof"] >= mlo) & (d["M_fof"] < mhi) & (d["m_tot_500c_bg"] > 0)
    if s.sum() < nmin:
        return None
    med = lambda a: np.nanmedian(a[s])
    return dict(
        fgas=med(d["m_gas_500c_bg"] / d["m_tot_500c_bg"]),
        cdm=med(d["m_dm_500c"] / d["m_dm_200c"]),  # DM back-reaction (contraction)
        cgas=med(d["m_gas_500c"] / d["m_gas_200c"]),  # gas central concentration
        fstar=med(d["m_star_500c"] / d["m_tot_500c_bg"]),  # stellar fraction
        lY=np.log10(med(d["Y_500c"])),
        lK=np.log10(med(d["K_mw_500c"])),  # entropy (heating/ejection)
    )


def _resid(a, c):
    return a - np.polyval(np.polyfit(c, a, 1), c)


def _resid_cols(Y, c):
    """Residualize every column of Y (n_run, n_ell) against c (n_run,)."""
    k0, k1 = np.polyfit(c, Y, 1)
    return Y - (np.outer(c, k0) + k1)


def multiR(SMv, fgas, X2):
    out = []
    for j in range(SMv.shape[1]):
        A = np.c_[fgas, X2, np.ones_like(X2)]
        coef = np.linalg.lstsq(A, SMv[:, j], rcond=None)[0]
        yp = A @ coef
        ss = 1 - ((SMv[:, j] - yp) ** 2).sum() / ((SMv[:, j] - SMv[:, j].mean()) ** 2).sum()
        out.append(np.sqrt(max(ss, 0)))
    return np.array(out)


def main():
    setup()

    design = json.load(open(f"{CACHE}/g1_design.json"))
    DESIGN = [
        dict(run=r[0], name=r[2], val=float(r[3]), fid=float(r[4]), fam=_family(r[2]))
        for r in design
    ]

    GS = dict(np.load(f"{CACHE}/g1_stats.npz", allow_pickle=True))
    ELL = GS["ell"]
    CL_BIND = np.load(f"{RUNS}/bind/run_0000/Cl_kappa.npz")["cl"]
    CL_DMO = np.load(f"{RUNS}/dmo/run_0000/Cl_kappa.npz")["cl"]
    S_FID = CL_BIND[ZIDX, ZIDX] / CL_DMO[ZIDX, ZIDX]

    RUNS_OK = [d for d in DESIGN if f"{d['run']}_clk" in GS]

    rows = [
        (summaries(d["run"]), d["fam"], (1 + GS[f"{d['run']}_clk"]) * S_FID)
        for d in RUNS_OK
    ]
    rows = [r for r in rows if r[0] is not None]
    H = {k: np.array([r[0][k] for r in rows]) for k in rows[0][0]}
    fam_ = np.array([r[1] for r in rows])
    SMv = np.array([r[2] for r in rows])  # (n_run, n_ell)

    lsel = (ELL >= 150) & (ELL <= 1e4)
    Lg = ELL[lsel]

    # (1) partial-correlation spectra: residualize S(l) and X against f_gas, correlate
    Sr = _resid_cols(SMv[:, lsel], H["fgas"])

    def pc_spec(x):
        xr = _resid(x, H["fgas"])
        return (xr @ Sr) / (np.sqrt((xr**2).sum()) * np.sqrt((Sr**2).sum(0)) + 1e-30)

    fg_c = H["fgas"] - H["fgas"].mean()
    S_c = SMv[:, lsel] - SMv[:, lsel].mean(0)
    r_fgas = (fg_c @ S_c) / (
        np.sqrt((fg_c**2).sum()) * np.sqrt((S_c**2).sum(0)) + 1e-30
    )

    # (2) 2-variable multi-R recovery with f_gas + c_dm
    mR_cdm = multiR(SMv[:, lsel], H["fgas"], H["cdm"])

    # (3) family-resolved residual off the 1-var bridge, per ell
    resid = _resid_cols(SMv[:, lsel], H["fgas"])

    # ---- plot ----
    fig, ax = plt.subplots(1, 3, figsize=TWO_COL)

    struct = [
        ("cdm", r"$c_{\rm dm}$", "#1b7837"),
        ("cgas", r"$c_{\rm gas}$", "#5aae61"),
        ("fstar", r"$f_\star$", "#00441b"),
    ]
    therm = [("lY", r"$\log Y$", "#c1272d"), ("lK", r"$\log K$", "#e08214")]
    for k, lab, c in struct:
        ax[0].plot(Lg, pc_spec(H[k]), color=c, lw=1.3, label=lab)
    for k, lab, c in therm:
        ax[0].plot(Lg, pc_spec(H[k]), color=c, lw=1.2, ls="--", label=lab)
    ax[0].axhline(0, color="0.7", lw=0.6)
    ax[0].axvspan(1000, 2000, color="0.6", alpha=0.10, lw=0)
    ax[0].set_xscale("log")
    ax[0].set_xlim(150, 1e4)
    ax[0].set_ylim(-0.8, 1.0)
    ax[0].set_xlabel(r"$\ell$")
    ax[0].set_ylabel(r"partial $r\,[S,\,X \mid f_{\rm gas}]$")
    ax[0].legend(ncol=2, loc="lower left")
    ax[0].text(230, 0.93, "structural", fontsize=6, color="#1b7837")
    ax[0].text(2200, -0.72, "thermal", fontsize=6, color="#c1272d")
    panel_label(ax[0], "(a)")

    ax[1].plot(Lg, np.abs(r_fgas), color="k", lw=1.5, label=r"$f_{\rm gas}$ only")
    ax[1].plot(Lg, mR_cdm, color="#1b7837", lw=1.5, label=r"$f_{\rm gas}+c_{\rm dm}$")
    ax[1].axvspan(1000, 2000, color="0.6", alpha=0.10, lw=0)
    ax[1].set_xscale("log")
    ax[1].set_xlim(150, 1e4)
    ax[1].set_ylim(0.4, 1.0)
    ax[1].set_xlabel(r"$\ell$")
    ax[1].set_ylabel(r"corr. with $S(\ell)$ ($|r|$, multi-$R$)")
    ax[1].legend(loc="lower left")
    panel_label(ax[1], "(b)")

    fam_colors = {"AGN": COLORS["highlight"], "SN / wind": COLORS["bind"], "other": COLORS["dmo"]}
    for fam in ("AGN", "SN / wind", "other"):
        m = fam_ == fam
        if m.any():
            ax[2].plot(Lg, resid[m].mean(0), color=fam_colors[fam], lw=1.4, label=f"{fam} (n={m.sum()})")
    ax[2].axhline(0, color="0.7", lw=0.6)
    ax[2].axvspan(1000, 2000, color="0.6", alpha=0.10, lw=0)
    ax[2].set_xscale("log")
    ax[2].set_xlim(150, 1e4)
    ax[2].set_xlabel(r"$\ell$")
    ax[2].set_ylabel(r"mean residual $S-{\rm bridge}(f_{\rm gas})$")
    ax[2].legend(loc="lower left")
    panel_label(ax[2], "(c)")

    fig.tight_layout(w_pad=1.1)
    save(fig, "figs/fig13_bridge_residual")

    li_hi = int(np.argmin(np.abs(Lg - 8000)))
    print(f"r(f_gas) at l~8k = {np.abs(r_fgas)[li_hi]:.2f} -> f_gas+c_dm multiR = {mR_cdm[li_hi]:.2f}")
    for fam in ("AGN", "SN / wind", "other"):
        m = fam_ == fam
        if m.any():
            print(
                f"  {fam:9s} n={m.sum():2d}  residual l~1.5k={resid[m].mean(0)[np.argmin(np.abs(Lg-1500))]:+.4f}"
                f"  l~8k={resid[m].mean(0)[li_hi]:+.4f}"
            )


if __name__ == "__main__":
    main()
