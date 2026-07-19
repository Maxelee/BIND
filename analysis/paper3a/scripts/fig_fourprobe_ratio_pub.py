"""Publication figure: the four-probe ejection signature — data/TNG
ratio vs scale for every probe, one mini-panel each, shared y axis.
The paper's motivator figure: the deficit deepens toward small scales /
high nu coherently across four independent measurement chains.

Run: python analysis/paper3a/scripts/fig_fourprobe_ratio_pub.py
Out: wp6_propagation/figures/fourprobe_ratio_pub.{pdf,png} + plans copy
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.inference.bblock import BBlock  # noqa: E402
from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.ksz import KszBlock  # noqa: E402
from analysis.paper3a.style import (  # noqa: E402
    COL, W_DOUBLE, apply, caveat_region, refline)

A = Path("/mnt/ceph/users/mlee1/paper3/A")
PLANS_FIG = Path("/mnt/home/mlee1/bind-paper3-plans/figures_AB")


def main() -> None:
    apply()
    fig, axes = plt.subplots(1, 4, figsize=(W_DOUBLE, 2.5), sharey=True)

    # ---- 1. X-ray f_gas vs mass ---------------------------------------
    ax = axes[0]
    ksz = KszBlock()
    fgas = FgasBlock(emu=ksz.emu)
    m = np.asarray(fgas.data.bin_centers, float)
    dat = np.asarray(fgas.data.values, float)
    err = np.asarray(fgas.data.stat_err, float)
    u_fid = np.full(31, 0.5)
    from analysis.paper3a.emulator import params_meta as pm
    u_fid[:30] = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    fid = fgas.predict(u_fid[None, :])[0]
    ax.errorbar(m, dat / fid, yerr=err / fid, fmt="o", ms=4,
                color=COL["data"], capsize=2.5, zorder=5)
    ax.set_xlabel(r"$\log_{10} M_{500c}\,[{\rm M_\odot}]$")
    ax.set_title(r"X-ray f$_{\rm gas}$ (eRASS1)", fontsize=8.5)
    ax.text(0.05, 0.06, r"$z < 0.2$", transform=ax.transAxes, fontsize=7)

    # ---- 2. kSZ CAP profile -------------------------------------------
    ax = axes[1]
    fc = json.loads((A / "wp5_chains" / "a5_ksz_firstcontact.json")
                    .read_text())
    r = np.asarray(fc["radii_arcmin"], float)
    dat = np.asarray(fc["data_values"], float)
    err = np.asarray(fc["data_err"], float)
    fid = np.asarray(fc["fid_pred_fsat0.2"], float)
    ax.errorbar(r, dat / fid, yerr=err / np.abs(fid), fmt="o", ms=4,
                color=COL["data"], capsize=2.5, zorder=5)
    ax.set_xlabel(r"$R_{\rm CAP}$ [arcmin]")
    ax.set_title("kSZ (DESI×ACT)", fontsize=8.5)
    ax.text(0.05, 0.06, r"$z_{\rm eff} = 0.74$", transform=ax.transAxes,
            fontsize=7)

    # ---- 3. gamma_t x y ------------------------------------------------
    ax = axes[2]
    pc = np.load(A / "wp6_propagation" / "a6_pandey_compare.npz",
                 allow_pickle=True)
    th = np.asarray(pc["theta_arcmin"], float)
    dat = np.asarray(pc["data_comb"], float)
    err = np.sqrt(np.diag(np.asarray(pc["cov_comb"], float)))
    fid = np.asarray(pc["xi_fid"], float)
    sel = np.asarray(pc["valid"], bool)
    ax.errorbar(th[sel], dat[sel] / fid[sel], yerr=err[sel] / fid[sel],
                fmt="o", ms=4, color=COL["data"], capsize=2.5, zorder=5)
    ax.set_xscale("log")
    caveat_region(ax, th[sel].min() * 0.8, 10.0, label="IA")
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_title(r"$\gamma_t \times y$ (DES×ACT)", fontsize=8.5)
    ax.text(0.05, 0.06, "DES $n(z)$", transform=ax.transAxes, fontsize=7)

    # ---- 4. kappa-peaks x y -------------------------------------------
    ax = axes[3]
    blk = BBlock()
    y_fid, _ = blk.model(np.zeros(2))
    sig = np.sqrt(blk.var_stat + np.diag(blk.cov_sys))
    x = np.arange(4)
    ax.errorbar(x, blk.data / y_fid, yerr=sig / y_fid, fmt="o", ms=4,
                color=COL["data"], capsize=2.5, zorder=5)
    ax.set_xticks(x, [r"$\nu$1–2", r"$\nu$2–3", r"$\nu$3–4",
                      r"$\nu{\geq}4$"], fontsize=7)
    ax.set_xlabel(r"peak significance")
    ax.set_title(r"$\kappa$-peaks $\times$ y (DES×ACT)", fontsize=8.5)
    ax.text(0.05, 0.06, "CAP 4$'$", transform=ax.transAxes, fontsize=7)

    for ax in axes:
        refline(ax, y=1.0, label="TNG" if ax is axes[3] else None)
    axes[0].set_ylabel("data / TNG-painted model")
    axes[0].set_ylim(0.0, 1.45)

    fig.tight_layout()
    out = A / "wp6_propagation" / "figures" / "fourprobe_ratio_pub"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=300)
    if PLANS_FIG.exists():
        shutil.copy(f"{out}.png", PLANS_FIG / "fourprobe_ratio_pub.png")
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    main()
