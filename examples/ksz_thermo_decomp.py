"""P4 -> full multi-probe: the (kappa, tau, y) THERMODYNAMIC DECOMPOSITION.

Three probes of the same halos, all from BIND's per-halo patches (no extra
ray-tracing), M*-matched (central M*>11.25, BGS-like):
  * kappa  (total mass)     = Sigma_total / Sigma_crit            [lensing]
  * tau    (gas density)    = sigma_T x_e Sigma_gas               [kSZ]
  * y      (gas pressure)   = sigma_T/(m_e c^2) int P_e dl         [tSZ]
=> decompose into the physical halo state:
  * total mass            <- kappa
  * gas fraction  f~gas   <- tau / kappa  (Sigma_gas / Sigma_total / f_b)
  * temperature   k_B T_e <- y / tau

The novelty: across the 256-node Sobol feedback sweep, kappa (total mass) is nearly
feedback-BLIND (narrow band) while f~gas and T_e fan out -- feedback redistributes
and heats baryons at ~fixed total mass. The (f~gas, T_e) anti-correlation is the AGN
signature (ejection lowers f~gas, heating raises T_e). This is the WL x SZ
thermodynamic decomposition (docs/wl_tsz_plan.md) that no single probe gives.

    python examples/ksz_thermo_decomp.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

CEPH = Path(os.environ.get("CEPH", "/mnt/home/mlee1/ceph"))
OUT = Path(os.environ.get("OUTPUT_ROOT", CEPH / "bind_science")) / "ksz_confront"

SIGMA_T, M_P, MSUN_G, MPC_CM = 6.6524e-25, 1.6726e-24, 1.989e33, 3.0857e24
X_E_PER_MASS, H, PIX = 0.88 / 1.6726e-24, 0.6774, 6.25 / 128.0
M_E_C2_OVER_KB = 5.93e9                          # K
F_B = 0.0490 / 0.3089
G_MPC = 4.301e-9                                 # Mpc (km/s)^2 / Msun
C_KMS = 299792.458
CI, Z_L, Z_S = 1, 0.26, 1.0                      # M*>11.25, lens z, source z


def _imp(n):
    s = importlib.util.spec_from_file_location(n, Path(__file__).resolve().parent / f"{n}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


CAP = _imp("ksz_cap_compare")


def _Dc(z):
    zz = np.linspace(0, z, 3000)
    return C_KMS / (100 * H) * np.trapezoid(1 / CAP._Ez(zz), zz)     # Mpc/h comoving


def _sigma_crit(zl, zs):
    Dl, Ds = _Dc(zl) / H / (1 + zl), _Dc(zs) / H / (1 + zs)
    Dls = (_Dc(zs) - _Dc(zl)) / H / (1 + zs)
    return C_KMS ** 2 / (4 * np.pi * G_MPC) * Ds / (Dl * Dls)        # Msun/Mpc^2


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    try:
        plt.style.use(["science", "no-latex"])
    except OSError:
        pass

    c = np.load(OUT / "bind_mstar_xprof_snap085.npz")
    x = c["x"]
    a = 1 / (1 + Z_L)
    area_cm2 = (PIX * a / H * MPC_CM) ** 2
    Sc = _sigma_crit(Z_L, Z_S)
    lM = np.nanmedian(c["logM200"][:, CI])
    theta = x * CAP._r200phys(lM, Z_L) / CAP._DA(Z_L) * CAP.ARCMIN

    tau = c["tau"][:, CI, :]                                          # (nodes, nr)
    yv = c["y"][:, CI, :]
    mtot = c["mtot_prof"][:, CI, :]                                   # Msun/h per pixel
    # kappa = Sigma_total / Sigma_crit
    kappa = (mtot / H / (PIX * a / H) ** 2) / Sc                      # dimensionless
    # gas mass per pixel from tau (invert tau_from_gas) -> f~gas = Sigma_gas/Sigma_tot/f_b
    gas = tau * H * area_cm2 / (SIGMA_T * X_E_PER_MASS * MSUN_G)      # Msun/h per pixel
    with np.errstate(divide="ignore", invalid="ignore"):
        fgas = (gas / mtot) / F_B
        Te = (yv / tau) * M_E_C2_OVER_KB
    rel = (theta > 1.2) & (theta < 6.5)                              # clean radial range

    def band(q):
        return np.nanmedian(q, 0), np.nanpercentile(q, 16, 0), np.nanpercentile(q, 84, 0)

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
    for a_, q, lab, col in [(ax[0], kappa, r"$\kappa$ (total mass)", "tab:purple"),
                            (ax[1], fgas, r"$\tilde f_{\rm gas}=\Sigma_{\rm gas}/\Sigma_{\rm tot}$", "tab:blue"),
                            (ax[2], Te, r"$k_B T_e \propto y/\tau$ [K]", "tab:red")]:
        m, lo, hi = band(q)
        a_.fill_between(theta[rel], lo[rel], hi[rel], color=col, alpha=.25)
        a_.plot(theta[rel], m[rel], "-", color=col, lw=1.8)
        a_.set_xlabel(r"$\theta$ [arcmin]"); a_.set_ylabel(lab)
        a_.set_yscale("log")
    ax[1].axhline(1, color="k", ls=":", lw=.8)
    fig.tight_layout()
    fig.savefig(OUT / "ksz_thermo_decomp.pdf", bbox_inches="tight")
    fig.savefig(OUT / "ksz_thermo_decomp.png", dpi=150, bbox_inches="tight")
    print(f"[decomp] wrote {OUT}/ksz_thermo_decomp.{{pdf,png}}")

    # feedback signature: Sobol spread of kappa vs f~gas vs T at theta~3'
    j = np.argmin(np.abs(theta - 3.0))
    def spread(q): m, lo, hi = band(q); return (hi[j] - lo[j]) / m[j]
    print(f"[decomp] fractional Sobol 16-84 spread at theta~3':")
    print(f"   kappa (mass)   = {spread(kappa):.2f}  <- feedback-BLIND")
    print(f"   f~gas          = {spread(fgas):.2f}  <- feedback-sensitive (ejection)")
    print(f"   T_e            = {spread(Te):.2f}  <- feedback-sensitive (heating)")
    r = np.corrcoef(np.log(fgas[:, j]), np.log(Te[:, j]))[0, 1]
    print(f"[decomp] corr(log f~gas, log T_e) across Sobol = {r:+.2f}  (neg = AGN ejection+heating)")

    # second figure: the feedback plane (f~gas, T_e) colored by AGN
    import pandas as pd
    df = pd.read_parquet(CEPH / "bind_sb35/analysis_cache/integrated.parquet",
                         columns=["run", "snap", "BlackHoleFeedbackFactor"]).drop_duplicates("run")
    runs = c["nodes"]
    agn = df.set_index("run").loc[runs, "BlackHoleFeedbackFactor"].to_numpy()
    fig2, ax2 = plt.subplots(figsize=(3.6, 3.0))
    sca = ax2.scatter(fgas[:, j], Te[:, j], c=np.log10(agn), cmap="viridis", s=14, edgecolor="k", lw=.2)
    fig2.colorbar(sca, label=r"$\log_{10}$ AGN feedback")
    ax2.set_xlabel(r"$\tilde f_{\rm gas}$ ($\theta\sim3'$)"); ax2.set_ylabel(r"$k_B T_e$ [K]")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    fig2.tight_layout(); fig2.savefig(OUT / "ksz_feedback_plane.pdf", bbox_inches="tight")
    fig2.savefig(OUT / "ksz_feedback_plane.png", dpi=150, bbox_inches="tight")
    print(f"[decomp] wrote {OUT}/ksz_feedback_plane.{{pdf,png}}")


if __name__ == "__main__":
    main()
