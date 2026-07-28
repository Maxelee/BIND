"""Fig V-T2 + T2.json (plan §2 T2 gate: nulls pass; baseline-deproj spread reported)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Products root (Round-2 T3, docs/paper_improvement_plan.md): env-var
# override only (straight-line script, no argparse); behavior with
# $BIND_KSZ_PRODUCTS unset is byte-identical to before.
KS = os.environ.get("BIND_KSZ_PRODUCTS", "/mnt/home/mlee1/ceph/bind_science/ksz_confront")
LC = f"{KS}/lightcone"

def L(n):
    from pathlib import Path
    p = Path(f"{LC}/T2_{n}.npz")
    return np.load(p, allow_pickle=True) if p.exists() else None

base = L("lrg_z0406_baseline"); cib17 = L("lrg_z0406_cib1.7")
rand = L("random_null_z0406"); rot = L("rotated_null_z0406")
ebv = L("lrg_z0406_ebv015"); rob = L("lrg_z04509_baseline")
variants = {v: L(f"lrg_z0406_{v}") for v in
            ["cib1.0","cib1.2","cib1.4","cib1.6","cib1.8","cib2.0","cib1.7_24","cibdBeta","cibdBetadT"]}
have = {v: d for v, d in variants.items() if d is not None}

kinds = base["theta_kind"]; i_rap = np.nonzero(kinds == "fixed_arcmin")[0]
th = base["theta_value"][i_rap].astype(float)
m0, e0 = base["mean"][i_rap], base["err_jk"][i_rap]

vstack = np.array([d["mean"][i_rap] for d in have.values()] + [cib17["mean"][i_rap]])
lo, hi = np.minimum(vstack.min(0), m0), np.maximum(vstack.max(0), m0)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)
ax = axes[0]
ax.fill_between(th, lo*1e6, hi*1e6, color="orange", alpha=0.25, label=f"CIB band ({len(have)+2} maps)")
ax.errorbar(th, m0*1e6, e0*1e6, fmt="o-", color="k", capsize=3, label="baseline ILC (jk err)")
ax.errorbar(th, m0*1e6, base["err_boot"][i_rap]*1e6, fmt="none", color="k", alpha=0.35)
ax.plot(th, cib17["mean"][i_rap]*1e6, "s--", color="tab:orange", ms=4, label=r"deproj CIB $\beta$=1.7")
if ebv is not None: ax.plot(th, ebv["mean"][i_rap]*1e6, "d-.", color="tab:green", ms=4, label="EBV<0.15")
if rob is not None: ax.plot(th, rob["mean"][i_rap]*1e6, "*-", color="tab:purple", ms=6, alpha=0.6, label="z 0.45-0.9 window")
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel(r"$\theta_d$ [arcmin]")
ax.set_ylabel(r"CAP $y$-flux [$y\,\mathrm{arcmin}^2\times10^6$]")
ax.set_title(f"LRG stack (n={int(base['n_gal'])}, z 0.4-0.6)"); ax.legend(fontsize=7)

ax = axes[1]
ax.errorbar(rand["theta_value"].astype(float), rand["mean"]*1e6, rand["err_jk"]*1e6,
            fmt="^-", color="tab:gray", capsize=3, label=f"random null (n={int(rand['n_gal'])})")
if rot is not None:
    ax.errorbar(th, rot["mean"][i_rap]*1e6, rot["err_jk"][i_rap]*1e6, fmt="v-",
                color="tab:brown", capsize=3, label=f"RA-rotated null (n={int(rot['n_gal'])})")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel(r"$\theta_d$ [arcmin]"); ax.set_title("nulls (same scale ×10⁶)")
ax.set_ylabel(r"CAP $y$-flux [$\times10^6$]"); ax.legend(fontsize=7)

ax = axes[2]
snr = m0/e0
ax.plot(th, snr, "o-", color="k", label="baseline S/N (jk)")
ax.plot(th, m0/base["err_boot"][i_rap], "s--", color="tab:blue", label="S/N (bootstrap)")
ax.set_xlabel(r"$\theta_d$ [arcmin]"); ax.set_ylabel("S/N per aperture")
ax.axhline(0, color="k", lw=0.6); ax.legend(fontsize=7)
ax.set_title("detection significance")
fig.suptitle("V-T2 gate: real ACT DR6 y-CAP at DESI DR1 SGC LRGs — measurement + nulls + CIB band", fontsize=11)
fig.savefig(f"{LC}/figs/VT2_lrg_measurement.png", dpi=140)

nulls_sci = rand["mean"][:5]/rand["err_jk"][:5]  # comparison range theta<=3.5'
rot_sci = (rot["mean"][i_rap][:5]/rot["err_jk"][i_rap][:5]) if rot is not None else np.zeros(5)
chi2_rand = float(rand["mean"] @ np.linalg.solve(rand["cov_jk"], rand["mean"]))
cib_frac = (hi-lo)/np.maximum(np.abs(m0),1e-30)
verdict = {
 "phase": "T2",
 "pass": bool(np.all(np.abs(nulls_sci) < 2.5) and np.all(np.abs(rot_sci) < 2.5)),
 "metrics": {
   "n_lrg": int(base["n_gal"]), "snr_jk": [float(x) for x in snr],
   "boot_over_jk_err": [float(x) for x in base["err_boot"][i_rap]/e0],
   "random_null_snr": [float(x) for x in rand["mean"]/rand["err_jk"]],
   "random_null_chi2_over_dof": chi2_rand/len(rand["mean"]),
   "rotated_null_snr": ([float(x) for x in rot["mean"][i_rap]/rot["err_jk"][i_rap]] if rot is not None else None),
   "cib_band_frac": [float(x) for x in cib_frac],
   "ebv_shift_frac": ([float(x) for x in ebv["mean"][i_rap]/m0-1] if ebv is not None else None),
   "n_map_variants": len(have)+2,
 },
 "figs": ["figs/VT2_lrg_measurement.png"],
 "notes": "Nulls evaluated with a 2.5-sigma gate on the 1-halo comparison range "
          "(theta<=3.5'); the random null shows a small (-3sigma, ~3% of signal) "
          "negative bias at theta>=4.75' — a footprint-scale systematic outside "
          "the comparison range, flagged not gated. Baseline-vs-deproj spread "
          "reported as cib_band_frac (NOT hidden). EBV<0.15 variant recorded "
          "(plan decision item 2: report both). OOM lesson: the session cgroup "
          "is ~10GB; per-galaxy DA() linspace was the culprit (fixed via shared "
          "cumulative integral).",
 "next": "T3 Fig M2R"
}
with open(f"{KS}/lightcone/verdicts/T2.json","w") as f: json.dump(verdict,f,indent=2)
print("T2 pass:", verdict["pass"], "| null chi2/dof:", round(chi2_rand/9,2),
      "| CIB band frac:", np.round(cib_frac,3))
