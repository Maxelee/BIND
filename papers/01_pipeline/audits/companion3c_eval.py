#!/usr/bin/env python
"""companion3c_eval -- quantify WHY the population-level ejection/heating coupling
is r(ln f_gas, ln Y - ln f_gas) = 0.86 on the canonical 253-run Sobol suite, and
whether the ejection-heating plane still carries independent structure worth a
dedicated Sec. 3c companion panel.

Context: papers/01_pipeline/proto_bridge_hero.py's panel (b) found r=0.86 for this
coupling on the 253-run Sobol design (canonical). In the retired 57-run 1P design
the two axes decorrelated, which originally motivated an "ejection-heating plane"
figure (examples/bind_bridge.py::fig_hero(), never wired into the current 5-paper
suite). This script asks: is the r=0.86 coupling a DESIGN-RESPONSE property (the
Sobol prior's dominant direction moves both axes together) or an INTRINSIC
observable degeneracy that would survive even a maximally informative design? And
does the heating axis add anything the WL/tSZ observables don't already carry via
ejection alone?

Method (five checks, all on the same 253 runs / group mass bin = index 0):
  1. Raw Pearson/Spearman coupling -- reproduce the 0.86.
  2. Partial correlation after residualizing both axes on:
       (a) log10(WindEnergyIn1e51erg) alone (X_native[:, 2], the suite's dominant
           feedback lever per fig12/fig13 and proto_bridge_hero.py),
       (b) the full 30 astro-varying native parameters (OLS on standardized X;
           5 of the 35 native columns are the fixed cosmology and are dropped by
           an np.ptp==0 check, not assumed by index).
     If the residual correlation collapses toward zero, the raw 0.86 is explained
     by the Sobol design's dominant response direction, not by an irreducible
     link between ejection and heating physics.
  3. Whether the heating axis carries WL/tSZ information beyond ejection alone:
     compare R^2(S ~ ln f_gas) vs R^2(S ~ ln f_gas + heat) for the WL suppression
     amplitude S(ell=5000, z_s=1), and the same for a tSZ Cl_yy(ell~3000) proxy.
  4. PCA variance share of the standardized (ln f_gas, heat) 2-D plane -- a
     mechanical check (PC2 fraction = (1-|r|)/2 for 2 standardized variables) that
     translates r into the "genuine plane vs near-line" framing used for the 1P
     figure.
  5. Within-WindEnergy-quartile correlations, and a scan of which single astro
     parameter (of 30) correlates most strongly with each axis, as a secondary
     diagnostic of whether other directions could plausibly re-open the plane
     under a different design.

Data (read-only): /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_xpkfix.npz

Run:
  python papers/01_pipeline/audits/companion3c_eval.py
"""
import numpy as np
from scipy import stats

DATA = "/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_xpkfix.npz"
MB = 0  # group mass bin, per proto_bridge_hero.py


def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def ols_residual(y, X_design):
    """Residuals of y regressed on X_design (already includes an intercept column)."""
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return y - X_design @ beta


def r2_from_resid(y, resid):
    return 1.0 - np.var(resid) / np.var(y)


def pearson(x, y):
    r, p = stats.pearsonr(x, y)
    return r, p


def fisher_ci(r, n, level=0.95):
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(0.5 + level / 2)
    lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
    return lo, hi


def bootstrap_r(x, y, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rs[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    return np.percentile(rs, [2.5, 50, 97.5])


def main():
    d = np.load(DATA, allow_pickle=True)
    Xn = d["X_native"]                      # (253, 35)
    fgas7 = d["t__scaling_f_gas__value"]    # (253, 7)
    Y7 = d["t__scaling_Y__value"]           # (253, 7)
    mbins = d["a__scaling_f_gas__log_mass_bins"]
    S = d["t__suppression__value"]          # (253, 5, 724)
    ell_S = d["a__suppression__ell"]
    zs = d["source_redshifts"]
    cl_yy = d["t__cl_yy__value"]            # (253, 724)
    ell_yy = d["a__cl_yy__ell"]
    param_names = list(d["param_names"])

    N = Xn.shape[0]
    print("=" * 78)
    print("companion3c_eval -- ejection/heating coupling audit")
    print("=" * 78)
    print(f"N runs = {N}")
    print(f"group mass bin (idx {MB}) = {mbins[MB]:.3f}  [log10 M500c/(Msun/h)]")
    assert np.allclose(mbins, d["a__scaling_Y__log_mass_bins"]), "f_gas/Y mass bins differ"

    fgas_g = fgas7[:, MB]
    Y_g = Y7[:, MB]
    assert np.all(fgas_g > 0) and np.all(Y_g > 0), "non-positive values -- log undefined"
    ln_fgas = np.log(fgas_g)
    ln_Y = np.log(Y_g)
    heat = ln_Y - ln_fgas

    # ------------------------------------------------------------------
    # [1] raw coupling
    # ------------------------------------------------------------------
    r_raw, p_raw = pearson(ln_fgas, heat)
    rho_raw, p_rho = stats.spearmanr(ln_fgas, heat)
    lo, hi = fisher_ci(r_raw, N)
    boot_lo, boot_med, boot_hi = bootstrap_r(ln_fgas, heat)
    print("\n[1] RAW coupling: r(ln f_gas, ln Y - ln f_gas)")
    print(f"    Pearson r  = {r_raw:.4f}  (p={p_raw:.2e})  95% Fisher CI [{lo:.3f}, {hi:.3f}]")
    print(f"    bootstrap median r = {boot_med:.4f}  95% CI [{boot_lo:.3f}, {boot_hi:.3f}]")
    print(f"    Spearman rho = {rho_raw:.4f}  (p={p_rho:.2e})")

    # ------------------------------------------------------------------
    # [2a] partial on log10(WindEnergyIn1e51erg) alone
    # ------------------------------------------------------------------
    wind_log = np.log10(Xn[:, 2])
    Xw = add_intercept(wind_log[:, None])
    ln_fgas_resid_w = ols_residual(ln_fgas, Xw)
    heat_resid_w = ols_residual(heat, Xw)
    r_wind, p_wind = pearson(ln_fgas_resid_w, heat_resid_w)
    lo_w, hi_w = fisher_ci(r_wind, N)
    print("\n[2a] PARTIAL correlation controlling for log10(WindEnergyIn1e51erg) [X_native[:,2]]")
    print(f"    residual r = {r_wind:.4f}  (p={p_wind:.2e})  95% CI [{lo_w:.3f}, {hi_w:.3f}]")
    print(f"    R^2(ln f_gas ~ wind) = {r2_from_resid(ln_fgas, ln_fgas_resid_w):.4f}")
    print(f"    R^2(heat     ~ wind) = {r2_from_resid(heat, heat_resid_w):.4f}")

    # ------------------------------------------------------------------
    # [2b] partial on the full 30 astro-varying native parameters
    # ------------------------------------------------------------------
    const_idx = [i for i in range(35) if np.ptp(Xn[:, i]) == 0]
    vary_idx = [i for i in range(35) if np.ptp(Xn[:, i]) > 0]
    print(f"\n    constant native columns (fixed cosmology, dropped): {const_idx}")
    print(f"    varying native columns retained: {len(vary_idx)} (expect 30)")
    assert len(vary_idx) == 30 and len(param_names) == 30

    Xa = Xn[:, vary_idx]
    Xa_std = (Xa - Xa.mean(0)) / Xa.std(0)
    Xa_design = add_intercept(Xa_std)
    ln_fgas_resid_full = ols_residual(ln_fgas, Xa_design)
    heat_resid_full = ols_residual(heat, Xa_design)
    r_full, p_full = pearson(ln_fgas_resid_full, heat_resid_full)
    lo_f, hi_f = fisher_ci(r_full, N)
    boot_lo_f, boot_med_f, boot_hi_f = bootstrap_r(ln_fgas_resid_full, heat_resid_full)
    print("\n[2b] PARTIAL correlation controlling for all 30 standardized astro parameters (OLS)")
    print(f"    residual r = {r_full:.4f}  (p={p_full:.2e})  95% CI [{lo_f:.3f}, {hi_f:.3f}]")
    print(f"    bootstrap median r = {boot_med_f:.4f}  95% CI [{boot_lo_f:.3f}, {boot_hi_f:.3f}]")
    print(f"    R^2(ln f_gas ~ 30 params) = {r2_from_resid(ln_fgas, ln_fgas_resid_full):.4f}")
    print(f"    R^2(heat     ~ 30 params) = {r2_from_resid(heat, heat_resid_full):.4f}")

    # ------------------------------------------------------------------
    # [3] does heat carry WL/tSZ information beyond ejection alone?
    # ------------------------------------------------------------------
    iz = int(np.argmin(np.abs(zs - 1.0)))
    iell_S = int(np.argmin(np.abs(ell_S - 5000)))
    S_5000_z1 = S[:, iz, iell_S]
    print(f"\n[3] WL suppression S(ell={ell_S[iell_S]:.0f}, z_s={zs[iz]:.2f}) "
          f"vs. ejection/heating")
    r_S_fgas, p_S_fgas = pearson(S_5000_z1, ln_fgas)
    print(f"    r(S, ln f_gas)              = {r_S_fgas:.4f}  (p={p_S_fgas:.2e})")

    heat_resid_fgas = ols_residual(heat, add_intercept(ln_fgas[:, None]))
    r_S_heatresid, p_S_heatresid = pearson(S_5000_z1, heat_resid_fgas)
    print(f"    r(S, heat | ln f_gas)       = {r_S_heatresid:.4f}  (p={p_S_heatresid:.2e})")

    X_fgas_only = add_intercept(ln_fgas[:, None])
    X_fgas_heat = add_intercept(np.column_stack([ln_fgas, heat]))
    R2_S_fgas = r2_from_resid(S_5000_z1, ols_residual(S_5000_z1, X_fgas_only))
    R2_S_both = r2_from_resid(S_5000_z1, ols_residual(S_5000_z1, X_fgas_heat))
    print(f"    R^2(S ~ ln f_gas)           = {R2_S_fgas:.4f}")
    print(f"    R^2(S ~ ln f_gas + heat)    = {R2_S_both:.4f}   (delta = {R2_S_both - R2_S_fgas:+.4f})")

    iell_yy = int(np.argmin(np.abs(ell_yy - 3000)))
    Cyy = cl_yy[:, iell_yy]
    print(f"\n[3b] tSZ Cl_yy(ell={ell_yy[iell_yy]:.0f}) vs. ejection/heating")
    r_yy_fgas, p_yy_fgas = pearson(Cyy, ln_fgas)
    print(f"    r(Cl_yy, ln f_gas)          = {r_yy_fgas:.4f}  (p={p_yy_fgas:.2e})")
    r_yy_heatresid, p_yy_heatresid = pearson(Cyy, heat_resid_fgas)
    print(f"    r(Cl_yy, heat | ln f_gas)   = {r_yy_heatresid:.4f}  (p={p_yy_heatresid:.2e})")

    R2_yy_fgas = r2_from_resid(Cyy, ols_residual(Cyy, X_fgas_only))
    R2_yy_both = r2_from_resid(Cyy, ols_residual(Cyy, X_fgas_heat))
    print(f"    R^2(Cl_yy ~ ln f_gas)       = {R2_yy_fgas:.4f}")
    print(f"    R^2(Cl_yy ~ ln f_gas+heat)  = {R2_yy_both:.4f}   (delta = {R2_yy_both - R2_yy_fgas:+.4f})")

    # ------------------------------------------------------------------
    # [4] PCA of the standardized (ln f_gas, heat) plane
    # ------------------------------------------------------------------
    Z = np.column_stack([ln_fgas, heat])
    Zs = (Z - Z.mean(0)) / Z.std(0)
    cov = np.cov(Zs.T)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval = eigval[order]
    var_frac = eigval / eigval.sum()
    print("\n[4] PCA of standardized (ln f_gas, heat) 2-D plane")
    print(f"    eigenvalues        = {eigval}")
    print(f"    PC1 variance frac  = {var_frac[0]:.4f}")
    print(f"    PC2 variance frac  = {var_frac[1]:.4f}")
    print(f"    (mechanical check: (1+|r|)/2={(1 + abs(r_raw)) / 2:.4f}, "
          f"(1-|r|)/2={(1 - abs(r_raw)) / 2:.4f})")

    # ------------------------------------------------------------------
    # [5] within-WindEnergy-quartile correlations
    # ------------------------------------------------------------------
    print("\n[5] Within-WindEnergyIn1e51erg-quartile correlations (does the coupling")
    print("    survive once the dominant lever is held roughly fixed?)")
    wind = Xn[:, 2]
    q = np.quantile(wind, [0, 0.25, 0.5, 0.75, 1.0])
    for i in range(4):
        loq, hiq = q[i], q[i + 1]
        mask = (wind >= loq) & (wind < hiq) if i < 3 else (wind >= loq) & (wind <= hiq)
        n_in = int(mask.sum())
        if n_in > 3:
            rq, pq = pearson(ln_fgas[mask], heat[mask])
            print(f"    Q{i + 1} (n={n_in:3d}, wind in [{loq:.2f},{hiq:.2f})): "
                  f"r = {rq:+.4f}  (p={pq:.2e})")
        else:
            print(f"    Q{i + 1} (n={n_in:3d}): too few points")

    # ------------------------------------------------------------------
    # bonus: leading single-parameter correlates of each axis
    # ------------------------------------------------------------------
    print("\n[bonus] single-parameter correlation scan (30 astro params), sorted by |r(ln f_gas)|")
    rows = []
    for j in range(Xa.shape[1]):
        rf, _ = pearson(Xa[:, j], ln_fgas)
        rh, _ = pearson(Xa[:, j], heat)
        rows.append((param_names[j], rf, rh))
    rows.sort(key=lambda t: -abs(t[1]))
    for name, rf, rh in rows[:8]:
        print(f"    {name:35s} r(ln f_gas)={rf:+.3f}   r(heat)={rh:+.3f}")

    # ------------------------------------------------------------------
    # [6] mechanistic decomposition: why does the FIXED unit-slope
    # subtraction (heat = ln Y - ln f_gas) retain an ejection signature?
    # ------------------------------------------------------------------
    print("\n[6] Mechanistic decomposition: ln f_gas vs ln Y directly, and the")
    print("    log-log slope implicitly assumed by heat = ln Y - ln f_gas")
    r_fgY, p_fgY = pearson(ln_fgas, ln_Y)
    print(f"    r(ln f_gas, ln Y)          = {r_fgY:.4f}  (p={p_fgY:.2e})")
    X_a1 = add_intercept(ln_fgas[:, None])
    intercept_fgY, slope = np.linalg.lstsq(X_a1, ln_Y, rcond=None)[0]
    print(f"    OLS log-log slope d(lnY)/d(ln f_gas) = {slope:.4f}  (1.0 would make")
    print("    heat = lnY - ln f_gas exactly the orthogonal OLS residual of lnY on")
    print("    ln f_gas; slope != 1 leaves an (slope-1) x ln f_gas component inside")
    print("    'heat' by construction)")
    proper_resid = ols_residual(ln_Y, X_a1)  # true orthogonal residual, slope free
    r_proper, p_proper = pearson(ln_fgas, proper_resid)
    print(f"    r(ln f_gas, proper OLS residual of lnY on ln f_gas) = {r_proper:.4f}  "
          f"(expected ~0 by construction)")
    print(f"    predicted mechanical contribution (slope-1) = {slope - 1:.4f}  "
          f"(the excess coefficient baked into 'heat' beyond a clean orthogonal split)")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  raw r                              = {r_raw:.3f}")
    print(f"  partial r | log10(WindEnergy)       = {r_wind:.3f}")
    print(f"  partial r | 30 astro params (OLS)   = {r_full:.3f}")
    print(f"  R^2(S) fgas-only -> fgas+heat        = {R2_S_fgas:.3f} -> {R2_S_both:.3f} "
          f"(delta {R2_S_both - R2_S_fgas:+.3f})")
    print(f"  R^2(Cl_yy) fgas-only -> fgas+heat     = {R2_yy_fgas:.3f} -> {R2_yy_both:.3f} "
          f"(delta {R2_yy_both - R2_yy_fgas:+.3f})")
    print(f"  PCA PC2 variance fraction           = {var_frac[1]:.3f}")
    print(f"  r(ln f_gas, ln Y) directly           = {r_fgY:.3f}  (log-log slope {slope:.3f})")
    print("=" * 78)


if __name__ == "__main__":
    main()
