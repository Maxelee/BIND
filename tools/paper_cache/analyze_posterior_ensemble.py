#!/usr/bin/env python3
"""Calibration analysis of the per-halo BIND posterior ensembles.

Input: the .npz written by build_posterior_ensemble.py
Output: PIT/rank statistics, coverage curves, sigma_post/sigma_pop ratios,
and a summary figure.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def pit_ranks(truth, draws, rng):
    """Randomized PIT: (#draws<truth + u*(#ties+1)) / (N+1)."""
    lt = (draws < truth[:, None]).sum(1)
    eq = (draws == truth[:, None]).sum(1)
    u = rng.random(len(truth))
    return (lt + u * (eq + 1)) / (draws.shape[1] + 1)

def coverage_curve(truth, draws, levels):
    out = []
    for q in levels:
        lo = np.quantile(draws, (1 - q) / 2, axis=1)
        hi = np.quantile(draws, (1 + q) / 2, axis=1)
        out.append(np.mean((truth >= lo) & (truth <= hi)))
    return np.array(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--fig", type=Path, default=None)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    truth, draws, names = d["truth"], d["draws"], [str(x) for x in d["names"]]
    suite, logm = d["suite"], d["log_m200c"]
    keep = np.isfinite(truth).all(1) & np.isfinite(draws).all((1, 2))
    truth, draws, suite, logm = truth[keep], draws[keep], suite[keep], logm[keep]
    n_h, N, n_s = draws.shape
    print(f"{n_h} halos x {N} draws x {n_s} summaries; "
          f"logM {logm.min():.2f}-{logm.max():.2f}; suites {dict(zip(*np.unique(suite, return_counts=True)))}")
    rng = np.random.default_rng(1)
    levels = np.linspace(0.05, 0.95, 19)

    rows = []
    cov_tab = {}
    for j, nm in enumerate(names):
        t, g = truth[:, j], draws[:, :, j]
        pit = pit_ranks(t, g, rng)
        D, p = stats.kstest(pit, "uniform")
        mu = g.mean(1)
        sd = g.std(1, ddof=1)
        frac_sd = np.median(sd / np.abs(mu))                    # per-halo posterior width
        z = (t - mu) / np.where(sd > 0, sd, np.nan)             # standardized truth residual
        # population scatter of the single-draw residual (what the paper quotes)
        r1 = g[:, 0] / t - 1.0
        pop_sd = np.diff(np.percentile(r1, [16, 84]))[0] / 2
        # population scatter of the truth itself at fixed mass (detrended in logM)
        lt = np.log10(np.clip(t, 1e-30, None))
        A = np.vstack([np.ones_like(logm), logm]).T
        res_t = lt - A @ np.linalg.lstsq(A, lt, rcond=None)[0]
        pop_truth = np.std(res_t) * np.log(10)                  # -> fractional
        cov = coverage_curve(t, g, levels)
        cov_tab[nm] = cov
        rows.append(dict(summary=nm, pit_D=D, pit_p=p,
                         frac_below=np.mean(pit < 1.0 / (N + 1)),
                         frac_above=np.mean(pit > N / (N + 1)),
                         sigma_post=frac_sd, z_rms=np.sqrt(np.nanmean(z**2)),
                         z_med=np.nanmedian(z),
                         sigma_pop_resid=pop_sd, sigma_pop_truth=pop_truth,
                         ratio_post_over_pop=frac_sd / pop_sd if pop_sd else np.nan,
                         cov68=cov[np.argmin(abs(levels - 0.65))],
                         cov90=cov[np.argmin(abs(levels - 0.9))]))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n=== per-summary calibration (all suites pooled) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== per-suite, integrated masses ===")
    for j, nm in enumerate(names[:3] + [names[-1]]):
        jj = names.index(nm)
        for s in ["CV", "1P", "Test"]:
            m = suite == s
            if m.sum() < 5:
                continue
            t, g = truth[m, jj], draws[m][:, :, jj]
            pit = pit_ranks(t, g, rng)
            D, _ = stats.kstest(pit, "uniform")
            mu, sd = g.mean(1), g.std(1, ddof=1)
            print(f"{nm:20s} {s:5s} n={m.sum():3d}  D={D:.3f}  "
                  f"edge_lo={np.mean(pit < 1/(N+1)):.3f} edge_hi={np.mean(pit > N/(N+1)):.3f}  "
                  f"sigma_post={np.median(sd/mu)*100:5.2f}%  "
                  f"bias(mean/truth-1)={np.median(mu/t-1)*100:+6.2f}%  "
                  f"z_rms={np.sqrt(np.mean(((t-mu)/sd)**2)):5.2f}")
    print("\n=== integrated masses by M200c bin (all suites) ===")
    edges=[13.0,13.3,13.6,15.0]
    for j,nm in enumerate(names[:3]):
        for i in range(len(edges)-1):
            m=(logm>=edges[i])&(logm<edges[i+1])
            if m.sum()<5: continue
            t,g=truth[m,j],draws[m][:,:,j]
            pit=pit_ranks(t,g,rng); mu,sd=g.mean(1),g.std(1,ddof=1)
            print(f"{nm:20s} [{edges[i]},{edges[i+1]}) n={m.sum():3d} D={stats.kstest(pit,'uniform')[0]:.3f} "
                  f"sigma_post={np.median(sd/mu)*100:5.2f}%  bias={np.median(mu/t-1)*100:+6.2f}%  "
                  f"z_rms={np.sqrt(np.mean(((t-mu)/sd)**2)):5.2f}  cov68={coverage_curve(t,g,[0.68])[0]:.2f}")

    out = a.npz.with_suffix(".calib.csv")
    df.to_csv(out, index=False)
    np.savez(a.npz.with_name(a.npz.stem + "_coverage.npz"),
             levels=levels, **{k: v for k, v in cov_tab.items()})
    print(f"\nwrote {out}")

    if a.fig:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = [names[0], names[1], names[2],
                "Sigma_Gas_0-0.25R200", "Sigma_Gas_0.5-1R200", "Sigma_Stars_0-0.25R200"]
        fig, ax = plt.subplots(2, len(show), figsize=(3.0 * len(show), 5.6))
        for k, nm in enumerate(show):
            j = names.index(nm)
            pit = pit_ranks(truth[:, j], draws[:, :, j], rng)
            ax[0, k].hist(pit, bins=20, range=(0, 1), density=True,
                          color="tab:blue", edgecolor="k", lw=0.4)
            ax[0, k].axhline(1, ls="--", c="k")
            ax[0, k].set_title(nm, fontsize=8)
            ax[0, k].set_xlabel("PIT")
            ax[1, k].plot(levels, cov_tab[nm], "o-", ms=3)
            ax[1, k].plot([0, 1], [0, 1], "k--")
            ax[1, k].set_xlabel("nominal"); ax[1, k].set_ylabel("empirical")
            ax[1, k].set_xlim(0, 1); ax[1, k].set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(a.fig, dpi=140)
        print(f"wrote {a.fig}")

if __name__ == "__main__":
    main()
