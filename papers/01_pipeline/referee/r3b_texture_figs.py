#!/usr/bin/env python3
"""Figures + headline numbers for the R3b multi-sample texture test.

Reads the npz written by ``r3b_texture_analysis.py`` and produces
  fig_r3b_1_oneoverN.png   -- does the texture shrink as 1/N, and what floor remains
  fig_r3b_2_mechanism.png  -- does the stochastic amplitude track gas diffuseness

Categorical colors are the Okabe-Ito colorblind-safe set, assigned in fixed order
(never cycled); one linear axis per panel; legend always present; grid recessive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIGS = Path("/mnt/home/mlee1/BIND/papers/01_pipeline/referee/figs")
WORK = Path("/mnt/home/mlee1/ceph/referee_work/texture")

# Okabe-Ito, fixed order
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

# comoving distance to snap_096 (z=0.0337, Om=0.3089, flat) in Mpc/h -> ell = k*chi
CHI_MPCH = 100.3


def load(path: Path):
    d = np.load(path, allow_pickle=True)
    slabs = sorted({k.split("/")[0] for k in d.files if "/" in k})
    return d, slabs


def stack(d, slabs, key):
    """Average a per-slab quantity over slabs, weighting by mode count."""
    vals, w = [], []
    for s in slabs:
        vals.append(d[f"{s}/{key}"])
        w.append(d[f"{s}/n_modes"])
    return np.array(vals), np.array(w)


def combine(d, slabs, key):
    """Slab-summed spectrum: LOS-independent slabs add incoherently in both
    signal and noise, so summing is the right way to build the plane-stack."""
    v, _ = stack(d, slabs, key)
    return v.sum(axis=0) if v.ndim == 2 else v.sum(axis=0)


def analyse(d, slabs, fld):
    k = d[f"{slabs[0]}/k"]
    A = combine(d, slabs, f"{fld}_auto")  # single-draw power  = |S|^2 + Pn
    X = combine(d, slabs, f"{fld}_cross")  # cross of draws     = |S|^2
    P2 = combine(d, slabs, f"{fld}_meanN2")
    P3 = combine(d, slabs, f"{fld}_meanN3")
    T = combine(d, slabs, f"{fld}_truth")
    C = combine(d, slabs, f"{fld}_canonical")
    Pn = A - X
    return dict(k=k, A=A, X=X, P2=P2, P3=P3, T=T, C=C, Pn=Pn)


def band(k, lo, hi):
    return (k >= lo) & (k <= hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(WORK / "r3b_spectra.npz"))
    ap.add_argument("--perhalo", default=str(WORK / "r3b_perhalo.npz"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)

    d, slabs = load(Path(args.spec))
    print("slabs:", slabs)
    R = {f: analyse(d, slabs, f) for f in ("gas", "y")}
    k = R["gas"]["k"]
    ell = k * CHI_MPCH

    summary = {"slabs": slabs, "chi_Mpch": CHI_MPCH}

    # ── Figure 1: 1/N scaling + the floor that survives averaging ────────────
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 11.0), sharex=True)
    for col, (fld, label) in enumerate(
        [("gas", r"gas surface density  ($\propto\tau$)"), ("y", r"Compton-$y$")]
    ):
        r = R[fld]
        ax = axes[0, col]
        ax.loglog(k, r["A"], color=OI["vermillion"], lw=2, label=r"single draw, $N{=}1$")
        ax.loglog(k, r["P3"], color=OI["blue"], lw=2, label=r"mean of $N{=}3$")
        ax.loglog(k, r["X"], color=OI["black"], lw=2, ls="--",
                  label=r"deterministic signal (draw$\times$draw cross)")
        ax.loglog(k, r["Pn"], color=OI["orange"], lw=2, label=r"stochastic texture $P_n$")
        ax.loglog(k, r["T"], color=OI["green"], lw=2, ls=":", label="hydro-pasted truth")
        ax.set_ylabel(r"$P(k)$  [arb.]")
        ax.set_title(label)
        ax.grid(alpha=0.18, lw=0.6)
        if col == 0:
            ax.legend(fontsize=7.6, frameon=False, loc="lower left")

        # how big is the stochastic component, and where does averaging put it?
        # NB the 1/N form is algebraically exact once |S|^2 is estimated by the
        # draw-cross-spectrum, so what carries information is the AMPLITUDE.
        ax = axes[1, col]
        fs = r["Pn"] / r["A"] * 100
        ax.axhspan(5, 10, color="0.85", zorder=0,
                   label=r"paper quoted 5-10% residual" if col == 0 else None)
        ax.loglog(k, fs, color=OI["vermillion"], lw=2, label=r"$N{=}1$  $P_n/P_{\rm tot}$")
        ax.loglog(k, fs / 2, color=OI["purple"], lw=2, label=r"$N{=}2$")
        ax.loglog(k, fs / 3, color=OI["blue"], lw=2, label=r"$N{=}3$")
        ax.set_ylabel(r"stochastic share of map power [%]")
        ax.set_ylim(1e-2, 60)
        ax.grid(alpha=0.18, lw=0.6)
        if col == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")

        # residual vs truth: single sample, N=3, and the N->inf floor
        ax = axes[2, col]
        ax.semilogx(k, (r["A"] - r["T"]) / r["T"] * 100, color=OI["vermillion"], lw=2,
                    label=r"single draw $-$ truth")
        ax.semilogx(k, (r["P3"] - r["T"]) / r["T"] * 100, color=OI["blue"], lw=2,
                    label=r"mean of 3 $-$ truth")
        ax.semilogx(k, (r["X"] - r["T"]) / r["T"] * 100, color=OI["black"], lw=2, ls="--",
                    label=r"$N\to\infty$ floor (model bias)")
        ax.semilogx(k, (r["C"] - r["T"]) / r["T"] * 100, color=OI["orange"], lw=2, ls="-.",
                    label="canonical paint $-$ truth\n(CAMELS-cosmology conditioning)")
        if fld == "gas":
            ax.axhline(7.77, color=OI["green"], lw=1.2, ls=":",
                       label=r"$(\Omega_b/\Omega_m)^2$ mismatch $=+7.8\%$")
        ax.axhline(0, color="0.55", lw=0.8)
        ax.set_ylim(-45, 30)
        ax.set_xlabel(r"$k$  [$h\,$Mpc$^{-1}$]")
        ax.set_ylabel("residual vs truth [%]")
        ax.grid(alpha=0.18, lw=0.6)
        if col == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")

        sec = axes[0, col].secondary_xaxis(
            "top", functions=(lambda x: x * CHI_MPCH, lambda x: x / CHI_MPCH)
        )
        sec.set_xlabel(r"$\ell \simeq k\chi$  (this plane)")

    fig.suptitle(
        "R3b multi-sample test: the sampler's stochastic texture is small, and the residual "
        "vs truth does NOT average away\n"
        "$N=3$ independent flow draws, identical halos, identical (fiducial) feedback parameters",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p1 = FIGS / f"fig_r3b_1_oneoverN{args.tag}.png"
    fig.savefig(p1, dpi=145)
    plt.close(fig)
    print("wrote", p1)

    # ── headline numbers ─────────────────────────────────────────────────────
    bands = {
        "low_k (k<0.5)": band(k, k.min(), 0.5),
        "mid_k (0.5-3)": band(k, 0.5, 3.0),
        "high_k (3-15)": band(k, 3.0, 15.0),
        "all (k<15)": band(k, k.min(), 15.0),
    }
    # empirical consistency check: the three pairwise draw-cross spectra all
    # estimate the same |S|^2, so their scatter bounds any draw-to-draw
    # correlation the decomposition would have missed.
    for fld in ("gas", "y"):
        ce = np.concatenate([d[f"{s}/{fld}_cross_each"] for s in slabs], axis=0)
        per_slab = np.array([d[f"{s}/{fld}_cross_each"] for s in slabs])  # (nslab,3,nbin)
        tot = per_slab.sum(axis=0)  # (3, nbin)
        rel = tot.std(axis=0) / np.abs(tot.mean(axis=0))
        summary[f"{fld}_cross_pair_relscatter_median"] = float(np.median(rel))
        del ce

    for fld in ("gas", "y"):
        r = R[fld]
        summary[fld] = {}
        for bn, m in bands.items():
            A, X, T, Pn, P3, C = (r[q][m] for q in ("A", "X", "T", "Pn", "P3", "C"))
            f_stoch = np.mean(Pn / A)
            res_single = np.mean((A - T) / T)
            res_mean3 = np.mean((P3 - T) / T)
            floor = np.mean((X - T) / T)
            res_canon = np.mean((C - T) / T)
            # the stochastic contribution to the residual, in PERCENTAGE POINTS
            # (a signed additive term; the residual itself can be either sign, so a
            # plain ratio is meaningless -- report pp and |pp| / |residual|)
            stoch_pp = np.mean(Pn / T) * 100
            frac_of_resid = abs(stoch_pp) / max(abs(res_single * 100), 1e-30)
            summary[fld][bn] = dict(
                f_stoch=float(f_stoch),
                resid_single_pct=float(res_single * 100),
                resid_mean3_pct=float(res_mean3 * 100),
                floor_pct=float(floor * 100),
                resid_canonical_pct=float(res_canon * 100),
                stoch_contribution_pp=float(stoch_pp),
                frac_of_residual_stochastic=float(frac_of_resid),
            )

    # ── Figure 2: mechanism -- does stochastic amplitude track diffuseness? ──
    ph = np.load(args.perhalo)
    gfrac = ph["g_noise"] / (ph["g_sig"] + 1e-30)
    tfrac = ph["t_noise"] / (ph["t_sig"] + 1e-30)
    lm = np.log10(ph["mass"])
    dif = ph["diffuse"]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    for ax, xv, xl in [
        (axes[0], lm, r"$\log_{10} M_{200c}\ [M_\odot/h]$"),
        (axes[1], dif, r"gas diffuseness  $1-M_{\rm gas}(<R_{200})/M_{\rm gas}^{\rm patch}$"),
    ]:
        for frac, c, nm in [
            (gfrac, OI["vermillion"], "gas patch"),
            (tfrac, OI["blue"], r"Compton-$y$ patch"),
        ]:
            ax.plot(xv, frac, ".", ms=1.7, alpha=0.16, color=c)
            qs = np.quantile(xv, np.linspace(0, 1, 13))
            cen, med = [], []
            for a, b in zip(qs[:-1], qs[1:]):
                m = (xv >= a) & (xv < b)
                if m.sum() > 8:
                    cen.append(0.5 * (a + b))
                    med.append(np.median(frac[m]))
            ax.plot(cen, med, "-o", color=c, lw=2, ms=5, label=nm)
        ax.set_xlabel(xl)
        ax.set_ylabel(r"per-halo stochastic fraction  $\sigma_{\rm draw}/{\rm rms}$")
        ax.grid(alpha=0.18, lw=0.6)
        ax.legend(fontsize=8.5, frameon=False)
    # correlation numbers, incl. the mass-controlled partial correlation
    rr = {}
    for nm, frac in [("gas", gfrac), ("y", tfrac)]:
        rr[nm] = {
            "spearman_vs_diffuse_RAW": float(_spearman(dif, frac)),
            "spearman_vs_logM": float(_spearman(lm, frac)),
            "partial_vs_diffuse_given_logM": float(_partial(dif, frac, lm)),
            "median": float(np.median(frac)),
        }
    rr["rho_diffuse_logM"] = float(_spearman(dif, lm))
    summary["perhalo"] = rr
    axes[1].set_title(
        "RAW $\\rho$ vs diffuseness: gas %.2f, $y$ %.2f\n(confounded: $\\rho$(diffuse, $\\log M$) = %.2f)"
        % (rr["gas"]["spearman_vs_diffuse_RAW"], rr["y"]["spearman_vs_diffuse_RAW"],
           rr["rho_diffuse_logM"]),
        fontsize=9,
    )

    # panel 3: the same relation AT FIXED MASS -- the confound removed
    ax = axes[2]
    edges = np.quantile(lm, np.linspace(0, 1, 6))
    edges[-1] += 1e-6
    shades = ["#08306b", "#2171b5", "#6baed6", "#bdd7e7", "#eff3ff"]
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (lm >= a) & (lm < b)
        if m.sum() < 40:
            continue
        xv, fv = dif[m], gfrac[m]
        qs = np.quantile(xv, np.linspace(0, 1, 7))
        cen, med = [], []
        for a2, b2 in zip(qs[:-1], qs[1:]):
            mm = (xv >= a2) & (xv < b2)
            if mm.sum() > 6:
                cen.append(0.5 * (a2 + b2))
                med.append(np.median(fv[mm]))
        ax.plot(cen, med, "-o", color=shades[i], lw=2, ms=4.5,
                label=r"$\log M$ %.2f--%.2f" % (a, b))
    ax.set_xlabel(r"gas diffuseness (at fixed halo mass)")
    ax.set_ylabel(r"gas $\sigma_{\rm draw}/{\rm rms}$")
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=7.4, frameon=False)
    ax.set_title(
        "Mass-controlled: partial $\\rho$ = %.2f (gas), %.2f ($y$)\n"
        "non-monotonic, turning over at high diffuseness"
        % (rr["gas"]["partial_vs_diffuse_given_logM"],
           rr["y"]["partial_vs_diffuse_given_logM"]),
        fontsize=9,
    )

    fig.suptitle(
        "Mechanism test: does the sampler's stochastic amplitude grow as the gas gets more diffuse?\n"
        "The raw trend is a halo-MASS effect; controlling for mass leaves no monotonic growth "
        "(partial $\\rho<0$).",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p2 = FIGS / f"fig_r3b_2_mechanism{args.tag}.png"
    fig.savefig(p2, dpi=145)
    plt.close(fig)
    print("wrote", p2)

    out = WORK / f"r3b_summary{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out)


def _partial(x, y, z):
    """Rank partial correlation of x,y controlling for z."""
    rxy, rxz, ryz = _spearman(x, y), _spearman(x, z), _spearman(y, z)
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2)))


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))


if __name__ == "__main__":
    main()
