#!/usr/bin/env python
"""Composite total-matter S(k) = P/P_DMO over a CAMELS suite, from eval outputs.

Loads per-sim composite.npz (+ optional hydro_replace.npy ceiling) and full_maps.npz
and plots truth / BIND / hydro-replaced suppression, mean +- sim scatter. Run on the
cached fm_two_head CV composites now, or on a fresh epoch047 run once it lands.

Run:  python composite_sk.py --suite CV --model fm_two_head
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from bind.metrics import power_spectrum_pylians_2d as PK

BOX = 50.0


def sk(field, dmo):
    k, p, _ = PK(field, box_size=BOX, MAS="None"); _, pd, _ = PK(dmo, box_size=BOX, MAS="None")
    return k, p / pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--suite", default="CV")
    ap.add_argument("--models", default="fm_two_head,fm_two_head_47",
                    help="comma list of model dirs to overlay (e.g. last.ckpt vs epoch047)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]

    sims = sorted(glob.glob(f"{args.suite_root}/{args.suite}/sim_*/snap_090"))
    truth_S, hydro_S = [], []
    bind_S = {m: [] for m in models}
    kref = None
    for s in sims:
        s = Path(s); base = s / "mass_threshold_1p000e13"
        present = [m for m in models if (base / m / "composite.npz").exists()]
        if not present:
            continue
        fm = np.load(s / "full_maps.npz")
        dmo = fm["dmo_fullbox"].astype(np.float64)
        truth = fm["truth_maps"].astype(np.float64).sum(0)
        k, st = sk(truth, dmo); kref = k; truth_S.append(st)
        for m in present:
            _, sb = sk(np.load(base / m / "composite.npz")["composite"].astype(np.float64).sum(0), dmo)
            bind_S[m].append(sb)
        hr = base / present[0] / "hydro_replace.npy"
        if hr.exists():
            h = np.load(hr).astype(np.float64); h = h.sum(0) if h.ndim == 3 else h
            _, sh = sk(h, dmo); hydro_S.append(sh)
    print(f"[composite_sk] {args.suite}: {len(truth_S)} sims; models {[ (m,len(bind_S[m])) for m in models]}")

    def at(a, q):
        return np.array(a).mean(0)[np.argmin(np.abs(kref - q))]
    print(f"{'':>16} {'k=2':>7} {'k=10':>7} {'k=30':>7} {'k=50':>7}")
    rows = [("truth", truth_S)] + [(m, bind_S[m]) for m in models if bind_S[m]]
    if hydro_S:
        rows.append(("hydro_replace", hydro_S))
    for nm, a in rows:
        print(f"{nm:>16} {at(a,2):7.3f} {at(a,10):7.3f} {at(a,30):7.3f} {at(a,50):7.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(7, 8), height_ratios=[2, 1], sharex=True)
    st_mean = np.array(truth_S).mean(0)
    series = [("truth", truth_S, "k", "Truth/DMO")]
    mcols = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
    for i, m in enumerate(models):
        if bind_S[m]:
            series.append((m, bind_S[m], mcols[i % len(mcols)], f"{m}/DMO"))
    if hydro_S:
        series.append(("hydro_replace", hydro_S, "0.5", "Hydro-replaced/DMO"))
    for nm, a, col, lab in series:
        a = np.array(a)
        ax[0].plot(kref, a.mean(0), color=col, label=lab, lw=1.8,
                   ls="--" if nm == "hydro_replace" else "-")
        if nm != "hydro_replace":
            ax[0].fill_between(kref, *np.percentile(a, [16, 84], axis=0), color=col, alpha=0.12)
        ax[1].plot(kref, a.mean(0) / st_mean, color=col, lw=1.6,
                   ls="--" if nm == "hydro_replace" else "-")
    ax[0].axhline(1, color="gray", ls=":"); ax[0].set_xscale("log"); ax[0].set_ylabel("P(k)/P_DMO(k)")
    ax[0].legend(); ax[0].set_title(f"{args.suite} composite suppression ({len(truth_S)} sims)")
    ax[1].axhline(1, color="k", lw=0.7); ax[1].fill_between(kref, 0.95, 1.05, color="g", alpha=0.1)
    ax[1].set_xscale("log"); ax[1].set_ylim(0.8, 1.2); ax[1].set_ylabel("P/P_truth"); ax[1].set_xlabel("k [h/Mpc]")
    fig.tight_layout()
    out = Path(args.out or f"ceph/fm_diag/composite_sk_{args.suite}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"[composite_sk] wrote {out}")


if __name__ == "__main__":
    main()
