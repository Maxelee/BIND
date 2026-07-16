"""Fold the field-level kappa x y cross into the per-halo (tau, y) multi-probe.

The per-halo CAP stacks (kSZ tau + tSZ y, around BGS-like halos) probe the SELECTED
halos' CGM; the field-level kappa x y cross probes how baryonic PRESSURE traces MASS
across the whole field -- and it is the single most feedback-sensitive observable
(Sobol spread 0.76). This combines them on the 123 Sobol nodes present in both the
per-halo cache and the assembled emulator dataset:

  * perhalo     = tau^CAP(theta) + y^CAP(theta)   (M*>11.25, BGS-like)
  * perhalo+ky  = + field-level kappa x y cross (ell-banded)

GP(30 params -> stacked observables) + emcee, constrained-DIRECTIONS corner. Shows
the kappa x y cross tightening the per-halo CGM constraint.

    python examples/perhalo_plus_kappay.py --sample
    python examples/perhalo_plus_kappay.py --plot
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

CEPH = Path(os.environ.get("CEPH", "/mnt/home/mlee1/ceph"))
DS = CEPH / "bind_sb35/emulator/emulator_dataset.npz"
MSTAR = CEPH / "bind_science/ksz_confront/bind_mstar_xprof_snap085.npz"
OUT = Path(os.environ.get("OUTPUT_ROOT", CEPH / "bind_science")) / "ksz_confront"
DATA_TH = np.array([2.5, 3.4, 4.2, 5.0, 5.9, 6.7, 7.5])
CI, Z = 1, 0.26                              # M*>11.25, lens z
NB, LMIN, LMAX = 6, 200, 6000
ERR_PH, ZI = 0.35, 3                         # per-halo kSZ/tSZ frac err; kappa x y source-z index


def _imp(n):
    s = importlib.util.spec_from_file_location(n, Path(__file__).resolve().parent / f"{n}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


MP = _imp("ksz_posterior_multiprobe"); CAP = _imp("ksz_cap_compare")


def _bands(ell, y):
    edges = np.geomspace(LMIN, LMAX, NB + 1); idx = np.digitize(ell, edges) - 1
    return np.array([y[..., idx == b].mean(-1) for b in range(NB)]).T


def _build():
    e = np.load(DS, allow_pickle=True)
    rid = e["run_ids"]; X = e["X_unit"]; params = [str(p) for p in e["param_names"]]
    ell = e["a__cl_kappa_y__ell"]; ky = e["t__cl_kappa_y__value"][:, ZI]    # (123, 724)
    kyok = e["t__cl_kappa_y__valid"]
    KY = np.log(np.abs(_bands(ell, ky)) + 1e-30)                            # (123, NB)
    m = np.load(MSTAR); nodes = list(m["nodes"]); x = m["x"]; da = CAP._DA(Z)
    PH, keep = [], []
    for i, r in enumerate(rid):
        if r not in nodes or not kyok[i]:
            continue
        j = nodes.index(int(r)); lM = m["logM200"][j, CI]
        if not np.isfinite(lM):
            continue
        th = x * CAP._r200phys(lM, Z) / da * CAP.ARCMIN
        tau, yv = m["tau"][j, CI, :], m["y"][j, CI, :]
        if not (np.isfinite(tau).all() and np.isfinite(yv).all()):
            continue
        tcap = [CAP.cap(t, th, tau) for t in DATA_TH]
        ycap = [CAP.cap(t, th, yv) for t in DATA_TH]
        if min(tcap) <= 0 or min(ycap) <= 0:
            continue
        PH.append(np.log(np.concatenate([tcap, ycap]))); keep.append(i)
    keep = np.array(keep)
    return X[keep], np.array(params), np.array(PH), KY[keep]


def do_sample():
    X, params, PH, KY = _build()
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd; lo, hi = Xs.min(0), Xs.max(0)
    print(f"[phk] {len(X)} matched nodes; per-halo dim={PH.shape[1]} kappaXy dim={KY.shape[1]}")
    gPH, ePH = MP._train_gps(Xs, PH)
    gKY, eKY = MP._train_gps(Xs, KY)
    dPH, dKY = np.median(PH, 0), np.median(KY, 0)
    CPH = np.diag(ERR_PH ** 2 + ePH)
    fky = np.sqrt(KY.shape[1]) / 21.0                                       # DESxACT-precision kappa x y
    CKY = np.diag(fky ** 2 + eKY)
    prior_sd = (X.max(0) - X.min(0)) / np.sqrt(12); res = {}
    for tag, g, dd, C in [("perhalo", gPH, dPH, CPH),
                          ("perhalo_ky", gPH + gKY, np.concatenate([dPH, dKY]),
                           np.block([[CPH, np.zeros((len(CPH), len(CKY)))],
                                     [np.zeros((len(CKY), len(CPH))), CKY]]))]:
        ch = MP._run_emcee(g, dd, np.linalg.inv(C), lo, hi, Xs.shape[1]) * sd + mu
        r = ch.std(0) / prior_sd; res[tag] = (ch, r)
        print(f"[phk] {tag:11s}: #params<0.85={int((r<0.85).sum())} best={r.min():.2f} ({params[r.argmin()]})")
    np.savez(OUT / "perhalo_plus_kappay.npz", params=np.array(params),
             prior_lo=X.min(0), prior_hi=X.max(0),
             **{f"chain_{k}": v[0] for k, v in res.items()},
             **{f"ratio_{k}": v[1] for k, v in res.items()})
    print(f"[phk] wrote {OUT}/perhalo_plus_kappay.npz")


def do_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner
    import scienceplots  # noqa: F401
    try:
        plt.style.use(["science", "no-latex"])
    except OSError:
        pass
    d = np.load(OUT / "perhalo_plus_kappay.npz", allow_pickle=True)
    params = list(d["params"]); lo, hi = d["prior_lo"], d["prior_hi"]
    pmean, psd = 0.5 * (lo + hi), (hi - lo) / np.sqrt(12); white = lambda c: (c - pmean) / psd
    Uj = white(d["chain_perhalo_ky"]); ev, evec = np.linalg.eigh(np.cov(Uj.T)); V = evec[:, :2]
    def tl(v, n=3):
        i = np.argsort(np.abs(v))[::-1][:n]; return " ".join(f"{'+' if v[j] > 0 else '-'}{params[j][:9]}" for j in i)
    labs = [f"dir{k+1} ({ev[k]:.2f})\n{tl(V[:, k])}" for k in range(2)]
    ck = dict(plot_datapoints=False, fill_contours=True, levels=(0.68, 0.95), smooth=1.2, bins=26, range=[(-2.2, 2.2)] * 2)
    fig = corner.corner(white(d["chain_perhalo"]) @ V, color="tab:blue", labels=labs, label_kwargs=dict(fontsize=7), **ck)
    corner.corner(Uj @ V, color="tab:red", fig=fig, **ck)
    fig.legend(handles=[plt.Line2D([], [], color="tab:blue", label=r"per-halo ($\tau$+$y$)"),
                        plt.Line2D([], [], color="tab:red", label=r"per-halo + $\kappa\times y$")],
               loc="upper right", fontsize=9, frameon=False)
    fig.savefig(OUT / "perhalo_kappay_corner.pdf", bbox_inches="tight")
    fig.savefig(OUT / "perhalo_kappay_corner.png", dpi=150, bbox_inches="tight")
    print(f"[phk] wrote {OUT}/perhalo_kappay_corner.{{pdf,png}}")
    for t in ("perhalo", "perhalo_ky"):
        r = d[f"ratio_{t}"]; print(f"  {t:11s}: #<0.85={int((r<0.85).sum())} median={np.median(r):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.sample:
        do_sample()
    if a.plot:
        do_plot()
    if not (a.sample or a.plot):
        ap.print_help()


if __name__ == "__main__":
    main()
