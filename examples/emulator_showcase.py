"""Presentation figures: the emulator reproduces every lightcone statistic, instantly.

Two figures for the paper / a talk:

1. **Showcase** (`emulator_showcase.png`) — for one held-out parameter point, overlay
   the emulator prediction (line + 1σ band) on the ray-traced truth for *every*
   statistic at once, annotated with the wall-clock prediction time.  The headline:
   the full multi-probe data vector that costs hours of painting + ray-tracing is
   reproduced in milliseconds.

2. **Response** (`emulator_response.png`) — against a held-out one-at-a-time suite
   (the 1P set when it exists, else the twobound corners), the emulator-predicted
   feedback response vs the truth, parameter by parameter.  The varied parameter of
   each comparison run is inferred from its offset from the fiducial, so this works
   for any OAT design.

Trains on the SB35 Sobol suite (or loads a saved bundle).  The comparison suite
must have its statistics computed; the 1P lightcones still need the paste→lux→stats
pipeline (`DESIGN=1P sbatch run_sobol_{paste,lux,stats}.sh`).

    python examples/emulator_showcase.py                       # twobound comparison
    python examples/emulator_showcase.py --compare_dir /ceph/bind_science/runs/1P
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

import bind
from bind.emulator import EmulatorDataset, assemble
from bind.emulator.core import Emulator
from bind.emulator.dataset import DEFAULT_DMO, params_to_unit

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "figures_lightcone"
DS = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")
BUNDLE = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator/lightcone_emulator_gp.pt")
TWOBOUND = Path("/mnt/home/mlee1/ceph/bind_science/runs/twobound")

# statistics to show, with the plot kind (the rest are skipped gracefully)
PANELS = [
    ("suppression", "logx"), ("cl_kappa_auto", "loglog"), ("cl_kappa_y", "logx_signed"),
    ("cl_yy", "loglog"), ("cl_kappa_tau", "logx_signed"), ("cl_tt", "loglog"),
    ("cl_yt", "logx_signed"), ("peak_counts", "logy"), ("minima_counts", "logy"),
    ("pdf", "logy"), ("mf_v0", "lin"), ("mf_v1", "lin"), ("mf_v2", "lin"),
    ("peak_R", "lin"), ("dm_pdf", "logy"), ("scaling_Y", "logy"),
]


def _emulator() -> Emulator:
    if BUNDLE.exists():
        print(f"[showcase] loading {BUNDLE}")
        return Emulator.load(BUNDLE)
    print("[showcase] training a GP emulator")
    ds = EmulatorDataset.load(DS)
    em = Emulator(backend="auto", n_components=12).fit(ds, verbose=False)
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    em.save(BUNDLE)
    return em


def _truth_at_zs(ds: EmulatorDataset, name: str, run: int, zs_plane: int):
    t = ds.targets.get(name)
    if t is None or not t.valid_mask()[run]:
        return None
    return t.value[run, zs_plane] if t.src_axis is not None else t.value[run]


def showcase_figure(em: Emulator, cmp_ds: EmulatorDataset, run: int, z_s: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zs_plane = int(np.argmin(np.abs(cmp_ds.source_redshifts - z_s)))

    t0 = time.perf_counter()
    pred = em.predict(cmp_ds.X_native[run], z_s=z_s)
    dt = 1e3 * (time.perf_counter() - t0)

    panels = [(n, k) for n, k in PANELS if n in pred]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    ell_vals = em.bin_values.get("suppression", em.bin_values.get("cl_kappa"))
    for ax, (name, kind) in zip(axes, panels):
        y = np.asarray(pred[name])
        # per-stat axis values (axis names can collide, e.g. peak nu 68 vs peak_R nu 14)
        x = ell_vals if name in ("suppression", "cl_kappa_auto") else em.bin_values.get(name)
        truth = _truth_at_zs(cmp_ds, name if name in cmp_ds.targets else "cl_kappa", run, zs_plane) \
            if name not in ("cl_kappa_auto", "suppression") else None
        if name == "suppression":
            t = cmp_ds.targets.get("suppression")
            truth = t.value[run, zs_plane] if t is not None and t.valid_mask()[run] else None
        if name == "cl_kappa_auto":
            t = cmp_ds.targets.get("cl_kappa")
            truth = (t.value[run, zs_plane, zs_plane] if t is not None and t.valid_mask()[run]
                     else None)
        if x is None or y.ndim != 1:
            ax.set_visible(False)
            continue
        err = pred.get(f"{name}_err")
        plot = {"loglog": ax.loglog, "logx": ax.semilogx, "logx_signed": ax.semilogx,
                "logy": ax.semilogy, "lin": ax.plot}.get(kind, ax.plot)
        plot(x, y, color="C0", lw=1.8, label="emulator")
        if err is not None and np.all(np.isfinite(err)) and kind not in ("loglog", "logy"):
            ax.fill_between(x, y - err, y + err, color="C0", alpha=0.25)
        if truth is not None and np.ndim(truth) == 1 and len(truth) == len(x):
            ax.plot(x, truth, "k.", ms=3, label="truth")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"BIND lightcone-statistics emulator — {len(panels)} statistics for one "
                 f"feedback model at $z_s={z_s}$, predicted in {dt:.0f} ms "
                 f"(vs hours of paint + ray-trace)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "emulator_showcase.png", dpi=130)
    plt.close(fig)
    print(f"[showcase] wrote emulator_showcase.png ({len(panels)} panels, {dt:.0f} ms/predict)")


# field statistics with a directly file-backed truth (scaling_* come from the
# parquet, not these run dirs, so they are excluded from the ratio figure)
RATIO_PANELS = ["suppression", "cl_kappa_auto", "cl_kappa_y", "cl_yy", "cl_kappa_tau",
                "cl_tt", "cl_yt", "peak_counts", "minima_counts", "pdf",
                "mf_v0", "mf_v1", "mf_v2", "peak_R", "dm_pdf", "moments"]


def _truth_panel(run_dir: Path, name: str, em: Emulator, zs_plane: int):
    """The 1-D truth vector for one panel from a run's stat files (None if absent)."""
    from bind.emulator.dataset import STAT_SPECS, _load_stat
    if name in ("suppression", "cl_kappa_auto"):
        v, _, _ = _load_stat(run_dir, STAT_SPECS["cl_kappa"])
        if v is None:
            return None
        auto = v[zs_plane, zs_plane]
        if name == "cl_kappa_auto":
            return auto
        dmo = em.cl_dmo[zs_plane]
        return auto / np.where(dmo > 0, dmo, np.nan)
    spec = STAT_SPECS.get(name)
    if spec is None:
        return None
    v, _, _ = _load_stat(run_dir, spec)
    if v is None:
        return None
    return np.take(v, zs_plane, axis=spec.src_axis) if spec.src_axis is not None else v


def twobound_ratio_figure(em: Emulator, param, fid_dir: Path, twobound_dir: Path,
                          z_s: float) -> None:
    """Per-statistic response to one parameter as a RATIO to fiducial, emulator vs truth.

    For the chosen parameter's lower & upper prior bound: the emulator ratio
    ``stat(bound)/stat(fiducial)`` (two lines) overlaid on the truth ratio
    ``stat_run(bound)/stat_run(fiducial)`` (two point sets).  Dividing by the
    fiducial cancels the shared shape and exposes the feedback response itself.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j = list(em.param_names).index(param) if isinstance(param, str) else int(param)
    pname = em.param_names[j]
    lo_dir = twobound_dir / f"run_{2 * j:04d}"
    hi_dir = twobound_dir / f"run_{2 * j + 1:04d}"
    lo_p, hi_p = np.load(lo_dir / "params.npy"), np.load(hi_dir / "params.npy")
    fid_p = bind.fiducial_params()
    zs_plane = int(np.argmin(np.abs(em.source_redshifts - z_s)))

    e_fid = em.predict(fid_p, z_s=z_s)
    e_lo = em.predict(lo_p, z_s=z_s)
    e_hi = em.predict(hi_p, z_s=z_s)
    ell_vals = em.bin_values.get("suppression", em.bin_values.get("cl_kappa"))

    panels = [n for n in RATIO_PANELS if n in e_fid and np.ndim(e_fid[n]) == 1]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    def _ratio(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.asarray(a, float) / np.asarray(b, float)
        return np.where(np.isfinite(r), r, np.nan)

    for ax, name in zip(axes, panels):
        x = ell_vals if name in ("suppression", "cl_kappa_auto") else em.bin_values.get(name)
        if x is None or len(x) != len(e_fid[name]):
            ax.set_visible(False)
            continue
        is_ell = name.startswith("cl_") or name == "suppression"
        # trust mask: drop the CIC-aliasing ell tail; keep finite-fiducial bins
        keep = (np.asarray(x) < 1.5e4) if is_ell else np.ones(len(x), bool)
        keep &= np.abs(e_fid[name]) > 1e-12 * np.nanmax(np.abs(e_fid[name]))
        xm = np.asarray(x)[keep]
        plot = ax.semilogx if is_ell else ax.plot
        r_lo, r_hi = _ratio(e_lo[name], e_fid[name])[keep], _ratio(e_hi[name], e_fid[name])[keep]
        plot(xm, r_lo, "C0-", lw=1.6, label="emu lower")
        plot(xm, r_hi, "C3-", lw=1.6, label="emu upper")
        t_fid = _truth_panel(fid_dir, name, em, zs_plane)
        t_lo = _truth_panel(lo_dir, name, em, zs_plane)
        t_hi = _truth_panel(hi_dir, name, em, zs_plane)
        tr = []
        if t_fid is not None and len(t_fid) == len(x):
            tf = np.asarray(t_fid)
            if t_lo is not None:
                tr_lo = _ratio(t_lo, tf)[keep]
                plot(xm, tr_lo, "C0.", ms=4, label="truth lower")
                tr.append(tr_lo)
            if t_hi is not None:
                tr_hi = _ratio(t_hi, tf)[keep]
                plot(xm, tr_hi, "C3.", ms=4, label="truth upper")
                tr.append(tr_hi)
        # robust y-limits from whatever ratios we have (around 1)
        allr = np.concatenate([r_lo, r_hi] + tr) if tr else np.concatenate([r_lo, r_hi])
        finite = allr[np.isfinite(allr)]
        if finite.size:
            lo, hi = np.nanpercentile(finite, [2, 98])
            pad = 0.15 * (hi - lo) + 1e-3
            ax.set_ylim(min(lo - pad, 0.97), max(hi + pad, 1.03))
        ax.axhline(1, ls=":", c="grey", lw=.7)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"Response to {pname} (ratio to fiducial), $z_s={z_s}$: "
                 f"emulator (lines) vs truth (points)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"emulator_ratio_{pname}.png", dpi=130)
    plt.close(fig)
    print(f"[ratio] wrote emulator_ratio_{pname}.png ({len(panels)} panels)")


def _varied_param(X_native: np.ndarray) -> np.ndarray:
    """Index of the astro param each OAT run varies (largest offset from fiducial)."""
    u = params_to_unit(X_native)
    u_fid = params_to_unit(bind.fiducial_params()[None])[0]
    return np.argmax(np.abs(u - u_fid), axis=1)


def response_figure(em: Emulator, cmp_ds: EmulatorDataset, z_s: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ell = cmp_ds.targets["suppression"].axes["ell"]
    band = (ell > 3e3) & (ell < 2e4)
    zs_plane = int(np.argmin(np.abs(cmp_ds.source_redshifts - z_s)))

    vparam = _varied_param(cmp_ds.X_native)
    u = params_to_unit(cmp_ds.X_native)
    # summary = WL suppression dip depth at z_s
    Strue = cmp_ds.targets["suppression"].value[:, zs_plane]      # (N, L)
    dip_true = np.nanmin(np.where(band, Strue, np.nan), 1)
    pred = em.predict(cmp_ds.X_native, z_s=z_s)["suppression"]    # (N, L)
    dip_pred = np.nanmin(np.where(band, pred, np.nan), 1)

    # the params with the most distinct levels in this suite (top 6 by count)
    uniq, counts = np.unique(vparam, return_counts=True)
    top = uniq[np.argsort(-counts)][:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, pidx in zip(axes.ravel(), top):
        sel = vparam == pidx
        xx = u[sel, pidx]
        o = np.argsort(xx)
        ax.plot(xx[o], dip_true[sel][o], "ks", ms=6, label="truth")
        ax.plot(xx[o], dip_pred[sel][o], "C0o-", ms=4, label="emulator")
        ax.axhline(1, ls=":", c="grey", lw=.8)
        ax.set_title(cmp_ds.param_names[pidx], fontsize=9)
        ax.set_xlabel("param (unit)")
        ax.set_ylabel(r"min $S(\ell)$")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"Feedback response of the WL suppression dip ($z_s={z_s}$): "
                 f"emulator vs truth on {cmp_ds.n_runs} held-out runs", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "emulator_response.png", dpi=130)
    plt.close(fig)
    r = np.corrcoef(dip_true, dip_pred)[0, 1]
    print(f"[response] wrote emulator_response.png — dip(S) corr(emu,truth)={r:.3f} "
          f"over {cmp_ds.n_runs} runs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare_dir", type=Path, default=TWOBOUND,
                    help="held-out OAT suite with computed stats (twobound, or 1P when ready)")
    ap.add_argument("--z_s", type=float, default=1.0)
    ap.add_argument("--run", type=int, default=0, help="comparison run for the showcase panel")
    ap.add_argument("--fiducial_dir", type=Path,
                    default=Path("/mnt/home/mlee1/ceph/bind_science/runs/bind/run_0000"),
                    help="fiducial-astro run with stats, for the ratio-to-fiducial figure")
    ap.add_argument("--ratio_param", default="BlackHoleRadiativeEfficiency",
                    help="parameter for the per-statistic ratio-to-fiducial figure")
    args = ap.parse_args()

    em = _emulator()
    print(em.describe())
    if not list(args.compare_dir.glob("run_*/Cl_kappa.npz")):
        print(f"\n[showcase] {args.compare_dir} has no computed statistics yet.")
        print("  Generate them with the lightcone pipeline, e.g. for the 1P set:")
        print("    DESIGN=1P sbatch run_sobol_paste.sh && DESIGN=1P sbatch run_sobol_lux.sh \\")
        print("      && DESIGN=1P sbatch run_sobol_stats.sh")
        return
    cmp_ds = assemble(args.compare_dir, dmo_dir=DEFAULT_DMO, parquet=None, verbose=False)
    print(f"\n[showcase] comparison suite: {cmp_ds.n_runs} runs from {args.compare_dir}")
    showcase_figure(em, cmp_ds, args.run, args.z_s)
    if "suppression" in cmp_ds.targets:
        response_figure(em, cmp_ds, args.z_s)
    # per-statistic response as a ratio to the fiducial truth (emulator vs truth)
    if (args.fiducial_dir / "Cl_kappa.npz").exists():
        twobound_ratio_figure(em, args.ratio_param, args.fiducial_dir,
                              args.compare_dir, args.z_s)
    else:
        print(f"[ratio] no fiducial stats at {args.fiducial_dir} — skipping ratio figure")


if __name__ == "__main__":
    main()
