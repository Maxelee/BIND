#!/usr/bin/env python3
"""F8-field1p-showcase: rebuild two paper figures from the paper cache of the
model selected via the PAPER_* env vars (see tools/paper_cache/paper_config.py).

1. fig7_1p_field_response  — 1P butterfly: per-pixel log10((hi+eps)/(lo+eps)) [dex]
   of the most-massive halo's surface-density maps between the 1P_p{j}_2 (prior max)
   and 1P_p{j}_n2 (prior min) variants, truth vs BIND, for the 6 hero params in
   field1p.npz. Layout: rows = params, 6 cols = {DM,Gas,Stars} x {Truth,BIND},
   symmetric per-channel color scale with bottom colorbars (matches the current
   paper figure); code copied from the fig6/field-response cell of
   examples/_build_paper_nbs.py.

2. fig1_showcase_composite — full-box showcase for CV/sim_0: top row truth
   (DMO + hydro DM/Gas/Stars), bottom row BIND composite (always rebuilt with
   build_bind_composite: shared paste, r200_factor 4, M200c >= 1e13 — the per-sim
   cached composite.npz may be legacy-paste, so it is never reused; DM panel =
   unblended pasted canvas ch 0 on black). Code copied from the section-1 cell
   of examples/_build_paper_nbs.py (2026-08-13 WORKLOG recipe).

CPU only. Writes pdf+png (dpi 300) to examples/paper_figures/.

Model selection is entirely via env vars (export before running):
  PAPER_SUITE_ROOT, PAPER_MODEL_SUBDIR, PAPER_MASS_DIR, PAPER_MODEL_TAG
"""
import os

_REQUIRED_ENV = ("PAPER_SUITE_ROOT", "PAPER_MODEL_SUBDIR", "PAPER_MASS_DIR",
                 "PAPER_MODEL_TAG")
_missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
if _missing:
    raise SystemExit(f"set env vars before running: {', '.join(_missing)}")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "paper_cache"))
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.cm import ScalarMappable
from scipy.stats import pearsonr

import paper_config as C

assert (C.CACHE_DIR / "field1p.npz").exists(), f"no field1p.npz in {C.CACHE_DIR}"
print(f"model tag {C.MODEL_TAG}: cache {C.CACHE_DIR}, suite {C.SUITE_ROOT} "
      f"({C.MODEL_SUBDIR}/{C.MASS_DIR})")

CACHE = C.CACHE_DIR
FIG_DIR = Path("/mnt/home/mlee1/vdm_bind2/examples/paper_figures")
FIG_DIR.mkdir(exist_ok=True)

# Paper style (identical to the notebook SETUP in _build_paper_nbs.py)
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "notebook"])
except Exception:
    pass

MASS_CH = C.MASS_CHANNELS
CH_DISPLAY = C.CH_DISPLAY
PARAM_LABELS = C.PARAM_LABELS
MIN_LOG_M200 = 13.0


def save_fig(fig, name, ext=("pdf", "png")):
    for e in ext:
        fig.savefig(FIG_DIR / f"{name}.{e}", dpi=300, bbox_inches="tight")
    print("  saved", name)


# ════════════════════════════════════════════════════════════════════════════
# Fig 7 — 1P field-level response butterfly (spine field1p.npz)
# ════════════════════════════════════════════════════════════════════════════
def fig7_field_response():
    f1 = dict(np.load(CACHE / "field1p.npz"))
    params = f1["params"]
    print(f"fig7: 1P params in field1p.npz = {list(params)}")
    nch = 3
    nrow = len(params)

    # Pass 1 — compute all log-ratio fields (builder recipe: eps = 1e-3 x the 99th
    # pctile of positive truth pixels, per param & channel; ratio in dex).
    fields, stats = {}, []
    for j in params:
        t_hi, t_lo = f1[f"p{j}_t_hi"], f1[f"p{j}_t_lo"]
        g_hi, g_lo = f1[f"p{j}_g_hi"], f1[f"p{j}_g_lo"]
        for c in range(nch):
            eps = 1e-3 * np.percentile(
                np.r_[t_hi[c][t_hi[c] > 0], t_lo[c][t_lo[c] > 0]]
                if (t_hi[c] > 0).any() else [1e-6], 99)
            tlr = np.log10((t_hi[c] + eps) / (t_lo[c] + eps))
            glr = np.log10((g_hi[c] + eps) / (g_lo[c] + eps))
            fields[(int(j), c)] = (tlr, glr)
            s_t, s_g = tlr.std(), glr.std()
            r = pearsonr(tlr.ravel(), glr.ravel())[0] if s_t > 1e-9 and s_g > 1e-9 else np.nan
            stats.append(dict(param=int(j), ch=MASS_CH[c], r=r,
                              A=(s_g / s_t if s_t > 1e-9 else np.nan),
                              t_std=s_t, g_std=s_g))

    # Shared symmetric per-channel color scale (matches the current paper layout:
    # one bottom colorbar per channel), from the pooled 99.5th |log-ratio| pctile.
    vch = []
    for c in range(nch):
        pool = np.concatenate([np.abs(np.r_[fields[(int(j), c)][0].ravel(),
                                            fields[(int(j), c)][1].ravel()]) for j in params])
        vch.append(float(np.quantile(pool, 0.995)) or 0.1)
    print("fig7: per-channel color range +/-", [f"{v:.3f}" for v in vch], "dex")

    fig, axes = plt.subplots(nrow, 2 * nch, figsize=(2.0 * 2 * nch, 2.0 * nrow),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.06})
    for ri, j in enumerate(params):
        for c in range(nch):
            tlr, glr = fields[(int(j), c)]
            v = vch[c]
            for ki, (fld, kind) in enumerate([(tlr, "Truth"), (glr, "BIND")]):
                ax = axes[ri, c * 2 + ki]
                ax.imshow(fld, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(f"{CH_DISPLAY[MASS_CH[c]]}\n{kind}", fontsize=9)
                if c == 0 and ki == 0:
                    ax.set_ylabel(PARAM_LABELS[int(j)], fontsize=9, rotation=0,
                                  ha="right", va="center", labelpad=8)
                if kind == "BIND":
                    s_t, s_g = tlr.std(), glr.std()
                    if s_t > 1e-9 and s_g > 1e-9:
                        ax.text(0.04, 0.96,
                                f"r={pearsonr(tlr.ravel(), glr.ravel())[0]:+.2f}\n"
                                f"A={s_g / s_t:.2f}",
                                transform=ax.transAxes, va="top", fontsize=6,
                                bbox=dict(fc="white", ec="none", alpha=0.7))
    for c in range(nch):
        sm = ScalarMappable(norm=Normalize(-vch[c], vch[c]), cmap="RdBu_r")
        cb = fig.colorbar(sm, ax=list(axes[:, c * 2:c * 2 + 2].ravel()),
                          location="bottom", fraction=0.025, pad=0.012, aspect=30)
        cb.set_label(rf"{CH_DISPLAY[MASS_CH[c]]}  $\log_{{10}}(\Sigma_{{\rm hi}}/\Sigma_{{\rm lo}})$ [dex]",
                     fontsize=8)
        cb.ax.tick_params(labelsize=7)
    save_fig(fig, "fig7_1p_field_response")
    plt.close(fig)
    return params, vch, stats


# ════════════════════════════════════════════════════════════════════════════
# Fig 1 — full-box showcase composite (spine CV/sim_0)
#   (verbatim recipe from the section-1 cell of examples/_build_paper_nbs.py)
# ════════════════════════════════════════════════════════════════════════════
def fig1_showcase():
    rec = C.discover_sims(("CV",))[0]
    assert rec["key"] == "CV/sim_0", rec["key"]
    fm = C.load_full_maps(rec)
    dmo, truth = fm["dmo_fullbox"], fm["truth_maps"]

    # Trained regime only: the BIND rows use halos with M200c >= 1e13 (MIN_LOG_M200).
    # The composite is ALWAYS rebuilt here with the standard shared-content paste
    # (r200_factor=4.0, paste_mode="shared") from generated_halos.npz — the per-sim
    # cached composite.npz may have been written with the legacy paste, so it is
    # never reused.
    from bind.inference.artifacts import load_halo_catalog, load_halo_cutouts
    from bind.inference.pipeline import build_bind_composite

    halos, _, _, _ = load_halo_catalog(rec["catalog"])
    masses = np.array([h["halo_mass"] for h in halos])
    keep = masses >= 10 ** MIN_LOG_M200
    n_pasted = int(keep.sum())
    print(f'{rec["key"]}: {n_pasted}/{len(halos)} halos with log10 M200c >= {MIN_LOG_M200}')
    idx = np.where(keep)[0]
    gen = C.load_generated(rec)[:, :C.N_MASS_CH]
    cutouts = load_halo_cutouts(rec["cutouts"])
    b = build_bind_composite(dmo, [halos[i] for i in idx], gen[idx],
                             [cutouts[i] for i in idx],
                             box_size=C.BOX_SIZE, npix=C.N_PIX_FULL,
                             patch_pix=C.PATCH_PIX, patch_mass_match=True,
                             taper_frac=0.15, r200_factor=4.0, paste_mode="shared")
    canvas, composite = b["hydro_canvas"], b["composite"]
    print("fig1: rebuilt composite with build_bind_composite "
          "(paste_mode=shared, r200_factor=4.0)")
    # canvas    = patches-on-black (taper-free)
    # composite = blended full-box map

    def showcase(bind_field, tag):
        fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(16, 8), sharex=True,
                                sharey=True, gridspec_kw={"wspace": 0.005, "hspace": 0.05})

        def _im(ax, img, title=None, vmax=None):
            pos = img[img > 0]
            vmax = vmax or np.quantile(pos, 0.999)
            vmin = max(np.quantile(pos, 0.05), 1e-8)
            ax.imshow(img, origin="lower", cmap="magma",
                      norm=SymLogNorm(linthresh=max(vmin, 1e-6), vmin=0, vmax=vmax, base=10),
                      extent=[0, 50, 0, 50])
            if title:
                ax.set_title(title)

        vmax = []
        for c in range(3):
            pos = np.concatenate([truth[c].ravel(), bind_field[c].ravel()])
            vmax.append(np.quantile(pos[pos > 0], 0.999))
        _im(axs[0, 0], dmo, "DMO (Nbody)")
        for c in range(3):
            _im(axs[0, c + 1], truth[c], CH_DISPLAY[MASS_CH[c]], vmax[c])
        _im(axs[1, 0], dmo)
        for c in range(3):
            _im(axs[1, c + 1], bind_field[c], vmax=vmax[c])

        for ax in axs[1]:
            ax.set_xlabel("X [Mpc/h]")
        axs[0, 0].set_ylabel("Y [Mpc/h]"); axs[1, 0].set_ylabel("Y [Mpc/h]")

        fig.text(0.01, 0.70, "Truth", va="center", ha="center", fontsize=20,
                 fontweight="bold", rotation="vertical")
        fig.text(0.01, 0.30, "BIND", va="center", ha="center", fontsize=20,
                 fontweight="bold", rotation="vertical")
        fig.subplots_adjust(left=0.06)
        plt.tight_layout()
        save_fig(fig, tag)
        plt.close(fig)

    # Composite variant, DM shown *unblended*: the DM panel is the raw pasted patches
    # (`canvas`), not composite[0] = (1-alpha)*DMO + alpha*canvas[0], so it shows what
    # the model generated instead of the DMO field it is pasted into. Gas/Stars are
    # unaffected by that fill -- their composite is alpha*canvas with a zero background.
    composite_dm_raw = np.stack([canvas[0], composite[1], composite[2]])
    showcase(composite_dm_raw, "fig1_showcase_composite")
    return n_pasted, len(halos)


if __name__ == "__main__":
    params, vch, stats = fig7_field_response()
    print("\nfig7 truth-vs-BIND per-panel stats (r = Pearson of log-ratio maps, "
          "A = std ratio BIND/Truth):")
    for s in stats:
        print(f"  p{s['param']:<3d} {s['ch']:<9s} r={s['r']:+.3f}  A={s['A']:.3f}  "
              f"sigma_truth={s['t_std']:.3f}  sigma_bind={s['g_std']:.3f} dex")
    n_pasted, n_tot = fig1_showcase()
    print(f"\nfig1: {n_pasted}/{n_tot} halos pasted (M200c >= 1e13), 2x4 panels")
