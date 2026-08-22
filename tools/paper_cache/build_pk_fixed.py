#!/usr/bin/env python3
"""Corrected full-box P(k) cache: shared-paste composites + true covering paint.

Companion to ``build_metric.py --metric pk`` (which reads the suite's saved
``composite.npz``, historically built with the legacy *averaging* paste that
loses high-k power wherever apertures overlap — see docs/circular_aperture.md,
"Shared-content overlap handling"). This builder recomputes per-sim total-matter
P(k) with the fixed paste, for three halo populations:

- ``fixed`` metric (CPU): ≥1e13 and ≥1e12 composites rebuilt from the suite's
  cached ``generated_halos.npz`` with ``build_bind_composite(paste_mode="shared")``,
  plus ``hr_ge13`` — the hydro-replace control (truth patches of the same ≥1e13
  halos, same shared paste), which the paper's Fig 5 uses instead of the legacy
  all-catalog-halo control from ``pk.npz``. Partials that predate ``hr_ge13``
  are recomputed automatically.
- ``cover`` metric (GPU): the full covering paint — ALL FoF halos ≥1e10, fresh
  generation per set-cover box, generated-closure (Rc) apertures with per-sim
  f_cos = Ω_b/Ω_m, covering-paint gas background. FoF group tables are read from
  COVER_FOF_ROOT with the CAMELS 1P N-body routing (astro-param 1P variations
  share the fiducial DMO).

``--reduce`` collates partials + the hydro-replace / legacy-BIND stacks from the
existing ``pk.npz`` into ``CACHE_DIR/pk_fixed.npz`` for the load-only notebook.

Env overrides (beyond paper_config's): PAPER_COVER_CKPT / PAPER_COVER_NORM
(default: the fm_thermo epoch064 EMA weights used by the dev suite),
COVER_FOF_ROOT (default CAMELS IllustrisTNG_DM L50n512).

Examples
--------
    python build_pk_fixed.py --metric fixed --pool 6
    python build_pk_fixed.py --metric cover --chunk 0 --n-chunks 3   # GPU
    python build_pk_fixed.py --metric fixed --reduce                 # writes pk_fixed.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_config as C  # noqa: E402
from bind.inference.artifacts import load_halo_catalog, load_halo_cutouts  # noqa: E402
from bind.inference.config import _resolve_1p_nbody_sim  # noqa: E402
from bind.inference.pipeline import build_bind_composite, circular_taper_weight  # noqa: E402
from bind.metrics import power_spectrum_pylians_2d  # noqa: E402

BOX, NPIX, PP = C.BOX_SIZE, C.N_PIX_FULL, C.PATCH_PIX
PPM = NPIX / BOX
MPP = BOX / NPIX
RF, TAP = 4.0, 0.15
COVER_FLOOR = 1e10
COVER_FOF_ROOT = Path(os.environ.get(
    "COVER_FOF_ROOT", "/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512"))
COVER_CKPT = os.environ.get(
    "PAPER_COVER_CKPT",
    "/mnt/home/mlee1/ceph/fm_runs/fm_thermo/checkpoints/kept/keep_epoch064_ema.ckpt")
COVER_NORM = os.environ.get(
    "PAPER_COVER_NORM", "/mnt/home/mlee1/ceph/fm_runs/fm_thermo/norm_stats.npz")

_yy, _xx = np.mgrid[0:PP, 0:PP]
RRi = np.round(np.hypot(_xx - PP // 2, _yy - PP // 2)).astype(int).ravel()
NRi = RRi.max() + 1

PASTE_KW = dict(box_size=BOX, npix=NPIX, patch_pix=PP,
                patch_mass_match=True, taper_frac=TAP, r200_factor=RF,
                paste_mode="shared")


def pk2d(field):
    k, p, _ = power_spectrum_pylians_2d(field, box_size=BOX, MAS="None", threads=1)
    return k, p


# ════════════════════════════════════════════════════════════════════════════
# fixed — shared-paste ≥1e13 / ≥1e12 composites from cached generations (CPU)
# ════════════════════════════════════════════════════════════════════════════
def compute_fixed(rec):
    fm = C.load_full_maps(rec)
    dmo = fm["dmo_fullbox"]
    halos, _, _, _ = load_halo_catalog(rec["catalog"])
    cutouts = load_halo_cutouts(rec["cutouts"])
    gen = C.load_generated(rec)[:, : C.N_MASS_CH]
    masses = np.array([h["halo_mass"] for h in halos])

    k, p_truth = pk2d(fm["truth_maps"].sum(0))
    _, p_dmo = pk2d(dmo)
    out = dict(k=k, truth=p_truth, dmo=p_dmo, n_ge12=len(halos),
               n_ge13=int((masses >= 1e13).sum()))

    for tag, mask in [("ge12", masses >= 1e12), ("ge13", masses >= 1e13)]:
        if not mask.any():
            out[tag] = np.full_like(p_truth, np.nan)
            continue
        idx = np.where(mask)[0]
        b = build_bind_composite(dmo, [halos[i] for i in idx], gen[idx],
                                 [cutouts[i] for i in idx], **PASTE_KW)
        _, out[tag] = pk2d(b["composite"].sum(0))

    # hydro-replace control at the trained cut: truth patches of the SAME >=1e13
    # halos, same shared paste (isolates model error from paste/aperture error)
    mask = masses >= 1e13
    if mask.any():
        idx = np.where(mask)[0]
        centers_pix = C.centers_to_pixels(
            np.array([halos[i]["halo_center"] for i in idx]))
        truth_patches = C.extract_truth_mass_patches(fm, centers_pix)
        b = build_bind_composite(dmo, [halos[i] for i in idx], truth_patches,
                                 [cutouts[i] for i in idx], **PASTE_KW)
        _, out["hr_ge13"] = pk2d(b["composite"].sum(0))
    else:
        out["hr_ge13"] = np.full_like(p_truth, np.nan)
    return out


# ════════════════════════════════════════════════════════════════════════════
# cover — true covering paint, ALL FoF halos ≥1e10, fresh generation (GPU)
# ════════════════════════════════════════════════════════════════════════════
def _fof_path(rec) -> Path:
    sim_id = rec["sim_id"]
    tab = f"fof_subhalo_tab_{C.SNAP.split('_')[-1]}.hdf5"
    if rec["suite"] == "CV":
        return COVER_FOF_ROOT / f"CV/CV_{sim_id.split('_')[-1]}" / tab
    if rec["suite"] == "Test":
        return COVER_FOF_ROOT / f"SB35/{sim_id.removeprefix('sim_')}" / tab
    # 1P — astro-param variations share the fiducial DMO. _resolve_1p_nbody_sim
    # keeps 'n1'/'n2' names verbatim, so mirror build_1p_specs and fall back to
    # the fiducial when the sim has no DM-only counterpart of its own.
    p = COVER_FOF_ROOT / f"1P/{_resolve_1p_nbody_sim(sim_id)}" / tab
    if not p.exists():
        p = COVER_FOF_ROOT / "1P/1P_p1_0" / tab
    return p


def _plan_cover(px, py, masses, r200):
    """Greedy set cover in descending mass; members must fit inside the box."""
    n = len(masses)
    ap = np.minimum(RF * r200 * PPM, PP // 2 - 2)
    order = np.argsort(-masses)
    covered = np.zeros(n, bool)
    host = np.full(n, -1)
    for oi in order:
        if covered[oi]:
            continue
        host[oi] = oi
        covered[oi] = True
        dx = (px - px[oi] + NPIX // 2) % NPIX - NPIX // 2
        dy = (py - py[oi] + NPIX // 2) % NPIX - NPIX // 2
        fits = (~covered) & (np.abs(dx) + ap < PP // 2 - 1) & (np.abs(dy) + ap < PP // 2 - 1)
        host[fits] = oi
        covered[fits] = True
    return host, ap


def _closure_radius(patch, cap, fcos):
    bar = (patch[1] + patch[2]).ravel()
    tot = patch.sum(0).ravel()
    fb = np.cumsum(np.bincount(RRi, weights=bar, minlength=NRi)) / np.maximum(
        np.cumsum(np.bincount(RRi, weights=tot, minlength=NRi)), 1e-30)
    hit = np.where(fb >= fcos)[0]
    return float(np.clip(hit[0] if len(hit) else NRi - 1, 2, cap))


def compute_cover(rec, model=None):
    import h5py
    from scipy.ndimage import gaussian_filter
    from bind.inference.pipeline import extract_multiscale

    fof = _fof_path(rec)
    if not fof.exists():
        print(f"  [cover] no FoF for {rec['key']}: {fof}")
        return None
    fm = C.load_full_maps(rec)
    dmo = fm["dmo_fullbox"]
    cat = C.load_catalog(rec)
    params = C.sim_params(cat).astype(np.float32)
    # Per-sim cosmic baryon fraction Ω_b/Ω_m (both vary across SB35 and the
    # cosmology 1P sims: realized range ≈ 0.06–0.68). Clip only guards against
    # a pathological params read.
    fcos = float(np.clip(params[6] / params[0], 0.02, 0.75))

    with h5py.File(fof, "r") as h:
        M200 = h["Group/Group_M_Crit200"][:] * 1e10
        POS = h["Group/GroupPos"][:] / 1e3
        R200 = h["Group/Group_R_Crit200"][:] / 1e3
    sel = np.where(M200 > COVER_FLOOR)[0]
    masses, r200 = M200[sel], R200[sel]
    px = (POS[sel, 0] * PPM).astype(int) % NPIX
    py = (POS[sel, 1] * PPM).astype(int) % NPIX

    host, _ = _plan_cover(px, py, masses, r200)
    covers = np.unique(host)
    cover_pos = {int(c): gi for gi, c in enumerate(covers)}

    cuts = []
    for c in covers:
        cond, ls = extract_multiscale(dmo, px[c], py[c], PP, MPP)
        cuts.append({"condition": cond, "large_scale": ls})
    G = model.generate(cuts, params, n_steps=20, batch_size=32, use_amp=True,
                       progress=False)[:, :3].astype(np.float32)

    # paint with Rc closure apertures + covering gas background
    gasbg = (fcos * gaussian_filter(dmo, 2.0, mode="wrap")).astype(np.float32)
    canvas = np.zeros((3, NPIX, NPIX), np.float32)
    wacc = np.zeros((NPIX, NPIX), np.float32)
    ar = np.arange(PP)
    wc = {}
    for j in range(len(masses)):
        h = int(host[j])
        dx = (px[j] - px[h] + NPIX // 2) % NPIX - NPIX // 2
        dy = (py[j] - py[h] + NPIX // 2) % NPIX - NPIX // 2
        patch = np.roll(G[cover_pos[h]], shift=(-dx, -dy), axis=(1, 2))
        cap = max(3, PP // 2 - 2 - max(abs(dx), abs(dy)))
        a = _closure_radius(patch, cap, fcos)
        key = round(a * 2) / 2
        if key not in wc:
            wc[key] = circular_taper_weight(PP, r_pix=key, taper_frac=TAP).astype(np.float32)
        w = wc[key]
        gg = np.ix_((px[j] - PP // 2 + ar) % NPIX, (py[j] - PP // 2 + ar) % NPIX)
        dw = dmo[gg]
        patch = patch * (float((dw * w).sum()) / (float((patch.sum(0) * w).sum()) + 1e-30))
        for ch in range(3):
            canvas[ch][gg] += patch[ch] * w
        wacc[gg] += w
    canvas /= np.where(wacc > 0, wacc, 1.0)[None]
    al = np.clip(wacc, 0, 1)
    comp = np.empty((3, NPIX, NPIX), np.float32)
    comp[0] = (1 - al) * (1 - fcos) * dmo + al * canvas[0]
    comp[1] = (1 - al) * gasbg + al * canvas[1]
    comp[2] = al * canvas[2]
    comp *= dmo.sum() / (comp.sum() + 1e-30)

    k, p_cover = pk2d(comp.sum(0))
    _, p_truth = pk2d(fm["truth_maps"].sum(0))
    _, p_dmo = pk2d(dmo)
    return dict(k=k, cover=p_cover, truth=p_truth, dmo=p_dmo,
                n_halos=len(masses), n_gens=len(covers), fcos=fcos)


# ════════════════════════════════════════════════════════════════════════════
# driver (partials + reduce), mirroring build_metric.py
# ════════════════════════════════════════════════════════════════════════════
def _partial_path(metric, rec):
    return C.PARTIAL_DIR / f"pk_{metric}" / f"{rec['suite']}__{rec['sim_id']}.npz"


def _build_one_fixed(key):
    rec = C.resolve_record(key)
    path = _partial_path("fixed", rec)
    if path.exists():
        with np.load(path) as d:
            if "hr_ge13" in d.files:
                return f"skip {key}"
        # partial predates the hr_ge13 control -> recompute
    try:
        d = compute_fixed(rec)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **d)
        return f"ok {key}"
    except Exception as exc:
        return f"FAIL {key}: {exc}\n{traceback.format_exc()}"


def reduce_all():
    out = {"k": None}
    for metric, keys in [("fixed", ("truth", "dmo", "ge12", "ge13", "hr_ge13")),
                         ("cover", ("truth", "dmo", "cover"))]:
        pdir = C.PARTIAL_DIR / f"pk_{metric}"
        per_suite: dict[str, dict[str, list]] = {}
        sims: dict[str, list] = {}
        for p in sorted(pdir.glob("*.npz")) if pdir.exists() else []:
            suite, sim = p.stem.split("__", 1)
            d = np.load(p)
            if out["k"] is None:
                out["k"] = d["k"]
            dd = per_suite.setdefault(suite, {kk: [] for kk in keys})
            for kk in keys:
                # tolerate partials that predate a key (e.g. hr_ge13)
                dd[kk].append(d[kk] if kk in d.files
                              else np.full_like(d["truth"], np.nan))
            sims.setdefault(suite, []).append(sim)
        for suite, dd in per_suite.items():
            for kk, lst in dd.items():
                out[f"{suite}_{metric}_{kk}"] = np.stack(lst)
            out[f"{suite}_{metric}_sims"] = np.array(sims[suite])
    # hydro-replace + legacy avg-paste BIND stacks from the existing pk metric
    pk_path = C.CACHE_DIR / "pk.npz"
    if pk_path.exists():
        pk = np.load(pk_path)
        for s in C.SUITES:
            for kk in ("truth", "hydro_replace", "bind"):
                if f"{s}_{kk}" in pk:
                    out[f"{s}_pkfile_{kk}"] = pk[f"{s}_{kk}"]
    path = C.CACHE_DIR / "pk_fixed.npz"
    np.savez_compressed(path, **{k: v for k, v in out.items() if v is not None})
    print(f"wrote {path}")
    for k in sorted(out):
        if k.endswith("_sims"):
            print(f"  {k}: {len(out[k])} sims")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["fixed", "cover"], default="fixed")
    ap.add_argument("--sim_ids", default=None, help="Comma list e.g. CV/sim_0,1P/1P_p3_n2")
    ap.add_argument("--suite", default="all", choices=["all", *C.SUITES])
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--reduce", action="store_true")
    args = ap.parse_args()

    if args.reduce:
        reduce_all()
        return

    if args.sim_ids:
        keys = [s.strip() for s in args.sim_ids.split(",") if s.strip()]
        keys = keys[args.chunk::args.n_chunks] if args.n_chunks > 1 else keys
    else:
        suites = C.SUITES if args.suite == "all" else (args.suite,)
        recs = C.discover_sims(suites)
        if args.n_chunks > 1:
            recs = recs[args.chunk::args.n_chunks]
        keys = [r["key"] for r in recs]
    print(f"[{args.metric}] {len(keys)} sims (chunk {args.chunk}/{args.n_chunks})")

    if args.metric == "fixed":
        if args.pool > 1:
            import multiprocessing as mp
            with mp.Pool(args.pool) as pool:
                for msg in pool.imap_unordered(_build_one_fixed, keys):
                    print(" ", msg, flush=True)
        else:
            for key in keys:
                print(" ", _build_one_fixed(key), flush=True)
    else:  # cover: one model instance, sequential over this chunk
        from bind.inference.paint import Model
        model = Model.from_files(COVER_CKPT, COVER_NORM, device="cuda")
        for key in keys:
            rec = C.resolve_record(key)
            path = _partial_path("cover", rec)
            if path.exists():
                print(f"  skip {key}", flush=True)
                continue
            try:
                d = compute_cover(rec, model=model)
                if d is None:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **d)
                print(f"  ok {key} (n={d['n_halos']}, gens={d['n_gens']})", flush=True)
            except Exception as exc:
                print(f"  FAIL {key}: {exc}", flush=True)


if __name__ == "__main__":
    main()
