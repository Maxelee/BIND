#!/usr/bin/env python3
"""Per-halo posterior ensembles for BIND (referee point I-4).

For each selected halo we hold the conditioning FIXED (same DMO cutout, same
large-scale context, same 35-dim parameter vector) and draw ``--n_samples``
independent flow-matching realizations.  From every draw we measure the same
summaries the paper quotes -- M_DM, M_gas, M_star inside R200c and the mean
surface density in four r/R200c annuli -- and we store the matched truth value.

That turns every halo into a one-dimensional posterior per summary, which is
what a PIT / rank histogram, a coverage curve and a sigma_post/sigma_pop ratio
need.  The paper's existing `fig2_pit_calibration` is a *marginal* PIT: it ranks
the truth of halo i against the population of single-draw generated values of
*other* halos in the same M200c bin, and therefore cannot say anything about the
per-halo conditional distribution.

Reads the cached suite-eval products (halo_cutouts.npz / halo_catalog.npz /
full_maps.npz) so no particle I/O or projection is needed -- only the sampler
runs on the GPU.

    python build_posterior_ensemble.py --n_halos 200 --n_samples 100

Writes an .npz with, per halo, the truth summary vector and the (n_samples,)
draws for each summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/home/mlee1/vdm_bind2")
sys.path.insert(0, str(REPO / "tools" / "paper_cache"))
import paper_config as C  # noqa: E402

# The model behind the paper's fm_thermo cache (mass_table.pkl), read from
# <sim>/<mass_dir>/<model>/summary.json:run_config.
DEF_SUITE_ROOT = Path("/mnt/home/mlee1/ceph/fm_lowmass")
DEF_MASS_DIR = "mass_threshold_1p000e12"
DEF_CKPT = Path("/mnt/home/mlee1/ceph/fm_runs/fm_thermo/checkpoints/kept/keep_epoch064_ema.ckpt")
DEF_NORM = Path("/mnt/home/mlee1/ceph/fm_runs/fm_thermo/norm_stats.npz")

CHS = C.MASS_CHANNELS                       # DM_hydro, Gas, Stars
ANNULI = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0)]   # r / R200c


def summary_vector(patches: np.ndarray, r_pix: float) -> np.ndarray:
    """(B, 3, H, W) physical maps -> (B, n_summ) summaries.

    Layout: [M_ch(<R200c) for ch in CHS] + [Sigma_ch(annulus) for ch, annulus]
    + [M_bar(<R200c)]  (M_bar = Gas + Stars).
    """
    rr = C.RR_PIX_PATCH / r_pix                    # r / R200c per pixel
    p = np.maximum(patches[:, : C.N_MASS_CH], 0.0)
    out = [p[:, c][:, rr < 1.0].sum(1) for c in range(C.N_MASS_CH)]
    for c in range(C.N_MASS_CH):
        for lo, hi in ANNULI:
            m = (rr >= lo) & (rr < hi)
            out.append(p[:, c][:, m].mean(1))
    out.append(out[1] + out[2])                    # baryons inside R200c
    return np.stack(out, axis=1)


def summary_names() -> list[str]:
    names = [f"M_{ch}_r200" for ch in CHS]
    names += [f"Sigma_{ch}_{lo:g}-{hi:g}R200" for ch in CHS for lo, hi in ANNULI]
    names += ["M_bar_r200"]
    return names


def pick_halos(rng, suite_root: Path, mass_dir: str, per_suite: dict,
               log_m_min: float, n_per_sim: int) -> list[dict]:
    """Stratified halo selection: n_per_sim halos spread over log M200c in each
    of the requested sims, for each suite."""
    sel = []
    for suite, n_sims in per_suite.items():
        if n_sims <= 0:
            continue
        root = suite_root / suite
        if not root.exists():
            print(f"[sel] WARN {root} missing"); continue
        sims = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name)
        step = max(1, len(sims) // n_sims)
        sims = sims[::step][:n_sims]
        for sd in sims:
            cat_p = sd / C.SNAP / mass_dir / "halo_catalog.npz"
            if not cat_p.exists():
                continue
            cat = np.load(cat_p)
            logm = np.log10(cat["masses"])
            ok = np.where(logm >= log_m_min)[0]
            if ok.size == 0:
                continue
            order = ok[np.argsort(logm[ok])]
            # spread the picks uniformly in rank over the sim's mass range
            idx = np.unique(np.linspace(0, order.size - 1, min(n_per_sim, order.size)).astype(int))
            for i in order[idx]:
                sel.append(dict(suite=suite, sim_id=sd.name, halo=int(i),
                                log_m200c=float(logm[i]), sim_dir=str(sd)))
    rng.shuffle(sel)
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--n_halos", type=int, default=200)
    ap.add_argument("--halo_slice", default=None,
                    help="A:B slice of the (deterministic, seed-fixed) halo list, "
                         "for splitting one run across cluster jobs; merge the "
                         "per-slice npz files with merge_ensemble_slices.py")
    ap.add_argument("--n_per_sim", type=int, default=8)
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--log_m_min", type=float, default=13.0)
    ap.add_argument("--suite_root", type=Path, default=DEF_SUITE_ROOT)
    ap.add_argument("--mass_dir", default=DEF_MASS_DIR)
    ap.add_argument("--checkpoint", type=Path, default=DEF_CKPT)
    ap.add_argument("--norm_stats", type=Path, default=DEF_NORM)
    ap.add_argument("--n_cv", type=int, default=9)
    ap.add_argument("--n_1p", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sel = pick_halos(rng, args.suite_root, args.mass_dir,
                     {"CV": args.n_cv, "1P": args.n_1p, "Test": args.n_test},
                     args.log_m_min, args.n_per_sim)[: args.n_halos]
    if args.halo_slice:
        a, b = (int(x) for x in args.halo_slice.split(":"))
        sel = sel[a:b]
        print(f"[slice] halos {a}:{b} of {args.n_halos}")
    print(f"[sel] {len(sel)} halos; "
          f"logM {min(s['log_m200c'] for s in sel):.2f}-{max(s['log_m200c'] for s in sel):.2f}")

    from bind.inference.paint import Model
    model = Model.from_files(args.checkpoint, args.norm_stats, device="auto")
    print(f"[model] {model}")

    names = summary_names()
    n_s = len(names)
    N = args.n_samples

    truth_all = np.full((len(sel), n_s), np.nan, np.float64)
    draws_all = np.full((len(sel), N, n_s), np.nan, np.float64)
    meta = {k: [] for k in ("suite", "sim_id", "halo", "log_m200c", "r200_pix")}

    # group by sim so each suite product is opened once
    by_sim: dict[str, list[int]] = {}
    for i, s in enumerate(sel):
        by_sim.setdefault(f"{s['suite']}|{s['sim_id']}", []).append(i)

    t0 = time.time()
    done = 0
    for key, idxs in by_sim.items():
        suite, sim_id = key.split("|")
        sd = args.suite_root / suite / sim_id / C.SNAP
        cat = np.load(sd / args.mass_dir / "halo_catalog.npz")
        cuts = np.load(sd / args.mass_dir / "halo_cutouts.npz")
        cond_all = cuts["condition"]        # read once (NpzFile views leak memory)
        ls_all = cuts["large_scale"]
        full = np.load(sd / "full_maps.npz")["truth_maps"]
        centers = C.centers_to_pixels(cat["centers"])
        r_pix_all = C.r200_pix_patch(cat)
        params = np.asarray(cat["params"], np.float64)

        for i in idxs:
            h = sel[i]["halo"]
            r_pix = float(r_pix_all[h])
            truth = np.stack([C.extract_patch(full[c], *centers[h]) for c in range(C.N_MASS_CH)])
            truth_all[i] = summary_vector(truth[None], r_pix)[0]

            cut = {"condition": cond_all[h], "large_scale": ls_all[h]}
            gen = model.generate([cut] * N, params[h], n_steps=args.n_steps,
                                 batch_size=args.batch_size, progress=False)
            draws_all[i] = summary_vector(gen, r_pix)

            meta["suite"].append(suite); meta["sim_id"].append(sim_id)
            meta["halo"].append(h); meta["log_m200c"].append(sel[i]["log_m200c"])
            meta["r200_pix"].append(r_pix)
            done += 1
            if done % 10 == 0 or done == len(sel):
                el = time.time() - t0
                print(f"[gen] {done}/{len(sel)}  {el:.1f}s  "
                      f"({el/done:.2f}s/halo, {el/done/N*1e3:.1f}ms/draw)", flush=True)
        del cuts, cond_all, ls_all, full

    # re-order the metadata to the row order actually written
    order = np.argsort([i for key, idxs in by_sim.items() for i in idxs])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        truth=truth_all, draws=draws_all, names=np.array(names),
        suite=np.array([sel[i]["suite"] for i in range(len(sel))]),
        sim_id=np.array([sel[i]["sim_id"] for i in range(len(sel))]),
        halo=np.array([sel[i]["halo"] for i in range(len(sel))]),
        log_m200c=np.array([sel[i]["log_m200c"] for i in range(len(sel))]),
        r200_pix=np.array([np.nan] * len(sel)),
        config=json.dumps({k: str(v) for k, v in vars(args).items()}),
    )
    print(f"[out] wrote {args.out}  ({len(sel)} halos x {N} draws x {n_s} summaries)")
    print(f"[out] total wall {time.time() - t0:.1f}s")
    _ = order


if __name__ == "__main__":
    main()
