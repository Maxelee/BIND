"""Resolution gate: paint a coarser TNG300-Dark run, compare halo-by-halo.

Decides whether BIND painting survives lower-resolution DMO conditioning
(docs/wl_tsz_plan.md §3.6) before any big-box run.  TNG300-2-Dark (8x) and
TNG300-3-Dark (64x) share initial phases with TNG300-1-Dark, so matched halos
are the *same objects* — compare BIND(test-res) vs BIND(fiducial-res) per halo.
The physics reference stays TNG300-1 hydro (low-res hydro is itself
resolution-shifted).

Produce the test-res composites with the existing staged scripts (sbatch env
overrides; same LC_SEED -> identical lightcone transforms -> matchable centers):

    ROOT=/mnt/home/mlee1/ceph/bind_resgate/n625
    SIM=/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output
    j1=$(SIM_ROOT=$SIM OUTPUT_ROOT=$ROOT sbatch --parsable --array=0-0 run_lightcone_project.sh)
    j2=$(OUTPUT_ROOT=$ROOT sbatch --parsable --dependency=afterok:$j1 --array=0-0 run_lightcone_generate.sh)
    OUTPUT_ROOT=$ROOT sbatch --dependency=afterok:$j2 --array=0-0 run_lightcone_recomposite.sh
    python examples/resolution_gate.py --test_root $ROOT     # (n1250 likewise)

Pass criterion (plan §3.6): median biases of M_gas/Y within a few % for
M > 10^13.5 — the high-nu drivers.  Expected failure order: thermo (y) before
mass; low-mass halos before clusters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PIX_MPCH = 6.25 / 128.0
R500_OVER_R200 = 0.659


def per_halo(snap_dir: Path):
    """Per-slab arrays: centers [Mpc/h], FoF mass, r200, M_gas/Y/f_gas in r500."""
    out = []
    for p in sorted(snap_dir.glob("composite_slab*.npz")):
        d = np.load(p)
        if "generated_patches" not in d.files or int(d["n_halos"]) == 0:
            out.append(None)
            continue
        gen = d["generated_patches"]
        thermo = d["thermo_patches"] if "thermo_patches" in d.files else None
        cen = np.asarray(d["halo_centers"], dtype=float)
        box = float(d["box_size"]) if "box_size" in d.files else 205.0
        if cen.max() > 1.05 * box:                       # pixel coords -> Mpc/h
            npix = (int(d["npix"]) if "npix" in d.files
                    else (d["composite"].shape[0] if "composite" in d.files else 4198))
            cen = cen * box / npix
        P = gen.shape[-1]
        yy, xx = np.mgrid[0:P, 0:P]
        rr = np.hypot(xx - P // 2, yy - P // 2) * PIX_MPCH
        n = gen.shape[0]
        Mg = np.zeros(n); Mt = np.zeros(n); Y = np.full(n, np.nan)
        for h in range(n):
            ap = rr <= R500_OVER_R200 * float(d["halo_r200"][h])
            Mg[h] = gen[h, 1][ap].sum()
            Mt[h] = gen[h, 0][ap].sum() + Mg[h] + gen[h, 2][ap].sum()
            if thermo is not None:
                Y[h] = thermo[h, 0][ap].sum() * PIX_MPCH ** 2
        out.append(dict(cen=cen, M=np.asarray(d["halo_masses"], dtype=float),
                        r200=np.asarray(d["halo_r200"], dtype=float),
                        Mg=Mg, fg=Mg / np.where(Mt > 0, Mt, 1.0), Y=Y))
    return out


def match(base, test, tol: float):
    """KD-tree match per slab; returns paired index arrays (base_slab, b_idx, t_idx)."""
    from scipy.spatial import cKDTree
    pairs = []
    for s, (b, t) in enumerate(zip(base, test)):
        if b is None or t is None:
            continue
        tree = cKDTree(t["cen"])
        dist, j = tree.query(b["cen"], distance_upper_bound=tol)
        ok = np.isfinite(dist)
        pairs.append((s, np.where(ok)[0], j[ok]))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_root", type=Path,
                   default=Path("/mnt/home/mlee1/ceph/bind_lightcone_tng"))
    p.add_argument("--test_root", type=Path, required=True)
    p.add_argument("--snap", type=int, default=96)
    p.add_argument("--tol", type=float, default=1.5, help="match radius [Mpc/h]")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    base = per_halo(args.base_root / f"snap_{args.snap:03d}")
    test = per_halo(args.test_root / f"snap_{args.snap:03d}")
    pairs = match(base, test, args.tol)
    nb = sum(len(b["M"]) for b in base if b is not None)
    nm = sum(len(i) for _, i, _ in pairs)
    print(f"[gate] snap {args.snap}: matched {nm}/{nb} base halos (tol {args.tol} Mpc/h)")

    M0, dM, dMg, dY, dfg = [], [], [], [], []
    for s, bi, ti in pairs:
        b, t = base[s], test[s]
        M0.append(b["M"][bi])
        dM.append(np.log10(t["M"][ti] / b["M"][bi]))
        dMg.append(np.log10(t["Mg"][ti] / np.where(b["Mg"][bi] > 0, b["Mg"][bi], np.nan)))
        dY.append(np.log10(t["Y"][ti] / b["Y"][bi]))
        dfg.append(t["fg"][ti] - b["fg"][bi])
    M0 = np.concatenate(M0); dM = np.concatenate(dM); dMg = np.concatenate(dMg)
    dY = np.concatenate(dY); dfg = np.concatenate(dfg)

    mb = np.logspace(13, 15, 9)
    print(f"\n{'M200c(base)':>12s} {'N':>6s} {'dlogM_FoF':>10s} {'dlogM_gas':>16s} "
          f"{'dlogY':>16s} {'d f_gas':>9s}")
    for lo, hi in zip(mb[:-1], mb[1:]):
        m = (M0 >= lo) & (M0 < hi)
        if m.sum() < 10:
            continue
        q = lambda x: np.nanpercentile(x[m], [16, 50, 84])
        a_, g, y = q(dM), q(dMg), q(dY)
        print(f"{np.sqrt(lo*hi):12.2e} {m.sum():6d} {a_[1]:10.3f} "
              f"{g[1]:7.3f} [{g[0]:+.2f},{g[2]:+.2f}] "
              f"{y[1]:7.3f} [{y[0]:+.2f},{y[2]:+.2f}] {np.nanmedian(dfg[m]):9.4f}")

    hi_m = M0 >= 10 ** 13.5
    bias_g = 10 ** np.nanmedian(dMg[hi_m]) - 1
    bias_y = 10 ** np.nanmedian(dY[hi_m]) - 1
    verdict = "PASS" if max(abs(bias_g), abs(bias_y)) < 0.05 else "FAIL"
    print(f"\n[gate] M>10^13.5 median bias: M_gas {bias_g:+.1%}, Y {bias_y:+.1%}  -> {verdict}"
          f"  (criterion: |bias| < 5%; see docs/wl_tsz_plan.md §3.6)")

    out = args.out or args.test_root
    np.savez(Path(out) / f"resolution_gate_snap{args.snap:03d}.npz",
             M_base=M0, dlogM=dM, dlogMgas=dMg, dlogY=dY, dfgas=dfg)
    print(f"[gate] wrote {out}/resolution_gate_snap{args.snap:03d}.npz")


if __name__ == "__main__":
    main()
