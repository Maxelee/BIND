"""Rebuild the fid/truth snap-096 halo-atlas caches with 200c thermodynamics added.

The atlas generator (examples/halo_atlas.py on analysis/sobol-sb35) computes the
thermo quantities (Y, mass-weighted T/K/Pe) only inside the 500c aperture; fig 3's
2026-08-05 single-aperture edit needs Y_200c. The aperture loop already runs over
both radii, so this script re-runs the identical reduction with the thermo block
evaluated at BOTH tags, adding Y_200c / T_mw_200c / K_mw_200c / Pe_mw_200c.

Safety: before replacing an existing cache file, every key it already carries must
be reproduced by the re-reduction bit-for-bit at float64 (max |rel diff| == 0) —
if not, the script refuses and leaves the original untouched (refuse-don't-guess).
Originals are kept as <name>.pre200c.npz.

    python papers/01_pipeline/build_atlas_200c.py            # fid + truth, snap 096
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
OUT = SCI / "halo_atlas"
SRC = {"fid": CEPH / "bind_lightcone_tng", "truth": SCI / "runs/truth/run_0000"}
SNAP = 96

PIX = 6.25 / 128.0
PIX_AREA = PIX * PIX
R500_FAC = 0.659
BG_ANN = (2.5, 3.0)


def reduce_file(path: Path) -> dict | None:
    """examples/halo_atlas.py::reduce_file with the thermo block at BOTH apertures."""
    d = np.load(path)
    if "generated_patches" not in d.files or int(d["n_halos"]) == 0:
        return None
    gen = d["generated_patches"].astype(np.float64)
    thr = d["thermo_patches"].astype(np.float64) if "thermo_patches" in d.files else None
    M = np.asarray(d["halo_masses"], float)
    r200 = np.asarray(d["halo_r200"], float)
    n, _, P, _ = gen.shape
    cen = P // 2
    yy, xx = np.mgrid[0:P, 0:P]
    rr = np.hypot(xx - cen, yy - cen) * PIX
    ann = (rr >= BG_ANN[0]) & (rr < BG_ANN[1])
    tot_pix = gen.sum(1)
    sig_tot_bg = tot_pix[:, ann].mean(1)
    sig_gas_bg = gen[:, 1][:, ann].mean(1)
    out = {"M_fof": M, "r200": r200,
           "sig_tot_bg": sig_tot_bg, "sig_gas_bg": sig_gas_bg}
    for tag, rad in (("500c", R500_FAC * r200), ("200c", r200)):
        m = rr[None] <= rad[:, None, None]
        npix = m.sum((1, 2))
        out[f"m_dm_{tag}"] = (gen[:, 0] * m).sum((1, 2))
        out[f"m_gas_{tag}"] = (gen[:, 1] * m).sum((1, 2))
        out[f"m_star_{tag}"] = (gen[:, 2] * m).sum((1, 2))
        out[f"m_tot_{tag}_bg"] = (out[f"m_dm_{tag}"] + out[f"m_gas_{tag}"]
                                  + out[f"m_star_{tag}"] - sig_tot_bg * npix)
        out[f"m_gas_{tag}_bg"] = out[f"m_gas_{tag}"] - sig_gas_bg * npix
        if thr is not None:
            gw = out[f"m_gas_{tag}"] + 1e-30
            out[f"Y_{tag}"] = (thr[:, 0] * m).sum((1, 2)) * PIX_AREA
            out[f"T_mw_{tag}"] = (thr[:, 1] * gen[:, 1] * m).sum((1, 2)) / gw
            out[f"K_mw_{tag}"] = (thr[:, 2] * gen[:, 1] * m).sum((1, 2)) / gw
            out[f"Pe_mw_{tag}"] = (thr[:, 3] * gen[:, 1] * m).sum((1, 2)) / gw
    return out


def main() -> None:
    for run, root in SRC.items():
        slabs = sorted((root / f"snap_{SNAP:03d}").glob("composite_slab*.npz"))
        assert slabs, f"no composite slabs under {root}/snap_{SNAP:03d}"
        parts = [r for r in (reduce_file(s) for s in slabs) if r is not None]
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

        ref_path = OUT / f"{run}_snap{SNAP:03d}.npz"
        ref = np.load(ref_path)
        cat["z"] = ref["z"]          # z/snap are manifest lookups, carried over
        cat["snap"] = ref["snap"]

        new_keys = sorted(set(cat) - set(ref.files))
        for k in ref.files:
            a, b = np.asarray(ref[k], float), np.asarray(cat[k], float)
            assert a.shape == b.shape, f"{run}:{k} shape {a.shape} != {b.shape}"
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.max(np.abs(np.where(a != 0, b / a - 1.0, b - a)))
            assert rel == 0.0, (
                f"{run}:{k} not reproduced (max rel diff {rel:.3e}) -- the slabs on disk "
                f"no longer match the shipped atlas; NOT replacing {ref_path.name}")
        print(f"[{run}] all {len(ref.files)} existing keys reproduced exactly; "
              f"adding {new_keys}")

        bak = ref_path.with_suffix(".pre200c.npz")
        if not bak.exists():
            shutil.copy2(ref_path, bak)
        tmp = ref_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **cat)
        tmp.replace(ref_path)
        n = len(cat["M_fof"])
        print(f"[{run}] wrote {ref_path.name}: {n} halos, "
              f"median Y_200c/Y_500c = {np.median(cat['Y_200c']/cat['Y_500c']):.3f}")


if __name__ == "__main__":
    main()
