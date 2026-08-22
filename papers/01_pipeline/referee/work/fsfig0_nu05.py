"""Build the fidswap nu05 per-realization shard + the ngal=10 peak cache.

Reuses papers/01_pipeline/nu_grid.py's own compute() verbatim (same grid, same
conventions, same engine) with the kappa cube re-pointed at the canonical
twobound replica.  Writes ONLY under referee_work/fidswap/ -- nothing is
touched inside bind_sb35 / bind_science.

    /mnt/home/mlee1/venvs/BIND_env/bin/python3 fsfig0_nu05.py [run_0049]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline")
import nu_grid as ng  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
FS = CEPH / "referee_work/fidswap"
OUT = FS / "nu05_shards"
OUT.mkdir(parents=True, exist_ok=True)

RUN = sys.argv[1] if len(sys.argv) > 1 else "run_0049"
TAG = "tb" + RUN.split("_")[1].lstrip("0")
KAPPA = SCI / f"runs/twobound/{RUN}/kappa_maps.npz"
assert KAPPA.exists(), KAPPA

# ── 1. the six nu05 statistics, per-realization draws (= sci_bind.npz's role) ──
dst = OUT / f"sci_bind_{TAG}.npz"
if dst.exists():
    print(f"{dst} exists -- skipping")
else:
    t0 = time.time()
    res = ng.compute(KAPPA, per_real=True)
    np.savez_compressed(dst, **res, src=str(KAPPA))
    print(f"wrote {dst} in {time.time()-t0:.0f}s "
          f"({', '.join(f'{k}{np.asarray(v).shape}' for k, v in res.items() if k != 'nu')})")

# ── 2. ngal=10 shape-noise peak/minima counts on the RELEASED 68-bin grid ─────
# (fig 4 uses the BIND side only for a diagnostic print; the truth side, which
#  sets the shape-noise band, is unaffected by the swap and reused as shipped.)
dst2 = OUT / f"peak_counts_ngal10_{TAG}.npz"
if dst2.exists():
    print(f"{dst2} exists -- skipping")
else:
    from bind.inference.stats import peak_counts
    ref = np.load(SCI / "runs/bind/run_0000/peak_counts_ngal10.npz")
    nu_ref = ref["nu"]
    d = nu_ref[1] - nu_ref[0]
    edges = np.concatenate([nu_ref - d / 2, [nu_ref[-1] + d / 2]])
    t0 = time.time()
    K = np.load(KAPPA)["kappa"]
    pk = np.empty((5, len(nu_ref)))
    pke = np.empty_like(pk)
    mn = np.empty_like(pk)
    mne = np.empty_like(pk)
    for zi in range(5):
        o = peak_counts(K[:, zi][:, None], fov_deg=5.0, smoothing_arcmin=2.0,
                        nu_bins=edges, shape_noise_ngal=10.0, sigma_e=0.26,
                        nu_norm="noise", return_realizations=True)
        pk[zi] = np.asarray(o["peak_counts"])[0]
        pke[zi] = np.asarray(o["peak_counts_err"])[0]
        mn[zi] = np.asarray(o["minima_counts"])[0]
        mne[zi] = np.asarray(o["minima_counts_err"])[0]
    np.savez_compressed(dst2, nu=nu_ref, peak_counts=pk, peak_counts_err=pke,
                        minima_counts=mn, minima_counts_err=mne,
                        smoothing_arcmin=2.0, shape_noise_ngal=10.0,
                        sigma_e=0.26, nu_norm="noise", src=str(KAPPA))
    print(f"wrote {dst2} in {time.time()-t0:.0f}s")
print("DONE")
