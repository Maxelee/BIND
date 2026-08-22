"""fidswap re-render: imgs/fig06b_full_hydro (the R1 three-rung ladder).

The ONLY change to the R1 machinery is the rung -> directory map:

    r1_common.RUNS['bind']  = ceph/bind_lightcone_tng      (retired fiducial)
 -> r1_common.RUNS['bind']  = bind_science/runs/twobound/run_0049

r1_stats.py and r1_ladder_fig.py are imported and driven UNMODIFIED; only
r1_common.RUNS / r1_common.WORK and the output stems are monkeypatched, so the
estimator code path is byte-identical to the shipped R1 figure.  Work products
go to referee_work/fidswap/r1/ (the four unaffected rungs + the 550-real
covariance are hard-linked in from referee_work/r1/, never recomputed);
referee_work/r1/ itself is left untouched.

    /mnt/home/mlee1/venvs/BIND_env/bin/python3 fsfig6b_r1ladder.py
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")

import r1_common as C  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
FS = CEPH / "referee_work/fidswap"
OLD_WORK = CEPH / "referee_work/r1"
NEW_WORK = FS / "r1"
REPLICA = os.environ.get("BIND_FID_REPLICA", "tb49")
RUN = {"tb18": "run_0018", "tb49": "run_0049", "tb53": "run_0053"}[REPLICA]

# ── the swap ────────────────────────────────────────────────────────────────
C.RUNS["bind"] = C.SCI / f"runs/twobound/{RUN}"
C.WORK = NEW_WORK
NEW_WORK.mkdir(parents=True, exist_ok=True)

# reuse every rung the swap does not touch (pasted / diffuse / hydro_full / dmo
# / the 550-real covariance) -- hard links, so nothing is copied or rebuilt.
for name in ("pre.json", "kappa_pasted.npz", "kappa_diffuse.npz",
             "kappa_hydro_full.npz", "kappa_dmo.npz", "y_pasted.npz",
             "y_diffuse.npz", "y_hydro_full.npz", "tau_pasted.npz",
             "tau_diffuse.npz", "tau_hydro_full.npz", "cov_pasted550.npz"):
    src, dst = OLD_WORK / name, NEW_WORK / name
    if src.exists() and not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
print(f"work dir {NEW_WORK}: {len(list(NEW_WORK.iterdir()))} products staged")

# ── recompute ONLY the bind rung (kappa / y / tau) against the new fiducial ──
import r1_stats as S  # noqa: E402

t0 = time.time()
for stage, fn in (("kappa", lambda: S.stage_kappa("bind", t0)),
                  ("y", lambda: S.stage_gas("bind", "y", t0)),
                  ("tau", lambda: S.stage_gas("bind", "tau", t0))):
    out = NEW_WORK / f"{stage}_bind.npz"
    if out.exists():
        print(f"skip {out.name} (exists)")
        continue
    print(f"=== {stage} bind ({C.RUNS['bind']}) -> {out.name}", flush=True)
    fn()
print(f"bind-rung stats rebuilt in {time.time()-t0:.0f}s")

# ── render, protecting the shipped figure + the shipped numbers json ────────
import r1_ladder_fig as L  # noqa: E402

L.WORK = NEW_WORK
L.OUT_STEM = L.IMGS / "fig06b_full_hydro_fidswap"
HERE = Path(__file__).resolve().parent
for zi in (1, 3):
    keep = HERE / f"r1_numbers_z{zi}.json"
    bak = HERE / f"r1_numbers_z{zi}.json.keep"
    if keep.exists() and not bak.exists():
        shutil.copy2(keep, bak)

sys.argv = ["r1_ladder_fig.py"]
L.main()

# restore the shipped jsons; keep the new ones under a fidswap name
for zi in (1, 3):
    new = HERE / f"r1_numbers_z{zi}.json"
    bak = HERE / f"r1_numbers_z{zi}.json.keep"
    if new.exists():
        shutil.move(new, FS / f"r1_numbers_z{zi}_{REPLICA}.json")
    if bak.exists():
        shutil.move(bak, new)
print(f"\nwrote {L.OUT_STEM}.png/.pdf and {FS}/r1_numbers_z1_{REPLICA}.json")
print("DONE fig06b")
