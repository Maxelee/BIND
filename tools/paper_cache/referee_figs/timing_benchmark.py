#!/usr/bin/env python3
"""Timing benchmark for the paper's computational-cost table (fm_two_head, A100).

Measures, on cached CV/sim_0 artifacts (no particle I/O):
  1. model load (checkpoint -> ready)
  2. single-halo generation latency (batch 1, n_steps=50, warm)
  3. batched generation throughput (batch 64, n_steps=50)
  4. full-box baryonification from cached prep: generate all >=1e13 halos of
     one 50 Mpc/h box + shared-content circular paste (GPU gen / CPU paste split)
  5. a 100-draw posterior ensemble for one halo (batch 50)

Run:  source ~/venvs/torch3/bin/activate && python timing_benchmark.py
"""
import json, os, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "paper_cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
os.environ.setdefault("PAPER_SUITE_ROOT", "/mnt/home/mlee1/ceph/fm_testsuite")
os.environ.setdefault("PAPER_MODEL_SUBDIR", "fm_two_head")
os.environ.setdefault("PAPER_MASS_DIR", "mass_threshold_1p000e13")
os.environ.setdefault("PAPER_MODEL_TAG", "fm_two_head")

import numpy as np
import torch
import paper_config as C
from bind.inference.artifacts import load_halo_catalog, load_halo_cutouts
from bind.inference.paint import Model
from bind.inference.pipeline import build_bind_composite

CKPT = "/mnt/home/mlee1/ceph/fm_runs/fm_two_head/checkpoints/last.ckpt"
NORM = "/mnt/home/mlee1/ceph/fm_runs/fm_two_head/norm_stats.npz"
SIM = "/mnt/home/mlee1/ceph/fm_testsuite/CV/sim_0/snap_090"
MD = f"{SIM}/mass_threshold_1p000e13"
N_STEPS = 50

def sync():
    torch.cuda.synchronize()

out = {}
t0 = time.perf_counter()
model = Model.from_files(CKPT, NORM, device="cuda")
sync(); out["model_load_s"] = time.perf_counter() - t0

cutouts = load_halo_cutouts(f"{MD}/halo_cutouts.npz")
halos, masses, r200s, positions = load_halo_catalog(f"{MD}/halo_catalog.npz")
params = np.asarray(halos[0]["params"], np.float32)
n_halos = len(masses)
print(f"[bench] {n_halos} halos >=1e13 in CV/sim_0; n_steps={N_STEPS}")

# warmup
_ = model.generate(cutouts[:2], params, n_steps=N_STEPS, batch_size=2, progress=False); sync()

# 2. single-halo latency (median of 10)
ts = []
for _ in range(10):
    t0 = time.perf_counter()
    _ = model.generate(cutouts[:1], params, n_steps=N_STEPS, batch_size=1, progress=False)
    sync(); ts.append(time.perf_counter() - t0)
out["single_halo_s"] = float(np.median(ts))

# 3. batched throughput (batch 64 over all halos, median of 3 passes)
ts = []
for _ in range(3):
    t0 = time.perf_counter()
    _ = model.generate(cutouts, params, n_steps=N_STEPS, batch_size=64, progress=False)
    sync(); ts.append(time.perf_counter() - t0)
out["batch64_all_s"] = float(np.median(ts))
out["batch64_per_halo_ms"] = 1e3 * out["batch64_all_s"] / n_halos

# 4. full-box baryonify from cached prep
t0 = time.perf_counter()
gen = model.generate(cutouts, params, n_steps=N_STEPS, batch_size=64, progress=False)
sync(); t_gen = time.perf_counter() - t0
fm = np.load(f"{SIM}/full_maps.npz")
dmo = fm["dmo_fullbox"]
t0 = time.perf_counter()
_ = build_bind_composite(dmo, halos, gen[:, :3], cutouts,
                         box_size=C.BOX_SIZE, npix=C.N_PIX_FULL,
                         patch_pix=C.PATCH_PIX, patch_mass_match=True,
                         taper_frac=0.15, r200_factor=4.0, paste_mode="shared")
t_paste = time.perf_counter() - t0
out["box_generate_s"] = t_gen
out["box_paste_s"] = t_paste
out["box_total_s"] = t_gen + t_paste

# 5. 100-draw ensemble for one halo
t0 = time.perf_counter()
for _ in range(2):
    _ = model.generate([cutouts[0]] * 50, params,
                       n_steps=N_STEPS, batch_size=50, progress=False)
sync(); out["ensemble100_one_halo_s"] = time.perf_counter() - t0

out["n_halos"] = int(n_halos)
out["gpu"] = torch.cuda.get_device_name(0)
print(json.dumps(out, indent=2))
open("/mnt/home/mlee1/ceph/paper_cache/fm_two_head/timing_benchmark.json", "w").write(json.dumps(out, indent=2))
