#!/usr/bin/env python
"""Local finite-difference Jacobian of the BOX matter-power suppression S(k) w.r.t.
the 35 feedback parameters at the CV fiducial -- the WL-target side of the gas->S(k)
chain on rigorous local-FD footing (matching the gas-side fd_jacobian_thermo.py).

S(k) is a FULL-BOX quantity (paste generated halo patches into the 50 Mpc/h DMO box,
P_hydro/P_DMO, sim-averaged) -- exactly box_supp_sobol.S_true. So we reuse its paste+Pk
primitives verbatim and only swap the cube patches for patches generated at theta_fid+-Delta
(fixed noise, fm_thermo mass channels). Central difference -> dS(k)/dtheta.

Modes:
  --validate D0,D1,...   (NO GPU)  recompute box S(k) from the EXISTING cube maps for
                         these designs and compare to box_supp_sobol.npz S_true -> checks
                         the paste/Pk reuse reproduces the cached pipeline.
  (compute, default)     (GPU)     generate patches at theta_fid+-Delta for all 35 params
                         (or a --params subset / --n_chunks shard), central-difference
                         box S(k); also saves the fiducial S(k). Output:
                         analysis_physics_cache/fd_sk_box[_shardN].npz
  --merge --shard_glob ... --output ...   combine shards.

Run (validate, no GPU):
    python tools/fd_sk_box.py --validate 0,1,2
Run (compute, GPU/SLURM):  sbatch run_fd_sk_box.sh
"""
import argparse
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import box_supp_sobol as B                       # paste + Pk primitives, load_sim_static
from fd_jacobian_thermo import (                 # generation machinery (already validated)
    normalize_params_fid, _sample_fixed_noise, MPC_PER_PIX, r200c_mpc_h,
)
from bind.data import NormStats, log_transform
from bind.inference.pipeline import _denormalize_to_physical
from bind.train import FlowMatchingLit

CV_ROOT = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
SNAP, MTAG = "snap_090", "mass_threshold_1p000e13"
AC = Path("analysis_physics_cache")
N_PARAMS = 35
K_TARGET = 10.0                                   # h/Mpc, the WL-relevant scale


def box_Sk(sims, patches3):
    """Sim-averaged box suppression S(k) from mass-channel patches (cube order),
    identical to box_supp_sobol.main()."""
    return np.mean([B.composite_pk(s, patches3[s['block']]) / s['pk_dmo'] for s in sims], 0)


# ---------------------------------------------------------------------------
def run_validate(designs):
    sims, _ = B.load_sim_static()
    cache = np.load(AC / "box_supp_sobol.npz", allow_pickle=True)
    k_box, S_true = cache["k_box"], cache["S_true"]
    ik = int(np.argmin(np.abs(k_box - K_TARGET)))
    map_files = sorted((B.SOBOL_ROOT / "maps").glob("gen_design*.npz"))
    print(f"validating box S(k) reuse at k~{k_box[ik]:.1f} h/Mpc")
    for d in designs:
        gen = np.load(map_files[d])["generated"][:, :3].astype(np.float32)
        S = box_Sk(sims, gen)
        print(f"  design {d:4d}: mine S(k_t)={S[ik]:.4f}  cache S_true={S_true[d][ik]:.4f}  "
              f"max|dS|={np.nanmax(np.abs(S - S_true[d])):.2e}")


# ---------------------------------------------------------------------------
def build_gen_inputs(sims):
    """Per-sim DMO conditions, large-scale, params, radii in cube (load_sim_static) order."""
    cond, ls, params, radii = [], [], [], []
    for s in sims:
        md = CV_ROOT / s["name"] / SNAP / MTAG
        cuts = np.load(md / "halo_cutouts.npz"); cat = np.load(md / "halo_catalog.npz")
        n = len(s["block"]); m = np.asarray(cat["masses"], float)[:n]
        cond.append(cuts["condition"][:n]); ls.append(cuts["large_scale"][:n])
        params.append(np.asarray(cat["params"], np.float32)[:n])
        radii.append((np.asarray(cat["radii"], float)[:n] / 1000.0 / MPC_PER_PIX)
                     if "radii" in cat.files else r200c_mpc_h(m) / MPC_PER_PIX)
    return (np.concatenate(cond).astype(np.float32), np.concatenate(ls).astype(np.float32),
            np.concatenate(params).astype(np.float32), np.concatenate(radii))


def run_compute(args):
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compute] device = {device}")
    run_dir = Path(args.run_dir)
    norm_stats = NormStats.load(run_dir / "norm_stats.npz")
    lit = FlowMatchingLit.load_from_checkpoint(str(run_dir / "checkpoints" / "last.ckpt"),
                                               map_location=device).eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm; model_fm.model.eval()
    out_ch = model_fm.out_channels
    print(f"[compute] out_channels = {out_ch}")

    sims, _ = B.load_sim_static()
    k_box = sims[0]["k"]; ik = int(np.argmin(np.abs(k_box - K_TARGET)))
    cond_raw, ls_raw, params, radii_pix = build_gen_inputs(sims)
    params[:, 14] = 0.0                                  # CAMELS p14 bug
    Nh = len(params)
    print(f"[compute] N_halo = {Nh} (cube order)")

    cond = ((log_transform(cond_raw) - norm_stats.cond_mean) /
            (norm_stats.cond_std + 1e-8))[:, None]
    ls = ((log_transform(ls_raw) - norm_stats.ls_mean[:, None, None]) /
          (norm_stats.ls_std[:, None, None] + 1e-8))
    p_fid = torch.tensor(normalize_params_fid(params[0], norm_stats), device=device)

    if args.params is not None:
        pidx = np.array([int(s) for s in args.params.split(",")], dtype=int)
    else:
        e = np.linspace(0, N_PARAMS, args.n_chunks + 1).astype(int)
        pidx = np.arange(e[args.chunk_id], e[args.chunk_id + 1])
    print(f"[compute] params {pidx.tolist()}")

    torch.manual_seed(args.noise_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.noise_seed)
    z = torch.randn(Nh, out_ch, 128, 128, device=device)

    def gen_patches(pvec):
        out = np.empty((Nh, 3, 128, 128), np.float32)
        with torch.no_grad():
            for s0 in range(0, Nh, args.batch_size):
                s1 = min(s0 + args.batch_size, Nh); bs = s1 - s0
                cb = torch.tensor(cond[s0:s1], dtype=torch.float32, device=device)
                lb = torch.tensor(ls[s0:s1], dtype=torch.float32, device=device)
                pb = pvec.unsqueeze(0).expand(bs, -1).contiguous()
                x = _sample_fixed_noise(model_fm, cb, lb, pb, z[s0:s1], args.n_steps)
                out[s0:s1] = _denormalize_to_physical(x.cpu().numpy(), norm_stats)[:, :3]
        return out

    t0 = time.time()
    S_fid = box_Sk(sims, gen_patches(p_fid))
    print(f"[compute] fiducial S(k~{k_box[ik]:.1f})={S_fid[ik]:.4f}  ({(time.time()-t0)/60:.1f} min)")

    J_Sk = np.full((N_PARAMS, len(k_box)), np.nan, np.float32)
    for n, j in enumerate(pidx):
        pp = p_fid.clone(); pp[j] += args.eps
        pm = p_fid.clone(); pm[j] -= args.eps
        Sp = box_Sk(sims, gen_patches(pp)); Sm = box_Sk(sims, gen_patches(pm))
        J_Sk[j] = (Sp - Sm) / (2 * args.eps)
        print(f"  param {j:2d} ({n+1}/{len(pidx)})  dS(k_t)/dtheta={J_Sk[j][ik]:+.4f}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, k_box=k_box, k_target=K_TARGET, ik=ik, J_Sk=J_Sk,
                        S_fid=S_fid, param_idxs=pidx,
                        meta=np.array(dict(eps=args.eps, n_steps=args.n_steps,
                                           noise_seed=args.noise_seed, model=str(run_dir)),
                                      dtype=object))
    print(f"[compute] wrote {out}")


def run_merge(args):
    files = sorted(glob(args.shard_glob))
    first = np.load(files[0], allow_pickle=True)
    J = np.full((N_PARAMS, len(first["k_box"])), np.nan, np.float32)
    seen = np.zeros(N_PARAMS, bool)
    for f in files:
        z = np.load(f, allow_pickle=True)
        for j in z["param_idxs"]:
            J[j] = z[f"J_Sk"][j]; seen[j] = True
    np.savez_compressed(args.output, k_box=first["k_box"], k_target=first["k_target"],
                        ik=first["ik"], J_Sk=J, S_fid=first["S_fid"], meta=first["meta"])
    print(f"[merge] wrote {args.output}  covering {seen.sum()}/{N_PARAMS} params")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=str, default=None, help="comma design ids (no GPU)")
    ap.add_argument("--run_dir", default="/mnt/home/mlee1/ceph/fm_runs/fm_thermo")
    ap.add_argument("--output", default="analysis_physics_cache/fd_sk_box.npz")
    ap.add_argument("--params", type=str, default=None)
    ap.add_argument("--n_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--noise_seed", type=int, default=42)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--shard_glob", type=str, default=None)
    args = ap.parse_args()
    if args.validate is not None:
        run_validate([int(x) for x in args.validate.split(",")])
    elif args.merge:
        run_merge(args)
    else:
        run_compute(args)


if __name__ == "__main__":
    main()
