"""Train the field-level WL emulator (``θ → κ`` conditional flow matching).

Plain PyTorch + (optional) DDP — launch with ``torchrun`` for multi-GPU.  bf16
autocast, EMA weights, linear-warmup→cosine LR, gradient clipping, and optional
activation checkpointing (needed at native 1024²).  The checkpoint is portable:
it embeds the arch config + the κ normalisation, so ``WLEmulator.load`` needs
only the ``.pt`` file.

    # single GPU
    python -m bind.wlemu.train --cache <cache_dir> --run_name fm_kappa

    # 4-GPU DDP (what run_wlemu_train.sh does)
    torchrun --nproc_per_node=4 -m bind.wlemu.train --cache <cache_dir> \
        --run_name fm_kappa1024 --batch_size 4 --grad_accum 4 --grad_checkpoint
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bind.wlemu.data import ASTRO_PARAM_NAMES, KappaCache, KappaCacheDataset
from bind.wlemu.model import FieldCFM, FieldUNet


@dataclass
class ArchConfig:
    """Everything ``WLEmulator`` needs to rebuild the network."""

    resolution: int
    base_ch: int = 128
    n_blocks: int = 2
    emb_dim: int = 256
    n_params: int = 30
    condition_redshift: bool = True
    ch_mult: tuple | None = None
    cfg_dropout: float = 0.1


# ──────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ddp_info():
    """(rank, world, local_rank) for either an ``srun`` or a ``torchrun`` launch.

    We are launched with ``srun python -m bind.wlemu.train`` (one task per GPU,
    matching run_train.sh), so derive the distributed rank from SLURM's env and
    populate the env vars ``torch.distributed`` expects.  Falls back to torchrun's
    env (RANK/WORLD_SIZE/LOCAL_RANK) and then to a 1-process serial run.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:           # torchrun
        return (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]),
                int(os.environ.get("LOCAL_RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    if world > 1:                                                     # srun SPMD
        rank = int(os.environ["SLURM_PROCID"])
        local = int(os.environ.get("SLURM_LOCALID", 0))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        os.environ["LOCAL_RANK"] = str(local)
        # Single-node DDP (run_train.sh uses --nodes=1); rendezvous on localhost.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29577"))
        return rank, world, local
    return 0, 1, 0


def _is_main():
    return _ddp_info()[0] == 0


def _log(*a):
    if _is_main():
        print(*a, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────
def _lr_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    rank, world, local = _ddp_info()
    ddp = world > 1
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)

    # Fixed input shapes (one resolution, one micro-batch) → let cuDNN autotune
    # the fastest conv algorithm and use the TF32 lanes for fp32 fallbacks.
    # NB: the cuDNN sub-library (libcudnn_cnn_train.so.8) must be on
    # LD_LIBRARY_PATH for conv-backward — run_wlemu_train.sh handles that for the
    # Flatiron nix torch build; otherwise the first loss.backward() raises
    # "FIND/GET was unable to find an engine to execute this computation".
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    cache = KappaCache.load(args.cache)
    resolution = cache.resolution
    train_idx, val_idx = cache.run_split(val_frac=args.val_frac, seed=args.seed)
    _log(f"[data] cache res {resolution}²  | {cache.n_runs} runs "
         f"({len(train_idx)} train / {len(val_idx)} val) · "
         f"{cache.n_real} real · {cache.n_z} z")

    train_ds = KappaCacheDataset(cache, run_idx=train_idx)
    val_ds = KappaCacheDataset(cache, run_idx=val_idx)

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        if ddp else None)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    cfg = ArchConfig(
        resolution=resolution, base_ch=args.base_ch, n_blocks=args.n_blocks,
        emb_dim=args.emb_dim, n_params=cache.params_unit.shape[1],
        condition_redshift=True, cfg_dropout=args.cfg_dropout)
    model = FieldUNet(
        resolution=cfg.resolution, base_ch=cfg.base_ch, n_blocks=cfg.n_blocks,
        emb_dim=cfg.emb_dim, n_params=cfg.n_params,
        condition_redshift=cfg.condition_redshift, dropout=args.dropout,
        grad_checkpoint=args.grad_checkpoint).to(device)
    # NB: channels-last (NHWC) was tried here and measured *slower* for this net
    # (GroupNorm-in-every-ResBlock + attention reshapes thrash the layout rather
    # than hitting faster kernels), so we keep the default NCHW.
    n_par = sum(p.numel() for p in model.parameters())
    _log(f"[model] FieldUNet {resolution}²  base_ch={cfg.base_ch}  "
         f"ch_mult={model.ch_mult}  params={n_par / 1e6:.1f}M  "
         f"grad_ckpt={args.grad_checkpoint}")

    raw_model = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=False)
    fm = FieldCFM(model, cfg_dropout=cfg.cfg_dropout)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

    from torch_ema import ExponentialMovingAverage
    ema = ExponentialMovingAverage(raw_model.parameters(), decay=args.ema_decay)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_factor(s, args.warmup_steps, total_steps))
    _log(f"[train] total_steps={total_steps}  steps/epoch={steps_per_epoch}  "
         f"global_batch={args.batch_size * world * args.grad_accum}")

    out_dir = Path(args.output_dir) / args.run_name
    if _is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    def save(tag, step, val_loss):
        if not _is_main():
            return
        with ema.average_parameters():
            ema_state = {k: v.detach().cpu().clone()
                         for k, v in raw_model.state_dict().items()}
        torch.save({
            "ema": ema_state,
            "model": {k: v.detach().cpu().clone()
                      for k, v in raw_model.state_dict().items()},
            "config": asdict(cfg),
            "norm": {"lam": cache.norm.lam, "mu": cache.norm.mu,
                     "sigma": cache.norm.sigma},
            "source_redshifts": cache.source_redshifts,
            "step": step, "val_loss": val_loss,
            "param_names": list(ASTRO_PARAM_NAMES),
        }, out_dir / f"{tag}.pt")

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    step = 0
    best_val = float("inf")
    t0 = time.time()
    model.train()
    data_iter = _infinite(train_loader, train_sampler)
    accum = 0
    opt.zero_grad(set_to_none=True)
    running = 0.0
    while step < total_steps:
        x, params, sf = next(data_iter)
        x = x.to(device, non_blocking=True)
        params = params.to(device, non_blocking=True)
        sf = sf.to(device, non_blocking=True)
        # Only the final micro-batch of an accumulation window needs to sync
        # gradients; skip the all-reduce on the others (3/4 of the comm at
        # grad_accum=4) via DDP's no_sync().
        is_last_micro = accum == args.grad_accum - 1
        sync_ctx = (model.no_sync() if (ddp and not is_last_micro)
                    else contextlib.nullcontext())
        with sync_ctx:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = fm.loss(x, params, sf) / args.grad_accum
            loss.backward()
        running += loss.item() * args.grad_accum
        accum += 1
        if accum == args.grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            ema.update()
            opt.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            if step % args.log_every == 0:
                rate = args.log_every * args.batch_size * world * args.grad_accum / (time.time() - t0)
                _log(f"step {step}/{total_steps}  loss {running / args.log_every:.4f}  "
                     f"lr {sched.get_last_lr()[0]:.2e}  {rate:.0f} img/s")
                running = 0.0
                t0 = time.time()

            if step % args.val_every == 0 or step == total_steps:
                vloss = _validate(fm, ema, raw_model, val_loader, device, use_amp,
                                  amp_dtype, args.val_batches)
                _log(f"  ↳ val {vloss:.4f}" + ("  *best*" if vloss < best_val else ""))
                if vloss < best_val:
                    best_val = vloss
                    save("best", step, vloss)
                save("last", step, vloss)
                model.train()

    save("last", step, best_val)
    _log(f"[done] best val {best_val:.4f} → {out_dir}")
    if ddp:
        torch.distributed.destroy_process_group()


def _infinite(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


@torch.no_grad()
def _validate(fm, ema, raw_model, loader, device, use_amp, amp_dtype, max_batches):
    raw_model.eval()
    losses = []
    with ema.average_parameters():
        for i, (x, params, sf) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            params = params.to(device)
            sf = sf.to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                losses.append(fm.loss(x, params, sf).item())
    return float(np.mean(losses)) if losses else float("nan")


def build_parser():
    p = argparse.ArgumentParser(description="Train the field-level WL κ emulator.")
    p.add_argument("--cache", required=True, help="cache dir from bind-wlemu-cache")
    p.add_argument("--run_name", required=True)
    p.add_argument("--output_dir", default="./runs")
    # architecture
    p.add_argument("--base_ch", type=int, default=128)
    p.add_argument("--n_blocks", type=int, default=2)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--cfg_dropout", type=float, default=0.1)
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="activation checkpointing (use at 1024²)")
    # optimisation
    p.add_argument("--batch_size", type=int, default=16, help="per-GPU micro-batch")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--max_steps", type=int, default=0, help="0 → use --epochs")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--warmup_steps", type=int, default=500)
    # data / val / logging
    p.add_argument("--val_frac", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=20)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
