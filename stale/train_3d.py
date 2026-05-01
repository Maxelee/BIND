"""3D multi-GPU training script for conditional flow matching."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from data_3d import AstroDataset3D, NormStats3D, compute_norm_stats_3d, load_file_list_3d
from model_3d import FlowMatching3D, StochasticInterpolant3D, UNet3D


class FlowMatchingLit3D(L.LightningModule):
    """Lightning wrapper for 3D flow matching + EMA."""

    def __init__(
        self,
        base_ch: int = 16,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        n_blocks: int = 2,
        emb_dim: int = 256,
        dropout: float = 0.1,
        n_params: int = 36,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ema_decay: float = 0.9999,
        cfg_dropout: float = 0.1,
        warmup_steps: int = 1000,
        star_occ_weight: float = 1.0,
        star_zero_norm: float | None = None,
        interpolant: str = "fm",
        sigma: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.unet = UNet3D(
            in_ch=7,
            out_ch=3,
            base_ch=base_ch,
            ch_mult=ch_mult,
            n_blocks=n_blocks,
            emb_dim=emb_dim,
            dropout=dropout,
            n_params=n_params,
        )

        if interpolant == "si":
            self.fm = StochasticInterpolant3D(
                self.unet,
                sigma=sigma,
                cfg_dropout=cfg_dropout,
                star_occ_weight=star_occ_weight,
                star_zero_norm=star_zero_norm,
            )
        else:
            self.fm = FlowMatching3D(
                self.unet,
                cfg_dropout=cfg_dropout,
                star_occ_weight=star_occ_weight,
                star_zero_norm=star_zero_norm,
            )

        self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=ema_decay)

    def training_step(self, batch, batch_idx):
        loss = self.fm.loss(batch["target"], batch["condition"], batch["large_scale"], batch["params"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.fm.loss(batch["target"], batch["condition"], batch["large_scale"], batch["params"])
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

    def on_train_start(self):
        self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=self.hparams.ema_decay)

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.unet.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        warmup = self.hparams.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            return 1.0

        warmup_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, self.trainer.estimated_stepping_batches - warmup),
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [warmup_sched, cosine_sched],
            milestones=[warmup],
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


class AstroDataModule3D(L.LightningDataModule):
    """Lightning DataModule for volumetric 3D halo training data."""

    def __init__(
        self,
        data_root: str,
        norm_stats_path: str | None = None,
        batch_size: int = 1,
        num_workers: int = 8,
        n_stats_samples: int = 256,
        train_fraction: float = 0.9,
        split_seed: int = 42,
        crop_size: int = 128,
        max_train_files: int | None = None,
        max_val_files: int | None = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.norm_stats_path = norm_stats_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_stats_samples = n_stats_samples
        self.train_fraction = train_fraction
        self.split_seed = split_seed
        self.crop_size = crop_size
        self.max_train_files = max_train_files
        self.max_val_files = max_val_files

    def setup(self, stage=None):
        train_files = load_file_list_3d(
            self.data_root,
            split="train",
            train_fraction=self.train_fraction,
            split_seed=self.split_seed,
        )
        val_files = load_file_list_3d(
            self.data_root,
            split="test",
            train_fraction=self.train_fraction,
            split_seed=self.split_seed,
        )

        if self.max_train_files:
            train_files = train_files[: self.max_train_files]
        if self.max_val_files:
            val_files = val_files[: self.max_val_files]

        stats_path = Path(self.norm_stats_path or Path(self.data_root) / "norm_stats_3d.npz")
        if stats_path.exists():
            self.norm_stats = NormStats3D.load(stats_path)
            print(f"Loaded 3D norm stats from {stats_path}")
        else:
            print(f"Computing 3D norm stats from {self.n_stats_samples} samples...")
            self.norm_stats = compute_norm_stats_3d(train_files, n_samples=self.n_stats_samples)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            self.norm_stats.save(stats_path)
            print(f"Saved 3D norm stats to {stats_path}")

        self.train_ds = AstroDataset3D(
            train_files,
            self.norm_stats,
            crop_size=self.crop_size,
            random_crop=True,
            augment_flip=True,
        )
        self.val_ds = AstroDataset3D(
            val_files,
            self.norm_stats,
            crop_size=self.crop_size,
            random_crop=False,
            augment_flip=False,
        )

        print(f"Train files: {len(train_files)} | Val files: {len(val_files)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


def parse_ch_mult(text: str) -> tuple[int, ...]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("ch_mult cannot be empty")
    return tuple(values)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/mnt/home/mlee1/ceph/train_data_1024/train_3d")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--n_stats_samples", type=int, default=256)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)

    parser.add_argument("--base_ch", type=int, default=16)
    parser.add_argument("--ch_mult", type=str, default="1,2,4")
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--star_occ_weight", type=float, default=1.0)
    parser.add_argument("--interpolant", type=str, default="fm", choices=["si", "fm"])
    parser.add_argument("--sigma", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bf16-mixed")

    parser.add_argument("--output_dir", type=str, default="/mnt/home/mlee1/ceph/fm_runs_3d")
    parser.add_argument("--run_name", type=str, default="fm3d_base")

    args = parser.parse_args()

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dm = AstroDataModule3D(
        data_root=args.data_root,
        norm_stats_path=str(run_dir / "norm_stats_3d.npz"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_stats_samples=args.n_stats_samples,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        crop_size=args.crop_size,
        max_train_files=args.max_train_files,
        max_val_files=args.max_val_files,
    )

    # Build/load normalization stats before constructing the model.
    dm.setup()
    ns = dm.norm_stats
    n_params = int(ns.param_min.shape[0])
    star_zero_norm = float((0.0 - ns.target_mean[2]) / (ns.target_std[2] + 1e-8))

    model = FlowMatchingLit3D(
        base_ch=args.base_ch,
        ch_mult=parse_ch_mult(args.ch_mult),
        n_blocks=args.n_blocks,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        n_params=n_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        cfg_dropout=args.cfg_dropout,
        warmup_steps=args.warmup_steps,
        star_occ_weight=args.star_occ_weight,
        star_zero_norm=star_zero_norm,
        interpolant=args.interpolant,
        sigma=args.sigma,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="epoch{epoch:03d}-val_loss{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=False,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
        callbacks=callbacks,
        default_root_dir=str(run_dir),
        log_every_n_steps=20,
        val_check_interval=1.0,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
