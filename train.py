"""Multi-GPU training script for conditional flow matching with PyTorch Lightning."""

import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader
from pathlib import Path

from data import (load_file_list, compute_norm_stats, AstroDataset, NormStats,
                  load_file_list_cube, compute_norm_stats_cube, CubeAstroDataset)
from model import UNet, FlowMatching, StochasticInterpolant


class FlowMatchingLit(L.LightningModule):
    """Lightning module wrapping Flow Matching training + EMA."""

    def __init__(self, base_ch=128, ch_mult=(1, 2, 4, 8), n_blocks=2,
                 emb_dim=512, attn_resolutions=(32, 16), dropout=0.1,
                 n_params=35, lr=1e-4, weight_decay=1e-4, ema_decay=0.9999,
                 cfg_dropout=0.1, warmup_steps=1000, n_sampling_steps=50,
                 star_occ_weight=1.0, star_zero_norm=None,
                 interpolant='fm', sigma=0.5, stars_two_head=False,
                 no_large_scale=False):
        super().__init__()
        self.save_hyperparameters()

        # Stars two-head mode: target gets a 4-channel layout
        # (DM_hydro, Gas, occupancy, conditional density), so the model needs
        # in_ch = 4 (state) + 1 (condition) + 3 (large_scale) = 8 and out_ch = 4.
        # When no_large_scale=True the large_scale channels are absent, reducing
        # in_ch by 3. All other code paths default to the original architecture.
        out_ch = 4 if stars_two_head else 3
        ls_ch = 0 if no_large_scale else 3
        in_ch = out_ch + 1 + ls_ch   # state + condition + large_scale

        self.unet = UNet(
            in_ch=in_ch, out_ch=out_ch, base_ch=base_ch, ch_mult=ch_mult,
            n_blocks=n_blocks, emb_dim=emb_dim,
            attn_resolutions=attn_resolutions, dropout=dropout,
            n_params=n_params,
        )
        if interpolant == 'si':
            # Stars two-head not wired into the SI branch yet; SI is unused
            # in current analyses. Flag it loudly if someone tries.
            assert not stars_two_head, (
                'stars_two_head=True is not implemented for the StochasticInterpolant '
                'branch. Use --interpolant fm.'
            )
            self.fm = StochasticInterpolant(self.unet, sigma=sigma,
                                            cfg_dropout=cfg_dropout,
                                            star_occ_weight=star_occ_weight,
                                            star_zero_norm=star_zero_norm)
        else:
            self.fm = FlowMatching(self.unet, cfg_dropout=cfg_dropout,
                                   star_occ_weight=star_occ_weight,
                                   star_zero_norm=star_zero_norm,
                                   out_channels=out_ch)
        self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=ema_decay)

    def training_step(self, batch, batch_idx):
        loss = self.fm.loss(
            batch['target'], batch['condition'],
            batch.get('large_scale'), batch['params'],
        )
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.fm.loss(
            batch['target'], batch['condition'],
            batch.get('large_scale'), batch['params'],
        )
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

    def on_train_start(self):
        # Re-init EMA after model has been moved to device by Lightning/DDP
        self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=self.hparams.ema_decay)

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema_state_dict'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # Linear warmup then cosine decay
        warmup = self.hparams.warmup_steps
        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            return 1.0
        warmup_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.estimated_stepping_batches - warmup,
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup_sched, cosine_sched], milestones=[warmup],
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


class AstroDataModule(L.LightningDataModule):

    def __init__(self, data_root, norm_stats_path=None, batch_size=64,
                 num_workers=8, n_stats_samples=10000, stars_two_head=False,
                 param_indices=None, no_large_scale=False):
        super().__init__()
        self.data_root = data_root
        self.norm_stats_path = norm_stats_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_stats_samples = n_stats_samples
        self.stars_two_head = stars_two_head
        self.param_indices = param_indices
        self.no_large_scale = no_large_scale

    def setup(self, stage=None):
        if self.no_large_scale:
            train_files = load_file_list_cube(self.data_root, 'train')
            test_files = load_file_list_cube(self.data_root, 'test')
        else:
            train_files = load_file_list(self.data_root, 'train')
            test_files = load_file_list(self.data_root, 'test')

        # Compute or load normalization stats
        stats_path = Path(self.norm_stats_path or
                          Path(self.data_root) / 'norm_stats.npz')
        if stats_path.exists():
            self.norm_stats = NormStats.load(stats_path)
            print(f'Loaded norm stats from {stats_path}')
            if self.stars_two_head and not self.norm_stats.stars_two_head:
                raise RuntimeError(
                    f'stars_two_head=True but {stats_path} was computed in '
                    f'single-head mode (lacks stars_occ/cond stats). Delete '
                    f'the file and re-run to recompute, or pass a different '
                    f'norm_stats_path.'
                )
        else:
            print(f'Computing norm stats from {self.n_stats_samples} samples '
                  f'(stars_two_head={self.stars_two_head}, '
                  f'no_large_scale={self.no_large_scale})...')
            compute_fn = (
                compute_norm_stats_cube if self.no_large_scale
                else compute_norm_stats
            )
            self.norm_stats = compute_fn(
                train_files, self.n_stats_samples,
                stars_two_head=self.stars_two_head,
            )
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            self.norm_stats.save(stats_path)
            print(f'Saved norm stats to {stats_path}')

        DatasetCls = CubeAstroDataset if self.no_large_scale else AstroDataset
        self.train_ds = DatasetCls(train_files, self.norm_stats,
                                   param_indices=self.param_indices)
        self.val_ds = DatasetCls(test_files, self.norm_stats,
                                 param_indices=self.param_indices)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True)


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_root', type=str,
                        default='/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    # Model
    parser.add_argument('--base_ch', type=int, default=128)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cfg_dropout', type=float, default=0.1)
    parser.add_argument('--star_occ_weight', type=float, default=1.0,
                        help='Extra loss weight for occupied stellar pixels (1=disabled). '
                             'Ignored when --stars_two_head is set.')
    parser.add_argument('--stars_two_head', action='store_true',
                        help='Split Stars target into (occupancy, conditional density) '
                             'and have the model predict both. Out_ch becomes 4. At '
                             'inference the two channels are recombined via a soft '
                             'multiplier before writing the standard 3-channel artifact.')
    parser.add_argument('--interpolant', type=str, default='fm', choices=['si', 'fm'],
                        help='si=stochastic interpolant (DMO→hydro), fm=original flow matching (noise→hydro)')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Stochastic interpolant noise amplitude (0=deterministic bridge)')
    parser.add_argument('--exclude_cosmo_params', action='store_true',
                        help='Drop the 5 cosmological parameters (indices 0,1,7,8) '
                             'from the conditioning vector, training with 31 params. We leave in Omega b')
    parser.add_argument('--no_large_scale', action='store_true',
                        help='Disable large-scale conditioning (for cube-projection datasets '
                             'that lack a large_scale field). Reduces UNet in_ch by 3.')
    # Training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    # Output
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/home/mlee1/ceph/fm_runs')
    parser.add_argument('--run_name', type=str, default='fm_base')
    args = parser.parse_args()

    # Cosmological parameter indices to exclude when --exclude_cosmo_params is set.
    COSMO_INDICES = [0, 1, 7, 8]
    if args.exclude_cosmo_params:
        import numpy as _np
        param_indices = [i for i in range(35) if i not in COSMO_INDICES]
        n_params = len(param_indices)          # 31
    else:
        param_indices = None
        n_params = 35

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dm = AstroDataModule(
        data_root=args.data_root,
        norm_stats_path=str(run_dir / 'norm_stats.npz'),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stars_two_head=args.stars_two_head,
        param_indices=param_indices,
        no_large_scale=args.no_large_scale,
    )

    # Compute/load norm stats up-front so we can derive star_zero_norm before
    # building the model (Lightning calls dm.setup() again during fit, but it
    # will load from the cached npz rather than recomputing).
    dm.setup()
    ns = dm.norm_stats
    # star_zero_norm is only meaningful in single-head mode (channel 2 = stars
    # density). In two-head mode channel 2 is occupancy and the loss-side
    # weighting is disabled.
    if args.stars_two_head:
        star_zero_norm = None
    else:
        star_zero_norm = float((0.0 - ns.target_mean[2]) / (ns.target_std[2] + 1e-8))

    model = FlowMatchingLit(
        base_ch=args.base_ch, n_blocks=args.n_blocks, emb_dim=args.emb_dim,
        dropout=args.dropout, cfg_dropout=args.cfg_dropout,
        lr=args.lr, weight_decay=args.weight_decay,
        ema_decay=args.ema_decay, warmup_steps=args.warmup_steps,
        star_occ_weight=args.star_occ_weight, star_zero_norm=star_zero_norm,
        interpolant=args.interpolant, sigma=args.sigma,
        stars_two_head=args.stars_two_head,
        no_large_scale=args.no_large_scale,
        n_params=n_params,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(run_dir / 'checkpoints'),
            filename='epoch{epoch:03d}-val_loss{val/loss:.4f}',
            monitor='val/loss', mode='min', save_top_k=3, auto_insert_metric_name=False,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices='auto',
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
        precision='bf16-mixed',
        gradient_clip_val=args.gradient_clip,
        callbacks=callbacks,
        default_root_dir=str(run_dir),
        log_every_n_steps=200,
        val_check_interval=1.0,
    )

    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
