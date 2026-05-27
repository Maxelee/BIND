"""Multi-GPU training script for 3D conditional flow matching."""

import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader
from pathlib import Path

from data_3d import (
    load_file_list_3d, compute_norm_stats_3d, AstroDataset3d, NormStats3d,
)
from model_3d import UNet3d, FlowMatching3d


class FlowMatching3dLit(L.LightningModule):

    def __init__(self, base_ch=128, ch_mult=(1, 2, 4, 8), n_blocks=2,
                 emb_dim=512, attn_resolutions=(16,), dropout=0.1,
                 n_params=35, lr=1e-4, weight_decay=1e-4, ema_decay=0.9999,
                 cfg_dropout=0.1, warmup_steps=1000, n_sampling_steps=50,
                 star_occ_weight=1.0, star_zero_norm=None,
                 input_resolution=128, use_checkpoint=True,
                 stars_two_head=False, compile_unet=False):
        super().__init__()
        self.save_hyperparameters()

        out_ch = 4 if stars_two_head else 3
        in_ch = out_ch + 1   # state + condition (no large_scale in 3D)

        self.unet = UNet3d(
            in_ch=in_ch, out_ch=out_ch, base_ch=base_ch, ch_mult=ch_mult,
            n_blocks=n_blocks, emb_dim=emb_dim,
            attn_resolutions=attn_resolutions, dropout=dropout,
            n_params=n_params, input_resolution=input_resolution,
            use_checkpoint=use_checkpoint,
        )
        self.fm = FlowMatching3d(self.unet, cfg_dropout=cfg_dropout,
                                 star_occ_weight=star_occ_weight,
                                 star_zero_norm=star_zero_norm,
                                 out_channels=out_ch)
        self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=ema_decay)

    def training_step(self, batch, batch_idx):
        loss = self.fm.loss(batch['target'], batch['condition'], batch['params'])
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.fm.loss(batch['target'], batch['condition'], batch['params'])
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

    def on_train_start(self):
        # torch_ema's shadow_params are plain tensors (not registered buffers),
        # so Lightning's .to(device) doesn't move them. Migrate them here so
        # update() runs on the same device as the unet parameters.
        device = next(self.unet.parameters()).device
        self.ema.shadow_params = [p.to(device) for p in self.ema.shadow_params]

        # Compile the unet on the first training start. Done here (after Lightning
        # has placed the model on the GPU and after weights are loaded) to avoid
        # state-dict prefix mismatches with checkpoints saved uncompiled.
        if self.hparams.compile_unet and not getattr(self, '_unet_compiled', False):
            self.unet = torch.compile(self.unet)
            self._unet_compiled = True

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema_state_dict'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Strip torch.compile's `_orig_mod.` prefix so a state_dict saved while
        # compiled loads cleanly into the uncompiled __init__ module (and vice
        # versa). The actual parameters are identical either way.
        sd = checkpoint.get('state_dict', {})
        if any('_orig_mod.' in k for k in sd):
            checkpoint['state_dict'] = {
                k.replace('._orig_mod.', '.'): v for k, v in sd.items()
            }
        if 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
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


class AstroDataModule3d(L.LightningDataModule):

    def __init__(self, data_root, norm_stats_path=None, batch_size=1,
                 num_workers=4, n_stats_samples=2000,
                 stars_two_head=False, interp_empty=True, interp_sigma=1.5):
        super().__init__()
        self.data_root = data_root
        self.norm_stats_path = norm_stats_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_stats_samples = n_stats_samples
        self.stars_two_head = stars_two_head
        self.interp_empty = interp_empty
        self.interp_sigma = interp_sigma

    def setup(self, stage=None):
        train_files = load_file_list_3d(self.data_root, 'train')
        test_files = load_file_list_3d(self.data_root, 'test')

        stats_path = Path(self.norm_stats_path or
                          Path(self.data_root) / 'norm_stats_3d.npz')
        if stats_path.exists():
            self.norm_stats = NormStats3d.load(stats_path)
            print(f'Loaded 3D norm stats from {stats_path}')
            if self.stars_two_head and not self.norm_stats.stars_two_head:
                raise RuntimeError(
                    f'stars_two_head=True but {stats_path} was computed in '
                    f'single-head mode. Delete it and re-run, or pass a '
                    f'different norm_stats_path.'
                )
            if self.interp_empty != self.norm_stats.interp_empty:
                raise RuntimeError(
                    f'interp_empty={self.interp_empty} but cached stats have '
                    f'interp_empty={self.norm_stats.interp_empty}. Stats '
                    f'depend on whether the field is filled. Delete '
                    f'{stats_path} and re-run.'
                )
        else:
            print(f'Computing 3D norm stats from {self.n_stats_samples} samples '
                  f'(stars_two_head={self.stars_two_head}, '
                  f'interp_empty={self.interp_empty}, sigma={self.interp_sigma})...')
            self.norm_stats = compute_norm_stats_3d(
                train_files, self.n_stats_samples,
                stars_two_head=self.stars_two_head,
                interp_empty=self.interp_empty,
                interp_sigma=self.interp_sigma,
            )
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            self.norm_stats.save(stats_path)
            print(f'Saved 3D norm stats to {stats_path}')

        self.train_ds = AstroDataset3d(train_files, self.norm_stats)
        self.val_ds = AstroDataset3d(test_files, self.norm_stats)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/mnt/home/mlee1/ceph/train_data_1024/train_3d')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--base_ch', type=int, default=128)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cfg_dropout', type=float, default=0.1)
    parser.add_argument('--star_occ_weight', type=float, default=1.0,
                        help='Loss upweight for occupied stellar voxels (1=disabled).')
    parser.add_argument('--stars_two_head', action='store_true',
                        help='Split Stars target into (occupancy, conditional density). '
                             'Out_ch becomes 4.')
    parser.add_argument('--input_resolution', type=int, default=128)
    parser.add_argument('--no_checkpoint', action='store_true',
                        help='Disable gradient checkpointing (uses much more memory).')
    parser.add_argument('--no_interp_empty', action='store_true',
                        help='Disable mask-aware Gaussian fill of empty voxels in '
                             'DM_hydro / Gas / DMO condition.')
    parser.add_argument('--interp_sigma', type=float, default=1.5,
                        help='Gaussian smoothing sigma for the empty-voxel fill.')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/home/mlee1/ceph/fm_runs_3d')
    parser.add_argument('--run_name', type=str, default='fm3d_base')
    parser.add_argument('--compile', action='store_true',
                        help='torch.compile() the unet at the start of training.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a .ckpt: load weights+EMA from it but '
                             'reset optimizer/scheduler/epoch (clean restart '
                             'from a known-good checkpoint).')
    args = parser.parse_args()

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dm = AstroDataModule3d(
        data_root=args.data_root,
        norm_stats_path=str(run_dir / 'norm_stats_3d.npz'),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stars_two_head=args.stars_two_head,
        interp_empty=not args.no_interp_empty,
        interp_sigma=args.interp_sigma,
    )
    dm.setup()
    ns = dm.norm_stats
    # star_zero_norm only meaningful in single-head mode (ch 2 = stars density).
    if args.stars_two_head:
        star_zero_norm = None
    else:
        star_zero_norm = float((0.0 - ns.target_mean[2]) / (ns.target_std[2] + 1e-8))

    model_kwargs = dict(
        base_ch=args.base_ch, n_blocks=args.n_blocks, emb_dim=args.emb_dim,
        dropout=args.dropout, cfg_dropout=args.cfg_dropout,
        lr=args.lr, weight_decay=args.weight_decay,
        ema_decay=args.ema_decay, warmup_steps=args.warmup_steps,
        star_occ_weight=args.star_occ_weight, star_zero_norm=star_zero_norm,
        input_resolution=args.input_resolution,
        use_checkpoint=not args.no_checkpoint,
        stars_two_head=args.stars_two_head,
        compile_unet=args.compile,
    )
    if args.resume_from:
        print(f'Loading weights + EMA from {args.resume_from} '
              f'(optimizer/scheduler/epoch reset)...')
        # map_location='cpu' is critical under DDP: without it, every rank
        # deserializes tensors onto cuda:0 (the device they were saved from on
        # rank 0), which OOMs with N_GPUS copies on a single GPU. Lightning's
        # trainer moves the model to each rank's correct GPU after this.
        model = FlowMatching3dLit.load_from_checkpoint(
            args.resume_from, map_location='cpu', **model_kwargs,
        )
    else:
        model = FlowMatching3dLit(**model_kwargs)

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
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        default_root_dir=str(run_dir),
        log_every_n_steps=50,
        val_check_interval=1.0,
    )

    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
