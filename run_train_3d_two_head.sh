#!/bin/bash
#SBATCH --job-name=fm3d_two_head_v2
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm3d_two_head_v2_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm3d_two_head_v2_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

mkdir -p /mnt/home/mlee1/ceph/logs

# Restart from epoch-6 best checkpoint of the v1 run (val_loss=0.2976) with:
#   --compile           torch.compile() the unet (~1.3–1.5× speedup)
#   --lr 5e-5           half the v1 LR — v1 spiked at ~step 3700 with lr=1e-4
#   --gradient_clip 0.5 tighter clip; bf16 + AdamW + two-head can spike on rare batches
#   --warmup_steps 5000 longer warmup so the cosine schedule eases in
#   --resume_from       loads weights+EMA from v1 best ckpt; optimizer is reset
#                       (don't carry the polluted Adam moments forward)
#
# Memory: keeping gradient checkpointing on (default). We tried --no_checkpoint
# both with batch=1+accumulate=2 and batch=1+accumulate=1 — both OOM'd at
# step-1 forward. At 128^3 with base_ch=128 and torch.compile overhead, the
# forward-pass activations alone don't fit on H100-80GB without checkpointing.
# expandable_segments:True still helps reduce fragmentation overhead.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun python train_3d.py \
    --data_root /mnt/home/mlee1/ceph/train_data_1024/train_3d \
    --batch_size 2 \
    --num_workers 8 \
    --base_ch 128 \
    --n_blocks 2 \
    --emb_dim 512 \
    --dropout 0.1 \
    --cfg_dropout 0.1 \
    --interp_sigma 1.5 \
    --lr 5e-5 \
    --gradient_clip 0.5 \
    --warmup_steps 5000 \
    --max_epochs 200 \
    --stars_two_head \
    --compile \
    --resume_from /mnt/home/mlee1/ceph/fm_runs_3d/fm3d_two_head/checkpoints/epoch006-val_loss0.2976.ckpt \
    --output_dir /mnt/home/mlee1/ceph/fm_runs_3d \
    --run_name fm3d_two_head_v2
