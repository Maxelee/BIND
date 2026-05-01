#!/bin/bash
#SBATCH --job-name=halo_server
#SBATCH --output=/mnt/home/mlee1/ceph/logs/halo_server_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/halo_server_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

mkdir -p /mnt/home/mlee1/ceph/logs

PORT=8765
NODE=$(hostname)

echo "============================================================"
echo "  BIND Halo Explorer server starting on node: $NODE"
echo "  Port: $PORT"
echo ""
echo "  On your laptop, run:"
echo "    ssh -L ${PORT}:${NODE}:${PORT} mlee1@gateway"
echo ""
echo "  Then open in Chrome:"
echo "    http://localhost:${PORT}"
echo "============================================================"

# Install server deps if missing (fast no-op if already installed)
pip install --quiet fastapi uvicorn pydantic

python serve_halo_explorer.py --port $PORT --img_px 220
