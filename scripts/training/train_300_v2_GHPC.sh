#!/bin/bash
#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=120000
#SBATCH -t 96:00:00
#SBATCH -J fb300v2
#SBATCH --gres=gpu:1
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fb300v2_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fb300v2_%j.err
# 300 epochs on the resynced corpus. ~58 h at 20000 samples/epoch; -t 96 leaves headroom.
# -n 1 -c 24 with srun follows the GHPC GPU template (one task, 24 CPUs for its dataloader).
set -eu
ROOT=/usr/home/qgg/qgeiss/flatbug-dir3
REPO=/usr/home/qgg/qgeiss/flat-bug-zoom
CFG=$REPO/scripts/training/fb_config_300_v2.yaml
NAME=fb300v2_$(date +%Y-%m-%d_%H-%M-%S)
test -d "$ROOT/flat-bug-data/yolo" || { echo "MISSING $ROOT/flat-bug-data/yolo - run resync_cvat_data.sh first"; exit 1; }
source /usr/home/qgg/qgeiss/.venv/bin/activate
export PYTHONPATH=$REPO/src
cd $REPO/scripts/training
echo "[$(date +%H:%M)] $NAME   cfg $CFG   data $ROOT"
srun fb_train -c "$CFG" -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
echo "[$(date +%H:%M)] done $NAME"
