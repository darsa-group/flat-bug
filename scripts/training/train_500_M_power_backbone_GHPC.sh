#!/bin/bash
#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=120000
#SBATCH -t 140:00:00
#SBATCH -J fb500Mbb
#SBATCH --gres=gpu:2
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fb500Mbb_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fb500Mbb_%j.err
# Backbone comparison: the recipe of flatbug-M-2026.09 (500 epochs, power LR anneal, zoom crops)
# on the same corpus, flatbug-dir4, with an older YOLO backbone as the only difference.
#
#   sbatch --export=ALL,BB=yolo11m scripts/training/train_500_M_power_backbone_GHPC.sh
#   sbatch --export=ALL,BB=yolov8m scripts/training/train_500_M_power_backbone_GHPC.sh
#
# BB picks fb_config_500_M_power_$BB.yaml, which differs from fb_config_500_M_power.yaml only in
# `model:`. Same node type, venv and GPU count as the YOLO26 run, so times compare too.
#
# REPO is a dedicated clone at branch train/backbones. fb_train refuses to start from a checkout
# with uncommitted changes or untracked files, and records the commit, config, environment and a
# checksummed inventory of every image in runs/segment/<name>/manifest.train.yaml.
#
# -t 140 h: the YOLO26-M run took ~99 h on these GPUs; YOLOv8-M and YOLO11-M are of similar size.
set -eu
BB=${BB:?set BB=yolo11m or BB=yolov8m}
ROOT=/usr/home/qgg/qgeiss/flatbug-dir4
REPO=/usr/home/qgg/qgeiss/flat-bug-backbones
CFG=$REPO/scripts/training/fb_config_500_M_power_$BB.yaml
NAME=fb500M_${BB}_$(date +%Y-%m-%d_%H-%M-%S)
test -d "$ROOT/flat-bug-data/yolo" || { echo "MISSING $ROOT/flat-bug-data/yolo"; exit 1; }
test -f "$CFG" || { echo "MISSING $CFG"; exit 1; }
source /usr/home/qgg/qgeiss/.venv/bin/activate
export PYTHONPATH=$REPO/src
export TORCHDYNAMO_DISABLE=1
cd $REPO/scripts/training
# The starting weights must be here before DDP starts, or each rank tries to download them.
test -f "$BB-seg.pt" || { echo "MISSING $REPO/scripts/training/$BB-seg.pt (download it on the login node)"; exit 1; }
echo "[$(date +%F_%H:%M)] $NAME   cfg $CFG   data $ROOT   repo $(git -C $REPO rev-parse --short HEAD)"
srun fb_train -c "$CFG" -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
echo "[$(date +%F_%H:%M)] done $NAME"
