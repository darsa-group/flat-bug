#!/bin/bash
#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=120000
#SBATCH -t 168:00:00
#SBATCH -J fb500M
#SBATCH --gres=gpu:2
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fb500M_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fb500M_%j.err
# 500 epochs, medium backbone, two GPUs, on the EXIF-corrected corpus.
#
# -t 168:00:00 (7 days), not the 96 h earlier scripts used: ghpc_gpu's MaxTime is 45-12:00:00,
# so the 96 h ceiling those scripts assumed was self-imposed and had no basis. At ~12 min an
# epoch this run needs ~100 h, which the real limit accommodates with room to spare.
#
# Why 500: the 300-epoch run did not converge. Loss reduction per 25-epoch window ACCELERATED
# through the last third, and validation mask mAP50-95 rose at all twelve validation points,
# with its largest late gain (+0.0085) in the final measured interval. Schedule length, not
# capacity, is the binding constraint.
set -eu
ROOT=/usr/home/qgg/qgeiss/flatbug-dir4
REPO=/usr/home/qgg/qgeiss/flat-bug-zoom
CFG=$REPO/scripts/training/fb_config_500_M.yaml
NAME=fb500M_$(date +%Y-%m-%d_%H-%M-%S)
test -d "$ROOT/flat-bug-data/yolo/insects" || { echo "MISSING $ROOT - run fb_prepare_data first"; exit 1; }
source /usr/home/qgg/qgeiss/.venv/bin/activate
export PYTHONPATH=$REPO/src
cd $REPO/scripts/training
echo "[$(date +%F_%T)] $NAME  cfg $CFG  data $ROOT"
grep '^path:' $ROOT/flat-bug-data/yolo/data.yaml
srun fb_train -c "$CFG" -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
echo "[$(date +%F_%T)] done $NAME"
