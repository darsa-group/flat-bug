#!/bin/bash
#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=120000
#SBATCH -t 48:00:00
#SBATCH -J fbLcont
#SBATCH --gres=gpu:2
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fbLcont_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fbLcont_%j.err
# 100 more epochs on the finished 300-epoch L model, on gpu02's two L40S.
#
# Here rather than GenomeDK: ghpc_gpu allows 45 days so this is ONE job with no chaining, the
# original weights and corpus are already on this filesystem, and GenomeDK's gpu-h200 queue
# went from an 8 h wait to 25 Sep when another user submitted 400 jobs. It will pend until the
# 500-epoch power run releases gpu02, around 22 Sep.
set -eu
ROOT=/usr/home/qgg/qgeiss/flatbug-dir4
REPO=/usr/home/qgg/qgeiss/flat-bug-zoom
CFG=$REPO/scripts/training/fb_config_L_cont100.yaml
NAME=fbLcont100_$(date +%Y-%m-%d_%H-%M-%S)
test -d "$ROOT/flat-bug-data/yolo" || { echo "MISSING $ROOT/flat-bug-data/yolo"; exit 1; }
source /usr/home/qgg/qgeiss/.venv/bin/activate
export PYTHONPATH=$REPO/src
export TORCHDYNAMO_DISABLE=1
cd $REPO/scripts/training
echo "[$(date +%F_%H:%M)] $NAME   cfg $CFG   data $ROOT"
srun fb_train -c "$CFG" -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
echo "[$(date +%F_%H:%M)] done $NAME"
