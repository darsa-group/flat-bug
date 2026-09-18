#!/bin/bash
#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=120000
#SBATCH -t 120:00:00
#SBATCH -J fb500Mpow
#SBATCH --gres=gpu:2
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fb500Mpow_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fb500Mpow_%j.err
# 500 epochs on gpu02's two L40S, in ONE job - ghpc_gpu allows 45 days, so unlike GenomeDK's
# 2 h gpu-short this needs no chaining and no resume at all.
#
# ~10 min/epoch measured on this node for the 300-epoch run, so ~83 h; -t 120 leaves headroom.
# -n 1 -c 24 with srun follows the GHPC GPU template: ONE task, which ultralytics forks into
# two ranks itself. Asking for -n 2 would start two independent trainings on the same GPUs.
set -eu
ROOT=/usr/home/qgg/qgeiss/flatbug-dir3
REPO=/usr/home/qgg/qgeiss/flat-bug-zoom
CFG=$REPO/scripts/training/fb_config_500_M_power.yaml
NAME=fb500Mpow_$(date +%Y-%m-%d_%H-%M-%S)
test -d "$ROOT/flat-bug-data/yolo" || { echo "MISSING $ROOT/flat-bug-data/yolo"; exit 1; }
source /usr/home/qgg/qgeiss/.venv/bin/activate
export PYTHONPATH=$REPO/src
export TORCHDYNAMO_DISABLE=1
cd $REPO/scripts/training
echo "[$(date +%F_%H:%M)] $NAME   cfg $CFG   data $ROOT"
srun fb_train -c "$CFG" -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
echo "[$(date +%F_%H:%M)] done $NAME"
