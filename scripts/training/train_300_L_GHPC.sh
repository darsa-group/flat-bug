#!/bin/bash
#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=120000
#SBATCH -t 96:00:00
#SBATCH -J fb300L
#SBATCH --gres=gpu:2
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fb300L_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fb300L_%j.err
# 300 epochs, LARGE backbone, on the EXIF-corrected corpus (flatbug-dir4).
#
# Submit with a dependency so it waits for both the medium run to finish and the dataset to be
# rebuilt, e.g.
#     sbatch --dependency=afterany:<M job>,afterok:<prep job> train_300_L_GHPC.sh
#
# ~70 h expected: L is 1.17x M's compute (measured), M ran ~12 min/epoch. That leaves ~24 h of
# margin against the 96 h limit.
set -eu
ROOT=/usr/home/qgg/qgeiss/flatbug-dir4
REPO=/usr/home/qgg/qgeiss/flat-bug-zoom
CFG=$REPO/scripts/training/fb_config_300_L.yaml
NAME=fb300L_$(date +%Y-%m-%d_%H-%M-%S)
test -d "$ROOT/flat-bug-data/yolo/insects" || { echo "MISSING $ROOT - run prep4 first"; exit 1; }
source /usr/home/qgg/qgeiss/.venv/bin/activate
export PYTHONPATH=$REPO/src
cd $REPO/scripts/training
echo "[$(date +%F_%T)] $NAME  cfg $CFG  data $ROOT"
grep '^path:' $ROOT/flat-bug-data/yolo/data.yaml
srun fb_train -c "$CFG" -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
echo "[$(date +%F_%T)] done $NAME"
