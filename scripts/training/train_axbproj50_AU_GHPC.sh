#!/bin/bash
# Containment arm. Compares against two finished runs on the same tree and data root:
#   control    fb_axb_excl_2026-09-05_10-11-39   artaxor-bbox excluded
#   reference  fb_axb_bbox_2026-09-06_10-48-25   artaxor-bbox as boxes-only, no containment
# Only fb_bbox_only_containment differs from the reference, so the pair isolates the penalty.
#
#   sbatch train_axbcontain50_AU_GHPC.sh
#
#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=96000
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH -J fb_axbproj
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_axbproj_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_axbproj_%j.err

source ~/.venv/bin/activate
export PYTHONPATH=$HOME/flat-bug-contain/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SRC=$HOME/flat-bug-contain
ROOT=$HOME/flatbug-dir2
CFG=$SRC/scripts/training/fb_config_axbproj50_GHPC.yaml
grep -q '^fb_bbox_only_projection: 1.0$' "$CFG" || { echo "FAIL: projection weight not set"; exit 1; }
grep -q '^fb_bbox_only_datasets: \["artaxor-bbox"\]$' "$CFG" || { echo "FAIL: bbox-only dataset not set"; exit 1; }
echo "config:"; grep -E '^fb_|^epochs' "$CFG"

NAME=fb_axbproj50_$(date +"%Y-%m-%d_%H-%M-%S")
cd $SRC/scripts/training
fb_train -c "$CFG" -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
