#!/bin/bash
# Zoom arm. Control is the finished fb_axb_excl_2026-09-05_10-11-39 - same tree, same config
# except fb_zoom_prob, same data root - so only this one arm needs running.
#
#   sbatch train_zoom50_AU_GHPC.sh
#
#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=96000
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH -J fb_zoom50
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_zoom50_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_zoom50_%j.err

source ~/.venv/bin/activate
export PYTHONPATH=$HOME/flat-bug-zoom/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SRC=$HOME/flat-bug-zoom
ROOT=$HOME/flatbug-dir2
CFG=$SRC/scripts/training/fb_config_zoom50_GHPC.yaml
grep -q '^fb_zoom_prob: 0.2$' "$CFG" || { echo "FAIL: fb_zoom_prob not set"; exit 1; }
echo "config:"; grep -E '^fb_|^epochs' "$CFG"

NAME=fb_zoom50_$(date +"%Y-%m-%d_%H-%M-%S")
cd $SRC/scripts/training
fb_train -c "$CFG" -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
