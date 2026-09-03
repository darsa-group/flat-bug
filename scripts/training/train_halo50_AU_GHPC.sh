#!/bin/bash
# 50-epoch A/B for the inpainting-halo fix.
#
#   sbatch --export=ALL,ARM=before train_halo50_AU_GHPC.sh   # develop, halo present
#   sbatch --export=ALL,ARM=after  train_halo50_AU_GHPC.sh   # patched, halo removed
#
# Both arms read the same prepared data and the same config; ARM only selects which source
# tree is on PYTHONPATH. The pyramid bug is deliberately left in place in both.

#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=96000
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH -J fb_halo
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_halo_%x_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_halo_%x_%j.err

source ~/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$ARM" in
  before) SRC=$HOME/flat-bug-halo-before ;;
  after)  SRC=$HOME/flat-bug-halo-after ;;
  *) echo "set ARM=before or ARM=after"; exit 1 ;;
esac
export PYTHONPATH=$SRC/src

ROOT=$HOME/flatbug-dir
CONFIG=$SRC/scripts/training/fb_config_halo50_GHPC.yaml
NAME=fb_halo_${ARM}_$(date +"%Y-%m-%d_%H-%M-%S")

cd $SRC/scripts/training
fb_train -c ${CONFIG} -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
