#!/bin/bash
# Mask-resolution / thin-structure experiment. Three arms sharing the already
# prepared data, differing only in mask_ratio and fb_thin_weight:
#   A (done)  fb_config_control_GHPC.yaml  mask_ratio 4, no weighting
#   B         fb_config_ratio2_GHPC.yaml   mask_ratio 2, no weighting
#   C         fb_config_thin_GHPC.yaml     mask_ratio 2, thin weight 4.0
# B isolates keeping appendages in the target, C isolates the loss weighting.
#   sbatch --export=FB_CONFIG=fb_config_ratio2_GHPC.yaml,FB_TAG=ratio2 train_thin_AU_GHPC.sh
#   sbatch --export=FB_CONFIG=fb_config_thin_GHPC.yaml,FB_TAG=thin     train_thin_AU_GHPC.sh

#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=96000
#SBATCH -t 96:00:00
#SBATCH --gres=gpu:1
#SBATCH -J fb_thin

source ~/.venv/bin/activate
export PYTHONPATH=$HOME/flat-bug-synth/src

ROOT=$HOME/flatbug-dir
CONFIG=$HOME/flat-bug-synth/scripts/training/${FB_CONFIG}
NAME=fb_${FB_TAG}_$(date +"%Y-%m-%d_%H-%M-%S")

# Same prepared data as the control arm on purpose: re-running fb_clone_data would
# repull from CVAT and any annotation edited since could differ between arms.
cd $HOME/flat-bug-synth/scripts/training
fb_train -c ${CONFIG} -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
