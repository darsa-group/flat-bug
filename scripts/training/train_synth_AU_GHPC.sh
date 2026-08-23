#!/bin/bash
# Stage 2: one training arm. Submit twice, once per config, so the synthetic
# and control runs differ only in fb_synth_prob.
#   sbatch --export=FB_CONFIG=fb_config_synth_GHPC.yaml,FB_TAG=synth  train_synth_AU_GHPC.sh
#   sbatch --export=FB_CONFIG=fb_config_control_GHPC.yaml,FB_TAG=ctrl train_synth_AU_GHPC.sh

#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=128000
#SBATCH -t 96:00:00
#SBATCH --gres=gpu:1
#SBATCH -J fb_synth_train

source ~/.venv/bin/activate
export PYTHONPATH=$HOME/flat-bug-synth/src

ROOT=$HOME/flatbug-dir
CONFIG=$HOME/flat-bug-synth/scripts/training/${FB_CONFIG}
NAME=fb_${FB_TAG}_$(date +"%Y-%m-%d_%H-%M-%S")

# The data is already cloned and prepared, so both arms train on exactly the same
# files. The train/val split itself is content-deterministic - fb_prepare_data
# assigns on md5(image bytes), so re-preparing would reproduce it - but
# re-running fb_clone_data would repull from CVAT, and any annotation edited
# since the last pull would differ between the two arms. A re-encoded image
# would also change its md5 and could legitimately flip subsets.
cd $HOME/flat-bug-synth/scripts/training
fb_train -c ${CONFIG} -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
