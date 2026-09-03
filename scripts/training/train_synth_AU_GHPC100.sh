#!/bin/bash
# Does synthetic touching-instance scenes help? 100-epoch A/B on the halo-fixed code.
#
#   sbatch --export=ALL,ARM=on  train_synth50_AU_GHPC.sh   # fb_synth_prob 0.4
#   sbatch --export=ALL,ARM=off train_synth50_AU_GHPC.sh   # fb_synth_prob 0.0
#
# Both arms run the SAME source tree and the same config file; ARM only overrides
# fb_synth_prob on the command line, so nothing else can drift between them.

#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=96000
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH -J fb_synth
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_synth_%x_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_synth_%x_%j.err

source ~/.venv/bin/activate
export PYTHONPATH=$HOME/flat-bug-synth-test/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$ARM" in
  on)  PROB=0.4 ;;
  off) PROB=0.0 ;;
  *) echo "set ARM=on or ARM=off"; exit 1 ;;
esac

ROOT=$HOME/flatbug-dir
SRC=$HOME/flat-bug-synth-test
NAME=fb_synth_${ARM}_$(date +"%Y-%m-%d_%H-%M-%S")

cd $SRC/scripts/training
fb_train -c $SRC/scripts/training/fb_config_synth_GHPC100.yaml \
  -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME} --fb_synth_prob ${PROB}
