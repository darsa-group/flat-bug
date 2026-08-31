#!/bin/bash
# PROTOTYPE - distance-saturating background penalty, dose-response on GHPC.
#
# Identical to the finished convnext_plain / convnext_instw arms except for --bg-gamma, so
# those two runs serve as the gamma=0 reference and the comparison is a dose-response
# rather than a single point.
#
#   sbatch --export=ALL,ARM=bg4       train_bgpen_GHPC.sh
#   sbatch --export=ALL,ARM=bg8       train_bgpen_GHPC.sh
#   sbatch --export=ALL,ARM=bg4_instw train_bgpen_GHPC.sh
#
# Motivation: on the trained convnext_plain checkpoint, 39.2% of false-positive PIXELS lie
# more than 20px from any annotated insect and 20.5% beyond 100px - free-standing
# foreground that a uniform loss treats exactly like a 1px halo. The weight ramps linearly
# to 50px and is flat beyond, and only the background portion is normalised, so foreground
# weight is untouched and the change is a redistribution within background.

#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=64000
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:1
#SBATCH -J fb_bgpen

source ~/.venv/bin/activate
export PYTHONPATH=$HOME/flat-bug-synth/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=$HOME/flatbug-dir
P=$HOME/flat-bug-synth/prototypes/semseg
OUT=$ROOT/semseg/${ARM}_$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p "$OUT"

case "$ARM" in
  bg4)        EXTRA="--bg-gamma 4" ;;
  bg8)        EXTRA="--bg-gamma 8" ;;
  bg4_instw)  EXTRA="--bg-gamma 4 --inst-weight" ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac

python $P/train.py \
  -d ${ROOT}/flat-bug-data/yolo/insects -o "$OUT" \
  -e 90 --steps 4000 --val-steps 800 -b 8 --workers 12 \
  --encoder tu-convnext_tiny \
  --seam-weight 60 --seam-sigma 6 \
  --synth-bank ${ROOT}/synth/crop_bank --synth-cache ${ROOT}/synth/cache --synth-prob 0.4 \
  --blur 0.3 --noise 0.3 --rotate 0.5 \
  --bg-saturate 50 \
  $EXTRA
