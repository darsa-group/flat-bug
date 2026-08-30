#!/bin/bash
# PROTOTYPE - four-arm semantic-segmentation experiment on AU GHPC.
#
#   A  convnext  plain          sbatch --export=ALL,ARM=convnext_plain  train_semseg_GHPC.sh
#   B  convnext  1/sqrt(area)   sbatch --export=ALL,ARM=convnext_instw  train_semseg_GHPC.sh
#   C  mit_b2 + distance map, plain          ARM=mit_plain
#   D  mit_b2 + distance map, 1/sqrt(area)   ARM=mit_instw
#
# All four share: seam weighting w0=60, synthetic touching scenes p=0.4, and the blur /
# noise / rotation augmentations. They differ only in encoder+distance-map and in whether
# the loss is weighted by 1/sqrt(instance area), so the 2x2 is attributable.

#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=64000
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:1
#SBATCH -J fb_semseg

source ~/.venv/bin/activate
export PYTHONPATH=$HOME/flat-bug-synth/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=$HOME/flatbug-dir
P=$HOME/flat-bug-synth/prototypes/semseg
OUT=$ROOT/semseg/${ARM}_$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p "$OUT"

case "$ARM" in
  convnext_plain) EXTRA="--encoder tu-convnext_tiny" ;;
  convnext_instw) EXTRA="--encoder tu-convnext_tiny --inst-weight" ;;
  mit_plain)      EXTRA="--encoder mit_b2 --dist-channel" ;;
  mit_instw)      EXTRA="--encoder mit_b2 --dist-channel --inst-weight" ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac

python $P/train.py \
  -d ${ROOT}/flat-bug-data/yolo/insects -o "$OUT" \
  -e 90 --steps 4000 --val-steps 800 -b 8 --workers 12 \
  --seam-weight 60 --seam-sigma 6 \
  --synth-bank ${ROOT}/synth/crop_bank --synth-cache ${ROOT}/synth/cache --synth-prob 0.4 \
  --blur 0.3 --noise 0.3 --rotate 0.5 \
  $EXTRA
