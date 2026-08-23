#!/bin/bash
# Stage 1: harvest the crop bank and cut/screen the background cache.
# Must finish before either training arm starts; the training script submits
# itself with --dependency=afterok on this job.

#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=128000
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1                # only for the flatbug screening pass
#SBATCH -J fb_synth_prep

source ~/.venv/bin/activate
export PYTHONPATH=$HOME/flat-bug-synth/src   # wins over the editable install in ~/flat-bug

ROOT=$HOME/flatbug-dir
DATA=$ROOT/flat-bug-data/yolo
OUT=$ROOT/synth
WEIGHTS=${FB_SCREEN_WEIGHTS:-$ROOT/flat_bug_M_v2.pt}
mkdir -p $OUT

python $HOME/flat-bug-synth/scripts/synthetic/build_crop_bank.py \
    -d $DATA -o $OUT/crop_bank \
    --min-size 32 --max-per-image 25 --max-per-dataset 700 --max-crops 20000 --seed 3

python - <<PY
from flat_bug.synthetic import build_cache
build_cache(data_dir="$DATA", cache_dir="$OUT/cache", split="train",
            tile=1536, per_dataset=6,
            screen_weights="$WEIGHTS", screen_device="cuda:0")
PY

echo "prep done: $(ls $OUT/crop_bank/crops | wc -l) crops"
ls -la $OUT/cache
