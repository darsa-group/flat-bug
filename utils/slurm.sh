#!/bin/bash
#SBATCH --job-name=fb_end_to_end_eval
#SBATCH --output=eval.out
#SBATCH --partition=GPUNodes
#SBATCH --time=23:00:00
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --nodelist=node5
#SBATCH --gres=gpu:0

echo "Starting job"
## Start of script ## 

# Activate micromamba environment
source /opt/anaconda-2022.05/bin/activate
eval "$(micromamba shell hook --shell=bash)"
micromamba activate fbc

# Move to the directory where the code is located
cd /home/altair/flat-bug

# Run the code
# fb_tune \
#     -i dev/fb_yolo/insects/images/train \
#     -a dev/fb_yolo/insects/labels/train/instances_default.json \
#     -o dev/tuning_single_v22 \
#     -w runs/segment/large/weights/best.pt \
#     --max-iter 10 \
#     --pop-size 1 \
#     --method bayesian \
#     --gpu cuda:0 \
#     -s 1 \
#     -n 3
fb_train \
    -d dev/fb_yolo \
    -c utils/default_train.yaml

## End of script ##
echo "Job finished"
exit 0
