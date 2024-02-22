#!/bin/bash
#SBATCH --job-name=fb_end_to_end_eval
#SBATCH --output=eval.out
#SBATCH --partition=GPUNodes
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --nodelist=node5
#SBATCH --gres=gpu:1

echo "Starting job"
## Start of script ## 

# The next three commands are only configured for my user on the ECE GPU cluster

# List the available conda environments
source /opt/anaconda-2022.05/bin/activate

# Activate the conda environment
conda activate /home/altair/.conda/envs/test

# Move to the directory where the code is located
cd /home/altair/flat-bug

# Run the code
# python src/bin/fb_predict.py -i dev/reference/val -w model_snapshots/fb_2024-02-09_best.pt -o dev/output --no-crops -p **.jpg -f --gpu cuda:0 --verbose
bash prototypes/end_to_end_eval.sh -w model_snapshots/fb_2024-02-19_best.pt -d dev

## End of script ##
echo "Job finished"
exit 0
