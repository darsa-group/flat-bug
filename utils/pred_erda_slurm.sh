#!/bin/bash
#SBATCH --job-name=predict_erda_ias
#SBATCH --output=pred_erda.out
#SBATCH --partition=GPUNodes
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --nodelist=node5
#SBATCH --gres=gpu:1

echo "Starting job"
## Start of script ## 

# List the available conda environments
source /opt/anaconda-2022.05/bin/activate

# Activate the conda environment
conda activate /home/altair/.conda/envs/test

# Move to the directory where the code is located
cd /home/altair/flat-bug

# Run the code
python src/bin/fb_predict_erda.py -f -i AMI/storage/ias/slovakia -o test/ias_output -w best.pt -s 0.5 -p \\/[^\\/\\.]*snapshot[\\/\\.]*\\.jpg$ -n 10

## End of script ##
echo "Job finished"
exit 0
