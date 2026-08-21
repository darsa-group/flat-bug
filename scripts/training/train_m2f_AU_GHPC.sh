#!/bin/bash

#SBATCH -p ghpc_gpu                 # Name of the queue
#SBATCH -N 1                        # Number of nodes(DO NOT CHANGE)
#SBATCH -n 16                       # Number of CPU cores
#SBATCH --mem=64000                 # Memory in MiB(10 GiB = 10 * 1024 MiB)
#SBATCH -t 96:00:00
#SBATCH --gres=gpu:1                # Request 1 GPU (DO NOT CHANGE)
#SBATCH -J flatbug_m2f              # Job name
#SBATCH -o m2f_%j.out               # Stdout+stderr, %j is the job id

# Mask2Former counterpart of train_AU_GHPC.sh. Options are documented in,
# and everything is done by, m2f_pipeline.sh.

set -euo pipefail

source ~/.venv/bin/activate

exec "$(dirname "$(readlink -f "$0")")/m2f_pipeline.sh"
