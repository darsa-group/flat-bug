#!/bin/bash

#SBATCH -p ghpc_gpu                 # Name of the queue
#SBATCH -N 1                       # Number of nodes(DO NOT CHANGE)
#SBATCH -n 16                       # Number of CPU cores
#SBATCH --mem=64000                 # Memory in MiB(10 GiB = 10 * 1024 MiB)
#SBATCH -t 96:00:00 
#SBATCH --gres=gpu:1           # Request 1 GPU (DO NOT CHANGE)
#SBATCH -J flatbug_gpu             # Job name

source ~/.venv/bin/activate
CONFIG=fb_config_M40S_GHPC.yaml
ROOT=~/flatbug-dir/

NAME=fb_M_$(date +"%Y-%m-%d_%H-%M-%S")
# source ${ROOT}/.venv/bin/activate

fb_clone_data -s ~/flat-bug/scripts/training/.secrets.yaml -o ${ROOT}/flat-bug-data/pre-pro/
fb_prepare_data -i ${ROOT}/flat-bug-data/pre-pro/  -o ${ROOT}/flat-bug-data/yolo/ -f
fb_train -c ${CONFIG} -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}

