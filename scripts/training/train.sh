#!/bin/bash

# # SBATCH -p ghpc_gpu                 # Name of the queue
# # SBATCH -N 1                       # Number of nodes(DO NOT CHANGE)
# # SBATCH -n 16                       # Number of CPU cores
# # SBATCH --mem=64000                 # Memory in MiB(10 GiB = 10 * 1024 MiB)
# # SBATCH -t 96:00:00 

#CONFIG=fb_config_L40S_fine-tune.yaml
CONFIG=fb_config_N40S.yaml

# Derive model size letter (N/S/M/L/X) from config filename, e.g. fb_config_N40S.yaml -> N
SIZE="N"
NAME="fb_${SIZE}_$(date +%Y-%m-%d_%H-%M-%S)"

ROOT=~/Desktop/flatbug-dir/
fb_clone_data -s ~/flat-bug/repos/scripts/training/.secrets.yaml -o ${ROOT}/flat-bug-data/pre-pro/
fb_prepare_data -i ${ROOT}/flat-bug-data/pre-pro/  -o ${ROOT}/flat-bug-data/yolo/ -f
fb_train -c ${CONFIG} -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}


fb_prepare_data.py -i ${ROOT}/flat-bug-data/pre-pro/  -o ${ROOT}/flat-bug-data/yolo/ -f
# fb_train.py -c ${ROOT}/scripts/training/${CONFIG} -d ${ROOT}/flat-bug-data/yolo/
fb_train -c ${ROOT}/scripts/training/${CONFIG} -d dev/fb_yolo
