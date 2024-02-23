#!/bin/bash

#SBATCH -p ghpc_gpu                 # Name of the queue
#SBATCH -N 1                       # Number of nodes(DO NOT CHANGE)
#SBATCH -n 16                       # Number of CPU cores
#SBATCH --mem=32000                 # Memory in MiB(10 GiB = 10 * 1024 MiB)
#SBATCH -t 48:00:00 

ROOT=/usr/home/qgg/$USER
source ${ROOT}/.venv/bin/activate
fb_train.py -c ${ROOT}/flat-bug/scripts/training/fb_config_L40S.yaml  -d ${ROOT}/flat-bug-data/yolo/
