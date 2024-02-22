#!/bin/bash
# sbatch -p ghpc_gpu train.sh
fb_train.py -c ./fb_config_L40S.yaml  -d flat-bug-data/yolo/
