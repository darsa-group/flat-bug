#!/bin/bash
python scripts/experiments/compare_backbone_sizes/orchestrate.py \
    -i "\$SLURM_TMPDIR/fb_yolo" \
    -o "$HOME/project/output/new_compare_backbone_sizes" \
    --soft \
    --slurm \
    slurm_setup=setup.txt \
    gres=gpu:t4:1 \
    cpus_per_task=12 \
    mem=32GB \
    time=02:00:00
