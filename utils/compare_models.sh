#!/bin/bash
# "$HOME/scratch/output/compare_backbone_v100/eval"
python scripts/experiments/compare_models.py \
    -i "\$SLURM_TMPDIR/fb_yolo/insects/images/val" \
    -g "\$SLURM_TMPDIR/fb_yolo/insects/labels/val/instances_default.json" \
    -o "$HOME/project/output/compare_backbone_v100/eval" \
    --tmp "\$SLURM_TMPDIR/job_output" \
    -d "$HOME/scratch/output/compare_backbone_v100/fb_compare_backbone_sizes_*" \
    --soft \
    --slurm \
    slurm_setup=setup.txt \
    gres=gpu:t4:1 \
    cpus_per_task=12 \
    mem=32GB \
    time=02:00:00
