#!/bin/bash
python scripts/experiments/compare_models.py \
    -i "\$SLURM_TMPDIR/fb_yolo/insects/images/val" \
    -g "\$SLURM_TMPDIR/fb_yolo/insects/labels/val/instances_default.json" \
    -o "$HOME/scratch/output/compare_backbone_v100/eval" \
    -d "$HOME/scratch/output/compare_backbone_v100/fb_compare_backbone_sizes_*" \
    --soft \
    --dry_run
