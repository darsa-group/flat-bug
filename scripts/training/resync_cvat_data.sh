#!/bin/bash
# Rebuild the training corpus from CVAT + S3. Run this BEFORE the 300-epoch job.
#
# Two stages:
#   fb_clone_data    pulls COCO annotations from every COMPLETED task in CVAT project 321224
#                    and the matching images from S3, into one directory per sub-dataset
#   fb_prepare_data  joins those into a single YOLO dataset, splitting train/val by md5 of the
#                    filename so the split is stable across resyncs
#
# What "completed" means, and why it is the exclusion mechanism: fb_clone_data keeps a task
# only if its status is 'completed' or every one of its jobs is. Setting a task to any other
# status removes it from the corpus without deleting anything. As of 2026-09-10 that excludes:
#     broto2025        100f  (validation)   <- deliberately excluded
#     2025-agrivolt    500f  (annotation)   <- still being curated
#     ArTaxOr         1050f  (annotation)   <- superseded by artaxor-seg + artaxor-bbox
#     aquamonitor     2981f  (annotation)
#     MAMBOcrops-bbox 7514f  (annotation)   <- only 3 of 38 jobs done, so it will NOT sync
# leaving 38 tasks / 21,930 images.
set -eu
ROOT=${1:-/usr/home/qgg/qgeiss/flatbug-dir3}
SECRETS=${2:-/usr/home/qgg/qgeiss/flat-bug-zoom/scripts/training/.secrets.yaml}
CLONE=$ROOT/coco
YOLO=$ROOT/flat-bug-data
echo "[$(date +%H:%M)] cloning CVAT project into $CLONE"
fb_clone_data -s "$SECRETS" -o "$CLONE" -f
echo "[$(date +%H:%M)] cloned: $(ls "$CLONE" | wc -l) sub-datasets"
echo "[$(date +%H:%M)] preparing YOLO dataset in $YOLO"
fb_prepare_data -i "$CLONE" -o "$YOLO" -p 0.15 -f
echo "[$(date +%H:%M)] train $(ls "$YOLO"/yolo/insects/images/train | wc -l) / val $(ls "$YOLO"/yolo/insects/images/val | wc -l)"
echo "[$(date +%H:%M)] dataset prefixes present:"
ls "$YOLO"/yolo/insects/images/train | sed 's/_[^_]*$//' | sed 's/\(.*\)_.*/\1/' | cut -d_ -f1 | sort -u | sed 's/^/    /'
echo RESYNC_DONE
