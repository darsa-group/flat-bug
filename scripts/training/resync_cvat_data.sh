#!/bin/bash
# Rebuild the training corpus from CVAT + S3. Run this BEFORE the 300-epoch job.
#
# Two stages:
#   fb_clone_data    pulls COCO annotations from every COMPLETED task in CVAT project 321224
#                    and the matching images from S3, into one directory per sub-dataset
#   fb_prepare_data  joins those into a single YOLO dataset, splitting train/val by md5 of the
#                    filename so the split is stable across resyncs
#
# What gets in: a task whose status is 'completed', plus - for a task still in annotation -
# the frames belonging to its individually completed JOBS, minus any frame marked deleted.
# So setting a task to a non-completed status no longer removes it wholesale; it narrows it to
# the work actually finished. As of 2026-09-10:
#     broto2025        100f  (validation)   <- excluded: no completed job
#     ArTaxOr         1050f  (annotation)   <- excluded: superseded by artaxor-seg/-bbox
#     aquamonitor     2981f  (annotation)   <- excluded: no completed job
#     2025-agrivolt    500f  (annotation)   <- PARTIAL, completed jobs only
#     MAMBOcrops-bbox 7514f  (annotation)   <- PARTIAL, completed jobs only
set -eu
ROOT=${1:-/usr/home/qgg/qgeiss/flatbug-dir3}
SECRETS=${2:-/usr/home/qgg/qgeiss/flat-bug/scripts/training/.secrets.yaml}
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
