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
# Running this as a batch job on the GHPC CPU partition needs a headless cv2: sky008 and
# friends have no libGL.so.1, so the venv's opencv-python fails to import and fb_prepare_data
# dies before it starts. The GPU nodes DO have libGL, so the venv is left alone and batch jobs
# opt in instead:
#     pip install --no-deps --target ~/opt/cv2-headless opencv-python-headless==4.13.0.92
#     export PYTHONPATH=~/opt/cv2-headless:$REPO/src
set -eu
ROOT=${1:-/usr/home/qgg/qgeiss/flatbug-dir3}
SECRETS=${2:-/usr/home/qgg/qgeiss/flat-bug/scripts/training/.secrets.yaml}
CLONE=$ROOT/coco
YOLO=$ROOT/flat-bug-data
echo "[$(date +%H:%M)] cloning CVAT project into $CLONE"
fb_clone_data -s "$SECRETS" -o "$CLONE" -f
echo "[$(date +%H:%M)] cloned: $(ls "$CLONE" | wc -l) sub-datasets"
echo "[$(date +%H:%M)] preparing YOLO dataset in $YOLO"
# -o "$YOLO/yolo": fb_prepare_data writes <out>/data.yaml and <out>/insects, so the "yolo"
# level has to be asked for explicitly. Every sbatch and flatbug-dir2 expect
# flat-bug-data/yolo/insects, and passing "$YOLO" instead produced a tree one level
# shallower that the -d path could not find.
fb_prepare_data -i "$CLONE" -o "$YOLO/yolo" -p 0.15 -f
echo "[$(date +%H:%M)] train $(ls "$YOLO"/yolo/insects/images/train | wc -l) / val $(ls "$YOLO"/yolo/insects/images/val | wc -l)"
echo "[$(date +%H:%M)] dataset prefixes present:"
ls "$YOLO"/yolo/insects/images/train | sed 's/_[^_]*$//' | sed 's/\(.*\)_.*/\1/' | cut -d_ -f1 | sort -u | sed 's/^/    /'
echo RESYNC_DONE
