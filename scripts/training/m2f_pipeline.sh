#!/bin/bash

# Shared body of the Mask2Former training + evaluation jobs.
#
# Site-agnostic on purpose: the cluster-specific parts (SBATCH header, how the
# environment is activated) live in the per-site wrappers that source this, so
# the two cannot drift apart. Run it directly for a non-SLURM machine.
#
# Everything is overridable from the environment, so a sweep needs no edits:
#   sbatch --export=ALL,IMAGE_SIZE=512,BATCH=8 scripts/training/train_m2f_AU_GHPC.sh
#
#   PREPARE_DATA=1   also run fb_clone_data + fb_prepare_data first (slow)
#   IMAGE_SIZE       training/tile resolution      (default 1024)
#   BATCH            samples per step              (default 2)
#   EPOCHS           number of epochs              (default 20)
#   BACKGROUND_DIR   insect-free images to mix in  (default: none)
#   EXCLUDE_DATASETS space-separated sub-dataset prefixes to hold out
#   RESUME           checkpoint to resume from
#   RUN_EVAL=0       skip the predict+evaluate stage

set -euo pipefail


ROOT=${ROOT:-~/flatbug-dir}
DATA=${DATA:-${ROOT}/flat-bug-data/yolo/}
NAME=${NAME:-m2f_$(date +"%Y-%m-%d_%H-%M-%S")}
OUT=${OUT:-${ROOT}/runs/${NAME}}

IMAGE_SIZE=${IMAGE_SIZE:-1024}
BATCH=${BATCH:-2}
EPOCHS=${EPOCHS:-20}
WORKERS=${WORKERS:-8}
# The model has 100 object queries; samples with more instances cannot be matched.
MAX_INSTANCES=${MAX_INSTANCES:-100}
RUN_EVAL=${RUN_EVAL:-1}

# transformers is an optional extra, so fail loudly rather than 40 minutes in.
python - <<'PY' || { echo "ERROR: install the extra first:  pip install -e '.[mask2former]'" >&2; exit 1; }
import transformers  # noqa: F401
PY

if [[ "${PREPARE_DATA:-0}" == "1" ]]; then
    echo "== Cloning and preparing data =="
    fb_clone_data -s ~/flat-bug/scripts/training/.secrets.yaml -o "${ROOT}/flat-bug-data/pre-pro/"
    fb_prepare_data -i "${ROOT}/flat-bug-data/pre-pro/" -o "${ROOT}/flat-bug-data/yolo/" -f
fi

mkdir -p "${OUT}"

TRAIN_CMD=(fb_train_m2f
    -d "${DATA}"
    -o "${OUT}"
    --image-size "${IMAGE_SIZE}"
    --batch "${BATCH}"
    --epochs "${EPOCHS}"
    --workers "${WORKERS}"
    --max-instances "${MAX_INSTANCES}"
    --val
    --device cuda:0
)
# `if` rather than `test && append`: under `set -e` a failing test as the last
# statement of a script aborts it, which is a trap for anyone reordering these.
if [[ -n "${BACKGROUND_DIR:-}" ]]; then
    TRAIN_CMD+=(--background-dir "${BACKGROUND_DIR}")
fi
if [[ -n "${RESUME:-}" ]]; then
    TRAIN_CMD+=(--resume "${RESUME}")
fi
if [[ -n "${EXCLUDE_DATASETS:-}" ]]; then
    # shellcheck disable=SC2206  # deliberate word splitting: space-separated prefixes
    TRAIN_CMD+=(--exclude-datasets ${EXCLUDE_DATASETS})
fi

{
    echo "name: ${NAME}"
    echo "date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "commit: $(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "job: ${SLURM_JOB_ID:-none}"
    echo "node: $(hostname)"
    echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    echo "data: ${DATA}"
    echo "command: ${TRAIN_CMD[*]}"
} | tee "${OUT}/metadata.yml"

echo "== Training =="
"${TRAIN_CMD[@]}"

if [[ "${RUN_EVAL}" != "1" ]]; then
    echo "Done (evaluation skipped). Results in ${OUT}"
    exit 0
fi

# Score the checkpoint the same way the YOLO pipeline is scored: tile, stitch,
# then compare the stitched result against the COCO ground truth.
VAL_IMAGES=${VAL_IMAGES:-${DATA}/insects/images/val}
VAL_GT=${VAL_GT:-${DATA}/insects/labels/val/instances_default.json}
N_EVAL=${N_EVAL:-200}

echo "== Predicting on ${N_EVAL} validation images =="
mkdir -p "${OUT}/preds" "${OUT}/eval"
fb_predict_m2f \
    -i "${VAL_IMAGES}" \
    -o "${OUT}/preds" \
    -w "${OUT}/best.pt" \
    -n "${N_EVAL}" \
    --batch "${BATCH}" \
    --no-crops --no-overviews \
    --device cuda:0

echo "== Evaluating =="
fb_evaluate \
    -p "${OUT}/preds/**/metadata_*.json" \
    -g "${VAL_GT}" \
    -I "${VAL_IMAGES}" \
    -o "${OUT}/eval" \
    --combine

echo "Done. Results in ${OUT}"
