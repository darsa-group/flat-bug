#!/bin/bash
## USAGE:
# -w weights (MANDATORY): the path to the weights file.
# -d directory (MANDATORY): the directory where the data is located and where the results will be saved the directory
#                           should have a "reference" directory with the ground truth json in
#                           "instances_default.json" and the matching images".
 
WEIGHTS=""
DIR=""

# Parse the command line arguments
while getopts "w:d:" flag
do
    case "${flag}" in
        w) WEIGHTS=${OPTARG};;
        d) DIR=${OPTARG};;
        *) echo "Usage: $0 -w weights -d directory"; exit 1;;
    esac
done

# Check for mandatory options
if [[ -z "$WEIGHTS" || -z "$DIR" ]]; then
    echo "Both -w (weights) and -d (directory) options are required."
    exit 1
fi

# fixme, R dependencies should be checked before
# fixme retrieve hyper parameters from CLI


MODEL_HASH=$(md5sum ${WEIGHTS} | cut -c1-8)
DATE=$(date -u  +"%Y-%m-%dT%H-%M-%SZ")
COMMIT_HASH=$(git rev-list --max-count=1 HEAD | cut -c1-8)

ID=${DATE}_${MODEL_HASH}_${COMMIT_HASH}_${SLURM_JOB_ID}

METADATA_FILE=${DIR}/${ID}/metadata.yml


echo "Saving outputs and results in ${DIR}/${ID}"

mkdir -p ${DIR}/${ID}/eval
mkdir -p ${DIR}/${ID}/preds
mkdir -p ${DIR}/${ID}/results

echo "date: ${DATE}" > ${METADATA_FILE}
echo "model: ${WEIGHTS}" >> ${METADATA_FILE}
echo "model_hash: ${MODEL_HASH}" >> ${METADATA_FILE}
echo "commit: ${COMMIT_HASH}" >> ${METADATA_FILE}
# todo add hyperparam here
 #todo, copy inference time to results?

# Run the model on the validation set
fb_predict.py -i "${DIR}/reference" -w "${WEIGHTS}" -o "$DIR/${ID}/preds" -p '**.jpg' -f --no-crops --gpu cuda:0 --no-crops --verbose &&
# Compare the predictions with the ground truth
fb_eval.py -p "${DIR}/${ID}/preds/**/**.json" -g "${DIR}/reference/instances_default.json" -I "${DIR}/reference" -P -o "${DIR}/${ID}/eval" &&
# Produce the evaluation metrics and figures
Rscript scripts/eval/eval-metrics.R --input_directory "${DIR}/${ID}/eval" --output_directory "${DIR}/${ID}/results"
