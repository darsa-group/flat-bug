#!/usr/bin/env bash

usage () {
  cat <<EOF
Usage: $0 -w weights -d directory [-c config.yaml] [-l local_directory] [-o output_directory] [-g PyTorch_device_string] [-p inference_file_regex_pattern]
    -w weights (MANDATORY):
        The path to the weights file.

    -d directory (MANDATORY): 
        The directory where the data is located, the directory should have a 'reference' directory 
        with the ground truth json in 'instances_default.json' and the matching images'.

    -c config (OPTIONAL):
        The path to the config file.

    -l local_directory (OPTIONAL): 
        The path to the local directory where the ground truth json is located.
        If not supplied, it is assumed to be a file named 'instances_default.json' in the directory.

    -o output_directory (OPTIONAL): 
        The path to the output directory where the results will be saved.
        If not supplied, it is assumed to be created in the parent directory of the directory (i.e. adjacent to the directory).

    -g PyTorch_device_string (OPTIONAL): 
        The PyTorch device string to use for inference.
        If not supplied, it is assumed to be cuda:0.

    -p inference_file_regex_pattern (OPTIONAL): 
        The regex pattern to use for selecting the inference files.
        If not supplied, it is assumed to be the default pattern.
EOF
}

WEIGHTS=""
CONFIG=""
DIR=""
GPU=""
LDIR=""
ODIR=""
IPAT=""

# Parse the command line arguments
while getopts "w:d:c:l:o:g:p:" flag
do
    case "${flag}" in
        w) WEIGHTS=${OPTARG};;
        d) DIR=${OPTARG};;
        c) CONFIG=${OPTARG};;
        l) LDIR=${OPTARG};;
        o) ODIR=${OPTARG};;
        g) GPU=${OPTARG};;
        p) IPAT=${OPTARG};;
        *) usage; return 1 2>/dev/null; exit 1;;
    esac
done

# If LDIR is not supplied, set it equal to DIR
if [[ -z "$LDIR" ]]; then
    LDIR="$DIR/instances_default.json"
fi

# Check for mandatory options
if [[ -z "$WEIGHTS" || -z "$DIR" ]]; then
    usage
    return 1 2>/dev/null
    exit 1
fi

# fixme, R dependencies should be checked before
# fixme retrieve hyper parameters from CLI

MODEL_HASH=$(md5sum ${WEIGHTS} | cut -c1-8)
DATE=$(date -u  +"%Y-%m-%dT%H-%M-%SZ")
COMMIT_HASH=$(git rev-list --max-count=1 HEAD | cut -c1-8)

ID=${DATE}_${MODEL_HASH}_${COMMIT_HASH}_${SLURM_JOB_ID}

if [ -z "$ODIR" ]; then
    ODIR="$DIR/../${ID}"
fi

METADATA_FILE="${ODIR}/metadata.yml"


echo "Saving outputs and results in ${ODIR}"

mkdir -p "${ODIR}/eval"
mkdir -p "${ODIR}/preds"
mkdir -p "${ODIR}/results"

echo "date: ${DATE}" > ${METADATA_FILE}
echo "model: ${WEIGHTS}" >> ${METADATA_FILE}
echo "model_hash: ${MODEL_HASH}" >> ${METADATA_FILE}
echo "commit: ${COMMIT_HASH}" >> ${METADATA_FILE}
# todo add hyperparam here

# Record start time
START_TIME=$(date +%s)

# Run the model on the validation set
# PREDICT_CMD="fb_predict.py -i \"${DIR}\" -w \"${WEIGHTS}\" -o \"${ODIR}/preds\"${GPU}${IPAT} --no-crops"
PREDICT_CMD=(fb_predict -i "${DIR}" -w "${WEIGHTS}" -o "${ODIR}/preds" --no-crops --no-overviews --fast)
if [[ -n "$GPU" ]]; then
    PREDICT_CMD+=("--gpu" "${GPU}")
fi
if [[ -n "$IPAT" ]]; then
    PREDICT_CMD+=("-p" "${IPAT}")
fi
if [[ -n "$CONFIG" ]]; then
    PREDICT_CMD+=("--config" "${CONFIG}")
fi
printf -v PREDICT_CMD_STR ' %q' "${PREDICT_CMD[@]}"
echo "Executing inference with:${PREDICT_CMD_STR}"
"${PREDICT_CMD[@]}" &&

# Compare the predictions with the ground truth
#EVAL_CMD="fb_eval.py -p \"${ODIR}/preds/coco_instances.json\" -g \"$LDIR\" -I \"${DIR}\" -P  -c -o \"${ODIR}/eval\""
EVAL_CMD=(fb_evaluate -p "${ODIR}/preds/coco_instances.json" -g "$LDIR" -I "${DIR}" -P -c -o "${ODIR}/eval" --combine)
if [[ -n "$CONFIG" ]]; then
    EVAL_CMD+=("--config" "${CONFIG}")
fi
printf -v EVAL_CMD_STR ' %q' "${EVAL_CMD[@]}"
echo "Executing evaluation with:${EVAL_CMD_STR}"
"${EVAL_CMD[@]}" &&

# Produce the evaluation metrics and figures
# METRIC_CMD="Rscript scripts/eval/eval-metrics.R --input_directory \"${ODIR}/eval\" --output_directory \"${ODIR}/results\""
METRIC_CMD=(Rscript scripts/eval/eval-metrics.R --input_directory "${ODIR}/eval" --output_directory "${ODIR}/results")
printf -v METRIC_CMD_STR ' %q' "${METRIC_CMD[@]}"
echo "Executing evaluation metrics and figure creation with:${METRIC_CMD_STR}"
"${METRIC_CMD[@]}" &&

# Record end time
END_TIME=$(date +%s)
EVAL_TIME=$((END_TIME - START_TIME))
# Print time taken in H:M:S
printf "Time taken: %02d:%02d:%02d\n" "$((EVAL_TIME/3600))" "$((EVAL_TIME%3600/60))" "$((EVAL_TIME%60))"
# Save time taken in metadata file
echo "time: ${EVAL_TIME}" >> ${METADATA_FILE}

echo "Evaluation complete. Results saved in ${ODIR}/results"
