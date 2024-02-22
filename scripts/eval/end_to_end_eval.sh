#!/bin/bash
## USAGE:
# -w weights (MANDATORY): the path to the weights file.
# -d directory (MANDATORY): the directory where the data is located and where the results will be saved the directory should have a "reference" directory with the ground truth json in "reference/gt/instances_default.json" and the images in "reference/val".
 
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

# Check that our current directory ends with "/flat-bug"
if [[ $PWD != */flat-bug ]]; then
    echo "Please run this script from the flat-bug directory"
    exit 1
fi

# Prepare the environment for end-to-end evaluation
python prepare_environment_for_end_to_end_eval.py --dir "$DIR"
# Run the model on the validation set
fb_predict.py -i "$DIR/reference/val" -w "$WEIGHTS" -o "$DIR/output" -p **.jpg -f --gpu cuda:0 --no-crops --no-overviews --verbose
# Compare the predictions with the ground truth
fb_eval.py -p "$DIR/output/**/**.json" -g "$DIR/reference/gt/instances_default.json" -I "$DIR/reference/val" -P -o "$DIR/eval"
# Produce the evaluation metrics and figures
Rscript prototypes/eval-metrics.R --input_directory "$DIR/eval" --output_directory "$DIR/results"
# Clean up the environment
python ./clean_up_end_to_end_eval.py --dir "$DIR" --clear-all
