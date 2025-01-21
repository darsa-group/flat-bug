#!/bin/bash

time_str () {
  date +"%d-%m-%Y_%H-%M-%S"
}

# Define job
BB_SIZE="" # Set size of YOLOv8 backbone model
RESUME_ERDA_DIR="" # Set to "" if the job isn't restarting a cancelled job
EXP_TYPE="leave_two_out_dataset_mapping" # Choose experiment type
EXP_SHORT_NAME="leave_two_out" # Set experiment 
ERDA_SHARELINK_ID="<INSERT>" # See page 7-8 in the ERDA user guide: https://erda.au.dk/public/au-erda-user-guide.pdf (if you do not use ERDA the synchronization code must be adjusted to your cloud storage service)

# Initialize standard variables
OUTPUT_DIR="/work/my_output"
mkdir -p $OUTPUT_DIR
if [[ -z $BB_SIZE ]]; then
  ERDA_DIR="${EXP_TYPE}_ucloud_output_$(time_str)"
else
  ERDA_DIR="${EXP_TYPE}_${BB_SIZE}_ucloud_output_$(time_str)"
fi
INIT_DIR="/work/fb_init"
FB_DIR="$INIT_DIR/fb_yolo"

echo "($(time_str)) Job started..."
echo "##########################################################################################"
echo "Environment state:"
printenv
echo "##########################################################################################"
echo ""
echo "($(time_str)) Contents of /work:"
ls -1a /work

## Resume experiment
if [[ ! -z "$RESUME_ERDA_DIR" ]]; then
  echo "($(time_str)) Pulling cancelled job..."
  ERDA_DIR=$RESUME_ERDA_DIR
  lftp -c "open -u $ERDA_SHARELINK_ID,$ERDA_SHARELINK_ID -p 2222 sftp://io.erda.au.dk; mirror --parallel=8 --verbose $RESUME_ERDA_DIR $OUTPUT_DIR;" >> /work/lftp.log 2>&1
  echo "($(time_str)) Cancelled job pulled into $OUTPUT_DIR:"
  ls -1a $OUTPUT_DIR
fi

## Setup data synchronization and job cleanup
LOCKFILE="/tmp/sync.lock"
SYNC_CMD="lftp -c \"open -u $ERDA_SHARELINK_ID,$ERDA_SHARELINK_ID -p 2222 sftp://io.erda.au.dk; mkdir -fp $ERDA_DIR; mirror --reverse --verbose $OUTPUT_DIR $ERDA_DIR;\""
update_crontab () {
  (crontab -l 2>/dev/null; echo "$1") | crontab -
}

cleanup () {
  echo "($(time_str)) Job finishing..."
  
  echo "($(time_str)) Stopping cronjobs."
  (crontab -l | grep -v "$SYNC_CMD") | crontab -
  echo "($(time_str)) Killing running synchronization processes." 
  pkill -f "$LOCKFILE"
  pkill -f "lftp"
  sleep 1 # Sleep for 1 second to allow processes to die
  echo "($(time_str)) Redirecting stdout/stderr back to original streams."

  echo "($(time_str)) Running final synchronization..."
  eval "flock $LOCKFILE $SYNC_CMD"
  echo "($(time_str)) Finished final synchronization."
  echo "($(time_str)) Job finished."  
}

# Cleanup
echo "($(time_str)) Trapping cleanup function."
trap cleanup EXIT SIGINT SIGTERM
# Synchronization with `lftp` and `cron`
echo "($(time_str)) Starting cron ERDA synchronization job."
sudo cron # Remember to start the `cron` daemon
touch /work/lftp.log # Initialize `lftp` log-file
update_crontab "*/30 * * * * flock -n $LOCKFILE $SYNC_CMD >> /work/lftp.log 2>&1"

echo "($(time_str)) Scheduled cron jobs:"
crontab -l

#############################################################################################
################################### EXPERIMENT CODE BELOW ###################################
#############################################################################################

### Run the experiments

# Swap some files to set some UCloud specific settings in flat-bug
rm flat-bug/scripts/experiments/compare_backbone_sizes/default.yaml
cp $INIT_DIR/ucloud_flatbug_backbone.yaml flat-bug/scripts/experiments/compare_backbone_sizes/default.yaml

rm flat-bug/scripts/experiments/leave_one_out_cv_and_finetuning/default.yaml
cp $INIT_DIR/ucloud_flatbug_leave1out.yaml flat-bug/scripts/experiments/leave_one_out_cv_and_finetuning/default.yaml

rm flat-bug/scripts/experiments/leave_two_out_dataset_mapping/default.yaml
cp $INIT_DIR/ucloud_flatbug_leave2out.yaml flat-bug/scripts/experiments/leave_two_out_dataset_mapping/default.yaml

# rm flat-bug/scripts/experiments/leave_one_out_cv_and_finetuning/orchestrate.py
# cp $INIT_DIR/leave_one_out_cv_and_finetuning_orchestrate_modified.py flat-bug/scripts/experiments/leave_one_out_cv_and_finetuning/orchestrate.py

echo "------------------------------------------------------------------------------------------------------------"
echo "($(time_str)) Starting experiment..."
echo ""

TRAIN_DIR="$OUTPUT_DIR/$EXP_TYPE"
cd "flat-bug"
mkdir -p "$TRAIN_DIR"

# # Start SLURM cluster
# init_slurm_cluster

# Run experiment
if [[ -z $BB_SIZE ]]; then
  python scripts/experiments/$EXP_TYPE/orchestrate.py -i "$FB_DIR" -o "$TRAIN_DIR" --try-resume --devices 0 1 2 3
else
  python scripts/experiments/$EXP_TYPE/orchestrate.py -i "$FB_DIR" -o "$TRAIN_DIR" --try-resume --devices 0 1 2 3 --try-resume --sizes $BB_SIZE
fi

# # Wait for experiment
# bash /work/fb_init/wait_squeue.sh

# Run eval
mkdir -p "$TRAIN_DIR/eval"
python scripts/experiments/compare_models.py -i "$FB_DIR/insects/images/val" -g "$FB_DIR/insects/labels/val/instances_default.json" -d "$TRAIN_DIR/fb_$EXP_SHORT_NAME*" -o "$TRAIN_DIR/eval" --devices 0 1 2 3
# --slurm cpus-per-task=48 time=05:00:00 gres=gpu:1

# # Wait for eval
# bash /work/fb_init/wait_squeue.sh

echo ""
echo "($(time_str)) Finished experiment."
echo "------------------------------------------------------------------------------------------------------------"

