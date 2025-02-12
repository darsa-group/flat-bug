#!/bin/bash

# This script checks the entire SLURM queue and blocks until it is empty.

# Check if squeue is available
if ! command -v squeue &> /dev/null; then
    echo "Warning: squeue command not found. Continuing without blocking."
    exit 0
fi

while true; do
    # Get the number of jobs in the entire SLURM queue, excluding the header
    job_count=$(($(squeue | wc -l) - 1))

    if [ "$job_count" -eq 0 ]; then
        echo "No jobs in the queue. Exiting."
        break
    else
        echo "$job_count jobs in the queue. Waiting..."
        # Sleep for 30 seconds before checking again
        sleep 30
    fi
done