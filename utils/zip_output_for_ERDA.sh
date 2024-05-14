#!/bin/bash

# Define the root directory
root_dir="/home/altair/flat-bug/test/ias_output_portugal_bulgaria"

# Signal that the zipping process has started
echo "Zipping process started for $root_dir"

# Loop through each subdirectory in root
for country in "$root_dir"/*/ ; do
    echo "Current country: $country"
    for trap in "$country"/*/ ; do
        echo "Current trap: $trap"
        # Loop through each subfolder ("X", "Y", "Z") in the current folder
        for subfolder in "$trap"*/; do
            # Zip the subfolder and store the zip file in the same directory as the subfolder
            # (cd "$folder" && zip -0 -r "${subfolder##*/}.zip" "${subfolder##*/}") &
            # echo "(cd \"$folder\" && zip -0 -r \"${subfolder##*/}.zip\" \"${subfolder##*/}\")"
            subfolder_trimmed="${subfolder%/}"
            subfolder_name="${subfolder_trimmed##*/}"
            echo "Subfolder: $subfolder"
            echo "Zipping: $subfolder_name"
            (cd "$trap" && zip -0 -r "$subfolder_name.zip" "$subfolder_name" > /dev/null 2>&1) &
        done
    done
done

# Wait for all background zipping processes to finish
wait
# Signal that all zipping processes have finished
echo "All zipping processes finished"
