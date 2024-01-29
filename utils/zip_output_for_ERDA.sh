#!/bin/bash

# Define the root directory
root_dir="/home/altair/flat-bug/test/output"

# Loop through each subdirectory in root
for folder in "$root_dir"/*/ ; do
    # Loop through each subfolder ("X", "Y", "Z") in the current folder
    for subfolder in "$folder"*/; do
        # Zip the subfolder and store the zip file in the same directory as the subfolder
        # (cd "$folder" && zip -0 -r "${subfolder##*/}.zip" "${subfolder##*/}") &
        # echo "(cd \"$folder\" && zip -0 -r \"${subfolder##*/}.zip\" \"${subfolder##*/}\")"
        subfolder_trimmed="${subfolder%/}"
        subfolder_name="${subfolder_trimmed##*/}"
        echo "Current folder: $folder"
        echo "Subfolder: $subfolder"
        echo "Zipping: $subfolder_name"
        (cd "$folder" && zip -0 -r "$subfolder_name.zip" "$subfolder_name" > /dev/null 2>&1) &
    done
done

# Wait for all background zipping processes to finish
wait
