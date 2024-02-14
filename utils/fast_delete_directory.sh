#!/bin/bash

TARGET_DIR="/home/altair/flat-bug/test/output"
EMPTY_DIR=$(mktemp -d)

if [ ! -d "$TARGET_DIR" ]; then
    echo "Target directory does not exist or is not a directory."
    exit 1
fi

TOTAL_ITEMS=$(find "$TARGET_DIR" -maxdepth 1 ! -path "$TARGET_DIR" | wc -l)
CURRENT_ITEM=0

find "$TARGET_DIR" -maxdepth 1 ! -path "$TARGET_DIR" -print0 | while IFS= read -r -d '' item; do
    CURRENT_ITEM=$((CURRENT_ITEM + 1))
    if [ -d "$item" ]; then
        echo "[$CURRENT_ITEM/$TOTAL_ITEMS] Processing directory: $item"
        rsync -a --delete "$EMPTY_DIR/" "$item/" --verbose && rmdir "$item"
    elif [ -f "$item" ]; then
        echo "[$CURRENT_ITEM/$TOTAL_ITEMS] Processing file: $item"
        rm "$item"
    fi
done

rm -r "$EMPTY_DIR"
echo "Deleted all files and directories in $TARGET_DIR"

