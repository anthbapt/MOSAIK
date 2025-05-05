#!/bin/bash

# Source root directory (adjust as needed)
SOURCE_DIR="/Volumes/Extreme SSD/CosMx/SBF_C018/RawFiles/TMA2/20250319_181208_S2/CellStatsDir/"

# Destination directory (where you want to copy the files)
DEST_DIR="/Volumes/Extreme SSD/CosMx/SBF_C018/flatFiles/TMA2/CellLabels"

# Create destination if it doesn't exist
mkdir -p "$DEST_DIR"

# Find and copy
find "$SOURCE_DIR" -mindepth 2 -maxdepth 2 -type f -name "CellLabels*.tif" | while read -r filepath; do
    filename=$(basename "$filepath")
    cp "$filepath" "$DEST_DIR/$filename"
done

echo "All CellLabels files copied to $DEST_DIR"

PARENT_DIR="$(dirname "$DEST_DIR")"

# Change to the parent directory
cd "$PARENT_DIR" || exit

gunzip *.csv.gz

for f in *-polygons.csv; do
    mv "$f" "${f%-polygons.csv}_polygons.csv"
done

echo "polygons file renamed"
