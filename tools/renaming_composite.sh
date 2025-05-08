#!/bin/bash

# Folder to process (current folder by default)
TARGET_DIR="${1:-.}"

# Loop through files in the target directory
for file in "$TARGET_DIR"/*; do
    # Skip if not a regular file
    [ -f "$file" ] || continue

    # Extract filename and directory
    filename=$(basename "$file")
    dir=$(dirname "$file")

    # Remove _composite or _composite_autocontrast
    newname="${filename/_composite_autocontrast/}"
    newname="${newname/_composite/}"

    # Replace 'F' with 'CellComposite_F' (only the first occurrence)
    newname="${newname/F/CellComposite_F}"

    # Rename the file
    mv "$file" "$dir/$newname"
    echo "Renamed: $filename → $newname"
done
