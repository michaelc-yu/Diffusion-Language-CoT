#!/bin/bash

# Directory to store PLAID weights
DEST_DIR="models/pretrained_plaid"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR" || exit 1

# Base URL for chunks
BASE_URL="https://github.com/igul222/plaid/releases/download/v1.0.0"

# Chunks to download
CHUNKS=("plaid1b_weights.tar.gz.00" "plaid1b_weights.tar.gz.01" "plaid1b_weights.tar.gz.02")

# Check if already extracted
if [ -f "pytorch_model.bin" ]; then
    echo "PLAID 1B model already extracted at $DEST_DIR"
    exit 0
fi

# Download chunks
for CHUNK in "${CHUNKS[@]}"; do
    if [ ! -f "$CHUNK" ]; then
        echo "Downloading $CHUNK ..."
        wget "$BASE_URL/$CHUNK"
    else
        echo "$CHUNK already exists, skipping..."
    fi
done

# Concatenate and extract
echo "Extracting model from chunks..."
cat plaid1b_weights.tar.gz.* | tar xvzf -

# Cleanup chunk files
echo "🧹 Cleaning up chunks..."
rm plaid1b_weights.tar.gz.*

echo "Done! PLAID 1B is available in: $DEST_DIR"

