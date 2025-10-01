#!/usr/bin/env bash
set -euo pipefail

echo "Preparing CellWhisperer model..."

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Check if CellWhisperer repository exists, if not clone it
CELLWHISPERER_DIR="$SCRIPT_DIR/CellWhisperer"
if [ ! -d "$CELLWHISPERER_DIR" ]; then
    cd "$SCRIPT_DIR"
    git clone https://github.com/mengerj/CellWhisperer.git
    cd "$CELLWHISPERER_DIR"
    git submodule update --init --recursive
fi
# Navigate to Geneformer module, install git-lfs there, and pull LFS files
GENEFORMER_DIR="$CELLWHISPERER_DIR/modules/Geneformer"

if [ -d "$GENEFORMER_DIR" ]; then
    cd "$GENEFORMER_DIR"
    # Install git lfs if not already available
    if ! command -v git-lfs &> /dev/null; then
        echo "Installing git-lfs..."
        # Try to install git-lfs (this might need adjustment based on the system)
        if command -v brew &> /dev/null; then
            brew install git-lfs
        elif command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y git-lfs
        elif command -v yum &> /dev/null; then
            sudo yum install -y git-lfs
        else
            echo "Please install git-lfs manually and rerun this script"
            exit 1
        fi
    fi
    echo "Installing git lfs..."
    git lfs install

    # Fix missing .gitattributes that prevents LFS files from being downloaded
    echo "Configuring git attributes for LFS..."
    echo "*.bin filter=lfs diff=lfs merge=lfs -text" > .gitattributes

    echo "Pulling git LFS files from Geneformer..."
    git lfs pull

    # Copy geneformer-12L-30M folder to script directory if it exists
    GENEFORMER_MODEL_DIR="$GENEFORMER_DIR/geneformer-12L-30M"
    TARGET_MODEL_DIR="$CELLWHISPERER_DIR/resources"
    # create target dir if it doesn't exist
    mkdir -p "$TARGET_MODEL_DIR"

    # The original condition [ ! -d "$TARGET_MODEL_DIR" ] is always false after mkdir -p,
    # so the copy never happens. We want to copy if the source exists and the target subdir does not.
    if [ -d "$GENEFORMER_MODEL_DIR" ] && [ ! -d "$TARGET_MODEL_DIR/geneformer-12L-30M" ]; then
        echo "Copying geneformer-12L-30M to script directory..."
        cp -r "$GENEFORMER_MODEL_DIR" "$TARGET_MODEL_DIR"
    fi
else
    echo "Warning: Geneformer directory not found at $GENEFORMER_DIR"
fi

echo "CellWhisperer preparation completed successfully!"
