#!/usr/bin/env bash
set -euo pipefail

echo "Preparing SCSA model..."

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Check if SCSA repository exists, if not clone it
SCSA_DIR="$SCRIPT_DIR/SCSA"
if [ ! -d "$SCSA_DIR" ]; then
    cd "$SCRIPT_DIR"
    git clone https://github.com/bioinfo-ibms-pumc/SCSA.git
else
    echo "SCSA repository already exists at $SCSA_DIR"
    cd "$SCSA_DIR"
    git pull || echo "Warning: git pull failed, using existing version"
fi

# Verify database file exists
if [ ! -f "$SCSA_DIR/whole_v2.db" ]; then
    echo "ERROR: SCSA database whole_v2.db not found at $SCSA_DIR/whole_v2.db"
    echo "The database should be included in the SCSA repository."
    exit 1
fi

echo "SCSA preparation completed successfully!"
echo "  Repository: $SCSA_DIR"
echo "  Database:   $SCSA_DIR/whole_v2.db"
echo ""
echo "Next steps:"
echo ""
echo "  1. Create a dedicated venv for SCSA (requires numpy<2.0):"
echo "     python -m venv $SCRIPT_DIR/scsa_venv"
echo "     $SCRIPT_DIR/scsa_venv/bin/pip install -r $SCRIPT_DIR/requirements_scsa.txt"
echo ""
echo "  2. (Optional) Create a dedicated venv for calmate label harmonisation:"
echo "     python -m venv $SCRIPT_DIR/calmate_venv"
echo "     $SCRIPT_DIR/calmate_venv/bin/pip install -r $SCRIPT_DIR/requirements_calmate.txt"
