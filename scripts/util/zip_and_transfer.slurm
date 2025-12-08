#!/bin/bash
#SBATCH --job-name=zip_transfer
#SBATCH --output=zip_transfer_%j.out
#SBATCH --error=zip_transfer_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# =============================================================================
# Configuration - Edit these variables
# =============================================================================

# Source directory to zip (local path)
SOURCE_DIR="/path/to/source/directory"

# Name for the archive file (without extension)
ARCHIVE_NAME="my_archive"

# Remote host (should be accessible via ssh without password)
REMOTE_HOST="imbi_gpu_H100"

# Target directory on remote host where the archive will be extracted
REMOTE_TARGET_DIR="/path/to/remote/destination"

# Temporary directory for creating the archive (local)
TEMP_DIR="/tmp"

# =============================================================================
# Script execution
# =============================================================================

echo "=========================================="
echo "Starting zip and transfer job"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

# Validate source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory '$SOURCE_DIR' does not exist!"
    exit 1
fi

# Create archive path
ARCHIVE_PATH="${TEMP_DIR}/${ARCHIVE_NAME}.tar.gz"

echo ""
echo "Step 1: Creating compressed archive..."
echo "Source: $SOURCE_DIR"
echo "Archive: $ARCHIVE_PATH"
echo ""

# Create tar.gz archive
tar -czf "$ARCHIVE_PATH" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create archive!"
    exit 1
fi

# Get archive size
ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
echo "Archive created successfully (Size: $ARCHIVE_SIZE)"

echo ""
echo "Step 2: Copying archive to remote host..."
echo "Remote: ${REMOTE_HOST}:${REMOTE_TARGET_DIR}/"
echo ""

# Copy archive to remote host with progress
rsync -avh --progress "$ARCHIVE_PATH" "${REMOTE_HOST}:${REMOTE_TARGET_DIR}/"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy archive to remote host!"
    rm -f "$ARCHIVE_PATH"
    exit 1
fi

echo "Archive copied successfully"

echo ""
echo "Step 3: Extracting archive on remote host..."
echo ""

# Extract archive on remote host and remove the archive file
ssh "$REMOTE_HOST" "cd ${REMOTE_TARGET_DIR} && tar -xzf ${ARCHIVE_NAME}.tar.gz && rm -f ${ARCHIVE_NAME}.tar.gz"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to extract archive on remote host!"
    echo "Archive remains at: ${REMOTE_HOST}:${REMOTE_TARGET_DIR}/${ARCHIVE_NAME}.tar.gz"
    rm -f "$ARCHIVE_PATH"
    exit 1
fi

echo "Archive extracted and cleaned up on remote host"

echo ""
echo "Step 4: Cleaning up local archive..."
rm -f "$ARCHIVE_PATH"
echo "Local archive removed"

echo ""
echo "=========================================="
echo "Transfer completed successfully!"
echo "Source: $SOURCE_DIR"
echo "Destination: ${REMOTE_HOST}:${REMOTE_TARGET_DIR}/$(basename "$SOURCE_DIR")"
echo "Finished: $(date)"
echo "=========================================="

