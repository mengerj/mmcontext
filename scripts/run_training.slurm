#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --mem=60G               # Max memory usage
#SBATCH --time=24:00:00         # Max job time

# ─────────────────────────────────────────────────────────────────────────────
# 0) Define a unified RUN_ID: real job ID under SLURM, or a timestamp locally
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RUN_ID="${SLURM_JOB_ID}"
else
  # local run: use date+seconds to make it unique
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1) Prepare output directory and log redirects
# ─────────────────────────────────────────────────────────────────────────────
BASE_OUT="outputs/$(date +%Y-%m-%d)/training/${RUN_ID}"
mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/train.out
exec 2>"$BASE_OUT"/train.err

# ─────────────────────────────────────────────────────────────────────────────
# 2) Configuration Variables - Customize these as needed
# ─────────────────────────────────────────────────────────────────────────────

# Project root path (adjust if running from different location)
#PROJECT_ROOT="."

# Configuration file path (relative to PROJECT_ROOT)
# Override this by setting CONFIG_FILE environment variable before running
CONFIG_FILE="/home/menger/git/mmcontext/conf/training/biobert_numeric.yaml"
#extend adata cache dir with run_id
ADATA_CACHE_DIR="/scratch/local/menger/data/from_nxtcloud/${RUN_ID}"
CACHE_DIR="/scratch/local/menger/cache/huggingface/${RUN_ID}"
# Optional command-line overrides
# These can be set as environment variables to override config file values
MODEL=null
EMBEDDING_METHOD=geneformer
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-}"

# Validate that config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi
echo "Using configuration file: $CONFIG_FILE"
if [[ -n "$EMBEDDING_METHOD" ]]; then
    echo "Overriding embedding_method: $EMBEDDING_METHOD"
fi
if [[ -n "$TEXT_ENCODER_NAME" ]]; then
    echo "Overriding text_encoder.name: $TEXT_ENCODER_NAME"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3) Setup cleanup trap and activate environment
# ─────────────────────────────────────────────────────────────────────────────

# Function to clean up scratch directories
cleanup() {
    echo "Cleaning up scratch local directories..."
    rm -rf $ADATA_CACHE_DIR
    rm -rf $CACHE_DIR
    echo "Cleaned up data/from_nxtcloud and cache/huggingface"
}

# Set trap to ensure cleanup runs on script exit (success, failure, or termination)
trap cleanup EXIT

echo "Starting training job with RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
echo "Configuration file: $CONFIG_FILE"
echo "Adata cache directory: $ADATA_CACHE_DIR"
echo "Cache directory: $CACHE_DIR"
source .venv/bin/activate
echo "venv activated"

#echo "installing accelerate"
#uv pip install accelerate
#echo "accelerate installed"

echo "Running training script with configuration from: $CONFIG_FILE"

# Change to project root directory to ensure relative paths work correctly
#cd "$PROJECT_ROOT"

# Build the command with optional overrides
TRAIN_CMD="python scripts/train.py --config-path=\"$(dirname "$CONFIG_FILE")\" --config-name=\"$(basename "$CONFIG_FILE" .yaml)\" hydra.run.dir=\"$BASE_OUT\" ++adata_cache_dir=\"$ADATA_CACHE_DIR\" ++cache_dir=\"$CACHE_DIR\""

# Add optional overrides if provided
if [[ -n "$EMBEDDING_METHOD" ]]; then
    TRAIN_CMD="$TRAIN_CMD embedding_method=\"$EMBEDDING_METHOD\""
fi
if [[ -n "$TEXT_ENCODER_NAME" ]]; then
    TRAIN_CMD="$TRAIN_CMD text_encoder.name=\"$TEXT_ENCODER_NAME\""
fi
if [[ -n "$MODEL" ]]; then
    TRAIN_CMD="$TRAIN_CMD model=\"$MODEL\""
fi
if [[ -n "$LEARNING_RATE" ]]; then
    TRAIN_CMD="$TRAIN_CMD trainer.learning_rate=\"$LEARNING_RATE\""
fi

echo "Executing: $TRAIN_CMD"
eval $TRAIN_CMD

# clean up scratch local
echo "Cleaning up scratch local"
rm -rf /scratch/local/menger/data/from_nxtcloud/${RUN_ID}
rm -rf /scratch/local/menger/cache/huggingface/${RUN_ID}
echo "Cleaned up data/from_nxtcloud and cache/huggingface"
