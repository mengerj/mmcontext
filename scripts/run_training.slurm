#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --mem=64G               # Request 64GB of host RAM, not GPU VRAM!
#SBATCH --time=04:00:00         # Max job time of 4 hours

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

# General training parameters
EMBEDDING_METHOD="scvi_fm"
FORCE_CUDA="true"           # Require CUDA for training
FORCE_REFRESH_CACHE="false" # Set to "true" to re-download datasets even if cached
TRAINER_FP16="true"

# Adapter dimensions
ADAPTER_HIDDEN_DIM="128"    # Can be "null" or a number like 1024
ADAPTER_OUTPUT_DIM="64"     # Usually 2048, 1024, etc.

# Learning rate and other trainer params
LEARNING_RATE="2e-4"        # Changed from 0.05 to avoid NaN issues
BATCH_SIZE="128"
EVAL_BATCH_SIZE="128"
NUM_EPOCHS="4"
WARMUP_RATIO="0.1"

# Note: Dataset-specific configurations (text_only, layer_axis, cs_length, etc.)
# are now handled in the YAML configuration files (conf/train_conf.yaml)
# rather than as command-line overrides.

# ─────────────────────────────────────────────────────────────────────────────
# 3) Prepare Training Parameters
# ─────────────────────────────────────────────────────────────────────────────
# No dataset overrides needed - configurations are in YAML files

# ─────────────────────────────────────────────────────────────────────────────
# 4) Activate environment and run training
# ─────────────────────────────────────────────────────────────────────────────

echo "Starting training job with RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
source .venv/bin/activate
echo "venv activated"

echo "Running training script with parameters:"
echo "  embedding_method: $EMBEDDING_METHOD"
echo "  force_cuda: $FORCE_CUDA"
echo "  force_refresh_cache: $FORCE_REFRESH_CACHE"
echo "  adapter.hidden_dim: $ADAPTER_HIDDEN_DIM"
echo "  adapter.output_dim: $ADAPTER_OUTPUT_DIM"
echo "  learning_rate: $LEARNING_RATE"
echo "  batch_size: $BATCH_SIZE"
echo "  trainer.fp16: $TRAINER_FP16"
echo "  Dataset configurations are loaded from conf/train_conf.yaml"

python3 scripts/train.py \
    embedding_method="$EMBEDDING_METHOD" \
    force_cuda="$FORCE_CUDA" \
    force_refresh_cache="$FORCE_REFRESH_CACHE" \
    adapter.hidden_dim="$ADAPTER_HIDDEN_DIM" \
    adapter.output_dim="$ADAPTER_OUTPUT_DIM" \
    trainer.learning_rate="$LEARNING_RATE" \
    trainer.per_device_train_batch_size="$BATCH_SIZE" \
    trainer.per_device_eval_batch_size="$EVAL_BATCH_SIZE" \
    trainer.num_train_epochs="$NUM_EPOCHS" \
    trainer.warmup_ratio="$WARMUP_RATIO" \
    trainer.fp16="$TRAINER_FP16" \
    hydra.run.dir="$BASE_OUT"
