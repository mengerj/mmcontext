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
EMBEDDING_METHOD="hvg"
TEXT_ONLY="false"
GENE_BASED_CELL_SENTENCE="false"
TRAINER_FP16="true"

# Dataset configuration
# You can define multiple datasets - each one will be added to the list
# Format: "name,type,caption,cs_len" (cs_len can be "null" or a number)
DATASETS=(
    "cellxgene_pseudo_bulk_100k,multiplets,natural_language_annotation,50"
    "geo_70k,multiplets,natural_language_annotation,50"
)

# Adapter dimensions
ADAPTER_HIDDEN_DIM="null"    # Can be "null" or a number like 1024
ADAPTER_OUTPUT_DIM="2048"    # Usually 2048, 1024, etc.

# Learning rate and other trainer params
LEARNING_RATE="2e-4"        # Changed from 0.05 to avoid NaN issues
BATCH_SIZE="512"
EVAL_BATCH_SIZE="512"
NUM_EPOCHS="4"
WARMUP_RATIO="0.1"

# ─────────────────────────────────────────────────────────────────────────────
# 3) Build Hydra Override Strings
# ─────────────────────────────────────────────────────────────────────────────

# Function to build datasets override string
build_datasets_override() {
    local datasets_str="datasets=["
    local first=true

    for dataset in "${DATASETS[@]}"; do
        IFS=',' read -r name type caption cs_len <<< "$dataset"

        if [ "$first" = true ]; then
            first=false
        else
            datasets_str="${datasets_str},"
        fi

        datasets_str="${datasets_str}{name:${name},type:${type},caption:${caption}"

        if [ "$cs_len" != "null" ]; then
            datasets_str="${datasets_str},cs_len:${cs_len}"
        fi

        datasets_str="${datasets_str}}"
    done

    datasets_str="${datasets_str}]"
    echo "$datasets_str"
}

DATASETS_OVERRIDE=$(build_datasets_override)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Activate environment and run training
# ─────────────────────────────────────────────────────────────────────────────

echo "Starting training job with RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
source .venv/bin/activate
echo "venv activated"

echo "Running training script with parameters:"
echo "  embedding_method: $EMBEDDING_METHOD"
echo "  text_only: $TEXT_ONLY"
echo "  gene_based_cell_sentence: $GENE_BASED_CELL_SENTENCE"
echo "  datasets: ${DATASETS_OVERRIDE}"
echo "  adapter.hidden_dim: $ADAPTER_HIDDEN_DIM"
echo "  adapter.output_dim: $ADAPTER_OUTPUT_DIM"
echo "  learning_rate: $LEARNING_RATE"
echo "  batch_size: $BATCH_SIZE"
echo "  trainer.fp16: $TRAINER_FP16"

python3 scripts/train.py \
    embedding_method="$EMBEDDING_METHOD" \
    text_only="$TEXT_ONLY" \
    gene_based_cell_sentence="$GENE_BASED_CELL_SENTENCE" \
    "$DATASETS_OVERRIDE" \
    adapter.hidden_dim="$ADAPTER_HIDDEN_DIM" \
    adapter.output_dim="$ADAPTER_OUTPUT_DIM" \
    trainer.learning_rate="$LEARNING_RATE" \
    trainer.per_device_train_batch_size="$BATCH_SIZE" \
    trainer.per_device_eval_batch_size="$EVAL_BATCH_SIZE" \
    trainer.num_train_epochs="$NUM_EPOCHS" \
    trainer.warmup_ratio="$WARMUP_RATIO" \
    trainer.fp16="$TRAINER_FP16" \
    hydra.run.dir="$BASE_OUT"
