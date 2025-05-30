#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --mem=64G               # Request 64GB of host RAM, not GPU VRAM!
#SBATCH --time=01:00:00         # Max job time of 1 hour

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

###
# 2. Set default overrides:
#    These can be customized as needed
###
EMBEDDING_METHOD="geneformer"
TEXT_ONLY="false"
GENE_BASED_CELL_SENTENCE="false"
DATASET_NAME="cellxgene_pseudo_bulk_350k"
TRAINER_FP16="true"

###
# 3. Activate environment
###
echo "Starting training job with RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
source .venv/bin/activate
echo "venv activated"

###
# 4. Run your training script with Hydra overrides
###
#accelerate launch --num_processes 2 scripts/train.py
echo "Running training script with parameters:"
echo "  embedding_method: $EMBEDDING_METHOD"
echo "  text_only: $TEXT_ONLY"
echo "  gene_based_cell_sentence: $GENE_BASED_CELL_SENTENCE"
echo "  dataset: $DATASET_NAME"
echo "  trainer.fp16: $TRAINER_FP16"

python3 scripts/train.py \
    ++embedding_method="$EMBEDDING_METHOD" \
    ++text_only="$TEXT_ONLY" \
    ++gene_based_cell_sentence="$GENE_BASED_CELL_SENTENCE" \
    ++datasets="[{name: $DATASET_NAME, type: multiplets, caption: natural_language_annotation}]" \
    ++trainer.fp16="$TRAINER_FP16" \
    ++hydra.run.dir="$BASE_OUT"
