#!/bin/bash
#SBATCH --job-name=eval_mmcontext
#SBATCH --output=eval_mmcontext.out
#SBATCH --error=eval_mmcontext.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16  # Increased for parallel processing
#SBATCH --mem=128G  # Increased memory for parallel processing

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
BASE_OUT="outputs/$(date +%Y-%m-%d)/evaluation/${RUN_ID}"
mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/eval.out
exec 2>"$BASE_OUT"/eval.err

# ─────────────────────────────────────────────────────────────────────────────
# 2) Set default overrides and parallel processing configuration
# ─────────────────────────────────────────────────────────────────────────────

# Parallel processing settings (can be overridden via command line)
ENABLE_PARALLEL="${ENABLE_PARALLEL:-false}"
MAX_WORKERS="${MAX_WORKERS:-}"  # Empty means auto-detect
# Model and dataset file overrides
MODEL_FILE="model_list.yaml"
DATASET_FILE="dataset_list_1.yaml"

# If running under SLURM, we can use the allocated CPU count
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    # Use SLURM allocated CPUs if MAX_WORKERS not explicitly set
    if [[ -z "$MAX_WORKERS" ]]; then
        MAX_WORKERS="$SLURM_CPUS_PER_TASK"
    fi
    echo "SLURM allocated CPUs: $SLURM_CPUS_PER_TASK"
fi

# Build evaluation overrides
EVAL_OVERRIDES=""
if [[ "$ENABLE_PARALLEL" == "true" ]]; then
    EVAL_OVERRIDES="${EVAL_OVERRIDES} eval.enable_parallel=true"
    if [[ -n "$MAX_WORKERS" ]]; then
        EVAL_OVERRIDES="${EVAL_OVERRIDES} eval.max_workers=${MAX_WORKERS}"
    fi
else
    EVAL_OVERRIDES="${EVAL_OVERRIDES} eval.enable_parallel=false"
fi

echo "Parallel processing enabled: $ENABLE_PARALLEL"
if [[ -n "$MAX_WORKERS" ]]; then
    echo "Max workers: $MAX_WORKERS"
else
    echo "Max workers: auto-detect"
fi
if [[ -n "$MODEL_FILE" ]]; then
    EVAL_OVERRIDES="${EVAL_OVERRIDES} models=${MODEL_FILE}"
fi
if [[ -n "$DATASET_FILE" ]]; then
    EVAL_OVERRIDES="${EVAL_OVERRIDES} datasets=${DATASET_FILE}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3) Activate environment
# ─────────────────────────────────────────────────────────────────────────────
echo "Starting evaluation job with RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
source .venv/bin/activate
echo "Virtual environment activated"

# ─────────────────────────────────────────────────────────────────────────────
# 4) Run your evaluation script with Hydra overrides
# ─────────────────────────────────────────────────────────────────────────────

echo "Running evaluation with overrides: $EVAL_OVERRIDES"

python scripts/eval.py \
    ++hydra.run.dir="$BASE_OUT" \
    $EVAL_OVERRIDES
