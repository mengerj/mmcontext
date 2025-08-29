#!/bin/bash
#SBATCH --job-name=embed_array
#SBATCH --partition=slurm
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00         # Max job time
#SBATCH --array=1-2%1           # Array job: 1-2 with max 1 concurrent job (adjust %N for more parallel jobs)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration: Model config files directory
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIG_DIR="conf/models"
MODEL_CONFIGS=("model_list_1" "model_list_2")  # Add more model configs here as needed

# Validate array task ID
if [[ ${SLURM_ARRAY_TASK_ID} -gt ${#MODEL_CONFIGS[@]} ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) exceeds number of model configs (${#MODEL_CONFIGS[@]})"
    exit 1
fi

# Get the model config for this array task (arrays are 1-indexed, bash arrays are 0-indexed)
MODEL_CONFIG=${MODEL_CONFIGS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "=== SLURM Array Job Configuration ==="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model Config: ${MODEL_CONFIG}"
echo "Model Config File: ${MODEL_CONFIG_DIR}/${MODEL_CONFIG}.yaml"

# Validate that the model config file exists
if [[ ! -f "${MODEL_CONFIG_DIR}/${MODEL_CONFIG}.yaml" ]]; then
    echo "ERROR: Model config file not found: ${MODEL_CONFIG_DIR}/${MODEL_CONFIG}.yaml"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 0) Define a unified RUN_ID: real job ID under SLURM, or a timestamp locally
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RUN_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
  # local run: use date+seconds to make it unique
  RUN_ID="$(date +%Y%m%d_%H%M%S)_${MODEL_CONFIG}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1) Prepare output directory and log redirects
# ─────────────────────────────────────────────────────────────────────────────
BASE_OUT="outputs/$(date +%Y-%m-%d)/embeddings/${RUN_ID}"
mkdir -p "$BASE_OUT"

# Redirect logs to the output directory (include model config in log names)
exec 1>"$BASE_OUT"/embed_${MODEL_CONFIG}.out
exec 2>"$BASE_OUT"/embed_${MODEL_CONFIG}.err

###
# 2. Set default overrides:
#    These can be customized as needed
#    Example: to force overwrite existing embeddings, add: run.overwrite=true
###

###
# 3. Activate environment
###
echo "Starting embedding job with RUN_ID: $RUN_ID"
echo "Model Config: $MODEL_CONFIG"
echo "Output directory: $BASE_OUT"
source .venv/bin/activate
echo "venv activated"

#echo "installing flash-attn"#
#uv pip install ninja packaging wheel setuptools
#uv pip install flash-attn==2.8.1 --no-build-isolation
#echo "flash-attn installed"

###
# 4. Run your training script with Hydra overrides
###

python3 scripts/embed.py \
    --config-path=../conf \
    --config-name=embed_conf \
    defaults=models/${MODEL_CONFIG} \
    ++hydra.run.dir="$BASE_OUT"
    # Add run.overwrite=true to force overwrite existing embeddings
    # Example: run.overwrite=true
