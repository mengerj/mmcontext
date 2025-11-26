#!/bin/bash
#SBATCH --job-name=combined_array
#SBATCH --partition=gpu
#SBATCH --time=24:00:00         # Max job time (3 days for combined pipeline)
#SBATCH --array=1-1%1           # Array job
#SBATCH --gres=gpu:2            # Request amount of GPUs
#SBATCH --mem=120G

# ─────────────────────────────────────────────────────────────────────────────
# Configuration: Model config files directory
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIG_DIR="conf/models"
MODEL_CONFIGS=("model_list_cxg_geo_1")  # Add more model configs here as needed
run_cellwhisperer=true
final_root="/scratch/global/menger/mmcontext_out/2411_results"

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
BASE_OUT="outputs/$(date +%Y-%m-%d)/combined/${RUN_ID}"
mkdir -p "$BASE_OUT"

# Redirect logs to the output directory (include model config in log names)
exec 1>"$BASE_OUT"/combined_${MODEL_CONFIG}.out
exec 2>"$BASE_OUT"/combined_${MODEL_CONFIG}.err

# ─────────────────────────────────────────────────────────────────────────────
# 2) Path configuration will be handled by the configuration file
# ─────────────────────────────────────────────────────────────────────────────

echo "=== Path Configuration ==="
echo "Paths will be determined by configuration file"
echo "Base Output: $BASE_OUT"

###
# 3. Activate environment
###
echo "Starting combined embedding + evaluation job with RUN_ID: $RUN_ID"
echo "Model Config: $MODEL_CONFIG"
echo "Output directory: $BASE_OUT"
source .venv/bin/activate
echo "venv activated"

# Optional: Install flash-attn if needed (uncomment if required)
#echo "installing flash-attn"
#uv pip install ninja packaging wheel setuptools
#uv pip install flash-attn==2.8.1 --no-build-isolation
#echo "flash-attn installed"

###
# 4. Run the combined pipeline script with Hydra overrides
###

python3 scripts/combined_pipeline.py \
    --config-path=../conf \
    --config-name=combined_conf \
    models=${MODEL_CONFIG} \
    settings.run_cellwhisperer=${run_cellwhisperer} \
    settings.final_root=${final_root} \
    ++hydra.run.dir="$BASE_OUT"
    # Add additional overrides as needed:
    # settings.enable_parallel=true
    # settings.max_workers=4
    # embed.overwrite=true
    # settings.skip_existing_evaluations=false

###
# 5. Cleanup: Remove computation directory if it's a temporary directory
###

# Extract computation root from the configuration
COMPUTATION_ROOT=$(python3 -c "
import sys
sys.path.append('.')
from hydra import compose, initialize_config_dir
from pathlib import Path
import os

# Initialize Hydra with the config directory
config_dir = Path.cwd() / 'conf'
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name='combined_conf', overrides=['models=${MODEL_CONFIG}'])
    print(cfg.settings.computation_root)
")

echo "Computation root from config: $COMPUTATION_ROOT"

# Only clean up if it's a local directory (contains /local/)
if [[ "$COMPUTATION_ROOT" == */local/* ]]; then
    echo "Cleaning up temporary computation directory: $COMPUTATION_ROOT"
    rm -rf "$COMPUTATION_ROOT"
    echo "Cleanup completed"
else
    echo "Computation directory is not temporary, skipping cleanup: $COMPUTATION_ROOT"
fi

echo "Combined pipeline job completed for model config: $MODEL_CONFIG"
