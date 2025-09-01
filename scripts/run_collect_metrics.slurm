#!/bin/bash
#SBATCH --job-name=collect_metrics
#SBATCH --output=collect_metrics.out
#SBATCH --error=collect_metrics.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

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
BASE_OUT="outputs/$(date +%Y-%m-%d)/metrics_collection/${RUN_ID}"
mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/collect_metrics.out
exec 2>"$BASE_OUT"/collect_metrics.err

# ─────────────────────────────────────────────────────────────────────────────
# 2) Activate environment
# ─────────────────────────────────────────────────────────────────────────────
echo "Starting metrics collection job with RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
source .venv/bin/activate
echo "Virtual environment activated"

# ─────────────────────────────────────────────────────────────────────────────
# 3) Run metrics collection script
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Step 1: Collecting metrics ==="
echo "Running collect_metrics.py with collect_metrics_conf.yaml"

python scripts/collect_metrics.py \
    --config-path=../conf \
    --config-name=collect_metrics_conf \
    ++hydra.run.dir="$BASE_OUT"

# Check if metrics collection was successful
if [[ $? -ne 0 ]]; then
    echo "ERROR: Metrics collection failed!"
    exit 1
fi

echo "Metrics collection completed successfully"

# ─────────────────────────────────────────────────────────────────────────────
# 4) Run plotting script
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Step 2: Creating plots ==="
echo "Running plot_metrics.py with collect_metrics_conf.yaml"

python scripts/plot_metrics.py \
    --config-path=../conf \
    --config-name=collect_metrics_conf \
    ++hydra.run.dir="$BASE_OUT"

# Check if plotting was successful
if [[ $? -ne 0 ]]; then
    echo "ERROR: Plotting failed!"
    exit 1
fi

echo "Plotting completed successfully"

# ─────────────────────────────────────────────────────────────────────────────
# 5) Summary
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Job Summary ==="
echo "RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
echo "Metrics collection and plotting completed successfully!"

# Print information about generated files
METRICS_OUTPUT_DIR=$(python -c "
import sys
sys.path.append('src')
from omegaconf import OmegaConf
cfg = OmegaConf.load('conf/collect_metrics_conf.yaml')
print(cfg.collect_metrics.output_dir)
" 2>/dev/null)

if [[ -n "$METRICS_OUTPUT_DIR" ]]; then
    echo "Generated files should be in: $METRICS_OUTPUT_DIR"
    if [[ -d "$METRICS_OUTPUT_DIR" ]]; then
        echo "Contents:"
        ls -la "$METRICS_OUTPUT_DIR"
        if [[ -d "$METRICS_OUTPUT_DIR/plots" ]]; then
            echo "Plots:"
            ls -la "$METRICS_OUTPUT_DIR/plots"
        fi
    fi
fi
