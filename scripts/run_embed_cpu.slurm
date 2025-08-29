#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --partition=slurm
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00         # Max job time

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
BASE_OUT="outputs/$(date +%Y-%m-%d)/embeddings/${RUN_ID}"
mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/embed.out
exec 2>"$BASE_OUT"/embed.err

###
# 2. Set default overrides:
#    These can be customized as needed
#    Example: to force overwrite existing embeddings, add: run.overwrite=true
###

###
# 3. Activate environment
###
echo "Starting embedding job with RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
source .venv/bin/activate
echo "venv activated"

#echo "installing flash-attn"#
#uv pip install ninja packaging wheel setuptools
#uv pip install flash-attn==2.8.1 --no-build-isolation
#echo "flash-attn installed"
###
# 4. Run your training script with Hydra overrides
##

python3 scripts/embed.py \
    ++hydra.run.dir="$BASE_OUT"
    # Add run.overwrite=true to force overwrite existing embeddings
    # Example: run.overwrite=true
