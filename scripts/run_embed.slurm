#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --partition=slurm
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
BASE_OUT="outputs/$(date +%Y-%m-%d)/embeddings/${RUN_ID}"
mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/embed.out
exec 2>"$BASE_OUT"/embed.err

###
# 2. Set default overrides:
#    These can be customized as needed
###

###
# 3. Activate environment
###
echo "Starting embedding job with RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
source .venv/bin/activate
echo "venv activated"

###
# 4. Run your training script with Hydra overrides
##

python3 scripts/embed.py \
    ++hydra.run.dir="$BASE_OUT"
