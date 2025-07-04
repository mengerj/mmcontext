#!/bin/bash
#SBATCH --job-name=eval_mmcontext
#SBATCH --output=eval_mmcontext.out
#SBATCH --error=eval_mmcontext.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

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

###
# 2. Set default overrides:
#    These can be customized as needed
###

###
# 3. Activate environment
###
echo "Starting evaluation job with RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
source .venv/bin/activate
echo "Virtual environment activated"

###
# 4. Run your evaluation script with Hydra overrides
###

python scripts/eval.py \
    ++hydra.run.dir="$BASE_OUT"
