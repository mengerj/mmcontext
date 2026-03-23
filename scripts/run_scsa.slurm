#!/bin/bash
#SBATCH --job-name=scsa_eval
#SBATCH --partition=slurm
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
final_root="/scratch/global/menger/mmcontext_out/2411_results"

# ─────────────────────────────────────────────────────────────────────────────
# 0) Define RUN_ID
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RUN_ID="${SLURM_JOB_ID}"
else
  RUN_ID="$(date +%Y%m%d_%H%M%S)_scsa"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1) Prepare output directory and log redirects
# ─────────────────────────────────────────────────────────────────────────────
BASE_OUT="outputs/$(date +%Y-%m-%d)/scsa/${RUN_ID}"
mkdir -p "$BASE_OUT"

exec 1>"$BASE_OUT"/scsa.out
exec 2>"$BASE_OUT"/scsa.err

echo "=== SCSA Evaluation Job ==="
echo "RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
echo "Results root: $final_root"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Activate environment
# ─────────────────────────────────────────────────────────────────────────────
source .venv/bin/activate
echo "venv activated"

# ─────────────────────────────────────────────────────────────────────────────
# 3) Run SCSA pipeline
# ─────────────────────────────────────────────────────────────────────────────
python3 scripts/run_scsa.py \
    settings.output_root=${final_root} \
    ++hydra.run.dir="$BASE_OUT"

echo "SCSA evaluation job completed"
