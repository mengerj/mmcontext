#!/bin/bash
#SBATCH --job-name=mmcontext
#SBATCH --output=mmcontext_slurm.out
#SBATCH --error=mmcontext_slurm.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --mem=64G               # Request 64GB of host RAM, not GPU VRAM!
#SBATCH --time=01:00:00         # Max job time of 1 hour

source .venv/bin/activate

# Now run your Python script
python3 scripts/train_hydra.py
