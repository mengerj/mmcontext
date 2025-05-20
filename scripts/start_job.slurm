#!/bin/bash
#SBATCH --job-name=mmcontext
#SBATCH --output=mmcontext_slurm.out
#SBATCH --error=mmcontext_slurm.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --time=04:00:00         # Max job time of 1 hour

source .venv/bin/activate

EMBEDDING_METHOD="geneformer"
# Now run your Python script
#accelerate launch --num_processes 2 scripts/train_hydra.py
python3 scripts/train_hydra.py ++embedding_method=$EMBEDDING_METHOD
