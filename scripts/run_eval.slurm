#!/bin/bash
#SBATCH --job-name=eval_mmcontext
#SBATCH --output=eval_mmcontext.out
#SBATCH --error=eval_mmcontext.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

# Environment setup
echo "Starting evaluation job"
source .venv/bin/activate
echo "Virtual environment activated"

# Run evaluations sequentially
echo "Evaluating HVG model"
python scripts/evaluation_workflow.py \
    ++model_name="jo-mengr/mmcontext-100k-natural_language_annotation-hvg-2048-text-unfrozen" \
    ++model_short_name="biobert_hvg"

echo "Evaluating PCA model"
python scripts/evaluation_workflow.py \
    ++model_name="jo-mengr/mmcontext-100k-natural_language_annotation-pca-2048-text-unfrozen" \
    ++model_short_name="biobert_pca"

echo "Evaluating scVI model"
python scripts/evaluation_workflow.py \
    ++model_name="jo-mengr/mmcontext-100k-natural_language_annotation-scvi-2048-text-unfrozen" \
    ++model_short_name="biobert_scvi"

echo "Evaluating Geneformer model"
python scripts/evaluation_workflow.py \
    ++model_name="jo-mengr/mmcontext-100k-natural_language_annotation-geneformer-2048-text-unfrozen" \
    ++model_short_name="biobert_geneformer"

echo "All evaluations completed"
