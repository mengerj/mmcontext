#!/usr/bin/env python
"""Train an MMContext model on the cxg_schaefer_tiny dataset.

Quick-start training script using sentence-transformers v5.4+ pipeline.
Supports two modes:

  1. **gene-list + text** (default) — gene-name strings as anchors, text
     descriptions as positives.  No VectorStore needed.
  2. **bimodal** — omics vectors from a VectorStore as anchors, text as
     positives.  Requires ``--vector-store`` pointing to a ``.mmap`` file
     that covers the ``sample_idx`` column of the dataset.

Usage::

    # Gene-list mode (default)
    python scripts/train_tiny.py --output-dir outputs/tiny_genelist

    # Bimodal mode
    python scripts/train_tiny.py --mode bimodal \
        --vector-store /path/to/store.mmap \
        --output-dir outputs/tiny_bimodal

    # Custom text encoder
    python scripts/train_tiny.py --text-model dmis-lab/biobert-base-cased-v1.2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.modules import Pooling, Normalize

from mmcontext.modules import MMContextModule, AdapterModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
HF_DATASET = "jo-mengr/cxg_schaefer_tiny"


def prepare_genelist_dataset(ds):
    """Reshape dataset for gene-list + text training.

    Returns a HF Dataset with columns ``anchor`` (gene names) and
    ``positive`` (text description).
    """
    # cell_sentence_1 = space-separated gene names → anchor
    ds = ds.rename_columns({"cell_sentence_1": "anchor"})
    ds = ds.select_columns(["anchor", "positive"])
    logger.info("Gene-list dataset: %d samples, columns=%s", len(ds), ds.column_names)
    return ds


def prepare_bimodal_dataset(ds):
    """Reshape dataset for bimodal (omics + text) training.

    Prefixes ``sample_idx`` with ``omics:`` so MMContextModule routes them
    through the VectorStore path.

    Returns a HF Dataset with columns ``anchor`` and ``positive``.
    """
    def prefix_omics(example):
        example["anchor"] = f"omics:{example['sample_idx']}"
        return example

    ds = ds.map(prefix_omics)
    ds = ds.select_columns(["anchor", "positive"])
    logger.info("Bimodal dataset: %d samples, columns=%s", len(ds), ds.column_names)
    return ds


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline(
    text_model: str,
    omics_dim: int | None = None,
    shared_dim: int = 256,
) -> SentenceTransformer:
    """Build the MMContext sentence-transformer pipeline.

    Parameters
    ----------
    text_model
        HuggingFace model name/path for the text encoder.
    omics_dim
        Dimension of omics vectors (only needed for bimodal mode).
        If None, the adapter omics head is sized to match text_dim.
    shared_dim
        Output dimension of the shared embedding space.
    """
    mmcontext = MMContextModule(model_name_or_path=text_model)
    text_dim = mmcontext.get_word_embedding_dimension()

    if omics_dim is None:
        omics_dim = text_dim

    adapter = AdapterModule(
        text_input_dim=text_dim,
        omics_input_dim=omics_dim,
        shared_dim=shared_dim,
    )
    pooling = Pooling(embedding_dimension=shared_dim, pooling_mode="mean")
    normalize = Normalize()

    pipeline = SentenceTransformer(modules=[mmcontext, adapter, pooling, normalize])
    logger.info(
        "Pipeline: text_dim=%d, omics_dim=%d, shared_dim=%d, params=%d",
        text_dim,
        omics_dim,
        shared_dim,
        sum(p.numel() for p in pipeline.parameters()),
    )
    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train MMContext on cxg_schaefer_tiny")
    parser.add_argument(
        "--mode",
        choices=["genelist", "bimodal"],
        default="genelist",
        help="Training mode (default: genelist)",
    )
    parser.add_argument(
        "--text-model",
        default="NeuML/pubmedbert-base-embeddings",
        help="HuggingFace text encoder (default: NeuML/pubmedbert-base-embeddings)",
    )
    parser.add_argument(
        "--vector-store",
        default=None,
        help="Path to .mmap VectorStore file. For bimodal mode: if omitted, "
             "the store is built automatically from adata_link + sample_idx "
             "columns using --obsm-key.",
    )
    parser.add_argument(
        "--obsm-key",
        default="X_scvi_fm",
        help="obsm key to extract when building VectorStore (default: X_scvi_fm). "
             "Other common choices: X_pca, X_geneformer, X_gs10k",
    )
    parser.add_argument("--omics-dim", type=int, default=None, help="Omics vector dimension")
    parser.add_argument("--shared-dim", type=int, default=256, help="Shared embedding dim (default: 256)")
    parser.add_argument("--output-dir", default="outputs/tiny_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--dataset", default=HF_DATASET, help="HuggingFace dataset name")
    parser.add_argument(
        "--use-mps-device",
        action=argparse.BooleanOptionalAction,
        default=torch.backends.mps.is_available(),
        help="Train on Apple Metal (MPS); default: on when MPS is available",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Weights & Biases project name. Enables wandb logging when set. "
             "You can also set WANDB_PROJECT env var instead.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name (auto-generated if omitted)",
    )
    args = parser.parse_args()

    # --- Dataset ---
    ds_raw = load_dataset(args.dataset, split="train")
    if args.mode == "bimodal":
        ds = prepare_bimodal_dataset(ds_raw)
    else:
        ds = prepare_genelist_dataset(ds_raw)

    # --- Pipeline ---
    pipeline = build_pipeline(
        text_model=args.text_model,
        omics_dim=args.omics_dim,
        shared_dim=args.shared_dim,
    )

    # Attach VectorStore for bimodal mode
    if args.mode == "bimodal":
        from mmcontext.io import VectorStore, prepare_vector_store

        if args.vector_store is not None:
            store = VectorStore.load(args.vector_store)
        else:
            # Auto-build from adata_link column
            store_path = os.path.join(args.output_dir, "vector_store.mmap")
            store = prepare_vector_store(
                ds_raw,
                obsm_key=args.obsm_key,
                output_path=store_path,
            )

        first_module = list(pipeline.children())[0]
        first_module.set_vector_store(store)

        # Infer omics_dim if not set
        if args.omics_dim is None:
            args.omics_dim = store.dim
            # Rebuild pipeline with correct omics_dim
            pipeline = build_pipeline(
                text_model=args.text_model,
                omics_dim=args.omics_dim,
                shared_dim=args.shared_dim,
            )
            first_module = list(pipeline.children())[0]
            first_module.set_vector_store(store)

        logger.info("VectorStore: %d vectors, dim=%d", len(store), store.dim)

    # --- Wandb setup ---
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    use_wandb = wandb_project is not None
    if use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name
        logger.info("W&B enabled: project=%s, run=%s", wandb_project, args.wandb_run_name or "(auto)")

    # --- Training ---
    loss = MultipleNegativesRankingLoss(pipeline)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available() and not args.use_mps_device,
        use_mps_device=args.use_mps_device,
        report_to="wandb" if use_wandb else "none",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        run_name=args.wandb_run_name,
    )

    trainer = SentenceTransformerTrainer(
        model=pipeline,
        args=training_args,
        train_dataset=ds,
        loss=loss,
    )

    logger.info("Starting training: mode=%s, epochs=%d, batch_size=%d", args.mode, args.epochs, args.batch_size)
    trainer.train()

    # --- Save final model ---
    save_path = os.path.join(args.output_dir, "final")
    pipeline.save(save_path)
    logger.info("Model saved to %s", save_path)

    # --- Verify reload ---
    logger.info("Verifying save/load roundtrip...")
    pipeline.eval()
    test_inputs = ["MALAT1 MT-CO3 GNAS SYT1 CALM1", "A cortical neuron expressing synaptic markers."]
    original_embs = pipeline.encode(test_inputs)

    loaded = SentenceTransformer(save_path)
    loaded.eval()
    loaded_embs = loaded.encode(test_inputs)

    max_diff = np.abs(original_embs - loaded_embs).max()
    logger.info("Save/load verification: max_diff=%.2e (should be < 1e-5)", max_diff)
    if max_diff > 1e-4:
        logger.warning("Save/load roundtrip difference is unexpectedly large!")
    else:
        logger.info("Save/load roundtrip OK")

    logger.info("Done.")


if __name__ == "__main__":
    main()
