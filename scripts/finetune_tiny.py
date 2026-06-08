#!/usr/bin/env python
"""Finetune an already-trained MMContext model with mined hard negatives.

Unlike :mod:`scripts.train_tiny`, this script does **not** use the
pre-provided hard negatives baked into the dataset (the ``negative_*_idx``
columns). Instead it loads a model that was already trained once and uses
:func:`sentence_transformers.util.mine_hard_negatives` to mine model-specific
hard negatives, then continues training on those. This only makes sense with a
model that already has a meaningful joint embedding space — mining against a
freshly initialised bimodal model would produce noise.

How mining works here: the dataset's ``anchor`` column (omics ids for the
bimodal modality, resolved through the attached VectorStore) is encoded as the
query, while the ``positive`` texts form the corpus. The closest-but-not-true
positives become the mined negatives — confusable text descriptions in the same
shared embedding space.

Usage::

    # 1) Produce a pretrained model with train_tiny first
    python scripts/train_tiny.py --epochs 1 --output-dir outputs/tiny_model

    # 2) Finetune it with mined hard negatives (bimodal, default)
    python scripts/finetune_tiny.py \
        --model-path outputs/tiny_model/final \
        --output-dir outputs/tiny_finetuned

    # Text modality
    python scripts/finetune_tiny.py --modality text \
        --model-path outputs/tiny_text/final \
        --output-dir outputs/tiny_text_finetuned
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.util import mine_hard_negatives

from mmcontext.embed import prepare_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
HF_DATASET = "jo-mengr/cxg_schaefer_tiny"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Finetune a pretrained MMContext model on mined hard negatives."""
    parser = argparse.ArgumentParser(description="Finetune MMContext with mined hard negatives")
    parser.add_argument(
        "--modality",
        choices=["text", "bimodal"],
        default="bimodal",
        help="Training modality (default: bimodal)",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/tiny_model/final",
        help="Path to the pretrained model to finetune (default: outputs/tiny_model/final).",
    )
    parser.add_argument(
        "--vector-store",
        default=None,
        help="Path to .mmap VectorStore file. For bimodal modality: if omitted, "
        "the store is built automatically from adata_link + sample_idx "
        "columns using --obsm-key.",
    )
    parser.add_argument(
        "--obsm-key",
        default="X_scvi_fm",
        help="obsm key to extract when building VectorStore (default: X_scvi_fm). "
        "Other common choices: X_pca, X_geneformer, X_gs10k",
    )
    parser.add_argument("--output-dir", default="outputs/tiny_finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
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
    parser.add_argument(
        "--freeze-text-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the text encoder to cut gradient/optimizer memory. "
        "On by default for finetuning; combine with --unfreeze-last-n to keep "
        "the top N layers trainable. Disable with --no-freeze-text-encoder.",
    )
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=1,
        help="With --freeze-text-encoder, keep the top N transformer layers "
        "(plus pooler) trainable. 0 freezes the whole encoder (default: 1).",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=torch.backends.mps.is_available(),
        help="Use bf16 mixed precision. Supported on MPS (macOS 14+) and CUDA; "
        "roughly halves activation memory. Default: on when MPS is available.",
    )

    # --- Hard-negative mining parameters ---
    mining = parser.add_argument_group("hard-negative mining")
    mining.add_argument("--num-negatives", type=int, default=3, help="Negatives to mine per anchor (default: 3)")
    mining.add_argument(
        "--range-min",
        type=int,
        default=1,
        help="Skip the N closest candidates (the closest is usually the true positive). Default: 1.",
    )
    mining.add_argument(
        "--range-max",
        type=int,
        default=None,
        help="Only consider candidates up to this rank (default: None = no upper bound).",
    )
    mining.add_argument(
        "--max-score",
        type=float,
        default=0.95,
        help="Drop candidates whose similarity exceeds this ceiling to avoid mining "
        "false negatives (descriptions that are actually correct). Default: 0.95.",
    )
    mining.add_argument(
        "--relative-margin",
        type=float,
        default=None,
        help="Negatives must score below relative_margin * positive_score (e.g. 0.95). Default: None.",
    )
    mining.add_argument(
        "--sampling-strategy",
        choices=["top", "random"],
        default="top",
        help="Which qualifying candidates to keep as negatives (default: top).",
    )
    mining.add_argument(
        "--use-faiss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use FAISS for the similarity search (recommended for large datasets). Default: off.",
    )
    args = parser.parse_args()

    # --- Dataset (raw) ---
    ds_raw = load_dataset(args.dataset, split="train")

    # --- Load pretrained model ---
    # Dimensions (text/omics/shared) come from the saved config; no rebuild needed.
    logger.info("Loading pretrained model from %s", args.model_path)
    model = SentenceTransformer(args.model_path)

    # --- Attach VectorStore (bimodal only), required before mining ---
    if args.modality == "bimodal":
        from mmcontext.io import VectorStore, prepare_vector_store

        if args.vector_store is not None:
            store = VectorStore.load(args.vector_store)
        else:
            store_path = os.path.join(args.output_dir, "vector_store.mmap")
            store = prepare_vector_store(
                ds_raw,
                obsm_key=args.obsm_key,
                output_path=store_path,
            )
        model[0].set_vector_store(store)
        logger.info("VectorStore: %d vectors, dim=%d", len(store), store.dim)

    # --- Freezing (memory savings) ---
    # Freezing the text encoder removes its gradients + Adam optimizer state;
    # keeping only the top N layers trainable is the usual fine-tuning sweet
    # spot. On by default here since we are continuing to train an aligned model.
    if args.freeze_text_encoder:
        text_module = model[0]
        # unfreeze_last_n == 0 freezes the whole encoder; > 0 keeps the top N trainable.
        text_module.freeze_all_but_top_layers(args.unfreeze_last_n)
        logger.info("Froze text encoder, keeping top %d layers trainable", args.unfreeze_last_n)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("Trainable params: %d / %d (%.1f%%)", trainable, total, 100.0 * trainable / total)

    # --- Prepare anchor/positive pairs (drop the dataset's own negatives) ---
    modality = "bimodal" if args.modality == "bimodal" else "text"
    primary_cell_sentence = "cell_sentence_1" if modality == "bimodal" else "cell_sentence_2"
    pairs = prepare_dataset(
        ds_raw,
        purpose="train",
        modality=modality,
        primary_cell_sentence=primary_cell_sentence,
        use_hard_negatives=False,
    )
    logger.info("Prepared %d anchor/positive pairs, columns=%s", len(pairs), pairs.column_names)

    # --- Mine hard negatives with the pretrained model ---
    model.eval()
    logger.info(
        "Mining hard negatives: num_negatives=%d, range_min=%d, range_max=%s, max_score=%s, "
        "relative_margin=%s, sampling=%s, faiss=%s",
        args.num_negatives,
        args.range_min,
        args.range_max,
        args.max_score,
        args.relative_margin,
        args.sampling_strategy,
        args.use_faiss,
    )
    mined = mine_hard_negatives(
        pairs,
        model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        num_negatives=args.num_negatives,
        range_min=args.range_min,
        range_max=args.range_max,
        max_score=args.max_score,
        relative_margin=args.relative_margin,
        sampling_strategy=args.sampling_strategy,
        output_format="n-tuple",
        batch_size=args.batch_size,
        use_faiss=args.use_faiss,
        verbose=True,
    )
    logger.info("Mined dataset: %d rows (from %d pairs), columns=%s", len(mined), len(pairs), mined.column_names)

    # --- Wandb setup ---
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    use_wandb = wandb_project is not None
    if use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name
        logger.info("W&B enabled: project=%s, run=%s", wandb_project, args.wandb_run_name or "(auto)")

    # --- Finetuning ---
    model.train()
    loss = MultipleNegativesRankingLoss(model)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        # bf16 (MPS macOS 14+ / CUDA) takes precedence; fall back to fp16 on CUDA only.
        bf16=args.bf16,
        fp16=(not args.bf16) and torch.cuda.is_available() and not args.use_mps_device,
        use_mps_device=args.use_mps_device,
        report_to="wandb" if use_wandb else "none",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        run_name=args.wandb_run_name,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=mined,
        loss=loss,
    )

    logger.info(
        "Starting finetuning: modality=%s, epochs=%d, batch_size=%d", args.modality, args.epochs, args.batch_size
    )
    trainer.train()

    # --- Save final model ---
    save_path = os.path.join(args.output_dir, "final")
    model.save(save_path)
    logger.info("Model saved to %s", save_path)

    # --- Verify reload ---
    logger.info("Verifying save/load roundtrip...")
    model.eval()
    test_inputs = ["MALAT1 MT-CO3 GNAS SYT1 CALM1", "A cortical neuron expressing synaptic markers."]
    original_embs = model.encode(test_inputs)

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
