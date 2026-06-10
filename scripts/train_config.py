#!/usr/bin/env python
"""Config-driven multi-dataset training for the MMContext ST pipeline.

Unlike ``train_tiny.py`` (single dataset, argparse), this script reads a plain
YAML config (no Hydra) and supports multiple omics + bio datasets, mixing
bimodal (omics-vector) and text anchors in one run. It reuses the existing
multi-dataset machinery: ``prepare_dataset``, ``prepare_vector_store``,
``get_loss`` / ``get_evaluator``, the unfreeze callbacks, and the Hub helpers.

Usage::

    python scripts/train_config.py --config conf/training/multi_example.yaml

See ``conf/training/multi_example.yaml`` for a documented config.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SequentialEvaluator
from transformers.integrations import WandbCallback

from mmcontext.callback import UnfreezeAdapterCallback, UnfreezeTextEncoderCallback
from mmcontext.embed import build_pipeline
from mmcontext.modules import MMContextModule
from mmcontext.training import (
    TrainConfig,
    assemble_training_data,
    build_merged_vector_store,
    load_config,
    load_raw_datasets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def build_or_load_model(cfg: TrainConfig, omics_dim: int | None) -> SentenceTransformer:
    """Build a fresh pipeline or load one to continue fine-tuning.

    Parameters
    ----------
    cfg : TrainConfig
        Run config.
    omics_dim : int or None
        Dimensionality of the merged VectorStore (``None`` for text-only runs).

    Returns
    -------
    SentenceTransformer
    """
    if cfg.model:
        logger.info("Loading existing model to continue training: %s", cfg.model)
        model = SentenceTransformer(cfg.model, trust_remote_code=True)
        if not isinstance(model[0], MMContextModule):
            raise TypeError(f"Loaded model's first module is {type(model[0]).__name__}, expected MMContextModule")
        if omics_dim is not None:
            adapter_dim = model[1].omics_input_dim
            if adapter_dim != omics_dim:
                raise ValueError(
                    f"Loaded adapter expects omics_input_dim={adapter_dim} but the merged "
                    f"VectorStore has dim={omics_dim}. Check obsm_key matches the loaded model."
                )
        return model

    return build_pipeline(
        text_model=cfg.text_encoder.name,
        omics_dim=omics_dim,
        shared_dim=cfg.shared_dim,
        adapter_hidden_dim=cfg.adapter_hidden_dim,
        max_seq_length=cfg.text_encoder.max_seq_length,
    )


def register_gene_special_token(model: SentenceTransformer, token: str) -> None:
    """Add *token* to the tokenizer and resize the encoder embeddings."""
    tokenizer = model[0].tokenizer
    num_added = tokenizer.add_tokens([token])
    if num_added > 0:
        model[0].auto_model.resize_token_embeddings(len(tokenizer))
        logger.info("Added special token %r and resized embeddings to %d", token, len(tokenizer))
    else:
        logger.info("Special token %r already present in tokenizer", token)


def main() -> None:
    """Run config-driven multi-dataset training."""
    parser = argparse.ArgumentParser(description="Config-driven MMContext training")
    parser.add_argument("--config", required=True, help="Path to the YAML training config")
    parser.add_argument(
        "--overwrite-store",
        action="store_true",
        help="Rebuild per-dataset vector stores even if cached .mmap files exist "
        "(also settable via overwrite_vector_store in the config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    # --- W&B ---
    use_wandb = cfg.trainer.wandb_project is not None
    if use_wandb:
        os.environ["WANDB_PROJECT"] = cfg.trainer.wandb_project
        if cfg.trainer.wandb_run_name:
            os.environ["WANDB_NAME"] = cfg.trainer.wandb_run_name
        logger.info("W&B enabled: project=%s", cfg.trainer.wandb_project)

    # --- Data + merged vector store ---
    omics_raw, bio_raw = load_raw_datasets(cfg)
    overwrite_store = args.overwrite_store or cfg.overwrite_vector_store
    merged_store, omics_dim = build_merged_vector_store(cfg, omics_raw, overwrite=overwrite_store)

    # --- Model ---
    model = build_or_load_model(cfg, omics_dim)
    if merged_store is not None:
        model[0].set_vector_store(merged_store)
    if cfg.gene_special_token:
        register_gene_special_token(model, cfg.gene_special_token)

    # --- Static text-encoder freezing ---
    if cfg.freeze_text_encoder:
        model[0].freeze_all_but_top_layers(cfg.unfreeze_last_n_layers)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("Trainable params: %d / %d (%.1f%%)", trainable, total, 100.0 * trainable / total)

    # --- Assemble datasets / losses / evaluators ---
    assembled = assemble_training_data(
        cfg,
        model,
        omics_raw,
        bio_raw,
        log_backend="wandb" if use_wandb else "auto",
    )
    if not assembled.train_datasets:
        raise ValueError("No training datasets were assembled — check omics_datasets / bio_datasets in the config.")
    logger.info("Training datasets: %s", list(assembled.train_datasets))

    dev_evaluator = SequentialEvaluator(assembled.evaluators) if assembled.evaluators else None
    eval_datasets = assembled.eval_datasets or None
    eval_strategy = cfg.trainer.eval_strategy
    eval_steps = cfg.trainer.eval_steps
    if not eval_datasets or dev_evaluator is None:
        logger.info("No eval datasets — disabling evaluation during training")
        eval_strategy, eval_steps = "no", None

    # --- Training arguments ---
    training_args = SentenceTransformerTrainingArguments(
        output_dir=cfg.trainer.output_dir,
        num_train_epochs=cfg.trainer.num_train_epochs,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        learning_rate=cfg.trainer.learning_rate,
        warmup_ratio=cfg.trainer.warmup_ratio,
        bf16=cfg.trainer.bf16,
        fp16=cfg.trainer.fp16,
        use_mps_device=cfg.trainer.use_mps_device,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=cfg.trainer.save_strategy,
        save_steps=cfg.trainer.save_steps,
        save_total_limit=cfg.trainer.save_total_limit,
        max_grad_norm=cfg.trainer.max_grad_norm,
        logging_steps=cfg.trainer.logging_steps,
        dataloader_num_workers=cfg.trainer.dataloader_num_workers,
        report_to="wandb" if use_wandb else "none",
        run_name=cfg.trainer.wandb_run_name,
    )
    if cfg.trainer.use_cosine_scheduler:
        training_args.set_lr_scheduler(name="cosine", warmup_ratio=cfg.trainer.warmup_ratio)

    if cfg.trainer.gradient_checkpointing:
        # Enable gradient flow for input_ids so partial unfreezing still trains.
        model[0].auto_model.enable_input_require_grads()
        model[0].auto_model.gradient_checkpointing_enable()

    # --- Callbacks ---
    callbacks = [UnfreezeTextEncoderCallback(unfreeze_epoch=cfg.unfreeze_epoch)]
    if use_wandb:
        callbacks.append(WandbCallback())
    if cfg.adapter_freezing is not None:
        callbacks.append(
            UnfreezeAdapterCallback(
                freeze_text_adapter=cfg.adapter_freezing.freeze_text_adapter,
                freeze_omics_adapter=cfg.adapter_freezing.freeze_omics_adapter,
                unfreeze_text_adapter_epoch=cfg.adapter_freezing.unfreeze_text_adapter_epoch,
                unfreeze_omics_adapter_epoch=cfg.adapter_freezing.unfreeze_omics_adapter_epoch,
            )
        )

    # --- Train ---
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=assembled.train_datasets,
        eval_dataset=eval_datasets,
        loss=assembled.losses,
        evaluator=dev_evaluator,
        callbacks=callbacks,
    )
    logger.info("Starting training on %d dataset(s)", len(assembled.train_datasets))
    trainer.train()

    # --- Save ---
    save_path = Path(cfg.trainer.output_dir) / "final"
    model.save(str(save_path))
    logger.info("Model saved to %s", save_path)

    # --- Save/load roundtrip sanity check ---
    verify_roundtrip(model, save_path)

    # --- Hub upload ---
    if cfg.push_to_hub:
        push_to_hub(cfg, model)

    logger.info("Done.")


def verify_roundtrip(model: SentenceTransformer, save_path: Path) -> None:
    """Encode a probe with the in-memory and reloaded model; warn on drift."""
    model.eval()
    probe = ["MALAT1 MT-CO3 GNAS SYT1 CALM1", "A cortical neuron expressing synaptic markers."]
    original = model.encode(probe)
    loaded = SentenceTransformer(str(save_path), trust_remote_code=True)
    loaded.eval()
    reloaded = loaded.encode(probe)
    max_diff = float(np.abs(original - reloaded).max())
    if max_diff > 1e-4:
        logger.warning("Save/load roundtrip difference is large: max_diff=%.2e", max_diff)
    else:
        logger.info("Save/load roundtrip OK (max_diff=%.2e)", max_diff)


def push_to_hub(cfg: TrainConfig, model: SentenceTransformer) -> None:
    """Upload the trained model to the Hub under a unique name."""
    from mmcontext.hub_utils import generate_unique_model_name, get_hf_username

    try:
        username = get_hf_username()
        model_name = generate_unique_model_name(
            cfg.text_encoder.name,
            obsm_key=cfg.obsm_key if cfg.omics_datasets else None,
            tag=cfg.tag,
            username=username,
        )
        repo_id = f"{username}/{model_name}"
        logger.info("Pushing model to Hub: %s", repo_id)
        model.push_to_hub(repo_id, private=True, exist_ok=True)
        logger.info("Uploaded model to https://huggingface.co/%s", repo_id)
    except Exception as e:
        logger.warning("Hub upload failed: %s. The model is saved locally and can be pushed manually.", e)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
