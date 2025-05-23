import logging
import os
import sys
from pathlib import Path

import hydra
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import (
    SentenceTransformer,
    # SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SequentialEvaluator
from transformers.integrations import WandbCallback

from mmcontext.callback import UnfreezeTextEncoderCallback
from mmcontext.eval import SystemMonitor
from mmcontext.models.mmcontextencoder import MMContextEncoder

# from mmcontext.pp.utils import consolidate_low_frequency_categories
from mmcontext.utils import get_evaluator, get_loss  # , load_test_adata_from_hf_dataset

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="train_conf", version_base=None)
def main(cfg: DictConfig):
    """
    Train the MMContext model using parameters specified in a Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        A Hydra DictConfig object containing all hyperparameters and settings
        for the dataset, model, and training process.
    """
    # Print out the loaded configuration for debugging
    # (comment out or remove if too verbose)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    # get the hydra output dir
    hydra_run_dir = HydraConfig.get().run.dir

    # -------------------------------------------------------------------------
    # 1. Create the model (MMContextEncoder => SentenceTransformer modules)
    # -------------------------------------------------------------------------
    if cfg.model:
        model = SentenceTransformer(cfg.model)
    else:
        input_dim_map = cfg.input_dim_map
        chosen_method = cfg.embedding_method
        if chosen_method not in input_dim_map:
            raise ValueError(f"Unknown embedding_method '{chosen_method}'. Allowed: {list(input_dim_map.keys())}")
        # Overwrite the model's embedding_dim with the mapped value
        cfg.adapter.omics_input_dim = input_dim_map[chosen_method]
        precomputed_key = f"X_{chosen_method}"
        enc = MMContextEncoder(
            text_encoder_name=cfg.text_encoder.name,
            adapter_hidden_dim=cfg.adapter.hidden_dim,
            adapter_output_dim=cfg.adapter.output_dim,
            output_token_embeddings=False,
            freeze_text_encoder=cfg.text_encoder.freeze_text_encoder,
            unfreeze_last_n_layers=cfg.text_encoder.unfreeze_last_n_layers,
            train_lookup=False,
        )

        # get the main cell sentence column
        if cfg.gene_based_cell_sentence:
            primary_cell_sentence = "cell_sentence_2"
            layer_axis = "var"
        else:
            primary_cell_sentence = "cell_sentence_1"
            layer_axis = "obs"
        # Get the correct columns for the given data format
    # -------------------------------------------------------------------------
    # 2. Load multiple datasets
    # -------------------------------------------------------------------------
    train_datasets = {}
    val_datasets = {}
    losses = {}
    evaluators = []

    for dataset_config in cfg.datasets:
        if dataset_config.type == "pairs" or dataset_config.type == "single":
            cell_sentences_cols = primary_cell_sentence
        elif dataset_config.type == "multiplets":
            cell_sentences_cols = [primary_cell_sentence, "negative_2"]
        dataset_name = f"{dataset_config.name}_{dataset_config.type}_{dataset_config.caption}"
        logger.info(f"Loading dataset: {dataset_name}")

        # Load the dataset
        dataset = load_dataset(f"jo-mengr/{dataset_name}")
        if not cfg.text_only:
            token_df = enc.get_initial_embeddings(
                dataset, layer_key=precomputed_key, download_dir="../../data/from_nxtcloud", axis=layer_axis
            )
            enc.register_initial_embeddings(token_df, data_origin=chosen_method)
        # Add the prefix expected by the model
        dataset_ready = enc.prepare_ds(
            dataset, cell_sentences_cols=cell_sentences_cols, prefix=not cfg.text_only
        )  # ,"negative_2"])

        # Add train split to train_datasets dictionary
        train_datasets[dataset_name] = dataset_ready["train"]

        # Add validation split to val_datasets dictionary
        val_datasets[dataset_name] = dataset_ready["val"]

        # Create loss function for this dataset type
        losses[dataset_name] = get_loss(dataset_type=dataset_config.type)

        evaluator = get_evaluator(
            dataset_type=dataset_config.type,
            dataset=dataset_ready["val"],
            batch_size=cfg.trainer.per_device_eval_batch_size,
        )
        evaluators.append(evaluator)
    # Build the sentence Trasnformer model
    modules = [enc]
    model = SentenceTransformer(modules=modules)
    # Combine all evaluators into a single sequential evaluator
    dev_evaluator = SequentialEvaluator(evaluators)

    # -------------------------------------------------------------------------
    # 3. Load test datasets
    # -------------------------------------------------------------------------
    test_datasets = {}
    for test_config in cfg.test_datasets:
        test_datasets[test_config.name] = load_dataset(test_config.name)["test"]

    # -------------------------------------------------------------------------
    # 4. Compute the correct embedding dimension based on method
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 5. Set up System Monitor
    # -------------------------------------------------------------------------
    monitor = SystemMonitor(logger=logging)
    monitor.start()

    # -------------------------------------------------------------------------
    # 6. Set up training arguments
    # -------------------------------------------------------------------------

    args = SentenceTransformerTrainingArguments(
        output_dir=hydra_run_dir,
        num_train_epochs=cfg.trainer.num_train_epochs,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        learning_rate=cfg.trainer.learning_rate,
        warmup_ratio=cfg.trainer.warmup_ratio,
        fp16=cfg.trainer.fp16,
        bf16=cfg.trainer.bf16,
        eval_strategy=cfg.trainer.eval_strategy,
        eval_steps=cfg.trainer.eval_steps,
        save_strategy=cfg.trainer.save_strategy,
        save_steps=cfg.trainer.save_steps,
        save_total_limit=cfg.trainer.save_total_limit,
        logging_steps=cfg.trainer.logging_steps,
        run_name=str(hydra_run_dir),
        dataloader_num_workers=cfg.trainer.dataloader_num_workers,
    )

    # -------------------------------------------------------------------------
    # 8. Create a trainer & train with multiple datasets
    # -------------------------------------------------------------------------
    unfreeze_callback = UnfreezeTextEncoderCallback(unfreeze_epoch=cfg.trainer.unfreeze_epoch)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_datasets,
        eval_dataset=val_datasets,
        loss=losses,
        evaluator=dev_evaluator,  # Pass the sequential evaluator instead of the dictionary
        callbacks=[unfreeze_callback, WandbCallback()],
    )
    trainer.train()

    # Evaluate on test datasets
    for test_dataset_config in cfg.test_datasets:
        test_dataset = test_datasets[test_dataset_config.name]
        test_evaluator = get_evaluator(
            dataset_type=test_dataset_config.type,
            dataset=test_dataset,
            batch_size=cfg.trainer.per_device_eval_batch_size,
        )
        test_results = test_evaluator(model)
        logger.info(f"Test results for {test_dataset_config.name}: {test_results}")

    # -------------------------------------------------------------------------
    # 9. Save the model and run system monitor
    # -------------------------------------------------------------------------
    # model[0].processor.omics_processor.clear_cache()  # Important or you will save anndata objects stored in the cache
    model_dir = Path(hydra_run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)

    monitor.stop()
    monitor.save(hydra_run_dir)
    monitor.plot_metrics(hydra_run_dir)
    logger.info(f"Training completed successfully. Model saved to {model_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
