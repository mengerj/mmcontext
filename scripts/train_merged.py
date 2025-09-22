import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, HfFolder
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SequentialEvaluator

# Import utility functions from the original training script
from train import (
    check_model_exists,
    check_revision_exists,
    generate_model_name,
    generate_revision_name,
    generate_unique_model_name,
    push_dataset_revision,
    resolve_torch_dtype_strings,
    validate_dataset_configurations,
)
from transformers.integrations import WandbCallback

from mmcontext.callback import UnfreezeAdapterCallback, UnfreezeTextEncoderCallback
from mmcontext.eval import SystemMonitor
from mmcontext.models.mmcontextencoder import MMContextEncoder
from mmcontext.utils import (
    get_evaluator,
    get_loss,
    resolve_negative_indices_and_rename,
    truncate_cell_sentences,
    truncate_semantic_cell_sentences_dataset,
)

logger = logging.getLogger(__name__)


def prepare_single_dataset(
    dataset,
    dataset_config: DictConfig,
    dataset_name: str,
    primary_cell_sentence: str,
    model: "SentenceTransformer",
    chosen_method: str = None,
    precomputed_key: str = None,
    force_refresh_cache: bool = False,
    adata_cache_dir: str = "data/from_nxtcloud",
) -> "DatasetDict":
    """
    Prepare a single dataset by applying all preprocessing steps

    This is similar to prepare_ds from the original script but focused on single dataset processing.
    """
    logger.info(f"Preparing dataset for merging: {dataset_name}")

    # Determine dataset-specific settings
    layer_axis = getattr(dataset_config, "layer_axis", "obs")
    dataset_text_only = getattr(dataset_config, "text_only", False)

    # Get dataset-specific cell sentence truncation parameters
    dataset_cs_length = getattr(dataset_config, "cs_length", None)
    dataset_cs_col = getattr(dataset_config, "cs_col", None)

    logger.info(
        f"Dataset '{dataset_name}' will be processed as: {'text_only' if dataset_text_only else 'numeric embeddings'}"
    )

    # Apply cell sentence truncation if configured (only for text_only datasets with cs_length specified)
    if dataset_cs_length and dataset_cs_length > 0 and dataset_cs_col:
        logger.info(
            f"Truncating cell sentences in column '{dataset_cs_col}' to {dataset_cs_length} tokens for text_only dataset"
        )

        # Apply truncation to all splits that contain the specified column
        truncated_dataset = {}
        for split_name, split_data in dataset.items():
            if dataset_cs_col in split_data.column_names and dataset_cs_col == "cell_sentence_2":
                truncated_dataset[split_name] = truncate_cell_sentences(
                    split_data,
                    dataset_cs_col,
                    dataset_cs_length,
                    filter_strings=dataset_config.get("gene_filter_strings", None),
                )
                logger.info(f"  Truncated {split_name} split")
            elif (
                dataset_cs_col in split_data.column_names
                and dataset_cs_col == "cell_sentence_3"
                or dataset_cs_col == "cell_sentence_4"
            ):
                truncated_dataset[split_name] = truncate_semantic_cell_sentences_dataset(
                    split_data,
                    "cell_sentence_3",
                    dataset_cs_length,
                    filter_strings=dataset_config.get("gene_filter_strings", None),
                )
                truncated_dataset[split_name] = truncate_semantic_cell_sentences_dataset(
                    truncated_dataset[split_name],
                    "cell_sentence_4",
                    dataset_cs_length,
                    filter_strings=dataset_config.get("gene_filter_strings", None),
                )
            else:
                truncated_dataset[split_name] = split_data
                logger.info(f"  Column '{dataset_cs_col}' not found in {split_name} split, keeping original")

        # Use the truncated dataset for the rest of the processing
        dataset = DatasetDict(truncated_dataset)
    elif dataset_text_only:
        logger.info(f"Text_only dataset '{dataset_name}' - no truncation applied (cs_length or cs_col not specified)")
    else:
        logger.info(f"Non-text_only dataset '{dataset_name}' - skipping cell sentence truncation")

    # Step 1: Handle embedding registration FIRST (needs access to raw dataset with all columns)
    if not dataset_text_only and hasattr(model[0], "get_initial_embeddings"):
        logger.info(f"Loading numeric embeddings for dataset '{dataset_name}' (before column selection)")
        token_df, _ = model[0].get_initial_embeddings(
            dataset,
            layer_key=precomputed_key,
            download_dir=f"{adata_cache_dir}/{dataset_name}",
            axis=layer_axis,
            overwrite=force_refresh_cache,
        )
        model[0].register_initial_embeddings(token_df, data_origin=chosen_method)
    elif not dataset_text_only and not hasattr(model[0], "get_initial_embeddings"):
        # If embedding_method is null, force text_only mode
        logger.error(
            f"Dataset '{dataset_name}' has no get_initial_embeddings method. Can't process numeric embeddings."
        )
        raise ValueError(
            f"Dataset '{dataset_name}' has no get_initial_embeddings method. Can't process numeric embeddings."
        )
    elif dataset_text_only:
        # In text_only mode, we'll use cell sentences directly
        logger.info(f"Dataset '{dataset_name}' using text_only mode - cell sentences will be processed as text")

    # Step 2: Select columns based on dataset configuration
    keep_columns = getattr(dataset_config, "keep_columns", None)
    index_column = getattr(dataset_config, "index_column", None)

    if keep_columns:
        # Automatically include index column if specified (needed for resolving indices)
        if index_column and index_column not in keep_columns:
            keep_columns = keep_columns + [index_column]
            logger.info(f"Added index column '{index_column}' to keep_columns")

        logger.info(f"Selecting columns: {keep_columns}")
        # Apply to all splits that contain the specified columns
        dataset_selected = {}
        for split_name, split_data in dataset.items():
            # Only keep columns that exist in this split
            available_columns = [col for col in keep_columns if col in split_data.column_names]
            if available_columns:
                dataset_selected[split_name] = split_data.select_columns(available_columns)
                logger.info(f"  Selected columns {available_columns} in {split_name} split")
            else:
                logger.warning(f"  No specified columns found in {split_name} split, keeping all")
                dataset_selected[split_name] = split_data
        dataset = DatasetDict(dataset_selected)
    else:
        logger.info("No keep_columns specified, using dataset as-is")

    # Step 3: Resolve negative indices for multiplet datasets and rename columns
    # Default to multiplets type since we're making it the default
    dataset_type = getattr(dataset_config, "type", "multiplets")
    if dataset_type == "multiplets":
        logger.info("Resolving negative indices and renaming columns for multiplet dataset")
        index_col_to_use = index_column if index_column else "sample_idx"
        dataset = resolve_negative_indices_and_rename(
            dataset,
            primary_cell_sentence_col=primary_cell_sentence,
            positive_col=dataset_config.get("positive_col", "positive"),
            negative_prefix="negative",
            index_col=index_col_to_use,
            remove_index_col=True,  # Remove index column after resolving
        )
        logger.info(
            f"Primary column '{primary_cell_sentence}' renamed to 'anchor', index column '{index_col_to_use}' removed"
        )

    # Step 4: Apply prefixes using the new simplified prefix_ds method
    prefix_columns = getattr(
        dataset_config, "prefix_columns", ["anchor", "negative_2"]
    )  # These are the omics representations by default

    if not dataset_text_only and hasattr(model[0], "prefix_ds"):
        logger.info(f"Applying prefixes to columns: {prefix_columns}")
        dataset_ready = model[0].prefix_ds(dataset, columns_to_prefix=prefix_columns)
    elif not dataset_text_only and not hasattr(model[0], "prefix_ds"):
        logger.warning(f"Model '{model[0].name}' has no prefix_ds method. Can't apply prefixes.")
        raise ValueError(f"Model '{model[0].name}' has no prefix_ds method. Can't apply prefixes.")
    else:
        logger.info(f"Dataset '{dataset_name}' is text_only - skipping prefixes")
        dataset_ready = dataset

    """
    # Add dataset source information to each row for tracking
    def add_dataset_source(batch):
        batch["dataset_source"] = [dataset_name] * len(batch["anchor"])
        return batch

    # Apply to all splits
    dataset_with_source = {}
    for split_name, split_data in dataset_ready.items():
        dataset_with_source[split_name] = split_data.map(
            add_dataset_source, batched=True, desc=f"Adding dataset source for {dataset_name}"
        )

    dataset_ready = DatasetDict(dataset_with_source)
    """
    # Log prepared dataset info
    logger.info(f"Dataset prepared - Keys: {list(dataset_ready.keys())}")
    for split_name, split_data in dataset_ready.items():
        logger.info(f"  Prepared {split_name} split: {len(split_data)} samples")
        if len(split_data) > 0:
            logger.info(f"    Prepared columns: {list(split_data.column_names)}")

    logger.info(f"Finished processing dataset: {dataset_name}")
    return dataset_ready


def merge_datasets(processed_datasets: list[DatasetDict], shuffle_seed: int = 42) -> DatasetDict:
    """
    Merge multiple processed datasets into a single dataset with randomly interleaved rows.

    Parameters
    ----------
    processed_datasets : list[DatasetDict]
        list of processed datasets to merge
    shuffle_seed : int, optional
        Random seed for shuffling (default: 42)

    Returns
    -------
    DatasetDict
        Merged dataset with randomly interleaved rows
    """
    logger.info(f"Merging {len(processed_datasets)} datasets with shuffle_seed={shuffle_seed}")

    merged_splits = {}

    # Get all available splits (train, val, etc.)
    all_splits = set()
    for dataset in processed_datasets:
        all_splits.update(dataset.keys())

    logger.info(f"Found splits to merge: {list(all_splits)}")

    # Merge each split separately
    for split_name in all_splits:
        split_datasets = []
        total_samples = 0

        for i, dataset in enumerate(processed_datasets):
            if split_name in dataset:
                split_datasets.append(dataset[split_name])
                total_samples += len(dataset[split_name])
                logger.info(f"  Dataset {i}: {len(dataset[split_name])} samples in {split_name} split")
            else:
                logger.warning(f"  Dataset {i}: No {split_name} split found, skipping")

        if split_datasets:
            # Concatenate all datasets for this split
            merged_split = concatenate_datasets(split_datasets)

            # Shuffle the merged dataset to randomly interleave rows
            merged_split = merged_split.shuffle(seed=shuffle_seed)

            merged_splits[split_name] = merged_split
            logger.info(f"  Merged {split_name} split: {len(merged_split)} total samples (shuffled)")
        else:
            logger.warning(f"  No datasets found for {split_name} split")

    merged_dataset = DatasetDict(merged_splits)

    # Log final merged dataset info
    logger.info("Final merged dataset:")
    for split_name, split_data in merged_dataset.items():
        logger.info(f"  {split_name} split: {len(split_data)} samples")
        if len(split_data) > 0:
            logger.info(f"    Columns: {list(split_data.column_names)}")

            # Log dataset source distribution
            if "dataset_source" in split_data.column_names:
                source_counts = {}
                for source in split_data["dataset_source"]:
                    source_counts[source] = source_counts.get(source, 0) + 1
                logger.info(f"    Dataset source distribution: {source_counts}")

    return merged_dataset


@hydra.main(config_path="../conf/training", config_name="train_conf", version_base=None)
def main(cfg: DictConfig):
    """
    Train the MMContext model using merged datasets from multiple sources.

    This script processes multiple datasets individually (applying truncation, prefixing,
    resolving indices, etc.) and then merges them into a single training dataset with
    randomly interleaved rows before training.

    Parameters
    ----------
    cfg : DictConfig
        A Hydra DictConfig object containing all hyperparameters and settings
        for the datasets, model, and training process.
    """
    # Print out the loaded configuration for debugging
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Validate dataset configurations before proceeding
    validate_dataset_configurations(cfg)

    # Check CUDA availability if force_cuda is enabled
    if getattr(cfg, "force_cuda", False):
        if not torch.cuda.is_available():
            error_msg = "CUDA is required (force_cuda=true) but not available on this system"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.info("CUDA is available and force_cuda=true - proceeding with training")
    else:
        cuda_status = "available" if torch.cuda.is_available() else "not available"
        logger.info(f"CUDA is {cuda_status}, force_cuda=false - proceeding with training")

    # get the hydra output dir
    hydra_run_dir = HydraConfig.get().run.dir
    monitor = SystemMonitor(logger=logger)
    monitor.start()

    try:
        # -------------------------------------------------------------------------
        # 1. Create the model (MMContextEncoder => SentenceTransformer modules)
        # -------------------------------------------------------------------------
        # Extract cache_dir from config for dataset loading
        cache_dir = getattr(cfg, "cache_dir", None)
        if cache_dir:
            logger.info(f"Using custom cache directory: {cache_dir}")
        adata_cache_dir = getattr(cfg, "adata_cache_dir", "data/from_nxtcloud")
        logger.info(f"Using adata cache directory: {adata_cache_dir}")

        # Whether to force redownload the dataset
        download_mode = "force_redownload" if getattr(cfg, "force_refresh_cache", False) else "reuse_dataset_if_exists"

        chosen_method = cfg.embedding_method
        input_dim_map = cfg.input_dim_map
        if chosen_method is not None and chosen_method not in input_dim_map:
            raise ValueError(f"Unknown embedding_method '{chosen_method}'. Allowed: {list(input_dim_map.keys())}")

        # Overwrite the model's embedding_dim with the mapped value (only if embedding_method is not null)
        if chosen_method is not None:
            cfg.adapter.omics_input_dim = input_dim_map[chosen_method]
            precomputed_key = f"X_{chosen_method}"
        else:
            precomputed_key = None

        if cfg.model:
            model = SentenceTransformer(cfg.model)
            model[0].freeze_text_encoder = cfg.text_encoder.freeze_text_encoder
            model[0]._manage_text_encoder_freezing()
        else:
            # Get text model kwargs if specified in config and resolve torch dtype strings
            text_model_kwargs = getattr(cfg.text_encoder, "model_kwargs", None)
            if text_model_kwargs:
                # Convert OmegaConf DictConfig to regular dict if needed
                if hasattr(text_model_kwargs, "_content"):
                    text_model_kwargs = dict(text_model_kwargs)
                text_model_kwargs = resolve_torch_dtype_strings(text_model_kwargs)
                logger.info(f"Using text model kwargs: {text_model_kwargs}")

            enc = MMContextEncoder(
                text_encoder_name=cfg.text_encoder.name,
                adapter_hidden_dim=cfg.adapter.hidden_dim,
                adapter_output_dim=cfg.adapter.output_dim,
                output_token_embeddings=False,
                freeze_text_encoder=cfg.text_encoder.freeze_text_encoder,
                unfreeze_last_n_layers=cfg.text_encoder.unfreeze_last_n_layers,
                train_lookup=False,
                joint_adapter_hidden_dim=None,
                text_model_kwargs=text_model_kwargs,
            )
            model = SentenceTransformer(modules=[enc])

        # -------------------------------------------------------------------------
        # 2. Process multiple omics datasets individually
        # -------------------------------------------------------------------------
        processed_datasets = []

        # Track which datasets are text_only vs numeric for model naming
        text_only_datasets = []
        numeric_datasets = []
        text_only_cs_lengths = []  # Track cs_length values from text_only datasets

        # Process omics datasets - these can be text_only or numeric
        if hasattr(cfg, "omics_datasets") and cfg.omics_datasets:
            for dataset_config in cfg.omics_datasets:
                # Construct dataset name - default to multiplets type
                base_name = dataset_config.name
                dataset_type = getattr(dataset_config, "type", "multiplets")
                dataset_name = f"{base_name}_{dataset_type}_{dataset_config.caption}"

                logger.info(f"Processing omics dataset: {dataset_name}")

                # Determine dataset-specific settings FIRST (needed for revision name and primary column)
                layer_axis = getattr(dataset_config, "layer_axis", "obs")  # Default to "obs"
                dataset_text_only = getattr(dataset_config, "text_only", False)

                # Determine primary cell sentence column based on layer_axis only
                if layer_axis == "var":
                    primary_cell_sentence = "cell_sentence_2"  # Gene-based
                elif layer_axis == "obs":
                    primary_cell_sentence = "cell_sentence_1"  # Cell-based
                else:
                    primary_cell_sentence = dataset_config.get("keep_columns", ["cell_sentence_1"])[0]

                logger.info(
                    f"Dataset '{dataset_name}' using layer_axis='{layer_axis}', primary_cell_sentence='{primary_cell_sentence}'"
                )

                # Track dataset mode for model naming
                dataset_cs_length = getattr(dataset_config, "cs_length", None)
                dataset_cs_col = getattr(dataset_config, "cs_col", None)
                if dataset_text_only:
                    text_only_datasets.append(dataset_name)
                    # Track cs_length if specified for text_only datasets
                    if dataset_cs_length and dataset_cs_length > 0:
                        text_only_cs_lengths.append(dataset_cs_length)
                else:
                    numeric_datasets.append(dataset_name)

                logger.info(
                    f"Dataset '{dataset_name}' will be processed as: {'text_only' if dataset_text_only else 'numeric embeddings'}"
                )

                # Generate revision name based on preprocessing parameters
                revision_name = generate_revision_name(dataset_config)

                # Check if processed revision already exists
                if check_revision_exists(dataset_name, revision_name) and dataset_text_only:
                    logger.info(
                        f"Found existing revision '{revision_name}' for dataset '{dataset_name}', loading directly"
                    )
                    dataset_ready = load_dataset(
                        f"jo-mengr/{dataset_name}",
                        revision=revision_name,
                        cache_dir=cache_dir,
                        download_mode=download_mode,
                    )
                    logger.info(f"Successfully loaded preprocessed dataset from revision '{revision_name}'")
                else:
                    if dataset_text_only:
                        logger.info(
                            f"Revision '{revision_name}' not found for dataset '{dataset_name}', processing from scratch"
                        )
                    # Load raw dataset and process it
                    dataset = load_dataset(f"jo-mengr/{dataset_name}", cache_dir=cache_dir, download_mode=download_mode)
                    logger.info(f"Raw dataset loaded - Name: {dataset_name}, Keys: {list(dataset.keys())}")

                    dataset_ready = prepare_single_dataset(
                        dataset,
                        dataset_config,
                        dataset_name,
                        primary_cell_sentence,
                        model,
                        chosen_method,
                        precomputed_key,
                        getattr(cfg, "force_refresh_cache", False),
                        adata_cache_dir,
                    )

                    # Push processed dataset as new revision if enabled
                    if (
                        getattr(cfg, "auto_push_processed_datasets", False)
                        and dataset_cs_length
                        and dataset_cs_length > 0
                        and dataset_cs_col
                    ):
                        push_success = push_dataset_revision(
                            dataset_ready,
                            dataset_name,
                            revision_name,
                            commit_message=f"Processed dataset with {revision_name} settings",
                        )
                        if push_success:
                            logger.info(f"Successfully pushed processed dataset as revision '{revision_name}'")
                        else:
                            logger.warning(f"Failed to push processed dataset as revision '{revision_name}'")
                    elif dataset_cs_length and dataset_cs_length > 0 and dataset_cs_col:
                        logger.info(
                            f"Dataset processed. Set auto_push_processed_datasets=true to automatically push as revision '{revision_name}'"
                        )
                    else:
                        logger.info(
                            f"Dataset '{dataset_name}' does not use cell sentence truncation - skipping revision upload."
                        )

                # Add to list of processed datasets for merging
                processed_datasets.append(dataset_ready)
                logger.info(f"Finished processing omics dataset: {dataset_name}\n")

        # -------------------------------------------------------------------------
        # 2.1. Load bio datasets (biological background knowledge, always text-only)
        # -------------------------------------------------------------------------
        if hasattr(cfg, "bio_datasets") and cfg.bio_datasets:
            logger.info("Loading bio datasets for training...")

            for bio_dataset_config in cfg.bio_datasets:
                dataset_id = bio_dataset_config.id
                dataset_name = bio_dataset_config.name
                logger.info(f"Loading bio dataset: {dataset_name} from {dataset_id}")

                # Bio datasets are always treated as text_only
                text_only_datasets.append(dataset_name)
                logger.info(f"Bio dataset '{dataset_name}' will be processed as text_only")

                # Load the dataset directly using the provided ID
                dataset = load_dataset(
                    dataset_id, revision=bio_dataset_config.revision, cache_dir=cache_dir, download_mode=download_mode
                )
                logger.info(f"Bio dataset loaded - Keys: {list(dataset.keys())}")

                # Log dataset splits and sizes
                for split_name, split_data in dataset.items():
                    logger.info(f"  {split_name} split: {len(split_data)} samples")
                    if len(split_data) > 0:
                        logger.info(f"    Columns: {list(split_data.column_names)}")

                """
                # Bio datasets don't need the same processing as omics datasets
                # Just add dataset source information
                def add_dataset_source(batch):
                    # Bio datasets might not have 'anchor' column, use appropriate column
                    batch_size = len(next(iter(batch.values())))
                    batch["dataset_source"] = [dataset_name] * batch_size
                    return batch

                # Apply to all splits
                dataset_with_source = {}
                for split_name, split_data in dataset.items():
                    dataset_with_source[split_name] = split_data.map(
                        add_dataset_source, batched=True, desc=f"Adding dataset source for {dataset_name}"
                    )

                dataset_ready = DatasetDict(dataset_with_source)
                """
                # Add to list of processed datasets for merging
                processed_datasets.append(dataset_ready)
                logger.info(f"Finished processing bio dataset: {dataset_name}\n")

        # -------------------------------------------------------------------------
        # 3. Merge all processed datasets
        # -------------------------------------------------------------------------
        if not processed_datasets:
            raise ValueError("No datasets were processed. Please check your configuration.")

        logger.info(f"Merging {len(processed_datasets)} processed datasets...")
        shuffle_seed = getattr(cfg, "shuffle_seed", 42)
        merged_dataset = merge_datasets(processed_datasets, shuffle_seed=shuffle_seed)

        # -------------------------------------------------------------------------
        # 4. Create loss function and evaluators for the merged dataset
        # -------------------------------------------------------------------------
        # Use the first dataset's type as the default (assuming all omics datasets have the same type)
        default_dataset_type = "multiplets"
        if hasattr(cfg, "omics_datasets") and cfg.omics_datasets:
            default_dataset_type = getattr(cfg.omics_datasets[0], "type", "multiplets")

        # Create single loss function for the merged dataset
        loss = get_loss(dataset_type=default_dataset_type)

        # Create evaluator for validation split if it exists
        evaluator = None
        if "val" in merged_dataset:
            evaluator = get_evaluator(
                dataset_type=default_dataset_type,
                dataset=merged_dataset["val"],
                batch_size=cfg.trainer.per_device_eval_batch_size,
                current_eval_name="merged_validation",
            )

        # -------------------------------------------------------------------------
        # 5. Set up training arguments
        # -------------------------------------------------------------------------
        # Generate unique model name to avoid conflicts
        unique_model_name = generate_unique_model_name(cfg)
        logger.info(f"Using simplified model name: {unique_model_name}")

        # Log dataset information for reference
        logger.info("Training datasets summary:")
        logger.info(f"  Text-only datasets: {text_only_datasets}")
        logger.info(f"  Numeric datasets: {numeric_datasets}")
        if text_only_cs_lengths:
            logger.info(f"  Text-only dataset cs_lengths: {text_only_cs_lengths}")
            if len(set(text_only_cs_lengths)) > 1:
                logger.info(f"  Multiple cs_length values detected: {set(text_only_cs_lengths)}")

        # Adjust evaluation strategy if there are no validation datasets
        eval_strategy = cfg.trainer.eval_strategy
        eval_steps = cfg.trainer.eval_steps
        if "val" not in merged_dataset or evaluator is None:
            logger.info("No validation dataset available - disabling evaluation during training")
            eval_strategy = "no"
            eval_steps = None

        args = SentenceTransformerTrainingArguments(
            output_dir=hydra_run_dir,
            num_train_epochs=cfg.trainer.num_train_epochs,
            per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
            learning_rate=cfg.trainer.learning_rate,
            warmup_ratio=cfg.trainer.warmup_ratio,
            fp16=cfg.trainer.fp16,
            bf16=cfg.trainer.bf16,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_strategy=cfg.trainer.save_strategy,
            save_steps=cfg.trainer.save_steps,
            save_total_limit=cfg.trainer.save_total_limit,
            max_grad_norm=cfg.trainer.max_grad_norm,
            logging_steps=cfg.trainer.logging_steps,
            run_name=unique_model_name,
            dataloader_num_workers=cfg.trainer.dataloader_num_workers,
        )

        if cfg.trainer.gradient_checkpointing:
            model[0].text_encoder.enable_input_require_grads()
            model[0].text_encoder.gradient_checkpointing_enable()

        # -------------------------------------------------------------------------
        # 6. Create a trainer & train with merged dataset
        # -------------------------------------------------------------------------
        unfreeze_callback = UnfreezeTextEncoderCallback(unfreeze_epoch=cfg.trainer.unfreeze_epoch)

        # Create callbacks list
        callbacks = [unfreeze_callback, WandbCallback()]

        # Add adapter callback if adapter freezing is configured
        if hasattr(cfg, "adapter_freezing"):
            adapter_callback = UnfreezeAdapterCallback(
                freeze_text_adapter=cfg.adapter_freezing.get("freeze_text_adapter", False),
                freeze_omics_adapter=cfg.adapter_freezing.get("freeze_omics_adapter", False),
                unfreeze_text_adapter_epoch=cfg.adapter_freezing.get("unfreeze_text_adapter_epoch", None),
                unfreeze_omics_adapter_epoch=cfg.adapter_freezing.get("unfreeze_omics_adapter_epoch", None),
            )
            callbacks.append(adapter_callback)
            logger.info("Adapter freezing callback added to training")
        else:
            logger.info("No adapter freezing configuration found, skipping adapter callback")

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=merged_dataset["train"],
            eval_dataset=merged_dataset.get("val", None),
            loss=loss,
            evaluator=evaluator,
            callbacks=callbacks,
        )
        trainer.train()

        # -------------------------------------------------------------------------
        # 7. Save the model
        # -------------------------------------------------------------------------
        model_dir = Path(hydra_run_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        print("unique_model_name", unique_model_name)
        model.save(model_dir)
        if cfg.get("push_to_hub", True):
            model.push_to_hub(f"jo-mengr/{unique_model_name}", private=True)
        logger.info(f"Training completed successfully. Model saved to {model_dir}")

    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        monitor.stop()
        monitor.save(hydra_run_dir)
        monitor.plot_metrics(hydra_run_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
