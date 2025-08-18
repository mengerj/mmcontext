import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, HfFolder
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import (
    SentenceTransformer,
    # SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.util import mine_hard_negatives
from transformers.integrations import WandbCallback

from mmcontext.callback import UnfreezeTextEncoderCallback
from mmcontext.eval import SystemMonitor
from mmcontext.models.mmcontextencoder import MMContextEncoder

# from mmcontext.pp.utils import consolidate_low_frequency_categories
from mmcontext.utils import (  # , load_test_adata_from_hf_dataset
    get_evaluator,
    get_loss,
    resolve_negative_indices_and_rename,
    truncate_cell_sentences,
)

logger = logging.getLogger(__name__)


def resolve_torch_dtype_strings(kwargs_dict: dict) -> dict:
    """
    Resolve string representations of torch dtypes to actual torch objects.

    Parameters
    ----------
    kwargs_dict : dict
        Dictionary that may contain torch dtype strings

    Returns
    -------
    dict
        Dictionary with torch dtype strings resolved to actual torch objects
    """
    if not kwargs_dict:
        return kwargs_dict

    resolved = {}
    for key, value in kwargs_dict.items():
        if isinstance(value, str) and value.startswith("torch."):
            # Map common torch dtype strings to actual objects
            dtype_mapping = {
                "torch.float32": torch.float32,
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
                "torch.int8": torch.int8,
                "torch.int16": torch.int16,
                "torch.int32": torch.int32,
                "torch.int64": torch.int64,
                "torch.bool": torch.bool,
                "torch.uint8": torch.uint8,
            }

            if value in dtype_mapping:
                resolved[key] = dtype_mapping[value]
                logger.info(f"Resolved torch dtype string '{value}' to {dtype_mapping[value]}")
            else:
                logger.warning(f"Unknown torch dtype string '{value}', keeping as string")
                resolved[key] = value
        else:
            resolved[key] = value

    return resolved


def check_model_exists(model_name: str, username: str = "jo-mengr") -> bool:
    """
    Check if a model already exists on Hugging Face Hub.

    Parameters
    ----------
    model_name : str
        Name of the model to check
    username : str, optional
        Username/organization on Hugging Face Hub (default: "jo-mengr")

    Returns
    -------
    bool
        True if the model exists, False otherwise
    """
    try:
        api = HfApi()
        full_model_name = f"{username}/{model_name}"
        # Try to get model info - if it exists, this won't raise an error
        api.model_info(full_model_name)
        return True
    except Exception:
        # If any error occurs (404, auth, etc.), assume model doesn't exist
        return False


def generate_unique_model_name(
    cfg: DictConfig,
    username: str = "jo-mengr",
) -> str:
    """
    Generate a unique model name, appending version numbers if needed.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model settings
    username : str, optional
        Username/organization on Hugging Face Hub (default: "jo-mengr")

    Returns
    -------
    str
        Unique model name that doesn't conflict with existing models
    """
    # Generate base model name
    base_name = generate_model_name(cfg)

    # Check if base name is available
    if not check_model_exists(base_name, username):
        logger.info(f"Model name '{base_name}' is available")
        return base_name

    # If base name exists, try with version numbers
    version = 2
    while True:
        versioned_name = f"{base_name}-v{version}"
        if not check_model_exists(versioned_name, username):
            logger.info(f"Model name '{base_name}' exists, using '{versioned_name}' instead")
            return versioned_name
        version += 1

        # Safety check to prevent infinite loop
        if version > 10:
            logger.warning(f"Reached version {version} for model {base_name}. Using timestamp suffix.")
            import time

            timestamp = int(time.time())
            return f"{base_name}-{timestamp}"


def generate_model_name(
    cfg: DictConfig,
    dataset_configs: list = None,
    cs_len: int = None,
    text_only_datasets: list = None,
    numeric_datasets: list = None,
) -> str:
    """
    Generate a simplified model name focusing on core architecture components.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model settings
    dataset_configs : list, optional
        List of dataset configurations (unused, kept for compatibility)
    cs_len : int, optional
        Length of cell sentences (unused, kept for compatibility)
    text_only_datasets : list, optional
        List of dataset names that are processed as text_only (unused, kept for compatibility)
    numeric_datasets : list, optional
        List of dataset names that use numeric embeddings (unused, kept for compatibility)

    Returns
    -------
    str
        Generated model name in format: mmcontext-{encoder}-{embedding_method}
    """
    # Get text encoder name (simplified)
    text_encoder_name = cfg.text_encoder.name
    if "pubmedbert" in text_encoder_name.lower():
        encoder_str = "pubmedbert"
    elif "biobert" in text_encoder_name.lower():
        encoder_str = "biobert"
    elif "modernbert" in text_encoder_name.lower():
        if "bioclinical" in text_encoder_name.lower():
            encoder_str = "biomodern"
        else:
            encoder_str = "modernbert"
    elif "qwen" in text_encoder_name.lower():
        encoder_str = "qwen"
    else:
        # Take the last part after '/'
        encoder_str = text_encoder_name.split("/")[-1]

    # Get embedding method
    embedding_method = cfg.embedding_method

    # Construct the simplified model name
    model_name = f"mmcontext-{encoder_str}-{embedding_method}"

    return model_name


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

            # Primary cell sentence column and layer axis are now determined per dataset
        # -------------------------------------------------------------------------
        # 2. Load multiple datasets
        # -------------------------------------------------------------------------
        train_datasets = {}
        val_datasets = {}
        losses = {}
        evaluators = []

        # Track which datasets are text_only vs numeric for model naming
        text_only_datasets = []
        numeric_datasets = []
        text_only_cs_lengths = []  # Track cs_length values from text_only datasets

        # Process omics datasets - these can be text_only or numeric
        if hasattr(cfg, "omics_datasets") and cfg.omics_datasets:
            for dataset_config in cfg.omics_datasets:
                # Construct dataset name
                base_name = dataset_config.name
                dataset_name = f"{base_name}_{dataset_config.type}_{dataset_config.caption}"

                logger.info(f"Loading omics dataset: {dataset_name}")

                # Load the dataset
                dataset = load_dataset(f"jo-mengr/{dataset_name}")
                logger.info(f"Raw dataset loaded - Keys: {list(dataset.keys())}")

                # Log dataset splits and sizes
                for split_name, split_data in dataset.items():
                    logger.info(f"  {split_name} split: {len(split_data)} samples")
                    if len(split_data) > 0:
                        logger.info(f"    Columns: {list(split_data.column_names)}")
                        # Log first sample for debugging
                        sample = split_data[0]
                        logger.info(f"    First sample keys: {list(sample.keys())}")
                        # Log sample content (truncated for readability)
                        for key, value in sample.items():
                            if isinstance(value, str):
                                preview = value[:100] + "..." if len(value) > 100 else value
                                logger.info(f"      {key}: {preview}")
                            else:
                                logger.info(f"      {key}: {type(value)} - {value}")

                # Determine dataset-specific settings
                layer_axis = getattr(dataset_config, "layer_axis", "obs")  # Default to "obs"

                # Determine primary cell sentence column based on layer_axis only
                if layer_axis == "var":
                    primary_cell_sentence = "cell_sentence_2"  # Gene-based
                else:
                    primary_cell_sentence = "cell_sentence_1"  # Cell-based

                logger.info(
                    f"Dataset '{dataset_name}' using layer_axis='{layer_axis}', primary_cell_sentence='{primary_cell_sentence}'"
                )
                eval_name = f"{dataset_name}_{primary_cell_sentence}"
                # Check if this dataset should be processed as text_only
                dataset_text_only = getattr(dataset_config, "text_only", False)

                # Get dataset-specific cell sentence truncation parameters
                dataset_cs_length = getattr(dataset_config, "cs_length", None)
                dataset_cs_col = getattr(dataset_config, "cs_col", None)

                # Track dataset mode for model naming
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

                # Apply cell sentence truncation if configured (only for text_only datasets with cs_length specified)

                if dataset_text_only and dataset_cs_length and dataset_cs_length > 0 and dataset_cs_col:
                    logger.info(
                        f"Truncating cell sentences in column '{dataset_cs_col}' to {dataset_cs_length} tokens for text_only dataset"
                    )

                    # Apply truncation to all splits that contain the specified column
                    truncated_dataset = {}
                    for split_name, split_data in dataset.items():
                        if dataset_cs_col in split_data.column_names:
                            truncated_dataset[split_name] = truncate_cell_sentences(
                                split_data,
                                dataset_cs_col,
                                dataset_cs_length,
                                filter_strings=dataset_config.get("gene_filter_strings", None),
                            )
                            logger.info(f"  Truncated {split_name} split")
                        else:
                            truncated_dataset[split_name] = split_data
                            logger.info(
                                f"  Column '{dataset_cs_col}' not found in {split_name} split, keeping original"
                            )

                    # Use the truncated dataset for the rest of the processing
                    dataset = DatasetDict(truncated_dataset)
                elif dataset_text_only:
                    logger.info(
                        f"Text_only dataset '{dataset_name}' - no truncation applied (cs_length or cs_col not specified)"
                    )
                else:
                    logger.info(f"Non-text_only dataset '{dataset_name}' - skipping cell sentence truncation")

                # Step 1: Handle embedding registration FIRST (needs access to raw dataset with all columns)
                if not dataset_text_only:
                    logger.info(f"Loading numeric embeddings for dataset '{dataset_name}' (before column selection)")
                    token_df, _ = enc.get_initial_embeddings(
                        dataset,
                        layer_key=precomputed_key,
                        download_dir=f"data/from_nxtcloud/{dataset_name}",
                        axis=layer_axis,
                        overwrite=getattr(cfg, "force_refresh_cache", False),  # Add this parameter to config
                    )
                    enc.register_initial_embeddings(token_df, data_origin=chosen_method)
                else:
                    # In text_only mode, we'll use cell sentences directly
                    logger.info(
                        f"Dataset '{dataset_name}' using text_only mode - cell sentences will be processed as text"
                    )

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
                if dataset_config.type == "multiplets":
                    logger.info("Resolving negative indices and renaming columns for multiplet dataset")
                    index_col_to_use = index_column if index_column else "sample_idx"
                    dataset = resolve_negative_indices_and_rename(
                        dataset,
                        primary_cell_sentence_col=primary_cell_sentence,
                        positive_col="positive",
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
                if dataset_text_only:
                    # For text_only datasets, don't add prefix
                    prefix_columns = []
                logger.info(f"Applying prefixes to columns: {prefix_columns}")

                dataset_ready = enc.prefix_ds(dataset, columns_to_prefix=prefix_columns)

                # Log prepared dataset info
                logger.info(f"Dataset prepared - Keys: {list(dataset_ready.keys())}")
                for split_name, split_data in dataset_ready.items():
                    logger.info(f"  Prepared {split_name} split: {len(split_data)} samples")
                    if len(split_data) > 0:
                        logger.info(f"    Prepared columns: {list(split_data.column_names)}")
                        # Log first prepared sample
                        prepared_sample = split_data[0]
                        logger.info(f"    First prepared sample keys: {list(prepared_sample.keys())}")
                        for key, value in prepared_sample.items():
                            if isinstance(value, str):
                                preview = value[:100] + "..." if len(value) > 100 else value
                                logger.info(f"      {key}: {preview}")
                            else:
                                logger.info(f"      {key}: {type(value)}")
                logger.info(f"Finished processing omics dataset: {dataset_name}\n")

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
                    current_eval_name=eval_name,
                )
                evaluators.append(evaluator)

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
                dataset = load_dataset(dataset_id, revision="hard_negatives")
                logger.info(f"Bio dataset loaded - Keys: {list(dataset.keys())}")

                # Log dataset splits and sizes
                for split_name, split_data in dataset.items():
                    logger.info(f"  {split_name} split: {len(split_data)} samples")
                    if len(split_data) > 0:
                        logger.info(f"    Columns: {list(split_data.column_names)}")

                if bio_dataset_config.anchor_col_name and bio_dataset_config.positive_col_name:
                    # Apply to all splits that contain the specified columns
                    dataset_processed = {}
                    for split_name, split_data in dataset.items():
                        dataset_with_negatives = split_data
                        # perform hard negative mining with the text model of interest
                        # dataset_with_negatives = mine_hard_negatives(
                        #    split_data,
                        #    model=SentenceTransformer(cfg.text_encoder.name),
                        #    anchor_column_name=bio_dataset_config.anchor_col_name,
                        #    positive_column_name=bio_dataset_config.positive_col_name,
                        # )
                        # rename negative col to negative_1 for the get_evaluator function
                        # dataset_with_negatives = dataset_with_negatives.rename_column("negative", "negative_1")
                        dataset_processed[split_name] = dataset_with_negatives
                    dataset_ready = DatasetDict(dataset_processed)
                else:
                    # If no keep_columns specified, use the dataset as-is
                    dataset_ready = dataset
                    logger.info("  No keep_columns specified, using dataset as-is")

                logger.info(f"Bio dataset prepared - Keys: {list(dataset_ready.keys())}")

                # Add only the train split to train_datasets (no validation for bio datasets)
                if "train" in dataset_ready:
                    train_datasets[dataset_name] = dataset_ready["train"]
                    logger.info(f"Added bio dataset '{dataset_name}' to training set")
                    # add a random sample of 1000 samples to the validation set
                    val_datasets[dataset_name] = dataset_ready["train"].select(
                        range(1000)
                    )  # add the training data also as evaluation just to check if these bio datasets are considered
                    evaluator = get_evaluator(
                        dataset_type=bio_dataset_config.type,
                        dataset=val_datasets[dataset_name],
                        batch_size=cfg.trainer.per_device_eval_batch_size,
                        current_eval_name=dataset_name,
                    )
                    evaluators.append(evaluator)
                else:
                    logger.warning(f"Bio dataset '{dataset_name}' has no 'train' split, skipping")

                # Create loss function for this dataset type
                losses[dataset_name] = get_loss(dataset_type=bio_dataset_config.type)

                # Note: No evaluators created for bio datasets
                logger.info(f"Finished processing bio dataset: {dataset_name}\n")

        # Build the sentence Trasnformer model
        modules = [enc]
        model = SentenceTransformer(modules=modules)

        # Add this after creating your model in train.py, around line 558-562

        # Diagnostic: Check parameter status before training
        print("\n=== GRADIENT DEBUGGING ===")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {trainable_params:,} / {total_params:,} trainable parameters")

        if trainable_params == 0:
            print("ERROR: No trainable parameters found!")
            print("Checking MMContextEncoder directly:")
            enc_trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
            enc_total = sum(p.numel() for p in enc.parameters())
            print(f"  MMContextEncoder: {enc_trainable:,} / {enc_total:,} trainable")

            # Print first few trainable parameters
            print("First few trainable parameters:")
            count = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}: {param.shape}")
                    count += 1
                    if count >= 5:
                        break

            if count == 0:
                print("  No trainable parameters found in model!")

        print("=== END GRADIENT DEBUGGING ===\n")

        # Combine all evaluators into a single sequential evaluator (only if we have evaluators)
        if evaluators:
            dev_evaluator = SequentialEvaluator(evaluators)
        else:
            dev_evaluator = None
            logger.info("No evaluators created - validation will be skipped")

        # -------------------------------------------------------------------------
        # 3. Load test datasets
        # -------------------------------------------------------------------------
        # test_datasets = {}
        # for test_config in cfg.test_datasets:
        #    test_datasets[test_config.name] = load_dataset(test_config.name)["test"]

        # -------------------------------------------------------------------------
        # 4. Compute the correct embedding dimension based on method
        # -------------------------------------------------------------------------

        # -------------------------------------------------------------------------
        # 6. Set up training arguments
        # -------------------------------------------------------------------------

        # Generate unique model name to avoid conflicts
        unique_model_name = generate_unique_model_name(cfg)
        logger.info(f"Using simplified model name: {unique_model_name}")

        # Log dataset information for reference (this detailed info can be used in model documentation)
        logger.info("Training datasets summary:")
        logger.info(f"  Text-only datasets: {text_only_datasets}")
        logger.info(f"  Numeric datasets: {numeric_datasets}")
        if text_only_cs_lengths:
            logger.info(f"  Text-only dataset cs_lengths: {text_only_cs_lengths}")
            if len(set(text_only_cs_lengths)) > 1:
                logger.info(f"  Multiple cs_length values detected: {set(text_only_cs_lengths)}")

        # Log omics datasets details
        if hasattr(cfg, "omics_datasets") and cfg.omics_datasets:
            logger.info("  Omics datasets configuration:")
            for dataset_config in cfg.omics_datasets:
                logger.info(
                    f"    - {dataset_config.name}: type={dataset_config.type}, "
                    f"text_only={getattr(dataset_config, 'text_only', False)}, "
                    f"layer_axis={getattr(dataset_config, 'layer_axis', 'obs')}"
                )

        # Log bio datasets details
        if hasattr(cfg, "bio_datasets") and cfg.bio_datasets:
            logger.info("  Bio datasets configuration:")
            for dataset_config in cfg.bio_datasets:
                logger.info(f"    - {dataset_config.name}: type={dataset_config.type}")

        # Adjust evaluation strategy if there are no validation datasets
        eval_strategy = cfg.trainer.eval_strategy
        eval_steps = cfg.trainer.eval_steps
        if not val_datasets or dev_evaluator is None:
            logger.info("No validation datasets available - disabling evaluation during training")
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
            gradient_checkpointing=cfg.trainer.gradient_checkpointing,
        )

        if cfg.trainer.gradient_checkpointing:
            model[
                0
            ].text_encoder.enable_input_require_grads()  # Enable gradient flow for input_ids, otherwise partial unfreezing will lead to no gradient flow
            model[0].text_encoder.gradient_checkpointing_enable()

        # -------------------------------------------------------------------------
        # 8. Create a trainer & train with multiple datasets
        # -------------------------------------------------------------------------

        unfreeze_callback = UnfreezeTextEncoderCallback(unfreeze_epoch=cfg.trainer.unfreeze_epoch)

        # Create callbacks list
        callbacks = [unfreeze_callback, WandbCallback()]

        # Add joint adapter callback if joint adapter is configured
        # if cfg.joint_adapter.hidden_dim is not None and cfg.joint_adapter.hidden_dim > 0:
        #    joint_adapter_callback = UnfreezeJointAdapterCallback(unfreeze_epoch=cfg.joint_adapter.unfreeze_epoch)
        #    callbacks.append(joint_adapter_callback)

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_datasets,
            eval_dataset=val_datasets,
            loss=losses,
            evaluator=dev_evaluator,  # Pass the sequential evaluator instead of the dictionary
            callbacks=callbacks,
        )
        trainer.train()

        model_dir = Path(hydra_run_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir)
        model.push_to_hub(f"jo-mengr/{unique_model_name}")
        logger.info(f"Training completed successfully. Model saved to {model_dir}")
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        monitor.stop()
        monitor.save(hydra_run_dir)
        monitor.plot_metrics(hydra_run_dir)
    # Evaluate on test datasets


#    for test_dataset_config in cfg.test_datasets:
#        test_dataset = test_datasets[test_dataset_config.name]
#        test_evaluator = get_evaluator(
#            dataset_type=test_dataset_config.type,
#            dataset=test_dataset,
#            batch_size=cfg.trainer.per_device_eval_batch_size,
#        )
#        test_results = test_evaluator(model)
#        logger.info(f"Test results for {test_dataset_config.name}: {test_results}")

# -------------------------------------------------------------------------
# 9. Save the model and run system monitor
# -------------------------------------------------------------------------
# model[0].processor.omics_processor.clear_cache()  # Important or you will save anndata objects stored in the cache


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
