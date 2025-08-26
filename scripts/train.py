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

from mmcontext.callback import UnfreezeAdapterCallback, UnfreezeTextEncoderCallback
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


def validate_dataset_configurations(cfg: DictConfig) -> None:
    """
    Validate Dataset Configurations

    Validate dataset configurations for consistency between text_only and layer_axis settings,
    and ensure embedding_method is specified when needed.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing dataset settings

    Raises
    ------
    ValueError
        If any dataset has inconsistent text_only and layer_axis settings,
        or if embedding_method is missing when datasets require numeric embeddings
    """
    errors = []

    # Check if any dataset requires numeric embeddings (text_only=false)
    requires_embedding_method = False

    # Validate omics datasets
    if hasattr(cfg, "omics_datasets") and cfg.omics_datasets:
        for i, dataset_config in enumerate(cfg.omics_datasets):
            dataset_name = dataset_config.name
            text_only = getattr(dataset_config, "text_only", False)
            layer_axis = getattr(dataset_config, "layer_axis", "obs")

            # Check if this dataset requires numeric embeddings
            if not text_only:
                requires_embedding_method = True

            if text_only and layer_axis != "var":
                errors.append(
                    f"Omics dataset '{dataset_name}' (index {i}): text_only=true requires layer_axis='var', "
                    f"but got layer_axis='{layer_axis}'"
                )
            elif not text_only and layer_axis != "obs":
                errors.append(
                    f"Omics dataset '{dataset_name}' (index {i}): text_only=false requires layer_axis='obs', "
                    f"but got layer_axis='{layer_axis}'"
                )

    # Check if embedding_method is specified when needed
    embedding_method = getattr(cfg, "embedding_method", None)
    if requires_embedding_method and embedding_method is None:
        errors.append(
            "embedding_method must be specified when any dataset has text_only=false. "
            f"Available methods: {list(cfg.input_dim_map.keys()) if hasattr(cfg, 'input_dim_map') else 'check input_dim_map in config'}"
        )

    # Bio datasets are always text_only, so no validation needed for them

    if errors:
        error_msg = "Dataset configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Dataset configuration validation passed")


def generate_revision_name(dataset_config: DictConfig) -> str:
    """
    Generate a revision name for a dataset based on preprocessing parameters.

    Parameters
    ----------
    dataset_config : DictConfig
        Dataset configuration containing layer_axis, gene_filter_strings, cs_length, etc.

    Returns
    -------
    str
        Revision name in format: {layer_axis}[_rps_rpl_mt][_cs{cs_length}]
    """
    parts = []

    # Add layer axis
    layer_axis = getattr(dataset_config, "layer_axis", "obs")
    parts.append(layer_axis)

    # Add gene filter strings if layer_axis is var
    if layer_axis == "var":
        gene_filter_strings = getattr(dataset_config, "gene_filter_strings", None)
        if gene_filter_strings:
            # Convert to lowercase and join with underscores
            filter_str = "_".join([s.lower() for s in gene_filter_strings])
            parts.append(filter_str)

        # Add cs_length if specified
        cs_length = getattr(dataset_config, "cs_length", None)
        if cs_length and cs_length > 0:
            parts.append(f"cs{cs_length}")

    return "_".join(parts)


def push_dataset_revision(
    dataset: "DatasetDict",
    dataset_name: str,
    revision: str,
    username: str = "jo-mengr",
    commit_message: str = None,
) -> bool:
    """
    Push a processed dataset as a new revision to HuggingFace Hub.

    Parameters
    ----------
    dataset : DatasetDict
        The processed dataset to push
    dataset_name : str
        Name of the dataset
    revision : str
        Revision name for the processed dataset
    username : str, optional
        Username/organization on Hugging Face Hub (default: "jo-mengr")
    commit_message : str, optional
        Commit message for the revision

    Returns
    -------
    bool
        True if push was successful, False otherwise
    """
    try:
        full_dataset_name = f"{username}/{dataset_name}"
        if commit_message is None:
            commit_message = f"Add preprocessed dataset revision: {revision}"

        logger.info(f"Pushing dataset '{dataset_name}' as revision '{revision}' to {full_dataset_name}")
        dataset.push_to_hub(
            full_dataset_name,
            revision=revision,
            commit_message=commit_message,
        )
        logger.info(f"Successfully pushed revision '{revision}' for dataset '{dataset_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to push revision '{revision}' for dataset '{dataset_name}': {e}")
        return False


def check_revision_exists(dataset_name: str, revision: str, username: str = "jo-mengr") -> bool:
    """
    Check if a specific revision exists for a dataset on Hugging Face Hub.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    revision : str
        Revision name to check
    username : str, optional
        Username/organization on Hugging Face Hub (default: "jo-mengr")

    Returns
    -------
    bool
        True if the revision exists, False otherwise
    """
    try:
        api = HfApi()
        full_dataset_name = f"{username}/{dataset_name}"
        # Try to get dataset info with specific revision
        api.dataset_info(full_dataset_name, revision=revision)
        return True
    except Exception:
        # If any error occurs (404, auth, etc.), assume revision doesn't exist
        return False


def prepare_ds(
    dataset,
    dataset_config: DictConfig,
    dataset_name: str,
    primary_cell_sentence: str,
    model: "SentenceTransformer",
    chosen_method: str = None,
    precomputed_key: str = None,
    force_refresh_cache: bool = False,
) -> "DatasetDict":
    """
    Prepare a dataset by applying all preprocessing steps.

    Parameters
    ----------
    dataset : DatasetDict
        Raw dataset loaded from HuggingFace
    dataset_config : DictConfig
        Dataset configuration
    dataset_name : str
        Name of the dataset
    primary_cell_sentence : str
        Primary cell sentence column name
    model : SentenceTransformer
        The model
    chosen_method : str, optional
        Embedding method for numeric datasets
    precomputed_key : str, optional
        Key for precomputed embeddings
    force_refresh_cache : bool, optional
        Whether to force refresh cache

    Returns
    -------
    DatasetDict
        Processed dataset ready for training
    """
    logger.info(f"Preparing dataset: {dataset_name}")

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
            download_dir=f"data/from_nxtcloud/{dataset_name}",
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

    # Log prepared dataset info
    logger.info(f"Dataset prepared - Keys: {list(dataset_ready.keys())}")
    for split_name, split_data in dataset_ready.items():
        logger.info(f"  Prepared {split_name} split: {len(split_data)} samples")
        if len(split_data) > 0:
            logger.info(f"    Prepared columns: {list(split_data.column_names)}")

    logger.info(f"Finished processing dataset: {dataset_name}")
    return dataset_ready


def generate_model_name(
    cfg: DictConfig,
) -> str:
    """
    Generate a simplified model name focusing on core architecture components.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model settings

    Returns
    -------
    str
        Generated model name in format: mmcontext-{encoder}-{embedding_method}[-{tag}]
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

    # Construct the base model name
    if embedding_method is not None:
        model_name = f"mmcontext-{encoder_str}-{embedding_method}"
    else:
        model_name = f"mmcontext-{encoder_str}"

    # Add custom tag if provided
    tag = getattr(cfg, "tag", None)
    if tag and tag.strip():  # Check if tag is not None and not empty/whitespace
        model_name = f"{model_name}-{tag.strip()}"

    return model_name


@hydra.main(config_path="../conf/training", config_name="train_conf", version_base=None)
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
                else:
                    primary_cell_sentence = "cell_sentence_1"  # Cell-based

                logger.info(
                    f"Dataset '{dataset_name}' using layer_axis='{layer_axis}', primary_cell_sentence='{primary_cell_sentence}'"
                )
                eval_name = f"{dataset_name}_{primary_cell_sentence}"

                # Track dataset mode for model naming
                dataset_cs_length = getattr(dataset_config, "cs_length", None)
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
                    dataset_ready = load_dataset(f"jo-mengr/{dataset_name}", revision=revision_name)
                    logger.info(f"Successfully loaded preprocessed dataset from revision '{revision_name}'")
                else:
                    if dataset_text_only:
                        logger.info(
                            f"Revision '{revision_name}' not found for dataset '{dataset_name}', processing from scratch"
                        )
                    # Load raw dataset and process it
                    dataset = load_dataset(f"jo-mengr/{dataset_name}")
                    logger.info(f"Raw dataset loaded - Name: {dataset_name}, Keys: {list(dataset.keys())}")

                    dataset_ready = prepare_ds(
                        dataset,
                        dataset_config,
                        dataset_name,
                        primary_cell_sentence,
                        model,
                        chosen_method,
                        precomputed_key,
                        getattr(cfg, "force_refresh_cache", False),
                    )

                    # Push processed dataset as new revision if enabled
                    if getattr(cfg, "auto_push_processed_datasets", False) and dataset_text_only:
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
                    elif dataset_text_only:
                        logger.info(
                            f"Dataset processed. Set auto_push_processed_datasets=true to automatically push as revision '{revision_name}'"
                        )
                    else:
                        logger.info(f"Dataset '{dataset_name}' is numeric - skipping revision upload.")

                # Log final dataset info
                logger.info(f"Dataset ready - Keys: {list(dataset_ready.keys())}")
                for split_name, split_data in dataset_ready.items():
                    logger.info(f"  {split_name} split: {len(split_data)} samples")
                    if len(split_data) > 0:
                        logger.info(f"    Columns: {list(split_data.column_names)}")

                # Add train split to train_datasets dictionary
                train_datasets[dataset_name] = dataset_ready["train"]

                # Add validation split to val_datasets dictionary
                val_datasets[dataset_name] = dataset_ready["val"]

                # Create loss function for this dataset type
                losses[dataset_name] = get_loss(dataset_type=dataset_type)

                evaluator = get_evaluator(
                    dataset_type=dataset_type,
                    dataset=dataset_ready["val"],
                    batch_size=cfg.trainer.per_device_eval_batch_size,
                    current_eval_name=eval_name,
                )
                evaluators.append(evaluator)

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
                dataset = load_dataset(dataset_id, revision=bio_dataset_config.revision)
                logger.info(f"Bio dataset loaded - Keys: {list(dataset.keys())}")

                # Log dataset splits and sizes
                for split_name, split_data in dataset.items():
                    logger.info(f"  {split_name} split: {len(split_data)} samples")
                    if len(split_data) > 0:
                        logger.info(f"    Columns: {list(split_data.column_names)}")

                dataset_ready = dataset

                # Add only the train split to train_datasets (no validation for bio datasets)
                if "train" in dataset_ready:
                    train_datasets[dataset_name] = dataset_ready["train"]
                    logger.info(f"Added bio dataset '{dataset_name}' to training set")
                    # add a random sample of 1000 samples to the validation set
                    val_datasets[dataset_name] = dataset_ready["train"].select(
                        range(1000)
                    )  # add the training data also as evaluation just to check if these bio datasets are considered
                    dataset_type = getattr(bio_dataset_config, "type", "multiplets")
                    evaluator = get_evaluator(
                        dataset_type=dataset_type,
                        dataset=val_datasets[dataset_name],
                        batch_size=cfg.trainer.per_device_eval_batch_size,
                        current_eval_name=dataset_name,
                    )
                    evaluators.append(evaluator)
                else:
                    logger.warning(f"Bio dataset '{dataset_name}' has no 'train' split, skipping")

                # Create loss function for this dataset type
                losses[dataset_name] = get_loss(dataset_type=dataset_type)

                # Note: No evaluators created for bio datasets
                logger.info(f"Finished processing bio dataset: {dataset_name}\n")

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
                    f"    - {dataset_config.name}:, "
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
        print("unique_model_name", unique_model_name)
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
