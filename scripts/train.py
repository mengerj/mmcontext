import logging
import os
import sys
from pathlib import Path

import hydra
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
from transformers.integrations import WandbCallback

from mmcontext.callback import UnfreezeTextEncoderCallback
from mmcontext.eval import SystemMonitor
from mmcontext.models.mmcontextencoder import MMContextEncoder

# from mmcontext.pp.utils import consolidate_low_frequency_categories
from mmcontext.utils import get_evaluator, get_loss  # , load_test_adata_from_hf_dataset

logger = logging.getLogger(__name__)


def truncate_cell_sentences(dataset, column_name: str, max_length: int):
    """
    Truncate cell sentences to the first max_length tokens efficiently.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace dataset containing cell sentences
    column_name : str
        Name of the column containing cell sentences to truncate
    max_length : int
        Maximum number of tokens/words to keep (first n elements)

    Returns
    -------
    Dataset
        Dataset with truncated cell sentences
    """

    def _truncate_batch(batch):
        truncated_sentences = []
        for sentence in batch[column_name]:
            # Split by spaces and take first max_length tokens
            tokens = sentence.split()[:max_length]
            # Join back with spaces
            truncated_sentences.append(" ".join(tokens))
        batch[column_name] = truncated_sentences
        return batch

    return dataset.map(_truncate_batch, batched=True, desc=f"Truncating {column_name} to {max_length} tokens")


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
    dataset_configs: list,
    cs_len: int = None,
    text_only_datasets: list = None,
    numeric_datasets: list = None,
    username: str = "jo-mengr",
) -> str:
    """
    Generate a unique model name, appending version numbers if needed.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model settings
    dataset_configs : list
        List of dataset configurations
    cs_len : int, optional
        Length of cell sentences when text_only is True
    text_only_datasets : list, optional
        List of dataset names that are processed as text_only
    numeric_datasets : list, optional
        List of dataset names that use numeric embeddings
    username : str, optional
        Username/organization on Hugging Face Hub (default: "jo-mengr")

    Returns
    -------
    str
        Unique model name that doesn't conflict with existing models
    """
    # Generate base model name
    base_name = generate_model_name(cfg, dataset_configs, cs_len, text_only_datasets, numeric_datasets)

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
    dataset_configs: list,
    cs_len: int = None,
    text_only_datasets: list = None,
    numeric_datasets: list = None,
) -> str:
    """
    Generate a descriptive model name based on configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model settings
    dataset_configs : list
        List of dataset configurations
    cs_len : int, optional
        Length of cell sentences when text_only is True
    text_only_datasets : list, optional
        List of dataset names that are processed as text_only
    numeric_datasets : list, optional
        List of dataset names that use numeric embeddings

    Returns
    -------
    str
        Generated model name
    """
    # Process dataset names
    dataset_parts = []
    captions = []

    for dataset_config in dataset_configs:
        # Get shortened dataset name by replacing specific parts
        dataset_name = dataset_config.name
        # Replace "cellxgene_pseudo_bulk" with "cg" while keeping any suffixes
        if dataset_name.startswith("cellxgene_pseudo_bulk"):
            shortened_name = dataset_name.replace("cellxgene_pseudo_bulk", "cg")
        else:
            shortened_name = dataset_name

        dataset_parts.append(shortened_name)

        # Shorten caption names for model naming
        caption = dataset_config.caption
        if caption == "natural_language_annotation":
            caption = "nla"
        captions.append(caption)

    # Combine dataset names
    datasets_str = "-".join(dataset_parts)

    # Get unique captions
    unique_captions = list(set(captions))
    captions_str = "-".join(unique_captions)

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
    else:
        # Take the last part after '/'
        encoder_str = text_encoder_name.split("/")[-1]

    # Get output dimension
    output_dim = cfg.adapter.output_dim

    # Determine method string based on dataset mix
    text_only_datasets = text_only_datasets or []
    numeric_datasets = numeric_datasets or []

    if text_only_datasets and numeric_datasets:
        # Mixed mode: some datasets are text_only, some are numeric
        method_str = f"mixed-{cfg.embedding_method}"
        if cs_len is not None:
            method_str = f"{method_str}-text_{cs_len}"
    elif text_only_datasets and not numeric_datasets:
        # All datasets are text_only
        method_str = "text_only"
        if cs_len is not None:
            method_str = f"text_only_{cs_len}"
        if not cfg.gene_based_cell_sentence:
            raise ValueError(
                "All datasets are text_only, but gene_based_cell_sentence is false. This will lead to training a text model on sample ids, which is not what you want."
            )
    else:
        # All datasets use numeric embeddings (or no datasets specified)
        method_str = cfg.embedding_method

    # Get cell sentence type
    if cfg.gene_based_cell_sentence:
        cell_sentence_str = "feat_cs"  # feature-based cell sentences (genes)
    else:
        cell_sentence_str = "sample_cs"  # sample-based cell sentences

    # Construct the model name
    model_name = f"mmcontext-{datasets_str}-{captions_str}-{encoder_str}-{output_dim}-{method_str}-{cell_sentence_str}"

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
            enc = MMContextEncoder(
                text_encoder_name=cfg.text_encoder.name,
                adapter_hidden_dim=cfg.adapter.hidden_dim,
                adapter_output_dim=cfg.adapter.output_dim,
                output_token_embeddings=False,
                freeze_text_encoder=cfg.text_encoder.freeze_text_encoder,
                unfreeze_last_n_layers=cfg.text_encoder.unfreeze_last_n_layers,
                train_lookup=False,
                joint_adapter_hidden_dim=None,
            )

            # get the main cell sentence column
            if hasattr(cfg, "cs_col") and cfg.cs_col:
                primary_cell_sentence = cfg.cs_col
            elif cfg.gene_based_cell_sentence:
                primary_cell_sentence = "cell_sentence_2"
            else:
                primary_cell_sentence = "cell_sentence_1"

            # Set layer axis based on gene_based_cell_sentence setting
            if cfg.gene_based_cell_sentence:
                layer_axis = "var"
            else:
                layer_axis = "obs"
            # Get the correct columns for the given data format
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

                # Check if this dataset should be processed as text_only
                dataset_text_only = getattr(dataset_config, "text_only", False)

                # Track dataset mode for model naming
                if dataset_text_only:
                    text_only_datasets.append(dataset_name)
                else:
                    numeric_datasets.append(dataset_name)

                logger.info(
                    f"Dataset '{dataset_name}' will be processed as: {'text_only' if dataset_text_only else 'numeric embeddings'}"
                )

                # Apply cell sentence truncation if configured (only for omics datasets when text_only=true)
                if cfg.cs_length and cfg.cs_length > 0 and dataset_text_only:
                    cs_col = getattr(cfg, "cs_col", "cell_sentence_2")  # Default to cell_sentence_2
                    logger.info(f"Truncating cell sentences in column '{cs_col}' to {cfg.cs_length} tokens")

                    # Apply truncation to all splits that contain the specified column
                    truncated_dataset = {}
                    for split_name, split_data in dataset.items():
                        if cs_col in split_data.column_names:
                            truncated_dataset[split_name] = truncate_cell_sentences(split_data, cs_col, cfg.cs_length)
                            logger.info(f"  Truncated {split_name} split")
                        else:
                            truncated_dataset[split_name] = split_data
                            logger.info(f"  Column '{cs_col}' not found in {split_name} split, keeping original")

                    # Use the truncated dataset for the rest of the processing
                    dataset = DatasetDict(truncated_dataset)

                # Handle embedding registration based on dataset-specific text_only setting
                if not dataset_text_only:
                    logger.info(f"Loading numeric embeddings for dataset '{dataset_name}'")
                    token_df, _ = enc.get_initial_embeddings(
                        dataset,
                        layer_key=precomputed_key,
                        download_dir=f"../../data/from_nxtcloud/{dataset_name}",
                        axis=layer_axis,
                    )
                    enc.register_initial_embeddings(token_df, data_origin=chosen_method)
                else:
                    # In text_only mode, we'll use cell sentences directly
                    logger.info(
                        f"Dataset '{dataset_name}' using text_only mode - cell sentences will be processed as text"
                    )

                # Add the prefix expected by the model (prefix=False for text_only datasets)
                # enc.prepare_ds is applied to all omics datasets
                dataset_ready = enc.prepare_ds(
                    dataset, primary_cell_sentence_col=primary_cell_sentence, prefix=not dataset_text_only
                )

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
                dataset = load_dataset(dataset_id)
                logger.info(f"Bio dataset loaded - Keys: {list(dataset.keys())}")

                # Log dataset splits and sizes
                for split_name, split_data in dataset.items():
                    logger.info(f"  {split_name} split: {len(split_data)} samples")
                    if len(split_data) > 0:
                        logger.info(f"    Columns: {list(split_data.column_names)}")

                # Use configurable select_columns instead of hardcoded remove_columns
                keep_columns = getattr(bio_dataset_config, "keep_columns", None)
                if keep_columns:
                    # Apply to all splits that contain the specified columns
                    dataset_processed = {}
                    for split_name, split_data in dataset.items():
                        # Only keep columns that exist in this split
                        available_columns = [col for col in keep_columns if col in split_data.column_names]
                        if available_columns:
                            dataset_processed[split_name] = split_data.select_columns(available_columns)
                            logger.info(f"  Kept columns {available_columns} in {split_name} split")
                        else:
                            logger.warning(f"  No specified columns found in {split_name} split, keeping all")
                            dataset_processed[split_name] = split_data
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
                else:
                    logger.warning(f"Bio dataset '{dataset_name}' has no 'train' split, skipping")

                # Create loss function for this dataset type
                losses[dataset_name] = get_loss(dataset_type=bio_dataset_config.type)

                # Note: No evaluators created for bio datasets
                logger.info(f"Finished processing bio dataset: {dataset_name}\n")

        # Build the sentence Trasnformer model
        modules = [enc]
        model = SentenceTransformer(modules=modules)

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
        cs_length_for_naming = cfg.cs_length if text_only_datasets and hasattr(cfg, "cs_length") else None
        omics_datasets = getattr(cfg, "omics_datasets", [])
        unique_model_name = generate_unique_model_name(
            cfg, omics_datasets, cs_length_for_naming, text_only_datasets, numeric_datasets
        )
        logger.info(f"Using model name: {unique_model_name}")
        logger.info(f"Text-only datasets: {text_only_datasets}")
        logger.info(f"Numeric datasets: {numeric_datasets}")

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
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        monitor.stop()
        monitor.save(hydra_run_dir)
        monitor.plot_metrics(hydra_run_dir)
        logger.info(f"Training completed successfully. Model saved to {model_dir}")

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
