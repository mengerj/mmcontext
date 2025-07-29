import logging
import os
import sys
from pathlib import Path

import hydra
from datasets import load_dataset
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
    cfg: DictConfig, dataset_configs: list, cs_len: int = None, username: str = "jo-mengr"
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
    username : str, optional
        Username/organization on Hugging Face Hub (default: "jo-mengr")

    Returns
    -------
    str
        Unique model name that doesn't conflict with existing models
    """
    # Generate base model name
    base_name = generate_model_name(cfg, dataset_configs, cs_len)

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


def generate_model_name(cfg: DictConfig, dataset_configs: list, cs_len: int = None) -> str:
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
        captions.append(dataset_config.caption)

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
    else:
        # Take the last part after '/'
        encoder_str = text_encoder_name.split("/")[-1]

    # Get output dimension
    output_dim = cfg.adapter.output_dim

    # Get embedding method or text_only
    if cfg.text_only:
        method_str = "text_only"
        if cs_len is not None:
            method_str = f"text_only_{cs_len}"
        if not cfg.gene_based_cell_sentence:
            raise ValueError(
                "text_only is true, but gene_based_cell_sentence is false. This will lead to training a text model on sample ids, which is not what you want."
            )
    else:
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
        cs_len = None

        for dataset_config in cfg.datasets:
            # Construct dataset name with optional cs_len suffix
            base_name = dataset_config.name

            dataset_name = f"{base_name}_{dataset_config.type}_{dataset_config.caption}"
            if hasattr(dataset_config, "cs_len") and dataset_config.cs_len is not None:
                dataset_name = f"{dataset_name}_cs{dataset_config.cs_len}"
            logger.info(f"Loading dataset: {dataset_name}")

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

            if not cfg.text_only:
                token_df, _ = enc.get_initial_embeddings(
                    dataset,
                    layer_key=precomputed_key,
                    download_dir=f"../../data/from_nxtcloud/{dataset_name}",
                    axis=layer_axis,
                )
                enc.register_initial_embeddings(token_df, data_origin=chosen_method)
            else:
                # Get the length of the cell sentences and validate consistency
                current_cs_len = len(dataset["train"][0][primary_cell_sentence].split(" "))
                if cs_len is None:
                    cs_len = current_cs_len
                    logger.info(f"Cell sentence length for text_only mode: {cs_len}")
                elif cs_len != current_cs_len:
                    raise ValueError(
                        f"Inconsistent cell sentence lengths across datasets: {cs_len} vs {current_cs_len} for dataset {dataset_name}"
                    )
            # Add the prefix expected by the model
            dataset_ready = enc.prepare_ds(
                dataset, primary_cell_sentence_col=primary_cell_sentence, prefix=not cfg.text_only
            )  # ,"negative_2"])

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
            logger.info(f"Finished processing dataset: {dataset_name}\n")

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
        unique_model_name = generate_unique_model_name(cfg, cfg.datasets, cs_len)
        logger.info(f"Using model name: {unique_model_name}")

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
            run_name=unique_model_name,
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

        model_dir = Path(hydra_run_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir)
        model.push_to_hub(f"jo-mengr/{unique_model_name}", private=True)
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
