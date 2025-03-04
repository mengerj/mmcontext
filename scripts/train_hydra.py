import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import anndata
import hydra
import numpy as np
import yaml
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import (
    SentenceTransformer,
    # SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from transformers.integrations import WandbCallback

from mmcontext.engine.callback import UnfreezeTextEncoderCallback
from mmcontext.eval import SystemMonitor, zero_shot_classification_roc
from mmcontext.models import MMContextEncoder
from mmcontext.pl import plot_umap

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
    # 1. Prepare directories and logging
    # -------------------------------------------------------------------------
    # setup_logging(logging_dir="../logs")
    # -------------------------------------------------------------------------
    # 2. Build the dataset name and load data
    # -------------------------------------------------------------------------
    dataset_name = f"{cfg.dataset.basename}_{cfg.dataset.type}_{cfg.dataset.caption}"
    logger.info(f"Starting training with dataset: {dataset_name}")
    dataset = load_dataset(f"jo-mengr/{dataset_name}")

    test_dataset = load_dataset(cfg.test_dataset.name)["test"]

    # -------------------------------------------------------------------------
    # 3. Compute the correct embedding dimension based on method
    # -------------------------------------------------------------------------
    input_dim_map = cfg.input_dim_map
    chosen_method = cfg.embedding_method
    if chosen_method not in input_dim_map:
        raise ValueError(f"Unknown embedding_method '{chosen_method}'. Allowed: {list(input_dim_map.keys())}")
    # Overwrite the model's embedding_dim with the mapped value
    cfg.adapter.omics_input_dim = input_dim_map[chosen_method]
    precomputed_key = f"X_{chosen_method}"

    # -------------------------------------------------------------------------
    # 4. Set up System Monitor
    # -------------------------------------------------------------------------
    monitor = SystemMonitor(logger=logging)
    monitor.start()

    # -------------------------------------------------------------------------
    # 5. Create the model (MMContextEncoder => SentenceTransformer modules)
    # -------------------------------------------------------------------------
    if cfg.model:
        model = SentenceTransformer(cfg.model)
    else:
        bimodal_model = MMContextEncoder(
            text_encoder_name=cfg.text_encoder.name,
            omics_input_dim=cfg.adapter.omics_input_dim,
            processor_obsm_key=precomputed_key,
            freeze_text_encoder=cfg.text_encoder.freeze_text_encoder,
            unfreeze_last_n_layers=cfg.text_encoder.unfreeze_last_n_layers,
            adapter_hidden_dim=cfg.adapter.hidden_dim,
            adapter_output_dim=cfg.adapter.output_dim,
        )
        modules = [bimodal_model]
        model = SentenceTransformer(modules=modules)

    # -------------------------------------------------------------------------
    # 6. Load train/val split
    # -------------------------------------------------------------------------
    train_dataset = dataset["train"]
    # randomly shuffle caption column of training data for a test purpose
    # import random
    # cap = train_dataset["caption"]
    # random.shuffle(cap)
    # replace dataset caption column with shuffled caption column
    # train_dataset = train_dataset.map(lambda example, idx: {"caption": cap[idx]}, with_indices=True)
    val_dataset = dataset["val"]

    # -------------------------------------------------------------------------
    # 7. Instantiate the loss function
    # -------------------------------------------------------------------------
    # For example, you could define a 'loss_name' field in your config
    # or just hardcode "contrastive" for now.
    # e.g. "cfg.loss.name"
    loss_obj = get_loss(dataset_type=cfg.dataset.type)
    loss = loss_obj(model)

    # -------------------------------------------------------------------------
    # 8. Set up training arguments
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
    )

    # -------------------------------------------------------------------------
    # 9. (Optional) Create an evaluator & evaluate the base model
    # -------------------------------------------------------------------------
    dev_evaluator = get_evaluator(
        dataset_type=cfg.dataset.type, dataset=val_dataset, batch_size=cfg.trainer.per_device_eval_batch_size
    )
    dev_evaluator(model)
    # -------------------------------------------------------------------------
    # 10. Create a trainer & train
    # -------------------------------------------------------------------------
    unfreeze_callback = UnfreezeTextEncoderCallback(unfreeze_epoch=cfg.trainer.unfreeze_epoch)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=dev_evaluator,
        extra_feature_keys=["omics_representation"],
        callbacks=[unfreeze_callback, WandbCallback()],
    )
    trainer.train()
    test_evaluator = get_evaluator(dataset_type=cfg.test_dataset.type, dataset=test_dataset)
    test_evaluator(model)

    # -------------------------------------------------------------------------
    # 11. Save the model and run system monitor
    # -------------------------------------------------------------------------
    model[0].processor.omics_processor.clear_cache()  # Important or you will save anndata objects stored in the cache
    model_dir = Path(hydra_run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)

    monitor.stop()
    monitor.save(hydra_run_dir)
    monitor.plot_metrics(hydra_run_dir)
    logger.info(f"Training completed successfully. Model saved to {model_dir}")

    # -------------------------------------------------------------------------
    # 12. Test on additional datasets & produce plots
    # -------------------------------------------------------------------------
    """
    for t_dataset in cfg.dataset.test_datasets:
        # extract the name from the dataset (after the /)
        dataset_name = t_dataset.split("/")[-1]
        test_dataset = load_dataset(t_dataset)["train"]
        # Get the anndata object from the test dataset
        adata = load_test_adata_from_hf_dataset(test_dataset)
        emb = model.encode(test_dataset["anndata_ref"])
        # Add the embeddings to the anndata object
        adata.obsm["mmcontext_emb"] = emb.astype(np.float32)

        adata.write_h5ad(f"{hydra_run_dir}/{dataset_name}_encoded.h5ad")
        mean_roc = zero_shot_classification_roc(
            adata, model, label_key="cluster_label", emb_key="mmcontext_emb", device=get_device(), text_template="{}"
        )
        logger.info(f"Mean ROC-AUC for {dataset_name}: {mean_roc}")
        # write mean of ROC-AUC to a file
        with open(Path(hydra_run_dir, "mean_roc.txt"), "a") as f:
            f.write(f"{dataset_name}: {mean_roc}\n")
        embedding_keys = ["mmcontext_emb", precomputed_key]
        for embedding_key in embedding_keys:
            plot_umap(
                adata,
                embedding_key=embedding_key,
                color_key=["cluster_label"],
                save_dir=hydra_run_dir,
                save_plot=True,
            )
        logger.info(f"Plots saved successfully to the output directory: {hydra_run_dir}")
    """


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
