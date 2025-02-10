import logging
import os
from datetime import datetime
from pathlib import Path

import anndata
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import BinaryClassificationEvaluator

from mmcontext.eval import SystemMonitor
from mmcontext.infer import MMContextInference
from mmcontext.models import MMContextEncoder
from mmcontext.pl import plot_umap
from mmcontext.pp import AnnDataSetConstructor, SimpleCaptionConstructor
from mmcontext.pp.utils import consolidate_low_frequency_categories

omics_model_cfg = {
    "embedding_dim": 50,
    "hidden_dim": 128,
    "num_layers": 1,
    "num_heads": 0,
    "use_self_attention": False,
    "activation": "relu",
    "dropout": 0.1,
}
dataset_name = "bowel_disease_pca"
text_encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
untrained_model_file = "data/models/mmcontext_model"
precomputed_key = "scvi"
save_dir = "out"
test_path = "data/test/bowel_disease.h5ad"
Path("logs").mkdir(exist_ok=True)
# set up a basic logger that writes to a file and the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/mmcontext_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),  # This will continue printing to console as well
    ],
)


def main():
    """Train MMContext model"""
    # Create logs directory if it doesn't exist
    logging.info("Starting training")
    dataset = load_dataset(f"jo-mengr/{dataset_name}")
    monitor = SystemMonitor(logger=logging)
    monitor.start()
    model = MMContextEncoder(
        text_encoder_name=text_encoder_name,
        processor_obsm_key="X_pp",
        omics_encoder_cfg=omics_model_cfg,
    )
    modules = [model]
    bimodal_model = SentenceTransformer(modules=modules)
    bimodal_model.save(untrained_model_file)
    model = SentenceTransformer(untrained_model_file)
    # embedding_dim = model[0].model.omics_encoder._get_sentence_embedding_dimension()

    # 3. Load the dataset
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    loss = losses.ContrastiveLoss(model=model)

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="../../data/models/mmcontext_trained",
        # Optional training parameters:
        num_train_epochs=5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="mmcontext",  # Will be used in W&B if `wandb` is installed
    )

    # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = BinaryClassificationEvaluator(
        sentences1=val_dataset["anndata_ref"],
        sentences2=val_dataset["caption"],
        labels=val_dataset["label"],
    )
    dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()
    model[0].processor.omics_processor.clear_cache()
    save_dir_date = Path(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir_date, exist_ok=True)
    model_dir = Path(save_dir_date, "model")
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)

    monitor.stop()
    monitor.save(save_dir_date)
    monitor.plot_metrics(save_dir_date)
    caption_constructor = SimpleCaptionConstructor(obs_keys=precomputed_key)
    constructor = AnnDataSetConstructor(
        caption_constructor=caption_constructor,
    )
    inferer = MMContextInference(
        file_path=test_path,
        constructor=constructor,
        model=model,
    )
    adata_val_new = inferer.encode(batch_size=64)
    adata_val_new = consolidate_low_frequency_categories(
        adata_val_new, ["cell_type", "dataset_id"], threshold=50, remove=True
    )
    adata_val_new.write_h5ad(f"{save_dir_date}/adata_val_encoded.h5ad")
    embedding_keys = ["omics_emb", "caption_emb", precomputed_key]
    for embedding_key in embedding_keys:
        plot_umap(
            adata_val_new,
            embedding_key=embedding_key,
            color_key=["cell_type", "dataset_id"],
            save_dir=save_dir_date,
            save_plot=True,
        )


if __name__ == "__main__":
    main()
