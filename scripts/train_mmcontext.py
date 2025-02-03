from mmcontext.models import MMContextEncoder
from mmcontext.pp import InitialEmbedder, AnnDataSetConstructor
from mmcontext.pp.utils import consolidate_low_frequency_categories
from mmcontext.pp.caption_constructors import SimpleCaptionConstructor
from mmcontext.eval import SystemMonitor
from mmcontext.infer import MMContextInference
from mmcontext.pl import plot_umap
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    losses
)
from sentence_transformers.evaluation import BinaryClassificationEvaluator

import anndata
from datetime import datetime
from pathlib import Path
import os
import logging
omics_model_cfg = {
    "embedding_dim": 50,
    "hidden_dim": 128,
    "num_layers": 1,
    "num_heads": 0,
    "use_self_attention": False,
    "activation": "relu",
    "dropout": 0.1,
}
train_name = 'cellxgene_spleen_10x3v3'
save_dir = "out/training/"
val_name = 'cellxgene_spleen_10x3v3'
train_path = f"data/train_data/{train_name}.h5ad"
val_path = f"data/test_data/{val_name}.h5ad"
text_encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
untrained_model_file = "data/models/mmcontext_model"
precomputed_key = "scvi"

Path("logs").mkdir(exist_ok=True)
#set up a basic logger that writes to a file and the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/mmcontext_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()  # This will continue printing to console as well
    ]
)



def main():
    # Create logs directory if it doesn't exist
    logging.info("Starting training")
    print("Starting training")
    monitor = SystemMonitor(logger = logging)
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
    embedding_dim = model[0].model.omics_encoder._get_sentence_embedding_dimension()
    adata_train = anndata.read_h5ad(train_path)
    adata_val = anndata.read_h5ad(val_path)
    init_embedder = InitialEmbedder(embedding_dim = embedding_dim, precomputed_key = precomputed_key)
    init_embedder.embed(adata_train)
    # You have to save the modifed adata files so they are used in training and inferece
    adata_train.write_h5ad(train_path)
    init_embedder.embed(adata_val)
    adata_val.write_h5ad(val_path)
    # Create caption constructor with desired obs keys
    caption_constructor = SimpleCaptionConstructor(
        obs_keys=['cell_type']
    )
    constructor = AnnDataSetConstructor(caption_constructor=caption_constructor)
    constructor.add_anndata(file_path=train_path)
    # Get train dataset
    train_dataset = constructor.get_dataset()
    constructor.clear()
    constructor.add_anndata(file_path=val_path)
    # Get val dataset
    val_dataset = constructor.get_dataset()
    loss = losses.ContrastiveLoss(model = model)

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
        #batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
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
        sentences1 = val_dataset["anndata_ref"],
        sentences2 = val_dataset["caption"],
        labels = val_dataset["label"],
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
    save_dir_date = Path(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir_date, exist_ok=True)
    model_dir = Path(save_dir_date, 'model')
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)

    monitor.stop()
    monitor.save(save_dir_date)
    monitor.plot_metrics(save_dir_date)
    constructor.clear()
    inferer = MMContextInference(
        file_path=val_path,
        constructor=constructor,
        model=model,
    )
    adata_val_new = inferer.encode(batch_size=64)
    adata_val_new = consolidate_low_frequency_categories(adata_val_new, ["cell_type","dataset_id"], threshold=50, remove=True)
    adata_val_new.write_h5ad(f"{save_dir_date}/adata_val_encoded.h5ad")
    embedding_keys = ["omics_emb","caption_emb","X_pp"]
    for embedding_key in embedding_keys:
        plot_umap(adata_val_new, embedding_key = embedding_key, color_key = ["cell_type","dataset_id"], save_dir = save_dir_date, save_plot = True)
if __name__ == "__main__":
    main()