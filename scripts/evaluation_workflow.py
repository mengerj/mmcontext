"""
evaluation_workflow.py

A Hydra-powered Python script to evaluate a model on a given dataset using multiple embeddings,
visualizations, and metrics. All plots and metrics can be saved to disk.

Usage:
------
1. Adjust the default config below or provide your own YAML config to Hydra.
2. Run:
   python evaluation_workflow.py
   or
   python evaluation_workflow.py model_name="some-other-model" dataset_name="some-other-dataset"

Configurable Parameters (via Hydra):
-----------------------------------
- model_name : str
    Name/path of the model to load via SentenceTransformer.
- dataset_name : str
    Name/path of the dataset on Hugging Face to load via load_dataset.
- embedding_keys : List[str]
    List of embedding keys in adata.obsm to visualize (UMAP).
- batch_key : str
    Column in adata.obs that indicates batch.
- label_key : str
    Column in adata.obs that indicates cell type or label of interest.
- scib_batch_key : str
    Batch key for scibEvaluator (can differ from batch_key if needed).
- scib_label_key : str
    Label key for scibEvaluator.
- output_dir : str
    Directory where all results (plots, CSVs) will be saved.
- max_cells_scib : int
    Maximum number of cells for scibEvaluator.
- n_top_genes_scib : int
    Number of top genes for scibEvaluator.
- plot_n_samples_pairwise : int
    Number of samples (subset) for the pairwise embedding analyses.
- model_device : str
    Device for the SentenceTransformer (e.g. "cpu", "cuda", "mps").
- zero_shot_text_template : str
    Template string for zero-shot text queries. Must contain "{}".
- save_plots : bool
    Toggle whether to save plots.
- save_csv : bool
    Toggle whether to save CSV files.
- do_annotation : bool
    Whether to run annotation with OmicsQueryAnnotator.

Example:
--------
python evaluation_workflow.py model_name="jo-mengr/mmcontext-geo7k-cellxgene3.5k-multiplets" \
    dataset_name="jo-mengr/geo_7k_cellxgene_3_5k_multiplets"
"""

import logging
import os

import anndata
import hydra
import numpy as np
import pandas as pd

# Import or define your own modules/functions
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer

from mmcontext.engine import OmicsQueryAnnotator
from mmcontext.eval import evaluate_annotation_accuracy, scibEvaluator, zero_shot_classification_roc
from mmcontext.eval.utils import create_emb_pair_dataframe
from mmcontext.pl import plot_umap, visualize_embedding_clusters
from mmcontext.pl.plotting import plot_embedding_similarity, plot_grouped_bar_chart
from mmcontext.pp.utils import consolidate_low_frequency_categories
from mmcontext.utils import load_test_adata_from_hf_dataset

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="eval_conf")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the Hydra-driven evaluation workflow.

    Parameters
    ----------
    cfg : DictConfig
        A Hydra DictConfig object containing all user-defined parameters.
    """
    logger.info("===== Starting Evaluation Workflow with Hydra Configuration =====")
    logger.info(OmegaConf.to_yaml(cfg))

    # use hydra run dir as output dir
    output_dir = HydraConfig.get().run.dir

    # 1. Load Dataset
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    dataset = load_dataset(cfg.dataset_name)
    test_dataset = dataset["test"]

    # 2. Load Model
    logger.info(f"Loading model: {cfg.model_name}")
    model = SentenceTransformer(cfg.model_name, device=cfg.model_device)
    # Load the underlying text model seperatly
    text_encoder_name = model[0].text_encoder_name
    text_model = SentenceTransformer(text_encoder_name, device=cfg.model_device)

    # 3. Prepare AnnData
    logger.info("Loading test AnnData from dataset...")
    adata = load_test_adata_from_hf_dataset(test_dataset)

    # 4. Ensure batch_key is categorical
    logger.info(f"Ensuring batch_key '{cfg.batch_key}' is categorical...")
    adata.obs[cfg.batch_key] = adata.obs[cfg.batch_key].astype("category")

    # 5. Generate Embeddings (for mmcontext_emb and mmcontext_text_emb)
    logger.info("Generating mmcontext_emb from test_dataset['anndata_ref']...")
    adata.obsm["mmcontext_emb"] = model.encode(test_dataset["anndata_ref"], device=cfg.model_device)

    if cfg.caption_key:
        logger.info("Generating mmcontext_text_emb from adata.obs['natural_language_annotation']...")
        text_annotations = adata.obs[cfg.caption_key].tolist()
        adata.obsm["mmcontext_text_emb"] = model.encode(text_annotations, device=cfg.model_device)

        logger.info("Generating text embeddings with original text model...")
        text_embeddings_original = text_model.encode(text_annotations)
        adata.obsm["text_emb_original"] = text_embeddings_original
    else:
        # remove mmcontext_text_emb if and text_emb_original from the list of keys
        cfg.embedding_keys = [
            key for key in cfg.embedding_keys if key not in ["mmcontext_text_emb", "text_emb_original"]
        ]

    # 6. Plot UMAP for all embedding_keys with label_key and batch_key
    logger.info("Plotting UMAP for each embedding and color key...")
    for emb_key in cfg.embedding_keys:
        # Consolidate low-frequency categories if desired
        # But do it only for plotting if you prefer to remove rare classes
        for label in [cfg.label_key, cfg.batch_key]:
            if label in adata.obs.columns:
                adata.obs[label] = adata.obs[label].astype("category")
                adata_cut = consolidate_low_frequency_categories(adata, label, threshold=10)
                logger.info(f"UMAP: embedding={emb_key}, color={label}")
                plot_umap(
                    adata_cut,
                    color_key=label,
                    embedding_key=emb_key,
                    save_plot=cfg.save_plots,
                    save_dir=output_dir,
                    nametag=f"{emb_key}_{label}",
                )

    if cfg.caption_key:
        # 7. Pairwise Embedding Analysis (mmcontext_emb vs mmcontext_text_emb)
        logger.info("Creating pairwise embedding dataframe for mmcontext_emb vs mmcontext_text_emb...")
        emb_pair_df = create_emb_pair_dataframe(
            adata,
            emb1_key="mmcontext_emb",
            emb2_key="mmcontext_text_emb",
            subset_size=1000,  # or rename to 'n_samples' param if required by your version
            label_keys=[cfg.batch_key, cfg.label_key],
        )

        #   a) visualize_embedding_clusters (method=umap and method=trimap)
        logger.info("Visualizing embedding clusters (UMAP) for the pair embeddings...")
        for method in ["umap", "trimap"]:
            visualize_embedding_clusters(
                emb_pair_df,
                method="umap",
                metric="cosine",
                n_samples=cfg.plot_n_samples_pairwise,  # newly renamed parameter in your code
                random_state=42,
                save_plot=cfg.save_plots,
                save_path=os.path.join(output_dir, f"pairwise_cluster_{method}.png") if cfg.save_plots else None,
            )

        #   b) plot_embedding_similarity with n_samples=10 and n_samples=200 for demonstration
        logger.info("Plotting embedding similarity with small subset...")
        plot_embedding_similarity(
            emb_pair_df,
            emb1_type="omics",
            emb2_type="text",
            n_samples=10,  # previously subset parameter
            label_key=cfg.label_key,
            save_plot=cfg.save_plots,
            save_dir=output_dir,
            nametag="n10",
        )

        logger.info("Plotting embedding similarity with larger subset (200)...")
        plot_embedding_similarity(
            emb_pair_df,
            emb1_type="omics",
            emb2_type="text",
            n_samples=200,
            label_key=cfg.label_key,
            save_plot=cfg.save_plots,
            save_dir=output_dir,
            nametag="n200",
        )
    # 8. scibEvaluator for metrics
    logger.info("Running scibEvaluator for batch integration and bio-conservation metrics...")
    evaluator = scibEvaluator(
        adata=adata,
        batch_key=cfg.batch_key,
        label_key=cfg.label_key,
        embedding_key=cfg.embedding_keys,
        n_top_genes=cfg.n_top_genes_scib,
        max_cells=cfg.max_cells_scib,
    )
    res = evaluator.evaluate()
    res_df = pd.DataFrame(res)
    logger.info("scibEvaluator results:")
    logger.info(f"\n{res_df}")

    #   a) Save results as CSV
    if cfg.save_csv:
        csv_path = os.path.join(output_dir, "scibEvaluator_results.csv")
        res_df.to_csv(csv_path, index=False)
        logger.info(f"Saved scibEvaluator results to {csv_path}")

    #   b) Visualize with plot_grouped_bar_chart
    #      We assume the user has a 'type' column or needs to create one.
    #      scibEvaluator results might already have 'type' or 'embedding' column.
    #      For demonstration, let's rename the column "embedding" to "type".

    plot_grouped_bar_chart(
        data=res_df,
        save_plot=cfg.save_plots,
        save_path=os.path.join(output_dir, "scibEvaluator_grouped_bar.png") if cfg.save_plots else None,
        title="Comparison of scibEvaluator Metrics",
    )

    # 9. Annotation and Query
    if cfg.do_annotation:
        logger.info("Annotating omics data with OmicsQueryAnnotator...")
        annotator = OmicsQueryAnnotator(model)
        # e.g. labels from adata.obs[LABEL_KEY], as user requested
        labels = adata.obs[cfg.label_key].values.tolist()

        # Now we can pass the text_template
        annotator.annotate_omics_data(adata, labels, text_template=cfg.zero_shot_text_template)

        logger.info("Plotting UMAP with color='best_label' for embedding='mmcontext_emb'...")
        plot_umap(adata, color_key="best_label", embedding_key="mmcontext_emb")
        if cfg.save_plots:
            import matplotlib.pyplot as plt

            outpath = os.path.join(output_dir, "umap_annotation_best_label.png")
            plt.savefig(outpath, dpi=150)
            logger.info(f"Saved: {outpath}")
            plt.close()

        logger.info("Evaluating annotation accuracy w.r.t. ground-truth label key (cfg.label_key)...")
        accuracy = evaluate_annotation_accuracy(
            adata=adata,
            true_key=cfg.label_key,
            inferred_key="best_label",
        )
        logger.info(f"Annotation accuracy: {accuracy}")

    logger.info("===== Workflow complete! =====")


if __name__ == "__main__":
    main()
