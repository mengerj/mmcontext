"""
evaluation_workflow.py

A Hydra-powered Python script to evaluate a model on multiple datasets using multiple embeddings,
visualizations, and metrics. All plots and metrics can be saved to disk.

Usage:
------
1. Adjust the default config below or provide your own YAML config to Hydra.
2. Run:
   python evaluation_workflow.py
   or
   python evaluation_workflow.py model_name="some-other-model" datasets.dataset1.name="some-other-dataset"

Configurable Parameters (via Hydra):
-----------------------------------
- model_name : str
    Name/path of the model to load via SentenceTransformer.
- datasets : dict
    Dictionary of dataset configurations, each with:
    - name : str
        Name/path of the dataset on Hugging Face to load via load_dataset.
    - batch_key : str
        Column in adata.obs that indicates batch.
    - label_key : str
        Column in adata.obs that indicates cell type or label of interest.
    - caption_key : str, optional
        Column in adata.obs with text annotations.
- embedding_keys : List[str]
    List of embedding keys in adata.obsm to visualize (UMAP).
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
- save_format : str
    Format for saving figures (e.g., "png", "pdf", "svg").
- results_db_path : str
    Path to the results database file for collecting metrics across runs.

Example:
--------
python evaluation_workflow.py model_name="jo-mengr/mmcontext-geo7k-cellxgene3.5k-multiplets"
"""

import logging
import os
from datetime import datetime

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
from mmcontext.eval.embedding_alignment import evaluate_modality_alignment
from mmcontext.eval.utils import create_emb_pair_dataframe
from mmcontext.pl import plot_umap, visualize_embedding_clusters
from mmcontext.pl.plotting import plot_embedding_similarity, plot_grouped_bar_chart
from mmcontext.pp.utils import consolidate_low_frequency_categories
from mmcontext.utils import load_test_adata_from_hf_dataset

logger = logging.getLogger(__name__)


def save_results_to_db(results_dict, db_path):
    """
    Save results to a centralized database file in long format.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing results to save
    db_path : str
        Path to the database file
    """
    # Extract metadata that applies to all rows
    metadata = {
        "model_name": results_dict["model_name"],
        "output_dir": results_dict["output_dir"],
        "timestamp": results_dict["timestamp"],
    }

    # Create rows for the long format
    rows = []
    for key, value in results_dict.items():
        # Skip metadata keys as they'll be added to each row
        if key in metadata:
            continue

        # Parse the key to extract dataset and metric names
        # Example key format: "bowel_disease_scib_nmi" or "bowel_disease_annotation_accuracy"
        parts = key.split("_")

        # Find the dataset name (everything before known metric types)
        metric_indicators = ["scib", "annotation", "modality", "zero", "full"]
        dataset_parts = []
        metric_parts = []
        found_metric = False

        for part in parts:
            if not found_metric and part not in metric_indicators:
                dataset_parts.append(part)
            else:
                found_metric = True
                metric_parts.append(part)

        dataset_name = "_".join(dataset_parts)
        metric_name = "_".join(metric_parts)

        # Create a row with all relevant information
        row = {
            **metadata,
            "dataset_name": dataset_name,
            "metric_name": metric_name,
            "metric_value": value,
        }
        rows.append(row)

    # Create a DataFrame from the rows
    results_df = pd.DataFrame(rows)

    # Check if the database file exists
    if os.path.exists(db_path):
        # Load existing database and append new results
        existing_df = pd.read_csv(db_path)
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        # Create new database
        updated_df = results_df
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Save the updated database without writing the index and only including header for new files
    updated_df.to_csv(db_path, index=False, mode="w")
    logger.info(f"Results saved to database: {db_path}")

    # Also save in a format suitable for funkyheatmap
    save_results_for_funkyheatmap(updated_df, db_path)


def save_results_for_funkyheatmap(results_df, db_path):
    """
    Save results in a format suitable for the funkyheatmap package.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe in long format
    db_path : str
        Path to the database file
    """
    # Create a unique model identifier
    results_df["model_id"] = results_df["model_name"].apply(lambda x: x.split("/")[-1])

    # Define metric groups
    metric_groups = {
        "bio_metrics": [
            "scib_ARI",
            "scib_NMI",
            "scib_ASW",
            "scib_Isolated_Labels_ASW",
            "scib_Isolated_Labels_F1",
            "scib_cLISI",
        ],
        "batch_metrics": ["scib_Graph_Connectivity", "scib_Silhouette_Batch", "scib_iLISI", "scib_PCR"],
        "alignment_metrics": ["modality_gap_irrelevant_score", "full_comparison_score"],
        "annotation_metrics": ["annotation_accuracy", "zero_shot_macro_auc"],
    }

    # Pivot the data to wide format with models as rows and metrics as columns
    # First, create a unique metric identifier by combining dataset_name and metric_name
    results_df["metric_id"] = results_df["dataset_name"] + "_" + results_df["metric_name"]

    # Pivot the data
    pivot_df = results_df.pivot(
        index=["model_id", "model_name", "timestamp"], columns="metric_id", values="metric_value"
    ).reset_index()

    # Rename the index column to 'id' for funkyheatmap
    pivot_df = pivot_df.rename(columns={"model_id": "id"})

    # Save the wide format data
    funky_db_path = os.path.splitext(db_path)[0] + "_funkyheatmap.csv"
    pivot_df.to_csv(funky_db_path, index=False)
    logger.info(f"Results saved in funkyheatmap format to: {funky_db_path}")

    # Create and save column_info DataFrame
    column_names = pivot_df.columns.tolist()

    # Determine the group for each column
    column_groups_list = []
    for col in column_names:
        if col in ["id", "model_name", "timestamp"]:
            group = "info"
        else:
            # Extract dataset and metric from the column name
            parts = col.split("_")
            dataset = parts[0]
            metric = "_".join(parts[1:])

            # Determine the metric group
            metric_group = None
            for group_name, metrics in metric_groups.items():
                if any(m.split("_", 1)[1] if m.startswith("scib_") else m == metric for m in metrics):
                    metric_group = group_name
                    break

            if metric_group:
                group = f"{dataset}_{metric_group}"
            else:
                group = f"{dataset}_other"

        column_groups_list.append(group)

    # Create column_info DataFrame
    column_info = pd.DataFrame(
        {
            "id": column_names,
            "name": [col.replace("_", " ").title() if col != "id" else "Model ID" for col in column_names],
            "geom": ["text" if col in ["id", "model_name", "timestamp"] else "bar" for col in column_names],
            "group": column_groups_list,
            "palette": [
                "black"
                if col in ["id", "model_name", "timestamp"]
                else "Blues"
                if any(m in col for m in metric_groups["batch_metrics"])
                else "Greens"
                if any(m in col for m in metric_groups["bio_metrics"])
                else "Reds"
                if any(m in col for m in metric_groups["alignment_metrics"])
                else "YlOrBr"
                if any(m in col for m in metric_groups["annotation_metrics"])
                else "Purples"
                for col in column_names
            ],
            "width": [3 if col == "id" else 2 if col in ["model_name", "timestamp"] else 1 for col in column_names],
            "legend": [False if col in ["id", "model_name", "timestamp"] else True for col in column_names],
        }
    )

    # Save column_info
    column_info_path = os.path.splitext(db_path)[0] + "_column_info.csv"
    column_info.to_csv(column_info_path, index=False)
    logger.info(f"Column info saved to: {column_info_path}")

    # Create and save column_groups DataFrame
    unique_groups = sorted(set(column_groups_list))

    # Extract dataset and metric group from each group
    group_info = []
    for group in unique_groups:
        if group == "info":
            level1 = "Info"
            level2 = "Model Information"
        else:
            parts = group.split("_")
            dataset = parts[0]
            metric_group = "_".join(parts[1:])

            level1 = dataset.title()

            if metric_group == "bio_metrics":
                level2 = "Biological Metrics"
            elif metric_group == "batch_metrics":
                level2 = "Batch Integration Metrics"
            elif metric_group == "alignment_metrics":
                level2 = "Alignment Metrics"
            elif metric_group == "annotation_metrics":
                level2 = "Annotation Metrics"
            else:
                level2 = "Other Metrics"

        group_info.append({"group": group, "level1": level1, "level2": level2})

    column_groups_df = pd.DataFrame(group_info)

    # Save column_groups
    column_groups_path = os.path.splitext(db_path)[0] + "_column_groups.csv"
    column_groups_df.to_csv(column_groups_path, index=False)
    logger.info(f"Column groups saved to: {column_groups_path}")


def evaluate_dataset(model, text_model, dataset_config, cfg, output_dir, results):
    """
    Evaluate a single dataset with the given model.

    Parameters
    ----------
    model : SentenceTransformer
        The MMContext model to evaluate
    text_model : SentenceTransformer
        The text-only model
    dataset_config : DictConfig
        Configuration for this dataset
    cfg : DictConfig
        Global configuration
    output_dir : str
        Base output directory
    results : dict
        Results dictionary to update

    Returns
    -------
    dict
        Updated results dictionary
    """
    dataset_name = dataset_config.name
    batch_key = dataset_config.batch_key
    label_key = dataset_config.label_key
    caption_key = dataset_config.caption_key if hasattr(dataset_config, "caption_key") else None

    # Get dataset-specific text template or use global default
    text_template = (
        dataset_config.zero_shot_text_template
        if hasattr(dataset_config, "zero_shot_text_template")
        else cfg.zero_shot_text_template
    )

    # Create dataset-specific output directory
    dataset_dir = os.path.join(output_dir, dataset_name.split("/")[-1])
    os.makedirs(dataset_dir, exist_ok=True)

    logger.info(f"Evaluating dataset: {dataset_name}")
    logger.info(f"  - batch_key: {batch_key}")
    logger.info(f"  - label_key: {label_key}")
    logger.info(f"  - caption_key: {caption_key}")
    logger.info(f"  - text_template: {text_template}")

    # 1. Load Dataset
    dataset = load_dataset(dataset_name)
    test_dataset = dataset["test"]
    # If the dataset has "pairs" in the name, filter the label column for == 1.0
    if "pairs" in dataset_name:
        test_dataset = test_dataset.filter(lambda x: x["label"] == 1.0)

    # 2. Prepare AnnData
    adata = load_test_adata_from_hf_dataset(test_dataset)

    # 3. Ensure batch_key is categorical
    if batch_key in adata.obs.columns:
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    else:
        logger.warning(f"batch_key '{batch_key}' not found in adata.obs")

    # 4. Generate Embeddings (for mmcontext_emb and mmcontext_text_emb)
    logger.info("Generating mmcontext_emb from test_dataset['anndata_ref']...")
    adata.obsm["mmcontext_emb"] = model.encode(test_dataset["anndata_ref"], device=cfg.model_device)

    # Determine which embedding keys to use for this dataset
    dataset_embedding_keys = list(cfg.embedding_keys)

    # 5. Generate text embeddings - either from caption_key or from label_key + template
    has_text_data = False

    if caption_key and caption_key in adata.obs.columns:
        # Use existing captions from the dataset
        logger.info(f"Generating text embeddings from adata.obs['{caption_key}']...")
        text_annotations = adata.obs[caption_key].tolist()
        has_text_data = True
    elif label_key in adata.obs.columns and text_template:
        # Generate captions from labels using the template
        logger.info(f"Generating text annotations from labels using template: '{text_template}'")
        labels = adata.obs[label_key].tolist()
        text_annotations = [text_template.format(label) for label in labels]
        has_text_data = True
    else:
        logger.warning("No caption_key or label_key available for text embeddings")

    if has_text_data:
        # Generate embeddings with both models
        logger.info("Generating mmcontext_text_emb...")
        adata.obsm["mmcontext_text_emb"] = model.encode(text_annotations, device=cfg.model_device)

        logger.info("Generating text embeddings with original text model...")
        text_embeddings_original = text_model.encode(text_annotations)
        adata.obsm["text_emb_original"] = text_embeddings_original
    else:
        # Remove text-related embeddings if text data is not available
        dataset_embedding_keys = [
            key for key in dataset_embedding_keys if key not in ["mmcontext_text_emb", "text_emb_original"]
        ]
        logger.warning("Skipping text embeddings due to lack of text data.")

    # 6. Plot UMAP for all embedding_keys with label_key and batch_key
    logger.info("Plotting UMAP for each embedding and color key...")
    for emb_key in dataset_embedding_keys:
        if emb_key not in adata.obsm:
            logger.warning(f"Embedding key '{emb_key}' not found in adata.obsm. Skipping.")
            continue

        # Consolidate low-frequency categories if desired
        for color_key in [label_key, batch_key]:
            if color_key in adata.obs.columns:
                adata.obs[color_key] = adata.obs[color_key].astype("category")
                adata_cut = consolidate_low_frequency_categories(adata, [color_key], threshold=10)
                logger.info(f"UMAP: embedding={emb_key}, color={color_key}")
                plot_umap(
                    adata_cut,
                    color_key=[color_key],
                    embedding_key=emb_key,
                    save_plot=cfg.save_plots,
                    save_dir=dataset_dir,
                    nametag=f"{emb_key}_{color_key}",
                    save_format=cfg.save_format,
                )

    # 7. Pairwise Embedding Analysis (if text data is available)
    if has_text_data and "mmcontext_text_emb" in adata.obsm:
        logger.info("Creating pairwise embedding dataframe for mmcontext_emb vs mmcontext_text_emb...")
        embedding_dict = {"omics": "mmcontext_emb", "text": "mmcontext_text_emb"}
        emb_pair_df = create_emb_pair_dataframe(
            adata,
            embedding_dict=embedding_dict,
            subset_size=1000,
            label_keys=[label_key] if label_key in adata.obs.columns else None,
        )

        modality_gap_irrelevant_score, full_comparison_score = evaluate_modality_alignment(emb_pair_df)

        # Add alignment scores to results
        dataset_key = dataset_name.split("/")[-1]
        results[f"{dataset_key}_modality_gap_irrelevant_score"] = modality_gap_irrelevant_score
        results[f"{dataset_key}_full_comparison_score"] = full_comparison_score

        #   a) visualize_embedding_clusters (method=umap and method=trimap)
        logger.info("Visualizing embedding clusters for the pair embeddings...")
        for method in ["umap"]:  # , "trimap"]:
            visualize_embedding_clusters(
                emb_pair_df,
                method=method,
                metric="cosine",
                n_samples=cfg.plot_n_samples_pairwise,
                random_state=42,
                save_plot=cfg.save_plots,
                save_format=cfg.save_format,
                save_dir=dataset_dir,
                nametag=f"pairwise_cluster_{method}",
            )

        #   b) plot_embedding_similarity with n_samples=10 and n_samples=200 for demonstration
        logger.info("Plotting embedding similarity with small subset...")
        plot_embedding_similarity(
            emb_pair_df,
            emb1_type="omics",
            emb2_type="text",
            n_samples=10,
            label_key=label_key if label_key in adata.obs.columns else None,
            save_plot=cfg.save_plots,
            save_dir=dataset_dir,
            nametag="n10",
            save_format=cfg.save_format,
        )

        logger.info("Plotting embedding similarity with larger subset...")
        plot_embedding_similarity(
            emb_pair_df,
            emb1_type="omics",
            emb2_type="text",
            n_samples=min(200, adata.n_obs),
            label_key=label_key if label_key in adata.obs.columns else None,
            save_plot=cfg.save_plots,
            save_dir=dataset_dir,
            nametag="n200",
            save_format=cfg.save_format,
        )

    # 8. scibEvaluator for metrics
    if batch_key in adata.obs.columns and label_key in adata.obs.columns:
        logger.info("Running scibEvaluator for batch integration and bio-conservation metrics...")
        # Filter embedding keys to only those available in adata.obsm
        available_embedding_keys = [key for key in dataset_embedding_keys if key in adata.obsm]

        evaluator = scibEvaluator(
            adata=adata,
            batch_key=batch_key,
            label_key=label_key,
            embedding_key=available_embedding_keys,
            n_top_genes=cfg.n_top_genes_scib,
            max_cells=cfg.max_cells_scib,
            in_parallel=False,
        )
        res = evaluator.evaluate()
        res_df = pd.DataFrame(res)
        logger.info("scibEvaluator results:")
        logger.info(f"\n{res_df}")

        # Extract mmcontext_emb results for the central database
        dataset_key = dataset_name.split("/")[-1]
        mmcontext_results = res_df[res_df["type"] == "mmcontext_emb"]
        for _, row in mmcontext_results.iterrows():
            results[f"{dataset_key}_scib_{row['metric']}"] = row["value"]

        #   a) Save all results as CSV in the output directory
        if cfg.save_csv:
            csv_path = os.path.join(dataset_dir, "scibEvaluator_results.csv")
            res_df.to_csv(csv_path, index=False)
            logger.info(f"Saved scibEvaluator results to {csv_path}")

        #   b) Visualize with plot_grouped_bar_chart
        plot_grouped_bar_chart(
            data=res_df,
            save_plot=cfg.save_plots,
            save_path=os.path.join(dataset_dir, f"scibEvaluator_grouped_bar.{cfg.save_format}")
            if cfg.save_plots
            else None,
            title=f"Comparison of scibEvaluator Metrics - {dataset_name}",
        )
    else:
        logger.warning(
            f"Skipping scibEvaluator: batch_key '{batch_key}' or label_key '{label_key}' not found in adata.obs"
        )

    # 9. Annotation and Query
    if cfg.do_annotation and label_key in adata.obs.columns:
        logger.info("Annotating omics data with OmicsQueryAnnotator...")
        annotator = OmicsQueryAnnotator(model, is_cosine=False)
        # e.g. labels from adata.obs[label_key]
        labels = adata.obs[label_key].values.tolist()

        # Now we can pass the text_template. It will modify the labels to align better with the text models expectations
        annotator.annotate_omics_data(adata, labels, text_template=text_template)

        logger.info("Plotting UMAP with color='best_label' for embedding='mmcontext_emb'...")
        plot_umap(
            adata,
            color_key=["best_label"],
            embedding_key="mmcontext_emb",
            save_plot=cfg.save_plots,
            save_dir=dataset_dir,
            save_format=cfg.save_format,
            nametag=f"Annotations_with_{label_key}",
        )

        logger.info("Evaluating annotation accuracy w.r.t. ground-truth label key...")
        accuracy = evaluate_annotation_accuracy(
            adata=adata,
            true_key=label_key,
            inferred_key="best_label",
        )
        logger.info(f"Annotation accuracy: {accuracy}")

        # Add annotation accuracy to results
        dataset_key = dataset_name.split("/")[-1]
        results[f"{dataset_key}_annotation_accuracy"] = accuracy

        logger.info("Computing zero-shot classification ROC...")
        macro_auc, auc_details = zero_shot_classification_roc(
            adata,
            model,
            label_key=label_key,
            emb_key="mmcontext_emb",
            text_template=text_template,
            device=cfg.model_device,
        )
        logger.info(f"Macro AUC: {macro_auc}")
        logger.info(f"Detail per label: {auc_details}")

        # Add zero-shot classification results to results
        results[f"{dataset_key}_zero_shot_macro_auc"] = macro_auc
    elif cfg.do_annotation:
        logger.warning(f"Skipping annotation: label_key '{label_key}' not found in adata.obs")

    return results


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

    # Initialize results dictionary
    results = {
        "model_name": cfg.model_name,
        "output_dir": output_dir,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Add model_short_name if provided in config
    if hasattr(cfg, "model_short_name"):
        results["model_short_name"] = cfg.model_short_name
    else:
        # Extract short name from model_name (last part after '/')
        results["model_short_name"] = cfg.model_name.split("/")[-1]
    # Load Model (once for all datasets)
    logger.info(f"Loading model: {cfg.model_name}")
    model = SentenceTransformer(cfg.model_name, device=cfg.model_device)

    # Load the underlying text model separately
    text_encoder_name = model[0].text_encoder_name
    text_model = SentenceTransformer(text_encoder_name, device=cfg.model_device)

    # Evaluate each dataset
    for dataset_key, dataset_config in cfg.datasets.items():
        logger.info(f"Processing dataset: {dataset_key}")
        results = evaluate_dataset(model, text_model, dataset_config, cfg, output_dir, results)

    # Save results to the central database
    if cfg.save_csv and hasattr(cfg, "results_db_path") and cfg.results_db_path:
        save_results_to_db(results, cfg.results_db_path)

    logger.info("===== Workflow complete! =====")


if __name__ == "__main__":
    main()
