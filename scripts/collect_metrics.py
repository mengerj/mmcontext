#!/usr/bin/env python3
"""Script to collect and fuse metrics from evaluation results."""

import shutil
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from omegaconf import DictConfig, OmegaConf


def load_metrics_file(file_path: Path) -> pd.DataFrame:
    """Load metrics file, return empty DataFrame if file doesn't exist."""
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()


def process_regular_metrics(
    df: pd.DataFrame, dataset_name: str, model_id: str, selected_metrics: list[str]
) -> list[dict[str, Any]]:
    """Process regular metrics and return list of result dictionaries."""
    results = []

    if df.empty:
        return results

    # Filter for selected metrics
    df_filtered = df[df["metric"].isin(selected_metrics)]

    for _, row in df_filtered.iterrows():
        results.append(
            {
                "dataset": dataset_name,
                "model": model_id,
                "label": row["label"],
                "label_kind": row["label_kind"],
                "metric": row["metric"],
                "value": row["value"],
            }
        )

    return results


def process_scib_metrics(
    df: pd.DataFrame,
    dataset_name: str,
    model_id: str,
    selected_metrics: list[str],
    types_as_models: list[str],
    model_embedding_type: str,
) -> list[dict[str, Any]]:
    """Process scib metrics with special type handling."""
    results = []

    if df.empty:
        return results

    # Filter for selected metrics
    df_filtered = df[df["metric"].isin(selected_metrics)]

    for _, row in df_filtered.iterrows():
        current_type = row["type"]

        if current_type in types_as_models:
            # Treat this type as a separate model
            model_name = current_type
            results.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "label": row["bio_label"],
                    "label_kind": "bio",
                    "metric": row["metric"],
                    "value": row["value"],
                }
            )

        elif current_type == model_embedding_type:
            # This is the actual model embedding - use original model name
            results.append(
                {
                    "dataset": dataset_name,
                    "model": model_id,
                    "label": row["bio_label"],
                    "label_kind": "bio",
                    "metric": row["metric"],
                    "value": row["value"],
                }
            )

    return results


def collect_all_metrics(cfg: DictConfig) -> pd.DataFrame:
    """Collect all metrics from evaluation results."""
    all_results = []

    eval_root = Path(cfg.collect_metrics.eval_root)

    for ds_cfg in cfg.datasets:
        dataset_name = ds_cfg.name

        for model_cfg in cfg.models:
            model_id = model_cfg.source
            text_only = model_cfg.get("text_only", False)
            if text_only:
                model_id = model_id + "_text_only"

            # Construct path to evaluation results
            model_dir = eval_root / dataset_name / Path(model_id).name.replace("/", "_")

            if not model_dir.exists():
                print(f"Warning: Model directory not found: {model_dir}")
                continue

            print(f"Processing: {dataset_name} / {model_id}")

            # Load regular metrics
            regular_metrics_file = model_dir / "eval" / "metrics.csv"
            regular_df = load_metrics_file(regular_metrics_file)

            # Process regular metrics
            regular_results = process_regular_metrics(
                regular_df, dataset_name, model_id, cfg.collect_metrics.regular_metrics
            )
            all_results.extend(regular_results)

            # Load scib metrics
            scib_metrics_file = model_dir / "eval" / "scib_metrics.csv"
            scib_df = load_metrics_file(scib_metrics_file)

            # Process scib metrics
            scib_results = process_scib_metrics(
                scib_df,
                dataset_name,
                model_id,
                cfg.collect_metrics.scib_metrics,
                cfg.collect_metrics.scib_types_as_models,
                cfg.collect_metrics.model_embedding_type,
            )
            all_results.extend(scib_results)

    # Convert to DataFrame
    if all_results:
        return pd.DataFrame(all_results)
    else:
        return pd.DataFrame(columns=["dataset", "model", "label", "label_kind", "metric", "value"])


def clean_model_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up model names for better visualization."""
    df = df.copy()

    # Clean up embedding model names first
    model_name_mapping = {
        "embedding_X_geneformer": "geneformer",
        "embedding_X_hvg": "hvg",
        "embedding_X_pca": "pca",
        "embedding_X_scvi_fm": "scvi_fm",
        "raw": "raw",
    }

    df["model"] = df["model"].replace(model_name_mapping)

    # Clean up long model names
    def shorten_model_name(name):
        # Skip if it's already a short embedding name
        if name in ["geneformer", "hvg", "pca", "scvi_fm", "raw"]:
            return name

        # Handle NeuML models (models that don't contain "mmcontext")
        if "mmcontext" not in name:
            if "NeuML" in name or "pubmedbert" in name:
                # Remove "_text_only" suffix and call it "pubmedbert"
                return "pubmedbert"

        # Find the position of cell_type or natural_language_annotation
        cell_type_pos = name.find("cell_type")
        cap_pos = name.find("natural_language_annotation")

        # Determine starting position
        if cell_type_pos != -1 and cap_pos != -1:
            start_pos = min(cell_type_pos, cap_pos)
        elif cell_type_pos != -1:
            start_pos = cell_type_pos
        elif cap_pos != -1:
            start_pos = cap_pos
        else:
            # If neither found, return original name
            return name

        # Extract the part starting from cell_type or natural_language_annotation
        shortened = name[start_pos:]

        # Apply replacements
        shortened = shortened.replace("cell_type", "ct")
        shortened = shortened.replace("natural_language_annotation", "cap")
        shortened = shortened.replace("pubmedbert", "")
        shortened = shortened.replace("sample_cs", "")
        shortened = shortened.replace("feat_cs", "")

        # Clean up extra dashes and underscores
        shortened = shortened.replace("--", "-")
        shortened = shortened.replace("__", "_")
        shortened = shortened.strip("-_")

        # Remove any remaining double dashes or underscores
        while "--" in shortened:
            shortened = shortened.replace("--", "-")
        while "__" in shortened:
            shortened = shortened.replace("__", "_")

        return shortened

    df["model"] = df["model"].apply(shorten_model_name)
    return df


def clean_metric_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up metric names by removing evaluator prefixes."""
    df = df.copy()

    # Remove prefixes like "scib/" and "LabelSimilarity/"
    df["metric"] = df["metric"].str.replace("scib/", "", regex=False)
    df["metric"] = df["metric"].str.replace("LabelSimilarity/", "", regex=False)

    return df


def create_metrics_visualization(df: pd.DataFrame, output_dir: Path, cfg: DictConfig) -> None:
    """Create a comprehensive visualization of all metrics."""
    # Clean up the data
    df_clean = clean_model_names(df)
    df_clean = clean_metric_names(df_clean)

    # Convert values to numeric to avoid categorical interpretation
    df_clean["value"] = pd.to_numeric(df_clean["value"], errors="coerce")

    # Remove any rows with NaN values (from conversion errors)
    df_clean = df_clean.dropna(subset=["value"])

    # For LabelSimilarity methods, create separate metrics for each label combination
    def create_combined_metric_name(row):
        # Only combine for metrics that have label and label_kind columns
        if pd.notna(row.get("label")) and pd.notna(row.get("label_kind")):
            return f"{row['metric']}_{row['label_kind']}_{row['label']}"
        else:
            return row["metric"]

    df_clean["combined_metric"] = df_clean.apply(create_combined_metric_name, axis=1)

    # Create legend for all datasets (shared across plots) - BEFORE using dataset_colors
    unique_datasets = sorted(df_clean["dataset"].unique())
    colors = sns.color_palette("husl", len(unique_datasets))
    dataset_colors = dict(zip(unique_datasets, colors, strict=False))

    plot_format = cfg.collect_metrics.get("plot_format", "png")
    create_separate_legend(unique_datasets, dataset_colors, output_dir, plot_format)

    # Separate scib and LabelSimilarity metrics
    scib_data = df_clean[
        df_clean["metric"].str.startswith("scib")
        | df_clean["metric"].str.contains("Bio_Conservation_Score|Batch_Integration_Score|Overall_Score")
    ].copy()
    labelsim_data = df_clean[
        df_clean["metric"].str.startswith("mean_auc") | df_clean["metric"].str.contains("LabelSimilarity")
    ].copy()

    # Create separate plots
    if not scib_data.empty:
        print("Creating scib metrics visualization...")
        create_single_metrics_plot(scib_data, output_dir, cfg, "scib_metrics", dataset_colors)

    if not labelsim_data.empty:
        print("Creating LabelSimilarity metrics visualization...")
        create_single_metrics_plot(labelsim_data, output_dir, cfg, "labelsim_metrics", dataset_colors)

    # Print summary for debugging
    print("Data summary:")
    print(f"  Datasets: {sorted(df_clean['dataset'].unique())}")
    print(f"  Models: {sorted(df_clean['model'].unique())}")
    print(f"  Original metrics: {sorted(df_clean['metric'].unique())}")
    print(f"  Combined metrics: {sorted(df_clean['combined_metric'].unique())}")
    print(f"  Value column dtype: {df_clean['value'].dtype}")


def create_single_metrics_plot(
    df_subset: pd.DataFrame, output_dir: Path, cfg: DictConfig, plot_name: str, dataset_colors: dict
) -> None:
    """Create a single metrics plot for a subset of data."""
    # Get unique metrics
    unique_metrics = sorted(df_subset["combined_metric"].unique())
    n_metrics = len(unique_metrics)

    if n_metrics == 0:
        print(f"No metrics to plot for {plot_name}!")
        return

    # Calculate subplot layout
    n_cols = 3  # Number of columns in subplot grid
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Handle different subplot configurations
    if n_metrics == 1:
        axes = np.array([[axes]])  # Single subplot
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Set style
    sns.set_style("whitegrid")

    # Track if we have any data to plot
    has_data = False

    # Plot each metric
    for i, metric in enumerate(unique_metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Filter data for this metric
        metric_data = df_subset[df_subset["combined_metric"] == metric]

        if metric_data.empty:
            ax.set_title(f"{metric} (No data)")
            ax.axis("off")
            continue

        has_data = True

        # Create bar plot
        try:
            sns.barplot(data=metric_data, x="model", y="value", hue="dataset", palette=dataset_colors, ax=ax)
        except Exception as e:
            print(f"Error plotting {metric}: {e}")
            ax.set_title(f"{metric} (Plot error)")
            ax.axis("off")
            continue

        # Customize the plot
        # Clean up the title for better readability
        title = metric.replace("_", " ").title()
        ax.set_title(f"{title}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)

        # Set consistent y-axis limits from 0 to 1
        ax.set_ylim(0, 1)

        # Rotate x-axis labels for better readability
        ax.tick_params(axis="x", rotation=45)

        # Remove legend from all subplots to save space
        if ax.get_legend() is not None:
            ax.legend().set_visible(False)

    # Hide empty subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")

    # Adjust layout more tightly
    plt.tight_layout(pad=1.5)

    if has_data:
        # Save the plot
        plot_format = cfg.collect_metrics.get("plot_format", "png")
        plot_file = output_dir / f"{plot_name}.{plot_format}"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Saved {plot_name} visualization to: {plot_file}")
    else:
        print(f"No data to plot for {plot_name} - skipping visualization")

    plt.close()


def create_separate_legend(
    unique_datasets: list, dataset_colors: dict, output_dir: Path, plot_format: str = "png"
) -> None:
    """Create a separate legend image."""
    # Create a new figure just for the legend
    fig_legend = plt.figure(figsize=(3, len(unique_datasets) * 0.3 + 0.5))

    # Create legend handles for all datasets
    legend_handles = []
    legend_labels = []
    for dataset in unique_datasets:
        legend_handles.append(Patch(color=dataset_colors[dataset]))
        legend_labels.append(dataset)

    # Create legend
    _legend = fig_legend.legend(
        legend_handles, legend_labels, title="Dataset", loc="center", frameon=True, fancybox=True, shadow=True
    )

    # Remove axes
    fig_legend.gca().set_axis_off()

    # Save legend
    legend_file = output_dir / f"legend.{plot_format}"
    plt.savefig(legend_file, dpi=300, bbox_inches="tight")
    print(f"Saved legend to: {legend_file}")

    plt.close(fig_legend)


@hydra.main(version_base=None, config_path="../conf", config_name="collect_metrics_conf")
def main(cfg: DictConfig) -> None:
    """Main function to collect and save metrics."""
    # Create output directory
    output_dir = Path(cfg.collect_metrics.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Collect all metrics
    print("Collecting metrics...")
    metrics_df = collect_all_metrics(cfg)

    if metrics_df.empty:
        print("No metrics found!")
        return

    # Save the fused metrics
    output_file = output_dir / "fused_metrics.csv"
    metrics_df.to_csv(output_file, index=False)
    print(f"Saved fused metrics to: {output_file}")
    print(f"Total rows: {len(metrics_df)}")

    # Create visualization
    print("Creating metrics visualization...")
    create_metrics_visualization(metrics_df, output_dir, cfg)

    # Save a copy of the configuration
    config_file = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_file)
    print(f"Saved configuration to: {config_file}")

    # Also save individual config files for reference
    config_dir = output_dir / "configs"
    config_dir.mkdir(exist_ok=True)

    # Copy the main config files
    conf_files = ["collect_metrics_conf.yaml", "dataset_list.yaml", "model_list.yaml"]

    for conf_file in conf_files:
        src = Path("conf") / conf_file
        if src.exists():
            shutil.copy2(src, config_dir / conf_file)

    print(f"Saved config copies to: {config_dir}")

    # Print summary
    print("\n--- Summary ---")
    print(f"Datasets: {metrics_df['dataset'].nunique()}")
    print(f"Models: {metrics_df['model'].nunique()}")
    print(f"Metrics: {metrics_df['metric'].nunique()}")
    print(f"Unique models: {sorted(metrics_df['model'].unique())}")


if __name__ == "__main__":
    main()
