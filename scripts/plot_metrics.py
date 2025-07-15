#!/usr/bin/env python3
"""Script to plot metrics from collected evaluation results."""

import argparse
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from omegaconf import DictConfig, OmegaConf

# Model name cleaning functions removed - using model_name from config instead


def clean_metric_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up metric names by removing evaluator prefixes."""
    df = df.copy()

    # Remove prefixes like "scib/" and "LabelSimilarity/"
    df["metric"] = df["metric"].str.replace("scib/", "", regex=False)
    df["metric"] = df["metric"].str.replace("LabelSimilarity/", "", regex=False)

    return df


def create_combined_metric_name(row):
    """Create combined metric names for visualization."""
    # Only combine for metrics that have label and label_kind columns
    # For LabelSimilarity/AUC metrics, only separate by label_kind, not individual label
    if pd.notna(row.get("label")) and pd.notna(row.get("label_kind")):
        if "LabelSimilarity" in row["metric"] or "mean_auc" in row["metric"]:
            return f"{row['metric']}_{row['label_kind']}"
        else:
            return f"{row['metric']}_{row['label_kind']}_{row['label']}"
    else:
        return row["metric"]


def create_single_metrics_plot(
    df_subset: pd.DataFrame,
    output_dir: Path,
    plot_name: str,
    dataset_colors: dict,
    plot_format: str = "png",
    vertical: bool = False,
    font_size: int = 12,
    fig_width: int = 6,
    fig_height: int = 5,
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width * n_cols, fig_height * n_rows))

    # Always flatten axes to a 1D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        # Single subplot case
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        # Single row or single column - already 1D, but make it a list
        axes = list(axes) if hasattr(axes, "__len__") else [axes]
    else:
        # Multi-row, multi-column - flatten to 1D
        axes = axes.flatten()

    # Set style
    sns.set_style("whitegrid")

    # Track if we have any data to plot
    has_data = False

    # Plot each metric
    for i, metric in enumerate(unique_metrics):
        ax = axes[i]  # Simple indexing now that axes is always 1D

        # Filter data for this metric
        metric_data = df_subset[df_subset["combined_metric"] == metric]

        if metric_data.empty:
            ax.set_title(f"{metric} (No data)", fontsize=font_size)
            ax.axis("off")
            continue

        has_data = True

        # Create bar plot - vertical or horizontal based on parameter
        try:
            if vertical:
                sns.barplot(
                    data=metric_data,
                    y="model_name",
                    x="value",
                    hue="dataset",
                    palette=dataset_colors,
                    ax=ax,
                    orient="h",
                )
            else:
                sns.barplot(data=metric_data, x="model_name", y="value", hue="dataset", palette=dataset_colors, ax=ax)
        except Exception as e:
            print(f"Error plotting {metric}: {e}")
            ax.set_title(f"{metric} (Plot error)", fontsize=font_size)
            ax.axis("off")
            continue

        # Customize the plot
        # Clean up the title for better readability
        title = metric.replace("_", " ").title()
        ax.set_title(f"{title}", fontsize=font_size + 2, fontweight="bold")

        if vertical:
            ax.set_xlabel("Value", fontsize=font_size)
            ax.set_ylabel("Model", fontsize=font_size)
            # Set consistent x-axis limits from 0 to 1
            ax.set_xlim(0, 1)
            # Rotate y-axis labels for better readability if needed
            ax.tick_params(axis="y", labelsize=font_size - 1)
            ax.tick_params(axis="x", labelsize=font_size - 1)
        else:
            ax.set_xlabel("Model", fontsize=font_size)
            ax.set_ylabel("Value", fontsize=font_size)
            # Set consistent y-axis limits from 0 to 1
            ax.set_ylim(0, 1)
            # Rotate x-axis labels for better readability
            ax.tick_params(axis="x", rotation=45, labelsize=font_size - 1)
            ax.tick_params(axis="y", labelsize=font_size - 1)

        # Remove legend from all subplots to save space
        if ax.get_legend() is not None:
            ax.legend().set_visible(False)

    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis("off")

    # Adjust layout more tightly
    plt.tight_layout(pad=1.5)

    if has_data:
        # Save the plot
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


def plot_metrics(
    metrics_file: Path,
    output_dir: Path,
    plot_format: str = "png",
    skip_scib: bool = False,
    skip_batch: bool = False,
    vertical: bool = False,
    font_size: int = 12,
    fig_width: int = 6,
    fig_height: int = 5,
) -> None:
    """Main plotting function."""
    # Load the metrics data
    print(f"Loading metrics from: {metrics_file}")
    df = pd.read_csv(metrics_file)

    # Clean up the data
    df_clean = clean_metric_names(df)

    # Convert values to numeric to avoid categorical interpretation
    df_clean["value"] = pd.to_numeric(df_clean["value"], errors="coerce")

    # Remove any rows with NaN values (from conversion errors)
    df_clean = df_clean.dropna(subset=["value"])

    # Filter out batch metrics if requested
    if skip_batch:
        print("Filtering out batch metrics (--skip-batch flag used)")
        df_clean = df_clean[df_clean["label_kind"] != "batch"]

    # Create combined metric names
    df_clean["combined_metric"] = df_clean.apply(create_combined_metric_name, axis=1)

    # Create legend for all datasets (shared across plots)
    unique_datasets = sorted(df_clean["dataset"].unique())
    colors = sns.color_palette("husl", len(unique_datasets))
    dataset_colors = dict(zip(unique_datasets, colors, strict=False))

    create_separate_legend(unique_datasets, dataset_colors, output_dir, plot_format)

    # Separate different types of metrics
    scib_data = df_clean[
        df_clean["metric"].str.startswith("scib")
        | df_clean["metric"].str.contains("Bio_Conservation_Score|Batch_Integration_Score|Overall_Score")
    ].copy()

    # Fix the filtering for LabelSimilarity metrics to include both mean_auc and accuracy_over_random
    labelsim_data = df_clean[df_clean["metric"].str.contains("mean_auc|accuracy_over_random")].copy()

    # Create separate plots for each metric type
    if not scib_data.empty and not skip_scib:
        print("Creating scib metrics visualization...")
        create_single_metrics_plot(
            scib_data,
            output_dir,
            "scib_metrics",
            dataset_colors,
            plot_format,
            vertical,
            font_size,
            fig_width,
            fig_height,
        )
    elif skip_scib:
        print("Skipping scib metrics visualization (--skip-scib flag used)")

    if not labelsim_data.empty:
        print("Creating separate plots for each LabelSimilarity metric...")

        # Get unique metrics and create separate plots for each
        unique_metrics = sorted(labelsim_data["metric"].unique())

        for metric in unique_metrics:
            metric_data = labelsim_data[labelsim_data["metric"] == metric].copy()
            if not metric_data.empty:
                # Deduplicate mean_auc data since it's calculated as overall metric across all labels
                if metric == "mean_auc":
                    print(f"Deduplicating {metric} data (removing duplicate rows with same values)...")
                    # Keep only one row per dataset/model combination for mean_auc
                    metric_data = metric_data.drop_duplicates(
                        subset=["dataset", "model", "model_name", "metric", "value"]
                    )
                    print(
                        f"  Reduced from {len(labelsim_data[labelsim_data['metric'] == metric])} to {len(metric_data)} rows"
                    )

                plot_name = f"{metric}_metrics"
                print(f"Creating plot for {metric}...")
                # Use vertical orientation for mean_auc to reduce whitespace
                use_vertical = vertical or (metric == "mean_auc")
                create_single_metrics_plot(
                    metric_data,
                    output_dir,
                    plot_name,
                    dataset_colors,
                    plot_format,
                    use_vertical,
                    font_size,
                    fig_width,
                    fig_height,
                )

    # Print summary for debugging
    print("Data summary:")
    print(f"  Datasets: {sorted(df_clean['dataset'].unique())}")
    print(f"  Model names: {sorted(df_clean['model_name'].unique())}")
    print(f"  Model sources: {sorted(df_clean['model'].unique())}")
    print(f"  Original metrics: {sorted(df_clean['metric'].unique())}")
    print(f"  Combined metrics: {sorted(df_clean['combined_metric'].unique())}")
    print(f"  Value column dtype: {df_clean['value'].dtype}")

    # Show what metrics are being plotted
    if not labelsim_data.empty:
        print(f"  LabelSimilarity metrics being plotted: {sorted(labelsim_data['metric'].unique())}")


def plot_metrics_from_config(cfg: DictConfig) -> None:
    """Plot metrics using configuration from config file."""
    # Get metrics file path
    metrics_file = Path(cfg.collect_metrics.output_dir) / "fused_metrics.csv"

    # Create plots output directory
    plots_output_dir = Path(cfg.collect_metrics.output_dir) / "plots"
    plots_output_dir.mkdir(parents=True, exist_ok=True)

    # Get plotting settings from config
    plot_format = cfg.collect_metrics.get("plot_format", "png")
    vertical = cfg.collect_metrics.get("vertical_plots", False)
    font_size = cfg.collect_metrics.get("font_size", 12)
    fig_width = cfg.collect_metrics.get("fig_width", 6)
    fig_height = cfg.collect_metrics.get("fig_height", 5)

    # Default skip options (can be overridden by command line)
    skip_scib = True  # Default to skipping scib since it has issues
    skip_batch = True  # Default to skipping batch since it has issues

    # Check if metrics file exists
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        print("Please run collect_metrics.py first to generate the metrics file.")
        return

    print(f"Using metrics file: {metrics_file}")
    print(f"Output directory: {plots_output_dir}")
    print(
        f"Plot settings: format={plot_format}, vertical={vertical}, font_size={font_size}, fig_size={fig_width}x{fig_height}"
    )

    # Plot metrics
    plot_metrics(
        metrics_file, plots_output_dir, plot_format, skip_scib, skip_batch, vertical, font_size, fig_width, fig_height
    )

    print(f"\nPlotting completed! Check output directory: {plots_output_dir}")


@hydra.main(version_base=None, config_path="../conf", config_name="collect_metrics_conf")
def main(cfg: DictConfig) -> None:
    """Main function using Hydra configuration."""
    plot_metrics_from_config(cfg)


def main_cli():
    """Main function with command line interface (for backward compatibility)."""
    parser = argparse.ArgumentParser(description="Plot metrics from collected evaluation results")
    parser.add_argument("--metrics_file", type=Path, required=True, help="Path to the fused_metrics.csv file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for plots")
    parser.add_argument(
        "--plot_format", type=str, default="png", choices=["png", "pdf", "svg"], help="Plot format (default: png)"
    )
    parser.add_argument("--skip-scib", action="store_true", help="Skip plotting scib metrics")
    parser.add_argument("--skip-batch", action="store_true", help="Skip plotting batch metrics")
    parser.add_argument("--vertical", action="store_true", help="Use vertical orientation (model names on y-axis)")
    parser.add_argument("--font-size", type=int, default=12, help="Font size for labels and titles (default: 12)")
    parser.add_argument("--fig-width", type=int, default=6, help="Figure width per subplot (default: 6)")
    parser.add_argument("--fig-height", type=int, default=5, help="Figure height per subplot (default: 5)")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if metrics file exists
    if not args.metrics_file.exists():
        print(f"Error: Metrics file not found: {args.metrics_file}")
        return

    # Plot metrics
    plot_metrics(
        args.metrics_file,
        args.output_dir,
        args.plot_format,
        args.skip_scib,
        args.skip_batch,
        args.vertical,
        args.font_size,
        args.fig_width,
        args.fig_height,
    )

    print(f"\nPlotting completed! Check output directory: {args.output_dir}")


if __name__ == "__main__":
    # Use Hydra by default, but allow CLI mode with --cli flag
    import sys

    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        main_cli()
    else:
        main()
