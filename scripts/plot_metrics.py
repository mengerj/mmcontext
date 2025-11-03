#!/usr/bin/env python3
"""Script to plot metrics from collected evaluation results."""

import argparse
import re
from itertools import combinations
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


def extract_base_model_name(model_name: str) -> str:
    """Extract base model name by removing version tags like _v1, _v2, etc."""
    # Remove version tags at the end (e.g., _v3, _v4, etc.)
    base_name = re.sub(r"_v\d+$", "", model_name)
    return base_name


# Short name extraction removed - now using displayed_name directly from config


def apply_model_order(df: pd.DataFrame, model_order: list = None) -> pd.DataFrame:
    """
    Sort dataframe by specified model order.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'model_name' column to sort by
    model_order : list, optional
        List of model names in desired order. Models not in this list will be
        sorted alphabetically and appended at the end.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by the specified model order
    """
    if model_order is None or len(model_order) == 0:
        return df

    # Convert model_order to a dict for efficient lookup
    order_dict = {model: i for i, model in enumerate(model_order)}

    # Create a sorting key based on model_order
    # Models not in model_order will be sorted alphabetically at the end
    def get_sort_key(row):
        model_name = row["model_name"]
        if model_name in order_dict:
            return (order_dict[model_name], "")  # Sort by order index first
        else:
            return (9999, model_name)  # Sort alphabetically at end

    df = df.copy()

    # Apply the sorting
    df["_sort_key"] = df.apply(get_sort_key, axis=1)
    df = df.sort_values("_sort_key")
    df = df.drop("_sort_key", axis=1)

    return df


def group_model_repetitions(df: pd.DataFrame, group_repetitions: bool = True, model_order: list = None) -> pd.DataFrame:
    """Group model repetitions and calculate statistics (mean, std, sem)."""
    if not group_repetitions:
        # No grouping needed, just return the dataframe
        return df

    # Add base model name column
    df = df.copy()
    df["base_model_name"] = df["model_name"].apply(extract_base_model_name)

    # Group by all columns except model_name and value, then aggregate
    # Don't include combined_metric since it's created later
    groupby_cols = ["dataset", "base_model_name", "dataset_label", "label", "label_kind", "metric"]

    # Check if we actually have repetitions to group
    repetition_counts = df.groupby(groupby_cols)["model_name"].nunique()
    has_repetitions = (repetition_counts > 1).any()

    if not has_repetitions:
        print("No model repetitions found - using original data")
        return df

    print(f"Found repetitions for: {repetition_counts[repetition_counts > 1].index.tolist()}")

    # Calculate statistics
    grouped_stats = (
        df.groupby(groupby_cols)
        .agg(
            {
                "value": ["mean", "std", "sem", "count"],
                "model": "first",  # Keep first model source for reference
                "model_name": lambda x: f"{x.iloc[0]} (n={len(x)})",  # Show count in name
            }
        )
        .reset_index()
    )

    # Flatten column names
    grouped_stats.columns = [
        "dataset",
        "base_model_name",
        "dataset_label",
        "label",
        "label_kind",
        "metric",
        "value_mean",
        "value_std",
        "value_sem",
        "value_count",
        "model",
        "model_name",
    ]

    # Rename columns to match original structure
    grouped_stats = grouped_stats.rename(
        columns={
            "value_mean": "value",
            "value_std": "value_std",
            "value_sem": "value_sem",
            "value_count": "n_repetitions",
        }
    )

    # Replace model_name with base_model_name for display
    # (base_model_name removes version tags like _v1, _v2, etc.)
    grouped_stats["model_name"] = grouped_stats["base_model_name"]

    return grouped_stats


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
    # Group all metrics by label_kind only (not individual labels) so they appear in the same plot
    if pd.notna(row.get("label")) and pd.notna(row.get("label_kind")):
        return f"{row['metric']}_{row['label_kind']}"
    else:
        return row["metric"]


def create_accuracy_plot_with_baseline(
    accuracy_data: pd.DataFrame,
    baseline_data: pd.DataFrame,
    output_dir: Path,
    plot_name: str,
    dataset_colors: dict,
    plot_format: str = "png",
    vertical: bool = False,
    font_size: int = 12,
    tick_label_size: int = None,
    fig_width: int = 6,
    fig_height: int = 5,
    show_error_bars: bool = True,
    axis_padding: float = 0.1,
    label_pad: float = 10,
    tick_label_pad: float = 5,
    y_axis_max: float = None,
    accuracy_over_random_y_max: float = None,
) -> None:
    """Create accuracy plots with baseline overlay as dotted lines."""
    # Get unique metrics
    unique_metrics = sorted(accuracy_data["combined_metric"].unique())
    n_metrics = len(unique_metrics)

    if n_metrics == 0:
        print(f"No accuracy metrics to plot for {plot_name}!")
        return

    # Calculate subplot layout - use fewer columns if we have fewer metrics
    n_cols = min(3, n_metrics)  # Don't use more columns than we have metrics
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create figure with subplots - adjust width based on actual number of columns used
    actual_width = fig_width * n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(actual_width, fig_height * n_rows), constrained_layout=True)

    # Always flatten axes to a 1D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = list(axes) if hasattr(axes, "__len__") else [axes]
    else:
        axes = axes.flatten()

    # Set style
    sns.set_style("whitegrid")

    # Track if we have any data to plot
    has_data = False

    # Plot each metric
    for i, metric in enumerate(unique_metrics):
        ax = axes[i]

        # Filter data for this metric
        metric_data = accuracy_data[accuracy_data["combined_metric"] == metric]

        # For baseline data, we need to match based on the same label_kind pattern
        # Extract the label_kind suffix from the accuracy metric (e.g., "accuracy_bio" -> "_bio")
        if "_" in metric:
            label_suffix = "_" + metric.split("_", 1)[1]  # Get everything after first underscore
            baseline_combined_metric = f"random_baseline_accuracy{label_suffix}"
        else:
            baseline_combined_metric = "random_baseline_accuracy"

        baseline_metric_data = baseline_data[baseline_data["combined_metric"] == baseline_combined_metric]

        if metric_data.empty:
            ax.set_title(f"{metric} (No data)", fontsize=font_size)
            ax.axis("off")
            continue

        has_data = True

        # Create bar plot - vertical or horizontal based on parameter
        try:
            # Check if we have error bar data (from grouped repetitions)
            has_error_data = "value_sem" in metric_data.columns and show_error_bars

            if vertical:
                bars = sns.barplot(
                    data=metric_data,
                    y="model_name",
                    x="value",
                    hue="dataset_label",
                    palette=dataset_colors,
                    ax=ax,
                    orient="h",
                )

                # Add error bars if available
                if has_error_data:
                    # Get bar positions from the plot
                    bar_positions = []
                    for patch in bars.patches:
                        bar_positions.append(
                            (patch.get_y() + patch.get_height() / 2, patch.get_x() + patch.get_width())
                        )

                    # Add error bars for each data point
                    for i, (_, row) in enumerate(metric_data.iterrows()):
                        if pd.notna(row["value_sem"]) and row["value_sem"] > 0 and i < len(bar_positions):
                            bar_y, bar_x = bar_positions[i]
                            ax.errorbar(
                                bar_x,
                                bar_y,
                                xerr=row["value_sem"],
                                fmt="none",
                                color="black",
                                capsize=3,
                                capthick=1,
                                alpha=0.8,
                            )

                # Add baseline lines within each bar
                if not baseline_metric_data.empty:
                    # Get bar positions and add baseline lines within each bar
                    for i, (_, row) in enumerate(metric_data.iterrows()):
                        if i < len(bars.patches):
                            patch = bars.patches[i]

                            # Find matching baseline for this dataset
                            matching_baseline = baseline_metric_data[
                                baseline_metric_data["dataset_label"] == row["dataset_label"]
                            ]

                            if not matching_baseline.empty:
                                baseline_value = matching_baseline["value"].iloc[0]

                                # For horizontal bars (vertical=True), draw vertical line within bar
                                bar_y_center = patch.get_y() + patch.get_height() / 2
                                bar_height = patch.get_height() * 0.8  # Make line slightly shorter than bar

                                # Draw vertical line within the bar at baseline value
                                ax.plot(
                                    [baseline_value, baseline_value],
                                    [bar_y_center - bar_height / 2, bar_y_center + bar_height / 2],
                                    color="gray",
                                    linewidth=3,
                                    alpha=0.9,
                                    zorder=10,
                                )
            else:
                bars = sns.barplot(
                    data=metric_data, x="model_name", y="value", hue="dataset_label", palette=dataset_colors, ax=ax
                )

                # Add error bars if available
                if has_error_data:
                    # Get bar positions from the plot
                    bar_positions = []
                    for patch in bars.patches:
                        bar_positions.append(
                            (patch.get_x() + patch.get_width() / 2, patch.get_y() + patch.get_height())
                        )

                    # Add error bars for each data point
                    for i, (_, row) in enumerate(metric_data.iterrows()):
                        if pd.notna(row["value_sem"]) and row["value_sem"] > 0 and i < len(bar_positions):
                            bar_x, bar_y = bar_positions[i]
                            ax.errorbar(
                                bar_x,
                                bar_y,
                                yerr=row["value_sem"],
                                fmt="none",
                                color="black",
                                capsize=3,
                                capthick=1,
                                alpha=0.8,
                            )

                # Add baseline lines within each bar
                if not baseline_metric_data.empty:
                    # Get bar positions and add baseline lines within each bar
                    for i, (_, row) in enumerate(metric_data.iterrows()):
                        if i < len(bars.patches):
                            patch = bars.patches[i]

                            # Find matching baseline for this dataset
                            matching_baseline = baseline_metric_data[
                                baseline_metric_data["dataset_label"] == row["dataset_label"]
                            ]

                            if not matching_baseline.empty:
                                baseline_value = matching_baseline["value"].iloc[0]

                                # For vertical bars (vertical=False), draw horizontal line within bar
                                bar_x_center = patch.get_x() + patch.get_width() / 2
                                bar_width = patch.get_width() * 0.8  # Make line slightly shorter than bar

                                # Draw horizontal line within the bar at baseline value
                                ax.plot(
                                    [bar_x_center - bar_width / 2, bar_x_center + bar_width / 2],
                                    [baseline_value, baseline_value],
                                    color="gray",
                                    linewidth=3,
                                    alpha=0.9,
                                    zorder=10,
                                )

        except Exception as e:
            print(f"Error plotting {metric}: {e}")
            ax.set_title(f"{metric} (Plot error)", fontsize=font_size)
            ax.axis("off")
            continue

        # Customize the plot
        title = metric.replace("_", " ").title()
        ax.set_title(f"{title}", fontsize=font_size + 2, fontweight="bold")

        # Use tick_label_size if provided, otherwise default to font_size - 1
        tick_size = tick_label_size if tick_label_size is not None else font_size - 1

        if vertical:
            ax.set_xlabel("Accuracy", fontsize=font_size)
            ax.set_ylabel("Model", fontsize=font_size, labelpad=label_pad)

            # Improve x-axis limits to reduce white space
            x_max = metric_data["value"].max()
            x_min = metric_data["value"].min()
            x_range = x_max - x_min

            # Determine fixed max value based on metric type
            fixed_max = None
            if "accuracy_over_random" in metric.lower() and accuracy_over_random_y_max is not None:
                fixed_max = accuracy_over_random_y_max
            elif y_axis_max is not None:
                fixed_max = y_axis_max

            # Add configurable padding
            if fixed_max is not None:
                ax.set_xlim(max(0, x_min - x_range * axis_padding), fixed_max)
            elif x_range > 0:
                padding = x_range * axis_padding
                ax.set_xlim(max(0, x_min - padding), x_max + padding)
            else:
                # If all values are the same, center around the value
                fallback_padding = x_max * 0.1 if x_max > 0 else 0.1
                ax.set_xlim(max(0, x_max - fallback_padding), x_max + fallback_padding)

            ax.tick_params(axis="y", labelsize=tick_size, pad=tick_label_pad)
            ax.tick_params(axis="x", labelsize=tick_size)
        else:
            ax.set_xlabel("Model", fontsize=font_size, labelpad=label_pad)
            ax.set_ylabel("Accuracy", fontsize=font_size)

            # Improve y-axis limits to reduce white space
            y_max = metric_data["value"].max()
            y_min = metric_data["value"].min()
            y_range = y_max - y_min

            # Determine fixed max value based on metric type
            fixed_max = None
            if "accuracy_over_random" in metric.lower() and accuracy_over_random_y_max is not None:
                fixed_max = accuracy_over_random_y_max
            elif y_axis_max is not None:
                fixed_max = y_axis_max

            # Add configurable padding
            if fixed_max is not None:
                ax.set_ylim(max(0, y_min - y_range * axis_padding), fixed_max)
            elif y_range > 0:
                padding = y_range * axis_padding
                ax.set_ylim(max(0, y_min - padding), y_max + padding)
            else:
                # If all values are the same, center around the value
                fallback_padding = y_max * 0.1 if y_max > 0 else 0.1
                ax.set_ylim(max(0, y_max - fallback_padding), y_max + fallback_padding)

            ax.tick_params(axis="x", rotation=45, labelsize=tick_size, pad=tick_label_pad)
            ax.tick_params(axis="y", labelsize=tick_size)

        # Remove legend from all subplots to save space
        if ax.get_legend() is not None:
            ax.legend().set_visible(False)

    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis("off")

    # Layout is handled by constrained_layout=True in subplots()

    if has_data:
        # Save the plot
        plot_file = output_dir / f"{plot_name}.{plot_format}"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Saved {plot_name} visualization to: {plot_file}")
    else:
        print(f"No data to plot for {plot_name} - skipping visualization")

    plt.close()


def create_single_metrics_plot(
    df_subset: pd.DataFrame,
    output_dir: Path,
    plot_name: str,
    dataset_colors: dict,
    plot_format: str = "png",
    vertical: bool = False,
    font_size: int = 12,
    tick_label_size: int = None,
    fig_width: int = 6,
    fig_height: int = 5,
    show_error_bars: bool = True,
    axis_padding: float = 0.1,
    label_pad: float = 10,
    tick_label_pad: float = 5,
    y_axis_max: float = None,
    accuracy_over_random_y_max: float = None,
) -> None:
    """Create a single metrics plot for a subset of data."""
    # Get unique metrics
    unique_metrics = sorted(df_subset["combined_metric"].unique())
    n_metrics = len(unique_metrics)

    if n_metrics == 0:
        print(f"No metrics to plot for {plot_name}!")
        return

    # Calculate subplot layout - use fewer columns if we have fewer metrics
    n_cols = min(3, n_metrics)  # Don't use more columns than we have metrics
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create figure with subplots - adjust width based on actual number of columns used
    actual_width = fig_width * n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(actual_width, fig_height * n_rows), constrained_layout=True)

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
            # Check if we have error bar data (from grouped repetitions)
            has_error_data = "value_sem" in metric_data.columns and show_error_bars

            if vertical:
                bars = sns.barplot(
                    data=metric_data,
                    y="model_name",
                    x="value",
                    hue="dataset_label",
                    palette=dataset_colors,
                    ax=ax,
                    orient="h",
                )

                # Add error bars if available
                if has_error_data:
                    # Get bar positions from the plot
                    bar_positions = []
                    for patch in bars.patches:
                        bar_positions.append(
                            (patch.get_y() + patch.get_height() / 2, patch.get_x() + patch.get_width())
                        )

                    # Add error bars for each data point
                    for i, (_, row) in enumerate(metric_data.iterrows()):
                        if pd.notna(row["value_sem"]) and row["value_sem"] > 0 and i < len(bar_positions):
                            bar_y, bar_x = bar_positions[i]
                            ax.errorbar(
                                bar_x,
                                bar_y,
                                xerr=row["value_sem"],
                                fmt="none",
                                color="black",
                                capsize=3,
                                capthick=1,
                                alpha=0.8,
                            )
            else:
                bars = sns.barplot(
                    data=metric_data, x="model_name", y="value", hue="dataset_label", palette=dataset_colors, ax=ax
                )

                # Add error bars if available
                if has_error_data:
                    # Get bar positions from the plot
                    bar_positions = []
                    for patch in bars.patches:
                        bar_positions.append(
                            (patch.get_x() + patch.get_width() / 2, patch.get_y() + patch.get_height())
                        )

                    # Add error bars for each data point
                    for i, (_, row) in enumerate(metric_data.iterrows()):
                        if pd.notna(row["value_sem"]) and row["value_sem"] > 0 and i < len(bar_positions):
                            bar_x, bar_y = bar_positions[i]
                            ax.errorbar(
                                bar_x,
                                bar_y,
                                yerr=row["value_sem"],
                                fmt="none",
                                color="black",
                                capsize=3,
                                capthick=1,
                                alpha=0.8,
                            )
        except Exception as e:
            print(f"Error plotting {metric}: {e}")
            ax.set_title(f"{metric} (Plot error)", fontsize=font_size)
            ax.axis("off")
            continue

        # Customize the plot
        # Clean up the title for better readability
        title = metric.replace("_", " ").title()
        ax.set_title(f"{title}", fontsize=font_size + 2, fontweight="bold")

        # Use tick_label_size if provided, otherwise default to font_size - 1
        tick_size = tick_label_size if tick_label_size is not None else font_size - 1

        if vertical:
            ax.set_xlabel("Value", fontsize=font_size)
            ax.set_ylabel("Model", fontsize=font_size, labelpad=label_pad)

            # Improve x-axis limits to reduce white space
            x_max = metric_data["value"].max()
            x_min = metric_data["value"].min()
            x_range = x_max - x_min

            # Determine fixed max value based on metric type
            fixed_max = None
            if "accuracy_over_random" in metric.lower() and accuracy_over_random_y_max is not None:
                fixed_max = accuracy_over_random_y_max
            elif y_axis_max is not None:
                fixed_max = y_axis_max

            # Add configurable padding
            if fixed_max is not None:
                ax.set_xlim(max(0, x_min - x_range * axis_padding), fixed_max)
            elif x_range > 0:
                padding = x_range * axis_padding
                ax.set_xlim(max(0, x_min - padding), x_max + padding)
            else:
                # If all values are the same, center around the value
                fallback_padding = x_max * 0.1 if x_max > 0 else 0.1
                ax.set_xlim(max(0, x_max - fallback_padding), x_max + fallback_padding)

            ax.tick_params(axis="y", labelsize=tick_size, pad=tick_label_pad)
            ax.tick_params(axis="x", labelsize=tick_size)
        else:
            ax.set_xlabel("Model", fontsize=font_size, labelpad=label_pad)
            ax.set_ylabel("Value", fontsize=font_size)

            # Improve y-axis limits to reduce white space
            y_max = metric_data["value"].max()
            y_min = metric_data["value"].min()
            y_range = y_max - y_min

            # Determine fixed max value based on metric type
            fixed_max = None
            if "accuracy_over_random" in metric.lower() and accuracy_over_random_y_max is not None:
                fixed_max = accuracy_over_random_y_max
            elif y_axis_max is not None:
                fixed_max = y_axis_max

            # Add configurable padding
            if fixed_max is not None:
                ax.set_ylim(max(0, y_min - y_range * axis_padding), fixed_max)
            elif y_range > 0:
                padding = y_range * axis_padding
                ax.set_ylim(max(0, y_min - padding), y_max + padding)
            else:
                # If all values are the same, center around the value
                fallback_padding = y_max * 0.1 if y_max > 0 else 0.1
                ax.set_ylim(max(0, y_max - fallback_padding), y_max + fallback_padding)

            ax.tick_params(axis="x", rotation=45, labelsize=tick_size, pad=tick_label_pad)
            ax.tick_params(axis="y", labelsize=tick_size)

        # Remove legend from all subplots to save space
        if ax.get_legend() is not None:
            ax.legend().set_visible(False)

    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis("off")

    # Layout is handled by constrained_layout=True in subplots()

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


def create_dataset_pair_scatter_plots(
    accuracy_data: pd.DataFrame,
    baseline_data: pd.DataFrame,
    output_dir: Path,
    plot_format: str = "png",
    font_size: int = 12,
    fig_width: int = 8,
    fig_height: int = 8,
) -> None:
    """
    Create scatter plots comparing accuracy between pairs of datasets.

    Parameters
    ----------
    accuracy_data : pd.DataFrame
        DataFrame with accuracy metrics, must have columns: dataset_label, model_name, metric, value
    baseline_data : pd.DataFrame
        DataFrame with baseline accuracy metrics, must have columns: dataset_label, metric, value
    output_dir : Path
        Directory to save plots
    plot_format : str
        Plot file format (png, pdf, svg)
    font_size : int
        Font size for labels and annotations
    fig_width : int
        Figure width in inches
    fig_height : int
        Figure height in inches
    """
    if accuracy_data.empty:
        print("No accuracy data available for scatter plots")
        return

    # Get unique dataset_label combinations
    unique_datasets = sorted(accuracy_data["dataset_label"].unique())

    if len(unique_datasets) < 2:
        print(f"Need at least 2 datasets for scatter plots, found {len(unique_datasets)}")
        return

    # Get unique combined metrics (this separates by label_kind if present)
    # If combined_metric column doesn't exist, create it
    accuracy_data = accuracy_data.copy()
    if "combined_metric" not in accuracy_data.columns:
        accuracy_data["combined_metric"] = accuracy_data.apply(create_combined_metric_name, axis=1)

    # Prepare baseline_data similarly
    baseline_data = baseline_data.copy()
    if "combined_metric" not in baseline_data.columns and not baseline_data.empty:
        baseline_data["combined_metric"] = baseline_data.apply(create_combined_metric_name, axis=1)

    unique_combined_metrics = sorted(accuracy_data["combined_metric"].unique())

    # Create pairs of datasets
    dataset_pairs = list(combinations(unique_datasets, 2))

    print(
        f"Creating scatter plots for {len(unique_combined_metrics)} metric(s) and {len(dataset_pairs)} dataset pair(s)..."
    )

    # Set style
    sns.set_style("whitegrid")

    # Process each combined metric separately (this handles different label_kinds)
    for combined_metric in unique_combined_metrics:
        metric_data = accuracy_data[accuracy_data["combined_metric"] == combined_metric].copy()

        # Get baseline for this combined metric (should match the accuracy metric pattern)
        # Extract label_kind suffix if present in combined_metric
        # Get the base metric name (before label_kind suffix)
        base_metric = metric_data["metric"].iloc[0] if not metric_data.empty else ""

        # Extract label_kind suffix from combined_metric
        if "_" in combined_metric and combined_metric != base_metric:
            # combined_metric is like "accuracy_bio", extract suffix
            label_suffix = "_" + combined_metric.split("_", 1)[1]
            baseline_combined_metric = f"random_baseline_accuracy{label_suffix}"
        else:
            baseline_combined_metric = "random_baseline_accuracy"

        # Filter baseline data for this combined metric
        baseline_metric_data = (
            baseline_data[baseline_data["combined_metric"] == baseline_combined_metric].copy()
            if not baseline_data.empty
            else pd.DataFrame()
        )

        # Create a scatter plot for each dataset pair
        for dataset1, dataset2 in dataset_pairs:
            # Filter data for each dataset
            data1 = metric_data[metric_data["dataset_label"] == dataset1].copy()
            data2 = metric_data[metric_data["dataset_label"] == dataset2].copy()

            # Get baseline values for each dataset
            baseline1 = baseline_metric_data[baseline_metric_data["dataset_label"] == dataset1]
            baseline2 = baseline_metric_data[baseline_metric_data["dataset_label"] == dataset2]

            baseline_value1 = baseline1["value"].iloc[0] if not baseline1.empty else None
            baseline_value2 = baseline2["value"].iloc[0] if not baseline2.empty else None

            # Find common models between the two datasets
            common_models = set(data1["model_name"].unique()) & set(data2["model_name"].unique())

            if len(common_models) == 0:
                print(f"  Skipping {dataset1} vs {dataset2}: no common models")
                continue

            # Create pivot tables for easier matching
            pivot1 = data1[data1["model_name"].isin(common_models)].pivot_table(
                index="model_name",
                values="value",
                aggfunc="mean",  # Use mean in case of duplicates
            )
            pivot2 = data2[data2["model_name"].isin(common_models)].pivot_table(
                index="model_name", values="value", aggfunc="mean"
            )

            # Merge to get matching pairs
            merged = pivot1.join(pivot2, how="inner", rsuffix="_2")
            merged.columns = ["x_value", "y_value"]

            if merged.empty:
                print(f"  Skipping {dataset1} vs {dataset2}: no matching data after merge")
                continue

            # Create the plot
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Plot points
            ax.scatter(
                merged["x_value"],
                merged["y_value"],
                s=100,
                alpha=0.7,
                color="black",
                edgecolors="white",
                linewidths=1.5,
                zorder=3,
            )

            # Add annotations for each point (using model_name which contains displayed_name)
            for model_name, row in merged.iterrows():
                ax.annotate(
                    model_name,
                    (row["x_value"], row["y_value"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=font_size - 2,
                    alpha=0.8,
                    zorder=4,
                )

            # Add baseline lines
            # Horizontal line for dataset2 baseline (y-axis)
            if baseline_value2 is not None:
                ax.axhline(
                    y=baseline_value2,
                    color="gray",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.6,
                    zorder=1,
                    label=f"{dataset2} baseline",
                )

            # Vertical line for dataset1 baseline (x-axis)
            if baseline_value1 is not None:
                ax.axvline(
                    x=baseline_value1,
                    color="gray",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.6,
                    zorder=1,
                    label=f"{dataset1} baseline",
                )

            # Set labels (no title for scatter plots)
            ax.set_xlabel(dataset1, fontsize=font_size, fontweight="bold")
            ax.set_ylabel(dataset2, fontsize=font_size, fontweight="bold")

            # Create a safe filename from dataset names
            safe_name1 = dataset1.replace(" ", "_").replace("/", "_")
            safe_name2 = dataset2.replace(" ", "_").replace("/", "_")
            safe_metric = combined_metric.replace(" ", "_").replace("/", "_")

            # Add grid
            ax.grid(True, alpha=0.3, zorder=0)

            # Set equal aspect ratio (1:1) so diagonal comparisons are fair
            # But allow for different axis ranges
            x_range = merged["x_value"].max() - merged["x_value"].min()
            y_range = merged["y_value"].max() - merged["y_value"].min()

            # Add some padding
            x_padding = x_range * 0.1 if x_range > 0 else 0.05
            y_padding = y_range * 0.1 if y_range > 0 else 0.05

            ax.set_xlim(merged["x_value"].min() - x_padding, merged["x_value"].max() + x_padding)
            ax.set_ylim(merged["y_value"].min() - y_padding, merged["y_value"].max() + y_padding)

            # Add diagonal line for reference (y = x)
            min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
            max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([min_val, max_val], [min_val, max_val], "k:", alpha=0.3, linewidth=1, zorder=2, label="y=x")

            # Save the plot
            plot_name = f"scatter_{safe_metric}_{safe_name1}_vs_{safe_name2}"
            plot_file = output_dir / f"{plot_name}.{plot_format}"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"  Saved scatter plot: {plot_name}")

            plt.close()


def plot_metrics(
    metrics_file: Path,
    output_dir: Path,
    plot_format: str = "png",
    skip_scib: bool = False,
    skip_batch: bool = False,
    vertical: bool = False,
    font_size: int = 12,
    tick_label_size: int = None,
    fig_width: int = 6,
    fig_height: int = 5,
    accuracy_metrics: list = None,
    baseline_accuracy_metric: str = None,
    group_repetitions: bool = True,
    show_error_bars: bool = True,
    axis_padding: float = 0.1,
    label_pad: float = 10,
    tick_label_pad: float = 5,
    y_axis_max: float = None,
    accuracy_over_random_y_max: float = None,
    model_order: list = None,
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

    # Create dataset-label combinations for datasets with multiple labels (needed before grouping)
    def create_dataset_label_name(row):
        """Create dataset name that includes label if dataset has multiple labels."""
        dataset = row["dataset"]
        label = row["label"]

        # Count unique labels for this dataset
        dataset_labels = df_clean[df_clean["dataset"] == dataset]["label"].nunique()

        if dataset_labels > 1:
            # Multiple labels - append label to dataset name
            return f"{dataset}_{label}"
        else:
            # Single label - keep original dataset name
            return dataset

    df_clean["dataset_label"] = df_clean.apply(create_dataset_label_name, axis=1)

    # Group model repetitions if requested
    if group_repetitions:
        print("Grouping model repetitions...")
        df_clean = group_model_repetitions(df_clean, group_repetitions=True, model_order=model_order)

    # Create combined metric names
    df_clean["combined_metric"] = df_clean.apply(create_combined_metric_name, axis=1)

    # Apply custom model order if specified
    if model_order:
        print(f"Applying custom model order: {model_order}")
        df_clean = apply_model_order(df_clean, model_order)

    # Create legend for all dataset-label combinations (shared across plots)
    unique_datasets = sorted(df_clean["dataset_label"].unique())
    colors = sns.color_palette("husl", len(unique_datasets))
    dataset_colors = dict(zip(unique_datasets, colors, strict=False))

    create_separate_legend(unique_datasets, dataset_colors, output_dir, plot_format)

    # Separate different types of metrics
    scib_data = df_clean[
        df_clean["metric"].str.startswith("scib")
        | df_clean["metric"].str.contains("Bio_Conservation_Score|Batch_Integration_Score|Overall_Score")
    ].copy()

    # Separate different types of metrics based on configuration
    if accuracy_metrics is None:
        accuracy_metrics = []

    # Clean metric names for comparison (remove prefixes)
    clean_accuracy_metrics = [
        metric.replace("LabelSimilarity/", "").replace("scib/", "") for metric in accuracy_metrics
    ]
    clean_baseline_metric = (
        baseline_accuracy_metric.replace("LabelSimilarity/", "").replace("scib/", "")
        if baseline_accuracy_metric
        else None
    )

    # Filter accuracy metrics
    accuracy_data = (
        df_clean[df_clean["metric"].isin(clean_accuracy_metrics)].copy() if clean_accuracy_metrics else pd.DataFrame()
    )

    # Filter baseline data
    baseline_data = (
        df_clean[df_clean["metric"] == clean_baseline_metric].copy() if clean_baseline_metric else pd.DataFrame()
    )

    # Filter regular metrics (excluding accuracy and baseline metrics)
    excluded_metrics = clean_accuracy_metrics + ([clean_baseline_metric] if clean_baseline_metric else [])
    regular_data = df_clean[~df_clean["metric"].isin(excluded_metrics)].copy()

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
            tick_label_size,
            fig_width,
            fig_height,
            show_error_bars,
            axis_padding,
            label_pad,
            tick_label_pad,
            y_axis_max,
            accuracy_over_random_y_max,
        )
    elif skip_scib:
        print("Skipping scib metrics visualization (--skip-scib flag used)")

    # Plot accuracy metrics with baseline overlay
    if not accuracy_data.empty:
        print("Creating accuracy metrics with baseline overlay...")

        # Get unique accuracy metrics and create separate plots for each
        unique_accuracy_metrics = sorted(accuracy_data["metric"].unique())

        for metric in unique_accuracy_metrics:
            metric_data = accuracy_data[accuracy_data["metric"] == metric].copy()
            if not metric_data.empty:
                # Create baseline plot
                plot_name = f"{metric}_metrics_with_baseline"
                print(f"Creating accuracy plot for {metric} with baseline overlay...")
                create_accuracy_plot_with_baseline(
                    metric_data,
                    baseline_data,
                    output_dir,
                    plot_name,
                    dataset_colors,
                    plot_format,
                    vertical,
                    font_size,
                    tick_label_size,
                    fig_width,
                    fig_height,
                    show_error_bars,
                    axis_padding,
                    label_pad,
                    tick_label_pad,
                    y_axis_max,
                    accuracy_over_random_y_max,
                )

                # Also create basic accuracy plot without baseline
                basic_plot_name = f"{metric}_metrics"
                print(f"Creating basic accuracy plot for {metric}...")
                create_single_metrics_plot(
                    metric_data,
                    output_dir,
                    basic_plot_name,
                    dataset_colors,
                    plot_format,
                    vertical,
                    font_size,
                    tick_label_size,
                    fig_width,
                    fig_height,
                    show_error_bars,
                    axis_padding,
                    label_pad,
                    tick_label_pad,
                    y_axis_max,
                    accuracy_over_random_y_max,
                )

        # Create scatter plots for dataset pairs
        print("\nCreating dataset pair scatter plots...")
        create_dataset_pair_scatter_plots(
            accuracy_data,
            baseline_data,
            output_dir,
            plot_format,
            font_size,
            fig_width,
            fig_height,
        )

    # Plot regular metrics (non-accuracy, non-baseline)
    if not regular_data.empty:
        print("Creating separate plots for each regular metric...")

        # Get unique metrics and create separate plots for each
        unique_metrics = sorted(regular_data["metric"].unique())

        for metric in unique_metrics:
            metric_data = regular_data[regular_data["metric"] == metric].copy()
            if not metric_data.empty:
                # Deduplicate mean_auc data since it's calculated as overall metric across all labels
                if metric == "mean_auc":
                    print(f"Deduplicating {metric} data (removing duplicate rows with same values)...")
                    # Keep only one row per dataset_label/model combination for mean_auc
                    metric_data = metric_data.drop_duplicates(
                        subset=["dataset_label", "model", "model_name", "metric", "value"]
                    )
                    print(
                        f"  Reduced from {len(regular_data[regular_data['metric'] == metric])} to {len(metric_data)} rows"
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
                    tick_label_size,
                    fig_width,
                    fig_height,
                    show_error_bars,
                    axis_padding,
                    label_pad,
                    tick_label_pad,
                    y_axis_max,
                    accuracy_over_random_y_max,
                )

    # Print summary for debugging
    print("Data summary:")
    print(f"  Original datasets: {sorted(df_clean['dataset'].unique())}")
    print(f"  Dataset-label combinations: {sorted(df_clean['dataset_label'].unique())}")
    print(f"  Model names: {sorted(df_clean['model_name'].unique())}")
    print(f"  Model sources: {sorted(df_clean['model'].unique())}")
    print(f"  Original metrics: {sorted(df_clean['metric'].unique())}")
    print(f"  Combined metrics: {sorted(df_clean['combined_metric'].unique())}")
    print(f"  Value column dtype: {df_clean['value'].dtype}")

    # Show what metrics are being plotted
    if not accuracy_data.empty:
        print(f"  Accuracy metrics being plotted: {sorted(accuracy_data['metric'].unique())}")
    if not regular_data.empty:
        print(f"  Regular metrics being plotted: {sorted(regular_data['metric'].unique())}")
    if not baseline_data.empty:
        print(f"  Baseline metric: {sorted(baseline_data['metric'].unique())}")


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
    tick_label_size = cfg.collect_metrics.get("tick_label_size", None)
    fig_width = cfg.collect_metrics.get("fig_width", 6)
    fig_height = cfg.collect_metrics.get("fig_height", 5)
    axis_padding = cfg.collect_metrics.get("axis_padding", 0.1)
    label_pad = cfg.collect_metrics.get("label_pad", 10)
    tick_label_pad = cfg.collect_metrics.get("tick_label_pad", 5)

    # Get accuracy metrics configuration
    accuracy_metrics = cfg.collect_metrics.get("accuracy_metrics", [])
    baseline_accuracy_metric = cfg.collect_metrics.get("baseline_accuracy_metric", None)

    # Get repetition grouping configuration
    group_repetitions = cfg.collect_metrics.get("group_repetitions", True)
    show_error_bars = cfg.collect_metrics.get("show_error_bars", True)

    # Get model order configuration
    model_order = cfg.collect_metrics.get("model_order", None)
    if model_order is not None and isinstance(model_order, list):
        # Convert OmegaConf list to Python list
        model_order = list(model_order)

    # Get y-axis max configuration
    y_axis_max = cfg.collect_metrics.get("y_axis_max", None)
    accuracy_over_random_y_max = cfg.collect_metrics.get("accuracy_over_random_y_max", None)

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
    print(f"Accuracy metrics: {accuracy_metrics}")
    print(f"Baseline accuracy metric: {baseline_accuracy_metric}")
    if model_order:
        print(f"Custom model order: {model_order}")

    # Plot metrics
    plot_metrics(
        metrics_file,
        plots_output_dir,
        plot_format,
        skip_scib,
        skip_batch,
        vertical,
        font_size,
        tick_label_size,
        fig_width,
        fig_height,
        accuracy_metrics,
        baseline_accuracy_metric,
        group_repetitions,
        show_error_bars,
        axis_padding,
        label_pad,
        tick_label_pad,
        y_axis_max,
        accuracy_over_random_y_max,
        model_order,
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
    parser.add_argument(
        "--tick-label-size", type=int, default=None, help="Font size for tick labels (default: font_size - 1)"
    )
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
        args.tick_label_size,
        args.fig_width,
        args.fig_height,
        None,  # accuracy_metrics
        None,  # baseline_accuracy_metric
        True,  # group_repetitions
        True,  # show_error_bars
        None,  # axis_padding (default)
        None,  # label_pad (default)
        None,  # tick_label_pad (default)
        None,  # y_axis_max (default)
        None,  # accuracy_over_random_y_max (default)
        None,  # model_order (not supported in CLI mode)
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
