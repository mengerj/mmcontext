#!/usr/bin/env python
"""
Funky Heatmap Generator from Evaluation Results

This script loads evaluation results and creates a clean funky heatmap visualization,
with one row per model-dataset combination and using short names for models and datasets.

Usage:
    python funky_heatmap_from_results.py --results_path PATH_TO_RESULTS_CSV [--output OUTPUT_FILE]

Author: Claude
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_evaluation_results(results_path: str) -> pd.DataFrame:
    """
    Load evaluation results from CSV file.

    Parameters
    ----------
    results_path : str
        Path to the evaluation results CSV file

    Returns
    -------
    pd.DataFrame
        Loaded evaluation results
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results_df = pd.read_csv(results_path)
    logger.info(f"Loaded {len(results_df)} evaluation results from {results_path}")
    return results_df


def transform_to_funkyheatmap_format(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transform evaluation results to a format suitable for funkyheatmap.

    Parameters
    ----------
    results_df : pd.DataFrame
        Evaluation results in long format

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple containing:
        - data: The main data for the heatmap
        - column_info: Information about the columns
        - column_groups: Information about column grouping
    """
    # Create a unique model identifier and short name
    if "model_short_name" not in results_df.columns:
        # Extract short name from model_name (last part after '/')
        results_df["model_short_name"] = results_df["model_name"].apply(lambda x: x.split("/")[-1])

    # Create dataset short names if not already present
    if "dataset_short_name" not in results_df.columns:
        # Use the dataset name as short name
        results_df["dataset_short_name"] = results_df["dataset_name"]

    # Create a unique ID for each model-dataset combination
    results_df["id"] = results_df["model_short_name"] + "_" + results_df["dataset_short_name"]

    # Define metric groups for better organization
    bio_metrics = ["ARI", "NMI", "ASW", "Isolated_Labels_ASW", "Isolated_Labels_F1", "cLISI"]
    batch_metrics = ["Graph_Connectivity", "Silhouette_Batch", "iLISI", "PCR"]
    alignment_metrics = ["modality_gap_irrelevant_score", "full_comparison_score"]
    annotation_metrics = ["annotation_accuracy", "zero_shot_macro_auc"]

    # Create a clean metric name without the 'scib_' prefix
    results_df["clean_metric_name"] = results_df["metric_name"].apply(lambda x: x[5:] if x.startswith("scib_") else x)

    # Create a wide format dataframe with one row per model-dataset combination
    # First, get unique model-dataset combinations
    unique_combinations = results_df[
        ["id", "model_name", "model_short_name", "dataset_name", "dataset_short_name"]
    ].drop_duplicates()

    # Create an empty dataframe with these combinations
    wide_df = unique_combinations.copy()

    # Add metrics as columns
    for _, row in results_df.iterrows():
        metric_name = row["clean_metric_name"]
        metric_value = row["metric_value"]
        model_dataset_id = row["id"]

        # Add the metric as a column to the wide dataframe
        mask = wide_df["id"] == model_dataset_id
        wide_df.loc[mask, metric_name] = metric_value

    # Ensure 'id' is the first column
    cols = ["id", "model_short_name", "dataset_short_name"]
    cols += [col for col in wide_df.columns if col not in cols + ["model_name", "dataset_name"]]
    wide_df = wide_df[cols]

    # Create column_info DataFrame
    column_names = wide_df.columns.tolist()

    # Determine the group for each column
    column_groups_list = []
    for col in column_names:
        if col in ["id", "model_short_name", "dataset_short_name"]:
            group = "info"
        elif any(metric in col for metric in bio_metrics):
            group = "bio"
        elif any(metric in col for metric in batch_metrics):
            group = "batch"
        elif any(metric in col for metric in alignment_metrics):
            group = "alignment"
        elif any(metric in col for metric in annotation_metrics):
            group = "annotation"
        else:
            group = "other"

        column_groups_list.append(group)

    # Create column_info DataFrame
    column_info = pd.DataFrame(
        {
            "id": column_names,
            "name": [col.upper() if col == "id" else col.replace("_", " ").title() for col in column_names],
            "geom": [
                "text" if col in ["id", "model_short_name", "dataset_short_name"] else "bar" for col in column_names
            ],
            "group": column_groups_list,
            "palette": [
                "black"
                if col in ["id", "model_short_name", "dataset_short_name"]
                else "Greens"
                if any(metric in col for metric in bio_metrics)
                else "Blues"
                if any(metric in col for metric in batch_metrics)
                else "Reds"
                if any(metric in col for metric in alignment_metrics)
                else "YlOrBr"
                if any(metric in col for metric in annotation_metrics)
                else "Purples"
                for col in column_names
            ],
            "width": [
                3 if col == "id" else 2 if col in ["model_short_name", "dataset_short_name"] else 1
                for col in column_names
            ],
            "legend": [
                False if col in ["id", "model_short_name", "dataset_short_name"] else True for col in column_names
            ],
        }
    )

    # Create column_groups DataFrame
    unique_groups = sorted(set(column_groups_list))

    group_info = []
    for group in unique_groups:
        if group == "info":
            level1 = "Info"
        elif group == "bio":
            level1 = "Biological"
        elif group == "batch":
            level1 = "Batch"
        elif group == "alignment":
            level1 = "Alignment"
        elif group == "annotation":
            level1 = "Annotation"
        else:
            level1 = "Other"

        group_info.append({"group": group, "level1": level1})

    column_groups_df = pd.DataFrame(group_info)

    return wide_df, column_info, column_groups_df


def create_funky_heatmap(
    data: pd.DataFrame,
    column_info: pd.DataFrame,
    column_groups: pd.DataFrame,
    output_file: str | None = None,
    width: int = 20,
    height: int = 12,
    dpi: int = 300,
) -> None:
    """
    Create a funky heatmap visualization.

    Parameters
    ----------
    data : pd.DataFrame
        The main data for the heatmap
    column_info : pd.DataFrame
        Information about the columns
    column_groups : pd.DataFrame
        Information about column grouping
    output_file : str, optional
        Path to save the heatmap. If None, the plot is displayed
    width : int, optional
        Width of the plot in inches (default: 20)
    height : int, optional
        Height of the plot in inches (default: 12)
    dpi : int, optional
        Resolution of the output figure in dots per inch (default: 300)
    """
    try:
        # Import the funkyheatmap package
        from funkyheatmappy import funky_heatmap
    except ImportError as e:
        logger.error("funkyheatmappy package not installed. Please install it with: pip install funkyheatmappy")
        raise ImportError(
            "funkyheatmappy package not installed. Please install it with: pip install funkyheatmappy"
        ) from e

    logger.info("Creating funky heatmap...")

    # Set up position arguments for better spacing
    position_args = {
        "expand_xmin": 0.2,
        "expand_xmax": 0.2,
        "expand_ymin": 0.2,
        "expand_ymax": 0.2,
        "col_annot_offset": 0.5,
        "col_annot_angle": 45,
        "row_height": 0.8,  # Increase row height for better readability
        "row_space": 0.2,  # Add space between rows
        "row_bigspace": 0.5,  # Add more space between groups of rows
        "col_width": 0.8,  # Increase column width for better readability
        "col_space": 0.2,  # Add space between columns
        "col_bigspace": 0.5,  # Add more space between groups of columns
    }

    # Create the plot
    fig = funky_heatmap(
        data=data,
        column_info=column_info,
        column_groups=column_groups,
        scale_column=True,
        add_abc=False,  # Disable ABC labels
        position_args=position_args,
    )

    # Adjust figure size
    fig.set_size_inches(width, height)

    # Save or display the plot
    if output_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Save the figure
        fig.savefig(output_file, bbox_inches="tight", dpi=dpi)
        logger.info(f"Funky heatmap saved to: {output_file}")
    else:
        # Display the plot
        plt.show()
        logger.info("Funky heatmap displayed")


def main(results_path: str, output_file: str | None = None, width: int = 20, height: int = 12, dpi: int = 300) -> None:
    """
    Main function to create a funky heatmap from evaluation results.

    Parameters
    ----------
    results_path : str
        Path to the evaluation results CSV file
    output_file : str, optional
        Path to save the heatmap. If None, a default path is used
    width : int, optional
        Width of the plot in inches (default: 20)
    height : int, optional
        Height of the plot in inches (default: 12)
    dpi : int, optional
        Resolution of the output figure in dots per inch (default: 300)
    """
    # Load evaluation results
    results_df = load_evaluation_results(results_path)

    # Transform to funkyheatmap format
    data, column_info, column_groups = transform_to_funkyheatmap_format(results_df)

    # Set default output file if not provided
    if output_file is None:
        output_dir = os.path.join(os.path.dirname(results_path), "figures")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "evaluation_funky_heatmap.pdf")

    # Create the funky heatmap
    create_funky_heatmap(
        data=data,
        column_info=column_info,
        column_groups=column_groups,
        output_file=output_file,
        width=width,
        height=height,
        dpi=dpi,
    )

    # Save the data for reference
    data_output_path = os.path.splitext(output_file)[0] + "_data.csv"
    column_info_output_path = os.path.splitext(output_file)[0] + "_column_info.csv"
    column_groups_output_path = os.path.splitext(output_file)[0] + "_column_groups.csv"

    data.to_csv(data_output_path, index=False)
    column_info.to_csv(column_info_output_path, index=False)
    column_groups.to_csv(column_groups_output_path, index=False)

    logger.info(f"Data saved to: {data_output_path}")
    logger.info(f"Column info saved to: {column_info_output_path}")
    logger.info(f"Column groups saved to: {column_groups_output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a funky heatmap from evaluation results.")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the evaluation results CSV file")
    parser.add_argument("--output", type=str, help="Path to save the heatmap")
    parser.add_argument("--width", type=int, default=20, help="Width of the plot in inches (default: 20)")
    parser.add_argument("--height", type=int, default=12, help="Height of the plot in inches (default: 12)")
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution of the output figure in dots per inch (default: 300)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(results_path=args.results_path, output_file=args.output, width=args.width, height=args.height, dpi=args.dpi)
