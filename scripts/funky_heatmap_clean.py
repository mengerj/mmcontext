"""
Clean Funky Heatmap Generator with Simulated Evaluation Data

This script creates a clean funky heatmap using simulated evaluation data,
with one dataset per row and simplified column organization.

Usage:
    python funky_heatmap_clean.py [--output OUTPUT_FILE] [--models MODELS] [--datasets DATASETS]

Author: Claude
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def simulate_evaluation_data(n_models: int = 4, n_datasets: int = 3) -> pd.DataFrame:
    """
    Simulate evaluation data with models and datasets.

    Parameters
    ----------
    n_models : int, optional
        Number of models to simulate (default: 4)
    n_datasets : int, optional
        Number of datasets to simulate (default: 3)

    Returns
    -------
    pd.DataFrame
        Simulated evaluation data in long format
    """
    # Define model names
    model_names = [
        "mmcontext-model-v1",
        "mmcontext-model-v2",
        "mmcontext-large",
        "mmcontext-small",
        "baseline-model",
        "competitor-model",
    ][:n_models]

    # Define dataset names
    dataset_names = ["dataset1", "dataset2", "dataset3", "dataset4", "dataset5"][:n_datasets]

    # Define metrics
    metrics = [
        # Biological metrics
        "ARI",
        "NMI",
        "ASW",
        # Batch metrics
        "Graph_Connectivity",
        "Silhouette_Batch",
        "iLISI",
        # Alignment metrics
        "modality_gap_score",
        "comparison_score",
        # Annotation metrics
        "annotation_accuracy",
        "zero_shot_auc",
    ]

    # Create rows for the data
    rows = []

    for model_name in model_names:
        for dataset_name in dataset_names:
            # Create a unique ID for this model-dataset combination
            model_dataset_id = f"{model_name}_{dataset_name}"

            # Generate values for each metric
            metric_values = {}

            # Make some models perform better than others
            base_value = 0.5
            if "large" in model_name:
                base_value = 0.8
            elif "small" in model_name:
                base_value = 0.6
            elif "baseline" in model_name:
                base_value = 0.4

            # Generate values for each metric with some randomness
            np.random.seed(hash(model_dataset_id) % 2**32)  # Different seed for each combination
            for metric in metrics:
                # Add some randomness
                value = np.clip(base_value + np.random.normal(0, 0.1), 0, 1)
                metric_values[metric] = value

            # Create a row
            row = {"id": model_dataset_id, "model": model_name, "dataset": dataset_name, **metric_values}
            rows.append(row)

    # Create a DataFrame
    df = pd.DataFrame(rows)

    return df


def prepare_for_funkyheatmap(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare evaluation data for funkyheatmap.

    Parameters
    ----------
    data : pd.DataFrame
        Simulated evaluation data

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple containing:
        - data: The main data for the heatmap
        - column_info: Information about the columns
        - column_groups: Information about column grouping
    """
    # Ensure 'id' is the first column
    cols = ["id", "model", "dataset"] + [col for col in data.columns if col not in ["id", "model", "dataset"]]
    data = data[cols]

    # Define metric groups
    bio_metrics = ["ARI", "NMI", "ASW"]
    batch_metrics = ["Graph_Connectivity", "Silhouette_Batch", "iLISI"]
    alignment_metrics = ["modality_gap_score", "comparison_score"]
    annotation_metrics = ["annotation_accuracy", "zero_shot_auc"]

    # Create column_info DataFrame
    column_info = pd.DataFrame(
        {
            "id": data.columns.tolist(),
            "name": [col.upper() if col == "id" else col.replace("_", " ").title() for col in data.columns],
            "geom": ["text" if col in ["id", "model", "dataset"] else "bar" for col in data.columns],
            "group": [
                "info"
                if col in ["id", "model", "dataset"]
                else "bio"
                if col in bio_metrics
                else "batch"
                if col in batch_metrics
                else "alignment"
                if col in alignment_metrics
                else "annotation"
                if col in annotation_metrics
                else "other"
                for col in data.columns
            ],
            "palette": [
                "black"
                if col in ["id", "model", "dataset"]
                else "Greens"
                if col in bio_metrics
                else "Blues"
                if col in batch_metrics
                else "Reds"
                if col in alignment_metrics
                else "YlOrBr"
                if col in annotation_metrics
                else "Purples"
                for col in data.columns
            ],
            "width": [3 if col == "id" else 2 if col in ["model", "dataset"] else 1 for col in data.columns],
            "legend": [False if col in ["id", "model", "dataset"] else True for col in data.columns],
        }
    )

    # Create column_groups DataFrame
    unique_groups = sorted(column_info["group"])

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

    return data, column_info, column_groups_df


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


def main(
    output_file: str | None = None,
    n_models: int = 4,
    n_datasets: int = 3,
    width: int = 20,
    height: int = 12,
    dpi: int = 300,
) -> None:
    """
    Main function to create a funky heatmap with simulated evaluation data.

    Parameters
    ----------
    output_file : str, optional
        Path to save the heatmap. If None, a default path is used
    n_models : int, optional
        Number of models to simulate (default: 4)
    n_datasets : int, optional
        Number of datasets to simulate (default: 3)
    width : int, optional
        Width of the plot in inches (default: 20)
    height : int, optional
        Height of the plot in inches (default: 12)
    dpi : int, optional
        Resolution of the output figure in dots per inch (default: 300)
    """
    # Simulate evaluation data
    logger.info(f"Simulating evaluation data with {n_models} models and {n_datasets} datasets...")
    data = simulate_evaluation_data(n_models=n_models, n_datasets=n_datasets)

    # Prepare data for funkyheatmap
    data, column_info, column_groups = prepare_for_funkyheatmap(data)

    # Set default output file if not provided
    if output_file is None:
        output_dir = "outputs/figures"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "clean_funky_heatmap.pdf")

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
    parser = argparse.ArgumentParser(description="Create a clean funky heatmap with simulated evaluation data.")
    parser.add_argument("--output", type=str, help="Path to save the heatmap")
    parser.add_argument("--models", type=int, default=4, help="Number of models to simulate (default: 4)")
    parser.add_argument("--datasets", type=int, default=3, help="Number of datasets to simulate (default: 3)")
    parser.add_argument("--width", type=int, default=20, help="Width of the plot in inches (default: 20)")
    parser.add_argument("--height", type=int, default=12, help="Height of the plot in inches (default: 12)")
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution of the output figure in dots per inch (default: 300)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        output_file=args.output,
        n_models=args.models,
        n_datasets=args.datasets,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
    )
