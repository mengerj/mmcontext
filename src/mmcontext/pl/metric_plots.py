import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def create_model_color_mapping(df, palette="husl"):
    """
    Create a consistent color mapping for models across plots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'model_short_name' column
    palette : str, optional
        Name of the seaborn color palette to use. Defaults to "husl".

    Returns
    -------
    dict
        Mapping of model names to colors
    """
    unique_models = sorted(df["model_short_name"].unique())
    colors = sns.color_palette(palette, n_colors=len(unique_models))
    return dict(zip(unique_models, colors, strict=False))


def plot_evaluation_results(
    csv_path,
    allowed_scib_metrics=None,
    # Plot appearance & saving
    figsize: tuple = (14, 8),
    dpi: int = 300,
    font_scale: float = 1.0,
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    legend_fontsize: int = 10,
    font_weight: str = "bold",
    save_plot: bool = False,
    save_dir: str | None = None,
    nametag: str | None = None,
    save_format: str = "png",
    color_mapping: dict | None = None,
):
    """
    Plot evaluation results as horizontal bar charts, side by side, sharing y-axis.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the evaluation results. Must have
        columns: ['dataset_name', 'metric_name', 'model_short_name', 'metric_value'].
    allowed_scib_metrics : set or list, optional
        If not None, only the scib metrics in this set/list are kept; all other
        'scib_' metrics are excluded. Non-scib metrics are always kept.
    figsize : tuple, optional
        Size of the figure (width, height). Defaults to (14, 8).
    dpi : int, optional
        Dots per inch for the figure. Defaults to 300.
    font_scale : float, optional
        Scale factor for all fonts. Defaults to 1.0.
    title_fontsize : int, optional
        Font size for the title. Defaults to 12.
    label_fontsize : int, optional
        Font size for axis labels. Defaults to 10.
    legend_fontsize : int, optional
        Font size for the legend. Defaults to 10.
    font_weight : str, optional
        Weight for the font used in the plot. Defaults to "bold".
    save_plot : bool, optional
        Whether to save the plot instead of displaying it. Defaults to False.
    save_dir : str or None, optional
        Directory where to save the plot. Required if save_plot=True.
    nametag : str or None, optional
        Additional tag to include in the saved filename. Defaults to None.
    save_format : str, optional
        Format for saving the figure (e.g., "png", "pdf"). Defaults to "png".
    color_mapping : dict, optional
        Mapping of model names to colors. If provided, these colors will be used
        for the models instead of generating new ones.

    Notes
    -----
    - Produces two subplots, side by side, one per dataset. Both subplots share
      the y-axis, meaning they have the same set and order of metrics listed.
    - Removes underscores and any leading 'scib_' from the metric names for
      more readable labels.
    - Only one legend is placed at the figure level, on the right side.
    """
    logger.info("Reading data from CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    # Create color mapping if not provided
    if color_mapping is None:
        color_mapping = create_model_color_mapping(df)

    # Optionally filter scib metrics
    if allowed_scib_metrics is not None:
        logger.debug("Filtering 'scib_' metrics not in %s", allowed_scib_metrics)
        df = df[~df["metric_name"].str.startswith("scib_") | df["metric_name"].isin(allowed_scib_metrics)]

    # Remove underscores and 'scib_' for display
    df["display_metric"] = df["metric_name"].str.replace("scib_", "", regex=False).str.replace("_", " ", regex=False)

    # Collect all metrics that appear so we have a consistent ordering
    all_metrics = df["display_metric"].unique().tolist()

    # Identify the datasets
    datasets = df["dataset_name"].unique()
    n_datasets = len(datasets)
    logger.info(f"Found {n_datasets} datasets: {datasets}")

    # Calculate number of rows and columns for subplots
    n_cols = min(3, n_datasets)  # Maximum 3 plots per row
    n_rows = (n_datasets + n_cols - 1) // n_cols  # Ceiling division

    # Adjust figsize based on number of subplots
    base_width = figsize[0] / 2  # Width per subplot from original figsize
    base_height = figsize[1]  # Height from original figsize
    adjusted_figsize = (base_width * n_cols, base_height * n_rows)

    # Set font scale using seaborn context
    with sns.plotting_context("notebook", font_scale=font_scale):
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=adjusted_figsize, sharey=True, dpi=dpi)

        # Convert axes to 1D array for easier iteration
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = np.ravel(axes)

        handles, labels = None, None

        for idx, (dataset, ax) in enumerate(zip(datasets, axes, strict=False)):
            logger.info(f"Plotting metrics for dataset: {dataset}")
            dataset_df = df[df["dataset_name"] == dataset]

            plot_df = pd.DataFrame(
                {
                    "Value": dataset_df["metric_value"],
                    "Display Metric": dataset_df["display_metric"],
                    "Model": dataset_df["model_short_name"],
                }
            )

            sns.barplot(
                data=plot_df,
                x="Value",
                y="Display Metric",
                hue="Model",
                order=all_metrics,
                ax=ax,
                palette=color_mapping,
            )

            ax.set_title(f"{dataset}", fontsize=title_fontsize, fontweight=font_weight)
            ax.set_xlabel("Score", fontsize=label_fontsize, fontweight=font_weight)
            # Only leftmost plots in each row get y-axis label
            ax.set_ylabel(
                "Individual metrics" if idx % n_cols == 0 else "", fontsize=label_fontsize, fontweight=font_weight
            )

            # Adjust tick label sizes
            ax.tick_params(axis="both", labelsize=label_fontsize)

            if ax.get_legend():
                handles, labels = ax.get_legend_handles_labels()
                ax.legend().remove()

        # Hide any unused subplots
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)

        # Create one figure-level legend on the right
        if handles and labels:
            fig.legend(
                handles,
                labels,
                title="Model",
                loc="center right",
                bbox_to_anchor=(1.15, 0.5),
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )

        # Adjust layout to leave space for legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Handle saving
        if save_plot:
            if not save_dir:
                raise ValueError("save_dir must be provided when save_plot=True")

            os.makedirs(save_dir, exist_ok=True)

            # Build filename
            base_name = "evaluation_results"
            if nametag:
                base_name += f"_{nametag}"
            save_path = os.path.join(save_dir, f"{base_name}.{save_format}")

            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
            logger.info("Displayed evaluation results plot interactively.")


def create_model_rank_table(csv_path, ascending=False):
    """
    Compute the mean rank of each model within each dataset, across all metrics.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing columns:
        ['model_name', 'model_short_name', 'dataset_name', 'output_dir',
         'timestamp', 'metric_name', 'metric_value'].
    ascending : bool, optional
        Whether lower metric_value is "better" (rank=1).
        By default, False, so higher metric_value is ranked as better.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
            * dataset_name
            * model_short_name
            * mean_rank
        and one row per (dataset, model). 'mean_rank' is rounded to two decimals.
        A smaller number = better rank.

    References
    ----------
    Data is read from the CSV file specified by 'csv_path'.
    Each row should represent one (model, dataset, metric) evaluation.

    Notes
    -----
    - If 'ascending=False' (default), higher metric_value -> better rank=1.
      If 'ascending=True', lower metric_value -> better rank=1.
    - Ranks are computed within each (dataset_name, metric_name) group.
    - If you want a different ranking method (e.g., "min", "max"), you can
      change 'method="dense"' to another strategy (like "average") in Pandas'
      rank call below.
    """
    logger.info("Reading CSV from %s", csv_path)
    df = pd.read_csv(csv_path)

    # 1) Rank within each (dataset_name, metric_name)
    #    Dense rank means ties get the same rank and the next rank is incremented by 1
    logger.info("Computing ranks (higher is better if ascending=False)")
    df["rank"] = df.groupby(["dataset_name", "metric_name"])["metric_value"].rank(method="dense", ascending=ascending)

    # 2) Compute mean rank for each (dataset, model_short_name)
    logger.info("Computing average of ranks across metrics per dataset+model")
    rank_summary = df.groupby(["dataset_name", "model_short_name"])["rank"].mean().reset_index()
    # rename the newly created column
    rank_summary.rename(columns={"rank": "mean_rank"}, inplace=True)

    # 3) Round ranks to 2 decimals
    rank_summary["mean_rank"] = rank_summary["mean_rank"].round(2)

    # 4) Sort by dataset_name and then mean_rank (so best rank=1 is first)
    rank_summary.sort_values(["dataset_name", "mean_rank"], inplace=True)
    rank_summary.reset_index(drop=True, inplace=True)

    logger.info("Returning the final rank table as a DataFrame")
    return rank_summary


def visualize_mean_ranks_line(
    rank_df,
    # Plot appearance & saving
    figsize: tuple = (8, 6),
    dpi: int = 300,
    font_scale: float = 1.0,
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    legend_fontsize: int = 10,
    font_weight: str = "bold",
    marker_size: float = 100,
    line_width: float = 2,
    save_plot: bool = False,
    save_dir: str | None = None,
    nametag: str | None = None,
    save_format: str = "png",
    color_mapping: dict | None = None,
):
    """
    Plot each dataset on the x-axis and the mean rank on the y-axis.

    Each line/color represents a different model, showing performance across datasets.

    Parameters
    ----------
    rank_df : pd.DataFrame
        Must have columns:
          * 'dataset_name' : str  (categorical on x-axis)
          * 'model_short_name' : str  (each line/color)
          * 'mean_rank' : float  (lower is better)
    figsize : tuple, optional
        Size of the figure (width, height). Defaults to (8, 6).
    dpi : int, optional
        Dots per inch for the figure. Defaults to 300.
    font_scale : float, optional
        Scale factor for all fonts. Defaults to 1.0.
    title_fontsize : int, optional
        Font size for the title. Defaults to 12.
    label_fontsize : int, optional
        Font size for axis labels. Defaults to 10.
    legend_fontsize : int, optional
        Font size for the legend. Defaults to 10.
    font_weight : str, optional
        Weight for the font used in the plot. Defaults to "bold".
    marker_size : float, optional
        Size of the markers in the plot. Defaults to 100.
    line_width : float, optional
        Width of the lines connecting points. Defaults to 2.
    save_plot : bool, optional
        Whether to save the plot instead of displaying it. Defaults to False.
    save_dir : str or None, optional
        Directory where to save the plot. Required if save_plot=True.
    nametag : str or None, optional
        Additional tag to include in the saved filename. Defaults to None.
    save_format : str, optional
        Format for saving the figure (e.g., "png", "pdf"). Defaults to "png".
    color_mapping : dict, optional
        Mapping of model names to colors. If provided, these colors will be used
        for the models instead of generating new ones.

    Notes
    -----
    - The y-axis is inverted so rank=1 is at the top.
    - Y-ticks are set to integer rank positions [#models ... 1].
    - Uses Seaborn's pointplot with lines connecting the same model across datasets.
    """
    # Count how many distinct models we have
    n_models = rank_df["model_short_name"].nunique()

    # Create color mapping if not provided
    if color_mapping is None:
        color_mapping = create_model_color_mapping(rank_df)

    # Set font scale using seaborn context
    with sns.plotting_context("notebook", font_scale=font_scale):
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Plot lines for each model across datasets
        sns.pointplot(
            data=rank_df,
            x="dataset_name",
            y="mean_rank",
            hue="model_short_name",
            markers=True,
            linestyles="-",
            errorbar=None,
            ax=ax,
            scale=marker_size / 100,  # Scale marker size
            linewidth=line_width,
            palette=color_mapping,
        )

        # Invert y-axis so rank=1 is at top
        ax.invert_yaxis()

        # Set integer y-ticks
        yticks = list(range(1, n_models + 1))[::-1]
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(t) for t in yticks], fontsize=label_fontsize)
        ax.set_ylim(n_models + 0.5, 0.5)

        # Labels and title
        ax.set_xlabel("Dataset", fontsize=label_fontsize, fontweight=font_weight)
        ax.set_ylabel("Mean rank", fontsize=label_fontsize, fontweight=font_weight)
        ax.set_title("Model performance across datasets", fontsize=title_fontsize, fontweight=font_weight)

        # Adjust x-tick labels
        plt.xticks(rotation=45, ha="right", fontsize=label_fontsize)

        # Move legend to the right
        ax.legend(
            title="Model", loc=None, bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize, title_fontsize=legend_fontsize
        )

        plt.tight_layout()

        # Handle saving
        if save_plot:
            if not save_dir:
                raise ValueError("save_dir must be provided when save_plot=True")

            os.makedirs(save_dir, exist_ok=True)

            # Build filename
            base_name = "mean_ranks_line"
            if nametag:
                base_name += f"_{nametag}"
            save_path = os.path.join(save_dir, f"{base_name}.{save_format}")

            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
            logger.info("Displayed mean ranks line plot interactively.")


def plot_evaluation_summary(
    csv_path,
    allowed_scib_metrics=None,
    # Plot appearance & saving
    eval_results_figsize: tuple = (14, 8),
    mean_ranks_figsize: tuple = (8, 6),
    dpi: int = 300,
    font_scale: float = 1.0,
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    legend_fontsize: int = 10,
    font_weight: str = "bold",
    marker_size: float = 100,
    line_width: float = 2,
    save_plot: bool = False,
    save_dir: str | None = None,
    nametag: str | None = None,
    save_format: str = "png",
    color_palette: str = "husl",
):
    """
    Create both evaluation results and mean ranks plots with consistent colors.

    This function reads the data once and ensures that both plots use the same
    colors for the same models.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the evaluation results. Must have
        columns: ['dataset_name', 'metric_name', 'model_short_name', 'metric_value'].
    allowed_scib_metrics : set or list, optional
        If not None, only the scib metrics in this set/list are kept; all other
        'scib_' metrics are excluded. Non-scib metrics are always kept.
    eval_results_figsize : tuple, optional
        Size of the evaluation results plot. Defaults to (14, 8).
    mean_ranks_figsize : tuple, optional
        Size of the mean ranks plot. Defaults to (8, 6).
    dpi : int, optional
        Dots per inch for the figure. Defaults to 300.
    font_scale : float, optional
        Scale factor for all fonts. Defaults to 1.0.
    title_fontsize : int, optional
        Font size for the title. Defaults to 12.
    label_fontsize : int, optional
        Font size for axis labels. Defaults to 10.
    legend_fontsize : int, optional
        Font size for the legend. Defaults to 10.
    font_weight : str, optional
        Weight for the font used in the plot. Defaults to "bold".
    marker_size : float, optional
        Size of the markers in the mean ranks plot. Defaults to 100.
    line_width : float, optional
        Width of the lines in the mean ranks plot. Defaults to 2.
    save_plot : bool, optional
        Whether to save the plot instead of displaying it. Defaults to False.
    save_dir : str or None, optional
        Directory where to save the plot. Required if save_plot=True.
    nametag : str or None, optional
        Additional tag to include in the saved filename. Defaults to None.
    save_format : str, optional
        Format for saving the figure (e.g., "png", "pdf"). Defaults to "png".
    color_palette : str, optional
        Name of the seaborn color palette to use. Defaults to "husl".

    Returns
    -------
    None
        Displays or saves both plots with consistent colors.
    """
    # Read data once
    logger.info("Reading data from CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    # Create consistent color mapping for both plots
    color_mapping = create_model_color_mapping(df, palette=color_palette)

    # Create evaluation results plot
    plot_evaluation_results(
        csv_path=csv_path,
        allowed_scib_metrics=allowed_scib_metrics,
        figsize=eval_results_figsize,
        dpi=dpi,
        font_scale=font_scale,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        legend_fontsize=legend_fontsize,
        font_weight=font_weight,
        save_plot=save_plot,
        save_dir=save_dir,
        nametag=f"{nametag}_eval_results" if nametag else "eval_results",
        save_format=save_format,
        color_mapping=color_mapping,
    )

    # Create rank table
    rank_df = create_model_rank_table(csv_path)

    # Create mean ranks line plot
    visualize_mean_ranks_line(
        rank_df=rank_df,
        figsize=mean_ranks_figsize,
        dpi=dpi,
        font_scale=font_scale,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        legend_fontsize=legend_fontsize,
        font_weight=font_weight,
        marker_size=marker_size,
        line_width=line_width,
        save_plot=save_plot,
        save_dir=save_dir,
        nametag=f"{nametag}_mean_ranks" if nametag else "mean_ranks",
        save_format=save_format,
        color_mapping=color_mapping,
    )
