import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_evaluation_results(csv_path):
    """
    Function to plot evaluation results from a CSV file.

    Creates barplots for each dataset with the different metrics and models.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create figure with two subplots side by side
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for each dataset
    datasets = df["dataset_short_name"].unique()
    axes = [ax1, ax2]

    for dataset, ax in zip(datasets, axes, strict=False):
        # Filter data for current dataset
        dataset_df = df[df["dataset_short_name"] == dataset]

        # Prepare data for plotting
        plot_data = []
        for _, row in dataset_df.iterrows():
            plot_data.append(
                {"Model": row["model_short_name"], "Metric": row["metric_name"], "Value": row["metric_value"]}
            )

        plot_df = pd.DataFrame(plot_data)

        # Create grouped bar plot
        sns.barplot(data=plot_df, x="Metric", y="Value", hue="Model", ax=ax)

        # Customize the plot
        ax.set_title(f"Metrics for {dataset}")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45)

        # Adjust legend
        ax.legend(title="Model")

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("evaluation_results.png", dpi=300, bbox_inches="tight")
    plt.close()
