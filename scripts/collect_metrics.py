#!/usr/bin/env python3
"""Script to collect and fuse metrics from evaluation results."""

import shutil
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf


def load_metrics_file(file_path: Path) -> pd.DataFrame:
    """Load metrics file, return empty DataFrame if file doesn't exist."""
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()


def process_regular_metrics(
    df: pd.DataFrame, dataset_name: str, model_id: str, model_name: str, selected_metrics: list[str]
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
                "model_name": model_name,
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
    model_name: str,
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
            results.append(
                {
                    "dataset": dataset_name,
                    "model": current_type,  # Use the type as the model source
                    "model_name": current_type,  # Use the type as the model name too
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
                    "model_name": model_name,
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
            # Use displayed_name if available, otherwise fallback to name, then source
            model_display_name = model_cfg.get("displayed_name", None)
            if model_display_name is None:
                model_display_name = model_cfg.get("name", model_cfg.source)

            # Get the name for directory path construction (use 'name' or 'source', not displayed_name)
            model_name_for_path = model_cfg.get("name", model_cfg.source)
            text_only = model_cfg.get("text_only", False)

            # Use model name (not displayed_name) for directory path construction
            model_dir_name = model_name_for_path
            if text_only:
                model_dir_name = model_dir_name + "_text_only"
                model_id = model_id + "_text_only"  # Keep for backward compatibility in results

            # Construct path to evaluation results using model name
            model_dir = eval_root / dataset_name / model_dir_name

            if not model_dir.exists():
                print(f"Warning: Model directory not found: {model_dir}")
                continue

            print(
                f"Processing: {dataset_name} / {model_id} (displayed_name: {model_display_name}, path_name: {model_name_for_path})"
            )

            # Load regular metrics
            regular_metrics_file = model_dir / "eval" / "metrics.csv"
            regular_df = load_metrics_file(regular_metrics_file)

            # Process regular metrics (use displayed_name for model_name column)
            regular_results = process_regular_metrics(
                regular_df, dataset_name, model_id, model_display_name, cfg.collect_metrics.regular_metrics
            )
            all_results.extend(regular_results)

            # Load scib metrics
            scib_metrics_file = model_dir / "eval" / "scib_metrics.csv"
            scib_df = load_metrics_file(scib_metrics_file)

            # Process scib metrics (use displayed_name for model_name column)
            scib_results = process_scib_metrics(
                scib_df,
                dataset_name,
                model_id,
                model_display_name,
                cfg.collect_metrics.scib_metrics,
                cfg.collect_metrics.scib_types_as_models,
                cfg.collect_metrics.model_embedding_type,
            )
            all_results.extend(scib_results)

    # Convert to DataFrame
    if all_results:
        return pd.DataFrame(all_results)
    else:
        return pd.DataFrame(columns=["dataset", "model", "model_name", "label", "label_kind", "metric", "value"])


# Model name cleaning functions removed - using model_name from config instead


def clean_metric_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up metric names by removing evaluator prefixes."""
    df = df.copy()

    # Remove prefixes like "scib/" and "LabelSimilarity/"
    df["metric"] = df["metric"].str.replace("scib/", "", regex=False)
    df["metric"] = df["metric"].str.replace("LabelSimilarity/", "", regex=False)

    return df


# Plotting functions have been moved to scripts/plot_metrics.py


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

    # Note: Use scripts/plot_metrics.py to create visualizations from the saved CSV

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
    print(f"Models: {metrics_df['model_name'].nunique()}")
    print(f"Metrics: {metrics_df['metric'].nunique()}")
    print(f"Unique model names: {sorted(metrics_df['model_name'].unique())}")
    print(f"Unique model sources: {sorted(metrics_df['model'].unique())}")


if __name__ == "__main__":
    main()
