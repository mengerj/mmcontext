#!/usr/bin/env python3
"""Script to collect and fuse metrics from evaluation results."""

import math
import re
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


def _resolve_model_order(model_order: list[str], cfg: DictConfig) -> list[str]:
    """Resolve model_order entries from config ``name`` to ``displayed_name``.

    Users may specify the order using either the internal ``name`` or the
    ``displayed_name``.  This builds a name -> displayed_name map from
    ``cfg.models`` and replaces any matching entries so the returned list
    uses the names that actually appear in the data.
    """
    if not model_order:
        return model_order

    name_to_display: dict[str, str] = {}
    for m in cfg.get("models", []):
        name = m.get("name", m.get("source"))
        display = m.get("displayed_name", name)
        if name and display:
            name_to_display[name] = display

    return [name_to_display.get(entry, entry) for entry in model_order]


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _format_cell(value: float, decimals: int, is_best: bool, bold_best: bool) -> str:
    """Format a single table cell value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    formatted = f"{value:.{decimals}f}"
    if is_best and bold_best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def _render_latex_table(
    pivot: pd.DataFrame,
    metric_display: dict[str, str],
    dataset: str,
    label: str,
    label_kind: str,
    decimals: int,
    bold_best: bool,
) -> str:
    """Render a single (dataset, label) pivot table as a LaTeX table string."""
    metrics = list(pivot.columns)
    n_metrics = len(metrics)
    col_spec = "l" + "c" * n_metrics

    col_maxes: dict[str, float] = {}
    for m in metrics:
        numeric = pd.to_numeric(pivot[m], errors="coerce")
        if numeric.notna().any():
            col_maxes[m] = numeric.max()

    header_names = [metric_display.get(m, _latex_escape(m)) for m in metrics]
    header_line = "Model & " + " & ".join(header_names) + " \\\\"

    body_lines: list[str] = []
    for model_name, row in pivot.iterrows():
        cells: list[str] = [_latex_escape(str(model_name))]
        for m in metrics:
            val = row[m]
            numeric_val = float(val) if pd.notna(val) else float("nan")
            is_best = not math.isnan(numeric_val) and m in col_maxes and abs(numeric_val - col_maxes[m]) < 1e-12
            cells.append(_format_cell(numeric_val, decimals, is_best, bold_best))
        body_lines.append(" & ".join(cells) + " \\\\")

    safe_dataset = _latex_escape(dataset)
    safe_label_name = _latex_escape(label)
    safe_kind = _latex_escape(label_kind)
    caption = (
        f"Evaluation results on the \\textit{{{safe_dataset}}} dataset "
        f"for the \\textit{{{safe_label_name}}} label ({safe_kind})."
    )
    tab_label = re.sub(r"[^a-zA-Z0-9_]", "_", f"{dataset}_{label}_{label_kind}")

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{tab_label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header_line,
        "\\midrule",
        *body_lines,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def generate_latex_tables(df: pd.DataFrame, cfg: DictConfig) -> None:
    r"""Generate LaTeX tables from the collected metrics DataFrame.

    Produces one .tex file per (dataset, label, label_kind) group and a
    combined ``all_tables.tex`` that ``\input``-s every individual file.
    """
    lt_cfg = cfg.collect_metrics.latex_tables
    output_dir = Path(cfg.collect_metrics.output_dir) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_metrics: list[str] = list(lt_cfg.metrics)
    metric_display: dict[str, str] = dict(lt_cfg.metric_display_names) if lt_cfg.metric_display_names else {}
    decimals: int = int(lt_cfg.decimal_places)
    bold_best: bool = bool(lt_cfg.bold_best)
    label_kind_filter: str | None = lt_cfg.get("label_kind_filter", None)
    model_order: list[str] = list(cfg.collect_metrics.model_order) if cfg.collect_metrics.get("model_order") else []
    model_order = _resolve_model_order(model_order, cfg)

    work = df[df["metric"].isin(selected_metrics)].copy()
    if label_kind_filter:
        work = work[work["label_kind"] == label_kind_filter]

    if work.empty:
        print("LaTeX tables: no data after filtering – nothing to write.")
        return

    tex_files: list[str] = []

    for (dataset, label, label_kind), grp in work.groupby(["dataset", "label", "label_kind"]):
        pivot = grp.pivot_table(index="model_name", columns="metric", values="value", aggfunc="first")
        pivot = pivot.reindex(columns=[m for m in selected_metrics if m in pivot.columns])

        ordered_models: list[str] = []
        remaining = set(pivot.index)
        for m in model_order:
            if m in remaining:
                ordered_models.append(m)
                remaining.discard(m)
        ordered_models.extend(sorted(remaining))
        pivot = pivot.reindex(ordered_models)

        tex = _render_latex_table(pivot, metric_display, dataset, label, label_kind, decimals, bold_best)
        fname = f"{dataset}_{label}_{label_kind}.tex"
        (output_dir / fname).write_text(tex)
        tex_files.append(fname)
        print(f"  Wrote {output_dir / fname}")

    combined = "\n".join(f"\\input{{tables/{f}}}" for f in sorted(tex_files)) + "\n"
    combined_path = output_dir / "all_tables.tex"
    combined_path.write_text(combined)
    print(f"  Wrote combined file: {combined_path}")


# Plotting functions have been moved to scripts/plot_metrics.py


@hydra.main(version_base=None, config_path="../conf/eval", config_name="collect_metrics_conf")
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

    # Generate LaTeX tables if enabled
    if cfg.collect_metrics.get("latex_tables", {}).get("enabled", False):
        print("\nGenerating LaTeX tables...")
        generate_latex_tables(metrics_df, cfg)

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
