"""
SCSA (Single-Cell RNA-seq Annotation) utilities for integration with mmcontext pipeline.

SCSA is a marker-gene-based cell type annotation tool that works at the cluster level.
Unlike embedding models, it produces direct predictions by matching differentially
expressed genes against curated cell marker databases.

This module provides functions to run the full SCSA pipeline and compute evaluation
metrics compatible with the mmcontext metrics collection system.
"""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


@dataclass
class HarmonizeResult:
    """Outcome of the calmate label-harmonisation step."""

    success: bool
    all_reviewed: bool
    mapped_csv: Path | None = None
    messages: list[str] = field(default_factory=list)


def _reorder_harmonized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns for the harmonised predictions CSV.

    Desired order (when present):
    ``cell_id``, ``true_labels``, ``true_labels_mapped``,
    ``predicted_labels``, ``predicted_labels_mapped``, then any remaining
    columns in their original order.
    """
    preferred: list[str] = []
    if "cell_id" in df.columns:
        preferred.append("cell_id")
    for col in [
        "true_labels",
        "true_labels_mapped",
        "predicted_labels",
        "predicted_labels_mapped",
    ]:
        if col in df.columns and col not in preferred:
            preferred.append(col)
    remaining = [c for c in df.columns if c not in preferred]
    return df[preferred + remaining]


def ensure_scsa_setup(modules_dir: Path) -> bool:
    """
    Ensure SCSA is properly set up by checking for the repository and database.

    Runs prepare_scsa.sh if the SCSA directory is missing.

    Parameters
    ----------
    modules_dir : Path
        Directory containing (or to contain) the SCSA installation.

    Returns
    -------
    bool
        True if setup is complete, False otherwise.
    """
    scsa_dir = modules_dir / "SCSA"
    scsa_db = scsa_dir / "whole_v2.db"

    if scsa_dir.exists() and scsa_db.exists():
        logger.info("SCSA already set up at %s", scsa_dir)
        return True

    prepare_script = modules_dir / "prepare_scsa.sh"
    if not prepare_script.exists():
        logger.error("SCSA prepare script not found at %s", prepare_script)
        return False

    try:
        logger.info("Running SCSA prepare script...")
        result = subprocess.run(
            ["bash", str(prepare_script)],
            cwd=str(modules_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error("SCSA prepare script failed (rc=%d): %s", result.returncode, result.stderr)
            return False

        logger.info("SCSA prepare script completed successfully")
        if result.stdout:
            logger.info("Prepare output: %s", result.stdout.strip())

        return scsa_dir.exists() and scsa_db.exists()

    except subprocess.TimeoutExpired:
        logger.error("SCSA prepare script timed out")
        return False
    except Exception as e:
        logger.error("Error running SCSA prepare script: %s", e)
        return False


def create_marker_csv(
    adata: ad.AnnData,
    output_csv: Path,
    cluster_col: str = "louvain",
    resolution: float = 1.0,
    n_top_genes: int = 2000,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    random_state: int = 0,
) -> tuple[ad.AnnData, str]:
    """
    Cluster cells (if needed) and create a marker-gene CSV in SCSA's expected format.

    The marker detection is performed on normalized (not scaled) data so that
    log fold changes are valid for SCSA's filtering.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with expression data.
    output_csv : Path
        Path to write the marker CSV.
    cluster_col : str
        Name of the cluster column to use or create.
    resolution : float
        Resolution for Louvain clustering if clustering is needed.
    n_top_genes : int
        Number of highly variable genes for preprocessing.
    n_neighbors : int
        Number of neighbors for the neighborhood graph.
    n_pcs : int
        Number of principal components.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[AnnData, str]
        The (possibly modified) AnnData and the cluster column name used.
    """
    needs_clustering = cluster_col not in adata.obs.columns

    if "gene_names" in adata.var.columns:
        adata.var.index = adata.var["gene_names"]

    if needs_clustering:
        logger.info("No cluster column '%s' found — running preprocessing + Louvain clustering", cluster_col)

        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
        adata = adata[:, adata.var.highly_variable].copy()

        adata_for_markers = adata.copy()

        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver="arpack", n_comps=n_pcs)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)
        sc.tl.louvain(adata, resolution=resolution, key_added=cluster_col, random_state=random_state)

        adata_for_markers.obs[cluster_col] = adata.obs[cluster_col]
    else:
        logger.info("Using existing cluster column '%s'", cluster_col)
        adata_for_markers = adata.copy()
        if "log1p" in adata.layers:
            adata_for_markers.X = adata.layers["log1p"].copy()
        elif adata.raw is not None:
            adata_for_markers = adata.raw.to_adata()
            adata_for_markers.obs[cluster_col] = adata.obs[cluster_col]

    n_clusters = len(adata.obs[cluster_col].unique())
    logger.info("Found %d clusters", n_clusters)

    logger.info("Computing marker genes per cluster (on normalized data)...")
    sc.tl.rank_genes_groups(adata_for_markers, groupby=cluster_col, method="t-test")

    result = adata_for_markers.uns["rank_genes_groups"]
    groups = result["names"].dtype.names

    dat = pd.DataFrame(
        {
            group + "_" + key[:1]: result[key][group]
            for group in groups
            for key in ["names", "logfoldchanges", "scores", "pvals"]
        }
    )

    dat.to_csv(output_csv, index=True)
    logger.info("Wrote marker CSV with %d genes and %d columns to %s", len(dat), len(dat.columns), output_csv)

    return adata, cluster_col


def run_scsa(
    csv_path: Path,
    db_path: Path,
    output_path: Path,
    scsa_dir: Path,
    *,
    scsa_python: str | Path | None = None,
    foldchange: float = 1.5,
    pvalue: float = 0.01,
    species: str = "Human",
    tissue: str = "All",
    gensymbol: bool = True,
) -> Path:
    """
    Run SCSA.py on a marker-gene CSV.

    SCSA requires numpy<2.0 and must be run with a dedicated Python
    interpreter from a separate virtual environment.

    Parameters
    ----------
    csv_path : Path
        Path to marker CSV produced by :func:`create_marker_csv`.
    db_path : Path
        Path to the SCSA database file.
    output_path : Path
        Path to write SCSA results.
    scsa_dir : Path
        Directory containing SCSA.py.
    scsa_python : str or Path or None
        Python executable from the SCSA venv.  When *None* the function
        looks for ``modules/scsa_venv/bin/python`` relative to *scsa_dir*'s
        parent, and falls back to ``sys.executable`` with a warning.
    foldchange : float
        Minimum log fold-change threshold.
    pvalue : float
        Maximum p-value threshold.
    species : str
        Species for database lookup.
    tissue : str
        Tissue type for database lookup.
    gensymbol : bool
        If True, use gene symbols instead of Ensembl IDs.

    Returns
    -------
    Path
        Path to the SCSA output file.
    """
    if scsa_python is None:
        default_venv_python = scsa_dir.parent / "scsa_venv" / "bin" / "python"
        if default_venv_python.exists():
            scsa_python = str(default_venv_python)
        else:
            logger.warning(
                "SCSA venv not found at %s — falling back to current interpreter. "
                "This will fail if numpy>=2.0 is installed. "
                "Create the venv with: python -m venv modules/scsa_venv && "
                "modules/scsa_venv/bin/pip install -r modules/requirements_scsa.txt",
                default_venv_python,
            )
            scsa_python = sys.executable
    else:
        scsa_python = str(scsa_python)

    cmd = [
        scsa_python,
        str(scsa_dir / "SCSA.py"),
        "-d", str(db_path),
        "-i", str(csv_path),
        "-s", "scanpy",
        "-o", str(output_path),
        "-f", str(foldchange),
        "-p", str(pvalue),
        "-g", species,
        "-k", tissue,
        "-T", "normal",
        "-t", "cellmarker",
        "-m", "txt",
    ]

    if gensymbol:
        cmd.append("-E")

    logger.info("Running SCSA: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(scsa_dir))

    if result.returncode != 0:
        logger.error("SCSA failed (rc=%d)\nstdout: %s\nstderr: %s", result.returncode, result.stdout, result.stderr)
        raise RuntimeError(f"SCSA failed with return code {result.returncode}: {result.stderr}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        logger.warning("SCSA output is empty or missing at %s", output_path)
        if result.stdout:
            logger.info("SCSA stdout: %s", result.stdout[:1000])

    logger.info("SCSA completed, output at %s", output_path)
    return output_path


def map_predictions_to_cells(
    scsa_result_file: Path,
    adata: ad.AnnData,
    cluster_col: str,
    annotation_key: str,
) -> pd.DataFrame:
    """
    Map SCSA cluster-level predictions to individual cells.

    For each cluster, the cell type with the highest Z-score is selected.
    Clusters without a prediction are labelled "Unknown".

    Parameters
    ----------
    scsa_result_file : Path
        Path to SCSA tab-separated output file.
    adata : AnnData
        AnnData with cluster assignments and true labels.
    cluster_col : str
        Column in ``adata.obs`` with cluster IDs.
    annotation_key : str
        Column in ``adata.obs`` with true cell-type labels.

    Returns
    -------
    DataFrame
        Columns: ``cell_id``, ``true_labels``, ``predicted_labels``.
    """
    if not scsa_result_file.exists() or scsa_result_file.stat().st_size == 0:
        logger.warning("SCSA result file empty or missing: %s — all cells will be 'Unknown'", scsa_result_file)
        return pd.DataFrame(
            {
                "cell_id": adata.obs.index,
                "true_labels": adata.obs[annotation_key].values,
                "predicted_labels": "Unknown",
            }
        )

    scsa_results = pd.read_csv(scsa_result_file, sep="\t")
    expected_cols = {"Cell Type", "Z-score", "Cluster"}
    if not expected_cols.issubset(scsa_results.columns):
        raise ValueError(
            f"SCSA output missing expected columns. Found {list(scsa_results.columns)}, need {expected_cols}"
        )

    # Pick the top prediction (highest Z-score) per cluster
    cluster_predictions: dict[str, str] = {}
    for cluster in scsa_results["Cluster"].unique():
        cluster_data = scsa_results[scsa_results["Cluster"] == cluster].copy()
        cluster_data["Z-score"] = pd.to_numeric(cluster_data["Z-score"], errors="coerce")
        cluster_data = cluster_data.dropna(subset=["Z-score"])
        if len(cluster_data) > 0:
            best = cluster_data.loc[cluster_data["Z-score"].idxmax()]
            cluster_predictions[str(cluster)] = best["Cell Type"]

    logger.info("SCSA predictions for %d clusters", len(cluster_predictions))

    if annotation_key not in adata.obs.columns:
        raise ValueError(f"Annotation key '{annotation_key}' not found. Available: {list(adata.obs.columns)}")

    cell_data = []
    unmapped = set()
    for idx, cluster in enumerate(adata.obs[cluster_col]):
        cstr = str(cluster)
        pred = cluster_predictions.get(cstr)
        if pred is None:
            unmapped.add(cstr)
            pred = "Unknown"
        cell_data.append(
            {
                "cell_id": adata.obs.index[idx],
                "true_labels": adata.obs[annotation_key].iloc[idx],
                "predicted_labels": pred,
            }
        )

    if unmapped:
        logger.warning("%d clusters had no SCSA prediction: %s", len(unmapped), sorted(unmapped))

    return pd.DataFrame(cell_data)


def compute_metrics(
    true_labels: np.ndarray | pd.Series,
    predicted_labels: np.ndarray | pd.Series,
    dataset_name: str,
    annotation_key: str,
) -> list[dict[str, Any]]:
    """
    Compute evaluation metrics directly from predictions.

    Produces rows in the same schema as the LabelSimilarity evaluator so that
    ``collect_metrics.py`` can fuse them with other model results.

    Parameters
    ----------
    true_labels : array-like
        Ground truth labels.
    predicted_labels : array-like
        Predicted labels.
    dataset_name : str
        Name of the dataset.
    annotation_key : str
        Name of the label column.

    Returns
    -------
    list[dict]
        One dict per metric, each with keys:
        ``dataset``, ``model``, ``model_name``, ``label``, ``label_kind``,
        ``metric``, ``value``.
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)

    unique_labels = np.unique(true_labels)
    n_labels = len(unique_labels)

    # Accuracy
    accuracy = float(np.mean(true_labels == predicted_labels))

    # Balanced accuracy (mean per-class recall)
    per_class_recall = []
    for label in unique_labels:
        mask = true_labels == label
        if mask.sum() > 0:
            per_class_recall.append(float(np.mean(predicted_labels[mask] == label)))
    balanced_accuracy = float(np.mean(per_class_recall)) if per_class_recall else 0.0

    # Macro F1
    f1_scores = []
    for label in unique_labels:
        mask_true = true_labels == label
        mask_pred = predicted_labels == label
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    # Random baseline
    random_baseline = 1.0 / n_labels if n_labels > 0 else 0.0
    accuracy_over_random = accuracy / random_baseline if random_baseline > 0 else 0.0

    metrics = {
        "LabelSimilarity/accuracy": accuracy,
        "LabelSimilarity/balanced_accuracy": balanced_accuracy,
        "LabelSimilarity/macro_f1": macro_f1,
        "LabelSimilarity/random_baseline_accuracy": random_baseline,
        "LabelSimilarity/accuracy_over_random": accuracy_over_random,
        "LabelSimilarity/n_labels": float(n_labels),
        # Embedding-only metrics — not computable for SCSA
        "LabelSimilarity/mean_auc": float("nan"),
        "LabelSimilarity/std_auc": float("nan"),
        "LabelSimilarity/topk_accuracy@1": float("nan"),
        "LabelSimilarity/topk_accuracy@3": float("nan"),
        "LabelSimilarity/topk_accuracy@5": float("nan"),
        "LabelSimilarity/mrr": float("nan"),
    }

    rows = []
    for metric_name, value in metrics.items():
        rows.append(
            {
                "dataset": dataset_name,
                "model": "scsa",
                "model_name": "scsa",
                "label": annotation_key,
                "label_kind": "LabelKind.BIO",
                "metric": metric_name,
                "value": value,
            }
        )

    return rows


def _find_calmate_cli(calmate_python: str | Path | None, modules_dir: Path) -> str | None:
    """Locate the ``calmate`` console-script in a venv.

    Resolution order:
    1. Derive from *calmate_python* (``…/bin/python`` → ``…/bin/calmate``)
    2. Auto-detect ``modules/calmate_venv/bin/calmate``
    """
    if calmate_python is not None:
        candidate = Path(calmate_python).parent / "calmate"
        if candidate.exists():
            return str(candidate)

    default = modules_dir / "calmate_venv" / "bin" / "calmate"
    if default.exists():
        return str(default)

    return None


def _run_calmate(calmate_cli: str, args: list[str], label: str) -> tuple[bool, str]:
    """Run a single calmate CLI command and return (ok, combined_output).

    Both stdout and stderr are always logged so that diagnostic messages
    from calmate (e.g. backend availability warnings) are visible.
    """
    cmd = [calmate_cli, *args]
    logger.info("Running calmate (%s): %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    if output.strip():
        for line in output.strip().splitlines():
            logger.info("calmate [%s]: %s", label, line)
    if result.returncode != 0:
        logger.error("calmate %s failed (rc=%d)", label, result.returncode)
        return False, output.strip()
    return True, output.strip()


def harmonize_labels(
    predictions_csv: Path,
    store_path: Path,
    modules_dir: Path,
    *,
    dataset_name: str,
    calmate_python: str | Path | None = None,
    cache_dir: str | None = None,
) -> HarmonizeResult:
    """Map SCSA predicted and ground-truth labels to Cell Ontology terms via calmate.

    Calls the calmate CLI twice for ``map`` (predicted + true labels) then twice
    for ``apply`` (predicted + true labels), yielding a final CSV with
    ``predicted_labels_mapped`` and ``true_labels_mapped`` columns.

    Parameters
    ----------
    predictions_csv:
        CSV produced by :func:`map_predictions_to_cells` with at least
        ``predicted_labels`` and ``true_labels`` columns.
    store_path:
        Path to the shared calmate mapping-store CSV.
    modules_dir:
        ``modules/`` directory, used to auto-detect the calmate venv.
    calmate_python:
        Explicit path to the Python interpreter inside the calmate venv.
        When *None*, ``modules/calmate_venv/bin/calmate`` is tried.
    cache_dir:
        Ontology / model cache directory forwarded to ``calmate map``.

    Returns
    -------
    HarmonizeResult
    """
    calmate_cli = _find_calmate_cli(calmate_python, modules_dir)
    if calmate_cli is None:
        return HarmonizeResult(
            success=False,
            all_reviewed=False,
            messages=[
                "Calmate venv not found. Create it with:",
                "  python -m venv modules/calmate_venv",
                "  modules/calmate_venv/bin/pip install -r modules/requirements_calmate.txt",
            ],
        )

    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_arg = ["--store", str(store_path)]
    diagnostics: list[str] = []

    # --- 1. Map predicted labels ---
    map_pred_args = [
        *store_arg,
        "map",
        str(predictions_csv),
        "-c",
        "predicted_labels",
        "-o",
        f"scsa:{dataset_name}",
    ]
    if cache_dir:
        map_pred_args += ["--cache-dir", cache_dir]
    ok, out = _run_calmate(calmate_cli, map_pred_args, "map predicted_labels")
    if out:
        diagnostics.append(f"[map predicted_labels] {out}")
    if not ok:
        return HarmonizeResult(success=False, all_reviewed=False, messages=[f"calmate map predicted_labels failed: {out}"])

    # --- 2. Map true labels ---
    map_true_args = [
        *store_arg,
        "map",
        str(predictions_csv),
        "-c",
        "true_labels",
        "-o",
        f"ground_truth:{dataset_name}",
    ]
    if cache_dir:
        map_true_args += ["--cache-dir", cache_dir]
    ok, out = _run_calmate(calmate_cli, map_true_args, "map true_labels")
    if out:
        diagnostics.append(f"[map true_labels] {out}")
    if not ok:
        return HarmonizeResult(success=False, all_reviewed=False, messages=[f"calmate map true_labels failed: {out}"])

    # --- 3. Check for unreviewed mappings ---
    has_unreviewed = False
    if store_path.exists():
        try:
            store_df = pd.read_csv(store_path)
            if "reviewed" in store_df.columns:
                reviewed_mask = store_df["reviewed"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
                n_unreviewed = int((~reviewed_mask).sum())
                has_unreviewed = n_unreviewed > 0
                logger.info("Calmate store: %d total, %d unreviewed", len(store_df), n_unreviewed)
        except Exception as exc:
            logger.warning("Could not read calmate store at %s: %s", store_path, exc)
            has_unreviewed = True

    if has_unreviewed:
        msgs = diagnostics + [
            "",
            f"{n_unreviewed} label mapping(s) need manual review before metrics can be computed.",
            "To review interactively:",
            f"  {calmate_cli} --store {store_path} review",
            "",
            "Then re-run with:",
            "  python scripts/run_scsa.py settings.skip_existing=false",
        ]
        return HarmonizeResult(success=True, all_reviewed=False, messages=msgs)

    # --- 4. Apply mappings (predicted) ---
    intermediate_csv = predictions_csv.with_name(predictions_csv.stem + "_mapped_pred.csv")
    apply_pred_args = [*store_arg, "apply", str(predictions_csv), "-c", "predicted_labels", "-o", str(intermediate_csv), "--all"]
    ok, out = _run_calmate(calmate_cli, apply_pred_args, "apply predicted_labels")
    if not ok:
        return HarmonizeResult(success=False, all_reviewed=True, messages=[f"calmate apply predicted_labels failed: {out}"])

    # --- 5. Apply mappings (true) ---
    final_csv = predictions_csv.with_name(predictions_csv.stem + "_harmonized.csv")
    apply_true_args = [*store_arg, "apply", str(intermediate_csv), "-c", "true_labels", "-o", str(final_csv), "--all"]
    ok, out = _run_calmate(calmate_cli, apply_true_args, "apply true_labels")
    if not ok:
        return HarmonizeResult(success=False, all_reviewed=True, messages=[f"calmate apply true_labels failed: {out}"])

    # Clean up intermediate file
    if intermediate_csv.exists():
        intermediate_csv.unlink()

    return HarmonizeResult(success=True, all_reviewed=True, mapped_csv=final_csv)


def process_scsa_dataset(
    ds_cfg: Any,
    run_cfg: Any,
    output_root: str,
    modules_dir: Path,
    scsa_cfg: Any,
    adata_cache: str,
    hf_cache: str | None = None,
    skip_existing: bool = True,
    temp_dir: str | None = None,
) -> tuple[str, bool, str | None, list[str]]:
    """
    Run the full SCSA pipeline for a single dataset.

    Loads the dataset, runs clustering + SCSA annotation, harmonises labels
    via calmate (if available), computes metrics, and writes results to
    ``{output_root}/{dataset_name}/scsa/eval/metrics.csv``.

    Parameters
    ----------
    ds_cfg : DictConfig
        Dataset configuration (from dataset_list.yaml).
    run_cfg : DictConfig
        Run configuration (seed, n_rows).
    output_root : str
        Root directory for evaluation results.
    modules_dir : Path
        Directory containing the SCSA installation.
    scsa_cfg : DictConfig
        SCSA-specific parameters (foldchange, pvalue, species, etc.).
    adata_cache : str
        AnnData cache directory.
    hf_cache : str or None
        HuggingFace cache directory.
    skip_existing : bool
        If True, skip datasets that already have SCSA results.
    temp_dir : str or None
        Directory for intermediate files. Defaults to a subdir of output_root.

    Returns
    -------
    tuple[str, bool, str | None, list[str]]
        ``(dataset_name, success, error_message, calmate_messages)``
    """
    from mmcontext.embed.dataset_utils import collect_adata_subset, load_generic_dataset
    from mmcontext.file_utils import save_table

    dataset_name = ds_cfg.name
    calmate_messages: list[str] = []

    out_dir = Path(output_root) / dataset_name / "scsa"
    eval_dir = out_dir / "eval"

    remap_only = bool(scsa_cfg.get("remap_only", False))

    # For full runs we can skip datasets that already have metrics.
    if not remap_only and skip_existing and (eval_dir / "metrics.csv").exists():
        logger.info("SCSA results already exist for %s — skipping", dataset_name)
        return dataset_name, True, "skipped_existing", calmate_messages

    if temp_dir is None:
        temp_dir_path = out_dir / "temp"
    else:
        temp_dir_path = Path(temp_dir) / dataset_name
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    scsa_dir = modules_dir / "SCSA"
    scsa_db = scsa_dir / "whole_v2.db"

    try:
        # ------------------------------------------------------------------
        # Fast path: re-run only the calmate mapping / metrics step
        # ------------------------------------------------------------------
        if remap_only:
            logger.info("remap_only=True — reusing existing SCSA predictions for %s", dataset_name)

            bio_labels = ds_cfg.get("bio_label_list", []) or []
            all_metric_rows: list[dict] = []

            calmate_store = Path(scsa_cfg.get("calmate_store", "modules/.calmate/mappings.csv"))
            calmate_cache = scsa_cfg.get("calmate_cache", None)
            calmate_python_path = scsa_cfg.get("calmate_python", None)

            # Load existing adata (if available) so we can update ontology labels.
            adata_path = eval_dir / "subset.h5ad"
            adata: ad.AnnData | None = None
            if adata_path.exists():
                adata = ad.read_h5ad(adata_path)
                logger.info("Loaded existing adata for remap_only from %s", adata_path)

            for annotation_key in bio_labels:
                pred_path = eval_dir / f"scsa_predictions_{annotation_key}.csv"
                if not pred_path.exists():
                    logger.warning(
                        "Prediction file %s not found for %s / %s — skipping",
                        pred_path,
                        dataset_name,
                        annotation_key,
                    )
                    continue

                logger.info("Re-running calmate label harmonisation for '%s'", annotation_key)
                harm = harmonize_labels(
                    predictions_csv=pred_path,
                    store_path=calmate_store,
                    modules_dir=modules_dir,
                    dataset_name=dataset_name,
                    calmate_python=calmate_python_path,
                    cache_dir=calmate_cache,
                )

                if not harm.success or not harm.all_reviewed:
                    calmate_messages.extend(harm.messages)
                    logger.warning(
                        "Skipping metrics for %s / %s — label harmonisation incomplete",
                        dataset_name,
                        annotation_key,
                    )
                    continue

                mapped_df = pd.read_csv(harm.mapped_csv)  # type: ignore[arg-type]
                mapped_df = _reorder_harmonized_columns(mapped_df)
                mapped_df.to_csv(harm.mapped_csv, index=False)  # keep on disk in the same order
                true_col = "true_labels_mapped" if "true_labels_mapped" in mapped_df.columns else "true_labels"
                pred_col = "predicted_labels_mapped" if "predicted_labels_mapped" in mapped_df.columns else "predicted_labels"
                logger.info("Harmonised CSV at %s (%d rows)", harm.mapped_csv, len(mapped_df))

                # Store mapped true labels back into adata for downstream use
                if adata is not None and true_col in mapped_df.columns:
                    ontology_col = f"{annotation_key}_ontology"
                    label_map = dict(zip(mapped_df["true_labels"], mapped_df[true_col]))
                    adata.obs[ontology_col] = adata.obs[annotation_key].map(label_map).fillna(adata.obs[annotation_key])
                    logger.info("Updated adata.obs['%s'] with ontology-mapped labels", ontology_col)

                metric_rows = compute_metrics(
                    mapped_df[true_col],
                    mapped_df[pred_col],
                    dataset_name,
                    annotation_key,
                )
                all_metric_rows.extend(metric_rows)

            if all_metric_rows:
                metrics_df = pd.DataFrame(all_metric_rows)
                save_table(metrics_df, eval_dir / "metrics", fmt="csv")
                logger.info("Saved %d metrics for %s to %s (remap_only)", len(all_metric_rows), dataset_name, eval_dir)

            if adata is not None:
                adata.write_h5ad(adata_path)
                logger.info("Re-saved adata with updated ontology columns to %s", adata_path)

            return dataset_name, True, None, calmate_messages

        # ------------------------------------------------------------------
        # Full pipeline: download data, run SCSA, run calmate, compute metrics
        # ------------------------------------------------------------------
        # Load HF dataset to get share links and sample IDs
        adata_download_dir = Path(adata_cache) / ds_cfg.name
        cache_dir = hf_cache

        raw_ds = load_generic_dataset(
            source=ds_cfg.source,
            fmt=ds_cfg.format,
            split=ds_cfg.get("split", "test"),
            max_rows=run_cfg.n_rows,
            seed=run_cfg.seed,
            cache_dir=cache_dir,
        )

        link_column = None
        if "share_link" in raw_ds.column_names:
            link_column = "share_link"
        elif "adata_link" in raw_ds.column_names:
            link_column = "adata_link"

        if link_column is None:
            return dataset_name, False, "No numeric data (adata_link/share_link) available for SCSA", calmate_messages

        from mmcontext.file_utils import collect_unique_links, download_and_extract_links

        links, _ = collect_unique_links(raw_ds, link_column=link_column)
        adata_download_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %d adata chunks to %s ...", len(links), adata_download_dir)
        download_and_extract_links(
            links,
            target_dir=adata_download_dir,
            overwrite=False,
        )

        sample_ids = raw_ds[ds_cfg.index_col]
        if isinstance(sample_ids[0], str) and "sample_idx:" in sample_ids[0]:
            sample_ids = [sid.split(":")[1] for sid in sample_ids]

        adata = collect_adata_subset(
            download_dir=adata_download_dir,
            sample_ids=sample_ids,
        )

        logger.info("Loaded AnnData for %s: %d obs x %d vars", dataset_name, adata.n_obs, adata.n_vars)

        # Run SCSA pipeline: cluster -> markers -> SCSA -> map
        marker_csv = temp_dir_path / f"{dataset_name}_markers.csv"
        scsa_output = temp_dir_path / f"{dataset_name}_scsa_result.txt"

        adata, cluster_col = create_marker_csv(
            adata,
            marker_csv,
            cluster_col=scsa_cfg.get("cluster_col", "louvain"),
            resolution=scsa_cfg.get("resolution", 1.0),
            n_top_genes=scsa_cfg.get("n_top_genes", 2000),
            random_state=run_cfg.seed,
        )

        scsa_python = scsa_cfg.get("scsa_python", None)

        run_scsa(
            csv_path=marker_csv,
            db_path=scsa_db,
            output_path=scsa_output,
            scsa_dir=scsa_dir,
            scsa_python=scsa_python,
            foldchange=scsa_cfg.get("foldchange", 1.5),
            pvalue=scsa_cfg.get("pvalue", 0.01),
            species=scsa_cfg.get("species", "Human"),
            tissue=scsa_cfg.get("tissue", "All"),
            gensymbol=scsa_cfg.get("gensymbol", True),
        )

        # Evaluate for each bio label
        bio_labels = ds_cfg.get("bio_label_list", []) or []
        all_metric_rows: list[dict] = []

        calmate_store = Path(scsa_cfg.get("calmate_store", "modules/.calmate/mappings.csv"))
        calmate_cache = scsa_cfg.get("calmate_cache", None)
        calmate_python_path = scsa_cfg.get("calmate_python", None)

        for annotation_key in bio_labels:
            if annotation_key not in adata.obs.columns:
                logger.warning("Annotation key '%s' not found in adata.obs for %s — skipping", annotation_key, dataset_name)
                continue

            logger.info("Mapping SCSA predictions for label '%s'", annotation_key)
            predictions_df = map_predictions_to_cells(scsa_output, adata, cluster_col, annotation_key)

            eval_dir.mkdir(parents=True, exist_ok=True)
            pred_path = eval_dir / f"scsa_predictions_{annotation_key}.csv"
            predictions_df.to_csv(pred_path, index=False)
            logger.info("Saved raw predictions to %s", pred_path)

            # --- Label harmonisation via calmate ---
            harm = harmonize_labels(
                predictions_csv=pred_path,
                store_path=calmate_store,
                modules_dir=modules_dir,
                dataset_name=dataset_name,
                calmate_python=calmate_python_path,
                cache_dir=calmate_cache,
            )

            if not harm.success or not harm.all_reviewed:
                calmate_messages.extend(harm.messages)
                logger.warning(
                    "Skipping metrics for %s / %s — label harmonisation incomplete",
                    dataset_name,
                    annotation_key,
                )
                continue

            # Read harmonised CSV and compute metrics on mapped columns
            mapped_df = pd.read_csv(harm.mapped_csv)  # type: ignore[arg-type]
            mapped_df = _reorder_harmonized_columns(mapped_df)
            mapped_df.to_csv(harm.mapped_csv, index=False)
            true_col = "true_labels_mapped" if "true_labels_mapped" in mapped_df.columns else "true_labels"
            pred_col = "predicted_labels_mapped" if "predicted_labels_mapped" in mapped_df.columns else "predicted_labels"
            logger.info("Harmonised CSV at %s (%d rows)", harm.mapped_csv, len(mapped_df))

            # Store mapped true labels back into adata for downstream use
            ontology_col = f"{annotation_key}_ontology"
            if true_col in mapped_df.columns:
                label_map = dict(zip(mapped_df["true_labels"], mapped_df[true_col]))
                adata.obs[ontology_col] = adata.obs[annotation_key].map(label_map).fillna(adata.obs[annotation_key])
                logger.info("Stored mapped true labels in adata.obs['%s']", ontology_col)

            metric_rows = compute_metrics(
                mapped_df[true_col],
                mapped_df[pred_col],
                dataset_name,
                annotation_key,
            )
            all_metric_rows.extend(metric_rows)

        if all_metric_rows:
            metrics_df = pd.DataFrame(all_metric_rows)
            save_table(metrics_df, eval_dir / "metrics", fmt="csv")
            logger.info("Saved %d metrics for %s to %s", len(all_metric_rows), dataset_name, eval_dir)

        # Re-save adata if ontology columns were added
        ontology_cols = [c for c in adata.obs.columns if c.endswith("_ontology")]
        if ontology_cols:
            adata_out = eval_dir / "subset.h5ad"
            adata.write_h5ad(adata_out)
            logger.info("Saved adata with ontology columns to %s", adata_out)

        logger.info("SCSA pipeline completed for %s", dataset_name)
        return dataset_name, True, None, calmate_messages

    except Exception as e:
        logger.error("SCSA pipeline failed for %s: %s", dataset_name, e, exc_info=True)
        return dataset_name, False, str(e), calmate_messages
