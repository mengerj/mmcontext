# embedding_benchmark/eval_pipeline.py
import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

import mmcontext.eval as ev_pkg
from mmcontext.eval.registry import get as get_evaluator

# from mmcontext.adata_utils import collect_adata_subset
from mmcontext.eval.utils import LabelKind, LabelSpec
from mmcontext.file_utils import save_table

# ---- discover all evaluator modules so decorators run ---------------
for m in pkgutil.walk_packages(ev_pkg.__path__, prefix=ev_pkg.__name__ + "."):
    importlib.import_module(m.name)

logger = logging.getLogger(__name__)


def run_scib_evaluation(cfg) -> None:
    """
    Run ScIB evaluation separately for all dataset/model combinations.

    ScIB has its own interface and internal parallelization, so it's handled
    separately from the main evaluation pipeline.

    Args:
        cfg: Hydra configuration object
    """
    print("\n=== Running ScIB Evaluation ===")

    # Create list of all dataset/model combinations
    tasks = []
    for ds_cfg in cfg.datasets:
        for model_cfg in cfg.models:
            tasks.append((ds_cfg, model_cfg))

    print(f"Total dataset/model combinations for ScIB: {len(tasks)}")

    successful = 0
    failed = 0
    skipped = 0

    for i, (ds_cfg, model_cfg) in enumerate(tasks):
        dataset_name = ds_cfg.name
        model_id = model_cfg.source
        model_name = model_cfg.get("name", model_cfg.source)  # Use name if available, fallback to source
        text_only = model_cfg.get("text_only", False)
        model_dir_name = model_name
        if text_only:
            model_id = model_id + "_text_only"
            model_dir_name = model_dir_name + "_text_only"

        print(f"\n=== [{i + 1}/{len(tasks)}] ScIB for: {dataset_name}/{model_id} (name: {model_name}) ===")

        emb_dir = Path(cfg.output.root) / dataset_name / model_dir_name

        # ── Check if eval directory exists and skip if requested ─────────────
        skip_if_eval_exists = cfg.eval.get("skip_if_eval_exists", False)
        eval_dir = emb_dir / "eval"

        if skip_if_eval_exists and eval_dir.exists():
            print(f"↷ Skipping {dataset_name}/{model_id} - eval directory already exists: {eval_dir}")
            skipped += 1
            continue

        # Check if required files exist
        embeddings_file = emb_dir / "embeddings.parquet"
        adata_file = emb_dir / "subset.h5ad"

        if not embeddings_file.exists() or not adata_file.exists():
            print(f"✗ Missing required files for {dataset_name}/{model_id}")
            failed += 1
            continue

        try:
            # Load data
            print(f"Loading data from: {emb_dir}")
            emb_df = pd.read_parquet(embeddings_file)
            E1 = np.vstack(emb_df["embedding"].to_numpy())
            adata = ad.read_h5ad(adata_file)

            print(f"Loaded embeddings: {E1.shape}, AnnData: {adata.n_obs} obs × {adata.n_vars} vars")

            # Pre-validate data for ScIB
            if E1.size == 0:
                print(f"✗ Empty embedding array for {dataset_name}/{model_id}")
                failed += 1
                continue

            if adata.n_obs == 0:
                print(f"✗ Empty AnnData for {dataset_name}/{model_id}")
                failed += 1
                continue

            # Run ScIB evaluation
            ScibClass = get_evaluator("scib")
            scib_evaluator = ScibClass()

            # Handle cases where label lists might be None or empty
            bio_labels = ds_cfg.bio_label_list or []
            batch_labels = ds_cfg.batch_label_list or []

            scib_results = scib_evaluator.compute_dataset_model(
                emb1=E1,
                adata=adata,
                dataset_name=dataset_name,
                model_id=model_id,
                model_name=model_name,
                bio_labels=bio_labels,
                batch_labels=batch_labels,
                **cfg.eval,
            )

            # Save results
            if scib_results:
                scib_df = pd.DataFrame(scib_results)
                eval_dir = emb_dir / "eval"
                eval_dir.mkdir(parents=True, exist_ok=True)
                save_table(scib_df, eval_dir / "scib_metrics", fmt="csv")
                print(f"✓ Saved {len(scib_results)} ScIB metrics")
                successful += 1
            else:
                print(f"✗ No ScIB results generated for {dataset_name}/{model_id}")
                failed += 1

        except Exception as e:
            print(f"✗ ScIB evaluation failed for {dataset_name}/{model_id}: {e}")

            # Save error to file
            try:
                eval_dir = emb_dir / "eval"
                eval_dir.mkdir(parents=True, exist_ok=True)
                error_results = [
                    {
                        "dataset": dataset_name,
                        "model": model_id,
                        "model_name": model_name,
                        "bio_label": "unknown",
                        "batch_label": "unknown",
                        "metric": "scib/error",
                        "value": str(e),
                        "data_id": "",
                        "hvg": "",
                        "type": "",
                    }
                ]
                error_df = pd.DataFrame(error_results)
                save_table(error_df, eval_dir / "scib_metrics", fmt="csv")
            except Exception as save_error:
                print(f"  Also failed to save error: {save_error}")

            failed += 1

    print("\n=== ScIB Evaluation Summary ===")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful / len(tasks) * 100:.1f}%")


def process_single_dataset_model(ds_cfg: Any, model_cfg: Any, eval_cfg: dict[str, Any], output_root: str) -> list[dict]:
    """
    Process a single dataset/model combination for standard evaluators only.

    ScIB evaluation is handled separately in run_scib_evaluation().

    Args:
        ds_cfg: Dataset configuration
        model_cfg: Model configuration
        eval_cfg: Evaluation configuration
        output_root: Output root directory

    Returns
    -------
        List of evaluation results as dictionaries
    """
    dataset_name = ds_cfg.name
    model_id = model_cfg.source
    model_name = model_cfg.get("name", model_cfg.source)  # Use name if available, fallback to source
    text_only = model_cfg.get("text_only", False)
    if text_only:
        model_name = model_name + "_text_only"

    plot_only = eval_cfg.get("plot_only", False)
    skip_missing_cache = eval_cfg.get("skip_missing_cache", True)

    if plot_only:
        logger.info(f"Plot-only mode: {model_name} for dataset: {dataset_name}")
    else:
        logger.info(f"Processing model: {model_name} for dataset: {dataset_name}")

    # Handle cases where label lists might be None or empty
    bio_labels = ds_cfg.bio_label_list or []
    batch_labels = ds_cfg.batch_label_list or []

    label_specs = [LabelSpec(n, LabelKind.BIO) for n in bio_labels] + [
        LabelSpec(n, LabelKind.BATCH) for n in batch_labels
    ]

    rows = []

    emb_dir = Path(output_root) / dataset_name / Path(model_name).name.replace("/", "_")
    logger.info(f"Looking for embeddings in: {emb_dir}")

    # ── Check if eval directory exists and skip if requested ─────────────
    skip_if_eval_exists = eval_cfg.get("skip_if_eval_exists", False)
    eval_dir = emb_dir / "eval"

    if skip_if_eval_exists and eval_dir.exists():
        logger.info(f"Skipping {dataset_name}/{model_name} - eval directory already exists: {eval_dir}")
        return []  # Return empty list to indicate skipped

    # ── Handle plot-only mode ───────────────────────────────────────
    if plot_only:
        return process_plot_only_mode(
            ds_cfg, model_cfg, eval_cfg, emb_dir, label_specs, dataset_name, model_id, model_name, skip_missing_cache
        )

    # ── load shared artefacts *once* per model ───────────────────────
    try:
        logger.info(f"Loading embeddings from: {emb_dir / 'embeddings.parquet'}")
        emb_df = pd.read_parquet(emb_dir / "embeddings.parquet")
        logger.info(f"✓ Loaded embeddings: {len(emb_df)} rows")

        E1 = np.vstack(emb_df["embedding"].to_numpy())
        logger.info(f"✓ Stacked embeddings: {E1.shape}")

        logger.info(f"Loading AnnData from: {emb_dir / 'subset.zarr'}")
        adata = ad.read_zarr(emb_dir / "subset.zarr")
        logger.info(f"✓ Loaded AnnData: {adata.n_obs} obs × {adata.n_vars} vars")

    except FileNotFoundError as e:
        logger.error(f"✗ Missing file for {dataset_name}/{model_name}: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Error loading data for {dataset_name}/{model_name}: {e}")
        raise

    # ──── Run standard evaluators ────────────────────
    evaluators = eval_cfg.get("suite", [])
    logger.info(f"Running evaluators: {evaluators}")

    # try to load label embeddings only once
    label_emb_cache = {}  # (kind, name) → tuple[ndarray, dict] | None
    # ------------- ITERATE over labels AND evaluators --------------
    for label_spec in label_specs:
        if label_spec.name not in adata.obs.columns:
            logger.info(f"  Skipping label {label_spec.name} - not found in adata.obs")
            continue  # skip silently
        logger.info(f"  Processing label: {label_spec.name} ({label_spec.kind})")
        y = adata.obs[label_spec.name].to_numpy()
        if (label_spec.kind, label_spec.name) not in label_emb_cache:
            prefix = "bio_label_embeddings" if label_spec.kind == LabelKind.BIO else "batch_label_embeddings"
            path = emb_dir / f"{prefix}_{label_spec.name}.parquet"
            mapping_path = emb_dir / f"{prefix}_{label_spec.name}_mapping.json"
            if path.exists():
                try:
                    df2 = pd.read_parquet(path)
                    embeddings = np.vstack(df2["embedding"].to_numpy())

                    # Load label mapping if available
                    label_to_index = None
                    if mapping_path.exists():
                        import json

                        with open(mapping_path) as f:
                            label_to_index = json.load(f)
                        logger.info(f"    ✓ Loaded label mapping for {label_spec.name}")
                    else:
                        logger.warning(f"    - No label mapping found for {label_spec.name}, using legacy format")

                    label_emb_cache[(label_spec.kind, label_spec.name)] = (embeddings, label_to_index)
                    logger.info(f"    ✓ Loaded label embeddings for {label_spec.name}")
                except Exception as e:
                    logger.error(f"    ✗ Error loading label embeddings for {label_spec.name}: {e}")
                    label_emb_cache[(label_spec.kind, label_spec.name)] = None
            else:
                logger.info(f"    - No label embeddings found for {label_spec.name}")
                label_emb_cache[(label_spec.kind, label_spec.name)] = None

        # Extract embeddings and mapping
        cache_entry = label_emb_cache[(label_spec.kind, label_spec.name)]
        if cache_entry is not None:
            E2, label_to_index = cache_entry
        else:
            E2, label_to_index = None, None

        for ev_name in evaluators:
            logger.info(f"    Running evaluator: {ev_name}")
            try:
                EvClass = get_evaluator(ev_name)
                ev = EvClass()

                if ev.requires_pair and E2 is None:
                    logger.info(f"      Skipping {ev_name} - requires pair but E2 is None")
                    continue

                result = ev.compute(
                    E1,
                    emb2=E2,
                    labels=y,
                    adata=adata,
                    label_kind=label_spec.kind,  # evaluators may ignore
                    label_key=label_spec.name,  # evaluators may ignore
                    out_dir=emb_dir,  # Pass output directory for caching
                    label_to_index=label_to_index,  # Pass label mapping for new format
                    **eval_cfg,
                )

                for key, val in result.items():
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_id,
                            "model_name": model_name,
                            "label": label_spec.name,
                            "label_kind": label_spec.kind,
                            "metric": f"{ev_name}/{key}",
                            "value": val,
                        }
                    )
                logger.info(f"      ✓ {ev_name} completed, {len(result)} metrics")

                if ev.produces_plot:
                    logger.info(f"      Generating plots for {ev_name}")
                    # …/eval/<Evaluator>/<label-name>/figure.png
                    plot_dir = (
                        emb_dir / "eval" / ev_name / label_spec.name  # <— NEW: sub-folder per label value
                    )
                    plot_dir.mkdir(parents=True, exist_ok=True)

                    ev.plot(
                        E1,
                        out_dir=plot_dir,
                        emb2=E2,
                        labels=y,  # ground-truth for *this* label column
                        adata=adata,
                        label_kind=label_spec.kind,  # "bio"  or  "batch"
                        label_key=label_spec.name,  # column name (e.g. "celltype")
                        label_to_index=label_to_index,  # Pass label mapping for new format
                        **eval_cfg,  # forward any extra Hydra knobs
                    )
                    logger.info(f"      ✓ Plots saved to {plot_dir}")
            except Exception as e:
                logger.error(f"      ✗ Error in evaluator {ev_name}: {e}")
                # Continue with next evaluator instead of crashing
                continue

    # ──── Save results ────────────────────────────────────────────────
    if rows:
        out_df = pd.DataFrame(rows)
        save_table(out_df, emb_dir / "eval/metrics", fmt="csv")
        logger.info(f"✓ Saved {len(rows)} evaluation results")

    logger.info(f"✓ Completed processing model {model_name} for dataset {dataset_name}")

    return rows


def process_plot_only_mode(
    ds_cfg: Any,
    model_cfg: Any,
    eval_cfg: dict[str, Any],
    emb_dir: Path,
    label_specs: list,
    dataset_name: str,
    model_id: str,
    model_name: str,
    skip_missing_cache: bool,
) -> list[dict]:
    """
    Process a single dataset/model combination in plot-only mode.

    This function regenerates plots using cached similarity matrices without recomputing embeddings.

    Args:
        ds_cfg: Dataset configuration
        model_cfg: Model configuration
        eval_cfg: Evaluation configuration
        emb_dir: Embedding directory path
        label_specs: List of label specifications
        dataset_name: Dataset name
        model_id: Model ID
        model_name: Model name
        skip_missing_cache: Whether to skip if cache is missing

    Returns
    -------
        List of evaluation results (empty in plot-only mode since we're not recomputing)
    """
    evaluators = eval_cfg.get("suite", [])
    logger.info(f"Plot-only mode: Regenerating plots for evaluators: {evaluators}")

    generated_any_plots = False

    for label_spec in label_specs:
        logger.info(f"  Processing label: {label_spec.name} ({label_spec.kind})")

        for ev_name in evaluators:
            try:
                EvClass = get_evaluator(ev_name)
                ev = EvClass()

                # Only process evaluators that produce plots and support plot-only mode
                if not ev.produces_plot:
                    logger.info(f"      Skipping {ev_name} - doesn't produce plots")
                    continue

                # Check if evaluator supports plot_only method (currently only LabelSimilarity)
                if not hasattr(ev, "plot_only"):
                    logger.info(f"      Skipping {ev_name} - doesn't support plot-only mode")
                    continue

                plot_dir = emb_dir / "eval" / ev_name / label_spec.name
                cache_file = plot_dir / "label_similarity_cache.pkl"

                if not cache_file.exists():
                    if skip_missing_cache:
                        logger.info(f"      Skipping {ev_name}/{label_spec.name} - no cache found")
                        continue
                    else:
                        logger.error(f"      ✗ Cache not found for {ev_name}/{label_spec.name}: {cache_file}")
                        continue

                logger.info(f"      Regenerating plots for {ev_name} using cached data")
                plot_dir.mkdir(parents=True, exist_ok=True)

                # Use plot_only method for supported evaluators
                ev.plot_only(
                    out_dir=plot_dir,
                    label_kind=label_spec.kind,
                    label_key=label_spec.name,
                    **eval_cfg,  # forward any extra Hydra knobs
                )

                logger.info(f"      ✓ Plots regenerated for {ev_name}")
                generated_any_plots = True

            except Exception as e:
                logger.error(f"      ✗ Error regenerating plots for {ev_name}: {e}")
                continue

    if generated_any_plots:
        logger.info(f"✓ Plot-only mode completed for {model_id} for dataset {dataset_name}")
    else:
        logger.warning(f"⚠ No plots were regenerated for {model_id} for dataset {dataset_name}")

    # Return empty list since we're not computing new metrics in plot-only mode
    return []


def _process_single_worker(args):
    """
    Worker function for parallel processing of standard evaluators.

    Args:
        args: Tuple of (ds_cfg, model_cfg, eval_cfg, output_root)

    Returns
    -------
        Tuple of (dataset_name, model_id, success, result_count, error_msg, was_skipped)
    """
    import os
    import sys
    import time
    import traceback

    ds_cfg, model_cfg, eval_cfg, output_root = args
    dataset_name = ds_cfg.name
    model_id = model_cfg.source
    model_name = model_cfg.get("name", model_cfg.source)  # Use name if available, fallback to source
    text_only = model_cfg.get("text_only", False)
    if text_only:
        model_id = model_id + "_text_only"

    # Configure logging for this worker process
    worker_logger = logging.getLogger(f"eval_worker_{os.getpid()}")
    worker_logger.setLevel(logging.INFO)

    # Create a handler that writes to stdout so we can see worker logs
    if not worker_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(f"[Worker {os.getpid()}] %(levelname)s - %(message)s"))
        worker_logger.addHandler(handler)

        worker_logger.info(f"Starting processing: {dataset_name}/{model_id} (name: {model_name})")
    start_time = time.time()

    try:
        results = process_single_dataset_model(ds_cfg, model_cfg, eval_cfg, output_root)

        # Check if this was skipped (empty results + skip flag enabled)
        was_skipped = False
        if len(results) == 0 and eval_cfg.get("skip_if_eval_exists", False):
            eval_dir = Path(output_root) / dataset_name / Path(model_name).name.replace("/", "_") / "eval"
            if eval_dir.exists():
                was_skipped = True

        elapsed_time = time.time() - start_time
        worker_logger.info(
            f"Completed processing: {dataset_name}/{model_id} (name: {model_name}) in {elapsed_time:.1f}s"
        )
        return dataset_name, model_id, True, len(results), None, was_skipped

    except Exception as e:
        # Get full traceback for debugging
        elapsed_time = time.time() - start_time
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        worker_logger.error(
            f"Error processing {dataset_name}/{model_id} (name: {model_name}) after {elapsed_time:.1f}s: {error_msg}"
        )
        return dataset_name, model_id, False, 0, str(e), False


def run_eval_suite(cfg) -> None:
    """
    Run the evaluation suite.

    Is called from the eval.py script.
    """
    import concurrent.futures
    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Check if we're in plot-only mode
    plot_only = cfg.eval.get("plot_only", False)

    if plot_only:
        print("\n=== Running Evaluation Suite in Plot-Only Mode ===")
        print("This will regenerate plots using cached similarity matrices.")
        print("Skipping ScIB evaluation (not supported in plot-only mode).")
    else:
        print("\n=== Running Full Evaluation Suite ===")

    # Create list of all dataset/model combinations
    tasks = []
    for ds_cfg in cfg.datasets:
        for model_cfg in cfg.models:
            tasks.append((ds_cfg, model_cfg, cfg.eval, cfg.output.root))

    print(f"Total dataset/model combinations to process: {len(tasks)}")

    # Check if parallel processing is enabled
    enable_parallel = cfg.eval.get("enable_parallel", True)
    max_workers = cfg.eval.get("max_workers", None)

    if max_workers is None:
        max_workers = mp.cpu_count()

    start_time = time.time()

    if not enable_parallel or len(tasks) == 1:
        # Sequential processing
        print("Running in sequential mode...")
        successful = 0
        failed = 0
        skipped = 0

        for i, (ds_cfg, model_cfg, eval_cfg, output_root) in enumerate(tasks):
            dataset_name = ds_cfg.name
            model_id = model_cfg.source
            model_name = model_cfg.get("name", model_cfg.source)  # Use name if available, fallback to source
            text_only = model_cfg.get("text_only", False)
            if text_only:
                model_id = model_id + "_text_only"

            task_start = time.time()
            print(
                f"\n=== [{i + 1}/{len(tasks)}] Processing dataset: {dataset_name}, model: {model_id} (name: {model_name}) ==="
            )
            try:
                results = process_single_dataset_model(ds_cfg, model_cfg, eval_cfg, output_root)
                task_time = time.time() - task_start

                if len(results) == 0 and eval_cfg.get("skip_if_eval_exists", False):
                    # Check if this was actually skipped (empty results + skip flag enabled)
                    eval_dir = Path(output_root) / dataset_name / Path(model_name).name.replace("/", "_") / "eval"
                    if eval_dir.exists():
                        skipped += 1
                        print(
                            f"↷ Skipped {model_id} (name: {model_name}) for dataset {dataset_name} - eval directory exists ({task_time:.1f}s)"
                        )
                        continue

                successful += 1
                print(
                    f"✓ Completed processing model {model_id} (name: {model_name}) for dataset {dataset_name} ({task_time:.1f}s)"
                )
                print(f"  → {len(results)} evaluation metrics")

                # Estimate remaining time
                if i > 0:
                    avg_time = (time.time() - start_time) / (i + 1)
                    remaining_time = avg_time * (len(tasks) - i - 1)
                    print(f"  → Estimated time remaining: {remaining_time:.1f}s")

            except Exception as e:
                failed += 1
                print(f"✗ Error processing model {model_id} (name: {model_name}) for dataset {dataset_name}: {e}")
                print("  Continuing with next combination...")
                continue

        total_time = time.time() - start_time
        print("\n=== Summary ===")
        print(f"Total tasks: {len(tasks)}")
        print(f"Successful: {successful}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful / len(tasks) * 100:.1f}%")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per task: {total_time / len(tasks):.1f}s")
    else:
        # Parallel processing
        print(f"Running in parallel mode with {max_workers} workers...")

        # Get timeout from config (default: 1 hour per task)
        task_timeout = cfg.eval.get("task_timeout", 3600)
        print(f"Task timeout: {task_timeout}s ({task_timeout / 60:.1f} minutes)")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for i, task in enumerate(tasks):
                ds_cfg, model_cfg, eval_cfg, output_root = task
                dataset_name = ds_cfg.name
                model_id = model_cfg.source
                model_name = model_cfg.get("name", model_cfg.source)  # Use name if available, fallback to source
                text_only = model_cfg.get("text_only", False)
                if text_only:
                    model_id = model_id + "_text_only"

                future = executor.submit(_process_single_worker, task)
                future_to_task[future] = (dataset_name, model_id, model_name, i, time.time())

            # Process completed tasks with timeout
            completed = 0
            successful = 0
            failed = 0
            skipped = 0
            timed_out = 0

            try:
                for future in as_completed(future_to_task, timeout=task_timeout + 60):  # Small buffer for cleanup
                    dataset_name, model_id, model_name, task_idx, submit_time = future_to_task[future]
                    completed += 1
                    task_time = time.time() - submit_time

                    try:
                        # Get result with timeout
                        dataset_name_result, model_id_result, success, result_count, error_msg, was_skipped = (
                            future.result(timeout=10)
                        )

                        if success:
                            if was_skipped:
                                skipped += 1
                                print(
                                    f"↷ [{completed}/{len(tasks)}] Skipped: {dataset_name_result}/{model_id_result} (name: {model_name}) - eval exists ({task_time:.1f}s)"
                                )
                            else:
                                successful += 1
                                print(
                                    f"✓ [{completed}/{len(tasks)}] Completed: {dataset_name_result}/{model_id_result} (name: {model_name}) ({task_time:.1f}s)"
                                )
                                print(f"  → {result_count} evaluation metrics")
                        else:
                            failed += 1
                            print(
                                f"✗ [{completed}/{len(tasks)}] Failed: {dataset_name_result}/{model_id_result} (name: {model_name}) ({task_time:.1f}s)"
                            )
                            print(f"  Error: {error_msg}")

                        # Progress update
                        if completed % max(1, len(tasks) // 10) == 0:  # Show progress every 10%
                            elapsed = time.time() - start_time
                            if completed > 0:
                                eta = (elapsed / completed) * (len(tasks) - completed)
                                print(f"  → Progress: {completed / len(tasks) * 100:.1f}% | ETA: {eta:.1f}s")

                    except concurrent.futures.TimeoutError:
                        timed_out += 1
                        failed += 1
                        print(
                            f"✗ [{completed}/{len(tasks)}] Timeout: {dataset_name}/{model_id} (name: {model_name}) (>{task_timeout}s)"
                        )
                        future.cancel()  # Try to cancel the future

                    except Exception as e:
                        failed += 1
                        print(
                            f"✗ [{completed}/{len(tasks)}] Exception in worker for {dataset_name}/{model_id} (name: {model_name}): {e}"
                        )

            except concurrent.futures.TimeoutError:
                print("⚠ Warning: Overall timeout reached. Some tasks may not have completed.")
                # Handle any remaining uncompleted tasks
                remaining_tasks = len(tasks) - completed
                if remaining_tasks > 0:
                    print(f"⚠ Warning: {remaining_tasks} tasks did not complete")
                    failed += remaining_tasks

            total_time = time.time() - start_time
            print("\n=== Summary ===")
            print(f"Total tasks: {len(tasks)}")
            print(f"Successful: {successful}")
            print(f"Skipped: {skipped}")
            print(f"Failed: {failed}")
            if timed_out > 0:
                print(f"Timed out: {timed_out}")
            print(f"Success rate: {successful / len(tasks) * 100:.1f}%")
            print(f"Total time: {total_time:.1f}s")
            print(f"Average time per task: {total_time / len(tasks):.1f}s")
            print(f"Speedup vs sequential: ~{max_workers}x (theoretical)")

    print("\n=== Evaluation suite completed ===")
