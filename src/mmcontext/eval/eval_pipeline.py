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


def process_single_dataset_model(
    ds_cfg: Any, model_cfg: Any, eval_cfg: dict[str, Any], output_root: str
) -> tuple[list[dict], list[dict]]:
    """
    Process a single dataset/model combination.

    Args:
        ds_cfg: Dataset configuration
        model_cfg: Model configuration
        eval_cfg: Evaluation configuration
        output_root: Output root directory

    Returns
    -------
        Tuple of (regular_results, scib_results) as lists of dictionaries
    """
    dataset_name = ds_cfg.name
    model_id = model_cfg.source
    text_only = model_cfg.get("text_only", False)
    if text_only:
        model_id = model_id + "_text_only"

    logger.info(f"Processing model: {model_id} for dataset: {dataset_name}")

    label_specs = [LabelSpec(n, LabelKind.BIO) for n in ds_cfg.bio_label_list] + [
        LabelSpec(n, LabelKind.BATCH) for n in ds_cfg.batch_label_list
    ]

    rows = []
    scib_rows = []  # Separate storage for scIB results

    emb_dir = Path(output_root) / dataset_name / Path(model_id).name.replace("/", "_")
    logger.info(f"Looking for embeddings in: {emb_dir}")

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
        logger.error(f"✗ Missing file for {dataset_name}/{model_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Error loading data for {dataset_name}/{model_id}: {e}")
        raise

    # ──── Run ScibBundle separately if in suite ──────────────────────
    if "scib" in eval_cfg.get("suite", []):
        logger.info(f"Running ScibBundle for {dataset_name}/{model_id}")
        try:
            ScibClass = get_evaluator("scib")
            scib_evaluator = ScibClass()

            scib_results = scib_evaluator.compute_dataset_model(
                emb1=E1,
                adata=adata,
                dataset_name=dataset_name,
                model_id=model_id,
                bio_labels=ds_cfg.bio_label_list,
                batch_labels=ds_cfg.batch_label_list,
                **eval_cfg,
            )

            scib_rows.extend(scib_results)
            logger.info(f"✓ ScibBundle completed for {dataset_name}/{model_id}")

        except Exception as e:
            logger.error(f"✗ ScibBundle failed for {dataset_name}/{model_id}: {e}")
            # Add error entry
            scib_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_id,
                    "bio_label": "unknown",
                    "batch_label": "unknown",
                    "metric": "scib/error",
                    "value": str(e),
                    "data_id": "",
                    "hvg": "",
                    "type": "",
                }
            )

    # ──── Run regular evaluators (excluding scib) ────────────────────
    # Filter out scib from regular evaluators
    regular_evaluators = [ev_name for ev_name in eval_cfg.get("suite", []) if ev_name != "scib"]
    logger.info(f"Running regular evaluators: {regular_evaluators}")

    # try to load label embeddings only once
    label_emb_cache = {}  # (kind, name) → ndarray | None
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
            if path.exists():
                try:
                    df2 = pd.read_parquet(path)
                    label_emb_cache[(label_spec.kind, label_spec.name)] = np.vstack(df2["embedding"].to_numpy())
                    logger.info(f"    ✓ Loaded label embeddings for {label_spec.name}")
                except Exception as e:
                    logger.error(f"    ✗ Error loading label embeddings for {label_spec.name}: {e}")
                    label_emb_cache[(label_spec.kind, label_spec.name)] = None
            else:
                logger.info(f"    - No label embeddings found for {label_spec.name}")
                label_emb_cache[(label_spec.kind, label_spec.name)] = None
        E2 = label_emb_cache[(label_spec.kind, label_spec.name)]

        for ev_name in regular_evaluators:
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
                    **eval_cfg,
                )

                for key, val in result.items():
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_id,
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
                        **eval_cfg,  # forward any extra Hydra knobs
                    )
                    logger.info(f"      ✓ Plots saved to {plot_dir}")
            except Exception as e:
                logger.error(f"      ✗ Error in evaluator {ev_name}: {e}")
                # Continue with next evaluator instead of crashing
                continue

    # ──── Save results ────────────────────────────────────────────────
    # Save regular evaluator results
    if rows:
        out_df = pd.DataFrame(rows)
        save_table(out_df, emb_dir / "eval/metrics", fmt="csv")
        logger.info(f"✓ Saved {len(rows)} regular evaluation results")

    # Save ScibBundle results separately
    if scib_rows:
        scib_df = pd.DataFrame(scib_rows)
        save_table(scib_df, emb_dir / "eval/scib_metrics", fmt="csv")
        logger.info(f"✓ ScibBundle results saved to {emb_dir / 'eval/scib_metrics.csv'}")

    logger.info(f"✓ Completed processing model {model_id} for dataset {dataset_name}")

    return rows, scib_rows


def _process_single_worker(args):
    """
    Worker function for parallel processing.

    Args:
        args: Tuple of (ds_cfg, model_cfg, eval_cfg, output_root)

    Returns
    -------
        Tuple of (dataset_name, model_id, success, regular_results, scib_results, error_msg)
    """
    import os
    import traceback

    ds_cfg, model_cfg, eval_cfg, output_root = args
    dataset_name = ds_cfg.name
    model_id = model_cfg.source
    text_only = model_cfg.get("text_only", False)
    if text_only:
        model_id = model_id + "_text_only"

    # Configure logging for this worker process
    worker_logger = logging.getLogger(f"eval_worker_{os.getpid()}")
    worker_logger.setLevel(logging.INFO)

    try:
        regular_results, scib_results = process_single_dataset_model(ds_cfg, model_cfg, eval_cfg, output_root)
        return dataset_name, model_id, True, len(regular_results), len(scib_results), None
    except Exception as e:
        # Get full traceback for debugging
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        worker_logger.error(f"Error processing {dataset_name}/{model_id}: {error_msg}")
        return dataset_name, model_id, False, 0, 0, str(e)


def run_eval_suite(cfg) -> None:
    """
    Run the evaluation suite.

    Is called from the eval.py script.
    """
    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

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

        for i, (ds_cfg, model_cfg, eval_cfg, output_root) in enumerate(tasks):
            dataset_name = ds_cfg.name
            model_id = model_cfg.source
            text_only = model_cfg.get("text_only", False)
            if text_only:
                model_id = model_id + "_text_only"

            task_start = time.time()
            print(f"\n=== [{i + 1}/{len(tasks)}] Processing dataset: {dataset_name}, model: {model_id} ===")
            try:
                regular_results, scib_results = process_single_dataset_model(ds_cfg, model_cfg, eval_cfg, output_root)
                task_time = time.time() - task_start
                successful += 1
                print(f"✓ Completed processing model {model_id} for dataset {dataset_name} ({task_time:.1f}s)")
                print(f"  → {len(regular_results)} regular metrics, {len(scib_results)} ScIB metrics")

                # Estimate remaining time
                if i > 0:
                    avg_time = (time.time() - start_time) / (i + 1)
                    remaining_time = avg_time * (len(tasks) - i - 1)
                    print(f"  → Estimated time remaining: {remaining_time:.1f}s")

            except Exception as e:
                failed += 1
                print(f"✗ Error processing model {model_id} for dataset {dataset_name}: {e}")
                print("  Continuing with next combination...")
                continue

        total_time = time.time() - start_time
        print("\n=== Summary ===")
        print(f"Total tasks: {len(tasks)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful / len(tasks) * 100:.1f}%")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per task: {total_time / len(tasks):.1f}s")
    else:
        # Parallel processing
        print(f"Running in parallel mode with {max_workers} workers...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for i, task in enumerate(tasks):
                ds_cfg, model_cfg, eval_cfg, output_root = task
                dataset_name = ds_cfg.name
                model_id = model_cfg.source
                text_only = model_cfg.get("text_only", False)
                if text_only:
                    model_id = model_id + "_text_only"

                future = executor.submit(_process_single_worker, task)
                future_to_task[future] = (dataset_name, model_id, i, time.time())

            # Process completed tasks
            completed = 0
            successful = 0
            failed = 0

            for future in as_completed(future_to_task):
                dataset_name, model_id, task_idx, submit_time = future_to_task[future]
                completed += 1
                task_time = time.time() - submit_time

                try:
                    dataset_name_result, model_id_result, success, regular_count, scib_count, error_msg = (
                        future.result()
                    )

                    if success:
                        successful += 1
                        print(
                            f"✓ [{completed}/{len(tasks)}] Completed: {dataset_name_result}/{model_id_result} ({task_time:.1f}s)"
                        )
                        print(f"  → {regular_count} regular metrics, {scib_count} ScIB metrics")
                    else:
                        failed += 1
                        print(
                            f"✗ [{completed}/{len(tasks)}] Failed: {dataset_name_result}/{model_id_result} ({task_time:.1f}s)"
                        )
                        print(f"  Error: {error_msg}")

                    # Progress update
                    if completed % max(1, len(tasks) // 10) == 0:  # Show progress every 10%
                        elapsed = time.time() - start_time
                        if completed > 0:
                            eta = (elapsed / completed) * (len(tasks) - completed)
                            print(f"  → Progress: {completed / len(tasks) * 100:.1f}% | ETA: {eta:.1f}s")

                except Exception as e:
                    failed += 1
                    print(f"✗ [{completed}/{len(tasks)}] Exception in worker for {dataset_name}/{model_id}: {e}")

            total_time = time.time() - start_time
            print("\n=== Summary ===")
            print(f"Total tasks: {len(tasks)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            print(f"Success rate: {successful / len(tasks) * 100:.1f}%")
            print(f"Total time: {total_time:.1f}s")
            print(f"Average time per task: {total_time / len(tasks):.1f}s")
            print(f"Speedup vs sequential: ~{max_workers}x (theoretical)")

    print("\n=== Evaluation suite completed ===")
