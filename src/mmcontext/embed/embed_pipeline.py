import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from mmcontext.embed.dataset_utils import (
    check_revision_exists,
    collect_adata_subset,
    generate_revision_name,
    load_generic_dataset,
    push_dataset_revision,
)
from mmcontext.embed.model_utils import embed_labels, load_st_model, prepare_model_and_embed
from mmcontext.file_utils import save_table
from mmcontext.utils import truncate_cell_sentences

logger = logging.getLogger(__name__)


def check_required_files_exist(out_dir: Path, output_format: str) -> bool:
    """
    Check if required embedding files already exist in the output directory.

    Parameters
    ----------
    out_dir : Path
        Output directory path
    output_format : str
        Output format (e.g., 'parquet', 'csv')

    Returns
    -------
    bool
        True if required files exist, False otherwise
    """
    # Check for main embeddings file
    embeddings_file = out_dir / f"embeddings.{output_format}"
    meta_file = out_dir / "meta.yaml"

    # Consider the output complete if both embeddings and meta files exist
    return embeddings_file.exists() and meta_file.exists()


def process_single_dataset_model(
    ds_cfg, model_cfg, run_cfg, output_root, output_format, adata_cache, hf_cache=None
) -> tuple[str, str, bool, str]:
    """
    Process a single dataset/model combination for embedding generation.

    Parameters
    ----------
    ds_cfg : dict
        Dataset configuration
    model_cfg : dict
        Model configuration
    run_cfg : dict
        Run configuration
    output_root : str
        Output root directory (resolved)
    output_format : str
        Output format (resolved)
    adata_cache : str
        AnnData cache directory (resolved)
    hf_cache : str, optional
        HuggingFace cache directory (resolved). If None, uses run_cfg.hf_cache for backward compatibility.

    Returns
    -------
    tuple[str, str, bool, str]
        Tuple of (dataset_name, model_id, success, error_msg)
    """
    dataset_name = ds_cfg.name
    model_id = model_cfg.source
    model_name = model_cfg.name
    text_only = model_cfg.get("text_only", False)

    # add a small string to the model indicating if it was used as text-only
    if text_only:
        model_name_for_path = model_name + "_text_only"
    else:
        model_name_for_path = model_name

    logger.info(f"Processing {dataset_name} + {model_name_for_path}")

    # Determine output directory
    out_dir = (
        Path(output_root)
        / ds_cfg.name  # <— dataset-specific folder
        / Path(model_name_for_path).name.replace("/", "_")
    )

    # Check if files already exist and skip if overwrite is False
    if not run_cfg.overwrite and check_required_files_exist(out_dir, output_format):
        logger.info(
            f"Embeddings already exist for {ds_cfg.name} + {model_name_for_path}. Skipping. Use overwrite=true to force regeneration."
        )
        return dataset_name, model_name_for_path, True, "skipped_existing"

    try:
        # Load dataset
        adata_download_dir = Path(adata_cache) / ds_cfg.name
        # Use hf_cache parameter if provided, otherwise fall back to run_cfg.hf_cache for backward compatibility
        cache_dir = hf_cache if hf_cache is not None else getattr(run_cfg, "hf_cache", None)

        # Check if we should use revisions
        use_revisions = getattr(run_cfg, "use_revisions", False)
        revision_loaded = False

        if use_revisions and ds_cfg.format == "hub" and text_only:
            # Generate revision name based on preprocessing parameters
            revision_name = generate_revision_name(ds_cfg)

            # Check if revision exists and load it if available
            if revision_name != "processed" and check_revision_exists(ds_cfg.source, revision_name):
                logger.info(f"Found existing revision '{revision_name}' for dataset '{ds_cfg.name}', loading directly")
                raw_ds = load_generic_dataset(
                    source=ds_cfg.source,
                    fmt=ds_cfg.format,
                    split=ds_cfg.get("split", "test"),
                    max_rows=run_cfg.n_rows,
                    seed=run_cfg.seed,
                    cache_dir=cache_dir,
                    revision=revision_name,
                )
                revision_loaded = True
                logger.info(f"Successfully loaded preprocessed dataset from revision '{revision_name}'")
            else:
                if revision_name != "processed":
                    logger.info(
                        f"Revision '{revision_name}' not found for dataset '{ds_cfg.name}', processing from scratch"
                    )
                # Load raw dataset
                raw_ds = load_generic_dataset(
                    source=ds_cfg.source,
                    fmt=ds_cfg.format,
                    split=ds_cfg.get("split", "test"),
                    max_rows=run_cfg.n_rows,
                    seed=run_cfg.seed,
                    cache_dir=cache_dir,
                )
        else:
            # Load raw dataset without revision
            raw_ds = load_generic_dataset(
                source=ds_cfg.source,
                fmt=ds_cfg.format,
                split=ds_cfg.get("split", "test"),
                max_rows=run_cfg.n_rows,
                seed=run_cfg.seed,
                cache_dir=cache_dir,
            )

        numeric_data_available = "share_link" in raw_ds.column_names

        # Load model
        st_model = load_st_model(model_id)
        if text_only:
            main_col = "cell_sentence_2"
        else:
            main_col = "cell_sentence_1"

        # truncate cell sentences (only if revision wasn't loaded)
        if not revision_loaded and ds_cfg.get("cs_length", None) is not None and main_col == "cell_sentence_2":
            cs_length = ds_cfg.cs_length
            filter_strings = ds_cfg.get("gene_filter_strings", None)
            logger.info(f"Applying cell sentence truncation (cs_length={cs_length})")
            raw_ds = truncate_cell_sentences(raw_ds, main_col, cs_length, filter_strings=filter_strings)

            # Push processed dataset as new revision if enabled
            if use_revisions and ds_cfg.format == "hub" and revision_name != "processed":
                push_success = push_dataset_revision(
                    raw_ds,
                    ds_cfg.source,
                    revision_name,
                    commit_message=f"Processed dataset with {revision_name} settings",
                )
                if push_success:
                    logger.info(f"Successfully pushed processed dataset as revision '{revision_name}'")
                else:
                    logger.warning(f"Failed to push processed dataset as revision '{revision_name}'")
        elif revision_loaded:
            logger.info("Skipping cell sentence truncation - using preprocessed revision")

        # Generate embeddings
        emb_df, path_map = prepare_model_and_embed(
            st_model,
            data=raw_ds,
            main_col=main_col,
            index_col=ds_cfg.index_col,
            batch_size=run_cfg.batch_size,
            num_workers=run_cfg.num_workers,
            layer_key=getattr(model_cfg, "layer_key", None),  # For text only models, layer_key is not needed.
            text_only=text_only,
            adata_download_dir=adata_download_dir,
        )

        if numeric_data_available:
            # get sample_ids but without "sample_idx: prefix"
            if "sample_idx:" in emb_df["sample_idx"][0]:
                sample_ids = [sid.split(":")[1] for sid in emb_df["sample_idx"].tolist()]
            elif "sample_idx" in emb_df.columns:
                sample_ids = emb_df["sample_idx"].tolist()
            else:
                raise ValueError(f"sample_idx column not found in emb_df: {emb_df.columns}")

            # Use path mapping if available (for local datasets), otherwise fall back to download_dir
            if path_map is not None:
                file_paths = list(path_map.values())
                adata_subset = collect_adata_subset(
                    sample_ids=sample_ids,
                    file_paths=file_paths,
                )
            else:
                adata_subset = collect_adata_subset(
                    download_dir=adata_download_dir,
                    sample_ids=sample_ids,
                )
        else:
            adata_subset = None

        # save the embeddings
        save_table(
            emb_df,
            out_path=out_dir / "embeddings",
            fmt=output_format,
        )

        # Save metadata
        (out_dir / "meta.yaml").write_text(
            f"model: {model_name_for_path}\ndataset: {ds_cfg.name}\nrows: {len(emb_df)}\n"
        )

        # Handle label embeddings if available
        if adata_subset is not None:
            subset_out = out_dir / "subset.h5ad"
            adata_subset.write_h5ad(subset_out)
            logger.info("Wrote subset AnnData → %s", subset_out)
            # Define label types and their output prefixes
            label_types = {"bio_label_list": "bio_label_embeddings", "batch_label_list": "batch_label_embeddings"}

            # Loop over each label type
            for label_list_attr, output_prefix in label_types.items():
                if hasattr(ds_cfg, label_list_attr):
                    label_cols = getattr(ds_cfg, label_list_attr)
                    if label_cols is not None:
                        for label_col in label_cols:
                            if label_col in adata_subset.obs.columns:
                                logger.info(f"Embedding {label_list_attr} from column: {label_col}")
                                label_emb_df, label_to_index = embed_labels(
                                    st_model,
                                    adata_subset,
                                    label_col,
                                    batch_size=run_cfg.batch_size,
                                    num_workers=run_cfg.num_workers,
                                )
                                # Save label embeddings
                                save_table(
                                    label_emb_df,
                                    out_path=out_dir / f"{output_prefix}_{label_col}",
                                    fmt=output_format,
                                )

                                # Save label mapping as JSON for downstream use
                                import json

                                mapping_path = out_dir / f"{output_prefix}_{label_col}_mapping.json"
                                with open(mapping_path, "w") as f:
                                    json.dump(label_to_index, f, indent=2)
                                logger.info(f"Saved label mapping to {mapping_path}")
                            else:
                                logger.warning(f"Label column {label_col} not found in adata.obs")

        logger.info(f"✓ Completed processing {dataset_name} + {model_name_for_path}")
        return dataset_name, model_name_for_path, True, None

    except Exception as e:
        logger.error(f"✗ Error processing {dataset_name} + {model_name_for_path}: {e}")
        return dataset_name, model_name_for_path, False, str(e)


def _process_single_worker(args):
    """
    Worker function for parallel processing of embedding generation.

    Parameters
    ----------
    args : tuple
        Tuple of (ds_cfg, model_cfg, run_cfg, output_root, output_format, adata_cache)

    Returns
    -------
    tuple[str, str, bool, str]
        Tuple of (dataset_name, model_id, success, error_msg)
    """
    import os
    import sys
    import time
    import traceback

    import torch

    ds_cfg, model_cfg, run_cfg, output_root, output_format, adata_cache = args
    dataset_name = ds_cfg.name
    model_id = model_cfg.source
    text_only = model_cfg.get("text_only", False)
    if text_only:
        model_name_for_path = model_id + "_text_only"
    else:
        model_name_for_path = model_id

    # Configure logging for this worker process
    worker_logger = logging.getLogger(f"embed_worker_{os.getpid()}")
    worker_logger.setLevel(logging.INFO)

    # Create a handler that writes to stdout so we can see worker logs
    if not worker_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(f"[Worker {os.getpid()}] %(levelname)s - %(message)s"))
        worker_logger.addHandler(handler)

    # Initialize CUDA context in worker process if available
    if torch.cuda.is_available():
        worker_logger.info(f"Initializing CUDA context - Device: {torch.cuda.get_device_name()}")
        # Clear any cached memory from previous runs
        torch.cuda.empty_cache()
        # Set memory growth to avoid OOM in multi-process scenarios
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory max
    else:
        worker_logger.warning("CUDA not available in worker process")

    worker_logger.info(f"Starting processing: {dataset_name}/{model_name_for_path}")
    start_time = time.time()

    try:
        dataset_name_result, model_id_result, success, error_msg = process_single_dataset_model(
            ds_cfg, model_cfg, run_cfg, output_root, output_format, adata_cache
        )

        elapsed_time = time.time() - start_time
        worker_logger.info(f"Completed processing: {dataset_name_result}/{model_id_result} in {elapsed_time:.1f}s")

        # Clean up GPU memory after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return dataset_name_result, model_id_result, success, error_msg

    except Exception as e:
        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get full traceback for debugging
        elapsed_time = time.time() - start_time
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        worker_logger.error(
            f"Error processing {dataset_name}/{model_name_for_path} after {elapsed_time:.1f}s: {error_msg}"
        )
        return dataset_name, model_name_for_path, False, str(e)


def embed_pipeline(cfg) -> None:
    """
    Orchestrate an end-to-end embedding generation run.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Instantiated Hydra config. The schema matches ``conf/config.yaml``.
    """
    import concurrent.futures
    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print("\n=== Running Embedding Pipeline ===")

    # Create list of all dataset/model combinations
    # Resolve config values in main process to avoid Hydra interpolation issues in workers
    tasks = []
    for ds_cfg in cfg.datasets:
        for model_cfg in cfg.models:
            tasks.append((ds_cfg, model_cfg, cfg.run, cfg.output.root, cfg.output.format, cfg.output.adata_cache))

    print(f"Total dataset/model combinations to process: {len(tasks)}")

    # Check if parallel processing is enabled
    enable_parallel = cfg.run.get("enable_parallel", False)
    max_workers = cfg.run.get("max_workers", None)

    if max_workers is None:
        max_workers = mp.cpu_count()

    start_time = time.time()

    if not enable_parallel or len(tasks) == 1:
        # Sequential processing
        print("Running in sequential mode...")
        successful = 0
        failed = 0
        skipped = 0

        for i, (ds_cfg, model_cfg, run_cfg, output_root, output_format, adata_cache) in enumerate(tasks):
            dataset_name = ds_cfg.name
            model_id = model_cfg.source
            model_name = model_cfg.name
            text_only = model_cfg.get("text_only", False)
            if text_only:
                model_name_for_path = model_name + "_text_only"
            else:
                model_name_for_path = model_name

            task_start = time.time()
            print(f"\n=== [{i + 1}/{len(tasks)}] Processing dataset: {dataset_name}, model: {model_name_for_path} ===")

            try:
                dataset_name_result, model_id_result, success, error_msg = process_single_dataset_model(
                    ds_cfg, model_cfg, run_cfg, output_root, output_format, adata_cache
                )
                task_time = time.time() - task_start

                if success:
                    if error_msg == "skipped_existing":
                        skipped += 1
                        print(f"↷ Skipped (already exists): {dataset_name_result}/{model_id_result} ({task_time:.1f}s)")
                    else:
                        successful += 1
                        print(f"✓ Completed: {dataset_name_result}/{model_id_result} ({task_time:.1f}s)")
                else:
                    failed += 1
                    print(f"✗ Failed: {dataset_name_result}/{model_id_result} ({task_time:.1f}s)")
                    print(f"  Error: {error_msg}")

                # Estimate remaining time
                if i > 0:
                    avg_time = (time.time() - start_time) / (i + 1)
                    remaining_time = avg_time * (len(tasks) - i - 1)
                    print(f"  → Estimated time remaining: {remaining_time:.1f}s")

            except Exception as e:
                failed += 1
                print(f"✗ Error processing {dataset_name}/{model_name_for_path}: {e}")
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

        # Get timeout from config (default: 2 hours per task for embedding)
        task_timeout = cfg.run.get("task_timeout", 7200)
        print(f"Task timeout: {task_timeout}s ({task_timeout / 60:.1f} minutes)")

        # Choose executor based on GPU availability and configuration
        use_threads_for_gpu = cfg.run.get("use_threads_for_gpu", True)  # Default to threads for GPU

        if torch.cuda.is_available() and use_threads_for_gpu:
            # Use ThreadPoolExecutor for GPU tasks to share GPU memory efficiently
            from concurrent.futures import ThreadPoolExecutor

            print("Using ThreadPoolExecutor for GPU processing (better memory sharing)")
            executor_class = ThreadPoolExecutor
            worker_func = process_single_dataset_model  # Use direct function for threads
            task_args_func = lambda task: task  # Pass args directly for threads
        else:
            # Use ProcessPoolExecutor for CPU tasks or when explicitly requested
            from concurrent.futures import ProcessPoolExecutor

            print("Using ProcessPoolExecutor")
            executor_class = ProcessPoolExecutor
            worker_func = _process_single_worker  # Use wrapper for processes
            task_args_func = lambda task: (task,)  # Wrap in tuple for processes

        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for i, task in enumerate(tasks):
                ds_cfg, model_cfg, run_cfg, output_root, output_format, adata_cache = task
                dataset_name = ds_cfg.name
                model_id = model_cfg.source
                model_name = model_cfg.name
                text_only = model_cfg.get("text_only", False)
                if text_only:
                    model_name_for_path = model_id + "_text_only"
                else:
                    model_name_for_path = model_id

                future = executor.submit(worker_func, *task_args_func(task))
                future_to_task[future] = (dataset_name, model_name_for_path, i, time.time())

            # Process completed tasks with timeout
            completed = 0
            successful = 0
            failed = 0
            skipped = 0
            timed_out = 0

            try:
                for future in as_completed(future_to_task, timeout=task_timeout + 60):  # Small buffer for cleanup
                    dataset_name, model_name_for_path, task_idx, submit_time = future_to_task[future]
                    completed += 1
                    task_time = time.time() - submit_time

                    try:
                        # Get result with timeout
                        dataset_name_result, model_id_result, success, error_msg = future.result(timeout=10)

                        if success:
                            if error_msg == "skipped_existing":
                                skipped += 1
                                print(
                                    f"↷ [{completed}/{len(tasks)}] Skipped: {dataset_name_result}/{model_id_result} ({task_time:.1f}s)"
                                )
                            else:
                                successful += 1
                                print(
                                    f"✓ [{completed}/{len(tasks)}] Completed: {dataset_name_result}/{model_id_result} ({task_time:.1f}s)"
                                )
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

                    except concurrent.futures.TimeoutError:
                        timed_out += 1
                        failed += 1
                        print(
                            f"✗ [{completed}/{len(tasks)}] Timeout: {dataset_name}/{model_name_for_path} (>{task_timeout}s)"
                        )
                        future.cancel()  # Try to cancel the future

                    except Exception as e:
                        failed += 1
                        print(
                            f"✗ [{completed}/{len(tasks)}] Exception in worker for {dataset_name}/{model_name_for_path}: {e}"
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

    print("\n=== Embedding pipeline completed ===")
