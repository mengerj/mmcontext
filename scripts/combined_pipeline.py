import logging
import shutil
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from mmcontext.embed.embed_pipeline import process_single_dataset_model as embed_single
from mmcontext.eval.eval_pipeline import process_single_dataset_model as eval_single
from mmcontext.eval.eval_pipeline import run_scib_evaluation
from mmcontext.file_utils import copy_resolved_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def process_combined_dataset_model(
    ds_cfg: Any,
    model_cfg: Any,
    cfg: DictConfig,
) -> tuple[str, str, bool, str, int]:
    """
    Process a single dataset/model combination through both embedding and evaluation steps.

    Parameters
    ----------
    ds_cfg : dict
        Dataset configuration
    model_cfg : dict
        Model configuration
    cfg : DictConfig
        Full configuration object with embed, eval, and settings sections

    Returns
    -------
    tuple[str, str, bool, str, int]
        Tuple of (dataset_name, model_name, success, error_msg, eval_result_count)
    """
    dataset_name = ds_cfg.name
    model_name = model_cfg.get("name", model_cfg.source)
    text_only = model_cfg.get("text_only", False)

    if text_only:
        model_name_for_path = model_name + "_text_only"
    else:
        model_name_for_path = model_name

    logger.info(f"Processing combined pipeline for {dataset_name} + {model_name_for_path}")

    try:
        # ═══════════════════════════════════════════════════════════════
        # STEP 0: CHECK IF EVALUATION ALREADY EXISTS (BEFORE EMBEDDING)
        # ═══════════════════════════════════════════════════════════════
        final_eval_dir = Path(cfg.settings.final_root) / dataset_name / model_name_for_path / "eval"
        skip_existing_eval = cfg.settings.get("skip_existing_evaluations", False)

        if skip_existing_eval and final_eval_dir.exists():
            logger.info(
                f"Evaluation already exists for {dataset_name}/{model_name_for_path} in final directory. Skipping entire pipeline."
            )
            return dataset_name, model_name_for_path, True, "evaluation_skipped_existing", 0

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: EMBEDDING GENERATION
        # ═══════════════════════════════════════════════════════════════
        logger.info(f"Step 1/2: Generating embeddings for {dataset_name}/{model_name_for_path}")

        embed_dataset_name, embed_model_name, embed_success, embed_error = embed_single(
            ds_cfg=ds_cfg,
            model_cfg=model_cfg,
            run_cfg=cfg.embed,
            output_root=cfg.settings.computation_root,
            output_format=cfg.settings.output_format,
            adata_cache=cfg.settings.adata_cache,
            hf_cache=cfg.settings.hf_cache,
        )

        if not embed_success:
            if embed_error == "skipped_existing":
                logger.info(f"Embeddings already exist for {dataset_name}/{model_name_for_path}")
            else:
                logger.error(f"Embedding failed for {dataset_name}/{model_name_for_path}: {embed_error}")
                return dataset_name, model_name_for_path, False, f"Embedding failed: {embed_error}", 0
        else:
            logger.info(f"✓ Embedding completed for {dataset_name}/{model_name_for_path}")

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: EVALUATION
        # ═══════════════════════════════════════════════════════════════
        logger.info(f"Step 2/2: Running evaluation for {dataset_name}/{model_name_for_path}")

        # Run evaluation using computation directory as source
        eval_results = eval_single(
            ds_cfg=ds_cfg,
            model_cfg=model_cfg,
            eval_cfg=cfg.eval,
            output_root=cfg.settings.computation_root,  # Read embeddings from computation directory
        )

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: MOVE EVALUATION RESULTS TO FINAL DIRECTORY
        # ═══════════════════════════════════════════════════════════════
        computation_eval_dir = Path(cfg.settings.computation_root) / dataset_name / model_name_for_path / "eval"

        if computation_eval_dir.exists():
            logger.info(f"Moving evaluation results from {computation_eval_dir} to {final_eval_dir}")

            # Create final directory structure
            final_eval_dir.parent.mkdir(parents=True, exist_ok=True)

            # Move evaluation directory to final location
            if final_eval_dir.exists():
                shutil.rmtree(final_eval_dir)  # Remove existing if present
            shutil.move(str(computation_eval_dir), str(final_eval_dir))

            logger.info(f"✓ Moved evaluation results to {final_eval_dir}")
        else:
            logger.warning(f"No evaluation directory found at {computation_eval_dir}")

        logger.info(f"✓ Combined pipeline completed for {dataset_name}/{model_name_for_path}")
        return dataset_name, model_name_for_path, True, None, len(eval_results)

    except Exception as e:
        logger.error(f"✗ Error in combined pipeline for {dataset_name}/{model_name_for_path}: {e}")
        return dataset_name, model_name_for_path, False, str(e), 0


def _process_combined_worker(args):
    """
    Worker function for parallel processing of combined pipeline.

    Parameters
    ----------
    args : tuple
        Tuple of (ds_cfg, model_cfg, cfg)

    Returns
    -------
    tuple[str, str, bool, str, int]
        Result from process_combined_dataset_model
    """
    import os
    import sys
    import time
    import traceback

    ds_cfg, model_cfg, cfg = args

    dataset_name = ds_cfg.name
    model_name = model_cfg.get("name", model_cfg.source)
    text_only = model_cfg.get("text_only", False)
    if text_only:
        model_name_for_path = model_name + "_text_only"
    else:
        model_name_for_path = model_name

    # Configure logging for this worker process
    worker_logger = logging.getLogger(f"combined_worker_{os.getpid()}")
    worker_logger.setLevel(logging.INFO)

    # Create a handler that writes to stdout so we can see worker logs
    if not worker_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(f"[Worker {os.getpid()}] %(levelname)s - %(message)s"))
        worker_logger.addHandler(handler)

    worker_logger.info(f"Starting combined processing: {dataset_name}/{model_name_for_path}")
    start_time = time.time()

    try:
        result = process_combined_dataset_model(ds_cfg, model_cfg, cfg)

        elapsed_time = time.time() - start_time
        worker_logger.info(
            f"Completed combined processing: {dataset_name}/{model_name_for_path} in {elapsed_time:.1f}s"
        )

        return result

    except Exception as e:
        # Get full traceback for debugging
        elapsed_time = time.time() - start_time
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        worker_logger.error(
            f"Error in combined processing {dataset_name}/{model_name_for_path} after {elapsed_time:.1f}s: {error_msg}"
        )
        return dataset_name, model_name_for_path, False, str(e), 0


def combined_pipeline(cfg: DictConfig) -> None:
    """
    Orchestrate the combined embedding + evaluation pipeline.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Instantiated Hydra config matching combined_conf.yaml schema.
    """
    import concurrent.futures
    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print("\n=== Running Combined Embedding + Evaluation Pipeline ===")

    # Create list of all dataset/model combinations
    tasks = []
    for ds_cfg in cfg.datasets:
        for model_cfg in cfg.models:
            tasks.append((ds_cfg, model_cfg, cfg))

    print(f"Total dataset/model combinations to process: {len(tasks)}")

    # Create output directories
    Path(cfg.settings.computation_root).mkdir(parents=True, exist_ok=True)
    Path(cfg.settings.final_root).mkdir(parents=True, exist_ok=True)

    # Check if parallel processing is enabled
    enable_parallel = cfg.settings.get("enable_parallel", False)
    max_workers = cfg.settings.get("max_workers", None)

    if max_workers is None:
        max_workers = mp.cpu_count()

    start_time = time.time()

    if not enable_parallel or len(tasks) == 1:
        # Sequential processing
        print("Running in sequential mode...")
        successful = 0
        failed = 0
        skipped = 0

        for i, (ds_cfg, model_cfg, task_cfg) in enumerate(tasks):
            dataset_name = ds_cfg.name
            model_name = model_cfg.get("name", model_cfg.source)
            text_only = model_cfg.get("text_only", False)
            if text_only:
                model_name_for_path = model_name + "_text_only"
            else:
                model_name_for_path = model_name

            task_start = time.time()
            print(f"\n=== [{i + 1}/{len(tasks)}] Processing: {dataset_name}/{model_name_for_path} ===")

            try:
                dataset_name_result, model_name_result, success, error_msg, eval_count = process_combined_dataset_model(
                    ds_cfg, model_cfg, task_cfg
                )
                task_time = time.time() - task_start

                if success:
                    if error_msg in ["skipped_existing", "evaluation_skipped_existing"]:
                        skipped += 1
                        print(f"↷ Skipped: {dataset_name_result}/{model_name_result} ({task_time:.1f}s)")
                    else:
                        successful += 1
                        print(f"✓ Completed: {dataset_name_result}/{model_name_result} ({task_time:.1f}s)")
                        print(f"  → {eval_count} evaluation metrics")
                else:
                    failed += 1
                    print(f"✗ Failed: {dataset_name_result}/{model_name_result} ({task_time:.1f}s)")
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

        # Get timeout from config
        task_timeout = cfg.settings.get("task_timeout", 10800)  # 3 hours default
        print(f"Task timeout: {task_timeout}s ({task_timeout / 60:.1f} minutes)")

        # Use ProcessPoolExecutor for CPU processing
        print("Using ProcessPoolExecutor for CPU processing")
        executor_class = ProcessPoolExecutor
        worker_func = _process_combined_worker
        task_args_func = lambda task: (task,)

        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for i, (ds_cfg, model_cfg, task_cfg) in enumerate(tasks):
                dataset_name = ds_cfg.name
                model_name = model_cfg.get("name", model_cfg.source)
                text_only = model_cfg.get("text_only", False)
                if text_only:
                    model_name_for_path = model_name + "_text_only"
                else:
                    model_name_for_path = model_name

                future = executor.submit(worker_func, *task_args_func((ds_cfg, model_cfg, task_cfg)))
                future_to_task[future] = (dataset_name, model_name_for_path, i, time.time())

            # Process completed tasks with timeout
            completed = 0
            successful = 0
            failed = 0
            skipped = 0
            timed_out = 0

            try:
                for future in as_completed(future_to_task, timeout=task_timeout + 60):
                    dataset_name, model_name_for_path, task_idx, submit_time = future_to_task[future]
                    completed += 1
                    task_time = time.time() - submit_time

                    try:
                        dataset_name_result, model_name_result, success, error_msg, eval_count = future.result(
                            timeout=10
                        )

                        if success:
                            if error_msg in ["skipped_existing", "evaluation_skipped_existing"]:
                                skipped += 1
                                print(
                                    f"↷ [{completed}/{len(tasks)}] Skipped: {dataset_name_result}/{model_name_result} ({task_time:.1f}s)"
                                )
                            else:
                                successful += 1
                                print(
                                    f"✓ [{completed}/{len(tasks)}] Completed: {dataset_name_result}/{model_name_result} ({task_time:.1f}s)"
                                )
                                print(f"  → {eval_count} evaluation metrics")
                        else:
                            failed += 1
                            print(
                                f"✗ [{completed}/{len(tasks)}] Failed: {dataset_name_result}/{model_name_result} ({task_time:.1f}s)"
                            )
                            print(f"  Error: {error_msg}")

                        # Progress update
                        if completed % max(1, len(tasks) // 10) == 0:
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
                        future.cancel()

                    except Exception as e:
                        failed += 1
                        print(
                            f"✗ [{completed}/{len(tasks)}] Exception in worker for {dataset_name}/{model_name_for_path}: {e}"
                        )

            except concurrent.futures.TimeoutError:
                print("⚠ Warning: Overall timeout reached. Some tasks may not have completed.")
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

    print("\n=== Combined pipeline completed ===")


@hydra.main(config_path="../conf", config_name="combined_conf")
def main(cfg: DictConfig) -> None:
    """Entry point launched by Hydra."""
    combined_pipeline(cfg)


if __name__ == "__main__":
    main()
