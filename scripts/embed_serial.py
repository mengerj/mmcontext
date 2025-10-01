#!/usr/bin/env python3
"""
Serial embedding script - no parallelization, GPU-friendly.

This script processes dataset/model combinations one by one in a simple loop,
avoiding all multiprocessing issues that can cause CUDA context problems.
"""

import logging
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mmcontext.embed.embed_pipeline import process_single_dataset_model
from mmcontext.file_utils import copy_resolved_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def embed_serial_pipeline(cfg: DictConfig) -> None:
    """
    Serial embedding pipeline - processes one dataset/model combination at a time.

    This avoids all multiprocessing and worker process issues that can cause
    CUDA context problems with GPU processing.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing datasets, models, and run settings
    """
    print("\n=== Running Serial Embedding Pipeline ===")

    # GPU information
    if torch.cuda.is_available():
        print(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 GPU memory available: {gpu_memory:.1f} GB")
        print("✅ Using serial processing - no CUDA context issues!")
    else:
        #print("💻 Using CPU processing")
        raise ValueError("No GPU detected")

    # Build list of tasks
    tasks = []
    for ds_cfg in cfg.datasets:
        for model_cfg in cfg.models:
            tasks.append((ds_cfg, model_cfg, cfg.run, cfg.output.root, cfg.output.format, cfg.output.adata_cache))

    print(f"📋 Processing {len(tasks)} dataset/model combinations")
    print(f"⚙️  Batch size: {cfg.run.batch_size}")
    print(f"👥 DataLoader workers: {cfg.run.num_workers}")

    # Process tasks serially
    successful = 0
    failed = 0
    total_start_time = time.time()

    for i, task in enumerate(tasks, 1):
        ds_cfg, model_cfg, run_cfg, output_root, output_format, adata_cache = task

        # Determine model identifier for logging
        model_id = model_cfg.source
        text_only = model_cfg.get("text_only", False)
        if text_only:
            model_id_display = f"{model_id}_text_only"
        else:
            model_id_display = model_id

        print(f"\n[{i}/{len(tasks)}] Processing: {ds_cfg.name} / {model_id_display}")
        task_start_time = time.time()

        try:
            # Process the single task
            result_dataset_name, result_model_id, success, error_msg = process_single_dataset_model(
                ds_cfg, model_cfg, run_cfg, output_root, output_format, adata_cache
            )

            task_duration = time.time() - task_start_time

            if success:
                print(f"✅ Completed in {task_duration:.1f}s: {result_dataset_name}/{result_model_id}")
                successful += 1
            else:
                print(f"❌ Failed after {task_duration:.1f}s: {error_msg}")
                failed += 1

        except Exception as e:
            task_duration = time.time() - task_start_time
            print(f"❌ Error after {task_duration:.1f}s: {str(e)}")
            failed += 1
            logger.exception(f"Unexpected error processing {ds_cfg.name}/{model_id_display}")

        # Clean up GPU memory between tasks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary
    total_duration = time.time() - total_start_time
    print("\n=== Summary ===")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_duration:.1f}s ({total_duration / 60:.1f} minutes)")
    print(f"⚡ Average time per task: {total_duration / len(tasks):.1f}s")

    if failed > 0:
        print(f"⚠️  {failed} tasks failed. Check logs above for details.")
    else:
        print("🎉 All tasks completed successfully!")


@hydra.main(config_path="../conf", config_name="embed_serial_conf")
def main(cfg: DictConfig) -> None:
    """Entry point launched by Hydra."""
    # Save the resolved config
    try:
        hydra_run_dir = Path.cwd()
        copy_resolved_config(cfg, hydra_run_dir)
        logger.info(f"Saved resolved config to {hydra_run_dir}")
    except Exception as e:
        logger.warning(f"Could not save resolved config: {e}")

    # Run the serial pipeline
    embed_serial_pipeline(cfg)


if __name__ == "__main__":
    main()
