"""
Standalone SCSA evaluation pipeline.

Runs the SCSA (Single-Cell RNA-seq Annotation) baseline method on the same
datasets used by the embedding pipeline and writes metrics in the identical
format so that ``collect_metrics.py`` can fuse them with other model results.

Usage::

    python scripts/run_scsa.py                     # use defaults from scsa_conf.yaml
    python scripts/run_scsa.py settings.output_root=/path/to/results
"""

import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mmcontext.embed.scsa_utils import ensure_scsa_setup, process_scsa_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


@hydra.main(config_path="../conf/eval", config_name="scsa_conf", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run SCSA baseline on every dataset in the shared dataset list."""
    print("\n=== Running SCSA Baseline Evaluation ===")

    modules_dir = Path(cfg.settings.modules_dir)
    if not ensure_scsa_setup(modules_dir):
        print("ERROR: SCSA setup failed. Run modules/prepare_scsa.sh manually.")
        return

    datasets = list(cfg.datasets)
    print(f"Datasets to process: {len(datasets)}")
    for ds in datasets:
        print(f"  - {ds.name}")

    Path(cfg.settings.output_root).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    successful = 0
    failed = 0
    skipped = 0
    needs_review = 0
    all_calmate_messages: list[str] = []

    for i, ds_cfg in enumerate(datasets):
        task_start = time.time()
        print(f"\n=== [{i + 1}/{len(datasets)}] Processing: {ds_cfg.name} ===")

        dataset_name, success, error_msg, calmate_msgs = process_scsa_dataset(
            ds_cfg=ds_cfg,
            run_cfg=cfg.embed,
            output_root=cfg.settings.output_root,
            modules_dir=modules_dir,
            scsa_cfg=cfg.scsa,
            adata_cache=cfg.settings.adata_cache,
            hf_cache=cfg.settings.get("hf_cache", None),
            skip_existing=cfg.settings.get("skip_existing", True),
            temp_dir=cfg.settings.get("temp_dir", None),
        )

        task_time = time.time() - task_start

        if calmate_msgs:
            needs_review += 1
            all_calmate_messages.extend([f"[{dataset_name}]"] + calmate_msgs + [""])

        if success:
            if error_msg == "skipped_existing":
                skipped += 1
                print(f"  Skipped (already exists): {dataset_name} ({task_time:.1f}s)")
            else:
                successful += 1
                print(f"  Completed: {dataset_name} ({task_time:.1f}s)")
        else:
            failed += 1
            print(f"  FAILED: {dataset_name} ({task_time:.1f}s)")
            print(f"    Error: {error_msg}")

    total_time = time.time() - start_time
    print("\n=== SCSA Evaluation Summary ===")
    print(f"Total datasets: {len(datasets)}")
    print(f"Successful:     {successful}")
    print(f"Skipped:        {skipped}")
    print(f"Failed:         {failed}")
    print(f"Total time:     {total_time:.1f}s")
    print(f"Output root:    {cfg.settings.output_root}")

    if all_calmate_messages:
        calmate_store = cfg.scsa.get("calmate_store", "modules/.calmate/mappings.csv")
        calmate_cli = "modules/calmate_venv/bin/calmate"
        print(f"\n{'=' * 60}")
        print(f"  LABEL MAPPING REVIEW REQUIRED")
        print(f"{'=' * 60}")
        print(f"\n{needs_review} dataset(s) have unreviewed or missing label mappings.")
        print("\nDetails:")
        for msg in all_calmate_messages:
            print(f"  {msg}")
        print("To review interactively:")
        print(f"  {calmate_cli} --store {calmate_store} review")
        print("\nThen re-run:")
        print("  python scripts/run_scsa.py settings.skip_existing=false")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
