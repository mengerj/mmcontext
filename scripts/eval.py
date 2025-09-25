# eval_main.py  (top-level file)
import logging

# from pathlib import Path
import hydra
from omegaconf import DictConfig

from mmcontext.eval.eval_pipeline import run_eval_suite, run_scib_evaluation

# from mmcontext.utils import copy_resolved_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="eval_conf")
def main(cfg: DictConfig) -> None:
    """
    Launch the evaluation suite.

    Runs both standard evaluators (in parallel) and ScIB evaluation (sequential).
    """
    # Save resolved config in BOTH the hydra run-dir and the
    # dataset/model output dir (same helper used earlier).
    # hydra_run_dir = Path.cwd()
    # named_output_dir = Path(cfg.output.root)
    # copy_resolved_config(cfg, hydra_run_dir, named_output_dir)

    print("=" * 60)
    print("MMCONTEXT EVALUATION PIPELINE")
    print("=" * 60)

    # Check evaluation mode
    plot_only = cfg.eval.get("plot_only", False)
    if plot_only:
        print("🎨 PLOT-ONLY MODE: Regenerating plots from cached data")

    # Check what evaluators are configured
    eval_suite = cfg.eval.get("suite", [])
    has_scib = "scib" in eval_suite and not plot_only  # Skip ScIB in plot-only mode
    standard_evaluators = [ev for ev in eval_suite if ev != "scib"]

    print(f"Configured evaluators: {eval_suite}")
    print(f"Standard evaluators: {standard_evaluators}")
    print(f"ScIB evaluation: {'enabled' if has_scib else 'disabled' if not plot_only else 'disabled (plot-only mode)'}")

    # Run standard evaluators (parallel)
    if standard_evaluators:
        if plot_only:
            print(f"\n{'=' * 20} REGENERATING PLOTS {'=' * 20}")
        else:
            print(f"\n{'=' * 20} STANDARD EVALUATORS {'=' * 20}")

        # Create a modified config for standard evaluators only
        eval_cfg_standard = cfg.eval.copy()
        eval_cfg_standard.suite = standard_evaluators

        cfg_standard = cfg.copy()
        cfg_standard.eval = eval_cfg_standard

        run_eval_suite(cfg_standard)
    else:
        print("\nNo standard evaluators configured, skipping...")

    # Run ScIB evaluation (sequential with internal parallelization)
    if has_scib:
        print(f"\n{'=' * 20} SCIB EVALUATION {'=' * 20}")
        run_scib_evaluation(cfg)
    elif plot_only and "scib" in eval_suite:
        print(f"\n{'=' * 20} SCIB EVALUATION {'=' * 20}")
        print("⚠ ScIB evaluation skipped in plot-only mode (not supported)")
    else:
        print("\nScIB evaluation disabled, skipping...")

    if plot_only:
        print(f"\n{'=' * 20} PLOT REGENERATION COMPLETE {'=' * 20}")
    else:
        print(f"\n{'=' * 20} EVALUATION COMPLETE {'=' * 20}")


if __name__ == "__main__":
    main()
