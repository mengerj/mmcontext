# eval_main.py  (top-level file)
import logging

# from pathlib import Path
import hydra
from omegaconf import DictConfig

from mmcontext.eval.eval_pipeline import run_eval_suite

# from mmcontext.file_utils import copy_resolved_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="eval_conf")
def main(cfg: DictConfig) -> None:
    """
    Launch the evaluation suite.

    Reuses the same Hydra config tree as the embedding stage.
    """
    # Save resolved config in BOTH the hydra run-dir and the
    # dataset/model output dir (same helper used earlier).
    # hydra_run_dir = Path.cwd()
    # named_output_dir = Path(cfg.output.root)
    # copy_resolved_config(cfg, hydra_run_dir, named_output_dir)

    run_eval_suite(cfg)


if __name__ == "__main__":
    main()
