import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mmcontext.embed.embed_pipeline import embed_pipeline
from mmcontext.file_utils import copy_resolved_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


@hydra.main(config_path="../conf", config_name="embed_conf")
def main(cfg: DictConfig) -> None:
    """Entry point launched by Hydra."""
    # ------------------------------------------------------------------ #
    # save the resolved config in both places *before* heavy work starts
    # ------------------------------------------------------------------ #
    # hydra_run_dir = Path.cwd()
    # named_output_dir = (
    #    Path(cfg.output.root) / cfg.data.dir
    # )
    # copy_resolved_config(cfg, hydra_run_dir)

    embed_pipeline(cfg)


if __name__ == "__main__":
    main()
