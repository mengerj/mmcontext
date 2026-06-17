"""mmcontext.embed — Pipeline building, dataset preparation, and hard-negative mining."""

from .dataset_prep import InferenceData, prepare_dataset, prepare_inference
from .mining import MiningConfig, mine_negatives, mining_report
from .pipeline import build_pipeline

__all__ = [
    "build_pipeline",
    "InferenceData",
    "MiningConfig",
    "mine_negatives",
    "mining_report",
    "prepare_dataset",
    "prepare_inference",
]
