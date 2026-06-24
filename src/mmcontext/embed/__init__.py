"""mmcontext.embed — Pipeline building, dataset preparation, embedding, and hard-negative mining."""

from .dataset_prep import InferenceData, prepare_dataset, prepare_inference
from .dataset_utils import load_generic_dataset
from .encode import create_label_dataset, embed_labels, load_st_model, prepare_model_and_embed
from .mining import MiningConfig, mine_negatives, mining_report
from .pipeline import build_pipeline

__all__ = [
    "build_pipeline",
    "create_label_dataset",
    "embed_labels",
    "InferenceData",
    "load_generic_dataset",
    "load_st_model",
    "MiningConfig",
    "mine_negatives",
    "mining_report",
    "prepare_dataset",
    "prepare_inference",
    "prepare_model_and_embed",
]
