"""Config-driven multi-dataset training helpers for the ST pipeline.

This module backs ``scripts/train_config.py``. It keeps the script thin by
encapsulating the reusable, testable parts of multi-dataset training on the
sentence-transformers v5.4+ pipeline (``MMContextModule`` + ``AdapterModule`` +
``Pooling`` + ``Normalize``):

* :class:`TrainConfig` and :func:`load_config` — a plain-YAML config (no Hydra).
* :func:`build_merged_vector_store` — build one namespaced
  :class:`~mmcontext.io.VectorStore` covering every bimodal omics dataset.
* :func:`assemble_training_data` — turn raw datasets into the
  ``train_dataset`` / ``eval_dataset`` / ``loss`` dicts (+ evaluators) that
  :class:`~sentence_transformers.SentenceTransformerTrainer` expects.

Omics datasets are processed via :func:`~mmcontext.embed.prepare_dataset`; bio
datasets are assumed to already be in ``anchor`` / ``positive`` / ``negative_*``
form and are passed through unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict
    from sentence_transformers import SentenceTransformer

    from mmcontext.io import VectorStore

logger = logging.getLogger(__name__)

NAMESPACE_SEP = ":"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """One training dataset entry (omics or bio)."""

    id: str
    name: str
    type: str = "multiplets"  # -> get_loss / get_evaluator
    # Omics-only fields
    modality: str = "bimodal"  # "bimodal" | "text"
    primary_cell_sentence: str = "cell_sentence_1"
    positive_col: str = "positive"
    use_hard_negatives: bool = True
    truncate: bool = False
    truncate_kwargs: dict | None = None
    # Bio-only field
    revision: str | None = None


@dataclass
class TextEncoderConfig:
    """Text encoder settings."""

    name: str = "NeuML/pubmedbert-base-embeddings"
    max_seq_length: int = 512


@dataclass
class AdapterFreezingConfig:
    """Per-modality adapter freeze/unfreeze schedule (maps to UnfreezeAdapterCallback)."""

    freeze_text_adapter: bool = False
    freeze_omics_adapter: bool = False
    unfreeze_text_adapter_epoch: float | None = None
    unfreeze_omics_adapter_epoch: float | None = None


@dataclass
class TrainerConfig:
    """SentenceTransformerTrainingArguments-facing settings."""

    output_dir: str = "outputs/multi"
    num_train_epochs: float = 2
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    bf16: bool = False
    fp16: bool = False
    eval_strategy: str = "steps"
    eval_steps: int | None = 200
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    logging_steps: int = 50
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    use_mps_device: bool = False
    use_cosine_scheduler: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None


@dataclass
class TrainConfig:
    """Top-level training config (one YAML file)."""

    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    model: str | None = None  # HF repo / local path to continue from, or None
    tag: str | None = None
    push_to_hub: bool = False

    obsm_key: str = "X_scvi_fm"
    shared_dim: int = 512
    adapter_hidden_dim: int | None = None

    freeze_text_encoder: bool = False
    unfreeze_last_n_layers: int = 0
    unfreeze_epoch: float = 1
    adapter_freezing: AdapterFreezingConfig | None = None

    gene_special_token: str | None = None
    delimiter: str = " "

    cache_dir: str | None = None
    adata_cache_dir: str = "data/from_nxtcloud"
    vector_store_dir: str = "data/vector_stores"
    force_refresh_cache: bool = False  # re-download HF datasets
    overwrite_vector_store: bool = False  # rebuild the VectorStore (e.g. after switching obsm_key)

    omics_datasets: list[DatasetConfig] = field(default_factory=list)
    bio_datasets: list[DatasetConfig] = field(default_factory=list)


def _coerce_scalar(value: Any, type_str: str) -> Any:
    """Coerce a scalar to the field's primitive type from its annotation string.

    Handles the common YAML gotcha where ``2e-5`` (no decimal point) is parsed
    as a string rather than a float. Annotations are strings here because this
    module uses ``from __future__ import annotations``.
    """
    if value is None:
        return None
    if type_str.startswith("bool"):
        return value
    try:
        if "float" in type_str:
            return float(value)
        if "int" in type_str:
            return int(value)
    except (TypeError, ValueError):
        return value
    return value


def _filter_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that are fields of *cls* (warn on the rest), coercing scalars."""
    fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
    unknown = set(data) - set(fields)
    if unknown:
        logger.warning("Ignoring unknown config keys for %s: %s", cls.__name__, sorted(unknown))
    return {k: _coerce_scalar(v, str(fields[k].type)) for k, v in data.items() if k in fields}


def config_from_dict(data: dict[str, Any]) -> TrainConfig:
    """Build a :class:`TrainConfig` from a plain (YAML-loaded) dict."""
    data = dict(data or {})

    text_encoder = TextEncoderConfig(**_filter_kwargs(TextEncoderConfig, data.pop("text_encoder", {}) or {}))
    trainer = TrainerConfig(**_filter_kwargs(TrainerConfig, data.pop("trainer", {}) or {}))

    adapter_freezing_raw = data.pop("adapter_freezing", None)
    adapter_freezing = (
        AdapterFreezingConfig(**_filter_kwargs(AdapterFreezingConfig, adapter_freezing_raw))
        if adapter_freezing_raw
        else None
    )

    omics = [DatasetConfig(**_filter_kwargs(DatasetConfig, d)) for d in (data.pop("omics_datasets", None) or [])]
    bio = [DatasetConfig(**_filter_kwargs(DatasetConfig, d)) for d in (data.pop("bio_datasets", None) or [])]

    return TrainConfig(
        text_encoder=text_encoder,
        trainer=trainer,
        adapter_freezing=adapter_freezing,
        omics_datasets=omics,
        bio_datasets=bio,
        **_filter_kwargs(TrainConfig, data),
    )


def load_config(path: str | Path) -> TrainConfig:
    """Load and validate a YAML training config.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.

    Returns
    -------
    TrainConfig
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return config_from_dict(raw)


# ---------------------------------------------------------------------------
# Raw dataset loading
# ---------------------------------------------------------------------------
def load_raw_datasets(cfg: TrainConfig) -> tuple[dict[str, DatasetDict], dict[str, DatasetDict]]:
    """Load every omics and bio dataset referenced by *cfg* from the Hub.

    Parameters
    ----------
    cfg : TrainConfig
        Run config.

    Returns
    -------
    tuple[dict[str, DatasetDict], dict[str, DatasetDict]]
        ``(omics_raw, bio_raw)`` keyed by ``DatasetConfig.name``.
    """
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    download_mode = "force_redownload" if cfg.force_refresh_cache else "reuse_dataset_if_exists"

    omics_raw: dict[str, DatasetDict] = {}
    for dcfg in cfg.omics_datasets:
        logger.info("Loading omics dataset '%s' from %s", dcfg.name, dcfg.id)
        omics_raw[dcfg.name] = load_dataset(dcfg.id, cache_dir=cfg.cache_dir, download_mode=download_mode)

    bio_raw: dict[str, DatasetDict] = {}
    for dcfg in cfg.bio_datasets:
        logger.info("Loading bio dataset '%s' from %s (revision=%s)", dcfg.name, dcfg.id, dcfg.revision)
        try:
            bio_raw[dcfg.name] = load_dataset(
                dcfg.id, revision=dcfg.revision, cache_dir=cfg.cache_dir, download_mode=download_mode
            )
        except DatasetNotFoundError as e:
            if dcfg.revision is None:
                raise
            # The pinned revision (e.g. "hard_negatives") may not exist for every
            # dataset; the main branch is expected to carry a negatives column too.
            logger.warning(
                "Revision %r not found for bio dataset '%s' (%s); falling back to the default branch.",
                dcfg.revision,
                dcfg.name,
                e,
            )
            bio_raw[dcfg.name] = load_dataset(dcfg.id, cache_dir=cfg.cache_dir, download_mode=download_mode)

    return omics_raw, bio_raw


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def build_merged_vector_store(
    cfg: TrainConfig,
    omics_raw: dict[str, DatasetDict],
    *,
    overwrite: bool = False,
) -> tuple[VectorStore | None, int | None]:
    """Build one namespaced VectorStore covering all bimodal omics datasets.

    Each per-dataset store is built on the **original** ``sample_idx`` values
    (so the lookup against ``adata.obs_names`` succeeds), then re-keyed to
    ``{name}:{sample_idx}`` and merged. Returns ``(None, None)`` when no omics
    dataset uses the bimodal modality.

    Parameters
    ----------
    cfg : TrainConfig
        Run config (uses ``obsm_key``, ``adata_cache_dir``, ``vector_store_dir``).
    omics_raw : dict[str, DatasetDict]
        Raw omics datasets keyed by ``DatasetConfig.name``.
    overwrite : bool, default False
        Rebuild per-dataset stores even if a cached ``.mmap`` exists.

    Returns
    -------
    tuple[VectorStore or None, int or None]
        The merged store and its dimensionality, or ``(None, None)``.
    """
    from datasets import concatenate_datasets

    from mmcontext.io import build_namespaced_vector_store, prepare_vector_store

    bimodal = [d for d in cfg.omics_datasets if d.modality == "bimodal"]
    if not bimodal:
        return None, None

    store_dir = Path(cfg.vector_store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    # Encode obsm_key in the filename so switching keys on the same data uses a
    # distinct cache file instead of silently reusing the previous key's vectors.
    key_slug = cfg.obsm_key.replace("/", "_")
    per_dataset: dict[str, tuple[VectorStore, list[str]]] = {}
    for dcfg in bimodal:
        dsdict = omics_raw[dcfg.name]
        splits = list(dsdict.values())
        combined = concatenate_datasets(splits) if len(splits) > 1 else splits[0]
        store = prepare_vector_store(
            combined,
            obsm_key=cfg.obsm_key,
            output_path=store_dir / f"{dcfg.name}__{key_slug}.mmap",
            cache_dir=cfg.adata_cache_dir,
            overwrite=overwrite,
        )
        orig_ids = [str(x) for x in combined["sample_idx"]]
        per_dataset[dcfg.name] = (store, orig_ids)

    merged = build_namespaced_vector_store(
        per_dataset,
        output_path=store_dir / "merged.mmap",
        namespace_sep=NAMESPACE_SEP,
    )
    return merged, merged.dim


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
@dataclass
class AssembledData:
    """Container for the dicts the trainer consumes."""

    train_datasets: dict[str, Dataset]
    eval_datasets: dict[str, Dataset]
    losses: dict[str, Any]
    evaluators: list[Any]


def _pick_val_split(dsdict: DatasetDict) -> Dataset | None:
    """Return the validation split of a DatasetDict, if present."""
    for key in ("val", "validation", "dev", "test"):
        if key in dsdict:
            return dsdict[key]
    return None


def _namespace_sample_ids(ds: Dataset, name: str, sample_id_col: str = "sample_idx") -> Dataset:
    """Prefix each ``sample_idx`` with ``{name}:`` so anchors match store keys."""
    return ds.map(
        lambda row: {sample_id_col: f"{name}{NAMESPACE_SEP}{row[sample_id_col]}"},
        desc=f"Namespacing sample_idx for {name}",
    )


def _select_training_columns(ds: Dataset, name: str) -> Dataset:
    """Keep only ``anchor``, ``positive`` and ``negative*`` columns.

    The ST trainer feeds every dataset column to the loss as text, so stray
    metadata columns would be misread as extra negatives. Bio datasets are
    passed through raw, so this drops anything that is not a training input.
    A single ``negative`` column is supported by both the loss and the
    TripletEvaluator (which prefers ``negative`` over ``negative_1``).
    """
    keep = [c for c in ds.column_names if c in ("anchor", "positive") or c.startswith("negative")]
    missing = {"anchor", "positive"} - set(keep)
    if missing:
        raise KeyError(
            f"Bio dataset '{name}' is missing required column(s) {sorted(missing)}. "
            f"Available columns: {ds.column_names}"
        )
    dropped = [c for c in ds.column_names if c not in keep]
    if dropped:
        logger.info("Bio dataset '%s': keeping %s, dropping %s", name, keep, dropped)
    return ds.select_columns(keep)


def assemble_training_data(
    cfg: TrainConfig,
    model: SentenceTransformer,
    omics_raw: dict[str, DatasetDict],
    bio_raw: dict[str, DatasetDict] | None = None,
    *,
    log_backend: str = "wandb",
) -> AssembledData:
    """Build the train/eval/loss dicts (+ evaluators) for multi-dataset training.

    Omics datasets are namespaced (bimodal) and passed through
    :func:`~mmcontext.embed.prepare_dataset`. Bio datasets are assumed to be in
    final ``anchor`` / ``positive`` / ``negative_*`` form and passed through;
    a small slice of their train split is reused for evaluation.

    Parameters
    ----------
    cfg : TrainConfig
        Run config.
    model : SentenceTransformer
        Built/loaded pipeline (its first module must already have the merged
        VectorStore attached for the bimodal path); passed to ``get_loss``.
    omics_raw : dict[str, DatasetDict]
        Raw omics datasets keyed by ``DatasetConfig.name``.
    bio_raw : dict[str, DatasetDict], optional
        Raw bio datasets keyed by ``DatasetConfig.name``.
    log_backend : str, default "wandb"
        Forwarded to :func:`~mmcontext.utils.get_loss`.

    Returns
    -------
    AssembledData
        ``train_datasets``, ``eval_datasets``, ``losses``, ``evaluators``.
    """
    from mmcontext.embed import prepare_dataset
    from mmcontext.utils import get_evaluator, get_loss

    bio_raw = bio_raw or {}
    train_datasets: dict[str, Dataset] = {}
    eval_datasets: dict[str, Dataset] = {}
    losses: dict[str, Any] = {}
    evaluators: list[Any] = []
    logging_steps = cfg.trainer.logging_steps
    eval_bs = cfg.trainer.per_device_eval_batch_size

    # -- Omics datasets -------------------------------------------------------
    for dcfg in cfg.omics_datasets:
        dsdict = omics_raw[dcfg.name]
        train_split = dsdict["train"]
        val_split = _pick_val_split(dsdict)

        if dcfg.modality == "bimodal":
            train_split = _namespace_sample_ids(train_split, dcfg.name)
            if val_split is not None:
                val_split = _namespace_sample_ids(val_split, dcfg.name)

        prep_kwargs = {
            "purpose": "train",
            "modality": dcfg.modality,
            "primary_cell_sentence": dcfg.primary_cell_sentence,
            "positive_col": dcfg.positive_col,
            "use_hard_negatives": dcfg.use_hard_negatives,
            "truncate": dcfg.truncate,
            "truncate_kwargs": dcfg.truncate_kwargs,
        }
        train_ready = prepare_dataset(train_split, **prep_kwargs)
        train_datasets[dcfg.name] = train_ready

        losses[dcfg.name] = get_loss(
            model=model,
            dataset_type=dcfg.type,
            dataset_name=dcfg.name,
            log_backend=log_backend,
            logging_steps=logging_steps,
        )

        if val_split is not None:
            val_ready = prepare_dataset(val_split, **prep_kwargs)
            eval_datasets[dcfg.name] = val_ready
            evaluators.append(
                get_evaluator(
                    dataset_type=dcfg.type,
                    dataset=val_ready,
                    batch_size=eval_bs,
                    current_eval_name=dcfg.name,
                )
            )

    # -- Bio datasets (already anchor/positive/negative) ----------------------
    for dcfg in cfg.bio_datasets:
        dsdict = bio_raw[dcfg.name]
        if "train" not in dsdict:
            logger.warning("Bio dataset '%s' has no 'train' split, skipping", dcfg.name)
            continue
        train_split = _select_training_columns(dsdict["train"], dcfg.name)
        train_datasets[dcfg.name] = train_split

        losses[dcfg.name] = get_loss(
            model=model,
            dataset_type=dcfg.type,
            dataset_name=dcfg.name,
            log_backend=log_backend,
            logging_steps=logging_steps,
        )

        # Reuse a slice of train as a sanity-check eval set (mirrors old train.py).
        val_size = min(1000, len(train_split))
        val_split = train_split.select(range(val_size))
        eval_datasets[dcfg.name] = val_split
        evaluators.append(
            get_evaluator(
                dataset_type=dcfg.type,
                dataset=val_split,
                batch_size=eval_bs,
                current_eval_name=dcfg.name,
            )
        )

    return AssembledData(
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        losses=losses,
        evaluators=evaluators,
    )
