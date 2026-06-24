#!/usr/bin/env python
"""Explore which hard negatives a trained model mines for each config dataset.

This is an *offline inspection* tool, not a training script. It reuses the
``scripts/train_config.py`` machinery (``load_config``, ``load_raw_datasets``,
``build_merged_vector_store``) to set up exactly the datasets + merged
VectorStore a real run would use, then mines model-specific hard negatives per
dataset with :func:`mmcontext.embed.mine_negatives` and writes a readable report
(:func:`mmcontext.embed.mining_report`) of what got chosen and how similar it is
to the true positive.

Mining is run **per dataset** (not over a merged corpus) because the
sentence-transformers trainer samples each batch from a single dataset, so
in-batch and mined negatives should both stay within-dataset.

You must point at an *already-trained* model — mining against a freshly
initialised bimodal model produces noise.

Usage::

    python scripts/explore_hard_negatives.py \
        --config conf/training/multi_example.yaml \
        --model jo-mengr/<some-trained-model> \
        --output-dir outputs/hard_neg_exploration \
        --num-negatives 3 --range-min 1 --max-score 0.95

Outputs (under --output-dir):
    <dataset>__mined.csv     per-anchor positive + mined negatives + sims/margins
    summary.md               score distributions + a few examples per dataset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from mmcontext.embed import MiningConfig, mine_negatives, mining_report, prepare_dataset
from mmcontext.training import (
    DatasetConfig,
    TrainConfig,
    _namespace_sample_ids,
    _select_training_columns,
    build_merged_vector_store,
    load_config,
    load_raw_datasets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore mined hard negatives for the config's datasets")
    parser.add_argument("--config", required=True, help="Path to the YAML training config")
    parser.add_argument(
        "--model",
        required=True,
        help="HF repo / local path of an ALREADY-TRAINED model to mine with "
        "(mining a fresh bimodal model produces noise).",
    )
    parser.add_argument("--output-dir", default="outputs/hard_neg_exploration", help="Where to write the report")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Subsample each dataset to the first N anchor/positive pairs before "
        "mining (also shrinks the corpus). Default: use the full dataset.",
    )
    parser.add_argument(
        "--report-rows",
        type=int,
        default=50,
        help="Rows per dataset to recompute similarities for in the CSV (default: 50).",
    )
    parser.add_argument("--overwrite-store", action="store_true", help="Rebuild cached vector stores")

    mining = parser.add_argument_group("hard-negative mining")
    mining.add_argument("--num-negatives", type=int, default=3)
    mining.add_argument("--range-min", type=int, default=1)
    mining.add_argument("--range-max", type=int, default=None)
    mining.add_argument("--max-score", type=float, default=0.95)
    mining.add_argument("--min-score", type=float, default=None)
    mining.add_argument("--absolute-margin", type=float, default=None)
    mining.add_argument("--relative-margin", type=float, default=None)
    mining.add_argument("--sampling-strategy", choices=["top", "random"], default="top")
    mining.add_argument("--use-faiss", action=argparse.BooleanOptionalAction, default=False)
    mining.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _prepare_omics_pairs(dcfg: DatasetConfig, dsdict):
    """Return an anchor/positive ``datasets.Dataset`` for one omics dataset."""
    train_split = dsdict["train"]
    if dcfg.modality == "bimodal":
        train_split = _namespace_sample_ids(train_split, dcfg.name)
    # Drop the dataset's own pre-baked negatives so we mine fresh, model-specific ones.
    return prepare_dataset(
        train_split,
        purpose="train",
        modality=dcfg.modality,
        primary_cell_sentence=dcfg.primary_cell_sentence,
        positive_col=dcfg.positive_col,
        use_hard_negatives=False,
    )


def _summarise(name: str, report: pd.DataFrame) -> dict:
    """Compute score/margin stats for one dataset's report."""
    neg_sim_cols = [c for c in report.columns if c.startswith("neg_") and c.endswith("_sim")]
    margin_cols = [c for c in report.columns if c.startswith("margin_")]
    neg_sims = report[neg_sim_cols].to_numpy().ravel() if neg_sim_cols else []
    margins = report[margin_cols].to_numpy().ravel() if margin_cols else []
    stat = {"dataset": name, "n_reported": len(report)}
    if len(report):
        stat["pos_sim_mean"] = round(float(report["pos_sim"].mean()), 4)
    if len(neg_sims):
        stat["neg_sim_mean"] = round(float(neg_sims.mean()), 4)
        stat["neg_sim_max"] = round(float(neg_sims.max()), 4)
    if len(margins):
        stat["margin_mean"] = round(float(margins.mean()), 4)
        stat["margin_min"] = round(float(margins.min()), 4)
    return stat


def main() -> None:
    """Mine and report hard negatives for every dataset in the config."""
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg: TrainConfig = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    omics_raw, bio_raw = load_raw_datasets(cfg)
    overwrite = args.overwrite_store or cfg.overwrite_vector_store
    merged_store, _ = build_merged_vector_store(cfg, omics_raw, overwrite=overwrite)

    logger.info("Loading model %s", args.model)
    model = SentenceTransformer(args.model, trust_remote_code=True)
    if merged_store is not None:
        model[0].set_vector_store(merged_store)
        logger.info("Attached merged VectorStore: %d vectors, dim=%d", len(merged_store), merged_store.dim)

    mining_cfg = MiningConfig(
        num_negatives=args.num_negatives,
        range_min=args.range_min,
        range_max=args.range_max,
        max_score=args.max_score,
        min_score=args.min_score,
        absolute_margin=args.absolute_margin,
        relative_margin=args.relative_margin,
        sampling_strategy=args.sampling_strategy,
        use_faiss=args.use_faiss,
        batch_size=args.batch_size,
    )
    logger.info("Mining config: %s", mining_cfg)

    # (name, anchor/positive pairs) for every dataset in the config.
    jobs: list[tuple[str, object]] = []
    for dcfg in cfg.omics_datasets:
        jobs.append((dcfg.name, _prepare_omics_pairs(dcfg, omics_raw[dcfg.name])))
    for dcfg in cfg.bio_datasets:
        if "train" not in bio_raw[dcfg.name]:
            logger.warning("Bio dataset '%s' has no train split, skipping", dcfg.name)
            continue
        jobs.append((dcfg.name, _select_training_columns(bio_raw[dcfg.name]["train"], dcfg.name)))

    stats: list[dict] = []
    for name, pairs in jobs:
        if args.max_pairs is not None and len(pairs) > args.max_pairs:
            pairs = pairs.select(range(args.max_pairs))
        logger.info("[%s] mining over %d pairs", name, len(pairs))
        mined = mine_negatives(pairs, model, mining_cfg, output_format="n-tuple", verbose=True)
        report = mining_report(mined, model, max_rows=args.report_rows)
        csv_path = out_dir / f"{name}__mined.csv"
        report.to_csv(csv_path, index=False)
        logger.info("[%s] wrote %s (%d rows; %d mined total)", name, csv_path, len(report), len(mined))
        stats.append({**_summarise(name, report), "n_pairs": len(pairs), "n_mined": len(mined)})

    _write_summary(out_dir / "summary.md", mining_cfg, stats)
    logger.info("Done. Report written to %s", out_dir)


def _write_summary(path: Path, mining_cfg: MiningConfig, stats: list[dict]) -> None:
    """Write a markdown summary with the score distributions and example rows."""
    lines = ["# Hard-negative mining exploration", "", "## Mining config", "", "```", repr(mining_cfg), "```", ""]
    summary_table = pd.DataFrame(stats).to_string(index=False) if stats else "(no datasets)"
    lines += ["## Per-dataset summary", "", "```", summary_table, "```", ""]
    lines += [
        "## Reading the numbers",
        "",
        "- `pos_sim`: cosine(anchor, true positive). Higher = the model already aligns the pair.",
        "- `neg_*_sim`: cosine(anchor, mined negative). Close to `pos_sim` ⇒ a genuinely hard negative.",
        "- `margin_*` = `pos_sim − neg_sim`. Small/negative margins are the hardest (and the most",
        "  likely false negatives — check that `max_score` is filtering those out).",
        "",
        "Per-dataset CSVs with every reported anchor/positive/negative are in this directory.",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
