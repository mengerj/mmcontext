"""Tests for config-driven multi-dataset training (src/mmcontext/training.py).

These cover the parts of the new pipeline that the single-dataset tests in
``test_st_integration.py`` do not exercise:

1. ``build_namespaced_vector_store`` — merging per-dataset stores under
   ``{name}:{id}`` keys without collisions.
2. ``assemble_training_data`` + a one-step multi-dataset
   ``SentenceTransformerTrainer`` run (dict of datasets, dict of losses,
   ``SequentialEvaluator``) mixing a bimodal omics dataset and a bio dataset.
3. End-to-end namespacing: ``omics:{name}:{id}`` anchors resolve in the merged
   store and ``model.encode`` succeeds.

All tests run offline: the conftest session fixture patches AutoModel /
AutoTokenizer with stubs (text hidden_size=32), and VectorStores are built from
synthetic numpy arrays.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import SequentialEvaluator

from mmcontext.embed import build_pipeline
from mmcontext.io import VectorStore, build_namespaced_vector_store
from mmcontext.training import (
    DatasetConfig,
    TrainConfig,
    TrainerConfig,
    assemble_training_data,
    config_from_dict,
)

OMICS_DIM = 8
SHARED_DIM = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def pipeline():
    """Obs pipeline built via the shared builder (stubbed text encoder)."""
    return build_pipeline(
        "bert-base-uncased",
        omics_dim=OMICS_DIM,
        shared_dim=SHARED_DIM,
        adapter_hidden_dim=SHARED_DIM,
        max_seq_length=8,
    )


def _make_store(ids, tmp_path, name="store", seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(ids), OMICS_DIM)).astype(np.float32)
    return VectorStore.from_numpy(data, ids, path=os.path.join(tmp_path, f"{name}.mmap")), data


def _omics_split(sample_ids):
    """A raw omics split with hard-negative idx columns referencing sample_ids."""
    n = len(sample_ids)
    return Dataset.from_dict(
        {
            "sample_idx": list(sample_ids),
            "cell_sentence_1": [f"GENE{i} GENE{(i + 1) % n}" for i in range(n)],
            "positive": [f"description of cell {i}" for i in range(n)],
            # odd negative -> resolves to positive of the referenced idx;
            # even negative -> resolves to cell_sentence_1 of the referenced idx.
            "negative_1_idx": [sample_ids[(i + 1) % n] for i in range(n)],
            "negative_2_idx": [sample_ids[(i + 2) % n] for i in range(n)],
            "adata_link": ["http://example/chunk.zarr"] * n,
        }
    )


# ---------------------------------------------------------------------------
# 1. build_namespaced_vector_store
# ---------------------------------------------------------------------------
class TestNamespacedStore:
    def test_merge_namespaces_and_preserves_vectors(self, tmp_path):
        store_a, data_a = _make_store(["0", "1"], tmp_path, name="a", seed=1)
        store_b, data_b = _make_store(["0", "1"], tmp_path, name="b", seed=2)  # same orig ids

        merged = build_namespaced_vector_store(
            {"ds_a": (store_a, ["0", "1"]), "ds_b": (store_b, ["0", "1"])},
            output_path=os.path.join(tmp_path, "merged.mmap"),
        )

        # No collisions despite overlapping original ids.
        assert len(merged) == 4
        for key in ("ds_a:0", "ds_a:1", "ds_b:0", "ds_b:1"):
            assert key in merged

        np.testing.assert_allclose(merged["ds_a:0"], data_a[0])
        np.testing.assert_allclose(merged["ds_b:1"], data_b[1])

    def test_dedup_preserves_order(self, tmp_path):
        store, data = _make_store(["0", "1", "2"], tmp_path, name="c", seed=3)
        merged = build_namespaced_vector_store(
            {"ds": (store, ["0", "0", "1", "2", "1"])},  # duplicates
            output_path=os.path.join(tmp_path, "merged_dedup.mmap"),
        )
        assert len(merged) == 3
        np.testing.assert_allclose(merged["ds:2"], data[2])

    def test_dim_mismatch_raises(self, tmp_path):
        store_a, _ = _make_store(["0"], tmp_path, name="d1", seed=4)
        other = np.random.default_rng(5).standard_normal((1, OMICS_DIM + 3)).astype(np.float32)
        store_b = VectorStore.from_numpy(other, ["0"], path=os.path.join(tmp_path, "d2.mmap"))
        with pytest.raises(ValueError, match="dim"):
            build_namespaced_vector_store(
                {"a": (store_a, ["0"]), "b": (store_b, ["0"])},
                output_path=os.path.join(tmp_path, "bad.mmap"),
            )

    def test_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            build_namespaced_vector_store({}, output_path=os.path.join(tmp_path, "x.mmap"))


# ---------------------------------------------------------------------------
# 2. assemble_training_data + multi-dataset training
# ---------------------------------------------------------------------------
class TestAssembleAndTrain:
    def _cfg(self):
        return TrainConfig(
            obsm_key="X_scvi_fm",
            shared_dim=SHARED_DIM,
            omics_datasets=[
                DatasetConfig(id="dummy/a", name="omics_a", type="multiplets", modality="bimodal"),
            ],
            bio_datasets=[
                DatasetConfig(id="dummy/bio", name="bio_x", type="multiplets"),
            ],
            trainer=TrainerConfig(per_device_eval_batch_size=2, logging_steps=1),
        )

    def _omics_raw(self):
        return {
            "omics_a": DatasetDict(
                train=_omics_split(["0", "1", "2", "3"]),
                val=_omics_split(["0", "1"]),
            )
        }

    def _bio_raw(self):
        return {
            "bio_x": DatasetDict(
                train=Dataset.from_dict(
                    {
                        "anchor": ["a neuron", "a B cell", "a fibroblast"],
                        "positive": ["SYT1+ neuron", "CD19+ B cell", "COL1A1+ fibroblast"],
                        "negative": ["unrelated 1", "unrelated 2", "unrelated 3"],  # singular name
                        "sample_id": ["x", "y", "z"],  # stray column that must be dropped
                    }
                )
            )
        }

    def test_assemble_shapes(self, pipeline, tmp_path):
        merged = build_namespaced_vector_store(
            {"omics_a": (_make_store(["0", "1", "2", "3"], tmp_path, name="oa")[0], ["0", "1", "2", "3"])},
            output_path=os.path.join(tmp_path, "merged.mmap"),
        )
        pipeline[0].set_vector_store(merged)

        assembled = assemble_training_data(
            self._cfg(), pipeline, self._omics_raw(), self._bio_raw(), log_backend="none"
        )

        assert set(assembled.train_datasets) == {"omics_a", "bio_x"}
        assert set(assembled.losses) == {"omics_a", "bio_x"}
        # Both datasets contribute an evaluator (omics from val, bio from train slice).
        assert len(assembled.evaluators) == 2

        # Omics anchors are namespaced omics ids; bio anchors are plain text.
        assert assembled.train_datasets["omics_a"]["anchor"][0] == "omics:omics_a:0"
        assert assembled.train_datasets["bio_x"]["anchor"][0] == "a neuron"

        # Bio dataset keeps anchor/positive/negative and drops the stray column.
        assert set(assembled.train_datasets["bio_x"].column_names) == {"anchor", "positive", "negative"}

    def test_multi_dataset_training_step(self, pipeline, tmp_path):
        merged = build_namespaced_vector_store(
            {"omics_a": (_make_store(["0", "1", "2", "3"], tmp_path, name="oa2")[0], ["0", "1", "2", "3"])},
            output_path=os.path.join(tmp_path, "merged2.mmap"),
        )
        pipeline[0].set_vector_store(merged)

        assembled = assemble_training_data(
            self._cfg(), pipeline, self._omics_raw(), self._bio_raw(), log_backend="none"
        )
        evaluator = SequentialEvaluator(assembled.evaluators)

        args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(tmp_path, "run"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            use_cpu=True,
            report_to="none",
        )
        trainer = SentenceTransformerTrainer(
            model=pipeline,
            args=args,
            train_dataset=assembled.train_datasets,
            loss=assembled.losses,
            evaluator=evaluator,
        )

        before = {n: p.clone() for n, p in pipeline.named_parameters() if p.requires_grad}
        trainer.train()
        delta = sum(
            (p - before[n]).abs().sum().item() for n, p in pipeline.named_parameters() if n in before
        )
        assert delta > 0, "No parameters changed during multi-dataset training"


# ---------------------------------------------------------------------------
# 3. Namespacing resolves end-to-end
# ---------------------------------------------------------------------------
class TestNamespacingResolves:
    def test_anchors_resolve_in_store(self, pipeline, tmp_path):
        merged = build_namespaced_vector_store(
            {"omics_a": (_make_store(["0", "1", "2", "3"], tmp_path, name="oa3")[0], ["0", "1", "2", "3"])},
            output_path=os.path.join(tmp_path, "merged3.mmap"),
        )
        pipeline[0].set_vector_store(merged)

        cfg = TrainConfig(
            omics_datasets=[DatasetConfig(id="dummy/a", name="omics_a", modality="bimodal")],
            trainer=TrainerConfig(per_device_eval_batch_size=2, logging_steps=1),
        )
        omics_raw = {"omics_a": DatasetDict(train=_omics_split(["0", "1", "2", "3"]))}
        assembled = assemble_training_data(cfg, pipeline, omics_raw, log_backend="none")

        anchors = assembled.train_datasets["omics_a"]["anchor"]
        for anchor in anchors:
            assert anchor.startswith("omics:omics_a:")
            assert anchor[len("omics:") :] in merged  # store key present

        emb = pipeline.encode(anchors)
        assert emb.shape == (len(anchors), SHARED_DIM)


# ---------------------------------------------------------------------------
# 4. Config parsing
# ---------------------------------------------------------------------------
class TestConfig:
    def test_config_from_dict(self):
        cfg = config_from_dict(
            {
                "obsm_key": "X_pca",
                "shared_dim": 128,
                "text_encoder": {"name": "some/model", "max_seq_length": 256},
                "omics_datasets": [{"id": "a/b", "name": "d1", "modality": "bimodal"}],
                "bio_datasets": [{"id": "c/d", "name": "bio", "revision": "main"}],
                "trainer": {"num_train_epochs": 5},
            }
        )
        assert cfg.obsm_key == "X_pca"
        assert cfg.shared_dim == 128
        assert cfg.text_encoder.max_seq_length == 256
        assert cfg.omics_datasets[0].name == "d1"
        assert cfg.bio_datasets[0].revision == "main"
        assert cfg.trainer.num_train_epochs == 5

    def test_config_ignores_unknown_keys(self):
        cfg = config_from_dict({"obsm_key": "X_pca", "bogus_key": 123})
        assert cfg.obsm_key == "X_pca"
        assert not hasattr(cfg, "bogus_key")
