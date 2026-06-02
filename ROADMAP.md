# MMContext Refactor — Roadmap & Technical Notes

This document captures the state of the codebase after the sentence-transformers
v5.4+ refactor (Phases 1–5), identifies remaining work, and provides context for
future development. It is structured as a reference for both human contributors
and AI-assisted editing sessions.

---

## 1 Architecture overview (post-refactor)

The new pipeline lives in two packages:

| Package             | Role                                                                                          |
| ------------------- | --------------------------------------------------------------------------------------------- |
| `mmcontext.modules` | ST pipeline modules: `MMContextModule` (InputModule), `AdapterModule`, `OmicsAttentionModule` |
| `mmcontext.io`      | `VectorStore` (mmap-backed lookup), `prepare_vector_store` (zarr → store builder)             |

A trained model is a standard `SentenceTransformer` with this module chain:

```
MMContextModule → [OmicsAttentionModule] → AdapterModule → Pooling → Normalize
```

`OmicsAttentionModule` is optional — include it for var-level (gene-vector)
inputs where the sequence length > 1; omit it for obs-level (single cell
embedding) inputs where it would degenerate to a feedforward layer.

### Key design decisions

- **`input_values` preprocess key** — omics preprocess writes `input_values`
  (not `token_embeddings`) so the ST training collator's `collect_features`
  suffix matching detects omics columns. Forward reads `input_values` and
  writes `token_embeddings` for downstream modules.
- **`modality_ids` tensor** — 0 = text, 1 = omics, 2 = pad. The
  `AdapterModule` uses this to route tokens through the correct projection head.
- **No Trainer subclass** — training uses the standard
  `SentenceTransformerTrainer` + `MultipleNegativesRankingLoss` with
  `(anchor, positive)` column layout.

---

## 2 What was completed (Phases 1–5)

| Phase | Deliverable                                                                                                                             | Tests                                                         |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1     | `VectorStore` — mmap-backed vector lookup with `from_numpy`, `from_adata`, `from_dict`, `load`                                          | `tests/test_vector_store.py` (in mmcontext_module tests)      |
| 2     | `MMContextModule` — text encoding via AutoModel/AutoTokenizer, omics via VectorStore or direct vectors, `preprocess`/`forward` contract | `tests/test_mmcontext_module.py`                              |
| 3     | `AdapterModule` — modality-aware projection (text head, omics head, shared dim), safetensors persistence                                | `tests/test_adapter_module.py`                                |
| 4     | `OmicsAttentionModule` — optional self-attention over variable-length omics sequences                                                   | `tests/test_omics_attention_module.py`                        |
| 5     | Full ST integration — pipeline construction, encode, save/load, training (text-only, bimodal, gene-list), `prepare_vector_store`        | `tests/test_st_integration.py`, `tests/test_prepare_store.py` |

### Supporting files

- `scripts/train_tiny.py` — end-to-end training script for `jo-mengr/cxg_schaefer_tiny` with wandb, MPS, and auto VectorStore preparation.
- `src/mmcontext/io/prepare_store.py` — memory-efficient store builder that reads only needed obsm rows from zarr (never loads full AnnData).

---

## 3 Legacy code (`_legacy/`)

The following modules were moved to `src/mmcontext/_legacy/` and are preserved
for backward compatibility with previously trained models:

| File                           | What it was                                                                  | Notes                                                                                                                 |
| ------------------------------ | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `mmcontextencoder.py`          | Dual-tower encoder (text + omics), `MMContextProcessor` tokenizer            | Central old architecture; `embed/model_utils.py` still imports `MMEnc.get_initial_embeddings_from_adata_link` from it |
| `adapters.py`                  | MLP adapter with identity/linear/2-layer modes                               | Superseded by `modules/adapter_module.py` which adds modality-awareness                                               |
| `omicsencoder.py`              | `MiniOmicsModel` — `nn.Embedding` lookup with HF `PreTrainedModel` interface | Superseded by `VectorStore` (no learnable embedding, just mmap lookup)                                                |
| `onehot.py`                    | `OneHotTextEncoder` — learnable embedding per unique sentence                | Useful for ablation experiments; no new equivalent                                                                    |
| `cell_sentence_transformer.py` | `OmicsEncoder` — full transformer over omics tokens with cross-attention     | Heavier than `OmicsAttentionModule`; cross-attention is not yet in new code                                           |

### Legacy tests (still functional)

These test files import from `_legacy` and exercise the old architecture. They
should continue to pass for backward compatibility:

- `test_mmcontext_encoder.py`
- `test_mmcontext_tokenizer.py`
- `test_miniOmicsEncoder.py`
- `test_register_initial_embeddings.py`
- `test_sentence_transformer_integration.py` (old integration tests)

---

## 4 Active non-module code — status & required changes

### `callback.py` — trainer callbacks

**Status:** Functional but coupled to old `MMContextEncoder` attribute paths
(`model[0].text_encoder`, `model[0].text_adapter`, `model[0].omics_adapter`).

**What needs to change:**

- `UnfreezeTextEncoderCallback` should look for `model[0].auto_model` (the
  `MMContextModule`'s text encoder) instead of `model[0].text_encoder`.
- `UnfreezeAdapterCallback` should look for the `AdapterModule` by iterating
  `model.children()` and checking `isinstance(m, AdapterModule)`, since it's
  no longer at `model[0]` — it's `model[1]` (or `model[2]` if
  OmicsAttentionModule is included).
- Consider making freeze/unfreeze methods on the modules themselves
  (`MMContextModule.freeze_text_encoder()` already exists;
  `AdapterModule` could add `freeze_text_head()` / `freeze_omics_head()`).

### `utils.py` — training utilities

**Useful functions that work with new architecture as-is:**

- `truncate_cell_sentences()` — gene-name truncation/filtering for HF datasets.
  Domain-specific and actively needed for dataset preprocessing.
- `truncate_semantic_cell_sentences_dataset()` — semantic-aware truncation.
- `resolve_negative_indices_and_rename()` — resolves `negative_*_idx` columns
  to actual text values. Needed for multiplet training with the real dataset.
- `get_evaluator()` — instantiates ST evaluators (BinaryClassification,
  Triplet) from dataset columns. Works with any ST model.
- `consolidate_low_frequency_categories()` — groups rare labels into "other".
- `get_device()` — simple CUDA/MPS/CPU device selection.

**Functions that reference old architecture:**

- `get_loss()` (both overloads) — hardcodes dataset_type → loss mapping.
  Logic is fine, but should be reviewed for new column naming conventions.
- `prepare_omics_resources()` — builds lookup dict with `"sample_idx:"` prefix.
  Superseded by `prepare_vector_store()` with `"omics:"` prefix, but the old
  function may still be needed for legacy model inference.

### `hub_utils.py` — HuggingFace Hub upload

**Status:** Template body references old module names. The upload logic
(`HfApi` calls) is architecture-agnostic and works.

**What needs to change:** Update the `MODEL_CARD_TEMPLATE` string to describe
the new pipeline modules.

### `file_utils.py` — I/O infrastructure

**Status:** Actively used by both old and new code.

**Key functions:**

- `download_and_extract_links()` — full-featured download with caching,
  Zenodo support, resume, retries. The new `io/prepare_store.py` has a
  lighter `_download_zarr()` that doesn't support all features. Consider
  converging on one implementation eventually.
- `remove_corrupted_null_arrays()` — zarr repair utility; used by
  `embed/dataset_utils.py`.
- `collect_unique_links()` — deduplicates adata links from a HF dataset.
- `load_test_adata()` — downloads + loads a test adata from HF.
- `save_table()` — generic table saver (CSV/parquet/AnnData).

---

## 5 Evaluation framework — analysis & adaptation needs

The `eval/` package is a **self-contained evaluation framework** with a
decorator-based registry pattern. It is largely architecture-agnostic — most
evaluators work on AnnData `.obsm` embeddings, not on the model directly.

### Architecture

```
eval/registry.py      — @register decorator, get(name) lookup
eval/base.py          — BaseEvaluator (abstract), EvalResult
eval/eval_pipeline.py — orchestrator: discover evaluators, run on datasets
eval/utils.py         — LabelKind, LabelSpec helpers
```

### Evaluators

| Evaluator             | Registry name                | What it measures                                  | Architecture dependency                          |
| --------------------- | ---------------------------- | ------------------------------------------------- | ------------------------------------------------ |
| `ARI`                 | `"ARI"`                      | Adjusted Rand Index (KMeans clustering vs labels) | None — operates on embeddings                    |
| `LabelSimilarity`     | (registered)                 | ROC-AUC of intra- vs inter-label cosine sim       | None                                             |
| `ScibBundle`          | `"scib"`                     | scIB benchmark metrics (batch/bio)                | None                                             |
| `UmapPlotter`         | (registered)                 | UMAP visualizations colored by labels             | Uses `pl/plotting.py`                            |
| `OmicsQueryAnnotator` | (not registered, standalone) | Zero-shot annotation via cosine similarity        | Needs `model.encode()` — works with any ST model |

### `embedding_alignment.py`

Standalone module (not registered as an evaluator) that computes cross-modal
alignment scores. Useful for measuring how well the shared space aligns
text and omics embeddings. Works on raw numpy arrays.

### What needs to change for new architecture

1. **Embedding pipeline (`embed/`)** — `embed/model_utils.py` imports
   `MMContextEncoder.get_initial_embeddings_from_adata_link` to register
   omics vectors before embedding. This needs a new path:
   - Load the ST model
   - Detect if it has an `MMContextModule` at position 0
   - Attach a `VectorStore` (built via `prepare_vector_store`) instead of
     calling `register_initial_embeddings`
   - The `prepare_model_and_embed()` function should be updated accordingly

2. **Eval pipeline** — The evaluators themselves don't need changes (they
   operate on `adata.obsm` arrays), but the `embed_pipeline.py` orchestrator
   that feeds them embeddings does (see point 1).

3. **`OmicsQueryAnnotator`** — Works with any model with `.encode()`.
   No changes needed for the new pipeline.

---

## 6 Missing features & test gaps

### Features not yet covered by new modules

- [ ] **Callbacks for new architecture** — `UnfreezeTextEncoderCallback` and
      `UnfreezeAdapterCallback` need to be updated for the new module layout
      (see §4).

- [ ] **Cross-attention** — The old `OmicsEncoder` in
      `cell_sentence_transformer.py` supported cross-attention between text
      and omics sequences. The new `OmicsAttentionModule` only does
      self-attention on omics tokens. Cross-attention would enable richer
      multimodal interaction but adds complexity.

- [ ] **`OneHotTextEncoder` equivalent** — Useful for ablation experiments
      (fast training without a real transformer). Currently only in `_legacy/`.
      Could be a simple fixture/utility rather than a full module.

- [ ] **Negative mining / hard negatives** — The training script currently
      uses `(anchor, positive)` pairs with in-batch negatives (MNR loss).
      The dataset has `negative_1_idx` and `negative_2_idx` columns.
      `resolve_negative_indices_and_rename()` in `utils.py` handles
      resolving these to actual text. Supporting `(anchor, positive, negative)`
      triplets would improve training quality.

- [ ] **Hub upload for new architecture** — `hub_utils.py` model card
      template references old module names.

- [ ] **Embed pipeline for new architecture** — `embed/model_utils.py`
      uses `MMContextEncoder.get_initial_embeddings_from_adata_link()`.
      Needs updating to use `VectorStore` + `prepare_vector_store`.

### Test gaps

- [ ] **`VectorStore` standalone tests** — Currently exercised through
      `test_mmcontext_module.py` and `test_prepare_store.py` fixtures, but
      there's no dedicated `test_vector_store.py` testing all construction
      methods (`from_numpy`, `from_adata`, `from_dict`, `from_dataframe`),
      edge cases, and persistence.

- [ ] **AdapterModule standalone tests** — verify existence of
      `tests/test_adapter_module.py` and confirm coverage.

- [ ] **Mixed-modality batches** — No test currently sends a batch where
      some samples are text and some are omics through the full pipeline
      in a single forward pass. This is an important edge case for
      training with heterogeneous datasets.

- [ ] **Gradient flow end-to-end** — Training tests verify parameters change,
      but don't check that gradients flow through specific module boundaries
      (e.g., from loss through Pooling → AdapterModule → MMContextModule's
      text encoder).

- [ ] **Multi-GPU / DDP** — No tests for distributed training.

- [ ] **Large-scale prepare_store** — `test_prepare_store.py` tests local
      zarr files; no integration test for the Zenodo download path
      (intentionally — network tests are fragile).

### Potential improvements

- [ ] **Converge download logic** — `file_utils.download_and_extract_links`
      and `io/prepare_store._download_zarr` both handle zarr downloads with
      different feature sets. Could share a common download backend.

- [ ] **`simulator.py` integration** — The old `LOSS_PRESETS` dict and
      `make_cluster_sampler` are valuable for generating synthetic test
      datasets. Consider moving to a `testing/` subpackage.

- [ ] **`sanity_helpers.py`** — Contains `plot_pca` and possibly other
      debug helpers not in `pl/`. Consider merging into `pl/` or a
      `debug/` module.

---

## 7 File map (post-cleanup)

```
src/mmcontext/
├── __init__.py                  # exports: modules, io, eval, pl, embed
├── _legacy/                     # old architecture (preserved for compat)
│   ├── adapters.py
│   ├── cell_sentence_transformer.py
│   ├── mmcontextencoder.py
│   ├── omicsencoder.py
│   └── onehot.py
├── modules/                     # NEW: ST v5.4+ pipeline modules
│   ├── mmcontext_module.py      #   InputModule (text + omics routing)
│   ├── adapter_module.py        #   modality-aware projection
│   └── omics_attention_module.py#   optional self-attention (var-level)
├── io/                          # NEW: data I/O
│   ├── vector_store.py          #   mmap-backed vector lookup
│   └── prepare_store.py         #   zarr → VectorStore builder
├── eval/                        # evaluation framework (active, mostly arch-agnostic)
│   ├── base.py, registry.py     #   BaseEvaluator + decorator registry
│   ├── eval_pipeline.py         #   orchestrator
│   ├── ari.py                   #   ARI evaluator
│   ├── label_similarity.py      #   label similarity evaluator
│   ├── embedding_alignment.py   #   cross-modal alignment scores
│   ├── scib_wrapper.py          #   scIB benchmark
│   ├── query_annotate.py        #   zero-shot annotation
│   ├── umap_plotter.py          #   UMAP visualizations
│   └── utils.py                 #   LabelKind, LabelSpec
├── embed/                       # embedding generation pipeline (active, needs update)
│   ├── embed_pipeline.py        #   orchestrator
│   ├── model_utils.py           #   model loading + embedding (imports _legacy)
│   ├── dataset_utils.py         #   adata/dataset loading
│   ├── cellwhisperer_utils.py   #   CellWhisperer-specific helpers
│   └── scsa_utils.py            #   scSA dataset helpers
├── pl/                          # plotting (active)
│   ├── plotting.py              #   UMAP + query-score plots
│   └── metric_plots.py          #   benchmark result bar charts
├── callback.py                  # trainer callbacks (needs update for new modules)
├── file_utils.py                # I/O infrastructure (active)
├── hub_utils.py                 # HF Hub upload (needs template update)
├── sanity_helpers.py            # debug plotting (standalone)
├── simulator.py                 # synthetic data generation (standalone, useful)
├── utils.py                     # training utilities (mixed: some active, some legacy)
└── models/                      # empty after cell_sentence_transformer.py moved
    └── __init__.py

scripts/
├── train_tiny.py                # NEW: training script for new architecture
├── train.py                     # old Hydra training script (uses _legacy)
└── train_merged.py              # old merged training script (uses _legacy)

tests/
├── test_mmcontext_module.py     # NEW: MMContextModule unit tests
├── test_adapter_module.py       # NEW: AdapterModule unit tests (verify exists)
├── test_omics_attention_module.py # NEW: OmicsAttentionModule unit tests
├── test_st_integration.py       # NEW: full pipeline integration + training
├── test_prepare_store.py        # NEW: VectorStore preparation tests
├── test_mmcontext_encoder.py    # legacy: old MMContextEncoder tests
├── test_mmcontext_tokenizer.py  # legacy: old tokenizer tests
├── test_miniOmicsEncoder.py     # legacy: old MiniOmicsModel tests
├── test_register_initial_embeddings.py  # legacy
├── test_sentence_transformer_integration.py  # legacy: old ST integration
└── conftest.py                  # shared fixtures (stub encoder, both old+new)
```
