# Migration Plan: Split evaluation/benchmarking into `mmcontext-benchmark`

**Status:** in progress — benchmark pipeline implemented (see §0); `mmcontext`
slim-down (§3–§5) still pending.
**Author:** Jonatan Menger (with Claude)
**Date:** 2026-06-03 (updated 2026-06-09)
**Goal:** Make `mmcontext` a focused model package (architecture + lightweight,
single-model evaluation runnable from a notebook). Move all multi-model
comparison, competitor integrations (SCSA, CellWhisperer), and the heavy
embedding/eval orchestration into a new sibling repo, `mmcontext-benchmark`.
Drop the `adata-hf-datasets` package dependency from `mmcontext`.

---

## 0. Implemented benchmark design (2026-06-09)

The benchmark repo was built around an **adapter + artifact-contract**
architecture rather than a straight file-move of the legacy orchestrators. What
landed (and where it diverged from the original §2–§3 plan):

- **Artifact contract** (`mmcontext_benchmark/artifacts.py`): one on-disk layout
  per `(dataset, model)` — embedders write `embeddings.parquet` + `subset.h5ad`
  - `*_label_embeddings_*`; direct classifiers write
    `predictions_<label>.parquet`; both write `meta.yaml`. Adapters and eval only
    talk through this, which is what makes different methods comparable.
- **Adapters + registry** (`adapters/`): one adapter per model _kind_. The
  `mmcontext` adapter **auto-detects** the old (registered-embeddings) vs new
  (VectorStore) approach by inspecting the loaded model's first module — so a
  single config entry works for both. External tools declare an explicit
  `kind:` and are dispatched to their adapter. This replaces the planned
  wholesale move of `model_utils`/`embed_pipeline` functions.
- **External tools = containers** (`containers/`): instead of porting
  `scsa_utils.py`/`cellwhisperer_utils.py` as in-process modules with their
  bespoke venv bootstrapping, each competitor ships as its **own image**
  (SCSA, CellWhisperer, and a separate **calmate** image for label
  harmonisation) with a self-contained entrypoint. The host adapter prepares
  inputs, runs the container (Docker locally / Apptainer on clusters), and reads
  artifacts back. This is the clean replacement for the "unsatisfying" venv +
  `subprocess` + interpreter-path approach.
- **Two metric tracks** (`eval/`): embedders → kept-in-`mmcontext`
  `LabelSimilarity`; classifiers (SCSA) → `eval/classification.py`, emitting the
  **same** `LabelSimilarity/...` keys so both fuse in `collect/`. CellWhisperer's
  dot+softmax+`logit_scale` is carried in `meta.yaml`. scIB is dropped (legacy).
- **Config**: Hydra is retained. `conf/benchmark.yaml` composes `datasets/` and
  `models/` lists; `scripts/run_benchmark.py` runs embed → eval → collect.

Consequence for this plan: the §3.2 "MOVE" rows are realised as _adapters +
container entrypoints_ in the benchmark repo, not as 1:1 file moves. The
`mmcontext`-side slim-down below is unchanged and still to do.

---

## 1. Guiding principles

1. **`mmcontext` = the model.** Modules, adapters, the VectorStore the model
   depends on, hub/training utilities, and a _slim_ eval surface that a user can
   run from `tutorials/evaluate_model.ipynb` on a single trained model.
2. **`mmcontext-benchmark` = the science of comparison.** Everything that loads
   _other_ models, runs multi-model embedding pipelines, computes comparative
   metrics, and reproduces paper figures. It depends on `mmcontext` (and on
   `adata-hf-datasets`), never the reverse.
3. **No new dependency edges into `mmcontext`.** After the split, nothing in
   `src/mmcontext` may import `adata_hf_datasets`, `scsa_utils`,
   `cellwhisperer_utils`, or `eval_pipeline`.
4. **Tests move with their code.** Each moved module takes its tests (or new
   ones) to the benchmark repo; kept modules keep/keep-improving their tests.

---

## 2. Key findings from the current tree

These shaped the boundaries below and are worth recording:

- **No `src/` file imports `adata_hf_datasets`.** The only references are in
  `pyproject.toml`, `egg-info`, docstrings, and _notebooks_. The package dep is
  effectively a notebook-only dependency today, so removing it from
  `pyproject.toml` is safe for the library. Users who need to _create_ test
  datasets install `adata-hf-datasets` themselves (as the kept
  `evaluate_model.ipynb` already instructs).
- **The kept eval modules are self-contained.** `query_annotate`,
  `label_similarity`, `ari`, `embedding_alignment`, `umap_plotter`,
  `scib_wrapper`, `evaluate_scib`, `base`, `registry`, `utils` only import from
  within `eval/`, plus `pl/` and `file_utils`. None imports `eval_pipeline.py`
  or anything in `embed/`. The split is clean.
- **`eval/eval_pipeline.py` is the multi-model orchestrator** (it special-cases
  CellWhisperer logit-scale extraction and walks the registry over many models).
  It is the natural seam — it moves; the registry + individual evaluators stay.
- **`embed/model_utils.py` mixes two concerns:** single-model encode helpers
  (`load_st_model`, `encode_adata`, `create_label_dataset`, `embed_labels`,
  `HFIndexedDataset`) vs. multi-model orchestration (`prepare_model_and_embed`,
  `prepare_model_and_ds`). It must be split, not moved wholesale.
- **`evaluate_model.ipynb` imports** `mmcontext.eval` (incl. registry `get`),
  `eval.query_annotate`, `pl`, `file_utils`, `utils` — all in the keep set. It
  is the canonical "evaluate one model" notebook and stays.
- **The two embed-data tutorials** are `tutorials/pretrained_inference.ipynb`
  and `tutorials/train_new.ipynb`; both import `adata_hf_datasets`. These are
  the two to remove. `tutorials/evaluate_model copy.ipynb` is a stray duplicate
  and should also go.

---

## 3. File-by-file disposition

### 3.1 KEEP in `mmcontext`

| Path                                                                              | Notes                                                                                       |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `src/mmcontext/modules/*`                                                         | Core architecture — untouched.                                                              |
| `src/mmcontext/io/vector_store.py`                                                | Required by `MMContextModule`.                                                              |
| `src/mmcontext/io/prepare_store.py`                                               | Keep the VectorStore-from-HF-dataset path (no `adata_hf` import; only a docstring mention). |
| `src/mmcontext/eval/{base,registry,utils}.py`                                     | Eval framework + label specs.                                                               |
| `src/mmcontext/eval/query_annotate.py`                                            | `OmicsQueryAnnotator` — used by the notebook; has tests.                                    |
| `src/mmcontext/eval/{ari,embedding_alignment,label_similarity}.py`                | Single-model metrics.                                                                       |
| `src/mmcontext/eval/{scib_wrapper,evaluate_scib}.py`                              | scIB metrics for one embedding.                                                             |
| `src/mmcontext/eval/umap_plotter.py`                                              | Single-model UMAP eval.                                                                     |
| `src/mmcontext/embed/dataset_utils.py`                                            | `SentenceDataset`, `load_generic_dataset`, `collect_adata_subset` — minimal loading.        |
| `src/mmcontext/embed/dataset_prep.py`                                             | `prepare_inference`, `InferenceData`, `prepare_dataset` — minimal data prep.                |
| **NEW** `src/mmcontext/embed/encode.py`                                           | Single-model encode helpers split out of `model_utils.py` (see 3.3).                        |
| `src/mmcontext/{callback,hub_utils,simulator,utils,file_utils,sanity_helpers}.py` | Core/training/util.                                                                         |
| `src/mmcontext/pl/*`                                                              | Plotting (used by kept eval + notebook).                                                    |
| `tutorials/evaluate_model.ipynb`                                                  | Canonical single-model evaluation notebook.                                                 |

### 3.2 MOVE to `mmcontext-benchmark`

| Path                                                                                                 | New home (benchmark)                          | Reason                                          |
| ---------------------------------------------------------------------------------------------------- | --------------------------------------------- | ----------------------------------------------- |
| `src/mmcontext/eval/eval_pipeline.py`                                                                | `mmcontext_benchmark/eval/pipeline.py`        | Multi-model comparison orchestrator.            |
| `src/mmcontext/embed/embed_pipeline.py`                                                              | `mmcontext_benchmark/embed/pipeline.py`       | Multi-model embedding pipeline.                 |
| `src/mmcontext/embed/scsa_utils.py`                                                                  | `mmcontext_benchmark/models/scsa.py`          | Competitor model (SCSA).                        |
| `src/mmcontext/embed/cellwhisperer_utils.py`                                                         | `mmcontext_benchmark/models/cellwhisperer.py` | Competitor model (CellWhisperer).               |
| `model_utils.py` → `prepare_model_and_embed`, `prepare_model_and_ds`, `process_single_dataset_model` | `mmcontext_benchmark/embed/orchestrate.py`    | Multi-model orchestration split.                |
| `tutorials/pretrained_inference.ipynb`, `tutorials/train_new.ipynb`                                  | `mmcontext-benchmark/examples/` (or delete)   | Embed-data tutorials using `adata_hf_datasets`. |

### 3.3 SPLIT `embed/model_utils.py`

Stays in `mmcontext` as `embed/encode.py` (single-model):
`load_st_model`, `encode_adata`, `create_label_dataset`, `embed_labels`,
`HFIndexedDataset`, `SentenceDataset`.

Moves to benchmark `embed/orchestrate.py` (multi-model):
`prepare_model_and_embed`, `prepare_model_and_ds`.

Update `src/mmcontext/embed/__init__.py` to export only kept symbols; remove
`process_cellwhisperer_dataset_model`, `ensure_cellwhisperer_setup`,
`embed_pipeline`, `process_single_dataset_model`, `prepare_model_and_embed`.

### 3.4 REMOVE

- `tutorials/pretrained_inference.ipynb`, `tutorials/train_new.ipynb`
  (after their useful content is ported to benchmark `examples/`).
- `tutorials/evaluate_model copy.ipynb` (stray duplicate).
- `adata-hf-datasets` from `pyproject.toml` `dependencies` and the
  `[tool.uv] sources.adata-hf-datasets` block.

---

## 4. Dependency changes (`pyproject.toml`)

Remove from `mmcontext`:

```toml
# dependencies
- "adata-hf-datasets @ git+https://github.com/mengerj/adata_hf_datasets.git",

# [tool.uv]
- sources.adata-hf-datasets = { git = "https://github.com/mengerj/adata_hf_datasets.git" }
```

**Training stays in `mmcontext`.** The package keeps its training code
(`callback.py`, training utilities, hub upload). `wandb` is used for training
logging, so it is **not removed** — instead it becomes an _optional_ extra so
inference-only users get a lighter install:

```toml
# move wandb out of core dependencies into:
optional-dependencies.train = [ "wandb>=0.27" ]
# (alternatively a broader `train` extra bundling hydra-core, wandb, accelerate)
```

Training code must import `wandb` lazily (inside the training/callback path,
guarded by a clear `ImportError` message pointing to `pip install
mmcontext[train]`) so that `import mmcontext` and inference never require it.

Candidates to also drop from `mmcontext` _once their only users have moved_
(verify with a final grep — do not remove blindly): `scib`, `louvain` (if only
used by scIB multi-model paths). Keep anything still imported by the kept
eval/model code. `scib`/`louvain` are used by the kept
`scib_wrapper`/`evaluate_scib`, so they **stay** unless those move too.
`hydra-core` stays if training configs use it.

`mmcontext-benchmark` `pyproject.toml` depends on:
`mmcontext @ git+https://github.com/mengerj/mmcontext.git`,
`adata-hf-datasets @ git+...`, plus the SCSA/CellWhisperer runtime deps.

---

## 5. Phased execution

Each phase is a separate PR targeting `dev-claude`, one logical change each, CI
green before merge (per `CLAUDE.md`).

**Phase 0 — Scaffold benchmark repo (no `mmcontext` change).**
Create `mmcontext-benchmark` skeleton (this plan ships a starter skeleton).
Wire CI, pre-commit, pytest. Add `mmcontext` as a git dependency.

**Phase 1 — Copy competitor + orchestration code into benchmark.**
Port `scsa_utils`, `cellwhisperer_utils`, `embed_pipeline`, `eval_pipeline`,
multi-model `model_utils` functions, and the two embed-data tutorials into the
benchmark repo. Fix imports to `mmcontext.*` for shared helpers (`eval.utils`,
`file_utils`, `pl`). Get benchmark tests passing against installed `mmcontext`.
_No deletion in `mmcontext` yet — both repos temporarily overlap._

**Phase 2 — Split `model_utils.py` in `mmcontext`.**
Branch `claude/<issue>-split-model-utils`. Create `embed/encode.py` with the
single-model helpers; update `embed/__init__.py`; update any internal imports.
Run tests.

**Phase 3 — Remove moved code from `mmcontext`.**
Delete `eval/eval_pipeline.py`, `embed/embed_pipeline.py`, `embed/scsa_utils.py`,
`embed/cellwhisperer_utils.py`, and the multi-model functions. Update
`__init__.py` exports. Remove the two tutorials + the stray copy.

**Phase 4 — Drop the dependency.**
Remove `adata-hf-datasets` from `pyproject.toml` and `[tool.uv]`. Re-lock.
Grep to confirm zero `adata_hf` / `scsa` / `cellwhisperer` references remain in
`src/`. Update README references.

**Phase 5 — Docs + notebook.**
Ensure `evaluate_model.ipynb` runs end-to-end against the slimmed package.
Add a README note pointing benchmarking/comparison users to
`mmcontext-benchmark`. Update `CLAUDE.md` package-structure section.

---

## 6. Test strategy

- **Kept code:** keep existing tests
  (`test_query_annotate.py`, module/adapter/encoder tests, `test_io`,
  `test_prepare_store.py`). Add a regression test asserting the slim public API:
  `import mmcontext; mmcontext.eval.get(...)` works and the moved symbols are
  _gone_ (`pytest.raises(ImportError)` / `AttributeError`).
- **Import-isolation test:** a test that imports every `src/mmcontext` module and
  asserts `adata_hf_datasets` was never imported (check `sys.modules`). This
  locks in the dependency removal.
- **Moved code:** create `tests/` in the benchmark repo. Reuse
  `mmcontext.simulator` for synthetic fixtures (per project convention — prefer
  the simulator over real datasets). Run on Python 3.11–3.13.
- **Sandbox note:** `torch` is too large to install in the Cowork sandbox, so
  the full test suite must run on Jo's machine / CI, not here.

---

## 7. Risks & mitigations

| Risk                                                     | Mitigation                                                                                                                             |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Hidden runtime coupling not caught by static grep        | Phase 1 keeps both copies until benchmark tests pass; only Phase 3 deletes.                                                            |
| `eval.utils.LabelKind` / `file_utils` used by moved code | These stay in `mmcontext`; benchmark imports them — intended one-way edge.                                                             |
| Dropping `scib`/`louvain` breaks kept scIB eval          | Keep them; only drop deps whose _only_ importer moved (verify by grep).                                                                |
| Notebook drift (Jo is mid-update)                        | Phase 5 is last; coordinate with the in-progress `evaluate_model.ipynb` rewrite.                                                       |
| Users relying on old `mmcontext.embed.embed_pipeline`    | Add a deprecation shim that raises `ImportError` with a message pointing to `mmcontext-benchmark` (one release), or note in CHANGELOG. |

---

## 8. Rollback

Each phase is an isolated PR. If a phase regresses, revert that PR; earlier
phases (scaffold, copies) are additive and safe to leave in place. The
benchmark repo is independent, so reverting `mmcontext` changes never breaks it
beyond a version pin bump.

---

## 9. Definition of done

- `grep -rn "adata_hf\|scsa\|cellwhisperer" src/mmcontext --include="*.py"`
  returns nothing (docstrings cleaned too).
- `pip install mmcontext` pulls no `adata-hf-datasets`.
- `tutorials/evaluate_model.ipynb` runs against the slim package.
- `mmcontext-benchmark` reproduces the multi-model comparison previously done by
  `eval_pipeline.py`, with its own passing tests.
- `CLAUDE.md` and `README` reflect the new boundary.
