# CLAUDE.md — mmcontext

## Project Overview

mmcontext is a Python library for multimodal contrastive learning that aligns text and omics (single-cell gene expression) embeddings using the sentence-transformers (>=5.4) framework. It enables joint embedding spaces where text descriptions and biological data can be compared directly.

**Repository**: `github.com/mengerj/mmcontext`
**Maintainer**: Jonatan Menger
**Python**: >=3.11, <3.14
**Key dependency**: `sentence-transformers>=5.4` (multimodal API)

## Package Structure

```
src/mmcontext/
├── modules/           # sentence-transformers InputModules
│   ├── mmcontext_module.py    # Core: text tokenization + omics vector pass-through
│   ├── adapter_module.py      # Projects omics vectors into shared embedding space
│   └── omics_attention_module.py  # Optional self-attention for omics tokens
├── embed/             # Embedding pipeline, dataset utilities, model utils
├── eval/              # Evaluation: kNN, ARI, scIB metrics, label similarity
├── io/                # VectorStore, data preparation
├── pl/                # Plotting utilities
├── _legacy/           # Archived modules (do not modify unless removing)
├── callback.py        # Training callbacks
├── hub_utils.py       # HuggingFace Hub integration
├── simulator.py       # Synthetic data generation for testing
└── utils.py           # Shared utilities
```

## Branch Strategy

- `main` — stable releases only
- `dev-claude` — integration branch for all agent and feature work
- Feature branches: always branch from `dev-claude`, never from `main`
- Branch naming: `claude/<issue-number>-<short-slug>` for agent-created branches

## Commands

```bash
# Install (editable, with test deps)
pip install -e ".[dev,test]"

# Run tests
pytest -v --color=yes

# Run tests with coverage
coverage run -m pytest -v --color=yes && coverage report

# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Format
ruff format src/ tests/
```

## Code Style

- **Formatter/linter**: ruff (line-length=120)
- **Docstrings**: numpy-style, required for public classes and functions (D100/D104/D105/D107 ignored)
- **Imports**: sorted by isort (via ruff)
- **Type hints**: use them for all public function signatures
- **Pre-commit**: prettier, pyproject-fmt, ruff, detect-private-key, check-ast

## Testing Conventions

- Tests live in `tests/` at repo root
- Use pytest fixtures; shared fixtures go in `conftest.py`
- Test files: `test_<module_name>.py`
- The `simulator.py` module generates synthetic data for tests — prefer it over loading real datasets
- Tests must pass on Python 3.11, 3.12, 3.13

## PR Conventions

- PRs target `dev-claude` (not `main`) unless it's a release
- PR description must reference the issue: `Fixes #<number>` or `Closes #<number>`
- All CI checks must pass before merge
- Keep PRs focused — one logical change per PR

## Issue Implementation Protocol

When implementing a feature from a GitHub issue (via @claude or otherwise):

1. **If the issue is ambiguous**: Post clarifying questions as a comment. Do NOT start implementation until the questions are answered.

2. **Plan first**: Before writing any code, post an implementation plan as a comment on the issue with a checkbox list:

   ```
   ## Implementation Plan
   - [ ] Step 1: description
   - [ ] Step 2: description
   - [ ] Step 3: description
   - [ ] Verify: run tests, check linting
   ```

   Wait for approval (a reply containing "approved", "go ahead", "LGTM", or "looks good").

3. **Implement**: Create a branch `claude/<issue-number>-<short-slug>` from `dev-claude`. Implement the plan step by step. Edit the plan comment to check off completed steps.

4. **Open PR**: Create a PR targeting `dev-claude` with `Fixes #<number>` in the description. Include a summary of what was done and any decisions made.

## Architecture Notes

The core pipeline follows the sentence-transformers module pattern:

```
Input → MMContextModule → AdapterModule → [OmicsAttentionModule] → Pooling → Loss
```

- **MMContextModule**: Handles both text (tokenize → AutoModel) and omics (VectorStore lookup or direct input) modalities. Outputs a unified features dict with `token_embeddings`, `attention_mask`, and `modality_ids`.
- **AdapterModule**: Projects omics vectors into the text model's embedding space.
- **OmicsAttentionModule**: Optional self-attention layer for omics tokens.
- Data stored in anndata format (`.h5ad`), converted via `adata-hf-datasets` for HuggingFace compatibility.

## What NOT to Change

- Do not modify files in `_legacy/` unless explicitly removing them
- Do not change the sentence-transformers module interface contracts (features dict keys)
- Do not add dependencies without discussion — the package already has heavy deps (torch, transformers, scanpy)
