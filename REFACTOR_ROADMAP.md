# mmcontext Refactor Roadmap — ST v5.4 Alignment

**Branch:** `dev-claude`
**Goal:** Refactor mmcontext to align with the sentence-transformers v5.4 multimodal API while preserving all core functionality. Test-driven development throughout.

## Architecture Overview

### Current → New

| Aspect | Current | Refactored |
|--------|---------|------------|
| Base class | `Module` | `InputModule` |
| Tokenization | `tokenize()` | `preprocess()` |
| Omics storage | `nn.Embedding` lookup | `VectorStore` (memory-mapped) |
| Omics key | `pixel_values` | `omics_values` |
| Adapters | Inside encoder | Separate `AdapterModule(Module)` |
| Modality flag | `omics_text_info` | `modality_ids` |
| OneHotTextEncoder | Included | Removed |
| Data loading | Coupled to encoder | `mmcontext.io` module |
| Pipeline | `[MMContextEncoder]` | `[MMContextModule, AdapterModule, Pooling, Normalize]` |
| Var support | Same class, no attention | Optional `OmicsAttentionModule` |

### Module Pipeline

```
SentenceTransformer(modules=[
    MMContextModule,          # InputModule: text encoder + omics pass-through
    OmicsAttentionModule,     # Module (OPTIONAL, var-only): self-attention on omics tokens
    AdapterModule,            # Module: modality-aware projection to shared space
    Pooling,                  # ST built-in: mean/cls/max pooling
    Normalize,                # ST built-in: L2 normalization
])
```

### Features Dict Contract

All modules communicate through a features dict. After `MMContextModule.forward()`:

```python
{
    "token_embeddings": Tensor(B, L, D_encoder),  # unified text + omics tokens
    "attention_mask": Tensor(B, L),                # 1 = real, 0 = pad
    "modality_ids": Tensor(B, L),                  # 0 = text, 1 = omics, 2 = pad
    "sentence_embedding": Tensor(B, D)             # set after Pooling
}
```

---

## Phases

### Phase 1: Foundation — VectorStore + IO Module

**Files created/modified:**
- `src/mmcontext/io/__init__.py` (new)
- `src/mmcontext/io/vector_store.py` (new)
- `src/mmcontext/io/adata_utils.py` (new — extracted from mmcontextencoder.py + file_utils.py)
- `tests/test_vector_store.py` (new)

**VectorStore responsibilities:**
- Create from AnnData (obsm/varm), DataFrame, dict, or numpy array
- Write embeddings to numpy memmap file + JSON index
- Batch lookup by sample IDs → numpy array
- Report dim, dtype, len
- Support both obs (1 vector/sample) and var (N vectors/sample)

**Tests (write FIRST):**
1. `test_from_numpy` — round-trip: write memmap, read back, values match
2. `test_from_adata_obs` — create from adata.obsm, lookup by obs index
3. `test_from_adata_var` — create from adata.varm, lookup by var index
4. `test_batch_lookup` — batch of IDs returns correct (N, D) array
5. `test_unknown_id_raises` — KeyError for missing IDs
6. `test_dim_property` — reports correct embedding dimension
7. `test_memory_efficiency` — creating store doesn't load full matrix into RAM (check process RSS)
8. `test_persistence` — store survives close/reopen cycle

**adata_utils responsibilities:**
- `load_embeddings_from_adata_link()` — extracted from `get_initial_embeddings_from_adata_link`
- `create_token_dataframe_from_obsm()` — extracted from encoder
- `build_embedding_df()` — extracted from file_utils

---

### Phase 2: Core Module — MMContextModule (InputModule)

**Files created/modified:**
- `src/mmcontext/modules/__init__.py` (new)
- `src/mmcontext/modules/mmcontext_module.py` (new)
- `tests/test_mmcontext_module.py` (new)

**MMContextModule responsibilities:**
- Extends `InputModule` from sentence-transformers
- `modalities` property returns `["text", "omics"]`
- `preprocess(inputs)` routes by input type:
  - `str` → text tokenization via AutoTokenizer
  - `dict` with `"omics_values"` key → omics vector packaging
  - `str` starting with prefix → VectorStore lookup (if store attached)
- `forward(features)` processes text through AutoModel, passes omics through as-is
- `set_vector_store(store)` / `remove_vector_store()` for attaching data
- `save()` / `load()` — saves config + text encoder weights (NOT omics data)
- Handles mixed batches (text + omics in same batch)
- `max_seq_length` property for ST compatibility
- Text encoder freezing/unfreezing logic

**Tests (write FIRST):**
1. `test_preprocess_text_only` — text strings → input_ids, attention_mask, modality_ids
2. `test_preprocess_omics_direct_vector` — raw vectors → omics_values, attention_mask, modality_ids
3. `test_preprocess_omics_via_store` — prefixed IDs + VectorStore → resolved vectors
4. `test_preprocess_mixed_batch` — text + omics in same batch → unified features
5. `test_preprocess_var_multiple_vectors` — list of gene vectors → variable-length tokens
6. `test_forward_text_only` — produces token_embeddings with correct shape
7. `test_forward_omics_only` — omics vectors pass through to token_embeddings
8. `test_forward_mixed_batch` — mixed batch produces unified token_embeddings + modality_ids
9. `test_forward_preserves_omics_values` — omics vectors appear unchanged in token_embeddings
10. `test_modalities_property` — returns ["text", "omics"]
11. `test_save_load_roundtrip` — config + weights survive save/load
12. `test_save_excludes_vector_store` — VectorStore data not in saved weights
13. `test_text_encoder_freezing` — freeze/unfreeze text encoder parameters
14. `test_max_seq_length` — property accessible and correct
15. `test_no_store_omics_id_raises` — prefixed IDs without VectorStore → clear error

---

### Phase 3: Modality-Aware AdapterModule

**Files created/modified:**
- `src/mmcontext/modules/adapter_module.py` (new — replaces adapters.py as pipeline Module)
- `tests/test_adapter_module.py` (new)

**AdapterModule responsibilities:**
- Extends `Module` from sentence-transformers
- Reads `modality_ids` from features dict
- Maintains separate projection weights: `text_proj` and `omics_proj`
- Each projection: Linear → ReLU → Linear → BatchNorm (configurable)
- Maps `D_text → D_shared` and `D_omics → D_shared`
- Pad tokens (modality_id=2) pass through as zeros
- `save()` / `load()` with config_keys for all dimensions
- `get_sentence_embedding_dimension()` returns D_shared

**Tests (write FIRST):**
1. `test_forward_text_only` — text tokens projected correctly
2. `test_forward_omics_only` — omics tokens projected correctly
3. `test_forward_mixed_batch` — text and omics tokens get different projections
4. `test_pad_tokens_stay_zero` — modality_id=2 tokens remain zero after projection
5. `test_output_dimension` — output matches D_shared regardless of input modality
6. `test_separate_weights` — text_proj and omics_proj have independent parameters
7. `test_gradient_flow` — gradients flow through both projections
8. `test_weights_update` — optimizer step changes both projection weights
9. `test_save_load_roundtrip` — config + weights survive save/load
10. `test_get_sentence_embedding_dimension` — returns D_shared

---

### Phase 4: OmicsAttentionModule (Optional, Var-only)

**Files created/modified:**
- `src/mmcontext/modules/omics_attention_module.py` (new)
- `tests/test_omics_attention_module.py` (new)

**OmicsAttentionModule responsibilities:**
- Extends `Module` from sentence-transformers
- Reads `modality_ids` from features dict
- Applies multi-head self-attention ONLY to omics tokens (modality_id=1)
- Text tokens (modality_id=0) pass through unchanged
- Configurable: num_layers, num_heads, hidden_dim, dropout
- Respects attention_mask for variable-length gene sequences
- `save()` / `load()` for persistence

**Tests (write FIRST):**
1. `test_text_passthrough` — text tokens unchanged after module
2. `test_omics_transformed` — omics tokens are modified by self-attention
3. `test_attention_mask_respected` — padded positions don't influence real tokens
4. `test_variable_length_sequences` — different-length gene sequences in same batch
5. `test_output_shape_preserved` — (B, L, D) shape unchanged
6. `test_gradient_flow` — gradients flow through attention layers
7. `test_save_load_roundtrip` — config + weights survive save/load
8. `test_single_token_sequence` — obs-like input (L=1) works without error

---

### Phase 5: SentenceTransformer Integration

**Files created/modified:**
- `tests/test_st_integration.py` (new — replaces test_sentence_transformer_integration.py)
- Minor adjustments to modules for compatibility

**Integration tests (write FIRST):**
1. `test_pipeline_construction` — modules compose into SentenceTransformer
2. `test_encode_text` — `model.encode(["text"])` produces correct shape
3. `test_encode_omics_direct` — `model.encode([{"omics_values": vector}])` works
4. `test_encode_omics_via_store` — `model.encode(["sample_idx:S1"])` with VectorStore
5. `test_encode_mixed` — mixed text + omics batch produces unified embeddings
6. `test_encode_normalize` — output vectors are L2-normalized
7. `test_save_load_full_pipeline` — save + load produces identical encode() results
8. `test_modules_json_structure` — saved modules.json has correct module chain
9. `test_load_with_trust_remote_code` — model loadable via SentenceTransformer(..., trust_remote_code=True)
10. `test_training_text_only` — SentenceTransformerTrainer runs with text data
11. `test_training_bimodal` — SentenceTransformerTrainer runs with mixed data
12. `test_precision_conversion` — fp32 → fp16 works across all modules
13. `test_max_seq_length_propagation` — max_seq_length accessible from SentenceTransformer
14. `test_obs_pipeline` — full obs pipeline: [MMContext, Adapter, Pooling, Normalize]
15. `test_var_pipeline` — full var pipeline: [MMContext, OmicsAttn, Adapter, Pooling, Normalize]

---

### Phase 6: Documentation + Cleanup

**Files modified:**
- `src/mmcontext/__init__.py` — update public API exports
- `src/mmcontext/modules/__init__.py` — export all modules
- `src/mmcontext/io/__init__.py` — export VectorStore and utilities
- Module docstrings — comprehensive docstrings with usage examples
- `README.md` — update architecture description and usage examples

**Cleanup:**
- Remove `OneHotTextEncoder` (`onehot.py`)
- Archive old `mmcontextencoder.py` (keep temporarily for reference, don't import)
- Archive old `adapters.py` (replaced by modules/adapter_module.py)
- Remove `test_adapter_callback.py` (empty)
- Update `pyproject.toml` if new dependencies needed

---

## Implementation Order & Dependencies

```
Phase 1 (VectorStore)
  ↓
Phase 2 (MMContextModule) ← depends on VectorStore
  ↓
Phase 3 (AdapterModule)   ← depends on features dict contract from Phase 2
  ↓
Phase 4 (OmicsAttention)  ← depends on features dict contract from Phase 2
  ↓
Phase 5 (ST Integration)  ← depends on all modules
  ↓
Phase 6 (Docs + Cleanup)  ← final polish
```

## Test Strategy

Each phase follows strict TDD:
1. Write test file with all tests (they will fail)
2. Implement the module until all tests pass
3. Run full test suite to check for regressions

**Stub strategy:** Tests use lightweight stubs (similar to existing `_TokStub`, `_TextEncStub`) to avoid downloading real models. The existing `conftest.py` pattern is extended for new modules.

**What's preserved from current tests:**
- Core encoding shapes and correctness (text, omics, mixed)
- Save/load round-trips
- Gradient flow through adapters
- ST integration (encode, train, normalize)
- Attention mask alignment
- Freezing/unfreezing behavior

**What's new:**
- VectorStore tests (memmap, lookup, persistence)
- Modality-aware adapter tests (separate projections)
- OmicsAttentionModule tests (self-attention on omics only)
- Direct vector input tests (no lookup table)
- Pipeline composition tests (modules.json)

## Open Items (Deferred)

- Training script rewrite (Phase 7, separate effort)
- Backward compatibility loading of old models (evaluate complexity later)
- Cross-attention in OmicsAttentionModule (future extension)
- Evaluation pipeline updates (depends on new model API)
