# mmcontext

[![Tests][badge-tests]][tests]

Multimodal contrastive learning aligning text and omics embeddings via [sentence-transformers](https://www.sbert.net/).

**mmcontext** enables joint embedding spaces where natural-language descriptions and biological data (single-cell gene expression) can be compared directly. Built on the sentence-transformers v5.4+ multimodal API.

![Conceptual Diagram](figs/concept.png)

## Paper

**mmcontext** is described in detail in our paper:
Menger, Jonatan, Sonia Maria Krissmer, Clemens Kreutz, Harald Binder, and Maren Hackenberg. "mmContext: an open framework for multimodal contrastive learning of omics and text data." bioRxiv (2025): 2025-12.
If you use **mmcontext** in your research, please cite our work.

## Installation

```bash
pip install mmcontext
```

Or install from source with development dependencies:

```bash
git clone https://github.com/mengerj/mmcontext.git
cd mmcontext
pip install -e ".[dev,test,eval,train]"
```

## Quick Start

### Using a Pre-trained Model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jo-mengr/mmcontext-pubmedbert-gs10k", trust_remote_code=True)

# Encode text queries
text_embeddings = model.encode(["CD4+ T cell", "B cell precursor"])
```

Pre-trained models are available on Hugging Face under the [jo-mengr](https://huggingface.co/jo-mengr) organization.

See the [pretrained inference tutorial](tutorials/pretrained_inference.ipynb) for a complete guide.

### Building a Pipeline

```python
from mmcontext.embed import build_pipeline

pipeline = build_pipeline(
    text_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    omics_dim=512,        # dimension of your omics vectors
    shared_dim=256,       # joint embedding space dimension
)
```

### Evaluation

```python
from mmcontext.eval import get

LabelSimilarity = get("LabelSimilarity")
evaluator = LabelSimilarity(similarity="cosine")
result = evaluator.compute(
    omics_embeddings=omics_emb,
    label_embeddings=label_emb,
    query_labels=labels,
    true_labels=true_labels,
    label_key="cell_type",
)
```

See the [evaluate model 2.0 notebook](tutorials/evaluate_model_2.0.ipynb) for a complete evaluation workflow.

## Architecture

mmcontext 2.0 uses the sentence-transformers module pipeline pattern:

```
Input -> MMContextModule -> AdapterModule -> Pooling -> Normalize
```

The **MMContextModule** handles both text (tokenize -> AutoModel) and omics (VectorStore lookup) modalities. The **AdapterModule** projects omics vectors into the text model's embedding space.

Data is stored in AnnData format (`.h5ad`/zarr), with a memory-efficient VectorStore for runtime omics lookups.

## Multi-Model Benchmarking

For systematic comparisons across multiple models and datasets, see the companion repository: [mmcontext-benchmark](https://github.com/mengerj/mmcontext-benchmark).

## Dataset Preparation

To prepare training datasets from AnnData objects, see [adata-hf-datasets](https://github.com/mengerj/adata_hf_datasets).

## Contributing

Contributions and suggestions are very welcome! Please open an [issue](https://github.com/mengerj/mmcontext/issues) to propose enhancements, report bugs, or ask questions.

## Citation

```bibtex
@article{menger2025mmcontext,
  author = {Menger, Jonatan and Krissmer, Sonia Maria and Kreutz, Clemens and Binder, Harald and Hackenberg, Maren},
  title = {mmContext: an open framework for multimodal contrastive learning of omics and text data},
  journal = {bioRxiv},
  year = {2025},
  doi = {10.1101/2025.12}
}
```

---

[badge-tests]: https://img.shields.io/github/actions/workflow/status/mengerj/mmcontext/test.yaml?branch=main
[tests]: https://github.com/mengerj/mmcontext/actions/workflows/test.yaml
