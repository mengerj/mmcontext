# Preprocessing: `pp`

```{eval-rst}
.. module:: mmcontext.pp
.. currentmodule:: mmcontext

```

This is the documentation of the preprocessing functionalities. This includes generation of data and or context embeddings, their integration into an anndata object, normalization and dimensionality alignment. Finally from the anndata object containing aligned embeddings, a pyTorch dataset can be generated which can be used to create a dataloader suitable for model training.
Classes from this module can be directly imported. eg:

```
from mmcontext.pp import Embedder
```

## Embedding Generation

The idea of this package is to use embeddings derived from pre-trained models of different modalities based on data and the respective context of that data. Therefore at the heart
of this package we have data embeddings, which could for example be computed with a pre-trained `scvi` model {cite:t}`lopez2018deep,gayoso2022python` and context embeddings, which could be computed with a LLM encoder applied to the available metadata. There are really no limits to which encoders to apply as long as they produce some embedding for each sample. By writing a class that inherits from {class}`mmcontext.pp.ContextEmbedder` or {class}`mmcontext.pp.DataEmbedder`, new methods can be implemented.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    pp.Embedder
    pp.ContextEmbedder
    pp.CategoryEmbedder
    pp.PlaceholderContextEmbedder
    pp.DataEmbedder
    pp.PlaceholderDataEmbedder
    pp.PCADataEmbedder
```

## Normalization and Alignment

Since different encoders produce embeddings of different dimensionalities, they have to be brought into a shared dimensionality in order to process them with the core model and to apply the contrastive learning loss objectives. By writing a class that inherits from {class}`mmcontext.pp.EmbeddingNormalizer` and {class}`mmcontext.pp.DimAligner`, new normalization and alignment methods can be implemented respectivly. All of these methods read from and write into the .obsm attribute of {class}`anndata.AnnData` objects.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    pp.EmbeddingNormalizer
    pp.MinMaxNormalizer
    pp.DimAligner
    pp.PCAReducer
```

## Construction of Dataset

With all dimensions aligned a dataset can be constructed with can be used to create a torch dataloader.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    pp.DataSetConstructor
    pp.EmbeddingDataset
```

```{bibliography}
:cited:
```
