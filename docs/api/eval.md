# Evaluation: `eval`

```{eval-rst}
.. module:: mmcontext.eval
.. currentmodule:: mmcontext

```

This is the documentation of the evaluation module, which comprises methods to evaluate the quality of embeddings and of reconstructed data. Method implemented by Luecken et. al {cite:p}`luecken2022benchmarking` are used to evaluate batch integration and conservation of biological information. To evaluate how well the reconstructed data matches the orginal data a variety of data properties are calculated and compared-
Classes and functions from this module can be directly imported. eg:

```
from mmcontext.eval import Evaluator
```

## Batch correction and bio-conservation (scib metrics)

The {class}`mmcontext.eval.scibEvaluator` implements the batch-integration and bio-conservation metrics used by Luecken et al. {cite:p}`luecken2022benchmarking` and also gives an average score. It can be applied to embeddings or expression data (both raw and reconstructed). The ilisi and clisi metrics can only be computed on unix systems.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    eval.scibEvaluator
```

## Comparison based on data characteristics

The class {class}`mmcontext.eval.DataProperties` can be used to add different original and reconstructed datasets and compute log2foldchanges between the properties of reconstructed and original data This allows to judge how realistic the reconstructed data is.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    eval.DataProperties
```
