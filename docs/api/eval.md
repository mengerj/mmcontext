# Evaluation: `eval`

```{eval-rst}
.. module:: mmcontext.eval
.. currentmodule:: mmcontext

```

The whole evaluation pipeline can be applied to a any Sentence Transformers model. Adapt conf/eval_conf.yaml to include your model repo id, as well as repo_ids of huggingface datasets to be used as evaluation datasets. Besides several UMAPs, which visualise the clustering of the different embedding spaces, several quantitative metrics are available.

```

```

Methods implemented by Luecken et. al {cite:p}`luecken2022benchmarking` are used to evaluate batch integration and conservation of biological information. To evaluate how well the reconstructed data matches the orginal data a variety of data properties are calculated and compared-
Classes and functions from this module can be directly imported. eg:

```
from mmcontext.eval import scibEvaluator
```

## Batch correction and bio-conservation (scib metrics)

The {class}`mmcontext.eval.scibEvaluator` implements the batch-integration and bio-conservation metrics used by Luecken et al. {cite:p}`luecken2022benchmarking` and also gives an average score. It can be applied to embeddings or expression data (both raw and reconstructed). The ilisi and clisi metrics can only be computed on unix systems.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    eval.scibEvaluator
```

## Annotation accuracy and ROC

The class {class}`mmcontext.eval.OmicsQueryAnnotator` can be used to query a dataset with natural language. The query-scores, which are the dot-product of the embeddings of a text-query and a reference omics dataset, indicate how well a description fits to a certain dataset. By calculating the query scores for a list of possible labels, we can annotate our data. The resulting adata object can be used to

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    eval.OmicsQueryAnnotator
```
