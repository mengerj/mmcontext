# Evaluation: `eval`

```{eval-rst}
.. module:: mmcontext.eval
.. currentmodule:: mmcontext

```

The whole evaluation pipeline can be applied to a any Sentence Transformers model. Adapt conf/eval_conf.yaml to include your model repo id, as well as repo_ids of huggingface datasets to be used as evaluation datasets. Besides several UMAPs, which visualise the clustering of the different embedding spaces, several quantitative metrics are available. The workflow can be run with:

```
python scripts/evaluation_workflow.py
```

## Batch correction and bio-conservation (scib metrics)

Methods implemented by Luecken et. al {cite:p}`luecken2022benchmarking` are used to evaluate batch integration and conservation of biological information. To evaluate how well the reconstructed data matches the orginal data a variety of data properties are calculated and compared-
Classes and functions from this module can be directly imported. eg:

```
from mmcontext.eval import scibEvaluator
```

The {class}`mmcontext.eval.evaluate_scib.scibEvaluator` implements the batch-integration and bio-conservation metrics used by Luecken et al. {cite:p}`luecken2022benchmarking` and also gives an average score. It can be applied to embeddings or expression data (both raw and reconstructed). The ilisi and clisi metrics can only be computed on unix systems.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    eval.scibEvaluator
```

## Annotation accuracy and ROC

The class {class}`mmcontext.eval.query_annotate.OmicsQueryAnnotator` can be used to query a dataset with natural language. The query-scores, which are the dot-product of the embeddings of a text-query and a reference omics dataset, indicate how well a description fits to a certain dataset. By calculating the query scores for a list of possible labels, we can annotate our data. The "query_score" attribute in adata.obs can be used to visualise the reponse with the plotting function {func}`mmcontext.pl.plotting.plot_query_scores_umap`. The annotation accuracy can be evaluated with {func}`mmcontext.eval.annotation_accuracy.evaluate_annotation_accuracy`.

With the function {func}`mmcontext.eval.classification_roc.zero_shot_classification_roc`, the area under the ROC curve can be used to judge the annotation quality (in addition to the accuracy). It is calculated for each label and then averaged over all labels to get one metric.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    eval.query_annotate.OmicsQueryAnnotator
    eval.annotation_accuracy.evaluate_annotation_accuracy
    mmcontext.eval.classification_roc.zero_shot_classification_roc
```

## Embedding alignment

As a further evaluation metric we quantify the alignment of cross-modal pairs in the shared latent space. The function {func}`mmcontext.eval.embedding_alignment.evaluate_modality_alignment` returns two metrics. The first is a modality-gap irrevelvant score. This means, only the distance between true cross-modal pairs and false cross-modal pairs is calculated. The other score also considers intra-modal false pairs. In a not well aligned latent space, intra-modal faulty pairs might be much closer than true cross-modal pairs. But for annotation, it is mostly important to have a good distinguishment between true and false cross-modal pairs.
