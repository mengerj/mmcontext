# Evaluation: `eval`

```{eval-rst}
.. module:: mmcontext.eval
.. currentmodule:: mmcontext

```

The whole evaluation pipeline can be applied to a any Sentence Transformers model. Adapt conf/eval_conf.yaml to include your model repo id, as well as repo_ids of huggingface datasets to be used as evaluation datasets. Besides several UMAPs, which visualise the clustering of the different embedding spaces, several quantitative metrics are available. The workflow can be run with:

```
python scripts/eval.py
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

    eval.evaluate_scib.scibEvaluator
```

## Annotation accuracy and ROC

The class {class}`mmcontext.eval.label_similarity.LabelSimilarity` can be used to evaluate the similarity of label embeddings and omics embeddings. Due to the contrastive learning objective, strings ob labels should be more similar to those cells that actually carry that label. The similarity between these embeddings can be used to compute ROC curves, for each label and for the whole dataset. This is a primary evaluation metric for classification tasks. Furthermore the annotation accuracy, also in comparison to a random annotation is returned. The .plot() method of this evaluator creates ROC curves, histograms of similarity scores and umaps, coloured by the similarity scores, and coloured by true and predicted labels.

The class {class}`mmcontext.eval.query_annotate.OmicsQueryAnnotator` can be used to query a dataset with a natural language query of choice. Provided a model, the similarites of samples and query are evaluated in the embedding space, which allows to visualise the response of the dataset to the query with the function {func}`mmcontext.pl.plotting.plot_query_scores_with_labels_umap`.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    eval.query_annotate.OmicsQueryAnnotator
    eval.label_similarity.LabelSimilarity
    pl.plotting.plot_query_scores_with_labels_umap
```

## Embedding alignment

As a further evaluation metric we quantify the alignment of cross-modal pairs in the shared latent space. The function {func}`mmcontext.eval.embedding_alignment.evaluate_modality_alignment` returns two metrics. The first is a modality-gap irrevelvant score. This means, only the distance between true cross-modal pairs and false cross-modal pairs is calculated. The other score also considers intra-modal false pairs. In a not well aligned latent space, intra-modal faulty pairs might be much closer than true cross-modal pairs. But for annotation, it is mostly important to have a good distinguishment between true and false cross-modal pairs.
