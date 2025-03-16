# A custom multi-modal Sentence Transformers Model: 'models'

```{eval-rst}
.. module:: mmcontext.models
.. currentmodule:: mmcontext

```

While the Sentence Transformers framework was primarly used for text embedding models, it can be extended to handle multi-modal inputs fairly simple. At the moment, there is a small change in the source code nescessary, so make sure you have the package installed from this fork:
https://github.com/mengerj/sentence-transformers/tree/master

Other than that, we have to create a custom model and a custom processor. I will soon create more detailed instructions on how to build these, but essentially the model has to be a torch.nn.module, with a forward and a tokenise method (and saving and loading methods). The tokenize method gets the raw inputs, and passes them to the correct processor. This processor has to return a feature matrix for the current batch and from thereone, the standard Sentence Transformers framework takes over. This allows us to easly load/save them on huggingface hub, to benifit from the by huggingface mainted repository and all of it's features.

## Model

The main model is {class}`mmcontext.models.MMContextEncoder.MMContextEncoder`. It combines the text encoder model, and adapter layers and expects to be used with a Sentence Transformer Trainer and a huggingface dataset. This huggingface dataset is contains a reference to the initial embedding vector (already pre-processed), a sample_id and captions. There are "pairs" datasets, which contain a data reference column, a caption column and a label column. Or there are "multiplets" datasets. These different datasets can be used with different losses, as implemented and documented in the Sentence Transformers framework. The processor is implemented in {class}`mmcontext.models.MMContextProcessor.MMContextProcessor` and retrieves the data provided by a share-url. Of course, traditional tokenizers can be implemented as well.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    models.MMContextEncoder.MMContextEncoder
    models.MMContextProcessor.MMContextProcessor
```
