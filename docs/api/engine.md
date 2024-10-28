# Preprocessing: `engine`

```{eval-rst}
.. module:: mmcontext.engine
.. currentmodule:: mmcontext

```

This is the documentation of the engine part, which comprises the core functionality accociated with model training. This includes initializing the model, defining the loss function and training the model. Furhtermore a pre-trained model can be loaded to perform inference.
Classes from this module can be directly imported. eg:

```
from mmcontext.engine import MMContextEncoder
```

## Model initialization

The {class}`mmcontext.engine.MMContextEncoder` is the main encoder which will be used to create embeddings based on the available data and context embeddings. It's structure is flexible and is build upon torchs {class}`torch.nn.TransformerEncoder` which creates stacks of the {class}`mmcontext.engine.CustomTransformerEncoderLayer` which can be configured to be

1. An MLP only model
2. To apply self attention (use_self_attention = True)
3. To apply cross attention (use_cross_attention = True)
4. To use both self and cross attention (both True)

The model takes two matrix inputs, `in_main` and `in_cross`. in_main will be passed through the MLP and optionally the self-attention layers, while in_cross is only used if cross-attention is used. In the end the model outputs embeddings of the same shape as in_main, updated based on the learning objective. The inputs are handled in the {class}`mmcontext.engine.Trainer` in dependancy with the settings used for the loss.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    engine.BaseModel
    engine.MMContextEncoder
    engine.CustomTransformerEncoderLayer
```

## Loss Configuration

The loss function is a central part of this project. The loss is implemented to be extendable and customizable by having a {class}`mmcontext.engine.LossManager` which you can use to add different losses to and which will be passed to the {class}`mmcontext.engine.Trainer` to compute the total loss (as a weighted average of the components) during Training. The current main implementation of a {class}`mmcontext.engine.LossFunction` is the {class}`mmcontext.engine.ContrastiveLoss`, which is a custom approach to contrastive learning. It's main configuration parameters are `target_mode` and `current_mode` which refer to the way in which the target similarity matrix and the current (during model training) similarity matrix are constructed. For example if `target_mode = 'context_context'`, the original context embeddings are used to create the target similarity matrix. Therefore during training, the loss is calcualted as the mean squared error between the current similarity matrix and the one based on the context. If `current_mode = 'data_data'`, the model would learn to find representations of the data that represent similarity found in the respective context.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    engine.LossFunction
    engine.LossManager
    engine.ContrastiveLoss
    engine.ReconstructionLoss
```

## Training the Model

The Trainer uses the defined model and loss the conduct training, aiming to iterativly minimize the loss. The {func}`mmcontext.engine.Trainer.fit` method can take a training and a validation dataloader as input. If a validation loader is given and a save_path is used, the weights of the best performing model can be saved to file. The {class}`mmcontext.engine.MMContextEncoder` has a method `load` to load weights from file.
Per default data embeddings are used for `in_main` while context embeddings are used as `in_cross`.

```{eval-rst}
.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    engine.Trainer
```
