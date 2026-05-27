"""mmcontext.modules — Sentence-transformers pipeline modules.

This package provides the modules that form the sentence-transformers pipeline
for multimodal (text + omics) encoding:

- :class:`MMContextModule` — InputModule: text encoder + omics pass-through
- :class:`AdapterModule` — modality-aware projection to shared space
- OmicsAttentionModule (Phase 4) — optional self-attention for var-based models
"""

from .adapter_module import AdapterModule
from .mmcontext_module import MMContextModule

__all__ = ["MMContextModule", "AdapterModule"]
