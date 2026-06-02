"""mmcontext.io — Data loading and vector storage utilities.

This module provides disk-backed vector storage for omics embeddings
and utilities for building stores from AnnData zarr archives.
"""

from .prepare_store import prepare_vector_store
from .vector_store import VectorStore

__all__ = ["VectorStore", "prepare_vector_store"]
