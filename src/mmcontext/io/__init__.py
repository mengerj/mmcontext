"""mmcontext.io — Data loading and vector storage utilities.

This module provides disk-backed vector storage for omics embeddings
and utilities for building stores from AnnData zarr archives.
"""

from ._adata_subset import collect_adata_subset
from .prepare_store import build_namespaced_vector_store, prepare_vector_store
from .vector_store import VectorStore

__all__ = ["VectorStore", "build_namespaced_vector_store", "collect_adata_subset", "prepare_vector_store"]
