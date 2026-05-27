"""mmcontext.io — Data loading and vector storage utilities.

This module provides disk-backed vector storage for omics embeddings
and utilities for loading data from AnnData objects.
"""

from .vector_store import VectorStore

__all__ = ["VectorStore"]
