# tests/test_evaluator.py

import logging
import sys
import warnings

import numpy as np
import pandas as pd

from mmcontext.eval import Evaluator  # Replace with the actual import path
from mmcontext.utils import create_test_emb_anndata  # Replace with the actual import path


def test_evaluator_metrics():
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Initialize logger
    logger = logging.getLogger("__name__")
    logger.setLevel(logging.INFO)
    logger.info("TEST: test_evaluator_metrics")

    # Create synthetic AnnData object
    n_samples = 200  # Increase sample size if necessary

    # Create pre-integration data
    adata = create_test_emb_anndata(n_samples=n_samples, emb_dim=20, data_key="d_emb", context_key="c_emb")
    # Simulate reconstruction
    adata.layers["reconstructed"] = adata.X.copy() + np.random.normal(0, 0.01, adata.X.shape)

    # Define keys
    embedding_key = "d_emb"
    batch_key = "batch"
    label_key = "cell_type"

    # Initialize Evaluator
    evaluator = Evaluator(
        adata=adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_key=embedding_key,
        n_top_genes=None,
        logger=logger,
    )

    # Evaluate metrics
    results_df = evaluator.evaluate()
    print(results_df)

    # Expected columns in the results DataFrame
    expected_columns = [
        "data_id",
        "hvg",
        "type",
        "ARI",
        "NMI",
        "ASW",
        "Isolated_Labels_ASW",
        "Isolated_Labels_F1",
        "Bio_Conservation_Score",
        "Graph_Connectivity",
        "Silhouette_Batch",
        "PCR",
        "Batch_Integration_Score",
        "Overall_Score",
    ]

    # Additional metrics that may fail on Windows
    ilisi_metrics = ["iLISI", "cLISI"]

    # Check if we are on Windows
    is_windows = sys.platform.startswith("win")

    # Assertions
    # Check if all expected columns are present
    for column in expected_columns:
        assert column in results_df.columns, f"Column '{column}' not found in results DataFrame"

    # Check that each metric has a value (is not None) in each row
    for idx, row in results_df.iterrows():
        print(f"Evaluating row {idx} with type '{row['type']}'")
        for metric in expected_columns:
            # Skip 'PCR' on 'raw' type since it's not computed
            if row["type"] == "raw" and metric == "PCR":
                continue  # PCR is not computed for 'raw' type
            # Handle Windows-specific metrics
            if is_windows and metric in ilisi_metrics:
                if pd.isnull(row[metric]):
                    print(f"{metric}: Not computed (expected on Windows)")
                else:
                    print(f"{metric}: {row[metric]} (computed on Windows)")
            else:
                # General assertion for other metrics
                assert row[metric] is not None, f"Metric '{metric}' has no value in row {idx}"
                print(f"{metric}: {row[metric]}")
