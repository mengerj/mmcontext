# tests/test_evaluator.py

"""
def test_evaluator_metrics():
    # Initialize logger
    logger = logging.getLogger('__name__')
    logger.info("TEST: test_evaluator_metrics")
    # Create synthetic AnnData object with batches and cell types
    n_samples = 100
    n_features = 50
    cell_types = ['TypeA', 'TypeB', 'TypeC']
    batch_categories = ['Batch1', 'Batch2']
    adata = create_test_anndata(
        n_samples=n_samples,
        n_features=n_features,
        cell_types=cell_types,
        batch_categories=batch_categories
    )
    # Compute PCA embeddings and store in adata.obsm['X_pca']
    #sc.pp.normalize_total(adata)
    #sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10)

    embedding_key = 'X_pca'
    batch_key = 'batch'
    label_key = 'cell_type'

    # List of metrics to test
    metrics_list = [
        'isolated_labels_asw_',
        'silhouette_',
        'pcr_',
        'isolated_labels_f1_',
        'nmi_',
        'ari_',
        'kBET_',
        'ilisi_',
        'clisi_',
    ]

    # Loop over metrics
    for metric in metrics_list:
        # Create config for this metric
        config = {
            'metrics': {metric: True}
        }
        evaluator = Evaluator(
            adata=adata,
            batch_key=batch_key,
            label_key=label_key,
            embedding_key=embedding_key,
            config=config,
            logger=logger
        )
        # Evaluate embeddings

        results = evaluator.evaluate_embeddings()
        metric_name = metric.rstrip('_')
        assert metric_name in results, f"Metric {metric_name} not computed"
        logger.info(f"Metric {metric_name}: Computed successfully. Result: {results[metric_name]}")
"""
