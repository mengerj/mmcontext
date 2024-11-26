import os
from pathlib import Path

import anndata

from mmcontext.eval import DataProperties, SystemMonitor, scibEvaluator
from mmcontext.pl.plotting import plot_umap
from mmcontext.pp import ConfigWorker, consolidate_low_frequency_categories


def eval_main(pp_hash, train_hash):
    """Main function for evaluating the inferred data obtained from the model."""
    model_out_dir = Path(f"out/processed/{pp_hash}/model_out/{train_hash}")
    # Track system usage
    monitor = SystemMonitor(interval=1)
    monitor.start()
    # Configure the config worker
    cfg_worker = ConfigWorker()
    eval_dir = model_out_dir / "eval"
    fig_dir = eval_dir / "figs"
    # Load the settings used to train the model
    settings_file = model_out_dir / "settings.yaml"
    cfg = cfg_worker.load_config(settings_file)
    data_dir = model_out_dir / "data"
    # Initialize the property evaluator
    dp = DataProperties(predefined_subset="microbiome")
    # get all the folders in data_dir (all the inferred data)
    filenames = os.listdir(data_dir)
    for filename in filenames:
        adata = anndata.read_zarr(data_dir / filename)
        # Initialize scibEvaluator
        evaluator = scibEvaluator(
            adata=adata,
            batch_key=cfg.get("data").get("batch_key"),
            label_key=cfg.get("data").get("cell_type_key"),
            embedding_key=adata.obsm.keys(),
        )

        res = evaluator.evaluate()
        # save results as csv
        res.to_csv(eval_dir / f"evaluation_{filename}.csv")
        # To get data properties and compare them, add data to the DataProperties object
        dp.add_orginal_data(adata.X.toarray(), id=filename)
        # Get layers with the string "reconstructed" in them
        reconstructed_keys = [layer for layer in adata.layers if "reconstructed" in layer]
        # Add reconstructed data to the DataProperties object
        for reconstructed_key in reconstructed_keys:
            dp.add_reconstructed_data(adata.layers[reconstructed_key])

        # plot umap
        # Remove low frequency categories for better visualization
        adata = consolidate_low_frequency_categories(
            adata, columns=[cfg.get("data").get("cell_type_key")], threshold=10, remove=True
        )
        for emb_key in [adata.obsm.keys()]:
            for label in [cfg.get("data").get("cell_type_key"), cfg.get("data").get("batch_key")]:
                plot_umap(
                    adata,
                    color_group=label,
                    embedding_key=emb_key,
                    save_plot=True,
                    save_dir=fig_dir / "umap",
                    nametag=emb_key,
                    sample_size=5000,
                )
    dp.compare_data_properties()
    # Save log2fc of all properties
    dp.log2fc_df.to_csv(eval_dir / "log2fc.csv")
    # Save mean and sdt of the mean log2fc of all properties
    dp.mean_std_df.to_csv(eval_dir / "mean_std_meanlog2fc.csv")
    # Plot a pca of the different data sets and their reconstructions
    dp.plot_pca(save_path=fig_dir / "pca.png")
    # Plot the metrics as boxplots
    dp.plot_metrics(save_dir=fig_dir)

    # Stop tracking system metrics
    monitor.stop()
    # Save system metrics in text form
    with open(eval_dir / "system_summary.txt", "w") as f:
        f.write(monitor.print_summary())
    # Save metric plots
    monitor.plot_metrics(save_dir=fig_dir)


if __name__ == "__main__":
    eval_main(pp_hash="ca46371a702966c1f7028a1bab4a0433", train_hash="01cc22c6ef6ee2ad05f8d61488da0460")
