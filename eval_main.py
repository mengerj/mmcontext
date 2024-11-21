import gzip
import os
import pickle
from pathlib import Path

import anndata

from mmcontext.eval import compare_data_properties, scibEvaluator
from mmcontext.pl.plotting import plot_umap
from mmcontext.pp import ConfigWorker, consolidate_low_frequency_categories


def eval_main(model_out_dir: str):
    """Main function for evaluating the inferred data obtained from the model."""
    # Configure the config worker
    cfg_worker = ConfigWorker()
    model_out_dir = Path(model_out_dir)
    eval_dir = model_out_dir / "eval"
    fig_dir = eval_dir / "figs"
    # Load the settings used to train the model
    settings_file = model_out_dir / "settings.yaml"
    cfg = cfg_worker.load_config(settings_file)
    data_dir = model_out_dir / "data"
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

        properties = compare_data_properties(adata.X.toarray(), adata.layers["reconstructed"], predefined_subset="all")
        # save properties as pickle
        with gzip.open(eval_dir / f"properties_{filename}.pkl.gz", "wb") as f:
            pickle.dump(properties, f)
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
                )


if __name__ == "__main__":
    eval_main("out/processed/ca46371a702966c1f7028a1bab4a0433/model_out/01cc22c6ef6ee2ad05f8d61488da0460")
