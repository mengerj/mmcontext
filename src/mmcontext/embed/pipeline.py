import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from mmcontext.embed.dataset_utils import collect_adata_subset, load_generic_dataset
from mmcontext.embed.model_utils import embed_labels, load_st_model, prepare_model_and_embed
from mmcontext.file_utils import save_table

logger = logging.getLogger(__name__)


def embed_pipeline(cfg) -> None:
    """
    Orchestrate an end-to-end embedding generation run.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Instantiated Hydra config. The schema matches ``conf/config.yaml``.
    """
    # --------------------------------------------------------------------- #
    #  loop over datasets and models
    # --------------------------------------------------------------------- #

    for ds_cfg in cfg.datasets:
        adata_download_dir = Path(cfg.output.adata_cache) / ds_cfg.name
        raw_ds = load_generic_dataset(
            source=ds_cfg.source,
            fmt=ds_cfg.format,
            split=ds_cfg.get("split", "test"),
            max_rows=cfg.run.n_rows,
            seed=cfg.run.seed,
        )
        numeric_data_available = "share_link" in raw_ds.column_names

        for model_cfg in cfg.models:
            model_id = model_cfg.source
            text_only = model_cfg.get("text_only", False)
            st_model = load_st_model(model_id)
            if text_only:
                main_col = "cell_sentence_2"
            else:
                main_col = "cell_sentence_1"
            emb_df, path_map = prepare_model_and_embed(
                st_model,
                data=raw_ds,
                main_col=main_col,
                index_col=ds_cfg.index_col,
                batch_size=cfg.run.batch_size,
                num_workers=cfg.run.num_workers,
                layer_key=ds_cfg.layer_key,
                text_only=text_only,
                adata_download_dir=adata_download_dir,
            )

            # add a small string to the model indicating if it was used as text-only
            if text_only:
                model_id = model_id + "_text_only"
            if numeric_data_available:
                # get sample_ids but without "sample_idx: prefix"
                if "sample_idx:" in emb_df["sample_idx"][0]:
                    sample_ids = [sid.split(":")[1] for sid in emb_df["sample_idx"].tolist()]
                elif "sample_idx" in emb_df.columns:
                    sample_ids = emb_df["sample_idx"].tolist()
                else:
                    raise ValueError(f"sample_idx column not found in emb_df: {emb_df.columns}")

                # Use path mapping if available (for local datasets), otherwise fall back to download_dir
                if path_map is not None:
                    file_paths = list(path_map.values())
                    adata_subset = collect_adata_subset(
                        sample_ids=sample_ids,
                        file_paths=file_paths,
                    )
                else:
                    adata_subset = collect_adata_subset(
                        download_dir=adata_download_dir,
                        sample_ids=sample_ids,
                    )
            else:
                adata_subset = None
            out_dir = (
                Path(cfg.output.root)
                / ds_cfg.name  # <— dataset-specific folder
                / Path(model_id).name.replace("/", "_")
            )
            # save the embeddings
            save_table(
                emb_df,
                out_path=out_dir / "embeddings",
                fmt=cfg.output.format,
            )

            # Handle label embeddings if available
            if adata_subset is not None:
                # Define label types and their output prefixes
                label_types = {"bio_label_list": "bio_label_embeddings", "batch_label_list": "batch_label_embeddings"}

                # Loop over each label type
                for label_list_attr, output_prefix in label_types.items():
                    if hasattr(ds_cfg, label_list_attr):
                        label_cols = getattr(ds_cfg, label_list_attr)
                        for label_col in label_cols:
                            if label_col in adata_subset.obs.columns:
                                logger.info(f"Embedding {label_list_attr} from column: {label_col}")
                                label_emb_df = embed_labels(
                                    st_model,
                                    adata_subset,
                                    label_col,
                                    batch_size=cfg.run.batch_size,
                                    num_workers=cfg.run.num_workers,
                                )
                                # Save label embeddings
                                save_table(
                                    label_emb_df,
                                    out_path=out_dir / f"{output_prefix}_{label_col}",
                                    fmt=cfg.output.format,
                                )
                            else:
                                logger.warning(f"Label column {label_col} not found in adata.obs")

            # if available, save the zarr store
            if adata_subset:
                subset_out = out_dir / "subset.zarr"
                adata_subset.write_zarr(subset_out)
                logger.info("Wrote subset AnnData → %s", subset_out)
            (out_dir / "meta.yaml").write_text(f"model: {model_id}\ndataset: {ds_cfg.name}\nrows: {len(emb_df)}\n")
