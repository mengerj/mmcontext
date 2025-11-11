#!/usr/bin/env python3
"""
Standalone CellWhisperer processing script.

This script runs CellWhisperer in isolation and can be executed independently
for debugging or called from the main pipeline via subprocess.

Usage:
    python scripts/run_cellwhisperer_standalone.py \
        --adata_path /path/to/data.h5ad \
        --cache_dir /path/to/cache \
        --output_file /path/to/results.pkl \
        --batch_size 8 \
        --annotation_keys "AIFI_L1,AIFI_L2"
"""

import argparse
import json
import logging
import pickle
import sys
import traceback
from pathlib import Path

# Add CellWhisperer to Python path if it exists
script_dir = Path(__file__).parent
repo_root = script_dir.parent
cellwhisperer_dir = repo_root / "modules" / "CellWhisperer"
if cellwhisperer_dir.exists():
    sys.path.insert(0, str(cellwhisperer_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_if_missing(url, dest, chunk_size=8192):
    """Simple download function"""
    import requests
    from tqdm import tqdm

    dest = Path(dest)
    if dest.exists():
        logger.info(f"File already exists: {dest}")
        return dest

    logger.info(f"Downloading {url} to {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))

            with open(dest, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
                    for chunk in r.iter_content(chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        logger.info(f"✓ Download completed: {dest}")
        return dest

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if dest.exists():
            dest.unlink()  # Clean up partial download
        raise


def load_cellwhisperer_model(cache_dir):
    """Load CellWhisperer model components."""
    logger.info("Loading CellWhisperer model...")

    # Import CellWhisperer utilities
    from cellwhisperer.utils.model_io import load_cellwhisperer_model

    # Download model checkpoint if needed
    ckpt_path = cache_dir / "cellwhisperer_emb" / "cellwhisperer_jointemb_v1.ckpt"
    url = "https://medical-epigenomics.org/papers/schaefer2024/data/models/cellwhisperer_clip_v1.ckpt"

    download_if_missing(url, ckpt_path)

    # Load model
    logger.info("Loading CellWhisperer model from checkpoint...")
    pl_model, tokenizer, transcriptome_processor = load_cellwhisperer_model(ckpt_path, eval=True, cache=False)
    model = pl_model.model

    logger.info("✓ CellWhisperer model loaded successfully")
    return pl_model, tokenizer, transcriptome_processor, model


def setup_device(pl_model):
    """Setup compute device and move model."""
    import torch

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        batch_size = 16  # Conservative for MPS
        logger.info("Using Mac MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        batch_size = 32
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        batch_size = 8
        logger.info("Using CPU device")

    # Move model to device
    logger.info(f"Moving model to {device}...")
    pl_model = pl_model.to(device)
    model = pl_model.model

    # Get logit scale
    logit_scale = pl_model.model.discriminator.temperature.exp()

    logger.info(f"✓ Model setup complete on {device}")
    return pl_model, model, logit_scale, device, batch_size


def create_embeddings(adata, model, transcriptome_processor, batch_size):
    """Create cell embeddings using CellWhisperer's transcriptome encoder."""
    import numpy as np
    import pandas as pd
    from cellwhisperer.utils.inference import adata_to_embeds
    from tqdm import tqdm

    logger.info(f"Creating embeddings for {adata.n_obs} cells...")
    # set adata.X to adata.layers["counts"] for cellwhisperer
    if "counts" in adata.layers:
        logger.info("Setting adata.X to adata.layers['counts']")
        adata.X = adata.layers["counts"]
    else:
        logger.error("counts layer not found in adata.layers, cellwhisperer requires raw counts")
        raise ValueError("counts layer not found in adata.layers")

    # Use CellWhisperer's adata_to_embeds function to get real embeddings
    embeddings_tensor = adata_to_embeds(adata, model, transcriptome_processor, batch_size)

    # Convert tensor to numpy and create DataFrame
    embeddings_array = embeddings_tensor.cpu().numpy()

    embeddings_list = []
    sample_indices = []

    for i, embedding in enumerate(embeddings_array):
        sample_indices.append(f"sample_idx:{i}")
        embeddings_list.append(embedding.tolist())

    embeddings_df = pd.DataFrame({"sample_idx": sample_indices, "embedding": embeddings_list})

    logger.info(f"✓ Created {len(embeddings_df)} embeddings with dimension {len(embeddings_list[0])}")
    return embeddings_df, embeddings_tensor


def create_label_embeddings(labels, model, tokenizer, batch_size):
    """Create embeddings for label strings using CellWhisperer text encoder."""
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

    logger.info(f"Creating label embeddings for {len(labels)} unique labels...")

    embeddings_list = []
    sample_indices = []

    # Process labels in batches
    for start_idx in tqdm(range(0, len(labels), batch_size), desc="Creating label embeddings"):
        end_idx = min(start_idx + batch_size, len(labels))
        batch_labels = labels[start_idx:end_idx]

        # Get text embeddings using model's embed_texts method
        with torch.no_grad():
            text_embeddings = model.embed_texts(batch_labels, chunk_size=len(batch_labels))
            text_embeddings = text_embeddings.cpu().numpy()

        # Store embeddings
        for i, embedding in enumerate(text_embeddings):
            sample_idx = start_idx + i
            sample_indices.append(sample_idx)
            embeddings_list.append(embedding.tolist())

    embeddings_df = pd.DataFrame({"sample_idx": sample_indices, "embedding": embeddings_list})

    logger.info(f"✓ Created {len(embeddings_df)} label embeddings with dimension {len(embeddings_list[0])}")
    return embeddings_df


def process_similarity_scores(
    adata, embeddings_tensor, annotation_keys, model, tokenizer, transcriptome_processor, logit_scale, batch_size
):
    """Process similarity scores for annotation keys."""
    from cellwhisperer.utils.inference import score_transcriptomes_vs_texts

    similarity_results = {}

    if not annotation_keys:
        logger.info("No annotation keys provided, skipping similarity scoring")
        return similarity_results

    for annotation_key in annotation_keys:
        if annotation_key not in adata.obs.columns:
            logger.warning(f"Annotation key '{annotation_key}' not found in adata.obs")
            continue

        logger.info(f"Processing similarity scores for {annotation_key}")

        # Get unique cell type labels
        cell_type_labels = adata.obs[annotation_key].unique().tolist()
        logger.info(f"Found {len(cell_type_labels)} unique labels for {annotation_key}")

        try:
            # Compute similarity scores
            logger.info("Computing transcriptome vs text similarity scores...")
            scores, grouping_keys = score_transcriptomes_vs_texts(
                transcriptome_input=embeddings_tensor,
                text_list_or_text_embeds=cell_type_labels,
                logit_scale=logit_scale,
                model=model,
                transcriptome_processor=transcriptome_processor,
                batch_size=batch_size,
                average_mode=None,
                score_norm_method="softmax",
            )

            # Get predicted labels
            predicted_indices = scores.argmax(dim=0)
            predicted_labels = [cell_type_labels[i] for i in predicted_indices]

            similarity_results[annotation_key] = {
                "cell_type_labels": cell_type_labels,
                "predicted_labels": predicted_labels,
                "true_labels": adata.obs[annotation_key].values.tolist(),
                "scores_shape": list(scores.shape),  # Can't pickle tensors easily
            }

            logger.info(f"✓ Processed similarity scores for {annotation_key}")

        except Exception as e:
            logger.error(f"Error processing {annotation_key}: {e}")
            logger.error(traceback.format_exc())
            similarity_results[annotation_key] = {
                "error": str(e),
                "cell_type_labels": cell_type_labels,
                "predicted_labels": [],
                "true_labels": adata.obs[annotation_key].values.tolist(),
                "scores_shape": [0, 0],
            }

    return similarity_results


def run_cellwhisperer_processing(
    adata_path: str, cache_dir: str, batch_size: int = 8, annotation_keys: list[str] = None, output_dir: str = None
) -> dict:
    """
    Run CellWhisperer processing as a Python function.

    Parameters
    ----------
    adata_path : str
        Path to AnnData file
    cache_dir : str
        Cache directory for models
    batch_size : int
        Batch size for processing
    annotation_keys : list[str]
        List of annotation keys to process for similarity scoring
    output_dir : str, optional
        Output directory for saving label embeddings

    Returns
    -------
    dict
        Results dictionary with embeddings_df, similarity_results, success, error, etc.
    """
    logger.info("Starting CellWhisperer processing...")
    logger.info(f"AnnData path: {adata_path}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Annotation keys: {annotation_keys}")

    try:
        # Load data
        import anndata as ad

        logger.info("Loading AnnData...")
        adata = ad.read_h5ad(adata_path)
        logger.info(f"✓ Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes")

        # Load model
        cache_dir_path = Path(cache_dir)
        pl_model, tokenizer, transcriptome_processor, model = load_cellwhisperer_model(cache_dir_path)

        # Setup device
        pl_model, model, logit_scale, device, recommended_batch_size = setup_device(pl_model)

        # Use provided batch size or recommended
        final_batch_size = batch_size if batch_size > 0 else recommended_batch_size
        logger.info(f"Using batch size: {final_batch_size}")

        # Create embeddings
        embeddings_df, embeddings_tensor = create_embeddings(adata, model, transcriptome_processor, final_batch_size)
        # Process similarity scores
        similarity_results = process_similarity_scores(
            adata,
            embeddings_tensor,
            annotation_keys or [],
            model,
            tokenizer,
            transcriptome_processor,
            logit_scale,
            final_batch_size,
        )

        # Generate label embeddings for evaluation
        label_embeddings_results = {}
        if output_dir and annotation_keys:
            import json

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            for annotation_key in annotation_keys:
                if annotation_key in adata.obs.columns:
                    logger.info(f"Generating label embeddings for {annotation_key}")

                    # Get unique labels
                    unique_labels = adata.obs[annotation_key].astype(str).unique().tolist()

                    # Create label embeddings
                    label_emb_df = create_label_embeddings(unique_labels, model, tokenizer, final_batch_size)

                    # Create label to index mapping
                    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

                    # Save label embeddings
                    label_emb_path = output_path / f"bio_label_embeddings_{annotation_key}.parquet"
                    label_emb_df.to_parquet(label_emb_path, index=False)

                    # Save label mapping
                    mapping_path = output_path / f"bio_label_embeddings_{annotation_key}_mapping.json"
                    with open(mapping_path, "w") as f:
                        json.dump(label_to_index, f, indent=2)

                    label_embeddings_results[annotation_key] = {
                        "embeddings_path": str(label_emb_path),
                        "mapping_path": str(mapping_path),
                        "n_labels": len(unique_labels),
                    }

                    logger.info(f"✓ Saved label embeddings for {annotation_key}: {len(unique_labels)} labels")

        # Return results
        results = {
            "embeddings_df": embeddings_df,
            "similarity_results": similarity_results,
            "label_embeddings_results": label_embeddings_results,
            "success": True,
            "error": None,
            "device": str(device),
            "batch_size": final_batch_size,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
        }

        logger.info("CellWhisperer processing completed successfully!")
        return results

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"CellWhisperer processing failed: {error_msg}")

        # Return error results
        results = {
            "embeddings_df": None,
            "similarity_results": {},
            "success": False,
            "error": error_msg,
            "device": "unknown",
            "batch_size": batch_size,
            "n_cells": 0,
            "n_genes": 0,
        }

        return results


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Run CellWhisperer processing")
    parser.add_argument("--adata_path", required=True, help="Path to AnnData file")
    parser.add_argument("--cache_dir", required=True, help="Cache directory for models")
    parser.add_argument("--output_file", required=True, help="Output pickle file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--annotation_keys", help="Comma-separated annotation keys")
    parser.add_argument("--output_dir", help="Output directory for label embeddings")

    args = parser.parse_args()

    # Parse annotation keys
    annotation_keys = []
    if args.annotation_keys:
        annotation_keys = [key.strip() for key in args.annotation_keys.split(",")]

    # Run processing
    results = run_cellwhisperer_processing(
        adata_path=args.adata_path,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        annotation_keys=annotation_keys,
        output_dir=args.output_dir,
    )

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    logger.info(f"✓ Results saved to {output_path}")

    # Exit with appropriate code
    if results["success"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
