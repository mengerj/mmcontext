"""
CellWhisperer embedding utilities for integration with mmcontext pipeline.

This module provides functions to generate embeddings using CellWhisperer models
and integrate them into the mmcontext embedding/evaluation pipeline.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def ensure_cellwhisperer_setup(modules_dir: Path) -> bool:
    """
    Ensure CellWhisperer is properly set up.

    This is done by cloning the repository, setting up the model, creating a virtual environment, and installing dependencies.

    Parameters
    ----------
    modules_dir : Path
        Directory where CellWhisperer should be cloned/installed

    Returns
    -------
    bool
        True if setup successful, False otherwise
    """
    cellwhisperer_dir = modules_dir / "CellWhisperer"
    cellwhisperer_requirements = modules_dir / "requirements_cellwhisperer.txt"
    cellwhisperer_venv = modules_dir / "cellwhisperer_venv"
    cellwhisperer_python = cellwhisperer_venv / "bin" / "python"

    # Check if CellWhisperer is already set up (repository + model + venv + python)
    setup_complete = (
        cellwhisperer_dir.exists()
        and (cellwhisperer_dir / "resources" / "geneformer-12L-30M").exists()
        and cellwhisperer_venv.exists()
        and cellwhisperer_python.exists()
    )

    if setup_complete:
        logger.info("CellWhisperer already set up with virtual environment")
        return True

    try:
        # Step 1: Run prepare script to clone repository and set up model
        prepare_script = modules_dir / "prepare_cellwhisperer.sh"
        if not prepare_script.exists():
            logger.error(f"Prepare script not found at {prepare_script}")
            return False

        logger.info("Running CellWhisperer prepare script...")
        result = subprocess.run(
            ["bash", str(prepare_script)],
            cwd=modules_dir.parent,  # Run from repo root
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"CellWhisperer prepare script failed with return code {result.returncode}")
            if result.stdout:
                logger.error(f"Prepare stdout: {result.stdout}")
            if result.stderr:
                logger.error(f"Prepare stderr: {result.stderr}")
            return False

        logger.info("CellWhisperer prepare script completed successfully")
        if result.stdout:
            logger.info(f"Prepare output: {result.stdout}")

        # Step 2: Create virtual environment
        logger.info("Creating CellWhisperer virtual environment...")
        if cellwhisperer_venv.exists():
            logger.info("Virtual environment already exists, removing...")
            import shutil

            shutil.rmtree(cellwhisperer_venv)

        result = subprocess.run(
            [sys.executable, "-m", "venv", str(cellwhisperer_venv)], capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            logger.error(f"Failed to create virtual environment: {result.stderr}")
            return False

        logger.info("✓ Virtual environment created successfully")

        # Step 3: Install requirements in the virtual environment
        if not cellwhisperer_requirements.exists():
            logger.error(f"Requirements file not found at {cellwhisperer_requirements}")
            return False

        logger.info("Installing CellWhisperer dependencies...")
        result = subprocess.run(
            [str(cellwhisperer_python), "-m", "pip", "install", "-r", str(cellwhisperer_requirements)],
            capture_output=True,
            text=True,
            timeout=1200,  # 20 minute timeout for package installation
        )

        if result.returncode != 0:
            logger.error(f"Failed to install requirements: {result.stderr}")
            if result.stdout:
                logger.error(f"Install stdout: {result.stdout}")
            return False

        logger.info("✓ CellWhisperer dependencies installed successfully")
        if result.stdout:
            logger.info(f"Install output: {result.stdout}")

        # Verify setup is complete
        if not cellwhisperer_python.exists():
            logger.error("CellWhisperer venv Python not found after setup")
            return False

        logger.info("✓ CellWhisperer setup completed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error("CellWhisperer setup timed out")
        return False
    except Exception as e:
        logger.error(f"Error running CellWhisperer setup: {e}")
        return False


def create_label_embeddings(labels, model, tokenizer, batch_size):
    """Create embeddings for label strings using CellWhisperer text encoder."""
    import pandas as pd

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


def create_cellwhisperer_embeddings_venv(
    adata_path: Path, modules_dir: Path, cache_dir: Path, batch_size: int = 32, annotation_keys: list[str] = None
) -> tuple[pd.DataFrame, dict]:
    """
    Create embeddings using CellWhisperer model running in separate venv.

    Parameters
    ----------
    adata_path : Path
        Path to saved AnnData file
    modules_dir : Path
        Directory containing CellWhisperer installation
    cache_dir : Path
        Cache directory for model checkpoints
    batch_size : int
        Batch size for processing
    annotation_keys : list[str]
        Annotation keys to process for similarity scoring

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (embeddings_df, similarity_results)
    """
    import pickle
    import tempfile

    logger.info("Running CellWhisperer in separate virtual environment...")

    # Create temporary files for communication
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        results_file = temp_dir / "cellwhisperer_results.pkl"

        # Get path to standalone script
        script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "run_cellwhisperer_standalone.py"
        if not script_path.exists():
            raise RuntimeError(f"CellWhisperer standalone script not found: {script_path}")

        # Prepare arguments
        args = [
            "--adata_path",
            str(adata_path),
            "--cache_dir",
            str(cache_dir),
            "--output_file",
            str(results_file),
            "--batch_size",
            str(batch_size),
        ]

        if annotation_keys:
            args.extend(["--annotation_keys", ",".join(annotation_keys)])
            # Pass project cache directory for label embeddings (not HuggingFace cache)
            current_dir = Path.cwd()
            if "runs" in str(current_dir):
                # We're in a Hydra run directory, go up to project root
                project_root = current_dir
                while project_root.name != "mmcontext" and project_root.parent != project_root:
                    project_root = project_root.parent
                project_cache_dir = project_root / "cache"
            else:
                # We're already in project root
                project_cache_dir = current_dir / "cache"
            args.extend(["--output_dir", str(project_cache_dir)])

        # Run CellWhisperer standalone script in venv
        cellwhisperer_python = modules_dir / "cellwhisperer_venv" / "bin" / "python"

        if not cellwhisperer_python.exists():
            raise RuntimeError(f"CellWhisperer venv Python not found at {cellwhisperer_python}")

        cmd = [str(cellwhisperer_python), str(script_path)] + args
        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=259200,  # 3 day timeout
        )

        logger.info(f"CellWhisperer script return code: {result.returncode}")
        if result.stdout:
            logger.info(f"CellWhisperer stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"CellWhisperer stderr: {result.stderr}")

        if result.returncode != 0:
            logger.error(f"CellWhisperer script failed with return code {result.returncode}")
            raise RuntimeError(f"CellWhisperer processing failed: {result.stderr}")

        # Load results
        if results_file.exists():
            with open(results_file, "rb") as f:
                results = pickle.load(f)

            if results["success"]:
                logger.info("✓ CellWhisperer processing completed successfully")
                logger.info(
                    f"Processed {results['n_cells']} cells with {results['n_genes']} genes on {results['device']}"
                )
                return results["embeddings_df"], results["similarity_results"]
            else:
                raise RuntimeError(f"CellWhisperer processing failed: {results['error']}")
        else:
            raise RuntimeError("CellWhisperer results file not found")

    return pd.DataFrame(), {}


def process_cellwhisperer_dataset_model(
    ds_cfg: Any,
    model_cfg: Any,
    run_cfg: Any,
    output_root: str,
    output_format: str,
    adata_cache: str,
    modules_dir: Path,
    hf_cache: str = None,
) -> tuple[str, str, bool, str]:
    """
    Process a single dataset with CellWhisperer model for embedding generation.

    This function follows the same pattern as process_single_dataset_model but
    uses CellWhisperer instead of sentence transformers.

    Parameters
    ----------
    ds_cfg : dict
        Dataset configuration
    model_cfg : dict
        Model configuration (should specify CellWhisperer)
    run_cfg : dict
        Run configuration
    output_root : str
        Output root directory
    output_format : str
        Output format (parquet/csv)
    adata_cache : str
        AnnData cache directory
    modules_dir : Path
        Directory containing CellWhisperer installation
    hf_cache : str, optional
        HuggingFace cache directory

    Returns
    -------
    tuple[str, str, bool, str]
        (dataset_name, model_name, success, error_msg)
    """
    from mmcontext.embed.dataset_utils import collect_adata_subset, load_generic_dataset
    from mmcontext.file_utils import save_table

    dataset_name = ds_cfg.name
    model_name = model_cfg.get("name", "cellwhisperer")

    logger.info(f"Processing CellWhisperer: {dataset_name} + {model_name}")

    # Determine output directory
    out_dir = Path(output_root) / ds_cfg.name / model_name

    # Check if files already exist and skip if overwrite is False
    from mmcontext.embed.embed_pipeline import check_required_files_exist

    if not run_cfg.overwrite and check_required_files_exist(out_dir, output_format):
        logger.info(f"CellWhisperer embeddings already exist for {ds_cfg.name} + {model_name}. Skipping.")
        return dataset_name, model_name, True, "skipped_existing"

    try:
        # Ensure CellWhisperer is set up
        if not ensure_cellwhisperer_setup(modules_dir):
            return dataset_name, model_name, False, "CellWhisperer setup failed"

        # Load dataset
        adata_download_dir = Path(adata_cache) / ds_cfg.name
        cache_dir = hf_cache if hf_cache is not None else getattr(run_cfg, "hf_cache", None)

        # Load raw dataset
        raw_ds = load_generic_dataset(
            source=ds_cfg.source,
            fmt=ds_cfg.format,
            split=ds_cfg.get("split", "test"),
            max_rows=run_cfg.n_rows,
            seed=run_cfg.seed,
            cache_dir=cache_dir,
        )

        # Check if numeric data is available
        numeric_data_available = "share_link" in raw_ds.column_names

        if not numeric_data_available:
            logger.warning("CellWhisperer requires numeric data (share_link), but none found")
            return dataset_name, model_name, False, "No numeric data available for CellWhisperer"

        # Get sample IDs and collect AnnData subset
        sample_ids = raw_ds[ds_cfg.index_col]
        if isinstance(sample_ids[0], str) and "sample_idx:" in sample_ids[0]:
            sample_ids = [sid.split(":")[1] for sid in sample_ids]

        adata_subset = collect_adata_subset(
            download_dir=adata_download_dir,
            sample_ids=sample_ids,
        )

        # Save AnnData to temporary file for CellWhisperer venv processing
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
            temp_adata_path = Path(tmp_file.name)
            adata_subset.write_h5ad(temp_adata_path)

        try:
            # Get annotation keys for similarity scoring
            annotation_keys = []
            if hasattr(ds_cfg, "bio_label_list") and ds_cfg.bio_label_list:
                annotation_keys.extend(ds_cfg.bio_label_list)

            # Create embeddings using CellWhisperer in separate venv
            embeddings_df, similarity_results = create_cellwhisperer_embeddings_venv(
                temp_adata_path,
                modules_dir,
                Path(cache_dir) if cache_dir else Path.cwd(),
                batch_size=run_cfg.batch_size,
                annotation_keys=annotation_keys,
            )

            # Create output directory
            out_dir.mkdir(parents=True, exist_ok=True)

            # Copy label embeddings to output directory if they were generated
            if annotation_keys:
                import shutil

                # Use project cache directory for label embeddings
                current_dir = Path.cwd()
                if "runs" in str(current_dir):
                    # We're in a Hydra run directory, go up to project root
                    project_root = current_dir
                    while project_root.name != "mmcontext" and project_root.parent != project_root:
                        project_root = project_root.parent
                    project_cache_dir = project_root / "cache"
                else:
                    # We're already in project root
                    project_cache_dir = current_dir / "cache"

                for annotation_key in annotation_keys:
                    # Look for label embedding files in project cache directory
                    label_emb_file = project_cache_dir / f"bio_label_embeddings_{annotation_key}.parquet"
                    mapping_file = project_cache_dir / f"bio_label_embeddings_{annotation_key}_mapping.json"

                    if label_emb_file.exists():
                        # Copy to output directory
                        shutil.copy2(label_emb_file, out_dir / label_emb_file.name)
                        logger.info(f"Copied label embeddings for {annotation_key}")
                    else:
                        logger.warning(f"Label embeddings file not found: {label_emb_file}")

                    if mapping_file.exists():
                        # Copy to output directory
                        shutil.copy2(mapping_file, out_dir / mapping_file.name)
                        logger.info(f"Copied label mapping for {annotation_key}")
                    else:
                        logger.warning(f"Label mapping file not found: {mapping_file}")
        finally:
            # Clean up temporary file
            if temp_adata_path.exists():
                temp_adata_path.unlink()

        # Save embeddings (output directory already created above)
        save_table(
            embeddings_df,
            out_path=out_dir / "embeddings",
            fmt=output_format,
        )

        # Save metadata
        (out_dir / "meta.yaml").write_text(f"model: {model_name}\ndataset: {ds_cfg.name}\nrows: {len(embeddings_df)}\n")

        # Save AnnData subset
        subset_out = out_dir / "subset.h5ad"
        adata_subset.write_h5ad(subset_out)
        logger.info("Wrote subset AnnData → %s", subset_out)

        # Save similarity results if available
        if similarity_results:
            import json

            for annotation_key, results in similarity_results.items():
                # Save predictions as CSV (separate from label embeddings)
                predictions_df = pd.DataFrame(
                    {
                        "cell_id": range(len(results["true_labels"])),
                        "true_labels": results["true_labels"],
                        "predicted_labels": results["predicted_labels"],
                    }
                )

                save_table(
                    predictions_df,
                    out_path=out_dir / f"cellwhisperer_predictions_{annotation_key}",
                    fmt=output_format,
                )

                # Save cell type labels
                with open(out_dir / f"cellwhisperer_labels_{annotation_key}.json", "w") as f:
                    json.dump(results["cell_type_labels"], f, indent=2)

        logger.info(f"✓ Completed CellWhisperer processing {dataset_name} + {model_name}")
        return dataset_name, model_name, True, None

    except Exception as e:
        logger.error(f"✗ Error processing CellWhisperer {dataset_name} + {model_name}: {e}")
        return dataset_name, model_name, False, str(e)
