"""Utilities for uploading MMContext models to Hugging Face Hub with custom model cards."""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# Embedding method descriptions
EMBEDDING_DESCRIPTIONS = {
    "pca": "Principal Component Analysis (PCA) dimensionality reduction selection of genes (from gs10k) (50-dimensional)",
    "hvg": "Highly Variable Genes (HVG) selection and processing",
    "scvi_fm": "scVI Foundation Model embeddings - pre-trained on cellxgene corpus (50-dimensional)",
    "geneformer": "Geneformer transformer-based gene embeddings (768-dimensional)",
    "gs": "Gene Set enrichment-based embeddings, based on genes used to train scvi model, subesetted to availble genes in training data (3936 genes)",
    "gs10k": "Gene Set enrichment-based embeddings (10k genes)",
    "random": "Random embeddings (for testing purposes)",
}

# Model card template
MODEL_CARD_TEMPLATE = """

This model uses a custom MMContextEncoder architecture for multimodal embedding generation, combining text and omics data representations.

## ⚠️ Important: Loading Instructions

**This model requires `trust_remote_code=True` to load properly.**

```python
from sentence_transformers import SentenceTransformer

# ✅ CORRECT: Load with trust_remote_code=True
model = SentenceTransformer('{repo_id}', trust_remote_code=True)

# Generate embeddings
texts = ["Cell type annotation", "Another description"]
embeddings = model.encode(texts)
print(f"Embeddings shape: {{embeddings.shape}}")
```

## Model Details

- **Architecture**: MMContextEncoder (custom multimodal architecture)
- **Text Encoder**: {text_encoder}
- **Omics Embedding Method**: {embedding_method}
- **Output Dimension**: {output_dim}
- **Pooling Strategy**: {pooling_mode}

### Omics Embedding Method: {embedding_method_name}

{embedding_description}

## Usage Tutorial

{tutorial_info}

## Model Architecture

The MMContextEncoder combines:
- **Text Branch**: {text_encoder} with optional adapter layers
- **Omics Branch**: Lookup-based encoder with precomputed {embedding_method} embeddings
- **Adapters**: Feed-forward projection layers for dimensionality alignment
- **Pooling**: {pooling_mode} pooling for sentence-level embeddings

## Files in this Repository

- `mmcontextencoder.py`: Main model implementation
- `adapters.py`: Adapter modules for dimensionality mapping
- `omicsencoder.py`: Omics data encoder
- `onehot.py`: One-hot text encoder
- `file_utils.py`: Utility functions{notebook_info}

## Training Details

{training_details}

---

*This model was trained using the MMContext framework for multimodal single-cell analysis.*
"""


def create_model_card(
    model_name: str,
    repo_id: str,
    text_encoder: str,
    embedding_method: str,
    output_dim: int,
    pooling_mode: str = "mean",
    training_details: str | None = None,
    tutorial_notebook: str | None = None,
    include_notebook: bool = False,
) -> str:
    """
    Create a custom model card for MMContext models.

    Parameters
    ----------
    model_name : str
        Name of the model
    repo_id : str
        Hugging Face repository ID
    text_encoder : str
        Name of the text encoder used
    embedding_method : str
        Omics embedding method used (e.g., 'scvi_fm', 'gs', 'pca')
    output_dim : int
        Output dimension of the model
    pooling_mode : str, optional
        Pooling strategy used
    training_details : str, optional
        Additional training details to include
    tutorial_notebook : str, optional
        Path or name of tutorial notebook
    include_notebook : bool, optional
        Whether a notebook is included in the repository

    Returns
    -------
    str
        Formatted model card content
    """
    # Get embedding method description
    embedding_description = EMBEDDING_DESCRIPTIONS.get(embedding_method, f"Custom embedding method: {embedding_method}")

    # Format embedding method name
    embedding_method_name = embedding_method.upper().replace("_", " ")

    # Tutorial information with proper links
    if tutorial_notebook:
        if include_notebook:
            # Create direct link to the notebook in the repository
            notebook_url = f"https://huggingface.co/{repo_id}/blob/main/{tutorial_notebook}"
            tutorial_info = f"📓 **Tutorial Notebook**: [{tutorial_notebook}]({notebook_url}) - Detailed usage examples and best practices"
        else:
            tutorial_info = f"📓 **Tutorial Notebook**: [{tutorial_notebook}]({tutorial_notebook})"
    else:
        tutorial_info = "📓 **Tutorial**: Please refer to the MMContext documentation for usage examples."

    # Notebook file info with clickable link
    notebook_info = ""
    if include_notebook and tutorial_notebook:
        notebook_url = f"https://huggingface.co/{repo_id}/blob/main/{tutorial_notebook}"
        notebook_info = f"\n- [`{tutorial_notebook}`]({notebook_url}): Tutorial notebook with usage examples"

    # Default training details
    if training_details is None:
        training_details = f"""- **Embedding Method**: {embedding_method_name}
- **Text Encoder**: {text_encoder}
- **Architecture**: Dual-tower with adapter layers
- **Pooling**: {pooling_mode} pooling"""

    return MODEL_CARD_TEMPLATE.format(
        repo_id=repo_id,
        text_encoder=text_encoder,
        embedding_method=embedding_method,
        embedding_method_name=embedding_method_name,
        embedding_description=embedding_description,
        output_dim=output_dim,
        pooling_mode=pooling_mode,
        tutorial_info=tutorial_info,
        notebook_info=notebook_info,
        training_details=training_details,
    )


def copy_mmcontext_files(model_dir: Path) -> None:
    """
    Copy all MMContext files to the top level of model directory for simple direct imports.

    Parameters
    ----------
    model_dir : Path
        Directory where the model is saved
    """
    # Define the source directory
    src_dir = Path(__file__).parent  # src/mmcontext directory

    # Required files for MMContextEncoder - copy all to top level
    required_files = [
        "mmcontextencoder.py",
        "adapters.py",
        "omicsencoder.py",
        "onehot.py",
        "file_utils.py",
    ]

    # Copy each required file directly to the top level of model directory
    for filename in required_files:
        src_file = src_dir / filename
        if not src_file.exists():
            logger.warning(f"Required file not found: {src_file}")
            continue

        dest_file = model_dir / filename

        # Copy the file as-is - relative imports work fine when files are in the same directory
        shutil.copy2(src_file, dest_file)
        logger.info(f"Copied: {src_file} -> {dest_file}")


def upload_model_to_hub(
    model: SentenceTransformer,
    repo_id: str,
    model_name: str | None = None,
    text_encoder: str | None = None,
    embedding_method: str | None = None,
    output_dim: int | None = None,
    pooling_mode: str = "mean",
    training_details: str | None = None,
    tutorial_notebook: str | None = None,
    notebook_path: str | Path | None = None,
    private: bool = False,
    commit_message: str | None = None,
) -> str:
    """
    Upload MMContext model to Hugging Face Hub with custom model card.

    Parameters
    ----------
    model : SentenceTransformer
        The trained model to upload
    repo_id : str
        Hugging Face repository ID (e.g., 'username/model-name')
    model_name : str, optional
        Display name for the model (defaults to repo_id)
    text_encoder : str, optional
        Name of text encoder (auto-detected if not provided)
    embedding_method : str, optional
        Omics embedding method (auto-detected if not provided)
    output_dim : int, optional
        Output dimension (auto-detected if not provided)
    pooling_mode : str, optional
        Pooling strategy (auto-detected if not provided)
    training_details : str, optional
        Additional training details
    tutorial_notebook : str, optional
        Name of tutorial notebook
    notebook_path : str or Path, optional
        Path to notebook file to upload
    private : bool, optional
        Whether to make repository private
    commit_message : str, optional
        Custom commit message

    Returns
    -------
    str
        URL of the uploaded model repository
    """
    # Auto-detect model parameters if not provided
    if model_name is None:
        model_name = repo_id.split("/")[-1].replace("-", " ").title()

    # Try to extract parameters from the model
    mmcontext_encoder = None
    for module in model.modules():
        if hasattr(module, "__class__") and "MMContextEncoder" in str(module.__class__):
            mmcontext_encoder = module
            break

    if mmcontext_encoder:
        if text_encoder is None:
            text_encoder = getattr(mmcontext_encoder, "text_encoder_name", "Unknown")
        if embedding_method is None:
            embedding_method = getattr(mmcontext_encoder, "_registered_data_origin", "unknown")
        if output_dim is None:
            output_dim = getattr(mmcontext_encoder, "_output_dim", "Unknown")
        if pooling_mode == "mean":  # Only override default
            pooling_mode = getattr(mmcontext_encoder, "pooling_mode", "mean")

    # Set defaults for missing values
    text_encoder = text_encoder or "Unknown"
    embedding_method = embedding_method or "unknown"
    output_dim = output_dim or "Unknown"

    # Create custom model card
    include_notebook = notebook_path is not None
    model_card_content = create_model_card(
        model_name=model_name,
        repo_id=repo_id,
        text_encoder=text_encoder,
        embedding_method=embedding_method,
        output_dim=output_dim,
        pooling_mode=pooling_mode,
        training_details=training_details,
        tutorial_notebook=tutorial_notebook,
        include_notebook=include_notebook,
    )

    # Use temporary directory for preparation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "model"

        # Save model to temporary directory
        logger.info("Preparing model for upload...")
        model.save_pretrained(temp_path)

        # Copy MMContext files to top level
        logger.info("Adding MMContext files to top level...")
        copy_mmcontext_files(temp_path)

        # Update modules.json to reference the top-level mmcontextencoder file
        modules_json_path = temp_path / "modules.json"
        if modules_json_path.exists():
            import json

            with open(modules_json_path) as f:
                modules_config = json.load(f)

            # Update the first module to use top-level mmcontextencoder.py
            if modules_config and len(modules_config) > 0:
                modules_config[0]["type"] = "mmcontextencoder.MMContextEncoder"

            with open(modules_json_path, "w") as f:
                json.dump(modules_config, f, indent=2)

            logger.info("Updated modules.json to reference top-level mmcontextencoder.py")

        # Create combined README (custom content + SentenceTransformer content)
        readme_path = temp_path / "README.md"

        # Check if SentenceTransformer already created a README
        existing_readme = ""
        if readme_path.exists():
            existing_readme = readme_path.read_text()
            logger.info("Found existing SentenceTransformer README, will combine with custom content")

        # Combine custom content with existing content
        if existing_readme:
            # Insert our custom content after the YAML frontmatter and widget section
            combined_content = insert_custom_content_after_yaml(existing_readme, model_card_content)
        else:
            combined_content = model_card_content

        readme_path.write_text(combined_content)
        logger.info("Created combined model card (custom + SentenceTransformer content)")

        # Copy notebook if provided
        if notebook_path:
            notebook_path = Path(notebook_path)
            if notebook_path.exists():
                dest_notebook = temp_path / notebook_path.name

                # Read notebook and replace model_id placeholder
                notebook_content = notebook_path.read_text()
                # Replace the placeholder model ID with actual repo_id
                notebook_content = notebook_content.replace('"jo-mengr/your-model-name"', f'"{repo_id}"')
                # Also replace the comment
                notebook_content = notebook_content.replace("# Update this!", "# Automatically set to this repository")
                dest_notebook.write_text(notebook_content)
                logger.info(f"Added tutorial notebook: {notebook_path.name}")
            else:
                logger.warning(f"Notebook not found: {notebook_path}")

        # Upload to Hub
        api = HfApi()

        # Create repository
        logger.info(f"Creating repository: {repo_id}")
        try:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            logger.info(f"Repository {repo_id} ready")
        except Exception as e:
            logger.warning(f"Repository creation: {e}")

        # Upload all files
        commit_msg = commit_message or f"Upload {model_name} with MMContext architecture"
        logger.info("Uploading model and files...")
        api.upload_folder(folder_path=temp_path, repo_id=repo_id, commit_message=commit_msg)

        logger.info(f"✅ Successfully uploaded model to https://huggingface.co/{repo_id}")
        return f"https://huggingface.co/{repo_id}"


def insert_custom_content_after_yaml(existing_readme: str, custom_content: str) -> str:
    """
    Insert custom content after the YAML frontmatter and widget section in SentenceTransformer README.

    Parameters
    ----------
    existing_readme : str
        Original README content from SentenceTransformer
    custom_content : str
        Custom content to insert

    Returns
    -------
    str
        Combined README with custom content inserted after YAML/widget section
    """
    import re

    # Find the end of the YAML frontmatter and widget section
    # Pattern: --- (YAML start) ... --- (YAML end) ... widget section ... first markdown heading or substantial content

    # First, find where the YAML frontmatter ends
    lines = existing_readme.split("\n")

    yaml_started = False
    yaml_ended = False
    widget_ended = False
    insert_position = 0

    for i, line in enumerate(lines):
        # Track YAML frontmatter
        if not yaml_started and line.strip() == "---":
            yaml_started = True
            continue
        elif yaml_started and not yaml_ended and line.strip() == "---":
            yaml_ended = True
            continue

        # After YAML ends, look for end of widget section
        if yaml_ended and not widget_ended:
            # Widget section ends when we hit a line that doesn't start with spaces/dashes
            # and isn't empty, and isn't part of the widget structure
            if (
                line.strip()
                and not line.startswith((" ", "-", "widget:"))
                and not line.strip().startswith("source_sentence:")
                and not line.strip().startswith("sentences:")
            ):
                # This is likely the start of the main content
                insert_position = i
                break

    # If we didn't find a good insertion point, insert after a reasonable number of lines
    if insert_position == 0:
        # Fallback: insert after first 50 lines or when we see a markdown header
        for i, line in enumerate(lines[:100]):
            if line.startswith("#") and i > 10:  # Found a markdown header after some content
                insert_position = i
                break
        if insert_position == 0:
            insert_position = min(50, len(lines))

    # Insert our custom content
    before_lines = lines[:insert_position]
    after_lines = lines[insert_position:]

    # Add our custom content with proper spacing
    custom_lines = [
        "",  # Empty line before our content
        "# MMContext Model Information",
        "",
        custom_content,
        "",  # Empty line after our content
        "---",
        "",
    ]

    # Combine all parts
    combined_lines = before_lines + custom_lines + after_lines
    return "\n".join(combined_lines)


def get_model_info_from_config(config_path: str | Path) -> dict[str, str]:
    """
    Extract model information from training configuration.

    Parameters
    ----------
    config_path : str or Path
        Path to training configuration file

    Returns
    -------
    dict[str, str]
        dictionary with model information
    """
    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Extract relevant information
        info = {}

        # Text encoder info
        if "text_encoder" in config:
            info["text_encoder"] = config["text_encoder"].get("name", "Unknown")

        # Embedding method
        if "data" in config:
            info["embedding_method"] = config["data"].get("embedding_method", "unknown")

        # Model architecture details
        if "adapter" in config:
            adapter_config = config["adapter"]
            info["adapter_hidden_dim"] = adapter_config.get("hidden_dim", "None")
            info["adapter_output_dim"] = adapter_config.get("output_dim", "None")

        return info

    except Exception as e:
        logger.warning(f"Could not extract config info: {e}")
        return {}
