#!/usr/bin/env python3
"""Script to load a model from a checkpoint and upload it to Hugging Face Hub.

Usage:
    python scripts/upload_checkpoint.py --output-date 2025-11-24 --job-id 10732 --model-dir checkpoint-10000
    python scripts/upload_checkpoint.py --load-dir outputs/2025-11-24/training/10732/checkpoint-10000 --repo-id jo-mengr/my-model-name
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from mmcontext.hub_utils import prepare_model_for_hub, upload_model_to_hub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


def main():
    """Helper function to upload a model from a checkpoint to Hugging Face Hub."""
    parser = argparse.ArgumentParser(description="Load a model from a checkpoint and upload it to Hugging Face Hub")
    parser.add_argument(
        "--load-dir",
        type=str,
        help="Full path to the checkpoint directory (e.g., outputs/2025-11-24/training/10732/checkpoint-10000)",
    )
    parser.add_argument(
        "--output-date",
        type=str,
        help="Output date (e.g., 2025-11-24). Used with --job-id and --model-dir to construct load-dir",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        help="Job ID (e.g., 10732). Used with --output-date and --model-dir to construct load-dir",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Model directory name (e.g., checkpoint-10000). Used with --output-date and --job-id to construct load-dir",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., jo-mengr/my-model-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make the repository private (default: True)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repository public (overrides --private)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        help="Custom commit message for the upload",
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        help="Name of text encoder (auto-detected if not provided)",
    )
    parser.add_argument(
        "--embedding-method",
        type=str,
        help="Omics embedding method (auto-detected if not provided)",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        help="Output dimension (auto-detected if not provided)",
    )
    parser.add_argument(
        "--training-details",
        type=str,
        help="Additional training details to include in model card",
    )

    args = parser.parse_args()

    # Determine load directory
    if args.load_dir:
        load_dir = Path(args.load_dir)
    elif args.output_date and args.job_id and args.model_dir:
        load_dir = Path(f"outputs/{args.output_date}/training/{args.job_id}/{args.model_dir}")
    else:
        logger.error(
            "Either --load-dir must be provided, or all of --output-date, --job-id, and --model-dir must be provided"
        )
        sys.exit(1)

    # Resolve to absolute path
    load_dir = load_dir.resolve()

    if not load_dir.exists():
        logger.error(f"Checkpoint directory does not exist: {load_dir}")
        sys.exit(1)

    logger.info(f"Loading model from: {load_dir}")

    # Load the model
    try:
        model = SentenceTransformer(str(load_dir))
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Determine privacy setting
    private = not args.public if args.public else args.private

    # Generate commit message if not provided
    commit_message = args.commit_message or f"Upload model from {load_dir.name}"

    # Prepare model for Hub upload
    logger.info("Preparing model for Hub upload (creating model card, metadata, etc.)...")
    prepared_model_dir = prepare_model_for_hub(
        model=model,
        output_dir=load_dir / "prepared_for_hub",
        repo_id=args.repo_id,
        model_name=None,  # Auto-detect from repo_id
        text_encoder=args.text_encoder,
        embedding_method=args.embedding_method,
        output_dim=args.output_dim,
        training_details=args.training_details,
        tutorial_notebook=None,
        notebook_path=None,
    )
    logger.info(f"Model prepared for Hub at: {prepared_model_dir}")

    # Upload to Hub
    try:
        repo_url = upload_model_to_hub(
            model=model,
            repo_id=args.repo_id,
            model_name=None,  # Auto-detect from repo_id
            text_encoder=args.text_encoder,
            embedding_method=args.embedding_method,
            output_dim=args.output_dim,
            training_details=args.training_details,
            tutorial_notebook=None,
            notebook_path=None,
            private=private,
            commit_message=commit_message,
            prepared_model_dir=prepared_model_dir,
        )
        logger.info(f"✅ Model successfully uploaded to Hub: {repo_url}")
    except Exception as e:
        logger.error(f"Failed to upload model to Hub: {e}")
        logger.error(f"Model has been prepared locally at: {prepared_model_dir}")
        logger.error("You can push it manually using:")
        logger.error("  from huggingface_hub import HfApi")
        logger.error("  api = HfApi()")
        logger.error(f"  api.create_repo(repo_id='{args.repo_id}', private={private}, exist_ok=True)")
        logger.error(f"  api.upload_folder(folder_path='{prepared_model_dir}', repo_id='{args.repo_id}')")
        sys.exit(1)


if __name__ == "__main__":
    main()
