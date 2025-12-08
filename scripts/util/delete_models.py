import os

import dotenv
from huggingface_hub import HfApi

dotenv.load_dotenv()

# ==== CONFIGURATION ====
# List of model repo names you want to KEEP
KEEP_MODELS = [
    "jo-mengr/mmcontext-pubmedbert-geneformer-v1-cxg_dim2048-v2",
    "jo-mengr/mmcontext-pubmedbert-geneformer-cxg_dim2048",
    "jo-mengr/mmcontext-pubmedbert-scvi_fm-cxg_dim2048",
    "jo-mengr/mmcontext-pubmedbert-pca-cxg_dim2048",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_dim2048",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_bio_dim2048-v3",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_geo_bio_dim2048",
    "jo-mengr/mmcontext-pubmedbert-cxg",
    "jo-mengr/mmcontext-qwen-cxg_100k",
    "jo-mengr/mmcontext-qwen-cxg_100k-v2",
    "jo-mengr/mmcontext-pubmedbert-cxg_100k-v5",
    "jo-mengr/mmcontext-pubmedbert-cxg_100k-v6",
    "jo-mengr/mmcontext-all-MiniLM-L6-v2-cxg_100k-v3",
    "jo-mengr/mmcontext-all-MiniLM-L6-v2-cxg_100k-v4",
    "jo-mengr/mmcontext-pubmedbert-geneformer-cxg",
    "jo-mengr/mmcontext-pubmedbert-geneformer-cxg_normlog-v2",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_normlogjo-mengr/mmcontext-biobert-geneformer-cxg",
    "jo-mengr/mmcontext-biobert-gs10k-cxg_dim2048",
    "jo-mengr/mmcontext-pubmedbert-semantic_100k",
    "jo-mengr/mmcontext-biobert-geneformer-v1-cxg_dim2048",
    "jo-mengr/mmcontext-pubmedbert-geneformer-cxg_normlog",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_normlog-v2",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_normlog",
]
# If True: only print which models would be deleted (safe mode).
# If False: actually delete the models.
DRY_RUN = True
# =======================


def main():
    """
    Delete all models from Hugging Face account except those in KEEP_MODELS.

    Notes
    -----
    - Requires `huggingface_hub` package.
    - Authentication uses HF_TOKEN environment variable (recommended).
    - DRY_RUN = True will only simulate deletions without executing them.
    """
    api = HfApi()
    token = os.getenv("HF_TOKEN")  # Must be set in your environment

    if not token:
        raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

    user = api.whoami(token=token)["name"]
    all_models = api.list_models(author=user, token=token)

    for model in all_models:
        repo_id = model.modelId
        if repo_id not in KEEP_MODELS:
            if DRY_RUN:
                print(f"[DRY RUN] Would delete {repo_id}")
            else:
                print(f"Deleting {repo_id}...")
                api.delete_repo(repo_id=repo_id, token=token, repo_type="model")

    print("Done. (Dry run mode)" if DRY_RUN else "Cleanup complete!")


if __name__ == "__main__":
    main()
