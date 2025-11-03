import os

import dotenv
from huggingface_hub import HfApi

dotenv.load_dotenv()

# ==== CONFIGURATION ====
# List of model repo names you want to KEEP
KEEP_MODELS = [
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_100k_unfreeze_full",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_100k_2layers_unfreeze_full",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_geo_unfreeze_full",
    "jo-mengr/mmcontext-pubmedbert-gs-cxg_geo_unfreeze_full",
    "jo-mengr/mmcontext-pubmedbert-gs-100k_ncbi",
    "jo-mengr/mmcontext-pubmedbert-gs-100k_mixed_no_bio",
    "jo-mengr/mmcontext-pubmedbert-semantic_100k",
    "jo-mengr/mmcontext-pubmedbert-gs-100k_ct",
    "jo-mengr/mmcontext-pubmedbert-gs-cxg_100k_ct_adapter_unfreeze",
    "jo-mengr/mmcontext-pubmedbert-gs-cxg_100k_ct_adapter",
    "jo-mengr/mmcontext-pubmedbert-gs-mixed_from_adapter",
    "jo-mengr/mmcontext-pubmedbert-gs-mixed-all-direct",
    "jo-mengr/mmcontext-pubmedbert-no_bio",
    "jo-mengr/mmcontext-pubmedbert-gs-adapter",
    "jo-mengr/mmcontext-pubmedbert-100k-v6",
    "jo-mengr/mmcontext-pubmedbert-gs-100k_adapter",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_100k_2layers_unfreeze_full",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_100k_unfreeze_full",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_geo_unfreeze_full",
    "jo-mengr/mmcontext-pubmedbert-gs-cxg_geo_unfreeze_full",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_geo_unfreeze_full-v2",
    "jo-mengr/mmcontext-pubmedbert-gs-cxg_geo_unfreeze_full-v2",
    "jo-mengr/mmcontext-pubmedbert-pca-cxg_geo_unfreeze_full",
    "jo-mengr/mmcontext-pubmedbert-scvi_fm-cxg_geo_unfreeze_full",
    "jo-mengr/mmcontext-pubmedbert-gs-cxg_geo_unfreeze_full_merged_ds",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_geo_unfreeze_full_merged_ds",
    "jo-mengr/mmcontext-pubmedbert-pca-cxg_geo_unfreeze_full_merged_ds",
    "jo-mengr/mmcontext-pubmedbert-scvi_fm-cxg_geo_unfreeze_full_merged_ds",
    "jo-mengr/mmcontext-pubmedbert-gs10k-cxg_100k_2layers_unfreeze_full",
    "jo-mengr/mmcontext-all-MiniLM-L6-v2-gs-cxg_100k_unfreeze_full",
    "jo-mengr/mmcontext-qwen-gs-cxg_100k_unfreeze_last",
    "jo-mengr/mmcontext-pubmedbert-gs-cxg_full_unfreeze_full",
]
# If True: only print which models would be deleted (safe mode).
# If False: actually delete the models.
DRY_RUN = False
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
