import os

import dotenv
from huggingface_hub import HfApi

dotenv.load_dotenv()

# ==== CONFIGURATION ====
# List of model repo names you want to EXCLUDE from privatization (keep public)
KEEP_PUBLIC_MODELS = [
    # Add any models you want to keep public here
    # Example:
    # "jo-mengr/mmcontext-qwen-scvi_fm",
    # "jo-mengr/mmcontext-biomodern-scvi_fm-v3",
]

# List of specific models to privatize (if empty, will privatize all models except those in KEEP_PUBLIC_MODELS)
SPECIFIC_MODELS_TO_PRIVATIZE = [
    # Add specific model repo names here if you only want to privatize certain models
    # If this list is empty, all models (except KEEP_PUBLIC_MODELS) will be privatized
    # Example:
    # "jo-mengr/mmcontext-pubmedbert-gs-100k_adapter",
    # "jo-mengr/mmcontext-pubmedbert-geneformer-100k_adapter",
]

# If True: only print which models would be privatized (safe mode).
# If False: actually privatize the models.
DRY_RUN = False

# Filter settings
ONLY_MMCONTEXT_MODELS = True  # If True, only process models with 'mmcontext' in the name
USERNAME_FILTER = "jo-mengr"  # Only process models from this username (set to None to process all your models)
# =======================


def main():
    """
    Privatize models from Hugging Face account.

    Notes
    -----
    - Requires `huggingface_hub` package.
    - Authentication uses HF_TOKEN environment variable (recommended).
    - DRY_RUN = True will only simulate privatization without executing them.
    - Models in KEEP_PUBLIC_MODELS will remain public.
    - If SPECIFIC_MODELS_TO_PRIVATIZE is not empty, only those models will be privatized.
    """
    api = HfApi()
    token = os.getenv("HF_TOKEN")  # Must be set in your environment

    if not token:
        raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

    user = api.whoami(token=token)["name"]
    print(f"Authenticated as: {user}")

    # Get all models for the user
    all_models = api.list_models(author=user, token=token)

    models_to_process = []

    for model in all_models:
        repo_id = model.modelId

        # Apply username filter
        if USERNAME_FILTER and not repo_id.startswith(f"{USERNAME_FILTER}/"):
            continue

        # Apply mmcontext filter
        if ONLY_MMCONTEXT_MODELS and "mmcontext" not in repo_id.lower():
            continue

        # Skip models that should remain public
        if repo_id in KEEP_PUBLIC_MODELS:
            print(f"Skipping {repo_id} (marked to keep public)")
            continue

        # If specific models are listed, only process those
        if SPECIFIC_MODELS_TO_PRIVATIZE:
            if repo_id not in SPECIFIC_MODELS_TO_PRIVATIZE:
                continue

        models_to_process.append(repo_id)

    print(f"\nFound {len(models_to_process)} models to privatize:")
    for repo_id in models_to_process:
        print(f"  - {repo_id}")

    if not models_to_process:
        print("No models found to privatize.")
        return

    print(f"\n{'[DRY RUN] ' if DRY_RUN else ''}Starting privatization process...")

    success_count = 0
    error_count = 0

    for repo_id in models_to_process:
        try:
            if DRY_RUN:
                print(f"[DRY RUN] Would privatize {repo_id}")
                success_count += 1
            else:
                print(f"Privatizing {repo_id}...")
                api.update_repo_visibility(repo_id=repo_id, private=True, token=token, repo_type="model")
                print(f"✓ Successfully privatized {repo_id}")
                success_count += 1
        except Exception as e:
            print(f"✗ Error privatizing {repo_id}: {str(e)}")
            error_count += 1

    print(f"\n{'Dry run complete!' if DRY_RUN else 'Privatization complete!'}")
    print(f"Successfully processed: {success_count}")
    if error_count > 0:
        print(f"Errors encountered: {error_count}")


if __name__ == "__main__":
    main()
