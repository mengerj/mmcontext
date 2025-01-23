import anndata
import pandas as pd


class SimpleCaptionConstructor:
    """Construct captions for each sample by concatenating values from specified obs keys"""

    def __init__(self, obs_keys: list[str], separator: str = " "):
        """
        Initialize the SimpleCaptionConstructor.

        Args:
            obs_keys: List of keys from adata.obs to include in the caption
            separator: String to use between concatenated values (default: space)
        """
        self.obs_keys = obs_keys
        self.separator = separator

    def construct_captions(self, adata: anndata.AnnData) -> None:
        """Include captions for each sample

        Construct captions by concatenating values from specified obs keys.
        Adds a 'caption' column to adata.obs.

        Args:
            adata: AnnData object to process

        Raises
        ------
            KeyError: If any of the specified obs_keys is not found in adata.obs
        """
        # Verify all keys exist
        missing_keys = [key for key in self.obs_keys if key not in adata.obs.columns]
        if missing_keys:
            raise KeyError(f"The following keys were not found in adata.obs: {missing_keys}")

        # Convert all values to strings and replace NaN with empty string
        str_values = [adata.obs[key].astype(str).replace("nan", "") for key in self.obs_keys]

        # Concatenate the values
        adata.obs["caption"] = pd.DataFrame(str_values).T.agg(self.separator.join, axis=1)
