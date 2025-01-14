import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class DataProperties:
    """
    Class to compute data properties for given data matrices and compare them.

    Parameters
    ----------
    predefined_subset : str, optional
        Choose one of the available predefined subsets. Currently:
        "proteomics", "microbiome", or "all". Default is 'proteomics'.
    custom_subset : list of str, optional
        Choose a set of properties by providing their names in a list.
    cor_method : str, optional
        Method used for correlations. Pairwise column and row correlations
        are calculated with the specified method. Default is 'spearman'.
    logger : logging.Logger, optional
        Logger to use. If None, a new logger is created.
    """

    def __init__(self, predefined_subset="proteomics", custom_subset=None, cor_method="spearman", logger=None):
        self.predefined_subset = predefined_subset
        self.custom_subset = custom_subset
        self.cor_method = cor_method
        self.logger = logger or logging.getLogger(__name__)
        self.properties = []  # Stores properties of both original and reconstructed data
        self.original_counter = 0  # Counter for original dataset IDs
        self.id = None

    def add_original_data(self, data, id=None):
        """
        Compute properties of the original data and store them.

        Parameters
        ----------
        data : numpy.ndarray
            Original data matrix (rows are features, columns are samples).
        id : str, optional
            Identifier for the original data. Default is '0'.
        """
        if id is None:
            id = str(self.original_counter)
            self.original_counter += 1
        self.id = id
        # always reset reconstruction counter when adding new original data
        self.reconstructed_counter = 0
        properties = self.compute_properties(data)
        self.properties.append({"id": id, "recon_id": "orginal_data", "type": "original", "properties": properties})

    def add_reconstructed_data(self, data, id=None):
        """
        Compute properties of the reconstructed data and store them.

        Parameters
        ----------
        data : numpy.ndarray
            Reconstructed data matrix (rows are features, columns are samples).
        id : str, optional
            Identifier for the reconstructed data. If not provided, an incrementing number as a string is used.
        """
        if self.id is None:
            self.logger.error("Please add an original dataset before adding reconstructed datasets. ")
            raise ValueError("Please add an original dataset before adding reconstructed datasets.")
        if id is None:
            id = str(self.reconstructed_counter)
            self.reconstructed_counter += 1
        properties = self.compute_properties(data)
        self.properties.append({"id": self.id, "recon_id": id, "type": "reconstructed", "properties": properties})

    def compare_data_properties(self):
        """
        Compare data properties between original and reconstructed datasets.

        Compute log2 fold changes between the properties of the original data
        and each reconstructed dataset.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing log2 fold changes, with one row per reconstructed dataset.
        """
        if not "original" and "reconstructed" in [prop["type"] for prop in self.properties]:
            self.logger.error("Please add at least one original and one reconstructed dataset before comparing.")
            return None
        # List to store log2FC data for each reconstructed dataset
        log2fc_list = []
        for original in self.properties:
            if original["type"] == "reconstructed":
                continue
            id = original["id"]
            original_props = original["properties"]
            prop_names = set(original_props.keys())
            for recon in self.properties:
                # skip if this is original data or it is reconstructed data belonging to a different og dataset
                if recon["type"] == "original" or recon["id"] != id:
                    continue
                recon_id = recon["recon_id"]
                recon_props = recon["properties"]
                # Get intersection of property names
                properties = prop_names & set(recon_props.keys())
                log2fc_dict = {"id": id, "recon_id": recon_id}

                for prop in properties:
                    try:
                        prop_value1 = original_props[prop]
                        prop_value2 = recon_props[prop]

                        # Handle p-values specially
                        if "pval" in prop:
                            # Clip p-values between 0.05 and 1
                            prop_value1 = max(prop_value1, 0.05)
                            prop_value2 = max(prop_value2, 0.05)

                        # Compute log2 fold change
                        if prop_value1 == 0 or prop_value2 == 0:
                            # Avoid returning NaN for log2(0/0)
                            if prop_value1 == 0 and prop_value2 == 0:
                                current_log2fc = np.float32(0.0)
                            else:
                                # Add small value to avoid division by zero
                                current_log2fc = np.sign(prop_value2 - prop_value1) * np.log2(
                                    np.divide(np.abs(prop_value2 + 1e-7), np.abs(prop_value1 + 1e-7))
                                )
                        else:
                            # Ensure direction is handled even if values are negative
                            current_log2fc = np.sign(prop_value2 - prop_value1) * np.log2(
                                np.divide(np.abs(prop_value2), np.abs(prop_value1))
                            )

                        log2fc_dict[prop] = current_log2fc

                    except Exception as e:
                        self.logger.error(f"Error computing log2FC for property '{prop}' in dataset '{recon_id}': {e}")
                        log2fc_dict[prop] = np.nan

                log2fc_list.append(log2fc_dict)

        # Convert the list of dictionaries to a DataFrame
        log2fc_df = pd.DataFrame(log2fc_list)
        # Compute meanLog2FC for each reconstructed dataset
        log2fc_df["meanLog2FC"] = log2fc_df.drop(columns=["id", "recon_id"]).abs().mean(axis=1, skipna=True)
        # Get the mean and sd of mean Log2FC for each unique original dataset
        mean_log2fc = log2fc_df.groupby("id")["meanLog2FC"].mean()
        sd_log2fc = log2fc_df.groupby("id")["meanLog2FC"].std(ddof=1)
        self.mean_std_df = pd.DataFrame({"meanLog2FC": mean_log2fc, "sdLog2FC": sd_log2fc})
        # Store the DataFrame for plotting
        self.log2fc_df = log2fc_df

        return log2fc_df

    def compute_properties(self, data):
        """
        Compute the selected data properties.

        Parameters
        ----------
        data : numpy.ndarray
            Data matrix to compute properties on.

        Returns
        -------
        dict
            A dictionary with data properties.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        self.data = data
        properties = self.select_properties()
        data_properties = {}
        for prop in properties:
            method_name = f"get_{prop}"
            method = getattr(self, method_name, None)
            if method:
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        result = method()
                        if w:
                            for warning in w:
                                self.logger.warning(f"Warning computing {prop}: {warning.message}")
                        if isinstance(result, dict) and "res" in result:
                            result = result.get("res")
                        if isinstance(result, np.ndarray):
                            if result.size > 1:
                                raise Exception(f"Property {prop} has more than one value. Skipping.")
                            else:
                                result = result.item()
                        # Ensure result is a numpy float
                        data_properties[prop] = np.float64(result)
                except Exception as e:
                    self.logger.error(f"Error computing {prop}: {e}")
                    data_properties[prop] = np.nan
                    continue
            else:
                self.logger.warning(f"Method {method_name} not found.")
        return data_properties

    def select_properties(self):
        """
        Select the properties to compute based on predefined or custom subset.

        Returns
        -------
        list of str
            List of property names to compute.
        """
        possible_properties = []
        for method in dir(self):
            if not method.startswith("__") and method.startswith("get_"):
                method_name = method[4:]
                possible_properties.append(method_name)

        if self.custom_subset:
            subset = self.custom_subset
        else:
            if self.predefined_subset == "all":
                subset = possible_properties
                self.logger.info("Using all available data properties.")
            elif self.predefined_subset == "proteomics":
                subset = [
                    "nFeatures",
                    "nSamples",
                    "corrpNAColAndColMeans",
                    "corrpNARowAndRowMeans",
                    "pNA",
                    "prctPC1",
                    "prctPC2",
                    "meanRowMeans",
                    "meanColMeans",
                    "meanRowSd",
                    "meanColSd",
                    "skewness",
                    "bimodalityColCorr",
                    "mean",
                    "variance",
                    "coefHclustCols",
                    "coefHclustRows",
                    "FCmax10pQuantileSdQuotient",
                ]
                self.logger.info("Using predefined proteomics subset.")
            elif self.predefined_subset == "microbiome":
                subset = [
                    "nFeatures",
                    "nSamples",
                    "p0",
                    "meanp0Row",
                    "sdp0Row",
                    "meanp0Col",
                    "sdp0Col",
                    "median",
                    "q95",
                    "q99",
                    "medianColSums",
                    "meanColSums",
                    "sdColSums",
                    "maxColSums",
                    "minColSums",
                    "corrColSumsP0Sample",
                    "bimodalityRowCorr",
                    "bimodalityColCorr",
                    "meanRowMeansLog2cpm",
                    "medianRowMediansLog2cpm",
                    "meanRowVarsLog2cpm",
                    "meanColCorr",
                    "meanRowCorr",
                    "sdRowMeansLog2cpm",
                    "sdRowMediansLog2cpm",
                    "sdRowVarsLog2cpm",
                    "sdColCorr",
                    "sdRowCorr",
                    "LinCoefPoly2",
                    "QuadCoefPoly2",
                    "poly1Xp0YRowMeans",
                    "coefHclustRows",
                    "coefHclustCols",
                ]
                self.logger.info("Using predefined microbiome subset.")
            else:
                self.logger.error("Unknown predefined subset.")
                subset = []
            self.subset = subset
        return subset

    def plot_metrics(self, title="Metrics Boxplot", save_dir=None):
        """
        Create boxplots of the metrics over the reconstructed datasets.

        Parameters
        ----------
        title : str, optional
            Title for the boxplot. Default is 'Metrics Boxplot'.
        save_dir:
            Directory to save the plot. If none, the plot will be displayed.
        """
        if not hasattr(self, "log2fc_df"):
            self.logger.error("Please run compare_data_properties() before plotting metrics.")
            return
        ids = self.log2fc_df["id"].unique()
        for id in ids:
            df_subset = self.log2fc_df[self.log2fc_df["id"] == id]
            df_subset = df_subset.drop(columns=["id"])
            # Melt the DataFrame for plotting
            df_melted = df_subset.melt(id_vars=["recon_id"], var_name="Metric", value_name="Log2FC")

            # Create the boxplot with metric names on the y-axis
            plt.figure(figsize=(8, max(6, len(df_melted["Metric"].unique()) * 0.4)))
            sns.boxplot(y="Metric", x="Log2FC", data=df_melted, orient="h")
            plt.title(f"{title}_orginal_dataID:{id}")
            plt.tight_layout()
        if save_dir:
            save_dir = Path(save_dir)
            plt.savefig(save_dir / f"{title}_{id}.png")
        else:
            plt.show()

    def plot_pca(self, save_path=None):
        """
        Perform PCA on the computed properties and plot the first two principal components.

        Datasets are colored by 'id', and different markers are used for original and reconstructed data.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot. If None, the plot will be displayed.
        """
        if not hasattr(self, "properties") or not self.properties:
            self.logger.error("No properties available. Please compute properties first.")
            return

        # Collect all datasets into a DataFrame
        data_list = []
        for item in self.properties:
            data_entry = {
                "id": item.get("id"),
                "recon_id": item.get("recon_id"),
                "type": item.get("type"),
            }
            # Flatten the properties dictionary into the data entry
            for prop_name, prop_value in item["properties"].items():
                data_entry[prop_name] = prop_value
            data_list.append(data_entry)

        # Create a DataFrame
        df = pd.DataFrame(data_list)
        # Drop columns with NaN values and log them
        df = df.dropna(axis=1, how="any")
        self.logger.warning(
            f"Property PCA Plot: Dropped columns with NaN values: {set(data_list[0].keys()) - set(df.columns)}"
        )
        # Extract the features (properties) for PCA
        feature_cols = [col for col in df.columns if col not in ["id", "recon_id", "type"]]
        X = df[feature_cols]

        # Handle missing values (fill with mean of each column)
        # X_filled = X.fillna(X.mean())

        # Perform PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)

        # Add principal components to the DataFrame
        df["PC1"] = principal_components[:, 0]
        df["PC2"] = principal_components[:, 1]

        # Plotting
        plt.figure(figsize=(10, 7))
        unique_ids = df["id"].unique()
        num_colors = len(unique_ids)
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(num_colors)]  # Cycle through colors if more than 10 ids

        # Create mappings for colors and markers
        id_to_color = {id: colors[i] for i, id in enumerate(unique_ids)}
        type_to_marker = {"original": "X", "reconstructed": "s"}  # 'X' for original, square for reconstructed

        # Plot data points
        for _idx, row in df.iterrows():
            plt.scatter(
                row["PC1"],
                row["PC2"],
                color=id_to_color[row["id"]],
                marker=type_to_marker[row["type"]],
                edgecolors="black",  # Black edge for visibility
                s=100,
            )  # Adjust size as needed

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA of Data Properties")

        # Create custom legend for colors (id)
        from matplotlib.patches import Patch

        color_patches = [Patch(facecolor=id_to_color[id], edgecolor="black", label=id) for id in unique_ids]

        # Create custom legend for markers (type)
        from matplotlib.lines import Line2D

        marker_lines = [
            Line2D([0], [0], marker=marker, color="black", linestyle="None", markersize=10, label=dataset_type)
            for dataset_type, marker in type_to_marker.items()
        ]

        # Combine legends
        first_legend = plt.legend(handles=color_patches, title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.gca().add_artist(first_legend)
        plt.legend(handles=marker_lines, title="Dataset Type", bbox_to_anchor=(1.05, 0.6), loc="upper left")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_p0(self):
        """
        Calculate the percent of zeros in the data matrix.

        Returns
        -------
        float
            Percent of zeros in the data matrix.
        """
        mtx = self.data
        return np.sum(mtx == 0) / mtx.size

    def _get_p0Row(self):
        """
        Calculate the percent of zeros per feature (row). Only used internally.

        Returns
        -------
        numpy.ndarray
            Percent of zeros per row.
        """
        mtx = self.data
        return np.sum(mtx == 0, axis=1) / mtx.shape[1]

    def _get_p0Col(self):
        """
        Calculate the percent of zeros per sample (column). Only used internally.

        Returns
        -------
        numpy.ndarray
            Percent of zeros per column.
        """
        mtx = self.data
        return np.sum(mtx == 0, axis=0) / mtx.shape[0]

    def get_meanp0Row(self):
        """
        Calculate the mean of percent of zeros per feature.

        Returns
        -------
        float
            Mean of percent of zeros per feature.
        """
        p0_row = self._get_p0Row()
        return np.nanmean(p0_row)

    def get_sdp0Row(self):
        """
        Calculate the standard deviation of percent of zeros per feature.

        Returns
        -------
        float
            Standard deviation of percent of zeros per feature.
        """
        p0_row = self._get_p0Row()
        return np.nanstd(p0_row, ddof=1)

    def get_meanp0Col(self):
        """
        Calculate the mean of percent of zeros per sample.

        Returns
        -------
        float
            Mean of percent of zeros per sample.
        """
        p0_col = self._get_p0Col()
        return np.nanmean(p0_col)

    def get_sdp0Col(self):
        """
        Calculate the standard deviation of percent of zeros per sample.

        Returns
        -------
        float
            Standard deviation of percent of zeros per sample.
        """
        p0_col = self._get_p0Col()
        return np.nanstd(p0_col, ddof=1)

    def get_median(self):
        """
        Calculate the median of the entire data matrix.

        Returns
        -------
        float
            Median of the data matrix.
        """
        mtx = self.data
        return np.nanmedian(mtx)

    def get_mean(self):
        """
        Calculate the mean of the entire data matrix.

        Returns
        -------
        float
            Mean of the data matrix.
        """
        mtx = self.data
        return np.nanmean(mtx)

    def get_q95(self):
        """
        Calculate the 95th percentile of the data matrix.

        Returns
        -------
        float
            95th percentile value.
        """
        mtx = self.data
        return np.nanpercentile(mtx, 95)

    def get_q99(self):
        """
        Calculate the 99th percentile of the data matrix.

        Returns
        -------
        float
            99th percentile value.
        """
        mtx = self.data
        return np.nanpercentile(mtx, 99)

    def _get_colSums(self):
        """
        Calculate the sum of each column. Only used internally.

        Returns
        -------
        numpy.ndarray
            Sum of each column.
        """
        mtx = self.data
        return np.nansum(mtx, axis=0)

    def get_medianColSums(self):
        """
        Calculate the median of the column sums.

        Returns
        -------
        float
            Median of the column sums.
        """
        col_sums = self._get_colSums()
        return np.nanmedian(col_sums)

    def get_sdColSums(self):
        """
        Calculate the standard deviation of the column sums.

        Returns
        -------
        float
            Standard deviation of the column sums.
        """
        col_sums = self._get_colSums()
        return np.nanstd(col_sums, ddof=1)

    def _get_colMeans(self):
        """
        Calculate the mean of each column. Only used internally.

        Returns
        -------
        numpy.ndarray
            Mean of each column.
        """
        mtx = self.data
        return np.nanmean(mtx, axis=0)

    # ... Implement other get_* methods similarly ...

    # Example of _get_log2cpm
    def _get_log2cpm(self):
        """
        Calculate log2 counts per million.

        Returns
        -------
        numpy.ndarray
            Log2 counts per million of the data matrix.
        """
        mtx = self.data.copy()
        if np.any(np.isnan(mtx)):
            mtx = np.nan_to_num(mtx)
            self.logger.info("NAs have been converted to 0 for the calculation of counts per million.")

        lib_sizes = np.sum(mtx, axis=0)
        # Ensure no division by zero by masking or replacing zero library sizes
        safe_lib_sizes = np.where(lib_sizes == 0, 1, lib_sizes)

        # Compute CPM
        cpm = (mtx / safe_lib_sizes) * 1e6

        # Set CPM to 0 where library sizes were 0
        cpm[:, lib_sizes == 0] = 0
        prior_count = 0.1
        cpm = cpm + prior_count
        log2cpm = np.log2(cpm)
        return log2cpm

    def get_maxColSums(self):
        """
        Calculate the maximum of the column sums.

        Returns
        -------
        float
            Maximum of the column sums.
        """
        col_sums = self._get_colSums()
        return np.nanmax(col_sums)

    def get_minColSums(self):
        """
        Calculate the minimum of the column sums.

        Returns
        -------
        float
            Minimum of the column sums.
        """
        col_sums = self._get_colSums()
        return np.nanmin(col_sums)

    def get_nFeatures(self):
        """
        Get the number of features (rows) in the data matrix.

        Returns
        -------
        int
            Number of features.
        """
        return self.data.shape[0]

    def get_nSamples(self):
        """
        Get the number of samples (columns) in the data matrix.

        Returns
        -------
        int
            Number of samples.
        """
        return self.data.shape[1]

    def _get_cpm(self):
        """
        Calculate counts per million (CPM).

        Returns
        -------
        numpy.ndarray
            Counts per million of the data matrix.
        """
        mtx = self.data.copy()
        if np.isnan(mtx).any():
            mtx = np.nan_to_num(mtx)
            self.logger.info("NAs have been converted to 0 for the calculation of counts per million.")
        lib_sizes = np.sum(mtx, axis=0)
        cpm = (mtx / lib_sizes) * 1e6
        return cpm

    def _get_rowMeansLog2cpm(self):
        """
        Calculate the mean of log2 CPM for each row. Only used internally.

        Returns
        -------
        numpy.ndarray
            Mean log2 CPM per row.
        """
        log2cpm = self._get_log2cpm()
        return np.nanmean(log2cpm, axis=1)

    def _get_rowMediansLog2cpm(self):
        """
        Calculate the median of log2 CPM for each row.

        Returns
        -------
        numpy.ndarray
            Median log2 CPM per row.
        """
        log2cpm = self._get_log2cpm()
        return np.nanmedian(log2cpm, axis=1)

    def _get_rowVarsLog2cpm(self):
        """
        Calculate the variance of log2 CPM for each row. Only used internally.

        Returns
        -------
        numpy.ndarray
            Variance of log2 CPM per row.
        """
        log2cpm = self._get_log2cpm()
        return np.nanvar(log2cpm, axis=1, ddof=1)

    def _get_rowMeans(self):
        """
        Calculate the mean of each row.

        Returns
        -------
        numpy.ndarray
            Mean of each row.
        """
        mtx = self.data
        return np.nanmean(mtx, axis=1)

    def _get_rowSd(self):
        """
        Calculate the standard deviation of each row.

        Returns
        -------
        numpy.ndarray
            Standard deviation of each row.
        """
        mtx = self.data
        return np.nanstd(mtx, axis=1, ddof=1)

    def _get_colSd(self):
        """
        Calculate the standard deviation of each column.

        Returns
        -------
        numpy.ndarray
            Standard deviation of each column.
        """
        mtx = self.data
        return np.nanstd(mtx, axis=0, ddof=1)

    def get_meanRowMeans(self):
        """
        Calculate the mean of the row means.

        Returns
        -------
        float
            Mean of the row means.
        """
        row_means = self._get_rowMeans()
        return np.nanmean(row_means)

    def get_meanColMeans(self):
        """
        Calculate the mean of the column means.

        Returns
        -------
        float
            Mean of the column means.
        """
        col_means = self._get_colMeans()
        return np.nanmean(col_means)

    def get_sdRowMeans(self):
        """
        Calculate the standard deviation of the row means.

        Returns
        -------
        float
            Standard deviation of the row means.
        """
        row_means = self._get_rowMeans()
        return np.nanstd(row_means, ddof=1)

    def get_sdColMeans(self):
        """
        Calculate the standard deviation of the column means.

        Returns
        -------
        float
            Standard deviation of the column means.
        """
        col_means = self._get_colMeans()
        return np.nanstd(col_means, ddof=1)

    def get_meanRowSd(self):
        """
        Calculate the mean of the row standard deviations.

        Returns
        -------
        float
            Mean of the row standard deviations.
        """
        row_sd = self._get_rowSd()
        return np.nanmean(row_sd)

    def get_meanColSd(self):
        """
        Calculate the mean of the column standard deviations.

        Returns
        -------
        float
            Mean of the column standard deviations.
        """
        col_sd = self._get_colSd()
        return np.nanmean(col_sd)

    def get_sdRowSd(self):
        """
        Calculate the standard deviation of the row standard deviations.

        Returns
        -------
        float
            Standard deviation of the row standard deviations.
        """
        row_sd = self._get_rowSd()
        return np.nanstd(row_sd, ddof=1)

    def get_sdColSd(self):
        """
        Calculate the standard deviation of the column standard deviations.

        Returns
        -------
        float
            Standard deviation of the column standard deviations.
        """
        col_sd = self._get_colSd()
        return np.nanstd(col_sd, ddof=1)

    def get_medianRowVars(self):
        """
        Calculate the median of the row variances.

        Returns
        -------
        float
            Median of the row variances.
        """
        row_vars = self._get_rowSd() ** 2
        return np.nanmedian(row_vars)

    def get_medianColVars(self):
        """
        Calculate the median of the column variances.

        Returns
        -------
        float
            Median of the column variances.
        """
        col_vars = self._get_colSd() ** 2
        return np.nanmedian(col_vars)

    def _get_rowCorr(self, nmaxFeature=500):
        """
        Calculate pairwise correlations of rows. Only used internally.

        Parameters
        ----------
        nmaxFeature : int, optional
            Maximum number of features to compute correlations on.

        Returns
        -------
        dict
            'res' : numpy.ndarray
                Correlation matrix.
            'seed' : int
                Random seed used.
        """
        mtx = self.data
        seed_used = None
        if mtx.shape[0] > nmaxFeature:
            seed_used = np.random.randint(0, 1e6)
            np.random.seed(seed_used)
            indices = np.random.choice(mtx.shape[0], nmaxFeature, replace=False)
            mtx_sub = mtx[indices, :]
        else:
            mtx_sub = mtx
        if self.cor_method == "pearson":
            corr_res = np.corrcoef(mtx_sub)
        elif self.cor_method == "spearman":
            corr_res, _ = scipy.stats.spearmanr(mtx_sub, axis=1)
        else:
            raise ValueError("Unsupported correlation method")
        return {"res": corr_res, "seed": seed_used}

    def _get_colCorr(self, nmaxSamples=100):
        """
        Calculate pairwise correlations of columns. Only used internally.

        Parameters
        ----------
        nmaxSamples : int, optional
            Maximum number of samples to compute correlations on.

        Returns
        -------
        dict
            'res' : numpy.ndarray
                Correlation matrix.
            'seed' : int
                Random seed used.
        """
        mtx = self.data
        seed_used = None
        if mtx.shape[1] > nmaxSamples:
            seed_used = np.random.randint(0, 1e6)
            np.random.seed(seed_used)
            indices = np.random.choice(mtx.shape[1], nmaxSamples, replace=False)
            mtx_sub = mtx[:, indices]
        else:
            mtx_sub = mtx
        if self.cor_method == "pearson":
            corr_res = np.corrcoef(mtx_sub, rowvar=False)
        elif self.cor_method == "spearman":
            corr_res, _ = scipy.stats.spearmanr(mtx_sub, axis=0)
        else:
            raise ValueError("Unsupported correlation method")
        return {"res": corr_res, "seed": seed_used}

    def get_corrColSumsP0Sample(self):
        """
        Calculate the correlation between percent zeros per sample and column sums.

        Returns
        -------
        float
            Spearman correlation coefficient.
        """
        p0_col = self._get_p0Col()
        col_sums = self._get_colSums()
        if np.all(p0_col == p0_col[0]) or np.all(col_sums == col_sums[0]):
            self.logger.info("Constant values in p0_col or col_sums. Returning NaN.")
            return np.nan
        corr, _ = scipy.stats.spearmanr(p0_col, col_sums)
        return corr

    def get_bimodalityRowCorr(self):
        """
        Calculate the bimodality index of the row correlations.

        Returns
        -------
        dict
            'res': float
                Bimodality index.
            'seed': int
                Seed used.
        """
        corr_res = self._get_rowCorr()
        if isinstance(corr_res["res"], float) and np.isnan(corr_res["res"]):
            return {"res": np.nan, "seed": corr_res["seed"]}
        corr_values = corr_res["res"].flatten()
        corr_values = corr_values[~np.isnan(corr_values)]
        seed_used = corr_res["seed"]
        try:
            gmm = GaussianMixture(n_components=2, covariance_type="spherical", max_iter=10000)
            gmm.fit(corr_values.reshape(-1, 1))
            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            sigma = np.sqrt(np.mean(variances))
            delta = np.abs(np.diff(means)) / sigma
            pi = gmm.weights_[0]
            bi = delta * np.sqrt(pi * (1 - pi))
            return {"res": bi, "seed": seed_used}
        except Exception as e:
            self.logger.error(f"Error computing bimodalityRowCorr: {e}")
            return {"res": np.nan, "seed": seed_used}

    def get_bimodalityColCorr(self):
        """
        Calculate the bimodality index of the column correlations.

        Returns
        -------
        dict
            'res': float
                Bimodality index.
            'seed': int
                Seed used.
        """
        corr_res = self._get_colCorr()
        if isinstance(corr_res["res"], float) and np.isnan(corr_res["res"]):
            return {"res": np.nan, "seed": corr_res["seed"]}
        corr_values = corr_res["res"].flatten()
        corr_values = corr_values[~np.isnan(corr_values)]
        seed_used = corr_res["seed"]

        try:
            gmm = GaussianMixture(n_components=2, covariance_type="spherical", max_iter=10000)
            gmm.fit(corr_values.reshape(-1, 1))
            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            sigma = np.sqrt(np.mean(variances))
            delta = np.abs(np.diff(means)) / sigma
            pi = gmm.weights_[0]
            bi = delta * np.sqrt(pi * (1 - pi))
            return {"res": bi, "seed": seed_used}
        except Exception as e:
            self.logger.error(f"Error computing bimodalityColCorr: {e}")
            return {"res": np.nan, "seed": seed_used}

    def get_meanRowMeansLog2cpm(self):
        """
        Calculate the mean of row means of log2 CPM.

        Returns
        -------
        float
            Mean of the row means of log2 CPM.
        """
        row_means_log2cpm = self._get_rowMeansLog2cpm()
        return np.nanmean(row_means_log2cpm)

    def get_medianRowMediansLog2cpm(self):
        """
        Calculate the median of row medians of log2 CPM.

        Returns
        -------
        float
            Median of the row medians of log2 CPM.
        """
        row_medians_log2cpm = self._get_rowMediansLog2cpm()
        return np.nanmedian(row_medians_log2cpm)

    def get_meanColSums(self):
        """
        Calculate the mean of the column sums.

        Returns
        -------
        float
            Mean of the column sums.
        """
        col_sums = self._get_colSums()
        return np.nanmean(col_sums)

    def get_meanRowVarsLog2cpm(self):
        """
        Calculate the mean of the row variances of log2 CPM.

        Returns
        -------
        float
            Mean of the row variances of log2 CPM.
        """
        row_vars_log2cpm = self._get_rowVarsLog2cpm()
        return np.nanmean(row_vars_log2cpm)

    def get_meanColCorr(self):
        """
        Calculate the mean of the column correlations.

        Returns
        -------
        dict
            'res': float
                Mean of column correlations.
            'seed': int
                Seed used.
        """
        corr_res = self._get_colCorr()
        mean_corr = np.nanmean(corr_res["res"])
        return {"res": mean_corr, "seed": corr_res["seed"]}

    def get_meanRowCorr(self):
        """
        Calculate the mean of the row correlations.

        Returns
        -------
        dict
            'res': float
                Mean of row correlations.
            'seed': int
                Seed used.
        """
        corr_res = self._get_rowCorr()
        mean_corr = np.nanmean(corr_res["res"])
        return {"res": mean_corr, "seed": corr_res["seed"]}

    def get_sdRowMeansLog2cpm(self):
        """
        Calculate the standard deviation of the row means of log2 CPM.

        Returns
        -------
        float
            Standard deviation of the row means of log2 CPM.
        """
        row_means_log2cpm = self._get_rowMeansLog2cpm()
        return np.nanstd(row_means_log2cpm, ddof=1)

    def get_sdRowMediansLog2cpm(self):
        """
        Calculate the standard deviation of the row medians of log2 CPM.

        Returns
        -------
        float
            Standard deviation of the row medians of log2 CPM.
        """
        row_medians_log2cpm = self._get_rowMediansLog2cpm()
        return np.nanstd(row_medians_log2cpm, ddof=1)

    def get_sdRowVarsLog2cpm(self):
        """
        Calculate the standard deviation of the row variances of log2 CPM.

        Returns
        -------
        float
            Standard deviation of the row variances of log2 CPM.
        """
        row_vars_log2cpm = self._get_rowVarsLog2cpm()
        return np.nanstd(row_vars_log2cpm, ddof=1)

    def get_sdColCorr(self):
        """
        Calculate the standard deviation of the column correlations.

        Returns
        -------
        dict
            'res': float
                Standard deviation of column correlations.
            'seed': int
                Seed used.
        """
        corr_res = self._get_colCorr()
        sd_corr = np.nanstd(corr_res["res"], ddof=1)
        return {"res": sd_corr, "seed": corr_res["seed"]}

    def get_sdRowCorr(self):
        """
        Calculate the standard deviation of the row correlations.

        Returns
        -------
        dict
            'res': float
                Standard deviation of row correlations.
            'seed': int
                Seed used.
        """
        corr_res = self._get_rowCorr()
        sd_corr = np.nanstd(corr_res["res"], ddof=1)
        return {"res": sd_corr, "seed": corr_res["seed"]}

    def get_LinCoefPoly2(self):
        """
        Calculate the linear coefficient of a quadratic fit of row means vs row variances of log2 CPM.

        Legacy name: LinearCoefPoly2XRowMeansLog2cpmYRowVarsLog2cpm

        Returns
        -------
        float
            Linear coefficient.
        """
        x = self._get_rowMeansLog2cpm()
        y = self._get_rowVarsLog2cpm()
        valid = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 3:
            return np.nan

        coeffs = np.polyfit(x, y, 2)
        return coeffs[1]

    def get_QuadCoefPoly2(self):
        """
        Calculate the quadratic coefficient of a quadratic fit of row means vs row variances of log2 CPM.

        Legacy Name: QuadraticCoefPoly2XRowMeansLog2cpmYRowVarsLog2cpm

        Returns
        -------
        float
            Quadratic coefficient.
        """
        x = self._get_rowMeansLog2cpm()
        y = self._get_rowVarsLog2cpm()
        valid = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 3:
            return np.nan
        if np.ptp(x) == 0 or np.ptp(y) == 0:  # Check range (max - min)
            return np.nan
        coeffs = np.polyfit(x, y, 2)
        return coeffs[0]

    def get_poly1Xp0YRowMeans(self):
        """
        Calculate the slope of the linear fit of percent zeros per row vs row means of log2 CPM.

        Legacy Name: slopeCoefPoly1Xp0RowYRowMeanslog2cpm

        Returns
        -------
        float
            Slope coefficient.
        """
        x = self._get_p0Row()
        y = self._get_rowMeansLog2cpm()
        valid = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 2:
            return np.nan
        if np.ptp(x) == 0 or np.ptp(y) == 0:  # Check range (max - min)
            return np.nan
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]

    def get_coefHclustRows(self, naToZero=False):
        """
        Calculate the cophenetic correlation coefficient for hierarchical clustering of rows.

        Parameters
        ----------
        naToZero : bool, optional
            If True, convert NaNs to zero.

        Returns
        -------
        dict
            'res': float
                Cophenetic correlation coefficient.
            'seed': int
                Seed used.
        """
        mtx = self.data.copy()
        if naToZero:
            mtx = np.nan_to_num(mtx)
        else:
            mtx = mtx[~np.isnan(mtx).any(axis=1)]
        if mtx.shape[0] > 500:
            seed_used = np.random.randint(0, 1e6)
            np.random.seed(seed_used)
            indices = np.random.choice(mtx.shape[0], 500, replace=False)
            mtx_sub = mtx[indices, :]
        else:
            mtx_sub = mtx
            seed_used = None
        try:
            Z = linkage(mtx_sub, method="average")
            c, coph_dists = cophenet(Z, pdist(mtx_sub))
            return {"res": c, "seed": seed_used}
        except Exception as e:
            self.logger.error(f"Error computing coefHclustRows: {e}")
            return {"res": np.nan, "seed": seed_used}

    def get_coefHclustCols(self, naToZero=False):
        """
        Calculate the cophenetic correlation coefficient for hierarchical clustering of columns.

        Parameters
        ----------
        naToZero : bool, optional
            If True, convert NaNs to zero.

        Returns
        -------
        dict
            'res': float
                Cophenetic correlation coefficient.
            'seed': int or None
                Seed used for random sampling.
        """
        mtx = self.data.copy()
        if naToZero:
            mtx = np.nan_to_num(mtx)
        else:
            mtx = mtx[:, ~np.isnan(mtx).any(axis=0)]  # Remove columns with NaNs

        if mtx.shape[1] > 500:
            seed_used = np.random.randint(0, 1e6)
            np.random.seed(seed_used)
            indices_cols = np.random.choice(mtx.shape[1], 500, replace=False)
            mtx_sub = mtx[:, indices_cols]
        else:
            mtx_sub = mtx
            seed_used = None
        try:
            Z = linkage(mtx_sub.T, method="average")  # Transpose to cluster columns
            c, coph_dists = cophenet(Z, pdist(mtx_sub.T))
            return {"res": c, "seed": seed_used}
        except Exception as e:
            self.logger.error(f"Error computing coefHclustCols: {e}")
            return {"res": np.nan, "seed": seed_used}

    def get_energy(self):
        """
        Calculate the energy of the data matrix (sum of squares of the elements).

        Returns
        -------
        float
            Energy of the data matrix.
        """
        mtx = self.data
        return np.nansum(mtx**2)

    def get_entropy(self, base=2, nbins=None):
        """
        Calculate the entropy of the data matrix.

        Parameters
        ----------
        base : float, optional
            The logarithmic base to use. Default is 2.
        nbins : int, optional
            Number of bins to discretize the data into.

        Returns
        -------
        float
            Entropy of the data matrix.
        """
        mtx = self.data.flatten()
        mtx = mtx[~np.isnan(mtx)]
        if nbins is None:
            nbins = len(np.unique(mtx))
        hist, _ = np.histogram(mtx, bins=nbins, density=True)
        hist = hist[hist > 0]  # Remove zero entries
        entropy = -np.sum(hist * np.log(hist) / np.log(base))
        return entropy

    def get_kurtosis(self):
        """
        Calculate the kurtosis of the data matrix.

        Returns
        -------
        float
            Kurtosis of the data matrix.
        """
        mtx = self.data.flatten()
        mtx = mtx[~np.isnan(mtx)]
        return scipy.stats.kurtosis(mtx, fisher=True, bias=False)

    def get_meanDeviation(self):
        """
        Calculate the mean absolute deviation of the data matrix.

        Returns
        -------
        float
            Mean absolute deviation.
        """
        mtx = self.data.flatten()
        mtx = mtx[~np.isnan(mtx)]
        return np.mean(np.abs(mtx - np.mean(mtx)))

    def get_skewness(self):
        """
        Calculate the skewness of the data matrix.

        Returns
        -------
        float
            Skewness of the data matrix.
        """
        mtx = self.data.flatten()
        mtx = mtx[~np.isnan(mtx)]
        return scipy.stats.skew(mtx, bias=False)

    def get_uniformity(self, nbins=None):
        """
        Calculate the uniformity of the data matrix.

        Parameters
        ----------
        nbins : int, optional
            Number of bins to discretize the data into.

        Returns
        -------
        float
            Uniformity measure.
        """
        mtx = self.data.flatten()
        mtx = mtx[~np.isnan(mtx)]
        if nbins is None:
            nbins = len(np.unique(mtx))
        hist, _ = np.histogram(mtx, bins=nbins, density=True)
        hist = hist / np.sum(hist)  # Normalize
        return np.sum(hist**2)

    def get_variance(self):
        """
        Calculate the variance of the data matrix.

        Returns
        -------
        float
            Variance of the data matrix.
        """
        mtx = self.data.flatten()
        return np.nanvar(mtx, ddof=1)

    def get_RMS(self):
        """
        Calculate the root mean square (RMS) of the data matrix.

        Returns
        -------
        float
            RMS of the data matrix.
        """
        mtx = self.data.flatten()
        return np.sqrt(np.nanmean(mtx**2))

    def get_prctPC1(self):
        """
        Calculate the percentage of variance explained by the first principal component.

        Returns
        -------
        float
            Percentage of variance explained by PC1.
        """
        mtx = self.data.copy().T  # Transpose to match R code
        mtx = mtx[:, ~np.isnan(mtx).any(axis=0)]  # Remove columns with NaNs
        mtx = mtx[:, np.nanvar(mtx, axis=0) != 0]  # Remove zero variance columns
        if mtx.shape[1] < 1:
            return np.nan
        try:
            pca = PCA(n_components=1)
            pca.fit(mtx)
            return pca.explained_variance_ratio_[0]
        except Exception as e:
            self.logger.error(f"Error computing prctPC1: {e}")
            return np.nan

    def get_prctPC2(self):
        """
        Calculate the percentage of variance explained by the second principal component.

        Returns
        -------
        float
            Percentage of variance explained by PC2.
        """
        mtx = self.data.copy().T  # Transpose to match R code
        mtx = mtx[:, ~np.isnan(mtx).any(axis=0)]  # Remove columns with NaNs
        mtx = mtx[:, np.nanvar(mtx, axis=0) != 0]  # Remove zero variance columns
        if mtx.shape[1] < 2:
            return np.nan
        try:
            pca = PCA(n_components=2)
            pca.fit(mtx)
            return pca.explained_variance_ratio_[1]
        except Exception as e:
            self.logger.error(f"Error computing prctPC2: {e}")
            return np.nan

    def get_pNA(self):
        """
        Calculate the percentage of NaNs in the data matrix.

        Returns
        -------
        float
            Percentage of NaNs in the data matrix.
        """
        mtx = self.data
        return np.mean(np.isnan(mtx))

    def _get_pNACol(self):
        """
        Calculate the percentage of NaNs in each column.

        Returns
        -------
        numpy.ndarray
            Percentage of NaNs per column.
        """
        mtx = self.data
        return np.mean(np.isnan(mtx), axis=0)

    def _get_pNARow(self):
        """
        Calculate the percentage of NaNs in each row.

        Returns
        -------
        numpy.ndarray
            Percentage of NaNs per row.
        """
        mtx = self.data
        return np.mean(np.isnan(mtx), axis=1)

    def get_meanPNACol(self):
        """
        Calculate the mean percentage of NaNs in the columns.

        Returns
        -------
        float
            Mean percentage of NaNs per column.
        """
        pNACol = self._get_pNACol()
        return np.mean(pNACol)

    def get_meanPNARow(self):
        """
        Calculate the mean percentage of NaNs in the rows.

        Returns
        -------
        float
            Mean percentage of NaNs per row.
        """
        pNARow = self._get_pNARow()
        return np.mean(pNARow)

    def get_pRowsWoNA(self):
        """
        Calculate the percentage of rows without any NaNs.

        Returns
        -------
        float
            Percentage of rows without NaNs.
        """
        mtx = self.data
        rows_no_na = np.sum(~np.isnan(mtx).any(axis=1))
        return rows_no_na / mtx.shape[0]

    def get_pvalCorrpNAColAndColMeans(self):
        """
        Calculate the p-value of the correlation between percentage of NaNs per column and column means.

        Returns
        -------
        float
            p-value of the correlation test.
        """
        x = self._get_pNACol()
        y = self._get_colMeans()
        if len(x) < 3:
            return np.nan
        if np.all(x == x[0]) or np.all(y == y[0]):
            self.logger.info("Constant values in pNACol or colMeans. Returning NaN.")
            return np.nan
        try:
            if self.cor_method == "spearman":
                corr_res = scipy.stats.spearmanr(x, y)
            if self.cor_method == "pearson":
                corr_res = scipy.stats.pearsonr(x, y)
            return corr_res.pvalue
        except Exception as e:
            self.logger.error(f"Error computing pvalCorrpNAColAndColMeans: {e}")
            return np.nan

    def get_pvalCorrpNARowAndRowMeans(self):
        """
        Calculate the p-value of the correlation between percentage of NaNs per row and row means.

        Returns
        -------
        float
            p-value of the correlation test.
        """
        x = self._get_pNARow()
        y = self._get_rowMeans()

        if np.all(x == x[0]) or np.all(y == y[0]):
            self.logger.info("Constant values in pNACol or colMeans. Returning NaN.")
            return np.nan

        if len(x) < 3:
            return np.nan
        try:
            if self.cor_method == "spearman":
                corr_res = scipy.stats.spearmanr(x, y)
            else:
                corr_res = scipy.stats.pearsonr(x, y)
            return corr_res.pvalue
        except Exception as e:
            self.logger.error(f"Error computing pvalCorrpNARowAndRowMeans: {e}")
            return np.nan

    def get_corrpNARowAndRowMeans(self):
        """
        Calculate the correlation between percentage of NaNs per row and row means.

        Returns
        -------
        float
            Correlation coefficient.
        """
        x = self._get_pNARow()
        y = self._get_rowMeans()
        if np.all(x == x[0]) or np.all(y == y[0]):
            self.logger.info("Constant values in pNACol or colMeans. Returning NaN.")
            return np.nan

        if len(x) < 3:
            return np.nan
        try:
            if self.cor_method == "spearman":
                corr_res = scipy.stats.spearmanr(x, y)
                return corr_res.correlation
            else:
                corr_res = scipy.stats.pearsonr(x, y)
                return corr_res[0]
        except Exception as e:
            self.logger.error(f"Error computing corrpNARowAndRowMeans: {e}")
            return np.nan

    def get_rowMeans10pQuantileSdQuotient(self):
        """
        Calculate the quotient of the mean row SD of the bottom 10% row means divided by overall mean row SD.

        Returns
        -------
        float
            Quotient value.
        """
        row_means = self._get_rowMeans()
        quantile_value = np.nanpercentile(row_means, 10)
        indices = np.where(row_means < quantile_value)[0]
        if len(indices) == 0:
            return np.nan
        quantile_mtx = self.data[indices, :]
        mean_row_sd_quantile = np.nanmean(np.nanstd(quantile_mtx, axis=1, ddof=1))
        mean_row_sd_all = self.get_meanRowSd()
        if mean_row_sd_all == 0:
            return np.nan
        return mean_row_sd_quantile / mean_row_sd_all

    def get_FCmax10pQuantileSdQuotient(self):
        """
        Calculate the quotient of mean row SDs for features with top 10% fold changes.

        Returns
        -------
        float
            Quotient value.

        Notes
        -----
        Assumes the data matrix is arranged with samples from group 1 followed by group 2.
        """
        mtx = self.data
        n_samples = mtx.shape[1]
        group_size = n_samples // 2
        if group_size < 1:
            self.logger.error("Not enough samples to divide into two groups.")
            return np.nan
        mean_group1 = np.nanmean(mtx[:, :group_size], axis=1)
        mean_group2 = np.nanmean(mtx[:, group_size : group_size * 2], axis=1)
        FCMeansDE = mean_group2 - mean_group1
        quantile_value = np.nanpercentile(FCMeansDE, 90)
        indices = np.where(FCMeansDE > quantile_value)[0]
        if len(indices) == 0:
            return np.nan
        quantile_mtx = mtx[indices, :]
        mean_row_sd_quantile = np.nanmean(np.nanstd(quantile_mtx, axis=1, ddof=1))
        mean_row_sd_all = self.get_meanRowSd()
        if mean_row_sd_all == 0:
            return np.nan
        return mean_row_sd_quantile / mean_row_sd_all
