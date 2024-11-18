import logging

import numpy as np
import scipy.stats
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def compare_data_properties(
    data1,
    data2,
    name1="original",
    name2="reconstructed",
    predefined_subset="simulationBenchmark",
    custom_subset=None,
    cor_method="spearman",
    logger=None,
):
    """
    Compute data properties for each of two datasets and compute log2 fold changes between each property.

    Parameters
    ----------
    data1 : numpy.ndarray
        First dataset (rows are features and columns are samples).
    data2 : numpy.ndarray
        Second dataset (rows are features and columns are samples).
    predefined_subset : str, optional
        Name of subset available for DataProperties.
    custom_subset : list of str, optional
        Optionally, a list of property names to compute.
    cor_method : str, optional
        Correlation method used for calculation of correlations.
    logger : logging.Logger, optional
        Logger to use. If None, a new logger is created.

    Returns
    -------
    dict
        Dictionary containing:
            - name1: properties of the first dataset.
            - name2: properties of the second dataset.
            - 'allLog2FC': log2 fold changes between the properties.
    """
    logger = logger or logging.getLogger(__name__)

    # Initialize DataProperties instances with the same settings
    dp = DataProperties(
        predefined_subset=predefined_subset, custom_subset=custom_subset, cor_method=cor_method, logger=logger
    )

    # Compute properties for each dataset
    properties1 = dp.compute_properties(data1)
    properties2 = dp.compute_properties(data2)

    # Get the intersection of property names
    properties = set(properties1.keys()) & set(properties2.keys())

    all_log2fc = {}
    for prop in properties:
        try:
            prop_value1 = properties1[prop]
            prop_value2 = properties2[prop]

            # Handle p-values specially
            if "pval" in prop:
                # Set significant p-values to 0.05 to reduce extreme log2FC values
                if prop_value1 < 0.05:
                    prop_value1 = 0.05
                if prop_value2 < 0.05:
                    prop_value2 = 0.05

            # If the property is a dict with 'res', extract 'res'
            if isinstance(prop_value1, dict) and "res" in prop_value1:
                prop_value1 = prop_value1["res"]
            if isinstance(prop_value2, dict) and "res" in prop_value2:
                prop_value2 = prop_value2["res"]

            # Compute log2 fold change
            current_log2fc = np.log2(np.divide(prop_value1, prop_value2))

            # Handle infinite values resulting from division by zero
            if np.isinf(current_log2fc).any():
                current_log2fc = np.nan

            all_log2fc[prop] = current_log2fc

        except Exception as e:
            logger.error(f"Error computing log2FC for property '{prop}': {e}")
            all_log2fc[prop] = np.nan

    return {name1: properties1, name2: properties2, "allLog2FC": all_log2fc}


class DataProperties:
    """
    Class to compute data properties for a given data matrix.

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
        self.data = data
        properties = self.select_properties()
        data_properties = {}
        for prop in properties:
            method_name = f"get_{prop}"
            method = getattr(self, method_name, None)
            if method:
                try:
                    result = method()
                    data_properties[prop] = result
                except Exception as e:
                    self.logger.error(f"Error computing {prop}: {e}")
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
        possible_properties = [
            method_name[4:]
            for method_name in dir(self)
            if callable(getattr(self, method_name)) and method_name.startswith("get_")
        ]

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
                    "p0Row",
                    "p0Col",
                    "median",
                    "q95",
                    "q99",
                    "rowMeansLog2cpm",
                    "rowMediansLog2cpm",
                    "rowVarsLog2cpm",
                    "colSums",
                    "medianColSums",
                    "meanColSums",
                    "sdColSums",
                    "colMeans",
                    "maxColSums",
                    "minColSums",
                    "colVarsLog2cpm",
                    "rowCorr",
                    "colCorr",
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
                    "LinearCoefPoly2XRowMeansLog2cpmYRowVarsLog2cpm",
                    "QuadraticCoefPoly2XRowMeansLog2cpmYRowVarsLog2cpm",
                    "slopeCoefPoly1Xp0RowYRowMeanslog2cpm",
                    "coefHclustRows",
                    "coefHclustCols",
                ]
                self.logger.info("Using predefined microbiome subset.")
            else:
                self.logger.error("Unknown predefined subset.")
                subset = []
        return subset

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

    def get_p0Row(self):
        """
        Calculate the percent of zeros per feature (row).

        Returns
        -------
        numpy.ndarray
            Percent of zeros per row.
        """
        mtx = self.data
        return np.sum(mtx == 0, axis=1) / mtx.shape[1]

    def get_p0Col(self):
        """
        Calculate the percent of zeros per sample (column).

        Returns
        -------
        numpy.ndarray
            Percent of zeros per column.
        """
        mtx = self.data
        return np.sum(mtx == 0, axis=0) / mtx.shape[0]

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

    def get_colSums(self):
        """
        Calculate the sum of each column.

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
        col_sums = self.get_colSums()
        return np.nanmedian(col_sums)

    def get_sdColSums(self):
        """
        Calculate the standard deviation of the column sums.

        Returns
        -------
        float
            Standard deviation of the column sums.
        """
        col_sums = self.get_colSums()
        return np.nanstd(col_sums, ddof=1)

    def get_colMeans(self):
        """
        Calculate the mean of each column.

        Returns
        -------
        numpy.ndarray
            Mean of each column.
        """
        mtx = self.data
        return np.nanmean(mtx, axis=0)

    # ... Implement other get_* methods similarly ...

    # Example of get_log2cpm
    def get_log2cpm(self):
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
        cpm = (mtx / lib_sizes) * 1e6
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
        col_sums = self.get_colSums()
        return np.nanmax(col_sums)

    def get_minColSums(self):
        """
        Calculate the minimum of the column sums.

        Returns
        -------
        float
            Minimum of the column sums.
        """
        col_sums = self.get_colSums()
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

    def get_cpm(self):
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

    def get_rowMeansLog2cpm(self):
        """
        Calculate the mean of log2 CPM for each row.

        Returns
        -------
        numpy.ndarray
            Mean log2 CPM per row.
        """
        log2cpm = self.get_log2cpm()
        return np.nanmean(log2cpm, axis=1)

    def get_rowMediansLog2cpm(self):
        """
        Calculate the median of log2 CPM for each row.

        Returns
        -------
        numpy.ndarray
            Median log2 CPM per row.
        """
        log2cpm = self.get_log2cpm()
        return np.nanmedian(log2cpm, axis=1)

    def get_rowMeansCpm(self):
        """
        Calculate the mean of CPM for each row.

        Returns
        -------
        numpy.ndarray
            Mean CPM per row.
        """
        cpm = self.get_cpm()
        return np.nanmean(cpm, axis=1)

    def get_rowMediansCpm(self):
        """
        Calculate the median of CPM for each row.

        Returns
        -------
        numpy.ndarray
            Median CPM per row.
        """
        cpm = self.get_cpm()
        return np.nanmedian(cpm, axis=1)

    def get_rowVarsLog2cpm(self):
        """
        Calculate the variance of log2 CPM for each row.

        Returns
        -------
        numpy.ndarray
            Variance of log2 CPM per row.
        """
        log2cpm = self.get_log2cpm()
        return np.nanvar(log2cpm, axis=1, ddof=1)

    def get_colMeansLog2cpm(self):
        """
        Calculate the mean of log2 CPM for each column.

        Returns
        -------
        numpy.ndarray
            Mean log2 CPM per column.
        """
        log2cpm = self.get_log2cpm()
        return np.nanmean(log2cpm, axis=0)

    def get_colMediansLog2cpm(self):
        """
        Calculate the median of log2 CPM for each column.

        Returns
        -------
        numpy.ndarray
            Median log2 CPM per column.
        """
        log2cpm = self.get_log2cpm()
        return np.nanmedian(log2cpm, axis=0)

    def get_colMeansCpm(self):
        """
        Calculate the mean of CPM for each column.

        Returns
        -------
        numpy.ndarray
            Mean CPM per column.
        """
        cpm = self.get_cpm()
        return np.nanmean(cpm, axis=0)

    def get_colMediansCpm(self):
        """
        Calculate the median of CPM for each column.

        Returns
        -------
        numpy.ndarray
            Median CPM per column.
        """
        cpm = self.get_cpm()
        return np.nanmedian(cpm, axis=0)

    def get_colVarsLog2cpm(self):
        """
        Calculate the variance of log2 CPM for each column.

        Returns
        -------
        numpy.ndarray
            Variance of log2 CPM per column.
        """
        log2cpm = self.get_log2cpm()
        return np.nanvar(log2cpm, axis=0, ddof=1)

    def get_rowMeans(self):
        """
        Calculate the mean of each row.

        Returns
        -------
        numpy.ndarray
            Mean of each row.
        """
        mtx = self.data
        return np.nanmean(mtx, axis=1)

    def get_rowSd(self):
        """
        Calculate the standard deviation of each row.

        Returns
        -------
        numpy.ndarray
            Standard deviation of each row.
        """
        mtx = self.data
        return np.nanstd(mtx, axis=1, ddof=1)

    def get_colSd(self):
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
        row_means = self.get_rowMeans()
        return np.nanmean(row_means)

    def get_meanColMeans(self):
        """
        Calculate the mean of the column means.

        Returns
        -------
        float
            Mean of the column means.
        """
        col_means = self.get_colMeans()
        return np.nanmean(col_means)

    def get_sdRowMeans(self):
        """
        Calculate the standard deviation of the row means.

        Returns
        -------
        float
            Standard deviation of the row means.
        """
        row_means = self.get_rowMeans()
        return np.nanstd(row_means, ddof=1)

    def get_sdColMeans(self):
        """
        Calculate the standard deviation of the column means.

        Returns
        -------
        float
            Standard deviation of the column means.
        """
        col_means = self.get_colMeans()
        return np.nanstd(col_means, ddof=1)

    def get_meanRowSd(self):
        """
        Calculate the mean of the row standard deviations.

        Returns
        -------
        float
            Mean of the row standard deviations.
        """
        row_sd = self.get_rowSd()
        return np.nanmean(row_sd)

    def get_meanColSd(self):
        """
        Calculate the mean of the column standard deviations.

        Returns
        -------
        float
            Mean of the column standard deviations.
        """
        col_sd = self.get_colSd()
        return np.nanmean(col_sd)

    def get_sdRowSd(self):
        """
        Calculate the standard deviation of the row standard deviations.

        Returns
        -------
        float
            Standard deviation of the row standard deviations.
        """
        row_sd = self.get_rowSd()
        return np.nanstd(row_sd, ddof=1)

    def get_sdColSd(self):
        """
        Calculate the standard deviation of the column standard deviations.

        Returns
        -------
        float
            Standard deviation of the column standard deviations.
        """
        col_sd = self.get_colSd()
        return np.nanstd(col_sd, ddof=1)

    def get_medianRowVars(self):
        """
        Calculate the median of the row variances.

        Returns
        -------
        float
            Median of the row variances.
        """
        row_vars = self.get_rowSd() ** 2
        return np.nanmedian(row_vars)

    def get_medianColVars(self):
        """
        Calculate the median of the column variances.

        Returns
        -------
        float
            Median of the column variances.
        """
        col_vars = self.get_colSd() ** 2
        return np.nanmedian(col_vars)

    def get_rowCorr(self, nmaxFeature=100):
        """
        Calculate pairwise correlations of rows.

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

    def get_colCorr(self, nmaxSamples=100):
        """
        Calculate pairwise correlations of columns.

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
        p0_col = self.get_p0Col()
        col_sums = self.get_colSums()
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
        corr_res = self.get_rowCorr()
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
        corr_res = self.get_colCorr()
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
        row_means_log2cpm = self.get_rowMeansLog2cpm()
        return np.nanmean(row_means_log2cpm)

    def get_medianRowMediansLog2cpm(self):
        """
        Calculate the median of row medians of log2 CPM.

        Returns
        -------
        float
            Median of the row medians of log2 CPM.
        """
        row_medians_log2cpm = self.get_rowMediansLog2cpm()
        return np.nanmedian(row_medians_log2cpm)

    def get_meanColSums(self):
        """
        Calculate the mean of the column sums.

        Returns
        -------
        float
            Mean of the column sums.
        """
        col_sums = self.get_colSums()
        return np.nanmean(col_sums)

    def get_meanRowVarsLog2cpm(self):
        """
        Calculate the mean of the row variances of log2 CPM.

        Returns
        -------
        float
            Mean of the row variances of log2 CPM.
        """
        row_vars_log2cpm = self.get_rowVarsLog2cpm()
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
        corr_res = self.get_colCorr()
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
        corr_res = self.get_rowCorr()
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
        row_means_log2cpm = self.get_rowMeansLog2cpm()
        return np.nanstd(row_means_log2cpm, ddof=1)

    def get_sdRowMediansLog2cpm(self):
        """
        Calculate the standard deviation of the row medians of log2 CPM.

        Returns
        -------
        float
            Standard deviation of the row medians of log2 CPM.
        """
        row_medians_log2cpm = self.get_rowMediansLog2cpm()
        return np.nanstd(row_medians_log2cpm, ddof=1)

    def get_sdRowVarsLog2cpm(self):
        """
        Calculate the standard deviation of the row variances of log2 CPM.

        Returns
        -------
        float
            Standard deviation of the row variances of log2 CPM.
        """
        row_vars_log2cpm = self.get_rowVarsLog2cpm()
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
        corr_res = self.get_colCorr()
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
        corr_res = self.get_rowCorr()
        sd_corr = np.nanstd(corr_res["res"], ddof=1)
        return {"res": sd_corr, "seed": corr_res["seed"]}

    def get_LinearCoefPoly2XRowMeansLog2cpmYRowVarsLog2cpm(self):
        """
        Calculate the linear coefficient of a quadratic fit of row means vs row variances of log2 CPM.

        Returns
        -------
        float
            Linear coefficient.
        """
        x = self.get_rowMeansLog2cpm()
        y = self.get_rowVarsLog2cpm()
        valid = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 3:
            return np.nan
        coeffs = np.polyfit(x, y, 2)
        return coeffs[1]

    def get_QuadraticCoefPoly2XRowMeansLog2cpmYRowVarsLog2cpm(self):
        """
        Calculate the quadratic coefficient of a quadratic fit of row means vs row variances of log2 CPM.

        Returns
        -------
        float
            Quadratic coefficient.
        """
        x = self.get_rowMeansLog2cpm()
        y = self.get_rowVarsLog2cpm()
        valid = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 3:
            return np.nan
        coeffs = np.polyfit(x, y, 2)
        return coeffs[0]

    def get_slopeCoefPoly1Xp0RowYRowMeanslog2cpm(self):
        """
        Calculate the slope of the linear fit of percent zeros per row vs row means of log2 CPM.

        Returns
        -------
        float
            Slope coefficient.
        """
        x = self.get_p0Row()
        y = self.get_rowMeansLog2cpm()
        valid = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 2:
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

    def get_pNACol(self):
        """
        Calculate the percentage of NaNs in each column.

        Returns
        -------
        numpy.ndarray
            Percentage of NaNs per column.
        """
        mtx = self.data
        return np.mean(np.isnan(mtx), axis=0)

    def get_pNARow(self):
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
        pNACol = self.get_pNACol()
        return np.mean(pNACol)

    def get_meanPNARow(self):
        """
        Calculate the mean percentage of NaNs in the rows.

        Returns
        -------
        float
            Mean percentage of NaNs per row.
        """
        pNARow = self.get_pNARow()
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
        x = self.get_pNACol()
        y = self.get_colMeans()
        if len(x) < 3:
            return np.nan
        try:
            if self.cor_method == "spearman":
                corr_res = scipy.stats.spearmanr(x, y)
            else:
                corr_res = scipy.stats.pearsonr(x, y)
            return corr_res.pvalue
        except Exception as e:
            self.logger.error(f"Error computing pvalCorrpNAColAndColMeans: {e}")
            return np.nan

    def get_corrpNAColAndColMeans(self):
        """
        Calculate the correlation between percentage of NaNs per column and column means.

        Returns
        -------
        float
            Correlation coefficient.
        """
        x = self.get_pNACol()
        y = self.get_colMeans()
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
            self.logger.error(f"Error computing corrpNAColAndColMeans: {e}")
            return np.nan

    def get_pvalCorrpNARowAndRowMeans(self):
        """
        Calculate the p-value of the correlation between percentage of NaNs per row and row means.

        Returns
        -------
        float
            p-value of the correlation test.
        """
        x = self.get_pNARow()
        y = self.get_rowMeans()
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
        x = self.get_pNARow()
        y = self.get_rowMeans()
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
        row_means = self.get_rowMeans()
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
