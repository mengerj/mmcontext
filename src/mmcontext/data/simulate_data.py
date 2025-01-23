import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


class InputExample:
    """Structure for one input example with texts, the label and a unique id"""

    def __init__(self, guid: str = "", texts: list = None, label: int | float = 0):
        """
        Creates one InputExample with the given texts, guid and label

        Parameters
        ----------
        guid : str, optional
            A unique id for the example.
        texts : list of dict or str
            The texts for the example. In our case, it can be: [omicsSample_dict, "caption string"].
        label : int or float, optional
            The label for the example. Default is 0.
        """
        self.guid = guid
        self.texts = texts if texts is not None else []
        self.label = label

    def __str__(self):
        return f"<InputExample> label: {self.label}, texts: {self.texts}"


def simulate_omics_dataset_with_captions(
    num_samples_per_cell_type: int,
    num_features: int,
    diff_factor: float = 1.0,
    base_mean: float = 2.0,
    base_std_dev: float = 1.0,
    base_zero_fraction: float = 0.1,
    random_state: int = 42,
) -> list[InputExample]:
    """Simulation of omics datasets

    Simulates omics samples for 10 different cell types, each with a distinct caption and
    a parameterized normal distribution for generating feature counts. For each sample,
    this function creates `InputExample` instances pairing the omics data with all 10 possible
    cell-type captions, where the correct one is labeled 1 and the others are labeled 0.

    Parameters
    ----------
    num_samples_per_cell_type : int
        Number of samples to generate for each cell type.
    num_features : int
        Number of features in each omics sample.
    diff_factor : float, optional
        A factor that determines how different the distributions are across the 10 cell types.
        Larger values produce larger differences in mean, std, and zero-fraction among cell types.
    base_mean : float, optional
        The base mean for the normal distribution for cell type 0. Subsequent cell types
        incrementally modify this based on the `diff_factor`.
    base_std_dev : float, optional
        The base standard deviation for cell type 0. Subsequent cell types incrementally modify
        this based on the `diff_factor`.
    base_zero_fraction : float, optional
        The base zero-fraction (fraction of features forced to zero) for cell type 0.
        Subsequent cell types incrementally modify this based on the `diff_factor`.
    random_state : int, optional
        Seed for reproducibility in random operations.

    Returns
    -------
    list[InputExample]
        A list of InputExample instances. Each instance has the omics data (dict) as the first
        text entry and the caption (str) as the second. The label is 1 if the caption matches
        the omics cell type, else 0.

    Notes
    -----
    Simulated data is artificially generated in-memory. No external data source is used.
    """
    logger.info("Starting simulation of omics dataset with captions.")
    np.random.seed(random_state)
    random.seed(random_state)

    # Predefine 10 different cell types with their base descriptions
    # We'll dynamically set their distribution parameters
    base_cell_types = [
        {"cell_type_name": f"CellType{i}", "caption": f"This is the description for cell type {i}"} for i in range(10)
    ]

    # Initialize a list to hold all InputExample objects
    dataset_examples = []

    # Create data for each cell type
    for i, cell_type_info in enumerate(base_cell_types):
        # Determine distribution parameters for this cell type
        mean = base_mean + i * diff_factor
        std_dev = base_std_dev + i * (diff_factor * 0.1)
        zero_fraction = base_zero_fraction + i * (diff_factor * 0.01)

        logger.debug(
            "CellType: %s | Mean: %.3f | StdDev: %.3f | ZeroFraction: %.3f",
            cell_type_info["cell_type_name"],
            mean,
            std_dev,
            zero_fraction,
        )

        # Generate samples for this cell type
        for sample_idx in range(num_samples_per_cell_type):
            # Generate feature counts from a normal distribution
            counts = np.random.normal(loc=mean, scale=std_dev, size=num_features)
            counts = np.abs(counts)  # Make all counts non-negative

            # Randomly set some values to zero
            num_zeros = int(zero_fraction * num_features)
            zero_indices = np.random.choice(range(num_features), size=num_zeros, replace=False)
            counts[zero_indices] = 0

            # Generate random feature IDs
            featureIDs = [str(random.randint(10000, 99999)) for _ in range(num_features)]
            feature_to_idx = {fid: idx for idx, fid in enumerate(featureIDs)}

            # Build the omics sample dictionary
            omics_sample = {
                "counts": counts,
                "featureIDs": featureIDs,
                "feature_to_idx": feature_to_idx,
                "sample_id": f"{cell_type_info['cell_type_name']}_sample_{sample_idx}",
                "is_omics": True,
            }

            # Now pair this omics_sample with all 10 possible captions
            for j, other_ct_info in enumerate(base_cell_types):
                # Label = 1 if it's the correct cell type, else 0
                label = 1 if j == i else 0
                example_id = f"{cell_type_info['cell_type_name']}_sample_{sample_idx}_caption_{j}"

                input_example = InputExample(
                    guid=example_id, texts=[omics_sample, other_ct_info["caption"]], label=label
                )
                dataset_examples.append(input_example)

    logger.info("Completed simulation. Generated %d InputExample instances.", len(dataset_examples))
    return dataset_examples
