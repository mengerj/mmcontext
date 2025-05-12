from collections.abc import Callable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

# ---------------------------------------------------------------------------
# Default vocabularies
# ---------------------------------------------------------------------------
CELL_TYPES_DEFAULT = [
    "T cell",
    "B cell",
    "NK cell",
    "Macrophage",
    "Dendritic cell",
    "Epithelial",
    "Fibroblast",
    "Endothelial",
    "Neuron",
    "Astrocyte",
]

TISSUES_DEFAULT = [
    "lung",
    "liver",
    "skin",
    "brain",
    "blood",
]

ASSAYS_DEFAULT = [
    "scRNA-seq",
    "ATAC-seq",
    "CITE-seq",
]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class OmicsCaptionSimulator:
    """Simulate an AnnData object **and** a contrastive HF‑Dataset.

    The HF dataset contains *positive* and *negative* (caption, token‑set)
    pairs with a binary `label` column.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_genes: int = 1000,
        *,
        cell_types: Sequence[str] = CELL_TYPES_DEFAULT,
        tissues: Sequence[str] = TISSUES_DEFAULT,
        assays: Sequence[str] = ASSAYS_DEFAULT,
        val_fraction: float = 0.2,
        rng_seed: int | None = 0,
        numeric_sampler: Callable[[int, int], np.ndarray] | None = None,
        dim_sample: int = 32,
        dim_gene: int = 16,
    ) -> None:
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.cell_types = list(cell_types)
        self.tissues = list(tissues)
        self.assays = list(assays)
        self.val_fraction = val_fraction
        self.rng = np.random.default_rng(rng_seed)
        self.numeric_sampler = numeric_sampler or (lambda n, d: self.rng.standard_normal((n, d)).astype(np.float32))
        self.dim_sample = dim_sample
        self.dim_gene = dim_gene

        # placeholders filled by ``simulate``
        self.adata: ad.AnnData | None = None
        self.hf_dataset: Dataset | None = None
        self.sample_embedding: pd.DataFrame | None = None
        self.gene_embedding: pd.DataFrame | None = None

    # -------------------------------------------------------------------
    # main
    # -------------------------------------------------------------------
    def simulate(self) -> "OmicsCaptionSimulator":
        """Run the complete simulation pipeline."""
        self._build_anndata()
        self._build_embeddings()
        self._build_hf_dataset()
        return self

    # -------------------------------------------------------------------
    # building blocks
    # -------------------------------------------------------------------
    def _build_anndata(self) -> None:
        sample_ids = [f"S{i + 1}" for i in range(self.n_samples)]
        obs = pd.DataFrame(
            {
                "sample_idx": sample_ids,
                "cell_type": self.rng.choice(self.cell_types, self.n_samples),
                "tissue": self.rng.choice(self.tissues, self.n_samples),
                "assay": self.rng.choice(self.assays, self.n_samples),
            }
        ).set_index("sample_idx")

        gene_ids = [f"g{i + 1}" for i in range(self.n_genes)]
        var = pd.DataFrame(index=gene_ids)
        var["gene_id"] = gene_ids

        X = np.zeros((self.n_samples, self.n_genes), dtype=np.float32)
        self.adata = ad.AnnData(X=X, obs=obs, var=var)

    def _build_embeddings(self) -> None:
        if self.adata is None:
            raise RuntimeError("Call _build_anndata first")
        self.sample_embedding = pd.DataFrame(
            self.numeric_sampler(self.n_samples, self.dim_sample),
            index=self.adata.obs_names,
        )
        self.gene_embedding = pd.DataFrame(
            self.numeric_sampler(self.n_genes, self.dim_gene),
            index=self.adata.var_names,
        )
        self.adata.obsm["sample_vec"] = self.sample_embedding.values
        self.adata.varm["gene_vec"] = self.gene_embedding.values

    # ------------------------------------------------------------------- #
    # HF dataset with **disjoint** sample splits
    # -------------------------------------------------------------------
    def _build_hf_dataset(self) -> None:
        if self.adata is None:
            raise RuntimeError("AnnData not initialised")
        obs = self.adata.obs
        gene_ids = list(self.adata.var_names)

        captions = [
            f"This is a {ct} from {ti} with {as_}"
            for ct, ti, as_ in zip(obs["cell_type"], obs["tissue"], obs["assay"], strict=False)
        ]
        gene_sents = [" ".join(self.rng.choice(gene_ids, 10, replace=False)) for _ in obs.index]
        sample_tokens = list(obs.index)

        # build row dicts --------------------------------------------------
        data_rows = []
        for sid, genes, cap in zip(sample_tokens, gene_sents, captions, strict=False):
            data_rows.append(
                {"sample_idx": sid, "cell_sentence_1": sid, "cell_sentence_2": genes, "captions": cap, "label": 1}
            )
            # negative row – mismatched caption
            wrong_cap = captions[self.rng.integers(self.n_samples)]
            data_rows.append(
                {"sample_idx": sid, "cell_sentence_1": sid, "cell_sentence_2": genes, "captions": wrong_cap, "label": 0}
            )

        full_ds = Dataset.from_list(data_rows)

        # -------- split by *sample_idx* so no ID leaks across splits ------
        unique_ids = np.array(sample_tokens)
        self.rng.shuffle(unique_ids)

        n_val = int(len(unique_ids) * self.val_fraction)
        val_ids = set(unique_ids[:n_val])
        train_ids = set(unique_ids[n_val:])

        ds_train = full_ds.filter(lambda row: row["sample_idx"] in train_ids)
        ds_val = full_ds.filter(lambda row: row["sample_idx"] in val_ids)

        self.hf_dataset = DatasetDict(train=ds_train, validation=ds_val)

    # -------------------------------------------------------------------
    # public getters
    # -------------------------------------------------------------------
    def get_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return numeric lookup tables (gene_df, sample_df)."""
        gene_df = pd.DataFrame({"token": self.gene_embedding.index, "embedding": self.gene_embedding.values.tolist()})
        sample_df = pd.DataFrame(
            {"token": self.sample_embedding.index, "embedding": self.sample_embedding.values.tolist()}
        )
        return gene_df, sample_df

    def get_hf_dataset(self) -> Dataset:
        """Return the HF dataset with positive and negative samples."""
        if self.hf_dataset is None:
            raise RuntimeError("Call simulate() first")
        return self.hf_dataset
