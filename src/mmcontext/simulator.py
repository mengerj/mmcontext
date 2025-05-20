import logging
from collections.abc import Callable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)
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

# --------------------------------------------------------------------------- #
# helper – standard column layouts for every ST loss family
# --------------------------------------------------------------------------- #
LOSS_PRESETS: dict[str, dict] = {
    # ------------------------------------------------ single sentence
    "single": {  # unlabeled
        "cols": ["sentence"],  # → Denoising / Contrastive-Tension
        "needs_class": False,
        "build": lambda samp, cap, lab: {"sentence": samp},
    },
    "single-class": {  # classification   Softmax / Triplet*
        "cols": ["sentence", "label"],
        "needs_class": True,
        "build": lambda samp, cap, lab: {"sentence": samp, "label": lab},
    },
    # ------------------------------------------------ anchor–positive pairs
    "pair": {  # MultipleNegativesRanking
        "cols": ["sentence1", "sentence2"],
        "needs_class": False,
        "build": lambda samp, cap, lab: {"sentence1": samp, "sentence2": cap},
    },
    "pair-binary": {  # Contrastive / OnlineContrastive
        "cols": ["sentence1", "sentence2", "label"],
        "needs_class": False,  # 1 / 0 already provided below
        "build": None,  # handled manually
    },
    # ------------------------------------------------ triplet
    "triplet": {  # BatchHard / TripletLoss
        "cols": ["anchor", "positive", "negative"],
        "needs_class": False,  # triplets self-contained
        "build": None,  # handled manually
    },
}


# -----------------------------------------------------------
# helper: build a label-aware numeric vector sampler
# -----------------------------------------------------------
def make_cluster_sampler(
    labels: Sequence[str | int],
    batch_labels: Sequence[str | int] | None = None,
    *,
    noise: float = 0.05,
    batch_shift: float = 0.2,
    rng_seed: int = 0,
) -> Callable[[int, int], np.ndarray]:
    """
    Build a sampler that adds **class** and **batch/assay** structure.

    Each sample is drawn from

    ``xᵢ = μ_class[labelᵢ] + γ_batch[batchᵢ] + εᵢ``
    with εᵢ ∼ 𝒩(0, noise² I).

    Parameters
    ----------
    labels
        Per-sample class labels (e.g. ``adata.obs["cell_type"]``).
    batch_labels
        Per-sample batch/assay labels (e.g. ``adata.obs["assay"]``).
        If *None* (default) the sampler assigns batches uniformly at random.
    noise
        Isotropic standard deviation of the residual noise ε.
    batch_shift
        Standard deviation of the batch shift vectors γ.
    rng_seed
        Seed passed to :pyclass:`numpy.random.default_rng` for reproducibility.

    Returns
    -------
    sampler : Callable[[int, int], np.ndarray]
        Function ``sampler(n, dim)``:

        * If ``n == len(labels)`` it returns the *sample matrix*
          (class + batch + noise).
        * Otherwise (e.g. for a gene matrix) it returns i.i.d. 𝒩(0, 1).

        The callable exposes
        ``sampler.batch_index`` (array of batch IDs, ordered like *labels*).

    Notes
    -----
    *Centroids* μ and *batch shifts* γ are created lazily the first time the
    sample matrix is requested, so the routine stays highly efficient for
    repeated calls with different dimensions.

    Examples
    --------
    >>> sampler = make_cluster_sampler(
    ...     labels=["T", "B", "T", "NK"],
    ...     batch_labels=["10x", "10x", "Smart-seq", "10x"],
    ...     rng_seed=42,
    ... )
    >>> X = sampler(4, 128)
    >>> sampler.batch_index
    array(['10x', '10x', 'Smart-seq', '10x'], dtype='<U9')
    """
    rng = np.random.default_rng(rng_seed)

    labels = np.asarray(labels)
    n_samples = len(labels)

    # ------------------------------------------------------------------ #
    # 1.  Batch assignments
    # ------------------------------------------------------------------ #
    if batch_labels is None:
        batch_labels = rng.choice(["batch0", "batch1"], size=n_samples)
        logger.info("No batch_labels provided – assigned %d random batches", 2)
    else:
        batch_labels = np.asarray(batch_labels)
        if batch_labels.shape != (n_samples,):
            raise ValueError("batch_labels must have the same length as labels")

    uniq_labels = np.unique(labels)
    uniq_batches = np.unique(batch_labels)

    centroids: dict[str, np.ndarray] = {}  # class centroids (lazy)
    batch_vectors: dict[str, np.ndarray] = {}  # batch shift vectors (lazy)

    def _sampler(n: int, dim: int) -> np.ndarray:
        nonlocal centroids, batch_vectors

        # ----------------------- sample matrix ------------------------ #
        if n == n_samples:
            if not centroids:
                logger.debug("Initialising %d class centroids in %d-D", len(uniq_labels), dim)
                centroids = {lab: rng.standard_normal(dim).astype(np.float32) for lab in uniq_labels}

            if not batch_vectors:
                logger.debug(
                    "Initialising %d batch vectors (shift σ = %.3f)",
                    len(uniq_batches),
                    batch_shift,
                )
                batch_vectors = {b: rng.standard_normal(dim).astype(np.float32) * batch_shift for b in uniq_batches}

            out = np.empty((n, dim), dtype=np.float32)
            for i, (lab, b) in enumerate(zip(labels, batch_labels, strict=True)):
                μ = centroids[lab]
                γ = batch_vectors[b]
                ε = noise * rng.standard_normal(dim).astype(np.float32)
                out[i] = μ + γ + ε
            return out

        # -------------------- everything else ------------------------- #
        return rng.standard_normal((n, dim), dtype=np.float32)

    # make batch labels available for diagnostics
    _sampler.batch_index = batch_labels
    return _sampler


'''
def make_cluster_sampler(labels, *, noise=0.05, rng_seed=0):
    """
    Sample a shifted normal distribution for each label.

    f(n, dim):
      • For *samples*  – returns tight clusters per label
      • For anything else (e.g. gene matrix) – iid N(0,1)
    """
    import numpy as np

    rng = np.random.default_rng(rng_seed)

    uniq = sorted(set(labels))
    centroids: dict[str, np.ndarray] = {}  # filled lazily when dim is known

    def _sampler(n, dim):
        nonlocal centroids
        if n == len(labels):  # we're sampling the *sample* matrix
            if not centroids:  # create fixed centres once
                centroids = {lab: rng.standard_normal(dim) for lab in uniq}

            vecs = np.empty((n, dim), dtype=np.float32)
            for i, lab in enumerate(labels):
                center = centroids[lab]
                vecs[i] = center + noise * rng.standard_normal(dim)
            return vecs
        else:  # e.g. gene embedding matrix
            return rng.standard_normal((n, dim)).astype(np.float32)

    return _sampler
'''


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
        use_gene_level: bool = False,
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
        self.use_gene_level = use_gene_level

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

    def _build_hf_dataset(
        self,
        preset: str = "pair-binary",
        *,
        class_encode: bool = True,
    ) -> None:
        """
        Build a HF dataset matching one of the Sentence-Transformers loss presets.

        Parameters
        ----------
        preset : str
            One of LOSS_PRESETS keys.
        class_encode : bool
            If the preset needs class labels, convert the string labels to integers
            via `datasets.ClassLabel`.
        """
        spec = LOSS_PRESETS[preset]  # raises KeyError if bad
        needs_cls = spec["needs_class"]

        # ------------------------------------------------ raw strings --------
        obs = self.adata.obs
        sample_ids = obs.index.tolist()  # S1 … SN
        gene_ids = list(self.adata.var_names)

        captions = [
            f"This is a {ct} from {ti} with {as_}"
            for ct, ti, as_ in zip(obs["cell_type"], obs["tissue"], obs["assay"], strict=False)
        ]
        cell_types = obs["cell_type"].tolist()
        gene_sents = [" ".join(self.rng.choice(gene_ids, 10, replace=False)) for _ in sample_ids]

        rows: list[dict] = []
        if self.use_gene_level:
            # Use gene sentences as primary data
            for sid, genes, _cap, ct in zip(sample_ids, gene_sents, captions, cell_types, strict=False):
                if preset == "pair-binary":  # (anchor,pos/neg)+binary label
                    rows.append(
                        {
                            "sentence1": genes,
                            "sentence2": ct,
                            "label": 1,
                            "sample_idx": sid,  # <─ keep ID for splitting
                        }
                    )
                    wrong_ct = self.rng.choice(cell_types)
                    rows.append(
                        {
                            "sentence1": genes,
                            "sentence2": wrong_ct,
                            "label": 0,
                            "sample_idx": sid,
                        }
                    )
                elif preset == "triplet":  # (a,p,n)
                    neg_sid = self.rng.choice(sample_ids)
                    rows.append(
                        {
                            "anchor": genes,
                            "positive": ct,
                            "negative": f"sample_idx:{neg_sid}",
                            "sample_idx": sid,
                        }
                    )
                else:  # single / single-class / pair
                    row = spec["build"](genes, ct, ct)
                    row["sample_idx"] = sid  # <─ always attach ID
                    rows.append(row)
        else:
            # Use sample IDs as primary data (original behavior)
            for sid, _genes, _cap, ct in zip(sample_ids, gene_sents, captions, cell_types, strict=False):
                if preset == "pair-binary":  # (anchor,pos/neg)+binary label
                    rows.append(
                        {
                            "sentence1": f"sample_idx:{sid}",
                            "sentence2": ct,
                            "label": 1,
                            "sample_idx": sid,  # <─ keep ID for splitting
                        }
                    )
                    wrong_ct = self.rng.choice(cell_types)
                    rows.append(
                        {
                            "sentence1": f"sample_idx:{sid}",
                            "sentence2": wrong_ct,
                            "label": 0,
                            "sample_idx": sid,
                        }
                    )
                elif preset == "triplet":  # (a,p,n)
                    neg_sid = self.rng.choice(sample_ids)
                    rows.append(
                        {
                            "anchor": f"sample_idx:{sid}",
                            "positive": ct,
                            "negative": f"sample_idx:{neg_sid}",
                            "sample_idx": sid,
                        }
                    )
                else:  # single / single-class / pair
                    row = spec["build"](f"sample_idx:{sid}", ct, ct)
                    row["sample_idx"] = sid  # <─ always attach ID
                    rows.append(row)

        ds_full = Dataset.from_list(rows)

        # ---------------- optional class-label encoding ----------------------
        if needs_cls and class_encode:
            ds_full = ds_full.class_encode_column("label")

        # ---------------- disjoint train / val split ------------------------
        unique_ids = np.array(sample_ids)
        self.rng.shuffle(unique_ids)

        n_val = int(len(unique_ids) * self.val_fraction)
        val_ids = set(unique_ids[:n_val])
        train_ids = set(unique_ids[n_val:])

        ds_train = ds_full.filter(lambda r: r["sample_idx"] in train_ids)
        ds_val = ds_full.filter(lambda r: r["sample_idx"] in val_ids)

        self.hf_dataset = DatasetDict(train=ds_train, val=ds_val)

    # -------------------------------------------------------------------
    # public getters
    # -------------------------------------------------------------------
    def get_dataframe(self) -> pd.DataFrame:
        """Return the numeric lookup table for the active level (gene or sample)."""
        if self.use_gene_level:
            return pd.DataFrame({"token": self.gene_embedding.index, "embedding": self.gene_embedding.values.tolist()})
        else:
            return pd.DataFrame(
                {"token": self.sample_embedding.index, "embedding": self.sample_embedding.values.tolist()}
            )

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
