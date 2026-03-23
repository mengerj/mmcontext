# mmcontext

[![Tests][badge-tests]][tests]

Align embeddings across multiple modalities using context-aware embeddings at the sample level.

**mmcontext** is built upon the excellent [sentence-transformers](https://www.sbert.net/) framework maintained by [Hugging Face](https://huggingface.co/). By leveraging their comprehensive documentation and extensive capabilities for text embeddings, **mmcontext** enables you to efficiently generate multi-modal embeddings without reinventing the wheel.

![Conceptual Diagram](figs/concept.png)

## Paper

**mmcontext** is described in detail in our paper:
Menger, Jonatan, Sonia Maria Krissmer, Clemens Kreutz, Harald Binder, and Maren Hackenberg. "mmContext: an open framework for multimodal contrastive learning of omics and text data." bioRxiv (2025): 2025-12.
If you use **mmcontext** in your research, please cite our work.

## Overview

**mmcontext** provides tools for multi-modal embedding generation, with different workflows depending on your needs:

- **[Using Pre-trained Models for Inference](#using-pre-trained-models-for-inference)** - Load and use pre-trained mmcontext models
- **[Training New Models](#training-new-models)** - Train custom mmcontext models on your datasets
- **[Reproducing Paper Results](#reproducing-paper-results)** - Reproduce experiments from the paper

## Using Pre-trained Models for Inference

You can use pre-trained models (currently limited to RNA-seq) to embed your own data into the joint latent space of an mmcontext model. You can then query the dataset with natural language, for annotation etc.

### Installation

For inference, you only need to install the package:

```bash
pip install git+https://github.com/mengerj/mmcontext.git@main
```

### Pre-trained Models

Pre-trained mmcontext models are available on Hugging Face under the [jo-mengr](https://huggingface.co/jo-mengr) organization. Browse available models at: **https://huggingface.co/jo-mengr**

### Tutorial

See the [Using a Pre-trained Model for Inference](tutorials/pretrained_inference.ipynb) tutorial for a complete guide on:

- Loading pre-trained models from Hugging Face
- Processing your adata to be suitable
- Generating embeddings for text and omics data
- Using models for downstream tasks

## Training New Models

Train your own mmcontext models on custom datasets.

### Tutorial

See the [Training a New Model](tutorials/train_new.ipynb) tutorial for:

- Step-by-step training guide
- Configuration and hyperparameters
- Best practices for multi-modal training

### Dataset Preparation

If you want to utilize the full pipeline used to process the training datasets, go to
[adata-hf-datasets](https://github.com/mengerj/adata_hf_datasets).
This package handles the conversion of AnnData objects to Hugging Face datasets with proper formatting for multimodal training.

## Reproducing Paper Results

To reproduce results from the paper, clone the repository and install in editable mode:

```bash
git clone https://github.com/mengerj/mmcontext.git
cd mmcontext

# Create a virtual environment (however you like)
# eg. with venv
python -m venv mmcontext
source mmcontext/bin/activate

# And install the package with pip
pip install -e .
```

### Training Models

To train a model as done for the paper, use the [`scripts/train.py`](scripts/train.py) script.
Try a small training run with

```bash
python scripts/train.py --config-name example_conf
```

Inspect the config at [`example_conf.yaml`](conf/training/example_conf.yaml). The configs to train the models presented in the paper, are [`basebert_numeric`](conf/training/basebert_numeric.yaml), for all models using numeric initial representations and [`basebert_text.yaml`](conf/training/basebert_text.yaml) for the model using cell_sentences (text only). All datasets used in training are hosted publically on the huggingface hub (with references to zenodo), therefore the training scripts can be launched without manually downloading any data.

For HPC systems with CUDA support, the recommended approach is to use the [`scripts/run_training.slurm`](scripts/run_training.slurm) SLURM script to launch training jobs. Training also works on CPU or MPS devices if CUDA is not available.

The SLURM script allows you to override configuration values from the command line, which is useful when launching several jobs with different configurations.

```bash
sbatch scripts/run_training.slurm
```

Before training, it is recommended to authenticate with Hugging Face (after activating the virtual environment used for mmcontext installation):

```bash
source .venv/bin/activate  # or your venv activation command
hf auth login
```

If you have a Weights & Biases (wandb) account, you can also log in so your training is tracked. From the command line, run:

```bash
wandb login
```

This will prompt you to enter your wandb API key.

Once training is complete, the finished models will be automatically uploaded to the Hugging Face Hub with metadata and model cards.

### Evaluating Models

Figure 1D of the paper investigates the latent space of one model in detail. This can be recreated with the [`evaluate_model.ipynb`](tutorials/evaluate_model.ipynb) notebook.

For Figure 1E, we evaluate several models on multiple datasets, with the [`scripts/embed_eval.py`](scripts/embed_eval.py) script. This runs both inference and evaluation pipelines in sequence.

The combined pipeline is configured using [`embed_eval_conf.yaml`](conf/eval/embed_eval_conf.yaml), which inherits from dataset and model configuration files that list the datasets and models to be evaluated. The configuration file contains additional parameters that are explained in the comments within the file itself.
The models and datasets evaluated in the paper are referenced in [model_list_cxg_geo_all.yaml](conf/models/model_list_cxg_geo_all.yaml) and [dataset_list.yaml](conf/datasets/dataset_list.yaml). These configs are imported in [embed_eval_conf.yaml](conf/eval/embed_eval_conf.yaml). To jointly embed data and evaluate with CellWhisperer, set `run_cellwhisperer: true`. It is highly recommended to use CUDA for CellWhisperer. The mmcontext models also run in reasonable time on MPS or CPU.

Run it locally with

```bash
python scripts/embed_eval.py
```

For HPC systems, you can run the combined pipeline as array jobs using [`scripts/run_combined_cpu.slurm`](scripts/run_embed_eval_cpu.slurm):

```bash
sbatch scripts/run_embed_eval_cpu.slurm
```

This allows you to process multiple model configurations in parallel across different array job tasks, by spreading the models across several config files and passing them as a list to the array job.

### SCSA Baseline

[SCSA](https://github.com/bioinfo-ibms-pumc/SCSA) (Single-Cell RNA-seq Annotation) is included as a non-embedding baseline. Unlike the mmcontext models, SCSA annotates cell types by clustering cells, detecting marker genes per cluster, and matching them against the CellMarker database using Z-scores. It runs as a standalone script on the same datasets.

SCSA requires `numpy<2.0` and therefore needs its own virtual environment:

```bash
# 1. Clone the SCSA repository
bash modules/prepare_scsa.sh

# 2. Create a dedicated venv for SCSA and install its dependencies
python -m venv modules/scsa_venv
modules/scsa_venv/bin/pip install -r modules/requirements_scsa.txt

# 3. (Optional) Create a dedicated venv for calmate label harmonisation
#    This maps SCSA's predicted labels and ground-truth labels to Cell Ontology
#    terms so that metrics are computed on normalised names.
python -m venv modules/calmate_venv
modules/calmate_venv/bin/pip install -r modules/requirements_calmate.txt

# 4. Run SCSA on all datasets (venvs are auto-detected from modules/)
python scripts/run_scsa.py

# 5. (Optional) After reviewing calmate mappings, you can recompute\n#    metrics without re-running SCSA itself by using:\n+#    python scripts/run_scsa.py scsa.remap_only=true settings.skip_existing=false
```

When calmate is available, the pipeline runs label harmonisation automatically. On the first run, many labels will need manual review. The script prints instructions at the end — run `modules/calmate_venv/bin/calmate --store modules/.calmate/mappings.csv review` to interactively approve the suggested ontology mappings, then re-run with `settings.skip_existing=false`. If you only want to redo the mapping/metrics step and reuse existing SCSA predictions, set `scsa.remap_only=true`; in that mode `settings.skip_existing` is ignored for SCSA and only controls whether other datasets are skipped.

SCSA is configured via [`scsa_conf.yaml`](conf/eval/scsa_conf.yaml), which shares the same dataset list as the embedding pipeline. Key parameters (fold-change threshold, p-value, clustering resolution, species) can be overridden from the command line. If the SCSA venv is in a non-default location, pass it explicitly: `python scripts/run_scsa.py scsa.scsa_python=/path/to/scsa_venv/bin/python`. Results are written to the same output directory tree as the other models, so `collect_metrics.py` picks them up automatically.

For HPC systems:

```bash
sbatch scripts/run_scsa.slurm
```

Figure Panel 1E is created by collecting the metrics from the big evaluation run. The config [`collect_metrics_conf.yaml`](conf/eval/collect_metrics_conf.yaml) has to point to the directory where the results from all datasets/models were stored. Then to collect and plot the metrics, run

```bash
python scripts/collect_metrics.py
python scripts/plot_metrics.py
```

## Core Components

The main model implementation is the **MMContextEncoder**, located in [`src/mmcontext/mmcontextencoder.py`](src/mmcontext/mmcontextencoder.py). This dual-tower encoder can process both text and omics data, enabling multi-modal embedding generation.

## Contributing

This package is under active development. Contributions and suggestions are very welcome! Please open an [issue](https://github.com/mengerj/mmcontext/issues) to:

- Propose enhancements
- Report bugs
- Discuss potential improvements
- Ask questions or seek help

We encourage community contributions and are happy to help you get started.

## Citation

If you find **mmcontext** useful for your research, please consider citing our paper (citation to be added upon publication):

```bibtex
@misc{mmcontext,
  author = {Jonatan Menger},
  title = {mmcontext: Multi-modal Contextual Embeddings},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  url = {https://github.com/mengerj/mmcontext}
}
```

## Contact

Encountered a bug or need help? Please use the [issue tracker](https://github.com/mengerj/mmcontext/issues). We are happy for any feedback.

---

[badge-tests]: https://img.shields.io/github/actions/workflow/status/mengerj/mmcontext/test.yaml?branch=main
[tests]: https://github.com/mengerj/mmcontext/actions/workflows/test.yaml
