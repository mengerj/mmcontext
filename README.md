# mmcontext

[![Tests][badge-tests]][tests]

Align embeddings across multiple modalities using context-aware embeddings at the sample level.

**mmcontext** is built upon the excellent [sentence-transformers](https://www.sbert.net/) framework maintained by [Hugging Face](https://huggingface.co/). By leveraging their comprehensive documentation and extensive capabilities for text embeddings, **mmcontext** enables you to efficiently generate multi-modal embeddings without reinventing the wheel.

![Conceptual Diagram](figs/concept.png)

## Paper

**mmcontext** is described in detail in our paper (citation to be added upon publication). If you use **mmcontext** in your research, please cite our work.

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

# Using uv (recommended)
uv pip install -e .

# Or using pip
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

For Figure 1E, we evaluate several models on multiple datasets, with the [`scripts/embed_eval.py`](scripts/combined_pipeline.py) script. This runs both inference and evaluation pipelines in sequence.

The combined pipeline is configured using [`embed_eval_conf.yaml`](conf/eval/combined_conf.yaml), which inherits from dataset and model configuration files that list the datasets and models to be evaluated. The configuration file contains additional parameters that are explained in the comments within the file itself.
The models and datasets evaluated in the paper are referenced in [model_list_cxg_geo_all.yaml](conf/models/model_list_cxg_geo_all.yaml) and [dataset_list.yaml](conf/datasets/dataset_list.yaml). These configs are imported in [embed_eval_conf.yaml](conf/eval/combined_conf.yaml). To jointly embed data and evaluate with CellWhisperer, set `run_cellwhisperer: true`. It is highly recommended to use CUDA for CellWhisperer. The mmcontext models also run in reasonable time on MPS or CPU.

For HPC systems, you can run the combined pipeline as array jobs using [`scripts/run_combined_cpu.slurm`](scripts/run_combined_cpu.slurm):

```bash
sbatch scripts/run_combined_cpu.slurm
```

This allows you to process multiple model configurations in parallel across different array job tasks, by spreading the models across several config files and passing them as a list to the array job.

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
