# Vision Transformers & Conformers with Grouped-Query Attention (GQA) on CIFAR-10

A comprehensive PyTorch-based benchmark comparing **Vision Transformers (ViT)** and **Conformer (Convolution-augmented Transformer)** classifiers on the **CIFAR-10** image dataset using alternative self-attention mechanisms: **Multi-Head Attention (MHA)**, **Multi-Query Attention (MQA)**, and **Grouped-Query Attention (GQA)**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Cloning the Repository](#cloning-the-repository)
  - [Environment Setup](#environment-setup)
    - [Option A: Conda Setup (Recommended)](#option-a-conda-setup-recommended)
    - [Option B: standard Virtualenv (venv) & pip](#option-b-standard-virtualenv-venv--pip)
  - [Running the Python Scripts](#running-the-python-scripts)
    - [Training](#training)
    - [Evaluation & Inference](#evaluation--inference)
- [Results & Benchmarks](#results--benchmarks)
  - [Quantitative Results (CIFAR-10)](#quantitative-results-cifar-10)
  - [Observations & Key Takeaways](#observations--key-takeaways)
  - [Visualization Plots](#visualization-plots)
- [Project Files Description](#project-files-description)

---

## Project Overview

In vanilla Transformer architectures, standard **Multi-Head Attention (MHA)** allocates unique key (K) and value (V) projections for every query (Q) head. While highly expressive, MHA incurs a significant computational footprint and substantial memory bandwidth overhead—particularly during auto-regressive generation or inference due to the key-value (KV) cache sizing.

To optimize the attention layer, this project provides a from-scratch PyTorch implementation of:

1. **Multi-Query Attention (MQA)**: All query heads share a single key-value projection, reducing parameters and saving memory bandwidth at a minor cost to model accuracy.
2. **Grouped-Query Attention (GQA)**: A generalization of MHA and MQA where query heads are split into $G$ groups, and each group shares a single key-value projection.

These self-attention schemas are evaluated in two image-classification paradigms:

- **Vision Transformer (ViT)**: A classic transformer applying patch projection followed by layers of self-attention and Multi-Layer Perceptrons (MLPs).
- **Conformer**: A macaron-style architecture that interleaves self-attention with depthwise and pointwise convolution blocks, capitalizing on both global attention and local translation-invariant convolutional inductive biases.

---

## Project Structure

```
gqa/
├── .gitignore              # Files and folders ignored by git
├── environment.yml         # Conda environment definition with dependencies
├── requirements.txt        # Pip package dependency specification
├── README.md               # Main repository documentation (this file)
├── configs/                # Configuration profiles for different models and training runs
│   ├── conformer/          # Configurations for Conformer runs
│   │   ├── model_config.yaml # Conformer model parameters
│   │   └── train_config.yaml # Conformer training hyperparameters
│   └── vit/         # Configurations for ViT runs
│       ├── model_config.yaml # ViT model parameters
│       └── train_config.yaml # ViT training hyperparameters
├── model/                  # Deep learning model library
│   ├── __init__.py         # Imports and registers classifiers
│   ├── model/              # Neural network configurations
│   │   ├── __init__.py     # Registry of models
│   │   ├── conformer.py    # ConformerClassifier wrapping ConformerBlock and MLP head
│   │   ├── vit.py   # ViTClassifier wrapping ViTBlock and MLP head
│   │   ├── blocks/         # High-level architecture layers
│   │   │   ├── __init__.py
│   │   │   ├── conformer.py# Macaron-style layers combining multi-FFNs, self-attention, and convolutions
│   │   │   └── vit.py      # Standard ViT blocks combining self-attention and feedforward layers
│   │   └── components/     # Low-level network primitives
│   │       ├── __init__.py
│   │       ├── attention.py# Einstein-sum-based GroupQueryAttention (GQA, MHA, MQA)
│   │       ├── conv.py     # Depthwise, pointwise, and gated convolutions for Conformer
│   │       └── feedforward.py # Position-wise Feed Forward Network (FFN)
│   └── wrappers/           # Engine and wrapper files
│       ├── __init__.py
│       └── wrapper.py      # ModelWrapper encapsulating training loops, evaluation, and scheduler logic
├── results/                # Metrics and plots directory
│   ├── results.pkl         # Serialized pickle file of empirical run benchmarks
│   └── samples/            # Pre-compiled evaluation plots comparing performance characteristics
│       ├── pdf/            # Vectorized PDF versions of metric plots
│       └── png/            # High-resolution PNG versions of metric plots
├── scripts/                # Execution entry points
│   ├── evaluate.py         # Loads checkpoint and evaluates model accuracy and speed
│   └── train.py            # Executes model training on CIFAR-10 via config parameters
├── test/                   # Benchmark utilities and evaluation plotting scripts (git-ignored)
│   ├── convert_to_pdf.py   # Utility to convert output plots to PDF
│   └── results.py          # Results database and script used to generate plots
└── utils/                  # Core utility library
    ├── __init__.py         # Entrypoint registering utility components
    ├── dataset.py          # Handles CIFAR-10 downloading and PyTorch DataLoader preparation
    ├── early_stopping.py   # Monitored state early-stopping implementation to prevent overfitting
    └── misc.py             # Random seeding, HPS config object wrapper, and YAML loading helpers
```

---

## Tech Stack

The workspace relies on a modern, robust, and highly-performant Python and deep learning tech stack:

- **Language**: Python 3.14+
- **Deep Learning Framework**: PyTorch 2.12.0+ (utilizes custom written layers and tensor operations)
- **Computer Vision**: Torchvision 0.27.0+ (CIFAR-10 loaders and transforms)
- **Learning Rate Scheduler**: Transformers 5.10.2+ (Cosine Annealing with Warmup scheduler)
- **Mathematical Processing**: NumPy 2.4.6+, SymPy 1.14.0+
- **Visualizations & Plotting**: Matplotlib 3.10.9+, Pillow 12.2.0+, pikepdf 10.9.1+
- **Configuration & CLI Utilities**: PyYAML 6.0.3+, argparse, tqdm, Rich 15.0.0+

---

## Getting Started

### Prerequisites

Make sure you have [Git](https://git-scm.com/) and either [Conda / Miniconda](https://docs.conda.io/) or Python 3.14+ installed on your machine.

### Cloning the Repository

```bash
git clone --filter=blob:none --sparse https://github.com/kaitosuzuki-CS/practice.git
cd practice
git sparse-checkout set gqa
cd gqa
```

### Environment Setup

#### Option A: Conda Setup (Recommended)

This approach automatically installs the correct Python version and pip dependencies inside an isolated conda environment:

```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the conda environment
conda activate gqa
```

#### Option B: standard Virtualenv (venv) & pip

Alternatively, set up a standard Python virtual environment and install the package list via pip:

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

---

## Running the Python Scripts

Both training and evaluation are conducted via modular scripts executed from the project root.

### Training

To train either model configuration, run the `scripts.train` entrypoint. You must supply a model config and a training config YAML file. Pre-configured files for both **Conformer** and **ViT** models are provided under the `configs/` directory.

To train the **Conformer** model:

```bash
python -m scripts.train \
  --model-config-path configs/conformer/model_config.yaml \
  --train-config-path configs/conformer/train_config.yaml
```

To train the **ViT** model:

```bash
python -m scripts.train \
  --model-config-path configs/vit/model_config.yaml \
  --train-config-path configs/vit/train_config.yaml
```

The model config file specifies:

- `mode`: `"conformer"` or `"vit"`
- `num_head_groups`: set to `1` for **MQA**, equal to `num_heads` for **MHA**, or intermediate divisors (e.g., `2`, `4`, `8`) for **GQA**.
- Architectural dimension values (`embed_dim`, `hidden_dim`, `num_layers`, `num_heads`, etc.)

### Evaluation & Inference

To load a saved model checkpoint and measure its validation classification accuracy and execution runtimes (eval latency), use the `scripts.evaluate` entrypoint:

```bash
python -m scripts.evaluate \
  --model-config-path configs/conformer/model_config.yaml \
  --train-config-path configs/conformer/train_config.yaml \
  --ckpt-path checkpoints/conformer/checkpoint_40.pt
```

This script automatically prints accuracy achievements and average processing speeds across both the training and validation splits.

---

## Results & Benchmarks

The architectures were profiled on CIFAR-10 with standard image parameters ($32 \times 32$ pixels, 3 channels) across the attention strategies.

### Quantitative Results (CIFAR-10)

#### 1. Vision Transformer (ViT)

_Base Hyperparameters: 12 Attention Heads, Embedding Dim: 192, 4 Layers, Hidden Dim: 384._

| Attention Scheme               | Parameter Count | Validation Accuracy | Latency (Runtime s) |
| :----------------------------- | :-------------: | :-----------------: | :-----------------: |
| **MHA** (12 heads, 12 groups)  |    1,454,858    |       60.67%        |        6.11s        |
| **GQA-8** (12 heads, 8 groups) |    1,356,554    |       59.59%        |        6.06s        |
| **GQA-4** (12 heads, 4 groups) |    1,307,402    |       59.94%        |        6.05s        |
| **GQA-2** (12 heads, 2 groups) |    1,282,826    |       58.85%        |        6.00s        |
| **MQA** (12 heads, 1 group)    |    1,270,538    |       58.80%        |      **5.81s**      |

#### 2. Conformer Classifier

_Base Hyperparameters: 12 Attention Heads, Embedding Dim: 192, 4 Layers, Hidden Dim: 384, Convolution Groups: 16._

| Attention Scheme               | Parameter Count | Validation Accuracy | Latency (Runtime s) |
| :----------------------------- | :-------------: | :-----------------: | :-----------------: |
| **MHA** (12 heads, 12 groups)  |    4,536,842    |     **77.75%**      |       10.12s        |
| **GQA-8** (12 heads, 8 groups) |    4,405,770    |       77.43%        |        9.97s        |
| **GQA-4** (12 heads, 4 groups) |    4,340,234    |       77.48%        |        9.94s        |
| **GQA-2** (12 heads, 2 groups) |    4,307,466    |       76.95%        |        9.92s        |
| **MQA** (12 heads, 1 group)    |  **4,291,082**  |       77.69%        |      **9.61s**      |

### Observations & Key Takeaways

1. **Conformer Performance Superiority**: Conformer models outperform the ViT models by an average of **+17% absolute accuracy**. Incorporating translation-invariant depthwise and pointwise convolution steps adds powerful local spatial features, allowing the networks to converge faster and achieve significantly higher accuracy on image datasets like CIFAR-10.
2. **Parameter and Computation Savings**:
   - Sharing Key-Value projection parameters via **MQA** reduces the ViT parameter count by **12.7%** and Conformer parameter count by **5.4%**.
   - These parameters savings directly translate to faster runtimes: MQA provides a **~5% speedup** in overall epoch-processing latency.
3. **Optimizing GQA (The Pareto Frontier)**:
   - **Grouped-Query Attention** operates as a tunable slider between performance (MHA) and efficiency (MQA).
   - In the Conformer experiments, **GQA-4** yields **77.48% accuracy** (retaining 99.65% of MHA performance) while cutting nearly 200,000 parameters and lowering processing latency.
   - For CIFAR-10, **MQA** acts as an exceptionally competitive configuration inside the Conformer block, achieving **77.69% accuracy** (99.92% of MHA) with the lowest possible weight footprints and latencies.

### Visualization Plots

Pre-generated evaluation curves are organized inside the `results/samples/` directory:

- **`vit_loss_curve.png` / `conformer_loss_curve.png`**: Multi-line plots detailing training loss convergence over 40 epochs.
- **`vit_accuracy.png` / `conformer_accuracy.png`**: Evaluation accuracies contrasted across MHA, MQA, and GQA settings.
- **`vit_parameter_count.png` / `conformer_parameter_count.png`**: Bar chart layouts illustrating parameter savings when sharing attention keys and values.
- **`*_parameter_count_vs_accuracy.png` / `*_parameter_count_vs_time.png`**: Compares the Pareto trade-offs of efficiency versus computational throughput.

---

## Project Files Description

### Configuration Profiles (`configs/`)

- **`configs/conformer/`**:
  - `model_config.yaml`: Configuration parameters for the Conformer model (embedding dim 128, 8 layers, 16 heads, 16 head groups, 32 conv groups).
  - `train_config.yaml`: Optimizer, learning rate, checkpoints, and data configurations for Conformer training (batch size 32, 40 epochs).
- **`configs/vit/`**:
  - `model_config.yaml`: Configuration parameters for the ViT model (embedding dim 128, 6 layers, 16 heads, 16 head groups).
  - `train_config.yaml`: Optimizer, learning rate, checkpoints, and data configurations for ViT training (batch size 32, 40 epochs).

### Execution entry points

- **`scripts/train.py`**: Reads specified model config and training parameters, initializes datasets, creates model wrappers, and runs the training loop.
- **`scripts/evaluate.py`**: Evaluates checkpoint model weights on CIFAR-10, reporting correct labels, overall accuracy, and runtime durations for the splits.

### Model Architecture Components (`model/model/`)

- **`vit.py` & `conformer.py`**: Classifier heads that manage image ingestion. `vit` uses standard Transformer patches and projections, whereas `conformer` integrates initial convolutive downsampling blocks (`in_conv`) before proceeding to the attention layers.
- **`blocks/vit.py`**: Combines LayerNorm, GroupQueryAttention, and Position-wise Feed Forward (GELU activated) into standard Transformer blocks.
- **`blocks/conformer.py`**: Implements the authentic Conformer block layout: a macaron-style wrapper containing a half-step `FFN1`, a global GroupQueryAttention block, a multi-stage `ConvBlock` layer, and a closing half-step `FFN2`.
- **`components/attention.py`**: Houses the core `GroupQueryAttention` layer. Written using custom weights and Einstein summation (`torch.einsum`) for clear multi-dimensional tensor contractions. Standard subclasses `MultiHeadAttention` and `MultiQueryAttention` inherit from GQA.
- **`components/conv.py`**: Gated activation 2D convolution. Consists of $1\times1$ Pointwise Conv expanding channels, a Gated Linear Unit (GLU) activation split, $3\times3$ Depthwise Conv with GroupNorm layers, and a $1\times1$ projection back.
- **`components/feedforward.py`**: Configurable Multi-Layer Perceptron used inside blocks.

### Core Utilities (`utils/`)

- **`dataset.py`**: Downloads and prepares the CIFAR-10 dataset using standard augmentation and normalization transforms.
- **`early_stopping.py`**: Tracks validation losses and automatically triggers early-stopping behavior, restoring the best epoch state when metrics plateau.
- **`misc.py`**: Config parsing (`HPS` mapping), manual seeding, and custom error validation.
