# Denoising Diffusion Probabilistic Model (DDPM)

## Project Overview

This project is an implementation of a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch. The model is trained to generate images by reversing a gradual noising process. This implementation provides scripts for training the model on different datasets (e.g., MNIST, CIFAR-100) and for generating images using a trained model.

## Table of Contents

- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Application](#application)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Project Files](#project-files)

## Project Structure

```
/
├── checkpoints/
│   ├── CIFAR100/
│   └── MNIST/
├── config/
│   ├── cifar100_config.json
│   └── mnist_config.json
├── model/
│   ├── blocks/
│   ├── components/
│   ├── layers/
│   ├── models/
│   └── main.py
├── samples/
│   ├── CIFAR100/
│   │   └── README.md
│   └── MNIST/
│       └── README.md
├── utils/
│   ├── dataset.py
│   ├── hps.py
│   ├── loss.py
│   └── misc.py
├── .gitignore
├── infer.py
├── requirements_conda.txt
├── requirements_pip.txt
├── test.py
└── train.py
```

## Tech Stack

- Python 3
- PyTorch
- Transformers (for learning rate scheduler)

## Application

The application consists of two main functionalities:

1.  **Training**: The `train.py` script trains the DDPM model on a specified dataset. It handles the training loop, validation, saving checkpoints, and early stopping.
2.  **Inference**: The `infer.py` script uses a trained model to generate images. It loads a model checkpoint and performs the reverse diffusion process to create images from noise.

## Getting Started

### Prerequisites

- Python 3.8+
- Conda (for CUDA setup) or pip

### Installation

You can set up the project using either Conda (recommended for CUDA) or pip.

#### Using Conda (with CUDA support)

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/DDPM.git
    cd DDPM
    ```

2.  **Create and activate a conda environment from `requirements_conda.txt`:**
    ```bash
    conda env create -f requirements_conda.txt
    conda activate ddpm
    ```

#### Using pip

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Hachiman-potassiumdesu/Practice.git
    cd Practice/DDPM
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies from `requirements_pip.txt`:**
    ```bash
    pip install -r requirements_pip.txt
    ```

### Running the Application

After installation, you can train the model and run inference as follows:

#### Train the model:

To train the model on the MNIST dataset:

```bash
python train.py --data mnist
```

To train the model on the CIFAR-100 dataset:

```bash
python train.py --data cifar100
```

Checkpoints will be saved in the `checkpoints` directory.

#### Run inference:

Before running inference, make sure you have a trained model checkpoint. Update the `ckpt_path` in the corresponding config file (`config/mnist_config.json` or `config/cifar100_config.json`) to point to your trained model.

To generate images using the MNIST model:

```bash
python infer.py --data mnist
```

To generate images using the CIFAR-100 model:

```bash
python infer.py --data cifar100
```

Generated images will be saved in the `samples` directory.

## Project Files

- **`train.py`**: The main script for training the DDPM model.
- **`infer.py`**: The main script for generating images using a trained model.
- **`test.py`**: Script for testing the model (if applicable).
- **`config/*.json`**: Configuration files for different datasets, containing hyperparameters for the model, training, and inference.
- **`model/`**: This directory contains the source code for the DDPM model architecture.
  - **`main.py`**: The main model file that defines the DDPM.
  - **`blocks/`**, **`components/`**, **`layers/`**, **`models/`**: These subdirectories contain different modules of the model, such as encoder/decoder blocks, attention mechanisms, and positional embeddings.
- **`utils/`**: This directory contains utility scripts for various tasks.
  - **`dataset.py`**: Script for creating and loading datasets.
  - **`hps.py`**: Script for loading hyperparameters from config files.
  - **`loss.py`**: Defines the loss function used for training.
  - **`misc.py`**: Contains miscellaneous utility functions like early stopping and setting random seeds.
- **`checkpoints/`**: Directory where model checkpoints are saved during training.
- **`samples/`**: Directory where generated images are saved during inference.
- **`.gitignore`**: Specifies which files and directories to ignore in Git version control.
