import argparse

import torch

from model import create_model_wrapper

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ViT or Conformer classifier with MHA, MQA, or GQA on CIFAR10"
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="configs/model_config.yaml",
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--train-config-path",
        type=str,
        default="configs/train_config.yaml",
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the pt/pth checkpoint file",
    )

    args = parser.parse_args()
    model_config_path = args.model_config_path
    train_config_path = args.train_config_path
    ckpt_path = args.ckpt_path

    wrapper = create_model_wrapper(model_config_path, train_config_path, device)
    wrapper.evaluate(ckpt_path)
