import os
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

parent_dir = Path(__file__).resolve().parent.parent


class HPS:
    def __init__(self, hps):
        for key, value in hps.items():
            if isinstance(value, dict):
                setattr(self, key, HPS(value))
            else:
                setattr(self, key, value)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(config_path):
    config_path = os.path.join(parent_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return HPS(config)
