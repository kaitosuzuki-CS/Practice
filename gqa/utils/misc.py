import os

import numpy as np
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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
