"""Seed management utilities for reproducibility."""
import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42

def set_all_seeds(seed: int = DEFAULT_SEED) -> None:
    """
    Set all seeds to make results reproducible.

    Args:
        seed: The random seed to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
