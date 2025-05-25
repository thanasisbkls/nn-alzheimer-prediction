"""
Random utilities for reproducibility

This module is used to set the random seed for reproducibility across all libraries.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries
    
    Args:
        seed: Random seed value
    """
    # Python's built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 