import torch.nn as nn
from collections import OrderedDict
from typing import Dict

def build_named_sequential(**layers) -> nn.Sequential:
    """
    Converts a standard dictionary into a PyTorch named Sequential block.
    This is highly useful for configuration frameworks like Hydra.
    """
    return nn.Sequential(OrderedDict(layers))
