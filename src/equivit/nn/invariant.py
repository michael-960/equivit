import torch
import torch.nn as nn
from typing import List



class Invariantization(nn.Module):
    """
    This layer takes a list of tensors, each of shape 
    :math:`(*, C_i, d_i)`, and returns a single tensor of shape 
    :math:`(*, \sum_i C_i)` by taking the norm over the last dimension of each tensor and concatenating the results.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: list of tensors, each of shape :math:`(*, C_i, d_i)`
        Returns:
            Tensor of shape :math:`(*, \sum_i C_i)`
        """
        return torch.cat([y.norm(dim=-1) for y in x], dim=-1)