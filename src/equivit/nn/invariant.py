import torch
import torch.nn as nn
from typing import List



class Invariantization(nn.Module):
    r"""
    This layer takes a list of tensors, each of shape 
    :math:`(*, C_i, d_i)`, and returns a single tensor of shape 
    :math:`(*, \sum_i C_i)` by taking the norm over the last dimension of each tensor and concatenating the results.

    The first tensor (trivial irrep) is left unchanged.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        r"""
        Args:
            x: list of tensors, each of shape :math:`(*, C_i, d_i)`
        Returns:
            Tensor of shape :math:`(*, \sum_i C_i)`
        """
        return torch.cat([y.norm(dim=-1) if i > 0 else y
                          for i,y in enumerate(x)], dim=-1)
