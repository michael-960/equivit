import torch
import torch.nn as nn
from typing import List



class Invariantization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: list of tensors, each of shape (*, C_i, d_i)
        Returns:
            Tensor of shape (*, sum_i C_i)
        """
        return torch.cat([y.norm(dim=-1) for y in x], dim=-1)