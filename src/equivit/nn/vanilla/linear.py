import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Tuple



class Linear(nn.Module):
    """
    This should be identical to nn.Linear.
    """
    def __init__(self, dim_in: int, dim_out: int, bias: bool=True):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(dim_out, dim_in))

        self._bias = bias

        if bias:
            self.bias = nn.Parameter(torch.zeros(dim_out))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # imitates source code of torch.nn.Linear
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        if self._bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: a tensor of shape :math:`(*, C_i)`
        
        Returns:
            a tensor of shape :math:`(*, C_o)`
        """

        y = x @ self.weights.T
        if self._bias:
            y += self.bias
        return y

