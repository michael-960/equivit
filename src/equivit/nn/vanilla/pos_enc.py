import torch
import torch.nn as nn
import math


class PositionEncoding(nn.Module):
    def __init__(self, 
        spatial_size: int, dim: int
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.pos_enc = nn.Parameter(torch.zeros(spatial_size**2, dim))

        self.reset_parameters()

    def reset_parameters(self):
        SQRT2_OVER_2 = math.sqrt(2)/2
        nn.init.trunc_normal_(self.pos_enc, std=SQRT2_OVER_2*0.2)

    def forward(self, x: torch.Tensor):
        r"""
        x: :math:`(*, L, C)`, where :math:`L = \text{spatial_size}^2`
        """
        return x + self.pos_enc
