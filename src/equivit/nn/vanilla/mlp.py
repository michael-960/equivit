import torch
import torch.nn as nn
from .linear import Linear
from typing import Tuple


class MLP(nn.Module):
    """
    """
    def __init__(self,
        dim_in: int, 
        dim_hidden: int, 
        dim_out: int,
        bias: bool = True,
        drop_probs: Tuple[float] = (0.,0.),
        norm_layer = None
    ):
        super().__init__()
        self.fc1 = Linear(dim_in, dim_hidden, bias=bias)

        self.act = nn.GELU()

        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = nn.Identity()

        # self.norm = norm_layer(C_A1_hidden, C_A2_hidden, C_E_hidden) if (norm_layer is not None) else nn.Identity()
        self.norm = norm_layer(dim_hidden) if (norm_layer is not None) else nn.Identity()

        self.fc2 = Linear(dim_hidden, dim_out, bias=bias)

        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: a tensor of shape :math:`(*, C_i)`
        
        Returns:
            a tensor of shape :math:`(*, C_o)`
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

