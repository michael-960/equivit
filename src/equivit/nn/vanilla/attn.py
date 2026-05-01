import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple

from .linear import Linear



class Attention(nn.Module):
    """
    Normal attention block. No equivariance.
    """
    def __init__(self,
        dim: int,
        num_heads: int,
        attn_bias: bool = True,
        attn_drop: float = 0.,
        proj_bias: bool = True,
        proj_drop: float = 0.
    ):
        super().__init__()
        assert dim % num_heads == 0, 'num_heads must divide dim'

        self.qkv = Linear(dim, dim*3, bias=attn_bias)
        self.attn_drop = attn_drop

        self.proj = Linear(dim, dim, bias=proj_bias)

        self.proj_drop = nn.Dropout(p=proj_drop)

        self.num_heads = num_heads

        self.dim_attn = dim // self.num_heads


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: a tensor of shape (*, L, C)
        Returns:
            a tensor of shape (*, L, C)
        """
        # slow implementation for clarity, must optimize later
        # (..., L, C)
        shape = x.shape

        # (*, L, 3*C)
        qkv = self.qkv(x)

        # (*, L, 3, num_heads, dim_attn)
        qkv = qkv.unflatten(-1, (3, self.num_heads, self.dim_attn))

        # (*, L, num_heads, dim_attn) 
        q = qkv[..., 0,:,:]
        k = qkv[..., 1,:,:]
        v = qkv[..., 2,:,:]

        y = F.scaled_dot_product_attention(
                        q.movedim(-3,-2), 
                        k.movedim(-3,-2), 
                        v.movedim(-3,-2),
                        dropout_p=self.attn_drop
                    ).movedim(-2, -3).reshape(*shape) # (*, L, C)

        y = self.proj(y)
        y = self.proj_drop(y)
        return y


