import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable

from .norm import LayerNorm, LayerScale
from .attn import Attention
from .drop import DropPath
from .mlp import MLP

from dataclasses import dataclass

from ..._core import MISSING


@dataclass
class TransformerBlockConfig:
    dim: int = MISSING
    num_heads: int = MISSING

    dim_mlp: int = MISSING

    attn_bias: bool = True
    attn_drop: float = 0.
    proj_bias: bool = True
    proj_drop: float = 0.

    mlp_bias: bool = True
    mlp_drop_probs: Tuple[float, float] = (0.,0.)

    activation: str = 'relu'
    activation_kw: dict = None

    ls_init_values: Optional[float] = None
    # norm_layer: Callable = None, 
    drop_path: float = 0.

    def validate(self):
        assert None not in [self.dim, self.num_heads, self.dim_mlp]



class TransformerBlock(nn.Module):
    """
    Normal transformer block. No equivariance.
    """
    def __init__(self, config: TransformerBlockConfig):

        super().__init__()

        config.validate()

        self.norm1 = LayerNorm(config.dim)

        self.attn = Attention(
                        dim=config.dim,
                        num_heads=config.num_heads,
                        attn_bias=config.attn_bias,
                        proj_bias=config.proj_bias,
                        attn_drop=config.attn_drop,
                        proj_drop=config.proj_drop
                    )

        if config.ls_init_values is not None:
            self.ls1 = LayerScale(config.dim, init_values=config.ls_init_values)
        else:
            self.ls1 = nn.Identity()

        self.drop_path_1 = DropPath(config.drop_path) if config.drop_path > 0. else nn.Identity()

        self.norm2 = LayerNorm(config.dim)

        self.mlp = MLP(
            dim_in=config.dim, dim_hidden=config.dim_mlp, dim_out=config.dim,
            bias=config.mlp_bias,
            drop_probs=config.mlp_drop_probs,
            activation=config.activation,
            activation_kw=config.activation_kw
        )

        if config.ls_init_values is not None:
            self.ls2 = LayerScale(config.dim, init_values=config.ls_init_values)
        else:
            self.ls2 = nn.Identity()

        self.drop_path_2 = DropPath(config.drop_path) if config.drop_path > 0. else nn.Identity()

        self.sample_drop_ratio = config.drop_path


    def forward(self, x: torch.Tensor):
        if self.training and self.sample_drop_ratio > 0.:
            y = self.drop_path_1(self.ls1(self.attn(self.norm1(x))))
            y = y + x
            z = self.drop_path_2(self.ls2(self.mlp(self.norm2(y))))
            z = z + y
            return z

        else:
            y = self.ls1(self.attn(self.norm1(x)))
            y = y + x
            z = self.ls2(self.mlp(self.norm2(y)))
            z = z + y
            return z


