import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple, List, Union, Literal, Optional, cast

from dataclasses import dataclass

from ..geometry import Group
from .norm import EquivariantLayerNorm, ListLayerScale 
from .attn import EquivariantIrrepwiseAttention, EquivariantCoupledAttention
from .drop import ListDropout, ListDropPath 
from .mlp import EquivariantMLP


from .._core import MISSING


@dataclass
class EquivariantTransformerBlockConfig:
    group: Group = MISSING

    dims: List[int] = MISSING

    num_heads: Union[int, List[int]] = MISSING

    homogeneous_space_copies: List[int] = MISSING

    attn_type: Literal['irrepwise', 'coupled'] = 'irrepwise'
    trivial_rep_attn_bias: bool = True
    attn_drop: float = 0.
    trivial_rep_proj_bias: bool = True
    proj_drop: float = 0.

    activation: str = 'relu'
    activation_kw: Optional[dict] = None

    trivial_rep_mlp_bias: bool = True
    mlp_drop_probs: Tuple[float, float] = (0.,0.)

    ls_init_values: Optional[float] = None
    # norm_layer: Callable = None, 
    drop_path: float=0.

    def validate(self):
        assert MISSING not in [self.group, self.dims, self.num_heads, self.homogeneous_space_copies]




class EquivariantTransformerBlock(nn.Module):
    """
    The layer applies the following operations in sequence:

    - :class:`EquivariantLayerNorm` 
    - :class:`EquivariantIrrepwiseAttention`  or :class:`EquivariantCoupledAttention` (depending on the value of ``attn_type``)
    - :class:`ListLayerScale` 
    - :class:`ListDropPath`
    - :class:`LayerNorm` 
    - :class:`EquivariantMLP` 
    - :class:`ListLayerScale` 
    - :class:`ListDropPath`

    Everything is equivariant.
    """
    def __init__(self, config: EquivariantTransformerBlockConfig):
        super().__init__()
        config.validate()

        self.dims = config.dims
        self.norm1 = EquivariantLayerNorm(config.dims)

        if config.attn_type == 'irrepwise':
            # assert isinstance(config.num_heads, list), "num_heads should be a list of the same length as dims for irrepwise attention"
            self.attn = EquivariantIrrepwiseAttention(
                            group=config.group,
                            dims=config.dims,
                            num_heads=config.num_heads,
                            trivial_rep_attn_bias=config.trivial_rep_attn_bias,
                            trivial_rep_proj_bias=config.trivial_rep_proj_bias,
                            attn_drop=config.attn_drop,
                            proj_drop=config.proj_drop
                        )
        elif config.attn_type == 'coupled':
            # assert isinstance(config.num_heads, int), "num_heads should be an integer for coupled attention"
            self.attn = EquivariantCoupledAttention(
                            group=config.group,
                            dims=config.dims,
                            num_heads=config.num_heads,
                            trivial_rep_attn_bias=config.trivial_rep_attn_bias,
                            trivial_rep_proj_bias=config.trivial_rep_proj_bias,
                            attn_drop=config.attn_drop,
                            proj_drop=config.proj_drop
                        )

        if config.ls_init_values is not None:
            self.ls1 = ListLayerScale(config.dims, init_values=config.ls_init_values)
        else:
            self.ls1 = nn.Identity()

        self.drop_path_1 = ListDropPath(config.drop_path) if config.drop_path > 0. else nn.Identity()

        self.norm2 = EquivariantLayerNorm(config.dims)

        self.mlp = EquivariantMLP(
            config.group,
            dims_in=config.dims,
            homogeneous_space_copies=config.homogeneous_space_copies,
            dims_out=config.dims,
            trivial_rep_bias=config.trivial_rep_mlp_bias,
            drop_probs=config.mlp_drop_probs,
            norm_layer=None,
            activation=config.activation,
            activation_kw=config.activation_kw
        )

        if config.ls_init_values is not None:
            self.ls2 = ListLayerScale(config.dims, init_values=config.ls_init_values)
        else:
            self.ls2 = nn.Identity()

        self.drop_path_2 = ListDropPath(config.drop_path) if config.drop_path > 0. else nn.Identity()

        self.sample_drop_ratio = config.drop_path

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        r"""
        Args:
            x: list of tensors, each of shape :math:`(*, L, C_i, d_i)`, where :math:`d_i` is the complex dimension of the :math:`i`-th irrep
        Returns: 
            list of tensors, each of shape :math:`(*, L, C_i, d_i)`
        """
        if self.training and self.sample_drop_ratio > 0.:
            y = self.drop_path_1(self.ls1(self.attn(self.norm1(x))))
            y = [u+v for u,v in zip(y,x)]
            z = self.drop_path_2(self.ls2(self.mlp(self.norm2(y))))
            z = [u+v for u,v in zip(z,y)]
            return z

        else:
            y = self.ls1(self.attn(self.norm1(x)))
            y = [u+v for u,v in zip(y,x)]
            z = self.ls2(self.mlp(self.norm2(y)))
            z = [u+v for u,v in zip(z,y)]
            return z

