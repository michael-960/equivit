import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple, Callable, List, Union, Literal

from dataclasses import dataclass

from ..geometry import Group
from .norm import EquivariantLayerNorm, ListLayerScale 
from .attn import EquivariantIrrepwiseAttention, EquivariantCoupledAttention
from .drop import ListDropout, ListDropPath 
from .mlp import EquivariantMLP


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
    def __init__(self, 
        group: Group,
        dims: List[int],
        num_heads: Union[int, List[int]],

        homogeneous_space_copies: List[int], # for nonlinearity in MLP

        attn_type: Literal['irrepwise', 'coupled'] = 'irrepwise', # only support 'irrepwise' for now
        trivial_rep_attn_bias: bool = True,
        attn_drop: float = 0.,
        trivial_rep_proj_bias: bool = True,
        proj_drop: float = 0.,

        trivial_rep_mlp_bias: bool = True,
        mlp_drop_probs: Tuple[float, float] = (0.,0.),

        ls_init_values=None,
        # norm_layer: Callable = None, 
        drop_path: float=0.
    ):
        super().__init__()

        self.dims = dims
        self.norm1 = EquivariantLayerNorm(dims)

        if attn_type == 'irrepwise':
            assert isinstance(num_heads, list), "num_heads should be a list of the same length as dims for irrepwise attention"
            self.attn = EquivariantIrrepwiseAttention(
                            group=group,
                            dims=dims,
                            num_heads=num_heads,
                            trivial_rep_attn_bias=trivial_rep_attn_bias,
                            trivial_rep_proj_bias=trivial_rep_proj_bias,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop
                        )
        elif attn_type == 'coupled':
            assert isinstance(num_heads, int), "num_heads should be an integer for coupled attention"
            self.attn = EquivariantCoupledAttention(
                            group=group,
                            dims=dims,
                            num_heads=num_heads,
                            trivial_rep_attn_bias=trivial_rep_attn_bias,
                            trivial_rep_proj_bias=trivial_rep_proj_bias,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop
                        )

        if ls_init_values is not None:
            self.ls1 = ListLayerScale(dims, init_values=ls_init_values)
        else:
            self.ls1 = nn.Identity()
        self.drop_path_1 = ListDropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = EquivariantLayerNorm(dims)

        self.mlp = EquivariantMLP(
            group,
            dims_in=dims,
            homogeneous_space_copies=homogeneous_space_copies,
            dims_out=dims,
            trivial_rep_bias=trivial_rep_mlp_bias,
            drop_probs=mlp_drop_probs,
            norm_layer=None
        )

        if ls_init_values is not None:
            self.ls2 = ListLayerScale(dims, init_values=ls_init_values)
        else:
            self.ls2 = nn.Identity()
        self.drop_path_2 = ListDropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.sample_drop_ratio = drop_path


    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape (*, L, Ci, di), where di is the dimension of the i-th irrep
        Returns: 
            list of tensors, each of shape (*, L, Ci, di)
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

