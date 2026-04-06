import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple, Callable, List

from ..geometry import Group
from .norm import EquivariantLayerNorm, ListLayerScale 
from .attn import EquivariantAttention
from .drop import ListDropout, ListDropPath 
from .mlp import EquivariantMLP


class EquivariantTranformerBlock(nn.Module):
    """
    LayerNorm -> Attention -> LayerScale -> DropPath
    -> LayerNorm -> MLP -> LayerScale -> DropPath

    Everything is equivariant.
    """
    def __init__(self, 
        group: Group,
        dims: List[int],
        num_heads: int,
        homogeneous_space_copies: List[int],

        trivial_rep_attn_bias: bool = True,
        attn_drop: float = 0.,
        trivial_rep_proj_bias: bool = True,
        proj_drop: float = 0.,

        trivial_rep_mlp_bias: bool = True,
        mlp_drop_probs: Tuple[float] = (0,0),

        ls_init_values=None,
        # norm_layer: Callable = HexLayerNorm, 
        drop_path: float=0.
    ):
        super().__init__()

        self.dims = dims
        self.norm1 = EquivariantLayerNorm(dims)

        self.attn = EquivariantAttention(
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
        x: list of tensors, each of shape (*, L, Ci, di), where di is the dimension of the i-th irrep
        return: list of tensors, each of shape (*, L, Ci, di)
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

