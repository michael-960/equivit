import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from .linear import EquivariantLinear
from .drop import ListDropout


# TODO: streams?
# Let's just use a simple for loop and hope that torch.compile can optimize it.


class EquivariantAttention(nn.Module):
    def __init__(self,
        dims: List[int],
        num_heads: int,
        trivial_rep_attn_bias: bool = True,
        attn_drop: float = 0.,
        trivial_rep_proj_bias: bool = True,
        proj_drop: float = 0.
    ):
        """
        This has little to do with group theory. 
        It's just a multi-head attention layer that processes each irrep separately.
        """
        super().__init__()
        for c in dims: assert c % num_heads == 0, 'num_heads must divide all input channels'

        self.dims = dims
        self.num_heads = num_heads
        self.attn_dims = [c // num_heads for c in dims]

        self.qkv = EquivariantLinear(
            dims_in=dims,
            dims_out=[c*3 for c in dims],
            trivial_rep_bias=trivial_rep_attn_bias
        )

        self.attn_drop = attn_drop

        self.proj = EquivariantLinear(
            dims_in=dims,
            dims_out=dims,
            trivial_rep_bias=trivial_rep_proj_bias
        )

        self.proj_drop = ListDropout(proj_drop)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        x: list of tensors, each of shape (*, L, Ci, di), where di is the dimension of the i-th irrep
        """

        # list of tensors of shape (*, L, 3*Ci, d)
        qkv = self.qkv(x)

        y = [None for _ in range(len(x))]

        for i in range(len(x)):
            # (*, L, 3, num_heads, Ci//num_heads*di)
            qkv_i = qkv[i].unflatten(-2, (3, self.num_heads, self.attn_dims[i])).flatten(-2, -1)

            # (*, L, num_heads, Ci//num_heads*di)
            q_i = qkv_i[..., 0,:,:]
            k_i = qkv_i[..., 1,:,:]
            v_i = qkv_i[..., 2,:,:]

            # TODO: check whether movedim affects performance
            y_i = F.scaled_dot_product_attention(
                            q_i.movedim(-3,-2),  # (*, num_heads, L, Ci//num_heads*di)
                            k_i.movedim(-3,-2), 
                            v_i.movedim(-3,-2), 
                            dropout_p=self.attn_drop
                        ).movedim(-2, -3).reshape(x[i].shape)

            y[i] = y_i

        y = self.proj(y)
        y = self.proj_drop(y)
        return y


