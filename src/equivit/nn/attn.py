import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from ..geometry import Group, IrrepType
from .linear import EquivariantLinear
from .drop import ListDropout

from .utils import assert_all_not_quaternionic


class EquivariantCoupledAttention(nn.Module):
    """
    This module does the following:

    Input: a list of tensors of shapes (*, L, C0, d0), (*, L, C1, d1), ..., 
    where di is the (complex) dimension of the i-th irrep.

    1. For each i, if the i-th irrep is of complex type, view the corresponding tensor as a float32 tensor of shape (*, L, Ci, di*2).
       (Let Di = di*2 for complex irreps and Di = di for real irreps, so that we can write the shape as (*, L, Ci, Di) for all i.)
    2. Reshape each tensor to (*, L, H, Ci/H, Di), where H is the number of attention heads. H must divide each Ci, and H is the same for all i.
    3. Flatten the last two dimensions to get (*, L, H, Ci/H * Di)
    4. Concatenate all tensors along the last dimension to get (*, L, H, sum_i Ci/H * Di)
    5. Apply multihead attention with H heads and head dimension sum_i Ci/H * Di
    6. Step 4. results in a tensor of shape (*, L, H, sum_i Ci/H * Di). Split this back into a list of tensors of shapes (*, L, H, Ci/H * Di) for each i.
    7. Reshape each tensor back to (*, L, Ci, Di)

    Compared to EquivariantIrrepwiseAttention, this allows for coupling between
    different irreps in the attention mechanism.

    This is the attention mechanism used in octic-vit (I think).
    """
    def __init__(self,
        group: Group,
        dims: List[int],
        num_heads: int,
        trivial_rep_attn_bias: bool = True,
        attn_drop: float = 0.,
        trivial_rep_proj_bias: bool = True,
        proj_drop: float = 0.
    ):
        super().__init__()
        assert_all_not_quaternionic(group)
        self.group = group
        self.irreps = group.real_irreps()
        self.num_irreps = len(self.irreps)
        assert len(dims) == self.num_irreps, f"Length of dims ({len(dims)}) should match the number of irreps ({self.num_irreps})"
        for i, c in enumerate(dims): assert c % num_heads == 0, f'num_heads ({num_heads}) does not divide dims[{i}] ({c})'

        self.is_complex = [irrep.rep_type is IrrepType.COMPLEX for irrep in self.irreps.values()]

        self.irrep_real_dims = [irrep.dim for irrep in self.irreps.values()]
        self.dims = dims
        self.split_dims = [c // num_heads * D for c,D in zip(dims, self.irrep_real_dims)]

        self.num_heads = num_heads
        self.attn_dims = [c // num_heads for c in dims]

        self.qkv = EquivariantLinear(
            group=group,
            dims_in=dims,
            dims_out=[c*3 for c in dims],
            trivial_rep_bias=trivial_rep_attn_bias
        )

        self.attn_drop = attn_drop

        self.proj = EquivariantLinear(
            group=group,
            dims_in=dims,
            dims_out=dims,
            trivial_rep_bias=trivial_rep_proj_bias
        )

        self.proj_drop = ListDropout(proj_drop)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape (*, L, Ci, di), where di is the (complex) dimension of the i-th irrep
        Returns:
            list of tensors, each of shape (*, L, Ci, di)
        """

        # (*, L, 3, H, Ci/H * Di)

        common_shape = x[0].shape[:-2] # (*, L)

        qkvs = [qkv.view(torch.float32).view(*common_shape, 3, self.num_heads, -1) # (*, L, 3, H, Ci/H * Di)
                for qkv in self.qkv(x)]

        qkvs = torch.cat(qkvs, dim=-1) # (*, L, 3, H, sum_i Ci/H * Di)

        q = qkvs.select(-3, 0) # (*, L, H, Q) Q:= sum_i Ci/H * Di
        k = qkvs.select(-3, 1) # (*, L, H, Q) 
        v = qkvs.select(-3, 2) # (*, L, H, Q)

        y = F.scaled_dot_product_attention(
                        q.movedim(-3,-2),  # (*, H, L, Q)
                        k.movedim(-3,-2), 
                        v.movedim(-3,-2),  
                        dropout_p=self.attn_drop).movedim(-2, -3) # (*, L, H, Q)

        y = torch.split(y, self.split_dims, dim=-1) # list of (*, L, H, Ci/H * Di)

        y = [z.reshape(*common_shape, self.dims[i], self.irrep_real_dims[i]) for i, z in enumerate(y)] 
        # list of (*, L, Ci, Di) where Di is the real dimension of the irrep

        y = [z.view(torch.complex64) if self.is_complex[i] else z for i, z in enumerate(y)]
        # list of real or complex tensors, each of shape (*, L, Ci, di) where di is the (complex) dimension of the irrep

        return self.proj_drop(self.proj(y))

class EquivariantIrrepwiseAttention(nn.Module):
    """
    Irrep-wise multihead attention.
    """
    def __init__(self,
        group: Group,
        dims: List[int],
        num_heads: List[int],
        trivial_rep_attn_bias: bool = True,
        attn_drop: float = 0.,
        trivial_rep_proj_bias: bool = True,
        proj_drop: float = 0.
    ):
        """
        Args:
            dims: list of input/output channels for each irrep
            num_heads: number of attention heads *per irrep* (must divide all input channels)
            trivial_rep_attn_bias: whether to include bias for the trivial representation in the attention linear layer computing q, k, v
            attn_drop: dropout probability for attention
            trivial_rep_proj_bias: whether to include bias for the trivial representation in the output projection linear layer
            proj_drop: dropout probability for the output projection

        Note: effectively, the total number of attention heads is \sum_i num_heads[i], since we are doing MHA separately for each irrep.
        """
        super().__init__()
        assert_all_not_quaternionic(group)
        self.group = group
        self.irreps = group.real_irreps()
        self.num_irreps = len(self.irreps)
        assert len(dims) == self.num_irreps, f"Length of dims ({len(dims)}) should match the number of irreps ({self.num_irreps})"
        assert len(num_heads) == self.num_irreps, f"Length of num_heads ({len(num_heads)}) should match the number of irreps ({self.num_irreps})"
        for i, (c,h) in enumerate(zip(dims, num_heads)): assert c % h == 0, f'num_heads[{i}] ({h}) does not divide dims[{i}] ({c})'

        self.is_complex = [irrep.rep_type is IrrepType.COMPLEX for irrep in self.irreps.values()]

        self.dims = dims
        self.num_heads = num_heads
        self.attn_dims = [c // h for c,h in zip(dims, num_heads)]

        self.qkv = EquivariantLinear(
            group=group,
            dims_in=dims,
            dims_out=[c*3 for c in dims],
            trivial_rep_bias=trivial_rep_attn_bias
        )

        self.attn_drop = attn_drop

        self.proj = EquivariantLinear(
            group=group,
            dims_in=dims,
            dims_out=dims,
            trivial_rep_bias=trivial_rep_proj_bias
        )

        self.proj_drop = ListDropout(proj_drop)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape (*, L, Ci, di), where di is the (complex) dimension of the i-th irrep
        Returns:
            list of tensors, each of shape (*, L, Ci, di)
        """

        # list of tensors of shape (*, L, 3*Ci, di)
        qkv = self.qkv(x)

        y = [None for _ in range(len(x))]

        for i in range(len(x)):
            _shape = x[i].shape # (*, L, Ci, di) complex64 or float32 depending on the irrep type

            # (*, L, 3, num_heads_i, Ci//num_heads_i * di)
            qkv_i = qkv[i].unflatten(-2, (3, self.num_heads[i], self.attn_dims[i])).flatten(-2, -1)

            # (*, L, num_heads_i, Ci//num_heads_i * di)  if real
            # (*, L, num_heads_i, Ci//num_heads_i * di*2) if complex 
            q_i = qkv_i.select(-3, 0).view(torch.float32)
            k_i = qkv_i.select(-3, 1).view(torch.float32)
            v_i = qkv_i.select(-3, 2).view(torch.float32)

            y_i = F.scaled_dot_product_attention(
                            q_i.movedim(-3,-2),  # (*, num_heads_i, L, Ci//num_heads*di) (real) or (*, num_heads_i, L, Ci//num_heads*di*2) (complex)
                            k_i.movedim(-3,-2), 
                            v_i.movedim(-3,-2), 
                            dropout_p=self.attn_drop
                        ).movedim(-2, -3)
            if self.is_complex[i]:
                # (*, L, num_heads_i, Ci//num_heads_i * di*2) float32
                # -> (*, L, num_heads_i, Ci//num_heads_i * di) complex64
                y_i = y_i.view(torch.complex64)                        

            # can we do view here?
            # y_i is not contiguous because of the scaled_dot_product_attention, so we would need to call contiguous() before view()
            y[i] = y_i.reshape(_shape) # (*, L, Ci, di)

        y = self.proj(y)
        y = self.proj_drop(y)
        return y


