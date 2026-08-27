import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from collections.abc import Sequence

from ..geometry import Group, IrrepType, GroupAction
from .linear import EquivariantLinear
from .drop import ListDropout

from .utils import assert_all_not_quaternionic

from . import functional as EF

from .init import complex_uniform_disk_, kaiming_uniform_, complex_kaiming_uniform_


class EquivariantCoupledAttention(nn.Module):
    r"""
    This module does the following:

    Input: a list of tensors of shapes :math:`(*, L, C_0, d_0), (*, L, C_1, d_1),\dotsb`, 
    where :math:`d_i` is the (complex) dimension of the :math:`i`-th irrep.

    1. For each :math:`i`, if the :math:`i`-th irrep is of complex type, view the corresponding tensor as a float32 tensor of shape :math:`(*, L, C_i, d_i\cdot 2)`.
       (Let :math:`D_i = d_i\cdot 2` for complex irreps and :math:`D_i = d_i` for real irreps, so that we can write the shape as :math:`(*, L, C_i, D_i)` for all :math:`i`.)
       
    2. Reshape each tensor to :math:`(*, L, H, \frac{C_i}{H}, D_i)`, where :math:`H` is the number of attention heads. :math:`H` must 
       divide each :math:`C_i`, and :math:`H` is the same for all :math:`i`.

    3. Flatten the last two dimensions to get :math:`(*, L, H, \frac{C_i}{H} \cdot D_i)`

    4. Concatenate all tensors along the last dimension to get a tensor of shape :math:`(*, L, H, \sum_i \frac{C_i}{H} \cdot Di)`

    5. Apply multihead attention with :math:`H` heads and head dimension :math:`\sum_i \frac{C_i}{H} \cdot Di`

    6. Step 5. results in a tensor of shape :math:`(*, L, H, \sum_i \frac{C_i}{H} \cdot D_i)`. Split this back into a list of tensors 
       of shapes :math:`(*, L, H, \frac{C_i}{H} \cdot D_i) for each :math:`i`.

    7. Reshape each tensor back to :math:`(*, L, C_i, D_i)`

    Compared to EquivariantIrrepwiseAttention, this allows for coupling between
    different irreps in the attention mechanism.

    This is the attention mechanism used in octic-vit.

    Args:
        group: the symmetry group
        dims: list of input/output channels for each irrep
        num_heads: number of attention heads (must divide all input channels)
        trivial_rep_attn_bias: whether to include bias for the trivial representation in the attention linear layer computing :math:`q, k, v`
        attn_drop: dropout probability for attention
        trivial_rep_proj_bias: whether to include bias for the trivial representation in the output projection linear layer
        proj_drop: dropout probability for the output projection
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

        self.dims = dims
        self.num_heads = num_heads
        self.trivial_rep_attn_bias = trivial_rep_attn_bias
        self.trivial_rep_proj_bias = trivial_rep_proj_bias


        self.is_complex = [irrep.rep_type is IrrepType.COMPLEX for irrep in self.irreps.values()]

        self.irrep_real_dims = [irrep.dim for irrep in self.irreps.values()]
        self.split_dims = [c // num_heads * D for c,D in zip(dims, self.irrep_real_dims)]

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
        self.proj_drop_p = proj_drop

        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.0
        with torch.no_grad():
            for i, weight in enumerate(self.qkv.weights):
                if self.is_complex[i]:
                    complex_kaiming_uniform_(weight, fan_in=self.dims[i], gain=gain)
                else:
                    kaiming_uniform_(weight, fan_in=self.dims[i], gain=gain)

            for i, weight in enumerate(self.proj.weights):
                if self.is_complex[i]:
                    complex_kaiming_uniform_(weight, fan_in=self.dims[i], gain=gain)
                else:
                    kaiming_uniform_(weight, fan_in=self.dims[i], gain=gain)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape :math:`(*, L, C_i, d_i)`, where :math:`d_i` is the (complex) dimension of the :math:`i`-th irrep
        Returns:
            list of tensors, each of shape :math:`(*, L, C_i, d_i)`
        """

        common_shape = x[0].shape[:-2] # (*, L)

        # qkvs = [qkv.view(torch.float32).view(*common_shape, 3, self.num_heads, -1) # (*, L, 3, H, Ci/H * Di)
                # for qkv in self.qkv(x)]
        qkvs = []
        for i, qkv_i in enumerate(self.qkv(x)):
            # qkv_i has shape (*, L, 3*Ci, di)
            if self.is_complex[i]:
                # qkv_i = torch.view_as_real(qkv_i).flatten(-2).view(*common_shape, 3, self.num_heads, -1)
                qkv_i = EF.to_real(qkv_i).flatten(-2).view(*common_shape, 3, self.num_heads, -1)
            else:
                qkv_i = qkv_i.view(*common_shape, 3, self.num_heads, -1)

            # qkv_i has shape (*, L, 3, H, Ci/H * Di)
            qkvs.append(qkv_i)

        qkvs = torch.cat(qkvs, dim=-1) # (*, L, 3, H, sum_i Ci/H * Di)

        q = qkvs.select(-3, 0) # (*, L, H, Q) Q:= sum_i Ci/H * Di
        k = qkvs.select(-3, 1) # (*, L, H, Q) 
        v = qkvs.select(-3, 2) # (*, L, H, Q)

        y = F.scaled_dot_product_attention(
                        q.movedim(-3,-2),  # (*, H, L, Q)
                        k.movedim(-3,-2), 
                        v.movedim(-3,-2),  
                        dropout_p=self.attn_drop if self.training else 0.
            ).movedim(-2, -3) # (*, L, H, Q)

        y = torch.split(y, self.split_dims, dim=-1) # list of (*, L, H, Ci/H * Di)

        y = [z.reshape(*common_shape, self.dims[i], self.irrep_real_dims[i]).contiguous() for i, z in enumerate(y)] 
        # list of (*, L, Ci, Di) where Di is the real dimension of the irrep

        # y = [torch.view_as_complex(z.unflatten(-1, (-1, 2))) if self.is_complex[i] else z for i, z in enumerate(y)]
        y = [EF.to_complex(z.unflatten(-1, (-1, 2))) if self.is_complex[i] else z for i, z in enumerate(y)]
        # list of real or complex tensors, each of shape (*, L, Ci, di) where di is the (complex) dimension of the irrep

        return self.proj_drop(self.proj(y))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(group={self.group}, dims={self.dims}, num_heads={self.num_heads}, trivial_rep_attn_bias={self.trivial_rep_attn_bias}, attn_drop={self.attn_drop}, trivial_rep_proj_bias={self.trivial_rep_proj_bias}, proj_drop={self.proj_drop_p})"


class EquivariantIrrepwiseAttention(nn.Module):
    r"""
    Irrep-wise multihead attention.

    Args:
        group: the symmetry group
        dims: a list :math:`C_0, C_1, \dotsb` of input/output channels for each irrep
        num_heads: a list :math:`h_0, h_1, \dotsb` specifying the number of attention heads in each irrep (:math:`h_i` must divide each input channel :math:`C_i`)
        trivial_rep_attn_bias: whether to include bias for the trivial representation in the attention linear layer computing :math:`q, k, v`
        attn_drop: dropout probability for attention
        trivial_rep_proj_bias: whether to include bias for the trivial representation in the output projection linear layer
        proj_drop: dropout probability for the output projection

    Note: 
        Effectively, the total number of attention heads is :math:`\sum_i h_i`, since we are doing MHA separately for each irrep.
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
        super().__init__()
        assert_all_not_quaternionic(group)
        self.group = group
        self.irreps = group.real_irreps()
        self.num_irreps = len(self.irreps)

        assert isinstance(dims, Sequence), f"dims should be a list of integers specifying the number of channels for each irrep, but got {dims}"
        assert isinstance(num_heads, Sequence), f"num_heads should be a list of integers specifying the number of attention heads for each irrep, but got {num_heads}"

        assert len(dims) == self.num_irreps, f"Length of dims ({len(dims)}) should match the number of irreps ({self.num_irreps})"
        assert len(num_heads) == self.num_irreps, f"Length of num_heads ({len(num_heads)}) should match the number of irreps ({self.num_irreps})"
        for i, (c,h) in enumerate(zip(dims, num_heads)): assert c % h == 0, f'num_heads[{i}] ({h}) does not divide dims[{i}] ({c})'

        self.is_complex = [irrep.rep_type is IrrepType.COMPLEX for irrep in self.irreps.values()]

        self.dims = list(dims)
        self.num_heads = list(num_heads)
        self.attn_dims = [c // h for c,h in zip(dims, num_heads)]

        self.trivial_rep_attn_bias = trivial_rep_attn_bias
        self.trivial_rep_proj_bias = trivial_rep_proj_bias


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
        self.proj_drop_p = proj_drop

        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.0
        with torch.no_grad():
            for i, weight in enumerate(self.qkv.weights):
                if self.is_complex[i]:
                    complex_kaiming_uniform_(weight, fan_in=self.dims[i], gain=gain)
                else:
                    kaiming_uniform_(weight, fan_in=self.dims[i], gain=gain)

            for i, weight in enumerate(self.proj.weights):
                if self.is_complex[i]:
                    complex_kaiming_uniform_(weight, fan_in=self.dims[i], gain=gain)
                else:
                    kaiming_uniform_(weight, fan_in=self.dims[i], gain=gain)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape :math:`(*, L, C_i, d_i)`, where :math:`d_i` is the (complex) dimension of the :math:`i`-th irrep
        Returns:
            list of tensors, each of shape :math:`(*, L, C_i, d_i)`
        """

        # list of tensors of shape (*, L, 3*Ci, di)
        qkv = self.qkv(x)

        y = [None for _ in range(len(x))]

        _attn_drop = self.attn_drop if self.training else 0.

        for i in range(len(x)):
            _shape = x[i].shape # (*, L, Ci, di) complex64 or float32 depending on the irrep type

            # (*, L, 3, num_heads_i, Ci//num_heads_i * di)
            qkv_i = qkv[i].unflatten(-2, (3, self.num_heads[i], self.attn_dims[i])).flatten(-2, -1)

            # (*, L, num_heads_i, Ci//num_heads_i * di)  if real
            # (*, L, num_heads_i, Ci//num_heads_i * di*2) if complex 

            # this breaks the graph!
            # q_i = qkv_i.select(-3, 0).view(torch.float32)
            # k_i = qkv_i.select(-3, 1).view(torch.float32)
            # v_i = qkv_i.select(-3, 2).view(torch.float32)

            q_i = qkv_i.select(-3, 0)
            k_i = qkv_i.select(-3, 1)
            v_i = qkv_i.select(-3, 2)

            if self.is_complex[i]:
                # q_i = torch.view_as_real(q_i).flatten(-2)
                # k_i = torch.view_as_real(k_i).flatten(-2)
                # v_i = torch.view_as_real(v_i).flatten(-2)
                q_i = EF.to_real(q_i).flatten(-2)
                k_i = EF.to_real(k_i).flatten(-2)
                v_i = EF.to_real(v_i).flatten(-2)


            y_i = F.scaled_dot_product_attention(
                            q_i.movedim(-3,-2),  # (*, num_heads_i, L, Ci//num_heads*di) (real) or (*, num_heads_i, L, Ci//num_heads*di*2) (complex)
                            k_i.movedim(-3,-2), 
                            v_i.movedim(-3,-2), 
                            dropout_p=_attn_drop
                        ).movedim(-2, -3)
            if self.is_complex[i]:
                # (*, L, num_heads_i, Ci//num_heads_i * di*2) float32
                # -> (*, L, num_heads_i, Ci//num_heads_i * di) complex64
                # y_i = y_i.view(torch.complex64)                        
                y_i = EF.to_complex(y_i.unflatten(-1, (-1, 2)))

            # can we do view here?
            # y_i is not contiguous because of the scaled_dot_product_attention, so we would need to call contiguous() before view()
            y[i] = y_i.reshape(_shape) # (*, L, Ci, di)

        y = self.proj(y)
        y = self.proj_drop(y)
        return y

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(group={self.group}, dims={self.dims}, num_heads={self.num_heads}, trivial_rep_attn_bias={self.trivial_rep_attn_bias}, attn_drop={self.attn_drop}, trivial_rep_proj_bias={self.trivial_rep_proj_bias}, proj_drop={self.proj_drop_p})"



class EquivariantAttention(nn.Module):
    r"""
    Equivariant multi-head self-attention in reduced parameter space.

    Let :math:`G` be a finite group and :math:`B` a finite :math:`G`-set. 
    Let :math:`V` be an orthogonal :math:`G`-representation.


    Suppose :math:`M, R: B \rightarrow \mathrm{End}(V)` are two maps such that

    .. math::
        M_{gb} = g M_b g^{-1}, \quad R_{gb} = g R_b g^{-1}

    for all :math:`g \in G` and :math:`b \in B`.

    This module implements the following multi-head self-attention:

    .. math::
        \mathrm{attn}(x)_i = \sum_{b\in B} 
            \frac{\sum_{j=1}^L e^{\braket{x_i, M_b x_j}} R_b x_j}
            {\sum_{j=1}^L e^{\braket{x_i, M_b x_j}}},

    where :math:`x = (x_1, \dotsb, x_L) \in \mathbb{R}^L\otimes V` is the input token sequence.
    """
    def __init__(self,
        action: GroupAction,
    ):
        super().__init__()
        self.reset_parameters()
        raise NotImplementedError("EquivariantAttention is not implemented yet.")

    def reset_parameters(self):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: list of tensors, each of shape :math:`(*, L, C_i, d_i)`, where :math:`d_i` is the (complex) dimension of the :math:`i`-th irrep
        Returns:
            list of tensors, each of shape :math:`(*, L, C_i, d_i)`
        """
        ...
