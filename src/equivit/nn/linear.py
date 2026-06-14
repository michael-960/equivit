import torch
import torch.nn as nn
import math
from typing import List

from ..geometry import Group, IrrepType

from .utils import assert_all_not_quaternionic

from .init import complex_uniform_disk_, kaiming_uniform_, complex_kaiming_uniform_, complex_trunc_normal_

from ._core import resolve_dims


class EquivariantLinear(nn.Module):
    r"""
    :math:`G`-equivariant linear layer applied to a list of tensors transforming in irreps of :math:`G`.

    Let :math:`G` be a group and let :math:`V_0, V_1, \dotsb, V_{M-1}` be the real irreps of :math:`G`.

    For an input list of tensors
    :math:`x_0, x_1, \dotsb, x_{M-1}`, 
    where :math:`x_i` has shape :math:`(*, C_i, d_i)` and transforms in the irrep :math:`V_i` 
    (i.e. the last dimension of :math:`x_i` transforms according to the representation matrix of :math:`V_i`), 
    this layer applies a linear transformation to each tensor that is equivariant under the :math:`G`-action.

    That is, it applys :math:`\bigoplus_i L_i` to the input, where 

    .. math::
        L_i\in \mathrm{Hom}_G(\mathbb{R}^{C_i}\otimes V_i, \mathbb{R}^{C_i'}\otimes V_i)

    is a learnable intertwiner.

    If :attr:`trivial_rep_bias` is True, then a learnable (per-channel) bias term is added to the output of the trivial representation (the zeroth irrep). 

    Args:
        group: the symmetry group :math:`G` that we want to respect
        dims_in: list of input channels :math:`C_0, C_1, \dotsb`, one for each irrep
        dims_out: list of output channels :math:`C_0', C_1', \dotsb`, one for each irrep
        trivial_rep_bias: whether to include bias for the trivial representation (the zeroth irrep)

    Note:
        Currently, we only support groups whose irreps are all of real or complex type. We do not support groups with quaternionic irreps yet.

        - If :math:`V_i` is of real type, then :math:`\mathrm{Hom}_G(\mathbb{R}^{C_i}\otimes V_i, \mathbb{R}^{C_i'}\otimes V_i)` is isomorphic to :math:`\mathbb{R}^{C_i'\times C_i}`.
        - If :math:`V_i` is of complex type, then :math:`\mathrm{Hom}_G(\mathbb{R}^{C_i}\otimes V_i, \mathbb{R}^{C_i'}\otimes V_i) \cong \mathbb{C}^{C_i'\times C_i}`.
          We deal with this by having complex-valued weights. Consequently, the input tensor :math:`x_i` for a complex irrep should be ``torch.complex64`` instead of ``torch.float32``.
          This also means that the representation matrices of the real irrep must be chosen tocommute with the standard complex structure.
          This is the case for the implementation of the irreps of :class:`equivit.geometry.DihedralGroup` and :class:`equivit.geometry.CyclicGroup`.
    """
    def __init__(self,
        group: Group,
        dims_in: List[int], 
        dims_out: List[int], 
        trivial_rep_bias: bool=True,
    ):
        super().__init__()
        assert_all_not_quaternionic(group)

        self.group = group
        irreps = group.real_irreps()

        self.num_irreps = len(irreps)
        self.dtypes = [torch.float32 if irrep.rep_type is IrrepType.REAL else torch.complex64 for irrep in irreps.values()]

        # assert len(dims_in) == self.num_irreps, "Length of dims_in should match number of irreps"
        # assert len(dims_out) == self.num_irreps, "Length of dims_out should match number of irreps"

        # self.dims_in = list(dims_in)
        # self.dims_out = list(dims_out)

        self.dims_in = resolve_dims(group, dims_in)
        self.dims_out = resolve_dims(group, dims_out)

        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(self.dims_out[i], self.dims_in[i], dtype=self.dtypes[i])) 
            for i in range(self.num_irreps)
        ])

        if trivial_rep_bias:
            self.bias = nn.Parameter(torch.zeros(self.dims_out[0], 1))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()


    def reset_parameters(self) -> None:
        gain = math.sqrt(2)
        with torch.no_grad():
            for i in range(self.num_irreps):
                if self.weights[i].numel() > 0:
                    if self.weights[i].dtype.is_complex:
                        # if complex, draw the weights from the unit disk
                        # complex_kaiming_uniform_(self.weights[i], fan_in=self.dims_in[i], gain=gain)
                        complex_trunc_normal_(self.weights[i], std=0.02, bound=0.04)
                    else:
                        # kaiming_uniform_(self.weights[i], fan_in=self.dims_in[i], gain=gain)
                        nn.init.trunc_normal_(self.weights[i], std=0.02)

            if self.bias is not None:
                if self.bias.numel() > 0:
                    # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[0])
                    # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    # nn.init.uniform_(self.bias, -bound, bound)

                    # let's just initiliaze the bias to zero
                    nn.init.zeros_(self.bias)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape :math:`(*, C_i, d_i)` for each irrep

        Returns: list of tensors, each of shape :math:`(*, C_i', d_i)` for each irrep (where :math:`C_i'` is the output dimension for that irrep)

        Note:
            - If the :math:`i`-th irrep is of real type, then the dtype of ``x[i]`` should be real
            - If the :math:`i`-th irrep is of complex type, then the dtype of ``x[i]`` should be complex.
        """
        outs = [None for _ in range(self.num_irreps)]

        for i in range(self.num_irreps):
            z = self.weights[i] @ x[i]
            outs[i] = z.view(-1).view(*z.shape) 
            # this is to ensure that z has the correct strides when there are 1's in z.shape
            # without this, PyTorch will throw an error if we later do torch.view_as_real(z) if z.shape[-1] == 1 and z.stride(-1) != 1
            if i == 0 and self.bias is not None:
                outs[i] += self.bias
        return outs


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(group={self.group}, dims_in={self.dims_in}, dims_out={self.dims_out}, trivial_rep_bias={self.bias is not None})"
