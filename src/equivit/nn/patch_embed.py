from ..geometry import Honeycomb, Triangle, decompose_set_action
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

from ..geometry import Group, Lattice, GroupAction, IrrepType

from .lattice_irrep_handler import GroupActionIrrepProjectionCalculator

from .utils import assert_all_not_quaternionic



# Patchembed can be decomposed into two steps:
# (*, Lpatch, C) -> ..., (*, C, Mi, di), ... -> ..., (*, Ci, di), ...
# where the Mi are the multiplicities of the irreps in the decomposition of the
# patch action, and the Ci are the desired output channels for each irrep. 
# note: if the Mi are large, these two steps should not be done independently


class EquivariantPatchEmbed(nn.Module):
    r"""
    Patch embedding layer that is equivariant to a given group action.

    Let :math:`G` be a group acting on a set :math:`X`. Let :math:`C(X,
    \mathbb{R})` be the space of real-valued functions on :math:`X`, which is a
    :math:`G`-representation.

    Let :math:`V_0, V_1, \dotsb, V_{M-1}` be the real irreps of :math:`G`.

    Choose an isomorphism :math:`\Phi` that decomposes :math:`C(X, \mathbb{R})` into irreps:

    .. math::
        \Phi: C(X, \mathbb{R}) \rightarrow \bigoplus_{i=0}^{M-1} \mathbb{R}^{D_i}\otimes V_i,

    where :math:`D_i` is the multiplicity of the irrep :math:`V_i` in :math:`C(X, \mathbb{R})`.

    For each irrep :math:`V_i`, let :math:`L_i\in
    \mathrm{Hom}_G(\mathbb{R}^{D_i}\otimes V_i, \mathbb{R}^{C_i}\otimes V_i)` be
    a learnable intertwiner. 

    This layer applies the following composition to an input :math:`f\in C(X, \mathbb{R})`:

    .. math::
        C(X, \mathbb{R}) \xrightarrow{\Phi} \bigoplus_{i=0}^{M-1} \mathbb{R}^{D_i}\otimes V_i 
        \xrightarrow{\bigoplus_i L_i} \bigoplus_{i=0}^{M-1} \mathbb{R}^{C_i}\otimes V_i.

    The second map is the same as applying a :class:`EquivariantLinear` layer without bias.

    Args:
        action: a group action.
        in_channels: an integer :math:`C` specifying the number of input channels
        out_channels: list of integers :math:`C_0, C_1, \dotsb, C_{M-1}` specifying the number of output channels for each irrep
        use_sparse: whether to use sparse matrices for the projection (can save memory and speed up computation for large groups, but may be slower for small groups)

    Note:
        This layer does not apply :math:`\Phi` and :math:`\bigoplus_i L_i`
        separately, but instead computes `\bigoplus_i L_i \circ \Phi` first.
    """
    def __init__(self, 
        action: GroupAction,
        in_channels: int, 
        out_channels: List[int],
        use_sparse: bool = True,
        # streams: List[torch.cuda.Stream]=None
    ):
        super().__init__()

        assert_all_not_quaternionic(action.group)

        self.action = action
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.proj_calc = GroupActionIrrepProjectionCalculator(action, use_sparse=use_sparse)

        self.L = self.action.num_elements

        irreps = action.group.real_irreps().values()

        self.irrep_dims = [irrep.dim if irrep.rep_type is IrrepType.REAL else irrep.dim//2 for irrep in irreps]
        self.dtypes = [torch.float32 if irrep.rep_type is IrrepType.REAL else torch.complex64
                       for irrep in irreps]
        self.is_complex = [irrep.rep_type is IrrepType.COMPLEX for irrep in irreps]

        assert len(self.out_channels) == self.proj_calc.num_irreps, f"Number of output channels ({len(self.out_channels)}) must match number of irreps ({self.proj_calc.num_irreps})"

        self.coefficients = nn.ParameterList(
            [nn.Parameter(
                torch.zeros((self.proj_calc.num_irrep_copies[i], in_channels*out_channels[i]), dtype=self.dtypes[i])
            ) 
                for i in range(self.proj_calc.num_irreps)]
        )
        self.reset_parameters()


    def reset_parameters(self):
        for coeff in self.coefficients:
            # xavier uniform for now, we should change this later
            nn.init.xavier_uniform_(coeff)

    def get_projections(self):
        filts = self.proj_calc(self.coefficients)
        # each entry has shape (Lpatch, C*Ci, di) -> (Lpatch*C, Ci*di)
        # note: di is the complex dimension of the irrep
        return [filt.view(self.L*self.in_channels, self.out_channels[i]*self.irrep_dims[i]) for i, filt in enumerate(filts)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: tensor of shape :math:`(*, L, C)`, where :math:`L` is the number of pixels in each patch and :math:`C` is the number of input channels.

        Returns: 
            a list of tensors, each of shape :math:`(*, C_i, d_i)`,
            where di is the dimension of each irrep and :math:`C_i` is the number of output channels for that irrep.

        Note: 
            An independent "patchification" module should be applied to the
            input before this module to rearrange the input into patches. This
            allows for more flexibility in the patch structure and the symmetries
            that can be respected.
        """

        outs: List = [None for _ in range(self.proj_calc.num_irreps)]

        x = x.flatten(-2) # (*, Lpatch*C)

        # each entry has shape (Lpatch*C, Ci*di)
        filts = self.get_projections()

        # Potential optimization possibilities:
        # 1. separate real and complex indices
        # 2. collect all filters into a single matrix of shape (Lpatch*C, sum_i Ci*di) and do a single matmul (maybe once for real irreps and once for complex irreps)
        for i in range(self.proj_calc.num_irreps):
            if self.is_complex[i]:
                outs[i] = (x @ filts[i].view(torch.float32)).view(torch.complex64).unflatten(-1, (self.out_channels[i], self.irrep_dims[i])) # (*, Ci, di)
            else:
                outs[i] = (x @ filts[i]).unflatten(-1, (self.out_channels[i], self.irrep_dims[i])) # (*, Ci, di)
        return outs





