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
    """
    Patch embedding layer that respects the symmetries of a lattice.
    We use the irreps of the symmetry group to project the input features.

    TODO: complex-type irreps
    """
    def __init__(self, 
        action: GroupAction,
        in_channels: int, 
        out_channels: List[int],
        use_sparse: bool = True,
        # streams: List[torch.cuda.Stream]=None
    ):
        """
        Args:
            action: a group action.
            in_channels: number of input channels
            out_channels: list of output channels for each irrep
        """
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
            x: tensor of shape (*, Lpatch, C), where Lpatch is the number of pixels in each patch and C is the number of input channels.
        Returns: 
            a list of tensors, each of shape (*, Ci, di),
            where di is the dimension of each irrep and Ci is the number of output channels for that irrep.

        Note: an independent "patchification" module should be applied to the
        input before this module to rearrange the input into patches. This
        allows for more flexibility in the patch structure and the symmetries
        that can be respected.
        """
        # 
        outs = [None for _ in range(self.proj_calc.num_irreps)]

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





