from ..geometry import Honeycomb, Triangle, decompose_set_action
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Union, Optional
import math

from ..geometry import Group, Lattice, GroupAction, IrrepType

from .lattice_irrep_handler import GroupActionIrrepProjectionCalculator

from .utils import assert_all_not_quaternionic

from .init import complex_uniform_disk_, kaiming_uniform_, complex_kaiming_uniform_

from ._core import resolve_dims

from . import functional as EF



# Patchembed can be decomposed into two steps:
# (*, Lpatch, C) -> ..., (*, C, Mi, di), ... -> ..., (*, Ci, di), ...
# where the Mi are the multiplicities of the irreps in the decomposition of the
# patch action, and the Ci are the desired output channels for each irrep. 
# note: if the Mi are large, these two steps should not be done independently


class EquivariantPatchEmbedTranspose(nn.Module):
    def __init__(self, 
        action: GroupAction,
        dims: Union[List[int], Dict[str, int]],
        out_channels: int, 
        use_sparse: bool = True,
    ):
        super().__init__()

        assert_all_not_quaternionic(action.group)

        self.action = action
        self.out_channels = out_channels
        self.dims = resolve_dims(action.group, dims)

        self.proj_calc = GroupActionIrrepProjectionCalculator(action, use_sparse=use_sparse)

        self.L = self.action.num_elements

        irreps = action.group.real_irreps().values()

        self.irrep_dims = [irrep.dim if irrep.rep_type is IrrepType.REAL else irrep.dim//2 for irrep in irreps]
        self.dtypes = [torch.float32 if irrep.rep_type is IrrepType.REAL else torch.complex64
                       for irrep in irreps]
        self.is_complex = [irrep.rep_type is IrrepType.COMPLEX for irrep in irreps]

        assert len(self.dims) == self.proj_calc.num_irreps, f"Number of output channels ({len(self.dims)}) must match number of irreps ({self.proj_calc.num_irreps})"

        self.coefficients = nn.ParameterList(
            [nn.Parameter(
                torch.zeros((self.proj_calc.num_irrep_copies[i], out_channels*self.dims[i]), dtype=self.dtypes[i])
            ) 
                for i in range(self.proj_calc.num_irreps)]
        )
        self.reset_parameters()


    def reset_parameters(self):
        gain = 1.0 # because no activation is applied after patch embedding and positional encoding
        with torch.no_grad():
            for i, coeff in enumerate(self.coefficients):
                # xavier uniform for now, we should change this later
                # this is wrong! 
                # nn.init.xavier_uniform_(coeff)
                if self.is_complex[i]:
                    fan_in = self.proj_calc.num_irrep_copies[i] * self.out_channels
                    complex_kaiming_uniform_(coeff, fan_in=fan_in, gain=gain)
                else:
                    fan_in = self.proj_calc.num_irrep_copies[i] * self.out_channels
                    kaiming_uniform_(coeff, fan_in=fan_in, gain=gain)

    def get_projections(self):
        filts = self.proj_calc(self.coefficients)
        # each entry has shape (Lpatch, C*Ci, di) -> (Lpatch*C, Ci*di)
        # note: di is the complex dimension of the irrep
        return [filt.view(self.L*self.out_channels, self.dims[i]*self.irrep_dims[i]) for i, filt in enumerate(filts)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape :math:`(*, C_i, d_i)`, 
            where :math:`d_i` is the dimension of the input irrep and :math:`C_i` is the number of input channels for that irrep. 

        Returns: 
            a tensor, of shape :math:`(*, L, C)`, where :math:`L` is the number of pixels in each patch and :math:`C` is the number of output channels.

        """

        # outs: List = [None for _ in range(self.proj_calc.num_irreps)]

        x = [z.flatten(-2) for z in x] # (*, Ci*di)

        # each entry has shape (Lpatch*C, Ci*di)
        filts = self.get_projections()

        common_shape = x[0].shape[:-1]
        out = x[0].new_zeros(size=(*common_shape, self.L*self.out_channels))

        for i in range(self.proj_calc.num_irreps):
            if self.is_complex[i]:
                # outs[i] = (x @ filts[i].view(torch.float32)).view(torch.complex64).unflatten(-1, (self.out_channels[i], self.irrep_dims[i])) # (*, Ci, di)
                # outs[i] = torch.view_as_complex(
                #                                 (x @ torch.view_as_real(filts[i]).flatten(-2,-1)).unflatten(-1, (-1, 2))
                #                 ).unflatten(-1, (self.dims[i], self.irrep_dims[i])) # (*, Ci, di)
                out += EF.to_real(x[i] @ filts[i].t())
            else:
                # outs[i] = (x @ filts[i]).unflatten(-1, (self.dims[i], self.irrep_dims[i])) # (*, Ci, di)
                out += x[i] @ filts[i].t()

        return out.unflatten(-1, (self.L, self.out_channels)) # (*, L, C)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(action={self.action}, out_channels={self.out_channels}, dims={self.dims})"
