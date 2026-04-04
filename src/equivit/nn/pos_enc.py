import math
import torch
import torch.nn as nn
from typing import List, Tuple

from ..geometry import Lattice, Group
from .lattice_irrep_handler import GroupActionIrrepProjectionCalculator


class EquivariantPositionalEncoding(nn.Module):
    def __init__(
        self,  
        lattice: Lattice, 
        dims: List[int],
        subgroup: tuple,
        streams: List[torch.cuda.Stream]=None
    ):
        super().__init__()
        self.dims = dims 

        self.proj_calc = GroupActionIrrepProjectionCalculator(
            lattice.action.pullback(lattice.symmetry_group.subgroup(*subgroup))
        )
        assert len(self.dims) == len(self.proj_calc.num_irreps), f"Number of dimensions ({len(self.dims)}) must match number of irreps ({len(self.proj_calc.num_irreps)})"
        self.L = self.proj_calc.L
        self.irrep_dims = self.proj_calc.irrep_dims
        self.group = self.proj_calc.group

        self.coefficients = nn.ParameterList(
            [nn.Parameter(
                torch.zeros((self.proj_calc.num_irrep_copies[i], self.dims[i]))
            ) 
                for i in range(self.proj_calc.num_irreps)]
        )

        if streams is None:
            self.streams = [None for _ in range(self.proj_calc.num_irreps)]
            self.has_streams = False
        else:
            self.streams = streams
            self.has_streams = True

        self.reset_parameters()

    def reset_parameters(self):
        for coeff in self.coefficients:
            # xavier uniform for now, we should change this later
            nn.init.xavier_uniform_(coeff)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape (*, L, Ci, di)
        Returns:
            list of tensors, each of shape (*, L, Ci, di) with positional encodings added
        """
        # (L, Ci, di)
        pos_enc = self.proj_calc(self.coefficients)

        for i in range(self.proj_calc.num_irreps):
            with torch.cuda.stream(self.streams[i]):
                x[i] = x[i] + pos_enc[i] # (*, L, Ci, di)

        return x
