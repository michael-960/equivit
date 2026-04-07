import math
import torch
import torch.nn as nn
from typing import List, Tuple

from ..geometry import Lattice, Group, AdvancedLattice, GroupElement
from .lattice_irrep_handler import GroupActionIrrepProjectionCalculator, InducedRepresentationInvariantSubspaceCalculator


# TODO: the current implementation does not include the hexvit case. We need to add this later.


class EquivariantPositionalEncoding(nn.Module):
    """
    Equivariant positional encoding.

    Let G be a group acting on a set X.

    Let (rho, V) be a G-representation (in our case, V is the direct sum of irreps of G, with multiplicities given by `dims`).

    This layer adds a learnable positional encoding f: X -> V. 

    The positional encoding vector itself is invariant under the G-action, i.e.
    f(g.x) = rho(g) f(g^{-1}.x) for all g in G, x in X.
    """
    def __init__(
        self,  
        lattice: Lattice, 
        dims: List[int],
        streams: List[torch.cuda.Stream]=None
    ):
        """
        Args:
            lattice: the lattice for which the positional encoding is defined
            dims: list of dimensions for each irrep in the representation V
            streams: list of CUDA streams to use for each irrep (optional)
        """
        super().__init__()
        self.dims = dims 
        self.proj_calc = GroupActionIrrepProjectionCalculator(lattice.action, streams=streams)

        assert len(self.dims) == len(self.proj_calc.num_irreps), f"Number of dimensions ({len(self.dims)}) must match number of irreps ({len(self.proj_calc.num_irreps)})"

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



# TODO: this is too similar to the above class.
# We should refactor the code to avoid duplication.

class EquivariantInducedPositionalEncoding(nn.Module):
    """
    This is a variant of the equivariant positional encoding that uses the
    induced representation of a subgroup of the symmetry group of the lattice.
    """
    def __init__(
        self,  
        lattice: Lattice, 
        subgroup_args: tuple,
        representatives: List[GroupElement],
        basepoints: List[int],
        dims: List[int],
        streams: List[torch.cuda.Stream]=None
    ):
        super().__init__()
        self.dims = dims 

        self.proj_calc = InducedRepresentationInvariantSubspaceCalculator(
                            lattice.action, 
                            subgroup_args=subgroup_args,
                            representatives=representatives,
                            basepoints=basepoints,
                            streams=streams
            )

        assert len(self.dims) == len(self.proj_calc.num_irreps), f"Number of dimensions ({len(self.dims)}) must match number of irreps ({len(self.proj_calc.num_irreps)})"

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
   