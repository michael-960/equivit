import math
import torch
import torch.nn as nn
from typing import List, Tuple

from ..geometry import Lattice, Group, AdvancedLattice, GroupElement, GroupAction, IrrepType, EquivariantPullbackBundle
from .lattice_irrep_handler import GroupActionIrrepProjectionCalculator, InducedRepresentationInvariantSubspaceCalculator

from .utils import assert_all_not_quaternionic


# TODO: the current implementation does not include the hexvit case. We need to add this later.


class EquivariantPositionalEncoding(nn.Module):
    """
    Equivariant positional encoding.

    Let G be a group acting on a set X.

    Let (rho, V) be a G-representation (in our case, V is the direct sum of irreps of G, with multiplicities given by `dims`).

    This layer adds a learnable positional encoding f: X -> V. 

    The positional encoding vector itself is invariant under the G-action, i.e.
    f(g.x) = rho(g) f(g^{-1}.x) for all g in G, x in X.

    TODO: complex-type irreps
    """
    def __init__(
        self,  
        action: GroupAction, 
        dims: List[int],
        use_sparse: bool = True
    ):
        """
        Args:
            lattice: the lattice for which the positional encoding is defined
            dims: list of dimensions for each irrep in the representation V
        """
        super().__init__()
        assert_all_not_quaternionic(action.group)

        self.dims = dims
        self.proj_calc = GroupActionIrrepProjectionCalculator(action, use_sparse=use_sparse)

        irreps = action.group.real_irreps().values()

        self.dtypes = [torch.float32 if irrep.rep_type is IrrepType.REAL else torch.complex64 for irrep in irreps]

        self.num_irreps =  len(irreps)

        assert len(self.dims) == self.num_irreps, f"Number of dimensions ({len(self.dims)}) must match number of irreps ({self.num_irreps})"

        self.coefficients = nn.ParameterList(
            [nn.Parameter(
                torch.zeros((self.proj_calc.num_irrep_copies[i], self.dims[i]), dtype=self.dtypes[i])
            ) 
                for i in range(self.num_irreps)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for coeff in self.coefficients:
            # xavier uniform for now, we should change this later
            nn.init.xavier_uniform_(coeff)

    def get_positional_encodings(self) -> List[torch.Tensor]:
        return self.proj_calc(self.coefficients)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape (*, L, Ci, di)
        Returns:
            list of tensors, each of shape (*, L, Ci, di) with positional encodings added
        """
        # (L, Ci, di) for each irrep
        # each tensor can be real or complex depending on the irrep type
        # if complex, di is the complex dimension
        pos_enc = self.get_positional_encodings()

        for i in range(self.num_irreps):
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
        # action: GroupAction, 
        # subgroup_args: tuple,
        # representatives: List[GroupElement],
        # basepoints: List[int],
        pullback_bundle: EquivariantPullbackBundle,
        dims: List[int],
        use_sparse: bool = True
    ):
        """
        Args:
            pullback_bundle: a G-equivariant principal H-bundle.
            dims: list of dimensions for each irrep of H
        """
        super().__init__()
        self.dims = dims 

        self.pullback_bundle = pullback_bundle

        self.proj_calc = InducedRepresentationInvariantSubspaceCalculator(
                            # action, 
                            # subgroup_args=subgroup_args,
                            # representatives=representatives,
                            # basepoints=basepoints,
                            pullback_bundle=self.pullback_bundle,
                            use_sparse=use_sparse
            )

        assert_all_not_quaternionic(pullback_bundle.subgroup)
        
        irreps = pullback_bundle.subgroup.real_irreps().values()

        self.dtypes = [torch.float32 if irrep.rep_type is IrrepType.REAL else torch.complex64 for irrep in irreps]

        self.num_irreps = len(irreps)

        assert len(self.dims) == self.num_irreps, f"Number of dimensions ({len(self.dims)}) must match number of irreps ({self.num_irreps})"

        self.coefficients = nn.ParameterList(
            [nn.Parameter(
                torch.zeros((self.proj_calc.num_irrep_copies[i], self.dims[i]), dtype=self.dtypes[i])
            ) 
                for i in range(self.num_irreps)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for coeff in self.coefficients:
            # xavier uniform for now, we should change this later
            nn.init.xavier_uniform_(coeff)

    def get_positional_encodings(self) -> List[torch.Tensor]:
        return self.proj_calc(self.coefficients)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape (*, L, Ci, di)
        Returns:
            list of tensors, each of shape (*, L, Ci, di) with positional encodings added
        """
        # (L, Ci, di)
        pos_enc = self.get_positional_encodings()

        for i in range(self.num_irreps):
            x[i] = x[i] + pos_enc[i] # (*, L, Ci, di)
        return x
   