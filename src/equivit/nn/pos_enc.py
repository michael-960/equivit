import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union

from ..geometry import Lattice, Group, GroupElement, GroupAction, IrrepType, EquivariantPullbackBundle
from .lattice_irrep_handler import GroupActionIrrepProjectionCalculator, InducedRepresentationInvariantSubspaceCalculator

from .utils import assert_all_not_quaternionic

from ._core import resolve_dims


# TODO: the current implementation does not include the hexvit case. We need to add this later.


class EquivariantPositionalEncoding(nn.Module):
    r"""
    Equivariant positional encoding.

    Let :math:`G` be a group acting on a set :math:`X`. Let :math:`(\rho, V)` be a :math:`G`-representation 
    (in our case, :math:`V` is the direct sum of irreps of :math:`G`, with multiplicities given by ``dims``).

    This layer adds a learnable positional encoding :math:`f: X \rightarrow V`. 

    The positional encoding vector itself is invariant under the :math:`G`-action, i.e.

    .. math::
        f(gx) = \rho(g) f(g^{-1}x) 

    for all :math:`g \in G, x \in X`.

    Args:
        action: the lattice for which the positional encoding is defined
        dims: list :math:`C_0, C_1, \ldots, C_{n-1}` of dimensions for each irrep in the representation V
        use_sparse: whether to use sparse tensors for the projections (can save memory)
    """
    def __init__(
        self,  
        action: GroupAction, 
        dims: Union[List[int], Dict[str, int]],
        use_sparse: bool = True
    ):
        super().__init__()
        assert_all_not_quaternionic(action.group)

        self.action = action
        self.use_sparse = use_sparse

        self.dims = resolve_dims(action.group, dims)
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
        with torch.no_grad():
            for coeff in self.coefficients:
                # xavier uniform for now, we should change this later
                # nn.init.xavier_uniform_(coeff)

                # small random values
                nn.init.uniform_(coeff, a=-0.05, b=0.05)

    def get_positional_encodings(self) -> List[torch.Tensor]:
        return self.proj_calc(self.coefficients)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape :math:`(*, L, C_i, d_i)`
        Returns:
            list of tensors, each of shape :math:`(*, L, C_i, d_i)` with positional encodings added

        Note: the input list will be modified in place
        """
        # (L, Ci, di) for each irrep
        # each tensor can be real or complex depending on the irrep type
        # if complex, di is the complex dimension
        pos_enc = self.get_positional_encodings()

        for i in range(self.num_irreps):
            x[i] = x[i] + pos_enc[i] # (*, L, Ci, di)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(action={self.action}, dims={self.dims}, use_sparse={self.use_sparse})"



# TODO: this is too similar to the above class.
# We should refactor the code to avoid duplication.

class EquivariantInducedPositionalEncoding(nn.Module):
    """
    This is a variant of the equivariant positional encoding that uses the
    induced representation of a subgroup of the symmetry group of the lattice.

    Args:
        pullback_bundle: a :math:`G`-equivariant principal :math:`H`-bundle.
        dims: list of dimensions for each irrep of :math:`H`.
        use_sparse: whether to use sparse tensors for the projections (can save memory).

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
            x: list of tensors, each of shape :math:`(*, L, C_i, d_i)`
        Returns:
            list of tensors, each of shape :math:`(*, L, C_i, d_i)` with positional encodings added
        """
        # (L, Ci, di)
        pos_enc = self.get_positional_encodings()

        for i in range(self.num_irreps):
            x[i] = x[i] + pos_enc[i] # (*, L, Ci, di)
        return x
   