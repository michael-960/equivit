from matplotlib import projections
import torch
import torch.nn as nn
from typing import List, Optional, Callable

from ..geometry import GroupAction, decompose_set_action, Group



class EquivariantNonlinear(nn.Module):
    """
    inverse Fourier transform -> pointwise nonlinearity -> Fourier transform
    """
    def __init__(self,
        group: Group,
        homogeneous_space_copies: List[int],
        activation: Callable=nn.ReLU(),
    ):
        """
        Args:
            group: the group for which the equivariant nonlinearity is defined
            homogeneous_space_copies: list of number of copies for each homogeneous space
            activation: activation function to use in the pointwise nonlinearity
        """
        super().__init__()
        self.group = group
        self.num_irreps = len(group.real_irreps())
        self.homogeneous_space_copies = homogeneous_space_copies
        self.num_homog_spaces = len(homogeneous_space_copies)
        
        self.activation = activation

        self.homog_actions = self.group.all_homogeneous_space_actions()
        assert len(homogeneous_space_copies) == len(self.homog_actions), "Length of homogeneous_space_copies must match the number of homogeneous spaces of the group."   

        self.fouriers = nn.ModuleList([Fourier(action) for action in self.homog_actions])

        # the i-th item of this list is the list of irrep multiplicities of the i-th homogeneous spac
        self.multiplicities = [
            fourier.irrep_multiplicities for fourier in self.fouriers
        ]

        # for the i-th irrep we need to allocate homogeneous_space_copies[n] * multiplicities[n][i]  channels of x[i]
        self.split_sizes = []
        for i in range(self.num_irreps):
            _ = []
            for n in range(len(self.homog_actions)):
                _.append(self.homogeneous_space_copies[n] * self.multiplicities[n][i])
            self.split_sizes.append(_)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        x: list of tensors, each of shape (*, Ci, di), where di is the dimension of the i-th irrep
        return: list of tensors, each of shape (*, Ci, di)
        """

        # TODO: too many for loops! optimize this

        # first, decompose each entry in x into chunks 
        x = [
            torch.split(y, self.split_sizes[i], dim=-2)
            for i, y in enumerate(x)
        ]

        # "transpose"
        x = list(zip(*x))

        outs = [None for _ in range(self.num_homog_spaces)]

        for n in range(len(self.homog_actions)):
            # transform to homogeneous space basis

            # each entry in x[n] has shape (*, homogeneous_space_copies[n]*multiplicities[n][i], di)
            x_homog = self.fouriers[n].inverse_transform([
                z.unflatten(-2, (self.homogeneous_space_copies[n], self.multiplicities[n][i])) for i, z in enumerate(x[n])
            ])
            # the output of inverse_transform has shape (*, homogeneous_space_copies[n], |X_n|) where X_n is the n-th homogeneous space

            # apply pointwise nonlinearity in homogeneous space
            x_homog = self.activation(x_homog)

            # transform back to real space
            # we get a list of tensors, each of shape (*, homogeneous_space_copies[n], multiplicities[n][i], di)
            y = self.fouriers[n].transform(x_homog)
            outs[n] = [z.flatten(-3, -2) for z in y]

        # recombine
        # outs[n][i] has shape (*, homogeneous_space_copies[n]*multiplicities[n][i], di)

        return [torch.cat([outs[n][i] for n in range(self.num_homog_spaces)], dim=-2) 
                for i in range(self.num_irreps)]


def normalize_columns(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=0)

class Fourier(nn.Module):
    """
    Given an action of a group G on a set X, 
    decompose the representation of G on functions on X into irreps, 
    and implement the corresponding Fourier transform and inverse Fourier transform.

    Note: this module should be used for small X (say |X| < 100), since the Fourier transform is
    implemented as a dense matrix multiplication.
    """
    def __init__(self, action: GroupAction):
        super().__init__()
        self.action = action
 
        projections = decompose_set_action(action)
        self.irrep_multiplicities = [len(p) for p in projections.values()]
        self.irrep_dims = [irrep.dim for irrep in action.group.real_irreps().values()]
        assert len(self.irrep_multiplicities) == len(self.irrep_dims), "Number of irreps in the decomposition does not match the number of irreps of the group. Something went wrong."
        self.split_sizes = [m*d for m, d in zip(self.irrep_multiplicities, self.irrep_dims)]
    
        all_projections = []
        for irrep_name, proj in projections.items():
            all_projections.extend([normalize_columns(p.to_dense()) for p in proj])
    
        matrix = torch.cat(all_projections, dim=-1)

        # (|X|, |X|)
        self.register_buffer('matrix', matrix.to(torch.float32))

    def transform(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        real space to frequency (or momentum or irrep) space
        x: (*, |X|)
        return: list of tensors, each of shape (*, Ri, di), where Ri is the multiplicity of the i-th irrep and di is the dimension of the i-th irrep.
        """
        y = torch.matmul(x, self.matrix)
        chunks = torch.split(y, self.split_sizes, dim=-1)
        return [chunk.unflatten(-1, (m, d)) 
                for chunk, m, d in zip(chunks, self.irrep_multiplicities, self.irrep_dims)
                ]

    def inverse_transform(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        frequency space to real space
        x: list of tensors, each of shape (*, Ri, di), where Ri is the multiplicity of the i-th irrep and di is the dimension of the i-th irrep.
        """
        return torch.matmul(torch.cat(
            [z.flatten(-2, -1) for z in x], dim=-1),
            self.matrix.t()
        )


