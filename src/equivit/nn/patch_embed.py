
from ..geometry import Honeycomb, Triangle, decompose_set_action
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

from ..geometry import Group, Lattice, GroupAction

from .lattice_irrep_handler import GroupActionIrrepProjectionCalculator


class Patchify(nn.Module):
    patch_inds: torch.Tensor
    def __init__(self):
        super().__init__()
        self._setup_patch_indices()

    def _setup_patch_indices(self):
        raise NotImplementedError("This method should be implemented by subclasses to set up the patch indices for the specific lattice structure of the patches.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should have shape (*, L, C)

        # assert H % self.patch_size == 0 and W % self.patch_size == 0, f"Image size ({H}x{W}) must be divisible by patch size ({self.patch_size})"
        # x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # (B, C, H//P, W//P, P, P)
        # x = x.contiguous().view(B, C, -1, self.patch_size*self.patch_size) # (B, C, num_patches, patch_size*patch_size)

        x_unfolded = x[...,self.patch_inds,:] # (*, num_patches, Lpatch, C)
        return x_unfolded


class SquarePatchify(Patchify):
    def __init__(self, img_size: int, patch_size: int):
        self.img_size = img_size
        self.patch_size = patch_size
        super().__init__()

    def _setup_patch_indices(self):
        L = self.img_size**2
        P = self.patch_size

        patch_inds = []

        for i in range(0, self.img_size, P):
            for j in range(0, self.img_size, P):
                _inds = []
                for di in range(P):
                    for dj in range(P):
                        _inds.append((i+di)*self.img_size + (j+dj))
                patch_inds.append(_inds)
        self.patch_inds = torch.tensor(patch_inds, dtype=torch.long)




class EquivariantPatchEmbed(nn.Module):
    """
    Patch embedding layer that respects the symmetries of a lattice.
    We use the irreps of the symmetry group to project the input features.
    """
    def __init__(self, 
        patch_lattice: Lattice, 
        in_channels: int, 
        out_channels: List[int],
        subgroup: tuple,
        streams: List[torch.cuda.Stream]=None
    ):
        """
        patch_lattice: the lattice structure of each patch. 
        in_channels: number of input channels
        out_channels: list of output channels for each irrep
        subgroup: a tuple specifying the subgroup of the full symmetry group of the patch lattice that we want to respect.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.proj_calc = GroupActionIrrepProjectionCalculator(
            patch_lattice.action.pullback(patch_lattice.symmetry_group.subgroup(*subgroup))
        )
        assert len(self.out_channels) == len(self.proj_calc.num_irreps), f"Number of output channels ({len(self.out_channels)}) must match number of irreps ({len(self.proj_calc.num_irreps)})"
        self.L = self.proj_calc.L
        self.irrep_dims = self.proj_calc.irrep_dims
        self.group = self.proj_calc.group

        self.coefficients = nn.ParameterList(
            [nn.Parameter(
                torch.zeros((self.proj_calc.num_irrep_copies[i], in_channels*out_channels[i]))
            ) 
                for i in range(self.proj_calc.num_irreps)]
        )

        # Ideally we want to parallelize the computation for different irreps
        # using different CUDA streams
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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: (*, Lpatch, C), where Lpatch is the number of pixels in each patch and C is the number of input channels.
        returns: a list of tensors, each of shape (*, Ci, di),
                where di is the dimension of each irrep and Ci is the number of output channels for that irrep.
        """
        # x_unfolded = x[...,self.patch_inds]

        # x should already have shape (*, Lpatch, C), where Lpatch is the number of pixels in each patch.
        # the 'patchification' of the input should be done outside of this module to allow more flexibility in the patch structure.

        outs = [None for _ in range(self.proj_calc.num_irreps)]

        x = x.flatten(-2) # (*, Lpatch*C)

        # each entry has shape (Lpatch, C*Ci, di) -> (Lpatch*C, Ci*di)
        filts = self.proj_calc(self.coefficients).view(self.L*self.in_channels, self.out_channels[i]*self.irrep_dims[i])

        for i in range(self.proj_calc.num_irreps):
            with torch.cuda.stream(self.streams[i]):
                outs[i] = (x @ filts[i]).unflatten(-1, (self.out_channels[i], self.irrep_dims[i])) # (*, Ci, di)
        return outs





