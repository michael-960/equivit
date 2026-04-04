from ..geometry import Honeycomb, Triangle, decompose_set_action
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

from ..geometry import Group, Lattice




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

        self.full_group = patch_lattice.symmetry_group
        self.subgroup_inclusion = self.full_group.subgroup(*subgroup)

        # The group that we will use to compute the irrep projections 
        # is the specified subgroup specified by the user.
        # This is a subgroup of the full symmetry group of the patch lattice.
        self.group = self.subgroup_inclusion.source
        self.irreps = self.group.real_irreps()
        self.num_irreps = len(self.irreps)
        self.irrep_dims = [irrep.dim for irrep in self.irreps.values()]

        assert len(self.out_channels) == len(self.num_irreps), f"Number of output channels ({len(self.out_channels)}) must match number of irreps ({len(self.num_irreps)})"

        # Pull back the action of the full symmetry group of the patch lattice 
        # to get an action of the subgroup on the patch lattice.
        self.action = patch_lattice.action.pullback(self.subgroup_inclusion)

        # Decompose the representation of the subgroup on the patch lattice 
        # into irreps to get the projections for each irrep.
        # This is a dictionary mapping each irrep name to a list of projections, each of shape (Lpatch, irrep_dim).
        # Note: each projection tensor is a sparse COO tensor.
        self.projection_bases = decompose_set_action(self.action)

        # Number of copies of each irrep in the representation of the subgroup on the patch lattice. 
        self.num_irrep_copies = [len(proj) for proj in self.projection_bases.values()]

        self.coefficients = nn.ParameterList(
            [nn.Parameter(
                torch.zeros((self.num_irrep_copies[i], 1, in_channels, 1, out_channels[i]))
            ) 
                for i in range(self.num_irreps)]
        )

        self.proj_indices = []
        self.proj_values = []

        for i, (irrep_name, projections) in enumerate(self.projection_bases.items()):
            max_l = max([p.indices().shape[1] for p in projections])
            d = self.irrep_dims[i]
            _inds = []
            _vals = []
            for p in projections:
                n_pad = max_l - p.indices().shape[1]
                _inds.append(torch.cat([
                                p.indices(), torch.zeros((1, n_pad), dtype=torch.long)
                            ], dim=1))
                _vals.append(torch.cat([
                    p.values(), torch.zeros((n_pad,d), dtype=p.values().dtype)], dim=0).to(torch.float32)
                    )

            # (n_copies, max_l)  
            self.proj_indices.append(torch.cat(_inds, dim=0))

            # (n_copies, max_l, 1, d, 1)
            self.proj_values.append(torch.stack(_vals, dim=0).unsqueeze(-2).unsqueeze(-1))

        # Ideally we want to parallelize the computation for different irreps
        # using different CUDA streams
        if streams is None:
            self.streams = [None for _ in range(self.num_irreps)]
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
        # x_unfolded = x[...,self.patch_inds]

        # x should already have shape (*, Lpatch, C), where Lpatch is the number of pixels in each patch.
        # the 'patchification' of the input should be done outside of this module to allow more flexibility in the patch structure.

        outs = [None for _ in range(self.num_irreps)]

        for i in range(self.num_irreps):
            with torch.cuda.stream(self.streams[i]):
                filt = torch.sparse_coo_tensor(
                    self.proj_indices[i].flatten().unsqueeze(0),  # (1, n_copies*max_l)
                    (self.coefficients[i] * self.proj_values[i]).flatten(0,1), # (n_copies*max_l, C, d, Ci)
                    size=(self.action.num_elements,  self.in_channels, self.irrep_dims[i], self.out_channels[i])
                ).to_dense() # (Lpatch, C, d, Ci)
                outs[i] = (x.flatten(-2) @ filt.flatten(0,1).flatten(1,2)).unflatten(-1, (self.irrep_dims[i], self.out_channels[i])) # (*, d, Ci)

        # Question: how should we synchronize the streams here? 
        if self.has_streams:
            default_stream = torch.cuda.default_stream()
            for stream in self.streams:
                default_stream.wait_stream(stream)

        return outs





