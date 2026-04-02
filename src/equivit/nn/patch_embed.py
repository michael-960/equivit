from ..lattices import HexGrid, AbstractTriangleGrid, AbstractHexagonGrid, Honeycomb
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

from ..geometry import Group, Lattice





class EquivariantPatchEmbed:
    """
    Patch embedding layer that respects the symmetries of the hexagonal lattice.
    We use the irreps of the symmetry group to project the input features.
    """

    def __init__(self, 
        patch_lattice: Lattice, 
        in_channels: int, 
        out_channels: List[int],
        subgroup: str = ''
    ):
        self.full_group = patch_lattice.symmetry_group
        self.group, self.group_inclusion = self.full_group.subgroup(subgroup)


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        x_unfolded = x[...,self.patch_inds]

        outs = [None for _ in range(self.n_groups)]
        streams = [torch.cuda.Stream() for _ in range(self.n_groups)]

        for i in range(self.n_groups):
            with torch.cuda.stream(streams[i]):
                outs[i] = torch.einsum('dczl,...cql->...dzq', projections[i], x_unfolded)

        torch.cuda.synchronize()

        return outs





class HexPatchEmbed(nn.Module):
    def __init__(
        self, 
        N: int, div: int,
        in_channels: int,
        out_channels: Tuple[int,int,int]
        # C_a1: int, C_a2: int, C_e: int
    ):
        super().__init__()
        assert N % div == 0
        assert (N//div) % 3 == 0

        Npatch = N // div
        self.m = Npatch // 3

        self.Npatch = Npatch

        self.in_channels = in_channels

        self.hexa = AbstractHexagonGrid(N)
        self.honeycomb = Honeycomb(div)
        self.triangle_patch_grid = AbstractTriangleGrid(Npatch)


        self._setup_patch_indices()

        multiplicity_a1 = self.triangle_patch_grid.filters_a1.shape[0]
        multiplicity_a2 = self.triangle_patch_grid.filters_a2.shape[0]
        multiplicity_e = self.triangle_patch_grid.filters_e.shape[0]

        self.register_buffer('filters_a1', self.triangle_patch_grid.filters_a1)
        self.register_buffer('filters_a2', self.triangle_patch_grid.filters_a2)
        self.register_buffer('filters_e', self.triangle_patch_grid.filters_e)

        C_a1, C_a2, C_e = out_channels        

        self.a1_weights = nn.Parameter(torch.zeros(C_a1, self.in_channels, multiplicity_a1))
        self.a2_weights = nn.Parameter(torch.zeros(C_a2, self.in_channels, multiplicity_a2))
        self.e_weights = nn.Parameter(torch.zeros(C_e, self.in_channels, multiplicity_e))

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.a1_weights)
        nn.init.xavier_uniform_(self.a2_weights)
        nn.init.xavier_uniform_(self.e_weights)

    def _setup_patch_indices(self):
        """
        For each lattice site in the honeycomb, we find the indices of the
        hexagonal lattice sites that belong to the corresponding patch.
        """
        patch_inds = []

        for q in range(self.honeycomb.L):
            i0, j0 = self.honeycomb.index_dec[2][q]
            w, _q = self.honeycomb.index_dec[4][q]
            i, j = i0*self.m, j0*self.m
            inds = []
            for _qpatch in range(self.triangle_patch_grid.L):
                (r,s) = self.triangle_patch_grid.index_1t2[_qpatch]
                if w == 0:
                    a = min([i+j, j-self.m+s, self.m-r])
                    inds.append(
                        self.hexa.index_3t1[i+j-a, j-self.m+s-a, self.m-r-a]
                    )
                else:
                    a = min([i+j, j+self.m-s, -self.m+r])
                    inds.append(
                        self.hexa.index_3t1[i+j-a, j+self.m-s-a, -self.m+r-a]
                    )

            patch_inds.append(inds)

        # return np.array(patch_inds)
        self.patch_inds = np.array(patch_inds)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        :param x: tensor of shape (*, C, L), where L is the number of pixels per patch
        :type x: torch.Tensor

        :return: 3-tuple of features (y_a1, y_a2, y_e), one for each irrep (A1, A2, and E). 
                The shape of each output tensor is 
                - (*, C_a1, L) for A1
                - (*, C_a2, L) for A2
                - (*, C_e, 2, L) for E
                where L is the number of patches
        :rtype: Tuple[torch.Tensor]
        """

        x_unfolded = x[...,self.patch_inds]


        # (C_a1, in_channels, Lpatch)
        a1_matrix = torch.matmul(self.a1_weights, self.filters_a1)
        
        # (C_a2, in_channels, Lpatch)
        a2_matrix = torch.matmul(self.a2_weights, self.filters_a2)

        # (C_e, in_channels, 2, Lpatch)
        e_matrix = torch.matmul(
            self.e_weights, self.filters_e.flatten(1,2)
            ).reshape(-1,self.in_channels,2,self.triangle_patch_grid.L)




        # (C_A1, in_channels, Lpatch), (*, in_channels, Lhoneycomb, Lpatch) --> (*, C_A1, Lhoneycomb)
        y_a1 = torch.einsum('dcl,...cql->...dq', a1_matrix, x_unfolded)

        y_a2 = torch.einsum('dcl,...cql->...dq', a2_matrix, x_unfolded)

        y_e = torch.einsum('dczl,...cql->...dzq', e_matrix, x_unfolded)

        return y_a1, y_a2, y_e


        