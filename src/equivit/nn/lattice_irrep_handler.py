from ..geometry import Honeycomb, Triangle, decompose_set_action
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

from ..geometry import GroupAction



class GroupActionIrrepProjectionCalculator(nn.Module):
    """
    Often we want to compute the irrep projections of a group action on a set (e.g. the pixels in a patch),
    and then linearly combine these projections, e.g. to obtain symmetry-adapted filters or positional encodings. 
    """
    def __init__(
        self, action: GroupAction, 
        use_sparse: bool = True,
        streams: Optional[List[torch.cuda.Stream]] = None
    ):
        super().__init__()
        # Decompose the representation of the subgroup on the patch lattice 
        # into irreps to get the projections for each irrep.
        # This is a dictionary mapping each irrep name to a list of projections, each of shape (Lpatch, irrep_dim).
        # Note: each projection tensor is a sparse COO tensor.
        self.action = action
        self.projection_bases = decompose_set_action(self.action)
        self.group = self.action.group


        self.num_irreps = len(self.projection_bases)

        # Number of copies of each irrep in the representation of the subgroup on the patch lattice. 
        self.num_irrep_copies = [len(proj) for proj in self.projection_bases.values()]

        self.L = self.action.num_elements
        self.irrep_dims = [irrep.dim for irrep in self.action.group.real_irreps(0).values()]

        self.use_sparse = use_sparse

        if self.use_sparse:
            # more memory efficient to store the projections as sparse tensors, since they are often very sparse
            # but this could compromise speed
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
                self.register_buffer(f'proj_indices{i}', torch.cat(_inds, dim=0))
                # each entry in list has shape (n_copies, max_l, 1, d)
                self.register_buffer(f'proj_values{i}', torch.stack(_vals, dim=0).unsqueeze(-2))
        else:
            # less memory efficient but potentially faster (and simpler)
            for i, (irrep_name, projections) in enumerate(self.projection_bases.items()):
                # each entry in list has shape (n_copies, Lpatch, d)
                self.register_buffer(f'projections{i}', torch.stack([p.to_dense() for p in projections], dim=0))

        if streams is None:
            self.streams = [None for _ in range(self.num_irreps)]
        else:
            self.streams = streams

    def forward(self, coefficients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        coefficients: list of tensors, each of shape (n_copies, Ci) for each irrep

        returns: list of tensors, each of shape (Lpatch, Ci, irrep_dim) 
                    for each irrep, obtained by linearly combining the projections with the coefficients.
        """
        filts = [None for _ in range(self.num_irreps)]

        if self.use_sparse:
            for i in range(self.num_irreps):
                C = coefficients[i].shape[1]
                with torch.cuda.stream(self.streams[i]):
                    filt = torch.sparse_coo_tensor(
                        getattr(self, f'proj_indices{i}').flatten().unsqueeze(0),  # (1, n_copies*max_l)
                        (coefficients[i].unsqueeze(-1).unsqueeze(-2) * getattr(self, f'proj_values{i}')).flatten(0,1), # (n_copies*max_l, Ci, d)
                        size=(self.L, C, self.irrep_dims[i])
                    ).to_dense() # (Lpatch, Ci, d)
                    filts[i] = filt
        else:
            for i in range(self.num_irreps):
                with torch.cuda.stream(self.streams[i]):
                    filt = torch.matmul(coefficients[i].t(), getattr(self, f'projections{i}').flatten(1,2))
                    # (Ci, Lpatch*d) -> (Ci, Lpatch, d)
                    filt = filt.unflatten(1, (self.L, self.irrep_dims[i]))
                    filt = filt.permute(1,0,2) # (Lpatch, Ci, d)
                    filts[i] = filt
        return filts
