from ..geometry import Honeycomb, Triangle, decompose_set_action, induce_and_find_invariant_vectors
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


from ..geometry import GroupAction, GroupRepresentation, GroupElement


class BasisHandler(nn.Module):
    """
    This module does the following:
    It stores a list of P tensors, which we will call basis tensors (although they may not be linearly independent), 
    each of shape (ni, L, di), where i = 0, ..., P-1.

    The forward() method takes as input a list of P tensors, each of shape 
    (ni, Ci), and returns a list of P tensors, each of shape (L, Ci, di).

    (ni, Ci) (ni, L, di) -> (L, Ci, di)

    The specific basis tensors should be specified by subclasses.

    It often happesn that the basis tesnors are sparse in the L dimension.
    The use_sparse flag makes it so that the basis tensors are stored as sparse
    tensors, which can save memory but may be slower.
    """
    def __init__(
        self, 
        irrep_dims: List[int],
        use_sparse: bool = True 
    ):
        super().__init__()
        self.use_sparse = use_sparse

        self.irrep_dims = irrep_dims

        basis_vectors_list = self.get_basis_vectors_list()

        self.num_irreps = len(basis_vectors_list)
        assert len(self.irrep_dims) == self.num_irreps, "Length of irrep_dims should match number of irreps (length of basis_vectors_list)."

        # deduce num_elements from the basis vectors
        self.num_elements = None
        for basis_vectors in basis_vectors_list:
            if len(basis_vectors) > 0:
                self.num_elements = basis_vectors[0].shape[0]
                break
        if self.num_elements is None:
            raise ValueError("Could not determine number of elements from basis vectors. Please ensure that get_basis_vectors_list() returns a list of lists of tensors.")

        self.num_irrep_copies = [len(basis_vectors) for basis_vectors in basis_vectors_list]

        self.setup_basis_tensors(basis_vectors_list, use_sparse=use_sparse)

    def get_basis_vectors_list(self):
        raise NotImplementedError("Subclasses should implement this method to return the list of basis vectors for each irrep.")

    def setup_basis_tensors(self, basis_vectors_list: List[list], use_sparse: bool):
        """
        basis_vectors_list: list of lists of sparse COO tensors, each of shape (|X|, irrep_dim), 
        where the outer list is over irreps.
        """
        if use_sparse:
            # more memory efficient to store the basis tensors as sparse tensors, since they are often very sparse
            # but this could compromise speed
            for i, basis_vectors in enumerate(basis_vectors_list):
                max_l = max([p.indices().shape[1] for p in basis_vectors])
                d = self.irrep_dims[i]
                _inds = []
                _vals = []
                for p in basis_vectors:
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
            for i, basis_vectors in enumerate(basis_vectors_list):
                # each entry in list has shape (n_copies, num_elements, d)
                self.register_buffer(f'projections{i}', torch.stack([p.to_dense() for p in basis_vectors], dim=0))


    def forward(self, coefficients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        coefficients: list of tensors, each of shape (ni, Ci) for each irrep

        returns: list of tensors, each of shape (Lpatch, Ci, irrep_dim) 
                    for each irrep, obtained by linearly combining the projections with the coefficients.
        """
        filts = [None for _ in range(self.num_irreps)]

        if self.use_sparse:
            for i in range(self.num_irreps):
                C = coefficients[i].shape[1]
                # with torch.cuda.stream(self.streams[i]):
                filt = torch.sparse_coo_tensor(
                    getattr(self, f'proj_indices{i}').flatten().unsqueeze(0),  # (1, n_copies*max_l)
                    (coefficients[i].unsqueeze(-2).unsqueeze(-1) * getattr(self, f'proj_values{i}')).flatten(0,1), # (n_copies*max_l, Ci, d)
                    size=(self.num_elements, C, self.irrep_dims[i])
                ).to_dense() # (Lpatch, Ci, d)
                filts[i] = filt
        else:
            for i in range(self.num_irreps):
                # with torch.cuda.stream(self.streams[i]):
                filt = torch.matmul(coefficients[i].t(), getattr(self, f'projections{i}').flatten(1,2))
                # (Ci, Lpatch*d) -> (Ci, Lpatch, d)
                filt = filt.unflatten(1, (self.num_elements, self.irrep_dims[i]))
                filt = filt.permute(1,0,2) # (Lpatch, Ci, d)
                filts[i] = filt
        return filts



class GroupActionIrrepProjectionCalculator(BasisHandler):
    """
    Given an action of a group G on a set X, this module calculates the
    projections onto the isotypic components of the induced representation of G
    on the space of functions X -> R.
    """
    def __init__(
        self, 
        action: GroupAction, 
        use_sparse: bool = True,
    ):
        self.action = action
        self.group = action.group
        irreps = self.action.group.real_irreps()
        super().__init__([irrep.dim for irrep in irreps.values()],
                         use_sparse=use_sparse)

    def get_basis_vectors_list(self):
        self.projection_bases = decompose_set_action(self.action)

        return list(self.projection_bases.values())



class InducedRepresentationInvariantSubspaceCalculator(BasisHandler):
    """
    Given:
        - an action of a group G on a set X
        - a normal subgroup H of G that contains all stabilizer subgroups
        - a choice of basepoint for each G-orbit in X
        - a choice of representative of each coset of H in G

    this module calculates the space C(G, V) of invariant functions G -> V for each irrep V of H, where the action 
    of G on C(G, V) is induced from that of H on V.
    """
    def __init__(
        self, 
        action: GroupAction,
        subgroup_args: tuple,
        representatives: List[GroupElement],
        basepoints: List[int],
        use_sparse: bool = True
    ):
        self.action = action
        self.subgroup_args = subgroup_args
        self.subgroup_incl = self.action.group.subgroup(*subgroup_args)
        self.subgroup = self.subgroup_incl.source
        self.representatives = representatives
        self.basepoints = basepoints

        irreps = self.subgroup.real_irreps()
        super().__init__([irrep.dim for irrep in irreps.values()], use_sparse)

    def get_basis_vectors_list(self):
        irreps = self.subgroup.real_irreps()
        invariant_vectors = []

        for irrep_name, irrep in irreps.items():
            # list of sparse COO tensors, each of shape (|X|, irrep_dim)
            basis_vectors = induce_and_find_invariant_vectors(
                self.action,
                self.subgroup_args,
                irrep,
                self.representatives,
                self.basepoints
            )
            invariant_vectors.append(basis_vectors)
        return invariant_vectors

