import numpy as np
import torch
from typing import Type, List

from .base import Group
from .action import GroupRepresentation, GroupAction, GroupElement


def find_irrep_components(
    rep: GroupRepresentation, 
    irrep: GroupRepresentation,
    clip_small_values=0.
):
    """
    rep: a representation over R 
    irrep: an irreducible representation over R
    group: the group class

    clip_small_values: small output entries (due to numerical error) will be set to zero

    Return: an array of shape (multiplicity, irrep_dim, rep_dim)

    warning: only works when both rep and irrep are orthogonal
    """
    assert rep.group is irrep.group
    group = irrep.group

    rep_dim = rep(group.identity()).shape[0]

    # If the irrep is of complex type, then Hom(irrep, irrep) is 2-dimensional over R. 
    # This would result in linearly dependent vectors if we didn't take care of it.

    # Calculate the Frobenius-Schur indicator
    # This is 1 if irrep is of real type, 0 if complex, 
    # and -2 (because we work over R) if quaternionic
    fb_ind = irrep.frobenius_schur_indicator()

    if abs(fb_ind) < 1e-7:
        rep_type = 'complex'
    elif abs(fb_ind-1) < 1e-7:
        rep_type = 'real'
    else:
        raise ValueError(f'The Frobenius-Schur indicator is {fb_ind}, which is not supported.')

    # calculate the matrices of the representation Hom(U, V),
    # where U is the irrep and V is the representation

    if rep_type == 'complex':
        complex_structure = irrep.complex_structure
        # replace the irrep by its complex counterpart (note: this is different from complexification)
        irrep = irrep.as_complex()

    _hom_rep = dict()
    for g in group:
        _hom_rep[g] = irrep(g.inv()).T[:,:,None,None] * rep(g)

    # compute the invariant subspace of Hom(irrep, rep)
    M = np.zeros((irrep.dim, irrep.dim, rep_dim, rep_dim), dtype=np.complex128 if rep_type=='complex' else np.float64)
    for g in group:
        M[:] += _hom_rep[g] / len(group)

    projection = M.transpose(0,2,1,3).reshape(irrep.dim*rep_dim, irrep.dim*rep_dim)
    
    eigres = np.linalg.eigh(projection)
    
    invariant_indices = [i for i in range(eigres[0].shape[0]) if np.abs(eigres[0][i] - 1.) < 1e-8]

    # (multiplicity, irrep_dim, rep_dim)
    res = eigres[1][:,invariant_indices].reshape(irrep.dim,rep_dim, -1).transpose(2,0,1)

    if rep_type == 'complex':
        res = complex_structure.dual_vector_c2r(res, axis=1)

    res = np.where(np.abs(res) < clip_small_values, 0., res)
    return res



def decompose_set_action(action: GroupAction):
    """
    Given a group action on a set, decompose the corresponding linear representation on
    the vector space spanned by the set elements into irreps. 

    action: a GroupAction object
    return: a dictionary mapping each irrep name to a list of projections, each
        of shape (n, irrep_dim), where n is the size of the set and irrep_dim is the
        dimension of the irrep. Each projection is a sparse matrix in COO format.
    """
    group = action.group

    irreps = group.real_irreps()

    # First, compute the orbits of the group action on the set. 
    orbits = action.orbits()

    irrep_projections = {irrep_name: [] for irrep_name in irreps.keys()}

    # The linear representation is a direct sum of the linear representations on
    # the orbits, so we can decompose each orbit separately and combine the
    # results.
    for orbit in orbits:
        orbit_rep = action.restrict_action(orbit).to_linear_representation()
        for irrep_name, irrep in irreps.items():
            projections = torch.tensor(find_irrep_components(orbit_rep, irrep, clip_small_values=1e-10))

            for i in range(projections.shape[0]):
                irrep_projections[irrep_name].append(
                    torch.sparse_coo_tensor(
                        indices=torch.tensor(orbit).unsqueeze(0),
                        values=projections[i].T,
                        size=(action.num_elements, irrep.dim)
                    ).coalesce()
                )

    return irrep_projections


def induce_and_find_invariant_vectors(
    action: GroupAction,
    subgroup_args: tuple,
    subgroup_representation: GroupRepresentation,
    representatives: List[GroupElement],
    basepoints: List[int]
):
    """
    Find a basis for the invariant subspace of the induced representation.

    basepoints: list of integers, one for each G-orbit of X. 

    Note: each basepoint is an integer in [0, |O|), where O is the corresponding G-orbit.
    """
    orbits = action.orbits()
    subgroup_incl = action.group.subgroup(*subgroup_args)
    subgroup = subgroup_incl.source

    dim = subgroup_representation(subgroup.identity()).shape[0]

    trivial_rep = list(action.group.real_irreps().values())[0]

    invariant_vectors = []

    for orbit, basepoint in zip(orbits, basepoints):
        restricted_action = action.restrict_action(orbit)
        induced_rep = restricted_action.induce_from(subgroup_args, subgroup_representation, representatives, basepoint)

        # shape: (number of invariant vectors, induced_rep_dim)
        inv_vecs = find_irrep_components(induced_rep, trivial_rep, clip_small_values=1e-9)[:,0,:].reshape(-1, len(orbit), dim)

        for i in range(inv_vecs.shape[0]):
            invariant_vectors.append(
                torch.sparse_coo_tensor(
                    indices=torch.tensor(orbit).unsqueeze(0),
                    values=inv_vecs[i],
                    size=(action.num_elements, dim)
                ).coalesce()
            )

    return invariant_vectors
