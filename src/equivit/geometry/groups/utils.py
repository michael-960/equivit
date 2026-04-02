import numpy as np
import torch
from typing import Type

from .base import Group
from .action import GroupRepresentation, GroupAction


def find_irrep_components(
    rep: GroupRepresentation, irrep: GroupRepresentation,
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

    if type(rep) is dict:
        raise TypeError(f'Representations can no longer be specified by a dictionary. Please pass an instance of GroupRepresentation instead.')
        # return find_irrep_components(
        #     lambda g: rep[g],
        #     irrep, group,
        #     clip_small_values=clip_small_values
        # )

    rep_dim = rep(group.identity()).shape[0]

    # If the irrep is of complex type, then Hom(irrep, irrep) is 2-dimensional over R. 
    # This would result in linearly dependent vectors if we didn't take care of it.

    # Calculate the Frobenius-Schur indicator
    # This is 1 if irrep is of real type, 0 if complex, and -1 if quaternionic
    fb_ind = irrep.frobenius_schur_indicator()

    if abs(fb_ind) < 1e-7:
        rep_type = 'complex'
    elif abs(fb_ind-1) < 1e-7:
        rep_type = 'real'
    elif abs(fb_ind+1) < 1e-7:
        # maybe we can deal with quaternionic irreps in the future
        raise ValueError(f'Quaternionic irreps are not supported (Frobenius-Schur indicator of the irrep is -1)')
    else:
        raise ValueError(f'The Frobenius-Schur indicator is {fb_ind}, which is impossible for an irrep.')

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
        res = complex_structure.vector_c2r(res, axis=1)

    res = np.where(np.abs(res) < clip_small_values, 0., res)
    return res



def decompose_set_action(action: GroupAction):
    """
    Given a group action on a set, decompose the corresponding representation on
    the vector space spanned by the set elements into irreps. 

    irreps: the Enum type containing the irreps of the group
    """
    group = action.group

    irreps = group.real_irreps()

    _dots = set(range(action.num_elements))
    orbits = []
    while len(_dots) > 0:
        dot = next(iter(_dots))
        orbit = []
        for g in group:
            i = action(g)[dot]
            if i not in orbit: orbit.append(i)
        orbits.append(orbit)
        _dots = _dots.difference(orbit)

    irrep_projections = {irrep_name: [] for irrep_name in irreps.keys()}
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
                    )
                )

    return irrep_projections

