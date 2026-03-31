import numpy as np
import torch
from typing import Type

from .base import Group


def find_irrep_components(
    rep, irrep, group, 
    clip_small_values=0.
):
    """
    rep: either a dictionary or a function that takes a group element and outputs a matrix
    irrep: an irreducible representation
    group: the group class

    clip_small_values: small output entries (due to numerical error) will be set to zero

    Return: an array of shape (multiplicity, irrep_dim, rep_dim)

    warning: only works when both rep and irrep are orthogonal
    """

    # if rep is a dict, lambdify it
    if type(rep) is dict:
        return find_irrep_components(
            lambda g: rep[g],
            irrep, group,
            clip_small_values=clip_small_values
        )


    rep_dim = rep(group.e).shape[0]


    # calculate the matrices of the representation Hom(U, V),
    # where U is the irrep and V is the representation
    _hom_rep = dict()
    for g in group:
        _hom_rep[g] = irrep(g.inv()).T[:,:,None,None] * rep(g)
    

    # compute the invariant subspace of Hom(U, V)
    M = np.zeros((irrep.dim, irrep.dim, rep_dim, rep_dim))
    for g in group:
        M[:] += _hom_rep[g] / len(group)

    projection = M.transpose(0,2,1,3).reshape(irrep.dim*rep_dim, irrep.dim*rep_dim)
    
    eigres = np.linalg.eigh(projection)
    
    invariant_indices = [i for i in range(eigres[0].shape[0]) if np.abs(eigres[0][i] - 1.) < 1e-8]

    res = eigres[1][:,invariant_indices].reshape(irrep.dim,rep_dim, -1).transpose(2,0,1)

    res = np.where(np.abs(res) < clip_small_values, 0., res)

    return res


def restrict_action(action_dict, indices):
    """
    Given a group action on a set, return the restricted action on a subset of the set.
    
    Return: a dictionary mapping group elements to the restricted action on the subset. 
    Each value of the dictionary is an array of shape (m,), where m is the size of the subset.
    """

    if type(indices) is not list:
        indices = list(indices)
        
    assert len(indices) == len(set(indices)), "repeated indices are not allowed"

    restricted_action_dict = dict()
    raw_index_to_new_index = {ind: i for i, ind in enumerate(indices)}

    try:
        for g in action_dict.keys():
            restricted_action_dict[g] = np.array(
                [raw_index_to_new_index[action_dict[g][i]] for i in indices]
            )
    except KeyError:
        raise ValueError("the subset is not invariant under the group action")

    return restricted_action_dict



def get_set_action_rep_matrices(action_dict, indices=None):
    """
    Given a group action on a set, return the representation matrices of the corresponding representation on the vector space spanned by the set elements.
    
    Return: a dictionary mapping group elements to representation matrices. 
    Each value of the dictionary is a permutation matrix of shape (n, n), where n is the size of the set.
    """

    if indices is not None:
        restricted_action_dict = restrict_action(action_dict, indices)
        return get_set_action_rep_matrices(restricted_action_dict, indices=None)
       
    rep_matrices = dict()
    n = len(action_dict[next(iter(action_dict.keys()))])

    all_inds = [i for i in range(n)]

    for g in action_dict.keys():
        rep_matrices[g] = np.zeros((n, n), dtype=np.float64)
        rep_matrices[g][action_dict[g], all_inds] = 1.

    return rep_matrices



def decompose_set_action(action_dict, group: Type[Group]):
    """
    Given a group action on a set, decompose the corresponding representation on
    the vector space spanned by the set elements into irreps. 

    irreps: the Enum type containing the irreps of the group
    """
    L = len(action_dict[next(iter(action_dict.keys()))])

    irreps = group.irreps()

    _dots = set(range(L))
    orbits = []
    while len(_dots) > 0:
        dot = next(iter(_dots))
        orbit = []
        for g in group:
            i = action_dict[g][dot]
            if i not in orbit: orbit.append(i)
        orbits.append(orbit)
        _dots = _dots.difference(orbit)

    irrep_projections = {irrep: [] for irrep in irreps}
    for orbit in orbits:
        _rep_matrices = get_set_action_rep_matrices(action_dict, orbit)
        for irrep in irreps:
            projections = torch.tensor(find_irrep_components(_rep_matrices, irrep, group, clip_small_values=1e-11))

            for i in range(projections.shape[0]):
                irrep_projections[irrep].append(
                    torch.sparse_coo_tensor(
                        indices=torch.tensor(orbit).unsqueeze(0),
                        values=projections[i].T,
                        size=(L, irrep.dim)
                    )
                )
    return irrep_projections

