import numpy as np


def find_irrep_component(
    rep, irrep, group, 
    clip_small_values=0.
):
    """
    rep: either a dictionary or a function that takes a group element and outputs a matrix
    irrep: an irreducible representation
    group: the group class

    clip_small_values: small output entries (due to numerical error) will be set to zero

    warning: only works when both rep and irrep are orthogonal
    """

    # if rep is a dict, lambdify it
    if type(rep) is dict:
        return find_irrep_component(
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