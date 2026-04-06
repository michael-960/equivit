import torch
from .base import Lattice
from ..groups import GroupAction, GroupRepresentation, GroupElement, find_irrep_components
from typing import List, Tuple, Dict, Any


class AdvancedLattice(Lattice):
    """
    An advanced lattice is a G-set X with the following additional data:
    - A choice of normal subgroup H of G that contains Stab(x) for all x in X.
    - For each G-orbit O of X, a choice of H-orbit O_0 \subset O
    - For each coset of H in G, a choice of representative g

    Given the above data, for each orbit O \subset X, there is a canonical G-set isomorphism 
    G / H -> O / H
    given by g.H \mapsto g.O_0

    where O_0 is the chosen H-orbit of O.

    If in addition we are given an H-representation (rho, V), 
    then we can construct a G-representation on the space of functions X -> V.
    """


    ...




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
