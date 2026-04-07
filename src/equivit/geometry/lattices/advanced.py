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



