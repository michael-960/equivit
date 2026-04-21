from __future__ import annotations
from enum import Enum
import numpy as np
from typing import Dict, List, Any, Union, overload, TYPE_CHECKING, Callable, Tuple

from .base import Group, GroupElement, GroupHomomorphism

from .representations import GroupRepresentation

if TYPE_CHECKING:
    import torch

# Group actions and representations



class GroupAction:
    r"""
    Group action on the set :math:`\{0, 1, 2, ..., n-1\}`

    The action is specified by a dictionary mapping each group element :math:`g` to a list of integers
    :math:`[m_0, m_1, m_2, ..., m_{n-1}]`, indicating that :math:`gx = m_x` for each :math:`x \in \{0, 1, 2, ..., n-1\}`.

    Args:
        group: the group that acts on the set
        action_dict: a dictionary mapping each group element to a list of
            integers specifying the action on the set. Each list should be a
            permutation of :math:`[0, 1, 2, ..., n-1]`.
    """
    def __init__(self, group: Group, action_dict: Dict[GroupElement, List[int]]):
        self.group = group
        self.action_dict = action_dict
        self.num_elements = len(action_dict[next(iter(action_dict.keys()))])

        _elements = [_ for _ in range(self.num_elements)]
        for g in group:
            assert len(action_dict[g]) == self.num_elements
            assert set(action_dict[g]) == set(_elements)

    def __call__(self, g: GroupElement) -> List[int]:
        assert g.group is self.group, f"Group element {g} does not belong to the group of this action ({self.group})."
        return self.action_dict[g]

    def validate(self):
        for g in self.group:
            for h in self.group:
                if not np.all(np.array(self(g*h)) == np.array(self(g))[np.array(self(h))]):
                    raise ValueError(f'The action is not valid: the action of {g}*{h} is not equal to the composition of the actions of {g} and {h}.')


    def restrict_action(self, indices: List[int]) -> GroupAction:
        """
        Return the restricted action on a subset of the set.

        Args:
            indices: a list of integers specifying the indices of the subset
                on which to restrict the action. The subset must be invariant
                under the group action.

        Returns: 
            The restricted group action on the subset specified by indices. The
            group is the same as the original action, and the action_dict is
            obtained by restricting the original action_dict to the specified
            indices. The elements are relabeled to be in the range ``[0, len(indices)-1]`` according to the order of indices.
        """
        if type(indices) is not list:
            indices = list(indices)
            
        assert len(indices) == len(set(indices)), "repeated indices are not allowed"

        restricted_action_dict = dict()
        raw_index_to_new_index = {ind: i for i, ind in enumerate(indices)}

        try:
            for g in self.action_dict.keys():
                restricted_action_dict[g] = np.array(
                    [raw_index_to_new_index[self.action_dict[g][i]] for i in indices]
                )
        except KeyError:
            raise ValueError("the subset is not invariant under the group action")

        return GroupAction(self.group, restricted_action_dict)

    def to_linear_representation(self) -> GroupRepresentation:
        r"""
        Returns: 
            an instance of :class:`GroupRepresentation` that consists of real matrices
            of the corresponding representation on the vector space spanned by the
            set elements.

        Note:
            - The resulting representation is given by permutation matrices, so they are very sparse.
            - This method should only be used for small sets.
        """
        rep_matrices = dict()

        all_inds = [i for i in range(self.num_elements)]

        for g in self.group:
            rep_matrices[g] = np.zeros((self.num_elements, self.num_elements), dtype=np.float64)
            rep_matrices[g][self.action_dict[g], all_inds] = 1.
        
        return GroupRepresentation(self.group, rep_matrices)


    def pullback(self, homomorphism: GroupHomomorphism) -> GroupAction:
        r"""
        Given a group action :math:`G\rightarrow \mathrm{Aut}(X)` and a homomorphism :math:`\varphi: H\rightarrow G`, 
        there is a natural action :math:`H\rightarrow \mathrm{Aut}(X)`, called the restriction or the pullback.

        Args:
            homomorphism: a group homomorphism :math:`\varphi: H\rightarrow G`
        
        Returns:
            The pullback of the action along the homomorphism, which is a group
            action of H on the same set X. 
        Note:
            This is different from restricting the action to a subset of X that is invariant under the subgroup H, 
            which is implemented in the :meth:`restrict_action` method.
        """
        action_dict = dict()
        for g in homomorphism.source:
            action_dict[g] = self.action_dict[homomorphism(g)]

        return GroupAction(homomorphism.source, action_dict)

    def irrep_multiplicities(self) -> Dict[str, int]:
        """
        Return the multiplicity of each real irrep in the 
        representation of the group on the vector space spanned functions from the set to :math:`\mathbb{R}`.

        Returns: 
            A dictionary mapping each irrep name to its multiplicity in the decomposition.
        """
        from .utils import decompose_set_action
        projections = decompose_set_action(self)

        return {irrep_name: len(proj) for irrep_name, proj in projections.items()}

    def orbits(self) -> List[List[int]]: 
        """
        Decompose the set into orbits under the group action, and return a list
        of orbits. Each orbit is represented as a list of indices of the
        elements in the orbit.
        """
        elements = [i for i in range(self.num_elements)]
        # _dots = set(range(self.num_elements))
        orbits = []
        while len(elements) > 0:
            x = elements[0]
            orbit = []
            for g in self.group:
                i = self(g)[x]
                if i not in orbit: orbit.append(i)
            orbits.append(orbit)
            # _dots = _dots.difference(orbit)
            elements = [e for e in elements if e not in orbit]
        return orbits

    def is_transitive(self) -> bool:
        """
        Return True if the group action is transitive, i.e., there is only one orbit.

        Returns:
            True if the group action is transitive, False otherwise.
        """
        return len(self.orbits()) == 1

    @overload
    def act_on_function(self, g: GroupElement, x: torch.Tensor, dim: int=-1) -> torch.Tensor: ...

    @overload
    def act_on_function(self, g: Any, x: np.ndarray, dim: int=-1) -> np.ndarray: ... 

    def act_on_function(self, g: Union[GroupElement,Any], x, dim: int=-1):
        r"""
        Natural :math:`G`-action on the vector space of functions :math:`X\rightarrow \mathbb{R}` given by
        
        .. math:: 
            (g\cdot f)(x) = f(g^{-1}x)
        """
        if isinstance(g, GroupElement):
            assert g.group is self.group, f"Group element {g} not in group {self.group}"
        else:
            g = self.group[g]

        ind_dict = self(g.inv())

        if dim < 0:
            dim = len(x.shape) + dim

        slices: List[Any] = [slice(None)]*len(x.shape)
        slices[dim] = ind_dict

        y = x[tuple(slices)]

        return y

    def quotient(self, relation: Callable[[int, int], bool]) -> Tuple[GroupAction, List[int]]:
        r"""
        Given an equivalence relation on the set, return the quotient action on the set of equivalence classes.

        Args:
            relation: a function that takes in two indices and returns True if they are equivalent, and False otherwise.
        
        Returns:
            A tuple of (quotient_action, quotient_map), where:
                - quotient_action is a GroupAction object representing the action of the group on the set of equivalence classes.
                - quotient_map is a list of integers of length num_elements,
                  mapping each element of the original set to the index of its
                  equivalence class in the quotient set. The indices of the
                  equivalence classes are assigned in the order they are
                  encountered when iterating through the original set.


        Note:
            - The relation is checked for reflexivity, symmetry and transitivity.
            - The relation must be compatible with the group action. I.e. :math:`i\sim j \Leftrightarrow gi \sim gj`.
        """

        # first, we check that relation is indeed an equivalence relation
        # this is O(L^3), so we might want to optimize this later
        for i in range(self.num_elements):
            assert relation(i, i), "relation is not reflexive"
            for j in range(self.num_elements):
                if relation(i, j):
                    assert relation(j, i), "relation is not symmetric"
                    for k in range(self.num_elements):
                        if relation(j, k):
                            assert relation(i, k), "relation is not transitive"

        # we check that the relation is compatible with the group action
        for g in self.group:
            for i in range(self.num_elements):
                for j in range(self.num_elements):
                    if relation(i, j):
                        assert relation(self(g)[i], self(g)[j]), "relation is not compatible with the group action"
                    else:
                        assert not relation(self(g)[i], self(g)[j]), "relation is not compatible with the group action"
    
        # we find the equivalence classes
        eq_classes = []
        seen = set()
        _eq_class_ind = 0
        quotient_map = dict()
        for i in range(self.num_elements):
            if i in seen:
                continue
            eq_class = [j for j in range(self.num_elements) if relation(i, j)]
            eq_classes.append(eq_class)
            for j in eq_class: 
                quotient_map[j] = _eq_class_ind
            _eq_class_ind += 1
            seen.update(eq_class)


        # we construct the quotient action
        quotient_action_dict = dict()
        for g in self.group:
            quotient_action_dict[g] = [quotient_map[self(g)[_eq_class[0]]] for _eq_class in eq_classes]

        return GroupAction(self.group, quotient_action_dict), [quotient_map[i] for i in range(self.num_elements)]

    def induce_from(self, 
        subgroup_args: tuple, 
        subgroup_representation: GroupRepresentation,
        representatives: List[GroupElement],
        base_point: int = 0,
    ) -> GroupRepresentation:
        r"""
        Consider the following scenario:
            - We have a transitive left :math:`G` action on a set :math:`X`.
            - Choose a base point :math:`x_0` in :math:`X`
            - H is a normal subgroup of :math:`G` that contains :math:`\mathrm{Stab}_G(x_0)`
                - Note: since H is normal and the G-action is transitive, H also contains :math:`\mathrm{Stab}_G(x)` for all :math:`x \in X`.
            - We are given an H-representation :math:`(\rho, V)`

        Given this data, we can construct a :math:`G`-representation on the 
        space of functions :math:`X\rightarrow V`.

        Special case: if the subgroup is the whole group, then the resulting representation is just 
        the tensor product of self.to_linear_representation() and the given representation of the group.

        Note:
            - The current implementation is extremely slow, so it should only be used for small :math:`|X|` and small :math:`\dim(V)`.
            - This is superseded by :class:`equivit.geometry.EquivariantPullbackBundle`.
        """
        # TODO: relate this to Ind_K^G(Res_K^H(V)), where K = Stab(x_0)
        assert self.is_transitive(), "Currently only transitive group actions are supported."    
        assert base_point in range(self.num_elements), f"base_point must be an integer in the range [0, {self.num_elements-1}]" 

        incl = self.group.subgroup(*subgroup_args)
        incl_image = [incl(h) for h in incl.source]
        incl_inverse = {incl(h): h for h in incl.source}


        assert self.group.is_normal(incl_image), "The subgroup must be normal."

        cosets = self.group.left_cosets(incl)

        stabilizer = [g for g in self.group if self(g)[base_point] == base_point]

        for s in stabilizer:
            assert s in incl_image, "The subgroup must contain the stabilizer of the base point."

        subgroup_action = self.pullback(incl) 
        # subgroup_orbits = subgroup_action.orbits()
        subgroup_orbits = []
        for coset in cosets:
            g = coset[0]
            orbit = [self(g*incl(h))[base_point] for h in incl.source]
            subgroup_orbits.append(orbit)

        assert len(representatives) == len(subgroup_orbits), "The number of representatives must match the number of orbits of the subgroup action."

        for r, orbit in zip(representatives, subgroup_orbits):
            assert self(r)[base_point] in orbit, "Each representative must map the base point to a different orbit of the subgroup action."

        orbit_dict = {} # dictionary that maps each element of the set (an integer) to the index of the orbit it belongs to under the subgroup action
        for orbit_ind, orbit in enumerate(subgroup_orbits):
            for p in orbit:
                orbit_dict[p] = orbit_ind


        rep_matrices = dict()
        subgroup_rep_matrix_at_e = subgroup_representation(incl.source.identity())
        subgroup_rep_dim = subgroup_rep_matrix_at_e.shape[0]
        rep_dim = self.num_elements * subgroup_rep_dim

        for g in self.group:
            rep_matrix = np.zeros((self.num_elements, subgroup_rep_dim, self.num_elements, subgroup_rep_dim), 
                                  dtype=subgroup_rep_matrix_at_e.dtype)

            for i in range(self.num_elements):
                orbit_ind = orbit_dict[i] # the orbit element that i belongs to
                j = self(g)[i] # the element that i is mapped to under g
                new_orbit_ind = orbit_dict[j] # the orbit that j belongs to

                a = representatives[orbit_ind] 
                b = representatives[new_orbit_ind]

                # we want to find h in the subgroup such that ga = bh
                h = b.inv() * g * a
                assert h in incl_image

                rep_matrix[j,:,i,:] = subgroup_representation(incl_inverse[h])
            rep_matrices[g] = rep_matrix.reshape(rep_dim, rep_dim)

        return GroupRepresentation(self.group, rep_matrices)