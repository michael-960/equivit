from __future__ import annotations
from enum import Enum
import numpy as np
from typing import Generic, TypeVar, Type, Tuple, Dict, Any, Literal, List, Optional

from .base import Group, GroupElement, GroupHomomorphism

# Group actions and representations



class GroupRepresentation:
    """
    An object that records the representation matrices of a representation of a finite group.
    """
    def __init__(self,
            group: Group, 
            rep_matrices: Dict[GroupElement, np.ndarray],
            complex_structure: Optional[ComplexStructure]=None
        ):
        """
        complex_structure: if the representation is over R and admits a complex structure, this argument specifies the complex structure. 
        It should not be specified for representations over C.
        """

        assert group.is_finite(), "Only representations of finite groups are supported for now."

        self.group = group

        id_mat = rep_matrices[group.identity()]
        self.dim = id_mat.shape[0]

        if np.isrealobj(id_mat):
            self.field = 'real'
        else:
            assert np.iscomplexobj(id_mat)
            self.field = 'complex'
            assert complex_structure is None, "complex_structure should not be specified for complex representations"

        self.complex_structure = complex_structure

        self._rep_matrices = dict()
        for g in group:
            mat = rep_matrices[g]
            assert mat.shape == (self.dim, self.dim)

            if self.field == 'real':
                assert np.isrealobj(mat)
            elif self.field == 'complex':
                if np.isrealobj(mat):
                    mat = mat.astype(np.complex128)
                assert np.iscomplexobj(mat)
            self._rep_matrices[g] = mat

    def __call__(self, g: GroupElement):
        # raise NotImplementedError("This method should be implemented by subclasses to return the representation matrix of the group element g in this representation.")
        return self._rep_matrices[g]

    def character(self, g: GroupElement):
        return self(g).trace()

    def all_characters(self):
        """
        Returns an array of characters, one for each group element.
        """
        return np.array([self.character(g) for g in self.group])

    def characters(self):
        """
        Returns an array of characters, one for each conjugacy class of the group. 
        """
        return np.array([self.character(cc[0]) for cc in self.group.conjugacy_classes()])

    def frobenius_schur_indicator(self):
        """
        Returns the Frobenius-Schur indicator of this representation. 
        """
        return np.mean([self.character(g*g) for g in self.group])

    def as_complex(self):
        """
        If this representation is real and admits a complex structure, 
        returns the corresponding complex representation.
        """
        if self.complex_structure is None:
            raise TypeError("This representation does not come with a complex structure.")

        return GroupRepresentation(
            self.group, 
            {g: self.complex_structure.endo_r2c(self(g), 0, 1) for g in self.group}
            )

    def validate(self, tol=1e-9):
        """
        Validate that this is a valid representation of the group.
        """
        for g in self.group:
            for h in self.group:
                error = np.max(np.abs((self(g*h) - self(g) @ self(h))))
                if error > tol:
                    raise ValueError(f'The representation is not valid: the representation matrix of {g}*{h} is not equal to the product of the representation matrices of {g} and {h}. The maximum absolute error (L-infinity) is {error}.')
    

class ComplexStructure:
    """
    If V is a real vector space, a complex structure is a linear map J V->V
    that squares to -1. 

    Subclasses should implement such a structure by directly specifying 
    how real vectors and endomorphisms are converted to complex ones and vice versa.

    The consistency these conversions should be taken care of by the 
    implementation. I.e., it will not be checked.
    """
    def vector_r2c(self, x: np.ndarray, axis: int):
        """
        Convert a 2n-dimensional real vector to an n-dimensional complex vector.
        """
        raise NotImplementedError()

    def vector_c2r(self, x: np.ndarray, axis: int):
        """
        Convert an n-dimensional complex vector to a 2n-dimensional complex vector.
        """
        raise NotImplementedError()

    def dual_vector_r2c(self, x: np.ndarray, axis: int): 
        """
        Suppose a is a dual vector. The corresponding complex dual vector b is defined by
        <b, v> = <a, v> + i<a, -Jv>, where J is the complex structure.
        """
        ...

    def dual_vector_c2r(self, x: np.ndarray, axis: int):
        """
        Suppose b is a complex dual vector. The corresponding real dual vector a is defined by
        <a, v> = Re(<b, v>), where J is the complex structure.
        """
        ...

    def endo_r2c(self, x: np.ndarray, axis1: int, axis2: int):
        """
        Convert a 2n-dimensional real endomorphism to an n-dimensional complex endomorphism.
        """
        raise NotImplementedError()

    def endo_c2r(self, x: np.ndarray, axis1: int, axis2: int):
        """
        Convert an n-dimensional complex endomorphism to a 2n-dimensional real endomorphism.
        """
        raise NotImplementedError()


class StandardComplexStructure(ComplexStructure):
    """
    Standard complex structure on R^{2n}
    """
    def __init__(self, complex_dim):
        self.complex_dim = complex_dim

    def vector_r2c(self, x: np.ndarray, axis: int):
        x_re, x_im = np.split(x, [self.complex_dim], axis=axis)
        return x_re + 1j*x_im

    def vector_c2r(self, x: np.ndarray, axis: int):
        return np.concat((x.real, x.imag), axis=axis)

    def dual_vector_r2c(self, x, axis):
        return self.vector_r2c(x, axis).conj()

    def dual_vector_c2r(self, x: np.ndarray, axis: int):
        return self.vector_c2r(x.conj(), axis)

    def endo_r2c(self, x: np.ndarray, axis1: int, axis2: int):
        x_re = np.split(
                    np.split(x, [self.complex_dim], axis=axis1)[0],
                    [self.complex_dim], axis=axis2)[0]

        x_im = np.split(
                    np.split(x, [self.complex_dim], axis=axis1)[1], 
                    [self.complex_dim], axis=axis2)[0]
        return x_re + 1j*x_im

    def endo_c2r(self, x, axis1: int, axis2: int):
        y = np.concat((x.real, -x.imag), axis=axis2)
        z = np.concat((x.imag, x.real), axis=axis2)
        return np.concat((y, z), axis=axis1)


def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])

class GroupAction:
    """
    Group action on the set 
    0, 1, 2, ..., n-1

    The action is specified by a dictionary mapping each group element g to a list of integers
        [m_0, m_1, m_2, ..., m_{n-1}],
    meaning that i is mapped to m_i under g.
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


    def restrict_action(self, indices):
        """
        Return the restricted action on a subset of the set.
        
        Return: a dictionary mapping group elements to the restricted action on the subset. 
        Each value of the dictionary is an array of shape (m,), where m is the size of the subset.
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

    def to_linear_representation(self):
        """
        Return the a GroupRepresentation object that consists of real matrices
        of the corresponding representation on the vector space spanned by the
        set elements.
        
        Return: a dictionary mapping group elements to representation matrices. 
        Each value of the dictionary is a permutation matrix of shape (n, n), where n is the size of the set.
        """
        rep_matrices = dict()

        all_inds = [i for i in range(self.num_elements)]

        for g in self.group:
            rep_matrices[g] = np.zeros((self.num_elements, self.num_elements), dtype=np.float64)
            rep_matrices[g][self.action_dict[g], all_inds] = 1.
        
        return GroupRepresentation(self.group, rep_matrices)


    def pullback(self, homomorphism: GroupHomomorphism):
        """
        Given a group action G->Aut(X) and a homomorphism H->G, 
        there is a natural action H->Aut(X), called the restriction or the pullback.
        Note that this is different from restricting the action to a subset of X that is invariant under the subgroup H, 
        which is implemented in the restrict_action method.
        """
        action_dict = dict()
        for g in homomorphism.source:
            action_dict[g] = self.action_dict[homomorphism(g)]

        return GroupAction(homomorphism.source, action_dict)

    def irrep_multiplicities(self):
        """
        Return the multiplicity of each real irrep in the 
        representation of the group on the vector space spanned functions from the set to R.

        Return: a dictionary mapping each irrep name to its multiplicity in the decomposition.
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
        """
        return len(self.orbits()) == 1

    def induce_from(self, 
        subgroup_args: tuple, 
        subgroup_representation: GroupRepresentation,
        representatives: List[GroupElement],
        base_point: int = 0,
    ) -> GroupRepresentation:
        """
            Consider the following scenario:
            - We have a transitive left G action on a set X.
            - Choose a base point x_0 in X
            - H is a normal subgroup of G that contains Stab(x_0)
                - Note: since H is normal and the G-action is transitive, H also contains Stab(x) for all x in X.
            - We are given an H-representation (rho, V)

            Given this data, we can construct a G-representation on the 
            space of functions X -> V.

            Special case: if the subgroup is the whole group, then the resulting representation is just 
            the tensor product of self.to_linear_representation() and the given representation of the group.

            The current implementation is extremely slow, so it should only be used for small |X| and small dim(V).
        """
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