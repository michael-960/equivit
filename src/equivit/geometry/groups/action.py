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
        assert g.group is self.group
        return self.action_dict[g]

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
        """
        action_dict = dict()
        for g in homomorphism.source:
            action_dict[g] = self.action_dict[homomorphism(g)]

        return GroupAction(homomorphism.source, action_dict)

    


