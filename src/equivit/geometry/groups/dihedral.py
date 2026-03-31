from enum import Enum
import typing
import numpy as np

from .base import Group, GroupIrreps


class Dihedral(Group):
    """
    Base class for dihedral groups. 
    The elements of the group are represented as words in the generators r and t,
    where r is a rotation and t is a reflection. 
    The relations are r^n = e, t^2 = e, and tr = rt^-1, where n is the order of the rotation element r.

    This is an abstract class.
    Subclasses should implement the specific dihedral group (e.g., D3, D4, D6)
    by defining the group elements as class attributes and implementing the
    conjugacy_classes, get_n, and word_element_lookup methods.
    """
    def __init__(self, s):
        self.word = s
        self.__class__.word_element_lookup()[s] = self

    @classmethod
    def reduce_word(cls, word: str) -> str:
        # Reduce the word by applying the relations of D6
        # r^n = e, t^2 = e, tr = rt^-1
        # This is a non-trivial task and may require multiple passes to fully reduce the word
        # For simplicity, we will implement a basic reduction that handles some cases

        assert set(word).issubset({'r', 't'}), "word can only contain 'r' and 't'"

        n = cls.get_n()

        while 'rt' in word:
            word = word.replace('rt', 't' + 'r' * (n - 1))

        # Handle r^6 = e
        while 'r' * n in word:
            word = word.replace('r' * n, '')

        # Handle t^2 = e
        while 'tt' in word:
            word = word.replace('tt', '')

        if word in cls.word_element_lookup().keys():
            return word

        return cls.reduce_word(word)


    @classmethod
    def from_word(cls, word: str):
        """
        word: a string containing t and r
        """
        return cls.word_element_lookup()[cls.reduce_word(word)]

    def __mul__(self, other: 'D6'):
        concatenated_word = self.word + other.word
        reduced_word = self.reduce_word(concatenated_word)
        return self.__class__.from_word(reduced_word)

    def __repr__(self):
        s = self.word
        if s == '':
            s = 'e'
        return f'{self.__class__.__name__}.{s}'

    def inv(self):
        if ('t' in self.word) or (self.word == ''):
            return self

        n = self.__class__.get_n()
        return self.__class__.from_word('r'*(n-len(self.word)))
    
    @classmethod
    def word_element_lookup(cls):
        raise NotImplementedError("This method should be implemented by subclasses to return a dictionary mapping reduced words to group elements.")

    @classmethod
    def get_n(cls):
        raise NotImplementedError("This method should be implemented by subclasses to return the order of the rotation element r.")

    @classmethod
    def identity(cls):
        return cls.from_word('')


_d2_elements = dict()

class D2(Dihedral):
    """
    D2 is isomorphic to the Klein four-group and C2 x C2.
    """
    e = ''
    r = 'r'
    t = 't'
    tr = 'tr'

    @classmethod
    def conjugacy_classes(cls):
        return [
            [cls.e],
            [cls.r],
            [cls.t],
            [cls.tr],
        ]

    @classmethod
    def get_n(cls):
        return 2

    @classmethod
    def word_element_lookup(cls):
        return _d2_elements

    @classmethod
    def irreps(cls):
        return D2Irreps


_d3_elements = dict()

class D3(Dihedral):
    e = ''
    r = 'r'
    rr = 'rr'
    t = 't'
    tr = 'tr'
    trr = 'trr'

    @classmethod
    def conjugacy_classes(cls):
        return [
            [cls.e],
            [cls.r, cls.rr],
            [cls.t, cls.tr, cls.trr],
        ]

    @classmethod
    def get_n(cls):
        return 3

    @classmethod
    def word_element_lookup(cls):
        return _d3_elements

    @classmethod
    def irreps(cls):
        return D3Irreps


_d4_elements = dict()

class D4(Dihedral):
    e = ''
    r = 'r'
    rr = 'rr'
    rrr = 'rrr'
    t = 't'
    tr = 'tr'
    trr = 'trr'
    trrr = 'trrr'

    @classmethod
    def conjugacy_classes(cls):
        return [
            [cls.e],
            [cls.r, cls.rrr],
            [cls.rr],
            [cls.t, cls.trr],
            [cls.tr, cls.trrr]
        ]

    @classmethod
    def get_n(cls):
        return 4

    @classmethod
    def word_element_lookup(cls):
        return _d4_elements
    
    @classmethod
    def irreps(cls):
        return D4Irreps




_d6_elements = dict()

class D6(Dihedral):

    e = ''
    r = 'r'
    rr = 'rr'
    rrr = 'rrr'
    rrrr = 'rrrr'
    rrrrr = 'rrrrr'

    t = 't'
    tr = 'tr'
    trr = 'trr'
    trrr = 'trrr'
    trrrr = 'trrrr'
    trrrrr = 'trrrrr'

    @classmethod
    def conjugacy_classes(cls):
        return [
            [cls.e],
            [cls.r, cls.rrrrr],
            [cls.rr, cls.rrrr],
            [cls.rrr],
            [cls.t, cls.trr, cls.trrrr],
            [cls.tr, cls.trrr, cls.trrrrr],
        ]

    @classmethod
    def get_n(cls):
        return 6

    @classmethod
    def word_element_lookup(cls):
        return _d6_elements
    
    @classmethod
    def irreps(cls):
        return D6Irreps


def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])


class DihedralIrreps(GroupIrreps):
    def __init__(
        self, name: str,
        r_matrix, t_matrix
    ):
        """
        r_matrix: image of r 
        t_matrix: image of t 
        """
        self.irrep_name = name
        self.setup_matrices(np.array(r_matrix, dtype=np.float64), np.array(t_matrix, dtype=np.float64))

    def setup_matrices(self, r_matrix, t_matrix):
        self.dim = r_matrix.shape[0]
        self._rep_matrices = dict()
        for g in self.__class__.group_class():
            num_r = g.word.count('r')

            mat = np.linalg.matrix_power(r_matrix, num_r)
            if 't' in g.word:
                mat = t_matrix @ mat
            self._rep_matrices[g] = mat

    def __call__(self, g):
        assert type(g) is self.__class__.group_class(), str(g) + ' is not an element of ' + str(self.__class__.group_class())
        return self._rep_matrices[g]

    @classmethod
    def group_class(cls):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}.{self.irrep_name}'    



class D2Irreps(DihedralIrreps):
    A = 'A', [[1]], [[1]]
    B1 = 'B1', [[1]], [[-1]]
    B2 = 'B2', [[-1]], [[-1]]
    B3 = 'B3', [[-1]], [[1]]

    @classmethod
    def group_class(cls):
        return D2


class D4Irreps(DihedralIrreps):
    A1 = 'A1', [[1]], [[1]]
    A2 = 'A2', [[1]], [[-1]]
    B1 = 'B1', [[-1]], [[1]]
    B2 = 'B2', [[-1]], [[-1]]
    E = 'E', [[0,-1],[1,0]], [[1, 0], [0, -1]]
    @classmethod
    def group_class(cls):
        return D4


class D3Irreps(DihedralIrreps):
    A1 = 'A1', [[1]], [[1]]
    A2 = 'A2', [[1]], [[-1]]
    E = 'E', rotation_matrix(np.pi*2/3), [[1,0], [0,-1]]

    @classmethod
    def group_class(cls):
        return D3


class D6Irreps(DihedralIrreps):
    A1 = 'A1', [[1]], [[1]]
    A2 = 'A2', [[1]], [[-1]]
    B1 = 'B1', [[-1]], [[1]]
    B2 = 'B2', [[-1]], [[-1]]
    E1 = 'E1', rotation_matrix(np.pi/3), [[1,0], [0,-1]]
    E2 = 'E2', rotation_matrix(np.pi*2/3), [[1,0], [0,-1]]

    @classmethod
    def group_class(cls):
        return D6



