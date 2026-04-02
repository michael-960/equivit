from enum import Enum
from typing import Tuple
import numpy as np

from .base import Group, GroupElement, GroupRepresentation, rotation_matrix



class DihedralGroup(Group):
    """
    Dihedral group of order 2n, denoted Dn.

    The elements of the group are represented as words in the generators r and t,
    where r is a rotation and t is a reflection. 
    The relations are r^n = e, t^2 = e, and tr = rt^-1, where n is the order of the rotation element r.
    """
    def __init__(self, n):
        super().__init__()
        self.n = n

        # generate elements
        for _ in self: ...

        self._real_irreps = None

    def __iter__(self):
        for num_t in [0,1]:
            for num_r in range(self.n):
                yield self.from_value((num_t, num_r))

    def __len__(self):
        return 2 * self.n

    def reduce_word(self, word: str) -> str:
        # Reduce the word by applying the relations of D6
        # r^n = e, t^2 = e, tr = rt^-1
        # This is a non-trivial task and may require multiple passes to fully reduce the word
        # For simplicity, we will implement a basic reduction that handles some cases

        assert set(word).issubset({'r', 't'}), "word can only contain 'r' and 't'"

        n = self.n

        while 'rt' in word:
            word = word.replace('rt', 't' + 'r' * (n - 1))

        # Handle r^6 = e
        while 'r' * n in word:
            word = word.replace('r' * n, '')

        # Handle t^2 = e
        while 'tt' in word:
            word = word.replace('tt', '')

        if word in self._alias_element_dict.keys():
            return word

        return self.reduce_word(word)

    def from_value(self, value: Tuple[int,int]):
        if value in self._value_element_dict.keys():
            return self._value_element_dict[value]

        _assert_string = f"value must be a tuple of the form (num_t, num_r) where num_t and num_r are integers with num_t in [0,1] and num_r in [0, {self.n-1}]"
        assert type(value) is tuple, _assert_string
        assert len(value) == 2, _assert_string
        assert type(value[0]) is int and type(value[1]) is int, _assert_string
        assert value[0] in [0,1], _assert_string
        assert value[1] in range(self.n), _assert_string

        element = GroupElement(self, value)

        # self._element_value_dict[value] = GroupElement(self, value)
        self._value_element_dict[value] = element
        self._alias_element_dict['t'*value[0]+'r'*value[1]] = element
        return element

    def multiply(self, g, h) -> GroupElement:
        if type(g) is GroupElement:
            g = g.value
            h = h.value

        num_t_g, num_r_g = g
        num_t_h, num_r_h = h

        num_t = (num_t_g + num_t_h) % 2

        if num_t_h == 1:
            num_r_g = (self.n - num_r_g) % self.n

        num_r = (num_r_g + num_r_h) % self.n

        return self.from_value((num_t, num_r))

    def inverse(self, g) -> GroupElement:
        if type(g) is GroupElement:
            g = g.value 

        num_t, num_r = g

        if num_t == 0:
            return self.from_value((0, (-num_r) % self.n))
        else:
            return self.from_value((1, num_r))

    def identity(self):
        return self.from_value((0,0))

    def conjugacy_classes(self):
        _classes = [[self.from_value((0,0))]]
        if self.n % 2 == 0:
            for i in range(1, self.n//2):
                _classes.append([self.from_value((0,i)), self.from_value((0,self.n-i))])
            _classes.append([self.from_value((0,self.n//2))])

            _classes.append([self.from_value((1, 2*z)) for z in range(self.n//2)])
            _classes.append([self.from_value((1, 2*z+1)) for z in range(self.n//2)])

        else:
            for i in range(1, (self.n+1)//2):
                _classes.append([self.from_value((0,i)), self.from_value((0,self.n-i))])
                
            _classes.append([self.from_value((1, z)) for z in range(self.n)])

        return _classes

    def element_repr(self, g):
        num_t, num_r = g.value
        s = 't'*num_t + 'r'*num_r
        return f'D{self.n}[{s}]'

    def real_irreps(self):
        """
        Returns the real irreducible representations of the dihedral group.
        For dihedral groups, the real irreps are also complex irreps.
        """
        if self._real_irreps is not None:
            return self._real_irreps

        irreps = dict()
        irreps['A1'] = DihedralRepresentation(self, [[1]], [[1]])
        irreps['A2'] = DihedralRepresentation(self, [[1]], [[-1]])

        if self.n % 2 == 0:
            irreps['B1'] = DihedralRepresentation(self, [[-1]], [[1]])
            irreps['B2'] = DihedralRepresentation(self, [[-1]], [[-1]])
            for k in range(1, self.n//2):
                theta = 2 * np.pi * k / self.n
                irreps[f'E{k}'] = DihedralRepresentation(self, rotation_matrix(theta), [[1,0], [0,-1]])
        else:
            for k in range(1, (self.n+1)//2):
                theta = 2 * np.pi * k / self.n
                irreps[f'E{k}'] = DihedralRepresentation(self, rotation_matrix(theta), [[1,0], [0,-1]])

        self._real_irreps = irreps
        return self._real_irreps





D2 = DihedralGroup(2)
D3 = DihedralGroup(3)
D4 = DihedralGroup(4)
D5 = DihedralGroup(5)
D6 = DihedralGroup(6)
D7 = DihedralGroup(7)
D8 = DihedralGroup(8)

D12 = DihedralGroup(12)

class DihedralRepresentation(GroupRepresentation):
    def __init__(self, group: DihedralGroup, r_matrix, t_matrix):
        self.group = group
        self.setup_matrices(np.array(r_matrix, dtype=np.float64), np.array(t_matrix, dtype=np.float64))

    def setup_matrices(self, r_matrix, t_matrix):
        self.dim = r_matrix.shape[0]
        self._rep_matrices = dict()
        for g in self.group:
            num_r = g.value[1]

            mat = np.linalg.matrix_power(r_matrix, num_r)
            if g.value[0] == 1:
                mat = t_matrix @ mat
            self._rep_matrices[g.value] = mat

    def __call__(self, g: GroupElement):
        return self._rep_matrices[g.value]




# class DihedralIrreps(GroupIrreps):
#     def __init__(
#         self, name: str,
#         r_matrix, t_matrix
#     ):
#         """
#         r_matrix: image of r 
#         t_matrix: image of t 
#         """
#         self.irrep_name = name
#         self.setup_matrices(np.array(r_matrix, dtype=np.float64), np.array(t_matrix, dtype=np.float64))

#     def setup_matrices(self, r_matrix, t_matrix):
#         self.dim = r_matrix.shape[0]
#         self._rep_matrices = dict()
#         for g in self.__class__.group_class():
#             num_r = g.word.count('r')

#             mat = np.linalg.matrix_power(r_matrix, num_r)
#             if 't' in g.word:
#                 mat = t_matrix @ mat
#             self._rep_matrices[g] = mat

#     def __call__(self, g):
#         assert type(g) is self.__class__.group_class(), str(g) + ' is not an element of ' + str(self.__class__.group_class())
#         return self._rep_matrices[g]

#     @classmethod
#     def group_class(cls):
#         raise NotImplementedError

#     def __repr__(self):
#         return f'{self.__class__.__name__}.{self.irrep_name}'    



# class D2Irreps(DihedralIrreps):
#     A = 'A', [[1]], [[1]]
#     B1 = 'B1', [[1]], [[-1]]
#     B2 = 'B2', [[-1]], [[-1]]
#     B3 = 'B3', [[-1]], [[1]]

#     @classmethod
#     def group_class(cls):
#         return D2


# class D4Irreps(DihedralIrreps):
#     A1 = 'A1', [[1]], [[1]]
#     A2 = 'A2', [[1]], [[-1]]
#     B1 = 'B1', [[-1]], [[1]]
#     B2 = 'B2', [[-1]], [[-1]]
#     E = 'E', [[0,-1],[1,0]], [[1, 0], [0, -1]]
#     @classmethod
#     def group_class(cls):
#         return D4


# class D3Irreps(DihedralIrreps):
#     A1 = 'A1', [[1]], [[1]]
#     A2 = 'A2', [[1]], [[-1]]
#     E = 'E', rotation_matrix(np.pi*2/3), [[1,0], [0,-1]]

#     @classmethod
#     def group_class(cls):
#         return D3


# class D6Irreps(DihedralIrreps):
#     A1 = 'A1', [[1]], [[1]]
#     A2 = 'A2', [[1]], [[-1]]
#     B1 = 'B1', [[-1]], [[1]]
#     B2 = 'B2', [[-1]], [[-1]]
#     E1 = 'E1', rotation_matrix(np.pi/3), [[1,0], [0,-1]]
#     E2 = 'E2', rotation_matrix(np.pi*2/3), [[1,0], [0,-1]]

#     @classmethod
#     def group_class(cls):
#         return D6



