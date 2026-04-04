import numpy as np
from .base import Group, GroupElement, CachedGroupMeta, GroupHomomorphism
from .action import rotation_matrix, GroupRepresentation, StandardComplexStructure, ComplexStructure, GroupAction
from typing import Optional, List



class CyclicGroup(Group, metaclass=CachedGroupMeta):
    """
    Cyclic group of order n.
    The elements of the group are represented as words in the generator r (rotation).
    This is an abstract class.
    Subclasses should define the group elements as class attributes and implement 
    the conjugacy_classes, get_n, and word_element_lookup methods.
    """
    def __init__(self, n: int):
        super().__init__()
        self.n = n
        # generate elements
        for _ in self: ...

        self._real_irreps = None


    def __iter__(self):
        for i in range(self.n):
            yield self.from_value(i)

    def __len__(self):
        return self.n

    def multiply(self, g: int, h: int) -> GroupElement:
        return self.from_value((g + h) % self.n)

    def inverse(self, g: int) -> GroupElement:
        if type(g) is GroupElement:
            g = g.value
        return self.from_value((-g) % self.n)
    
    def from_value(self, value) -> GroupElement:
        if value in self._value_element_dict.keys():
            return self._value_element_dict[value]

        assert type(value) is int, "value must be an integer"
        assert value in range(self.n), f"value must be an integer in the range [0, {self.n-1}]"

        element = GroupElement(self, value)

        # self._element_value_dict[value] = GroupElement(self, value)
        self._value_element_dict[value] = element
        self._alias_element_dict['r'*value] = element
        return element

    def identity(self):
        return self.from_value(0)

    def element_repr(self, g):
        s = 'r'*g.value
        return f'C{self.n}[{s}]'

    def conjugacy_classes(self):
        # cyclic groups are abelian, so each element forms its own conjugacy class
        return [[self.from_value(i)] for i in range(self.n)]

    def real_irreps(self):
        """
        Returns the real irreducible representations of the cyclic group.
        For cyclic groups, some real irreps are not complex irreps.
        """
        if self._real_irreps is not None:
            return self._real_irreps

        irreps = dict()

        irreps['A'] = cyclic_group_representation(self, [[1]])

        if self.n % 2 == 0:
            irreps['B'] = cyclic_group_representation(self, [[-1]])

        for k in range(1, (self.n+1)//2):
            theta = 2 * np.pi * k / self.n
            irreps[f'E{k}'] = cyclic_group_representation(
                self, rotation_matrix(theta), complex_structure=StandardComplexStructure(1))

        self._real_irreps = irreps
        return self._real_irreps

    def complex_irreps(self):
        """
        Returns the complex irreducible representations of the cyclic group.
        For cyclic groups, all complex irreps are 1-dimensional and are given by the characters.
        """

        irreps = dict()

        for k in range(self.n):
            theta = 2 * np.pi * k / self.n
            irrep = cyclic_group_representation(self, [[np.exp(1j * theta)]])
            irreps[f'{k}'] = irrep

        return irreps

    def subgroup(self, m: int) -> GroupHomomorphism:
        """
        Returns the subgroup of Cn of order m (and index n/m).
        """
        assert self.n % m == 0, f"Subgroups of {self} are indexed by factors of {self.n}. {m} is not a factor of {self.n}"

        k = self.n // m
        Ck = CyclicGroup(m)
        mapping = dict()
        for a in range(m):
            mapping[Ck.from_value(a)] = self.from_value(a*k % self.n)

        return GroupHomomorphism(Ck, self, mapping)

    def is_finite(self) -> bool:
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}({self.n})'



TRIVIAL_GROUP = C1 = CyclicGroup(1)
C2 = CyclicGroup(2) 
C3 = CyclicGroup(3) 
C4 = CyclicGroup(4)
C5 = CyclicGroup(5)
C6 = CyclicGroup(6)
C7 = CyclicGroup(7)
C8 = CyclicGroup(8)

C12 = CyclicGroup(12)


def cyclic_group_action(
        group: CyclicGroup, 
        r_action: List[int]
) -> GroupAction:
    action_dict = dict()
    r_action_ = np.array(r_action)
    for g in group:
        q = np.arange(len(r_action))
        for _ in range(g.value):
            q = r_action_[q]
        action_dict[g] = q.tolist()

    return GroupAction(group, action_dict)


def cyclic_group_representation(
        group: CyclicGroup, r_matrix: np.ndarray, 
        complex_structure: Optional[ComplexStructure]=None
) -> GroupRepresentation:
    """
    A representation of Cn is specified by the image of the generator r.
    """
    rep_matrices = dict()
    for g in group:
        num_r = g.value
        mat = np.linalg.matrix_power(r_matrix, num_r)
        rep_matrices[g] = mat

    return GroupRepresentation(group, rep_matrices, complex_structure=complex_structure)

