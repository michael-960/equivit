import numpy as np
from .base import Group, GroupElement, rotation_matrix, GroupRepresentation

class CyclicGroup(Group):
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

    def multiply(self, g: int, h: int) -> int:
        return self.from_value((g + h) % self.n)

    def inverse(self, g: int) -> int:
        if type(g) is GroupElement:
            g = g.value
        return self.from_value((-g) % self.n)
    
    def from_value(self, value):
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

        irreps['A'] = CyclicGroupRepresentation(self, [[1]])

        if self.n % 2 == 0:
            irreps['B'] = CyclicGroupRepresentation(self, [[-1]])


        for k in range(1, (self.n+1)//2):
            theta = 2 * np.pi * k / self.n
            irreps[f'E{k}'] = CyclicGroupRepresentation(self, rotation_matrix(theta))

        self._real_irreps = irreps
        return self._real_irreps



TRIVIAL_GROUP = C1 = CyclicGroup(1)
C2 = CyclicGroup(2) 
C3 = CyclicGroup(3) 
C4 = CyclicGroup(4)
C5 = CyclicGroup(5)
C6 = CyclicGroup(6)
C7 = CyclicGroup(7)
C8 = CyclicGroup(8)

C12 = CyclicGroup(12)


class CyclicGroupRepresentation(GroupRepresentation):
    def __init__(self, group: CyclicGroup, r_matrix):
        self.group = group
        self.setup_matrices(np.array(r_matrix, dtype=np.float64))

    def setup_matrices(self, r_matrix):
        self.dim = r_matrix.shape[0]
        self._rep_matrices = dict()
        for g in self.group:
            num_r = g.value
            mat = np.linalg.matrix_power(r_matrix, num_r)

            self._rep_matrices[g.value] = mat

    def __call__(self, g: GroupElement):
        return self._rep_matrices[g.value]

