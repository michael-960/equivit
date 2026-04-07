from enum import Enum
from typing import Tuple, Optional, List, Literal
import numpy as np

from .base import Group, GroupElement, CachedGroupMeta, GroupHomomorphism
from .action import GroupRepresentation, rotation_matrix, ComplexStructure, GroupAction
from .cyclic import CyclicGroup

from ...registry import GROUP


# @GROUP.register('DihedralGroup')
class DihedralGroup(Group, metaclass=CachedGroupMeta):
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
        # Note: this method is not used anywhere else in the code, 
        # TODO: remove this

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
        irreps['A1'] = dihedral_group_representation(self, [[1]], [[1]])
        irreps['A2'] = dihedral_group_representation(self, [[1]], [[-1]])

        if self.n % 2 == 0:
            irreps['B1'] = dihedral_group_representation(self, [[-1]], [[1]])
            irreps['B2'] = dihedral_group_representation(self, [[-1]], [[-1]])
            for k in range(1, self.n//2):
                theta = 2 * np.pi * k / self.n
                irreps[f'E{k}'] = dihedral_group_representation(self, rotation_matrix(theta), [[1,0], [0,-1]])
        else:
            for k in range(1, (self.n+1)//2):
                theta = 2 * np.pi * k / self.n
                irreps[f'E{k}'] = dihedral_group_representation(self, rotation_matrix(theta), [[1,0], [0,-1]])

        self._real_irreps = irreps
        return self._real_irreps

    def cyclic_subgroup(self) -> GroupHomomorphism:
        """
        Return the subgroup Cn
        """
        Cn = CyclicGroup(self.n)
        return GroupHomomorphism(Cn, self, lambda x: self.from_value((0, x.value)))

    def subgroup(self, type_: Literal['C', 'D'], m: int, q: Optional[int]=None) -> GroupHomomorphism:
        """
        Dn has two kinds of subgroups: cyclic subgroups of order m dividing n, and dihedral subgroups of order 2m where m divides n.

        For each m dividing n, there is a unique cyclic subgroup of order m, isomorphic to Cm, generated by r^(n/m).

        For each m dividing n, there are k dihedral subgroups of order 2m, where k := n/m. Each of these is 
        generated by r^k and tr^q for some q in [0, k-1].

        Args:
            type_: the type of subgroup, either 'C' for cyclic or 'D' for dihedral
            m: the order of the cyclic subgroup or half the order of the dihedral subgroup (must divide n)

            q: only for dihedral subgroups, an integer in [0, n/m-1] specifying the reflection generator
        Returns:
            a GroupHomomorphism representing the inclusion of the subgroup into Dn 
        """
        if type_ == 'C':
            assert q is None, f"q must not be specified for cyclic subgroups"
            i = self.cyclic_subgroup()
            j = i.source.subgroup(m)
            return i.compose(j)

        elif type_ == 'D':
            # The dihedral subgroup labeled by (m, q)
            # is generated by r^k and tr^q, where k := n/m
            # note that if q - q' is divisible by k, then the two subgroups (m, q) and (m, q') are the same
            # so we can always choose q in the range [0, k-1]
            assert q is not None, f"q must be specified for dihedral subgroups"
            assert self.n % m == 0, f"m ({m}) must divide n ({self.n})"
            k = self.n // m
            assert q in range(k), f"q ({q}) must be an integer satisfying 0 <= q < n/m"

            Dm = DihedralGroup(m)

            def mapping(g: GroupElement):
                num_t, num_r = g.value
                return self.from_value((num_t, (num_t*q + num_r*k) % self.n))

            return GroupHomomorphism(Dm, self, mapping)
        else:
            raise ValueError(f"Subgroup type must be either 'C' or 'D', not {type_}")

    def is_finite(self) -> bool:
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}({self.n})'

    def subgroups_up_to_conjugacy(self) -> List[tuple]:
        subgroups = []
        for m in range(1, self.n+1):
            if self.n % m == 0:
                subgroups.append(('C', m))
        for m in range(1, self.n+1):
            if self.n % m == 0:
                k = self.n // m
                if k % 2 == 0:
                    # if k is even, there are two conjugacy classes of dihedral subgroups of order 2m, represented by (m, 0) and (m, 1)
                    subgroups.append(('D', m, 0))
                    subgroups.append(('D', m, 1))
                else:
                    # if k is odd, there is only one conjugacy class of dihedral subgroups of order 2m, represented by (m, 0)
                    subgroups.append(('D', m, 0))
        return subgroups


D2 = DihedralGroup(2)
D3 = DihedralGroup(3)
D4 = DihedralGroup(4)
D5 = DihedralGroup(5)
D6 = DihedralGroup(6)
D7 = DihedralGroup(7)
D8 = DihedralGroup(8)

D12 = DihedralGroup(12)

def dihedral_group_action(
        group: DihedralGroup, 
        r_action: List[int],
        t_action: List[int],
) -> GroupAction:
    """
    Factory function to create a group action of a dihedral group from the actions of the generators r and t.
    Args:
        group: a dihedral group Dn
        r_action: a list of length L specifying the action of the rotation generator r on the set {0,...,L-1}
        t_action: a list of length L specifying the action of the reflection generator t     
    """
    action_dict = dict()
    r_action_ = np.array(r_action)
    t_action_ = np.array(t_action)

    for g in group:
        num_t, num_r = g.value

        q = np.arange(len(r_action))
        for _ in range(num_r):
            q = r_action_[q]

        if num_t == 1:
            q = t_action_[q]
        
        action_dict[g] = q.tolist()

    return GroupAction(group, action_dict)


def dihedral_group_representation(
        group: DihedralGroup, 
        r_matrix: np.ndarray, t_matrix: np.ndarray,
        complex_structure: Optional[ComplexStructure]=None
):
    """
    Factory function to create a group representation of a dihedral group from the representation matrices of the generators r and t.
    """
    rep_matrices = dict()
    for g in group:
        num_r = g.value[1]
        mat = np.linalg.matrix_power(r_matrix, num_r)
        if g.value[0] == 1:
            mat = t_matrix @ mat
        rep_matrices[g] = mat

    return GroupRepresentation(group, rep_matrices, complex_structure=complex_structure)