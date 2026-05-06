from enum import Enum
from typing import Tuple, Optional, List, Literal, Dict
import numpy as np

from .base import Group, GroupElement, CachedGroupMeta, GroupHomomorphism, rotation_matrix
from .action import GroupAction
from .representations import GroupRepresentation, IrrepType, RealIrrep, ComplexStructure
from .cyclic import CyclicGroup



class DihedralGroup(Group, metaclass=CachedGroupMeta):
    r"""
    Dihedral group of order :math:`2n`, denoted :math:`D_n`.

    The elements of the group are represented as words in the generators :math:`r` and :math:`t`,
    where :math:`r` is a rotation (whose order is :math:`n`) and :math:`t` is a reflection. 

    The relations are :math:`r^n = e, t^2 = e`, and :math:`tr = rt^{-1}`
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

    def complex_irreps(self):
        # irreps = self.real_irreps()
        raise NotImplementedError()

    def real_irreps(self) -> Dict[str, RealIrrep]:
        """
        Returns the real irreducible representations of the dihedral group.
        For dihedral groups, the real irreps are also complex irreps.
        """
        if self._real_irreps is not None:
            return self._real_irreps

        irreps = dict()

        irreps['A1'] = RealIrrep.from_rep(dihedral_group_representation(self, [[1]], [[1]]), 
                                          name='A1', rep_type=IrrepType.REAL)

        irreps['A2'] = RealIrrep.from_rep(dihedral_group_representation(self, [[1]], [[-1]]), 
                                          name='A2', rep_type=IrrepType.REAL)

        if self.n % 2 == 0: 
            irreps['B1'] = RealIrrep.from_rep(dihedral_group_representation(self, [[-1]], [[1]]), name='B1', rep_type=IrrepType.REAL)

            irreps['B2'] = RealIrrep.from_rep(dihedral_group_representation(self, [[-1]], [[-1]]), name='B2', rep_type=IrrepType.REAL)
            for k in range(1, self.n//2):
                theta = 2 * np.pi * k / self.n
                irreps[f'E{k}'] = RealIrrep.from_rep(dihedral_group_representation(self, rotation_matrix(theta), [[1,0], [0,-1]]),
                                                     name=f'E{k}', rep_type=IrrepType.REAL)
        else:
            for k in range(1, (self.n+1)//2):
                theta = 2 * np.pi * k / self.n
                irreps[f'E{k}'] = RealIrrep.from_rep(dihedral_group_representation(self, rotation_matrix(theta), [[1,0], [0,-1]]),
                                                     name=f'E{k}', rep_type=IrrepType.REAL)

        self._real_irreps = irreps
        return self._real_irreps

    def cyclic_subgroup(self) -> GroupHomomorphism:
        """
        Return the subgroup :math:`C_n`
        """
        Cn = CyclicGroup(self.n)
        return GroupHomomorphism(Cn, self, lambda x: self.from_value((0, x.value)))

    def subgroup(self, type_: Literal['C', 'D'], m: int, q: Optional[int]=None) -> GroupHomomorphism:
        r"""
        :math:`D_n` has two kinds of subgroups: cyclic subgroups of order :math:`m` dividing :math:`n`, and dihedral subgroups of order :math:`2m` where :math:`m` divides :math:`n`.

        For each :math:`m` dividing :math:`n`, there is a unique cyclic subgroup of order :math:`m`, isomorphic to :math:`C_m`, generated by :math:`r^{n/m}`.

        For each :math:`m` dividing :math:`n`, there are :math:`k` dihedral subgroups of order :math:`2m`, where :math:`k := n/m`. Each of these is 
        generated by :math:`r^k` and :math:`tr^q` for some :math:`q \in \{0, 1, \dotsb, k-1\}`.

        Args:
            type\_: the type of subgroup, either 'C' for cyclic or 'D' for dihedral
            m: the order of the cyclic subgroup or half the order of the dihedral subgroup (must divide :math:`n`)

            q: only for dihedral subgroups, an integer in :math:`\{0, 1, \dotsb, n/m-1\}` specifying the reflection generator
        Returns:
            a GroupHomomorphism representing the inclusion of the subgroup into :math:`D_n` 
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

    def get_subgroup_name(self, subgroup_elements: List[GroupElement]) -> Tuple[int]:
        assert len(subgroup_elements) == len(set(subgroup_elements)), "Duplicate elements in subgroup_elements"

        subgroup_elements = set(subgroup_elements)

        for m in range(1, self.n+1):
            if self.n % m == 0:
                subgroup_incl = self.subgroup('C', m)
                if subgroup_elements == set([subgroup_incl(g) for g in subgroup_incl.source]):
                    return ('C', m)
                
                k = self.n // m
                for q in range(k):
                    subgroup_incl = self.subgroup('D', m, q)
                    if subgroup_elements == set([subgroup_incl(g) for g in subgroup_incl.source]):
                        return ('D', m, q)

                
        raise ValueError(f"The provided subgroup_elements do not correspond to any subgroup of {self}")


    @classmethod
    def parse_args(cls, n: int) -> tuple:
        return (n,)


D1 = DihedralGroup(1)
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