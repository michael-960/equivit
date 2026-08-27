from typing import Dict
from .base import Group, GroupElement, CachedGroupMeta, GroupHomomorphism, rotation_matrix
from .representations import RealIrrep


class SymmetricGroup(Group, metaclass=CachedGroupMeta):
    r"""
    Symmetric group of order :math:`n`.
    Stub.
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

    def multiply(self, g: tuple, h: tuple) -> GroupElement:
        raise NotImplementedError("The multiplication of symmetric groups is not implemented yet.")

    def inverse(self, g: tuple) -> GroupElement:
        if type(g) is GroupElement:
            g = g.value

        return self.from_value(g[::-1])
    
    def from_value(self, value: tuple[int]) -> GroupElement:
        if value in self._value_element_dict.keys():
            return self._value_element_dict[value]

        assert type(value) is tuple, "value must be a tuple of integers"
        assert len(value) <= self.n, "value must be a tuple of integers of length at most n"
        assert all(type(v) is int for v in value), "value must be a tuple of integers"
        assert all(0 <= v < self.n for v in value), "value must be a tuple of integers in the range [0, n)"
        assert len(set(value)) == len(value), "value must be a tuple of distinct integers"

        element = GroupElement(self, value)

        # self._element_value_dict[value] = GroupElement(self, value)
        self._value_element_dict[value] = element
        self._alias_element_dict['r'*value] = element
        return element

    def identity(self):
        return self.from_value(())

    def element_repr(self, g):
        return f'S{self.n}[{g.value}]'

    def conjugacy_classes(self):
        # cyclic groups are abelian, so each element forms its own conjugacy class
        raise NotImplementedError("The conjugacy classes of symmetric groups are not implemented yet.")

    def real_irreps(self) -> Dict[str, RealIrrep]:
        """
        Returns the real irreducible representations of the symmetric group.
        These are indexed by Young diagrams.
        """
        raise NotImplementedError("The real irreps of symmetric groups are not implemented yet.")

    def complex_irreps(self):
        """
        Returns the complex irreducible representations of the symmetric group.
        """
        raise NotImplementedError("The complex irreps of symmetric groups are not implemented yet.")

