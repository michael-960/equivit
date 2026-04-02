from __future__ import annotations
from enum import Enum
import numpy as np
from typing import Generic, TypeVar, Type, Tuple, Dict, Any, Literal, List, Optional



class Group:
    def __init__(self):
        self._value_element_dict = {}
        # self._element_value_dict = {}
        self._alias_element_dict = {}

    def __getitem__(self, key):
        if key in self._value_element_dict.keys():
            return self._value_element_dict[key]

        if key in self._alias_element_dict.keys():
            return self._alias_element_dict[key]

        # return self.from_value(key)
        raise KeyError(f"Key {key} not found in group element lookup.")

    def __len__(self) -> int:
        raise ValueError("This method should be implemented by subclasses to return the number of elements in the group.")

    def __contains__(self, g):
        if not (type(g) is GroupElement): return False

        return g.group is self


    def order(self) -> int:
        return len(self)

    def multiply(self, g: Any, h: Any):
        raise NotImplementedError("This method should be implemented by subclasses to return the product of two group elements g and h.")

    def inverse(self, g: Any):
        raise NotImplementedError("This method should be implemented by subclasses to return the inverse of a group element g.")

    def from_value(self, value: Any) -> GroupElement:
        raise NotImplementedError("This method should be implemented by subclasses to construct a group element from a value.")

    def real_irreps(self):
        raise NotImplementedError("This method should be implemented by subclasses to return the class that contains the irreducible representations of the group.")
        
    def conjugacy_classes(self):
        raise NotImplementedError("This method should be implemented by subclasses to return the conjugacy classes of the group.")

    def identity(self) -> GroupElement:
        """Returns the identity element of the group."""
        raise NotImplementedError("This method should be implemented by subclasses to return the identity element of the group.")

    def subgroup(self, name: str) -> Tuple[Group, Dict[GroupElement, GroupElement]]:
        """Returns a subgroup of the group given a name together with the inclusion map."""
        raise NotImplementedError("This method should be implemented by subclasses to return a subgroup of the group given its name.")

    def element_repr(self, g: GroupElement) -> str:
        """Returns a string representation of the group element g."""
        return f'{self.__class__.__name__}[{str(g.value)}]'


class GroupElement:
    def __init__(self, group: Group, value: Any):
        self.group = group
        self.value = value

    def __mul__(self, other: GroupElement):
        assert self.group is other.group, "Group elements must belong to the same group to be multiplied."
        return self.group.multiply(self.value, other.value)

    def inv(self):
        return self.group.inverse(self.value)

    def __eq__(self, other):
        return self.group is other.group and self.value == other.value

    def __repr__(self):
        return self.group.element_repr(self)

    def __hash__(self):
        return hash((self.group, self.value))



