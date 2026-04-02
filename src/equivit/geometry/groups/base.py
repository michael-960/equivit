from __future__ import annotations
from enum import Enum
import numpy as np
from typing import Generic, TypeVar, Type, Tuple, Dict, Any



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


    def order(self):
        return len(self)

    def multiply(self, g: Any, h: Any):
        raise NotImplementedError("This method should be implemented by subclasses to return the product of two group elements g and h.")

    def inverse(self, g: Any):
        raise NotImplementedError("This method should be implemented by subclasses to return the inverse of a group element g.")

    def from_value(self, value: Any) -> GroupElement:
        raise NotImplementedError("This method should be implemented by subclasses to construct a group element from a value.")

    def irreps(cls) -> Type[GroupIrreps]:
        raise NotImplementedError("This method should be implemented by subclasses to return the class that contains the irreducible representations of the group.")
        
    def conjugacy_classes(cls):
        raise NotImplementedError("This method should be implemented by subclasses to return the conjugacy classes of the group.")

    def identity(cls) -> GroupElement:
        """Returns the identity element of the group."""
        raise NotImplementedError("This method should be implemented by subclasses to return the identity element of the group.")

    def subgroup(cls, name: str) -> Tuple[Group, Dict[Group, Group]]:
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



# class GroupIrreps(Enum):
#     dim: int
#     def character(self, g):
#         return self(g).trace()

#     def characters(self):
#         """
#         Returns an array of characters, one for each conjugacy class of the group. 
#         """
#         return np.array([self.character(cc[0]) for cc in self.__class__.group_class().conjugacy_classes()])

#     def all_characters(self):
#         """
#         Returns an array of characters, one for each group element.
#         """
#         return np.array([self.character(g) for g in self.__class__.group_class()])

#     def __call__(self, g):
#         raise NotImplementedError("This method should be implemented by subclasses to return the representation matrix of the group element g in this irrep.")

#     @classmethod
#     def group_class(cls) -> Type[Group]:
#         raise NotImplementedError



class GroupRepresentation:
    group: Group

    def __call__(self, g: GroupElement):
        raise NotImplementedError("This method should be implemented by subclasses to return the representation matrix of the group element g in this representation.")

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
            - 1 if the representation is real
            - 0 if the representation is complex
            - -1 if the representation is quaternionic
        """
        return np.mean([self.character(g*g) for g in self.group])


def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
