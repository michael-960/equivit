from __future__ import annotations
from enum import Enum
import numpy as np
from typing import Generic, TypeVar, Type, Tuple, Dict, Any, Literal, List, Optional, Union, Callable



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
        raise NotImplementedError("This method should be implemented by subclasses to return the number of elements in the group.")

    def __contains__(self, g):
        if not (type(g) is GroupElement): return False

        return g.group is self

    def is_finite(self) -> bool:
        return False

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
        """
        Returns the conjugacy classes of the group.
        """
        raise NotImplementedError("This method should be implemented by subclasses to return the conjugacy classes of the group.")

    def identity(self) -> GroupElement:
        """Returns the identity element of the group."""
        raise NotImplementedError("This method should be implemented by subclasses to return the identity element of the group.")

    def subgroup(self, *args) -> GroupHomomorphism:
        """
        Given some arguments specifying a subgroup, return the inclusion map as an injective group homomorphism.
        The subgroup can be recovered as the domain (source) of the homomorphism.
        """
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



class GroupHomomorphism:
    def __init__(self, 
        source: Group, target: Group, 
        mapping: Union[Dict[GroupElement, GroupElement], Callable[[GroupElement], GroupElement]]
    ):
        self.source = source
        self.target = target

        if type(mapping) is dict:
        # self._mapping = dict()
            for k, v in mapping.items():
                assert k.group is source
                assert v.group is target
                # self._mapping[k] = v
        self._mapping = mapping

    def __call__(self, g: GroupElement) -> GroupElement:
        assert g.group is self.source

        if type(self._mapping) is dict:
            return self._mapping[g]
        h = self._mapping(g)
        assert h.group is self.target
        return h

    def compose(self, f: GroupHomomorphism) -> GroupHomomorphism:
        """
        self \circ f
        """
        assert f.target is self.source, f"Source of the second morphism ({self.source}) must coincide with the target of the first morphism ({f.target})."

        def mapping(g):
            return self(f(g))

        return GroupHomomorphism(f.source, self.target, mapping)

    def validate(self, g: GroupElement, h: GroupElement) -> bool:
        """
        Check whether f(gh) = f(g)f(h).
        """
        return self(g) * self(h) is self(g * h)

    def validate_all(self):
        """
        Check the homomorphism property for all pairs of elements in the source group.
        Note: this is only possible for finite groups.
        """
        assert len(self.source) > 0, "Cannot validate homomorphism property for infinite groups"
        for g in self.source:
            for h in self.source:
                if not self.validate(g,h):
                    return False
        return True




class CachedGroupMeta(type):
    """
    Metaclass for caching groups like Cn or Dn.
    """
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls._cache = dict()

    def __call__(cls, *args):
        if args not in cls._cache:
            cls._cache[args] = super().__call__(*args)

        return cls._cache[args]
