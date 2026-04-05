from __future__ import annotations
from enum import Enum
import numpy as np
from typing import Generic, TypeVar, Type, Tuple, Dict, Any, Literal, List, Optional, Union, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .action import GroupAction






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

    def complex_irreps(self):
        """
        Returns the complex irreducible representations of the group.
        """
        raise NotImplementedError("This method should be implemented by subclasses to return the class that contains the complex irreducible representations of the group.")

    def real_irreps(self):
        raise NotImplementedError("This method should be implemented by subclasses to return the class that contains the irreducible representations of the group.")

        # TODO: maybe we can compute the real irreps automatically from the complex irreps

        # complex_irreps = self.complex_irreps()
        # real_irreps = dict()

        # for irrep_name, irrep in complex_irreps.items():

    def subgroups_up_to_conjugacy(self) -> List[tuple]:
        """
        Returns a list of subgroups of the group, up to conjugacy. 
        Note: the output of this method should be a list of argument tuples to
        be passed to the subgroup method to obtain the corresponding subgroup
        homomorphisms.

        Note: two subgroups H, K are conjugate if there exists g in G such that gHg^{-1} = K.
        """
        raise NotImplementedError("This method should be implemented by subclasses to return a list of subgroups of the group, up to conjugacy.")

    def homogeneous_space_action(self, *subgroup_args) -> 'GroupAction':
        """
        Given some arguments specifying a subgroup, return the homogeneous space action of the group on the left cosets of the subgroup.
        The subgroup can be recovered as the stabilizer of the identity coset.

        Note: current implementation only works for finite groups, since we need to enumerate the cosets to compute the action.        """
        assert self.is_finite(), "homogeneous_space_action is only implemented for finite groups."

        from .action import GroupAction 

        inclusion = self.subgroup(*subgroup_args)
        subgroup = inclusion.source
        elements = [g for g in self]

        # compute left cosets of the subgroup
        cosets = []
        while len(elements) > 0:
            g = elements[0]
            coset = []
            for h in subgroup:
                coset.append(g * inclusion(h))

            elements = [a for a in elements if a not in coset]
            cosets.append(coset)

        # set up a lookup dictionary to find the index of the coset that each group element belongs to
        element_orbit_dict = {}
        for g in self:
            _coset_found = False
            for i, coset in enumerate(cosets):
                if g in coset:
                    element_orbit_dict[g] = i
                    _coset_found = True
                    break
            if not _coset_found:
                raise ValueError(f"Group element {g} not found in any coset, this should not happen.")

        # set up action dict
        action_dict = {}

        for g in self:
            action_dict[g] = []
            for i, coset in enumerate(cosets):
                action_dict[g].append(element_orbit_dict[g*coset[0]])
        
        return GroupAction(self, action_dict)

    def all_homogeneous_space_actions(self) -> 'List[GroupAction]':
        """
        Return the homogeneous space actions of the group on the left cosets of all subgroups.
        If two subgroups are conjugate, then the corresponding homogeneous space actions are isomorphic, so we only return one of them.

        Note: this is only implemented for finite groups (specifically for those whose subgroups_up_to_conjugacy method is implemented).
        """
        assert self.is_finite(), "all_homogeneous_space_actions is only implemented for finite groups."

        actions = []
        for subgroup_args in self.subgroups_up_to_conjugacy():
            actions.append(self.homogeneous_space_action(*subgroup_args))

        return actions
    

 




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
