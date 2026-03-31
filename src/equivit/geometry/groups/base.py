from __future__ import annotations
from enum import Enum
import numpy as np
from typing import Generic, TypeVar, Type




class Group(Enum):

    def __mul__(self, other):
        raise NotImplementedError("This method should be implemented by subclasses to define the group multiplication.")

    def inv(self):
        raise NotImplementedError("This method should be implemented by subclasses to return the inverse of the group element.")

    @classmethod
    def irreps(cls) -> Type[GroupIrreps]:
        raise NotImplementedError("This method should be implemented by subclasses to return the class that contains the irreducible representations of the group.")
        
    @classmethod
    def conjugacy_classes(cls):
        raise NotImplementedError("This method should be implemented by subclasses to return the conjugacy classes of the group.")

    @classmethod
    def identity(cls):
        """Returns the identity element of the group."""
        raise NotImplementedError("This method should be implemented by subclasses to return the identity element of the group.")

    @classmethod
    def from_word(cls, word: str):
        """Constructs a group element from a word in the generators."""
        raise NotImplementedError("This method should be implemented by subclasses to construct a group element from a word in the generators.")


class GroupIrreps(Enum):
    dim: int
    def character(self, g):
        return self(g).trace()

    def characters(self):
        """
        Returns an array of characters, one for each conjugacy class of the group. 
        """
        return np.array([self.character(cc[0]) for cc in self.__class__.group_class().conjugacy_classes()])

    def all_characters(self):
        """
        Returns an array of characters, one for each group element.
        """
        return np.array([self.character(g) for g in self.__class__.group_class()])

    def __call__(self, g):
        raise NotImplementedError("This method should be implemented by subclasses to return the representation matrix of the group element g in this irrep.")

    @classmethod
    def group_class(cls) -> Type[Group]:
        raise NotImplementedError




class TrivialGroup(Group):
    e = ''

    @classmethod
    def conjugacy_classes(cls):
        return [[cls.e]]

    @classmethod
    def identity(cls):
        return cls.e

    @classmethod
    def irreps(cls):
        return TrivialGroupIrreps


class TrivialGroupIrreps(GroupIrreps):
    A = 'A'

    def __init__(self, name: str):
        self.irrep_name = name
        self.dim = 1

    def __call__(self, g: TrivialGroup):
        return np.array([[1.]])