from enum import Enum
from .action import GroupRepresentation
import numpy as np


# THIS FILE IS EXPERIMENTAL AND NOT USED ANYWHERE ELSE IN THE CODEBASE YET.
# Idea: we often want to attach some extra information to each irrep. 

class RepresentationType(Enum):
    REAL = 'real'
    COMPLEX = 'complex'
    QUATERNIONIC = 'quaternionic'


class Intertwiner:
    source: GroupRepresentation
    target: GroupRepresentation
    ...


class Irrep:
    """
    Abstract base class
    """
    ...

class ComplexIrrep(GroupRepresentation):
    def __init__(self, name: str, rep_type: RepresentationType, rep_matrices: np.ndarray):
        ...

    def validate_irrep_type(self):
        """
        Validate that the irrep type is consistent with the character of the representation. 
        For example, if the character is real-valued, then the irrep type should be either REAL or QUATERNIONIC, but not COMPLEX. 
        """
        # something to do with the Frobenius-Schur indicator of the representation.
        ...


    def to_real_irrep(self):
        """
        Convert this complex irrep to a real irrep. 
        """
        if self.rep_type is RepresentationType.REAL:
            # If the complex irrep is of real type, then it has a real structure C that commutes with the group action
            # The C-invariant subspace is then the corresponding real irrep.
            ...

        elif self.rep_type is RepresentationType.COMPLEX or self.rep_type is RepresentationType.QUATERNIONIC:
            # If the complex irrep is of complex or quaternionic type, then it has no real
            # structure, and the corresponding real irrep is just the same
            # representation but interpreted as a real representation.
            ...

        else:
            raise ValueError(f'Invalid representation type: {self.rep_type}')


class RealIrrep(GroupRepresentation):
    rep_type: RepresentationType

    def to_complex_irrep(self):
        """
        Convert this real irrep to a complex irrep. 
        """
        if self.rep_type is RepresentationType.REAL:
            # return the complexification (this means we return the same matrices but interpret them as complex matrices instead of real matrices)
            ...

        elif self.rep_type is RepresentationType.COMPLEX or self.rep_type is RepresentationType.QUATERNIONIC:
            # in this case this irrep has a complex structure J that commutes with the group action, 
            # and we can use it to define a complex irrep structure on the same vector space.
            # In this case, the complex dimension of the complex irrep is half the real dimension of the original irrep.
            ...

        else:
            raise ValueError(f'Invalid representation type: {self.rep_type}')
