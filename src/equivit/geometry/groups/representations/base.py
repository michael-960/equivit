from enum import Enum
from ..base import Group, GroupElement, GroupHomomorphism
from typing import Dict, Any, Optional, List
import numpy as np

from .complex_structure import ComplexStructure




class IrrepType(Enum):
    REAL = 'real'
    COMPLEX = 'complex'
    QUATERNIONIC = 'quaternionic'



class GroupRepresentation:
    r"""
    An object that records the representation matrices of a representation of a finite group.

    The representation can be either over :math:`\mathbb{R}` or over :math:`\mathbb{C}`. 
    If the representation is over :math:`\mathbb{R}` and admits a complex structure, then the complex structure can be specified by the complex_structure argument.

    Args:
        group: the group of the representation
        rep_matrices: a dictionary mapping group elements to representation
                        matrices. Each matrix should be of shape :math:`(d, d)`, where :math:`d`
                        is the dimension of the representation. 
        complex_structure: if the representation is over :math:`\mathbb{R}` and admits a complex structure, this argument specifies the complex structure. 
                           It should not be specified for representations over :math:`\mathbb{C}`.

    Note:
        We rely on the user to ensure that the representation matrices satisfy the group representation property, i.e., :math:`\rho(g h) = \rho(g) \rho(h)` for all group elements :math:`g, h`.

    """
    def __init__(self,
            group: Group, 
            rep_matrices: Dict[GroupElement, np.ndarray],
            complex_structure: Optional[ComplexStructure]=None
        ):

        assert group.is_finite(), "Only representations of finite groups are supported for now."

        self.group = group

        id_mat = rep_matrices[group.identity()]
        self.dim = id_mat.shape[0]

        if np.isrealobj(id_mat):
            self.field = 'real'
        else:
            assert np.iscomplexobj(id_mat)
            self.field = 'complex'
            assert complex_structure is None, "complex_structure should not be specified for complex representations"

        self.complex_structure = complex_structure

        self._rep_matrices = dict()
        for g in group:
            mat = rep_matrices[g]
            assert mat.shape == (self.dim, self.dim)

            if self.field == 'real':
                assert np.isrealobj(mat)
            elif self.field == 'complex':
                if np.isrealobj(mat):
                    mat = mat.astype(np.complex128)
                assert np.iscomplexobj(mat)
            self._rep_matrices[g] = mat

    def __call__(self, g: GroupElement):
        # raise NotImplementedError("This method should be implemented by subclasses to return the representation matrix of the group element g in this representation.")
        return self._rep_matrices[g]

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
        """
        return np.mean([self.character(g*g) for g in self.group])

    def validate(self, tol=1e-9):
        """
        Validate that this is a valid representation of the group.
        """
        for g in self.group:
            for h in self.group:
                error = np.max(np.abs((self(g*h) - self(g) @ self(h))))
                if error > tol:
                    raise ValueError(f'The representation is not valid: the representation matrix of {g}*{h} is not equal to the product of the representation matrices of {g} and {h}. The maximum absolute error (L-infinity) is {error}.')

    def as_complex(self):
        """
        If this representation is real and admits a complex structure, 
        returns the corresponding complex representation.
        """
        if self.field == 'complex':
            raise TypeError("This representation is already a complex representation.")
        if self.complex_structure is None:
            raise TypeError("This representation does not come with a complex structure.")

        return GroupRepresentation(
            self.group, 
            {g: self.complex_structure.endo_r2c(self(g), 0, 1) for g in self.group}
            )
    
    def pullback(self, homomorphism: GroupHomomorphism) -> 'GroupRepresentation':
        """
        Pull back this representation along the given group homomorphism. 
        The resulting representation is a representation of the source group of the homomorphism.
        """
        return GroupRepresentation(
            homomorphism.source, 
            {h: self(homomorphism(h)) for h in homomorphism.source}
        )



class ComplexIrrep(GroupRepresentation):
    def __init__(self,
        group: Group, 
        name: str, 
        rep_type: IrrepType, 
        rep_matrices: np.ndarray
    ):
        super().__init__(group, rep_matrices, complex_structure=None)
        self.name = name
        self.rep_type = rep_type

    def validate_irrep_type(self):
        """
        Validate that the irrep type is consistent with the character of the representation. 
        For example, if the character is real-valued, then the irrep type should be either REAL or QUATERNIONIC, but not COMPLEX. 
        """
        # something to do with the Frobenius-Schur indicator of the representation.
        fb_ind = self.frobenius_schur_indicator()
        if abs(fb_ind) < 1e-7:
            assert self.rep_type is IrrepType.COMPLEX, f'Frobenius-Schur indicator is {fb_ind}, so the representation should be of complex type'
        elif abs(fb_ind-1) < 1e-7:
            assert self.rep_type is IrrepType.REAL, f'Frobenius-Schur indicator is {fb_ind}, so the representation should be of real type'
        elif abs(fb_ind+1) < 1e-7:
            assert self.rep_type is IrrepType.QUATERNIONIC, f'Frobenius-Schur indicator is {fb_ind}, so the representation should be of quaternionic type'
        else:
            raise ValueError(f'Invalid Frobenius-Schur indicator: {fb_ind}')

    def __repr__(self):
        return f'ComplexIrrep(group={self.group}, name={self.name}, dim={self.dim}, type={self.rep_type})'

    def to_real_irrep(self):
        """
        Convert this complex irrep to a real irrep. 
        """
        raise NotImplementedError("Conversion from complex irrep to real irrep is not implemented yet.")
        if self.rep_type is IrrepType.REAL:
            # If the complex irrep is of real type, then it has a real structure C that commutes with the group action
            # The C-invariant subspace is then the corresponding real irrep.
            ...

        elif self.rep_type is IrrepType.COMPLEX or self.rep_type is IrrepType.QUATERNIONIC:
            # If the complex irrep is of complex or quaternionic type, then it has no real
            # structure, and the corresponding real irrep is just the same
            # representation but interpreted as a real representation.
            ...

        else:
            raise ValueError(f'Invalid representation type: {self.rep_type}')

    @classmethod
    def from_rep(cls, rep: GroupRepresentation, name: str, rep_type: IrrepType):
        """
        Create a ComplexIrrep object from a GroupRepresentation object.
        """
        assert rep.field == 'complex', "The representation must be a complex representation to be converted to a ComplexIrrep."
        return cls(rep.group, name=name, rep_type=rep_type, rep_matrices={g: rep(g) for g in rep.group})



class RealIrrep(GroupRepresentation):
    def __init__(self,
        group: Group, 
        name: str, 
        rep_type: IrrepType, 
        rep_matrices: np.ndarray,
        complex_structure: Optional[ComplexStructure]=None
    ):
        if rep_type is IrrepType.REAL:
            assert complex_structure is None, "A real irrep of real type should not have a complex structure."
        else:
            assert complex_structure is not None, "A real irrep of complex or quaternionic type should have a complex structure."
        super().__init__(group, rep_matrices, complex_structure=complex_structure)
        self.name = name
        self.rep_type = rep_type

    def validate_irrep_type(self):
        """
        Validate that the irrep type is consistent with the character of the representation. 
        For example, if the character is real-valued, then the irrep type should be either REAL or QUATERNIONIC, but not COMPLEX. 
        """
        fb_ind = self.frobenius_schur_indicator()
        if abs(fb_ind) < 1e-7:
            assert self.rep_type is IrrepType.COMPLEX, f'Frobenius-Schur indicator is {fb_ind}, so the representation should be of complex type'
        elif abs(fb_ind-1) < 1e-7:
            assert self.rep_type is IrrepType.REAL, f'Frobenius-Schur indicator is {fb_ind}, so the representation should be of real type'
        elif abs(fb_ind+2) < 1e-7:
            assert self.rep_type is IrrepType.QUATERNIONIC, f'Frobenius-Schur indicator is {fb_ind}, so the representation should be of quaternionic type'
        else:
            raise ValueError(f'Invalid Frobenius-Schur indicator: {fb_ind}')


    def to_complex_irrep(self):
        """
        Convert this real irrep to a complex irrep. 
        """
        raise NotImplementedError("Conversion from real irrep to complex irrep is not implemented yet.")

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

    def __repr__(self):
        return f'RealIrrep(group={self.group}, name={self.name}, dim={self.dim}, type={self.rep_type})'

    @classmethod
    def from_rep(cls, rep: GroupRepresentation, name: str, rep_type: IrrepType):
        """
        Create a RealIrrep object from a GroupRepresentation object.
        """
        assert rep.field == 'real', "The representation must be a real representation to be converted to a RealIrrep."
        return cls(rep.group, name=name, rep_type=rep_type, rep_matrices={g: rep(g) for g in rep.group}, complex_structure=rep.complex_structure)


