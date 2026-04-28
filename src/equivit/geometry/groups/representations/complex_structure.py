import numpy as np
import warnings


class ComplexStructure:
    """
    If V is a real vector space, a complex structure is a linear map J: V->V
    that squares to -1. 

    For us, a complex structure will mean something more: 
    it will specify how to convert vectors and endomorphisms of the real vector space to complex ones.

    Subclasses should implement such a structure by directly specifying 
    how real vectors and endomorphisms are converted to complex ones and vice versa.

    The consistency these conversions should be taken care of by the 
    implementation. I.e., it will not be checked.
    """
    def vector_r2c(self, x: np.ndarray, axis: int):
        """
        Convert a 2n-dimensional real vector to an n-dimensional complex vector.
        """
        raise NotImplementedError()

    def vector_c2r(self, x: np.ndarray, axis: int):
        """
        Convert an n-dimensional complex vector to a 2n-dimensional complex vector.
        """
        raise NotImplementedError()

    def dual_vector_r2c(self, x: np.ndarray, axis: int): 
        """
        Suppose a is a dual vector. The corresponding complex dual vector b is defined by
        <b, v> = <a, v> + i<a, -Jv>, where J is the complex structure.
        """
        ...

    def dual_vector_c2r(self, x: np.ndarray, axis: int):
        """
        Suppose b is a complex dual vector. The corresponding real dual vector a is defined by
        <a, v> = Re(<b, v>), where J is the complex structure.
        """
        ...

    def endo_r2c(self, x: np.ndarray, axis1: int, axis2: int):
        """
        Convert a 2n-dimensional real endomorphism to an n-dimensional complex endomorphism.
        """
        raise NotImplementedError()

    def endo_c2r(self, x: np.ndarray, axis1: int, axis2: int):
        """
        Convert an n-dimensional complex endomorphism to a 2n-dimensional real endomorphism.
        """
        raise NotImplementedError()


class StandardComplexStructure(ComplexStructure):
    """
    Standard complex structure on R^{2n}

    TODO: Fix the implementation for complex_dim > 1
    For example, a real tensor x = [x_0, x_1, x_2, x_3]
    should be identified with the complex tensor
    [x_0+ix_1, x_2+ix_3] if complex_dim=2, instead of 
    [x_0+ix_2, x_1+ix_3] as the current implementation does.
    """
    def __init__(self, complex_dim):
        if complex_dim != 1:
            warnings.warn("StandardComplexStructure will not play well with torch.view_as_complex and torch.view_as_real for complex_dim > 1.")
        self.complex_dim = complex_dim

    def vector_r2c(self, x: np.ndarray, axis: int):
        x_re, x_im = np.split(x, [self.complex_dim], axis=axis)
        return x_re + 1j*x_im

    def vector_c2r(self, x: np.ndarray, axis: int):
        return np.concatenate((x.real, x.imag), axis=axis)

    def dual_vector_r2c(self, x, axis):
        return self.vector_r2c(x, axis).conj()

    def dual_vector_c2r(self, x: np.ndarray, axis: int):
        return self.vector_c2r(x.conj(), axis)

    def endo_r2c(self, x: np.ndarray, axis1: int, axis2: int):
        x_re = np.split(
                    np.split(x, [self.complex_dim], axis=axis1)[0],
                    [self.complex_dim], axis=axis2)[0]

        x_im = np.split(
                    np.split(x, [self.complex_dim], axis=axis1)[1], 
                    [self.complex_dim], axis=axis2)[0]
        return x_re + 1j*x_im

    def endo_c2r(self, x, axis1: int, axis2: int):
        y = np.concatenate((x.real, -x.imag), axis=axis2)
        z = np.concatenate((x.imag, x.real), axis=axis2)
        return np.concatenate((y, z), axis=axis1)

   

